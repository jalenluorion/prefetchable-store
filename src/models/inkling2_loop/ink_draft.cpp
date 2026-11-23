// Inkling 2 Mini's draft head (ink_draft.h): weights, the BFMMLA bf16 products, context K/V, the block, the Markov RNN, the
// output layer and the confidence gate.
#include "ink_draft.h"

#include <arm_neon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../../format/files.h"
#include "dmeng/kernels_batch.h"

namespace dm {

// ------------------------------------------------------------------------------------------------ bf16
static inline uint16_t f2bf(float f) {  // round to nearest even (MLX's astype(bfloat16))
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0xffff) ? 0x40 : 0));
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
static inline float bfr(float f) { return bf16_to_f32(f2bf(f)); }
static inline void bfr_n(float* x, size_t n) {
  for (size_t i = 0; i < n; i++) x[i] = bfr(x[i]);
}

// rmsnorm as MLX's rms(): fp32 x / sqrt(mean(x^2) + eps) * w, rounded to bf16 (w null: no weight)
static void rms_bf(const float* x, const float* w, int n, float eps, float* out) {
  float ss = 0;
  for (int i = 0; i < n; i++) ss += x[i] * x[i];
  const float r = 1.f / std::sqrt(ss / n + eps);
  for (int i = 0; i < n; i++) out[i] = bfr(x[i] * r * (w ? w[i] : 1.f));
}

// ------------------------------------------------------------------------------------------------ bf16 products (BFMMLA)
// Inputs packed in slot pairs: P[(sp * K/4 + kc) * 8 ..] = {x[2 sp][4 kc .. +3], x[2 sp + 1][4 kc .. +3]} (bf16), pairs
// padded to a multiple of 4 with zeros. One BFMMLA: rows (r, r+1) x 4 inputs . slots (s, s+1) -> 2 x 2 fp32 sums.
static void pack_pairs(const uint16_t* X, int n, int K, uint16_t* P) {
  const int npp = ((n + 1) / 2 + 3) / 4 * 4, KC = K / 4;
  for (int sp = 0; sp < npp; sp++) {
    const uint16_t* a = 2 * sp < n ? X + (size_t)(2 * sp) * K : nullptr;
    const uint16_t* b = 2 * sp + 1 < n ? X + (size_t)(2 * sp + 1) * K : nullptr;
    uint16_t* d = P + (size_t)sp * KC * 8;
    for (int kc = 0; kc < KC; kc++, d += 8) {
      if (a) std::memcpy(d, a + kc * 4, 8);
      else std::memset(d, 0, 8);
      if (b) std::memcpy(d + 4, b + kc * 4, 8);
      else std::memset(d + 4, 0, 8);
    }
  }
}
static size_t packed_elems(int n, int K) { return (size_t)(((n + 1) / 2 + 3) / 4 * 4) * K * 2; }

// Y[s][r] (s in [2 sp0, 2 sp0 + 2 NP) and < n) = W[r] . x_s, rows [r0, r1), r1 - r0 a multiple of 4, K a multiple of 8
template <int NP>
static void gemm_tile(const uint16_t* W, int K, int r0, int r1, const uint16_t* P, int sp0, int n, float* const* Y) {
  const int KC = K / 4;
  for (int r = r0; r < r1; r += 4) {
    float32x4_t acc[2][NP];
    for (int a = 0; a < 2; a++)
      for (int s = 0; s < NP; s++) acc[a][s] = vdupq_n_f32(0);
    const uint16_t* w0 = W + (size_t)r * K;
    const uint16_t* w1 = w0 + K;
    const uint16_t* w2 = w1 + K;
    const uint16_t* w3 = w2 + K;
    const uint16_t* pb = P + (size_t)sp0 * KC * 8;
    for (int k = 0; k < K; k += 8) {
      const uint64x2_t a0 = vreinterpretq_u64_u16(vld1q_u16(w0 + k)), a1 = vreinterpretq_u64_u16(vld1q_u16(w1 + k));
      const uint64x2_t a2 = vreinterpretq_u64_u16(vld1q_u16(w2 + k)), a3 = vreinterpretq_u64_u16(vld1q_u16(w3 + k));
      const bfloat16x8_t A00 = vreinterpretq_bf16_u64(vzip1q_u64(a0, a1)), A01 = vreinterpretq_bf16_u64(vzip2q_u64(a0, a1));
      const bfloat16x8_t A10 = vreinterpretq_bf16_u64(vzip1q_u64(a2, a3)), A11 = vreinterpretq_bf16_u64(vzip2q_u64(a2, a3));
      const uint16_t* pk = pb + (size_t)(k / 4) * 8;
      for (int s = 0; s < NP; s++) {
        const bfloat16x8_t p0 = vreinterpretq_bf16_u16(vld1q_u16(pk + (size_t)s * KC * 8));
        const bfloat16x8_t p1 = vreinterpretq_bf16_u16(vld1q_u16(pk + (size_t)s * KC * 8 + 8));
        acc[0][s] = vbfmmlaq_f32(acc[0][s], A00, p0);
        acc[1][s] = vbfmmlaq_f32(acc[1][s], A10, p0);
        acc[0][s] = vbfmmlaq_f32(acc[0][s], A01, p1);
        acc[1][s] = vbfmmlaq_f32(acc[1][s], A11, p1);
      }
    }
    for (int a = 0; a < 2; a++)
      for (int s = 0; s < NP; s++) {
        const int s0 = 2 * (sp0 + s), rr = r + 2 * a;
        float v[4];
        vst1q_f32(v, acc[a][s]);
        if (s0 < n) Y[s0][rr] = v[0], Y[s0][rr + 1] = v[2];
        if (s0 + 1 < n) Y[s0 + 1][rr] = v[1], Y[s0 + 1][rr + 1] = v[3];
      }
  }
}

struct GJob {  // one matrix [R, K] bf16 for n packed inputs; Y[s] + r
  const uint16_t* W;
  int R, K, n;
  const uint16_t* P;
  float* const* Y;
};
struct GCtx {
  const GJob* j;
  int nj, chunk;
  std::vector<int> start;
  alignas(128) std::atomic<int> next{0};
};
static void phase_gemm(void* p, int, int) {
  GCtx& C = *(GCtx*)p;
  const int tot = C.start[C.nj];
  for (int t; (t = C.next.fetch_add(1, std::memory_order_relaxed)) < tot;) {
    int ji = 0;
    while (C.start[ji + 1] <= t) ji++;
    const GJob& J = C.j[ji];
    const int r0 = (t - C.start[ji]) * C.chunk, r1 = std::min(J.R, r0 + C.chunk);
    const int npp = ((J.n + 1) / 2 + 3) / 4 * 4;
    const int used = (J.n + 1) / 2;  // pairs holding a slot
    for (int sp = 0; sp < npp && sp < used; sp += 4) {
      const int left = used - sp;
      if (left >= 3) gemm_tile<4>(J.W, J.K, r0, r1, J.P, sp, J.n, J.Y);
      else if (left == 2) gemm_tile<2>(J.W, J.K, r0, r1, J.P, sp, J.n, J.Y);
      else gemm_tile<1>(J.W, J.K, r0, r1, J.P, sp, J.n, J.Y);
    }
  }
}
static void run_gemm(Pool& pool, const GJob* j, int nj, int chunk = 32) {
  GCtx C;
  C.j = j;
  C.nj = nj;
  C.chunk = chunk;
  C.start.resize(nj + 1);
  C.start[0] = 0;
  for (int i = 0; i < nj; i++) C.start[i + 1] = C.start[i] + (j[i].R + chunk - 1) / chunk;
  pool.run(phase_gemm, &C);
}

// ------------------------------------------------------------------------------------------------ load
InkDraft::~InkDraft() {
  free_region(wreg_, wreg_bytes_);
  free_region(freg_, freg_bytes_);
  for (int i = 0; i < 3; i++) {
    free_region((uint8_t*)ck_[i], cbytes_[i]);
    free_region((uint8_t*)cv_[i], cbytes_[i]);
  }
  if (umap_) munmap(umap_, umap_bytes_);
  if (mtp_fd_ >= 0) close(mtp_fd_);
}

static std::vector<float> to_f32(const uint16_t* p, size_t n) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; i++) v[i] = bf16_to_f32(p[i]);
  return v;
}

bool InkDraft::load(const InkDraftOpts& o, Pool* pool, std::string& err) {
  o_ = o;
  pool_ = pool;
  H_ = o.H;
  VU_ = o.VU;
  if (!read_all(err)) return false;
  if (o.exact) {  // the released bf16 unembedding, mapped
    ms::SafeTensors st;
    if (!st.open(o.resident, err)) return false;
    const ms::StTensor* t = st.find("model.llm.unembed.weight");
    if (!t || t->dtype != "BF16" || t->shape.size() != 2 || t->shape[1] != H_) return (err = "resident.safetensors: no bf16 unembedding"), false;
    const int fd = open(o.resident.c_str(), O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    umap_bytes_ = sb.st_size;
    umap_ = mmap(nullptr, umap_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (umap_ == MAP_FAILED) return (umap_ = nullptr), (err = "mmap " + o.resident), false;
    U_.kind = Weight::BF16;
    U_.rows = VU_;
    U_.cols = H_;
    U_.w = (const uint8_t*)umap_ + st.data_offset(*t);
  } else if (!make_fold(err)) {
    return false;
  }
  // context K / V
  const int KVD = NKV_ * HD_;
  for (int i = 0; i < 3; i++) {
    ccap_[i] = i == 1 ? o.ctx : std::min(WIN_, o.ctx);
    cbytes_[i] = (size_t)ccap_[i] * KVD * 2;
    ck_[i] = (uint16_t*)alloc_region(cbytes_[i], Mem::PLAIN, &err);
    cv_[i] = (uint16_t*)alloc_region(cbytes_[i], Mem::PLAIN, &err);
    if (!ck_[i] || !cv_[i]) return false;
  }
  for (auto& x : hx_) x.init(H_, o.prec);
  hl_.resize((size_t)7 * VU_);
  lg_.resize(VU_);
  last_.assign((size_t)4 * H_, 0.f);
  return true;
}

bool InkDraft::read_all(std::string& err) {
  ms::SafeTensors st;
  if (!st.open(o_.mtp, err)) return false;
  // bf16 matrices into one region (64-byte aligned each); markov_w1 stays in the file
  struct M {
    std::string name;
    const uint16_t** dst;
  };
  std::vector<M> mats;
  for (int i = 0; i < 3; i++) {
    const std::string p = "layers." + std::to_string(i) + ".";
    Layer& L = L_[i];
    mats.push_back({p + "self_attn.q_proj.weight", &L.q});
    mats.push_back({p + "self_attn.k_proj.weight", &L.k});
    mats.push_back({p + "self_attn.v_proj.weight", &L.v});
    mats.push_back({p + "self_attn.o_proj.weight", &L.o});
    mats.push_back({p + "mlp.gate_proj.weight", &L.g});
    mats.push_back({p + "mlp.up_proj.weight", &L.u});
    mats.push_back({p + "mlp.down_proj.weight", &L.d});
    mats.push_back({p + "self_attn.k_proj_ctx.weight", &L.kc});
    mats.push_back({p + "self_attn.v_proj_ctx.weight", &L.vc});
  }
  mats.push_back({"markov_head.rnn_proj.weight", &rnn_w_});
  mats.push_back({"markov_head.hidden_out.weight", &ho_w_});
  size_t tot = 0;
  for (auto& m : mats) {
    const ms::StTensor* t = st.find(m.name);
    if (!t || t->dtype != "BF16") return (err = "mtp.safetensors: missing bf16 " + m.name), false;
    tot += (t->end - t->begin + 63) / 64 * 64;
  }
  wreg_bytes_ = tot;
  wreg_ = alloc_region(tot, o_.mem, &err);
  if (!wreg_) return false;
  size_t off = 0;
  for (auto& m : mats) {
    const ms::StTensor* t = st.find(m.name);
    if (!st.read(*t, wreg_ + off, err)) return false;
    *m.dst = (const uint16_t*)(wreg_ + off);
    off += (t->end - t->begin + 63) / 64 * 64;
  }
  wbytes_ = off;
  auto vec = [&](const std::string& n, size_t want, std::vector<float>& out) {
    const ms::StTensor* t = st.find(n);
    if (!t || t->dtype != "BF16" || (size_t)t->numel() != want) return (err = "mtp.safetensors: bad " + n), false;
    std::vector<uint16_t> b(want);
    if (!st.read(*t, b.data(), err)) return false;
    out = to_f32(b.data(), want);
    wbytes_ += want * 2;
    return true;
  };
  for (int i = 0; i < 3; i++) {
    const std::string p = "layers." + std::to_string(i) + ".";
    Layer& L = L_[i];
    if (!vec(p + "input_layernorm.weight", H_, L.ln_in) || !vec(p + "post_attention_layernorm.weight", H_, L.ln_post) ||
        !vec(p + "self_attn.q_norm.weight", HD_, L.qn) || !vec(p + "self_attn.k_norm.weight", HD_, L.kn) ||
        !vec(p + "self_attn.k_norm_ctx.weight", HD_, L.kcn) || !vec(p + "dflare_norm.weight", H_, L.dfl))
      return false;
  }
  std::vector<float> fw;
  if (!vec("norm.weight", H_, norm_) || !vec("mask_embed", H_, mask_) || !vec("fuse_weight", 16, fw) ||
      !vec("markov_head.dflare_norm.weight", H_, mk_dfl_) || !vec("markov_head.rnn_proj.bias", 3 * MR_, rnn_b_) ||
      !vec("markov_head.hidden_out.bias", H_, ho_b_))
    return false;
  // alpha = softmax(fuse_weight) per row in fp32, then bf16 (as the reference)
  alpha_.resize(16);
  for (int r = 0; r < 4; r++) {
    float mx = -INFINITY, z = 0;
    for (int t = 0; t < 4; t++) mx = std::max(mx, fw[r * 4 + t]);
    for (int t = 0; t < 4; t++) z += std::exp(fw[r * 4 + t] - mx);
    for (int t = 0; t < 4; t++) alpha_[r * 4 + t] = bfr(std::exp(fw[r * 4 + t] - mx) / z);
  }
  const ms::StTensor* w1 = st.find("markov_head.markov_w1.weight");
  if (!w1 || w1->dtype != "BF16" || w1->shape.size() != 2 || w1->shape[1] != MR_) return (err = "mtp.safetensors: bad markov_w1"), false;
  w1_off_ = st.data_offset(*w1);
  mtp_fd_ = open(o_.mtp.c_str(), O_RDONLY);
  if (mtp_fd_ < 0) return (err = "open " + o_.mtp), false;
  return true;
}

// F = unembed . hidden_out.W [VU, 256] (bf16), c = unembed . hidden_out.b [VU]: computed once from the released bf16
// tensors with fp32 sums, kept in o.fold (magic, VU, rank, the two source sizes, F, c)
bool InkDraft::make_fold(std::string& err) {
  struct stat s1, s2;
  if (stat(o_.mtp.c_str(), &s1) || stat(o_.resident.c_str(), &s2)) return (err = "stat mtp / resident"), false;
  const uint64_t key[4] = {0x31444c4f464b4e49ull /* INKFOLD1 */, ((uint64_t)VU_ << 32) | (uint64_t)MR_, (uint64_t)s1.st_size, (uint64_t)s2.st_size};
  const size_t fb = (size_t)VU_ * MR_ * 2;
  freg_bytes_ = (fb + 63) / 64 * 64;
  freg_ = alloc_region(freg_bytes_, o_.mem, &err);
  if (!freg_) return false;
  fc_.resize(VU_);
  bool have = false;
  if (FILE* f = fopen(o_.fold.c_str(), "rb")) {
    uint64_t k[4];
    have = fread(k, 8, 4, f) == 4 && !std::memcmp(k, key, sizeof k) && fread(freg_, 1, fb, f) == fb && fread(fc_.data(), 4, VU_, f) == (size_t)VU_;
    fclose(f);
  }
  if (!have) {
    ms::SafeTensors st;
    if (!st.open(o_.resident, err)) return false;
    const ms::StTensor* t = st.find("model.llm.unembed.weight");
    if (!t || t->dtype != "BF16" || t->shape.size() != 2 || t->shape[1] != H_) return (err = "resident.safetensors: no bf16 unembedding"), false;
    const int fd = open(o_.resident.c_str(), O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    void* map = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return (err = "mmap " + o_.resident), false;
    const uint16_t* U = (const uint16_t*)((const uint8_t*)map + st.data_offset(*t));
    // inputs: the 256 columns of hidden_out.W and its bias, as bf16 rows of H
    const int n = MR_ + 1;
    std::vector<uint16_t> X((size_t)n * H_);
    for (int m = 0; m < MR_; m++)
      for (int h = 0; h < H_; h++) X[(size_t)m * H_ + h] = ho_w_[(size_t)h * MR_ + m];
    for (int h = 0; h < H_; h++) X[(size_t)MR_ * H_ + h] = f2bf(ho_b_[h]);
    std::vector<uint16_t> P(packed_elems(n, H_));
    pack_pairs(X.data(), n, H_, P.data());
    const int RB = 8192;  // rows per pass
    std::vector<float> Yb((size_t)n * RB);
    std::vector<float*> Y(n);
    uint16_t* F = (uint16_t*)freg_;
    for (int r0 = 0; r0 < VU_; r0 += RB) {
      const int nr = std::min(RB, VU_ - r0), nr4 = (nr + 3) / 4 * 4;
      for (int i = 0; i < n; i++) Y[i] = Yb.data() + (size_t)i * RB;
      std::vector<uint16_t> tail;
      const uint16_t* Wb = U + (size_t)r0 * H_;
      if (nr4 != nr) {  // the last block padded to 4 rows
        tail.assign((size_t)nr4 * H_, 0);
        std::memcpy(tail.data(), Wb, (size_t)nr * H_ * 2);
        Wb = tail.data();
      }
      GJob j{Wb, nr4, H_, n, P.data(), Y.data()};
      run_gemm(*pool_, &j, 1, 16);
      for (int r = 0; r < nr; r++) {
        for (int m = 0; m < MR_; m++) F[(size_t)(r0 + r) * MR_ + m] = f2bf(Yb[(size_t)m * RB + r]);
        fc_[r0 + r] = Yb[(size_t)MR_ * RB + r];
      }
    }
    munmap(map, sb.st_size);
    const std::string tmp = o_.fold + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    bool ok = f && fwrite(key, 8, 4, f) == 4 && fwrite(freg_, 1, fb, f) == fb && fwrite(fc_.data(), 4, VU_, f) == (size_t)VU_;
    if (f) ok = fclose(f) == 0 && ok;
    if (ok) rename(tmp.c_str(), o_.fold.c_str());
    else fprintf(stderr, "draft head: could not write the fold cache %s (kept in memory)\n", o_.fold.c_str());
  }
  F_.kind = Weight::BF16;
  F_.rows = VU_;
  F_.cols = MR_;
  F_.w = freg_;
  wbytes_ += fb + (size_t)VU_ * 4;
  return true;
}

size_t InkDraft::state_bytes() const {
  size_t n = 0;
  for (int i = 0; i < 3; i++) n += 2 * cbytes_[i];
  return n + pend_.capacity() * 4;
}

// ------------------------------------------------------------------------------------------------ features
void InkDraft::add_taps(const float* rows, int n, int p0) {
  if (p0 != next_pos()) truncate(p0);  // a gap cannot happen in order; a restart drops what was there
  const size_t row = (size_t)4 * H_;
  pend_.resize((size_t)(npend_ + n) * row);
  float* d = pend_.data() + (size_t)npend_ * row;
  for (size_t i = 0; i < (size_t)n * row; i++) d[i] = bfr(rows[i]);  // the reference feeds the features in bf16
  npend_ += n;
}

void InkDraft::truncate(int pos) {
  if (pos >= next_pos()) return;
  if (pos >= nproj_) {
    npend_ = pos - nproj_;
    return;
  }
  // projected positions are rolled back: their context K/V will be overwritten (the global layer's by position, the rings
  // by slot); the features of the new last position are not kept, so drafting waits for the next one
  nproj_ = pos;
  npend_ = 0;
  if (pos == 0) last_.assign(last_.size(), 0.f);
}

// rope (neox halves) at position pos on n heads of HD (bf16 values in and out)
static void rope_bf(float* x, int nh, int HD, int pos, double theta) {
  const int half = HD / 2;
  float c[64], s[64];
  for (int i = 0; i < half; i++) {
    const double inv = 1.0 / std::pow(theta, (double)(2 * i) / HD);
    const float ang = (float)((double)pos * inv);
    c[i] = std::cos(ang);
    s[i] = std::sin(ang);
  }
  for (int h = 0; h < nh; h++) {
    float* v = x + (size_t)h * HD;
    for (int i = 0; i < half; i++) {
      const float x1 = v[i], x2 = v[i + half];
      v[i] = bfr(x1 * c[i] - x2 * s[i]);
      v[i + half] = bfr(x2 * c[i] + x1 * s[i]);
    }
  }
}

void InkDraft::project(int n) {
  if (n <= 0) return;
  auto t0 = clk::now();
  const int KVD = NKV_ * HD_;
  const size_t row = (size_t)4 * H_;
  xb_.resize((size_t)3 * n * H_);
  std::vector<float> m(H_), hb(H_);
  for (int i = 0; i < 3; i++)
    for (int s = 0; s < n; s++) {  // mix_norm(features, i, dflare_norm_i)
      const float* f = pend_.data() + (size_t)s * row;
      for (int h = 0; h < H_; h++) {
        float t = 0;
        for (int k = 0; k < 4; k++) t += f[(size_t)k * H_ + h] * alpha_[i * 4 + k];
        m[h] = t;
      }
      rms_bf(m.data(), L_[i].dfl.data(), H_, eps_, hb.data());
      uint16_t* d = xb_.data() + ((size_t)i * n + s) * H_;
      for (int h = 0; h < H_; h++) d[h] = f2bf(hb[h]);
    }
  const size_t pe = packed_elems(n, H_);
  pk_.resize(3 * pe);
  for (int i = 0; i < 3; i++) pack_pairs(xb_.data() + (size_t)i * n * H_, n, H_, pk_.data() + i * pe);
  std::vector<float> out((size_t)6 * n * KVD);
  std::vector<float*> Y((size_t)6 * n);
  for (int j = 0; j < 6 * n; j++) Y[j] = out.data() + (size_t)j * KVD;
  GJob jb[6];
  for (int i = 0; i < 3; i++) {
    jb[2 * i] = {L_[i].kc, KVD, H_, n, pk_.data() + i * pe, Y.data() + (size_t)(2 * i) * n};
    jb[2 * i + 1] = {L_[i].vc, KVD, H_, n, pk_.data() + i * pe, Y.data() + (size_t)(2 * i + 1) * n};
  }
  run_gemm(*pool_, jb, 6);
  for (int i = 0; i < 3; i++)
    for (int s = 0; s < n; s++) {
      const int pos = nproj_ + s;
      float* k = Y[(size_t)(2 * i) * n + s];
      float* v = Y[(size_t)(2 * i + 1) * n + s];
      bfr_n(k, KVD);
      for (int h = 0; h < NKV_; h++) rms_bf(k + (size_t)h * HD_, L_[i].kcn.data(), HD_, eps_, k + (size_t)h * HD_);
      rope_bf(k, NKV_, HD_, pos, theta_);
      const int slot = pos % ccap_[i];
      uint16_t* kd = ck_[i] + (size_t)slot * KVD;
      uint16_t* vd = cv_[i] + (size_t)slot * KVD;
      for (int x = 0; x < KVD; x++) {
        kd[x] = f2bf(k[x]);
        vd[x] = f2bf(v[x]);
      }
    }
  std::memcpy(last_.data(), pend_.data() + (size_t)(n - 1) * row, row * 4);
  pend_.erase(pend_.begin(), pend_.begin() + (size_t)n * row);
  npend_ -= n;
  nproj_ += n;
  n_proj += n;
  ms_proj += ms_since(t0);
}

// ------------------------------------------------------------------------------------------------ the block
namespace {
struct AttnB {  // the block's attention for one layer: per KV head, 3 query heads x 8 slots over context + block keys
  const float* q;  // [8][NH * HD]
  const float* k;  // block keys [8][NKV * HD]
  const float* v;
  const uint16_t *ck, *cv;  // context [cap][NKV * HD]
  int cap, lo, p, NH, NKV, HD, BS;
  float* o;  // [8][NH * HD]
  std::vector<std::vector<float>>* sc;
  alignas(128) std::atomic<int> next{0};
};
void phase_attn_b(void* pp, int tid, int) {
  AttnB& X = *(AttnB*)pp;
  const int rep = X.NH / X.NKV, HD = X.HD, KVD = X.NKV * HD, nq = rep * X.BS;
  const int nc = X.p - X.lo + 1, n = nc + X.BS;
  const float scale = 1.f / std::sqrt((float)HD);
  std::vector<float>& sc = (*X.sc)[tid];
  if ((int)sc.size() < nq * n + nq * HD) sc.resize((size_t)nq * n + nq * HD + 4096);
  for (int j; (j = X.next.fetch_add(1, std::memory_order_relaxed)) < X.NKV;) {
    // queries (slot s, head j * rep + r) at index s * rep + r
    const float* qv[64];
    for (int s = 0; s < X.BS; s++)
      for (int r = 0; r < rep; r++) qv[s * rep + r] = X.q + (size_t)s * X.NH * HD + (size_t)(j * rep + r) * HD;
    float kf[256];
    for (int i = 0; i < n; i++) {
      const float* kr;
      if (i < nc) {
        const uint16_t* kb = X.ck + (size_t)((X.lo + i) % X.cap) * KVD + (size_t)j * HD;
        for (int d = 0; d < HD; d++) kf[d] = bf16_to_f32(kb[d]);
        kr = kf;
      } else {
        kr = X.k + (size_t)(i - nc) * KVD + (size_t)j * HD;
      }
      for (int qi = 0; qi < nq; qi++) {
        const float* q = qv[qi];
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0;
        for (int d = 0; d < HD; d += 8) {
          a0 = vfmaq_f32(a0, vld1q_f32(q + d), vld1q_f32(kr + d));
          a1 = vfmaq_f32(a1, vld1q_f32(q + d + 4), vld1q_f32(kr + d + 4));
        }
        sc[(size_t)qi * n + i] = vaddvq_f32(vaddq_f32(a0, a1)) * scale;
      }
    }
    float* acc = sc.data() + (size_t)nq * n;
    for (int qi = 0; qi < nq; qi++) {
      float* s = sc.data() + (size_t)qi * n;
      float mx = -INFINITY, z = 0;
      for (int i = 0; i < n; i++) mx = std::max(mx, s[i]);
      for (int i = 0; i < n; i++) z += (s[i] = std::exp(s[i] - mx));
      const float iz = 1.f / z;
      for (int i = 0; i < n; i++) s[i] *= iz;
    }
    std::fill(acc, acc + (size_t)nq * HD, 0.f);
    float vf[256];
    for (int i = 0; i < n; i++) {
      const float* vr;
      if (i < nc) {
        const uint16_t* vb = X.cv + (size_t)((X.lo + i) % X.cap) * KVD + (size_t)j * HD;
        for (int d = 0; d < HD; d++) vf[d] = bf16_to_f32(vb[d]);
        vr = vf;
      } else {
        vr = X.v + (size_t)(i - nc) * KVD + (size_t)j * HD;
      }
      for (int qi = 0; qi < nq; qi++) {
        const float32x4_t pw = vdupq_n_f32(sc[(size_t)qi * n + i]);
        float* a = acc + (size_t)qi * HD;
        for (int d = 0; d < HD; d += 4) vst1q_f32(a + d, vfmaq_f32(vld1q_f32(a + d), pw, vld1q_f32(vr + d)));
      }
    }
    for (int s = 0; s < X.BS; s++)
      for (int r = 0; r < rep; r++) {
        float* o = X.o + (size_t)s * X.NH * HD + (size_t)(j * rep + r) * HD;
        const float* a = acc + (size_t)(s * rep + r) * HD;
        for (int d = 0; d < HD; d++) o[d] = bfr(a[d]);
      }
  }
}
}  // namespace

void InkDraft::block(int p, const float* emb) {
  const int H = H_, QD = NH_ * HD_, KVD = NKV_ * HD_, B = BS_;
  x_.resize((size_t)B * H);
  resid_.resize((size_t)B * H);
  hn_.resize((size_t)B * H);
  q_.resize((size_t)B * QD);
  k_.resize((size_t)B * KVD);
  v_.resize((size_t)B * KVD);
  ao_.resize((size_t)B * QD);
  att_.resize((size_t)B * H);
  gg_.resize((size_t)B * I_);
  uu_.resize((size_t)B * I_);
  act_.resize((size_t)B * I_);
  hfin_.resize((size_t)B * H);
  for (int h = 0; h < H; h++) x_[h] = bfr(emb[h]);
  for (int s = 1; s < B; s++) std::memcpy(x_.data() + (size_t)s * H, mask_.data(), sizeof(float) * H);
  static thread_local std::vector<std::vector<float>> scs;
  scs.resize(pool_->size());
  auto packf = [&](const float* x, int n, int K, uint16_t* P) {  // fp32 (bf16 values) [n, K] -> packed
    xb_.resize((size_t)n * K);
    for (size_t i = 0; i < (size_t)n * K; i++) xb_[i] = f2bf(x[i]);
    pack_pairs(xb_.data(), n, K, P);
  };
  auto ptrs = [&](float* base, int ld) {
    std::vector<float*> v(B);
    for (int s = 0; s < B; s++) v[s] = base + (size_t)s * ld;
    return v;
  };
  std::vector<uint16_t> P1(packed_elems(B, H)), P2(packed_elems(B, std::max(QD, I_)));
  for (int li = 0; li < 3; li++) {
    const Layer& W = L_[li];
    if (li == 0) resid_ = x_;
    else
      for (size_t i = 0; i < resid_.size(); i++) resid_[i] = bfr(resid_[i] + x_[i]);
    for (int s = 0; s < B; s++) rms_bf(resid_.data() + (size_t)s * H, W.ln_in.data(), H, eps_, hn_.data() + (size_t)s * H);
    packf(hn_.data(), B, H, P1.data());
    {
      auto yq = ptrs(q_.data(), QD), yk = ptrs(k_.data(), KVD), yv = ptrs(v_.data(), KVD);
      GJob j[3] = {{W.q, QD, H, B, P1.data(), yq.data()}, {W.k, KVD, H, B, P1.data(), yk.data()}, {W.v, KVD, H, B, P1.data(), yv.data()}};
      run_gemm(*pool_, j, 3);
    }
    bfr_n(q_.data(), q_.size());
    bfr_n(k_.data(), k_.size());
    bfr_n(v_.data(), v_.size());
    for (int s = 0; s < B; s++) {
      float* q = q_.data() + (size_t)s * QD;
      float* k = k_.data() + (size_t)s * KVD;
      for (int h = 0; h < NH_; h++) rms_bf(q + (size_t)h * HD_, W.qn.data(), HD_, eps_, q + (size_t)h * HD_);
      for (int h = 0; h < NKV_; h++) rms_bf(k + (size_t)h * HD_, W.kn.data(), HD_, eps_, k + (size_t)h * HD_);
      rope_bf(q, NH_, HD_, p + 1 + s, theta_);
      rope_bf(k, NKV_, HD_, p + 1 + s, theta_);
    }
    {
      AttnB X;
      X.q = q_.data();
      X.k = k_.data();
      X.v = v_.data();
      X.ck = ck_[li];
      X.cv = cv_[li];
      X.cap = ccap_[li];
      X.lo = li == 1 ? 0 : std::max(0, (p + 1) - (WIN_ - 1));  // window_left = WIN - 1 from the block's first slot
      X.p = p;
      X.NH = NH_;
      X.NKV = NKV_;
      X.HD = HD_;
      X.BS = B;
      X.o = ao_.data();
      X.sc = &scs;
      pool_->run(phase_attn_b, &X);
    }
    packf(ao_.data(), B, QD, P2.data());
    {
      auto ya = ptrs(att_.data(), H);
      GJob j{W.o, H, QD, B, P2.data(), ya.data()};
      run_gemm(*pool_, &j, 1);
    }
    for (size_t i = 0; i < resid_.size(); i++) resid_[i] = bfr(resid_[i] + bfr(att_[i]));
    for (int s = 0; s < B; s++) rms_bf(resid_.data() + (size_t)s * H, W.ln_post.data(), H, eps_, hn_.data() + (size_t)s * H);
    packf(hn_.data(), B, H, P1.data());
    {
      auto yg = ptrs(gg_.data(), I_), yu = ptrs(uu_.data(), I_);
      GJob j[2] = {{W.g, I_, H, B, P1.data(), yg.data()}, {W.u, I_, H, B, P1.data(), yu.data()}};
      run_gemm(*pool_, j, 2);
    }
    for (size_t i = 0; i < act_.size(); i++) {
      const float g = bfr(gg_[i]), u = bfr(uu_[i]);
      act_[i] = bfr(g * (1.f / (1.f + std::exp(-g))) * u);
    }
    packf(act_.data(), B, I_, P2.data());
    {
      auto yx = ptrs(x_.data(), H);
      GJob j{W.d, H, I_, B, P2.data(), yx.data()};
      run_gemm(*pool_, &j, 1);
    }
    bfr_n(x_.data(), x_.size());
  }
  std::vector<float> t(H);
  for (int s = 0; s < B; s++) {
    for (int h = 0; h < H; h++) t[h] = bfr(resid_[(size_t)s * H + h] + x_[(size_t)s * H + h]);
    rms_bf(t.data(), norm_.data(), H, eps_, hfin_.data() + (size_t)s * H);
  }
}

// one step of the Markov RNN for slot j (previous token prev): state_ updated, latent [256] out (bf16 values)
void InkDraft::rnn_step(int prev, int j, float* latent) {
  std::vector<float> in((size_t)2 * MR_ + 2 * H_), pr(3 * MR_);
  uint16_t w1[256];
  if (pread(mtp_fd_, w1, MR_ * 2, (off_t)(w1_off_ + (uint64_t)prev * MR_ * 2)) != MR_ * 2) std::memset(w1, 0, sizeof w1);
  std::memcpy(in.data(), state_.data(), sizeof(float) * MR_);
  for (int i = 0; i < MR_; i++) in[MR_ + i] = bf16_to_f32(w1[i]);
  std::memcpy(in.data() + 2 * MR_, hfin_.data() + (size_t)j * H_, sizeof(float) * H_);
  std::memcpy(in.data() + 2 * MR_ + H_, f_rnn_.data(), sizeof(float) * H_);
  Weight W;
  W.kind = Weight::BF16;
  W.rows = 3 * MR_;
  W.cols = 2 * MR_ + 2 * H_;
  W.w = rnn_w_;
  Job jb{&W, nullptr, in.data(), pr.data()};
  run_rows(*pool_, &jb, 1, 32);
  for (int i = 0; i < 3 * MR_; i++) pr[i] = bfr(bfr(pr[i]) + rnn_b_[i]);
  for (int i = 0; i < MR_; i++) {
    const float g = 1.f / (1.f + std::exp(-pr[i]));
    state_[i] = bfr(g * state_[i] + (1.f - g) * std::tanh(pr[MR_ + i]));
    latent[i] = bfr(std::tanh(pr[2 * MR_ + i]));
  }
}

namespace {
struct PickCtx {  // phase 0: lg = softcap(a + b + c) (bf16-rounded first when exact; a / c may be null), its max
  float* lg;
  const float *a, *b, *c;
  float sc;
  bool exact;
  int n, nt;
  float mx[64];
  int am[64];
  double z[64];
  int phase;
};
void phase_pick(void* p, int tid, int nth) {
  PickCtx& C = *(PickCtx*)p;
  const int per = (C.n + nth - 1) / nth, a = tid * per, b = std::min(C.n, a + per);
  if (C.phase == 0) {
    float m = -INFINITY;
    int am = -1;
    const float sc = C.sc, isc = 1.f / C.sc;
    for (int i = a; i < b; i++) {
      float x = (C.a ? C.a[i] : 0.f) + C.b[i] + (C.c ? C.c[i] : 0.f);
      if (C.exact) x = bfr(x);
      const float v = sc * std::tanh(x * isc);
      C.lg[i] = v;
      if (v > m) m = v, am = i;
    }
    C.mx[tid] = m;
    C.am[tid] = am;
  } else {
    const float m = C.mx[0];
    double z = 0;
    for (int i = a; i < b; i++) z += std::exp(C.lg[i] - m);
    C.z[tid] = z;
  }
}
}  // namespace

float InkDraft::pick(float* lg, const float* a, const float* b, const float* c, int32_t* id) {
  PickCtx C;
  C.lg = lg;
  C.a = a;
  C.b = b;
  C.c = c;
  C.sc = o_.softcap;
  C.exact = o_.exact;
  C.n = VU_;
  C.nt = pool_->size();
  C.phase = 0;
  pool_->run(phase_pick, &C);
  float m = -INFINITY;
  int am = 0;
  for (int t = 0; t < C.nt; t++)
    if (C.am[t] >= 0 && C.mx[t] > m) m = C.mx[t], am = C.am[t];  // ties: the lowest index (chunks in order)
  C.mx[0] = m;
  C.phase = 1;
  pool_->run(phase_pick, &C);
  double z = 0;
  for (int t = 0; t < C.nt; t++) z += C.z[t];
  *id = am;
  return (float)(1.0 / z);
}

int InkDraft::draft(int32_t tok, const float* emb, int max, float thr, int32_t* out, float* prob, float* logits1) {
  if (nproj_ + npend_ == 0 || max <= 0) return 0;
  auto t0 = clk::now();
  max = std::min(max, BS_ - 1);
  while (npend_ > 0) project(std::min(npend_, 64));
  const int p = nproj_ - 1;
  // f_rnn = mix_norm(features(p), 3, markov dflare_norm)
  f_rnn_.resize(H_);
  {
    std::vector<float> m(H_);
    for (int h = 0; h < H_; h++) {
      float t = 0;
      for (int k = 0; k < 4; k++) t += last_[(size_t)k * H_ + h] * alpha_[12 + k];
      m[h] = t;
    }
    rms_bf(m.data(), mk_dfl_.data(), H_, eps_, f_rnn_.data());
  }
  auto t1 = clk::now();
  block(p, emb);
  auto t2 = clk::now();
  ms_block += ms_between(t1, t2);
  state_.assign(MR_, 0.f);
  int prev = tok, nd = 0, nb = 0;  // nb: drafts whose head(h_j) is computed
  std::vector<float> lat(MR_), z(H_), b(H_);
  for (int j = 1; j <= max; j++) {
    rnn_step(prev, j, lat.data());
    float* lg = lg_.data();
    if (o_.exact) {
      Job jo{nullptr, nullptr, lat.data(), b.data()};
      Weight Wo;
      Wo.kind = Weight::BF16;
      Wo.rows = H_;
      Wo.cols = MR_;
      Wo.w = ho_w_;
      jo.w = &Wo;
      run_rows(*pool_, &jo, 1, 64);
      const float* hj = hfin_.data() + (size_t)j * H_;
      for (int h = 0; h < H_; h++) z[h] = bfr(hj[h] + bfr(bfr(b[h]) + ho_b_[h]));
      Job jh{&U_, nullptr, z.data(), lg};
      run_rows(*pool_, &jh, 1, 128);
    } else {
      if (j > nb) {  // head(h_j): draft 1 alone, then every remaining one in one batch
        const int n = j == 1 ? 1 : max - 1;
        Weight hv = *o_.head;
        hv.rows = VU_;
        std::vector<const XQ*> xq(n);
        std::vector<const float*> xf(n);
        std::vector<float*> y(n);
        for (int i = 0; i < n; i++) {
          const float* hj = hfin_.data() + (size_t)(j + i) * H_;
          hx_[i].prep(hj, H_);
          xq[i] = &hx_[i].q;
          xf[i] = hj;
          y[i] = hl_.data() + (size_t)(j - 1 + i) * VU_;
        }
        const XQB* pb = nullptr;
        if (n >= 6 && o_.prec == Prec::I16) {
          hxb_.reserve(H_, n);
          for (int i = 0; i < n; i++) xqb_set_from(hxb_, i, hx_[i].q);
          xqb_pad(hxb_);
          pb = &hxb_;
        }
        JobN jn{&hv, n, xq.data(), xf.data(), y.data(), pb};
        run_rows_n(*pool_, &jn, 1, 64);
        nb = j - 1 + n;
        if (n > 1) n_head_batched++;
      }
      Job jf{&F_, nullptr, lat.data(), lg};
      run_rows(*pool_, &jf, 1, 512);
    }
    int32_t id;
    const float pr = o_.exact ? pick(lg, nullptr, lg, nullptr, &id) : pick(lg, hl_.data() + (size_t)(j - 1) * VU_, lg, fc_.data(), &id);
    if (j == 1 && logits1) std::memcpy(logits1, lg, sizeof(float) * VU_);
    if (pr < thr) break;
    out[nd] = id;
    prob[nd] = pr;
    nd++;
    prev = id;
  }
  ms_head += ms_since(t2);
  ms_total += ms_since(t0);
  n_calls++;
  n_drafts += nd;
  return nd;
}

void InkDraft::reset_stats() {
  n_calls = n_proj = n_drafts = n_head_batched = 0;
  ms_proj = ms_block = ms_head = ms_total = 0;
}

std::string InkDraft::stats_json() const {
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(3);
  auto per = [&](double v) { return v / std::max<long>(1, n_calls); };
  o << "{\"calls\":" << n_calls << ",\"drafts\":" << n_drafts << ",\"positions_projected\":" << n_proj << ",\"head_batches\":" << n_head_batched
    << ",\"ms_per_call\":" << per(ms_total) << ",\"ms_projection_per_call\":" << per(ms_proj) << ",\"ms_block_per_call\":" << per(ms_block)
    << ",\"ms_head_per_call\":" << per(ms_head) << ",\"weight_gb\":" << wbytes_ / 1e9 << ",\"state_mb\":" << state_bytes() / 1e6
    << ",\"exact\":" << (o_.exact ? "true" : "false") << "}";
  return o.str();
}

// The block's bf16 products on synthetic weights of the same shapes (3 layers: q k v | o | gate up | down, one fork-join
// each, as block() runs them), n inputs: ms per block and weight GB/s, median of reps. dmeng --draft-bench.
std::string ink_draft_bench(Pool& pool, int reps) {
  const int H = 2304, QD = 3072, KVD = 1024, I = 9216;
  struct Mx {
    int R, K;
    std::vector<uint16_t> w;
  };
  std::vector<Mx> m;
  uint32_t st = 12345;
  auto rnd = [&]() {
    st = st * 1664525u + 1013904223u;
    return f2bf(((int)(st >> 9) - (1 << 22)) * (1.f / (1 << 26)));
  };
  const int shp[7][2] = {{QD, H}, {KVD, H}, {KVD, H}, {H, QD}, {I, H}, {I, H}, {H, I}};
  size_t bytes = 0;
  for (int l = 0; l < 3; l++)
    for (auto& sh : shp) {
      Mx x{sh[0], sh[1], std::vector<uint16_t>((size_t)sh[0] * sh[1])};
      for (auto& v : x.w) v = rnd();
      bytes += x.w.size() * 2;
      m.push_back(std::move(x));
    }
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(2);
  o << "{\"weight_gb\":" << bytes / 1e9;
  for (int n : {1, 2, 4, 8}) {
    std::vector<uint16_t> X((size_t)n * I), P(packed_elems(n, I));
    for (auto& v : X) v = rnd();
    std::vector<float> Y((size_t)n * I);
    std::vector<float*> yp(n);
    for (int i = 0; i < n; i++) yp[i] = Y.data() + (size_t)i * I;
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
      auto t0 = clk::now();
      for (int l = 0; l < 3; l++) {
        Mx* L = &m[(size_t)l * 7];
        for (int g : {0, 3, 4, 6}) {  // fork-joins: (q k v) (o) (gate up) (down)
          GJob j[3];
          int nj = g == 0 ? 3 : g == 4 ? 2 : 1;
          pack_pairs(X.data(), n, L[g].K, P.data());
          for (int q = 0; q < nj; q++) j[q] = {L[g + q].w.data(), L[g + q].R, L[g + q].K, n, P.data(), yp.data()};
          run_gemm(pool, j, nj);
        }
      }
      t.push_back(ms_since(t0));
    }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2];
    o << ",\"n" << n << "_ms\":" << med << ",\"n" << n << "_gb_s\":" << bytes / 1e6 / med;
  }
  o << "}";
  return o.str();
}

}  // namespace dm
