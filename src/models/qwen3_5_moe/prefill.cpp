// Batched prefill for Qwen3.6-35B-A3B, layer-major: every layer runs over a whole span of positions (sub-chunks bound
// the scratch), so its weights are read once per sub-chunk and its experts once per span, each expert applied to all the
// positions that picked it. Every per-position computation is the decode path's (qwen3_5_moe.cpp): the same quantised
// activations, the same integer group dots (i8mm here, SDOT in decode: exact either way), the same float epilogues
// (kernels_batch.h), and the shared inline helpers of qwen3_5_moe.h for the rest. So a prefill of n positions leaves the
// same bits (logits, recurrent states, KV) as n single steps; tools/dmserve.cpp --check verifies it.
#include <arm_neon.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../../core/gpu_gemm.h"
#include "dmeng/kernels_batch.h"
#include "qwen3_5_moe.h"

namespace dm {
namespace qwen35 {
namespace {

inline double msd(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

// ------------------------------------------------------------------------------------------------ generic phases
template <class F>
struct ForCtx {
  const F* f;
  int n;
  alignas(128) std::atomic<int> next{0};
};
template <class F>
void for_phase(void* p, int tid, int nt) {
  auto& C = *(ForCtx<F>*)p;
  for (int i; (i = C.next.fetch_add(1, std::memory_order_relaxed)) < C.n;) (*C.f)(i);
}
template <class F>
void pfor(Pool& pool, int n, const F& f) {  // f(i) for i in [0, n), dynamic over the pool's threads
  if (n <= 0) return;
  ForCtx<F> C;
  C.f = &f;
  C.n = n;
  pool.run(for_phase<F>, &C);
}

struct GJob {
  const Weight* m;
  const XQB* x;
  int r0, r1, p1;
  float* Y;
  size_t ldy;
};
void run_gjobs(Pool& pool, const std::vector<GJob>& J) {
  pfor(pool, (int)J.size(), [&](int i) {
    const GJob& j = J[i];
    if (j.p1 <= 0) return;
    if (j.m->kind == Weight::AFF8)
      gemm_aff8_i16(j.m->c, (const _Float16*)j.m->s, (const _Float16*)j.m->b, j.m->cols, j.r0, j.r1, *j.x, 0, j.p1, j.Y + j.r0, j.ldy);
    else
      gemm_aff4_i16(j.m->c, (const _Float16*)j.m->s, (const _Float16*)j.m->b, j.m->cols, j.r0, j.r1, *j.x, 0, j.p1, j.Y + j.r0, j.ldy);
  });
}
void add_gemm(std::vector<GJob>& J, const Weight& m, const XQB& x, int n, float* Y, size_t ldy, int RB = 64) {
  for (int r = 0; r < m.rows; r += RB) J.push_back({&m, &x, r, std::min(m.rows, r + RB), n, Y, ldy});
}

// A set of products computed together, optionally split with the GPU: the GPU takes the first frac of each affine (8- or
// 4-bit) matrix's rows (whole 64-row blocks; the same exact arithmetic, src/core/gpu_gemm.h), the CPU the rest, at the
// same time. Every (row, position) output is computed once, by one side, with the same bits either way. Without the GPU
// (gpu null or frac 0) the jobs are exactly add_gemm's over the items in order. An item's GPU input may differ from the
// CPU's: gx read through gmap (an expert's positions gathered from the layer's MoE input on the GPU, while the CPU reads
// its own gathered copy; the same quantised values). If the GPU does not complete a set (gpu_gemm.h finish), the CPU
// computes the GPU's rows too.
struct GItem {
  const Weight* m;
  const XQB* x;  // the CPU's input, positions [0, n)
  float* Y;
  size_t ldy;
  int n;
  const XQB* gx = nullptr;  // the GPU's input (default x) ...
  const int* gmap = nullptr;  // ... at these positions of gx (n of them; default 0..n-1)
};
struct GpuAcct {
  double ms_wait = 0, ms_stage = 0;  // waiting for the GPU's rows after the CPU's; staging copies
  uint64_t sets = 0, failed = 0;
  double macs_gpu = 0, macs_cpu = 0;
};
void run_split(Pool& pool, GpuGemm* gpu, double frac, const std::vector<GItem>& items, std::vector<GJob>& J, GpuAcct* acct) {
  J.clear();
  const bool use = gpu && gpu->ok() && frac > 0;
  std::vector<std::pair<int, int>> tk;  // (ticket, item)
  std::vector<int> rgs(items.size(), 0);
  if (use) {
    auto ts = clk::now();
    gpu->begin();
    for (size_t i = 0; i < items.size(); i++) {
      const GItem& it = items[i];
      const Weight& m = *it.m;
      int rg = (m.kind == Weight::AFF8 || m.kind == Weight::AFF4) && it.n > 0 ? ((int)(m.rows * frac) / 64) * 64 : 0;
      if (rg > 0) {
        const int s = gpu->stage(it.gx ? *it.gx : *it.x);
        const int t = gpu->add(m.kind == Weight::AFF8 ? 8 : 4, m.c, (const _Float16*)m.s, (const _Float16*)m.b, m.cols, rg, it.n, s, it.gmap);
        if (t < 0) rg = 0;  // not inside a wrapped region: the CPU computes the whole matrix
        else tk.push_back({t, (int)i});
      }
      rgs[i] = rg;
    }
    const int nc = gpu->prepare();
    if (nc == 0 && !tk.empty()) {  // no GPU buffers: everything on the CPU
      tk.clear();
      std::fill(rgs.begin(), rgs.end(), 0);
    }
    if (!tk.empty()) {
      pfor(pool, nc, [&](int i) { gpu->copy(i); });
      gpu->submit();
    }
    if (acct) acct->ms_stage += msd(ts);
  }
  for (size_t i = 0; i < items.size(); i++) {
    const GItem& it = items[i];
    const Weight& m = *it.m;
    for (int r = rgs[i]; r < m.rows; r += 64) J.push_back({&m, it.x, r, std::min(m.rows, r + 64), it.n, it.Y, it.ldy});
    if (acct) {
      acct->macs_gpu += (double)rgs[i] * m.cols * it.n;
      acct->macs_cpu += (double)(m.rows - rgs[i]) * m.cols * it.n;
    }
  }
  run_gjobs(pool, J);
  if (tk.empty()) return;
  auto t = clk::now();
  const bool ok = gpu->finish();
  if (acct) {
    acct->ms_wait += msd(t);
    acct->sets++;
  }
  if (ok) {
    pfor(pool, (int)tk.size(), [&](int u) {
      const GItem& it = items[tk[u].second];
      const int rg = rgs[tk[u].second];
      const float* y = gpu->out(tk[u].first);
      for (int i = 0; i < it.n; i++) std::memcpy(it.Y + (size_t)i * it.ldy, y + (size_t)i * rg, (size_t)rg * 4);
    });
  } else {  // the GPU's rows on the CPU (the same bits)
    if (acct) acct->failed++;
    J.clear();
    for (auto& p : tk) {
      const GItem& it = items[p.second];
      for (int r = 0; r < rgs[p.second]; r += 64) J.push_back({it.m, it.x, r, r + 64, it.n, it.Y, it.ldy});
    }
    run_gjobs(pool, J);
  }
}

// copy position si of src into position di of dst (same K)
void xqb_copy(XQB& dst, int di, const XQB& src, int si) {
  const int NG = src.K / XB, KB = src.K / 8;
  std::memcpy(dst.d + (size_t)di * NG, src.d + (size_t)si * NG, NG * 4);
  std::memcpy(dst.gsum + (size_t)di * NG, src.gsum + (size_t)si * NG, NG * 4);
  const int8_t* sp[4] = {src.hn, src.ln, src.hp, src.lp};
  int8_t* dp[4] = {dst.hn, dst.ln, dst.hp, dst.lp};
  const size_t so = (size_t)(si >> 1) * KB * 16 + (si & 1) * 8, dof = (size_t)(di >> 1) * KB * 16 + (di & 1) * 8;
  for (int q = 0; q < 4; q++)
    for (int b = 0; b < KB; b++) std::memcpy(dp[q] + dof + (size_t)b * 16, sp[q] + so + (size_t)b * 16, 8);
}

struct PrefillCounters {
  double ms_embed = 0, ms_attn_proj = 0, ms_attn_core = 0, ms_attn_out = 0, ms_router = 0, ms_experts = 0, ms_wait = 0,
         ms_cache = 0, ms_head = 0, ms_names = 0, ms_total = 0, ms_conv = 0, ms_heads = 0, ms_full = 0;
  bool gpu = false;  // the GPU split was on (then the JSON also reports "gpu": the split's account)
  GpuAcct ga, ge;    // the split of the attention projections / of the experts (shared + routed)
  double gpu_frac = 0, gpu_frac_experts = 0;
  int gpu_kernel = -1;
  uint64_t positions = 0, chunks = 0, layer_units = 0, union_sum = 0, hits = 0, misses = 0, late = 0;
  uint64_t named = 0, named_used = 0, named_unused = 0, used_not_named = 0;  // early names (stream 1, 2) vs the layer's use
  uint64_t bytes_read = 0;
  void reset() { *this = PrefillCounters(); }
  std::string json() const;
};

}  // namespace

struct Prefill {
  Qwen35* M = nullptr;
  PrefillCounters k;
  // run positions ids[0..n_all) from the state; logits of the last one into M->lgts if want_logits. False: cancel was
  // set (checked after every layer); the state is then back at the span's start (fixed part restored, pos unchanged)
  bool forward_chunk(const int* ids, int n_all, bool want_logits);
  const std::atomic<bool>* cancel = nullptr;
  const PrefillProgress* progress = nullptr;  // (positions done incl. this span's layers so far, total, -)
  int prog_base = 0, prog_total = 0;
  std::vector<uint8_t> snap;                  // the fixed state at the span's start (for a cancel inside the span)
  std::vector<int> capture;  // positions relative to the span start, in (0, n], whose DeltaNet states go to sink
  LM::StateSink sink;
  // scratch (sized for the sub-chunk)
  int cap = 0;
  std::vector<float> X, Hn, Y, A, B, Cv, O, Z, BB, AA, Lg, Pw, Shw, Eo, G, U, Part, Qn, Cs, Sn;
  std::vector<int32_t> Pk;
  std::vector<int16_t> Rec;
  XQB qa, qo;
  std::vector<XQB> qe, qd;  // per expert of a layer: gathered MoE inputs, activations for the down projection
  std::vector<std::vector<int>> epos;  // per expert: (position * K + pick slot) of the sub-chunk, ascending
  std::vector<std::pair<int, int>> names_next;  // stream 1, 2: (slot, expert) of layer L+1 named at layer L
  void reserve(int n);
};

namespace {
std::string PrefillCounters::json() const {
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(3);
  s << "{\"positions\":" << positions << ",\"chunks\":" << chunks << ",\"ms\":" << ms_total << ",\"tok_s\":" << positions / std::max(1e-9, ms_total / 1e3)
    << ",\"ms_by_stage\":{\"embed\":" << ms_embed << ",\"attn_proj\":" << ms_attn_proj << ",\"attn_core\":" << ms_attn_core << ",\"attn_out\":" << ms_attn_out
    << ",\"router\":" << ms_router << ",\"names\":" << ms_names << ",\"experts\":" << ms_experts << ",\"wait\":" << ms_wait << ",\"cache\":" << ms_cache
    << ",\"head\":" << ms_head;
  s << ",\"of_core_conv\":" << ms_conv << ",\"of_core_heads\":" << ms_heads << ",\"of_core_full\":" << ms_full << "}";
  if (gpu) {  // per part: GPU share of the products' multiply-adds, time waiting for the GPU after the CPU's rows, staging
    auto one = [&](const char* name, double f, const GpuAcct& a) {
      s << "\"" << name << "\":{\"frac\":" << f << ",\"gmac_gpu\":" << a.macs_gpu / 1e9 << ",\"gmac_cpu\":" << a.macs_cpu / 1e9 << ",\"wait_ms\":" << a.ms_wait
        << ",\"stage_ms\":" << a.ms_stage << ",\"sets\":" << a.sets << ",\"failed\":" << a.failed << "}";
    };
    s << ",\"gpu\":{\"kernel\":\"" << (gpu_kernel == 1 ? "matrix" : "integer") << "\",";
    one("attn", gpu_frac, ga);
    s << ",";
    one("experts", gpu_frac_experts, ge);
    s << "}";
  }
  s
    << ",\"mean_union_per_layer\":" << (double)union_sum / std::max<uint64_t>(1, layer_units) << ",\"hits\":" << hits << ",\"late\":" << late << ",\"misses\":" << misses
    << ",\"named\":" << named << ",\"named_used\":" << named_used << ",\"named_unused\":" << named_unused << ",\"used_not_named\":" << used_not_named
    << ",\"mb_read\":" << bytes_read / MB << "}";
  return s.str();
}

}  // namespace

void Prefill::reserve(int n) {
  if (n <= cap) return;
  const QCfg& c = M->c;
  const int H = c.H, C = c.conv_dim(), ZD = c.NVH * c.DV, QD = c.NQ * c.HD;
  X.resize((size_t)n * H);
  Hn.resize((size_t)n * H);
  Y.resize((size_t)n * H);
  A.resize((size_t)n * std::max(C, 2 * QD));            // GDN qkv / full-attention q + gate
  B.resize((size_t)n * 2 * c.NKV * c.HD);                // full-attention k, v
  Cv.resize((size_t)n * C);
  O.resize((size_t)n * std::max(ZD, QD));
  Z.resize((size_t)n * ZD);
  BB.resize((size_t)n * c.NVH);
  AA.resize((size_t)n * c.NVH);
  Lg.resize((size_t)n * (c.NE + 1));
  Pk.resize((size_t)n * c.K);
  Pw.resize((size_t)n * c.K);
  Shw.resize(n);
  Eo.resize((size_t)n * (c.K + 1) * H);
  G.resize((size_t)n * (c.K + 1) * std::max(c.I, c.SI));
  U.resize((size_t)n * (c.K + 1) * std::max(c.I, c.SI));
  Qn.resize((size_t)n * QD);
  Cs.resize((size_t)n * 512);
  Sn.resize((size_t)n * 512);
  qe.resize(c.NE + 1);
  qd.resize(c.NE + 1);
  epos.resize(c.NE);
  cap = n;
}

bool Prefill::forward_chunk(const int* ids, int n_all, bool want_logits) {
  Qwen35& Mo = *M;
  const QCfg& c = Mo.c;
  Pool& pool = *Mo.core->pool;
  QState& Sst = *Mo.S;
  const PrefillOpts& o = Mo.po;
  const int H = c.H, NL = c.NL, NE = c.NE, K = c.K, C = c.conv_dim(), ZD = c.NVH * c.DV, HD = c.HD, QD = c.NQ * HD;
  const int base = Sst.pos, SUB = std::max(8, std::min(o.chunk, n_all));
  if (base + n_all > Sst.ctx) {
    fprintf(stderr, "context full (%d tokens)\n", Sst.ctx);
    abort();
  }
  reserve(SUB);
  if (X.size() < (size_t)n_all * H) X.resize((size_t)n_all * H);  // the residual stream of every position of the span
  if (cancel) {  // a cancel inside the span puts the state back here
    snap.resize(Mo.fixed_bytes(Sst));
    Mo.save_fixed(Sst, snap.data());
  }
  auto t_all = clk::now();
  // embeddings
  auto t0 = clk::now();
  stage_mark(ST_EMBED);
  pfor(pool, n_all, [&](int i) {
    Mo.rows.row(ids[i], X.data() + (size_t)i * H);
  });
  k.ms_embed += msd(t0);
  const int half = c.ROT / 2;
  static const int split_min = getenv("DMENG_SPLIT_MIN") ? atoi(getenv("DMENG_SPLIT_MIN")) : 2 * 256;
  constexpr int CHK = 256;
  std::vector<GJob> J;
  GpuGemm* const gpu = (o.gpu_frac > 0 || o.gpu_frac_experts > 0) ? Mo.gpu.get() : nullptr;  // the GPU's share, if on

  // Layer-major: each layer runs over the whole span (sub-chunks of SUB positions bound the scratch), so a layer's
  // experts are read once per span, not once per chunk.
  std::vector<std::pair<int, int>> named_here;
  // sub-chunk boundaries: every SUB positions, and at each capture position (a layer's state is copied there)
  std::vector<int> bnd;
  for (int p = SUB; p < n_all; p += SUB) bnd.push_back(p);
  for (int p : capture)
    if (p > 0 && p < n_all) bnd.push_back(p);
  bnd.push_back(n_all);
  std::sort(bnd.begin(), bnd.end());
  bnd.erase(std::unique(bnd.begin(), bnd.end()), bnd.end());
  auto captured = [&](int e) { return std::find(capture.begin(), capture.end(), e) != capture.end(); };
  Cache::ChunkClock chunk_clock(Mo.core->cache, Mo.core->tok_clock + n_all - 1);  // the span is one token to the policy
  for (int L = 0; L < NL; L++) {
   if (Mo.core->cache.budget_req.load(std::memory_order_relaxed) >= 0) Mo.core->cache.apply_budget();
   if (L > 0) {
     if (progress && *progress) (*progress)(prog_base + (int)((long)n_all * L / NL), prog_total, 0.0);
     if (cancel && cancel->load()) {  // stop within one layer: names for this layer released, the fixed state restored
       for (auto& se : names_next) Mo.core->cache.unpin_pf(L, se.first, se.second);
       names_next.clear();
       Mo.load_fixed(Sst, snap.data(), base);
       k.ms_total += msd(t_all);
       return false;
     }
   }
   for (size_t bi = 0, s0 = 0; bi < bnd.size(); s0 = bnd[bi++]) {
    const int n = bnd[bi] - (int)s0, pos0 = base + (int)s0;
    const long clk0 = Mo.core->tok_clock + (long)s0;
    float* Xb = X.data() + (size_t)s0 * H;
    const bool last_sub = bi + 1 == bnd.size();
    // rope tables per position (as attn_full builds them)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < half; j++) {
        double inv = std::pow(c.theta, -2.0 * j / c.ROT);
        float ang = (float)(pos0 + i) * (float)inv;
        Cs[(size_t)i * 512 + j] = std::cos(ang);
        Sn[(size_t)i * 512 + j] = std::sin(ang);
      }
    const Lw& W = Mo.lw[L];
    GpuGemm* const Ga = n >= o.gpu_min && o.gpu_frac > 0 ? gpu : nullptr;           // attention projections
    GpuGemm* const Ge = n >= o.gpu_min && o.gpu_frac_experts > 0 ? gpu : nullptr;   // shared + routed experts
    // ---- attention input
    auto ta = clk::now();
    stage_mark(ST_ATTN_PROJ);
    qa.reserve(H, n);
    pfor(pool, n, [&](int i) {
      rmsnorm(Xb + (size_t)i * H, W.ln_in, H, c.eps, Hn.data() + (size_t)i * H);
      xqb_set(qa, i, Hn.data() + (size_t)i * H);
    });
    xqb_pad(qa);
    if (!W.full) {
      // ---- Gated DeltaNet
      run_split(pool, Ga, o.gpu_frac, {{&W.qkv, &qa, A.data(), (size_t)C, n}, {&W.z, &qa, Z.data(), (size_t)ZD, n}}, J, &k.ga);
      pfor(pool, n, [&](int i) {
        const float* h = Hn.data() + (size_t)i * H;
        for (int r = 0; r < c.NVH; r++) BB[(size_t)i * c.NVH + r] = dot_bf16((const uint16_t*)W.wb.w + (size_t)r * H, h, H);
        for (int r = 0; r < c.NVH; r++) AA[(size_t)i * c.NVH + r] = dot_bf16((const uint16_t*)W.wa.w + (size_t)r * H, h, H);
      });
      auto tb = clk::now();
      stage_mark(ST_CONV);
      k.ms_attn_proj += std::chrono::duration<double, std::milli>(tb - ta).count();
      // conv: channel by channel, positions in order (phase_conv per position)
      const int CK = c.CK;
      float* st = Sst.ls[L].conv.data();
      const float* w = W.conv;
      const int CB = 256;
      pfor(pool, (C + CB - 1) / CB, [&](int blk) {
        for (int i = 0; i < n; i++)  // positions in order; each channel's own sequence is what phase_conv computes
          for (int ch = blk * CB; ch < std::min(C, blk * CB + CB); ch++)
            Cv[(size_t)i * C + ch] = conv_channel(st, w + (size_t)ch * CK, A[(size_t)i * C + ch], C, CK, ch);
      });
      k.ms_conv += msd(tb);
      auto th0 = clk::now();
      stage_mark(ST_ATTN_CORE);
      // heads: head by head, positions in order (the decode path's gdn::head per position)
      const gdn::Weights gw{W.conv, W.A_log, W.dt_bias, W.gnorm};
      pfor(pool, c.NVH, [&](int h) {
        float* S = Sst.ls[L].S.data() + (size_t)h * c.DV * c.DK;
        for (int i = 0; i < n; i++)
          gdn::head(Mo.gd, gw, S, Cv.data() + (size_t)i * C, Z.data() + (size_t)i * ZD + h * c.DV, BB[(size_t)i * c.NVH + h],
                    AA[(size_t)i * c.NVH + h], h, O.data() + (size_t)i * ZD + h * c.DV);
      });
      k.ms_heads += msd(th0);
      qo.reserve(ZD, n);
      pfor(pool, n, [&](int i) { xqb_set(qo, i, O.data() + (size_t)i * ZD); });
      xqb_pad(qo);
      auto tc = clk::now();
      stage_mark(ST_ATTN_OUT);
      k.ms_attn_core += std::chrono::duration<double, std::milli>(tc - tb).count();
      run_split(pool, Ga, o.gpu_frac, {{&W.o, &qo, Y.data(), (size_t)H, n}}, J, &k.ga);
      k.ms_attn_out += msd(tc);
      if (sink && captured(bnd[bi])) {
        const LState& ls = Sst.ls[L];
        sink(base + bnd[bi], Mo.fixed_off[L], ls.conv.data(), ls.conv.size() * 4);
        sink(base + bnd[bi], Mo.fixed_off[L] + ls.conv.size() * 4, ls.S.data(), ls.S.size() * 4);
      }
    } else {
      // ---- gated full attention
      const int KD = c.NKV * HD;
      float* Kp = B.data();
      float* Vp = B.data() + (size_t)n * KD;
      run_split(pool, Ga, o.gpu_frac, {{&W.q, &qa, A.data(), (size_t)2 * QD, n}, {&W.k, &qa, Kp, (size_t)KD, n}, {&W.v, &qa, Vp, (size_t)KD, n}}, J,
                &k.ga);
      auto tb = clk::now();
      stage_mark(ST_ATTN_CORE);
      k.ms_attn_proj += std::chrono::duration<double, std::milli>(tb - ta).count();
      LState& S = Sst.ls[L];
      // K (normed, rotated) and V into the cache at their positions (attn_full)
      pfor(pool, n, [&](int i) {
        const int pos = pos0 + i;
        const float* kp = Kp + (size_t)i * KD;
        const float* vp = Vp + (size_t)i * KD;
        for (int j = 0; j < c.NKV; j++) {
          float kk[1024], ss = 0;
          for (int t = 0; t < HD; t++) {
            kk[t] = kp[j * HD + t];
            ss += kk[t] * kk[t];
          }
          float r = 1.f / std::sqrt(ss / HD + c.eps);
          for (int t = 0; t < HD; t++) kk[t] = kk[t] * r * W.kn[t];
          {
            const float* cs = Cs.data() + (size_t)i * 512;
            const float* sn = Sn.data() + (size_t)i * 512;
            for (int t = 0; t < half; t++) {
              float a = kk[t], b = kk[t + half];
              kk[t] = a * cs[t] - b * sn[t];
              kk[t + half] = b * cs[t] + a * sn[t];
            }
          }
          _Float16* kd = S.K + ((size_t)j * Sst.ctx + pos) * HD;
          _Float16* vd = S.V + ((size_t)j * Sst.ctx + pos) * HD;
          for (int t = 0; t < HD; t++) {
            kd[t] = (_Float16)kk[t];
            vd[t] = (_Float16)vp[j * HD + t];
          }
        }
      });
      // queries: q_norm + rope, as both decode paths build them
      pfor(pool, n * c.NQ, [&](int jh) {
        const int i = jh / c.NQ, h = jh % c.NQ;
        const float* qs = A.data() + (size_t)i * 2 * QD + (size_t)h * 2 * HD;
        float ss = 0;
        for (int t = 0; t < HD; t++) ss += qs[t] * qs[t];
        float r = 1.f / std::sqrt(ss / HD + c.eps);
        float* q = Qn.data() + (size_t)i * QD + (size_t)h * HD;
        for (int t = 0; t < HD; t++) q[t] = qs[t] * r * W.qn[t];
        const float* cs = Cs.data() + (size_t)i * 512;
        const float* sn = Sn.data() + (size_t)i * 512;
        for (int t = 0; t < half; t++) {
          float a = q[t], b = q[t + half];
          q[t] = a * cs[t] - b * sn[t];
          q[t + half] = b * cs[t] + a * sn[t];
        }
      });
      const float scale = 1.f / std::sqrt((float)HD);
      const int rep = c.NQ / c.NKV;
      // short contexts (T <= split_min): one head per job, phase_full_heads
      int n_short = 0;
      while (n_short < n && pos0 + n_short + 1 <= split_min) n_short++;
      pfor(pool, n_short * c.NQ, [&](int jh) {
        const int i = jh / c.NQ, h = jh % c.NQ, T = pos0 + i + 1;
        static thread_local std::vector<float> sc;
        if ((int)sc.size() < T) sc.resize(T + 1024);
        _Float16 q16[1024];
        const float* q = Qn.data() + (size_t)i * QD + (size_t)h * HD;
        for (int t = 0; t < HD; t++) q16[t] = (_Float16)q[t];
        int kvh = h / rep;
        const _Float16* Kc = S.K + (size_t)kvh * Sst.ctx * HD;
        const _Float16* Vc = S.V + (size_t)kvh * Sst.ctx * HD;
        float mx = -INFINITY;
        for (int t = 0; t < T; t++) {
          const _Float16* kr = Kc + (size_t)t * HD;
          float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
          for (int u = 0; u < HD; u += 8) {
            float16x8_t kv = vld1q_f16((const float16_t*)kr + u), qv = vld1q_f16((const float16_t*)q16 + u);
            s0 = vfmlalq_low_f16(s0, kv, qv);
            s1 = vfmlalq_high_f16(s1, kv, qv);
          }
          sc[t] = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
          mx = std::max(mx, sc[t]);
        }
        double zs = 0;
        for (int t = 0; t < T; t++) {
          sc[t] = std::exp(sc[t] - mx);
          zs += sc[t];
        }
        float iz = (float)(1.0 / zs);
        float32x4_t acc[256];
        const int NA = HD / 4;
        for (int u = 0; u < NA; u++) acc[u] = vdupq_n_f32(0);
        for (int t = 0; t < T; t++) {
          const _Float16* vr = Vc + (size_t)t * HD;
          float32x4_t pv = vdupq_n_f32(sc[t] * iz);
          for (int u = 0; u < HD; u += 8) {
            float16x8_t vv = vld1q_f16((const float16_t*)vr + u);
            acc[u / 4] = vfmaq_f32(acc[u / 4], vcvt_f32_f16(vget_low_f16(vv)), pv);
            acc[u / 4 + 1] = vfmaq_f32(acc[u / 4 + 1], vcvt_high_f32_f16(vv), pv);
          }
        }
        float* o = O.data() + (size_t)i * QD + (size_t)h * HD;
        const float* gate = A.data() + (size_t)i * 2 * QD + (size_t)h * 2 * HD + HD;
        for (int u = 0; u < HD; u += 4) vst1q_f32(o + u, acc[u / 4]);
        for (int u = 0; u < HD; u++) o[u] *= sigm(gate[u]);
      });
      // long contexts: split-K per query (phase_split + the combination), queries in blocks to bound the partials
      const int QB = 32;
      for (int qb0 = n_short; qb0 < n; qb0 += QB) {
        const int qb1 = std::min(n, qb0 + QB);
        const int maxT = pos0 + qb1, maxch = (maxT + CHK - 1) / CHK, W2 = HD + 2;
        Part.resize((size_t)QB * maxch * c.NQ * W2);
        // one (query, chunk, kv head): phase_split's arithmetic for that query
        auto one = [&](int i, int ch, int kvh) {
          const int T = pos0 + i + 1;
          static thread_local std::vector<_Float16> q16v(8 * 256);
          static thread_local std::vector<float> scv(8 * 256);
          _Float16* q16 = q16v.data();
          float* sc = scv.data();
          const float* qn = Qn.data() + (size_t)i * QD;
          for (int g = 0; g < rep; g++)
            for (int u = 0; u < HD; u++) q16[(size_t)g * HD + u] = (_Float16)qn[(size_t)(kvh * rep + g) * HD + u];
          int t0 = ch * CHK, t1 = std::min(T, t0 + CHK), nn = t1 - t0;
          const _Float16* Kc = S.K + ((size_t)kvh * Sst.ctx + t0) * HD;
          const _Float16* Vc = S.V + ((size_t)kvh * Sst.ctx + t0) * HD;
          // scores: each (key, head) dot is phase_split's (two FMLAL accumulators over the dimensions in order, then
          // their sum); four heads run side by side so their chains overlap and the key row is loaded once for them
          for (int t = 0; t < nn; t++) {
            const float16_t* kr = (const float16_t*)(Kc + (size_t)t * HD);
            int g = 0;
            for (; g + 4 <= rep; g += 4) {
              const float16_t* q0 = (const float16_t*)q16 + (size_t)g * HD;
              float32x4_t a0 = vdupq_n_f32(0), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
              for (int u = 0; u < HD; u += 8) {
                float16x8_t kv = vld1q_f16(kr + u);
                float16x8_t x0 = vld1q_f16(q0 + u), x1 = vld1q_f16(q0 + HD + u), x2 = vld1q_f16(q0 + 2 * HD + u), x3 = vld1q_f16(q0 + 3 * HD + u);
                a0 = vfmlalq_low_f16(a0, kv, x0); b0 = vfmlalq_high_f16(b0, kv, x0);
                a1 = vfmlalq_low_f16(a1, kv, x1); b1 = vfmlalq_high_f16(b1, kv, x1);
                a2 = vfmlalq_low_f16(a2, kv, x2); b2 = vfmlalq_high_f16(b2, kv, x2);
                a3 = vfmlalq_low_f16(a3, kv, x3); b3 = vfmlalq_high_f16(b3, kv, x3);
              }
              sc[(size_t)(g + 0) * CHK + t] = vaddvq_f32(vaddq_f32(a0, b0)) * scale;
              sc[(size_t)(g + 1) * CHK + t] = vaddvq_f32(vaddq_f32(a1, b1)) * scale;
              sc[(size_t)(g + 2) * CHK + t] = vaddvq_f32(vaddq_f32(a2, b2)) * scale;
              sc[(size_t)(g + 3) * CHK + t] = vaddvq_f32(vaddq_f32(a3, b3)) * scale;
            }
            for (; g < rep; g++) {
              const float16_t* qq = (const float16_t*)q16 + (size_t)g * HD;
              float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
              for (int u = 0; u < HD; u += 8) {
                float16x8_t kv = vld1q_f16(kr + u), qv = vld1q_f16(qq + u);
                s0 = vfmlalq_low_f16(s0, kv, qv);
                s1 = vfmlalq_high_f16(s1, kv, qv);
              }
              sc[(size_t)g * CHK + t] = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
            }
          }
          float* part = Part.data() + (size_t)(i - qb0) * maxch * c.NQ * W2;
          for (int g = 0; g < rep; g++) {
            float* sg = sc + (size_t)g * CHK;
            float mx = -INFINITY;
            for (int t = 0; t < nn; t++) mx = std::max(mx, sg[t]);
            double z = 0;
            for (int t = 0; t < nn; t++) z += (sg[t] = std::exp(sg[t] - mx));
            float* out = part + ((size_t)ch * c.NQ + kvh * rep + g) * W2;
            out[0] = mx;
            out[1] = (float)z;
          }
          // values: each output element is the same fused multiply-add sequence over t as phase_split's (starting from
          // 0), kept in registers per 8 dimensions x 8 heads instead of in memory; the weights transposed to [t][head]
          if (rep == 8) {
            static thread_local std::vector<float> wtv(256 * 8);
            float* wt = wtv.data();
            for (int t = 0; t < nn; t++)
              for (int g = 0; g < 8; g++) wt[t * 8 + g] = sc[(size_t)g * CHK + t];
            for (int u = 0; u < HD; u += 8) {
              float32x4_t a0[8], a1[8];
              for (int g = 0; g < 8; g++) a0[g] = a1[g] = vdupq_n_f32(0);
              for (int t = 0; t < nn; t++) {
                float16x8_t vv = vld1q_f16((const float16_t*)(Vc + (size_t)t * HD) + u);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(vv)), hi = vcvt_high_f32_f16(vv);
                float32x4_t w0 = vld1q_f32(wt + t * 8), w1 = vld1q_f32(wt + t * 8 + 4);
                a0[0] = vfmaq_laneq_f32(a0[0], lo, w0, 0); a1[0] = vfmaq_laneq_f32(a1[0], hi, w0, 0);
                a0[1] = vfmaq_laneq_f32(a0[1], lo, w0, 1); a1[1] = vfmaq_laneq_f32(a1[1], hi, w0, 1);
                a0[2] = vfmaq_laneq_f32(a0[2], lo, w0, 2); a1[2] = vfmaq_laneq_f32(a1[2], hi, w0, 2);
                a0[3] = vfmaq_laneq_f32(a0[3], lo, w0, 3); a1[3] = vfmaq_laneq_f32(a1[3], hi, w0, 3);
                a0[4] = vfmaq_laneq_f32(a0[4], lo, w1, 0); a1[4] = vfmaq_laneq_f32(a1[4], hi, w1, 0);
                a0[5] = vfmaq_laneq_f32(a0[5], lo, w1, 1); a1[5] = vfmaq_laneq_f32(a1[5], hi, w1, 1);
                a0[6] = vfmaq_laneq_f32(a0[6], lo, w1, 2); a1[6] = vfmaq_laneq_f32(a1[6], hi, w1, 2);
                a0[7] = vfmaq_laneq_f32(a0[7], lo, w1, 3); a1[7] = vfmaq_laneq_f32(a1[7], hi, w1, 3);
              }
              for (int g = 0; g < 8; g++) {
                float* out = part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2;
                vst1q_f32(out + u, a0[g]);
                vst1q_f32(out + u + 4, a1[g]);
              }
            }
          } else {
            for (int g = 0; g < rep; g++) std::fill(part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2, part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2 + HD, 0.f);
            for (int t = 0; t < nn; t++) {
              const float16_t* vr = (const float16_t*)(Vc + (size_t)t * HD);
              for (int u = 0; u < HD; u += 8) {
                float16x8_t vv = vld1q_f16(vr + u);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(vv)), hi = vcvt_high_f32_f16(vv);
                for (int g = 0; g < rep; g++) {
                  float* out = part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2;
                  float w2 = sc[(size_t)g * CHK + t];
                  vst1q_f32(out + u, vfmaq_n_f32(vld1q_f32(out + u), lo, w2));
                  vst1q_f32(out + u + 4, vfmaq_n_f32(vld1q_f32(out + u + 4), hi, w2));
                }
              }
            }
          }
        };
        // jobs: (chunk, kv head, group of queries); a chunk's keys and values stay in cache for the whole group
        const int nq = qb1 - qb0, cj = maxch * c.NKV;
        const int groups = std::max(1, std::min(nq, (2 * pool.size() + cj - 1) / cj));
        pfor(pool, cj * groups, [&](int jb) {
          const int grp = jb / cj, ch = (jb % cj) / c.NKV, kvh = jb % c.NKV;
          const int i0 = qb0 + nq * grp / groups, i1 = qb0 + nq * (grp + 1) / groups;
          for (int i = i0; i < i1; i++)
            if (ch * CHK < pos0 + i + 1) one(i, ch, kvh);
        });
        pfor(pool, (qb1 - qb0) * c.NQ, [&](int jh) {
          const int i = qb0 + jh / c.NQ, h = jh % c.NQ, T = pos0 + i + 1, nch = (T + CHK - 1) / CHK;
          const float* part = Part.data() + (size_t)(i - qb0) * maxch * c.NQ * W2;
          float mx = -INFINITY;
          for (int ch = 0; ch < nch; ch++) mx = std::max(mx, part[((size_t)ch * c.NQ + h) * W2]);
          std::vector<double> acc(HD, 0.0);
          double z = 0;
          for (int ch = 0; ch < nch; ch++) {
            const float* pp = part + ((size_t)ch * c.NQ + h) * W2;
            double f = std::exp((double)pp[0] - mx);
            z += f * pp[1];
            for (int u = 0; u < HD; u++) acc[u] += f * pp[2 + u];
          }
          float* oh = O.data() + (size_t)i * QD + (size_t)h * HD;
          const float* gate = A.data() + (size_t)i * 2 * QD + (size_t)h * 2 * HD + HD;
          for (int u = 0; u < HD; u++) oh[u] = (float)(acc[u] / z) * sigm(gate[u]);
        });
      }
      k.ms_full += msd(tb);
      qo.reserve(QD, n);
      pfor(pool, n, [&](int i) { xqb_set(qo, i, O.data() + (size_t)i * QD); });
      xqb_pad(qo);
      auto tc = clk::now();
      stage_mark(ST_ATTN_OUT);
      k.ms_attn_core += std::chrono::duration<double, std::milli>(tc - tb).count();
      run_split(pool, Ga, o.gpu_frac, {{&W.o, &qo, Y.data(), (size_t)H, n}}, J, &k.ga);
      k.ms_attn_out += msd(tc);
    }
    // ---- residual, MoE input
    auto tr = clk::now();
    stage_mark(ST_MOE_ROUTER);
    pfor(pool, n, [&](int i) {
      float* x = Xb + (size_t)i * H;
      const float* y = Y.data() + (size_t)i * H;
      for (int j = 0; j < H; j++) x[j] += y[j];
      float* hm = Hn.data() + (size_t)i * H;
      rmsnorm(x, W.ln_post, H, c.eps, hm);
      xqb_set(qa, i, hm);
    });
    xqb_pad(qa);
    // ---- released router per position (Model::moe)
    pfor(pool, n, [&](int i) {
      const float* hm = Hn.data() + (size_t)i * H;
      float* lg = Lg.data() + (size_t)i * (NE + 1);
      for (int e = 0; e < NE; e++) lg[e] = dot_bf16((const uint16_t*)W.router.w + (size_t)e * H, hm, H);
      lg[NE] = dot_bf16((const uint16_t*)W.shgate.w, hm, H);
      float mx = -INFINITY;
      for (int e = 0; e < NE; e++) mx = std::max(mx, lg[e]);
      double zsum = 0;
      for (int e = 0; e < NE; e++) zsum += std::exp((double)(lg[e] - mx));
      static thread_local std::vector<int> idxv;
      idxv.resize(NE);
      int* idx = idxv.data();
      for (int e = 0; e < NE; e++) idx[e] = e;
      std::partial_sort(idx, idx + K, idx + NE, [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
      int32_t* pk = Pk.data() + (size_t)i * K;
      float* pwl = Pw.data() + (size_t)i * K;
      double s = 0;
      for (int j = 0; j < K; j++) {
        pk[j] = idx[j];
        pwl[j] = (float)(std::exp((double)(lg[idx[j]] - mx)) / zsum);
        s += pwl[j];
      }
      if (c.norm_topk)
        for (int j = 0; j < K; j++) pwl[j] = (float)(pwl[j] / s);
      Shw[i] = sigm(lg[NE]);
    });
    k.ms_router += msd(tr);
    if (Mo.core->pick_log)  // the released picks by position (early names for drafted positions copy them)
      for (int i = 0; i < n; i++) Mo.core->log_picks(pos0 + i, L, Pk.data() + (size_t)i * K);
    if (o.record) {
      if (Rec.size() < (size_t)n_all * NL * K) Rec.resize((size_t)n_all * NL * K);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < K; j++) Rec[((size_t)((int)s0 + i) * NL + L) * K + j] = (int16_t)Pk[(size_t)i * K + j];
    }
    // ---- the layer's experts: union, slots, early names for the next layer
    auto tcache = clk::now();
    stage_mark(ST_MOE_CACHE);
    Cache& Ca = Mo.core->cache;
    const long clock_last = clk0 + n - 1;
    Ca.now_step = clock_last * NL + L;
    for (int e = 0; e < NE; e++) epos[e].clear();
    for (int i = 0; i < n; i++)
      for (int j = 0; j < K; j++) epos[Pk[(size_t)i * K + j]].push_back(i * K + j);
    std::vector<int> uni;
    for (int e = 0; e < NE; e++)
      if (!epos[e].empty()) uni.push_back(e);
    k.layer_units++;
    k.union_sum += uni.size();
    // the names issued for this layer at the previous one: score them, then they are unpinned after use
    if (bi == 0) named_here.swap(names_next);
    if (bi == 0 && !named_here.empty()) {  // scored against the first sub-chunk's use
      std::vector<uint8_t> nm(NE, 0);
      for (auto& se : named_here) {
        const Slot& s = Ca.slot(L, se.first);
        if (s.layer == L && s.e >= 0) nm[s.e] = 1;
      }
      for (int e = 0; e < NE; e++) {
        bool used = !epos[e].empty();
        if (nm[e] && used) k.named_used++;
        if (nm[e] && !used) k.named_unused++;
        if (!nm[e] && used) k.used_not_named++;
      }
    }
    std::vector<int> slot(NE, -1);
    for (int e : uni) {
      int x = Ca.slot_of(L, e);
      if (x >= 0) {
        if (Ca.mem == Mem::FILE && Ca.ready(L, x)) Ca.file_hit_check(L, x);
        if (Ca.ready(L, x)) k.hits++;
        else k.late++;
        Ca.slot(L, x).pins++;
      } else {
        x = Ca.victim(L, 0);
        if (x < 0) x = Ca.steal(L);
        if (x < 0) {
          fprintf(stderr, "prefill: no expert slot free (budget too small for a chunk's union)\n");
          abort();
        }
        Ca.issue(L, x, e, true);
        Ca.slot(L, x).pins = 1;
        k.misses++;
        k.bytes_read += c.rec;
      }
      slot[e] = x;
    }
    // early names for layer L + 1 (issued behind this layer's own reads)
    if (bi == 0 && o.stream && L + 1 < NL) {
      auto tn = clk::now();
      stage_mark(ST_PREFETCH);
      std::vector<uint8_t> want(NE, 0);
      if (o.stream == 1) {
        std::fill(want.begin(), want.end(), 1);
      } else {
        const Lw& Wn = Mo.lw[L + 1];
        std::vector<int32_t> top((size_t)n * K);
        pfor(pool, n, [&](int i) {
          const float* hm = Hn.data() + (size_t)i * H;
          static thread_local std::vector<float> lg2v;
          static thread_local std::vector<int> idxv;
          lg2v.resize(NE);
          idxv.resize(NE);
          float* lg2 = lg2v.data();
          int* idx = idxv.data();
          for (int e = 0; e < NE; e++) lg2[e] = dot_bf16((const uint16_t*)Wn.router.w + (size_t)e * H, hm, H);
          for (int e = 0; e < NE; e++) idx[e] = e;
          std::partial_sort(idx, idx + K, idx + NE, [&](int a, int b) { return lg2[a] > lg2[b] || (lg2[a] == lg2[b] && a < b); });
          for (int j = 0; j < K; j++) top[(size_t)i * K + j] = idx[j];
        });
        for (int32_t e : top) want[e] = 1;
      }
      for (int e = 0; e < NE; e++)
        if (want[e]) {
          bool held = Ca.slot_of(L + 1, e) >= 0;
          const int sl = Ca.name(L + 1, e, clock_last * NL + L + 1);
          if (sl >= 0) {
            names_next.push_back({sl, e});
            k.named++;
            if (!held) k.bytes_read += c.rec;
          }
        }
      k.ms_names += msd(tn);
      stage_mark(ST_MOE_CACHE);
    }
    k.ms_cache += msd(tcache);
    // ---- experts in waves: whatever has landed, then the rest as it lands
    auto texp = clk::now();
    stage_mark(ST_MOE_EXPERTS);
    double wait_ms = 0;
    std::vector<uint8_t> done(NE, 0);
    bool shared_done = false;
    size_t left = uni.size();
    const int I = c.I, SI = c.SI, GI = std::max(I, SI);
    while (left > 0 || !shared_done) {
      std::vector<int> wave;
      for (int e : uni)
        if (!done[e] && Ca.ready(L, slot[e])) wave.push_back(e);
      if (wave.empty() && shared_done) {
        for (int e : uni)
          if (!done[e]) {
            auto tw = clk::now();
            Ca.wait(L, slot[e]);
            wait_ms += msd(tw);
            wave.push_back(e);
            break;
          }
        for (int e : uni)  // and whatever else landed meanwhile
          if (!done[e] && Ca.ready(L, slot[e]) && std::find(wave.begin(), wave.end(), e) == wave.end()) wave.push_back(e);
      }
      // assignment offsets of this wave (row a of G/U = one (position, pick) of one expert)
      std::vector<int> off(wave.size() + 1, 0);
      for (size_t w = 0; w < wave.size(); w++) off[w + 1] = off[w] + (int)epos[wave[w]].size();
      const int na = off[wave.size()];
      std::vector<Weight> mats(wave.size() * 3);
      const ExpertFormat& F = Mo.core->mi.formats[0];
      for (size_t w = 0; w < wave.size(); w++) F.mats(Ca.rec(L, slot[wave[w]]), mats[w * 3 + 0], mats[w * 3 + 1], mats[w * 3 + 2]);
      // gather each expert's positions
      pfor(pool, (int)wave.size(), [&](int w) {
        const int e = wave[w];
        XQB& q = qe[e];
        q.reserve(H, (int)epos[e].size());
        for (size_t t = 0; t < epos[e].size(); t++) xqb_copy(q, (int)t, qa, epos[e][t] / K);
        xqb_pad(q);
      });
      // gate and up (+ the shared expert's in the first wave); with the GPU on, its rows read each expert's positions
      // straight from the layer's MoE input qa (gmap: the position of each assignment)
      const bool do_shared = !shared_done;
      float* Gs = G.data() + (size_t)n * K * GI;  // the shared expert's rows sit after all routed assignments
      float* Us = U.data() + (size_t)n * K * GI;
      GpuGemm* const Gw = Ge && na + (do_shared ? n : 0) >= o.gpu_min_wave ? Ge : nullptr;  // this wave's GPU, if any
      std::vector<int> gmap;
      std::vector<GItem> gi;
      if (Gw) {
        gmap.resize(na);
        for (size_t w = 0; w < wave.size(); w++)
          for (size_t t = 0; t < epos[wave[w]].size(); t++) gmap[off[w] + t] = epos[wave[w]][t] / K;
      }
      for (size_t w = 0; w < wave.size(); w++) {
        const int e = wave[w], ne = (int)epos[e].size();
        const int* gm = Gw ? gmap.data() + off[w] : nullptr;
        gi.push_back({&mats[w * 3 + 0], &qe[e], G.data() + (size_t)off[w] * GI, (size_t)GI, ne, &qa, gm});
        gi.push_back({&mats[w * 3 + 1], &qe[e], U.data() + (size_t)off[w] * GI, (size_t)GI, ne, &qa, gm});
      }
      if (do_shared) {
        gi.push_back({&W.sg, &qa, Gs, (size_t)GI, n});
        gi.push_back({&W.su, &qa, Us, (size_t)GI, n});
      }
      run_split(pool, Gw, o.gpu_frac_experts, gi, J, &k.ge);
      // activations, quantised for the down projection
      pfor(pool, (int)wave.size() + (do_shared ? 1 : 0), [&](int w) {
        if (w == (int)wave.size()) {
          qd[NE].reserve(SI, n);
          std::vector<float> act(SI);
          for (int i = 0; i < n; i++) {
            const float* g = Gs + (size_t)i * GI;
            const float* u = Us + (size_t)i * GI;
            for (int j = 0; j < SI; j++) act[j] = silu(g[j]) * u[j];
            xqb_set(qd[NE], i, act.data());
          }
          xqb_pad(qd[NE]);
          return;
        }
        const int e = wave[w], ne = (int)epos[e].size();
        qd[e].reserve(I, ne);
        std::vector<float> act(I);
        for (int t = 0; t < ne; t++) {
          const float* g = G.data() + (size_t)(off[w] + t) * GI;
          const float* u = U.data() + (size_t)(off[w] + t) * GI;
          for (int j = 0; j < I; j++) act[j] = silu(g[j]) * u[j];
          xqb_set(qd[e], t, act.data());
        }
        xqb_pad(qd[e]);
      });
      // down projections into a staging area (rows of G reused: an assignment's H outputs), then into place
      std::vector<float> D((size_t)na * H);
      gi.clear();
      for (size_t w = 0; w < wave.size(); w++) {
        const int e = wave[w], ne = (int)epos[e].size();
        gi.push_back({&mats[w * 3 + 2], &qd[e], D.data() + (size_t)off[w] * H, (size_t)H, ne});
      }
      if (do_shared) gi.push_back({&W.sd, &qd[NE], Eo.data() + (size_t)K * H, (size_t)(K + 1) * H, n});
      run_split(pool, Gw, o.gpu_frac_experts, gi, J, &k.ge);
      pfor(pool, (int)wave.size(), [&](int w) {
        const int e = wave[w];
        for (size_t t = 0; t < epos[e].size(); t++) {
          const int a = epos[e][t];  // position * K + pick slot
          const int i = a / K, j = a % K;
          std::memcpy(Eo.data() + ((size_t)i * (K + 1) + j) * H, D.data() + (size_t)(off[w] + t) * H, (size_t)H * 4);
        }
      });
      for (int e : wave) {
        done[e] = 1;
        left--;
      }
      shared_done = true;
    }
    k.ms_wait += wait_ms;
    k.ms_experts += msd(texp) - wait_ms;
    // ---- fixed-order combination (Model::moe), residual
    auto tcmb = clk::now();
    stage_mark(ST_MOE_COMBINE);
    pfor(pool, n, [&](int i) {
      const float* pwl = Pw.data() + (size_t)i * K;
      const float* eout = Eo.data() + (size_t)i * (K + 1) * H;
      float* x = Xb + (size_t)i * H;
      float* yout = Y.data() + (size_t)i * H;
      const float shw = Shw[i];
      for (int r = 0; r < H; r++) {
        float acc = 0;
        for (int j = 0; j < K; j++) acc += pwl[j] * eout[(size_t)j * H + r];
        yout[r] = acc + shw * eout[(size_t)K * H + r];
      }
      for (int r = 0; r < H; r++) x[r] += yout[r];
    });
    k.ms_experts += msd(tcmb);
    // ---- the cache's bookkeeping, position by position as token-by-token prefill would have done it
    auto tcb = clk::now();
    stage_mark(ST_MOE_CACHE);
    for (int e : uni) {
      for (int a : epos[e]) {
        const long t = clk0 + a / K;
        Ca.use(L, slot[e], t, t * NL + L);
      }
      Ca.unpin(L, slot[e]);
    }
    if (last_sub) {
      for (auto& se : named_here) Ca.unpin_pf(L, se.first, se.second);
      named_here.clear();
    }
    k.ms_cache += msd(tcb);
   }
  }
  if (o.record) fwrite(Rec.data(), 2, (size_t)n_all * NL * K, o.record);
  // ---- head on the last position
  if (want_logits) {
    auto th = clk::now();
    stage_mark(ST_HEAD);
    rmsnorm(X.data() + (size_t)(n_all - 1) * H, Mo.fnorm, H, c.eps, Mo.h.data());
    Mo.xin.prep(Mo.h.data(), H);
    const int V = c.V, RB = 64;
    pfor(pool, (V + RB - 1) / RB, [&](int b) {
      int r0 = b * RB, r1 = std::min(V, r0 + RB);
      mv(Mo.head, r0, r1, &Mo.xin.q, nullptr, Mo.lgts.data() + r0);
    });
    k.ms_head += msd(th);
  }
  Sst.pos += n_all;
  Mo.core->tok_clock += n_all;
  k.positions += n_all;
  k.chunks++;
  stage_mark(ST_OTHER);
  k.ms_total += msd(t_all);
  if (progress && *progress) (*progress)(prog_base + n_all, prog_total, 0.0);
  return true;
}


// ------------------------------------------------------------------------------------------------ the model's entry
Qwen35::Qwen35() = default;
Qwen35::~Qwen35() = default;

bool Qwen35::set_gpu_frac(double f, bool experts, std::string& err) {
  if (!(f >= 0 && f < 1)) {
    err = experts ? "prefill_gpu_experts must be in [0, 1)" : "prefill_gpu_frac must be in [0, 1)";
    return false;
  }
  if (f > 0 && !gpu) {
#if defined(DM_METAL)
    if (!R.base || !core) {
      err = "prefill GPU split: load and bind the model first";
      return false;
    }
    std::unique_ptr<GpuGemm> g(new GpuGemm());
    if (!g->init()) {
      err = "prefill GPU split: GPU (Metal) init failed";
      return false;
    }
    // the weights where they are, no copies: the resident file's region and the expert slab(s) (locked or plain
    // memory; FILE and purgeable records are not wrapped, so those experts stay on the CPU)
    const bool res = g->wrap(R.base, R.size);
    bool slab = false;
    for (int p = 0; p < core->cache.npools; p++)
      if (core->cache.P[p].slab) slab = g->wrap(core->cache.P[p].slab, core->cache.P[p].slab_bytes) || slab;
    if (!res && !slab) {
      err = "prefill GPU split: neither the resident weights nor the expert slab could be wrapped";
      return false;
    }
    fprintf(stderr, "dmeng: prefill GPU split on (resident %s, expert slab %s; %.2f GB wrapped)\n", res ? "wrapped" : "not wrapped",
            slab ? "wrapped" : "not wrapped", g->wrapped_bytes() / 1e9);
    gpu = std::move(g);
#else
    err = "prefill GPU split: this build has no Metal (DM_METAL off)";
    return false;
#endif
  }
  (experts ? po.gpu_frac_experts : po.gpu_frac) = f;
  return true;
}

ModelMem Qwen35::mem(const State& s0) const {
  const QState& s = (const QState&)s0;
  ModelMem m;
  m.resident = R.size;
  for (const LState& st : s.ls) {
    m.kv_reserved += 2 * st.kv_bytes;
    m.recurrent += (st.conv.size() + st.S.size()) * sizeof(float);
  }
  if (P) {
    auto add = [&](const auto& v) { m.prefill_scratch += v.capacity() * sizeof(v[0]); };
    add(P->X); add(P->Hn); add(P->Y); add(P->A); add(P->B); add(P->Cv); add(P->O); add(P->Z); add(P->BB); add(P->AA); add(P->Lg);
    m.prefill_scratch += gpu ? gpu->buffer_bytes() : 0;  // the GPU split's staging and output buffers
    add(P->Pw); add(P->Shw); add(P->Eo); add(P->G); add(P->U); add(P->Part); add(P->Qn); add(P->Cs); add(P->Sn); add(P->Pk); add(P->Rec);
    m.prefill_xqb = P->qa.cap + P->qo.cap;
    for (const XQB& q : P->qe) m.prefill_xqb += q.cap;
    for (const XQB& q : P->qd) m.prefill_xqb += q.cap;
  }
  return m;
}

bool Qwen35::prefill(State& s0, const int32_t* ids, int n, bool want_logits, const std::vector<int>& caps, const LM::StateSink& sink,
                     PrefillStats* st, const std::atomic<bool>* cancel) {
  S = (QState*)&s0;
  greedy_ = -1;  // the logits (if wanted) come from the full head
  // one position per pass: the decode path (the batched kernels are I16, and take inputs in multiples of 256)
  const bool dims_ok = c.H % 256 == 0 && c.I % 256 == 0 && c.SI % 256 == 0 && (c.NVH * c.DV) % 256 == 0 && (c.NQ * c.HD) % 256 == 0;
  const bool tbt = po.chunk <= 1 || prec != Prec::I16 || !dims_ok;
  if (tbt && !po.batched_always) return Model::prefill(s0, ids, n, want_logits, caps, sink, st, cancel);
  if (!P) {
    P.reset(new Prefill());
    P->M = this;
  }
  P->k.reset();
  P->k.gpu = (po.gpu_frac > 0 || po.gpu_frac_experts > 0) && gpu;
  P->k.gpu_frac = po.gpu_frac;
  if (gpu) gpu->set_kernel(po.gpu_kernel);
  P->k.gpu_kernel = gpu ? gpu->kernel() : -1;
  P->k.gpu_frac_experts = po.gpu_frac_experts;
  // per-layer caps would refuse the next layer's early names (names never borrow past a cap): off for the pass
  Cache::CapsOff caps_off(core->cache);
  const uint64_t b0 = core->cache.st.bytes_urgent + core->cache.st.bytes_pf;
  int done = 0;
  bool ok = true;
  while (done < n) {
    int m = tbt ? 1 : std::min(std::max(1, po.span), n - done);
    if (!tbt && n - done - m > 0 && n - done - m < 64) m = n - done;  // no tiny last span
    P->capture.clear();
    for (int c : caps)
      if (c > done && c <= done + m) P->capture.push_back(c - done);
    P->sink = sink;
    const int mode = po.stream;
    if (mode == 3) po.stream = m >= po.whole_from ? 1 : 2;
    P->cancel = cancel;
    P->progress = st ? &st->progress : nullptr;
    P->prog_base = done;
    P->prog_total = n;
    const bool whole = P->forward_chunk((const int*)ids + done, m, done + m == n && want_logits);
    po.stream = mode;
    P->capture.clear();
    P->sink = nullptr;
    if (!whole) {
      ok = false;
      break;
    }
    done += m;
    if (cancel && cancel->load() && done < n) {
      ok = false;
      break;
    }
  }
  if (st) {
    st->positions = done;
    st->chunks = (int)P->k.chunks;
    st->ms = P->k.ms_total;
    st->ms_wait = P->k.ms_wait;
    st->mb_read = (core->cache.st.bytes_urgent + core->cache.st.bytes_pf - b0) / MB;
    st->json = P->k.json();
  }
  if (po.release) {  // the scratch back to the system (re-created by the next prefill)
    P.reset();
    if (gpu) gpu->trim();
  }
  return ok;
}

}  // namespace qwen35
}  // namespace dm
