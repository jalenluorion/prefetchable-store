// The screened output head (include/dmeng/screen_head.h).
//
// Exact logit of row r (I16 input, mv_aff8): the float evaluation of T = sum_g I_g s_g d_g + b_g gsum_g, with I_g =
// sum_i c_i x_i (an exact int32), c the 8-bit codes, x the group's integers (256 xh + xl), d_g its scale. The head is
// split into nibble planes (weights.h, AFF8S): c = 16 hi + lo, lo in [0, 15]. With the high nibbles known,
//   I_g = 16 H_g + 7.5 S_g + e_g,  |e_g| <= min(7.5 A_g, ||lo - 7.5||_2 ||x_g||_2)
// (H_g = sum_i hi_i x_i, S_g = sum_i x_i, A_g = sum_i |x_i|; the 2-norm of each row's and group's low nibbles stored
// rounded up). Summed with |s_g| d_g, this gives every row an interval around its midpoint C. Float rounding: both kernels'
// errors are below 2^-20 of U = sum_g |I_g s_g d_g| + |b_g gsum_g| <= 255 AL + B (AL = sum_g |s_g| d_g A_g, B = sum_g
// |b_g gsum_g|), the midpoints' terms below 248 AL; the slack used, 4e-5 (300 AL + B), is about 40 times all of it.
//
// A row is out of the top k when its upper end is below a lower bound on the k-th largest logit: the k-th largest lower
// end, or the k-th largest exact logit of any k rows (the k rows with the largest midpoints are computed exactly). The
// rows that stay are computed with the full head's row kernel.
#include "dmeng/screen_head.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dm {
namespace {

// integer dot of one 64-group's high nibbles (AFF8S plane layout: natural order) with the I16 planes
inline int32x4_t hdot(const uint8_t* c, const int8_t* xh, const int8_t* xl) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  const uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  const int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(v0, m)), q1 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
  const int8x16_t q2 = vreinterpretq_s8_u8(vandq_u8(v1, m)), q3 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
  int32x4_t a = vdupq_n_s32(0), b = vdupq_n_s32(0);
  a = vdotq_s32(a, q0, vld1q_s8(xh));
  b = vdotq_s32(b, q0, vld1q_s8(xl));
  a = vdotq_s32(a, q1, vld1q_s8(xh + 16));
  b = vdotq_s32(b, q1, vld1q_s8(xl + 16));
  a = vdotq_s32(a, q2, vld1q_s8(xh + 32));
  b = vdotq_s32(b, q2, vld1q_s8(xl + 32));
  a = vdotq_s32(a, q3, vld1q_s8(xh + 48));
  b = vdotq_s32(b, q3, vld1q_s8(xl + 48));
  return vaddq_s32(vshlq_n_s32(a, 8), b);
}

inline float32x4_t u8x4(const uint8_t* p) {  // four bytes as floats
  uint32_t w;
  std::memcpy(&w, p, 4);
  return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(w)))));
}

constexpr int NTMAX = 64;

// the k largest values seen (unsorted) and their rows
struct TopK {
  float v[ScreenHead::KMAX];
  int r[ScreenHead::KMAX];
  int n = 0, imin = 0;
  float vmin = INFINITY;
  void clear() { n = 0, vmin = INFINITY; }
  inline void push(float x, int row, int k) {
    if (n < k) {
      v[n] = x;
      r[n++] = row;
      if (n == k) rescan(k);
    } else if (x > vmin) {
      v[imin] = x;
      r[imin] = row;
      rescan(k);
    }
  }
  void rescan(int k) {
    imin = 0;
    for (int i = 1; i < k; i++)
      if (v[i] < v[imin]) imin = i;
    vmin = v[imin];
  }
};

// per token: the input's group sums and norms, scaled as the levels use them
struct Tok {
  float s75[64], a75[64], q2[64], aa[64];
};

struct LevelCtx {
  ScreenHead* M;
  const XQ* x;
  const Tok* t;
  int k, njobs;
  TopK lo[NTMAX], mid[NTMAX];  // per thread: the k largest lower ends, the k largest midpoints
  alignas(128) std::atomic<int> next{0};
};

void phase_coarse(void* p, int tid, int) {
  LevelCtx& C = *(LevelCtx*)p;
  ScreenHead& M = *C.M;
  const Weight& w = *M.head;
  const XQ& x = *C.x;
  const Tok& T = *C.t;
  const int NG = w.cols / XB, k = C.k;
  TopK& tlo = C.lo[tid];
  TopK& tmid = C.mid[tid];
  tlo.clear();
  tmid.clear();
  const float32x4_t quarter = vdupq_n_f32(0.25f);
  for (int b; (b = C.next.fetch_add(1, std::memory_order_relaxed)) < C.njobs;) {
    const int r0 = b * 64, r1 = std::min(M.V, r0 + 64);
    for (int r = r0; r < r1; r++) {
      const uint8_t* ch = aff8s_hi(w, r);
      const uint8_t* e1 = M.e1.data() + (size_t)r * NG;
      const _Float16* s = (const _Float16*)w.s + (size_t)r * NG;
      const _Float16* bi = (const _Float16*)w.b + (size_t)r * NG;
      float32x4_t cacc = vdupq_n_f32(0), racc = cacc, aacc = cacc, bacc = cacc;
      int g = 0;
      for (; g + 4 <= NG; g += 4) {
        const int32x4_t a0 = hdot(ch + 32 * g, x.hn + XB * g, x.ln + XB * g);
        const int32x4_t a1 = hdot(ch + 32 * (g + 1), x.hn + XB * (g + 1), x.ln + XB * (g + 1));
        const int32x4_t a2 = hdot(ch + 32 * (g + 2), x.hn + XB * (g + 2), x.ln + XB * (g + 2));
        const int32x4_t a3 = hdot(ch + 32 * (g + 3), x.hn + XB * (g + 3), x.ln + XB * (g + 3));
        const float32x4_t pf = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)));
        const float32x4_t mid = vfmaq_n_f32(vld1q_f32(T.s75 + g), pf, 16.f);
        const float32x4_t sd = vmulq_f32(vcvt_f32_f16(vld1_f16((const float16_t*)s + g)), vld1q_f32(x.d + g));
        const float32x4_t bv = vcvt_f32_f16(vld1_f16((const float16_t*)bi + g)), gs = vld1q_f32(x.gsum + g);
        const float32x4_t asd = vabsq_f32(sd);
        cacc = vfmaq_f32(cacc, mid, sd);
        cacc = vfmaq_f32(cacc, bv, gs);
        racc = vfmaq_f32(racc, asd, vminq_f32(vld1q_f32(T.a75 + g), vmulq_f32(vmulq_f32(u8x4(e1 + g), quarter), vld1q_f32(T.q2 + g))));
        aacc = vfmaq_f32(aacc, asd, vld1q_f32(T.aa + g));
        bacc = vfmaq_f32(bacc, vabsq_f32(bv), vabsq_f32(gs));
      }
      float c = vaddvq_f32(cacc), R = vaddvq_f32(racc), AL = vaddvq_f32(aacc), B = vaddvq_f32(bacc);
      for (; g < NG; g++) {  // groups beyond a multiple of 4
        const float mid = 16.f * (float)vaddvq_s32(hdot(ch + 32 * g, x.hn + XB * g, x.ln + XB * g)) + T.s75[g];
        const float sd = (float)s[g] * x.d[g], bv = (float)bi[g];
        c += mid * sd + bv * x.gsum[g];
        R += std::fabs(sd) * std::min(T.a75[g], 0.25f * (float)e1[g] * T.q2[g]);
        AL += std::fabs(sd) * T.aa[g];
        B += std::fabs(bv) * std::fabs(x.gsum[g]);
      }
      const float sl = 4e-5f * (300.f * AL + B) + 1e-30f, rad = R * (1.f + 1e-5f) + sl;
      M.hi[r] = c + rad;
      if (r >= M.VO) continue;
      tlo.push(c - rad, r, k);
      tmid.push(c, r, k);
    }
  }
}

struct ExactCtx {
  ScreenHead* M;
  const XQ* x;
  float* lg;
  float lb, cap;
  int nch, k;
  static constexpr int CH = 1024;
  float bv[NTMAX], hmax[NTMAX];  // per thread: the best stored value; the largest upper end of a row left out
  int bi[NTMAX], nc[NTMAX];
  TopK top[NTMAX];  // per thread (cap > 0): the k largest stored values
  alignas(128) std::atomic<int> next{0};
};

void phase_exact(void* p, int tid, int) {
  ExactCtx& E = *(ExactCtx*)p;
  ScreenHead& M = *E.M;
  const Weight& w = *M.head;
  float best = -INFINITY, hmax = -INFINITY;
  int bidx = -1, nc = 0;
  TopK& top = E.top[tid];
  top.clear();
  for (int ch; (ch = E.next.fetch_add(1, std::memory_order_relaxed)) < E.nch;) {
    const int r0 = ch * ExactCtx::CH, r1 = std::min(M.VO, r0 + ExactCtx::CH);
    for (int r = r0; r < r1; r++) {
      if (M.hi[r] >= E.lb) {
        float v;
        mv_aff8s_i16(w.c, w.srows, (const _Float16*)w.s, (const _Float16*)w.b, w.cols, r, r + 1, *E.x, &v);  // the full head's row
        if (E.cap > 0) {
          v = softcap(v, E.cap);
          top.push(v, r, E.k);
        }
        E.lg[r] = v;
        nc++;
        if (v > best || (v == best && (bidx < 0 || r < bidx))) {
          best = v;
          bidx = r;
        }
      } else {
        E.lg[r] = -INFINITY;
        hmax = std::max(hmax, M.hi[r]);
      }
    }
  }
  E.bv[tid] = best;
  E.bi[tid] = bidx;
  E.nc[tid] = nc;
  E.hmax[tid] = hmax;
}

// a lower bound on the k-th largest logit from a level's results: the k-th largest lower end, or the smallest exact logit
// of the k rows with the largest midpoints, whichever is higher
float floor_of(const ScreenHead& M, const LevelCtx& C, int nt, int k, const XQ& x) {
  float all[NTMAX * ScreenHead::KMAX];
  int n = 0;
  for (int t = 0; t < nt; t++)
    for (int i = 0; i < C.lo[t].n; i++) all[n++] = C.lo[t].v[i];
  if (n < k) return -INFINITY;
  std::nth_element(all, all + (k - 1), all + n, [](float a, float b) { return a > b; });
  const float lb = all[k - 1];
  std::pair<float, int> mids[NTMAX * ScreenHead::KMAX];
  int m = 0;
  for (int t = 0; t < nt; t++)
    for (int i = 0; i < C.mid[t].n; i++) mids[m++] = {C.mid[t].v[i], C.mid[t].r[i]};
  std::nth_element(mids, mids + (k - 1), mids + m, [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
  float ek = INFINITY;
  const Weight& w = *M.head;
  for (int i = 0; i < k; i++) {
    float v;
    mv_aff8s_i16(w.c, w.srows, (const _Float16*)w.s, (const _Float16*)w.b, w.cols, mids[i].second, mids[i].second + 1, x, &v);
    ek = std::min(ek, v);
  }
  return std::max(lb, ek);
}

}  // namespace

bool ScreenHead::build(Pool& pool, Weight& w, int vrows, int vout, std::string& err) {
  if (w.kind != Weight::AFF8 || w.cols % XB || w.cols / XB > 64 || pool.size() > NTMAX) {
    err = "screened head: needs an 8-bit head with whole 64-groups (at most 4096 inputs)";
    return false;
  }
  head = &w;
  V = vrows;
  VO = vout;
  hi.assign(V, 0.f);
  // a test input (fixed, with a few large entries) and the full head's logits on the original bytes
  const int H = w.cols, NG = H / XB;
  std::vector<float> xt(H);
  uint64_t st = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < H; i++) {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    xt[i] = ((int)(st >> 40) % 2001 - 1000) / 400.f * ((i % 97) == 0 ? 20.f : 1.f);
  }
  XBuf xb;
  xb.init(H, Prec::I16);
  xb.prep(xt.data(), H);
  std::vector<float> ref(V), got(V);
  {
    Job j = {&w, &xb.q, nullptr, ref.data()};
    run_rows(pool, &j, 1, 64);
  }
  // per row and group: 4 ||lo - 7.5||_2 of the codes' low nibbles, rounded up (at most 240)
  e1.assign((size_t)V * NG, 0);
  parallel_for(pool, (V + 255) / 256, [&](int ch) {
    for (int r = ch * 256; r < std::min(V, ch * 256 + 256); r++) {
      const uint8_t* c = w.c + (size_t)r * H;
      for (int g = 0; g < NG; g++) {
        double q = 0;
        for (int i = 0; i < XB; i++) {
          const double a = (c[g * XB + i] & 15) - 7.5;
          q += a * a;
        }
        e1[(size_t)r * NG + g] = (uint8_t)std::min(255.0, std::ceil(std::sqrt(q) * 4.0 * (1.0 + 1e-9)));
      }
    }
  });
  if (!aff8s_split(w, pool, err)) return false;
  auto same = [&](const float* a, const char* what) {
    if (std::memcmp(a, ref.data(), sizeof(float) * V) != 0) {
      err = std::string("screened head: the split layout's ") + what + " differs from the 8-bit head";
      return false;
    }
    return true;
  };
  {  // the one-position kernel
    Job j = {&w, &xb.q, nullptr, got.data()};
    run_rows(pool, &j, 1, 64);
    if (!same(got.data(), "matrix-vector kernel")) return false;
  }
  {  // several positions: the multi-vector path (3) and the packed gemm (6)
    std::vector<std::vector<float>> outs(6, std::vector<float>(V));
    const XQ* xs[6];
    float* os[6];
    for (int i = 0; i < 6; i++) xs[i] = &xb.q, os[i] = outs[i].data();
    JobN j3 = {&w, 3, xs, nullptr, os};
    run_rows_n(pool, &j3, 1, 64);
    for (int i = 0; i < 3; i++)
      if (!same(outs[i].data(), "multi-vector kernel")) return false;
    XQB pb;
    pb.reserve(H, 6);
    for (int i = 0; i < 6; i++) xqb_set_from(pb, i, xb.q);
    pb.N = 6;
    xqb_pad(pb);
    for (auto& o : outs) std::fill(o.begin(), o.end(), 0.f);
    JobN j6 = {&w, 6, xs, nullptr, os, &pb};
    run_rows_n(pool, &j6, 1, 64);
    for (int i = 0; i < 6; i++)
      if (!same(outs[i].data(), "packed gemm")) return false;
  }
  for (int k : {1, 20}) {  // the screen itself: every top-k row's exact logit, the same argmax
    const int b = run(pool, xb.q, k, got.data());
    std::vector<int> idx(VO);
    for (int i = 0; i < VO; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int c) { return ref[a] > ref[c] || (ref[a] == ref[c] && a < c); });
    if (b != idx[0]) {
      err = "screened head: argmax differs on the test input";
      return false;
    }
    for (int j = 0; j < k; j++)
      if (std::memcmp(&got[idx[j]], &ref[idx[j]], 4) != 0) {
        err = "screened head: a top-k logit differs on the test input";
        return false;
      }
  }
  reset_stats();
  return true;
}

int ScreenHead::run(Pool& pool, const XQ& x, int k, float* lg, float cap) {
  auto t0 = clk::now();
  k = std::max(1, std::min(k, std::min(KMAX, VO)));
  const int NG = head->cols / XB, nt = pool.size();
  Tok T;
  for (int g = 0; g < NG; g++) {
    int64_t S = 0, A = 0;
    double Q = 0;
    for (int i = 0; i < XB; i++) {
      const int v = 256 * (int)x.hn[g * XB + i] + (int)x.ln[g * XB + i];
      S += v;
      A += v < 0 ? -v : v;
      Q += (double)v * v;
    }
    T.s75[g] = 7.5f * (float)S;
    T.a75[g] = 7.5f * (float)A;
    T.aa[g] = (float)A;
    T.q2[g] = (float)(std::sqrt(Q) * (1.0 + 1e-6));
  }
  LevelCtx C1;
  C1.M = this;
  C1.x = &x;
  C1.t = &T;
  C1.k = k;
  C1.njobs = (V + 63) / 64;
  pool.run(phase_coarse, &C1);
  const float lb2 = floor_of(*this, C1, nt, k, x);
  auto t1 = clk::now();
  ExactCtx E;
  E.M = this;
  E.x = &x;
  E.lg = lg;
  E.lb = lb2;
  E.cap = cap;
  E.k = k;
  E.nch = (VO + ExactCtx::CH - 1) / ExactCtx::CH;
  pool.run(phase_exact, &E);
  float best = -INFINITY, hmax = -INFINITY;
  int b = -1, nc = 0;
  ok = true;
  if (cap > 0) {  // capped values can tie where the raw ones did not: every row left out must stay below the k-th kept one
    float all[NTMAX * KMAX];
    int n = 0;
    for (int t = 0; t < nt; t++) {
      hmax = std::max(hmax, E.hmax[t]);
      for (int i = 0; i < E.top[t].n; i++) all[n++] = E.top[t].v[i];
    }
    if (n < k) {
      ok = false;
    } else if (hmax > -INFINITY) {
      std::nth_element(all, all + (k - 1), all + n, [](float a, float c) { return a > c; });
      ok = softcap(hmax, cap) < all[k - 1];
    }
    if (!ok) n_fallback += 1;
  }
  for (int t = 0; t < nt; t++) {
    nc += E.nc[t];
    if (E.bi[t] >= 0 && (b < 0 || E.bv[t] > best || (E.bv[t] == best && E.bi[t] < b))) {
      best = E.bv[t];
      b = E.bi[t];
    }
  }
  n_cand += nc;
  n_calls += 1;
  ms_coarse += std::chrono::duration<double, std::milli>(t1 - t0).count();
  ms_exact += ms_since(t1);
  return b;
}

}  // namespace dm
