// The exact greedy head (moved from the Qwen3.6 model so every family has it): the argmax of an 8-bit head's logits
// without computing every row. A 4-bit copy derived at load (MLX 4-bit of the same codes' top nibbles) gives coarse logits
// of every row; a per-row bound |exact - coarse| (the copy's error norms times the input's, the bias difference, float
// slack) leaves the rows that can still be the maximum; those are computed exactly with the full head's own row kernel and
// the argmax taken among them (lowest index on a tie, as a full argmax). Bit-identical to argmax(full head) by construction.
#include "dmeng/greedy_head.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace dm {
namespace {

struct H4Ctx {
  GreedyHead* M;
  int V, H, G;
  alignas(128) std::atomic<int> next{0};
};
void phase_build_h4(void* p, int, int) {
  H4Ctx& C = *(H4Ctx*)p;
  GreedyHead& M = *C.M;
  const uint8_t* q8 = M.head8->c;
  const _Float16* s8 = (const _Float16*)M.head8->s;
  const _Float16* b8 = (const _Float16*)M.head8->b;
  constexpr int CH = 256;
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) * CH < C.V;) {
    const int r0 = ch * CH, r1 = std::min(C.V, r0 + CH);
    for (int r = r0; r < r1; r++) {
      double dn = 0, bm = 0;
      for (int g = 0; g < C.G; g++) {
        double dg = 0;
        const size_t gi = (size_t)r * C.G + g;
        const float s = (float)s8[gi], b = (float)b8[gi];
        const _Float16 s4 = (_Float16)(16.f * s), b4 = (_Float16)(b + 8.f * s);
        M.h4s[gi] = s4;
        M.h4b[gi] = b4;
        bm = std::max(bm, (double)std::fabs(b - (float)b4));
        const uint8_t* q = q8 + (size_t)r * C.H + g * XB;
        uint8_t* o = M.h4c.data() + (size_t)r * (C.H / 2) + g * (XB / 2);
        for (int i = 0; i < XB / 2; i++) {
          const int a = q[2 * i] >> 4, bb = q[2 * i + 1] >> 4;
          o[i] = (uint8_t)(a | (bb << 4));  // MLX 4-bit packing: element 2i low nibble, 2i + 1 high nibble
        }
        for (int i = 0; i < XB; i++) {
          const double w8 = (double)s * q[i] + b, w4 = (double)(float)s4 * (q[i] >> 4) + (double)(float)b4;
          dn += (w8 - w4) * (w8 - w4);
          dg += (w8 - w4) * (w8 - w4);
        }
        _Float16 hh = (_Float16)(std::sqrt(dg) * (1.0 + 1e-3));
        if ((double)(float)hh < std::sqrt(dg)) hh = (_Float16)((float)hh * 1.001f + 1e-7f);  // never below the true norm
        M.h_gnorm[gi] = hh;
      }
      M.h_dnorm[r] = (float)(std::sqrt(dn) * (1.0 + 1e-6));
      M.h_bmax[r] = (float)bm;
    }
  }
}

}  // namespace

bool GreedyHead::build(Pool& pool, const Weight& head, int vrows, int vout, Prec prec, std::string& err) {
  if (prec != Prec::I16 || head.kind != Weight::AFF8 || head.cols % XB) {
    err = "exact greedy head: needs I16 activations and an 8-bit head";
    return false;
  }
  head8 = &head;
  V = vrows;
  VO = vout;
  const int H = head.cols, G = H / XB;
  h4c.assign((size_t)V * H / 2, 0);
  h4s.resize((size_t)V * G);
  h4b.resize((size_t)V * G);
  h_dnorm.assign(V, 0.f);
  h_bmax.assign(V, 0.f);
  h_gnorm.assign((size_t)V * G, (_Float16)0.f);
  coarse.assign(V, 0.f);
  cand.reserve(V);
  H4Ctx C;
  C.M = this;
  C.V = V;
  C.H = H;
  C.G = G;
  pool.run(phase_build_h4, &C);
  head4.kind = Weight::AFF4;
  head4.rows = V;
  head4.cols = H;
  head4.c = h4c.data();
  head4.s = h4s.data();
  head4.b = h4b.data();
  return true;
}

namespace {
struct CandCtx {
  GreedyHead* M;
  int n;
  alignas(128) std::atomic<int> next{0};
};
void phase_cand(void* p, int, int) {
  CandCtx& C = *(CandCtx*)p;
  GreedyHead& M = *C.M;
  constexpr int CH = 16;
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) * CH < C.n;) {
    for (int i = ch * CH; i < std::min(C.n, ch * CH + CH); i++) {
      const int r = M.cand[i];
      mv(*M.head8, r, r + 1, M.xq, nullptr, M.out + r);  // the full head's own row computation
    }
  }
}

// per row: B_r = sum_g ||w8 - w4||_g ||xhat_g|| + max_g|b8 - b4| * 32 sum_g d_g + float slack; and the best lower bound
struct BoundCtx {
  static constexpr int CH = 2048;
  GreedyHead* M;
  int G, nch;
  const float* xg;
  float bterm;
  std::vector<float> lo, bound;
  alignas(128) std::atomic<int> next{0};
};
void phase_bound(void* p, int, int) {
  BoundCtx& C = *(BoundCtx*)p;
  GreedyHead& M = *C.M;
  const int V = M.V;
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < C.nch;) {
    const int r0 = ch * BoundCtx::CH, r1 = std::min(V, r0 + BoundCtx::CH);
    float lo = -INFINITY;
    for (int r = r0; r < r1; r++) {
      const _Float16* gn = M.h_gnorm.data() + (size_t)r * C.G;
      float acc = 0;
      for (int g = 0; g < C.G; g++) acc += (float)gn[g] * C.xg[g];
      const float B = acc * (1.f + 1e-5f) + M.h_bmax[r] * C.bterm + 0.01f + 1e-3f * std::fabs(M.coarse[r]);
      C.bound[r] = B;
      lo = std::max(lo, M.coarse[r] - B);
    }
    C.lo[ch] = lo;
  }
}

}  // namespace

int GreedyHead::run(Pool& pool, const XQ& xin, float* lg) {
  // coarse logits of every row from the 4-bit copy, on the same quantised input as the exact head
  auto t0 = clk::now();
  Job jh = {&head4, &xin, nullptr, coarse.data()};
  run_rows(pool, &jh, 1, 64);
  t_head_coarse += ms_since(t0);
  auto t1 = clk::now();
  // |exact - coarse| <= ||w8 - w4||_2 ||xhat||_2 + max_g|b8 - b4| * 32 * sum_g d_g  (+ float slack); xhat = the
  // quantised input the scale terms see (d_g * (hi * 256 + lo)), the bias terms see the fp32 sums (|x - xhat| <= d_g / 2)
  const XQ& q = xin;
  const int G = head8->cols / XB;
  float xg[64];
  double dsum = 0;
  for (int g = 0; g < G; g++) {
    double s2 = 0;
    for (int i = 0; i < XB; i++) {
      const double v = (double)q.hn[g * XB + i] * 256.0 + (double)q.ln[g * XB + i];
      s2 += v * v;
    }
    xg[g] = (float)((double)q.d[g] * std::sqrt(s2) * (1.0 + 1e-6));
    dsum += q.d[g];
  }
  BoundCtx B;
  B.M = this;
  B.G = G;
  B.xg = xg;
  B.bterm = (float)(32.0 * dsum);
  B.nch = (V + BoundCtx::CH - 1) / BoundCtx::CH;
  B.lo.assign(B.nch, -INFINITY);
  B.bound.resize(V);
  pool.run(phase_bound, &B);
  float lo_best = -INFINITY;
  for (float v : B.lo) lo_best = std::max(lo_best, v);
  lo = lo_best;
  cand.clear();
  for (int r = 0; r < V; r++)
    if (coarse[r] + B.bound[r] >= lo_best) cand.push_back(r);
  CandCtx C;
  C.M = this;
  C.n = (int)cand.size();
  xq = &xin;
  out = lg;
  pool.run(phase_cand, &C);
  int b = -1;
  for (int r : cand)  // ascending rows: the lowest index wins a tie, as the full argmax
    if (r < VO && (b < 0 || lg[r] > lg[b])) b = r;
  hd_cand += cand.size();
  hd_calls += 1;
  t_head_exact += ms_since(t1);
  return b;
}


}  // namespace dm
