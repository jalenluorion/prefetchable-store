// Half-precision products on the GPU for the fast prompt path (see gpu_half.h). Objective-C++ with ARC (-fobjc-arc).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_half.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dm {

// The kernel: one threadgroup (4 simdgroups) computes a TR x TQ tile (rows x positions) of one job, over the input in
// steps of 32: the weights' codes dequantised (code * scale + bias, fp32, rounded to half) into threadgroup memory, the
// positions' inputs (half, gathered through the job's map) beside them, then 8 x 8 half multiply-accumulates into fp32.
// Modes: 0 fp32 out [n][ld] (stored transposed straight from the matrix units); 1 gate and up of an expert at once, the
// SwiGLU activation half into the rows the out map names; 2 half out into the rows the out map names (an expert's down).
// The combination (kernel hc) sums a position's rows in pick order with fp32 fmas: a fixed order, so a run's bits do not
// depend on the GPU's scheduling.
static const char* HSRC = R"(
#include <metal_stdlib>
using namespace metal;
// ks: the input split in ks parts (mode 0 only), part p's partial sums at y + p * ps floats
struct HJob { ulong co, so, bo, co2, so2, bo2, x, xm, y, ym; uint R, K, N, fmt, tr, tbase, ld, ks, nq, ps, pad0, pad1; };
constant ulong NOMAP = 0xFFFFFFFFFFFFFFFFul;

// WPT consecutive codes of one row (inside one group of 64) into registers
template <uint WPT>
struct WReg {
  uint4 c[WPT / 16 > 0 ? WPT / 16 : 1];
  float s, b;
  inline void load(device const uchar* w, ulong co, ulong so, ulong bo, uint fmt, uint row, uint K, uint k) {
    const uint NG = K / 64;
    s = float(((device const half*)(w + so))[(ulong)row * NG + k / 64]);
    b = float(((device const half*)(w + bo))[(ulong)row * NG + k / 64]);
    if (fmt == 8) {
      device const uint4* p = (device const uint4*)(w + co + (ulong)row * K + k);
      for (uint i = 0; i < WPT / 16; i++) c[i] = p[i];
    } else {
      device const uint4* p = (device const uint4*)(w + co + (ulong)row * (K / 2) + k / 2);
      for (uint i = 0; i < WPT / 32; i++) c[i] = p[i];
    }
  }
  inline void store(uint fmt, threadgroup half* dst) const {
    if (fmt == 8) {
      for (uint i = 0; i < WPT / 16; i++)
        for (uint q = 0; q < 4; q++) {
          const uint v = c[i][q];
          for (uint t = 0; t < 4; t++) dst[16 * i + 4 * q + t] = half(fma(float((v >> (8 * t)) & 255u), s, b));
        }
    } else {
      for (uint i = 0; i < WPT / 32; i++)
        for (uint q = 0; q < 4; q++) {
          const uint v = c[i][q];
          for (uint t = 0; t < 8; t++) dst[32 * i + 8 * q + t] = half(fma(float((v >> (4 * t)) & 15u), s, b));
        }
    }
  }
};

template <uint TR, uint TQ, uint SR, uint SQ, uint MODE>
inline void hg_impl(device const uchar* w, device uchar* st, device const HJob* jobs, device const uint* tj, uint t, uint tid, uint sg,
                    threadgroup half* Tg, threadgroup uint* src, threadgroup uint* dstr) {
  // MODE 1: the tile's first TR / 2 rows are gate rows, the rest the same rows of up (one accumulator set, as mode 0)
  constexpr uint BK = 64, NT = 128, FR = TR / SR / 8, FQ = TQ / SQ / 8, WPT = TR * BK / NT, XPT = TQ * BK / NT;
  constexpr uint TRo = MODE == 1 ? TR / 2 : TR;  // output rows per tile
  threadgroup half* Wt = Tg;
  threadgroup half* Xt = Tg + TR * BK;
  const HJob j = jobs[tj[t]];
  const uint local = t - j.tbase, rt = local % j.tr, rest = local / j.tr, qt = rest % j.nq, kp = rest / j.nq;
  const uint r0 = rt * TRo, q0 = qt * TQ;
  const uint Kp = j.K / j.ks, ka = kp * Kp, kb = ka + Kp;  // this tile's part of the input
  for (uint u = tid; u < TQ; u += NT) {
    const uint q = min(q0 + u, j.N - 1);
    src[u] = j.xm != NOMAP ? ((device const uint*)(st + j.xm))[q] : q;
    if (MODE != 0) dstr[u] = q0 + u < j.N ? ((device const uint*)(st + j.ym))[q0 + u] : 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 acc[FR][FQ];
  for (uint a = 0; a < FR; a++)
    for (uint b = 0; b < FQ; b++) acc[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const uint sr = (sg % SR) * (TR / SR), sq = (sg / SR) * (TQ / SQ);
  const uint K = j.K;
  const uint we = tid * WPT, wr = we / BK, wk = we % BK;
  const bool up = MODE == 1 && wr >= TRo;
  const uint row = min(r0 + (up ? wr - TRo : wr), j.R - 1);
  const ulong co = up ? j.co2 : j.co, so = up ? j.so2 : j.so, bo = up ? j.bo2 : j.bo;
  const uint xe = tid * XPT, xp = xe / BK, xk = xe % BK;
  device const half* xrow = (device const half*)(st + j.x) + (ulong)src[xp] * K + xk;
  WReg<WPT> g1;
  half4 xr[XPT / 4];
  g1.load(w, co, so, bo, j.fmt, row, K, ka + wk);
  for (uint i = 0; i < XPT / 4; i++) xr[i] = ((device const half4*)(xrow + ka))[i];
  for (uint k0 = ka; k0 < kb; k0 += BK) {
    g1.store(j.fmt, Wt + wr * BK + wk);
    for (uint i = 0; i < XPT / 4; i++) ((threadgroup half4*)(Xt + xp * BK + xk))[i] = xr[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + BK < kb) {  // the next step's codes and inputs in flight while this one multiplies
      g1.load(w, co, so, bo, j.fmt, row, K, k0 + BK + wk);
      for (uint i = 0; i < XPT / 4; i++) xr[i] = ((device const half4*)(xrow + k0 + BK))[i];
    }
    for (uint k8 = 0; k8 < BK; k8 += 8) {
      simdgroup_half8x8 A[FR], B[FQ];
      for (uint a = 0; a < FR; a++) simdgroup_load(A[a], Wt + (sr + 8 * a) * BK + k8, BK);
      for (uint b = 0; b < FQ; b++) simdgroup_load(B[b], Xt + (sq + 8 * b) * BK + k8, BK, ulong2(0, 0), true);
      for (uint a = 0; a < FR; a++)
        for (uint b = 0; b < FQ; b++) simdgroup_multiply_accumulate(acc[a][b], A[a], B[b], acc[a][b]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (MODE == 0) {
    device float* Y = (device float*)(st + j.y) + (ulong)kp * j.ps;
    for (uint a = 0; a < FR; a++)
      for (uint b = 0; b < FQ; b++)
        simdgroup_store(acc[a][b], Y + (ulong)(q0 + sq + 8 * b) * j.ld + r0 + sr + 8 * a, j.ld, ulong2(0, 0), true);
    return;
  }
  // modes 1 and 2: the tile through threadgroup memory ([row][position] fp32, over the operand tiles), then into the rows
  // the out map names, consecutive threads on consecutive rows
  threadgroup float* C = (threadgroup float*)Tg;
  for (uint a = 0; a < FR; a++)
    for (uint b = 0; b < FQ; b++) simdgroup_store(acc[a][b], C + (sr + 8 * a) * TQ + sq + 8 * b, TQ);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device half* Y = (device half*)(st + j.y);
  for (uint e = tid; e < TRo * TQ; e += NT) {
    const uint r = e % TRo, q = e / TRo, d = dstr[q];
    if (r0 + r < j.R && d != 0xFFFFFFFFu) {
      float v = C[r * TQ + q];
      if (MODE == 1) v = v / (1.0f + exp(-v)) * C[(r + TRo) * TQ + q];
      Y[(ulong)d * j.ld + r0 + r] = half(v);
    }
  }
}

#define HG(NAME, TR, TQ, SR, SQ, MODE)                                                                                          \
  kernel void NAME(device const uchar* w [[buffer(0)]], device uchar* st [[buffer(1)]], device const HJob* jobs [[buffer(2)]],  \
                   device const uint* tj [[buffer(3)]], constant uint& t0 [[buffer(4)]], uint3 tgi [[threadgroup_position_in_grid]], \
                   uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {                      \
    /* the operand tiles (half), overlaid by the output tile (fp32) in modes 1 and 2 */                                          \
    threadgroup half Tg[(TR + TQ) * 64 > (MODE == 0 ? 0 : 2 * TR * TQ) ? (TR + TQ) * 64 : 2 * TR * TQ];                           \
    threadgroup uint src[TQ], dstr[TQ];                                                                                          \
    hg_impl<TR, TQ, SR, SQ, MODE>(w, st, jobs, tj, t0 + tgi.x, tid, sg, Tg, src, dstr);                                         \
  }
HG(hg_a0, 64, 64, 2, 2, 0)
HG(hg_b0, 128, 32, 4, 1, 0)
HG(hg_c0, 128, 16, 4, 1, 0)
HG(hg_a1, 64, 64, 2, 2, 1)
HG(hg_b1, 64, 32, 2, 2, 1)
HG(hg_c1, 64, 16, 4, 1, 1)
HG(hg_a2, 64, 64, 2, 2, 2)
HG(hg_b2, 128, 32, 4, 1, 2)
HG(hg_c2, 128, 16, 4, 1, 2)

// Attention over listed keys (QSA's selected positions; a dense prefix is the list 0..p): one threadgroup per (position,
// key head), its G <= 16 query heads at once, the keys in chunks of 32 (half keys and values as the cache holds them,
// scores and the running softmax in fp32: max, rescale, sum), the output divided by the sum at the end. D = 256.
// The Gated DeltaNet's heads over n positions in order (gdn.h head_io's arithmetic in fp32, other summation orders): one
// threadgroup per value head, four threads per state row (32 of its 128 key columns each, in registers); per position
// the l2-normed q and k, beta and the decay, S = g S, S += k (v - S k) beta, y = S q, then the gated RMSNorm over the head.
struct GD { uint n, NKH, NVH, C, ldp, zoff, boff, aoff, ldo, sig; float eps; uint pad; ulong S, cv, pj, out, A, dt, gn; };
kernel void gdn(device uchar* st [[buffer(0)]], constant GD& p [[buffer(1)]], uint h [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
  constexpr uint DK = 128, DV = 128;
  threadgroup float qk[2 * DK];
  threadgroup float ys[DV];
  threadgroup float red[2];
  const uint row = tid / 4, part = tid % 4, kh = h / (p.NVH / p.NKH);
  device float* S = (device float*)(st + p.S) + ((ulong)h * DV + row) * DK + part * 32;
  float s[32];
  for (uint j = 0; j < 32; j++) s[j] = S[j];
  device const float* A = (device const float*)(st + p.A);
  device const float* dt = (device const float*)(st + p.dt);
  device const float* gn = (device const float*)(st + p.gn);
  const float Ah = exp(A[h]);
  for (uint t = 0; t < p.n; t++) {
    device const float* cv = (device const float*)(st + p.cv) + (ulong)t * p.C;
    device const float* pj = (device const float*)(st + p.pj) + (ulong)t * p.ldp;
    if (tid < 2 * DK) qk[tid] = tid < DK ? cv[kh * DK + tid] : cv[p.NKH * DK + kh * DK + tid - DK];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < 2) {  // the norms of q (simdgroup 0) and k (1)
      float a = 0;
      for (uint i = ln; i < DK; i += 32) a += qk[sg * DK + i] * qk[sg * DK + i];
      a = simd_sum(a);
      if (ln == 0) red[sg] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rq = 1.0f / sqrt(red[0] + 1e-6f) / sqrt((float)DK), rk = 1.0f / sqrt(red[1] + 1e-6f);
    const float beta = 1.0f / (1.0f + exp(-pj[p.boff + h]));
    const float xa = pj[p.aoff + h] + dt[h];
    const float g = exp(-Ah * (xa > 20.0f ? xa : log(1.0f + exp(xa))));
    float pr = 0;
    for (uint j = 0; j < 32; j++) {
      s[j] *= g;
      pr += s[j] * qk[DK + part * 32 + j] * rk;
    }
    pr += simd_shuffle_xor(pr, 1);
    pr += simd_shuffle_xor(pr, 2);
    const float v = cv[2 * p.NKH * DK + h * DV + row];
    const float delta = (v - pr) * beta;
    float o = 0;
    for (uint j = 0; j < 32; j++) {
      s[j] += qk[DK + part * 32 + j] * rk * delta;
      o += s[j] * qk[part * 32 + j] * rq;
    }
    o += simd_shuffle_xor(o, 1);
    o += simd_shuffle_xor(o, 2);
    if (part == 0) ys[row] = o;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      float a = 0;
      for (uint i = ln; i < DV; i += 32) a += ys[i] * ys[i];
      a = simd_sum(a);
      if (ln == 0) red[0] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) {
      const float r = 1.0f / sqrt(red[0] / DV + p.eps), z = pj[p.zoff + h * DV + row];
      const float gate = p.sig ? 1.0f / (1.0f + exp(-z)) : z / (1.0f + exp(-z));
      ((device float*)(st + p.out))[(ulong)t * p.ldo + h * DV + row] = o * r * gn[row] * gate;
    }
  }
  for (uint j = 0; j < 32; j++) S[j] = s[j];
}

struct QA { uint n, nkv, G, ctx; ulong q, sel0, sel, out; float scale; uint pad; };
kernel void qa(device const half* Kc [[buffer(0)]], device const half* Vc [[buffer(1)]], device uchar* st [[buffer(2)]],
               constant QA& p [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
  constexpr uint D = 256, KC = 32, NT = 128;
  threadgroup half Qs[16 * D];           // the query heads (rows >= G zero)
  threadgroup half KV[KC * D];           // a chunk's keys, then its values; at the end the output rows (fp32, 16 x D)
  threadgroup float Ss[16 * KC];         // scores
  threadgroup half Ps[16 * KC];          // probabilities
  threadgroup float Dg[2 * 64];          // the rows' rescale factors as two 8 x 8 diagonals
  threadgroup float mrow[16], lrow[16];
  threadgroup uint ks[KC];
  const uint i = tg / p.nkv, g = tg % p.nkv;
  device const uint* sel0 = (device const uint*)(st + p.sel0);
  device const uint* sel = (device const uint*)(st + p.sel) + sel0[i];
  const uint nk = sel0[i + 1] - sel0[i];
  device const half* q = (device const half*)(st + p.q) + ((ulong)i * p.nkv * p.G + (ulong)g * p.G) * D;
  for (uint e = tid; e < 16 * D; e += NT) Qs[e] = e / D < p.G ? q[e] : half(0);
  if (tid < 16) mrow[tid] = -INFINITY, lrow[tid] = 0.0f;
  device const half* Kg = Kc + (ulong)g * p.ctx * D;
  device const half* Vg = Vc + (ulong)g * p.ctx * D;
  simdgroup_float8x8 O[2][8];  // rows 0-15 x this simdgroup's 64 columns
  for (uint a = 0; a < 2; a++)
    for (uint b = 0; b < 8; b++) O[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c0 = 0; c0 < nk; c0 += KC) {
    const uint nc = min(KC, nk - c0);
    if (tid < KC) ks[tid] = tid < nc ? sel[c0 + tid] : sel[c0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < KC * D / 8; e += NT) {  // keys, 8 halves per load
      const uint r = e / (D / 8), c = e % (D / 8);
      ((threadgroup half4*)KV)[2 * e] = ((device const half4*)(Kg + (ulong)ks[r] * D))[2 * c];
      ((threadgroup half4*)KV)[2 * e + 1] = ((device const half4*)(Kg + (ulong)ks[r] * D))[2 * c + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {  // scores: simdgroup sg the keys [8 sg, 8 sg + 8) for both row blocks
      simdgroup_float8x8 S0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f), S1 = S0;
      for (uint k8 = 0; k8 < D; k8 += 8) {
        simdgroup_half8x8 A0, A1, B;
        simdgroup_load(A0, Qs + k8, D);
        simdgroup_load(A1, Qs + 8 * D + k8, D);
        simdgroup_load(B, KV + (8 * sg) * D + k8, D, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(S0, A0, B, S0);
        simdgroup_multiply_accumulate(S1, A1, B, S1);
      }
      simdgroup_store(S0, Ss + 8 * sg, KC);
      simdgroup_store(S1, Ss + 8 * KC + 8 * sg, KC);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {  // the running softmax of row tid
      const uint r = tid;
      float mx = mrow[r];
      for (uint j = 0; j < nc; j++) mx = max(mx, Ss[r * KC + j] * p.scale);
      const float corr = exp(mrow[r] - mx);
      float l = lrow[r] * corr;
      for (uint j = 0; j < KC; j++) {
        const float e = j < nc ? exp(Ss[r * KC + j] * p.scale - mx) : 0.0f;
        l += e;
        Ps[r * KC + j] = half(e);
      }
      mrow[r] = mx, lrow[r] = l;
      for (uint c = 0; c < 8; c++) Dg[(r / 8) * 64 + (r % 8) * 8 + c] = c == r % 8 ? corr : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < KC * D / 8; e += NT) {  // values over the keys' tile
      const uint r = e / (D / 8), c = e % (D / 8);
      ((threadgroup half4*)KV)[2 * e] = ((device const half4*)(Vg + (ulong)ks[r] * D))[2 * c];
      ((threadgroup half4*)KV)[2 * e + 1] = ((device const half4*)(Vg + (ulong)ks[r] * D))[2 * c + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
      simdgroup_float8x8 D0, D1;
      simdgroup_load(D0, Dg, 8);
      simdgroup_load(D1, Dg + 64, 8);
      for (uint b = 0; b < 8; b++) {
        simdgroup_multiply(O[0][b], D0, O[0][b]);
        simdgroup_multiply(O[1][b], D1, O[1][b]);
      }
      for (uint k8 = 0; k8 < KC; k8 += 8) {
        simdgroup_half8x8 P0, P1, Vb;
        simdgroup_load(P0, Ps + k8, KC);
        simdgroup_load(P1, Ps + 8 * KC + k8, KC);
        for (uint b = 0; b < 8; b++) {
          simdgroup_load(Vb, KV + k8 * D + 64 * sg + 8 * b, D);
          simdgroup_multiply_accumulate(O[0][b], P0, Vb, O[0][b]);
          simdgroup_multiply_accumulate(O[1][b], P1, Vb, O[1][b]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float* Of = (threadgroup float*)KV;  // 16 x D fp32 = 16 KB = the tile's memory
  for (uint b = 0; b < 8; b++) {
    simdgroup_store(O[0][b], Of + 64 * sg + 8 * b, D);
    simdgroup_store(O[1][b], Of + 8 * D + 64 * sg + 8 * b, D);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* out = (device float*)(st + p.out) + ((ulong)i * p.nkv * p.G + (ulong)g * p.G) * D;
  for (uint e = tid; e < p.G * D; e += NT) out[e] = Of[e] / lrow[e / D];
}

// the combination: y[i][r] = fma chain over the picks in order of w[i K + j] * e[i K + j][r], then shw[i] * e[P K + i][r]
kernel void hc(device uchar* st [[buffer(1)]], constant uint4& d [[buffer(5)]], constant ulong4& o [[buffer(6)]],
               uint2 g [[thread_position_in_grid]]) {
  const uint H = d.x, K = d.y, P = d.z, sh = d.w, r = g.x, i = g.y;
  if (r >= H || i >= P) return;
  device const half* e = (device const half*)(st + o.x);
  device const float* w = (device const float*)(st + o.y);
  device const float* shw = (device const float*)(st + o.z);
  device float* y = (device float*)(st + o.w);
  float acc = 0.0f;
  for (uint j = 0; j < K; j++) {
    const float wj = w[(ulong)i * K + j];
    if (wj != 0.0f) acc = fma(wj, float(e[((ulong)i * K + j) * H + r]), acc);
  }
  if (sh) acc = fma(shw[i], float(e[((ulong)P * K + i) * H + r]), acc);
  y[(ulong)i * H + r] = acc;
}
)";

namespace {
struct HJob {  // the kernel's HJob (10 x 8 + 12 x 4 = 128 bytes)
  uint64_t co, so, bo, co2, so2, bo2, x, xm, y, ym;
  uint32_t R, K, N, fmt, tr, tbase, ld, ks, nq, ps, pad0, pad1;
};
static_assert(sizeof(HJob) == 128, "HJob layout");
constexpr uint64_t NOMAP = ~0ull;
constexpr int TRv[3][3] = {{64, 128, 128}, {32, 32, 32}, {64, 128, 128}}, TQv[3] = {64, 32, 16};  // output rows per tile [mode][variant]
inline int variant(int n) { return n > 32 ? 0 : n > 16 ? 1 : 2; }
inline size_t al(size_t n) { return (n + 255) & ~size_t(255); }
using hclk = std::chrono::steady_clock;
inline double ms_from(hclk::time_point a) { return std::chrono::duration<double, std::milli>(hclk::now() - a).count(); }
}  // namespace

struct HImpl {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> ps[3][3];  // [mode][variant]
  id<MTLComputePipelineState> comb, qa, gd;
  size_t page = 16384, max_len = 0;
  struct WB {
    id<MTLBuffer> b;
    const uint8_t* base;
    size_t len;
  };
  std::vector<WB> wb;
  // the set being built
  struct In {
    const float* const* rows;
    int n, K;
    size_t off;
  };
  std::vector<In> ins;
  struct Map {
    const int* p;
    int n;
    size_t off;
  };
  std::vector<Map> maps;
  std::vector<HJob> jobs;
  std::vector<int> job_wb, job_var, job_mode, job_in, job_map, job_n;
  std::vector<uint32_t> tiles;
  struct Grp {
    int wb, mode, var;
    uint32_t t0, nt;
  };
  std::vector<Grp> grps;
  id<MTLBuffer> st = nil, jb = nil, tt = nil;
  size_t st_cap = 0, jb_cap = 0, tt_cap = 0;
  struct Moe {  // the MoE block in flight (moe_begin .. moe_end): offsets in st, the waves' command buffers
    bool live = false, sh = false, sh_done = false;
    int P = 0, H = 0, K = 0, I = 0;
    const Weight *sg = nullptr, *su = nullptr, *sd = nullptr;
    size_t ox = 0, oact = 0, oe = 0, ow = 0, osw = 0, oy = 0, omap = 0, omap_end = 0;
    std::vector<id<MTLCommandBuffer>> cbs;
    std::vector<id<MTLBuffer>> keep;
  } moe;
  bool grow(id<MTLBuffer> __strong& b, size_t& cap, size_t need) {
    if (need <= cap && b) return true;
    b = nil;
    size_t n = std::max<size_t>(need + need / 8, 1 << 20);
    b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    cap = b ? n : 0;
    return b != nil;
  }
  int find(const void* p, size_t n) const {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < wb.size(); i++)
      if (b >= wb[i].base && b + n <= wb[i].base + wb[i].len) return (int)i;
    return -1;
  }
  int find_w(const Weight& w) const {  // one buffer holding the codes, scales and biases
    if ((w.kind != Weight::AFF8 && w.kind != Weight::AFF4) || w.cols % 64) return -1;
    const size_t cb = (size_t)w.rows * (w.kind == Weight::AFF8 ? w.cols : w.cols / 2), sb = (size_t)w.rows * (w.cols / 64) * 2;
    for (size_t i = 0; i < wb.size(); i++) {
      auto in = [&](const void* p, size_t nb) { return (const uint8_t*)p >= wb[i].base && (const uint8_t*)p + nb <= wb[i].base + wb[i].len; };
      if (in(w.c, cb) && in(w.s, sb) && in(w.b, sb)) return (int)i;
    }
    return -1;
  }
  void clear() {
    ins.clear();
    maps.clear();
    jobs.clear();
    job_wb.clear();
    job_var.clear();
    job_mode.clear();
    job_in.clear();
    job_map.clear();
    job_n.clear();
    tiles.clear();
    grps.clear();
  }
  // a job (weights in buffer wbi) -> its index
  int push(const Weight& g, const Weight* u, int wbi, int R, int n, int mode, int var) {
    const uint8_t* B = wb[wbi].base;
    HJob j{};
    j.co = (uint64_t)(g.c - B);
    j.so = (uint64_t)((const uint8_t*)g.s - B);
    j.bo = (uint64_t)((const uint8_t*)g.b - B);
    if (u) {
      j.co2 = (uint64_t)(u->c - B);
      j.so2 = (uint64_t)((const uint8_t*)u->s - B);
      j.bo2 = (uint64_t)((const uint8_t*)u->b - B);
    }
    j.R = (uint32_t)R;
    j.K = (uint32_t)g.cols;
    j.N = (uint32_t)n;
    j.fmt = g.kind == Weight::AFF8 ? 8 : 4;
    j.tr = (uint32_t)((R + TRv[mode][var] - 1) / TRv[mode][var]);
    j.nq = (uint32_t)((n + TQv[var] - 1) / TQv[var]);
    j.ks = 1;
    j.xm = NOMAP;
    j.ym = NOMAP;
    jobs.push_back(j);
    job_wb.push_back(wbi);
    job_var.push_back(var);
    job_mode.push_back(mode);
    return (int)jobs.size() - 1;
  }
  // the tile table grouped by (buffer, mode, variant)
  void tile() {
    std::vector<int> order(jobs.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      if (job_wb[a] != job_wb[b]) return job_wb[a] < job_wb[b];
      if (job_mode[a] != job_mode[b]) return job_mode[a] < job_mode[b];
      return job_var[a] < job_var[b];
    });
    for (int i : order) {
      HJob& j = jobs[i];
      const uint32_t nt = j.tr * j.nq * j.ks;
      j.tbase = (uint32_t)tiles.size();
      if (grps.empty() || grps.back().wb != job_wb[i] || grps.back().mode != job_mode[i] || grps.back().var != job_var[i])
        grps.push_back({job_wb[i], job_mode[i], job_var[i], j.tbase, 0});
      grps.back().nt += nt;
      for (uint32_t t = 0; t < nt; t++) tiles.push_back((uint32_t)i);
    }
  }
  // encode the groups of modes [m0, m1] (a barrier between modes)
  void encode(id<MTLComputeCommandEncoder> en, int m0, int m1) {
    int last = -1;
    for (const Grp& G : grps) {
      if (G.mode < m0 || G.mode > m1) continue;
      if (last >= 0 && last != G.mode) [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
      last = G.mode;
      [en setComputePipelineState:ps[G.mode][G.var]];
      [en setBuffer:wb[G.wb].b offset:0 atIndex:0];
      for (uint32_t a = 0; a < G.nt; a += 65535) {
        const uint32_t t0 = G.t0 + a, nt = std::min<uint32_t>(65535, G.nt - a);
        [en setBytes:&t0 length:4 atIndex:4];
        [en dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      }
    }
  }
  bool commit_wait(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted;
  }
};

bool GpuHalf::init() {
  @autoreleasepool {
    if (impl) return true;
    HImpl* g = new HImpl();
    g->dev = MTLCreateSystemDefaultDevice();
    if (!g->dev) {
      delete g;
      return false;
    }
    NSError* err = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:HSRC] options:co error:&err];
    if (!lib) {
      fprintf(stderr, "gpu_half: %s\n", err ? err.localizedDescription.UTF8String : "kernel compile failed");
      delete g;
      return false;
    }
    const char* nm[3][3] = {{"hg_a0", "hg_b0", "hg_c0"}, {"hg_a1", "hg_b1", "hg_c1"}, {"hg_a2", "hg_b2", "hg_c2"}};
    for (int m = 0; m < 3; m++)
      for (int v = 0; v < 3; v++) {
        g->ps[m][v] = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:nm[m][v]]] error:&err];
        if (!g->ps[m][v] || g->ps[m][v].maxTotalThreadsPerThreadgroup < 128) {
          fprintf(stderr, "gpu_half: kernel %s unavailable\n", nm[m][v]);
          delete g;
          return false;
        }
      }
    g->comb = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"hc"] error:&err];
    g->qa = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"qa"] error:&err];
    g->gd = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gdn"] error:&err];
    if (g->gd && g->gd.maxTotalThreadsPerThreadgroup < 512) g->gd = nil;
    if (!g->qa) fprintf(stderr, "gpu_half: no attention kernel (%s)\n", err ? err.localizedDescription.UTF8String : "");
    g->q = [g->dev newCommandQueue];
    if (!g->comb || !g->q) {
      fprintf(stderr, "gpu_half: %s\n", err ? err.localizedDescription.UTF8String : "no combination kernel or queue");
      delete g;
      return false;
    }
    g->page = (size_t)getpagesize();
    g->max_len = (size_t)g->dev.maxBufferLength;
    impl = g;
    return true;
  }
}

GpuHalf::~GpuHalf() { delete (HImpl*)impl; }

bool GpuHalf::wrap(const void* base, size_t bytes) {
  @autoreleasepool {
    if (!impl || !base || !bytes) return false;
    HImpl& g = *(HImpl*)impl;
    if ((uintptr_t)base % g.page) return false;
    for (auto& w : g.wb)
      if (w.base == base) return true;  // already
    const size_t len = (bytes + g.page - 1) / g.page * g.page;
    const size_t W = std::min(len, g.max_len / g.page * g.page);
    if (W < (size_t)64 << 20 && W < len) return false;
    const size_t stride = W == len ? len : W / 2;
    bool any = false;
    for (size_t o = 0; o < len; o += stride) {
      const size_t l = std::min(W, len - o);
      id<MTLBuffer> b = [g.dev newBufferWithBytesNoCopy:(void*)((const uint8_t*)base + o) length:l options:MTLResourceStorageModeShared
                                            deallocator:nil];
      if (!b) return any;
      g.wb.push_back({b, (const uint8_t*)base + o, l});
      any = true;
      if (o + l >= len) break;
    }
    return any;
  }
}

bool GpuHalf::holds(const Weight& w) const { return impl && ((const HImpl*)impl)->find_w(w) >= 0; }

size_t GpuHalf::wrapped_bytes() const {
  if (!impl) return 0;
  size_t s = 0;
  for (auto& w : ((const HImpl*)impl)->wb) s += w.len;
  return s;
}

void GpuHalf::begin() { ((HImpl*)impl)->clear(); }

int GpuHalf::input(const float* const* rows, int n, int K) {
  HImpl& g = *(HImpl*)impl;
  for (size_t i = 0; i < g.ins.size(); i++)
    if (g.ins[i].rows == rows && g.ins[i].n == n && g.ins[i].K == K) return (int)i;
  g.ins.push_back({rows, n, K, 0});
  return (int)g.ins.size() - 1;
}

int GpuHalf::add(const Weight& w, int R, int in, int n, const int* map) {
  HImpl& g = *(HImpl*)impl;
  if (in < 0 || in >= (int)g.ins.size() || n <= 0 || R <= 0 || R > w.rows || g.ins[in].K != w.cols) return -1;
  const int wbi = g.find_w(w);
  if (wbi < 0) return -1;
  const int t = g.push(w, nullptr, wbi, R, n, 0, variant(n));
  int mi = -1;
  if (map) {
    for (size_t i = 0; i < g.maps.size() && mi < 0; i++)
      if (g.maps[i].p == map && g.maps[i].n == n) mi = (int)i;
    if (mi < 0) {
      g.maps.push_back({map, n, 0});
      mi = (int)g.maps.size() - 1;
    }
  }
  g.job_in.push_back(in);
  g.job_map.push_back(mi);
  g.job_n.push_back(n);
  flops += 2.0 * R * w.cols * n;
  return t;
}

bool GpuHalf::run(Pool& pool) {
  @autoreleasepool {
    HImpl& g = *(HImpl*)impl;
    if (g.jobs.empty()) return true;
    const auto t0 = hclk::now();
    size_t off = 0;
    auto take = [&](size_t n) {
      const size_t o = off;
      off += al(n);
      return o;
    };
    for (auto& I : g.ins) I.off = take((size_t)I.n * I.K * 2);
    for (auto& M : g.maps) M.off = take((size_t)M.n * 4);
    for (size_t i = 0; i < g.jobs.size(); i++) {
      HJob& j = g.jobs[i];
      const int v = g.job_var[i];
      j.x = g.ins[g.job_in[i]].off;
      j.xm = g.job_map[i] >= 0 ? g.maps[g.job_map[i]].off : NOMAP;
      j.ld = (uint32_t)(j.tr * TRv[0][v]);
      const size_t np = (size_t)j.nq * TQv[v];
      // few tiles over a long input (the low-rank mixes: 320 rows of 10,240 inputs): the input split in ks parts, their
      // partial sums added by out()'s reader (fills the GPU; the sum order is fixed)
      const uint32_t tiles = j.tr * j.nq;
      j.ks = 1;
      while (tiles * j.ks * 2 <= 160 && j.K % (j.ks * 2 * 64) == 0 && j.K / (j.ks * 2) >= 512) j.ks *= 2;
      j.ps = (uint32_t)(np * j.ld);
      j.y = take(np * j.ld * 4 * j.ks);
    }
    g.tile();
    if (!g.grow(g.st, g.st_cap, off) || !g.grow(g.jb, g.jb_cap, g.jobs.size() * sizeof(HJob)) || !g.grow(g.tt, g.tt_cap, g.tiles.size() * 4)) {
      failed++;
      return false;
    }
    std::memcpy(g.jb.contents, g.jobs.data(), g.jobs.size() * sizeof(HJob));
    std::memcpy(g.tt.contents, g.tiles.data(), g.tiles.size() * 4);
    uint8_t* S = (uint8_t*)g.st.contents;
    // staging: the inputs to half, row by row (the caller's threads), and the maps
    std::vector<std::pair<int, int>> rowsj;
    for (size_t i = 0; i < g.ins.size(); i++)
      for (int r = 0; r < g.ins[i].n; r += 16) rowsj.push_back({(int)i, r});
    parallel_for(pool, (int)rowsj.size(), [&](int u) {
      const auto& I = g.ins[rowsj[u].first];
      for (int r = rowsj[u].second; r < std::min(I.n, rowsj[u].second + 16); r++) {
        _Float16* d = (_Float16*)(S + I.off) + (size_t)r * I.K;
        const float* s = I.rows[r];
        for (int k = 0; k < I.K; k++) d[k] = (_Float16)s[k];
      }
    });
    for (auto& M : g.maps) {
      uint32_t* d = (uint32_t*)(S + M.off);
      for (int t = 0; t < M.n; t++) d[t] = (uint32_t)M.p[t];
    }
    ms_stage += ms_from(t0);
    const auto t1 = hclk::now();
    id<MTLCommandBuffer> cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setBuffer:g.st offset:0 atIndex:1];
    [en setBuffer:g.jb offset:0 atIndex:2];
    [en setBuffer:g.tt offset:0 atIndex:3];
    g.encode(en, 0, 0);
    [en endEncoding];
    const bool ok = g.commit_wait(cb);
    ms_gpu += ms_from(t1);
    sets++;
    if (!ok) failed++;
    return ok;
  }
}

void GpuHalf::row(int t, int q, float* dst) const {
  const HImpl& g = *(const HImpl*)impl;
  const HJob& j = g.jobs[t];
  const float* y = (const float*)((const uint8_t*)g.st.contents + j.y) + (size_t)q * j.ld;
  if (j.ks == 1) {
    std::memcpy(dst, y, (size_t)j.R * 4);
    return;
  }
  for (uint32_t r = 0; r < j.R; r++) {
    float a = y[r];
    for (uint32_t p = 1; p < j.ks; p++) a += y[(size_t)p * j.ps + r];
    dst[r] = a;
  }
}

// A MoE block in waves: moe_begin stages the inputs and sizes the buffers, each moe_wave encodes and commits its experts'
// gate-up and down (the shared expert's with the first wave; no wait: the GPU runs a wave while the caller waits for the
// next records to land), moe_end commits the combination, waits for every wave and copies the outputs
bool GpuHalf::moe_begin(Pool& pool, const float* const* x, int P, int H, int K, int I, const Weight* sh_gate, const Weight* sh_up,
                        const Weight* sh_down, int SI) {
  @autoreleasepool {
    if (!impl || P <= 0) return false;
    HImpl& g = *(HImpl*)impl;
    HImpl::Moe& m = g.moe;
    m = HImpl::Moe();
    const auto t0 = hclk::now();
    m.sh = sh_gate && sh_up && sh_down;
    if (m.sh && (g.find_w(*sh_gate) < 0 || g.find_w(*sh_up) < 0 || g.find_w(*sh_down) < 0 || SI != I)) return false;
    m.P = P, m.H = H, m.K = K, m.I = I, m.sg = sh_gate, m.su = sh_up, m.sd = sh_down;
    const int A = P * K + (m.sh ? P : 0);
    size_t off = 0;
    auto take = [&](size_t n) {
      const size_t o = off;
      off += al(n);
      return o;
    };
    m.ox = take((size_t)P * H * 2), m.oact = take((size_t)A * I * 2), m.oe = take((size_t)A * H * 2), m.ow = take((size_t)P * K * 4);
    m.osw = take((size_t)P * 4), m.oy = take((size_t)P * H * 4);
    m.omap = take((size_t)A * 8 + (size_t)P * K * 512);  // the users' maps (and slack for the alignment of each)
    m.omap_end = off;
    if (!g.grow(g.st, g.st_cap, off)) {
      failed++;
      return false;
    }
    uint8_t* S = (uint8_t*)g.st.contents;
    parallel_for(pool, (P + 15) / 16, [&](int u) {
      for (int i = u * 16; i < std::min(P, u * 16 + 16); i++) {
        _Float16* d = (_Float16*)(S + m.ox) + (size_t)i * H;
        for (int k = 0; k < H; k++) d[k] = (_Float16)x[i][k];
      }
    });
    m.live = true;
    ms_stage += ms_from(t0);
    return true;
  }
}

bool GpuHalf::moe_wave(Pool& pool, const Exp* const* ex, int ne) {
  @autoreleasepool {
    HImpl& g = *(HImpl*)impl;
    HImpl::Moe& m = g.moe;
    if (!m.live) return false;
    const auto t0 = hclk::now();
    g.clear();
    uint8_t* S = (uint8_t*)g.st.contents;
    auto add2 = [&](const Weight& gw, const Weight& uw, const Weight& dw, int n, size_t mpos, size_t mrow) {
      const int v = variant(n);
      const int a = g.push(gw, &uw, g.find_w(gw), m.I, n, 1, v);
      if (g.job_wb.back() != g.find_w(uw)) return false;  // gate and up must share a buffer
      g.jobs[a].x = m.ox;
      g.jobs[a].xm = mpos;
      g.jobs[a].y = m.oact;
      g.jobs[a].ym = mrow;
      g.jobs[a].ld = m.I;
      const int b = g.push(dw, nullptr, g.find_w(dw), m.H, n, 2, v);
      g.jobs[b].x = m.oact;
      g.jobs[b].xm = mrow;
      g.jobs[b].y = m.oe;
      g.jobs[b].ym = mrow;
      g.jobs[b].ld = m.H;
      flops += 2.0 * n * 3.0 * m.I * m.H;
      return true;
    };
    auto map = [&](size_t n) {
      const size_t o = m.omap;
      m.omap += al(n * 4);
      return o;
    };
    std::vector<std::pair<size_t, size_t>> mp(ne);
    for (int k = 0; k < ne; k++) {
      const Exp& e = *ex[k];
      if (e.pos.empty()) continue;
      if (g.find_w(*e.gate) < 0 || g.find_w(*e.up) < 0 || g.find_w(*e.down) < 0 || e.gate->cols != m.H || e.down->cols != m.I || e.gate->rows != m.I)
        return (m.live = false), false;
      mp[k] = {map(e.pos.size()), map(e.row.size())};
      if (m.omap > m.omap_end || !add2(*e.gate, *e.up, *e.down, (int)e.pos.size(), mp[k].first, mp[k].second)) return (m.live = false), false;
    }
    if (m.sh && !m.sh_done) {
      const size_t a = map(m.P), b = map(m.P);
      uint32_t* dp = (uint32_t*)(S + a);
      uint32_t* dr = (uint32_t*)(S + b);
      for (int i = 0; i < m.P; i++) dp[i] = (uint32_t)i, dr[i] = (uint32_t)(m.P * m.K + i);
      if (m.omap > m.omap_end || !add2(*m.sg, *m.su, *m.sd, m.P, a, b)) return (m.live = false), false;
      m.sh_done = true;
    }
    if (g.jobs.empty()) return true;
    parallel_for(pool, ne, [&](int k) {
      const Exp& e = *ex[k];
      if (e.pos.empty()) return;
      uint32_t* dp = (uint32_t*)(S + mp[k].first);
      uint32_t* dr = (uint32_t*)(S + mp[k].second);
      for (size_t t = 0; t < e.pos.size(); t++) dp[t] = (uint32_t)e.pos[t], dr[t] = (uint32_t)e.row[t];
    });
    g.tile();
    id<MTLBuffer> jb = [g.dev newBufferWithBytes:g.jobs.data() length:g.jobs.size() * sizeof(HJob) options:MTLResourceStorageModeShared];
    id<MTLBuffer> tt = [g.dev newBufferWithBytes:g.tiles.data() length:g.tiles.size() * 4 options:MTLResourceStorageModeShared];
    if (!jb || !tt) return (m.live = false), false;
    id<MTLCommandBuffer> cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setBuffer:g.st offset:0 atIndex:1];
    [en setBuffer:jb offset:0 atIndex:2];
    [en setBuffer:tt offset:0 atIndex:3];
    g.encode(en, 1, 1);
    [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
    g.encode(en, 2, 2);
    [en endEncoding];
    [cb commit];
    m.cbs.push_back(cb);
    m.keep.push_back(jb);
    m.keep.push_back(tt);
    ms_stage += ms_from(t0);
    return true;
  }
}

bool GpuHalf::moe_end(Pool& pool, const float* w, const float* shw, float* const* y) {
  @autoreleasepool {
    HImpl& g = *(HImpl*)impl;
    HImpl::Moe& m = g.moe;
    const auto t1 = hclk::now();
    bool ok = m.live;
    if (ok) {
      uint8_t* S = (uint8_t*)g.st.contents;
      std::memcpy(S + m.ow, w, (size_t)m.P * m.K * 4);
      if (m.sh) std::memcpy(S + m.osw, shw, (size_t)m.P * 4);
      id<MTLCommandBuffer> cb = [g.q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setBuffer:g.st offset:0 atIndex:1];
      [en setComputePipelineState:g.comb];
      const uint32_t d[4] = {(uint32_t)m.H, (uint32_t)m.K, (uint32_t)m.P, m.sh ? 1u : 0u};
      const uint64_t o[4] = {m.oe, m.ow, m.osw, m.oy};
      [en setBytes:d length:sizeof d atIndex:5];
      [en setBytes:o length:sizeof o atIndex:6];
      [en dispatchThreads:MTLSizeMake(m.H, m.P, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [en endEncoding];
      [cb commit];
      m.cbs.push_back(cb);
    }
    for (id<MTLCommandBuffer> c : m.cbs) {  // (in order on one queue: the last one done means all done; each checked)
      [c waitUntilCompleted];
      ok = ok && c.status == MTLCommandBufferStatusCompleted;
    }
    ms_gpu += ms_from(t1);
    sets++;
    if (ok) {
      const auto t2 = hclk::now();
      const float* Y = (const float*)((const uint8_t*)g.st.contents + m.oy);
      const int P = m.P, H = m.H;
      parallel_for(pool, (P + 15) / 16, [&](int u) {
        for (int i = u * 16; i < std::min(P, u * 16 + 16); i++) std::memcpy(y[i], Y + (size_t)i * H, sizeof(float) * H);
      });
      ms_stage += ms_from(t2);
    } else if (m.live) {
      failed++;
    }
    m = HImpl::Moe();
    return ok;
  }
}

bool GpuHalf::moe(Pool& pool, const float* const* x, int P, int H, int K, int I, const std::vector<Exp>& ex, const float* w, const Weight* sh_gate,
                  const Weight* sh_up, const Weight* sh_down, int SI, const float* shw, float* const* y) {
  if (!moe_begin(pool, x, P, H, K, I, sh_gate, sh_up, sh_down, SI)) return false;
  std::vector<const Exp*> e(ex.size());
  for (size_t i = 0; i < ex.size(); i++) e[i] = &ex[i];
  moe_wave(pool, e.data(), (int)e.size());
  return moe_end(pool, w, shw, y);
}

bool GpuHalf::attn(Pool& pool, const _Float16* K, const _Float16* V, int ctx, int D, int NQ, int NKV, int n, const float* const* q,
                   const std::vector<int>* const* sel, float* const* out) {
  @autoreleasepool {
    HImpl& g = *(HImpl*)impl;
    const int G = NKV > 0 ? NQ / NKV : 0;
    if (!g.qa || D != 256 || G < 1 || G > 16 || NQ != G * NKV || n <= 0) return false;
    const auto t0 = hclk::now();
    // the cache's keys and values where they are (page-aligned regions), wrapped for this call
    const size_t kvb = ((size_t)NKV * ctx * D * 2 + g.page - 1) / g.page * g.page;
    if ((uintptr_t)K % g.page || (uintptr_t)V % g.page) return false;
    id<MTLBuffer> kb = [g.dev newBufferWithBytesNoCopy:(void*)K length:kvb options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> vb = [g.dev newBufferWithBytesNoCopy:(void*)V length:kvb options:MTLResourceStorageModeShared deallocator:nil];
    if (!kb || !vb) return false;
    size_t nsel = 0;
    for (int i = 0; i < n; i++) nsel += sel[i]->size();
    size_t off = 0;
    auto take = [&](size_t b) {
      const size_t o = off;
      off += al(b);
      return o;
    };
    const size_t oq = take((size_t)n * NQ * D * 2), os0 = take((size_t)(n + 1) * 4), os = take(nsel * 4), oo = take((size_t)n * NQ * D * 4);
    if (!g.grow(g.st, g.st_cap, off)) return false;
    uint8_t* S = (uint8_t*)g.st.contents;
    uint32_t* s0 = (uint32_t*)(S + os0);
    s0[0] = 0;
    for (int i = 0; i < n; i++) s0[i + 1] = s0[i] + (uint32_t)sel[i]->size();
    parallel_for(pool, n, [&](int i) {
      _Float16* d = (_Float16*)(S + oq) + (size_t)i * NQ * D;
      for (int k = 0; k < NQ * D; k++) d[k] = (_Float16)q[i][k];
      uint32_t* sd = (uint32_t*)(S + os) + s0[i];
      for (size_t t = 0; t < sel[i]->size(); t++) sd[t] = (uint32_t)(*sel[i])[t];
    });
    struct QA {
      uint32_t n, nkv, G, ctx;
      uint64_t q, sel0, sel, out;
      float scale;
      uint32_t pad;
    } P{(uint32_t)n, (uint32_t)NKV, (uint32_t)G, (uint32_t)ctx, oq, os0, os, oo, 1.f / std::sqrt((float)D), 0};
    ms_stage += ms_from(t0);
    const auto t1 = hclk::now();
    id<MTLCommandBuffer> cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:g.qa];
    [en setBuffer:kb offset:0 atIndex:0];
    [en setBuffer:vb offset:0 atIndex:1];
    [en setBuffer:g.st offset:0 atIndex:2];
    [en setBytes:&P length:sizeof P atIndex:3];
    [en dispatchThreadgroups:MTLSizeMake((size_t)n * NKV, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [en endEncoding];
    const bool ok = g.commit_wait(cb);
    ms_gpu += ms_from(t1);
    flops += 2.0 * 2.0 * nsel * NKV * 16 * D;
    sets++;
    if (!ok) {
      failed++;
      return false;
    }
    const float* O = (const float*)(S + oo);
    parallel_for(pool, n, [&](int i) { std::memcpy(out[i], O + (size_t)i * NQ * D, sizeof(float) * NQ * D); });
    return true;
  }
}

bool GpuHalf::gdn(Pool& pool, int n, int NKH, int NVH, int DK, int DV, float* S, const float* cv, int C, const float* pj, int ldp, int zoff, int boff,
                  int aoff, float* out, int ldo, const float* A_log, const float* dt_bias, const float* gnorm, bool sigmoid_gate, float eps) {
  @autoreleasepool {
    HImpl& g = *(HImpl*)impl;
    if (!g.gd || DK != 128 || DV != 128 || n <= 0 || NVH % NKH) return false;
    const auto t0 = hclk::now();
    size_t off = 0;
    auto take = [&](size_t b) {
      const size_t o = off;
      off += al(b);
      return o;
    };
    const size_t sb = (size_t)NVH * DV * DK * 4;
    const size_t oS = take(sb), ocv = take((size_t)n * C * 4), opj = take((size_t)n * ldp * 4), oo = take((size_t)n * ldo * 4), oA = take(NVH * 4),
                 odt = take(NVH * 4), ogn = take(DV * 4);
    if (!g.grow(g.st, g.st_cap, off)) return false;
    uint8_t* B = (uint8_t*)g.st.contents;
    std::memcpy(B + oS, S, sb);
    parallel_for(pool, n, [&](int t) {
      std::memcpy(B + ocv + (size_t)t * C * 4, cv + (size_t)t * C, (size_t)C * 4);
      std::memcpy(B + opj + (size_t)t * ldp * 4, pj + (size_t)t * ldp, (size_t)ldp * 4);
    });
    std::memcpy(B + oA, A_log, NVH * 4);
    std::memcpy(B + odt, dt_bias, NVH * 4);
    std::memcpy(B + ogn, gnorm, DV * 4);
    struct GD {
      uint32_t n, NKH, NVH, C, ldp, zoff, boff, aoff, ldo, sig;
      float eps;
      uint32_t pad;
      uint64_t S, cv, pj, out, A, dt, gn;
    } P{(uint32_t)n, (uint32_t)NKH, (uint32_t)NVH, (uint32_t)C, (uint32_t)ldp, (uint32_t)zoff, (uint32_t)boff, (uint32_t)aoff, (uint32_t)ldo,
        sigmoid_gate ? 1u : 0u, eps, 0, oS, ocv, opj, oo, oA, odt, ogn};
    ms_stage += ms_from(t0);
    const auto t1 = hclk::now();
    id<MTLCommandBuffer> cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:g.gd];
    [en setBuffer:g.st offset:0 atIndex:0];
    [en setBytes:&P length:sizeof P atIndex:1];
    [en dispatchThreadgroups:MTLSizeMake(NVH, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    [en endEncoding];
    const bool ok = g.commit_wait(cb);
    ms_gpu += ms_from(t1);
    sets++;
    if (!ok) {
      failed++;
      return false;
    }
    std::memcpy(S, B + oS, sb);
    parallel_for(pool, n, [&](int t) { std::memcpy(out + (size_t)t * ldo, B + oo + (size_t)t * ldo * 4, (size_t)NVH * DV * 4); });
    return true;
  }
}

size_t GpuHalf::buffer_bytes() const {
  if (!impl) return 0;
  const HImpl& g = *(const HImpl*)impl;
  return g.st_cap + g.jb_cap + g.tt_cap;
}

void GpuHalf::trim() {
  if (!impl) return;
  HImpl& g = *(HImpl*)impl;
  g.st = g.jb = g.tt = nil;
  g.st_cap = g.jb_cap = g.tt_cap = 0;
}

}  // namespace dm
