// Exact batched matrix products on the GPU (see gpu_gemm.h). Objective-C++ with ARC (-fobjc-arc).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "dmeng/workcount.h"
#include "gpu_gemm.h"

#include <unistd.h>
#include <cstdlib>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dm {

// The arithmetic, per (row r, position q) of a matrix of K inputs in groups of 64:
//   dot_g  = sum over the group of int(code) * int(act), act = hi * 256 + lo (the XQB natural-order planes): an exact
//            int32, as the CPU's USMMLA (8-bit codes) / SMMLA (4-bit codes) sums ((hi sum) << 8 + lo sum), in any order
//            (|dot| < 64 * 255 * 32767 < 2^31)
//   lane   = g mod 4;  a = fma(float(dot), float(scale_g) * d_g, a);  a = fma(float(bias_g), gsum_g, a)   (groups in whole fours)
//   y      = (lane0 + lane1) + (lane2 + lane3)
//   tail   (K not a multiple of 256: 1-3 groups more) y = y + fma(float(dot) * float(scale_g), d_g, float(bias_g) * gsum_g)
//            in group order: the one-vector kernels' scalar tail
// which is gemm_tile_loop's epilogue (vfmaq_f32 per lane, vaddvq_f32 at the end) operation for operation, compiled with
// MTLMathModeSafe so no operation is reassociated, contracted or approximated: the same bits.
//
// One dispatch covers many products (a "set"): a tile table maps each threadgroup (16 rows x 16 positions of one
// product) to its job; a job names its weights (offsets in the weight buffer bound for the dispatch), its staged
// activations (offsets in the staging buffer), an optional position map (gathered inputs) and its output.
static const char* GSRC = R"(
#include <metal_stdlib>
using namespace metal;
struct Job { ulong co, so, bo, xh, xl, xd, xg, xm, y; uint R, K, N, fmt, tr, tbase; };
kernel void gd(device const uchar* w [[buffer(0)]], device const uchar* st [[buffer(1)]], device const Job* jobs [[buffer(2)]],
               device const uint* tj [[buffer(3)]], device float* out [[buffer(4)]], constant uint& t0 [[buffer(5)]],
               uint2 tgi [[threadgroup_position_in_grid]], uint2 lt [[thread_position_in_threadgroup]],
               uint tid [[thread_index_in_threadgroup]]) {
  const uint t = t0 + tgi.x;
  const Job j = jobs[tj[t]];
  const uint R = j.R, K = j.K, N = j.N, NG = K / 64, NG4 = NG & ~3u, KB = K / 8, local = t - j.tbase;
  const uint rt = local % j.tr, qt = local / j.tr;
  const ulong rb = j.fmt == 8 ? (ulong)K : (ulong)K / 2;
  device const uchar* codes = w + j.co;
  device const half* sc = (device const half*)(w + j.so);
  device const half* bi = (device const half*)(w + j.bo);
  device const char* XH = (device const char*)(st + j.xh);
  device const char* XL = (device const char*)(st + j.xl);
  device const float* d = (device const float*)(st + j.xd);
  device const float* gs = (device const float*)(st + j.xg);
  const bool mapped = j.xm != 0xFFFFFFFFFFFFFFFFul;
  device const uint* xm = (device const uint*)(st + (mapped ? j.xm : 0ul));
  threadgroup uchar wt[16][64];
  threadgroup short xt[16][64];
  threadgroup uint src[16];
  if (tid < 16) {
    const uint qq = qt * 16 + tid;
    src[tid] = qq < N ? (mapped ? xm[qq] : qq) : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = rt * 16 + lt.x, q = qt * 16 + lt.y, sq = src[lt.y];
  float4 acc = float4(0.0f);
  float y = 0.0f;
  for (uint g = 0; g < NG; g++) {
    for (uint k = tid; k < 1024; k += 256) {
      const uint rr = k / 64, kk = k % 64, row = rt * 16 + rr, pos = qt * 16 + rr;
      uchar c = 0;
      if (row < R) {
        if (j.fmt == 8) {
          c = codes[(ulong)row * rb + g * 64 + kk];
        } else {
          const uchar b = codes[(ulong)row * rb + g * 32 + kk / 2];
          c = (kk & 1) ? (uchar)(b >> 4) : (uchar)(b & 15);
        }
      }
      wt[rr][kk] = c;
      short v = 0;
      if (pos < N) {
        const uint s = src[rr];
        const ulong o = ((ulong)(s >> 1) * KB + g * 8 + kk / 8) * 16 + (s & 1) * 8 + (kk & 7);
        v = (short)(int(XH[o]) * 256 + int(XL[o]));
      }
      xt[rr][kk] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int dot = 0;
    for (uint kk = 0; kk < 64; kk++) dot += int(wt[lt.x][kk]) * int(xt[lt.y][kk]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r < R && q < N) {
      if (g < NG4) {
        const uint lane = g & 3;
        float m = float(sc[(ulong)r * NG + g]) * d[(ulong)sq * NG + g];
        float a = acc[lane];
        a = fma(float(dot), m, a);
        a = fma(float(bi[(ulong)r * NG + g]), gs[(ulong)sq * NG + g], a);
        acc[lane] = a;
      } else {
        if (g == NG4) y = (acc.x + acc.y) + (acc.z + acc.w);
        const float ds = float(dot) * float(sc[(ulong)r * NG + g]);
        const float bg = float(bi[(ulong)r * NG + g]) * gs[(ulong)sq * NG + g];
        y = y + fma(ds, d[(ulong)sq * NG + g], bg);
      }
    }
  }
  if (r < R && q < N) out[j.y + (ulong)q * R + r] = NG4 == NG ? (acc.x + acc.y) + (acc.z + acc.w) : y;
}

// The same products through the GPU's fp32 matrix units (simdgroup 8 x 8 multiply-accumulate). Per 64-input group the
// two plane sums hs = sum code * hi and ls = sum code * lo are formed in fp32 from exact small integers (codes 0..255,
// hi -126..127, lo -128..127, held exactly in half): every product and every partial sum is an integer of magnitude at
// most 64 * 255 * 128 < 2^24, so each fp32 multiply / add / fma on them is exact whatever the order or fusion. Then
// dot = int(hs) * 256 + int(ls) is the CPU's (sum_hi << 8) + sum_lo exactly, and the epilogue is gd's, so the outputs are
// the same bits. Tile (16 SR) rows x (16 SQ) positions, SR x SQ = 4 simdgroups of 16 x 16 outputs each.
template <uint SR, uint SQ>
static void gm_impl(device const uchar* w, device const uchar* st, device const Job* jobs, device const uint* tj, device float* out,
                    uint t, uint tid, uint sg, uint ln, threadgroup half* wt, threadgroup half* xh, threadgroup half* xl,
                    threadgroup float* cs, threadgroup uint* src) {
  constexpr uint TR = 16 * SR, TQ = 16 * SQ, NT = 128;
  const Job j = jobs[tj[t]];
  const uint R = j.R, K = j.K, N = j.N, NG = K / 64, NG4 = NG & ~3u, KB = K / 8, local = t - j.tbase;
  const uint rt = local % j.tr, qt = local / j.tr;
  const ulong rb = j.fmt == 8 ? (ulong)K : (ulong)K / 2;
  device const uchar* codes = w + j.co;
  device const half* sc = (device const half*)(w + j.so);
  device const half* bi = (device const half*)(w + j.bo);
  device const char* XH = (device const char*)(st + j.xh);
  device const char* XL = (device const char*)(st + j.xl);
  device const float* d = (device const float*)(st + j.xd);
  device const float* gs = (device const float*)(st + j.xg);
  const bool mapped = j.xm != 0xFFFFFFFFFFFFFFFFul;
  device const uint* xm = (device const uint*)(st + (mapped ? j.xm : 0ul));
  for (uint u = tid; u < TQ; u += NT) {
    const uint qq = qt * TQ + u;
    src[u] = qq < N ? (mapped ? xm[qq] : qq) : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sr = (sg % SR) * 16, sp = (sg / SR) * 16;  // this simdgroup's 16 x 16 block inside the tile
  threadgroup float* csh = cs + sg * 512;              // its plane sums: hi [16][16], then lo [16][16] (over the tiles)
  float4 acc[8];
  float yt[8];
  for (uint i = 0; i < 8; i++) acc[i] = float4(0.0f), yt[i] = 0.0f;
  for (uint g = 0; g < NG; g++) {
    for (uint k = tid; k < TR * 64; k += NT) {  // weights [row][k] (zero past R)
      const uint rr = k / 64, kk = k % 64, row = rt * TR + rr;
      uchar c = 0;
      if (row < R) {
        if (j.fmt == 8) {
          c = codes[(ulong)row * rb + g * 64 + kk];
        } else {
          const uchar b = codes[(ulong)row * rb + g * 32 + kk / 2];
          c = (kk & 1) ? (uchar)(b >> 4) : (uchar)(b & 15);
        }
      }
      wt[rr * 64 + kk] = half(c);
    }
    for (uint k = tid; k < TQ * 64; k += NT) {  // activations [k][position] (zero past N)
      const uint pp = k / 64, kk = k % 64, pos = qt * TQ + pp;
      half vh = 0, vl = 0;
      if (pos < N) {
        const uint s = src[pp];
        const ulong o = ((ulong)(s >> 1) * KB + g * 8 + kk / 8) * 16 + (s & 1) * 8 + (kk & 7);
        vh = half(XH[o]);
        vl = half(XL[o]);
      }
      xh[kk * TQ + pp] = vh;
      xl[kk * TQ + pp] = vl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 ch[2][2], cl[2][2];
    for (uint a = 0; a < 2; a++)
      for (uint b = 0; b < 2; b++) {
        ch[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        cl[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      }
    for (uint k8 = 0; k8 < 64; k8 += 8) {
      simdgroup_half8x8 A[2], BH[2], BL[2];
      for (uint a = 0; a < 2; a++) simdgroup_load(A[a], wt + (sr + 8 * a) * 64 + k8, 64);
      for (uint b = 0; b < 2; b++) {
        simdgroup_load(BH[b], xh + k8 * TQ + sp + 8 * b, TQ);
        simdgroup_load(BL[b], xl + k8 * TQ + sp + 8 * b, TQ);
      }
      for (uint a = 0; a < 2; a++)
        for (uint b = 0; b < 2; b++) {
          simdgroup_multiply_accumulate(ch[a][b], A[a], BH[b], ch[a][b]);
          simdgroup_multiply_accumulate(cl[a][b], A[a], BL[b], cl[a][b]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // every simdgroup done with the tiles: their memory takes the sums
    for (uint a = 0; a < 2; a++)
      for (uint b = 0; b < 2; b++) {
        simdgroup_store(ch[a][b], csh + (8 * a) * 16 + 8 * b, 16);
        simdgroup_store(cl[a][b], csh + 256 + (8 * a) * 16 + 8 * b, 16);
      }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; i++) {  // the thread's 8 outputs of the block: element ln + 32 i = (row e / 16, position e % 16)
      const uint e = ln + 32 * i, row = rt * TR + sr + e / 16, pl = sp + e % 16, pos = qt * TQ + pl;
      if (row < R && pos < N) {
        const int dot = int(csh[e]) * 256 + int(csh[256 + e]);
        const uint sq = src[pl], lane = g & 3;
        if (g < NG4) {
          float m = float(sc[(ulong)row * NG + g]) * d[(ulong)sq * NG + g];
          float v = acc[i][lane];
          v = fma(float(dot), m, v);
          v = fma(float(bi[(ulong)row * NG + g]), gs[(ulong)sq * NG + g], v);
          acc[i][lane] = v;
        } else {  // the scalar tail
          if (g == NG4) yt[i] = (acc[i].x + acc[i].y) + (acc[i].z + acc[i].w);
          const float ds = float(dot) * float(sc[(ulong)row * NG + g]);
          const float bg = float(bi[(ulong)row * NG + g]) * gs[(ulong)sq * NG + g];
          yt[i] = yt[i] + fma(ds, d[(ulong)sq * NG + g], bg);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = 0; i < 8; i++) {
    const uint e = ln + 32 * i, row = rt * TR + sr + e / 16, pos = qt * TQ + sp + e % 16;
    if (row < R && pos < N) out[j.y + (ulong)pos * R + row] = NG4 == NG ? (acc[i].x + acc[i].y) + (acc[i].z + acc[i].w) : yt[i];
  }
}
#define GM(NAME, SR, SQ)                                                                                                         \
  kernel void NAME(device const uchar* w [[buffer(0)]], device const uchar* st [[buffer(1)]], device const Job* jobs [[buffer(2)]], \
                   device const uint* tj [[buffer(3)]], device float* out [[buffer(4)]], constant uint& t0 [[buffer(5)]],          \
                   uint3 tgi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                          \
                   uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {                           \
    /* one area: the weight tile and both activation tiles (half), then (overlaid) the 4 simdgroups' sums (2048 floats) */   \
    threadgroup float tg[(16 * SR * 64 + 2 * 64 * 16 * SQ) / 2 > 2048 ? (16 * SR * 64 + 2 * 64 * 16 * SQ) / 2 : 2048];               \
    threadgroup uint src[16 * SQ];                                                                                                 \
    threadgroup half* th = (threadgroup half*)tg;                                                                                  \
    gm_impl<SR, SQ>(w, st, jobs, tj, out, t0 + tgi.x, tid, sg, ln, th, th + 16 * SR * 64, th + 16 * SR * 64 + 64 * 16 * SQ, tg, src); \
  }
GM(gm2x2, 2, 2)
GM(gm4x1, 4, 1)

// NVFP4 (Inkling's experts), on the matrix units: per 16-input sub-block s of 64-input block b, the integer dot of the
// E2M1 codes doubled (small integers, NV_E2M1X2) with hi * 256 + lo, formed as the two plane sums in fp32 (every product
// and partial sum an integer of magnitude at most 16 * 12 * 128 < 2^24: exact), then gemm_nvfp4_i16's epilogue
// operation for operation: v = float(dot) * (scale_s * d_b) with the E4M3 scale read as fp16 bits ((c & 0x7F) << 7 |
// (c & 0x80) << 8, i.e. e4m3 / 256); lane s of accumulator A1 for odd blocks inside a full group of four, else A0;
// y = ((A0 + A1) summed pairwise) * post, post = gmul * 128 (Job.bo carries its bits). Tile 32 rows x 32 positions.
constant char NVX2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
kernel void gn(device const uchar* w [[buffer(0)]], device const uchar* st [[buffer(1)]], device const Job* jobs [[buffer(2)]],
               device const uint* tj [[buffer(3)]], device float* out [[buffer(4)]], constant uint& t0 [[buffer(5)]],
               uint3 tgi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
               uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {
  constexpr uint SR = 2, TR = 32, TQ = 32, NT = 128;
  threadgroup half wt[TR * 16];
  threadgroup half xh[16 * TQ];
  threadgroup half xl[16 * TQ];
  threadgroup float cs[4 * 512];
  threadgroup uint src[TQ];
  const uint t = t0 + tgi.x;
  const Job j = jobs[tj[t]];
  const uint R = j.R, K = j.K, N = j.N, NB = K / 64, NS = K / 16, KB = K / 8, local = t - j.tbase;
  const uint rt = local % j.tr, qt = local / j.tr;
  const ulong rb = (ulong)K / 2;
  device const uchar* codes = w + j.co;
  device const uchar* scl = w + j.so;
  const float post = as_type<float>((uint)j.bo);
  device const char* XH = (device const char*)(st + j.xh);
  device const char* XL = (device const char*)(st + j.xl);
  device const float* d = (device const float*)(st + j.xd);
  const bool mapped = j.xm != 0xFFFFFFFFFFFFFFFFul;
  device const uint* xm = (device const uint*)(st + (mapped ? j.xm : 0ul));
  for (uint u = tid; u < TQ; u += NT) {
    const uint qq = qt * TQ + u;
    src[u] = qq < N ? (mapped ? xm[qq] : qq) : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sr = (sg % SR) * 16, sp = (sg / SR) * 16;
  threadgroup float* csh = cs + sg * 512;
  float4 a0[8], a1[8];
  for (uint i = 0; i < 8; i++) a0[i] = a1[i] = float4(0.0f);
  for (uint b = 0; b < NB; b++) {
    const bool use1 = (b & ~3u) + 4 <= NB && (b & 1u);
    for (uint s = 0; s < 4; s++) {
      const uint e0 = b * 64 + s * 16;
      for (uint k = tid; k < TR * 16; k += NT) {
        const uint rr = k / 16, kk = k % 16, row = rt * TR + rr, e = e0 + kk;
        half c = 0;
        if (row < R) {
          const uchar by = codes[(ulong)row * rb + e / 2];
          c = half(NVX2[(e & 1) ? (by >> 4) : (by & 15)]);
        }
        wt[rr * 16 + kk] = c;
      }
      for (uint k = tid; k < TQ * 16; k += NT) {
        const uint pp = k / 16, kk = k % 16, pos = qt * TQ + pp, e = e0 + kk;
        half vh = 0, vl = 0;
        if (pos < N) {
          const uint q = src[pp];
          const ulong o = ((ulong)(q >> 1) * KB + e / 8) * 16 + (q & 1) * 8 + (e & 7);
          vh = half(XH[o]);
          vl = half(XL[o]);
        }
        xh[kk * TQ + pp] = vh;
        xl[kk * TQ + pp] = vl;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_float8x8 ch[2][2], cl[2][2];
      for (uint a = 0; a < 2; a++)
        for (uint c2 = 0; c2 < 2; c2++) {
          ch[a][c2] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
          cl[a][c2] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
      for (uint k8 = 0; k8 < 16; k8 += 8) {
        simdgroup_half8x8 A[2], BH[2], BL[2];
        for (uint a = 0; a < 2; a++) simdgroup_load(A[a], wt + (sr + 8 * a) * 16 + k8, 16);
        for (uint c2 = 0; c2 < 2; c2++) {
          simdgroup_load(BH[c2], xh + k8 * TQ + sp + 8 * c2, TQ);
          simdgroup_load(BL[c2], xl + k8 * TQ + sp + 8 * c2, TQ);
        }
        for (uint a = 0; a < 2; a++)
          for (uint c2 = 0; c2 < 2; c2++) {
            simdgroup_multiply_accumulate(ch[a][c2], A[a], BH[c2], ch[a][c2]);
            simdgroup_multiply_accumulate(cl[a][c2], A[a], BL[c2], cl[a][c2]);
          }
      }
      for (uint a = 0; a < 2; a++)
        for (uint c2 = 0; c2 < 2; c2++) {
          simdgroup_store(ch[a][c2], csh + (8 * a) * 16 + 8 * c2, 16);
          simdgroup_store(cl[a][c2], csh + 256 + (8 * a) * 16 + 8 * c2, 16);
        }
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = 0; i < 8; i++) {
        const uint e = ln + 32 * i, row = rt * TR + sr + e / 16, pl = sp + e % 16, pos = qt * TQ + pl;
        if (row < R && pos < N) {
          const int dot = int(csh[e]) * 256 + int(csh[256 + e]);
          const ushort cb = scl[(ulong)row * NS + b * 4 + s];
          const float sc = float(as_type<half>((ushort)(((cb & 0x7F) << 7) | ((cb & 0x80) << 8))));
          const float m = sc * d[(ulong)src[pl] * NB + b];
          if (use1) a1[i][s] = fma(float(dot), m, a1[i][s]);  // (the CPU's vmul + vadd contract to one FMA under -ffp-contract=fast)
          else a0[i][s] = fma(float(dot), m, a0[i][s]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint i = 0; i < 8; i++) {
    const uint e = ln + 32 * i, row = rt * TR + sr + e / 16, pos = qt * TQ + sp + e % 16;
    if (row < R && pos < N) {
      const float4 tt = a0[i] + a1[i];
      out[j.y + (ulong)pos * R + row] = ((tt.x + tt.y) + (tt.z + tt.w)) * post;
    }
  }
}

// NVFP4 again, the same bits as gn with fewer steps: one fp32 product per 16-input sub-block instead of the two plane
// products. The activation enters as the integer v = hi * 256 + lo (|v| <= 32384, exact in fp32) and the code as its
// doubled E2M1 value (|c| <= 12): every product (< 2^19) and every partial sum of a sub-block (< 16 * 12 * 32384 < 2^23)
// is an integer below 2^24, so the fp32 matrix product is the exact integer sub-block dot in any order, which is
// float(dot) of gemm_nvfp4_i16. A whole 64-input block is staged per barrier (gn: 16 inputs); the sums are read from the
// simdgroup matrices' own elements (thread_elements) instead of a store to threadgroup memory; each thread's rows' four
// E4M3 scales of the block are one 32-bit load. Then gn's epilogue, operation for operation. Tile (16 SR) rows x (16 SQ)
// positions, SR x SQ = 4 simdgroups of 16 x 16 outputs; each thread holds 8 outputs (2 rows x 4 positions).
template <uint SR, uint SQ>
static void gn2_impl(device const uchar* w, device const uchar* st, device const Job* jobs, device const uint* tj, device float* out,
                     uint t, uint tid, uint sg, uint ln, threadgroup float* wt, threadgroup float* xt, threadgroup uint* src) {
  constexpr uint TR = 16 * SR, TQ = 16 * SQ, NT = 128;
  const Job j = jobs[tj[t]];
  const uint R = j.R, K = j.K, N = j.N, NB = K / 64, NS = K / 16, KB = K / 8, local = t - j.tbase;
  const uint rt = local % j.tr, qt = local / j.tr;
  const ulong rb = (ulong)K / 2;
  device const uchar* codes = w + j.co;
  device const uchar* scl = w + j.so;
  const float post = as_type<float>((uint)j.bo);
  device const char* XH = (device const char*)(st + j.xh);
  device const char* XL = (device const char*)(st + j.xl);
  device const float* d = (device const float*)(st + j.xd);
  const bool mapped = j.xm != 0xFFFFFFFFFFFFFFFFul;
  device const uint* xm = (device const uint*)(st + (mapped ? j.xm : 0ul));
  for (uint u = tid; u < TQ; u += NT) {
    const uint qq = qt * TQ + u;
    src[u] = qq < N ? (mapped ? xm[qq] : qq) : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint sr = (sg % SR) * 16, sp = (sg / SR) * 16;
  // this thread's elements of an 8 x 8 result: row fm, columns fn and fn + 1
  const uint qid = ln / 4, fm = (qid & 4) + ((ln / 2) % 4), fn = (qid & 2) * 2 + (ln % 2) * 2;
  uint rows[2], poss[4];
  for (uint a = 0; a < 2; a++) rows[a] = rt * TR + sr + 8 * a + fm;
  for (uint c = 0; c < 2; c++)
    for (uint k = 0; k < 2; k++) poss[2 * c + k] = sp + 8 * c + fn + k;  // inside the tile
  float4 a0[8], a1[8];  // output (a, c, k) at 4 a + 2 c + k
  for (uint i = 0; i < 8; i++) a0[i] = a1[i] = float4(0.0f);
  for (uint b = 0; b < NB; b++) {
    const bool use1 = (b & ~3u) + 4 <= NB && (b & 1u);
    for (uint k = tid; k < TR * 8; k += NT) {  // weights: row k / 8, codes 8 (k % 8) .. + 7 of the block (4 bytes)
      const uint rr = k / 8, wq = k % 8, row = rt * TR + rr;
      uint by4 = 0;
      if (row < R) by4 = *(device const uint*)(codes + (ulong)row * rb + b * 32 + wq * 4);
      threadgroup float* o = wt + rr * 64 + wq * 8;
      for (uint e = 0; e < 8; e++) o[e] = float(NVX2[(by4 >> (4 * e)) & 15u]);
    }
    for (uint k = tid; k < TQ * 8; k += NT) {  // activations: position k / 8, inputs 8 (k % 8) .. + 7 (8 bytes per plane)
      const uint pp = k / 8, kq = k % 8, pos = qt * TQ + pp;
      uint2 h = uint2(0), l = uint2(0);
      if (pos < N) {
        const uint q = src[pp];
        const ulong o = ((ulong)(q >> 1) * KB + b * 8 + kq) * 16 + (q & 1) * 8;
        h = *(device const uint2*)(XH + o);
        l = *(device const uint2*)(XL + o);
      }
      for (uint e = 0; e < 8; e++) {
        const uint sh = 8 * (e & 3);
        const int hv = (int)(char)(((e < 4 ? h.x : h.y) >> sh) & 255u), lv = (int)(char)(((e < 4 ? l.x : l.y) >> sh) & 255u);
        xt[(kq * 8 + e) * TQ + pp] = float(hv * 256 + lv);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 sc[2];  // the four sub-block scales of each of the thread's rows (E4M3 read as fp16 bits: e4m3 / 256)
    for (uint a = 0; a < 2; a++) {
      uint s4 = 0;
      if (rows[a] < R) s4 = *(device const uint*)(scl + (ulong)rows[a] * NS + b * 4);
      for (uint s = 0; s < 4; s++) {
        const ushort cb = (ushort)((s4 >> (8 * s)) & 255u);
        sc[a][s] = float(as_type<half>((ushort)(((cb & 0x7F) << 7) | ((cb & 0x80) << 8))));
      }
    }
    float dp[4];
    for (uint u = 0; u < 4; u++) dp[u] = qt * TQ + poss[u] < N ? d[(ulong)src[poss[u]] * NB + b] : 0.0f;
    for (uint s = 0; s < 4; s++) {
      simdgroup_float8x8 acc[2][2];
      for (uint a = 0; a < 2; a++)
        for (uint c = 0; c < 2; c++) acc[a][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint k8 = 0; k8 < 16; k8 += 8) {
        simdgroup_float8x8 A[2], B[2];
        for (uint a = 0; a < 2; a++) simdgroup_load(A[a], wt + (sr + 8 * a) * 64 + s * 16 + k8, 64);
        for (uint c = 0; c < 2; c++) simdgroup_load(B[c], xt + (s * 16 + k8) * TQ + sp + 8 * c, TQ);
        for (uint a = 0; a < 2; a++)
          for (uint c = 0; c < 2; c++) simdgroup_multiply_accumulate(acc[a][c], A[a], B[c], acc[a][c]);
      }
      for (uint a = 0; a < 2; a++)
        for (uint c = 0; c < 2; c++) {
          const float2 e2 = float2(acc[a][c].thread_elements()[0], acc[a][c].thread_elements()[1]);
          for (uint k = 0; k < 2; k++) {
            const uint i = 4 * a + 2 * c + k;
            const float m = sc[a][s] * dp[2 * c + k];
            if (use1) a1[i][s] = fma(e2[k], m, a1[i][s]);
            else a0[i][s] = fma(e2[k], m, a0[i][s]);
          }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint a = 0; a < 2; a++)
    for (uint c = 0; c < 2; c++)
      for (uint k = 0; k < 2; k++) {
        const uint i = 4 * a + 2 * c + k, row = rows[a], pos = qt * TQ + poss[2 * c + k];
        if (row < R && pos < N) {
          const float4 tt = a0[i] + a1[i];
          out[j.y + (ulong)pos * R + row] = ((tt.x + tt.y) + (tt.z + tt.w)) * post;
        }
      }
}
#define GN2(NAME, SR, SQ)                                                                                                         \
  kernel void NAME(device const uchar* w [[buffer(0)]], device const uchar* st [[buffer(1)]], device const Job* jobs [[buffer(2)]], \
                   device const uint* tj [[buffer(3)]], device float* out [[buffer(4)]], constant uint& t0 [[buffer(5)]],          \
                   uint3 tgi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],                          \
                   uint sg [[simdgroup_index_in_threadgroup]], uint ln [[thread_index_in_simdgroup]]) {                           \
    threadgroup float wt[16 * SR * 64];                                                                                            \
    threadgroup float xt[64 * 16 * SQ];                                                                                            \
    threadgroup uint src[16 * SQ];                                                                                                 \
    gn2_impl<SR, SQ>(w, st, jobs, tj, out, t0 + tgi.x, tid, sg, ln, wt, xt, src);                                                  \
  }
GN2(gn2x2, 2, 2)
GN2(gn4x1, 4, 1)

// MXFP8 (Inkling's last layers' experts): the one-vector kernel's fp32 operations replayed per output, on the GPU's
// scalar units (the sum is not exact, so not on the matrix units). mv_mxfp8 per 32-input group g: the E4M3 code as fp16
// bits ((c & 0x80) << 8 | (c & 0x7F) << 7, e4m3 / 256), times the fp16 input, added by FMLAL into lane e % 4 of a (e / 4
// even) or b (odd), elements in order from 0; a product of two fp16 values is exact in fp32 and FMLAL is fused, so each
// step is fma(code, x, lane). dot = a + b per lane, the group's value (d0 + d1) + (d2 + d3) (two pairwise adds), lane
// g % 4 of acc = fma(value, 2^(s - 127) as the bits s << 23, acc); y = ((acc0 + acc1) + (acc2 + acc3)) * 256. Groups in
// whole fours only (add_mxfp8 refuses others). Inputs: an fp16 stage (rows of K halves), positions through the map.
// Tile TR rows x TQ positions, 128 threads, each 2 rows x 4 positions; 4 groups (128 inputs) staged per barrier.
template <uint TR, uint TQ>
static void gx_impl(device const uchar* w, device const uchar* st, device const Job* jobs, device const uint* tj, device float* out,
                    uint t, uint tid, threadgroup half* wt, threadgroup half* xt, threadgroup uint* src) {
  constexpr uint NT = 128, RG = TR / 2, KC = 128;  // RG row pairs; positions in fours: NT / RG of them
  const Job j = jobs[tj[t]];
  const uint R = j.R, K = j.K, N = j.N, NG = K / 32, local = t - j.tbase;
  const uint rt = local % j.tr, qt = local / j.tr;
  device const uchar* codes = w + j.co;
  device const uchar* scl = w + j.so;
  device const half* X = (device const half*)(st + j.xh);
  const bool mapped = j.xm != 0xFFFFFFFFFFFFFFFFul;
  device const uint* xm = (device const uint*)(st + (mapped ? j.xm : 0ul));
  for (uint u = tid; u < TQ; u += NT) {
    const uint qq = qt * TQ + u;
    src[u] = qq < N ? (mapped ? xm[qq] : qq) : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r0 = 2 * (tid % RG), p0 = 4 * (tid / RG);  // inside the tile
  float4 acc[2][4];
  for (uint a = 0; a < 2; a++)
    for (uint q = 0; q < 4; q++) acc[a][q] = float4(0.0f);
  for (uint k0 = 0; k0 < K; k0 += KC) {
    for (uint k = tid; k < TR * (KC / 4); k += NT) {  // codes: row k / 32, 4 bytes
      const uint rr = k / (KC / 4), cq = k % (KC / 4), row = rt * TR + rr;
      uint c4 = 0;
      if (row < R) c4 = *(device const uint*)(codes + (ulong)row * K + k0 + cq * 4);
      for (uint e = 0; e < 4; e++) {
        const ushort c = (ushort)((c4 >> (8 * e)) & 255u);
        wt[rr * KC + cq * 4 + e] = as_type<half>((ushort)(((c & 0x80) << 8) | ((c & 0x7F) << 7)));
      }
    }
    for (uint k = tid; k < TQ * (KC / 4); k += NT) {  // inputs: position k / 32, 4 halves
      const uint pp = k / (KC / 4), cq = k % (KC / 4), pos = qt * TQ + pp;
      half4 v = half4(0.0h);
      if (pos < N) v = *(device const half4*)(X + (ulong)src[pp] * K + k0 + cq * 4);
      *(threadgroup half4*)(xt + pp * KC + cq * 4) = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint gg = 0; gg < KC / 32 && k0 + 32 * gg < K; gg++) {
      const uint g = k0 / 32 + gg;
      for (uint a = 0; a < 2; a++) {
        const uint row = rt * TR + r0 + a;
        const float es = as_type<float>((uint)(row < R ? scl[(ulong)row * NG + g] : 0) << 23);
        threadgroup const half* wr = wt + (r0 + a) * KC + 32 * gg;
        for (uint q = 0; q < 4; q++) {
          threadgroup const half* xr = xt + (p0 + q) * KC + 32 * gg;
          float la[4] = {0.0f, 0.0f, 0.0f, 0.0f}, lb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (uint e = 0; e < 32; e++) {
            const float pr = float(wr[e]), xv = float(xr[e]);
            if ((e >> 2) & 1u) lb[e & 3] = fma(pr, xv, lb[e & 3]);
            else la[e & 3] = fma(pr, xv, la[e & 3]);
          }
          const float d0 = la[0] + lb[0], d1 = la[1] + lb[1], d2 = la[2] + lb[2], d3 = la[3] + lb[3];
          const float v = (d0 + d1) + (d2 + d3);
          acc[a][q][g & 3] = fma(v, es, acc[a][q][g & 3]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint a = 0; a < 2; a++)
    for (uint q = 0; q < 4; q++) {
      const uint row = rt * TR + r0 + a, pos = qt * TQ + p0 + q;
      if (row < R && pos < N) out[j.y + (ulong)pos * R + row] = ((acc[a][q].x + acc[a][q].y) + (acc[a][q].z + acc[a][q].w)) * 256.0f;
    }
}
#define GX(NAME, TR, TQ)                                                                                                          \
  kernel void NAME(device const uchar* w [[buffer(0)]], device const uchar* st [[buffer(1)]], device const Job* jobs [[buffer(2)]], \
                   device const uint* tj [[buffer(3)]], device float* out [[buffer(4)]], constant uint& t0 [[buffer(5)]],          \
                   uint3 tgi [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {                        \
    threadgroup half wt[TR * 128];                                                                                                 \
    threadgroup half xt[TQ * 128];                                                                                                 \
    threadgroup uint src[TQ];                                                                                                      \
    gx_impl<TR, TQ>(w, st, jobs, tj, out, t0 + tgi.x, tid, wt, xt, src);                                                           \
  }
GX(gx64x16, 64, 16)
GX(gx32x32, 32, 32)
)";

namespace {
struct Job {  // the kernel's Job, same layout (9 x 8 + 6 x 4 = 96 bytes)
  uint64_t co, so, bo, xh, xl, xd, xg, xm, y;
  uint32_t R, K, N, fmt, tr, tbase;
};
static_assert(sizeof(Job) == 96, "Job layout");
constexpr uint64_t NOMAP = ~0ull;
}  // namespace

struct GImpl {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  // gd (int dots, 16 x 16 tiles), gm2x2 (matrix units, 32 x 32), gm4x1 (64 x 16), gn (NVFP4, the first kernel), gn2x2
  // (NVFP4, 32 x 32), gn4x1 (NVFP4, 64 x 16)
  id<MTLComputePipelineState> ps[8];  // + gx64x16, gx32x32 (MXFP8)
  int nv_kernel = 2;  // NVFP4: 2 the one-product kernels (gn2x2 / gn4x1 by positions), 1 gn (DMENG_GPU_NV=1: the first kernel)
  int kernel = 1;
  size_t page = 16384, max_len = 0;
  struct WB {
    id<MTLBuffer> b;
    const uint8_t* base;
    size_t len;
  };
  std::vector<WB> wb;
  struct Stage {
    const XQB* x;
    size_t oh, ol, od, og;
    std::vector<const _Float16*> rows;  // an fp16 stage (x null): rows of K halves, at oh
    int K = 0;
  };
  std::vector<Stage> stages;
  struct Map {
    const int* p;
    int n;
    size_t off;
  };
  std::vector<Map> maps;
  std::vector<Job> jobs;
  std::vector<int> job_wb, job_map, job_var;
  std::vector<uint32_t> tiles;
  struct Grp {
    int wb, var;
    uint32_t t0, nt;
  };
  std::vector<Grp> grps;
  id<MTLBuffer> st = nil, jb = nil, tt = nil, ob = nil;
  size_t st_cap = 0, jb_cap = 0, tt_cap = 0, ob_cap = 0;
  id<MTLCommandBuffer> cb = nil;
  bool warned = false;
  bool grow(id<MTLBuffer> __strong& b, size_t& cap, size_t need) {
    if (need <= cap && b) return true;
    size_t n = std::max<size_t>(need + need / 4, 1 << 16);
    b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    cap = b ? n : 0;
    return b != nil;
  }
};

bool GpuGemm::init() {
  @autoreleasepool {
    if (impl) return true;
    GImpl* g = new GImpl();
    g->dev = MTLCreateSystemDefaultDevice();
    if (!g->dev) {
      delete g;
      return false;
    }
    NSError* err = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
    if (@available(iOS 18.0, macOS 15.0, *)) {
      co.mathMode = MTLMathModeSafe;  // no reassociation, contraction or approximation
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      co.fastMathEnabled = NO;  // (iOS 17: the same rule under its older name)
#pragma clang diagnostic pop
    }
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:GSRC] options:co error:&err];
    if (!lib) {
      fprintf(stderr, "gpu_gemm: %s\n", err ? err.localizedDescription.UTF8String : "kernel compile failed");
      delete g;
      return false;
    }
    NSString* names[8] = {@"gd", @"gm2x2", @"gm4x1", @"gn", @"gn2x2", @"gn4x1", @"gx64x16", @"gx32x32"};
    for (int i = 0; i < 8; i++) g->ps[i] = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:names[i]] error:&err];
    g->q = [g->dev newCommandQueue];
    if (!g->ps[0] || !g->q || g->ps[0].maxTotalThreadsPerThreadgroup < 256) {
      delete g;
      return false;
    }
    if (!g->ps[1] || !g->ps[2] || g->ps[1].maxTotalThreadsPerThreadgroup < 128 || g->ps[2].maxTotalThreadsPerThreadgroup < 128)
      g->kernel = 0;  // no matrix variant on this device: the integer kernel
    if (const char* e = getenv("DMENG_GPU_NV")) g->nv_kernel = atoi(e) == 1 ? 1 : 2;
    if (!g->ps[4] || !g->ps[5] || g->ps[4].maxTotalThreadsPerThreadgroup < 128 || g->ps[5].maxTotalThreadsPerThreadgroup < 128) g->nv_kernel = 1;
    g->page = (size_t)getpagesize();
    g->max_len = (size_t)g->dev.maxBufferLength;
    impl = g;
    return true;
  }
}

GpuGemm::~GpuGemm() {
  if (!impl) return;
  GImpl* g = (GImpl*)impl;
  if (g->cb) [g->cb waitUntilCompleted];
  delete g;
}

bool GpuGemm::wrap(const void* base, size_t bytes) {
  @autoreleasepool {
    if (!impl || !base || !bytes) return false;
    GImpl& g = *(GImpl*)impl;
    if ((uintptr_t)base % g.page) return false;
    const size_t len = (bytes + g.page - 1) / g.page * g.page;
    // one buffer if the device allows it; else overlapping windows (stride half a window), so any range of up to half
    // a window lies inside one of them
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

static int find_wb(const GImpl& g, const void* p, size_t n) {
  const uint8_t* b = (const uint8_t*)p;
  for (size_t i = 0; i < g.wb.size(); i++)
    if (b >= g.wb[i].base && b + n <= g.wb[i].base + g.wb[i].len) return (int)i;
  return -1;
}

bool GpuGemm::holds(const void* p, size_t n) const { return impl && find_wb(*(const GImpl*)impl, p, n) >= 0; }

void GpuGemm::touch() {
  if (!impl) return;
  GImpl& g = *(GImpl*)impl;
  id<MTLBuffer> o = [g.dev newBufferWithLength:16384 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [g.q commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  for (auto& w : g.wb) [bl copyFromBuffer:w.b sourceOffset:0 toBuffer:o destinationOffset:0 size:16];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

size_t GpuGemm::wrapped_bytes() const {
  if (!impl) return 0;
  size_t s = 0;
  for (auto& w : ((const GImpl*)impl)->wb) s += w.len;
  return s;
}

void GpuGemm::begin() {
  GImpl& g = *(GImpl*)impl;
  g.stages.clear();
  g.maps.clear();
  g.jobs.clear();
  g.job_wb.clear();
  g.job_map.clear();
  g.job_var.clear();
  g.tiles.clear();
  g.grps.clear();
}

int GpuGemm::stage(const XQB& x) {
  GImpl& g = *(GImpl*)impl;
  for (size_t i = 0; i < g.stages.size(); i++)
    if (g.stages[i].x == &x) return (int)i;
  DM_WC("gpu.stage_pos", x.N);
  g.stages.push_back({&x, 0, 0, 0, 0, {}, 0});
  return (int)g.stages.size() - 1;
}

int GpuGemm::stage_f16(const _Float16* const* rows, int n, int K) {
  GImpl& g = *(GImpl*)impl;
  if (n <= 0 || K <= 0 || K % 4) return -1;
  GImpl::Stage S{nullptr, 0, 0, 0, 0, std::vector<const _Float16*>(rows, rows + n), K};
  g.stages.push_back(std::move(S));
  return (int)g.stages.size() - 1;
}

int GpuGemm::add_mxfp8(const uint8_t* codes, const uint8_t* scales_e8m0, int K, int R, int n, int s, const int* map) {
  GImpl& g = *(GImpl*)impl;
  if (R <= 0 || n <= 0 || K % 128 || s < 0 || s >= (int)g.stages.size() || g.stages[s].x || g.stages[s].K != K || !g.ps[6] || !g.ps[7] ||
      g.ps[6].maxTotalThreadsPerThreadgroup < 128 || g.ps[7].maxTotalThreadsPerThreadgroup < 128)
    return -1;
  const size_t cb = (size_t)R * K, sb = (size_t)R * (K / 32);
  int w = -1;
  for (size_t i = 0; i < g.wb.size() && w < 0; i++) {
    const GImpl::WB& b = g.wb[i];
    auto in = [&](const void* p, size_t nb) { return (const uint8_t*)p >= b.base && (const uint8_t*)p + nb <= b.base + b.len; };
    if (in(codes, cb) && in(scales_e8m0, sb)) w = (int)i;
  }
  if (w < 0) return -1;
  const uint8_t* B = g.wb[w].base;
  Job j{};
  j.co = (uint64_t)(codes - B);
  j.so = (uint64_t)(scales_e8m0 - B);
  j.xm = (uint64_t)s;
  j.R = (uint32_t)R;
  j.K = (uint32_t)K;
  j.N = (uint32_t)n;
  j.fmt = 32;
  const int var = n <= 16 ? 6 : 7;
  j.tr = (uint32_t)((R + (var == 6 ? 63 : 31)) / (var == 6 ? 64 : 32));
  int mi = -1;
  if (map) {
    for (size_t i = 0; i < g.maps.size() && mi < 0; i++)
      if (g.maps[i].p == map && g.maps[i].n == n) mi = (int)i;
    if (mi < 0) {
      g.maps.push_back({map, n, 0});
      mi = (int)g.maps.size() - 1;
    }
  }
  g.job_var.push_back(var);
  g.jobs.push_back(j);
  g.job_wb.push_back(w);
  g.job_map.push_back(mi);
  return (int)g.jobs.size() - 1;
}

int GpuGemm::add(int fmt, const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int R, int n, int s,
                 const int* map) {
  if (fmt == 8) {
    DM_WC("gpu.aff8.row_pos", (uint64_t)R * n);
    DM_WC("gpu.aff8.adds", 1);
  } else {
    DM_WC("gpu.aff4.row_pos", (uint64_t)R * n);
    DM_WC("gpu.aff4.adds", 1);
  }
  GImpl& g = *(GImpl*)impl;
  if (R <= 0 || n <= 0 || K % 64 || (fmt != 8 && fmt != 4) || s < 0) return -1;
  const size_t NG = K / 64, cb = (size_t)R * (fmt == 8 ? K : K / 2), sb = (size_t)R * NG * 2;
  // one buffer holding all three (the windows overlap, so try each that holds the codes)
  int w = -1;
  for (size_t i = 0; i < g.wb.size() && w < 0; i++) {
    const GImpl::WB& b = g.wb[i];
    auto in = [&](const void* p, size_t nb) { return (const uint8_t*)p >= b.base && (const uint8_t*)p + nb <= b.base + b.len; };
    if (in(codes, cb) && in(scales, sb) && in(biases, sb)) w = (int)i;
  }
  if (w < 0) return -1;
  const uint8_t* B = g.wb[w].base;
  Job j{};
  j.co = (uint64_t)(codes - B);
  j.so = (uint64_t)((const uint8_t*)scales - B);
  j.bo = (uint64_t)((const uint8_t*)biases - B);
  j.xm = (uint64_t)s;  // the stage index until prepare() writes the offsets
  j.R = (uint32_t)R;
  j.K = (uint32_t)K;
  j.N = (uint32_t)n;
  j.fmt = (uint32_t)fmt;
  const int var = g.kernel == 0 ? 0 : n <= 16 ? 2 : 1;
  static const int TRv[3] = {16, 32, 64};
  j.tr = (uint32_t)((R + TRv[var] - 1) / TRv[var]);
  int mi = -1;
  if (map) {
    for (size_t i = 0; i < g.maps.size() && mi < 0; i++)
      if (g.maps[i].p == map && g.maps[i].n == n) mi = (int)i;
    if (mi < 0) {
      g.maps.push_back({map, n, 0});
      mi = (int)g.maps.size() - 1;
    }
  }
  g.job_var.push_back(var);
  g.jobs.push_back(j);
  g.job_wb.push_back(w);
  g.job_map.push_back(mi);
  return (int)g.jobs.size() - 1;
}

int GpuGemm::add_nvfp4(const uint8_t* codes, const uint8_t* scales_e4m3, float gmul, int K, int R, int n, int s, const int* map) {
  GImpl& g = *(GImpl*)impl;
  if (R <= 0 || n <= 0 || K % 64 || s < 0 || !g.ps[3] || g.ps[3].maxTotalThreadsPerThreadgroup < 128) return -1;
  const size_t cb = (size_t)R * (K / 2), sb = (size_t)R * (K / 16);
  int w = -1;
  for (size_t i = 0; i < g.wb.size() && w < 0; i++) {
    const GImpl::WB& b = g.wb[i];
    auto in = [&](const void* p, size_t nb) { return (const uint8_t*)p >= b.base && (const uint8_t*)p + nb <= b.base + b.len; };
    if (in(codes, cb) && in(scales_e4m3, sb)) w = (int)i;
  }
  if (w < 0) return -1;
  const uint8_t* B = g.wb[w].base;
  Job j{};
  j.co = (uint64_t)(codes - B);
  j.so = (uint64_t)(scales_e4m3 - B);
  const float post = gmul * 128.f;  // as gemm_nvfp4_i16
  uint32_t pb;
  std::memcpy(&pb, &post, 4);
  j.bo = pb;
  j.xm = (uint64_t)s;
  j.R = (uint32_t)R;
  j.K = (uint32_t)K;
  j.N = (uint32_t)n;
  j.fmt = 16;
  const int var = g.nv_kernel == 1 ? 3 : n <= 16 ? 5 : 4;
  j.tr = (uint32_t)((R + (var == 5 ? 63 : 31)) / (var == 5 ? 64 : 32));
  int mi = -1;
  if (map) {
    for (size_t i = 0; i < g.maps.size() && mi < 0; i++)
      if (g.maps[i].p == map && g.maps[i].n == n) mi = (int)i;
    if (mi < 0) {
      g.maps.push_back({map, n, 0});
      mi = (int)g.maps.size() - 1;
    }
  }
  g.job_var.push_back(var);
  g.jobs.push_back(j);
  g.job_wb.push_back(w);
  g.job_map.push_back(mi);
  return (int)g.jobs.size() - 1;
}

int GpuGemm::prepare() {
  GImpl& g = *(GImpl*)impl;
  if (g.jobs.empty()) return 0;
  size_t off = 0;
  auto take = [&](size_t n) {
    size_t o = off;
    off += (n + 255) & ~size_t(255);
    return o;
  };
  for (GImpl::Stage& S : g.stages) {
    if (!S.x) {
      S.oh = take(S.rows.size() * (size_t)S.K * 2);
      continue;
    }
    const XQB& x = *S.x;
    const size_t NG = x.K / 64;
    S.oh = take((size_t)x.NP * x.K);
    S.ol = take((size_t)x.NP * x.K);
    S.od = take((size_t)x.NP * NG * 4);
    S.og = take((size_t)x.NP * NG * 4);
  }
  for (GImpl::Map& m : g.maps) m.off = take((size_t)m.n * 4);
  const size_t st_need = off;
  // outputs, then the tile table grouped by weight buffer
  uint64_t y = 0;
  std::vector<int> order(g.jobs.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return g.job_wb[a] != g.job_wb[b] ? g.job_wb[a] < g.job_wb[b] : g.job_var[a] < g.job_var[b];
  });
  static const uint32_t TQv[8] = {16, 32, 16, 32, 32, 16, 16, 32};
  for (size_t i = 0; i < g.jobs.size(); i++) {
    Job& j = g.jobs[i];
    const GImpl::Stage& S = g.stages[(size_t)j.xm];
    j.xh = S.oh;
    j.xl = S.ol;
    j.xd = S.od;
    j.xg = S.og;
    j.xm = g.job_map[i] >= 0 ? g.maps[g.job_map[i]].off : NOMAP;
    j.y = y;
    y += (uint64_t)j.N * j.R;
  }
  for (int i : order) {
    Job& j = g.jobs[i];
    const int var = g.job_var[i];
    const uint32_t nt = j.tr * ((j.N + TQv[var] - 1) / TQv[var]);
    j.tbase = (uint32_t)g.tiles.size();
    if (g.grps.empty() || g.grps.back().wb != g.job_wb[i] || g.grps.back().var != var) g.grps.push_back({g.job_wb[i], var, j.tbase, 0});
    g.grps.back().nt += nt;
    for (uint32_t t = 0; t < nt; t++) g.tiles.push_back((uint32_t)i);
  }
  if (!g.grow(g.st, g.st_cap, st_need) || !g.grow(g.ob, g.ob_cap, y * 4) || !g.grow(g.jb, g.jb_cap, g.jobs.size() * sizeof(Job)) ||
      !g.grow(g.tt, g.tt_cap, g.tiles.size() * 4)) {
    g.jobs.clear();
    return 0;
  }
  std::memcpy(g.jb.contents, g.jobs.data(), g.jobs.size() * sizeof(Job));
  std::memcpy(g.tt.contents, g.tiles.data(), g.tiles.size() * 4);
  return (int)g.stages.size() + 1;
}

void GpuGemm::copy(int i) {
  GImpl& g = *(GImpl*)impl;
  if (i == (int)g.stages.size()) {  // the position maps (small)
    for (const GImpl::Map& m : g.maps) {
      uint32_t* d = (uint32_t*)((uint8_t*)g.st.contents + m.off);
      for (int t = 0; t < m.n; t++) d[t] = (uint32_t)m.p[t];
    }
    return;
  }
  const GImpl::Stage& S = g.stages[i];
  if (!S.x) {
    uint8_t* d = (uint8_t*)g.st.contents + S.oh;
    for (size_t r = 0; r < S.rows.size(); r++) std::memcpy(d + r * (size_t)S.K * 2, S.rows[r], (size_t)S.K * 2);
    return;
  }
  const XQB& x = *S.x;
  const size_t NG = x.K / 64;
  uint8_t* d = (uint8_t*)g.st.contents;
  std::memcpy(d + S.oh, x.hn, (size_t)x.NP * x.K);
  std::memcpy(d + S.ol, x.ln, (size_t)x.NP * x.K);
  std::memcpy(d + S.od, x.d, (size_t)x.NP * NG * 4);
  std::memcpy(d + S.og, x.gsum, (size_t)x.NP * NG * 4);
}

void GpuGemm::submit() {
  DM_WC("gpu.sets", 1);
  @autoreleasepool {
    GImpl& g = *(GImpl*)impl;
    if (g.jobs.empty()) return;
    g.cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> en = [g.cb computeCommandEncoder];
    [en setBuffer:g.st offset:0 atIndex:1];
    [en setBuffer:g.jb offset:0 atIndex:2];
    [en setBuffer:g.tt offset:0 atIndex:3];
    [en setBuffer:g.ob offset:0 atIndex:4];
    for (const GImpl::Grp& G : g.grps) {
      [en setComputePipelineState:g.ps[G.var]];
      [en setBuffer:g.wb[G.wb].b offset:0 atIndex:0];
      const MTLSize tpg = G.var == 0 ? MTLSizeMake(16, 16, 1) : MTLSizeMake(128, 1, 1);  // (gn: 128 as well)
      for (uint32_t a = 0; a < G.nt; a += 65535) {
        const uint32_t t0 = G.t0 + a, nt = std::min<uint32_t>(65535, G.nt - a);
        [en setBytes:&t0 length:4 atIndex:5];
        [en dispatchThreadgroups:MTLSizeMake(nt, 1, 1) threadsPerThreadgroup:tpg];
      }
    }
    [en endEncoding];
    [g.cb commit];
  }
}

bool GpuGemm::finish() {
  @autoreleasepool {
    GImpl& g = *(GImpl*)impl;
    if (!g.cb) return true;
    [g.cb waitUntilCompleted];
    const bool ok = g.cb.status == MTLCommandBufferStatusCompleted;
    if (!ok && !g.warned) {
      fprintf(stderr, "gpu_gemm: a set did not complete (%s); its rows are computed on the CPU\n",
              g.cb.error ? g.cb.error.localizedDescription.UTF8String : "no error given");
      g.warned = true;
    }
    g.cb = nil;
    return ok;
  }
}

const float* GpuGemm::out(int t) const {
  const GImpl& g = *(const GImpl*)impl;
  return (const float*)g.ob.contents + g.jobs[t].y;
}

void GpuGemm::set_kernel(int k) {
  GImpl& g = *(GImpl*)impl;
  g.kernel = k != 0 && g.ps[1] && g.ps[2] ? 1 : 0;
}

int GpuGemm::kernel() const { return impl ? ((const GImpl*)impl)->kernel : -1; }

void GpuGemm::set_nv_kernel(int k) {
  GImpl& g = *(GImpl*)impl;
  g.nv_kernel = k == 1 || !g.ps[4] || !g.ps[5] ? 1 : 2;
}

int GpuGemm::nv_kernel() const { return impl ? ((const GImpl*)impl)->nv_kernel : -1; }

size_t GpuGemm::buffer_bytes() const {
  if (!impl) return 0;
  const GImpl& g = *(const GImpl*)impl;
  return g.st_cap + g.jb_cap + g.tt_cap + g.ob_cap;
}

void GpuGemm::trim() {
  if (!impl) return;
  GImpl& g = *(GImpl*)impl;
  if (g.cb) [g.cb waitUntilCompleted];
  g.cb = nil;
  g.st = g.jb = g.tt = g.ob = nil;
  g.st_cap = g.jb_cap = g.tt_cap = g.ob_cap = 0;
}

int GpuGemm::tickets() const { return impl ? (int)((const GImpl*)impl)->jobs.size() : 0; }

}  // namespace dm
