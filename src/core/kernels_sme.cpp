// SME2 variant of the batched I16 kernels (kernels_batch.h), for measurement against the i8mm kernels.
// The I16 activation of an input is the 15-bit integer v = hi * 256 + lo (|v| <= 32384), and a code is 0..255, so one
// int16 SMOPA (za32: 16 rows x 16 positions x 2 inputs per instruction) gives the same exact int32 group totals as the two
// int8 planes. Per 64-row x 16-position tile the streaming part writes the integer total of every 64-input group to a
// buffer; the float epilogue then runs in normal mode with the batch-1 kernels' operation order (NEON), so the outputs are
// bit-identical to mv_aff8 / mv_aff4. Weights are widened / unpacked to int16 and interleaved per 64-row block on the fly
// (no second copy of the resident weights); activations are interleaved per 16-position tile once per call.
#include <arm_neon.h>
#include <arm_sme.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "dmeng/kernels_batch.h"

namespace dm {

static inline int perm_pos2(int i) {
  int half = i & 32, j = i & 31;
  return half + ((j & 1) ? 16 + (j >> 1) : (j >> 1));
}

// int totals T[g][64 rows][16 positions] for all K/64 groups of one 64-row x 16-position tile
// wl: [K/2][4 blocks][16 rows][2] int16 (64 rows); xl: [K/2][16 positions][2] int16
__arm_new("za") __arm_locally_streaming static void sme_tile(const int16_t* wl, const int16_t* xl, int K, int32_t* T) {
  const svbool_t ph = svptrue_b16(), p32 = svptrue_b32();
  const int NG = K / 64;
  for (int g = 0; g < NG; g++) {
    svzero_za();
    for (int kq = g * 32; kq < g * 32 + 32; kq++) {
      svint16_t b = svld1_s16(ph, xl + (size_t)kq * 32);
      const int16_t* w = wl + (size_t)kq * 128;
      svmopa_za32_s16_m(0, ph, ph, svld1_s16(ph, w), b);
      svmopa_za32_s16_m(1, ph, ph, svld1_s16(ph, w + 32), b);
      svmopa_za32_s16_m(2, ph, ph, svld1_s16(ph, w + 64), b);
      svmopa_za32_s16_m(3, ph, ph, svld1_s16(ph, w + 96), b);
    }
    int32_t* o = T + (size_t)g * 64 * 16;
    for (int r = 0; r < 16; r++) {
      svst1_hor_za32(0, r, p32, o + (0 * 16 + r) * 16);
      svst1_hor_za32(1, r, p32, o + (1 * 16 + r) * 16);
      svst1_hor_za32(2, r, p32, o + (2 * 16 + r) * 16);
      svst1_hor_za32(3, r, p32, o + (3 * 16 + r) * 16);
    }
  }
}

static inline float32x4_t f16x4s(const _Float16* p) { return vcvt_f32_f16(vld1_f16((const float16_t*)p)); }

// FMT 8: AFF8 (codes natural order), 4: AFF4 (nibbles; perm order of the activations)
template <int FMT>
static void gemm_sme(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x, int p0,
                     int p1, float* Y, size_t ldy) {
  const int NG = K / 64, KB = K / 8;
  const size_t rb = FMT == 8 ? (size_t)K : (size_t)K / 2;
  static thread_local std::vector<int16_t> wl, xl;
  static thread_local std::vector<int32_t> T;
  wl.resize((size_t)K * 64);
  xl.resize((size_t)K * 16);
  T.resize((size_t)NG * 64 * 16);
  const int8_t* XH = FMT == 8 ? x.hn : x.hp;
  const int8_t* XL = FMT == 8 ? x.ln : x.lp;
  for (int rblk = r0; rblk < r1; rblk += 64) {
    const int nr = std::min(64, r1 - rblk);
    // weights of 64 rows -> int16, interleaved [K/2][4][16][2]; for AFF4 in the perm order the activations use
    for (int rr = 0; rr < 64; rr++) {
      const int r = rblk + rr;
      int16_t* base = wl.data() + (size_t)(rr / 16) * 32 + (rr % 16) * 2;
      if (rr >= nr) {
        for (int k = 0; k < K; k++) base[(size_t)(k / 2) * 128 + (k & 1)] = 0;
        continue;
      }
      const uint8_t* c = codes + (size_t)r * rb;
      for (int k = 0; k < K; k++) {
        int v;
        if (FMT == 8) {
          v = c[k];
        } else {  // perm position k inside its group of 64: which nibble of which byte
          const int g = k / 64, j = k % 64, half = j / 32, q = j % 32;  // q < 16: low nibble of byte (half*16 + q), else high of byte (half*16 + q - 16)
          const uint8_t b = c[g * 32 + half * 16 + (q & 15)];
          v = q < 16 ? (b & 15) : (b >> 4);
        }
        base[(size_t)(k / 2) * 128 + (k & 1)] = (int16_t)v;
      }
    }
    for (int pt = p0; pt < p1; pt += 16) {
      // activations of 16 positions -> v = hi * 256 + lo, [K/2][16][2]
      for (int pp = 0; pp < 16; pp++) {
        const int p = pt + pp;
        for (int k = 0; k < K; k++) {
          int v = 0;
          if (p < x.NP) {
            const size_t o = ((size_t)(p >> 1) * KB + (k >> 3)) * 16 + (p & 1) * 8 + (k & 7);
            v = (int)XH[o] * 256 + (int)XL[o];
          }
          xl[(size_t)(k / 2) * 32 + pp * 2 + (k & 1)] = (int16_t)v;
        }
      }
      sme_tile(wl.data(), xl.data(), K, T.data());
      // epilogue: the batch-1 kernels' lane update per (row, position)
      for (int rr = 0; rr < nr; rr++) {
        const int r = rblk + rr;
        const _Float16* s = scales + (size_t)r * NG;
        const _Float16* bi = biases + (size_t)r * NG;
        for (int pp = 0; pp < 16; pp++) {
          const int p = pt + pp;
          if (p < p0 || p >= p1) continue;
          float32x4_t acc = vdupq_n_f32(0);
          for (int g = 0; g < NG; g += 4) {
            int32_t d4[4];
            for (int q = 0; q < 4; q++) d4[q] = T[((size_t)(g + q) * 64 + rr) * 16 + pp];
            float32x4_t dots = vcvtq_f32_s32(vld1q_s32(d4));
            acc = vfmaq_f32(acc, dots, vmulq_f32(f16x4s(s + g), vld1q_f32(x.d + (size_t)p * NG + g)));
            acc = vfmaq_f32(acc, f16x4s(bi + g), vld1q_f32(x.gsum + (size_t)p * NG + g));
          }
          Y[(size_t)(p - p0) * ldy + (r - r0)] = vaddvq_f32(acc);
        }
      }
    }
  }
}

void gemm_aff8_i16_sme(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x, int p0,
                       int p1, float* Y, size_t ldy) {
  gemm_sme<8>(codes, scales, biases, K, r0, r1, x, p0, p1, Y, ldy);
}
void gemm_aff4_i16_sme(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x, int p0,
                       int p1, float* Y, size_t ldy) {
  gemm_sme<4>(codes, scales, biases, K, r0, r1, x, p0, p1, Y, ldy);
}

}  // namespace dm
