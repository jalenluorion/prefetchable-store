// Copied from moved-store (branch port-ios, base unified-wip bdeb350) src/core for the P0 phone bench; unchanged.
// Batched I16 kernels for prefill (see include/dmeng/kernels_batch.h): i8mm integer group dots + the batch-1 kernels'
// own float epilogue per (row, position), so the outputs are bit-identical to mv_aff8 / mv_aff4.
#include "dmeng/kernels_batch.h"

#include <arm_neon.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dm {

static inline size_t up64b(size_t n) { return (n + 63) & ~size_t(63); }

void XQB::reserve(int K_, int N_) {
  K = K_;
  N = N_;
  NP = (N_ + 7) & ~7;
  const int NG = K / XB;
  size_t need = 2 * up64b(sizeof(float) * (size_t)NP * NG) + 4 * up64b((size_t)NP * K);
  if (need > cap) {
    std::free(mem);
    mem = std::aligned_alloc(64, need);
    cap = need;
  }
  uint8_t* m = (uint8_t*)mem;
  d = (float*)m; m += up64b(sizeof(float) * (size_t)NP * NG);
  gsum = (float*)m; m += up64b(sizeof(float) * (size_t)NP * NG);
  hn = (int8_t*)m; m += up64b((size_t)NP * K);
  ln = (int8_t*)m; m += up64b((size_t)NP * K);
  hp = (int8_t*)m; m += up64b((size_t)NP * K);
  lp = (int8_t*)m;
}

XQB::~XQB() { std::free(mem); }

static inline int perm_pos(int i) {
  int half = i & 32, j = i & 31;
  return half + ((j & 1) ? 16 + (j >> 1) : (j >> 1));
}

// byte offset of input k (in the plane's order) of position i inside a packed plane
static inline size_t pk_off(int K, int i, int k) { return ((size_t)(i >> 1) * (K / 8) + (k >> 3)) * 16 + (i & 1) * 8 + (k & 7); }

void xqb_set(XQB& q, int i, const float* x) {
  const int NG = q.K / XB;
  for (int b = 0; b < NG; b++) {
    const float* xb = x + XB * b;
    float s = 0.f, amax = 0.f;  // the same loop as xq_block (same order of the fp32 sum)
    for (int j = 0; j < XB; j++) {
      s += xb[j];
      amax = std::fmax(amax, std::fabs(xb[j]));
    }
    q.gsum[(size_t)i * NG + b] = s;
    float dd = amax / 32384.f, id = dd > 0 ? 1.f / dd : 0.f;
    q.d[(size_t)i * NG + b] = dd;
    for (int j = 0; j < XB; j++) {
      int v = (int)std::lrintf(xb[j] * id);
      v = v > 32384 ? 32384 : (v < -32384 ? -32384 : v);
      int hi = (v + 128) >> 8, lo = v - (hi << 8);
      q.hn[pk_off(q.K, i, XB * b + j)] = (int8_t)hi;
      q.ln[pk_off(q.K, i, XB * b + j)] = (int8_t)lo;
      q.hp[pk_off(q.K, i, XB * b + perm_pos(j))] = (int8_t)hi;
      q.lp[pk_off(q.K, i, XB * b + perm_pos(j))] = (int8_t)lo;
    }
  }
}

void xqb_pad(XQB& q) {
  const int NG = q.K / XB;
  for (int i = q.N; i < q.NP; i++) {
    for (int b = 0; b < NG; b++) q.d[(size_t)i * NG + b] = q.gsum[(size_t)i * NG + b] = 0.f;
    for (int k = 0; k < q.K; k++) q.hn[pk_off(q.K, i, k)] = q.ln[pk_off(q.K, i, k)] = q.hp[pk_off(q.K, i, k)] = q.lp[pk_off(q.K, i, k)] = 0;
  }
}

static inline float32x4_t f16x4(const _Float16* p) { return vcvt_f32_f16(vld1_f16((const float16_t*)p)); }

// 4 x 4 int32 transpose: in V[g] = [c0, c1, c2, c3] for g = 0..3, out W[c] = [g0, g1, g2, g3]
static inline void tr4(const int32x4_t* V, int32x4_t* W) {
  int32x4_t t0 = vtrn1q_s32(V[0], V[1]), t1 = vtrn2q_s32(V[0], V[1]);
  int32x4_t t2 = vtrn1q_s32(V[2], V[3]), t3 = vtrn2q_s32(V[2], V[3]);
  W[0] = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
  W[2] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(t0), vreinterpretq_s64_s32(t2)));
  W[1] = vreinterpretq_s32_s64(vzip1q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
  W[3] = vreinterpretq_s32_s64(vzip2q_s64(vreinterpretq_s64_s32(t1), vreinterpretq_s64_s32(t3)));
}

// Tile: 4 rows (2 row pairs) x 8 positions (4 position pairs). FMT 8 = AFF8 (USMMLA, natural order), 4 = AFF4 (SMMLA, perm).
template <int FMT>
static void gemm_tile_loop(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1,
                           const XQB& x, int p0, int p1, float* Y, size_t ldy) {
  const int NG = K / XB, KB = K / 8;
  if (NG % 4 != 0 || (r0 & 3) || (r1 & 3) || (p0 & 1)) {
    fprintf(stderr, "gemm: unsupported shape (K %d, rows %d-%d, p0 %d)\n", K, r0, r1, p0);
    abort();
  }
  const size_t rb = FMT == 8 ? (size_t)K : (size_t)K / 2;
  const int8_t* XH = FMT == 8 ? x.hn : x.hp;
  const int8_t* XL = FMT == 8 ? x.ln : x.lp;
  const int pend = (p1 + 7) & ~7;  // the padded positions exist in x (zero); their outputs are not written
  alignas(64) float32x4_t F[4][8];
  alignas(64) int32x4_t T[4][2][4];
  for (int pt = p0; pt < pend; pt += 8) {
    for (int r = r0; r < r1; r += 4) {
      const uint8_t* c[4] = {codes + (size_t)r * rb, codes + (size_t)(r + 1) * rb, codes + (size_t)(r + 2) * rb, codes + (size_t)(r + 3) * rb};
      for (int a = 0; a < 4; a++)
        for (int b = 0; b < 8; b++) F[a][b] = vdupq_n_f32(0);
      for (int g0 = 0; g0 < NG; g0 += 4) {
        for (int gg = 0; gg < 4; gg++) {
          const int g = g0 + gg;
          int32x4_t ah[2][4], al[2][4];
          for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++) ah[i][j] = al[i][j] = vdupq_n_s32(0);
          const int8_t* xh = XH + ((size_t)(pt >> 1) * KB + (size_t)g * 8) * 16;
          const int8_t* xl = XL + ((size_t)(pt >> 1) * KB + (size_t)g * 8) * 16;
          const size_t pstride = (size_t)KB * 16;  // next position pair
          if (FMT == 8) {
            for (int kb = 0; kb < 8; kb++) {
              uint8x16_t w0 = vcombine_u8(vld1_u8(c[0] + 64 * g + 8 * kb), vld1_u8(c[1] + 64 * g + 8 * kb));
              uint8x16_t w1 = vcombine_u8(vld1_u8(c[2] + 64 * g + 8 * kb), vld1_u8(c[3] + 64 * g + 8 * kb));
              for (int j = 0; j < 4; j++) {
                int8x16_t vh = vld1q_s8(xh + j * pstride + kb * 16), vl = vld1q_s8(xl + j * pstride + kb * 16);
                ah[0][j] = vusmmlaq_s32(ah[0][j], w0, vh);
                al[0][j] = vusmmlaq_s32(al[0][j], w0, vl);
                ah[1][j] = vusmmlaq_s32(ah[1][j], w1, vh);
                al[1][j] = vusmmlaq_s32(al[1][j], w1, vl);
              }
            }
          } else {
            const uint8x16_t m = vdupq_n_u8(0x0F);
            for (int half = 0; half < 2; half++) {  // bytes 0-15 then 16-31 of the group
              uint8x16_t b0 = vld1q_u8(c[0] + 32 * g + 16 * half), b1 = vld1q_u8(c[1] + 32 * g + 16 * half);
              uint8x16_t b2 = vld1q_u8(c[2] + 32 * g + 16 * half), b3 = vld1q_u8(c[3] + 32 * g + 16 * half);
              // pair rows: [row a bytes 0-7 | row b bytes 0-7] and [.. 8-15 | .. 8-15]
              uint8x16_t p0lo = vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(b0), vreinterpretq_u64_u8(b1)));
              uint8x16_t p0hi = vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(b0), vreinterpretq_u64_u8(b1)));
              uint8x16_t p1lo = vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(b2), vreinterpretq_u64_u8(b3)));
              uint8x16_t p1hi = vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(b2), vreinterpretq_u64_u8(b3)));
              // k-blocks of this half: low nibbles of bytes 0-7, 8-15, then high nibbles of bytes 0-7, 8-15
              for (int q = 0; q < 4; q++) {
                const int kb = half * 4 + q;
                uint8x16_t s0 = q == 0 ? p0lo : q == 1 ? p0hi : q == 2 ? p0lo : p0hi;
                uint8x16_t s1 = q == 0 ? p1lo : q == 1 ? p1hi : q == 2 ? p1lo : p1hi;
                int8x16_t w0 = vreinterpretq_s8_u8(q < 2 ? vandq_u8(s0, m) : vshrq_n_u8(s0, 4));
                int8x16_t w1 = vreinterpretq_s8_u8(q < 2 ? vandq_u8(s1, m) : vshrq_n_u8(s1, 4));
                for (int j = 0; j < 4; j++) {
                  int8x16_t vh = vld1q_s8(xh + j * pstride + kb * 16), vl = vld1q_s8(xl + j * pstride + kb * 16);
                  ah[0][j] = vmmlaq_s32(ah[0][j], w0, vh);
                  al[0][j] = vmmlaq_s32(al[0][j], w0, vl);
                  ah[1][j] = vmmlaq_s32(ah[1][j], w1, vh);
                  al[1][j] = vmmlaq_s32(al[1][j], w1, vl);
                }
              }
            }
          }
          for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++) T[gg][i][j] = vaddq_s32(vshlq_n_s32(ah[i][j], 8), al[i][j]);
        }
        // epilogue for groups g0..g0+3: per (row, position) the batch-1 kernel's lane update
        for (int i = 0; i < 2; i++) {
          const float32x4_t sA = f16x4(scales + (size_t)(r + 2 * i) * NG + g0), sB = f16x4(scales + (size_t)(r + 2 * i + 1) * NG + g0);
          const float32x4_t bA = f16x4(biases + (size_t)(r + 2 * i) * NG + g0), bB = f16x4(biases + (size_t)(r + 2 * i + 1) * NG + g0);
          for (int j = 0; j < 4; j++) {
            int32x4_t V[4] = {T[0][i][j], T[1][i][j], T[2][i][j], T[3][i][j]}, W[4];
            tr4(V, W);  // W[0] = (row 2i, pos 2j), W[1] = (2i, 2j+1), W[2] = (2i+1, 2j), W[3] = (2i+1, 2j+1)
            const int pa = pt + 2 * j, pb = pa + 1;
            const float32x4_t dA = vld1q_f32(x.d + (size_t)pa * NG + g0), dB = vld1q_f32(x.d + (size_t)pb * NG + g0);
            const float32x4_t gA = vld1q_f32(x.gsum + (size_t)pa * NG + g0), gB = vld1q_f32(x.gsum + (size_t)pb * NG + g0);
            float32x4_t& f0 = F[2 * i][2 * j];
            float32x4_t& f1 = F[2 * i][2 * j + 1];
            float32x4_t& f2 = F[2 * i + 1][2 * j];
            float32x4_t& f3 = F[2 * i + 1][2 * j + 1];
            f0 = vfmaq_f32(f0, vcvtq_f32_s32(W[0]), vmulq_f32(sA, dA));
            f0 = vfmaq_f32(f0, bA, gA);
            f1 = vfmaq_f32(f1, vcvtq_f32_s32(W[1]), vmulq_f32(sA, dB));
            f1 = vfmaq_f32(f1, bA, gB);
            f2 = vfmaq_f32(f2, vcvtq_f32_s32(W[2]), vmulq_f32(sB, dA));
            f2 = vfmaq_f32(f2, bB, gA);
            f3 = vfmaq_f32(f3, vcvtq_f32_s32(W[3]), vmulq_f32(sB, dB));
            f3 = vfmaq_f32(f3, bB, gB);
          }
        }
      }
      for (int b = 0; b < 8; b++) {
        const int p = pt + b;
        if (p < p0 || p >= p1) continue;
        float* y = Y + (size_t)(p - p0) * ldy + (r - r0);
        for (int a = 0; a < 4; a++) y[a] = vaddvq_f32(F[a][b]);
      }
    }
  }
}

void gemm_aff8_i16(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x,
                   int p0, int p1, float* Y, size_t ldy) {
  gemm_tile_loop<8>(codes, scales, biases, K, r0, r1, x, p0, p1, Y, ldy);
}
void gemm_aff4_i16(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x,
                   int p0, int p1, float* Y, size_t ldy) {
  gemm_tile_loop<4>(codes, scales, biases, K, r0, r1, x, p0, p1, Y, ldy);
}

}  // namespace dm
