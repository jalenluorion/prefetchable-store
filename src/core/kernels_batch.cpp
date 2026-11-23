// Batched I16 kernels for prefill (see include/dmeng/kernels_batch.h): i8mm integer group dots + the batch-1 kernels'
// own float epilogue per (row, position), so the outputs are bit-identical to mv_aff8 / mv_aff4.
#include "dmeng/kernels_batch.h"

#include <arm_neon.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

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

// ------------------------------------------------------------------------------------------------ NVFP4 (Inkling experts)
void xqb_set_from(XQB& q, int i, const XQ& x) {
  const int NG = q.K / XB;
  std::memcpy(q.d + (size_t)i * NG, x.d, sizeof(float) * NG);
  std::memcpy(q.gsum + (size_t)i * NG, x.gsum, sizeof(float) * NG);
  const size_t base = (size_t)(i >> 1) * (q.K / 8) * 16 + (i & 1) * 8;
  for (int kb = 0; kb < q.K / 8; kb++) {  // 8-byte chunks into the pair-interleaved planes
    const size_t o = base + (size_t)kb * 16;
    std::memcpy(q.hn + o, x.hn + 8 * kb, 8);
    std::memcpy(q.ln + o, x.ln + 8 * kb, 8);
    std::memcpy(q.hp + o, x.hp + 8 * kb, 8);
    std::memcpy(q.lp + o, x.lp + 8 * kb, 8);
  }
}

static const int8_t NV_E2M1X2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static inline float32x4_t nv_e4m3x4(const uint8_t* p) {  // = kernels.cpp e4m3x4
  uint32_t w;
  std::memcpy(&w, p, 4);
  uint16x4_t u = vget_low_u16(vmovl_u8(vcreate_u8(w)));
  uint16x4_t f = vorr_u16(vshl_n_u16(vand_u16(u, vdup_n_u16(0x7F)), 7), vshl_n_u16(vand_u16(u, vdup_n_u16(0x80)), 8));
  return vcvt_f32_f16(vreinterpret_f16_u16(f));
}
static inline void nv_e4m3x16(const uint8_t* p, float32x4_t out[4]) {  // = kernels.cpp e4m3x16
  uint8x16_t v = vld1q_u8(p);
  const uint16x8_t m7 = vdupq_n_u16(0x7F), m8 = vdupq_n_u16(0x80);
  uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_high_u8(v);
  lo = vorrq_u16(vshlq_n_u16(vandq_u16(lo, m7), 7), vshlq_n_u16(vandq_u16(lo, m8), 8));
  hi = vorrq_u16(vshlq_n_u16(vandq_u16(hi, m7), 7), vshlq_n_u16(vandq_u16(hi, m8), 8));
  float16x8_t a = vreinterpretq_f16_u16(lo), b = vreinterpretq_f16_u16(hi);
  out[0] = vcvt_f32_f16(vget_low_f16(a));
  out[1] = vcvt_high_f32_f16(a);
  out[2] = vcvt_f32_f16(vget_low_f16(b));
  out[3] = vcvt_high_f32_f16(b);
}

void gemm_nvfp4_i16(const uint8_t* codes, const uint8_t* sc, float gmul, int K, int r0, int r1, const XQB& x, int p0, int p1,
                    float* Y, size_t ldy) {
  const float post = gmul * 128.f;  // as mv_nvfp4
  const int NB = K / XB, rb = K / 2, rs = K / 16, KB = K / 8, NG = K / XB;
  const int8x16_t lut = vld1q_s8(NV_E2M1X2);
  const uint8x16_t m = vdupq_n_u8(0x0F);
  const int q0 = p0 >> 1, q1 = (p1 + 1) >> 1;  // position pairs
  const size_t pstride = (size_t)KB * 16;
  for (int qs = q0; qs < q1; qs += 4) {  // up to 4 pairs (8 positions) per tile
    const int nq = std::min(4, q1 - qs);
    int r = r0;
    for (; r + 2 <= r1; r += 2) {
      const uint8_t* ca = codes + (size_t)r * rb;
      const uint8_t* cb = ca + rb;
      const uint8_t* sa = sc + (size_t)r * rs;
      const uint8_t* sbp = sa + rs;
      float32x4_t A0[2][8], A1[2][8];
      for (int a = 0; a < 2; a++)
        for (int i = 0; i < 2 * nq; i++) A0[a][i] = A1[a][i] = vdupq_n_f32(0);
      float32x4_t svA[4], svB[4];
      for (int b = 0; b < NB; b++) {
        const bool full = (b & ~3) + 4 <= NB;
        if (full && (b & 3) == 0) {
          nv_e4m3x16(sa + 4 * b, svA);
          nv_e4m3x16(sbp + 4 * b, svB);
        }
        const float32x4_t scA = full ? svA[b & 3] : nv_e4m3x4(sa + 4 * b), scB = full ? svB[b & 3] : nv_e4m3x4(sbp + 4 * b);
        int8x16_t W[8];  // row pair, perm-order 8-input chunks: low nibbles of bytes 0-7, 8-15, high of 0-7, 8-15; then 16-31
        for (int h = 0; h < 2; h++) {
          uint8x16_t va = vld1q_u8(ca + 32 * b + 16 * h), vb = vld1q_u8(cb + 32 * b + 16 * h);
          uint8x16_t plo = vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(va), vreinterpretq_u64_u8(vb)));
          uint8x16_t phi = vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(va), vreinterpretq_u64_u8(vb)));
          W[4 * h + 0] = vqtbl1q_s8(lut, vandq_u8(plo, m));
          W[4 * h + 1] = vqtbl1q_s8(lut, vandq_u8(phi, m));
          W[4 * h + 2] = vqtbl1q_s8(lut, vshrq_n_u8(plo, 4));
          W[4 * h + 3] = vqtbl1q_s8(lut, vshrq_n_u8(phi, 4));
        }
        for (int j = 0; j < nq; j++) {
          const int8_t* xh = x.hp + (size_t)(qs + j) * pstride + (size_t)b * 8 * 16;
          const int8_t* xl = x.lp + (size_t)(qs + j) * pstride + (size_t)b * 8 * 16;
          int32x4_t T[4];
          for (int s = 0; s < 4; s++) {  // sub-block s = chunks (s / 2) * 4 + s % 2 and that + 2
            const int k0 = (s >> 1) * 4 + (s & 1), k1 = k0 + 2;
            int32x4_t hi = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), W[k0], vld1q_s8(xh + 16 * k0)), W[k1], vld1q_s8(xh + 16 * k1));
            int32x4_t lo = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), W[k0], vld1q_s8(xl + 16 * k0)), W[k1], vld1q_s8(xl + 16 * k1));
            T[s] = vaddq_s32(vshlq_n_s32(hi, 8), lo);  // [row a pos a, row a pos b, row b pos a, row b pos b]
          }
          int32x4_t D[4];
          tr4(T, D);
          const int pa = 2 * (qs + j), pb = pa + 1;
          const float da = x.d[(size_t)pa * NG + b], db = x.d[(size_t)pb * NG + b];
          const float32x4_t v0 = vmulq_f32(vcvtq_f32_s32(D[0]), vmulq_n_f32(scA, da));
          const float32x4_t v1 = vmulq_f32(vcvtq_f32_s32(D[1]), vmulq_n_f32(scA, db));
          const float32x4_t v2 = vmulq_f32(vcvtq_f32_s32(D[2]), vmulq_n_f32(scB, da));
          const float32x4_t v3 = vmulq_f32(vcvtq_f32_s32(D[3]), vmulq_n_f32(scB, db));
          float32x4_t(*A)[8] = (full && (b & 1)) ? A1 : A0;  // the batch-1 kernel: blocks b, b + 2 -> acc0, b + 1, b + 3 -> acc1
          A[0][2 * j] = vaddq_f32(A[0][2 * j], v0);
          A[0][2 * j + 1] = vaddq_f32(A[0][2 * j + 1], v1);
          A[1][2 * j] = vaddq_f32(A[1][2 * j], v2);
          A[1][2 * j + 1] = vaddq_f32(A[1][2 * j + 1], v3);
        }
      }
      for (int i = 0; i < 2 * nq; i++) {
        const int p = 2 * qs + i;
        if (p < p0 || p >= p1) continue;
        Y[(size_t)(p - p0) * ldy + (r - r0)] = vaddvq_f32(vaddq_f32(A0[0][i], A1[0][i])) * post;
        Y[(size_t)(p - p0) * ldy + (r + 1 - r0)] = vaddvq_f32(vaddq_f32(A0[1][i], A1[1][i])) * post;
      }
    }
    if (r < r1) {  // an odd last row: the batch-1 kernel on each position's own planes is not available here; scalar-exact
      // path: unpack the position's planes back to an XQ view (perm planes are pair-interleaved: copy 8-byte chunks)
      std::vector<int8_t> hp(K), lp(K);
      XQ v;
      v.K = K;
      v.prec = Prec::I16;
      for (int p = std::max(p0, 2 * qs); p < std::min(p1, 2 * (qs + nq)); p++) {
        for (int kb = 0; kb < KB; kb++) {
          const size_t o = (size_t)(p >> 1) * pstride + (size_t)kb * 16 + (p & 1) * 8;
          std::memcpy(hp.data() + 8 * kb, x.hp + o, 8);
          std::memcpy(lp.data() + 8 * kb, x.lp + o, 8);
        }
        v.hp = hp.data();
        v.lp = lp.data();
        v.d = x.d + (size_t)p * NG;
        v.gsum = x.gsum + (size_t)p * NG;
        mv_nvfp4(codes, sc, gmul, K, r, r + 1, v, Y + (size_t)(p - p0) * ldy + (r - r0));
      }
    }
  }
}


// ------------------------------------------------------------------------------------------------ multi-vector matvecs
// The weights of a row are loaded once and dotted with up to 8 input vectors; each vector gets exactly the arithmetic
// (same operations, same order) of the one-vector kernel (kernels.cpp, I16 activations; MXFP8 on the fp16 plane), so a
// position computed with others is bit-identical to it computed alone. ys[i][r - r0] = W[r] . x_i.
static inline float32x4_t mvm_f16x4(const _Float16* p) { return vcvt_f32_f16(vld1_f16((const float16_t*)p)); }

// ------------------------------------------------------------------------------------------------ AFF8, I16
void mv_aff8_multi(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ* const* xs,
               int n, float* const* ys) {
  const int NG = K / XB;
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * K;
    const _Float16* s = scales + (size_t)r * NG;
    const _Float16* bi = biases + (size_t)r * NG;
    float32x4_t acc[8];
    for (int i = 0; i < n; i++) acc[i] = vdupq_n_f32(0);
    int g = 0;
    for (; g + 4 <= NG; g += 4) {
      uint8x16_t v[4][4];
      for (int q = 0; q < 4; q++)
        for (int k = 0; k < 4; k++) v[q][k] = vld1q_u8(c + 64 * (g + q) + 16 * k);
      const float32x4_t sv = mvm_f16x4(s + g), bv = mvm_f16x4(bi + g);
      for (int i = 0; i < n; i++) {
        const XQ& x = *xs[i];
        int32x4_t a[4];
        for (int q = 0; q < 4; q++) {
          const int8_t* xh = x.hn + XB * (g + q);
          const int8_t* xl = x.ln + XB * (g + q);
          int32x4_t hi = vdupq_n_s32(0), lo = vdupq_n_s32(0);
          for (int k = 0; k < 4; k++) {
            hi = vusdotq_s32(hi, v[q][k], vld1q_s8(xh + 16 * k));
            lo = vusdotq_s32(lo, v[q][k], vld1q_s8(xl + 16 * k));
          }
          a[q] = vaddq_s32(vshlq_n_s32(hi, 8), lo);
        }
        float32x4_t dots = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(a[0], a[1]), vpaddq_s32(a[2], a[3])));
        acc[i] = vfmaq_f32(acc[i], dots, vmulq_f32(sv, vld1q_f32(x.d + g)));
        acc[i] = vfmaq_f32(acc[i], bv, vld1q_f32(x.gsum + g));
      }
    }
    for (int i = 0; i < n; i++) {
      const XQ& x = *xs[i];
      float t = vaddvq_f32(acc[i]);
      for (int gg = g; gg < NG; gg++) {
        int32x4_t hi = vdupq_n_s32(0), lo = vdupq_n_s32(0);
        for (int k = 0; k < 4; k++) {
          uint8x16_t vv = vld1q_u8(c + 64 * gg + 16 * k);
          hi = vusdotq_s32(hi, vv, vld1q_s8(x.hn + XB * gg + 16 * k));
          lo = vusdotq_s32(lo, vv, vld1q_s8(x.ln + XB * gg + 16 * k));
        }
        int32x4_t a = vaddq_s32(vshlq_n_s32(hi, 8), lo);
        t += (float)vaddvq_s32(a) * (float)s[gg] * x.d[gg] + (float)bi[gg] * x.gsum[gg];
      }
      ys[i][r - r0] = t;
    }
  }
}

// ------------------------------------------------------------------------------------------------ NVFP4, I16
static const int8_t MVM_E2M1X2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static inline float32x4_t mvm_e4m3x4(const uint8_t* p) {
  uint32_t w;
  std::memcpy(&w, p, 4);
  uint16x4_t u = vget_low_u16(vmovl_u8(vcreate_u8(w)));
  uint16x4_t f = vorr_u16(vshl_n_u16(vand_u16(u, vdup_n_u16(0x7F)), 7), vshl_n_u16(vand_u16(u, vdup_n_u16(0x80)), 8));
  return vcvt_f32_f16(vreinterpret_f16_u16(f));
}
static inline void mvm_e4m3x16(const uint8_t* p, float32x4_t out[4]) {
  uint8x16_t v = vld1q_u8(p);
  const uint16x8_t m7 = vdupq_n_u16(0x7F), m8 = vdupq_n_u16(0x80);
  uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_high_u8(v);
  lo = vorrq_u16(vshlq_n_u16(vandq_u16(lo, m7), 7), vshlq_n_u16(vandq_u16(lo, m8), 8));
  hi = vorrq_u16(vshlq_n_u16(vandq_u16(hi, m7), 7), vshlq_n_u16(vandq_u16(hi, m8), 8));
  float16x8_t a = vreinterpretq_f16_u16(lo), b = vreinterpretq_f16_u16(hi);
  out[0] = vcvt_f32_f16(vget_low_f16(a));
  out[1] = vcvt_high_f32_f16(a);
  out[2] = vcvt_f32_f16(vget_low_f16(b));
  out[3] = vcvt_high_f32_f16(b);
}
struct MvmNvBlk {  // one 64-block of codes, decoded once
  int8x16_t l0, h0, l1, h1;
};
static inline MvmNvBlk mvm_nv_decode(const uint8_t* c, int8x16_t lut) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  return {vqtbl1q_s8(lut, vandq_u8(v0, m)), vqtbl1q_s8(lut, vshrq_n_u8(v0, 4)), vqtbl1q_s8(lut, vandq_u8(v1, m)),
          vqtbl1q_s8(lut, vshrq_n_u8(v1, 4))};
}
static inline int32x4_t mvm_nv_dot(const MvmNvBlk& q, const int8_t* xh, const int8_t* xl) {  // = nv_block_int<true>
  int32x4_t a = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q.l0, vld1q_s8(xh)), q.h0, vld1q_s8(xh + 16));
  int32x4_t b = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q.l1, vld1q_s8(xh + 32)), q.h1, vld1q_s8(xh + 48));
  int32x4_t al = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q.l0, vld1q_s8(xl)), q.h0, vld1q_s8(xl + 16));
  int32x4_t bl = vdotq_s32(vdotq_s32(vdupq_n_s32(0), q.l1, vld1q_s8(xl + 32)), q.h1, vld1q_s8(xl + 48));
  a = vaddq_s32(vshlq_n_s32(a, 8), al);
  b = vaddq_s32(vshlq_n_s32(b, 8), bl);
  return vpaddq_s32(a, b);
}

void mv_nvfp4_multi(const uint8_t* codes, const uint8_t* sc, float gmul, int K, int r0, int r1, const XQ* const* xs, int n,
                float* const* ys) {
  const float post = gmul * 128.f;
  const int NB = K / XB, rb = K / 2, rs = K / 16;
  const int8x16_t lut = vld1q_s8(MVM_E2M1X2);
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * rb;
    const uint8_t* s = sc + (size_t)r * rs;
    float32x4_t acc0[8], acc1[8];
    for (int i = 0; i < n; i++) acc0[i] = acc1[i] = vdupq_n_f32(0);
    auto blk = [&](int i, const MvmNvBlk& q, int bb, float32x4_t scale) {
      const XQ& x = *xs[i];
      int32x4_t d = mvm_nv_dot(q, x.hp + XB * bb, x.lp + XB * bb);
      return vmulq_f32(vcvtq_f32_s32(d), vmulq_n_f32(scale, x.d[bb]));
    };
    int b = 0;
    for (; b + 4 <= NB; b += 4) {
      float32x4_t sv[4];
      mvm_e4m3x16(s + 4 * b, sv);
      MvmNvBlk q0 = mvm_nv_decode(c + 32 * b, lut), q1 = mvm_nv_decode(c + 32 * (b + 1), lut), q2 = mvm_nv_decode(c + 32 * (b + 2), lut),
            q3 = mvm_nv_decode(c + 32 * (b + 3), lut);
      for (int i = 0; i < n; i++) {
        acc0[i] = vaddq_f32(acc0[i], blk(i, q0, b, sv[0]));
        acc1[i] = vaddq_f32(acc1[i], blk(i, q1, b + 1, sv[1]));
        acc0[i] = vaddq_f32(acc0[i], blk(i, q2, b + 2, sv[2]));
        acc1[i] = vaddq_f32(acc1[i], blk(i, q3, b + 3, sv[3]));
      }
    }
    for (; b < NB; b++) {
      MvmNvBlk q = mvm_nv_decode(c + 32 * b, lut);
      float32x4_t sv = mvm_e4m3x4(s + 4 * b);
      for (int i = 0; i < n; i++) acc0[i] = vaddq_f32(acc0[i], blk(i, q, b, sv));
    }
    for (int i = 0; i < n; i++) ys[i][r - r0] = vaddvq_f32(vaddq_f32(acc0[i], acc1[i])) * post;
  }
}

// ------------------------------------------------------------------------------------------------ MXFP8 (fp16 plane)
static inline float16x8_t mvm_e4m3x8_f16(int8x8_t v) {
  int16x8_t t = vshlq_n_s16(vmovl_s8(v), 7);
  return vreinterpretq_f16_s16(vandq_s16(t, vdupq_n_s16((int16_t)0xBFFF)));
}
static inline float32x4_t mvm_e8m0x4(const uint8_t* p) {
  uint32_t w;
  std::memcpy(&w, p, 4);
  uint32x4_t u = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(w))));
  return vreinterpretq_f32_u32(vshlq_n_u32(u, 23));
}
struct MvmFp8Grp {
  float16x8_t c[4];  // the 32 codes of a group as fp16 x 2^-8
};
static inline MvmFp8Grp mvm_fp8_decode(const int8_t* c) {
  int8x16_t v0 = vld1q_s8(c), v1 = vld1q_s8(c + 16);
  return {{mvm_e4m3x8_f16(vget_low_s8(v0)), mvm_e4m3x8_f16(vget_high_s8(v0)), mvm_e4m3x8_f16(vget_low_s8(v1)), mvm_e4m3x8_f16(vget_high_s8(v1))}};
}
static inline float32x4_t mvm_fp8_dot(const MvmFp8Grp& g, const _Float16* x) {  // = dotfp8_group
  float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
  for (int k = 0; k < 2; k++) {
    float16x8_t x0 = vld1q_f16((const float16_t*)x + 16 * k), x1 = vld1q_f16((const float16_t*)x + 16 * k + 8);
    a = vfmlalq_low_f16(a, g.c[2 * k], x0);
    b = vfmlalq_high_f16(b, g.c[2 * k], x0);
    a = vfmlalq_low_f16(a, g.c[2 * k + 1], x1);
    b = vfmlalq_high_f16(b, g.c[2 * k + 1], x1);
  }
  return vaddq_f32(a, b);
}

void mv_mxfp8_multi(const uint8_t* codes, const uint8_t* sc, int K, int r0, int r1, const XQ* const* xs, int n, float* const* ys) {
  const int NG = K / 32;
  for (int r = r0; r < r1; r++) {
    const int8_t* c = (const int8_t*)codes + (size_t)r * K;
    const uint8_t* s = sc + (size_t)r * NG;
    float32x4_t acc[8];
    for (int i = 0; i < n; i++) acc[i] = vdupq_n_f32(0);
    int g = 0;
    for (; g + 4 <= NG; g += 4) {
      MvmFp8Grp q0 = mvm_fp8_decode(c + 32 * g), q1 = mvm_fp8_decode(c + 32 * (g + 1)), q2 = mvm_fp8_decode(c + 32 * (g + 2)), q3 = mvm_fp8_decode(c + 32 * (g + 3));
      const float32x4_t es = mvm_e8m0x4(s + g);
      for (int i = 0; i < n; i++) {
        const _Float16* x = xs[i]->fn;
        float32x4_t dots = vpaddq_f32(vpaddq_f32(mvm_fp8_dot(q0, x + 32 * g), mvm_fp8_dot(q1, x + 32 * (g + 1))),
                                      vpaddq_f32(mvm_fp8_dot(q2, x + 32 * (g + 2)), mvm_fp8_dot(q3, x + 32 * (g + 3))));
        acc[i] = vfmaq_f32(acc[i], dots, es);
      }
    }
    for (int i = 0; i < n; i++) {
      float t = vaddvq_f32(acc[i]);
      for (int gg = g; gg < NG; gg++) {
        uint32_t bits = (uint32_t)s[gg] << 23;
        float sc1;
        std::memcpy(&sc1, &bits, 4);
        t += vaddvq_f32(mvm_fp8_dot(mvm_fp8_decode(c + 32 * gg), xs[i]->fn + 32 * gg)) * sc1;
      }
      ys[i][r - r0] = t * 256.f;
    }
  }
}

}  // namespace dm
