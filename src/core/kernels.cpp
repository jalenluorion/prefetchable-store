// NEON matrix-vector kernels for the store formats (see include/dmeng/kernels.h). Written for Apple M4 P-cores:
// SDOT/USDOT (dotprod, i8mm) for the integer paths, FMLAL (fp16fml) for the fp16 path.
#include "dmeng/kernels.h"

#include <arm_neon.h>

#include <cmath>
#include <cstring>

namespace dm {

static inline size_t up64(size_t n) { return (n + 63) & ~size_t(63); }

size_t xq_bytes(int K) {
  return 2 * up64(sizeof(float) * (K / XB)) + 4 * up64(K) + 2 * up64(2 * K);
}

void xq_bind(XQ& q, void* mem, int K, Prec p) {
  uint8_t* m = (uint8_t*)mem;
  q.K = K;
  q.prec = p;
  q.gsum = (float*)m; m += up64(sizeof(float) * (K / XB));
  q.d = (float*)m; m += up64(sizeof(float) * (K / XB));
  q.hp = (int8_t*)m; m += up64(K);
  q.lp = (int8_t*)m; m += up64(K);
  q.hn = (int8_t*)m; m += up64(K);
  q.ln = (int8_t*)m; m += up64(K);
  q.fp = (_Float16*)m; m += up64(2 * K);
  q.fn = (_Float16*)m;
}

// position of natural element i (0..63) in the perm order of a 64 block
static inline int perm_pos(int i) {
  int half = i & 32, j = i & 31;
  return half + ((j & 1) ? 16 + (j >> 1) : (j >> 1));
}

// The I16 block with NEON: the same values as the scalar code below (the sum in order; the max is order-free; each
// product x * id rounded once and converted with ties to even as lrintf; the fp16 conversions round to nearest even
// as the casts), the perm order by unzipping even and odd elements of each 32-element half
static void xq_block_i16(XQ& q, int b, const float* x) {
  float s = 0.f;
  for (int i = 0; i < XB; i++) s += x[i];
  q.gsum[b] = s;
  float32x4_t xv[16], m = vdupq_n_f32(0.f);
  for (int k = 0; k < 16; k++) {
    xv[k] = vld1q_f32(x + 4 * k);
    m = vmaxq_f32(m, vabsq_f32(xv[k]));
  }
  const float amax = vmaxvq_f32(m);
  _Float16 *fp = q.fp + XB * b, *fn = q.fn + XB * b;
  float16x8_t h[8];
  for (int k = 0; k < 8; k++) {
    h[k] = vcombine_f16(vcvt_f16_f32(xv[2 * k]), vcvt_f16_f32(xv[2 * k + 1]));
    vst1q_f16((float16_t*)fn + 8 * k, h[k]);
  }
  for (int half = 0; half < 2; half++) {  // 32 elements = h[4 half .. 4 half + 3]
    const float16x8_t* g = h + 4 * half;
    float16_t* o = (float16_t*)fp + 32 * half;
    vst1q_f16(o, vuzp1q_f16(g[0], g[1]));
    vst1q_f16(o + 8, vuzp1q_f16(g[2], g[3]));
    vst1q_f16(o + 16, vuzp2q_f16(g[0], g[1]));
    vst1q_f16(o + 24, vuzp2q_f16(g[2], g[3]));
  }
  const float d = amax / 32384.f, id = d > 0 ? 1.f / d : 0.f;
  q.d[b] = d;
  const int32x4_t lim = vdupq_n_s32(32384), nlim = vdupq_n_s32(-32384);
  int8x16_t hi[4], lo[4];
  for (int k = 0; k < 4; k++) {  // 16 elements
    int16x8_t v16[2];
    for (int u = 0; u < 2; u++) {
      int32x4_t a0 = vcvtnq_s32_f32(vmulq_n_f32(xv[4 * k + 2 * u], id));
      int32x4_t a1 = vcvtnq_s32_f32(vmulq_n_f32(xv[4 * k + 2 * u + 1], id));
      a0 = vmaxq_s32(vminq_s32(a0, lim), nlim);
      a1 = vmaxq_s32(vminq_s32(a1, lim), nlim);
      v16[u] = vcombine_s16(vmovn_s32(a0), vmovn_s32(a1));
    }
    int16x8_t h0 = vshrq_n_s16(vaddq_s16(v16[0], vdupq_n_s16(128)), 8), h1 = vshrq_n_s16(vaddq_s16(v16[1], vdupq_n_s16(128)), 8);
    int16x8_t l0 = vsubq_s16(v16[0], vshlq_n_s16(h0, 8)), l1 = vsubq_s16(v16[1], vshlq_n_s16(h1, 8));
    hi[k] = vcombine_s8(vmovn_s16(h0), vmovn_s16(h1));
    lo[k] = vcombine_s8(vmovn_s16(l0), vmovn_s16(l1));
  }
  int8_t *hp = q.hp + XB * b, *lp = q.lp + XB * b, *hn = q.hn + XB * b, *ln = q.ln + XB * b;
  for (int k = 0; k < 4; k++) {
    vst1q_s8(hn + 16 * k, hi[k]);
    vst1q_s8(ln + 16 * k, lo[k]);
  }
  for (int half = 0; half < 2; half++) {
    vst1q_s8(hp + 32 * half, vuzp1q_s8(hi[2 * half], hi[2 * half + 1]));
    vst1q_s8(hp + 32 * half + 16, vuzp2q_s8(hi[2 * half], hi[2 * half + 1]));
    vst1q_s8(lp + 32 * half, vuzp1q_s8(lo[2 * half], lo[2 * half + 1]));
    vst1q_s8(lp + 32 * half + 16, vuzp2q_s8(lo[2 * half], lo[2 * half + 1]));
  }
}

static void xq_block_ref(XQ& q, int b, const float* x);
void xq_block(XQ& q, int b, const float* x) {
  if (q.prec == Prec::I16) return xq_block_i16(q, b, x);
  xq_block_ref(q, b, x);
}

// the scalar definition (every precision; the I16 path above gives its values)
static void xq_block_ref(XQ& q, int b, const float* x) {
  float s = 0.f, amax = 0.f;
  for (int i = 0; i < XB; i++) {
    s += x[i];
    amax = std::fmax(amax, std::fabs(x[i]));
  }
  q.gsum[b] = s;
  int8_t *hp = q.hp + XB * b, *lp = q.lp + XB * b, *hn = q.hn + XB * b, *ln = q.ln + XB * b;
  _Float16 *fp = q.fp + XB * b, *fn = q.fn + XB * b;
  for (int i = 0; i < XB; i++) {
    _Float16 h = (_Float16)x[i];
    fn[i] = h;
    fp[perm_pos(i)] = h;
  }
  if (q.prec == Prec::I8) {
    float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
    q.d[b] = d;
    for (int i = 0; i < XB; i++) {
      int v = (int)std::lrintf(x[i] * id);
      v = v > 127 ? 127 : (v < -127 ? -127 : v);
      hn[i] = (int8_t)v;
      hp[perm_pos(i)] = (int8_t)v;
    }
  } else if (q.prec == Prec::I16) {
    float d = amax / 32384.f, id = d > 0 ? 1.f / d : 0.f;
    q.d[b] = d;
    for (int i = 0; i < XB; i++) {
      int v = (int)std::lrintf(x[i] * id);
      v = v > 32384 ? 32384 : (v < -32384 ? -32384 : v);
      int hi = (v + 128) >> 8, lo = v - (hi << 8);
      hn[i] = (int8_t)hi;
      ln[i] = (int8_t)lo;
      hp[perm_pos(i)] = (int8_t)hi;
      lp[perm_pos(i)] = (int8_t)lo;
    }
  } else {
    q.d[b] = 1.f;
  }
}

void xq_prep(XQ& q, const float* x) {
  for (int b = 0; b < q.K / XB; b++) xq_block(q, b, x + XB * b);
}

static inline float32x4_t f16x4_to_f32(const _Float16* p) { return vcvt_f32_f16(vld1_f16((const float16_t*)p)); }

// ------------------------------------------------------------------------------------------------ AFF4
// one 64-group of 4-bit codes (32 bytes) dotted with the perm-order int8 plane(s): int32x4 partial lanes
static inline int32x4_t dot4_i8(const uint8_t* c, const int8_t* x) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  int32x4_t a = vdupq_n_s32(0);
  a = vdotq_s32(a, vreinterpretq_s8_u8(vandq_u8(v0, m)), vld1q_s8(x));
  a = vdotq_s32(a, vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), vld1q_s8(x + 16));
  a = vdotq_s32(a, vreinterpretq_s8_u8(vandq_u8(v1, m)), vld1q_s8(x + 32));
  a = vdotq_s32(a, vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), vld1q_s8(x + 48));
  return a;
}
static inline int32x4_t dot4_i16(const uint8_t* c, const int8_t* xh, const int8_t* xl) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(v0, m)), h0 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
  int8x16_t l1 = vreinterpretq_s8_u8(vandq_u8(v1, m)), h1 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
  int32x4_t a = vdupq_n_s32(0), b = vdupq_n_s32(0);
  a = vdotq_s32(a, l0, vld1q_s8(xh));
  b = vdotq_s32(b, l0, vld1q_s8(xl));
  a = vdotq_s32(a, h0, vld1q_s8(xh + 16));
  b = vdotq_s32(b, h0, vld1q_s8(xl + 16));
  a = vdotq_s32(a, l1, vld1q_s8(xh + 32));
  b = vdotq_s32(b, l1, vld1q_s8(xl + 32));
  a = vdotq_s32(a, h1, vld1q_s8(xh + 48));
  b = vdotq_s32(b, h1, vld1q_s8(xl + 48));
  return vaddq_s32(vshlq_n_s32(a, 8), b);
}
static inline float32x4_t dot4_f16(const uint8_t* c, const _Float16* x) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  uint8x16_t q[4] = {vandq_u8(v0, m), vshrq_n_u8(v0, 4), vandq_u8(v1, m), vshrq_n_u8(v1, 4)};
  float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
  for (int k = 0; k < 4; k++) {
    float16x8_t c0 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(q[k])));
    float16x8_t c1 = vcvtq_f16_u16(vmovl_high_u8(q[k]));
    float16x8_t x0 = vld1q_f16((const float16_t*)x + 16 * k), x1 = vld1q_f16((const float16_t*)x + 16 * k + 8);
    a = vfmlalq_low_f16(a, c0, x0);
    b = vfmlalq_high_f16(b, c0, x0);
    a = vfmlalq_low_f16(a, c1, x1);
    b = vfmlalq_high_f16(b, c1, x1);
  }
  return vaddq_f32(a, b);
}

// I16, two rows at a time: each group's activation planes are loaded once for both rows (the single-row loop loads them
// per row: 8 of its 24 instructions per 32 code bytes). Per row the same exact int32 group totals and the same float
// operations in the same order as the single-row loop below, so the outputs are bit-identical.
static void mv_aff4_i16_pairs(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1,
                              const XQ& x, float* y) {
  const int NG = K / XB, rb = K / 2;
  const uint8x16_t m = vdupq_n_u8(0x0F);
  for (int r = r0; r + 1 < r1; r += 2) {
    const uint8_t* c0 = codes + (size_t)r * rb;
    const uint8_t* c1 = c0 + rb;
    const _Float16 *s0 = scales + (size_t)r * NG, *s1 = s0 + NG;
    const _Float16 *b0 = biases + (size_t)r * NG, *b1 = b0 + NG;
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0;
    for (int g = 0; g < NG; g += 4) {
      int32x4_t t0[4], t1[4];
      for (int q = 0; q < 4; q++) {
        const int8_t* xh = x.hp + XB * (g + q);
        const int8_t* xl = x.lp + XB * (g + q);
        const int8x16_t h0 = vld1q_s8(xh), h1 = vld1q_s8(xh + 16), h2 = vld1q_s8(xh + 32), h3 = vld1q_s8(xh + 48);
        const int8x16_t l0 = vld1q_s8(xl), l1 = vld1q_s8(xl + 16), l2 = vld1q_s8(xl + 32), l3 = vld1q_s8(xl + 48);
        for (int rr = 0; rr < 2; rr++) {
          const uint8_t* c = (rr ? c1 : c0) + 32 * (g + q);
          const uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
          const int8x16_t n0 = vreinterpretq_s8_u8(vandq_u8(v0, m)), n1 = vreinterpretq_s8_u8(vshrq_n_u8(v0, 4));
          const int8x16_t n2 = vreinterpretq_s8_u8(vandq_u8(v1, m)), n3 = vreinterpretq_s8_u8(vshrq_n_u8(v1, 4));
          int32x4_t a = vdupq_n_s32(0), b = vdupq_n_s32(0);
          a = vdotq_s32(a, n0, h0);
          b = vdotq_s32(b, n0, l0);
          a = vdotq_s32(a, n1, h1);
          b = vdotq_s32(b, n1, l1);
          a = vdotq_s32(a, n2, h2);
          b = vdotq_s32(b, n2, l2);
          a = vdotq_s32(a, n3, h3);
          b = vdotq_s32(b, n3, l3);
          (rr ? t1 : t0)[q] = vaddq_s32(vshlq_n_s32(a, 8), b);
        }
      }
      const float32x4_t xd = vld1q_f32(x.d + g), xg = vld1q_f32(x.gsum + g);
      const float32x4_t d0 = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(t0[0], t0[1]), vpaddq_s32(t0[2], t0[3])));
      acc0 = vfmaq_f32(acc0, d0, vmulq_f32(f16x4_to_f32(s0 + g), xd));
      acc0 = vfmaq_f32(acc0, f16x4_to_f32(b0 + g), xg);
      const float32x4_t d1 = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(t1[0], t1[1]), vpaddq_s32(t1[2], t1[3])));
      acc1 = vfmaq_f32(acc1, d1, vmulq_f32(f16x4_to_f32(s1 + g), xd));
      acc1 = vfmaq_f32(acc1, f16x4_to_f32(b1 + g), xg);
    }
    y[r - r0] = vaddvq_f32(acc0);
    y[r + 1 - r0] = vaddvq_f32(acc1);
  }
}

void mv_aff4(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ& x,
             float* y) {
  const int NG = K / XB, rb = K / 2;
  if (x.prec == Prec::I16 && NG % 4 == 0 && r1 - r0 >= 2) {  // pairs of rows; an odd last row below
    const int np = (r1 - r0) & ~1;
    mv_aff4_i16_pairs(codes, scales, biases, K, r0, r0 + np, x, y);
    r0 += np;
    y += np;
  }
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * rb;
    const _Float16* s = scales + (size_t)r * NG;
    const _Float16* bi = biases + (size_t)r * NG;
    float32x4_t acc = vdupq_n_f32(0);
    int g = 0;
    if (x.prec == Prec::F16) {
      for (; g + 4 <= NG; g += 4) {
        float32x4_t d0 = dot4_f16(c + 32 * g, x.fp + XB * g), d1 = dot4_f16(c + 32 * (g + 1), x.fp + XB * (g + 1));
        float32x4_t d2 = dot4_f16(c + 32 * (g + 2), x.fp + XB * (g + 2)), d3 = dot4_f16(c + 32 * (g + 3), x.fp + XB * (g + 3));
        float32x4_t dots = vpaddq_f32(vpaddq_f32(d0, d1), vpaddq_f32(d2, d3));
        acc = vfmaq_f32(acc, dots, f16x4_to_f32(s + g));
        acc = vfmaq_f32(acc, f16x4_to_f32(bi + g), vld1q_f32(x.gsum + g));
      }
      float t = vaddvq_f32(acc);
      for (; g < NG; g++) t += vaddvq_f32(dot4_f16(c + 32 * g, x.fp + XB * g)) * (float)s[g] + (float)bi[g] * x.gsum[g];
      y[r - r0] = t;
      continue;
    }
    const bool i16 = x.prec == Prec::I16;
    for (; g + 4 <= NG; g += 4) {
      int32x4_t a0, a1, a2, a3;
      if (i16) {
        a0 = dot4_i16(c + 32 * g, x.hp + XB * g, x.lp + XB * g);
        a1 = dot4_i16(c + 32 * (g + 1), x.hp + XB * (g + 1), x.lp + XB * (g + 1));
        a2 = dot4_i16(c + 32 * (g + 2), x.hp + XB * (g + 2), x.lp + XB * (g + 2));
        a3 = dot4_i16(c + 32 * (g + 3), x.hp + XB * (g + 3), x.lp + XB * (g + 3));
      } else {
        a0 = dot4_i8(c + 32 * g, x.hp + XB * g);
        a1 = dot4_i8(c + 32 * (g + 1), x.hp + XB * (g + 1));
        a2 = dot4_i8(c + 32 * (g + 2), x.hp + XB * (g + 2));
        a3 = dot4_i8(c + 32 * (g + 3), x.hp + XB * (g + 3));
      }
      float32x4_t dots = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)));
      acc = vfmaq_f32(acc, dots, vmulq_f32(f16x4_to_f32(s + g), vld1q_f32(x.d + g)));
      acc = vfmaq_f32(acc, f16x4_to_f32(bi + g), vld1q_f32(x.gsum + g));
    }
    float t = vaddvq_f32(acc);
    for (; g < NG; g++) {
      int32x4_t a = i16 ? dot4_i16(c + 32 * g, x.hp + XB * g, x.lp + XB * g) : dot4_i8(c + 32 * g, x.hp + XB * g);
      t += (float)vaddvq_s32(a) * (float)s[g] * x.d[g] + (float)bi[g] * x.gsum[g];
    }
    y[r - r0] = t;
  }
}

// ------------------------------------------------------------------------------------------------ AFF8
static inline int32x4_t dot8_i8(const uint8_t* c, const int8_t* x) {
  int32x4_t a = vdupq_n_s32(0);
  for (int k = 0; k < 4; k++) a = vusdotq_s32(a, vld1q_u8(c + 16 * k), vld1q_s8(x + 16 * k));
  return a;
}
static inline int32x4_t dot8_i16(const uint8_t* c, const int8_t* xh, const int8_t* xl) {
  int32x4_t a = vdupq_n_s32(0), b = vdupq_n_s32(0);
  for (int k = 0; k < 4; k++) {
    uint8x16_t v = vld1q_u8(c + 16 * k);
    a = vusdotq_s32(a, v, vld1q_s8(xh + 16 * k));
    b = vusdotq_s32(b, v, vld1q_s8(xl + 16 * k));
  }
  return vaddq_s32(vshlq_n_s32(a, 8), b);
}
static inline float32x4_t dot8_f16(const uint8_t* c, const _Float16* x) {
  float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
  for (int k = 0; k < 4; k++) {
    uint8x16_t v = vld1q_u8(c + 16 * k);
    float16x8_t c0 = vcvtq_f16_u16(vmovl_u8(vget_low_u8(v))), c1 = vcvtq_f16_u16(vmovl_high_u8(v));
    float16x8_t x0 = vld1q_f16((const float16_t*)x + 16 * k), x1 = vld1q_f16((const float16_t*)x + 16 * k + 8);
    a = vfmlalq_low_f16(a, c0, x0);
    b = vfmlalq_high_f16(b, c0, x0);
    a = vfmlalq_low_f16(a, c1, x1);
    b = vfmlalq_high_f16(b, c1, x1);
  }
  return vaddq_f32(a, b);
}

void mv_aff8(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ& x,
             float* y) {
  const int NG = K / XB;
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * K;
    const _Float16* s = scales + (size_t)r * NG;
    const _Float16* bi = biases + (size_t)r * NG;
    float32x4_t acc = vdupq_n_f32(0);
    int g = 0;
    if (x.prec == Prec::F16) {
      for (; g + 4 <= NG; g += 4) {
        float32x4_t dots = vpaddq_f32(vpaddq_f32(dot8_f16(c + 64 * g, x.fn + XB * g), dot8_f16(c + 64 * (g + 1), x.fn + XB * (g + 1))),
                                      vpaddq_f32(dot8_f16(c + 64 * (g + 2), x.fn + XB * (g + 2)), dot8_f16(c + 64 * (g + 3), x.fn + XB * (g + 3))));
        acc = vfmaq_f32(acc, dots, f16x4_to_f32(s + g));
        acc = vfmaq_f32(acc, f16x4_to_f32(bi + g), vld1q_f32(x.gsum + g));
      }
      float t = vaddvq_f32(acc);
      for (; g < NG; g++) t += vaddvq_f32(dot8_f16(c + 64 * g, x.fn + XB * g)) * (float)s[g] + (float)bi[g] * x.gsum[g];
      y[r - r0] = t;
      continue;
    }
    const bool i16 = x.prec == Prec::I16;
    for (; g + 4 <= NG; g += 4) {
      int32x4_t a0, a1, a2, a3;
      if (i16) {
        a0 = dot8_i16(c + 64 * g, x.hn + XB * g, x.ln + XB * g);
        a1 = dot8_i16(c + 64 * (g + 1), x.hn + XB * (g + 1), x.ln + XB * (g + 1));
        a2 = dot8_i16(c + 64 * (g + 2), x.hn + XB * (g + 2), x.ln + XB * (g + 2));
        a3 = dot8_i16(c + 64 * (g + 3), x.hn + XB * (g + 3), x.ln + XB * (g + 3));
      } else {
        a0 = dot8_i8(c + 64 * g, x.hn + XB * g);
        a1 = dot8_i8(c + 64 * (g + 1), x.hn + XB * (g + 1));
        a2 = dot8_i8(c + 64 * (g + 2), x.hn + XB * (g + 2));
        a3 = dot8_i8(c + 64 * (g + 3), x.hn + XB * (g + 3));
      }
      float32x4_t dots = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)));
      acc = vfmaq_f32(acc, dots, vmulq_f32(f16x4_to_f32(s + g), vld1q_f32(x.d + g)));
      acc = vfmaq_f32(acc, f16x4_to_f32(bi + g), vld1q_f32(x.gsum + g));
    }
    float t = vaddvq_f32(acc);
    for (; g < NG; g++) {
      int32x4_t a = i16 ? dot8_i16(c + 64 * g, x.hn + XB * g, x.ln + XB * g) : dot8_i8(c + 64 * g, x.hn + XB * g);
      t += (float)vaddvq_s32(a) * (float)s[g] * x.d[g] + (float)bi[g] * x.gsum[g];
    }
    y[r - r0] = t;
  }
}

// ------------------------------------------------------------------------------------------------ AFF8, split planes
// mv_aff8's I16 path with the code bytes rebuilt from the two nibble planes (weights.h, AFF8S; vsli / vsri): every int32
// lane, and so every float operation after it, is mv_aff8's.
static inline int32x4_t dot8s_i16(const uint8_t* h, const uint8_t* l, const int8_t* xh, const int8_t* xl) {
  const uint8x16_t h0 = vld1q_u8(h), h1 = vld1q_u8(h + 16), l0 = vld1q_u8(l), l1 = vld1q_u8(l + 16);
  const uint8x16_t v[4] = {vsliq_n_u8(l0, h0, 4), vsriq_n_u8(h0, l0, 4), vsliq_n_u8(l1, h1, 4), vsriq_n_u8(h1, l1, 4)};
  int32x4_t a = vdupq_n_s32(0), b = vdupq_n_s32(0);
  for (int k = 0; k < 4; k++) {
    a = vusdotq_s32(a, v[k], vld1q_s8(xh + 16 * k));
    b = vusdotq_s32(b, v[k], vld1q_s8(xl + 16 * k));
  }
  return vaddq_s32(vshlq_n_s32(a, 8), b);
}

void mv_aff8s_i16(const uint8_t* codes, int rows, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ& x,
                  float* y) {
  const int NG = K / XB;
  const size_t half = (size_t)K / 2;
  for (int r = r0; r < r1; r++) {
    const int b0 = r & ~63, nb = rows - b0 < 64 ? rows - b0 : 64;
    const uint8_t* ch = codes + (size_t)b0 * K + (size_t)(r & 63) * half;
    const uint8_t* cl = ch + (size_t)nb * half;
    const _Float16* s = scales + (size_t)r * NG;
    const _Float16* bi = biases + (size_t)r * NG;
    float32x4_t acc = vdupq_n_f32(0);
    int g = 0;
    for (; g + 4 <= NG; g += 4) {
      int32x4_t a0 = dot8s_i16(ch + 32 * g, cl + 32 * g, x.hn + XB * g, x.ln + XB * g);
      int32x4_t a1 = dot8s_i16(ch + 32 * (g + 1), cl + 32 * (g + 1), x.hn + XB * (g + 1), x.ln + XB * (g + 1));
      int32x4_t a2 = dot8s_i16(ch + 32 * (g + 2), cl + 32 * (g + 2), x.hn + XB * (g + 2), x.ln + XB * (g + 2));
      int32x4_t a3 = dot8s_i16(ch + 32 * (g + 3), cl + 32 * (g + 3), x.hn + XB * (g + 3), x.ln + XB * (g + 3));
      float32x4_t dots = vcvtq_f32_s32(vpaddq_s32(vpaddq_s32(a0, a1), vpaddq_s32(a2, a3)));
      acc = vfmaq_f32(acc, dots, vmulq_f32(f16x4_to_f32(s + g), vld1q_f32(x.d + g)));
      acc = vfmaq_f32(acc, f16x4_to_f32(bi + g), vld1q_f32(x.gsum + g));
    }
    float t = vaddvq_f32(acc);
    for (; g < NG; g++) {
      int32x4_t a = dot8s_i16(ch + 32 * g, cl + 32 * g, x.hn + XB * g, x.ln + XB * g);
      t += (float)vaddvq_s32(a) * (float)s[g] * x.d[g] + (float)bi[g] * x.gsum[g];
    }
    y[r - r0] = t;
  }
}

// ------------------------------------------------------------------------------------------------ NVFP4
// E2M1 nibble -> 2 x value as int8 (0, .5, 1, 1.5, 2, 3, 4, 6 and negatives)
static const int8_t E2M1X2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// four E4M3 bytes -> float32 x 2^-8 (exact: the E4M3 bits shifted into an fp16, subnormals included)
static inline float32x4_t e4m3x4(const uint8_t* p) {
  uint32_t w;
  std::memcpy(&w, p, 4);
  uint16x4_t u = vget_low_u16(vmovl_u8(vcreate_u8(w)));
  uint16x4_t f = vorr_u16(vshl_n_u16(vand_u16(u, vdup_n_u16(0x7F)), 7), vshl_n_u16(vand_u16(u, vdup_n_u16(0x80)), 8));
  return vcvt_f32_f16(vreinterpret_f16_u16(f));
}

// sixteen E4M3 bytes -> four float32x4 (x 2^-8)
static inline void e4m3x16(const uint8_t* p, float32x4_t out[4]) {
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

// one 64 block of NVFP4 codes (32 bytes = groups of 16: 4 per block) -> the 4 group dots
static inline float32x4_t nv_block_f16(const uint8_t* c, const _Float16* xp16, int8x16_t lut) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  int8x16_t q[4] = {vqtbl1q_s8(lut, vandq_u8(v0, m)), vqtbl1q_s8(lut, vshrq_n_u8(v0, 4)),
                    vqtbl1q_s8(lut, vandq_u8(v1, m)), vqtbl1q_s8(lut, vshrq_n_u8(v1, 4))};
  // q[0], q[1]: bytes 0-7 -> group 0, bytes 8-15 -> group 1; q[2], q[3]: groups 2, 3
  const float16_t* xp = (const float16_t*)xp16;
  float32x4_t g[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
  for (int k = 0; k < 4; k++) {
    float16x8_t c0 = vcvtq_f16_s16(vmovl_s8(vget_low_s8(q[k]))), c1 = vcvtq_f16_s16(vmovl_high_s8(q[k]));
    float16x8_t x0 = vld1q_f16(xp + 16 * k), x1 = vld1q_f16(xp + 16 * k + 8);
    int ga = (k >> 1) * 2;
    g[ga] = vfmlalq_low_f16(g[ga], c0, x0);
    g[ga] = vfmlalq_high_f16(g[ga], c0, x0);
    g[ga + 1] = vfmlalq_low_f16(g[ga + 1], c1, x1);
    g[ga + 1] = vfmlalq_high_f16(g[ga + 1], c1, x1);
  }
  return vpaddq_f32(vpaddq_f32(g[0], g[1]), vpaddq_f32(g[2], g[3]));
}
template <bool I16>
static inline int32x4_t nv_block_int(const uint8_t* c, const int8_t* xh, const int8_t* xl, int8x16_t lut) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  int8x16_t l0 = vqtbl1q_s8(lut, vandq_u8(v0, m)), h0 = vqtbl1q_s8(lut, vshrq_n_u8(v0, 4));
  int8x16_t l1 = vqtbl1q_s8(lut, vandq_u8(v1, m)), h1 = vqtbl1q_s8(lut, vshrq_n_u8(v1, 4));
  int32x4_t a = vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, vld1q_s8(xh)), h0, vld1q_s8(xh + 16));
  int32x4_t b = vdotq_s32(vdotq_s32(vdupq_n_s32(0), l1, vld1q_s8(xh + 32)), h1, vld1q_s8(xh + 48));
  if (I16) {
    int32x4_t al = vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, vld1q_s8(xl)), h0, vld1q_s8(xl + 16));
    int32x4_t bl = vdotq_s32(vdotq_s32(vdupq_n_s32(0), l1, vld1q_s8(xl + 32)), h1, vld1q_s8(xl + 48));
    a = vaddq_s32(vshlq_n_s32(a, 8), al);
    b = vaddq_s32(vshlq_n_s32(b, 8), bl);
  }
  return vpaddq_s32(a, b);  // [group 0, 1, 2, 3]
}

template <int P>  // 0 F16, 1 I8, 2 I16
static void nv_rows(const uint8_t* codes, const uint8_t* sc, float post, int K, int r0, int r1, const XQ& x, float* y) {
  const int NB = K / XB, rb = K / 2, rs = K / 16;
  const int8x16_t lut = vld1q_s8(E2M1X2);
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * rb;
    const uint8_t* s = sc + (size_t)r * rs;
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0;
    int b = 0;
    auto blk = [&](int bb, float32x4_t scale) -> float32x4_t {
      if (P == 0) return vmulq_f32(nv_block_f16(c + 32 * bb, x.fp + XB * bb, lut), scale);
      int32x4_t d = nv_block_int<P == 2>(c + 32 * bb, x.hp + XB * bb, x.lp + XB * bb, lut);
      return vmulq_f32(vcvtq_f32_s32(d), vmulq_n_f32(scale, x.d[bb]));
    };
    for (; b + 4 <= NB; b += 4) {  // 16 scales converted at once
      float32x4_t sv[4];
      e4m3x16(s + 4 * b, sv);
      acc0 = vaddq_f32(acc0, blk(b, sv[0]));
      acc1 = vaddq_f32(acc1, blk(b + 1, sv[1]));
      acc0 = vaddq_f32(acc0, blk(b + 2, sv[2]));
      acc1 = vaddq_f32(acc1, blk(b + 3, sv[3]));
    }
    for (; b < NB; b++) acc0 = vaddq_f32(acc0, blk(b, e4m3x4(s + 4 * b)));
    y[r - r0] = vaddvq_f32(vaddq_f32(acc0, acc1)) * post;
  }
}

// The I16 rows, cheaper per weight byte (the expert kernel of decode, 44 of Inkling's 48 applications): two rows at a
// time share the activation planes and block scales; the low plane's SDOTs accumulate into the high plane's sums
// shifted by 8 (integer: the same (hi << 8) + lo); the scales are converted by a sign-extending shift (exact, as the
// fp16 route above). Every float operation, its operands and its order are nv_rows<2>'s (bit-identical: checked on
// random records and inputs, every E4M3 scale code, K = 2304, 1152, 3072, 320, 192, odd row ranges).
static inline void e4m3x16s(const uint8_t* p, float32x4_t out[4]) {
  int8x16_t v = vld1q_s8((const int8_t*)p);
  const int16x8_t m = vdupq_n_s16((int16_t)0xBFFF);
  float16x8_t a = vreinterpretq_f16_s16(vandq_s16(vshll_n_s8(vget_low_s8(v), 7), m));
  float16x8_t b = vreinterpretq_f16_s16(vandq_s16(vshll_high_n_s8(v, 7), m));
  out[0] = vcvt_f32_f16(vget_low_f16(a));
  out[1] = vcvt_high_f32_f16(a);
  out[2] = vcvt_f32_f16(vget_low_f16(b));
  out[3] = vcvt_high_f32_f16(b);
}
static inline void e4m3x8(const uint8_t* p, float32x4_t out[2]) {
  float16x8_t a = vreinterpretq_f16_s16(vandq_s16(vshll_n_s8(vld1_s8((const int8_t*)p), 7), vdupq_n_s16((int16_t)0xBFFF)));
  out[0] = vcvt_f32_f16(vget_low_f16(a));
  out[1] = vcvt_high_f32_f16(a);
}
static inline float32x4_t e4m3x4s(const uint8_t* p) {
  uint32_t w;
  std::memcpy(&w, p, 4);
  int16x4_t u = vget_low_s16(vshll_n_s8(vreinterpret_s8_u32(vdup_n_u32(w)), 7));
  return vcvt_f32_f16(vreinterpret_f16_s16(vand_s16(u, vdup_n_s16((int16_t)0xBFFF))));
}
// the 4 group dots of one 64 block, I16: the high plane's sums shifted by 8, then the low plane's SDOTs accumulated
// into them (integer: equal to (hi << 8) + lo)
static inline int32x4_t blk_i16(const uint8_t* c, const int8x16_t xh[4], const int8x16_t xl[4], int8x16_t lut) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  int8x16_t l0 = vqtbl1q_s8(lut, vandq_u8(v0, m)), h0 = vqtbl1q_s8(lut, vshrq_n_u8(v0, 4));
  int8x16_t l1 = vqtbl1q_s8(lut, vandq_u8(v1, m)), h1 = vqtbl1q_s8(lut, vshrq_n_u8(v1, 4));
  int32x4_t a = vshlq_n_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, xh[0]), h0, xh[1]), 8);
  int32x4_t b = vshlq_n_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), l1, xh[2]), h1, xh[3]), 8);
  a = vdotq_s32(vdotq_s32(a, l0, xl[0]), h0, xl[1]);
  b = vdotq_s32(vdotq_s32(b, l1, xl[2]), h1, xl[3]);
  return vpaddq_s32(a, b);
}
// two rows at a time: the activation planes and block scales loaded once for both
static void nv_rows_i16(const uint8_t* codes, const uint8_t* sc, float post, int K, int r0, int r1, const XQ& x, float* y) {
  const int NB = K / XB, rb = K / 2, rs = K / 16;
  const int8x16_t lut = vld1q_s8(E2M1X2);
  int r = r0;
  for (; r + 2 <= r1; r += 2) {
    const uint8_t* cA = codes + (size_t)r * rb;
    const uint8_t* cB = cA + rb;
    const uint8_t* sA = sc + (size_t)r * rs;
    const uint8_t* sB = sA + rs;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, b0 = a0, b1 = a0;
    int b = 0;
    auto blk2 = [&](int bb, float32x4_t sa, float32x4_t sb, float32x4_t& accA, float32x4_t& accB) {
      const int8_t* xh = x.hp + XB * bb;
      const int8_t* xl = x.lp + XB * bb;
      int8x16_t H[4] = {vld1q_s8(xh), vld1q_s8(xh + 16), vld1q_s8(xh + 32), vld1q_s8(xh + 48)};
      int8x16_t L[4] = {vld1q_s8(xl), vld1q_s8(xl + 16), vld1q_s8(xl + 32), vld1q_s8(xl + 48)};
      const float d = x.d[bb];
      int32x4_t dA = blk_i16(cA + 32 * bb, H, L, lut), dB = blk_i16(cB + 32 * bb, H, L, lut);
      accA = vaddq_f32(accA, vmulq_f32(vcvtq_f32_s32(dA), vmulq_n_f32(sa, d)));
      accB = vaddq_f32(accB, vmulq_f32(vcvtq_f32_s32(dB), vmulq_n_f32(sb, d)));
    };
    for (const int nb4 = NB & ~3; b < nb4; b += 2) {  // as mv_nvfp4: blocks 4i, 4i + 2 into the first sums, 4i + 1,
      float32x4_t sa[2], sb[2];                        // 4i + 3 into the second, the rest into the first; scales
      e4m3x8(sA + 4 * b, sa);                          // converted 8 at a time (fewer live registers than 16)
      e4m3x8(sB + 4 * b, sb);
      blk2(b, sa[0], sb[0], a0, b0);
      blk2(b + 1, sa[1], sb[1], a1, b1);
    }
    for (; b < NB; b++) blk2(b, e4m3x4s(sA + 4 * b), e4m3x4s(sB + 4 * b), a0, b0);
    y[r - r0] = vaddvq_f32(vaddq_f32(a0, a1)) * post;
    y[r + 1 - r0] = vaddvq_f32(vaddq_f32(b0, b1)) * post;
  }
  for (; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * rb;
    const uint8_t* s = sc + (size_t)r * rs;
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = acc0;
    int b = 0;
    auto blk = [&](int bb, float32x4_t scale) -> float32x4_t {
      const int8_t* xh = x.hp + XB * bb;
      const int8_t* xl = x.lp + XB * bb;
      int8x16_t H[4] = {vld1q_s8(xh), vld1q_s8(xh + 16), vld1q_s8(xh + 32), vld1q_s8(xh + 48)};
      int8x16_t L[4] = {vld1q_s8(xl), vld1q_s8(xl + 16), vld1q_s8(xl + 32), vld1q_s8(xl + 48)};
      return vmulq_f32(vcvtq_f32_s32(blk_i16(c + 32 * bb, H, L, lut)), vmulq_n_f32(scale, x.d[bb]));
    };
    for (; b + 4 <= NB; b += 4) {
      float32x4_t sv[4];
      e4m3x16s(s + 4 * b, sv);
      acc0 = vaddq_f32(acc0, blk(b, sv[0]));
      acc1 = vaddq_f32(acc1, blk(b + 1, sv[1]));
      acc0 = vaddq_f32(acc0, blk(b + 2, sv[2]));
      acc1 = vaddq_f32(acc1, blk(b + 3, sv[3]));
    }
    for (; b < NB; b++) acc0 = vaddq_f32(acc0, blk(b, e4m3x4s(s + 4 * b)));
    y[r - r0] = vaddvq_f32(vaddq_f32(acc0, acc1)) * post;
  }
}

void mv_nvfp4(const uint8_t* codes, const uint8_t* sc, float gmul, int K, int r0, int r1, const XQ& x, float* y) {
  const float post = gmul * 128.f;  // x 0.5 (the table holds 2x) x 256 (the e4m3 conversion returns 2^-8 x)
  if (x.prec == Prec::F16) nv_rows<0>(codes, sc, post, K, r0, r1, x, y);
  else if (x.prec == Prec::I8) nv_rows<1>(codes, sc, post, K, r0, r1, x, y);
  else nv_rows_i16(codes, sc, post, K, r0, r1, x, y);
}

// ------------------------------------------------------------------------------------------------ MXFP8
// eight E4M3 codes (int8) -> fp16 x 2^-8: sign-extend, shift the 7 magnitude bits into place, clear bit 14
static inline float16x8_t e4m3x8_f16(int8x8_t v) {
  int16x8_t t = vshlq_n_s16(vmovl_s8(v), 7);
  return vreinterpretq_f16_s16(vandq_s16(t, vdupq_n_s16((int16_t)0xBFFF)));
}
static inline float32x4_t e8m0x4(const uint8_t* p) {
  uint32_t w;
  std::memcpy(&w, p, 4);
  uint32x4_t u = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(w))));
  return vreinterpretq_f32_u32(vshlq_n_u32(u, 23));  // 2^(e - 127); e = 0 gives 0 instead of 2^-127
}
static inline float32x4_t dotfp8_group(const int8_t* c, const _Float16* x) {  // one group of 32
  float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
  for (int k = 0; k < 2; k++) {
    int8x16_t v = vld1q_s8(c + 16 * k);
    float16x8_t c0 = e4m3x8_f16(vget_low_s8(v)), c1 = e4m3x8_f16(vget_high_s8(v));
    float16x8_t x0 = vld1q_f16((const float16_t*)x + 16 * k), x1 = vld1q_f16((const float16_t*)x + 16 * k + 8);
    a = vfmlalq_low_f16(a, c0, x0);
    b = vfmlalq_high_f16(b, c0, x0);
    a = vfmlalq_low_f16(a, c1, x1);
    b = vfmlalq_high_f16(b, c1, x1);
  }
  return vaddq_f32(a, b);
}

void mv_mxfp8(const uint8_t* codes, const uint8_t* sc, int K, int r0, int r1, const XQ& x, float* y) {
  const int NG = K / 32;
  for (int r = r0; r < r1; r++) {
    const int8_t* c = (const int8_t*)codes + (size_t)r * K;
    const uint8_t* s = sc + (size_t)r * NG;
    float32x4_t acc = vdupq_n_f32(0);
    int g = 0;
    for (; g + 4 <= NG; g += 4) {
      float32x4_t dots = vpaddq_f32(vpaddq_f32(dotfp8_group(c + 32 * g, x.fn + 32 * g), dotfp8_group(c + 32 * (g + 1), x.fn + 32 * (g + 1))),
                                    vpaddq_f32(dotfp8_group(c + 32 * (g + 2), x.fn + 32 * (g + 2)), dotfp8_group(c + 32 * (g + 3), x.fn + 32 * (g + 3))));
      acc = vfmaq_f32(acc, dots, e8m0x4(s + g));
    }
    float t = vaddvq_f32(acc);
    for (; g < NG; g++) {
      uint32_t bits = (uint32_t)s[g] << 23;
      float sc1;
      std::memcpy(&sc1, &bits, 4);
      t += vaddvq_f32(dotfp8_group(c + 32 * g, x.fn + 32 * g)) * sc1;
    }
    y[r - r0] = t * 256.f;
  }
}

// ------------------------------------------------------------------------------------------------ F32
void mv_f32(const float* W, int K, int r0, int r1, const float* x, float* y) {
  for (int r = r0; r < r1; r++) {
    const float* w = W + (size_t)r * K;
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    int k = 0;
    for (; k + 16 <= K; k += 16) {
      a0 = vfmaq_f32(a0, vld1q_f32(w + k), vld1q_f32(x + k));
      a1 = vfmaq_f32(a1, vld1q_f32(w + k + 4), vld1q_f32(x + k + 4));
      a2 = vfmaq_f32(a2, vld1q_f32(w + k + 8), vld1q_f32(x + k + 8));
      a3 = vfmaq_f32(a3, vld1q_f32(w + k + 12), vld1q_f32(x + k + 12));
    }
    float t = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; k < K; k++) t += w[k] * x[k];
    y[r - r0] = t;
  }
}

}  // namespace dm
