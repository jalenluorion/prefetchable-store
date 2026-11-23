// The one-vector NVFP4 kernel with I16 activations (mv_nvfp4, the decode expert kernel) against the kernel it replaced
// (nv_rows<2> of the v1 tree, copied below as the reference): every output bit for bit, on random records (every E4M3
// scale code but NaN), random inputs with outliers, Inkling's K (2304, 1152) and others (3072, 320, 192), odd row ranges.
// Also the single-thread instructions and cycles per weight byte of both. No model.   nv_i16_check
#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "dmeng/kernels.h"
#include "dmeng/weights.h"

using namespace dm;
extern "C" int thread_selfcounts(int type, void* buf, size_t nbytes);

namespace v1 {
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

// v1's xq_block (the scalar definition), I16 path
static inline int perm_pos(int i) {
  int half = i & 32, j = i & 31;
  return half + ((j & 1) ? 16 + (j >> 1) : (j >> 1));
}
static void xq_block_v1(XQ& q, int b, const float* x) {
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
}
}  // namespace v1

int main() {
  std::mt19937 rng(5);
  long bad = 0, tot = 0;
  for (int K : {2304, 1152, 3072, 320, 192}) {
    for (int t = 0; t < 20; t++) {
      XBuf xb;
      xb.init(K, Prec::I16);
      std::vector<float> x(K);
      std::normal_distribution<float> nd(0, t % 3 == 0 ? 10.f : 1.f);
      for (auto& v : x) v = nd(rng) * ((rng() % 7 == 0) ? 50.f : 1.f);
      xb.prep(x.data(), K);
      const int R = 257, r0 = t % 3, r1 = R - (t % 2);
      std::vector<uint8_t> c((size_t)R * K / 2), s((size_t)R * K / 16);
      for (auto& b : c) b = (uint8_t)rng();
      for (auto& b : s) {
        b = (uint8_t)rng();
        if ((b & 0x7f) == 0x7f) b ^= 1;
      }
      std::vector<float> y0(R), y1(R);
      v1::nv_rows<2>(c.data(), s.data(), 0.37f * 128.f, K, r0, r1, xb.q, y0.data());
      mv_nvfp4(c.data(), s.data(), 0.37f, K, r0, r1, xb.q, y1.data());
      for (int i = 0; i < r1 - r0; i++, tot++) bad += std::memcmp(&y0[i], &y1[i], 4) != 0;
    }
  }
  // the activation quantisation (xq_block, I16): every field of every block against v1's scalar code, on random
  // vectors with outliers, exact zeros, all-zero blocks, values at rounding ties and huge / tiny magnitudes
  long xbad = 0, xtot = 0;
  for (int t = 0; t < 4000; t++) {
    const int K2 = 2304;
    std::vector<float> x(K2);
    std::normal_distribution<float> nd(0, 1.f);
    const int kind = t % 8;
    for (int i = 0; i < K2; i++) {
      float v = nd(rng) * std::ldexp(1.f, (int)(rng() % 40) - 20);
      if (kind == 1 && rng() % 3 == 0) v = 0.f;
      if (kind == 2 && (i / 64) % 2 == 0) v = 0.f;
      if (kind == 3) v = (float)((int)(rng() % 200) - 100) * 0.5f;  // halves: ties after scaling
      if (kind == 4) v = (rng() % 50 == 0) ? 1e30f * nd(rng) : nd(rng) * 1e-30f;
      x[i] = v;
    }
    XBuf a, b2;
    a.init(K2, Prec::I16);
    b2.init(K2, Prec::I16);
    a.q.K = b2.q.K = K2;
    for (int blk = 0; blk < K2 / XB; blk++) {
      v1::xq_block_v1(a.q, blk, x.data() + XB * blk);
      xq_block(b2.q, blk, x.data() + XB * blk);
    }
    xtot++;
    xbad += std::memcmp(a.q.gsum, b2.q.gsum, 4 * (K2 / XB)) || std::memcmp(a.q.d, b2.q.d, 4 * (K2 / XB)) ||
            std::memcmp(a.q.hp, b2.q.hp, K2) || std::memcmp(a.q.lp, b2.q.lp, K2) || std::memcmp(a.q.hn, b2.q.hn, K2) ||
            std::memcmp(a.q.ln, b2.q.ln, K2) || std::memcmp(a.q.fp, b2.q.fp, 2 * K2) || std::memcmp(a.q.fn, b2.q.fn, 2 * K2);
  }
  printf("{\"xq_vectors\":%ld,\"xq_vectors_differ\":%ld}\n", xtot, xbad);
  bad += xbad;
  const int K = 2304, R = 4608;
  XBuf xb;
  xb.init(K, Prec::I16);
  std::vector<float> x(K);
  std::normal_distribution<float> nd;
  for (auto& v : x) v = nd(rng);
  xb.prep(x.data(), K);
  std::vector<uint8_t> c((size_t)R * K / 2), s((size_t)R * K / 16);
  for (auto& b : c) b = (uint8_t)rng();
  for (auto& b : s) b = (uint8_t)(0x30 + rng() % 16);
  std::vector<float> y(R);
  double ib[2], cb[2];
  for (int v = 0; v < 2; v++) {
    uint64_t a[2], b[2];
    thread_selfcounts(1, a, sizeof a);
    for (int r = 0; r < 20; r++) v ? mv_nvfp4(c.data(), s.data(), 1.f, K, 0, R, xb.q, y.data()) : v1::nv_rows<2>(c.data(), s.data(), 128.f, K, 0, R, xb.q, y.data());
    thread_selfcounts(1, b, sizeof b);
    const double by = 20.0 * (c.size() + s.size());
    ib[v] = (b[0] - a[0]) / by;
    cb[v] = (b[1] - a[1]) / by;
  }
  printf("{\"rows\":%ld,\"rows_differ\":%ld,\"ins_per_byte\":{\"v1\":%.3f,\"now\":%.3f},\"cyc_per_byte\":{\"v1\":%.3f,\"now\":%.3f}}\n", tot, bad, ib[0], ib[1], cb[0], cb[1]);
  printf("%s\n", bad ? "NV I16 CHECK FAILED" : "NV I16 CHECK PASSED");
  return bad ? 1 : 0;
}
