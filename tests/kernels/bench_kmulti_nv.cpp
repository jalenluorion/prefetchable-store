// Inkling's decode expert kernels (one position, I16 activations for NVFP4, fp16 for MXFP8) on real records: the engine's
// kernels and candidate rewrites, sized before any engine change. Per candidate: instructions and cycles per weight byte on
// one thread in cache (a 2304 x 2304 gate-up matrix, repeated), GB/s on all threads streaming over a set of experts larger
// than the caches (rows claimed 64 at a time, gate-up then down of each expert), and agreement with the engine's kernel
// (bits) or, for the candidates that round differently, the error of both against a float64 reference on real records.
// The name (bench_kmulti_*) lets a guard script treat it as a model process (bench/README.md). No model process.
//   bench_kmulti_nv MODEL_DIR [threads] [reps]
//
// Candidates (NVFP4):
//   K0  mv_nvfp4 (the engine: two rows share the activation planes; per 64 block: 4 lookups, 8 SDOT, 2 shifts, a pairwise
//       add to the 4 group sums, then convert, scale x activation scale, multiply, add)
//   K1  K0 with four rows sharing the activation planes (same operations per row, fewer loads)
//   G2  group-lane layout: the codes reordered within each 64 block at load time, and the activation planes in the
//       matching order, so each SDOT lane accumulates one group of 16: no pairwise add, one multiply-add joins the planes
//       (integers exact: the same group sums, the same float operations in the same order: bit-identical to K0); scales
//       converted by an unsigned shift (the scales of every NVFP4 record are positive: checked)
//   G4  G2 with four rows
//   G2F G2 with the float epilogue fused (convert, scale x activation scale, one FMA): one rounding fewer, not K0's bits
//   G2I the integer fold (scale mantissa as an integer multiply, exponent as a shift, per row minimum exponent folded
//       into the end): sizing only
// Candidates (MXFP8):
//   M0  mv_mxfp8 (the engine)
//   M1  four accumulators per group instead of two (shorter FMLAL chains): rounds differently
#include <arm_neon.h>
#include <fcntl.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "dmeng/kernels.h"
#include "dmeng/weights.h"

using namespace dm;
using sclk = std::chrono::steady_clock;
extern "C" int thread_selfcounts(int type, void* buf, size_t nbytes);

constexpr int H = 2304, I = 1152;
// the store's record layout (moved-store.json nv_* / mx_*)
constexpr size_t NV_REC = 4489216, NV_GU_C = 0, NV_GU_S = 2654208, NV_DN_C = 2985984, NV_DN_S = 4313088, NV_TR = 4478976;
constexpr size_t MX_REC = 8224768, MX_GU_C = 0, MX_GU_S = 5308416, MX_DN_C = 5474304, MX_DN_S = 8128512;

static const int8_t LUT[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ------------------------------------------------------------------------------------------------ the group-lane layout
// position p (0..63) of a block holds natural weight n(p): SDOT operand q = p / 16, lane i = (p % 16) / 4 takes group i
static inline int gl_nat(int p) { return 16 * ((p & 15) >> 2) + 4 * (p >> 4) + (p & 3); }
static void gl_codes(const uint8_t* in, uint8_t* out, int K) {  // one row
  auto nib = [&](int b, int n) { const uint8_t v = in[32 * b + n / 2]; return (n & 1) ? v >> 4 : v & 15; };
  for (int b = 0; b < K / 64; b++)
    for (int j = 0; j < 16; j++) {
      out[32 * b + j] = (uint8_t)(nib(b, gl_nat(j)) | nib(b, gl_nat(16 + j)) << 4);
      out[32 * b + 16 + j] = (uint8_t)(nib(b, gl_nat(32 + j)) | nib(b, gl_nat(48 + j)) << 4);
    }
}
struct XG {
  std::vector<int8_t> h, l;
};
static void gl_x(const XQ& x, int K, XG& g) {
  g.h.resize(K);
  g.l.resize(K);
  for (int b = 0; b < K / 64; b++)
    for (int p = 0; p < 64; p++) {
      g.h[64 * b + p] = x.hn[64 * b + gl_nat(p)];
      g.l[64 * b + p] = x.ln[64 * b + gl_nat(p)];
    }
}

// ------------------------------------------------------------------------------------------------ NVFP4 candidates
static inline void e4m3x8(const uint8_t* p, float32x4_t out[2]) {  // K0's (sign-extending)
  float16x8_t a = vreinterpretq_f16_s16(vandq_s16(vshll_n_s8(vld1_s8((const int8_t*)p), 7), vdupq_n_s16((int16_t)0xBFFF)));
  out[0] = vcvt_f32_f16(vget_low_f16(a));
  out[1] = vcvt_high_f32_f16(a);
}
static inline void e4m3x8u(const uint8_t* p, float32x4_t out[2]) {  // positive scales: one unsigned shift
  float16x8_t a = vreinterpretq_f16_u16(vshll_n_u8(vld1_u8(p), 7));
  out[0] = vcvt_f32_f16(vget_low_f16(a));
  out[1] = vcvt_high_f32_f16(a);
}
static inline int32x4_t blk_k0(const uint8_t* c, const int8x16_t* xh, const int8x16_t* xl, int8x16_t lut) {
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
static inline int32x4_t blk_gl(const uint8_t* c, const int8x16_t* xh, const int8x16_t* xl, int8x16_t lut) {
  const uint8x16_t m = vdupq_n_u8(0x0F);
  uint8x16_t v0 = vld1q_u8(c), v1 = vld1q_u8(c + 16);
  int8x16_t l0 = vqtbl1q_s8(lut, vandq_u8(v0, m)), h0 = vqtbl1q_s8(lut, vshrq_n_u8(v0, 4));
  int8x16_t l1 = vqtbl1q_s8(lut, vandq_u8(v1, m)), h1 = vqtbl1q_s8(lut, vshrq_n_u8(v1, 4));
  int32x4_t hi = vdotq_s32(vdotq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, xh[0]), h0, xh[1]), l1, xh[2]), h1, xh[3]);
  int32x4_t lo = vdotq_s32(vdotq_s32(vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, xl[0]), h0, xl[1]), l1, xl[2]), h1, xl[3]);
  return vmlaq_n_s32(lo, hi, 256);  // the group sums, (hi << 8) + lo
}

enum Epi { EXACT, FUSED };
// R rows at a time sharing the activation planes; GL: group-lane codes and planes; U: unsigned scale shift
template <int R, bool GL, bool U, Epi E>
static void nv_rows(const uint8_t* codes, const uint8_t* sc, float post, int K, int r0, int r1, const int8_t* XH, const int8_t* XL,
                    const float* XD, float* y) {
  const int NB = K / 64, rb = K / 2, rs = K / 16;
  const int8x16_t lut = vld1q_s8(LUT);
  int r = r0;
  for (; r + R <= r1; r += R) {
    const uint8_t* c[R];
    const uint8_t* s[R];
    float32x4_t a0[R], a1[R];
    for (int k = 0; k < R; k++) {
      c[k] = codes + (size_t)(r + k) * rb;
      s[k] = sc + (size_t)(r + k) * rs;
      a0[k] = a1[k] = vdupq_n_f32(0);
    }
    auto blk = [&](int bb, const float32x4_t* sv, float32x4_t* acc) {
      const int8_t* xh = XH + 64 * bb;
      const int8_t* xl = XL + 64 * bb;
      const int8x16_t Hv[4] = {vld1q_s8(xh), vld1q_s8(xh + 16), vld1q_s8(xh + 32), vld1q_s8(xh + 48)};
      const int8x16_t Lv[4] = {vld1q_s8(xl), vld1q_s8(xl + 16), vld1q_s8(xl + 32), vld1q_s8(xl + 48)};
      const float d = XD[bb];
      for (int k = 0; k < R; k++) {
        const int32x4_t g = GL ? blk_gl(c[k] + 32 * bb, Hv, Lv, lut) : blk_k0(c[k] + 32 * bb, Hv, Lv, lut);
        if (E == EXACT) acc[k] = vaddq_f32(acc[k], vmulq_f32(vcvtq_f32_s32(g), vmulq_n_f32(sv[k], d)));
        else acc[k] = vfmaq_f32(acc[k], vcvtq_f32_s32(g), vmulq_n_f32(sv[k], d));
      }
    };
    int b = 0;
    for (const int nb4 = NB & ~3; b < nb4; b += 2) {
      float32x4_t sv[R][2], s0[R], s1[R];
      for (int k = 0; k < R; k++) {
        if (U) e4m3x8u(s[k] + 4 * b, sv[k]);
        else e4m3x8(s[k] + 4 * b, sv[k]);
        s0[k] = sv[k][0];
        s1[k] = sv[k][1];
      }
      blk(b, s0, a0);
      blk(b + 1, s1, a1);
    }
    for (; b < NB; b++) {
      float32x4_t s0[R];
      for (int k = 0; k < R; k++) {
        float32x4_t t[2];
        e4m3x8(s[k] + 4 * b - 4, t);  // (the 4 scales of block b are the high half of the 8 from b - 1; NB >= 1)
        s0[k] = t[1];
      }
      blk(b, s0, a0);
    }
    for (int k = 0; k < R; k++) y[r + k - r0] = vaddvq_f32(vaddq_f32(a0[k], a1[k])) * post;
  }
  for (; r < r1; r++) nv_rows<1, GL, U, E>(codes, sc, post, K, r, r + 1, XH, XL, XD, y + (r - r0));
}

// the integer fold (sizing): per group, (dot x mantissa) << (exponent - the row's minimum), one float conversion and FMA
// with the activation scale per block; the row's 2^(Emin - 7) / 8 and the global scale at the end
static void nv_rows_ifold(const uint8_t* codes, const uint8_t* sc, const int8_t* emin, float gmul, int K, int r0, int r1, const int8_t* XH,
                          const int8_t* XL, const float* XD, float* y) {
  const int NB = K / 64, rb = K / 2, rs = K / 16;
  const int8x16_t lut = vld1q_s8(LUT);
  const uint8x16_t widx = {0, 255, 255, 255, 1, 255, 255, 255, 2, 255, 255, 255, 3, 255, 255, 255};
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * rb;
    const uint8_t* s = sc + (size_t)r * rs;
    const int32x4_t em = vdupq_n_s32(emin[r]);
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < NB; b++) {
      const int8_t* xh = XH + 64 * b;
      const int8_t* xl = XL + 64 * b;
      const int8x16_t Hv[4] = {vld1q_s8(xh), vld1q_s8(xh + 16), vld1q_s8(xh + 32), vld1q_s8(xh + 48)};
      const int8x16_t Lv[4] = {vld1q_s8(xl), vld1q_s8(xl + 16), vld1q_s8(xl + 32), vld1q_s8(xl + 48)};
      const int32x4_t g = blk_gl(c + 32 * b, Hv, Lv, lut);
      uint32_t w;
      std::memcpy(&w, s + 4 * b, 4);
      const int32x4_t sv = vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(w)), widx));
      const int32x4_t M = vorrq_s32(vandq_s32(sv, vdupq_n_s32(7)), vdupq_n_s32(8));
      const int32x4_t sh = vsubq_s32(vshrq_n_s32(sv, 3), em);
      const int32x4_t t = vshlq_s32(vmulq_s32(g, M), sh);
      acc = vfmaq_n_f32(acc, vcvtq_f32_s32(t), XD[b]);
    }
    y[r - r0] = vaddvq_f32(acc) * std::ldexp(gmul / 2.f / 8.f, emin[r] - 7);
  }
}

// ------------------------------------------------------------------------------------------------ MXFP8 candidates
static inline float16x8_t e4m3x8_f16(int8x8_t v) {
  return vreinterpretq_f16_s16(vandq_s16(vshlq_n_s16(vmovl_s8(v), 7), vdupq_n_s16((int16_t)0xBFFF)));
}
static inline float32x4_t mx_group4(const int8_t* c, const _Float16* x) {  // four accumulators
  float32x4_t a = vdupq_n_f32(0), b = a, e = a, f = a;
  int8x16_t v0 = vld1q_s8(c), v1 = vld1q_s8(c + 16);
  float16x8_t c0 = e4m3x8_f16(vget_low_s8(v0)), c1 = e4m3x8_f16(vget_high_s8(v0));
  float16x8_t c2 = e4m3x8_f16(vget_low_s8(v1)), c3 = e4m3x8_f16(vget_high_s8(v1));
  const float16_t* xp = (const float16_t*)x;
  float16x8_t x0 = vld1q_f16(xp), x1 = vld1q_f16(xp + 8), x2 = vld1q_f16(xp + 16), x3 = vld1q_f16(xp + 24);
  a = vfmlalq_low_f16(a, c0, x0);
  b = vfmlalq_high_f16(b, c0, x0);
  e = vfmlalq_low_f16(e, c1, x1);
  f = vfmlalq_high_f16(f, c1, x1);
  a = vfmlalq_low_f16(a, c2, x2);
  b = vfmlalq_high_f16(b, c2, x2);
  e = vfmlalq_low_f16(e, c3, x3);
  f = vfmlalq_high_f16(f, c3, x3);
  return vaddq_f32(vaddq_f32(a, e), vaddq_f32(b, f));
}
static inline float32x4_t e8m0x4(const uint8_t* p) {
  uint32_t w;
  std::memcpy(&w, p, 4);
  uint32x4_t u = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(w))));
  return vreinterpretq_f32_u32(vshlq_n_u32(u, 23));
}
static void mx_rows4(const uint8_t* codes, const uint8_t* sc, int K, int r0, int r1, const _Float16* xf, float* y) {
  const int NG = K / 32;
  for (int r = r0; r < r1; r++) {
    const int8_t* c = (const int8_t*)codes + (size_t)r * K;
    const uint8_t* s = sc + (size_t)r * NG;
    float32x4_t acc = vdupq_n_f32(0);
    for (int g = 0; g < NG; g += 4) {
      float32x4_t dots = vpaddq_f32(vpaddq_f32(mx_group4(c + 32 * g, xf + 32 * g), mx_group4(c + 32 * (g + 1), xf + 32 * (g + 1))),
                                    vpaddq_f32(mx_group4(c + 32 * (g + 2), xf + 32 * (g + 2)), mx_group4(c + 32 * (g + 3), xf + 32 * (g + 3))));
      acc = vfmaq_f32(acc, dots, e8m0x4(s + g));
    }
    y[r - r0] = vaddvq_f32(acc) * 256.f;
  }
}

// ------------------------------------------------------------------------------------------------ data
struct NvMat {
  const uint8_t *c, *s;
  int rows, K;
  float gmul;
  const int8_t* emin;
};
static double e4m3_val(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  const double v = e ? std::ldexp(1.0 + m / 8.0, e - 7) : std::ldexp(m / 8.0, -6);
  return (b & 0x80) ? -v : v;
}
static double e2m1_val(int n) { return LUT[n] / 2.0; }
static double e4m3w(int8_t b) {  // an MXFP8 weight
  return e4m3_val((uint8_t)b);
}

static bool read_all(int fd, uint8_t* dst, size_t n, off_t off) {
  size_t done = 0;
  while (done < n) {
    ssize_t r = pread(fd, dst + done, n - done, off + (off_t)done);
    if (r <= 0) return false;
    done += (size_t)r;
  }
  return true;
}

static void set_qos() { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: bench_kmulti_nv MODEL_DIR [threads] [reps]\n");
    return 2;
  }
  const std::string md = argv[1];
  const int NT = argc > 2 ? atoi(argv[2]) : 8, REPS = argc > 3 ? atoi(argv[3]) : 5;
  set_qos();
  const int NEV = 40, NEM = 24;  // 180 MB / 197 MB of records
  std::vector<uint8_t> nv((size_t)NEV * NV_REC), mx((size_t)NEM * MX_REC);
  {
    int fd = open((md + "/store/layer_00.bin").c_str(), O_RDONLY);
    int fm = open((md + "/store/layer_21.bin").c_str(), O_RDONLY);
    if (fd < 0 || fm < 0 || !read_all(fd, nv.data(), nv.size(), 0) || !read_all(fm, mx.data(), mx.size(), 0)) {
      fprintf(stderr, "cannot read the layer files\n");
      return 1;
    }
    close(fd);
    close(fm);
  }
  // the matrices; the group-lane copies; each row's minimum scale exponent; the scales' sign bits
  std::vector<NvMat> mats;
  std::vector<uint8_t> nvg(nv.size());
  std::vector<int8_t> emin;
  long neg = 0;
  for (int e = 0; e < NEV; e++) {
    const uint8_t* r = nv.data() + (size_t)e * NV_REC;
    float t[4];
    std::memcpy(t, r + NV_TR, 16);
    const float gm[2] = {t[0] * (6.f * 448.f / t[2]) / (6.f * 448.f), t[1] * (6.f * 448.f / t[3]) / (6.f * 448.f)};
    mats.push_back({r + NV_GU_C, r + NV_GU_S, 2 * I, H, gm[0], nullptr});
    mats.push_back({r + NV_DN_C, r + NV_DN_S, H, I, gm[1], nullptr});
  }
  emin.resize(mats.size() * H);
  for (size_t i = 0; i < mats.size(); i++) {
    NvMat& m = mats[i];
    const size_t off = (size_t)(m.c - nv.data());
    for (int r = 0; r < m.rows; r++) gl_codes(m.c + (size_t)r * m.K / 2, nvg.data() + off + (size_t)r * m.K / 2, m.K);
    for (int r = 0; r < m.rows; r++) {
      int lo = 99;
      for (int g = 0; g < m.K / 16; g++) {
        const uint8_t b = m.s[(size_t)r * m.K / 16 + g];
        neg += b >> 7;
        lo = std::min(lo, (b >> 3) & 15);
      }
      emin[i * H + r] = (int8_t)lo;
    }
    m.emin = emin.data() + i * H;
  }
  auto gl_of = [&](const NvMat& m) { return nvg.data() + (m.c - nv.data()); };

  // activations: gate-up input (H) and down input (I): normal with outliers (as the engine's I16 check)
  std::mt19937 rng(7);
  XBuf xh, xi;
  xh.init(H, Prec::I16);
  xi.init(I, Prec::I16);
  std::vector<float> fx(H);
  std::normal_distribution<float> nd(0, 1);
  for (auto& v : fx) v = nd(rng) * ((rng() % 31 == 0) ? 20.f : 1.f);
  xh.prep(fx.data(), H);
  for (int k = 0; k < I; k++) fx[k] = nd(rng) * 0.3f * ((rng() % 31 == 0) ? 20.f : 1.f);
  xi.prep(fx.data(), I);
  XG gh, gi;
  gl_x(xh.q, H, gh);
  gl_x(xi.q, I, gi);
  XBuf xhf, xif;  // MXFP8: fp16 activations
  xhf.init(H, Prec::F16);
  xif.init(I, Prec::F16);
  xhf.prep(fx.data(), H);  // (values only matter for the error columns)
  xif.prep(fx.data(), I);

  struct Cand {
    const char* name;
    bool exact_expected;
  };
  const Cand NV[] = {{"K0", true}, {"K1", true}, {"G2", true}, {"G4", true}, {"G2F", false}, {"G2I", false},
                     {"K0U", true}, {"K1U", true}, {"K1UF", false}, {"G4F", false}};
  auto run_nv = [&](int ci, const NvMat& m, int r0, int r1, float* y) {
    const bool gu = m.K == H;
    const XQ& q = gu ? xh.q : xi.q;
    const XG& g = gu ? gh : gi;
    const uint8_t* cg = gl_of(m);
    const float post = m.gmul * 128.f;
    switch (ci) {
      case 0: mv_nvfp4(m.c, m.s, m.gmul, m.K, r0, r1, q, y); break;
      case 1: nv_rows<4, false, false, EXACT>(m.c, m.s, post, m.K, r0, r1, q.hp, q.lp, q.d, y); break;
      case 2: nv_rows<2, true, true, EXACT>(cg, m.s, post, m.K, r0, r1, g.h.data(), g.l.data(), q.d, y); break;
      case 3: nv_rows<4, true, true, EXACT>(cg, m.s, post, m.K, r0, r1, g.h.data(), g.l.data(), q.d, y); break;
      case 4: nv_rows<2, true, true, FUSED>(cg, m.s, post, m.K, r0, r1, g.h.data(), g.l.data(), q.d, y); break;
      case 5: nv_rows_ifold(cg, m.s, m.emin, m.gmul, m.K, r0, r1, g.h.data(), g.l.data(), q.d, y); break;
      case 6: nv_rows<2, false, true, EXACT>(m.c, m.s, post, m.K, r0, r1, q.hp, q.lp, q.d, y); break;
      case 7: nv_rows<4, false, true, EXACT>(m.c, m.s, post, m.K, r0, r1, q.hp, q.lp, q.d, y); break;
      case 8: nv_rows<4, false, true, FUSED>(m.c, m.s, post, m.K, r0, r1, q.hp, q.lp, q.d, y); break;
      case 9: nv_rows<4, true, true, FUSED>(cg, m.s, post, m.K, r0, r1, g.h.data(), g.l.data(), q.d, y); break;
    }
  };
  const int NC = 10;

  // (a) bits / error against float64 on every row of 8 experts (gate-up and down)
  printf("{\"scales_with_sign_bit\":%ld}\n", neg);
  std::vector<double> ref;
  std::vector<float> y0(H), y1(H);
  double err[NC] = {0}, errmax[NC] = {0}, sref = 0;
  long differ[NC] = {0}, nrows = 0;
  for (int mi = 0; mi < 16; mi++) {
    const NvMat& m = mats[mi];
    const XQ& q = m.K == H ? xh.q : xi.q;
    for (int r = 0; r < m.rows; r++) {
      double t = 0;
      for (int k = 0; k < m.K; k++) {
        const uint8_t byte = m.c[(size_t)r * m.K / 2 + k / 2];
        const int n = (k & 1) ? byte >> 4 : byte & 15;
        const double w = e2m1_val(n) * e4m3_val(m.s[(size_t)r * m.K / 16 + k / 16]) * (double)m.gmul;
        const double xv = ((double)q.hn[k] * 256.0 + q.ln[k]) * (double)q.d[k / 64];
        t += w * xv;
      }
      ref.push_back(t);
      sref += std::fabs(t);
    }
    run_nv(0, m, 0, m.rows, y0.data());
    for (int ci = 0; ci < NC; ci++) {
      run_nv(ci, m, 0, m.rows, y1.data());
      for (int r = 0; r < m.rows; r++) {
        const double e = std::fabs((double)y1[r] - ref[ref.size() - m.rows + r]);
        err[ci] += e;
        errmax[ci] = std::max(errmax[ci], e / (std::fabs(ref[ref.size() - m.rows + r]) + 1e-30));
        differ[ci] += std::memcmp(&y0[r], &y1[r], 4) != 0;
      }
    }
    nrows += m.rows;
  }
  for (int ci = 0; ci < NC; ci++)
    printf("{\"cand\":\"%s\",\"rows\":%ld,\"rows_differ_from_K0\":%ld,\"mean_abs_err_over_mean_abs_ref\":%.3e,\"max_rel_err\":%.3e}\n", NV[ci].name, nrows,
           differ[ci], err[ci] / sref, errmax[ci]);

  // (b) one thread, in cache: a gate-up matrix repeated
  double ins_b[NC], cyc_b[NC];
  {
    const NvMat& m = mats[0];
    const double by = (double)m.rows * (m.K / 2 + m.K / 16);
    for (int ci = 0; ci < NC; ci++) {
      for (int w = 0; w < 3; w++) run_nv(ci, m, 0, m.rows, y1.data());
      ins_b[ci] = cyc_b[ci] = 1e9;
      for (int trial = 0; trial < 9; trial++) {  // the least cycles of 9 trials (a P core, undisturbed)
        uint64_t a[2], b[2];
        thread_selfcounts(1, a, sizeof a);
        for (int rep = 0; rep < 5; rep++) run_nv(ci, m, 0, m.rows, y1.data());
        thread_selfcounts(1, b, sizeof b);
        if ((b[1] - a[1]) / (5 * by) < cyc_b[ci]) ins_b[ci] = (b[0] - a[0]) / (5 * by), cyc_b[ci] = (b[1] - a[1]) / (5 * by);
      }
    }
  }
  // (c) all threads streaming over the set
  auto stream = [&](auto&& kernel_of_chunk, size_t nchunks, double bytes) {
    std::vector<double> gbs{0.0};
    if (NT <= 0) return std::make_pair(0.0, 0.0);  // threads 0: one-thread columns only
    gbs.clear();
    for (int rep = 0; rep < REPS; rep++) {
      std::atomic<size_t> next{0};
      auto t0 = sclk::now();
      std::vector<std::thread> th;
      for (int t = 0; t < NT; t++)
        th.emplace_back([&]() {
          set_qos();
          std::vector<float> y(64);
          for (size_t ch; (ch = next.fetch_add(1)) < nchunks;) kernel_of_chunk(ch, y.data());
        });
      for (auto& t : th) t.join();
      gbs.push_back(bytes / std::chrono::duration<double>(sclk::now() - t0).count() / 1e9);
    }
    std::sort(gbs.begin(), gbs.end());
    return std::make_pair(gbs.back(), gbs[gbs.size() / 2]);
  };
  std::vector<std::pair<int, int>> nvch;  // (matrix, first row)
  double nvbytes = 0;
  for (size_t i = 0; i < mats.size(); i++) {
    for (int r = 0; r < mats[i].rows; r += 64) nvch.push_back({(int)i, r});
    nvbytes += (double)mats[i].rows * (mats[i].K / 2 + mats[i].K / 16);
  }
  for (int ci = 0; ci < NC; ci++) {
    auto res = stream([&](size_t ch, float* y) { const NvMat& m = mats[nvch[ch].first]; run_nv(ci, m, nvch[ch].second, std::min(m.rows, nvch[ch].second + 64), y); },
                      nvch.size(), nvbytes);
    printf("{\"fmt\":\"nvfp4\",\"cand\":\"%s\",\"ins_per_byte_1t\":%.3f,\"cyc_per_byte_1t\":%.3f,\"gb_s_%dt_best\":%.1f,\"gb_s_median\":%.1f}\n", NV[ci].name,
           ins_b[ci], cyc_b[ci], NT, res.first, res.second);
  }

  // MXFP8: M0 engine, M1 four accumulators
  struct MxMat {
    const uint8_t *c, *s;
    int rows, K;
  };
  std::vector<MxMat> mm;
  for (int e = 0; e < NEM; e++) {
    const uint8_t* r = mx.data() + (size_t)e * MX_REC;
    mm.push_back({r + MX_GU_C, r + MX_GU_S, 2 * I, H});
    mm.push_back({r + MX_DN_C, r + MX_DN_S, H, I});
  }
  auto run_mx = [&](int ci, const MxMat& m, int r0, int r1, float* y) {
    const XQ& q = m.K == H ? xhf.q : xif.q;
    if (ci == 0) mv_mxfp8(m.c, m.s, m.K, r0, r1, q, y);
    else mx_rows4(m.c, m.s, m.K, r0, r1, q.fn, y);
  };
  {
    double e2[2] = {0, 0}, sr = 0;
    long dif = 0;
    for (int mi = 0; mi < 4; mi++) {
      const MxMat& m = mm[mi];
      const XQ& q = m.K == H ? xhf.q : xif.q;
      run_mx(0, m, 0, m.rows, y0.data());
      run_mx(1, m, 0, m.rows, y1.data());
      for (int r = 0; r < m.rows; r++) {
        double t = 0;
        for (int k = 0; k < m.K; k++)
          t += e4m3w((int8_t)m.c[(size_t)r * m.K + k]) * std::ldexp(1.0, (int)m.s[(size_t)r * m.K / 32 + k / 32] - 127) * (double)q.fn[k];
        sr += std::fabs(t);
        e2[0] += std::fabs(y0[r] - t);
        e2[1] += std::fabs(y1[r] - t);
        dif += std::memcmp(&y0[r], &y1[r], 4) != 0;
      }
    }
    printf("{\"fmt\":\"mxfp8\",\"rows_differ_M1\":%ld,\"err_M0\":%.3e,\"err_M1\":%.3e}\n", dif, e2[0] / sr, e2[1] / sr);
  }
  std::vector<std::pair<int, int>> mxch;
  double mxbytes = 0;
  for (size_t i = 0; i < mm.size(); i++) {
    for (int r = 0; r < mm[i].rows; r += 64) mxch.push_back({(int)i, r});
    mxbytes += (double)mm[i].rows * (mm[i].K + mm[i].K / 32);
  }
  for (int ci = 0; ci < 2; ci++) {
    const MxMat& m0 = mm[0];
    const double by = (double)m0.rows * (m0.K + m0.K / 32);
    for (int w = 0; w < 3; w++) run_mx(ci, m0, 0, m0.rows, y1.data());
    uint64_t a[2], b[2], ba[2] = {0, ~0ull};
    for (int trial = 0; trial < 9; trial++) {
      thread_selfcounts(1, a, sizeof a);
      for (int rep = 0; rep < 3; rep++) run_mx(ci, m0, 0, m0.rows, y1.data());
      thread_selfcounts(1, b, sizeof b);
      if (b[1] - a[1] < ba[1]) ba[0] = (b[0] - a[0]) * 5, ba[1] = (b[1] - a[1]) * 5;
    }
    a[0] = a[1] = 0;
    b[0] = ba[0];
    b[1] = ba[1];
    auto res = stream([&](size_t ch, float* y) { const MxMat& m = mm[mxch[ch].first]; run_mx(ci, m, mxch[ch].second, std::min(m.rows, mxch[ch].second + 64), y); },
                      mxch.size(), mxbytes);
    printf("{\"fmt\":\"mxfp8\",\"cand\":\"M%d\",\"ins_per_byte_1t\":%.3f,\"cyc_per_byte_1t\":%.3f,\"gb_s_%dt_best\":%.1f,\"gb_s_median\":%.1f}\n", ci,
           (b[0] - a[0]) / (15 * by), (b[1] - a[1]) / (15 * by), NT, res.first, res.second);
  }
  // AFF4 at the same shapes (the 35B's kernel): the nv codes read as affine 4-bit with fp16 scales / biases
  {
    std::vector<_Float16> sb((size_t)2 * I * H / 64 * 2, (_Float16)0.01f);
    const XQ& q = xh.q;
    const NvMat& m = mats[0];
    const double by = (double)m.rows * (m.K / 2 + m.K / 64 * 4);
    uint64_t a[2], b[2], ba[2] = {0, ~0ull};
    for (int w = 0; w < 3; w++) mv_aff4(m.c, sb.data(), sb.data() + sb.size() / 2, m.K, 0, m.rows, q, y1.data());
    for (int trial = 0; trial < 9; trial++) {
      thread_selfcounts(1, a, sizeof a);
      for (int rep = 0; rep < 5; rep++) mv_aff4(m.c, sb.data(), sb.data() + sb.size() / 2, m.K, 0, m.rows, q, y1.data());
      thread_selfcounts(1, b, sizeof b);
      if (b[1] - a[1] < ba[1]) ba[0] = (b[0] - a[0]) * 6, ba[1] = (b[1] - a[1]) * 6;
    }
    a[0] = a[1] = 0;
    b[0] = ba[0];
    b[1] = ba[1];
    double ab = 0;
    for (auto& mt : mats) ab += (double)mt.rows * (mt.K / 2 + mt.K / 64 * 4);
    auto res = stream(
        [&](size_t ch, float* y) {
          const NvMat& mt = mats[nvch[ch].first];
          const int r0 = nvch[ch].second, r1 = std::min(mt.rows, r0 + 64);
          // (scales / biases: one shared row set, in cache: the AFF4 bytes per row are counted as if read)
          const XQ& qq = mt.K == H ? xh.q : xi.q;
          mv_aff4(mt.c, sb.data(), sb.data() + sb.size() / 2, mt.K, r0, r1, qq, y);
        },
        nvch.size(), nvbytes);
    printf("{\"fmt\":\"aff4\",\"cand\":\"A0\",\"ins_per_byte_1t\":%.3f,\"cyc_per_byte_1t\":%.3f,\"gb_s_%dt_best_codes_only\":%.1f,\"gb_s_median\":%.1f}\n",
           (b[0] - a[0]) / (30 * by), (b[1] - a[1]) / (30 * by), NT, res.first, res.second);
    (void)ab;
  }
  return 0;
}
