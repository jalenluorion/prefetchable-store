// One decode position through the 35B's 4-bit affine expert kernel (I16 activations), NEON (mv_aff4) against streaming
// SVE on the SME units (512-bit SDOT on the unpacked nibbles, the group totals to a buffer, the float epilogue in normal
// mode with mv_aff4's operation order), alone and side by side: which cores and units together read the most expert bytes
// per second when the kernel is instruction-bound. Weights: 384 MB of real code bytes from the store's layer files (any
// layer file of the model folder) as 2048-input rows, fp16 scales and biases; every SME row checked bit for bit against
// mv_aff4. No model process. The name matches engine/bench/guard.py's model-process pattern, so the queue waits for it.
//   bench_kmulti_sme4 LAYER_FILE [seconds per cell]
#include <arm_neon.h>
#include <arm_sme.h>
#include <pthread.h>
#include <pthread/qos.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "dmeng/kernels.h"

using namespace dm;
using sclk = std::chrono::steady_clock;

static constexpr int K = 2048, NG = K / 64, RB = K / 2, CH = 64;  // inputs, groups, code bytes per row, rows per claim

// the activation planes rearranged for 64-byte code vectors (two groups): for byte j (0..31) of a group's 32 code bytes,
// the low nibble pairs with perm position j < 16 ? j : j + 16, the high nibble with j < 16 ? j + 16 : j + 32
struct XS {
  alignas(64) int8_t lh[K / 2], ll[K / 2], hh[K / 2], hl[K / 2];
};
static void prep_xs(const XQ& x, XS& s) {
  for (int g = 0; g < NG; g++)
    for (int j = 0; j < 32; j++) {
      const int pl = g * 64 + (j < 16 ? j : j + 16), ph = g * 64 + (j < 16 ? j + 16 : j + 32);
      s.lh[g * 32 + j] = x.hp[pl];
      s.ll[g * 32 + j] = x.lp[pl];
      s.hh[g * 32 + j] = x.hp[ph];
      s.hl[g * 32 + j] = x.lp[ph];
    }
}

// group totals of rows [r0, r1): T[(r - r0) * NG + g]
__arm_locally_streaming static void sme_totals(const uint8_t* codes, int r0, int r1, const XS& x, int32_t* T) {
  const svbool_t p8 = svptrue_b8(), p32 = svptrue_b32();
  const svbool_t plo = svwhilelt_b32(0, 8), phi = svnot_b_z(p32, plo);
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * RB;
    int32_t* t = T + (size_t)(r - r0) * NG;
    for (int g = 0; g < NG; g += 2) {
      const svuint8_t v = svld1_u8(p8, c + 32 * g);
      const svint8_t lo = svreinterpret_s8_u8(svand_n_u8_x(p8, v, 0x0F)), hi = svreinterpret_s8_u8(svlsr_n_u8_x(p8, v, 4));
      svint32_t a = svdot_s32(svdup_n_s32(0), lo, svld1_s8(p8, x.lh + 32 * g));
      a = svdot_s32(a, hi, svld1_s8(p8, x.hh + 32 * g));
      svint32_t b = svdot_s32(svdup_n_s32(0), lo, svld1_s8(p8, x.ll + 32 * g));
      b = svdot_s32(b, hi, svld1_s8(p8, x.hl + 32 * g));
      const svint32_t s = svadd_s32_x(p32, svlsl_n_s32_x(p32, a, 8), b);
      t[g] = (int32_t)svaddv_s32(plo, s);
      t[g + 1] = (int32_t)svaddv_s32(phi, s);
    }
  }
}

// mv_aff4's float part (I16) on the totals: groups of four, the same operations in the same order
static void epilogue(const int32_t* T, const _Float16* sc, const _Float16* bi, int r0, int r1, const XQ& x, float* y) {
  for (int r = r0; r < r1; r++) {
    const int32_t* t = T + (size_t)(r - r0) * NG;
    const _Float16* s = sc + (size_t)r * NG;
    const _Float16* b = bi + (size_t)r * NG;
    float32x4_t acc = vdupq_n_f32(0);
    for (int g = 0; g < NG; g += 4) {
      const float32x4_t dots = vcvtq_f32_s32(vld1q_s32(t + g));
      acc = vfmaq_f32(acc, dots, vmulq_f32(vcvt_f32_f16(vld1_f16((const float16_t*)s + g)), vld1q_f32(x.d + g)));
      acc = vfmaq_f32(acc, vcvt_f32_f16(vld1_f16((const float16_t*)b + g)), vld1q_f32(x.gsum + g));
    }
    y[r - r0] = vaddvq_f32(acc);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: bench_kmulti_sme4 LAYER_FILE [seconds]\n");
    return 2;
  }
  const double secs = argc > 2 ? atof(argv[2]) : 2.0;
  const size_t rows = (size_t)384 * 1024 * 1024 / RB;
  std::vector<uint8_t> codes(rows * RB);
  {
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(codes.data(), 1, codes.size(), f) != codes.size()) {
      fprintf(stderr, "need 384 MB from %s\n", argv[1]);
      return 1;
    }
    fclose(f);
  }
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  std::vector<_Float16> sc(rows * NG), bi(rows * NG);
  for (auto& v : sc) v = (_Float16)(0.01f * U(rng));
  for (auto& v : bi) v = (_Float16)(0.05f * U(rng));
  std::vector<float> xf(K);
  for (auto& v : xf) v = U(rng) * (U(rng) > 0.95f ? 8.f : 1.f);
  std::vector<uint8_t> xm(xq_bytes(K) + 64);
  XQ xq;
  xq_bind(xq, (void*)(((uintptr_t)xm.data() + 63) & ~(uintptr_t)63), K, Prec::I16);
  xq_prep(xq, xf.data());
  static XS xs;
  prep_xs(xq, xs);
  const double bytes_row = RB + NG * 4.0;
  std::vector<float> yref(rows), y(rows);
  // the check: SME totals + epilogue == mv_aff4, every row of the first 4096
  {
    std::vector<int32_t> T(4096 * NG);
    mv_aff4(codes.data(), sc.data(), bi.data(), K, 0, 4096, xq, yref.data());
    sme_totals(codes.data(), 0, 4096, xs, T.data());
    epilogue(T.data(), sc.data(), bi.data(), 0, 4096, xq, y.data());
    if (std::memcmp(y.data(), yref.data(), 4096 * 4) != 0) {
      fprintf(stderr, "SME rows differ from mv_aff4\n");
      return 1;
    }
    printf("check: 4096 rows bit-identical to mv_aff4\n");
  }
  struct Cell {
    const char* name;
    int neon, sme_p, sme_e;
  } cells[] = {{"NEON x4", 4, 0, 0}, {"NEON x8", 8, 0, 0}, {"NEON x10", 10, 0, 0}, {"SME P", 0, 1, 0}, {"SME E", 0, 0, 1},
               {"SME P + E", 0, 1, 1}, {"SME P + NEON x4", 4, 1, 0}, {"SME P + E + NEON x4", 4, 1, 1}, {"SME P + NEON x8", 8, 1, 0},
               {"SME P + E + NEON x8", 8, 1, 1}, {"SME P + E + NEON x6", 6, 1, 1}};
  for (const Cell& cl : cells) {
    std::atomic<size_t> next{0};
    std::atomic<bool> stop{false};
    std::atomic<double> by_neon{0}, by_sme{0};
    auto neon = [&]() {
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      std::vector<float> out(CH);
      double n = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        const size_t r0 = (next.fetch_add(CH, std::memory_order_relaxed)) % rows;
        mv_aff4(codes.data(), sc.data(), bi.data(), K, (int)r0, (int)(r0 + CH), xq, out.data());
        n += CH;
      }
      double e = by_neon.load();
      while (!by_neon.compare_exchange_weak(e, e + n)) {}
    };
    auto sme = [&](bool p) {
      pthread_set_qos_class_self_np(p ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_UTILITY, 0);
      std::vector<int32_t> T(CH * NG);
      std::vector<float> out(CH);
      double n = 0;
      while (!stop.load(std::memory_order_relaxed)) {
        const size_t r0 = (next.fetch_add(CH, std::memory_order_relaxed)) % rows;
        sme_totals(codes.data(), (int)r0, (int)(r0 + CH), xs, T.data());
        epilogue(T.data(), sc.data(), bi.data(), (int)r0, (int)(r0 + CH), xq, out.data());
        n += CH;
      }
      double e = by_sme.load();
      while (!by_sme.compare_exchange_weak(e, e + n)) {}
    };
    std::vector<std::thread> th;
    for (int i = 0; i < cl.neon; i++) th.emplace_back(neon);
    if (cl.sme_p) th.emplace_back(sme, true);
    if (cl.sme_e) th.emplace_back(sme, false);
    const auto t0 = sclk::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(secs));
    stop.store(true);
    for (auto& t : th) t.join();
    const double s = std::chrono::duration<double>(sclk::now() - t0).count();
    printf("%-22s total %6.1f GB/s   NEON %6.1f   SME %6.1f\n", cl.name, (by_neon.load() + by_sme.load()) * bytes_row / s / 1e9,
           by_neon.load() * bytes_row / s / 1e9, by_sme.load() * bytes_row / s / 1e9);
    fflush(stdout);
  }
  return 0;
}
