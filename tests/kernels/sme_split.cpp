// SME2 for prefill, the split nobody tried: one thread per core cluster in streaming mode on whole-matrix pieces (the
// shared kernels_sme.cpp gemm), the other cores on the NEON i8mm gemm, rows claimed from the two ends of each matrix.
// Prefill-shaped work: 8 positions per weight pass, the real matrix shapes of both models (8-bit resident attention
// matrices; the 35B's 4-bit experts), a set larger than the caches. Reported: weight GB/s and int8 MACs/s for NEON alone,
// SME alone (one thread at user-interactive QoS, one at utility: macOS places them on the P and the E cluster), and the
// split; every output compared bit for bit with the NEON pass. No model process.
//   build_cs/sme_split [reps]
#include <arm_neon.h>
#include <pthread.h>
#include <pthread/qos.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "dmeng/kernels_batch.h"

using namespace dm;
using sclk = std::chrono::steady_clock;

struct Mat {
  int rows, cols, fmt;  // fmt 8 (AFF8) or 4 (AFF4)
  std::vector<uint8_t> c;
  std::vector<_Float16> s, b;
  size_t bytes() const { return c.size() + (s.size() + b.size()) * 2; }
};

static void gemm_neon(const Mat& m, int r0, int r1, const XQB& x, float* Y, size_t ldy) {
  if (m.fmt == 8) gemm_aff8_i16(m.c.data(), m.s.data(), m.b.data(), m.cols, r0, r1, x, 0, 8, Y + r0, ldy);
  else gemm_aff4_i16(m.c.data(), m.s.data(), m.b.data(), m.cols, r0, r1, x, 0, 8, Y + r0, ldy);
}
static void gemm_sme(const Mat& m, int r0, int r1, const XQB& x, float* Y, size_t ldy) {
  if (m.fmt == 8) gemm_aff8_i16_sme(m.c.data(), m.s.data(), m.b.data(), m.cols, r0, r1, x, 0, 8, Y + r0, ldy);
  else gemm_aff4_i16_sme(m.c.data(), m.s.data(), m.b.data(), m.cols, r0, r1, x, 0, 8, Y + r0, ldy);
}

// one pass over every matrix: each matrix's 64-row blocks are claimed one by one by NEON threads from the front and in
// runs of `sme_run` blocks by SME threads from the back (a flag per block: no block twice)
struct Pass {
  const std::vector<Mat>* ms;
  const XQB* x;
  std::vector<std::vector<float>>* Y;
  std::vector<std::unique_ptr<std::atomic<int>[]>> taken;
  std::vector<int> nb;
  std::atomic<int> cur_front{0}, cur_back{0};
  int sme_run = 16;
};

static void neon_worker(Pass& P, int qos) {
  pthread_set_qos_class_self_np((qos_class_t)qos, 0);
  const auto& ms = *P.ms;
  for (size_t mi = 0; mi < ms.size(); mi++) {
    const Mat& m = ms[mi];
    for (int b = 0; b < P.nb[mi]; b++) {
      if (P.taken[mi][b].exchange(1, std::memory_order_acq_rel)) continue;
      gemm_neon(m, b * 64, std::min(m.rows, b * 64 + 64), *P.x, (*P.Y)[mi].data(), m.rows);
    }
  }
}
static void sme_worker(Pass& P, int qos) {
  pthread_set_qos_class_self_np((qos_class_t)qos, 0);
  const auto& ms = *P.ms;
  for (size_t mi = 0; mi < ms.size(); mi++) {
    const Mat& m = ms[mi];
    for (int b = P.nb[mi] - 1; b >= 0;) {  // from the back: a run of consecutive free blocks, one kernel call
      int hi = b;
      while (b >= 0 && hi - b < P.sme_run && !P.taken[mi][b].exchange(1, std::memory_order_acq_rel)) b--;
      if (hi > b) gemm_sme(m, (b + 1) * 64, std::min(m.rows, (hi + 1) * 64), *P.x, (*P.Y)[mi].data(), m.rows);
      if (hi == b) b--;  // the block at b was taken by another thread: look further front
    }
  }
}

static double run_pass(const std::vector<Mat>& ms, const XQB& x, std::vector<std::vector<float>>& Y, int n_neon, int n_sme, int sme_run) {
  Pass P;
  P.ms = &ms;
  P.x = &x;
  P.Y = &Y;
  P.sme_run = sme_run;
  for (auto& m : ms) {
    const int nb = (m.rows + 63) / 64;
    P.nb.push_back(nb);
    P.taken.emplace_back(new std::atomic<int>[nb]);
    for (int i = 0; i < nb; i++) P.taken.back()[i].store(0);
  }
  auto t0 = sclk::now();
  std::vector<std::thread> th;
  for (int i = 0; i < n_sme; i++) th.emplace_back(sme_worker, std::ref(P), i == 0 ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_UTILITY);
  for (int i = 0; i < n_neon; i++) th.emplace_back(neon_worker, std::ref(P), QOS_CLASS_USER_INTERACTIVE);
  for (auto& t : th) t.join();
  return std::chrono::duration<double, std::milli>(sclk::now() - t0).count();
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 7;
  std::mt19937 rng(1);
  auto mk = [&](int rows, int cols, int fmt) {
    Mat m{rows, cols, fmt, {}, {}, {}};
    m.c.resize((size_t)rows * cols / (fmt == 8 ? 1 : 2));
    for (auto& v : m.c) v = (uint8_t)(rng() & 0xFF);
    m.s.resize((size_t)rows * (cols / 64));
    m.b.resize(m.s.size());
    for (auto& v : m.s) v = (_Float16)(0.001f + 0.0001f * (rng() % 10));
    for (auto& v : m.b) v = (_Float16)(-0.1f + 0.001f * (rng() % 100));
    return m;
  };
  struct Set {
    const char* name;
    std::vector<Mat> ms;
    int K;
  };
  std::vector<Set> sets(3);
  sets[0].name = "inkling attention (AFF8: q, g 3072x2304, k, v 1024x2304, o 2304x3072) x 8 layers";
  sets[0].K = 2304;
  for (int l = 0; l < 8; l++) {
    sets[0].ms.push_back(mk(3072, 2304, 8));
    sets[0].ms.push_back(mk(3072, 2304, 8));
    sets[0].ms.push_back(mk(1024, 2304, 8));
    sets[0].ms.push_back(mk(1024, 2304, 8));
  }
  sets[1].name = "qwen3.6 linear attention (AFF8: qkv 8192x2048, z 4096x2048) x 8 layers";
  sets[1].K = 2048;
  for (int l = 0; l < 8; l++) {
    sets[1].ms.push_back(mk(8192, 2048, 8));
    sets[1].ms.push_back(mk(4096, 2048, 8));
  }
  sets[2].name = "qwen3.6 experts (AFF4: gate, up 512x2048) x 96";
  sets[2].K = 2048;
  for (int e = 0; e < 96; e++) {
    sets[2].ms.push_back(mk(512, 2048, 4));
    sets[2].ms.push_back(mk(512, 2048, 4));
  }
  printf("{\"reps\":%d,\"positions\":8,\"results\":[\n", reps);
  bool first = true;
  for (auto& S : sets) {
    XQB x;
    x.reserve(S.K, 8);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xv(S.K);
    for (int i = 0; i < 8; i++) {
      for (auto& v : xv) v = nd(rng);
      xqb_set(x, i, xv.data());
    }
    xqb_pad(x);
    double gb = 0, macs = 0;
    for (auto& m : S.ms) gb += m.bytes() / 1e9, macs += (double)m.rows * m.cols * 8;
    std::vector<std::vector<float>> ref(S.ms.size()), out(S.ms.size());
    for (size_t i = 0; i < S.ms.size(); i++) ref[i].assign((size_t)S.ms[i].rows * 8, 0.f), out[i] = ref[i];
    run_pass(S.ms, x, ref, 8, 0, 16);  // the reference: NEON, 8 threads
    struct Mode {
      const char* name;
      int neon, sme, run;
    } modes[] = {{"neon_8", 8, 0, 16},       {"neon_10", 10, 0, 16},   {"sme_1", 0, 1, 16},      {"sme_2_one_per_cluster", 0, 2, 16},
                 {"split_6neon_2sme", 6, 2, 16}, {"split_8neon_2sme", 8, 2, 16}, {"split_8neon_1sme", 8, 1, 16}};
    for (auto& md : modes) {
      std::vector<double> t;
      for (int r = 0; r < reps; r++) {
        for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
        t.push_back(run_pass(S.ms, x, out, md.neon, md.sme, md.run));
      }
      bool same = true;
      for (size_t i = 0; i < out.size() && same; i++) same = std::memcmp(out[i].data(), ref[i].data(), out[i].size() * 4) == 0;
      std::sort(t.begin(), t.end());
      const double ms = t[t.size() / 2];
      printf("%s{\"set\":\"%s\",\"mode\":\"%s\",\"weight_gb\":%.3f,\"ms\":%.2f,\"weight_gb_s\":%.1f,\"gmacs_s\":%.0f,\"bits_equal_neon\":%s}\n",
             first ? "" : ",", S.name, md.name, gb, ms, gb / (ms / 1e3), macs / (ms / 1e3) / 1e9, same ? "true" : "false");
      first = false;
      fflush(stdout);
    }
  }
  printf("]}\n");
  return 0;
}
