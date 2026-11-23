// Inkling's expert kernels at prefill user counts: how fast is one expert's work as a function of how many positions use
// it (8 to 512 users: a 256-position chunk gives 16 per expert, an 8K span's application up to ~500 if it ran as one).
// NVFP4 (layers 0-19): the packed i8mm gemm (gemm_nvfp4_i16, what Core::moe_chunk uses from 6 users) and the multi-vector
// kernel in groups of 8 (mv_nvfp4_multi, below 6 users); MXFP8 (layers 20-23): the multi-vector kernel in groups of 8
// (mv_mxfp8_multi, the only exact batched path: the reference is the fp16 FMLAL chain of mv_mxfp8). Real shapes (gate-up
// 2I x H fused, down H x I; H 2304, I 1152), random records, a set larger than the caches, all threads. Reported: int8 /
// fp16 MACs per second and the expert compute per prompt token this implies (44 NVFP4 + 4 MXFP8 applications x 8 picks).
// Every kernel's outputs are compared bit for bit with the one-vector kernel's (mv_nvfp4 / mv_mxfp8). No model process.
//   expert_users [reps] [threads]
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
#include "dmeng/weights.h"

using namespace dm;
using sclk = std::chrono::steady_clock;

constexpr int H = 2304, I = 1152;

struct Exp {  // one expert's record: fused gate-up (2I x H) and down (H x I)
  std::vector<uint8_t> gc, gs, dc, ds;
};

static void make(Exp& e, bool fp8, std::mt19937& rng) {
  auto codes = [&](std::vector<uint8_t>& v, size_t n) {
    v.resize(n);
    for (auto& b : v) {
      b = (uint8_t)(rng() & 0xFF);
      if (fp8 && (b & 0x7F) == 0x7F) b ^= 1;  // E4M3: no NaN codes
    }
  };
  auto scales = [&](std::vector<uint8_t>& v, size_t n) {
    v.resize(n);
    for (auto& b : v) b = fp8 ? (uint8_t)(120 + rng() % 8) : (uint8_t)(0x30 + rng() % 16);  // E8M0 near 2^-4 / E4M3 near 2^-3
  };
  const size_t gr = 2 * I, dr = H;
  codes(e.gc, gr * H / (fp8 ? 1 : 2));
  scales(e.gs, gr * H / (fp8 ? 32 : 16));
  codes(e.dc, dr * I / (fp8 ? 1 : 2));
  scales(e.ds, dr * I / (fp8 ? 32 : 16));
}

// work list: (expert, matrix, row block); threads claim them one by one
struct Pass {
  const std::vector<Exp>* ex;
  bool fp8, packed;
  int U;
  const XQB *xg, *xd;                      // packed inputs (gate-up, down)
  const std::vector<XBuf> *qg, *qd;        // one-vector inputs
  std::vector<std::vector<float>>* Y;      // [expert * 2 + m] [U][rows]
  int nblk_g, nblk_d;
  std::atomic<int> next{0};
};

static void worker(Pass& P) {
  const int RB = 64, total = (int)P.ex->size() * (P.nblk_g + P.nblk_d);
  std::vector<const XQ*> xs(8);
  std::vector<float*> ys(8);
  std::vector<float> tmp((size_t)P.U * RB);
  for (int j; (j = P.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
    const int per = P.nblk_g + P.nblk_d, e = j / per, b = j % per;
    const bool g = b < P.nblk_g;
    const Exp& E = (*P.ex)[e];
    const int rows = g ? 2 * I : H, K = g ? H : I, r0 = (g ? b : b - P.nblk_g) * RB, r1 = std::min(rows, r0 + RB);
    const uint8_t* c = g ? E.gc.data() : E.dc.data();
    const uint8_t* s = g ? E.gs.data() : E.ds.data();
    float* Y = (*P.Y)[(size_t)e * 2 + (g ? 0 : 1)].data();
    if (P.packed) {
      gemm_nvfp4_i16(c, s, 1.f, K, r0, r1, g ? *P.xg : *P.xd, 0, P.U, Y + r0, rows);
      continue;
    }
    const std::vector<XBuf>& q = g ? *P.qg : *P.qd;
    for (int u0 = 0; u0 < P.U; u0 += 8) {
      const int nu = std::min(8, P.U - u0);
      for (int k = 0; k < nu; k++) {
        xs[k] = &q[u0 + k].q;
        ys[k] = Y + (size_t)(u0 + k) * rows + r0;
      }
      if (P.fp8) mv_mxfp8_multi(c, s, K, r0, r1, xs.data(), nu, ys.data());
      else mv_nvfp4_multi(c, s, 1.f, K, r0, r1, xs.data(), nu, ys.data());
    }
  }
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 5;
  const int nth = argc > 2 ? atoi(argv[2]) : (int)std::thread::hardware_concurrency();
  std::mt19937 rng(7);
  printf("{\"reps\":%d,\"threads\":%d,\"H\":%d,\"I\":%d,\"results\":[\n", reps, nth, H, I);
  bool first = true;
  for (int fp8 = 0; fp8 < 2; fp8++) {
    const int NEXP = fp8 ? 24 : 40;  // 197 / 180 MB of records: larger than the caches
    std::vector<Exp> ex(NEXP);
    for (auto& e : ex) make(e, fp8, rng);
    const double rec_mb = (double)(ex[0].gc.size() + ex[0].gs.size() + ex[0].dc.size() + ex[0].ds.size()) / 1e6;
    for (int U : {1, 4, 8, 16, 32, 64, 128, 256, 512}) {
      std::normal_distribution<float> nd(0.f, 1.f);
      std::vector<XBuf> qg(U), qd(U);
      std::vector<float> xv(H);
      XQB xg, xd;
      xg.reserve(H, U);
      xd.reserve(I, U);
      for (int u = 0; u < U; u++) {
        qg[u].init(H, Prec::I16);
        qd[u].init(I, Prec::I16);
        for (auto& v : xv) v = nd(rng);
        qg[u].prep(xv.data(), H);
        xqb_set_from(xg, u, qg[u].q);
        for (int k = 0; k < I; k++) xv[k] = nd(rng) * 0.1f;
        qd[u].prep(xv.data(), I);
        xqb_set_from(xd, u, qd[u].q);
      }
      xqb_pad(xg);
      xqb_pad(xd);
      // the one-vector reference of expert 0
      std::vector<float> rg((size_t)U * 2 * I), rd((size_t)U * H);
      for (int u = 0; u < U; u++) {
        if (fp8) {
          mv_mxfp8(ex[0].gc.data(), ex[0].gs.data(), H, 0, 2 * I, qg[u].q, rg.data() + (size_t)u * 2 * I);
          mv_mxfp8(ex[0].dc.data(), ex[0].ds.data(), I, 0, H, qd[u].q, rd.data() + (size_t)u * H);
        } else {
          mv_nvfp4(ex[0].gc.data(), ex[0].gs.data(), 1.f, H, 0, 2 * I, qg[u].q, rg.data() + (size_t)u * 2 * I);
          mv_nvfp4(ex[0].dc.data(), ex[0].ds.data(), 1.f, I, 0, H, qd[u].q, rd.data() + (size_t)u * H);
        }
      }
      for (int packed = 0; packed < (fp8 ? 1 : 2); packed++) {
        std::vector<std::vector<float>> Y((size_t)NEXP * 2);
        for (int e = 0; e < NEXP; e++) {
          Y[(size_t)e * 2].assign((size_t)U * 2 * I, 0.f);
          Y[(size_t)e * 2 + 1].assign((size_t)U * H, 0.f);
        }
        std::vector<double> t;
        for (int r = 0; r < reps; r++) {
          Pass P;
          P.ex = &ex;
          P.fp8 = fp8;
          P.packed = packed;
          P.U = U;
          P.xg = &xg;
          P.xd = &xd;
          P.qg = &qg;
          P.qd = &qd;
          P.Y = &Y;
          P.nblk_g = (2 * I + 63) / 64;
          P.nblk_d = (H + 63) / 64;
          auto t0 = sclk::now();
          std::vector<std::thread> th;
          for (int k = 0; k < nth; k++) th.emplace_back(worker, std::ref(P));
          for (auto& x : th) x.join();
          t.push_back(std::chrono::duration<double, std::milli>(sclk::now() - t0).count());
        }
        const bool same = std::memcmp(Y[0].data(), rg.data(), rg.size() * 4) == 0 && std::memcmp(Y[1].data(), rd.data(), rd.size() * 4) == 0;
        std::sort(t.begin(), t.end());
        const double ms = t[t.size() / 2];
        const double macs = (double)NEXP * U * 3.0 * I * H;
        const double gmacs = macs / (ms / 1e3) / 1e9;
        const double ms_tok = (fp8 ? 4.0 : 44.0) * 8 * 3.0 * I * H / (gmacs * 1e9) * 1e3;  // this format's applications per prompt token
        printf("%s{\"fmt\":\"%s\",\"kernel\":\"%s\",\"users\":%d,\"experts\":%d,\"record_mb\":%.2f,\"ms\":%.2f,\"gmacs_s\":%.1f,"
               "\"record_gb_s\":%.1f,\"ms_per_prompt_token\":%.2f,\"bits_equal_one_vector\":%s}\n",
               first ? "" : ",", fp8 ? "mxfp8" : "nvfp4", packed ? "gemm_nvfp4_i16" : (fp8 ? "mv_mxfp8_multi" : "mv_nvfp4_multi"), U, NEXP,
               rec_mb, ms, gmacs, NEXP * rec_mb / 1e3 / (ms / 1e3), ms_tok, same ? "true" : "false");
        first = false;
        fflush(stdout);
      }
    }
  }
  printf("]}\n");
  return 0;
}
