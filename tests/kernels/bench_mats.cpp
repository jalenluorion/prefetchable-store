// The decode fork-joins of the in-memory 8-bit matrices, Flash-Next's shapes beside the 35B's, in one process: per
// fork-join (one run_rows_q call with the input's quantisation, as the models call it) ms and GB/s over many copies of
// the matrices (a decode token streams ~4 GB between two uses of a matrix, so every call reads from memory, not cache).
// Random codes, scales and biases: the kernels' time does not depend on the values.
//
// usage: bench_mats [threads (8)] [copies' bytes per case, MB (400)] [rounds (20)]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "dmeng/pool.h"
#include "dmeng/weights.h"

using namespace dm;

struct Mat {
  int rows, cols;
};
struct Case {
  const char* model;
  const char* name;
  std::vector<Mat> mats;  // one fork-join over these (one input)
  int chunk;
  int per_token;          // calls per decode token
};

int main(int argc, char** argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : 8;
  const double mb_case = argc > 2 ? atof(argv[2]) : 400;
  const int rounds = argc > 3 ? atoi(argv[3]) : 20;
  // Flash-Next: H 2560, 4 streams (10,240), low rank 320; DeltaNet 16 key / 48 value heads of 128; QSA 24 q heads of 256
  // (gated: 2 x), 2 KV heads, indexer 4 + 1 heads of 128. 35B: H 2048; DeltaNet 16 / 32 heads of 128; 16 q heads of 256.
  const std::vector<Case> cases = {
      {"flash", "DeltaNet in-projection (qkv, z, b, a)", {{10240, 2560}, {6144, 2560}, {48, 2560}, {48, 2560}}, 32, 36},
      {"flash", "DeltaNet out-projection", {{2560, 6144}}, 32, 36},
      {"flash", "QSA in-projection (q, k, v, indexer)", {{12288, 2560}, {512, 2560}, {512, 2560}, {640, 2560}}, 32, 12},
      {"flash", "QSA out-projection", {{2560, 6144}}, 32, 12},
      {"flash", "hyper-connection down + inject", {{320, 10240}, {4, 10240}}, 16, 96},
      {"flash", "hyper-connection up", {{10240, 320}}, 32, 96},
      {"q36", "DeltaNet in-projection (qkv, z, b, a)", {{8192, 2048}, {4096, 2048}, {32, 2048}, {32, 2048}}, 32, 30},
      {"q36", "DeltaNet out-projection", {{2048, 4096}}, 32, 30},
      {"q36", "full attention in-projection (q, k, v)", {{8192, 2048}, {512, 2048}, {512, 2048}}, 32, 10},
      {"q36", "full attention out-projection", {{2048, 4096}}, 32, 10},
  };
  Pool pool(threads);
  std::mt19937 rng(1);
  printf("{\"threads\":%d,\"cases\":[\n", threads);
  for (size_t ci = 0; ci < cases.size(); ci++) {
    const Case& C = cases[ci];
    size_t bytes1 = 0;
    for (auto& m : C.mats) bytes1 += (size_t)m.rows * m.cols * (1 + 4.0 / 64);
    const int copies = std::max(1, (int)(mb_case * 1e6 / bytes1));
    const int K = C.mats[0].cols;
    std::vector<std::vector<uint8_t>> mem;
    std::vector<std::vector<Weight>> W(copies);
    for (int c = 0; c < copies; c++)
      for (auto& m : C.mats) {
        const size_t n = (size_t)m.rows * m.cols, g = n / 64;
        mem.emplace_back(n + 4 * g);
        auto& b = mem.back();
        for (size_t i = 0; i < n; i++) b[i] = (uint8_t)rng();
        uint16_t* sb = (uint16_t*)(b.data() + n);
        for (size_t i = 0; i < 2 * g; i++) sb[i] = 0x2000 + (rng() & 0x3ff);  // small positive fp16
        Weight w;
        w.kind = Weight::AFF8;
        w.rows = m.rows;
        w.cols = m.cols;
        w.c = b.data();
        w.s = sb;
        w.b = sb + g;
        W[c].push_back(w);
      }
    std::vector<float> x(K), y(16384 * 4);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    XBuf xb;
    xb.init(K, Prec::I16);
    std::vector<double> t;
    double bytes = 0;
    for (int r = 0; r < rounds + 1; r++)
      for (int c = 0; c < copies; c++) {
        std::vector<Job> jb;
        size_t o = 0;
        for (auto& w : W[c]) {
          jb.push_back({&w, &xb.q, x.data(), y.data() + o});
          o += w.rows;
        }
        XBuf* qs[1] = {&xb};
        const float* xs[1] = {x.data()};
        const int ks[1] = {K};
        double b0 = 0;
        auto t0 = std::chrono::steady_clock::now();
        run_rows_q(pool, jb.data(), (int)jb.size(), C.chunk, false, &b0, qs, xs, ks, 1);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (r > 0) t.push_back(ms), bytes = b0;  // round 0 warms the pages
      }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2], p10 = t[t.size() / 10];
    printf("{\"model\":\"%s\",\"stage\":\"%s\",\"mb\":%.3f,\"ms_median\":%.4f,\"ms_p10\":%.4f,\"gb_s_median\":%.1f,\"gb_s_p10\":%.1f,\"calls_per_token\":%d,"
           "\"ms_per_token\":%.2f,\"copies\":%d}%s\n",
           C.model, C.name, bytes / 1e6, med, p10, bytes / 1e6 / med, bytes / 1e6 / p10, C.per_token, med * C.per_token, copies,
           ci + 1 < cases.size() ? "," : "");
    fflush(stdout);
  }
  printf("]}\n");
  return 0;
}
