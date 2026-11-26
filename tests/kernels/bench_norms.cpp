// The decode's norms on one thread: rmsnorm one row at a time against rmsnorm_n (the rows' sums side by side), for the
// row groups the models norm together (Inkling's three attention inputs and the gates' pairs, Flash-Next's streams, the
// 35B's verify positions). First every group's output is checked bit for bit against single rmsnorm calls (random
// rows, weights and none), then each form is timed in the cache (the rows are a decode step's working set).
//
// usage: bench_norms [hidden (2304)] [rounds (20000)]      exit 1 if any output differs
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "dmeng/weights.h"

using namespace dm;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const int H = argc > 1 ? atoi(argv[1]) : 2304;
  const int R = argc > 2 ? atoi(argv[2]) : 20000;
  std::mt19937 g(7);
  std::normal_distribution<float> nd(0.f, 3.f);
  const int NMAX = 8;
  std::vector<float> x((size_t)NMAX * H), w((size_t)NMAX * H), a((size_t)NMAX * H), b((size_t)NMAX * H);
  for (auto& v : x) v = nd(g);
  for (auto& v : w) v = 1.f + 0.1f * nd(g);
  const float* xs[NMAX];
  const float* ws[NMAX];
  float* as[NMAX];
  float* bs[NMAX];
  for (int s = 0; s < NMAX; s++) xs[s] = x.data() + (size_t)s * H, ws[s] = w.data() + (size_t)s * H, as[s] = a.data() + (size_t)s * H, bs[s] = b.data() + (size_t)s * H;
  int bad = 0;
  for (int n = 1; n <= NMAX; n++)
    for (int weighted = 0; weighted < 2; weighted++)
      for (int len : {H, H - 1, 17, 1}) {
        for (int s = 0; s < n; s++) rmsnorm(xs[s], weighted ? ws[s] : nullptr, len, 1e-6f, as[s]);
        rmsnorm_n(xs, weighted ? ws : nullptr, n, len, 1e-6f, bs);
        for (int s = 0; s < n; s++)
          if (memcmp(as[s], bs[s], sizeof(float) * len)) bad++, printf("DIFFER n %d weighted %d len %d row %d\n", n, weighted, len, s);
      }
  printf("bits: %s (rows 1-%d, with and without weights, lengths %d, %d, 17, 1)\n", bad ? "DIFFER" : "identical", NMAX, H, H - 1);
  printf("%-6s %12s %12s %8s\n", "rows", "single us", "side by side", "ratio");
  volatile float sink = 0;
  for (int n : {2, 3, 4, 8}) {
    double t1 = 0, t2 = 0;
    for (int rep = 0; rep < 3; rep++) {  // alternating, the best of three
      auto t0 = clk::now();
      for (int r = 0; r < R; r++)
        for (int s = 0; s < n; s++) rmsnorm(xs[s], ws[s], H, 1e-6f, as[s]);
      sink = sink + a[0];
      auto tm = clk::now();
      for (int r = 0; r < R; r++) rmsnorm_n(xs, ws, n, H, 1e-6f, bs);
      sink = sink + b[0];
      auto te = clk::now();
      const double u1 = std::chrono::duration<double, std::micro>(tm - t0).count() / R, u2 = std::chrono::duration<double, std::micro>(te - tm).count() / R;
      t1 = rep ? std::min(t1, u1) : u1;
      t2 = rep ? std::min(t2, u2) : u2;
    }
    printf("%-6d %12.2f %12.2f %8.2f\n", n, t1, t2, t1 / t2);
  }
  return bad ? 1 : 0;
}
