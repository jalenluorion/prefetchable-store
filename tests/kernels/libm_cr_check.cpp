// is libm tanhf / expf the float rounding of the double function on every input? and per-call costs
#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
  long dt = 0, de = 0;
  float tlo = INFINITY, thi = -INFINITY, elo = INFINITY, ehi = -INFINITY;
  int shown = 0;
  for (uint64_t u = 0; u < (1ull << 32); u++) {
    uint32_t w = (uint32_t)u;
    float x;
    std::memcpy(&x, &w, 4);
    if (std::isnan(x)) continue;
    float a = tanhf(x), b = (float)std::tanh((double)x);
    if (std::memcmp(&a, &b, 4)) { dt++; tlo = std::fmin(tlo, x); thi = std::fmax(thi, x); if (shown < 6) shown++, printf("tanh %a: libm %a double %a\n", x, a, b); }
    a = expf(x), b = (float)std::exp((double)x);
    if (std::memcmp(&a, &b, 4)) { de++; elo = std::fmin(elo, x); ehi = std::fmax(ehi, x); if (shown < 12) shown++, printf("exp %a: libm %a double %a\n", x, a, b); }
  }
  printf("{\"tanhf_ne_double_rounded\":%ld,\"range\":[%g,%g],\"expf_ne_double_rounded\":%ld,\"range_e\":[%g,%g]}\n", dt, tlo, thi, de, elo, ehi);
  // costs on a P core: activations like the engine's (|x| up to ~30)
  const int N = 1 << 16;
  std::vector<float> in(N), out(N);
  for (int i = 0; i < N; i++) in[i] = (i - N / 2) * (30.f / N);
  auto t = [&](auto f) { auto t0 = std::chrono::steady_clock::now(); for (int r = 0; r < 200; r++) f(); return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (200.0 * N); };
  volatile float sink = 0;
  double c1 = t([&] { for (int i = 0; i < N; i++) out[i] = tanhf(in[i]); sink = out[7]; });
  double c2 = t([&] { for (int i = 0; i < N; i++) out[i] = expf(in[i]); sink = out[7]; });
  double c3 = t([&] { int n = N; vvtanhf(out.data(), in.data(), &n); sink = out[7]; });
  double c4 = t([&] { for (int i = 0; i < N; i++) out[i] = (float)std::tanh((double)in[i]); sink = out[7]; });
  printf("{\"ns_tanhf\":%.2f,\"ns_expf\":%.2f,\"ns_vvtanhf\":%.2f,\"ns_double_tanh\":%.2f}\n", c1, c2, c3, c4);
}
