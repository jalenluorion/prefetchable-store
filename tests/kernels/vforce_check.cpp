// every float input: Accelerate vForce vvtanhf / vvexpf against libm tanhf / expf, bit for bit (NaN: both NaN)
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
  const int N = 1 << 20;
  std::vector<float> in(N), a(N), b(N);
  long dt = 0, de = 0, dtn = 0, den = 0;
  float tlo = INFINITY, thi = -INFINITY, elo = INFINITY, ehi = -INFINITY;
  int shown = 0;
  for (uint64_t base = 0; base < (1ull << 32); base += N) {
    for (int i = 0; i < N; i++) {
      uint32_t u = (uint32_t)(base + i);
      std::memcpy(&in[i], &u, 4);
    }
    for (int f = 0; f < 2; f++) {
      int n = N;
      if (f == 0) vvtanhf(b.data(), in.data(), &n);
      else vvexpf(b.data(), in.data(), &n);
      for (int i = 0; i < N; i++) {
        a[i] = f == 0 ? tanhf(in[i]) : expf(in[i]);
        const bool an = std::isnan(a[i]), bn = std::isnan(b[i]);
        bool diff = (an || bn) ? an != bn : std::memcmp(&a[i], &b[i], 4) != 0;
        if (!diff) continue;
        const float x = in[i];
        if (f == 0) {
          dt++;
          if (std::fabs(x) < 40.f) dtn++;
          tlo = std::fmin(tlo, x), thi = std::fmax(thi, x);
        } else {
          de++;
          if (std::fabs(x) < 40.f) den++;
          elo = std::fmin(elo, x), ehi = std::fmax(ehi, x);
        }
        if (shown < 12) shown++, printf("%s(%a = %.9g): libm %a vforce %a\n", f ? "exp" : "tanh", x, x, a[i], b[i]);
      }
    }
  }
  printf("{\"tanhf_inputs_differ\":%ld,\"tanh_differ_abs_lt_40\":%ld,\"tanh_min\":%g,\"tanh_max\":%g,\"expf_inputs_differ\":%ld,\"exp_differ_abs_lt_40\":%ld,\"exp_min\":%g,\"exp_max\":%g}\n",
         dt, dtn, tlo, thi, de, den, elo, ehi);
}
