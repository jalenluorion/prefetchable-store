#include "dmeng/sampler.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace dm {

int argmax(const float* lg, int V) {
  int b = 0;
  for (int i = 1; i < V; i++)
    if (lg[i] > lg[b]) b = i;
  return b;
}

int sample(const float* lg, int V, double temp, int top_k, double top_p, std::mt19937_64& rng) {
  if (temp <= 0) return argmax(lg, V);
  std::vector<int> idx(V);
  for (int i = 0; i < V; i++) idx[i] = i;
  int kk = top_k > 0 ? std::min(top_k, V) : V;
  if (kk < V || top_p < 1.0) {
    int part = kk < V ? kk : std::min(V, 4096);
    std::partial_sort(idx.begin(), idx.begin() + part, idx.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    if (kk == V) kk = part;  // top-p only: the mass beyond the first 4096 is negligible at any sane temperature
  }
  double mx = -1e30;
  for (int j = 0; j < kk; j++) mx = std::max(mx, (double)lg[idx[j]] / temp);
  std::vector<double> p(kk);
  double z = 0;
  for (int j = 0; j < kk; j++) z += (p[j] = std::exp(lg[idx[j]] / temp - mx));
  int keep = kk;
  if (top_p < 1.0) {
    double cum = 0;
    for (int j = 0; j < kk; j++) {
      if (cum >= top_p * z) {
        keep = j;
        break;
      }
      cum += p[j];
    }
  }
  double zz = 0;
  for (int j = 0; j < keep; j++) zz += p[j];
  double u = std::uniform_real_distribution<double>(0, zz)(rng), acc = 0;
  for (int j = 0; j < keep; j++) {
    acc += p[j];
    if (u < acc) return idx[j];
  }
  return idx[keep - 1];
}

}  // namespace dm
