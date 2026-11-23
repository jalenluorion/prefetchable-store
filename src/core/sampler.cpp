#include "dmeng/sampler.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace dm {

// the first index of the largest logit (as the scalar loop "if (lg[i] > lg[b]) b = i"): the maximum with NEON, then
// the first index holding it (logits are never NaN; +0 and -0 compare equal in both forms)
int argmax(const float* lg, int V) {
  if (V < 16) {
    int b = 0;
    for (int i = 1; i < V; i++)
      if (lg[i] > lg[b]) b = i;
    return b;
  }
  float32x4_t m0 = vld1q_f32(lg), m1 = m0, m2 = m0, m3 = m0;
  int i = 0;
  for (; i + 16 <= V; i += 16) {
    m0 = vmaxq_f32(m0, vld1q_f32(lg + i));
    m1 = vmaxq_f32(m1, vld1q_f32(lg + i + 4));
    m2 = vmaxq_f32(m2, vld1q_f32(lg + i + 8));
    m3 = vmaxq_f32(m3, vld1q_f32(lg + i + 12));
  }
  float m = vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)));
  for (; i < V; i++) m = lg[i] > m ? lg[i] : m;
  const float32x4_t mv = vdupq_n_f32(m);
  for (i = 0; i + 16 <= V; i += 16) {
    const uint32x4_t e = vorrq_u32(vorrq_u32(vceqq_f32(vld1q_f32(lg + i), mv), vceqq_f32(vld1q_f32(lg + i + 4), mv)),
                                   vorrq_u32(vceqq_f32(vld1q_f32(lg + i + 8), mv), vceqq_f32(vld1q_f32(lg + i + 12), mv)));
    if (vmaxvq_u32(e)) break;
  }
  for (; i < V; i++)
    if (lg[i] == m) return i;
  return 0;
}

template <class G>
static int sample_t(const float* lg, int V, double temp, int top_k, double top_p, G& rng) {
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

int sample(const float* lg, int V, double temp, int top_k, double top_p, std::mt19937_64& rng) { return sample_t(lg, V, temp, top_k, top_p, rng); }

namespace {
struct DrawAt {  // a URNG that replays Draws from index k (the engine's range, so the distribution consumes it alike)
  using result_type = std::mt19937_64::result_type;
  Draws* d;
  size_t k;
  static constexpr result_type min() { return std::mt19937_64::min(); }
  static constexpr result_type max() { return std::mt19937_64::max(); }
  result_type operator()() { return d->at(k++); }
};
}  // namespace

int sample_at(const float* lg, int V, double temp, int top_k, double top_p, Draws& d, size_t k) {
  DrawAt g{&d, k};
  return sample_t(lg, V, temp, top_k, top_p, g);
}

}  // namespace dm
