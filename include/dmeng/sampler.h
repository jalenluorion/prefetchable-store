// Token sampling: greedy (lowest index on ties), or temperature, then top-k, then top-p on the tempered distribution
// (the releases' order).
#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace dm {

int argmax(const float* lg, int V);
int sample(const float* lg, int V, double temp, int top_k, double top_p, std::mt19937_64& rng);

// The draws of a sampled generation by token index. sample() takes exactly one 64-bit output of the engine per sampled
// token (uniform_real_distribution<double> on mt19937_64), so the k-th sampled token of a generation always uses the k-th
// output of mt19937_64(seed). Draws keeps them by index, so a token's draw does not depend on how many positions the pass
// that produced it carried: draft and verify under sampling then gives the tokens sampled decoding gives for the seed.
struct Draws {
  std::mt19937_64 gen;
  std::vector<uint64_t> v;
  explicit Draws(uint64_t seed = 0) : gen(seed) {}
  void reset(uint64_t seed) {
    gen.seed(seed);
    v.clear();
  }
  uint64_t at(size_t k) {
    while (v.size() <= k) v.push_back(gen());
    return v[k];
  }
};
// sample() with the draws from index k on (the k-th sampled token of the generation)
int sample_at(const float* lg, int V, double temp, int top_k, double top_p, Draws& d, size_t k);

}  // namespace dm
