// Token sampling: greedy (lowest index on ties), or temperature, then top-k, then top-p on the tempered distribution
// (the releases' order).
#pragma once
#include <cstdint>
#include <random>

namespace dm {

int argmax(const float* lg, int V);
int sample(const float* lg, int V, double temp, int top_k, double top_p, std::mt19937_64& rng);

}  // namespace dm
