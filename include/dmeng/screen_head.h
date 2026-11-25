// The screened output head: the exact top-k rows (and their exact logits) of an 8-bit head without computing every row.
// The head's codes are split in place into nibble planes (Weight::AFF8S; no copy of the weights). A pass over the high
// planes (half the code bytes, plus the scales and biases) gives every row an interval that holds its exact logit (the
// low nibbles' reach, bounded per group, plus a float slack far above either kernel's rounding); the rows that can still
// be in the top k are computed with the full head's own row kernel. The argmax (k = 1) and the top k's rows, order and
// logits (a top-k sampler's candidates and probabilities) are the full head's, bit for bit. Extra memory: one byte per
// row and 64-input group (the bound) and a float per row.
#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "dmeng/weights.h"

namespace dm {

struct ScreenHead {
  static constexpr int KMAX = 64;
  Weight* head = nullptr;  // the model's head, AFF8S after build()
  int V = 0, VO = 0;       // rows; rows that count (the unpadded vocabulary)
  std::vector<float> hi;     // the last run's upper end per row
  std::vector<uint8_t> e1;  // [V][groups] 4 ||lo - 7.5||_2 of the low nibbles, rounded up
  double n_cand = 0, n_calls = 0, ms_coarse = 0, ms_exact = 0, n_fallback = 0;
  bool ok = true;  // the last run's answer holds (always, without a cap)
  // split the head in place (it must be AFF8 with I16 activations; its memory writable) and check the split kernels
  // against the original bytes on a test input (bit for bit; false if not)
  bool build(Pool& pool, Weight& head, int vrows, int vout, std::string& err);
  // logits of head . x into lg[0, VO): the exact value for every row that can be among the top k (k <= KMAX) of those
  // rows, -inf for the rest. Returns the argmax row (lowest index on ties). cap > 0: the values are cap tanh(v / cap)
  // (a softcap: monotone, but it can make distinct logits equal), and ok says whether every row left out is still below
  // the k-th kept one after the cap; when it is false the caller computes the full head.
  int run(Pool& pool, const XQ& x, int k, float* lg, float cap = 0.f);
  // run() for n inputs at once (a multi-position verify pass): the coarse level reads the head's planes once for all of
  // them; each input's lg, argmax (best[i]) and ok (oks[i], with a cap) equal run()'s on that input alone, bit for bit
  void run_n(Pool& pool, const XQ* const* xs, int n, int k, float* const* lg, int* best, bool* oks, float cap = 0.f);
  double bytes = 0;  // head bytes the runs read (statistics: the coarse planes, scales, biases and bounds, the exact rows)
  double coarse_bytes() const;
  double row_bytes() const;
  void reset_stats() { n_cand = n_calls = ms_coarse = ms_exact = n_fallback = bytes = 0; }

 private:
  struct Scratch;  // run_n's per-input lists and bounds (screen_head.cpp)
  std::shared_ptr<Scratch> sc_;
};

}  // namespace dm
