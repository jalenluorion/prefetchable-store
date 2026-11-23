// The exact greedy head: argmax of an 8-bit output head's logits from a 4-bit copy's coarse logits plus a certified
// bound, the remaining rows computed exactly (greedy_head.cpp). Shared by the model families (Want::GREEDY).
#pragma once
#include <string>
#include <vector>

#include "dmeng/weights.h"

namespace dm {

struct GreedyHead {
  const Weight* head8 = nullptr;
  int V = 0, VO = 0;  // head rows; rows that count for the argmax (the unpadded vocabulary)
  Weight head4;
  std::vector<uint8_t> h4c;
  std::vector<_Float16> h4s, h4b, h_gnorm;
  std::vector<float> h_dnorm, h_bmax, coarse;
  std::vector<int> cand;
  const XQ* xq = nullptr;
  float* out = nullptr;
  double hd_cand = 0, hd_calls = 0, t_head_coarse = 0, t_head_exact = 0;
  float lo = 0;  // the last run's best lower bound: every row not in `cand` has a logit below it
  bool ok() const { return head8 != nullptr; }
  // derive the 4-bit copy and the bounds (once, at load)
  bool build(Pool& pool, const Weight& head, int vrows, int vout, Prec prec, std::string& err);
  // the argmax row over [0, VO) of head . xin; lg receives the exact logits of the candidate rows (others untouched)
  int run(Pool& pool, const XQ& xin, float* lg);
  void reset_stats() { hd_cand = hd_calls = t_head_coarse = t_head_exact = 0; }
};

}  // namespace dm
