// One query head of gated softmax attention over fp16 keys and values, in one pass: shared by the 35B (its decode and
// batched-prefill heads up to split_min positions) and Flash-Next (QSA over the selected positions). Scores: the fp16
// query against each key with two FMLAL accumulators over the dimensions in order, their sum, x 1/sqrt(HD); softmax
// with the max subtracted, exp in float, the sum in double; values: fused multiply-adds of the normalised weights in
// position order, four lanes at a time; then the output times sigmoid(gate). Positions come from pos(t), t < n
// (contiguous for the 35B, a selection for Flash-Next). The 35B's split-K form for long contexts (attn_split.h) sums in
// another order and stays its own: Flash-Next attends a bounded selection in one pass, and its bits must not move.
#pragma once
#include <arm_neon.h>

#include <cmath>
#include <vector>

#include "dmeng/weights.h"

namespace dm {
namespace gated_attn {

template <class Pos>
inline void head_f16(const _Float16* q16, const _Float16* K, const _Float16* V, int HD, int n, Pos pos, const float* gate, float* o,
                     std::vector<float>& sc) {
  if ((int)sc.size() < n) sc.resize(n + 1024);
  const float scale = 1.f / std::sqrt((float)HD);
  float mx = -INFINITY;
  for (int t = 0; t < n; t++) {
    const float16_t* kr = (const float16_t*)(K + (size_t)pos(t) * HD);
    float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
    for (int i = 0; i < HD; i += 8) {
      float16x8_t kv = vld1q_f16(kr + i), qv = vld1q_f16((const float16_t*)q16 + i);
      s0 = vfmlalq_low_f16(s0, kv, qv);
      s1 = vfmlalq_high_f16(s1, kv, qv);
    }
    sc[t] = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
    mx = std::max(mx, sc[t]);
  }
  double zs = 0;
  for (int t = 0; t < n; t++) {
    sc[t] = std::exp(sc[t] - mx);
    zs += sc[t];
  }
  const float iz = (float)(1.0 / zs);
  float32x4_t acc[256];
  for (int i = 0; i < HD / 4; i++) acc[i] = vdupq_n_f32(0);
  for (int t = 0; t < n; t++) {
    const _Float16* vr = V + (size_t)pos(t) * HD;
    const float32x4_t pv = vdupq_n_f32(sc[t] * iz);
    for (int i = 0; i < HD; i += 8) {
      float16x8_t vv = vld1q_f16((const float16_t*)vr + i);
      acc[i / 4] = vfmaq_f32(acc[i / 4], vcvt_f32_f16(vget_low_f16(vv)), pv);
      acc[i / 4 + 1] = vfmaq_f32(acc[i / 4 + 1], vcvt_high_f32_f16(vv), pv);
    }
  }
  for (int i = 0; i < HD; i += 4) vst1q_f32(o + i, acc[i / 4]);
  for (int i = 0; i < HD; i++) o[i] *= sigm(gate[i]);
}

}  // namespace gated_attn
}  // namespace dm
