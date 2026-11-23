// One split-K job of the decode attention (a chunk of CHK keys for the query heads of one KV head), arithmetic identical
// to the original phase_split body (qwen3_5_moe.cpp) element by element:
//   - each (key, head) score: two FMLAL accumulators over the head dimensions in order, their sum, times the scale;
//   - per head: the max over the chunk, exp(score - max) and their double sum in key order;
//   - each output element: the fused multiply-add sequence over the keys in order, starting from 0.
// What changes is only the schedule: four heads' score dots run side by side (independent chains overlap, the key row is
// loaded once for them), and the value sums for 8 dimensions x 8 heads stay in registers over the whole chunk (the
// weights transposed to [key][head]) instead of being loaded and stored for every key.
#pragma once
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace dm {

constexpr int SPLIT_CHK = 256;

// q: [NQ][HD] fp32 normed + rotated queries; K, V: the KV head's cache rows [ctx][HD] (fp16), so row t is at (kvh * ctx + t);
// part: [nch][NQ][HD + 2] (max, sum, output) as phase_split writes it.
static inline void split_chunk(const float* q, const _Float16* Kall, const _Float16* Vall, int ctx, int T, int ch, int kvh, int NQ, int NKV,
                               int HD, float* part) {
  const int rep = NQ / NKV, W = HD + 2, CHK = SPLIT_CHK;
  const float scale = 1.f / std::sqrt((float)HD);
  static thread_local std::vector<_Float16> q16v;
  static thread_local std::vector<float> scv, wtv;
  q16v.resize((size_t)rep * HD);
  scv.resize((size_t)rep * CHK);
  wtv.resize((size_t)CHK * 8);
  _Float16* q16 = q16v.data();
  float* sc = scv.data();
  for (int g = 0; g < rep; g++)
    for (int i = 0; i < HD; i++) q16[(size_t)g * HD + i] = (_Float16)q[(size_t)(kvh * rep + g) * HD + i];
  const int t0 = ch * CHK, t1 = std::min(T, t0 + CHK), n = t1 - t0;
  const _Float16* K = Kall + ((size_t)kvh * ctx + t0) * HD;
  const _Float16* V = Vall + ((size_t)kvh * ctx + t0) * HD;
  for (int t = 0; t < n; t++) {
    const float16_t* kr = (const float16_t*)(K + (size_t)t * HD);
    int g = 0;
    for (; g + 4 <= rep; g += 4) {
      const float16_t* q0 = (const float16_t*)q16 + (size_t)g * HD;
      float32x4_t a0 = vdupq_n_f32(0), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
      for (int u = 0; u < HD; u += 8) {
        float16x8_t kv = vld1q_f16(kr + u);
        float16x8_t x0 = vld1q_f16(q0 + u), x1 = vld1q_f16(q0 + HD + u), x2 = vld1q_f16(q0 + 2 * HD + u), x3 = vld1q_f16(q0 + 3 * HD + u);
        a0 = vfmlalq_low_f16(a0, kv, x0); b0 = vfmlalq_high_f16(b0, kv, x0);
        a1 = vfmlalq_low_f16(a1, kv, x1); b1 = vfmlalq_high_f16(b1, kv, x1);
        a2 = vfmlalq_low_f16(a2, kv, x2); b2 = vfmlalq_high_f16(b2, kv, x2);
        a3 = vfmlalq_low_f16(a3, kv, x3); b3 = vfmlalq_high_f16(b3, kv, x3);
      }
      sc[(size_t)(g + 0) * CHK + t] = vaddvq_f32(vaddq_f32(a0, b0)) * scale;
      sc[(size_t)(g + 1) * CHK + t] = vaddvq_f32(vaddq_f32(a1, b1)) * scale;
      sc[(size_t)(g + 2) * CHK + t] = vaddvq_f32(vaddq_f32(a2, b2)) * scale;
      sc[(size_t)(g + 3) * CHK + t] = vaddvq_f32(vaddq_f32(a3, b3)) * scale;
    }
    for (; g < rep; g++) {
      const float16_t* qq = (const float16_t*)q16 + (size_t)g * HD;
      float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
      for (int u = 0; u < HD; u += 8) {
        float16x8_t kv = vld1q_f16(kr + u), qv = vld1q_f16(qq + u);
        s0 = vfmlalq_low_f16(s0, kv, qv);
        s1 = vfmlalq_high_f16(s1, kv, qv);
      }
      sc[(size_t)g * CHK + t] = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
    }
  }
  for (int g = 0; g < rep; g++) {
    float* sg = sc + (size_t)g * CHK;
    float mx = -INFINITY;
    for (int t = 0; t < n; t++) mx = std::max(mx, sg[t]);
    double z = 0;
    for (int t = 0; t < n; t++) z += (sg[t] = std::exp(sg[t] - mx));
    float* out = part + ((size_t)ch * NQ + kvh * rep + g) * W;
    out[0] = mx;
    out[1] = (float)z;
  }
  if (rep == 8) {
    float* wt = wtv.data();
    for (int t = 0; t < n; t++)
      for (int g = 0; g < 8; g++) wt[t * 8 + g] = sc[(size_t)g * CHK + t];
    for (int u = 0; u < HD; u += 8) {
      float32x4_t a0[8], a1[8];
      for (int g = 0; g < 8; g++) a0[g] = a1[g] = vdupq_n_f32(0);
      for (int t = 0; t < n; t++) {
        float16x8_t vv = vld1q_f16((const float16_t*)(V + (size_t)t * HD) + u);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(vv)), hi = vcvt_high_f32_f16(vv);
        float32x4_t w0 = vld1q_f32(wt + t * 8), w1 = vld1q_f32(wt + t * 8 + 4);
        a0[0] = vfmaq_laneq_f32(a0[0], lo, w0, 0); a1[0] = vfmaq_laneq_f32(a1[0], hi, w0, 0);
        a0[1] = vfmaq_laneq_f32(a0[1], lo, w0, 1); a1[1] = vfmaq_laneq_f32(a1[1], hi, w0, 1);
        a0[2] = vfmaq_laneq_f32(a0[2], lo, w0, 2); a1[2] = vfmaq_laneq_f32(a1[2], hi, w0, 2);
        a0[3] = vfmaq_laneq_f32(a0[3], lo, w0, 3); a1[3] = vfmaq_laneq_f32(a1[3], hi, w0, 3);
        a0[4] = vfmaq_laneq_f32(a0[4], lo, w1, 0); a1[4] = vfmaq_laneq_f32(a1[4], hi, w1, 0);
        a0[5] = vfmaq_laneq_f32(a0[5], lo, w1, 1); a1[5] = vfmaq_laneq_f32(a1[5], hi, w1, 1);
        a0[6] = vfmaq_laneq_f32(a0[6], lo, w1, 2); a1[6] = vfmaq_laneq_f32(a1[6], hi, w1, 2);
        a0[7] = vfmaq_laneq_f32(a0[7], lo, w1, 3); a1[7] = vfmaq_laneq_f32(a1[7], hi, w1, 3);
      }
      for (int g = 0; g < 8; g++) {
        float* out = part + ((size_t)ch * NQ + kvh * rep + g) * W + 2;
        vst1q_f32(out + u, a0[g]);
        vst1q_f32(out + u + 4, a1[g]);
      }
    }
  } else {
    for (int g = 0; g < rep; g++) std::fill(part + ((size_t)ch * NQ + kvh * rep + g) * W + 2, part + ((size_t)ch * NQ + kvh * rep + g) * W + 2 + HD, 0.f);
    for (int t = 0; t < n; t++) {
      const float16_t* vr = (const float16_t*)(V + (size_t)t * HD);
      for (int u = 0; u < HD; u += 8) {
        float16x8_t vv = vld1q_f16(vr + u);
        float32x4_t lo = vcvt_f32_f16(vget_low_f16(vv)), hi = vcvt_high_f32_f16(vv);
        for (int g = 0; g < rep; g++) {
          float* out = part + ((size_t)ch * NQ + kvh * rep + g) * W + 2;
          float w = sc[(size_t)g * CHK + t];
          vst1q_f32(out + u, vfmaq_n_f32(vld1q_f32(out + u), lo, w));
          vst1q_f32(out + u + 4, vfmaq_n_f32(vld1q_f32(out + u + 4), hi, w));
        }
      }
    }
  }
}

}  // namespace dm
