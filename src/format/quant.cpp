// Compiled with -ffp-contract=off and without fast-math: every operation is one IEEE float32 operation, as in the kernel.
#include "quant.h"

#include <cmath>
#include <cstring>

namespace ms {

uint16_t f32_to_f16_bits(float x) {
  _Float16 h = (_Float16)x;  // round to nearest even (fcvt)
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
float f16_bits_to_f32(uint16_t b) {
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return (float)h;
}
uint16_t f32_to_bf16_bits(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}
float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

static inline float load(const void* w, Src src, size_t i) {
  switch (src) {
    case Src::F16: return f16_bits_to_f32(((const uint16_t*)w)[i]);
    case Src::BF16: return bf16_bits_to_f32(((const uint16_t*)w)[i]);
    default: return ((const float*)w)[i];
  }
}

void quantize_rows(const void* w, Src src, int R, int C, int bits, Rule rule, uint8_t* codes, uint16_t* scales,
                   uint16_t* biases, int r0, int r1, QStats* st) {
  (void)R;
  const int G = 64, NG = C / G;
  const float n_bins = (float)((1 << bits) - 1);
  const float eps = 1e-7f;
  const float tmax = rule == Rule::F16 ? 65504.0f : bf16_bits_to_f32(0x7f7f);  // Limits<T>::max
  float v[64];
  for (int r = r0; r < r1; r++) {
    for (int g = 0; g < NG; g++) {
      size_t base = (size_t)r * C + (size_t)g * G;
      float w_min = tmax, w_max = 0.0f;
      for (int k = 0; k < G; k++) {
        float x = load(w, src, base + k);
        if (rule == Rule::F16 && src != Src::F16) x = f16_bits_to_f32(f32_to_f16_bits(x));  // the astype(float16)
        v[k] = x;
        w_min = std::fmin(w_min, x);
        w_max = std::fmax(w_max, x);
      }
      float scale = std::fmax((w_max - w_min) / n_bins, eps);
      bool side = std::fabs(w_min) > std::fabs(w_max);
      scale = side ? scale : -scale;
      float edge = side ? w_min : w_max;
      float q0 = std::round(edge / scale);
      bool at_zero = q0 == 0.0f;
      scale = at_zero ? scale : edge / q0;
      float bias = at_zero ? 0.0f : edge;
      uint16_t sb, bb;
      if (rule == Rule::F16) {
        sb = f32_to_f16_bits(scale);
        bb = f32_to_f16_bits(bias);
      } else {
        float s16 = bf16_bits_to_f32(f32_to_bf16_bits(scale)), b16 = bf16_bits_to_f32(f32_to_bf16_bits(bias));
        sb = f32_to_f16_bits(s16);
        bb = f32_to_f16_bits(b16);
        if (st) st->f16_inexact += (f16_bits_to_f32(sb) != s16) + (f16_bits_to_f32(bb) != b16);
      }
      scales[(size_t)r * NG + g] = sb;
      biases[(size_t)r * NG + g] = bb;
      if (bits == 8) {
        uint8_t* c = codes + base;
        for (int k = 0; k < G; k++) {
          float q = std::fmin(std::round((v[k] - bias) / scale), n_bins);
          c[k] = q > 0.0f ? (uint8_t)q : 0;
        }
      } else {  // 4-bit: two per byte, element 2j in the low nibble
        uint8_t* c = codes + base / 2;
        for (int k = 0; k < G; k += 2) {
          float q0f = std::fmin(std::round((v[k] - bias) / scale), n_bins);
          float q1f = std::fmin(std::round((v[k + 1] - bias) / scale), n_bins);
          uint8_t a = q0f > 0.0f ? (uint8_t)q0f : 0, b = q1f > 0.0f ? (uint8_t)q1f : 0;
          c[k / 2] = (uint8_t)(a | (b << 4));
        }
      }
    }
  }
}

}  // namespace ms
