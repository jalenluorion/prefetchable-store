// The affine quantiser of MLX (mx.quantize / nn.quantize, group 64, 8 or 4 bits), reproduced bit for bit so the engine
// derives its 8-bit resident and router files in C++ (FORMAT.md section 6). The arithmetic is MLX's Metal kernel
// `affine_quantize` (mlx/backend/metal/kernels/quantized.h, MLX 0.32): per group of 64 along a row, in float32,
//   w_min = min(values, T max), w_max = max(values, 0)               (w_max starts at 0, w_min at the dtype's max)
//   scale = max((w_max - w_min) / n_bins, 1e-7); side = |w_min| > |w_max|; scale = side ? scale : -scale
//   edge = side ? w_min : w_max; q0 = round(edge / scale)
//   scale = q0 == 0 ? scale : edge / q0;  bias = q0 == 0 ? 0 : edge
//   code = min(round((w - bias) / scale), n_bins)                     (round = half away from zero)
//   stored scale / bias = T(scale), T(bias)                           (T = the input dtype; round to nearest even)
// Two input rules exist in our files:
//   F16  (Qwen families, routers): the weight is cast to f16 first (mx.quantize(w.astype(float16))); T = f16.
//   BF16 (Inkling resident): nn.quantize of the bf16 weight; T = bf16; the engine widens the bf16 scales / biases to f16.
#pragma once
#include <cstdint>

namespace ms {

enum class Src { F16, BF16, F32 };
enum class Rule { F16, BF16 };

struct QStats {
  uint64_t f16_inexact = 0;  // BF16 rule: scales / biases that do not survive the widening to f16 exactly
};

// Quantises rows [r0, r1) of W [R, C] (C % 64 == 0). codes: [R, C] bytes (8-bit) or [R, C / 2] (4-bit, element 2j in the
// low nibble of byte j); scales, biases: [R, C / 64] f16 bit patterns. Pointers address row 0 of each array.
void quantize_rows(const void* w, Src src, int R, int C, int bits, Rule rule, uint8_t* codes, uint16_t* scales,
                   uint16_t* biases, int r0, int r1, QStats* st = nullptr);

uint16_t f32_to_f16_bits(float x);
float f16_bits_to_f32(uint16_t h);
uint16_t f32_to_bf16_bits(float x);   // round to nearest even (NaN kept quiet)
float bf16_bits_to_f32(uint16_t b);

}  // namespace ms
