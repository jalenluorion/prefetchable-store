// Batch-1 matrix-vector kernels on packed weights, NEON, operating on the store's record bytes as they are.
//
// Formats (matrices [rows = out, cols = in], groups along cols):
//   AFF4   MLX affine 4-bit, group 64: codes 2 per byte (element 2j in the low nibble of byte j), fp16 scale + fp16
//          bias per group; w = code * scale + bias. (q36 routed experts)
//   AFF8   MLX affine 8-bit, group 64: one code byte per element, fp16 scale + bias. (q36 resident weights)
//   NVFP4  E2M1 codes 2 per byte (low nibble = even element), one E4M3 scale per 16, per-matrix float32 global
//          multiplier; w = e2m1 * e4m3 * gmul. (Inkling layers 0-19)
//   MXFP8  E4M3 codes, one UE8M0 scale per 32; w = e4m3 * 2^(e - 127). (Inkling layers 20-23)
//
// Precision of the activation side (the weights are never dequantised to memory):
//   F16   x rounded to fp16, products widened and accumulated in fp32 (FMLAL), as MLX's Metal kernels do
//   I16   x quantised per 64 inputs to 15 bits (two int8 planes, two SDOTs): finer than fp16 for the large entries
//   I8    x quantised per 64 inputs to int8 (one SDOT): llama.cpp's Q8 activations
// MXFP8 codes are not integers, so MXFP8 always uses the F16 path.
#pragma once
#include <cstddef>
#include <cstdint>

namespace dm {

enum class Prec : int { F16 = 0, I8 = 1, I16 = 2 };
enum class Fmt : int { AFF4 = 0, AFF8 = 1, NVFP4 = 2, MXFP8 = 3 };

constexpr int XB = 64;  // activation block (quantisation and bias-sum granularity)

// A prepared input vector (K a multiple of 64). Both code orders are kept: "perm" pairs with 4-bit codes (per 32
// inputs: the 16 even entries, then the 16 odd ones), "nat" with byte codes.
struct XQ {
  int K = 0;
  Prec prec = Prec::I16;
  float* gsum = nullptr;     // [K/64] sum of x per block (affine bias term), from the fp32 x
  float* d = nullptr;        // [K/64] block scale of the integer planes
  int8_t* hp = nullptr;      // [K] perm order, high (or only) int8 plane
  int8_t* lp = nullptr;      // [K] perm order, low plane (I16)
  int8_t* hn = nullptr;      // [K] natural order
  int8_t* ln = nullptr;
  _Float16* fp = nullptr;    // [K] perm order fp16 (F16)
  _Float16* fn = nullptr;    // [K] natural order fp16
};

// Bytes needed for an XQ of K inputs (one allocation, 64-byte aligned pieces).
size_t xq_bytes(int K);
void xq_bind(XQ& q, void* mem, int K, Prec p);
// Prepare block b (inputs [64 b, 64 b + 64)) from x64 (the 64 fp32 values of that block).
void xq_block(XQ& q, int b, const float* x64);
void xq_prep(XQ& q, const float* x);  // all blocks

// y[r - r0] = W[r] . x for rows r in [r0, r1). Row r of the codes starts at codes + r * row_bytes.
void mv_aff4(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ& x, float* y);
void mv_aff8(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ& x, float* y);
// AFF8 split into nibble planes (weights.h, Weight::AFF8S; rows = the matrix's rows), I16 activations: bit-identical to
// mv_aff8 on the unsplit bytes.
void mv_aff8s_i16(const uint8_t* codes, int rows, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ& x,
                  float* y);
void mv_nvfp4(const uint8_t* codes, const uint8_t* scales_e4m3, float gmul, int K, int r0, int r1, const XQ& x, float* y);
void mv_mxfp8(const uint8_t* codes, const uint8_t* scales_e8m0, int K, int r0, int r1, const XQ& x, float* y);
void mv_f32(const float* W, int K, int r0, int r1, const float* x, float* y);

}  // namespace dm
