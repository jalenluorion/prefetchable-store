// Batched (many positions) matrix kernels for prefill, bit-identical to the batch-1 kernels of kernels.h.
//
// Why bit-identical is possible: in the I16 path each output is sum over 64-input groups of
//   (integer group dot) * (weight scale * activation scale) + weight bias * activation group sum,
// where the group dot is an exact int32 and the float part is accumulated in four lanes (group g into lane g mod 4) and
// summed pairwise at the end. Here the integer group dots come from i8mm (SMMLA / USMMLA: 2 rows x 2 positions x 8 inputs
// per instruction), which gives the same exact integers in any order, and the float part per (row, position) is the very
// same sequence of IEEE operations as mv_aff8 / mv_aff4. So prefill through these kernels and decode through mv give the
// same bits, and the engine's lossless guarantee carries over (checked by bench/bench_batch.cpp and the prefill check).
#pragma once
#include <cstddef>
#include <cstdint>

#include "dmeng/kernels.h"

namespace dm {

// Activations of N positions, quantised exactly as xq_block does (I16: per 64 inputs, two int8 planes), packed for i8mm:
// for each pair of positions and each block of 8 inputs, 16 bytes = position 2p's 8 values then position 2p+1's.
// Natural order (pairs with byte codes, AFF8) and the 4-bit perm order (pairs with nibble codes, AFF4) are both kept.
struct XQB {
  int K = 0, N = 0, NP = 0;  // inputs, positions, positions padded to a multiple of 8
  float* d = nullptr;        // [NP][K/64] block scales
  float* gsum = nullptr;     // [NP][K/64] block sums of the fp32 inputs
  int8_t *hn = nullptr, *ln = nullptr, *hp = nullptr, *lp = nullptr;  // [NP/2][K/8][16]
  void* mem = nullptr;
  size_t cap = 0;
  void reserve(int K, int N);  // (re)allocates when needed; keeps the memory
  ~XQB();
};
// Quantise position i (x: its K fp32 values) into the batch. Bit-identical to xq_block over its 64-blocks.
void xqb_set(XQB& q, int i, const float* x);
// Zero the padding positions [N, NP).
void xqb_pad(XQB& q);

// Y[(i - p0) * ldy + (r - r0)] = W[r] . x_i for rows r in [r0, r1) and positions i in [p0, p1). r0, r1 multiples of 4;
// p0 even; K a multiple of 64 (whole fours of 64-input groups in four lanes, then 1-3 groups as the one-vector kernels'
// scalar tail). Loops position tiles outside, row tiles inside: call it on row blocks of ~32-128 rows
// (one block per thread job) so a block's weights stay in L1/L2 while the positions stream past them.
void gemm_aff8_i16(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x,
                   int p0, int p1, float* Y, size_t ldy);
void gemm_aff4_i16(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x,
                   int p0, int p1, float* Y, size_t ldy);

// NVFP4 (E2M1 codes 2 per byte, an E4M3 scale per 16 inputs, gmul the tensor's global multiplier): Y as gemm_aff8_i16, bit-
// identical to mv_nvfp4 with I16 inputs. Integer sub-block dots from SMMLA (2 rows x 2 positions x 8 inputs, perm order), the
// float part per (row, position) the batch-1 kernel's operations. Any r1 - r0 (an odd last row runs the batch-1 path), any
// p0 / p1 (positions are taken in pairs; the partner of an odd last one is read but not written: keep it zero, xqb_pad).
// K a multiple of 64. Cost grows with the position pairs used, not with the 8-position padding.
void gemm_nvfp4_i16(const uint8_t* codes, const uint8_t* scales_e4m3, float gmul, int K, int r0, int r1, const XQB& x, int p0,
                    int p1, float* Y, size_t ldy);
// Position i of a batch from an XQ already quantised (I16: the same bytes, d and gsum copied; no requantisation, so the batch
// and the one-position kernels see identical inputs). Cheaper than xqb_set when the XQ exists.
void xqb_set_from(XQB& q, int i, const XQ& x);

// Multi-vector matvecs (decode / verify of a few positions): up to 8 inputs, the row's weights loaded once, each input's
// result bit-identical to the one-vector kernel's (I16 inputs; MXFP8 reads the fp16 plane). ys[i] gets rows [r0, r1).
void mv_aff8_multi(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQ* const* xs, int n,
                   float* const* ys);
void mv_nvfp4_multi(const uint8_t* codes, const uint8_t* scales_e4m3, float gmul, int K, int r0, int r1, const XQ* const* xs, int n,
                    float* const* ys);
void mv_mxfp8_multi(const uint8_t* codes, const uint8_t* scales_e8m0, int K, int r0, int r1, const XQ* const* xs, int n, float* const* ys);

// The same results through SME2 (int16 SMOPA for the group totals, the same NEON float epilogue): kernels_sme.cpp.
void gemm_aff8_i16_sme(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x,
                       int p0, int p1, float* Y, size_t ldy);
void gemm_aff4_i16_sme(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r0, int r1, const XQB& x,
                       int p0, int p1, float* Y, size_t ldy);

}  // namespace dm
