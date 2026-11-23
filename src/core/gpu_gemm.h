// Exact batched matrix products on the GPU (Metal), for prefill beside the CPU kernels: the 8-bit (AFF8) and 4-bit
// (AFF4) affine formats, on macOS and iOS.
//
// The kernel (gpu_gemm.mm) computes, per (row, position), the same exact int32 group dots and the same float epilogue as
// gemm_aff8_i16 / gemm_aff4_i16 (kernels_batch.cpp, gemm_tile_loop): per 64-input group the integer dot of the codes
// (8-bit bytes, or 4-bit nibbles: element 2j in the low nibble of byte j) with the 15-bit activations (hi * 256 + lo of
// the XQB planes), then lane g mod 4 updated with fma(dot, scale * d) and fma(bias, gsum), the four lanes summed
// pairwise at the end; IEEE fp32 with Metal's safe math mode. So its outputs are the CPU kernel's bits, and a prefill can
// give the GPU the first rows of a matrix and the CPU the rest, at the same time.
//
// One set of products at a time: begin(), stage() the inputs (an XQB, optionally read through a position map, so an
// expert's positions are gathered on the GPU from the layer's MoE input), add() the products, prepare() + copy(i) (the
// staging copies, run by the caller's threads) + submit(); the caller computes its own rows meanwhile, then finish()
// waits and out(ticket) holds the GPU's rows. finish() false (the GPU refused or failed the work, e.g. an iOS app in the
// background): the caller computes those rows itself, so a result never depends on the GPU's availability.
//
// Weights are read where they are: regions (the resident weights, the expert slab) are wrapped without a copy
// (page-aligned memory); a matrix outside every wrapped region is not taken (add() returns -1).
//
// Built when DM_METAL is defined (CMake's DM_METAL on Apple platforms, the Swift package for iOS / macOS); without it the
// stubs below report no GPU, and the prefill stays on the CPU.
#pragma once
#include <cstddef>
#include <cstdint>

#include "dmeng/kernels_batch.h"

namespace dm {

#if defined(DM_METAL)
struct GpuGemm {
  void* impl = nullptr;
  bool init();                                     // the device and the kernel; false: no GPU
  bool wrap(const void* base, size_t bytes);       // a page-aligned region (several allowed); false: not wrapped
  bool ok() const { return impl != nullptr; }
  bool holds(const void* p, size_t n) const;       // [p, p + n) inside one wrapped buffer
  size_t wrapped_bytes() const;
  // 1 (default): the matrix-unit kernel (gm: fp32 simdgroup multiply-accumulate on exact small integers); 0: the
  // integer kernel (gd). The same bits either way; set between sets.
  void set_kernel(int k);
  int kernel() const;
  void begin();
  int stage(const XQB& x);  // the activations of x (copied by copy()); once per XQB per set
  // rows [0, R) of a K-input matrix (fmt 8 or 4) against stage s at positions [0, n), or at positions map[0..n) of it;
  // -1: not taken (not inside one wrapped buffer). The map must stay valid until copy() has run.
  int add(int fmt, const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int R, int n, int s,
          const int* map = nullptr);
  // rows [0, R) of a K-input NVFP4 matrix (E2M1 codes, an E4M3 scale per 16 inputs, a global multiplier): the bits of
  // gemm_nvfp4_i16 (kernels_batch.h); -1: not taken (not wrapped, or no matrix-unit kernel on this device)
  int add_nvfp4(const uint8_t* codes, const uint8_t* scales_e4m3, float gmul, int K, int R, int n, int s, const int* map = nullptr);
  int prepare();         // sizes the buffers; returns the number of copy tasks
  void copy(int i);      // staging copy i (any thread; distinct i in parallel)
  void submit();
  bool finish();         // wait; false: the GPU did not complete the set
  const float* out(int ticket) const;  // [n][R] of the ticket
  int tickets() const;
  size_t buffer_bytes() const;  // the staging / output / table buffers held between sets
  void trim();                  // give those buffers back (re-made by the next set)
  GpuGemm() = default;
  GpuGemm(const GpuGemm&) = delete;
  GpuGemm& operator=(const GpuGemm&) = delete;
  ~GpuGemm();
};
#else
struct GpuGemm {  // built without Metal: never ok, so never used
  bool init() { return false; }
  bool wrap(const void*, size_t) { return false; }
  bool ok() const { return false; }
  bool holds(const void*, size_t) const { return false; }
  size_t wrapped_bytes() const { return 0; }
  void set_kernel(int) {}
  int kernel() const { return -1; }
  void begin() {}
  int stage(const XQB&) { return -1; }
  int add(int, const uint8_t*, const _Float16*, const _Float16*, int, int, int, int, const int* = nullptr) { return -1; }
  int add_nvfp4(const uint8_t*, const uint8_t*, float, int, int, int, int, const int* = nullptr) { return -1; }
  int prepare() { return 0; }
  void copy(int) {}
  void submit() {}
  bool finish() { return false; }
  const float* out(int) const { return nullptr; }
  int tickets() const { return 0; }
  size_t buffer_bytes() const { return 0; }
  void trim() {}
};
#endif

}  // namespace dm
