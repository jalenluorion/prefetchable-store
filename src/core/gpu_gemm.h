// Exact 8-bit batched matrix products on the GPU (Metal), for prefill beside the CPU kernels.
//
// The kernel (gpu_gemm.mm) computes, per (row, position), the same exact int32 group dots and the same float epilogue as
// mv_aff8 / gemm_aff8_i16 (kernels_batch.h): per 64-input group the integer dot of the 8-bit codes with the 15-bit
// activations (hi * 256 + lo of the XQB planes), then lane g mod 4 updated with fma(dot, scale * d) and fma(bias, gsum),
// the four lanes summed pairwise at the end; IEEE fp32 with Metal's safe math mode. So its outputs are the CPU kernel's
// bits, and a prefill can give the GPU the first rows of an 8-bit matrix and the CPU the rest, at the same time.
//
// Built on macOS when CMake's DM_METAL is on (the default there); without it the stubs below report no GPU, and the
// prefill stays on the CPU.
#pragma once
#include <cstddef>
#include <cstdint>

#include "dmeng/kernels_batch.h"

namespace dm {

#if defined(DM_METAL)
struct GpuGemm {
  void* impl = nullptr;
  // wrap a page-aligned region (the resident weights) so matrices inside it need no copy
  bool init(const void* res_base, size_t res_bytes);
  bool ok() const { return impl != nullptr; }
  bool holds(const void* p, size_t n) const;  // [p, p + n) inside the wrapped region
  // stage the activations of positions [0, x.N) of x (I16, natural order) for the GPU; once per XQB per sub-chunk
  void stage(const XQB& x, int slot);
  // enqueue rows [0, r1) of an 8-bit matrix (codes / scales / biases inside the wrapped region) against a staged slot;
  // the output rows land in an internal buffer; returns a ticket
  int enqueue(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r1, int n, int slot);
  void commit();  // submit everything enqueued since the last commit
  // wait for the submitted work, then copy the ticket's output into Y[i * ldy + r] for r < r1, i < n
  void finish(int ticket, float* Y, size_t ldy);
  void wait();
  GpuGemm() = default;
  GpuGemm(const GpuGemm&) = delete;
  GpuGemm& operator=(const GpuGemm&) = delete;
  ~GpuGemm();
};
#else
struct GpuGemm {  // built without Metal: never ok, so never used
  bool init(const void*, size_t) { return false; }
  bool ok() const { return false; }
  bool holds(const void*, size_t) const { return false; }
  void stage(const XQB&, int) {}
  int enqueue(const uint8_t*, const _Float16*, const _Float16*, int, int, int, int) { return -1; }
  void commit() {}
  void finish(int, float*, size_t) {}
  void wait() {}
};
#endif

}  // namespace dm
