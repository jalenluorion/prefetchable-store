// run_rows_n with the GPU's share (gpu_gemm.h): for every job with a packed input (JobN::xb) and an 8-bit affine matrix, the GPU
// computes the first frac of the matrix's rows (whole 64-row blocks) while the CPU computes the rest through run_rows_n,
// at the same time. The GPU kernel computes gemm_aff8_i16's bits, which are mv's bits (weights.h
// mv_multi), so the outputs do not depend on the split. Jobs without a packed input, other formats, matrices outside
// the wrapped regions and the reference path stay on the CPU; if the GPU does not complete the set, the CPU computes
// its rows too. The per-position-output form of the 35B prefill's run_split (qwen3_5_moe/prefill.cpp), for Inkling's
// batched projections.
#pragma once
#include <cstdint>

#include "dmeng/pool.h"
#include "dmeng/weights.h"
#include "gpu_gemm.h"

namespace dm {

struct GpuSplitAcct {
  double ms_wait = 0, ms_stage = 0;  // waiting for the GPU's rows after the CPU's; staging copies
  uint64_t sets = 0, failed = 0;
  double macs_gpu = 0, macs_cpu = 0;
};

void run_rows_n_split(Pool& pool, GpuGemm* gpu, double frac, const JobN* jobs, int nj, int chunk, bool ref, double* bytes,
                      GpuSplitAcct* acct);

}  // namespace dm
