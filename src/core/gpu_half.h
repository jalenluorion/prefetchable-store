// Half-precision products on the GPU for the fast prompt path (fast mode only; exact mode never uses it): the weights as
// stored (8- and 4-bit affine records, groups of 64 along the input) dequantised to half inside the kernel, the inputs
// rounded to half, the products accumulated in fp32 on the matrix units. This is the old runner's standard (MLX's
// quantised matmul in fp16), not the CPU's bits: a prompt computed this way leaves a state close to exact prefill's, and
// fast mode's decode from it is checked against exact prefill's (tests and bench/prefill/fast_quality.sh).
//
// Two kinds of work, each one GPU command buffer, synchronous for the caller:
//   gemm sets  begin(); input() the fp32 rows of each input once; add() products (rows [0, R) of a wrapped matrix against
//              an input, optionally at a list of its positions); run() (staging by the caller's threads, then the GPU);
//              row(ticket, q) gives position q's outputs.
//   MoE blocks moe(): P positions' routed experts (each expert computed for the positions that picked it, gate and up
//              fused with the activation, then down) and the shared expert, combined per position in pick order on the
//              GPU (fixed order: the same bits from run to run), into fp32 outputs.
// Weights are read where they are: the resident weights and the expert slab are wrapped without a copy (page-aligned).
// A matrix outside every wrapped region, or of another kind, is not taken (add() returns -1; moe() returns false): the
// caller computes it on the CPU.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "dmeng/pool.h"
#include "dmeng/weights.h"

namespace dm {

#if defined(DM_METAL)
struct GpuHalf {
  void* impl = nullptr;
  bool init();
  bool wrap(const void* base, size_t bytes);
  bool holds(const Weight& w) const;  // an 8- or 4-bit affine matrix inside one wrapped region
  size_t wrapped_bytes() const;
  void touch();  // one GPU use of every wrapped buffer (their pages made resident for the GPU now)
  // ---- gemm sets
  void begin();
  int input(const float* const* rows, int n, int K);  // n fp32 rows of K values (pointers kept until run())
  // rows [0, R) of w (R <= w.rows) against input `in` at its positions [0, n), or at map[0..n) of them; -1: not taken
  int add(const Weight& w, int R, int in, int n, const int* map = nullptr);
  bool run(Pool& pool);                       // false: the GPU did not complete the set (the caller computes it)
  void row(int ticket, int q, float* dst) const;  // position q's R outputs of the ticket (fp32)
  // ---- a MoE block: x [P][H] fp32 inputs; per routed expert e its record's gate / up / down and its users (position
  // i, pick j: row i * K + j); w [P][K] the routing weights (0: left out); the shared expert (sh_* null: none) with its
  // scalar gates shw [P]. y[i] = fma chain over j of w[i K + j] * down_e(act) in pick order, then + shw[i] * shared.
  struct Exp {
    const Weight *gate, *up, *down;
    std::vector<int> pos, row;  // users: position in the block, assignment row (i * K + j)
  };
  bool moe(Pool& pool, const float* const* x, int P, int H, int K, int I, const std::vector<Exp>& ex, const float* w,
           const Weight* sh_gate, const Weight* sh_up, const Weight* sh_down, int SI, const float* shw, float* const* y);
  // the same block in waves (the old runner's pipelined union: compute the records that have landed while the rest are
  // read): moe_begin stages the inputs; each moe_wave commits its experts (and, with the first, the shared expert) without
  // waiting; moe_end adds the combination, waits for all and writes y. false anywhere: the block is dropped (caller: CPU)
  bool moe_begin(Pool& pool, const float* const* x, int P, int H, int K, int I, const Weight* sh_gate, const Weight* sh_up,
                 const Weight* sh_down, int SI);
  bool moe_wave(Pool& pool, const Exp* const* ex, int ne);
  bool moe_end(Pool& pool, const float* w, const float* shw, float* const* y);
  // ---- attention of n positions over listed keys (QSA's selection; all earlier keys while within its budget): K, V the
  // cache's half keys and values [NKV][ctx][D] (page-aligned regions), q[i] the position's NQ query heads [NQ][D] (fp32,
  // normed and rotated), sel[i] its keys; out[i] [NQ][D] = softmax(q K^T / sqrt(D)) V per head (fp32; the caller gates)
  bool attn(Pool& pool, const _Float16* K, const _Float16* V, int ctx, int D, int NQ, int NKV, int n, const float* const* q,
            const std::vector<int>* const* sel, float* const* out);
  // ---- the Gated DeltaNet's heads over n positions in order (gdn.h head_io, in fp32 with other summation orders; key and
  // value heads of 128): S the state [NVH][DV][DK] (read, then written back), cv[t] the convolved q, k, v [C], pj[t] the
  // projections (z at zoff [NVH DV], beta inputs at boff, alpha inputs at aoff [NVH]); out[t] the gated-normed heads [NVH DV]
  bool gdn(Pool& pool, int n, int NKH, int NVH, int DK, int DV, float* S, const float* cv, int C, const float* pj, int ldp, int zoff, int boff,
           int aoff, float* out, int ldo, const float* A_log, const float* dt_bias, const float* gnorm, bool sigmoid_gate, float eps);
  // counters
  double ms_stage = 0, ms_gpu = 0, flops = 0;
  long sets = 0, failed = 0;
  size_t buffer_bytes() const;
  void trim();  // give the staging / output buffers back
  GpuHalf() = default;
  GpuHalf(const GpuHalf&) = delete;
  GpuHalf& operator=(const GpuHalf&) = delete;
  ~GpuHalf();
};
#else
struct GpuHalf {
  struct Exp {
    const Weight *gate, *up, *down;
    std::vector<int> pos, row;
  };
  bool init() { return false; }
  bool wrap(const void*, size_t) { return false; }
  bool holds(const Weight&) const { return false; }
  size_t wrapped_bytes() const { return 0; }
  void touch() {}
  void begin() {}
  int input(const float* const*, int, int) { return -1; }
  int add(const Weight&, int, int, int, const int* = nullptr) { return -1; }
  bool run(Pool&) { return false; }
  void row(int, int, float*) const {}
  bool moe(Pool&, const float* const*, int, int, int, int, const std::vector<Exp>&, const float*, const Weight*, const Weight*,
           const Weight*, int, const float*, float* const*) {
    return false;
  }
  bool moe_begin(Pool&, const float* const*, int, int, int, int, const Weight*, const Weight*, const Weight*, int) { return false; }
  bool attn(Pool&, const _Float16*, const _Float16*, int, int, int, int, int, const float* const*, const std::vector<int>* const*, float* const*) {
    return false;
  }
  bool gdn(Pool&, int, int, int, int, int, float*, const float*, int, const float*, int, int, int, int, float*, int, const float*, const float*,
           const float*, bool, float) {
    return false;
  }
  bool moe_wave(Pool&, const Exp* const*, int) { return false; }
  bool moe_end(Pool&, const float*, const float*, float* const*) { return false; }
  double ms_stage = 0, ms_gpu = 0, flops = 0;
  long sets = 0, failed = 0;
  size_t buffer_bytes() const { return 0; }
  void trim() {}
};
#endif

// run_rows_n with the products on the GPU in half precision (fast prompt path): every job whose matrix the GPU takes
// (8- / 4-bit affine, wrapped, whole rows) runs there, the rest on the CPU (their exact kernels). false: the GPU did not
// complete its set (its jobs' outputs are not written; the caller computes them). gpu null: run_rows_n.
bool run_rows_n_half(Pool& pool, GpuHalf* gpu, const JobN* jobs, int nj, int chunk, double* bytes);

}  // namespace dm
