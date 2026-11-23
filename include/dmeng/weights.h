// Shared building blocks of the engine: memory regions, weight views over stored bytes, the resident file reader,
// embedding rows read from disk, row phases (several matrices in one fork-join) and the small numeric helpers every
// model uses. Nothing here knows which model runs.
#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "dmeng/kernels.h"
#include "dmeng/kernels_batch.h"
#include "dmeng/pool.h"

namespace dm {

using clk = std::chrono::steady_clock;
inline double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
inline double ms_between(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// ---------------------------------------------------------------------------------------------- memory
// PLAIN: anonymous pages, resident as touched. MLOCK: wired at allocation. PURGEABLE (expert slots only): each slot is
// its own VM object; idle slots are marked volatile so the system may drop them instead of compressing them.
// FILE (expert slots only): a held record is a view of the store file's pages in the OS file cache (store.h, Cache).
enum class Mem : int { PLAIN = 0, MLOCK = 1, PURGEABLE = 2, FILE = 3 };
uint8_t* alloc_region(size_t bytes, Mem mode, std::string* err);
void free_region(uint8_t* p, size_t bytes);
Mem mem_of(const std::string& s);  // "plain" | "mlock" | "purgeable" | "file"

// ---------------------------------------------------------------------------------------------- weights
// One matrix [rows = outputs, cols = inputs] over stored bytes. The quantised kinds share the kernels' Fmt numbering.
struct Weight {
  enum Kind : int { NONE = -1, AFF4 = 0, AFF8 = 1, NVFP4 = 2, MXFP8 = 3, BF16 = 4, F32 = 5 };
  Kind kind = NONE;
  int rows = 0, cols = 0;
  const uint8_t* c = nullptr;  // codes (quantised kinds)
  const void* s = nullptr;     // scales: fp16 (affine), E4M3 (NVFP4), E8M0 (MXFP8)
  const void* b = nullptr;     // fp16 biases (affine)
  const void* w = nullptr;     // bf16 or f32 values
  float gmul = 1.f;            // NVFP4 per-matrix global multiplier
  bool quantised() const { return kind >= AFF4 && kind <= MXFP8; }  // needs a prepared XQ input
  size_t bytes() const;                                             // bytes a full pass reads
};

// y[r - r0] = W[r] . x for rows [r0, r1). Quantised kinds read the prepared xq; bf16 / f32 read the fp32 xf.
// ref = the exact reference path (fp32 activations, scalar kernels) for AFF8 / NVFP4 / MXFP8 when xf is given.
void mv(const Weight& w, int r0, int r1, const XQ* xq, const float* xf, float* y, bool ref = false);

// exact reference kernels (fp32 activations)
void kref_aff8(const Weight& w, int r0, int r1, const float* x, float* y);
void kref_nvfp4(const uint8_t* codes, const uint8_t* sc, float gmul, int K, int r0, int r1, const float* x, float* y);
void kref_mxfp8(const uint8_t* codes, const uint8_t* sc, int K, int r0, int r1, const float* x, float* y);

// An XQ with its own (64-byte aligned) memory.
struct XBuf {
  XQ q;
  void* mem = nullptr;
  int cap = 0;
  XBuf() = default;
  XBuf(const XBuf&) = delete;
  XBuf& operator=(const XBuf&) = delete;
  XBuf(XBuf&& o) noexcept : q(o.q), mem(o.mem), cap(o.cap) { o.mem = nullptr; }
  ~XBuf();
  void init(int K, Prec p);
  void reset(int K) { q.K = K; }                // the vector length of the next prep / xq_block calls
  void prep(const float* x, int K);             // quantise x (K values, K <= cap)
};

// ---------------------------------------------------------------------------------------------- the DMENGRES1 file
// A text header (cfg lines, one line per tensor: name dtype offset bytes ndim shape...) and the payload, read into one
// region. Used for the 8-bit resident weights and the 8-/4-bit router files.
struct ResFile {
  struct T {
    std::string dtype;
    size_t off = 0, nbytes = 0;
    std::vector<long> shape;
  };
  std::map<std::string, std::string> cfg;
  std::map<std::string, T> t;
  uint8_t* base = nullptr;
  size_t size = 0, hdr = 0;
  Mem mem = Mem::PLAIN;
  // Mem::FILE: the whole file mapped PROT_READ / MAP_SHARED (clean file-backed pages; base = map + hdr), no copy
  uint8_t* map = nullptr;
  size_t map_size = 0;
  ResFile() = default;
  ResFile(const ResFile&) = delete;
  ResFile& operator=(const ResFile&) = delete;
  ~ResFile();
  bool load(const std::string& path, Mem mode, std::string& err);
  static bool read_cfg(const std::string& path, std::map<std::string, std::string>& cfg, std::string& err);  // header only
  bool has(const std::string& n) const { return t.count(n) > 0; }
  bool has_cfg(const std::string& k) const { return cfg.count(k) > 0; }
  const T& info(const std::string& n) const;
  const void* ptr(const std::string& n) const { return base + info(n).off; }
  const float* f32(const std::string& n) const;
  // NAME.q4/.s/.b -> AFF4; NAME.q/.s/.b -> AFF8 (AFF4 when the file's cfg says bits 4); else NAME as bf16 / f32
  Weight weight(const std::string& n) const;
  Weight bf16_row(const std::string& n) const;  // a 1-D bf16 tensor as a one-row matrix
  long geti(const std::string& k) const;
  double getf(const std::string& k) const;
  std::string gets(const std::string& k) const;
};

// The embedding table stays on disk; one bf16 row is read per token.
struct EmbedRows {
  int fd = -1;
  long off = 0;
  int H = 0;
  bool open(const std::string& path, long offset, int hidden, std::string& err);
  void row(int tok, float* out) const;
  ~EmbedRows();
};

// ---------------------------------------------------------------------------------------------- row phases
// Several matrices, each with its own input, computed in one fork-join; rows claimed in chunks. Each output row is
// computed by one thread start to finish, so the result does not depend on the thread count or scheduling.
struct Job {
  const Weight* w;
  const XQ* xq;     // input of the quantised kinds
  const float* xf;  // fp32 input (bf16 / f32 weights; every kind in the reference path)
  float* out;
};
void run_rows(Pool& pool, const Job* jobs, int n, int chunk, bool ref = false, double* bytes = nullptr);
// The same with the inputs' quantisation inside the fork-join: nq inputs xs[i] (Ks[i] values) into qs[i] block by block
// (xq_block, the blocks XBuf::prep would do), then the rows, each row chunk once every block is done (a counter): the
// serial prep before a row phase becomes part of it. The quantised bytes and the rows are XBuf::prep's and run_rows'.
void run_rows_q(Pool& pool, const Job* jobs, int n, int chunk, bool ref, double* bytes, XBuf* const* qs, const float* const* xs,
                const int* Ks, int nq);
// One matrix with a per-chunk epilogue run by the thread that computed the chunk, in the same fork-join:
// post(ctx, r0, r1) right after rows [r0, r1) are written (an output head's softcap, say). The rows and the epilogue's
// arithmetic are those of run_rows + a serial loop over the same rows: only where they run changes.
void run_rows_post(Pool& pool, const Job& job, int chunk, bool ref, double* bytes, void (*post)(void*, int, int), void* ctx);

// The same for several positions at once (multi-position verify, batched prefill): one matrix, n inputs, out[i] + r.
// AFF8 / NVFP4 / MXFP8 with I16 inputs go through the multi-vector kernels (8 inputs per weight pass), AFF8 with a packed
// batch (xb, n >= 6) through the i8mm gemm; everything else, and the reference path, through mv per input. Each input's
// rows are bit-identical to mv alone, whatever n, the chunking or the thread count.
void mv_multi(const Weight& w, int r0, int r1, const XQ* const* xq, const float* const* xf, int n, float* const* y, bool ref = false);
struct JobN {
  const Weight* w;
  int n;
  const XQ* const* xq;   // [n] quantised inputs
  const float* const* xf;  // [n] fp32 inputs
  float* const* out;     // [n] outputs (rows written at out[i] + r)
  const XQB* xb = nullptr;  // the same inputs packed (optional)
};
void run_rows_n(Pool& pool, const JobN* jobs, int nj, int chunk, bool ref = false, double* bytes = nullptr);

// ---------------------------------------------------------------------------------------------- numerics
float dot_bf16(const uint16_t* w, const float* x, int n);
// out = x / rms(x) * w (w null: no weight); the sum of squares in double, the scale in float
inline void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float r = 1.f / std::sqrt((float)(ss / n) + eps);
  for (int i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.f);
}
inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline float softcap(float x, float c) { return c * std::tanh(x / c); }
inline float gelu_erf(float x) { return 0.5f * x * (1.f + std::erf(x * 0.70710678118654752f)); }
// v + a * b with the product rounded on its own (never fused, whatever -ffp-contract says): for the places where the
// arithmetic of record has an unfused multiply-add
inline float add_unfused(float v, float a, float b) {
  volatile float p = a * b;
  return v + p;
}
inline float bf16_to_f32(uint16_t v) {
  uint32_t b = (uint32_t)v << 16;
  float f;
  __builtin_memcpy(&f, &b, 4);
  return f;
}

}  // namespace dm
