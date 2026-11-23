// Internal interfaces of the P0 bench: weight buffers of this model's shapes, a timed CPU pass, a background CPU load.
#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "dmeng/kernels.h"
#include "dmeng/kernels_batch.h"
#include "dmeng/pool.h"
#include "util.h"

namespace dmb {

struct Case {
  std::string name;  // short id
  std::string what;  // which matrix of Qwen3.6-35B-A3B
  dm::Fmt fmt;
  int rows, cols;
};
std::vector<Case> model_cases();
const Case* find_case(const std::string& id);

// nmat matrices of one case laid out back to back (codes, fp16 scales, fp16 biases; 64-byte aligned), filling at least
// total_bytes, so a pass over them streams from DRAM as a token's weights do.
struct MatBuf {
  Case c;
  uint8_t* buf = nullptr;
  size_t stride = 0, code_bytes = 0, scale_bytes = 0, mat_bytes = 0;
  int nmat = 0;
  void make(const Case& c, size_t total_bytes, uint64_t seed);
  void release();
  const uint8_t* codes(int m) const { return buf + (size_t)m * stride; }
  const _Float16* scales(int m) const { return (const _Float16*)(codes(m) + code_bytes); }
  const _Float16* biases(int m) const { return (const _Float16*)(codes(m) + code_bytes + scale_bytes); }
  double macs_per_pass() const { return (double)c.rows * c.cols * nmat; }
  double bytes_per_pass() const { return (double)mat_bytes * nmat; }
};
size_t mat_bytes(const Case& c);

// one pass of every matrix of b through the pool (every thread a contiguous row range of every matrix, I16 activations)
double cpu_pass(dm::Pool& pool, const MatBuf& b, const dm::XQ& x, float* y);

// A CPU kernel load on its own thread (and pool) until stop(): counts bytes and passes so a caller can take windows.
struct CpuLoad {
  std::thread th;
  std::atomic<bool> stop_{false};
  std::atomic<int64_t> bytes{0}, passes{0};
  std::atomic<bool> ready{false};
  MatBuf b;
  int threads = 2;
  std::string qos = "ui";
  void start(const Case& c, int threads, const std::string& qos, size_t mb);
  void stop();
  struct Mark { double t; int64_t bytes; };
  Mark mark() const;
  static double gbps(const Mark& a, const Mark& b) { return (b.bytes - a.bytes) / (b.t - a.t) / 1e9; }
};

// Metal part of the kernel bench (metal.mm): appends "metal": {...} members to j
void metal_bench(J& j, const Spec& sp, dmb_log_fn log);

}  // namespace dmb
