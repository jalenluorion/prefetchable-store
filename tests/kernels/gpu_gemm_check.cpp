// The GPU's batched products (src/core/gpu_gemm.h) against the CPU kernels, bit for bit, and their rates on this machine
// (src/core/gpu_check.cpp; the same test the C API runs on the phone as dm_gpu_check). Exit 0 only if every output's
// bits match.
//   gpu_gemm_check [cpu_threads (4)] [seconds per rate (2)]
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../../src/core/gpu_check.h"

int main(int argc, char** argv) {
  const int threads = argc > 1 ? atoi(argv[1]) : 4;
  const double secs = argc > 2 ? atof(argv[2]) : 2;
  std::string text;
  const std::string js = dm::gpu_gemm_selftest(threads, secs, &text);
  printf("%s%s\n", text.c_str(), js.c_str());
  return js.find("\"exact\":true") != std::string::npos ? 0 : 1;
}
