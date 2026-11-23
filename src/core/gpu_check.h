// A self-test of the prefill's GPU products on the device it runs on (no model needed): both GPU kernels against the
// CPU kernels bit for bit (8- and 4-bit, direct and gathered inputs, odd sizes), then rates for orientation: each GPU
// kernel alone on a 4-bit projection (8192 x 2048, 256 positions) and on an expert-shaped set (160 experts, 15
// positions each), the CPU's 4-bit batched kernel alone on `threads`, and both at once (with the balanced GPU share
// they imply). About 8 x secs seconds. Returns JSON ({"gpu":false} without a GPU); text: a readable summary.
#pragma once
#include <string>

namespace dm {
std::string gpu_gemm_selftest(int threads, double secs, std::string* text = nullptr);
}
