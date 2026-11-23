// run_rows_n with the GPU's share (gpu_split.h).
#include "gpu_split.h"

#include <chrono>
#include <cstring>
#include <utility>
#include <vector>

namespace dm {

static double ms_from(std::chrono::steady_clock::time_point a) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
}

void run_rows_n_split(Pool& pool, GpuGemm* gpu, double frac, const JobN* jobs, int nj, int chunk, bool ref, double* bytes,
                      GpuSplitAcct* acct) {
  if (!gpu || !gpu->ok() || !(frac > 0) || ref) {
    run_rows_n(pool, jobs, nj, chunk, ref, bytes);
    return;
  }
  std::vector<JobN> J(jobs, jobs + nj);
  std::vector<std::pair<int, int>> tk;  // (ticket, job)
  auto ts = std::chrono::steady_clock::now();
  gpu->begin();
  for (int i = 0; i < nj; i++) {
    JobN& j = J[i];
    const Weight& m = *j.w;
    const bool fits = j.xb && j.n > 0 && j.r0 == 0 && j.r1 < 0 && m.kind == Weight::AFF8;  // the CPU's packed path is gemm_aff8_i16 (8-bit)
    const int rg = fits ? ((int)(m.rows * frac) / 64) * 64 : 0;
    if (rg <= 0) continue;
    const int s = gpu->stage(*j.xb);
    const int t = gpu->add(8, m.c, (const _Float16*)m.s, (const _Float16*)m.b, m.cols, rg, j.n, s);
    if (t < 0) continue;  // not inside a wrapped region: the CPU computes the whole matrix
    tk.push_back({t, i});
    j.r0 = rg;  // the CPU's rows
  }
  const int nc = tk.empty() ? 0 : gpu->prepare();
  if (nc == 0 && !tk.empty()) {  // no GPU buffers: everything on the CPU
    for (auto& p : tk) J[p.second].r0 = 0;
    tk.clear();
  }
  if (!tk.empty()) {
    parallel_for(pool, nc, [&](int i) { gpu->copy(i); });
    gpu->submit();
  }
  if (acct) {
    acct->ms_stage += ms_from(ts);
    for (int i = 0; i < nj; i++) {
      const Weight& m = *J[i].w;
      acct->macs_gpu += (double)J[i].r0 * m.cols * J[i].n;
      acct->macs_cpu += (double)(m.rows - J[i].r0) * m.cols * J[i].n;
    }
  }
  run_rows_n(pool, J.data(), nj, chunk, ref, bytes);
  if (tk.empty()) return;
  auto tw = std::chrono::steady_clock::now();
  const bool ok = gpu->finish();
  if (acct) {
    acct->ms_wait += ms_from(tw);
    acct->sets++;
  }
  if (ok) {
    parallel_for(pool, (int)tk.size(), [&](int u) {
      const JobN& j = J[tk[u].second];
      const int rg = j.r0;
      const float* y = gpu->out(tk[u].first);
      for (int k = 0; k < j.n; k++) std::memcpy(j.out[k], y + (size_t)k * rg, (size_t)rg * 4);
    });
  } else {  // the GPU's rows on the CPU (the same bits)
    if (acct) acct->failed++;
    std::vector<JobN> R;
    for (auto& p : tk) {
      JobN j = J[p.second];
      j.r1 = j.r0;
      j.r0 = 0;
      R.push_back(j);
    }
    run_rows_n(pool, R.data(), (int)R.size(), chunk, ref, nullptr);
  }
}

}  // namespace dm
