// run_rows_n with the GPU's share (gpu_split.h).
#include "gpu_split.h"
#include "gpu_half.h"

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
    // the CPU's packed path is gemm_aff8_i16 / gemm_aff4_i16 (8- or 4-bit, inputs a multiple of 64): the GPU's bits are
    // that kernel's, which are mv's only there (other input widths stay on the CPU)
    const bool fits = j.xb && j.n > 0 && j.r0 == 0 && j.r1 < 0 && (m.kind == Weight::AFF8 || m.kind == Weight::AFF4) && m.cols % XB == 0;
    const int rg = fits ? ((int)(m.rows * frac) / 64) * 64 : 0;
    if (rg <= 0) continue;
    const int s = gpu->stage(*j.xb);
    const int t = gpu->add(m.kind == Weight::AFF8 ? 8 : 4, m.c, (const _Float16*)m.s, (const _Float16*)m.b, m.cols, rg, j.n, s);
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

bool run_rows_n_half(Pool& pool, GpuHalf* gpu, const JobN* jobs, int nj, int chunk, double* bytes) {
  if (!gpu) {
    run_rows_n(pool, jobs, nj, chunk, false, bytes);
    return true;
  }
  std::vector<JobN> cpu;
  std::vector<std::pair<int, int>> tk;  // (ticket, job)
  gpu->begin();
  for (int i = 0; i < nj; i++) {
    const JobN& j = jobs[i];
    int t = -1;
    if (j.n > 0 && j.r0 == 0 && j.r1 < 0 && gpu->holds(*j.w)) t = gpu->add(*j.w, j.w->rows, gpu->input(j.xf, j.n, j.w->cols), j.n);
    if (t >= 0) tk.push_back({t, i});
    else cpu.push_back(j);
  }
  const bool ok = tk.empty() || gpu->run(pool);
  if (!cpu.empty()) run_rows_n(pool, cpu.data(), (int)cpu.size(), chunk, false, bytes);
  if (!ok) return false;
  parallel_for(pool, (int)tk.size() * 8, [&](int u) {
    const JobN& j = jobs[tk[u / 8].second];
    for (int k = u % 8; k < j.n; k += 8) gpu->row(tk[u / 8].first, k, j.out[k]);
  });
  if (bytes)
    for (auto& p : tk) *bytes += (double)jobs[p.second].w->bytes();
  return true;
}

}  // namespace dm
