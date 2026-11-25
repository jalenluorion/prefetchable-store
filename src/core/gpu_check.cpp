// The GPU's batched products (gpu_gemm.h) checked against the CPU kernels on the device itself, and their rates (see
// gpu_check.h). No model needed: random affine weights in a wrapped page-aligned region, activations quantised by
// xqb_set.
#include "gpu_check.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "dmeng/kernels_batch.h"
#include "dmeng/pool.h"
#include "dmeng/weights.h"
#include "gpu_gemm.h"

namespace dm {

namespace {
using sclk = std::chrono::steady_clock;
double secs_since(sclk::time_point t) { return std::chrono::duration<double>(sclk::now() - t).count(); }

struct Mat {
  int fmt, R, K;
  const uint8_t* c;
  const _Float16 *s, *b;
};
struct Arena {
  uint8_t* base = nullptr;
  size_t bytes = 0, off = 0;
  std::mt19937 rng{7};
  bool init(size_t n) {
    std::string err;
    base = alloc_region(n, Mem::PLAIN, &err);
    bytes = n;
    if (!base) return false;
    for (size_t i = 0; i < n; i += 4) {
      const uint32_t v = rng();
      std::memcpy(base + i, &v, std::min<size_t>(4, n - i));
    }
    return true;
  }
  ~Arena() { free_region(base, bytes); }
  uint8_t* raw(size_t n) {  // random bytes
    if (off + n + 1024 > bytes) {
      fprintf(stderr, "gpu_check: test region too small\n");
      abort();
    }
    uint8_t* p = base + off;
    off += (n + 255) & ~size_t(255);
    return p;
  }
  Mat mat(int fmt, int R, int K) {  // codes random; fp16 scales and biases set to sane values
    const size_t NG = K / 64, cb = (size_t)R * (fmt == 8 ? K : K / 2);
    if (off + cb + 4 * R * NG + 1024 > bytes) {
      fprintf(stderr, "gpu_check: test region too small\n");
      abort();
    }
    Mat m{fmt, R, K, base + off, nullptr, nullptr};
    off += (cb + 255) & ~size_t(255);
    _Float16* S = (_Float16*)(base + off);
    off += (R * NG * 2 + 255) & ~size_t(255);
    _Float16* B = (_Float16*)(base + off);
    off += (R * NG * 2 + 255) & ~size_t(255);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t i = 0; i < R * NG; i++) {
      S[i] = (_Float16)(0.01f * std::fabs(nd(rng)) + 1e-4f);
      B[i] = (_Float16)(0.1f * nd(rng));
    }
    m.s = S;
    m.b = B;
    return m;
  }
};

void cpu_gemm(const Mat& m, int r0, int r1, const XQB& x, int n, float* Y, size_t ldy) {
  if (m.fmt == 8) gemm_aff8_i16(m.c, m.s, m.b, m.K, r0, r1, x, 0, n, Y, ldy);
  else gemm_aff4_i16(m.c, m.s, m.b, m.K, r0, r1, x, 0, n, Y, ldy);
}

// the CPU kernel on rows of m, 64-row jobs over a pool, for `secs`: GMAC/s
struct CpuRun {
  const Mat* m;
  const XQB* x;
  int n;
  float* Y;
  std::atomic<int> next{0};
};
void cpu_task(void* p, int, int) {
  CpuRun& c = *(CpuRun*)p;
  for (int b; (b = c.next.fetch_add(1)) * 64 < c.m->R;) cpu_gemm(*c.m, b * 64, std::min(c.m->R, b * 64 + 64), *c.x, c.n, c.Y + b * 64, c.m->R);
}
double cpu_rate(Pool& pool, const Mat& m, const XQB& x, int n, double secs, const std::atomic<bool>* stop = nullptr) {
  std::vector<float> Y((size_t)n * m.R);
  double macs = 0;
  auto t0 = sclk::now();
  while (secs_since(t0) < secs && !(stop && stop->load())) {
    CpuRun c{&m, &x, n, Y.data()};
    pool.run(cpu_task, &c);
    macs += (double)m.R * m.K * n;
  }
  return macs / secs_since(t0) / 1e9;
}

void fill_x(XQB& x, int K, int n, std::mt19937& rng, float scale) {
  std::normal_distribution<float> nd(0.f, scale);
  std::vector<float> v(K);
  x.reserve(K, n);
  for (int q = 0; q < n; q++) {
    for (auto& a : v) a = nd(rng) * (q % 7 == 3 ? 100.f : 1.f);
    if (q % 11 == 5) std::fill(v.begin(), v.end(), 0.f);
    xqb_set(x, q, v.data());
  }
  xqb_pad(x);
}

void gather(XQB& dst, const XQB& src, const std::vector<int>& map) {  // the planes copied, as the prefill's xqb_copy
  const int K = src.K, NG = K / 64, KB = K / 8, n = (int)map.size();
  dst.reserve(K, n);
  for (int t = 0; t < n; t++) {
    const int si = map[t];
    std::memcpy(dst.d + (size_t)t * NG, src.d + (size_t)si * NG, NG * 4);
    std::memcpy(dst.gsum + (size_t)t * NG, src.gsum + (size_t)si * NG, NG * 4);
    const int8_t* sp[4] = {src.hn, src.ln, src.hp, src.lp};
    int8_t* dp[4] = {dst.hn, dst.ln, dst.hp, dst.lp};
    const size_t so = (size_t)(si >> 1) * KB * 16 + (si & 1) * 8, dof = (size_t)(t >> 1) * KB * 16 + (t & 1) * 8;
    for (int q = 0; q < 4; q++)
      for (int b = 0; b < KB; b++) std::memcpy(dp[q] + dof + (size_t)b * 16, sp[q] + so + (size_t)b * 16, 8);
  }
  xqb_pad(dst);
}

void run_set(GpuGemm& g) {
  const int nc = g.prepare();
  for (int i = 0; i < nc; i++) g.copy(i);
  g.submit();
}
}  // namespace

// The CPU's batched kernels (gemm_aff8_i16 / gemm_aff4_i16) against the one-vector kernels (mv_aff8 / mv_aff4, I16) they
// must equal, bit for bit, on input widths with and without a tail of groups (K not a multiple of 256). Outputs that differ.
long cpu_gemm_vs_mv(std::ostringstream& tx, long* checked) {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.f, 1.f);
  long bad = 0;
  struct C { int fmt, R, K, n; };
  for (const C& p : std::vector<C>{{8, 64, 640, 13}, {4, 64, 640, 13}, {8, 32, 320, 9}, {4, 32, 320, 9}, {8, 16, 128, 5},
                                   {4, 16, 192, 4}, {8, 16, 2048, 7}, {4, 16, 2560, 3}}) {
    const int NG = p.K / 64;
    std::vector<uint8_t> codes((size_t)p.R * (p.fmt == 8 ? p.K : p.K / 2));
    for (auto& c : codes) c = (uint8_t)rng();
    std::vector<_Float16> sc((size_t)p.R * NG), bi((size_t)p.R * NG);
    for (auto& v : sc) v = (_Float16)(0.001f + 0.01f * (rng() % 1000) / 1000.f);
    for (auto& v : bi) v = (_Float16)(-0.05f + 0.1f * (rng() % 1000) / 1000.f);
    XQB xb;
    xb.reserve(p.K, p.n);
    std::vector<XBuf> xq(p.n);
    std::vector<float> v(p.K);
    for (int q = 0; q < p.n; q++) {
      for (auto& a : v) a = nd(rng) * (q % 3 == 1 ? 30.f : 1.f);
      xq[q].init(p.K, Prec::I16);
      xq[q].prep(v.data(), p.K);
      xqb_set(xb, q, v.data());
    }
    xqb_pad(xb);
    std::vector<float> Y((size_t)p.n * p.R), y1(p.R);
    if (p.fmt == 8) gemm_aff8_i16(codes.data(), sc.data(), bi.data(), p.K, 0, p.R, xb, 0, p.n, Y.data(), p.R);
    else gemm_aff4_i16(codes.data(), sc.data(), bi.data(), p.K, 0, p.R, xb, 0, p.n, Y.data(), p.R);
    long b = 0;
    for (int q = 0; q < p.n; q++) {
      if (p.fmt == 8) mv_aff8(codes.data(), sc.data(), bi.data(), p.K, 0, p.R, xq[q].q, y1.data());
      else mv_aff4(codes.data(), sc.data(), bi.data(), p.K, 0, p.R, xq[q].q, y1.data());
      for (int r = 0; r < p.R; r++) b += std::memcmp(&y1[r], &Y[(size_t)q * p.R + r], 4) != 0;
    }
    if (b) tx << "CPU gemm against mv (fmt " << p.fmt << ", K " << p.K << "): " << b << " differ\n";
    *checked += (long)p.n * p.R;
    bad += b;
  }
  return bad;
}

std::string gpu_gemm_selftest(int threads, double secs, std::string* text) {
  std::ostringstream js, tx;
  js.setf(std::ios::fixed);
  js.precision(2);
  tx.setf(std::ios::fixed);
  tx.precision(1);
  long cpu_checked = 0;
  const long cpu_bad = cpu_gemm_vs_mv(tx, &cpu_checked);
  tx << "CPU gemm against the one-vector kernels: " << cpu_checked << " outputs, " << cpu_bad << " differ\n";
  GpuGemm g;
  if (!g.init()) {
    if (text) *text = "GPU CHECK: no GPU (no Metal device, a build without DM_METAL, or the kernel did not compile)\n";
    return "{\"gpu\":false}";
  }
  Arena A;
  if (!A.init((size_t)224 << 20) || !g.wrap(A.base, A.bytes)) {
    if (text) *text = "GPU CHECK FAILED: could not wrap a test region\n";
    return "{\"gpu\":true,\"exact\":false,\"error\":\"wrap\"}";
  }
  // 1. exactness: both kernels, 8- and 4-bit, direct and gathered inputs, odd sizes, several products per set
  struct P {
    int fmt, R, K, n, src_n;
    bool mapped;
  };
  const std::vector<P> cases = {{8, 4096, 2048, 256, 256, false}, {4, 4096, 2048, 256, 256, false}, {8, 2048, 4096, 37, 37, false},
                                {4, 2048, 4096, 37, 37, false},   {4, 512, 2048, 15, 300, true},     {4, 512, 2048, 1, 300, true},
                                {8, 1024, 512, 29, 29, false},    {4, 1024, 512, 3, 3, false},       {4, 256, 2048, 300, 300, false},
                                {8, 64, 256, 17, 40, true},
                                // inputs not a multiple of 256 (whole fours of groups, then the scalar tail): Flash-Next's
                                // 640-wide expert / shared down projections and its 320-wide mixer, and tails without fours
                                {4, 2560, 640, 29, 29, false},    {8, 2560, 640, 37, 37, false},     {8, 4096, 320, 40, 40, false},
                                {4, 512, 640, 15, 300, true},     {8, 256, 128, 9, 9, false},        {4, 512, 192, 6, 6, false}};
  long checked = 0, bad = 0;
  bool complete = true;
  for (int rep = 0; rep < 4; rep++) {
    const int kern = rep % 2;
    g.set_kernel(kern);
    if (g.kernel() != kern) {
      tx << "kernel " << kern << " unavailable on this device\n";
      bad++;
      continue;
    }
    A.off = 0;
    std::vector<XQB> xs(cases.size()), gx(cases.size());
    std::vector<std::vector<int>> maps(cases.size());
    std::vector<Mat> ms;
    std::vector<int> tk(cases.size()), rgs(cases.size());
    g.begin();
    for (size_t i = 0; i < cases.size(); i++) {
      const P& p = cases[i];
      ms.push_back(A.mat(p.fmt, p.R, p.K));
      fill_x(xs[i], p.K, p.src_n, A.rng, 1.f + rep);
      if (p.mapped) {
        maps[i].resize(p.n);
        for (auto& m : maps[i]) m = (int)(A.rng() % p.src_n);
      }
      rgs[i] = std::max(64, (p.R / 2) / 64 * 64);  // the GPU takes rows [0, rg) as the prefill does
      tk[i] = g.add(p.fmt, ms[i].c, ms[i].s, ms[i].b, p.K, rgs[i], p.n, g.stage(xs[i]), p.mapped ? maps[i].data() : nullptr);
    }
    run_set(g);
    if (!g.finish()) {
      complete = false;
      break;
    }
    for (size_t i = 0; i < cases.size(); i++) {
      const P& p = cases[i];
      const XQB* xin = &xs[i];
      if (p.mapped) {
        gather(gx[i], xs[i], maps[i]);
        xin = &gx[i];
      }
      std::vector<float> ref((size_t)p.n * rgs[i]);
      cpu_gemm(ms[i], 0, rgs[i], *xin, p.n, ref.data(), rgs[i]);
      const float* y = g.out(tk[i]);
      long b = 0;
      for (size_t u = 0; u < ref.size(); u++) b += std::memcmp(&ref[u], &y[u], 4) != 0;
      if (b) tx << "kernel " << kern << " case " << i << " (fmt " << p.fmt << ", R " << p.R << ", K " << p.K << ", n " << p.n << "): " << b << " differ\n";
      checked += (long)ref.size();
      bad += b;
    }
  }
  // 1b. NVFP4 (Inkling's experts): gn against gemm_nvfp4_i16, random codes, E4M3 scales in a sane range, odd position
  // counts, a block count that is not a multiple of four, a gathered case
  if (complete) {
    struct NV {
      int R, K, n, src_n;
      bool mapped;
    };
    const std::vector<NV> nc = {{512, 2048, 37, 37, false}, {256, 2048, 300, 300, false}, {1024, 512, 6, 6, false},
                                {768, 2880, 19, 40, true},  {128, 2880, 1, 1, false}};
    A.off = 0;
    std::vector<XQB> xs(nc.size()), gx(nc.size());
    std::vector<std::vector<int>> maps(nc.size());
    std::vector<const uint8_t*> cs(nc.size()), ss(nc.size());
    std::vector<int> tk(nc.size());
    const float gmul = 0.37f;
    g.set_kernel(1);
    g.begin();
    for (size_t i = 0; i < nc.size(); i++) {
      const NV& p = nc[i];
      cs[i] = A.raw((size_t)p.R * p.K / 2);
      uint8_t* sc = A.raw((size_t)p.R * p.K / 16);
      for (size_t u = 0; u < (size_t)p.R * p.K / 16; u++) sc[u] = (uint8_t)(0x20 + A.rng() % 0x38) | (A.rng() % 5 == 0 ? 0x80 : 0);
      ss[i] = sc;
      fill_x(xs[i], p.K, p.src_n, A.rng, 1.f);
      if (p.mapped) {
        maps[i].resize(p.n);
        for (auto& m : maps[i]) m = (int)(A.rng() % p.src_n);
      }
      tk[i] = g.add_nvfp4(cs[i], ss[i], gmul, p.K, p.R, p.n, g.stage(xs[i]), p.mapped ? maps[i].data() : nullptr);
      if (tk[i] < 0) tx << "NVFP4 case " << i << " not taken\n", bad++;
    }
    run_set(g);
    if (!g.finish()) complete = false;
    for (size_t i = 0; complete && i < nc.size(); i++) {
      if (tk[i] < 0) continue;
      const NV& p = nc[i];
      const XQB* xin = &xs[i];
      if (p.mapped) {
        gather(gx[i], xs[i], maps[i]);
        xin = &gx[i];
      }
      std::vector<float> ref((size_t)p.n * p.R);
      gemm_nvfp4_i16(cs[i], ss[i], gmul, p.K, 0, p.R, *xin, 0, p.n, ref.data(), p.R);
      const float* y = g.out(tk[i]);
      long b = 0;
      for (size_t u = 0; u < ref.size(); u++) b += std::memcmp(&ref[u], &y[u], 4) != 0;
      if (b) tx << "NVFP4 case " << i << " (R " << p.R << ", K " << p.K << ", n " << p.n << "): " << b << " differ\n";
      checked += (long)ref.size();
      bad += b;
    }
  }
  const bool exact = complete && bad == 0 && checked > 0 && cpu_bad == 0;
  js << "{\"gpu\":true,\"exact\":" << (exact ? "true" : "false") << ",\"outputs_compared\":" << checked << ",\"outputs_differ\":" << bad
     << ",\"complete\":" << (complete ? "true" : "false");
  tx << "GPU CHECK " << (exact ? "PASSED" : "FAILED") << " (" << checked << " outputs of both kernels compared with the CPU kernels, " << bad
     << " differ" << (complete ? "" : "; a set did not complete") << ")\n";
  // 2. rates (orientation): GPU alone per kernel on a projection (4-bit 8192 x 2048, N 256) and an expert-shaped set
  // (160 experts x gate + up 256 rows of 2048 and down 1024 rows of 512, 15 positions each, 4-bit); the CPU's 4-bit
  // kernel alone on `threads`; both at once (the GPU taking its share while the CPU runs)
  A.off = 0;
  const int NP = 256;
  const Mat big = A.mat(4, 8192, 2048);
  XQB xb;
  fill_x(xb, 2048, NP, A.rng, 1.f);
  const int E = 160, ne = 15, src_n = 472;
  std::vector<Mat> ex;
  for (int e = 0; e < E; e++) {
    ex.push_back(A.mat(4, 256, 2048));
    ex.push_back(A.mat(4, 256, 2048));
    ex.push_back(A.mat(4, 1024, 512));
  }
  XQB xa, xd;
  fill_x(xa, 2048, src_n, A.rng, 1.f);
  fill_x(xd, 512, E * ne, A.rng, 1.f);
  std::vector<int> emap(E * ne);
  for (auto& m : emap) m = (int)(A.rng() % src_n);
  auto gpu_proj = [&](double s, const std::atomic<bool>* stop) {
    double macs = 0;
    auto t0 = sclk::now();
    while (secs_since(t0) < s && !(stop && stop->load())) {
      g.begin();
      g.add(4, big.c, big.s, big.b, 2048, 8192, NP, g.stage(xb));
      run_set(g);
      g.finish();
      macs += 8192.0 * 2048 * NP;
    }
    return macs / secs_since(t0) / 1e9;
  };
  auto gpu_experts = [&](double s) {
    double macs = 0;
    auto t0 = sclk::now();
    while (secs_since(t0) < s) {
      g.begin();
      const int sa = g.stage(xa);
      for (int e = 0; e < E; e++) {
        g.add(4, ex[3 * e].c, ex[3 * e].s, ex[3 * e].b, 2048, 256, ne, sa, emap.data() + e * ne);
        g.add(4, ex[3 * e + 1].c, ex[3 * e + 1].s, ex[3 * e + 1].b, 2048, 256, ne, sa, emap.data() + e * ne);
      }
      run_set(g);
      g.finish();
      g.begin();
      const int sd = g.stage(xd);
      for (int e = 0; e < E; e++) g.add(4, ex[3 * e + 2].c, ex[3 * e + 2].s, ex[3 * e + 2].b, 512, 1024, ne, sd, emap.data() + e * ne);
      run_set(g);
      g.finish();
      macs += (double)E * ne * (2.0 * 256 * 2048 + 1024.0 * 512);
    }
    return macs / secs_since(t0) / 1e9;
  };
  const double s = std::max(0.5, secs);
  js << ",\"rates_gmacs\":{";
  tx << "rates (GMAC/s, " << s << " s each):";
  for (int kern : {0, 1}) {
    g.set_kernel(kern);
    if (g.kernel() != kern) continue;
    const char* name = kern ? "matrix" : "integer";
    const double rp = gpu_proj(s, nullptr), re = gpu_experts(s);
    js << "\"gpu_" << name << "_proj\":" << rp << ",\"gpu_" << name << "_experts\":" << re << ",";
    tx << " GPU " << name << " projection " << rp << ", experts " << re << ";";
  }
  g.set_kernel(1);
  Pool pool(std::max(1, threads));
  const double c_alone = cpu_rate(pool, big, xb, NP, s);
  // together: the CPU on its pool in a thread, the GPU (default kernel) on this one, for the same time
  double c_tog = 0;
  std::thread ct([&] {
    Pool p2(std::max(1, threads));
    c_tog = cpu_rate(p2, big, xb, NP, s, nullptr);
  });
  const double g_tog = gpu_proj(s, nullptr);
  ct.join();
  js << "\"cpu_threads\":" << threads << ",\"cpu_alone_proj\":" << c_alone << ",\"together_gpu_proj\":" << g_tog << ",\"together_cpu_proj\":" << c_tog
     << ",\"together_sum\":" << g_tog + c_tog << ",\"split_frac\":" << g_tog / std::max(1e-9, g_tog + c_tog) << "}}";
  tx << " CPU (" << threads << " threads) " << c_alone << "; together GPU " << g_tog << " + CPU " << c_tog << " = " << g_tog + c_tog
     << " (balanced GPU share " << g_tog / std::max(1e-9, g_tog + c_tog) << ")\n";
  if (text) *text = tx.str();
  return js.str();
}

}  // namespace dm
