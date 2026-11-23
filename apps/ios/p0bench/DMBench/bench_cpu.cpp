// P0 kernel bench, CPU part (dmb_kernel_bench). Spec keys (defaults for the phone):
//   cell=3        seconds per sweep cell
//   threads=1:2:4:6   thread counts; qos=ui:default   QoS classes of the caller and the pool's workers
//   mb=384        weight buffer per case (DRAM-fed: far above the caches)
//   cases=all     or a list a4gu:a4dn:...
//   batch=1       the batched (prefill) kernels at N = 8 / 64 / 256
//   sustain=60    seconds of the sustained runs (0 = none); metal=1 the Metal part (metal.mm)
//   bgthreads=4   E-core cells at QoS background (0 = none)
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "bench.h"

using namespace dm;

namespace dmb {

std::vector<Case> model_cases() {
  return {
      {"a4gu", "routed expert gate/up 512x2048 aff4 (store record)", Fmt::AFF4, 512, 2048},
      {"a4dn", "routed expert down 2048x512 aff4 (store record)", Fmt::AFF4, 2048, 512},
      {"a8qkv", "linear-attention qkv 8192x2048 aff8 (Mac resident)", Fmt::AFF8, 8192, 2048},
      {"a8out", "attention out_proj 2048x4096 aff8 (Mac resident)", Fmt::AFF8, 2048, 4096},
      {"a8rt", "router 256x2048 aff8", Fmt::AFF8, 256, 2048},
      {"a8head", "LM head slice 32768x2048 aff8 (head = 248320 rows)", Fmt::AFF8, 32768, 2048},
      {"a4qkv", "linear-attention qkv 8192x2048 aff4 (6 GB phone resident)", Fmt::AFF4, 8192, 2048},
      {"a4out", "attention out_proj 2048x4096 aff4 (6 GB phone resident)", Fmt::AFF4, 2048, 4096},
      {"a4head", "LM head slice 32768x2048 aff4", Fmt::AFF4, 32768, 2048},
  };
}
const Case* find_case(const std::string& id) {
  static std::vector<Case> cs = model_cases();
  for (auto& c : cs)
    if (c.name == id) return &c;
  return nullptr;
}

size_t mat_bytes(const Case& c) {
  size_t code = (size_t)c.rows * c.cols / (c.fmt == Fmt::AFF4 ? 2 : 1);
  size_t sc = (size_t)c.rows * (c.cols / 64) * 2;
  return code + 2 * sc;
}

void MatBuf::make(const Case& c_, size_t total, uint64_t seed) {
  c = c_;
  code_bytes = (size_t)c.rows * c.cols / (c.fmt == Fmt::AFF4 ? 2 : 1);
  scale_bytes = (size_t)c.rows * (c.cols / 64) * 2;
  mat_bytes = code_bytes + 2 * scale_bytes;
  stride = (mat_bytes + 63) & ~size_t(63);
  nmat = (int)std::max<size_t>(1, total / stride);
  buf = (uint8_t*)std::aligned_alloc(16384, ((size_t)nmat * stride + 16383) & ~size_t(16383));
  std::mt19937_64 rng(seed);
  for (size_t i = 0; i < (size_t)nmat * stride / 8; i++) ((uint64_t*)buf)[i] = rng();
  std::normal_distribution<float> nd(0.f, 1.f);
  for (int m = 0; m < nmat; m++) {
    _Float16* s = (_Float16*)(buf + (size_t)m * stride + code_bytes);
    size_t ns = scale_bytes / 2;
    for (size_t i = 0; i < ns; i++) s[i] = (_Float16)(0.002f * nd(rng));        // scales (either sign, as in the store)
    for (size_t i = 0; i < ns; i++) s[ns + i] = (_Float16)(0.01f * nd(rng));    // biases
  }
}
void MatBuf::release() {
  std::free(buf);
  buf = nullptr;
}

struct PassCtx {
  const MatBuf* b;
  const XQ* x;
  float* y;
};
static void pass_task(void* p, int tid, int nt) {
  PassCtx& C = *(PassCtx*)p;
  const MatBuf& b = *C.b;
  int a = (int)((long)b.c.rows * tid / nt), e = (int)((long)b.c.rows * (tid + 1) / nt);
  a &= ~3;
  e = tid == nt - 1 ? b.c.rows : (e & ~3);
  float* y = C.y + (size_t)tid * 32768;
  for (int m = 0; m < b.nmat; m++) {
    if (b.c.fmt == Fmt::AFF4) mv_aff4(b.codes(m), b.scales(m), b.biases(m), b.c.cols, a, e, *C.x, y);
    else mv_aff8(b.codes(m), b.scales(m), b.biases(m), b.c.cols, a, e, *C.x, y);
  }
}

double cpu_pass(Pool& pool, const MatBuf& b, const XQ& x, float* y) {
  PassCtx C{&b, &x, y};
  double t0 = now();
  pool.run(pass_task, &C);
  return now() - t0;
}

struct XHolder {
  std::vector<uint8_t> mem;
  XQ x;
  explicit XHolder(int K, uint64_t seed = 7) {
    mem.resize(xq_bytes(K) + 64);
    void* p = (void*)(((uintptr_t)mem.data() + 63) & ~uintptr_t(63));
    xq_bind(x, p, K, Prec::I16);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> xf(K);
    for (auto& v : xf) v = nd(rng) * (rng() % 50 == 0 ? 20.f : 1.f);
    xq_prep(x, xf.data());
  }
};

void CpuLoad::start(const Case& c, int nt, const std::string& q, size_t mb) {
  threads = nt;
  qos = q;
  b.make(c, mb << 20, 99);
  stop_ = false;
  th = std::thread([this] {
    set_qos(qos);
    g_worker_qos = qos_value(qos);
    Pool pool(threads);
    XHolder xh(b.c.cols);
    std::vector<float> y((size_t)threads * 32768);
    ready = true;
    while (!stop_.load()) {
      cpu_pass(pool, b, xh.x, y.data());
      bytes.fetch_add((int64_t)b.bytes_per_pass());
      passes.fetch_add(1);
    }
  });
  while (!ready.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
}
void CpuLoad::stop() {
  stop_ = true;
  if (th.joinable()) th.join();
  b.release();
  ready = false;
}
CpuLoad::Mark CpuLoad::mark() const { return {now(), bytes.load()}; }

// ------------------------------------------------------------------------------------------------- the sweep cells
struct CellOut {
  double gbps = 0, gmacs = 0, us_per_mat = 0, secs = 0;
  int passes = 0, th0 = 0, th1 = 0;
  std::vector<std::array<double, 3>> series;  // t, GB/s over that second, thermal
};

static CellOut run_cell(const MatBuf& b, int nt, const std::string& qos, double secs, bool series) {
  set_qos(qos);
  g_worker_qos = qos_value(qos);
  CellOut o;
  o.th0 = thermal();
  Pool pool(nt);
  XHolder xh(b.c.cols);
  std::vector<float> y((size_t)nt * 32768);
  cpu_pass(pool, b, xh.x, y.data());  // warm (pages, pool)
  double t0 = now(), tsec = t0, tot = 0;
  int64_t bsec = 0;
  int n = 0;
  while (true) {
    tot += cpu_pass(pool, b, xh.x, y.data());
    n++;
    bsec += (int64_t)b.bytes_per_pass();
    double t = now();
    if (series && t - tsec >= 1.0) {
      o.series.push_back({t - t0, bsec / (t - tsec) / 1e9, (double)thermal()});
      tsec = t;
      bsec = 0;
    }
    if (t - t0 >= secs && n >= 2) break;
  }
  o.secs = tot;
  o.passes = n;
  o.gbps = b.bytes_per_pass() * n / tot / 1e9;
  o.gmacs = b.macs_per_pass() * n / tot / 1e9;
  o.us_per_mat = tot / n / b.nmat * 1e6;
  o.th1 = thermal();
  set_qos("ui");
  return o;
}

static void cell_json(J& j, const CellOut& o) {
  j.kv("gbps", o.gbps).kv("gmacs", o.gmacs).kv("us_per_matrix", o.us_per_mat).kv("passes", o.passes)
      .kv("thermal_start", thermal_name(o.th0)).kv("thermal_end", thermal_name(o.th1));
  if (!o.series.empty()) {
    j.key("series").arr();
    for (auto& s : o.series) j.arr().v(s[0]).v(s[1]).v(thermal_name((int)s[2])).end_arr();
    j.end_arr();
  }
}

// raw read bandwidth (bw_cpu.cpp), NEON loads over a buffer, per thread count
struct BwCtx { const uint8_t* p; size_t n; uint64_t out[64 * 16]; };
static void bw_task(void* v, int tid, int nt) {
  BwCtx& c = *(BwCtx*)v;
  size_t a = c.n * tid / nt & ~size_t(63), b = c.n * (tid + 1) / nt & ~size_t(63);
  uint32x4_t s0 = vdupq_n_u32(0), s1 = s0, s2 = s0, s3 = s0;
  for (size_t i = a; i < b; i += 64) {
    s0 = vaddq_u32(s0, vld1q_u32((const uint32_t*)(c.p + i)));
    s1 = vaddq_u32(s1, vld1q_u32((const uint32_t*)(c.p + i + 16)));
    s2 = vaddq_u32(s2, vld1q_u32((const uint32_t*)(c.p + i + 32)));
    s3 = vaddq_u32(s3, vld1q_u32((const uint32_t*)(c.p + i + 48)));
  }
  c.out[tid * 16] = vaddvq_u32(vaddq_u32(vaddq_u32(s0, s1), vaddq_u32(s2, s3)));
}

// batched kernels (prefill): row blocks of 64 rows as jobs
struct GemmCtx {
  const Case* c;
  const uint8_t* codes;
  const _Float16 *sc, *bi;
  const XQB* x;
  float* Y;
  int N;
  std::atomic<int> next{0};
};
static void gemm_task(void* p, int, int) {
  GemmCtx& g = *(GemmCtx*)p;
  for (int b; (b = g.next.fetch_add(1)) * 64 < g.c->rows;) {
    int a = b * 64, e = std::min(g.c->rows, a + 64);
    if (g.c->fmt == Fmt::AFF4) gemm_aff4_i16(g.codes, g.sc, g.bi, g.c->cols, a, e, *g.x, 0, g.N, g.Y + a, g.c->rows);
    else gemm_aff8_i16(g.codes, g.sc, g.bi, g.c->cols, a, e, *g.x, 0, g.N, g.Y + a, g.c->rows);
  }
}

}  // namespace dmb

using namespace dmb;

extern "C" char* dmb_kernel_bench(const char* spec, dmb_log_fn log) {
  Spec sp(spec);
  start_pressure_source();
  set_qos("ui");
  const double cell = sp.num("cell", 3);
  const std::vector<int> threads = sp.ints("threads", {1, 2, 4, 6});
  std::vector<std::string> qoss;
  {
    std::string q = sp.str("qos", "ui:default");
    size_t p = 0;
    while (p < q.size()) {
      size_t e = q.find(':', p);
      if (e == std::string::npos) e = q.size();
      qoss.push_back(q.substr(p, e - p));
      p = e + 1;
    }
  }
  const size_t mb = (size_t)sp.num("mb", 384);
  const double sustain = sp.num("sustain", 60);
  const int bgthreads = (int)sp.num("bgthreads", 4);
  std::vector<Case> cases;
  if (sp.str("cases", "all") == "all") cases = model_cases();
  else {
    std::string q = sp.str("cases", "");
    size_t p = 0;
    while (p < q.size()) {
      size_t e = q.find(':', p);
      if (e == std::string::npos) e = q.size();
      if (auto* c = find_case(q.substr(p, e - p))) cases.push_back(*c);
      p = e + 1;
    }
  }
  J j;
  j.obj();
  j.key("features").raw(features_json());
  j.kv("spec", std::string(spec ? spec : "")).kv("thermal_at_start", thermal_name(thermal()))
      .kv("footprint_start", footprint()).kv("available_start", available());
  logf(log, "KBENCH start: features %s thermal %s", features_json().c_str(), thermal_name(thermal()));
  double T0 = now();

  // 1. raw read bandwidth
  {
    size_t n = mb << 20;
    uint8_t* p = (uint8_t*)std::aligned_alloc(16384, n);
    std::memset(p, 1, n);
    j.key("bandwidth").arr();
    for (const std::string& q : qoss)
      for (int nt : threads) {
        set_qos(q);
        g_worker_qos = qos_value(q);
        Pool pool(nt);
        BwCtx c{p, n, {}};
        pool.run(bw_task, &c);
        std::vector<double> ts;
        double t0 = now();
        while (now() - t0 < cell || ts.size() < 3) {
          double a = now();
          pool.run(bw_task, &c);
          ts.push_back(now() - a);
        }
        std::sort(ts.begin(), ts.end());
        double g = n / ts[ts.size() / 2] / 1e9;
        j.obj().kv("threads", nt).kv("qos", q).kv("gbps", g).kv("thermal", thermal_name(thermal())).end_obj();
        logf(log, "KBENCH bw qos=%s threads=%d %.1f GB/s", q.c_str(), nt, g);
      }
    j.end_arr();
    set_qos("ui");
    std::free(p);
  }

  // 2. batch-1 kernels, DRAM-fed sweep
  int best_t = threads.back();
  double best_g = 0;
  j.key("matvec").arr();
  for (const Case& c : cases) {
    MatBuf b;
    b.make(c, mb << 20, 1);
    for (const std::string& q : qoss)
      for (int nt : threads) {
        CellOut o = run_cell(b, nt, q, cell, false);
        j.obj().kv("case", c.name).kv("what", c.what).kv("rows", c.rows).kv("cols", c.cols)
            .kv("fmt", c.fmt == Fmt::AFF4 ? "aff4_g64" : "aff8_g64").kv("threads", nt).kv("qos", q)
            .kv("matrix_bytes", (int64_t)b.mat_bytes).kv("nmat", b.nmat);
        cell_json(j, o);
        j.end_obj();
        logf(log, "KBENCH mv %s qos=%s t=%d %.1f GB/s %.1f GMAC/s %.1f us/mat [%s]", c.name.c_str(), q.c_str(), nt, o.gbps,
             o.gmacs, o.us_per_mat, thermal_name(o.th1));
        if (c.name == "a4gu" && q == "ui" && o.gbps > best_g) best_g = o.gbps, best_t = nt;
      }
    if (bgthreads > 0 && (c.name == "a4gu" || c.name == "a8qkv")) {
      CellOut o = run_cell(b, bgthreads, "background", cell, false);
      j.obj().kv("case", c.name).kv("what", c.what).kv("rows", c.rows).kv("cols", c.cols)
          .kv("fmt", c.fmt == Fmt::AFF4 ? "aff4_g64" : "aff8_g64").kv("threads", bgthreads).kv("qos", "background")
          .kv("matrix_bytes", (int64_t)b.mat_bytes).kv("nmat", b.nmat);
      cell_json(j, o);
      j.end_obj();
      logf(log, "KBENCH mv %s qos=background t=%d %.1f GB/s", c.name.c_str(), bgthreads, o.gbps);
    }
    b.release();
  }
  j.end_arr();
  j.kv("best_threads_experts", best_t);

  // 3. batched kernels (compute-bound: one matrix, stays in the caches as a prefill chunk's would per row block)
  if (sp.num("batch", 1) > 0) {
    j.key("batched").arr();
    struct BC { const char* id; std::vector<int> Ns; };
    std::vector<BC> bcs = {{"a4gu", {8, 64, 256}}, {"a4qkv", {64, 256}}, {"a8qkv", {64, 256}}};
    std::vector<int> bth = {threads.size() > 1 ? 2 : threads[0], threads.back()};
    bth.erase(std::unique(bth.begin(), bth.end()), bth.end());
    for (auto& bc : bcs) {
      const Case& c = *find_case(bc.id);
      MatBuf b;
      b.make(c, 1, 3);
      for (int N : bc.Ns) {
        XQB xb;
        xb.reserve(c.cols, N);
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> X((size_t)N * c.cols);
        for (auto& v : X) v = nd(rng) * (rng() % 50 == 0 ? 20.f : 1.f);
        for (int i = 0; i < N; i++) xqb_set(xb, i, X.data() + (size_t)i * c.cols);
        xqb_pad(xb);
        std::vector<float> Y((size_t)xb.NP * c.rows);
        // exactness: position 0 through the batch kernel vs the batch-1 kernel
        size_t bad = 0;
        {
          GemmCtx g{&c, b.codes(0), b.scales(0), b.biases(0), &xb, Y.data(), N};
          gemm_task(&g, 0, 1);
          XHolder xh(c.cols);
          xq_prep(xh.x, X.data());
          std::vector<float> y1(c.rows);
          if (c.fmt == Fmt::AFF4) mv_aff4(b.codes(0), b.scales(0), b.biases(0), c.cols, 0, c.rows, xh.x, y1.data());
          else mv_aff8(b.codes(0), b.scales(0), b.biases(0), c.cols, 0, c.rows, xh.x, y1.data());
          for (int r = 0; r < c.rows; r++) bad += std::memcmp(&y1[r], &Y[r], 4) != 0;
        }
        for (int nt : bth) {
          set_qos("ui");
          g_worker_qos = qos_value("ui");
          Pool pool(nt);
          std::vector<double> ts;
          double t0 = now();
          while (now() - t0 < cell || ts.size() < 3) {
            GemmCtx g{&c, b.codes(0), b.scales(0), b.biases(0), &xb, Y.data(), N};
            double a = now();
            pool.run(gemm_task, &g);
            ts.push_back(now() - a);
          }
          std::sort(ts.begin(), ts.end());
          double t = ts[ts.size() / 2], gm = (double)c.rows * c.cols * N / t / 1e9;
          j.obj().kv("case", c.name).kv("N", N).kv("threads", nt).kv("gmacs", gm).kv("ms", t * 1e3)
              .kv("bit_mismatch_vs_mv_pos0", (int64_t)bad).kv("thermal", thermal_name(thermal())).end_obj();
          logf(log, "KBENCH gemm %s N=%d t=%d %.1f GMAC/s (pos0 mismatches %zu)", c.name.c_str(), N, nt, gm, bad);
        }
      }
      b.release();
    }
    j.end_arr();
  }

  // 4. sustained CPU: the experts' kernel at the best thread count, per-second rate and thermal state
  if (sustain > 0) {
    MatBuf b;
    b.make(*find_case("a4gu"), mb << 20, 1);
    CellOut o = run_cell(b, best_t, "ui", sustain, true);
    j.key("sustained_cpu").obj().kv("case", "a4gu").kv("threads", best_t).kv("qos", "ui").kv("seconds", sustain);
    cell_json(j, o);
    j.end_obj();
    logf(log, "KBENCH sustained cpu a4gu t=%d %.0f s: %.1f GB/s mean, first %.1f last %.1f [%s]", best_t, sustain, o.gbps,
         o.series.empty() ? 0 : o.series.front()[1], o.series.empty() ? 0 : o.series.back()[1], thermal_name(o.th1));
    b.release();
  }

  // 5. Metal
  if (sp.num("metal", 1) > 0) {
    Spec sp2 = sp;
    sp2.kv["cputhreads"] = std::to_string(best_t);
    metal_bench(j, sp2, log);
  }
  j.kv("seconds_total", now() - T0).kv("thermal_at_end", thermal_name(thermal()))
      .kv("memory_warnings", (int)g_mem_warnings.load()).kv("pressure_events", pressure_events());
  j.end_obj();
  logf(log, "KBENCH done in %.0f s, thermal %s", now() - T0, thermal_name(thermal()));
  return dup(j.s);
}
