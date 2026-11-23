#include "dmeng/pool.h"

#include <pthread.h>
#include <pthread/qos.h>

#include <chrono>
#include <cstdlib>
#if DM_COUNTERS
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#endif

namespace dm {

void set_qos_interactive() { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }

static inline void cpu_relax() { __builtin_arm_yield(); }

#if DM_COUNTERS
// The counting build (cmake -DDM_COUNTERS=ON): per-thread instructions / cycles of the pool's work, the main thread's
// serial work and its spin-waits, printed to stderr at exit when DM_COUNTS is set. One line
//   DMCOUNTS {"work_ins", "work_cyc", "main_work_ins", "main_serial_ins", "main_serial_cyc", "spin_ins", "spin_cyc", "runs"}
// and per phase function "DMFN <image> <base> <addr> <ins> <cyc> <runs>". Useful work = work + main serial.
extern "C" int thread_selfcounts(int type, void* buf, size_t nbytes);
extern "C" void dmc_now(uint64_t* a) { thread_selfcounts(1, a, 2 * sizeof(uint64_t)); }
static std::atomic<uint64_t> g_work_i{0}, g_work_c{0}, g_spin_i{0}, g_spin_c{0}, g_mwork_i{0}, g_mwork_c{0}, g_runs{0};
static uint64_t g_m0[2] = {0, 0}, g_ml[2] = {0, 0};
static bool g_started = false;
struct FnCnt { void* fn = nullptr; std::atomic<uint64_t> i{0}, c{0}, n{0}; };
static FnCnt g_fn[128];
static int g_nfn = 0, g_cur = 0;
static int fn_index(void* fn) {
  for (int k = 0; k < g_nfn; k++) if (g_fn[k].fn == fn) return k;
  if (g_nfn == 128) return 127;
  g_fn[g_nfn].fn = fn;
  return g_nfn++;
}
// per stage (stage_mark): the main thread's total between marks, its share of the pool work and its spin-waits, and the
// pool work of every thread for runs dispatched in that stage
static int g_stage = ST_OTHER, g_run_stage = ST_OTHER;
static uint64_t g_st_last[2] = {0, 0};
static bool g_st_on = false;
static uint64_t g_st_main_i[ST_N], g_st_main_c[ST_N], g_st_mw_i[ST_N], g_st_mw_c[ST_N], g_st_sp_i[ST_N], g_st_sp_c[ST_N];
static std::atomic<uint64_t> g_st_w_i[ST_N], g_st_w_c[ST_N];
static const char* const kStage[ST_N] = {"other", "embed", "depth_attn", "msg_gates", "norms_quant", "attn_proj", "kv_prep",
                                         "attn_core", "attn_out", "convs", "moe_router", "moe_cache", "moe_experts",
                                         "moe_combine", "head", "span_copies"};
void stage_mark(int s) {
  uint64_t x[2];
  dmc_now(x);
  if (g_st_on) {
    g_st_main_i[g_stage] += x[0] - g_st_last[0];
    g_st_main_c[g_stage] += x[1] - g_st_last[1];
  }
  g_st_on = true;
  g_st_last[0] = x[0];
  g_st_last[1] = x[1];
  g_stage = s;
}
extern "C" void dmc_spin(const uint64_t* a, const uint64_t* b) {
  g_spin_i += b[0] - a[0]; g_spin_c += b[1] - a[1];
  g_st_sp_i[g_stage] += b[0] - a[0]; g_st_sp_c[g_stage] += b[1] - a[1];  // main-thread waits only (run(), expert reads)
}
struct DmcReport {
  ~DmcReport() {
    if (!getenv("DM_COUNTS")) return;
    const double mi = (double)(g_ml[0] - g_m0[0]), mc = (double)(g_ml[1] - g_m0[1]);
    fprintf(stderr, "DMCOUNTS {\"work_ins\": %llu, \"work_cyc\": %llu, \"main_work_ins\": %llu, \"main_serial_ins\": %.0f, \"main_serial_cyc\": %.0f, \"spin_ins\": %llu, \"spin_cyc\": %llu, \"runs\": %llu}\n",
            (unsigned long long)g_work_i.load(), (unsigned long long)g_work_c.load(), (unsigned long long)g_mwork_i.load(),
            mi - (double)g_mwork_i.load() - (double)g_spin_i.load(), mc - (double)g_mwork_c.load() - (double)g_spin_c.load(),
            (unsigned long long)g_spin_i.load(), (unsigned long long)g_spin_c.load(), (unsigned long long)g_runs.load());
    // per stage: serial = the main thread outside pool runs and waits; pool = all threads' work in the stage's runs
    for (int k = 1; k < ST_N; k++)
      fprintf(stderr, "DMSTAGE %s %.0f %.0f %llu %llu\n", kStage[k],
              (double)g_st_main_i[k] - (double)g_st_mw_i[k] - (double)g_st_sp_i[k], (double)g_st_main_c[k] - (double)g_st_mw_c[k] - (double)g_st_sp_c[k],
              (unsigned long long)g_st_w_i[k].load(), (unsigned long long)g_st_w_c[k].load());
    for (int k = 0; k < g_nfn; k++) {
      Dl_info d;
      const bool ok = dladdr(g_fn[k].fn, &d);
      fprintf(stderr, "DMFN %s %p %p %llu %llu %llu\n", ok ? d.dli_fname : "?", ok ? d.dli_fbase : nullptr, g_fn[k].fn,
              (unsigned long long)g_fn[k].i.load(), (unsigned long long)g_fn[k].c.load(), (unsigned long long)g_fn[k].n.load());
    }
  }
} g_dmc_report;
#endif

static int spin_default() {
  const char* e = getenv("DMENG_SPIN_US");
  return e ? atoi(e) : 2000;
}

Pool::Pool(int nthreads, int spin_us, int qos)
    : n_(nthreads < 1 ? 1 : nthreads), spin_us_(spin_us < 0 ? spin_default() : spin_us), qos_(qos) {
  for (int t = 1; t < n_; t++) th_.emplace_back([this, t] { worker(t); });
}

Pool::~Pool() {
  stop_.store(true);
  {
    std::lock_guard<std::mutex> lk(mu_);
    gen_.fetch_add(1);
  }
  cv_.notify_all();
  for (auto& t : th_) t.join();
}

void Pool::run(TaskFn fn, void* ctx) {
  if (n_ == 1) {
    fn(ctx, 0, 1);
    return;
  }
  fn_ = fn;
  ctx_ = ctx;
#if DM_COUNTERS
  g_cur = fn_index((void*)fn);
  g_fn[g_cur].n++;
  g_run_stage = g_stage;
#endif
  done_.store(0, std::memory_order_relaxed);
  park_.store(false, std::memory_order_relaxed);
  gen_.fetch_add(1, std::memory_order_seq_cst);  // bump first, then look for sleepers (see worker())
  if (sleeping_.load(std::memory_order_seq_cst) > 0) {
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();
  }
#if DM_COUNTERS
  uint64_t a[2], b[2], c[2];
  dmc_now(a);
  if (!g_started) { g_started = true; g_m0[0] = a[0]; g_m0[1] = a[1]; }
#endif
  fn(ctx, 0, n_);
#if DM_COUNTERS
  dmc_now(b);
#endif
  while (done_.load(std::memory_order_acquire) != n_ - 1) cpu_relax();
#if DM_COUNTERS
  dmc_now(c);
  g_work_i += b[0] - a[0]; g_work_c += b[1] - a[1]; g_mwork_i += b[0] - a[0]; g_mwork_c += b[1] - a[1];
  g_fn[g_cur].i += b[0] - a[0]; g_fn[g_cur].c += b[1] - a[1];
  g_st_w_i[g_run_stage] += b[0] - a[0]; g_st_w_c[g_run_stage] += b[1] - a[1];
  g_st_mw_i[g_run_stage] += b[0] - a[0]; g_st_mw_c[g_run_stage] += b[1] - a[1];
  dmc_spin(b, c);
  g_ml[0] = c[0]; g_ml[1] = c[1];
  g_runs++;
#endif
}

void Pool::worker(int tid) {
  if (qos_ >= 0) pthread_set_qos_class_self_np((qos_class_t)qos_, 0);
  else set_qos_interactive();
  uint64_t seen = 0;  // the generation at construction: a run() issued before this thread starts is not missed
  for (;;) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t g;
    int iter = 0;
    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
      cpu_relax();
      if (++iter == 4096 || park_.load(std::memory_order_relaxed)) {
        iter = 0;
        if (park_.load(std::memory_order_relaxed) || std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
          std::unique_lock<std::mutex> lk(mu_);
          sleeping_.fetch_add(1, std::memory_order_seq_cst);
          cv_.wait(lk, [&] { return gen_.load(std::memory_order_seq_cst) != seen; });
          sleeping_.fetch_sub(1);
          t0 = std::chrono::steady_clock::now();
        }
      }
    }
    seen = g;
    if (stop_.load()) return;
#if DM_COUNTERS
    uint64_t a[2], b[2];
    dmc_now(a);
#endif
    fn_(ctx_, tid, n_);
#if DM_COUNTERS
    dmc_now(b);
    g_work_i += b[0] - a[0]; g_work_c += b[1] - a[1];
    g_fn[g_cur].i += b[0] - a[0]; g_fn[g_cur].c += b[1] - a[1];
    g_st_w_i[g_run_stage] += b[0] - a[0]; g_st_w_c[g_run_stage] += b[1] - a[1];
#endif
    done_.fetch_add(1, std::memory_order_acq_rel);
  }
}

}  // namespace dm
