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
                                         "moe_combine", "head", "span_copies", "prefetch"};
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


#if DM_POOLTRACE
// The join trace (cmake -DDM_POOLTRACE=ON, a diagnostic build): per phase function, where a run's time goes. Per run:
// the dispatch (t0), each thread's entry and exit of the phase function and its cycles there, the caller's own exit,
// the join's end. Summed per function and printed at exit as "DMJOIN" lines: runs, mean run time, the join's wait after
// the median thread's exit and after the caller's exit, the latest entry after dispatch, how often a thread entered
// after every other had left, and per thread its busy time and clock (cycles / busy time: P-core about 4 GHz, E-core
// below 3).
}  // namespace dm
#include <chrono>
#include <cstdio>
#include <dlfcn.h>
#include <algorithm>
extern "C" int thread_selfcounts(int type, void* buf, size_t nbytes);
namespace dm {
static inline int64_t pt_now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct PtRun { int64_t t0 = 0, cend = 0, jend = 0, ent[32] = {0}, ex[32] = {0}; uint64_t cyc[32] = {0}; };
static PtRun g_pt;
struct PtFn {
  void* fn = nullptr;
  int stage = 0;  // the counting build's stage at dispatch (0 without DM_COUNTERS)
  double runs = 0, t_run = 0, w_med = 0, w_caller = 0, late_ent = 0, late_after_all = 0, skipped = 0, busy[32] = {0}, cyc[32] = {0}, gt3[32] = {0};
};
static PtFn g_ptf[256];
static int g_npt = 0;
static int g_ptn = 0;
static void pt_close(void* fn, int n, int stage) {
  int k = 0;
  while (k < g_npt && (g_ptf[k].fn != fn || g_ptf[k].stage != stage)) k++;
  if (k == g_npt) {
    if (g_npt == 256) return;
    g_ptf[g_npt].fn = fn;
    g_ptf[g_npt++].stage = stage;
  }
  PtFn& F = g_ptf[k];
  PtRun& R = g_pt;
  int64_t ex[32];
  int64_t maxent = 0, maxex = 0;
  int ne = 0;
  for (int t = 0; t < n; t++) {
    if (!R.ent[t]) continue;  // skipped the run (run_any)
    ex[ne++] = R.ex[t];
    maxent = std::max(maxent, R.ent[t] - R.t0);
    maxex = std::max(maxex, R.ex[t]);
  }
  std::sort(ex, ex + ne);
  F.skipped += n - ne;
  F.runs++;
  F.t_run += R.jend - R.t0;
  F.w_med += maxex - ex[ne / 2];
  F.w_caller += R.jend - R.cend;
  F.late_ent += maxent;
  for (int t = 1; t < n; t++) {  // a worker that entered after every other thread had left
    int64_t others = 0;
    for (int u = 0; u < n; u++) if (u != t) others = std::max(others, R.ex[u]);
    if (R.ent[t] && R.ent[t] > others) F.late_after_all++;
  }
  for (int t = 0; t < n; t++) {
    const double b = (double)(R.ex[t] - R.ent[t]);
    F.busy[t] += b;
    F.cyc[t] += (double)R.cyc[t];
    if (b > 0 && R.cyc[t] / b > 3.3) F.gt3[t]++;
  }
}
struct PtReport {
  ~PtReport() {
    for (int k = 0; k < g_npt; k++) {
      const PtFn& F = g_ptf[k];
      Dl_info d;
      const bool ok = dladdr(F.fn, &d);
      fprintf(stderr, "DMJOIN %s %p %p stage %d runs %.0f run_us %.2f wait_after_median_us %.2f wait_after_caller_us %.2f latest_entry_us %.2f late_after_all %.0f skipped_per_run %.2f |", ok ? d.dli_fname : "?",
              ok ? d.dli_fbase : nullptr, F.fn, F.stage, F.runs, F.t_run / F.runs / 1e3, F.w_med / F.runs / 1e3, F.w_caller / F.runs / 1e3, F.late_ent / F.runs / 1e3, F.late_after_all, F.skipped / F.runs);
      for (int t = 0; t < g_ptn; t++) fprintf(stderr, " t%d busy_us %.2f ghz %.2f pcore %.2f;", t, F.busy[t] / F.runs / 1e3, F.busy[t] > 0 ? F.cyc[t] / F.busy[t] : 0, F.gt3[t] / F.runs);
      fprintf(stderr, "\n");
    }
  }
} g_pt_report;
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

void Pool::run(TaskFn fn, void* ctx) { dispatch(fn, ctx, false); }
void Pool::run_any(TaskFn fn, void* ctx) { dispatch(fn, ctx, true); }

// A run's state word: the low 32 bits of its generation (tag), the kind (ALL: every thread runs fn; else claimed work),
// CLOSED (an any-run whose caller has returned from fn: late workers skip it) and the count of workers inside fn.
static constexpr uint64_t ST_ALL = 1ull << 31, ST_CLOSED = 1ull << 30, ST_COUNT = (1ull << 30) - 1;

void Pool::dispatch(TaskFn fn, void* ctx, bool any) {
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
#if DM_POOLTRACE
  g_ptn = n_;
  for (int t = 0; t < n_; t++) g_pt.ent[t] = g_pt.ex[t] = 0, g_pt.cyc[t] = 0;  // a worker that skips the run stays 0
  g_pt.t0 = pt_now();
#endif
  done_.store(0, std::memory_order_relaxed);
  park_.store(false, std::memory_order_relaxed);
  const uint64_t g = gen_.load(std::memory_order_relaxed) + 1;
  st_.store(((g & 0xFFFFFFFFull) << 32) | (any ? 0 : ST_ALL), std::memory_order_release);  // before the generation
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
#if DM_POOLTRACE
  uint64_t pc0[2], pc1[2];
  thread_selfcounts(1, pc0, sizeof pc0);
  g_pt.ent[0] = pt_now();
#endif
  fn(ctx, 0, n_);
#if DM_POOLTRACE
  g_pt.ex[0] = g_pt.cend = pt_now();
  thread_selfcounts(1, pc1, sizeof pc1);
  g_pt.cyc[0] = pc1[1] - pc0[1];
#endif
#if DM_COUNTERS
  dmc_now(b);
#endif
  if (any) {  // close the run, then wait for the workers inside fn; any later one skips it
    st_.fetch_or(ST_CLOSED, std::memory_order_acq_rel);
    while (st_.load(std::memory_order_acquire) & ST_COUNT) cpu_relax();
  } else {
    while (done_.load(std::memory_order_acquire) != n_ - 1) cpu_relax();
  }
#if DM_POOLTRACE
  g_pt.jend = pt_now();
#if DM_COUNTERS
  pt_close((void*)fn, n_, g_run_stage);
#else
  pt_close((void*)fn, n_, 0);
#endif
#endif
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
    // this generation's run: an every-thread run (fn, then done_), or an any-run entered only while it is open (a
    // tagged compare-and-swap, so a late worker never touches the context of a run its caller has left); a tag that
    // is not g's: the caller has moved on past g (an any-run this worker missed), nothing to do
    bool all = false;
    {
      uint64_t st = st_.load(std::memory_order_acquire);
      bool in = false;
      for (;;) {
        if ((st >> 32) != (g & 0xFFFFFFFFull) || (!(st & ST_ALL) && (st & ST_CLOSED))) break;
        if (st & ST_ALL) {
          in = all = true;
          break;
        }
        if (st_.compare_exchange_weak(st, st + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
          in = true;
          break;
        }
      }
      if (!in) continue;
    }
#if DM_COUNTERS
    uint64_t a[2], b[2];
    dmc_now(a);
#endif
#if DM_POOLTRACE
    uint64_t pc0[2], pc1[2];
    thread_selfcounts(1, pc0, sizeof pc0);
    g_pt.ent[tid] = pt_now();
#endif
    fn_(ctx_, tid, n_);
#if DM_POOLTRACE
    g_pt.ex[tid] = pt_now();
    thread_selfcounts(1, pc1, sizeof pc1);
    g_pt.cyc[tid] = pc1[1] - pc0[1];
#endif
#if DM_COUNTERS
    dmc_now(b);
    g_work_i += b[0] - a[0]; g_work_c += b[1] - a[1];
    g_fn[g_cur].i += b[0] - a[0]; g_fn[g_cur].c += b[1] - a[1];
    g_st_w_i[g_run_stage] += b[0] - a[0]; g_st_w_c[g_run_stage] += b[1] - a[1];
#endif
    if (all) done_.fetch_add(1, std::memory_order_acq_rel);
    else st_.fetch_sub(1, std::memory_order_release);
  }
}

}  // namespace dm
