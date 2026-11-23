// Copied from moved-store (branch port-ios, base unified-wip bdeb350) src/core for the P0 phone bench; one change: g_worker_qos.
#include "dmeng/pool.h"

#include <pthread.h>
#include <pthread/qos.h>

#include <chrono>

namespace dm {

int g_worker_qos = QOS_CLASS_USER_INTERACTIVE;

void set_qos_interactive() { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }

static inline void cpu_relax() { __builtin_arm_yield(); }

Pool::Pool(int nthreads, int spin_us) : n_(nthreads < 1 ? 1 : nthreads), spin_us_(spin_us) {
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
  done_.store(0, std::memory_order_relaxed);
  gen_.fetch_add(1, std::memory_order_seq_cst);  // bump first, then look for sleepers (see worker())
  if (sleeping_.load(std::memory_order_seq_cst) > 0) {
    std::lock_guard<std::mutex> lk(mu_);
    cv_.notify_all();
  }
  fn(ctx, 0, n_);
  while (done_.load(std::memory_order_acquire) != n_ - 1) cpu_relax();
}

void Pool::worker(int tid) {
  pthread_set_qos_class_self_np((qos_class_t)g_worker_qos, 0);
  uint64_t seen = 0;  // the generation at construction: a run() issued before this thread starts is not missed
  for (;;) {
    auto t0 = std::chrono::steady_clock::now();
    uint64_t g;
    int iter = 0;
    while ((g = gen_.load(std::memory_order_acquire)) == seen) {
      cpu_relax();
      if (++iter == 4096) {
        iter = 0;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::microseconds(spin_us_)) {
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
    fn_(ctx_, tid, n_);
    done_.fetch_add(1, std::memory_order_acq_rel);
  }
}

}  // namespace dm
