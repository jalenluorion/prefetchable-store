// A small fork-join pool for one-token decode: the caller is thread 0, workers spin on a generation counter so a
// dispatch costs about a microsecond, and park on a condition variable after `spin_us` without work.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace dm {

using TaskFn = void (*)(void* ctx, int tid, int nthreads);

class Pool {
 public:
  // qos: QoS class (qos_class_t) of the workers; -1 = user-interactive. The caller (thread 0) keeps its own.
  // spin_us < 0: DMENG_SPIN_US from the environment, else 2000
  explicit Pool(int nthreads, int spin_us = -1, int qos = -1);
  ~Pool();
  int size() const { return n_; }
  // Runs fn(ctx, tid, n) on all n threads (tid 0 = the caller) and returns when every thread is done.
  void run(TaskFn fn, void* ctx);
  // For fn whose work is claimed from shared counters (any thread may do any part, a thread may do none, and a part
  // only ever waits for parts claimed before it): the caller runs fn too and returns once it has left fn and every
  // worker that entered has left; a worker that arrives after that skips the run. A slow, sleeping or descheduled
  // worker costs only the part it claimed, never a wait at the join. Which thread computes a part changes, not how.
  void run_any(TaskFn fn, void* ctx);
  // Idle workers stop spinning and sleep until the next run() (the caller is about to wait on something slow: an
  // expert read). The next run() wakes them (a condition-variable notify instead of a spin hand-off).
  void park() { park_.store(true, std::memory_order_relaxed); }

 private:
  void worker(int tid);
  void dispatch(TaskFn fn, void* ctx, bool any);
  int n_, spin_us_, qos_;
  std::vector<std::thread> th_;
  alignas(128) std::atomic<uint64_t> gen_{0};
  alignas(128) std::atomic<int> done_{0};
  alignas(128) std::atomic<uint64_t> st_{0};  // the current run's tag, kind, closed bit and workers inside (pool.cpp)
  alignas(128) std::atomic<int> sleeping_{0};
  TaskFn fn_ = nullptr;
  void* ctx_ = nullptr;
  std::atomic<bool> stop_{false};
  std::atomic<bool> park_{false};
  std::mutex mu_;
  std::condition_variable cv_;
};

void set_qos_interactive();

// f(i) for i in [0, n), the indices claimed one by one by the pool's threads (each once, on any thread): for loops whose
// iterations are independent (every position's own arithmetic), so the result is the serial loop's.
template <class F>
void parallel_for(Pool& pool, int n, const F& f) {
  if (n <= 0) return;
  if (n == 1 || pool.size() == 1) {
    for (int i = 0; i < n; i++) f(i);
    return;
  }
  struct Ctx {
    const F* f;
    int n;
    alignas(128) std::atomic<int> next{0};
  } c{&f, n};
  pool.run_any(
      [](void* p, int, int) {
        Ctx& c = *(Ctx*)p;
        for (int i; (i = c.next.fetch_add(1, std::memory_order_relaxed)) < c.n;) (*c.f)(i);
      },
      &c);
}

// Stages for the counting build (DM_COUNTERS): the main thread marks which stage runs; its serial instructions / cycles
// and the pool work it dispatches are summed per stage and printed at exit ("DMSTAGE ..."). A no-op in any other build.
enum Stage {
  ST_OTHER, ST_EMBED, ST_DEPTH, ST_GATES, ST_NORMQ, ST_ATTN_PROJ, ST_KV_PREP, ST_ATTN_CORE, ST_ATTN_OUT, ST_CONV,
  ST_MOE_ROUTER, ST_MOE_CACHE, ST_MOE_EXPERTS, ST_MOE_COMBINE, ST_HEAD, ST_SPAN, ST_PREFETCH, ST_N
};
#if DM_COUNTERS
void stage_mark(int s);
#else
inline void stage_mark(int) {}
#endif

}  // namespace dm
