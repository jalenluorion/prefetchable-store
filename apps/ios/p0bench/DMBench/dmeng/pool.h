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
  explicit Pool(int nthreads, int spin_us = 2000);
  ~Pool();
  int size() const { return n_; }
  // Runs fn(ctx, tid, n) on all n threads (tid 0 = the caller) and returns when every thread is done.
  void run(TaskFn fn, void* ctx);

 private:
  void worker(int tid);
  int n_, spin_us_;
  std::vector<std::thread> th_;
  alignas(128) std::atomic<uint64_t> gen_{0};
  alignas(128) std::atomic<int> done_{0};
  alignas(128) std::atomic<int> sleeping_{0};
  TaskFn fn_ = nullptr;
  void* ctx_ = nullptr;
  std::atomic<bool> stop_{false};
  std::mutex mu_;
  std::condition_variable cv_;
};

void set_qos_interactive();
// P0 bench addition (not in the engine): the QoS class new workers take (default user-interactive, as the engine)
extern int g_worker_qos;

}  // namespace dm
