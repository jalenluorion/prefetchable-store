// The expert cache's size from the machine, for the serving worker (tools/dmserve.cpp --cache-gb auto).
//
// At start: the ceiling = the largest locked cache that leaves `reserve` GB of the machine's memory to the rest of the system:
//     ceiling = physical memory - reserve - the process's other memory (measured after the resident weights are loaded,
//               plus KV for `kv_positions`, checkpoints, prefill scratch and a margin), clamped to [min, the store's size]
// Slots for the ceiling are reserved (virtual); only the start size is locked:
//     start = min(ceiling, memory free now - the process's other memory still to come - headroom)
// so a machine whose other programs use more than the reserve does not go into pressure at start.
// Then a watcher thread follows what macOS reports (kern.memorystatus_vm_pressure_level and kern.memorystatus_level,
// checked every `period_s`, and woken at once by the system's memory-pressure notifications): under pressure (warn or
// critical, or the free level below `low_pct`) the budget shrinks by `step` GB per check down to `min`; when the
// pressure is normal and the free level has stayed at or above `high_pct` for `grow_after_s`, it grows by `step` GB per
// check up to the ceiling, never back into a size that met pressure until `retry_s` of calm. Only the budget moves (Cache::set_budget, applied by
// the compute thread at a token or prefill-layer boundary, or by the worker's idle loop): never what is computed.
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <thread>

#include "dmeng/core.h"

namespace dm {

struct MemRule {
  double reserve_gb = 5.5;    // memory left to the rest of the system (kernel, apps)
  double min_gb = 2.0;        // the least cache the watcher shrinks to
  double margin_gb = 0.3;     // the process's other memory: misc. margin
  double scratch_gb = 0.5;    // prefill scratch while a prefill runs
  double headroom_gb = 3.0;   // the start also leaves this much of what is free now: macOS reports warn pressure / a free
                              // level under 20% with ~2.5-3.4 GB left on a 16 GB Mac (measured), so the start does not press
  int kv_positions = 16384;   // KV planned for (pos_bytes each)
};

struct MemSnap {
  double total_gb = 0, avail_gb = 0;  // physical; free + inactive + speculative + purgeable
  int level_pct = -1, pressure = -1;  // kern.memorystatus_level (free %), kern.memorystatus_vm_pressure_level (1/2/4)
  double footprint_gb = 0;            // this process
  std::string json() const;
};
MemSnap mem_snapshot();

// the ceiling by the rule above (slots reserved), and the start (locked now): the ceiling, or what is free now minus the
// process's memory still to come and the headroom, if less; why: the numbers that went into it (JSON)
double auto_cache_gb(const MemRule& r, size_t pos_bytes, double ckpt_gb, double store_gb, double& start_gb, std::string& why);
double store_bytes_gb(const std::string& model_dir, const std::string& layer_file, int layers);

class MemWatch {
 public:
  MemWatch(Cache& c, double start_gb, double max_gb, double min_gb)
      : c_(c), max_b_((int64_t)(max_gb * 1e9)), min_b_((int64_t)(min_gb * 1e9)), start_b_((int64_t)(start_gb * 1e9)) {}
  ~MemWatch() { stop(); }
  int low_pct = 20, high_pct = 35;
  double step_gb = 1.0, period_s = 2.0, grow_after_s = 30.0;
  double retry_s = 600;  // a size that met pressure is not grown into again for this long (calm)
  std::function<void()> wake;                    // a budget was requested (the worker applies it when idle)
  std::function<void(const std::string&)> note;  // one JSON line per change
  void start();
  void stop();
  int64_t target() const { return target_.load(); }
  double floor_gb() const { return min_b_ / 1e9; }  // after start(): the least budget the watcher sets

 private:
  Cache& c_;
  int64_t max_b_, min_b_, start_b_;
  std::atomic<int64_t> target_{0};
  std::atomic<bool> quit_{false}, kick_{false};
  std::thread th_;
  void* src_ = nullptr;  // dispatch source (memory pressure)
  void loop();
};

}  // namespace dm
