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
  // The start, one of two rules (the manifest's profile "cache_start"; Preset::cache_start, set by set_start_rule):
  //  headroom (built-in): what is free now (the resident weights already counted in it) less the process's memory still
  //    to come and headroom_gb: the rule the 35B's and Inkling's published caches come from (6.6 and 7.8 GB on a quiet
  //    16 GB Mac; macOS reports warn pressure / a free level under 20% with ~2.5-3.4 GB left on a 16 GB Mac, measured);
  //  free_level: the size that leaves the kernel's free level (kern.memorystatus_level) at `floor_pct` once the cache and
  //    the process's memory still to come are resident: (level now - floor_pct) % of physical memory - still to come.
  //    Measured on the base M4 16 GB with Flash-Next only (9.8 GB in all at a 3.5 GB cache): from a 42-44% level at
  //    start, 3.5 GB ran with no swap growth and 4.5 GB grew swap by 130-400 MB with compute 13-55% slower (the level
  //    fell to ~9%, as this rule predicts: 8.5%); the 14% floor gives 3.5 GB from a 43% start level (0.62 GB of routers
  //    and 0.81 GB of scratch, margin and KV still to come). Flash-Next's manifest declares it.
  // No single rule reproduced all three models' measured sizes (engine/CONSISTENCY-AUDIT item 1; FINISH.md, the
  // integration record): the rule is a declared per-model value until the two smaller models are measured under it.
  std::string start_rule = "headroom";
  double headroom_gb = 3.0;
  int floor_pct = 14;
  double later_gb = 0;        // memory the front end loads after sizing the cache (the router files), counted as to come
  // never a size the OS will not lock: the wired memory the system allows (vm.global_user_wire_limit) minus what is
  // wired now (kernel, every process, this one's resident weights) minus this margin (Flash-Next's 5.75 GB failed mlock)
  double wire_margin_gb = 0.5;
  int kv_positions = 16384;   // per-position state planned for at the start (pos_bytes each); the state's memory is
                              // reserved virtually for the whole context and becomes resident as positions fill
};

struct MemSnap {
  double total_gb = 0, avail_gb = 0;  // physical; free + inactive + speculative + purgeable
  int level_pct = -1, pressure = -1;  // kern.memorystatus_level (free %), kern.memorystatus_vm_pressure_level (1/2/4)
  double wired_gb = 0, wire_limit_gb = 0;  // the system's wired memory now; vm.global_user_wire_limit
  double footprint_gb = 0;            // this process
  std::string json() const;
};
MemSnap mem_snapshot();

// the ceiling by the rule above (slots reserved), and the start (locked now): the ceiling, or what is free now minus the
// process's memory still to come and the headroom, if less; why: the numbers that went into it (JSON)
double auto_cache_gb(const MemRule& r, size_t pos_bytes, double ckpt_gb, double store_gb, double& start_gb, std::string& why);
// the start rule from its resolved spelling (Preset::cache_start: "headroom:G" | "free_level:P"); false: not a rule
bool set_start_rule(MemRule& r, const std::string& spec, std::string& err);
// the memory watcher's shrink level for the start rule: below it the cache shrinks (headroom: 20, the level macOS warns
// at with that rule's 2.5-3.4 GB left; free_level: the floor less 2)
int watch_low_pct(const MemRule& r);
double store_bytes_gb(const std::string& model_dir, const std::string& layer_file, int layers);

class MemWatch {
 public:
  MemWatch(Cache& c, double start_gb, double max_gb, double min_gb)
      : c_(c), max_b_((int64_t)(max_gb * 1e9)), min_b_((int64_t)(min_gb * 1e9)), start_b_((int64_t)(start_gb * 1e9)) {}
  ~MemWatch() { stop(); }
  int low_pct = 12, high_pct = 35;  // below low_pct (before the ~9% where compute slows) the cache shrinks
  double step_gb = 1.0, period_s = 2.0, grow_after_s = 30.0;
  double retry_s = 600;  // a size that met pressure is not grown into again for this long (calm)
  std::function<void()> wake;                    // a budget was requested (the worker applies it when idle)
  std::function<void(const std::string&)> note;  // one JSON line per change
  void start();
  void stop();
  int64_t target() const { return target_.load(); }
  double floor_gb() const { return min_b_ / 1e9; }  // after start(): the least budget the watcher sets
  // the context: the per-position state grows with it, so the ceiling follows. kv_planned: the bytes the start planned
  // for; set_positions (any thread; the worker, before a request with its positions + tokens to come, and after it): the
  // ceiling is lowered by the state beyond the plan, and a budget above it shrinks at the next check (why "context")
  void kv_plan(size_t pos_bytes, int planned_positions) {
    pos_b_ = (int64_t)pos_bytes;
    plan_b_ = (int64_t)pos_bytes * planned_positions;
  }
  void set_positions(int n) {
    kv_extra_.store(std::max<int64_t>(0, pos_b_ * n - plan_b_));
    kick_.store(true);
  }

 private:
  Cache& c_;
  int64_t max_b_, min_b_, start_b_;
  std::atomic<int64_t> target_{0};
  int64_t pos_b_ = 0, plan_b_ = 0;
  std::atomic<int64_t> kv_extra_{0};
  std::atomic<bool> quit_{false}, kick_{false};
  std::thread th_;
  void* src_ = nullptr;  // dispatch source (memory pressure)
  void loop();
};

}  // namespace dm
