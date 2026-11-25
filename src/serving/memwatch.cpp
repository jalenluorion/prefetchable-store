// The expert cache's size from the machine (src/serving/memwatch.h).
#include "memwatch.h"

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/stat.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>

namespace dm {

static int sysctl_int(const char* name) {
  int v = -1;
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, nullptr, 0) == 0 ? v : -1;
}

MemSnap mem_snapshot() {
  MemSnap s;
  uint64_t mem = 0;
  size_t n = sizeof mem;
  if (sysctlbyname("hw.memsize", &mem, &n, nullptr, 0) == 0) s.total_gb = mem / 1e9;
  vm_statistics64_data_t vm;
  mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
  vm_size_t pg = 16384;
  host_page_size(mach_host_self(), &pg);
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &cnt) == KERN_SUCCESS)
    s.avail_gb = (double)(vm.free_count + vm.inactive_count + vm.speculative_count + vm.purgeable_count) * pg / 1e9;
  s.level_pct = sysctl_int("kern.memorystatus_level");
  s.pressure = sysctl_int("kern.memorystatus_vm_pressure_level");
  task_vm_info_data_t v;
  mach_msg_type_number_t c2 = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &c2) == KERN_SUCCESS) s.footprint_gb = v.phys_footprint / 1e9;
  s.wired_gb = (double)vm.wire_count * pg / 1e9;
  uint64_t wl = 0;
  n = sizeof wl;
  if (sysctlbyname("vm.global_user_wire_limit", &wl, &n, nullptr, 0) == 0) s.wire_limit_gb = wl / 1e9;
  return s;
}

std::string MemSnap::json() const {
  char b[256];
  snprintf(b, sizeof b, "{\"total_gb\":%.2f,\"avail_gb\":%.2f,\"free_level_pct\":%d,\"pressure\":%d,\"footprint_gb\":%.2f,\"wired_gb\":%.2f,\"wire_limit_gb\":%.2f}",
           total_gb, avail_gb, level_pct, pressure, footprint_gb, wired_gb, wire_limit_gb);
  return b;
}

double store_bytes_gb(const std::string& dir, const std::string& layer_file, int layers) {
  double b = 0;
  char buf[4096];
  for (int L = 0; L < layers; L++) {
    snprintf(buf, sizeof buf, layer_file.c_str(), L);
    struct stat st;
    if (stat((dir + "/" + buf).c_str(), &st) == 0) b += st.st_size;
  }
  return b / 1e9;
}

double auto_cache_gb(const MemRule& r, size_t pos_bytes, double ckpt_gb, double store_gb, double& start_gb, std::string& why) {
  const MemSnap s = mem_snapshot();
  const double kv = (double)pos_bytes * r.kv_positions / 1e9;
  const double other = s.footprint_gb + kv + ckpt_gb + r.scratch_gb + r.margin_gb + r.later_gb;
  double c = s.total_gb - r.reserve_gb - other;
  const double raw = c;
  c = std::max(r.min_gb, c);
  if (store_gb > 0) c = std::min(c, store_gb);
  // never more than the system will lock (the cache is the only memory still to be wired)
  const double wire = s.wire_limit_gb > 0 ? s.wire_limit_gb - s.wired_gb - r.wire_margin_gb : 1e9;
  c = std::min(c, std::max(0.0, wire));
  // the start: the kernel's free level kept at floor_pct once the cache and the process's memory still to come are in
  const double to_come = other - s.footprint_gb;
  const double start_raw = s.level_pct >= 0 ? (s.level_pct - r.floor_pct) / 100.0 * s.total_gb - to_come
                                            : s.avail_gb - to_come - r.headroom_gb;
  start_gb = std::min(c, std::max(r.min_gb, start_raw));
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(2);
  o << "{\"rule\":\"physical - reserve - other\",\"physical_gb\":" << s.total_gb << ",\"reserve_gb\":" << r.reserve_gb << ",\"other_gb\":" << other
    << ",\"resident_now_gb\":" << s.footprint_gb << ",\"kv_gb\":" << kv << ",\"later_gb\":" << r.later_gb << ",\"ckpt_gb\":" << ckpt_gb << ",\"scratch_gb\":" << r.scratch_gb
    << ",\"margin_gb\":" << r.margin_gb << ",\"wire_room_gb\":" << wire << ",\"floor_pct\":" << r.floor_pct << ",\"unclamped_gb\":" << raw << ",\"store_gb\":" << store_gb << ",\"ceiling_gb\":" << c << ",\"headroom_gb\":" << r.headroom_gb << ",\"start_unclamped_gb\":" << start_raw
    << ",\"start_gb\":" << start_gb << ",\"at_start\":" << s.json()
    << "}";
  why = o.str();
  return c;
}

void MemWatch::start() {
  // the cache's own floor: each pool keeps at least what one step or one prefill layer pins (Cache::SlotPool::min_cap);
  // Inkling's two record formats put it at ~3.6 GB, the 35B's at ~0.9 GB
  int64_t floor_b = 0;
  for (int p = 0; p < c_.npools; p++) floor_b += (int64_t)c_.P[p].min_cap * (int64_t)c_.P[p].rec;
  min_b_ = std::max(min_b_, floor_b);
  start_b_ = std::max(start_b_, min_b_);
  target_.store(start_b_);
  quit_.store(false);
  // the system's pressure notifications wake the watcher at once (it then decides from the levels, as on a timer)
  dispatch_source_t ds = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                                DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
                                                dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
  if (ds) {
    std::atomic<bool>* k = &kick_;
    dispatch_source_set_event_handler(ds, ^{
      k->store(true);
    });
    dispatch_resume(ds);
    src_ = (void*)ds;
  }
  th_ = std::thread([this] { loop(); });
}

void MemWatch::stop() {
  if (!th_.joinable()) return;
  quit_.store(true);
  th_.join();
  if (src_) {
    dispatch_source_cancel((dispatch_source_t)src_);
    src_ = nullptr;
  }
}

void MemWatch::loop() {
  using clk = std::chrono::steady_clock;
  auto calm_since = clk::now(), pressed_at = clk::now();
  const int64_t step = (int64_t)(step_gb * 1e9);
  int64_t limit = max_b_;  // the size growth stops at: below the last size that met pressure, until retry_s of calm
  while (!quit_.load()) {
    for (int i = 0; i < (int)(period_s * 20) && !quit_.load() && !kick_.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (quit_.load()) break;
    const bool kicked = kick_.exchange(false);
    const MemSnap s = mem_snapshot();
    const bool pressed = s.pressure >= 2 || (s.level_pct >= 0 && s.level_pct < low_pct);
    const bool calm = s.pressure == 1 && s.level_pct >= high_pct;
    if (!calm) calm_since = clk::now();
    const int64_t cur = target_.load();
    int64_t next = cur;
    std::string why;
    const int64_t ceil_now = std::max(min_b_, max_b_ - kv_extra_.load());  // the context's state beyond the plan
    if (pressed) {
      limit = std::max(min_b_, std::min(limit, cur - step));
      pressed_at = clk::now();
    } else if (limit < max_b_ && std::chrono::duration<double>(clk::now() - pressed_at).count() >= retry_s) {
      limit = std::min(max_b_, limit + step);
      pressed_at = clk::now();
    }
    if (pressed && cur > min_b_) {
      next = std::max(min_b_, cur - step);
      why = s.pressure >= 2 ? "pressure" : "low_free_level";
    } else if (cur > ceil_now) {
      next = ceil_now;
      why = "context";
    } else if (calm && cur < std::min(limit, ceil_now) && std::chrono::duration<double>(clk::now() - calm_since).count() >= grow_after_s) {
      next = std::min(std::min(limit, ceil_now), cur + step);
      why = "calm";
      calm_since = clk::now();  // one step per grow_after_s
    }
    if (next != cur) {
      target_.store(next);
      c_.set_budget(next);
      if (wake) wake();
      if (note) {
        std::ostringstream o;
        o << "{\"budget_gb\":" << next / 1e9 << ",\"from_gb\":" << cur / 1e9 << ",\"grow_limit_gb\":" << limit / 1e9 << ",\"in_force_gb\":" << c_.budget_bytes() / 1e9 << ",\"held_gb\":" << c_.held_bytes() / 1e9 << ",\"why\":\"" << why << "\",\"kicked\":" << (kicked ? "true" : "false")
          << ",\"machine\":" << s.json() << "}";
        note(o.str());
      }
    }
  }
}

}  // namespace dm
