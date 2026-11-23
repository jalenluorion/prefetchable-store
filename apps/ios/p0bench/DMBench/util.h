// Shared helpers of the P0 bench (implemented in util.mm).
#pragma once
#include <atomic>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "include/dmbench.h"

namespace dmb {

double now();                        // seconds, monotonic
int thermal();                       // 0 nominal, 1 fair, 2 serious, 3 critical (NSProcessInfo)
const char* thermal_name(int t);
int64_t footprint();                 // phys_footprint (TASK_VM_INFO), bytes
int64_t available();                 // os_proc_available_memory (iOS; 0 elsewhere)
struct Events { int64_t faults = 0, pageins = 0, cow = 0; };
Events events();                     // task_events_info (process-wide)
struct Cpu { double user = 0, sys = 0; };
Cpu thread_cpu();                    // this thread's user / system time (THREAD_BASIC_INFO), seconds
Cpu task_cpu();                      // the process's live threads' user / system time (TASK_THREAD_TIMES_INFO + TASK_BASIC_INFO)
int battery_pct();                   // -1 unknown (filled by the app through the spec when known)
extern std::atomic<int> g_mem_warnings;
extern std::atomic<int> g_foreground;
int pressure_events();               // DISPATCH_SOURCE_TYPE_MEMORYPRESSURE events since start (warn + critical)
void start_pressure_source();

void set_qos(const std::string& name);          // this thread
int qos_value(const std::string& name);          // ui | default | utility | background -> qos_class_t
std::string features_json();                     // hw.optional.arm.* and core counts (sysctl)

// key=value,key=value spec
struct Spec {
  std::map<std::string, std::string> kv;
  explicit Spec(const char* s);
  double num(const std::string& k, double d) const;
  std::string str(const std::string& k, const std::string& d) const;
  std::vector<int> ints(const std::string& k, std::vector<int> d) const;  // "1:2:4"
};

// minimal JSON writer
struct J {
  std::string s;
  bool after_key = false;
  std::vector<bool> stack;
  void sep();
  J& obj();
  J& end_obj();
  J& arr();
  J& end_arr();
  J& key(const char* k);
  J& v(double x);
  J& v(int64_t x);
  J& v(int x) { return v((int64_t)x); }
  J& v(const std::string& x);
  J& v(const char* x) { return v(std::string(x)); }
  J& vb(bool b);
  J& raw(const std::string& x);
  template <class T> J& kv(const char* k, T x) { key(k); return v(x); }
};

void logf(dmb_log_fn log, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
char* dup(const std::string& s);

}  // namespace dmb
