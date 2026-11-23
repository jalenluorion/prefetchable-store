#import <Foundation/Foundation.h>

#include "util.h"

#include <TargetConditionals.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/sysctl.h>

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
#include <os/proc.h>
#endif

namespace dmb {

std::atomic<int> g_mem_warnings{0};
std::atomic<int> g_foreground{1};
static std::atomic<int> g_pressure{0};
static dispatch_source_t g_src = nullptr;

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int thermal() {
  switch ([NSProcessInfo processInfo].thermalState) {
    case NSProcessInfoThermalStateNominal: return 0;
    case NSProcessInfoThermalStateFair: return 1;
    case NSProcessInfoThermalStateSerious: return 2;
    case NSProcessInfoThermalStateCritical: return 3;
  }
  return -1;
}
const char* thermal_name(int t) {
  static const char* n[] = {"nominal", "fair", "serious", "critical"};
  return t >= 0 && t < 4 ? n[t] : "unknown";
}

int64_t footprint() {
  task_vm_info_data_t vi;
  mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &c) != KERN_SUCCESS) return -1;
  return (int64_t)vi.phys_footprint;
}

int64_t available() {
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  return (int64_t)os_proc_available_memory();
#else
  return 0;
#endif
}

Events events() {
  task_events_info_data_t e;
  mach_msg_type_number_t c = TASK_EVENTS_INFO_COUNT;
  Events r;
  if (task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&e, &c) == KERN_SUCCESS) {
    r.faults = e.faults;
    r.pageins = e.pageins;
    r.cow = e.cow_faults;
  }
  return r;
}

Cpu thread_cpu() {
  thread_basic_info_data_t b;
  mach_msg_type_number_t c = THREAD_BASIC_INFO_COUNT;
  mach_port_t t = mach_thread_self();
  Cpu r;
  if (thread_info(t, THREAD_BASIC_INFO, (thread_info_t)&b, &c) == KERN_SUCCESS) {
    r.user = b.user_time.seconds + b.user_time.microseconds * 1e-6;
    r.sys = b.system_time.seconds + b.system_time.microseconds * 1e-6;
  }
  mach_port_deallocate(mach_task_self(), t);
  return r;
}

Cpu task_cpu() {
  Cpu r;
  task_thread_times_info_data_t tt;
  mach_msg_type_number_t c = TASK_THREAD_TIMES_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_THREAD_TIMES_INFO, (task_info_t)&tt, &c) == KERN_SUCCESS) {
    r.user += tt.user_time.seconds + tt.user_time.microseconds * 1e-6;
    r.sys += tt.system_time.seconds + tt.system_time.microseconds * 1e-6;
  }
  mach_task_basic_info_data_t bi;
  c = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&bi, &c) == KERN_SUCCESS) {
    r.user += bi.user_time.seconds + bi.user_time.microseconds * 1e-6;  // terminated threads
    r.sys += bi.system_time.seconds + bi.system_time.microseconds * 1e-6;
  }
  return r;
}

int battery_pct() { return -1; }

int pressure_events() { return g_pressure.load(); }

void start_pressure_source() {
  if (g_src) return;
  g_src = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                                 DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
                                 dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
  dispatch_source_set_event_handler(g_src, ^{ g_pressure.fetch_add(1); });
  dispatch_resume(g_src);
}

int qos_value(const std::string& n) {
  if (n == "ui") return QOS_CLASS_USER_INTERACTIVE;
  if (n == "initiated") return QOS_CLASS_USER_INITIATED;
  if (n == "default") return QOS_CLASS_DEFAULT;
  if (n == "utility") return QOS_CLASS_UTILITY;
  if (n == "background") return QOS_CLASS_BACKGROUND;
  return QOS_CLASS_USER_INTERACTIVE;
}
void set_qos(const std::string& n) { pthread_set_qos_class_self_np((qos_class_t)qos_value(n), 0); }

static int64_t sysi(const char* name) {
  int64_t v = 0;
  size_t sz = sizeof(v);
  if (sysctlbyname(name, &v, &sz, nullptr, 0) != 0) return -1;
  if (sz == 4) return (int64_t)(int32_t)v;
  return v;
}
static std::string syss(const char* name) {
  char b[256] = {0};
  size_t sz = sizeof(b) - 1;
  if (sysctlbyname(name, b, &sz, nullptr, 0) != 0) return "";
  return b;
}

std::string features_json() {
  J j;
  j.obj();
  const char* f[] = {"hw.optional.arm.FEAT_DotProd", "hw.optional.arm.FEAT_I8MM", "hw.optional.arm.FEAT_FHM",
                     "hw.optional.arm.FEAT_FP16", "hw.optional.arm.FEAT_BF16", "hw.optional.arm.FEAT_SME",
                     "hw.optional.arm.FEAT_SME2", "hw.optional.arm.FEAT_SVE", "hw.ncpu", "hw.perflevel0.physicalcpu",
                     "hw.perflevel1.physicalcpu", "hw.perflevel0.l2cachesize", "hw.perflevel1.l2cachesize",
                     "hw.l1dcachesize", "hw.cachelinesize", "hw.pagesize", "hw.memsize", "hw.cpufamily",
                     "hw.perflevel0.cpusperl2", "hw.nperflevels"};
  for (const char* n : f) j.kv(n, sysi(n));
  j.kv("hw.machine", syss("hw.machine"));
  j.kv("kern.osversion", syss("kern.osversion"));
  j.kv("kern.osproductversion", syss("kern.osproductversion"));
  j.kv("machdep.cpu.brand_string", syss("machdep.cpu.brand_string"));
  j.end_obj();
  return j.s;
}

Spec::Spec(const char* s) {
  std::string t = s ? s : "";
  size_t p = 0;
  while (p < t.size()) {
    size_t q = t.find(',', p);
    if (q == std::string::npos) q = t.size();
    std::string a = t.substr(p, q - p);
    size_t e = a.find('=');
    if (e != std::string::npos) kv[a.substr(0, e)] = a.substr(e + 1);
    else if (!a.empty()) kv[a] = "1";
    p = q + 1;
  }
}
double Spec::num(const std::string& k, double d) const {
  auto it = kv.find(k);
  return it == kv.end() ? d : atof(it->second.c_str());
}
std::string Spec::str(const std::string& k, const std::string& d) const {
  auto it = kv.find(k);
  return it == kv.end() ? d : it->second;
}
std::vector<int> Spec::ints(const std::string& k, std::vector<int> d) const {
  auto it = kv.find(k);
  if (it == kv.end()) return d;
  std::vector<int> r;
  const std::string& s = it->second;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(':', p);
    if (q == std::string::npos) q = s.size();
    r.push_back(atoi(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return r;
}

void J::sep() {
  if (after_key) {
    after_key = false;
    return;
  }
  if (!stack.empty()) {
    if (!stack.back()) s += ",";
    stack.back() = false;
  }
}
J& J::obj() { sep(); s += "{"; stack.push_back(true); return *this; }
J& J::end_obj() { s += "}"; stack.pop_back(); return *this; }
J& J::arr() { sep(); s += "["; stack.push_back(true); return *this; }
J& J::end_arr() { s += "]"; stack.pop_back(); return *this; }
J& J::key(const char* k) {
  sep();
  s += "\"";
  s += k;
  s += "\":";
  after_key = true;
  return *this;
}
J& J::v(double x) {
  sep();
  char b[64];
  if (std::isfinite(x)) snprintf(b, sizeof b, "%.6g", x);
  else snprintf(b, sizeof b, "null");
  s += b;
  return *this;
}
J& J::v(int64_t x) { sep(); s += std::to_string(x); return *this; }
J& J::v(const std::string& x) {
  sep();
  s += "\"";
  for (char c : x) {
    if (c == '"' || c == '\\') { s += '\\'; s += c; }
    else if (c == '\n') s += "\\n";
    else if ((unsigned char)c < 0x20) s += ' ';
    else s += c;
  }
  s += "\"";
  return *this;
}
J& J::vb(bool b) { sep(); s += b ? "true" : "false"; return *this; }
J& J::raw(const std::string& x) { sep(); s += x; return *this; }

void logf(dmb_log_fn log, const char* fmt, ...) {
  char b[2048];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  if (log) log(b);
  else { fputs(b, stdout); fputc('\n', stdout); fflush(stdout); }
}

char* dup(const std::string& s) {
  char* p = (char*)malloc(s.size() + 1);
  memcpy(p, s.c_str(), s.size() + 1);
  return p;
}

}  // namespace dmb

extern "C" {
void dmb_free(char* p) { free(p); }
void dmb_note_memory_warning(void) { dmb::g_mem_warnings.fetch_add(1); }
void dmb_set_foreground(int fg) { dmb::g_foreground.store(fg); }
}
