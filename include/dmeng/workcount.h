// Work counters for same-work comparisons between two builds (tests/work_compare.py): kernel calls by kind, the
// (row, position) products and weight bytes they cover, the GPU's rows, fork-joins, packed-input copies, expert reads and
// names, moe waves. Off unless DMENG_WORKCOUNT is set (one predictable branch per counted call); dmeng prints the
// prefill's counters as a WORK line on stderr. Counting only: nothing computed depends on it.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace dm {
namespace wc {
inline bool on() {
  static const bool v = std::getenv("DMENG_WORKCOUNT") != nullptr;
  return v;
}
struct Reg {
  std::mutex mu;
  std::map<std::string, std::atomic<uint64_t>*> m;
};
inline Reg& reg() {
  static Reg r;
  return r;
}
inline std::atomic<uint64_t>& slot(const char* k) {
  Reg& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  auto it = r.m.find(k);
  if (it == r.m.end()) it = r.m.emplace(k, new std::atomic<uint64_t>(0)).first;
  return *it->second;
}
inline void reset() {
  Reg& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  for (auto& kv : r.m) kv.second->store(0);
}
inline std::string json() {
  Reg& r = reg();
  std::lock_guard<std::mutex> g(r.mu);
  std::string o = "{";
  for (auto& kv : r.m) {
    if (!kv.second->load()) continue;
    o += (o.size() > 1 ? ",\"" : "\"") + kv.first + "\":" + std::to_string(kv.second->load());
  }
  return o + "}";
}
}  // namespace wc
}  // namespace dm

#define DM_WC(key, v)                                                         \
  do {                                                                        \
    if (::dm::wc::on()) {                                                     \
      static std::atomic<uint64_t>& wc_s_ = ::dm::wc::slot(key);              \
      wc_s_.fetch_add((uint64_t)(v), std::memory_order_relaxed);              \
    }                                                                         \
  } while (0)
