// P0 file-cache test (dmb_filecache_test): is the OS file cache a usable expert cache for a CPU engine on iOS?
// The store files are mapped PROT_READ, MAP_SHARED (clean file-backed pages: never written, so never dirty or COW);
// "records" are 1,769,472-byte ranges (108 pages of 16 KB, page aligned) at offsets i * 1,769,472 of each file. The
// files are only read (open O_RDONLY, pread, mmap PROT_READ); nothing is written or deleted.
// Variants:
//   grow   (i)   sweep new records with 8 threads until the resident set plateaus, a memory warning, or the end; then a
//                decode-like random pattern; then re-touch of resident records (warm rate)
//   dirty  (ii)  the same with `dirty` GB (default 1.5) of dirty memory held
//   paths  (iii) each access path's own cost per record (wall, thread user/system CPU, faults, page-ins; cold and
//                resident): a pread F_NOCACHE into a private buffer (today's path), b buffered pread into a private
//                buffer, c mmap + touch, d madvise(MADV_WILLNEED) + touch; does b populate the cache the mapping sees;
//                WILLNEED with 1-16 records in flight; 8-thread throughput per path; and the interference with a CPU
//                kernel load (the experts' kernel on `cput` threads at QoS user-interactive) per path
//   bg     (iv)  touch `bgtouch` GB, wait for the app to be backgrounded and brought back, measure what stayed
// Every second: resident GB (mincore over all mapped ranges), phys_footprint, os_proc_available_memory, memory warnings
// (UIApplication + dispatch memory-pressure source), thermal, faults / page-ins, cold and warm touch rates.
#import <Foundation/Foundation.h>

#include <arm_neon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <mutex>
#include <random>

#include "bench.h"

namespace dmb {

static const size_t REC = 1769472;

struct Map {
  std::string path;
  int fd = -1, fdnc = -1;
  uint8_t* p = nullptr;
  size_t size = 0;
  int nrec = 0;
};

struct FC {
  std::vector<Map> maps;
  std::vector<std::pair<int, int>> recs;  // (map, record)
  size_t page = 16384;
  dmb_log_fn log = nullptr;
  std::atomic<uint64_t> sink{0};
  // worker counters
  std::atomic<int64_t> cold_bytes{0}, warm_bytes{0}, cold_ns{0}, warm_ns{0}, touched_bytes{0};
  std::atomic<bool> stop{false};
  std::string phase = "init";
  std::mutex mu;
  std::vector<int> early;  // records touched first by the sweep (reclaim witness)
  uint8_t* rp(int r) const { return maps[recs[r].first].p + (size_t)recs[r].second * REC; }
  off_t off(int r) const { return (off_t)recs[r].second * REC; }
  int fd(int r, bool nocache) const { return nocache ? maps[recs[r].first].fdnc : maps[recs[r].first].fd; }
};

static size_t incore_range(const uint8_t* p, size_t len, size_t page, std::vector<char>& vec) {
  size_t np = (len + page - 1) / page;
  vec.resize(np);
  if (mincore((void*)p, len, vec.data()) != 0) return 0;
  size_t n = 0;
  for (size_t i = 0; i < np; i++) n += (vec[i] & MINCORE_INCORE) != 0;
  return n;
}
static double rec_frac(FC& F, int r) {
  thread_local std::vector<char> v;
  return (double)incore_range(F.rp(r), REC, F.page, v) / (REC / F.page);
}
static double resident_gb(FC& F) {
  std::vector<char> v;
  size_t pages = 0;
  for (auto& m : F.maps) {
    const size_t chunk = (size_t)256 << 20;
    for (size_t o = 0; o < m.size; o += chunk) pages += incore_range(m.p + o, std::min(chunk, m.size - o), F.page, v);
  }
  return (double)pages * F.page / 1e9;
}
static double set_resident_gb(FC& F, const std::vector<int>& rs) {
  std::vector<char> v;
  size_t pages = 0;
  for (int r : rs) pages += incore_range(F.rp(r), REC, F.page, v);
  return (double)pages * F.page / 1e9;
}

static uint64_t sum_bytes(const uint8_t* p, size_t n) {
  uint64x2_t s0 = vdupq_n_u64(0), s1 = s0;
  for (size_t i = 0; i < n; i += 64) {
    uint8x16_t a = vld1q_u8(p + i), b = vld1q_u8(p + i + 16), c = vld1q_u8(p + i + 32), d = vld1q_u8(p + i + 48);
    s0 = vpadalq_u32(s0, vpaddlq_u16(vpaddlq_u8(veorq_u8(a, b))));
    s1 = vpadalq_u32(s1, vpaddlq_u16(vpaddlq_u8(veorq_u8(c, d))));
  }
  return vgetq_lane_u64(vaddq_u64(s0, s1), 0) + vgetq_lane_u64(vaddq_u64(s0, s1), 1);
}

// touch a record through the mapping, classified cold / warm by mincore just before
static void touch_rec(FC& F, int r) {
  bool warm = rec_frac(F, r) >= 0.999;
  double t0 = now();
  F.sink.fetch_add(sum_bytes(F.rp(r), REC), std::memory_order_relaxed);
  int64_t ns = (int64_t)((now() - t0) * 1e9);
  (warm ? F.warm_bytes : F.cold_bytes).fetch_add(REC);
  (warm ? F.warm_ns : F.cold_ns).fetch_add(ns);
  F.touched_bytes.fetch_add(REC);
}

struct Sample {
  double t, resident, early_res, touched, cold_gbps, warm_gbps;
  int64_t fp, av, faults, pageins;
  int warn, press, th;
  std::string phase;
};

struct Sampler {
  FC& F;
  std::thread th;
  std::atomic<bool> stop{false};
  std::vector<Sample> s;
  std::mutex mu;
  double t0;
  explicit Sampler(FC& f) : F(f) {}
  void start() {
    t0 = now();
    th = std::thread([this] {
      set_qos("initiated");
      int64_t cb = 0, wb = 0, cn = 0, wn = 0;
      while (!stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        Sample x;
        x.t = now() - t0;
        x.resident = resident_gb(F);
        std::vector<int> e;
        {
          std::lock_guard<std::mutex> lk(F.mu);
          e = F.early;
          x.phase = F.phase;
        }
        x.early_res = e.empty() ? 0 : set_resident_gb(F, e);
        x.touched = F.touched_bytes.load() / 1e9;
        int64_t c1 = F.cold_bytes.load(), w1 = F.warm_bytes.load(), cn1 = F.cold_ns.load(), wn1 = F.warm_ns.load();
        // per-thread rates: bytes / summed touch time (8 threads touch in parallel; the aggregate is touched GB/s)
        x.cold_gbps = cn1 > cn ? (double)(c1 - cb) / ((cn1 - cn) / 1e9) / 1e9 : 0;
        x.warm_gbps = wn1 > wn ? (double)(w1 - wb) / ((wn1 - wn) / 1e9) / 1e9 : 0;
        cb = c1; wb = w1; cn = cn1; wn = wn1;
        x.fp = footprint();
        x.av = available();
        Events ev = events();
        x.faults = ev.faults;
        x.pageins = ev.pageins;
        x.warn = g_mem_warnings.load();
        x.press = pressure_events();
        x.th = thermal();
        {
          std::lock_guard<std::mutex> lk(mu);
          s.push_back(x);
        }
        logf(F.log, "FILECACHE t=%.0f %s resident %.2f GB (early set %.2f) touched %.2f GB cold %.2f warm %.2f GB/s/thread fp %.2f avail %.2f GB warn %d/%d %s",
             x.t, x.phase.c_str(), x.resident, x.early_res, x.touched, x.cold_gbps, x.warm_gbps, x.fp / 1e9, x.av / 1e9,
             x.warn, x.press, thermal_name(x.th));
      }
    });
  }
  void end() {
    stop = true;
    if (th.joinable()) th.join();
  }
  void json(J& j) {
    std::lock_guard<std::mutex> lk(mu);
    j.key("samples_columns").arr();
    for (const char* c : {"t", "phase", "resident_gb", "early_set_resident_gb", "touched_gb", "cold_gbps_per_thread",
                          "warm_gbps_per_thread", "footprint", "available", "faults", "pageins", "warnings", "pressure", "thermal"})
      j.v(c);
    j.end_arr();
    j.key("samples").arr();
    for (auto& x : s)
      j.arr().v(x.t).v(x.phase).v(x.resident).v(x.early_res).v(x.touched).v(x.cold_gbps).v(x.warm_gbps).v(x.fp).v(x.av)
          .v(x.faults).v(x.pageins).v(x.warn).v(x.press).v(thermal_name(x.th)).end_arr();
    j.end_arr();
  }
};

static void set_phase(FC& F, const char* p) {
  std::lock_guard<std::mutex> lk(F.mu);
  F.phase = p;
  logf(F.log, "FILECACHE phase %s", p);
}

// run `nt` threads of fn(tid) until F.stop
template <class Fn>
static void run_threads(FC& F, int nt, Fn fn) {  // the caller clears F.stop before starting its controllers
  std::vector<std::thread> ts;
  for (int t = 0; t < nt; t++) ts.emplace_back([&, t] { set_qos("ui"); fn(t); });
  for (auto& t : ts) t.join();
}

// the sweep: new records in a shuffled order until the resident set plateaus / a warning / the end / the time limit
static void sweep(FC& F, Sampler& S, J& j, int nt, double max_s, double target_gb, std::vector<int>* touched_out) {
  std::vector<int> order(F.recs.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
  std::shuffle(order.begin(), order.end(), std::mt19937(42));
  std::atomic<size_t> next{0};
  F.stop = false;
  const double t0 = now();
  std::atomic<double> first_warn{0};
  int w0 = g_mem_warnings.load() + pressure_events();
  std::string why = "end of the mapped set";
  std::thread ctl([&] {
    double best = 0, best_t = now(), touched_at_best = 0;
    while (!F.stop.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      double t = now();
      if (t - t0 > max_s) { why = "time limit"; F.stop = true; break; }
      if (target_gb > 0 && F.touched_bytes.load() / 1e9 >= target_gb) { why = "target touched"; F.stop = true; break; }
      int w = g_mem_warnings.load() + pressure_events();
      if (w > w0 && first_warn.load() == 0) first_warn = t;
      if (first_warn.load() > 0 && t - first_warn.load() > 10) { why = "memory warning (+10 s)"; F.stop = true; break; }
      double res = 0;
      {
        std::lock_guard<std::mutex> lk(S.mu);
        if (!S.s.empty()) res = S.s.back().resident;
      }
      double touched = F.touched_bytes.load() / 1e9;
      if (res > best + 0.1) { best = res; best_t = t; touched_at_best = touched; }
      if (target_gb <= 0 && t - best_t > 15 && touched - touched_at_best > 1.5) { why = "plateau (no +0.1 GB in 15 s while touching 1.5 GB more)"; F.stop = true; break; }
    }
  });
  std::mutex tm;
  run_threads(F, nt, [&](int) {
    for (;;) {
      if (F.stop.load()) return;
      size_t i = next.fetch_add(1);
      if (i >= order.size()) { F.stop = true; return; }
      touch_rec(F, order[i]);
      if (i < 600) {  // the first ~1 GB touched: watched for reclaim
        std::lock_guard<std::mutex> lk(F.mu);
        F.early.push_back(order[i]);
      }
      if (touched_out) {
        std::lock_guard<std::mutex> lk(tm);
        touched_out->push_back(order[i]);
      }
    }
  });
  F.stop = true;
  ctl.join();
  double res = resident_gb(F);
  j.key("sweep").obj().kv("stopped_by", why).kv("seconds", now() - t0).kv("touched_gb", F.touched_bytes.load() / 1e9)
      .kv("resident_gb_at_end", res).kv("records_touched", (int64_t)std::min(next.load(), order.size()))
      .kv("cold_gbps_per_thread", F.cold_ns.load() ? F.cold_bytes.load() / (F.cold_ns.load() / 1e9) / 1e9 : 0.0)
      .kv("aggregate_gbps", F.touched_bytes.load() / (now() - t0) / 1e9).end_obj();
  logf(F.log, "FILECACHE sweep done: %s, touched %.2f GB in %.0f s, resident %.2f GB", why.c_str(), F.touched_bytes.load() / 1e9,
       now() - t0, res);
}

static void timed_random(FC& F, int nt, double secs, const std::vector<int>& pool, const char* key, J& j) {
  int64_t cb = F.cold_bytes, wb = F.warm_bytes, cn = F.cold_ns, wn = F.warm_ns, tb = F.touched_bytes;
  F.stop = false;
  double t0 = now();
  std::thread timer([&] {
    while (now() - t0 < secs && !F.stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    F.stop = true;
  });
  run_threads(F, nt, [&](int t) {
    std::mt19937_64 rng(1000 + t);
    while (!F.stop.load()) {
      int r = pool.empty() ? (int)(rng() % F.recs.size()) : pool[rng() % pool.size()];
      touch_rec(F, r);
    }
  });
  timer.join();
  double el = now() - t0;
  int64_t c = F.cold_bytes - cb, w = F.warm_bytes - wb;
  j.key(key).obj().kv("seconds", el).kv("threads", nt).kv("pool_records", (int64_t)pool.size())
      .kv("aggregate_gbps", (F.touched_bytes - tb) / el / 1e9).kv("cold_gb", c / 1e9).kv("warm_gb", w / 1e9)
      .kv("cold_gbps_per_thread", F.cold_ns - cn > 0 ? c / ((F.cold_ns - cn) / 1e9) / 1e9 : 0.0)
      .kv("warm_gbps_per_thread", F.warm_ns - wn > 0 ? w / ((F.warm_ns - wn) / 1e9) / 1e9 : 0.0)
      .kv("resident_gb_after", resident_gb(F)).end_obj();
  logf(F.log, "FILECACHE %s: %.2f GB/s aggregate, cold %.2f GB warm %.2f GB", key, (F.touched_bytes - tb) / el / 1e9, c / 1e9, w / 1e9);
}

// ----------------------------------------------------------------------------------------------- read paths (iii)
enum Path { PA = 0, PB, PC, PD };
static const char* PNAME[] = {"a_pread_nocache", "b_pread_buffered", "c_mmap_touch", "d_madvise_willneed_touch"};

static void do_path(FC& F, int path, int r, uint8_t* buf) {
  switch (path) {
    case PA: case PB: {
      int fd = F.fd(r, path == PA);
      size_t got = 0;
      while (got < REC) {
        ssize_t n = pread(fd, buf + got, REC - got, F.off(r) + (off_t)got);
        if (n <= 0) break;
        got += (size_t)n;
      }
      F.sink.fetch_add(buf[0] + buf[REC - 1], std::memory_order_relaxed);
      break;
    }
    case PC: F.sink.fetch_add(sum_bytes(F.rp(r), REC), std::memory_order_relaxed); break;
    case PD:
      madvise(F.rp(r), REC, MADV_WILLNEED);
      F.sink.fetch_add(sum_bytes(F.rp(r), REC), std::memory_order_relaxed);
      break;
  }
}

static std::vector<int> pick_cold(FC& F, int n, std::mt19937_64& rng, std::vector<char>& used) {
  std::vector<int> out;
  for (int tries = 0; tries < 200000 && (int)out.size() < n; tries++) {
    int r = (int)(rng() % F.recs.size());
    if (used[r]) continue;
    if (rec_frac(F, r) == 0) {
      out.push_back(r);
      used[r] = 1;
    }
  }
  return out;
}

// a record with no resident page now (the cache evicts as it goes, so records can come back cold)
static int next_cold(FC& F, std::mt19937_64& rng) {
  int r = 0;
  for (int tries = 0; tries < 4000; tries++) {
    r = (int)(rng() % F.recs.size());
    if (rec_frac(F, r) == 0) return r;
  }
  return r;
}

struct Agg { std::vector<double> wall, user, sys; double faults = 0, pageins = 0, frac_after = 0; int n = 0; };
static void agg_json(J& j, const char* k, Agg& a) {
  auto med = [](std::vector<double> v) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
  auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size(); };
  j.key(k).obj().kv("n", a.n).kv("wall_ms_median", med(a.wall)).kv("wall_ms_mean", mean(a.wall)).kv("user_ms_mean", mean(a.user))
      .kv("sys_ms_mean", mean(a.sys)).kv("faults_per_record", a.n ? a.faults / a.n : 0).kv("pageins_per_record", a.n ? a.pageins / a.n : 0)
      .kv("resident_frac_after", a.n ? a.frac_after / a.n : 0)
      .kv("gbps_single_thread", mean(a.wall) > 0 ? REC / (mean(a.wall) * 1e-3) / 1e9 : 0).end_obj();
}

static void paths_test(FC& F, const Spec& sp, J& j) {
  const int nper = (int)sp.num("nper", 24);
  std::mt19937_64 rng(7);
  std::vector<char> used(F.recs.size(), 0);
  uint8_t* buf = (uint8_t*)std::aligned_alloc(16384, REC);
  std::memset(buf, 0, REC);
  set_qos("ui");
  // A. single-record costs, cold then resident (second access of the same records)
  set_phase(F, "paths-single");
  j.key("single").obj();
  for (int p = 0; p < 4; p++) {
    std::vector<int> rs = pick_cold(F, nper, rng, used);
    Agg cold, again;
    for (int pass = 0; pass < 2; pass++)
      for (int r : rs) {
        Agg& a = pass == 0 ? cold : again;
        Cpu c0 = thread_cpu();
        Events e0 = events();
        double t0 = now();
        do_path(F, p, r, buf);
        double w = (now() - t0) * 1e3;
        Cpu c1 = thread_cpu();
        Events e1 = events();
        a.wall.push_back(w);
        a.user.push_back((c1.user - c0.user) * 1e3);
        a.sys.push_back((c1.sys - c0.sys) * 1e3);
        a.faults += e1.faults - e0.faults;
        a.pageins += e1.pageins - e0.pageins;
        a.frac_after += rec_frac(F, r);
        a.n++;
      }
    j.key(PNAME[p]).obj();
    agg_json(j, "cold", cold);
    agg_json(j, "second_access", again);
    j.end_obj();
    logf(F.log, "FILECACHE path %s: cold %.2f ms (sys %.2f) then %.2f ms (sys %.2f), resident after %.2f", PNAME[p],
         cold.wall.empty() ? 0 : cold.wall[cold.wall.size() / 2], cold.sys.empty() ? 0 : cold.sys[0],
         again.wall.empty() ? 0 : again.wall[again.wall.size() / 2], again.sys.empty() ? 0 : again.sys[0],
         cold.n ? cold.frac_after / cold.n : 0);
  }
  // a resident record through path a / b (copy out of the cache) and through c (in place): records made resident by c
  {
    std::vector<int> rs = pick_cold(F, nper, rng, used);
    for (int r : rs) do_path(F, PC, r, buf);
    for (int p : {PA, PB, PC}) {
      Agg a;
      for (int r : rs) {
        Cpu c0 = thread_cpu();
        Events e0 = events();
        double t0 = now();
        do_path(F, p, r, buf);
        a.wall.push_back((now() - t0) * 1e3);
        Cpu c1 = thread_cpu();
        Events e1 = events();
        a.user.push_back((c1.user - c0.user) * 1e3);
        a.sys.push_back((c1.sys - c0.sys) * 1e3);
        a.faults += e1.faults - e0.faults;
        a.pageins += e1.pageins - e0.pageins;
        a.frac_after += rec_frac(F, r);
        a.n++;
      }
      agg_json(j, (std::string("resident_via_") + PNAME[p]).c_str(), a);
    }
  }
  j.end_obj();
  // B. WILLNEED with k records in flight: advise k, then touch them; and advise k, then wait for mincore (no touch)
  set_phase(F, "paths-willneed");
  j.key("willneed").arr();
  for (int k : {1, 4, 8, 16}) {
    std::vector<int> rs = pick_cold(F, k, rng, used);
    double t0 = now();
    for (int r : rs) madvise(F.rp(r), REC, MADV_WILLNEED);
    double ta = now();
    for (int r : rs) F.sink.fetch_add(sum_bytes(F.rp(r), REC));
    double t1 = now();
    std::vector<int> rs2 = pick_cold(F, k, rng, used);
    double u0 = now();
    for (int r : rs2) madvise(F.rp(r), REC, MADV_WILLNEED);
    double fr = 0;
    while (now() - u0 < 5) {
      fr = 0;
      for (int r : rs2) fr += rec_frac(F, r);
      if (fr >= rs2.size() * 0.999) break;
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    double u1 = now();
    j.obj().kv("in_flight", k).kv("advise_ms", (ta - t0) * 1e3).kv("advise_then_touch_gbps", k * REC / (t1 - t0) / 1e9)
        .kv("advise_then_wait_gbps", k * REC * (fr / std::max<size_t>(1, rs2.size())) / (u1 - u0) / 1e9)
        .kv("wait_resident_frac", fr / std::max<size_t>(1, rs2.size())).end_obj();
    logf(F.log, "FILECACHE willneed k=%d: advise+touch %.2f GB/s, advise+wait %.2f GB/s (frac %.2f)", k, k * REC / (t1 - t0) / 1e9,
         k * REC * (fr / std::max<size_t>(1, rs2.size())) / (u1 - u0) / 1e9, fr / std::max<size_t>(1, rs2.size()));
  }
  j.end_arr();
  // C. throughput, 8 threads per path, random cold records
  set_phase(F, "paths-throughput");
  const int nt = (int)sp.num("threads", 8);
  const double tsec = sp.num("tput", 6);
  j.key("throughput").arr();
  for (int p = 0; p < 4; p++) {
    std::atomic<int64_t> bytes{0};
    Cpu c0 = task_cpu();
    Events e0 = events();
    F.stop = false;
    double t0 = now();
    std::thread timer([&] { while (now() - t0 < tsec && !F.stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50)); F.stop = true; });
    run_threads(F, nt, [&](int t) {
      uint8_t* b = (uint8_t*)std::aligned_alloc(16384, REC);
      std::mt19937_64 rr(500 + 17 * p + t);
      while (!F.stop.load()) {
        do_path(F, p, next_cold(F, rr), b);
        bytes.fetch_add(REC);
      }
      std::free(b);
    });
    F.stop = true;
    timer.join();
    double el = now() - t0;
    Cpu c1 = task_cpu();
    Events e1 = events();
    double gb = bytes.load() / 1e9;
    j.obj().kv("path", PNAME[p]).kv("threads", nt).kv("seconds", el).kv("gbps", gb / el)
        .kv("cpu_user_s_per_gb", (c1.user - c0.user) / std::max(gb, 1e-9)).kv("cpu_sys_s_per_gb", (c1.sys - c0.sys) / std::max(gb, 1e-9))
        .kv("faults_per_record", (e1.faults - e0.faults) / std::max(1.0, gb * 1e9 / REC)).kv("pageins_per_record", (e1.pageins - e0.pageins) / std::max(1.0, gb * 1e9 / REC))
        .kv("footprint", footprint()).kv("available", available()).end_obj();
    logf(F.log, "FILECACHE throughput %s x%d: %.2f GB/s, cpu user %.2f sys %.2f s/GB", PNAME[p], nt, gb / el,
         (c1.user - c0.user) / std::max(gb, 1e-9), (c1.sys - c0.sys) / std::max(gb, 1e-9));
  }
  j.end_arr();
  // D. interference: the experts' CPU kernel on cput threads, alone and beside each read path (4 reader threads)
  set_phase(F, "paths-interference");
  const int cput = (int)sp.num("cput", 2);
  const double win = sp.num("window", 8);
  const int readers = (int)sp.num("readers", 4);
  CpuLoad L;
  L.start(*find_case("a4gu"), cput, "ui", 256);
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // a hot set made resident through the mapping (for the resident variants)
  std::vector<int> hot;
  for (int i = 0; i < (int)(sp.num("hotgb", 0.8) * 1e9 / REC); i++) hot.push_back((int)(rng() % F.recs.size()));
  for (int r : hot) do_path(F, PC, r, buf);
  j.key("interference").obj().kv("compute", "a4gu aff4 512x2048 matvec, DRAM-fed").kv("compute_threads", cput)
      .kv("compute_qos", "ui").kv("readers", readers).kv("window_s", win);
  j.key("windows").arr();
  struct W { const char* name; int path; bool resident; double rate; };  // rate: GB/s over all readers, 0 = as fast as it goes
  const double paced = sp.num("paced", 1.0);
  std::vector<W> ws = {{"baseline", -1, false, 0}, {"a_pread_nocache cold", PA, false, 0}, {"b_pread_buffered cold", PB, false, 0},
                       {"c_mmap_touch cold", PC, false, 0}, {"d_madvise_touch cold", PD, false, 0},
                       {"b_pread_buffered resident paced", PB, true, paced}, {"c_mmap_touch resident paced", PC, true, paced},
                       {"b_pread_buffered resident unpaced", PB, true, 0}, {"c_mmap_touch resident unpaced", PC, true, 0},
                       {"baseline again", -1, false, 0}};
  for (auto& w : ws) {
    if (w.resident)
      for (int r : hot) do_path(F, PC, r, buf);  // (re)make the hot set resident before the window
    const std::vector<int>& list = hot;
    std::atomic<int64_t> bytes{0};
    Cpu c0 = task_cpu();
    Events e0 = events();
    int64_t fp0 = footprint(), av0 = available();
    auto m0 = L.mark();
    double t0 = now();
    std::vector<std::thread> rt;
    F.stop = false;
    if (w.path >= 0)
      for (int t = 0; t < readers; t++)
        rt.emplace_back([&, t] {
          set_qos("ui");
          uint8_t* b = (uint8_t*)std::aligned_alloc(16384, REC);
          std::mt19937_64 rr(77 + t);
          const double per = w.rate > 0 ? REC / (w.rate * 1e9 / readers) : 0;  // seconds per record per reader
          double due = now();
          while (!F.stop.load()) {
            int r = w.resident ? list[rr() % list.size()] : next_cold(F, rr);
            do_path(F, w.path, r, b);
            bytes.fetch_add(REC);
            if (per > 0) {
              due += per;
              double sl = due - now();
              if (sl > 0) std::this_thread::sleep_for(std::chrono::microseconds((int64_t)(sl * 1e6)));
            }
          }
          std::free(b);
        });
    while (now() - t0 < win) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    F.stop = true;
    for (auto& t : rt) t.join();
    auto m1 = L.mark();
    double el = now() - t0;
    Cpu c1 = task_cpu();
    Events e1 = events();
    double comp = CpuLoad::gbps(m0, m1);
    j.obj().kv("window", w.name).kv("target_read_gbps", w.rate).kv("compute_gbps", comp).kv("read_gbps", bytes.load() / el / 1e9)
        .kv("task_cpu_user_s", c1.user - c0.user).kv("task_cpu_sys_s", c1.sys - c0.sys).kv("faults", e1.faults - e0.faults)
        .kv("pageins", e1.pageins - e0.pageins).kv("footprint_start", fp0).kv("footprint_end", footprint())
        .kv("available_start", av0).kv("available_end", available()).kv("hot_set_resident_gb", set_resident_gb(F, hot))
        .kv("thermal", thermal_name(thermal())).end_obj();
    logf(F.log, "FILECACHE interference %-34s compute %.2f GB/s, reads %.2f GB/s, sys %.2f s, hot resident %.2f GB [%s]", w.name, comp,
         bytes.load() / el / 1e9, c1.sys - c0.sys, set_resident_gb(F, hot), thermal_name(thermal()));
  }
  j.end_arr();
  j.end_obj();
  L.stop();
  std::free(buf);
}

}  // namespace dmb

using namespace dmb;

extern "C" char* dmb_filecache_test(const char* variant_c, const char* const* paths, int npaths, const char* spec, dmb_log_fn log) {
  std::string variant = variant_c ? variant_c : "grow";
  Spec sp(spec);
  start_pressure_source();
  FC F;
  F.log = log;
  F.page = (size_t)getpagesize();
  J j;
  j.obj();
  j.kv("variant", variant).kv("spec", std::string(spec ? spec : "")).kv("page_bytes", (int64_t)F.page).kv("record_bytes", (int64_t)REC)
      .kv("mapping", "PROT_READ, MAP_SHARED, whole file").kv("footprint_start", footprint()).kv("available_start", available())
      .kv("thermal_start", thermal_name(thermal()));
  // map
  size_t mapped = 0;
  std::string map_err;
  const double max_map_gb = sp.num("mapgb", 1e9);
  for (int i = 0; i < npaths; i++) {
    Map m;
    m.path = paths[i];
    m.fd = open(paths[i], O_RDONLY);
    if (m.fd < 0) { map_err = std::string("open ") + paths[i] + ": " + strerror(errno); continue; }
    m.fdnc = open(paths[i], O_RDONLY);
    fcntl(m.fdnc, F_NOCACHE, 1);
    struct stat st;
    fstat(m.fd, &st);
    m.size = (size_t)st.st_size;
    if (mapped / 1e9 + m.size / 1e9 > max_map_gb) { close(m.fd); close(m.fdnc); break; }
    void* p = mmap(nullptr, m.size, PROT_READ, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) {
      map_err = std::string("mmap ") + paths[i] + " after " + std::to_string(mapped / 1e9) + " GB: " + strerror(errno);
      close(m.fd);
      close(m.fdnc);
      break;
    }
    m.p = (uint8_t*)p;
    m.nrec = (int)(m.size / REC);
    mapped += m.size;
    F.maps.push_back(m);
  }
  for (int mi = 0; mi < (int)F.maps.size(); mi++)
    for (int r = 0; r < F.maps[mi].nrec; r++) F.recs.push_back({mi, r});
  j.kv("files_given", npaths).kv("files_mapped", (int)F.maps.size()).kv("mapped_gb", mapped / 1e9).kv("map_error", map_err)
      .kv("records", (int64_t)F.recs.size());
  logf(log, "FILECACHE %s: mapped %d of %d files, %.2f GB, %zu records%s%s; resident now %.2f GB", variant.c_str(), (int)F.maps.size(),
       npaths, mapped / 1e9, F.recs.size(), map_err.empty() ? "" : "; ", map_err.c_str(), resident_gb(F));
  j.kv("resident_gb_at_start", resident_gb(F));
  if (F.recs.empty()) {
    j.end_obj();
    return dup(j.s);
  }
  Sampler S(F);
  S.start();
  const int nt = (int)sp.num("threads", 8);
  uint8_t* dirty = nullptr;
  size_t dirty_bytes = 0;
  if (variant == "dirty") {
    dirty_bytes = (size_t)(sp.num("dirty", 1.5) * 1e9);
    set_phase(F, "dirty-alloc");
    dirty = (uint8_t*)malloc(dirty_bytes);
    for (size_t i = 0; i < dirty_bytes; i += 4096) dirty[i] = (uint8_t)i;
    std::memset(dirty, 1, dirty_bytes);
    j.kv("dirty_gb", dirty_bytes / 1e9).kv("footprint_after_dirty", footprint()).kv("available_after_dirty", available());
  }
  if (variant == "grow" || variant == "dirty") {
    set_phase(F, "sweep");
    sweep(F, S, j, nt, sp.num("sweep", 240), 0, nullptr);
    set_phase(F, "decode-random");
    timed_random(F, nt, sp.num("decode", 30), {}, "decode_random_all", j);
    // re-touch: the records resident now
    std::vector<int> res;
    for (int r = 0; r < (int)F.recs.size(); r++)
      if (rec_frac(F, r) >= 0.999) res.push_back(r);
    set_phase(F, "retouch-resident");
    timed_random(F, nt, sp.num("retouch", 15), res, "retouch_resident", j);
    j.kv("resident_records_before_retouch", (int64_t)res.size());
    set_phase(F, "idle");
    std::this_thread::sleep_for(std::chrono::milliseconds((int)(sp.num("idle", 10) * 1000)));
  } else if (variant == "paths") {
    paths_test(F, sp, j);
  } else if (variant == "bg") {
    set_phase(F, "bg-fill");
    std::vector<int> touched;
    sweep(F, S, j, nt, sp.num("sweep", 120), sp.num("bgtouch", 2.5), &touched);
    double r_set = set_resident_gb(F, touched), r_all = resident_gb(F);
    j.kv("before_background_touched_set_resident_gb", r_set).kv("before_background_resident_gb", r_all);
    set_phase(F, "bg-wait");
    logf(log, "FILECACHE ready for background: touched set resident %.2f GB, all %.2f GB", r_set, r_all);
    double tw = now(), tb = 0, tf = 0;
    while (now() - tw < sp.num("bgwait", 240)) {
      if (g_foreground.load() == 0) { tb = now(); break; }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (tb > 0)
      while (now() - tb < 900) {
        if (g_foreground.load() == 1) { tf = now(); break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    double r_set2 = set_resident_gb(F, touched), r_all2 = resident_gb(F);
    j.kv("backgrounded", tb > 0).kv("background_seconds", tf > 0 ? tf - tb : -1.0)
        .kv("after_return_touched_set_resident_gb", r_set2).kv("after_return_resident_gb", r_all2)
        .kv("footprint_after_return", footprint()).kv("available_after_return", available());
    logf(log, "FILECACHE back after %.1f s: touched set resident %.2f GB (was %.2f), all %.2f GB (was %.2f)", tf > 0 ? tf - tb : -1.0,
         r_set2, r_set, r_all2, r_all);
    set_phase(F, "bg-retouch");
    timed_random(F, nt, sp.num("retouch", 10), touched, "retouch_after_return", j);
  }
  S.end();
  S.json(j);
  j.kv("memory_warnings", g_mem_warnings.load()).kv("pressure_events", pressure_events()).kv("footprint_end", footprint())
      .kv("available_end", available()).kv("thermal_end", thermal_name(thermal())).kv("sink", (int64_t)(F.sink.load() & 0xffff));
  j.end_obj();
  for (auto& m : F.maps) {
    munmap(m.p, m.size);
    close(m.fd);
    close(m.fdnc);
  }
  if (dirty) free(dirty);
  return dup(j.s);
}
