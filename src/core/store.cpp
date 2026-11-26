// Record formats, the reader threads and the expert slots.
#include "dmeng/workcount.h"
#include "dmeng/store.h"

#include "../format/json.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <mach/vm_map.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>

#if DM_COUNTERS
#include <cstdint>
extern "C" void dmc_now(uint64_t* a);                         // src/core/pool.cpp, the counting build only
extern "C" void dmc_spin(const uint64_t* a, const uint64_t* b);
#endif

namespace dm {

// ---------------------------------------------------------------------------------------------- record formats
static float nv_gmul(const uint8_t* rec, long trailer, int which) {
  float t[4];
  std::memcpy(t, rec + trailer, 16);
  if (!(t[2 + which] > 0.f)) return 0.f;               // an unread slot (timing runs with skipped reads)
  float gs = t[which] * (6.f * 448.f / t[2 + which]);  // the release's global scale, float32
  return gs / (6.f * 448.f);                           // w = e2m1 x e4m3 x global scale / (448 x 6)
}

void ExpertFormat::mats(const uint8_t* r, Weight& gu, Weight& up, Weight& dn) const {
  const Weight::Kind k = (Weight::Kind)(int)fmt;
  gu = Weight();
  up = Weight();
  dn = Weight();
  gu.kind = dn.kind = k;  // up stays NONE for a fused gate_up matrix
  gu.rows = fused ? 2 * I : I;
  gu.cols = H;
  gu.c = r + gu_c;
  gu.s = r + gu_s;
  gu.b = fmt == Fmt::AFF4 ? r + gu_b : nullptr;
  if (!fused) {
    up.kind = k;
    up.rows = I;
    up.cols = H;
    up.c = r + up_c;
    up.s = r + up_s;
    up.b = fmt == Fmt::AFF4 ? r + up_b : nullptr;
  }
  dn.rows = H;
  dn.cols = I;
  dn.c = r + dn_c;
  dn.s = r + dn_s;
  dn.b = fmt == Fmt::AFF4 ? r + dn_b : nullptr;
  if (fmt == Fmt::NVFP4) {
    gu.gmul = nv_gmul(r, trailer, 0);
    dn.gmul = nv_gmul(r, trailer, 1);
  }
}

// ---------------------------------------------------------------------------------------------- readers
static inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
}

void Readers::start(int n, std::atomic<int>** pending, bool skip) {
  pending_ = pending;
  skip_ = skip;
  stop_ = false;
  for (int i = 0; i < 3; i++) {
    q_ns[i].store(0);
    svc_ns[i].store(0);
    njobs[i].store(0);
  }
  for (int i = 0; i < n; i++) th_.emplace_back([this] { loop(); });
}

void Readers::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : th_) t.join();
  th_.clear();
  if (drop_th_.joinable()) {
    {
      std::lock_guard<std::mutex> lk(drop_mu_);
      drop_stop_ = true;
    }
    drop_cv_.notify_all();
    drop_th_.join();
  }
}

void Readers::drops_start(std::atomic<uint8_t>* state) {
  drop_state_ = state;
  drop_stop_ = false;
  drop_th_ = std::thread([this] { drop_loop(); });
}

void Readers::drop_pages(int key, const uint8_t* p, size_t n) {
  if (!drop_state_) {  // (no drop thread: at once)
    msync((void*)p, n, MS_INVALIDATE);
    return;
  }
  drop_state_[key].store(1, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(drop_mu_);
    drops_.push_back({key, p, n});
  }
  drop_cv_.notify_one();
}

void Readers::drops_flush() {
  std::unique_lock<std::mutex> lk(drop_mu_);
  drop_cv_.wait(lk, [&] { return (drops_.empty() && !drop_busy_) || !drop_th_.joinable(); });
}

void Readers::drop_loop() {
  pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);  // kernel work on clean pages, off the compute's and readers' way
  for (;;) {
    Drop d;
    {
      std::unique_lock<std::mutex> lk(drop_mu_);
      drop_busy_ = false;
      drop_cv_.notify_all();  // (drops_flush)
      drop_cv_.wait(lk, [&] { return drop_stop_ || !drops_.empty(); });
      if (drops_.empty()) return;  // stopping, nothing left
      d = drops_.front();
      drops_.pop_front();
      drop_busy_ = true;
    }
    uint8_t want = 1;
    if (drop_state_[d.key].compare_exchange_strong(want, 2, std::memory_order_acq_rel)) {
      msync((void*)d.p, d.n, MS_INVALIDATE);
      drop_state_[d.key].store(0, std::memory_order_release);
      drops_done.fetch_add(1, std::memory_order_relaxed);
    }
  }
}

void Readers::settle_drop(int key) {
  uint8_t want = 1;
  if (drop_state_[key].compare_exchange_strong(want, 0, std::memory_order_acq_rel)) {
    drops_cancelled.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  while (drop_state_[key].load(std::memory_order_acquire) == 2) __builtin_arm_yield();
}

void Readers::tick_(int64_t t) {
  if (dt_last_) dt_ns_[dt_state_] += (uint64_t)std::max<int64_t>(0, t - dt_last_);
  dt_last_ = t;
  const int inf = inflight_[0] + inflight_[1] + inflight_[2];
  const int s = inf > 0 ? 1 + std::min(inf, 4) : (urgent_.empty() && normal_.empty() && bg_.empty() ? 0 : 1);
  if (s >= 2 && dt_state_ < 2) dt_ns_[6]++;
  dt_state_ = s;
}

void Readers::depth_times(uint64_t out[7]) {
  std::lock_guard<std::mutex> lk(mu_);
  const int64_t t = now_ns();
  for (int i = 0; i < 7; i++) out[i] = dt_ns_[i];
  if (dt_last_) out[dt_state_] += (uint64_t)std::max<int64_t>(0, t - dt_last_);
}

bool Readers::saturated() {
  std::lock_guard<std::mutex> lk(mu_);
  return inflight_[0] + inflight_[1] + inflight_[2] >= (int)th_.size() && !(urgent_.empty() && normal_.empty());
}

bool Readers::urgent_backlog() {
  std::lock_guard<std::mutex> lk(mu_);
  return inflight_[0] + inflight_[1] + inflight_[2] >= (int)th_.size() && !urgent_.empty();
}

uint64_t Readers::busy_wall() {
  std::lock_guard<std::mutex> lk(mu_);
  return busy_wall_ns_;
}

void Readers::submit_cls(const Job& j, int cls) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto& q = cls == 0 ? urgent_ : cls == 1 ? normal_ : bg_;
    q.push_back(j);
    q.back().t_sub = now_ns();
    tick_(q.back().t_sub);
  }
  cv_.notify_one();
}

bool Readers::promote(int pool, int slot) {
  std::lock_guard<std::mutex> lk(mu_);
  bool moved = false;
  for (auto* q : {&normal_, &bg_})
    for (auto it = q->begin(); it != q->end();) {
      if (it->pool == pool && it->slot == slot) {
        urgent_.push_back(*it);
        if (q == &bg_) bg_promoted.fetch_add(1, std::memory_order_relaxed);
        it = q->erase(it);
        moved = true;
      } else {
        ++it;
      }
    }
  if (moved) tick_(now_ns());
  if (moved) cv_.notify_all();  // an urgent job may start even where the speculative limits hold readers back
  return moved;
}

bool Readers::cancel(int pool, int slot) {
  std::lock_guard<std::mutex> lk(mu_);
  int q = 0;
  for (auto* d : {&normal_, &bg_})
    for (auto& j : *d) q += j.pool == pool && j.slot == slot;
  if (q == 0 || q != pending_[pool][slot].load(std::memory_order_acquire)) return false;  // a part has started
  for (auto* d : {&normal_, &bg_})
    for (auto it = d->begin(); it != d->end();) it = it->pool == pool && it->slot == slot ? d->erase(it) : it + 1;
  tick_(now_ns());
  return true;
}

void Readers::cancel_stale(long step, std::vector<std::pair<int, int>>& out) {
  std::lock_guard<std::mutex> lk(mu_);
  out.clear();
  for (auto* d : {&normal_, &bg_})
    for (auto& j : *d)
      if (j.step <= step && std::find(out.begin(), out.end(), std::make_pair(j.pool, j.slot)) == out.end()) out.push_back({j.pool, j.slot});
  size_t w = 0;
  for (size_t i = 0; i < out.size(); i++) {
    int q = 0;
    for (auto* d : {&normal_, &bg_})
      for (auto& j : *d) q += j.pool == out[i].first && j.slot == out[i].second;
    if (q == pending_[out[i].first][out[i].second].load(std::memory_order_acquire)) out[w++] = out[i];
  }
  out.resize(w);
  if (out.empty()) return;
  for (auto* d : {&normal_, &bg_})
    for (auto it = d->begin(); it != d->end();)
      it = std::find(out.begin(), out.end(), std::make_pair(it->pool, it->slot)) != out.end() ? d->erase(it) : it + 1;
  tick_(now_ns());
}

bool Readers::can_spec() const {
  if (normal_.empty() || inflight_[1] >= max_spec) return false;
  if (excl && inflight_[0] > 0) return false;
  return true;
}

bool Readers::can_bg() const {
  return !bg_.empty() && urgent_.empty() && normal_.empty() && inflight_[0] == 0 && inflight_[1] <= bg_spec_gate && inflight_[2] < max_bg;
}

bool Readers::bg_open() {
  std::lock_guard<std::mutex> lk(mu_);
  return urgent_.empty() && normal_.empty() && inflight_[0] == 0 && inflight_[1] <= bg_spec_gate;
}

int Readers::bg_queued() {
  std::lock_guard<std::mutex> lk(mu_);
  return (int)bg_.size() + inflight_[2];
}

// Mem::FILE: are all pages of [p, p + n) resident? (mincore; n a multiple of the page size)
static bool all_resident(const uint8_t* p, size_t n) {
  static thread_local std::vector<char> v;
  const size_t pg = (size_t)getpagesize(), np = (n + pg - 1) / pg;
  v.resize(np);
  if (mincore((void*)p, n, v.data()) != 0) return false;
  for (size_t i = 0; i < np; i++)
    if (!(v[i] & MINCORE_INCORE)) return false;
  return true;
}

void Readers::loop() {
  // the reads are DMA; the thread only issues them. This QoS lets macOS keep these off the compute cores' way.
  int q = qos;
  if (q < 0) {  // DMENG_READER_QOS: interactive | initiated (default) | utility | background (placement experiments)
    const char* e = getenv("DMENG_READER_QOS");
    const std::string v = e ? e : "";
    q = v == "interactive" ? QOS_CLASS_USER_INTERACTIVE : v == "utility" ? QOS_CLASS_UTILITY : v == "background" ? QOS_CLASS_BACKGROUND
                                                                                                              : QOS_CLASS_USER_INITIATED;
  }
  pthread_set_qos_class_self_np((qos_class_t)q, 0);
  // io_low (CoreOpts::reader_io_low; default on iOS only): the readers' disk I/O one tier below the threads' own
  // (IOPOL_STANDARD; the CPU QoS above is kept). iOS (P2): at the app's tier, the main thread's page-ins waited behind the
  // queued 1.77 MB records for seconds. Off on the Mac: there the readers are the critical path, and a lower tier is
  // throttled whenever another process touches the drive.
  if (io_low) setiopolicy_np(IOPOL_TYPE_DISK, IOPOL_SCOPE_THREAD, IOPOL_STANDARD);
  std::vector<uint8_t> scratch;  // Mem::FILE: the pread's destination (the pages stay in the file cache)
  for (;;) {
    Job j;
    int cls;
    int64_t t0;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !urgent_.empty() || can_spec() || can_bg(); });
      if (stop_) return;
      if (!urgent_.empty()) {
        cls = 0;
        j = urgent_.front();
        urgent_.pop_front();
        spec_inflight_at_urgent.fetch_add(inflight_[1], std::memory_order_relaxed);
      } else if (!can_spec()) {  // the background class (first in, first out)
        cls = 2;
        j = bg_.front();
        bg_.pop_front();
      } else {
        cls = 1;
        auto best = normal_.begin();
        if (prio)
          for (auto it = normal_.begin(); it != normal_.end(); ++it)
            if (it->step < best->step || (it->step == best->step && it->pp > best->pp)) best = it;
        j = *best;
        normal_.erase(best);
      }
      t0 = now_ns();
      if (inflight_[0] + inflight_[1] + inflight_[2] == 0) busy_t0_ = t0;
      inflight_[cls]++;
      tick_(t0);
    }
    q_ns[cls].fetch_add((uint64_t)std::max<int64_t>(0, t0 - j.t_sub), std::memory_order_relaxed);
    // tests: DMENG_TEST_SPEC_READ_US holds every speculative read (a name's) that long in flight, as a busy machine does
    static const int spec_us = getenv("DMENG_TEST_SPEC_READ_US") ? atoi(getenv("DMENG_TEST_SPEC_READ_US")) : 0;
    if (spec_us > 0 && cls != 0) std::this_thread::sleep_for(std::chrono::microseconds(spec_us));
    bool skip_piece = skip_;
    if (j.key >= 0 && drop_state_) settle_drop(j.key);  // the spare tier: a drop of this record's pages is not wanted now
    if (!skip_piece && !j.dst && file_skip_resident && all_resident(j.view, j.n)) {
      skip_piece = true;
      file_resident_skips.fetch_add(1, std::memory_order_relaxed);
      file_bytes_skipped.fetch_add(j.n, std::memory_order_relaxed);
      if (j.prob && file_prefault) {  // a probation read found resident: its pages mapped here, not in the compute threads
        const size_t pg = (size_t)getpagesize();
        uint64_t acc = 0;
        for (size_t o = 0; o < j.n; o += pg) acc += ((const volatile uint8_t*)j.view)[o];
        (void)acc;
      }
    }
    if (!skip_piece) {
      uint8_t* dst = j.dst;
      if (!dst) {
        if (scratch.size() < j.n) scratch.resize(j.n);
        dst = scratch.data();
      }
      int64_t slot_end = 0;  // drive_bps (bench only): this read's slot on the slower drive's clock
      if (drive_bps > 0) {
        std::lock_guard<std::mutex> g(drive_mu_);
        slot_end = std::max(now_ns(), drive_t_) + (int64_t)((double)j.n * 1e9 / drive_bps);
        drive_t_ = slot_end;
      }
      size_t done = 0;
      while (done < j.n) {
        ssize_t r = pread(j.fd, dst + done, j.n - done, j.off + (long)done);
        if (r <= 0) {
          fprintf(stderr, "read failed (fd %d off %ld): %s\n", j.fd, j.off, strerror(errno));
          abort();
        }
        done += (size_t)r;
      }
      if (slot_end > 0) {
        const int64_t left = slot_end - now_ns();
        if (left > 0) std::this_thread::sleep_for(std::chrono::nanoseconds(left));
      }
      bytes.fetch_add(j.n, std::memory_order_relaxed);
      bytes_cls[cls].fetch_add(j.n, std::memory_order_relaxed);
      if (j.pool >= 0 && j.pool < 4) bytes_pool[j.pool].fetch_add(j.n, std::memory_order_relaxed);
      if (!j.dst && file_prefault) {  // map the cached pages here (soft faults), not in the compute threads
        const size_t pg = (size_t)getpagesize();
        uint64_t acc = 0;
        for (size_t o = 0; o < j.n; o += pg) acc += ((const volatile uint8_t*)j.view)[o];
        (void)acc;
      }
      // a slab read with the spare tier on: the slab holds the record, so the file cache does not (one place)
      if (j.dst && j.drop_view && j.view) msync((void*)j.view, j.n, MS_INVALIDATE);
    }
    const int64_t t1 = now_ns();
    busy_ns.fetch_add((uint64_t)(t1 - t0), std::memory_order_relaxed);
    svc_ns[cls].fetch_add((uint64_t)(t1 - t0), std::memory_order_relaxed);
    njobs[cls].fetch_add(1, std::memory_order_relaxed);
    jobs.fetch_add(1, std::memory_order_relaxed);
    pending_[j.pool][j.slot].fetch_sub(1, std::memory_order_release);
    bool wake;
    {
      std::lock_guard<std::mutex> lk(mu_);
      inflight_[cls]--;
      if (inflight_[0] + inflight_[1] + inflight_[2] == 0) busy_wall_ns_ += (uint64_t)(t1 - busy_t0_);
      tick_(t1);
      wake = ((max_spec < (1 << 30) || excl) && !normal_.empty()) || !bg_.empty();
    }
    if (wake) cv_.notify_all();
  }
}

// ---------------------------------------------------------------------------------------------- cache
bool Cache::init(int NL_, int NE_, int K_, const std::vector<int>& step_layer_, const std::vector<int>& layer_pool_,
                 const std::vector<size_t>& pool_rec, const std::vector<double>& share, const std::string& dir,
                 const std::string& layer_file, double budget, int readers, int split_, Mem mode, bool skip, std::string& err) {
  NL = NL_;
  NE = NE_;
  K = K_;
  step_layer = step_layer_;
  steps = (int)step_layer.size();
  layer_pool = layer_pool_;
  npools = (int)pool_rec.size();
  if (npools < 1 || npools > MAXP || (int)share.size() != npools) {
    err = "cache: 1 to 4 record formats, one budget share each";
    return false;
  }
  mem = mode;
  split = std::max(1, split_);
  skip_reads = skip;
  share_ = share;
  if (mode == Mem::FILE) file_under = false;  // (the whole cache is the file tier already)
  const bool tier = file_under && prob_bytes > 0;  // the spare tier: probation slots past each pool's slab
  prob_slots_ = 0;
  for (int p = 0; p < npools; p++) {
    int nl = 0;
    for (int L = 0; L < NL; L++) nl += layer_pool[L] == p;
    SlotPool& S = P[p];
    S.rec = pool_rec[p];
    S.n = nl ? (int)std::floor(budget * share[p] / (double)S.rec) : 0;
    S.n = std::min(S.n, nl * NE);
    if (nl && S.n < 2 * K + 2) {
      err = "expert budget too small for record format " + std::to_string(p);
      return false;
    }
    S.slab_bytes = (size_t)S.n * S.rec;
    if (mode == Mem::FILE) {
      // bookkeeping only: no slab
    } else if (mode == Mem::PURGEABLE) {
      S.preg.resize(S.n);
      S.nonvol.assign(S.n, 1);
      for (int i = 0; i < S.n; i++) {
        vm_address_t a = 0;
        if (vm_allocate(mach_task_self(), &a, S.rec, VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE) != KERN_SUCCESS) {
          err = "vm_allocate (purgeable) failed";
          return false;
        }
        S.preg[i] = (uint8_t*)a;
      }
    } else if (S.n) {
      S.slab = alloc_region(S.slab_bytes, defer_lock && mode == Mem::MLOCK ? Mem::PLAIN : mode, &err);
      if (!S.slab) return false;
      S.locked.assign(S.n, mode == Mem::MLOCK && !defer_lock ? 1 : 0);  // (alloc_region locked the whole slab, or failed)
      if (mode == Mem::MLOCK && !defer_lock) locked_bytes += (int64_t)S.slab_bytes;
    }
    // probation: as many slots as the pool's share of the budget holds records (at most the pool's records), no memory
    S.np = tier && nl ? (int)std::min<int64_t>((int64_t)std::floor((double)prob_bytes * share[p] / (double)S.rec), (int64_t)nl * NE - S.n) : 0;
    S.np = std::max(0, S.np);
    prob_slots_ += S.np;
    S.s.assign(S.n + S.np, Slot());
    S.pending.reset(new std::atomic<int>[std::max(1, S.n + S.np)]);
    for (int i = 0; i < S.n + S.np; i++) S.pending[i].store(0);
    S.free_.clear();
    for (int i = S.n - 1; i >= 0; i--) S.free_.push_back(i);
    S.pfree.clear();
    S.pfifo.clear();
    for (int i = S.n + S.np - 1; i >= S.n; i--) S.pfree.push_back(i);
    S.cap = S.n;
    // the pools' run-time floor, one rule for every family: one step's pins (2K + 2). Every multi-position MoE runs through
    // Core::moe_chunk, which runs a union larger than the free slots in batches, and early names are only asked for
    S.min_cap = std::min(S.n, 2 * K + 2);
  }
  tab.assign((size_t)NL * NE, -1);
  ev_step.assign((size_t)NL * NE, -1);
  n_reads.assign((size_t)NL * NE, 0);
  ev_pf.assign((size_t)NL * NE, 0);
  pred.assign((size_t)NL * NE, -1.f);
  char buf[4096];
  for (int L = 0; L < NL; L++) {
    snprintf(buf, sizeof buf, layer_file.c_str(), L);
    std::string p = dir + "/" + (L < (int)layer_paths.size() ? layer_paths[L] : std::string(buf));
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) {
      err = "cannot open " + p;
      return false;
    }
    const bool under = prob_slots_ > 0;  // the spare tier: buffered reads, the file mapped (probation records are read there)
    if (mode != Mem::FILE && !under) fcntl(fd, F_NOCACHE, 1);  // FILE: buffered reads, the file cache is the tier
    fcntl(fd, F_RDAHEAD, 0);
    fds.push_back(fd);
    if (mode == Mem::FILE || under) {
      const off_t sz = lseek(fd, 0, SEEK_END);
      void* m = sz > 0 ? mmap(nullptr, (size_t)sz, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
      if (m == MAP_FAILED) {
        err = "cannot map " + p + ": " + strerror(errno);
        return false;
      }
      fmap.push_back((uint8_t*)m);
      fmap_bytes.push_back((size_t)sz);
    }
  }
  file_under = prob_slots_ > 0;  // (the tier is on only with probation slots)
  pageins_at_init = task_pageins();
  for (int p = 0; p < npools; p++) pend_[p] = P[p].pending.get();
  if (file_under) {
    drop_state_.reset(new std::atomic<uint8_t>[(size_t)NL * NE]);
    for (size_t k = 0; k < (size_t)NL * NE; k++) drop_state_[k].store(0);
    rd.drops_start(drop_state_.get());
  }
  rd.start(readers, pend_, skip);
  return true;
}

void Cache::spare_pause() {
  if (!file_under || spare_paused.exchange(true)) return;
  for (int fd : fds) fcntl(fd, F_NOCACHE, 1);  // reads stop leaving pages in the file cache
  // the probation records' pages go at the compute thread's next step boundary (tier_drain): it knows what is in use
}

// ---------------------------------------------------------------------------------------------- the spare tier
uint64_t Cache::task_pageins() {
  task_events_info_data_t e;
  mach_msg_type_number_t c = TASK_EVENTS_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&e, &c) != KERN_SUCCESS) return 0;
  return (uint64_t)e.pageins;
}

int Cache::prob_held(int p) const {
  int k = 0;
  for (int i = P[p].n; i < P[p].n + P[p].np; i++) k += P[p].s[i].layer >= 0;
  return k;
}

uint64_t Cache::prob_held_bytes() const {
  uint64_t b = 0;
  for (int p = 0; p < npools; p++) b += (uint64_t)prob_held(p) * P[p].rec;
  return b;
}

void Cache::free_slot(int p, int slot) {
  SlotPool& S = P[p];
  if (slot >= S.n) S.pfree.push_back(slot);
  else if (slot < S.cap) S.free_.push_back(slot);
}

// a slot for a read into probation: a free one, else the oldest read whose record is not pinned and has landed (its pages
// dropped: the budget); -1 when every probation record is pinned or in flight (the read then goes to the slab)
int Cache::prob_take(int L) {
  const int p = layer_pool[L];
  SlotPool& S = P[p];
  if (!S.pfree.empty()) {
    const int v = S.pfree.back();
    S.pfree.pop_back();
    return v;
  }
  for (auto it = S.pfifo.begin(); it != S.pfifo.end();) {
    const int s = it->first;
    const Slot& x = S.s[s];
    if (x.layer < 0 || x.pseq != it->second) {  // promoted, dropped or read again since
      it = S.pfifo.erase(it);
      continue;
    }
    if (x.pins > 0 || S.pending[s].load(std::memory_order_relaxed) != 0) {
      ++it;
      continue;
    }
    S.pfifo.erase(it);
    drop(L, s, true);  // (its pages invalidated)
    st.n_prob_budget_drops++;
    return s;
  }
  return -1;
}

int Cache::prob_check(int L, int slot) {
  SlotPool& S = pool(L);
  Slot& x = S.s[slot];
  if (S.pending[slot].load(std::memory_order_acquire) != 0 || skip_reads) return slot;  // in flight: the reader maps it
  const uint8_t* v = fmap[L] + (size_t)x.e * S.rec;
  if (all_resident(v, S.rec)) return slot;
  st.n_prob_reclaimed++;  // the OS took (part of) it before this use
  if (x.pins == 0) {      // a miss like any other
    drop(L, slot, true);
    free_slot(layer_pool[L], slot);
    return -1;
  }
  // pinned (a name's, a loop hold's): read again into the same slot now, urgent (pins and statistics unchanged)
  const size_t R = S.rec, piece = ((R / split) + 16383) & ~size_t(16383);
  const int parts = (int)((R + piece - 1) / piece);
  S.pending[slot].store(parts, std::memory_order_release);
  for (int i = 0; i < parts; i++) {
    size_t o = (size_t)i * piece, len = std::min(piece, R - o);
    Readers::Job j{fds[L], (long)((size_t)x.e * R + o), nullptr, len, layer_pool[L], slot};
    j.view = v + o;
    j.prob = true;
    j.key = L * NE + x.e;
    rd.submit(j, true);
  }
  return slot;
}

void Cache::tier_drain() {
  bool left = false;
  for (int p = 0; p < npools; p++) {
    SlotPool& S = P[p];
    for (int i = S.n; i < S.n + S.np; i++) {
      Slot& x = S.s[i];
      if (x.layer < 0) continue;
      if (x.pins > 0 || S.pending[i].load(std::memory_order_acquire) != 0) {
        left = true;
        continue;
      }
      drop(x.layer, i, false);
      S.pfree.push_back(i);
    }
    S.pfifo.clear();  // (rebuilt by nothing: no read lands in probation any more)
  }
  prom_q_.clear();
  prob_drained_ = !left;
}

void Cache::tier_step() {
  if (spare_paused.load(std::memory_order_relaxed)) {
    if (!prob_drained_) tier_drain();
    prom_q_.clear();
    return;
  }
  if (prom_q_.empty() || apart_on) return;  // (a prompt apart: its records move after it)
  const auto t0 = clk::now();
  size_t w = 0;
  for (size_t i = 0; i < prom_q_.size(); i++) {
    const int k = prom_q_[i], L = k / NE;
    const int s = tab[(size_t)k];
    if (s < 0 || !is_prob(L, s) || !pool(L).s[s].promote) continue;  // gone meanwhile (the budget, the OS)
    if (pool(L).s[s].pins > 0 || pool(L).pending[s].load(std::memory_order_acquire) != 0) {
      prom_q_[w++] = k;  // in use (a hold, a name): at a later boundary
      continue;
    }
    promote_one(L, s);
  }
  prom_q_.resize(w);
  st.promote_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// probation slot -> the slab slot a miss would have taken (victim: a free slot first, else the lowest keep score; caps
// kept, never borrowed past), copied from the mapping at memory speed; its file pages dropped after the copy
void Cache::promote_one(int L, int s) {
  SlotPool& S = pool(L);
  Slot& x = S.s[s];
  const int e = x.e;
  const uint8_t* src = fmap[L] + (size_t)e * S.rec;
  if (!all_resident(src, S.rec)) {  // the OS took it: no copy from the drive on the compute thread (a later use misses)
    st.n_prob_reclaimed++;
    drop(L, s, true);
    free_slot(layer_pool[L], s);
    return;
  }
  prob_bypass_ = true;
  const int v = victim(L, 1);  // (reserve 1: a capped layer gives its own record, as a name's request; no borrowing)
  prob_bypass_ = false;
  if (v < 0) {  // no slab slot to give now: stays in probation, tried again at its next reuse
    x.promote = false;
    st.n_promote_refused++;
    return;
  }
  if (mem == Mem::PURGEABLE) touch_nonvolatile(L, v);
  std::memcpy(S.row(v), src, S.rec);
  rd.drop_pages(L * NE + e, src, S.rec);  // one place: the slab (the pages dropped by the readers' drop thread)
  Slot& y = S.s[v];
  y = x;
  y.pins = y.pf_pins = 0;
  y.pf_unused = y.apart = y.promote = false;
  y.ptok = -1;
  y.puses = 0;
  y.pseq = 0;
  tab[(size_t)L * NE + e] = v;
  if (!lheld.empty()) lheld[L]++;
  hz_enter(L, v, e);
  x = Slot();
  S.pfree.push_back(s);
  st.n_promoted++;
}

void Cache::hz_enter(int L, int slot, int e) {
  SlotPool& S = pool(L);
  if (hazard_evict && hzm_ok[L] && slot < S.cap) {  // the layer's minimum, with the new record
    const double v = keep_score(L, e, tok_now);
    const int id = L * NE + e;
    if (hzm_slot[L] < 0 || v < hzm_v[L] || (v == hzm_v[L] && id < hzm_id[L])) {
      hzm_v[L] = v;
      hzm_id[L] = id;
      hzm_slot[L] = slot;
    }
    if (rk.on) {  // and in the ranker's ordered list (an older entry of the same record goes)
      auto& h = hzs[L];
      for (size_t i = 0; i < h.size(); i++)
        if (h[i].id == id) {
          h.erase(h.begin() + i);
          break;
        }
      const HzE x2{v, id, slot};
      h.insert(std::lower_bound(h.begin(), h.end(), x2, [](const HzE& a, const HzE& b) { return a.v < b.v || (a.v == b.v && a.id < b.id); }), x2);
    }
  }
}

Cache::~Cache() {
  rd.stop();
  free_region(core_slab, core_slab_bytes);
  for (size_t i = 0; i < fmap.size(); i++) munmap(fmap[i], fmap_bytes[i]);
  for (int fd : fds) close(fd);
  for (int p = 0; p < npools; p++) {
    free_region(P[p].slab, P[p].slab_bytes);
    for (auto r : P[p].preg) vm_deallocate(mach_task_self(), (vm_address_t)r, P[p].rec);
  }
}

void Cache::drop(int L, int slot, bool count_pf_unused) {
  Slot& b = pool(L).s[slot];
  if (b.layer >= 0) {
    const size_t k = (size_t)b.layer * NE + b.e;
    const bool pr = slot >= pool(L).n;  // a probation record (the spare tier)
    tab[k] = -1;
    hz_gone(b.layer, slot);
    if (!lheld.empty() && !pr) lheld[b.layer]--;
    ev_step[k] = now_step;
    ev_pf[k] = evict_pf_ctx;
    if (count_pf_unused && b.pf_unused) st.n_pf_unused_evicted++;
    if (b.bg_unused) st.n_bg_unused_evicted++;
    if (!floor_until.empty()) floor_until[k] = -1;  // (a later read of the record has no floor)
    // a probation record's pages always leave the file cache with it (the budget is the engine's); Mem::FILE: the hint
    if (pr) rd.drop_pages((int)k, fmap[b.layer] + (size_t)b.e * pool(L).rec, pool(L).rec);
    else if (mem == Mem::FILE && file_evict_hint == 1) madvise(fmap[b.layer] + (size_t)b.e * pool(L).rec, pool(L).rec, MADV_DONTNEED);
    else if (mem == Mem::FILE && file_evict_hint == 2) msync(fmap[b.layer] + (size_t)b.e * pool(L).rec, pool(L).rec, MS_INVALIDATE);
  }
  b = Slot();
}

namespace {
struct PolicyTimer {  // adds the scope's wall time to a counter (ms)
  double& acc;
  clk::time_point t0 = clk::now();
  explicit PolicyTimer(double& a) : acc(a) {}
  ~PolicyTimer() { acc += std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }
};
}  // namespace

void Cache::pre_use(int L, const int32_t* picks, int n, long t) {
  if (!hazard_evict || apart_on) return;
  PolicyTimer pt(st.policy_ms);
  for (int j = 0; j < n; j++) hz_touch(L, picks[j], t);
}

int Cache::victim(int L, int reserve, int fit_names) {
  PolicyTimer pt(st.policy_ms);
  if (prob_slots_ && !prob_bypass_ && prob_on()) {  // the spare tier: every read lands in probation (the slab untouched)
    const int v = prob_take(L);
    if (v >= 0) return v;
  }
  const bool capped = !lcap.empty() && lheld[L] >= lcap[L];  // per-layer caps: at its cap a layer evicts its own
  const bool for_name = reserve > 0;
  if (capped && for_name && cap_fit && fit_names >= 0)
    reserve = std::min(reserve, std::max(0, lcap[L] - fit_names - (layer_hold.empty() ? 0 : layer_hold[L])));
  int v = victim_in(L, reserve, capped);
  if (v < 0 && capped && !for_name) {  // the cap is soft for a slot the step needs now (its pins, a chunk's union):
    v = victim_in(L, reserve, false);     // the pool's choice (a free slot first); names (reserve > 0) still refuse
    if (v >= 0) st.n_cap_borrow++;
  }
  return v;
}

int Cache::victim_in(int L, int reserve, bool capped) {
  SlotPool& S = pool(L);
  if (!capped && !S.free_.empty()) {
    int v = S.free_.back();
    S.free_.pop_back();
    return v;
  }
  if (apart_on && !capped) {  // a prompt apart: its own earlier reads go first
    const int v = apart_victim(L, reserve);
    if (v >= 0) return v;
  }
  int best = -1, unpinned = 0;
  double bs = 0;
  // a slot past a lowered budget (still held because it was in use when the budget fell) is given back when it comes up
  // here and the choice is made again; the last such slot is kept for reuse if nothing within the budget can go (a step
  // whose pins fill the budget must still get its slot: the budget is then exceeded by that slot, as before)
  int spare = -1;
  auto past_cap = [&](int v) {
    if (v < S.cap) return false;
    drop(L, v, true);
    if (spare >= 0) slot_memory(layer_pool[L], spare, false);
    spare = v;
    return true;
  };
  auto settle = [&](int v) {  // v: the in-budget choice (or -1); returns the slot to use
    if (v >= 0) {
      if (spare >= 0) slot_memory(layer_pool[L], spare, false);
      return v;
    }
    return spare;  // dropped already; its memory stays in use
  };
  if (rk.on) {  // the learned ranker over hazard's shortlist
    hz_tick(tok_now);
    do best = rk_victim(L, capped, reserve);
    while (best >= 0 && past_cap(best));
    if (best >= 0) drop(L, best, true);
    return settle(best);
  }
  if (hazard_evict) {  // the hazard keep-score (it carries the last use): the lowest goes, ties to the lower record id
    hz_tick(tok_now);
    do best = hz_victim(L, capped, reserve);
    while (best >= 0 && past_cap(best));
    if (hz_check) {
      const int ref = hz_scan(L, capped, reserve);
      if (ref != best) {
        fprintf(stderr, "hazard victim: incremental %d, scan %d (layer %d, token %ld, reserve %d)\n", best, ref, L, tok_now,
                reserve);
        abort();
      }
    }
    if (best >= 0) drop(L, best, true);
    return settle(best);
  }
  for (int i = 0; i < S.cap; i++) {
    const Slot& x = S.s[i];
    if (x.pins > 0 || S.pending[i].load(std::memory_order_relaxed) != 0 || (capped && x.layer != L)) continue;
    unpinned++;
    if (learned_evict && x.layer >= 0) {  // the learned keep-score: the lowest goes (recency breaks ties)
      const double v = keep_score(x.layer, x.e, tok_now);
      if (best < 0 || v < bs || (v == bs && x.last < S.s[best].last)) {
        best = i;
        bs = v;
      }
      continue;
    }
    if (best < 0 || x.key < S.s[best].key || (x.key == S.s[best].key && x.last < S.s[best].last)) best = i;
  }
  if (best < 0 || unpinned <= reserve) return -1;
  drop(L, best, true);
  return best;
}

// the hazard victim as the slot scan chose it: over the unpinned, landed slots of the pool (with caps: the layer's own),
// the lowest keep score, ties to the lower record id; -1 when none or when the unpinned slots are <= reserve
int Cache::hz_scan(int L, bool capped, int reserve) const {
  const SlotPool& S = pool(L);
  int best = -1, unpinned = 0;
  double bs = 0;
  for (int i = 0; i < S.cap; i++) {
    const Slot& x = S.s[i];
    if (x.pins > 0 || S.pending[i].load(std::memory_order_relaxed) != 0 || (capped && x.layer != L)) continue;
    unpinned++;
    if (x.layer < 0) continue;
    const double v = keep_score(x.layer, x.e, tok_now);
    if (best < 0 || v < bs || (v == bs && x.layer * NE + x.e < S.s[best].layer * NE + S.s[best].e)) {
      best = i;
      bs = v;
    }
  }
  return best < 0 || unpinned <= reserve ? -1 : best;
}

// layer L's lowest (keep score, record id) over its held records in slots below the pool's cap (eligible: also not
// pinned and landed); records in increasing id, so a strict < keeps the lower id on a tie
void Cache::hz_layer_min(int L, bool eligible, double& v, int& id, int& slot) const {
  const SlotPool& S = pool(L);
  const int32_t* tb = tab.data() + (size_t)L * NE;
  slot = -1;
  for (int e = 0; e < NE; e++) {
    const int s = tb[e];
    if (s < 0 || s >= S.cap) continue;
    if (eligible && (S.s[s].pins > 0 || S.pending[s].load(std::memory_order_relaxed) != 0)) continue;
    const double x = keep_score(L, e, tok_now);
    if (slot < 0 || x < v) {
      v = x;
      id = L * NE + e;
      slot = s;
    }
  }
}

int Cache::hz_victim(int L, bool capped, int reserve) {
  SlotPool& S = pool(L);
  const std::vector<int>& ls = pool_layers[layer_pool[L]];
  const int n = capped ? 1 : (int)ls.size();
  struct Cand {
    double v;
    int id, slot;
    bool eligible_min;  // searched over eligible records already
  };
  static thread_local std::vector<Cand> c;
  c.resize(n);
  for (int i = 0; i < n; i++) {
    const int l = capped ? L : ls[i];
    if (!hzm_ok[l]) {
      hz_layer_min(l, false, hzm_v[l], hzm_id[l], hzm_slot[l]);
      hzm_ok[l] = 1;
    }
    c[i] = {hzm_v[l], hzm_id[l], hzm_slot[l], false};
  }
  int best = -1;
  for (;;) {
    int b = -1;
    for (int i = 0; i < n; i++)
      if (c[i].slot >= 0 && (b < 0 || c[i].v < c[b].v || (c[i].v == c[b].v && c[i].id < c[b].id))) b = i;
    if (b < 0) break;
    const int s = c[b].slot;
    if (c[b].eligible_min || (S.s[s].pins == 0 && S.pending[s].load(std::memory_order_relaxed) == 0)) {
      best = s;
      break;
    }
    hz_layer_min(capped ? L : ls[b], true, c[b].v, c[b].id, c[b].slot);  // the winner is pinned or in flight
    c[b].eligible_min = true;
  }
  if (best < 0) return -1;
  if (reserve > 0) {  // the scan's condition: more than `reserve` unpinned, landed slots (free ones count, caps: own layer)
    int unpinned = 0;
    if (capped) {
      const int32_t* tb = tab.data() + (size_t)L * NE;
      for (int e = 0; e < NE && unpinned <= reserve; e++) {
        const int s = tb[e];
        if (s >= 0 && s < S.cap && S.s[s].pins == 0 && S.pending[s].load(std::memory_order_relaxed) == 0) unpinned++;
      }
    } else {
      for (int i = 0; i < S.cap && unpinned <= reserve; i++)
        if (S.s[i].pins == 0 && S.pending[i].load(std::memory_order_relaxed) == 0) unpinned++;
    }
    if (unpinned <= reserve) return -1;
  }
  return best;
}

int Cache::pick_victim(int L, int reserve, double* val) const {
  const SlotPool& S = pool(L);
  if (!S.free_.empty()) {
    if (val) *val = -1.0;
    return S.free_.back();
  }
  const long t = now_step / steps;
  const int cur = step_layer[now_step % steps];  // the running layer's estimates are about this pass, not its next
  int best = -1, unpinned = 0;
  double bv = 0;
  for (int i = 0; i < S.cap; i++) {
    const Slot& x = S.s[i];
    if (x.pins > 0 || S.pending[i].load(std::memory_order_relaxed) != 0) continue;
    unpinned++;
    double v;
    if (x.layer < 0) {
      v = -1.0;
    } else {
      float pr = pred[(size_t)x.layer * NE + x.e];
      const float u = use_tab[vbin(x.key, t)];
      v = (pr >= 0.f && x.layer != cur) ? pr + u : 2.f * u;  // next pass + the passes after it (frequency estimate)
    }
    if (best < 0 || v < bv || (v == bv && (x.key < S.s[best].key || (x.key == S.s[best].key && x.last < S.s[best].last)))) {
      best = i;
      bv = v;
    }
  }
  if (best < 0 || unpinned <= reserve) return -1;
  if (val) *val = bv;
  return best;
}

void Cache::evict(int L, int slot) {
  SlotPool& S = pool(L);
  if (!S.free_.empty() && S.free_.back() == slot) {
    S.free_.pop_back();
    return;
  }
  drop(L, slot, true);
}

int Cache::steal(int L) {
  SlotPool& S = pool(L);
  int best = -1;
  for (int i = 0; i < S.cap; i++) {
    const Slot& x = S.s[i];
    if (x.pins == 0 || x.pins != x.pf_pins || S.pending[i].load(std::memory_order_relaxed) != 0) continue;
    if (best < 0 || x.key < S.s[best].key || (x.key == S.s[best].key && x.last < S.s[best].last)) best = i;
  }
  if (best < 0) return -1;
  drop(L, best, false);
  st.n_pf_stolen++;
  return best;
}

// A miss that finds no landed slot to evict or steal: the slots not pinned by a step may still be reads in flight, the
// names' (a run-ahead's names are unpinned at their step, a router's or an oracle's stay pinned by the name only). Once one
// lands, victim or steal can take it: wait for the first such read (promoted to urgent). Which slot a miss gets never
// changes the outputs, so waiting is always correct; only a pool whose every slot a step itself pins has none.
bool Cache::wait_name_read(int L) {
  SlotPool& S = pool(L);
  for (int i = 0; i < S.cap; i++) {
    const Slot& x = S.s[i];
    if (x.pins == x.pf_pins && S.pending[i].load(std::memory_order_acquire) != 0) {
      wait(L, i);
      return true;
    }
  }
  return false;
}

void Cache::issue(int L, int slot, int e, bool urgent, long step, float pp) {
  DM_WC("cache.issue", 1);
  SlotPool& S = pool(L);
  Slot& x = S.s[slot];
  x.layer = L;
  x.e = e;
  x.key = -1e300;
  x.last = -1;
  const bool pr = slot >= S.n;  // into probation (the spare tier): not a slab slot, not counted in a layer's cap
  x.apart = !pr && apart_on && !bg_issue;
  if (x.apart) apart_q[layer_pool[L]].push_back(slot);
  tab[(size_t)L * NE + e] = slot;
  if (!lheld.empty() && !pr) lheld[L]++;
  hz_enter(L, slot, e);
  if (pr) {
    x.ptok = -1;
    x.puses = 0;
    x.promote = false;
    x.pseq = ++prob_seq_;
    S.pfifo.push_back({slot, x.pseq});
  }
  (urgent ? st.bytes_urgent : st.bytes_pf) += S.rec;
  n_reads[(size_t)L * NE + e]++;
  if (bg_issue) st.bytes_bg += S.rec;
  if (skip_reads) return;  // timing without reads: the slot keeps whatever bytes it has
  const int sp = bg_issue ? std::max(1, bg_split) : split;
  const size_t R = S.rec, piece = ((R / sp) + 16383) & ~size_t(16383);
  const int parts = (int)((R + piece - 1) / piece);
  S.pending[slot].store(parts, std::memory_order_release);
  uint8_t* dst = mem == Mem::FILE || pr ? nullptr : S.row(slot);
  const uint8_t* view = !fmap.empty() ? fmap[L] + (size_t)e * R : nullptr;  // (FILE, or the spare tier's mapping)
  // a slab read while the spare tier is on (probation full of pins): its buffered pread's pages dropped after it
  const bool drop_view = dst && view && prob_slots_ && !spare_paused.load(std::memory_order_relaxed);
  for (int i = 0; i < parts; i++) {
    size_t o = (size_t)i * piece, len = std::min(piece, R - o);
    Readers::Job j{fds[L], (long)((size_t)e * R + o), dst ? dst + o : nullptr, len, layer_pool[L], slot};
    if (view) j.view = view + o;
    j.drop_view = drop_view;
    j.prob = pr;
    j.key = prob_slots_ ? L * NE + e : -1;
    j.step = urgent ? 0 : step;
    j.pp = pp;
    rd.submit_cls(j, urgent ? 0 : read_bg || bg_issue ? 2 : 1);
  }
}

void Cache::wait(int L, int slot) {
  if (ready(L, slot)) return;
  if (rd.promote(layer_pool[L], slot)) st.n_late_queued++;
  auto t0 = clk::now();
#if DM_COUNTERS
  uint64_t sa_[2], sb_[2];
  dmc_now(sa_);
#endif
  if (park_us >= 0 && on_long_wait) {
    const auto lim = t0 + std::chrono::microseconds(park_us);
    while (!ready(L, slot) && clk::now() < lim) __builtin_arm_yield();
    if (!ready(L, slot)) on_long_wait(long_wait_ctx);
  }
  while (!ready(L, slot)) __builtin_arm_yield();
#if DM_COUNTERS
  dmc_now(sb_);
  dmc_spin(sa_, sb_);
#endif
  st.wait_ms += ms_since(t0);
}

void Cache::use(int L, int slot, long t_use, long step) {
  Slot& x = pool(L).s[slot];
  const long t = chunk_t >= 0 ? chunk_t : t_use;  // the time-ordered statistics' clock
  if (hazard_evict && x.layer >= 0 && !apart_on) {
    if (chunk_t >= 0) chunk_touch.push_back({step, x.layer, x.e, t_use});  // replayed in time order at the pass's end
    else hz_touch(x.layer, x.e, t);
  }
  if (!apart_on) x.apart = false;  // a decode step's record now
  if (learned_evict && x.layer >= 0) {  // the policy's use statistics (decayed counts at three half-lives, last use)
    static const double hl[3] = {2.0, 8.0, 32.0};
    const size_t k = (size_t)x.layer * NE + x.e;
    for (int h = 0; h < 3; h++) lc[h][k] = lc[h][k] * std::exp2(-((double)t - lc_t[k]) / hl[h]) + 1.0;
    lc_t[k] = (double)t;
    l_last[k] = (double)t;
  }
  if (rk.on && x.layer >= 0) {  // the ranker's use statistics (eviction_v3.py Feats.step_end)
#pragma clang fp contract(off)
    const size_t N = (size_t)NL * NE, k = (size_t)x.layer * NE + x.e;
    const double dt = (double)t - rk_ct[k];
    for (size_t i = 0; i < rk.hl_c.size(); i++) rk_cnt[i * N + k] = rk_cnt[i * N + k] * std::exp2(-dt / rk.hl_c[i]) + 1.0;
    rk_ct[k] = (double)t;
    rk_l2[k] = rk_l1[k];
    rk_l1[k] = (double)t;
    rk_gen[x.layer]++;
  }
  // decayed count at t: v = 2^(key - t / hl); after this use v + 1
  double v = x.key < -1e299 ? 0.0 : std::exp2(x.key - (double)t_use / half_life);
  x.key = std::log2(v + 1.0) + (double)t_use / half_life;
  x.last = step;
  x.pf_unused = false;
  if (x.bg_unused) {
    x.bg_unused = false;
    st.n_bg_used++;
  }
  if (slot >= pool(L).n && x.layer >= 0) {  // a probation record: served from the file cache after its first use;
    if (x.puses++ > 0) {                    // a use at a later token than its first (the statistics' clock) promotes it
      st.n_prob_hit++;
      st.bytes_prob_hit += pool(L).rec;
    }
    if (!apart_on) {
      if (x.ptok < 0) x.ptok = t;
      else if (t > x.ptok && !x.promote && prob_on()) {
        x.promote = true;
        prom_q_.push_back(x.layer * NE + x.e);
      }
    }
  }
}

int Cache::name(int L, int e, long step, int reserve, int fit_names) {
  DM_WC("cache.name_calls", 1);
  SlotPool& S = pool(L);
  int sl = tab[(size_t)L * NE + e];
  if (sl >= 0) {
    S.s[sl].pins++;
    S.s[sl].pf_pins++;
    return sl;
  }
  if (name_gate && !skip_reads && rd.urgent_backlog()) {
    st.n_pf_gated++;
    return -1;
  }
  evict_pf_ctx = true;
  int v = victim(L, reserve >= 0 ? reserve : 2 * K + extra_reserve[layer_pool[L]], fit_names);  // a name leaves the slots the misses and holds need
  evict_pf_ctx = false;
  if (v < 0) {
    st.n_pf_dropped++;
    return -1;
  }
  if (mem == Mem::PURGEABLE) touch_nonvolatile(L, v);
  issue(L, v, e, false, step);
  S.s[v].pins = 1;
  S.s[v].pf_pins = 1;
  S.s[v].pf_unused = true;
  st.n_pf_issued++;
  return v;
}

bool Cache::read_bg_value(int L, int e, double value, long until) {
  const size_t k = (size_t)L * NE + e;
  if (tab[k] >= 0 || !hazard_evict || rk.on || mem == Mem::PURGEABLE) return false;
  if (floor_v.empty()) {
    floor_v.assign((size_t)NL * NE, 0.f);
    floor_until.assign((size_t)NL * NE, -1);
  }
  SlotPool& S = pool(L);
  // per-layer caps: a layer at its cap takes a background read's slot from its own records, as a name does (the same
  // reserve, the same value test); below its cap the pool's choice, as without caps
  const bool capped = !lcap.empty() && lheld[L] >= lcap[L];
  int v = -1;
  if (prob_on()) v = prob_take(L);  // the spare tier: into probation (no slab record displaced: no value test)
  if (v >= 0) {
  } else if (!capped && !S.free_.empty()) {
    v = S.free_.back();
    S.free_.pop_back();
  } else {
    PolicyTimer pt(st.policy_ms);
    hz_tick(tok_now);
    // the reserve: a name pins its slot until its step runs, so names leave 2K + the holds unpinned; a background read
    // pins nothing (a step's miss takes its slot once landed, or borrows past a capped layer's cap while it is in flight),
    // so in a layer at its cap it leaves K: the 2K + holds of a name refused every background read in the
    // layers the routers had just named (8 of a 20-slot cap pinned at the 35B's phone size)
    int reserve = 2 * K + extra_reserve[layer_pool[L]];
    if (capped) reserve = std::min(reserve, K);
    v = hz_victim(L, capped, reserve);
    if (v < 0) return false;
    const Slot& x = S.s[v];
    if (x.layer >= 0 && keep_score(x.layer, x.e, tok_now) >= value) {
      st.n_bg_refused++;
      return false;
    }
    evict_pf_ctx = true;
    drop(L, v, true);
    evict_pf_ctx = false;
  }
  if (floor_until[k] < now_step) floor_keys.push_back((int)k);
  floor_v[k] = (float)value;
  floor_until[k] = until;
  bg_issue = true;
  issue(L, v, e, false, until);
  bg_issue = false;
  S.s[v].pins = 0;
  S.s[v].bg_unused = true;
  if (!hzm_ok.empty()) hzm_ok[L] = 0;  // its score has the floor now
  st.n_bg_issued++;
  return true;
}

void Cache::expire_floors() {
  size_t w = 0;
  for (size_t i = 0; i < floor_keys.size(); i++) {
    const int k = floor_keys[i];
    if (floor_until[k] >= now_step) {
      floor_keys[w++] = k;
      continue;
    }
    floor_until[k] = -1;
    if (!hzm_ok.empty()) hzm_ok[k / NE] = 0;
  }
  floor_keys.resize(w);
}

bool Cache::drop_name(int L, int slot, int e) {
  unpin_pf(L, slot, e);
  const Slot& x = pool(L).s[slot];
  if (x.layer != L || x.e != e || x.pins > 0 || ready(L, slot)) return false;
  return cancel(L, slot);
}

void Cache::unpin_pf(int L, int slot, int e) {
  Slot& x = pool(L).s[slot];
  if (x.layer != L || x.e != e || x.pf_pins <= 0) return;  // taken since
  x.pf_pins--;
  x.pins--;
}

bool Cache::cancel(int L, int slot) {
  if (skip_reads || !rd.cancel(layer_pool[L], slot)) return false;
  SlotPool& S = pool(L);
  Slot& b = S.s[slot];
  if (b.layer >= 0) {
    tab[(size_t)b.layer * NE + b.e] = -1;
    n_reads[(size_t)b.layer * NE + b.e]--;  // (cancelled before it ran: not a read)
    if (!lheld.empty() && slot < S.n) lheld[b.layer]--;
    hz_gone(b.layer, slot);
  }
  b = Slot();
  S.pending[slot].store(0, std::memory_order_release);
  free_slot(layer_pool[L], slot);
  st.bytes_pf -= S.rec;
  st.n_pf_cancelled++;
  return true;
}

int Cache::cancel_stale(long step) {
  if (skip_reads) return 0;
  static thread_local std::vector<std::pair<int, int>> v;
  rd.cancel_stale(step, v);
  for (auto& ps : v) {
    SlotPool& S = P[ps.first];
    Slot& b = S.s[ps.second];
    if (b.layer >= 0) {
      tab[(size_t)b.layer * NE + b.e] = -1;
      n_reads[(size_t)b.layer * NE + b.e]--;
      if (!lheld.empty() && ps.second < S.n) lheld[b.layer]--;
      hz_gone(b.layer, ps.second);
    }
    b = Slot();
    S.pending[ps.second].store(0, std::memory_order_release);
    free_slot(ps.first, ps.second);
    st.bytes_pf -= S.rec;
    st.n_pf_cancelled++;
  }
  return (int)v.size();
}

void Cache::learn_use(int L, const int32_t* picks, long t) {
  for (int e = 0; e < NE; e++) {
    int sl = tab[(size_t)L * NE + e];
    if (sl < 0 || !ready(L, sl)) continue;
    bool used = false;
    for (int j = 0; j < K; j++) used |= picks[j] == e;
    int b = vbin(pool(L).s[sl].key, t);
    use_cnt[b] += 1;
    use_pos[b] += used;
  }
}

void Cache::refresh_use_tab() {
  // per-bin rate with a weak prior, made non-decreasing in the count (pool adjacent violators)
  double r[VB], w[VB];
  for (int b = 0; b < VB; b++) {
    w[b] = use_cnt[b] + 2.0;
    r[b] = (use_pos[b] + 2.0 * (b == 0 ? 0.02 : std::min(0.9, 0.02 * std::pow(1.6, b)))) / w[b];
  }
  int lo[VB], hi[VB], nb = 0;
  double rv[VB], wv[VB];
  for (int b = 0; b < VB; b++) {
    lo[nb] = hi[nb] = b;
    rv[nb] = r[b];
    wv[nb] = w[b];
    nb++;
    while (nb > 1 && rv[nb - 2] > rv[nb - 1]) {
      double ww = wv[nb - 2] + wv[nb - 1];
      rv[nb - 2] = (rv[nb - 2] * wv[nb - 2] + rv[nb - 1] * wv[nb - 1]) / ww;
      wv[nb - 2] = ww;
      hi[nb - 2] = hi[nb - 1];
      nb--;
    }
  }
  for (int k = 0; k < nb; k++)
    for (int b = lo[k]; b <= hi[k]; b++) use_tab[b] = (float)rv[k];
}

void Cache::touch_nonvolatile(int L, int slot) {
  SlotPool& S = pool(L);
  if (S.preg.empty() || slot >= S.n || S.nonvol[slot]) return;  // (probation slots have no memory of their own)
  int state = VM_PURGABLE_NONVOLATILE;
  vm_purgable_control(mach_task_self(), (vm_address_t)S.preg[slot], VM_PURGABLE_SET_STATE, &state);
  S.nonvol[slot] = 1;
  if ((state & VM_PURGABLE_STATE_MASK) == VM_PURGABLE_EMPTY && S.s[slot].layer >= 0) {
    // the system took this record: it is not held any more
    st.n_purged++;
    tab[(size_t)S.s[slot].layer * NE + S.s[slot].e] = -1;
    if (!lheld.empty()) lheld[S.s[slot].layer]--;
    hz_gone(S.s[slot].layer, slot);
    int p = S.s[slot].pins;
    S.s[slot] = Slot();
    S.s[slot].pins = p;
  }
}

void Cache::token_end_volatile(long t) {
  // records not used in the last two tokens become volatile (the system may drop them instead of compressing them)
  for (int p = 0; p < npools; p++) {
    SlotPool& S = P[p];
    if (S.preg.empty()) continue;
    for (int i = 0; i < S.n; i++) {
      if (!S.nonvol[i] || S.s[i].pins > 0 || S.pending[i].load(std::memory_order_acquire) != 0 || S.s[i].layer < 0) continue;
      if (S.s[i].last >= 0 && S.s[i].last / steps >= t - 1) continue;
      int state = VM_PURGABLE_VOLATILE;
      vm_purgable_control(mach_task_self(), (vm_address_t)S.preg[i], VM_PURGABLE_SET_STATE, &state);
      S.nonvol[i] = 0;
    }
  }
}

int Cache::held(int p) const {  // the slab's records (probation: prob_held)
  int k = 0;
  for (int i = 0; i < P[p].n; i++) k += P[p].s[i].layer >= 0;
  return k;
}

// ---------------------------------------------------------------------------------------------- Mem::FILE
void Cache::file_hit_check(int L, int slot) {
  if (mem != Mem::FILE || file_check <= 0 || skip_reads) return;
  SlotPool& S = pool(L);
  const Slot& x = S.s[slot];
  if (core_n && core_tab[(size_t)L * NE + x.e] >= 0) return;  // served from the locked core
  const uint8_t* v = fmap[L] + (size_t)x.e * S.rec;
  st.file_checks++;
  bool ok;
  if (file_check >= 2) {
    ok = all_resident(v, S.rec);
  } else {
    const size_t pg = (size_t)getpagesize();
    char a = 0, b = 0;
    ok = mincore((void*)v, pg, &a) == 0 && mincore((void*)(v + S.rec - pg), pg, &b) == 0 && (a & MINCORE_INCORE) && (b & MINCORE_INCORE);
  }
  if (ok) return;
  // the OS dropped (part of) it: read it again now, as an urgent miss into the same slot (pins and policy unchanged)
  st.file_refaults++;
  st.bytes_refault += S.rec;
  const size_t R = S.rec, piece = ((R / split) + 16383) & ~size_t(16383);
  const int parts = (int)((R + piece - 1) / piece);
  S.pending[slot].store(parts, std::memory_order_release);
  for (int i = 0; i < parts; i++) {
    size_t o = (size_t)i * piece, len = std::min(piece, R - o);
    Readers::Job j{fds[L], (long)((size_t)x.e * R + o), nullptr, len, layer_pool[L], slot};
    j.view = v + o;
    rd.submit(j, true);
  }
}

uint64_t Cache::file_resident_bytes() const {
  const size_t pg = (size_t)getpagesize();
  std::vector<char> v;
  uint64_t n = 0;
  for (size_t i = 0; i < fmap.size(); i++) {
    v.resize((fmap_bytes[i] + pg - 1) / pg);
    if (mincore(fmap[i], fmap_bytes[i], v.data()) != 0) continue;
    for (char c : v) n += (c & MINCORE_INCORE) ? pg : 0;
  }
  return n;
}

uint64_t Cache::held_bytes() const {
  uint64_t b = 0;
  for (int p = 0; p < npools; p++) b += (uint64_t)held(p) * P[p].rec;
  return b;
}

// ---------------------------------------------------------------------------------------------- run-time budget
bool Cache::slot_memory(int p, int i, bool keep) {
  SlotPool& S = P[p];
  if (mem == Mem::FILE) return true;  // bookkeeping only
  if (mem == Mem::PURGEABLE) {
    int state = keep ? VM_PURGABLE_NONVOLATILE : VM_PURGABLE_EMPTY;
    vm_purgable_control(mach_task_self(), (vm_address_t)S.preg[i], VM_PURGABLE_SET_STATE, &state);
    S.nonvol[i] = keep;
    return true;
  }
  uint8_t* r = S.slab + (size_t)i * S.rec;  // records are whole pages (FORMAT.md: multiples of 16 KB)
  if (keep) {
    madvise(r, S.rec, MADV_FREE_REUSE);
    if (mem == Mem::MLOCK && !S.locked[i]) {
      if (mlock(r, S.rec) != 0) {
        lock_refused++;
        return false;
      }
      S.locked[i] = 1;
      locked_bytes += (int64_t)S.rec;
    }
  } else {
    if (mem == Mem::MLOCK && S.locked[i]) {
      munlock(r, S.rec);
      S.locked[i] = 0;
      locked_bytes -= (int64_t)S.rec;
    }
    madvise(r, S.rec, MADV_FREE_REUSABLE);
  }
  return true;
}

std::string Cache::lock_json() const {
  char b[160];
  snprintf(b, sizeof b, "{\"locked\":%s,\"locked_gb\":%.4f,\"budget_gb\":%.4f,\"refused\":%llu}", mem == Mem::MLOCK ? "true" : "false",
           locked_bytes / 1e9, budget_bytes() / 1e9, (unsigned long long)lock_refused);
  return b;
}

void Cache::apply_budget() {
  const int64_t req = budget_req.exchange(-1, std::memory_order_acq_rel);
  bool again = false;
  for (int p = 0; p < npools; p++) {
    SlotPool& S = P[p];
    if (!S.n) continue;
    int want = S.cap;
    if (req >= 0) want = std::max(S.min_cap, std::min(S.n, (int)std::floor((double)req * share_[p] / (double)S.rec)));
    if (want > S.cap) {  // grow: slots [cap, want) come back (those still holding a record stay held)
      for (int i = S.cap; i < want; i++) {
        if (S.s[i].layer < 0 && S.s[i].pins == 0 && S.pending[i].load(std::memory_order_acquire) == 0) {
          if (!slot_memory(p, i, true)) {  // the OS refused the lock: the budget stops below this slot
            slot_memory(p, i, false);
            want = i;
            break;
          }
          S.free_.push_back(i);
        }
      }
      S.cap = want;
      continue;
    }
    if (want < S.cap && mem == Mem::FILE) file_compact(p, want);
    if (want < S.cap) {
      S.free_.erase(std::remove_if(S.free_.begin(), S.free_.end(), [&](int i) { return i >= want; }), S.free_.end());
      S.cap = want;
    }
    // slots >= cap: memory given back; one still holding a record is dropped first once unpinned and landed (else at
    // the next boundary). Idempotent for slots given back already.
    for (int i = S.cap; i < S.n; i++) {
      Slot& x = S.s[i];
      if (x.pins > 0 || S.pending[i].load(std::memory_order_acquire) != 0) {
        again = true;
        continue;
      }
      if (x.layer >= 0) drop(x.layer, i, true);
      slot_memory(p, i, false);
    }
  }
  if (caps_auto) caps_auto_apply();
  hz_dirty_all();  // the caps moved
  if (again) {  // try again at the next boundary unless a newer request came in meanwhile
    int64_t none = -1;
    int64_t cur = 0;
    for (int p = 0; p < npools; p++) cur += (int64_t)P[p].cap * (int64_t)P[p].rec;
    budget_req.compare_exchange_strong(none, req >= 0 ? req : cur);
  }
}

void Cache::lock_start(int64_t bytes) {
  // a refused lock stops the start budget there (never below a pool's floor: those slots are used unlocked, reported)
  int64_t got = 0;
  bool refused = false;
  for (int p = 0; p < npools; p++) {
    SlotPool& S = P[p];
    if (!S.n) continue;
    const int want = std::max(S.min_cap, std::min(S.n, (int)std::floor((double)bytes * share_[p] / (double)S.rec)));
    int i = 0;
    for (; i < want; i++)
      if (!slot_memory(p, i, true) && i >= S.min_cap) {
        slot_memory(p, i, false);
        refused = true;
        break;
      }
    got += (int64_t)i * (int64_t)S.rec;
  }
  set_budget(refused ? std::min(bytes, got) : bytes);
  apply_budget();  // the slots past the start budget: never locked, given back as reusable
}

void Cache::lend(int64_t bytes) {
  bytes = std::max<int64_t>(0, bytes);
  if (bytes == lent) return;
  if (lent == 0) lend_base = budget_bytes();  // the budget without the loan (restored exactly: no slot lost to rounding)
  const int64_t base = lend_base;
  lent = bytes;
  set_budget(std::max<int64_t>(0, base - bytes));
  apply_budget();
}

void Cache::lend_hold() {
  if (!lent || mem == Mem::FILE || mem == Mem::PURGEABLE) return;
  for (int p = 0; p < npools; p++) {
    SlotPool& S = P[p];
    for (int i = S.cap; i < S.n; i++)
      if (S.s[i].layer < 0 && S.s[i].pins == 0 && S.pending[i].load(std::memory_order_acquire) == 0) slot_memory(p, i, false);
  }
}

int64_t Cache::budget_bytes() const {
  int64_t b = 0;
  for (int p = 0; p < npools; p++) b += (int64_t)P[p].cap * (int64_t)P[p].rec;
  return b;
}

// FILE shrink: the policy chooses what goes (lowest keep value first, as eviction would), then the survivors above
// `want` move into free slots below it (bookkeeping only: no bytes move). Pinned or in-flight entries stay.
void Cache::file_compact(int p, int want) {
  SlotPool& S = P[p];
  if (hazard_evict) hz_tick(tok_now);
  std::vector<std::pair<double, int>> v;  // (value, slot) of unpinned landed entries
  int held_n = 0;
  for (int i = 0; i < S.n; i++) {
    const Slot& x = S.s[i];
    if (x.layer < 0) continue;
    held_n++;
    if (x.pins > 0 || S.pending[i].load(std::memory_order_acquire) != 0) continue;
    const double val = learned_evict || hazard_evict ? keep_score(x.layer, x.e, tok_now) : x.key;
    v.push_back({val, i});
  }
  std::sort(v.begin(), v.end(), [&](const std::pair<double, int>& a, const std::pair<double, int>& b) {
    return a.first < b.first || (a.first == b.first && S.s[a.second].last < S.s[b.second].last);
  });
  for (size_t k = 0; k < v.size() && held_n > want; k++, held_n--) {
    const int i = v[k].second;
    drop(S.s[i].layer, i, true);
  }
  std::vector<int> low;  // empty slots below want
  for (int i = 0; i < want; i++)
    if (S.s[i].layer < 0 && S.s[i].pins == 0 && S.pending[i].load(std::memory_order_acquire) == 0) low.push_back(i);
  for (int i = want; i < S.n && !low.empty(); i++) {
    Slot& x = S.s[i];
    if (x.layer < 0 || x.pins > 0 || S.pending[i].load(std::memory_order_acquire) != 0) continue;
    const int j = low.back();
    low.pop_back();
    S.s[j] = x;
    tab[(size_t)x.layer * NE + x.e] = j;
    x = Slot();
  }
  hz_dirty_all();  // records moved slots
  S.free_.clear();  // rebuilt from what is empty below want
  for (int i = want - 1; i >= 0; i--)
    if (S.s[i].layer < 0 && S.s[i].pins == 0 && S.pending[i].load(std::memory_order_acquire) == 0) S.free_.push_back(i);
}

bool Cache::core_init(int64_t bytes, std::string& err) {
  if (mem != Mem::FILE || bytes <= 0) return true;
  const size_t R = P[0].rec;
  core_n = (int)std::min<int64_t>(bytes / (int64_t)R, (int64_t)P[0].n / 2);
  if (core_n <= 0) return true;
  core_slab_bytes = (size_t)core_n * R;
  core_slab = alloc_region(core_slab_bytes, Mem::MLOCK, &err);
  if (!core_slab) {
    core_n = 0;
    return false;
  }
  core_tab.assign((size_t)NL * NE, -1);
  core_of.assign(core_n, -1);
  return true;
}

int Cache::core_held() const {
  int k = 0;
  for (int32_t x : core_of) k += x >= 0;
  return k;
}

// the hottest held records of pool 0 by decayed count into the core; at most core_swaps copies per refresh (pread from
// the file: usually a copy out of the file cache), each core record pinned in the policy while it is in the core
void Cache::core_refresh() {
  SlotPool& S = P[0];
  const double t = (double)tok_now / half_life;
  std::vector<std::pair<double, int>> hot;  // (decayed count now, slot)
  for (int i = 0; i < S.n; i++) {
    const Slot& x = S.s[i];
    if (x.layer < 0 || layer_pool[x.layer] != 0 || S.pending[i].load(std::memory_order_acquire) != 0 || x.key < -1e299) continue;
    hot.push_back({x.key - t, i});
  }
  const int top = std::min<int>(core_n, (int)hot.size());
  std::partial_sort(hot.begin(), hot.begin() + top, hot.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first > b.first; });
  std::vector<uint8_t> want((size_t)NL * NE, 0);
  for (int k = 0; k < top; k++) want[(size_t)S.s[hot[k].second].layer * NE + S.s[hot[k].second].e] = 1;
  int swaps = 0;
  for (int k = 0; k < top && swaps < core_swaps; k++) {
    const Slot& x = S.s[hot[k].second];
    const size_t key = (size_t)x.layer * NE + x.e;
    if (core_tab[key] >= 0) continue;
    int c = -1;  // an empty core slot, else one whose record fell out of the top
    for (int j = 0; j < core_n && c < 0; j++)
      if (core_of[j] < 0) c = j;
    for (int j = 0; j < core_n && c < 0; j++)
      if (!want[core_of[j]]) c = j;
    if (c < 0) break;
    if (core_of[c] >= 0) {  // the old record goes back to the file tier (unpinned)
      const int oL = core_of[c] / NE, oe = core_of[c] % NE, os = tab[core_of[c]];
      core_tab[core_of[c]] = -1;
      if (os >= 0) unpin(oL, os);
      (void)oe;
    }
    uint8_t* dst = core_slab + (size_t)c * S.rec;
    size_t done = 0;
    while (done < S.rec) {
      ssize_t r = pread(fds[x.layer], dst + done, S.rec - done, (off_t)((size_t)x.e * S.rec + done));
      if (r <= 0) break;
      done += (size_t)r;
    }
    if (done < S.rec) {
      core_of[c] = -1;
      continue;
    }
    core_of[c] = (int32_t)key;
    core_tab[key] = c;
    S.s[hot[k].second].pins++;
    core_swapped++;
    swaps++;
  }
}

void Cache::token_end_file(long t) {
  if (mem != Mem::FILE) return;
  if (core_n && t % core_every == 0) core_refresh();
  if (!fa.on) return;
  fa.tokens++;
  // resident fraction of a rotating sample of held records (first + last page)
  SlotPool& S = P[0];
  const size_t pg = (size_t)getpagesize();
  int seen = 0, res = 0;
  for (int k = 0; k < S.n && seen < fa.sample; k++) {
    const int i = (fa.cursor + k) % S.n;
    const Slot& x = S.s[i];
    if (x.layer < 0 || S.pending[i].load(std::memory_order_acquire) != 0) continue;
    if (core_n && core_tab[(size_t)x.layer * NE + x.e] >= 0) continue;
    const uint8_t* v = fmap[x.layer] + (size_t)x.e * S.rec;
    char a = 0, b = 0;
    seen++;
    res += mincore((void*)v, pg, &a) == 0 && mincore((void*)(v + S.rec - pg), pg, &b) == 0 && (a & MINCORE_INCORE) && (b & MINCORE_INCORE);
  }
  fa.cursor = (fa.cursor + std::max(1, fa.sample * 2)) % std::max(1, S.n);
  if (seen) fa.res_frac = 0.8 * fa.res_frac + 0.2 * ((double)res / seen);
  if (st.file_refaults < fa.last_refaults) fa.last_refaults = 0;  // statistics were reset
  const double r = (double)(st.file_refaults - fa.last_refaults);
  fa.last_refaults = st.file_refaults;
  fa.refault_ema = 0.875 * fa.refault_ema + 0.125 * r;
  if (fa.tokens % fa.every) return;
  int64_t mx = fa.max_bytes;
  if (mx <= 0) {
    mx = 0;
    for (int p = 0; p < npools; p++) mx += (int64_t)P[p].n * (int64_t)P[p].rec;
  }
  const int64_t cur = budget_bytes();
  int64_t nb = cur;
  if (fa.refault_ema > fa.refault_hi) {
    const int64_t target = (int64_t)(fa.res_frac * (double)held_bytes());
    nb = std::max(fa.min_bytes, std::min((int64_t)(cur * (1.0 - fa.shrink_min)), target));
  } else if (fa.refault_ema < fa.refault_lo && cur < mx) {
    nb = std::min(mx, cur + (int64_t)(fa.grow * (double)mx));
  }
  if (std::llabs(nb - cur) >= (int64_t)S.rec) {  // at least one record of change
    (nb < cur ? fa.shrinks : fa.grows)++;
    set_budget(nb);
    apply_budget();
    fa.trace.push_back({t, budget_bytes()});
  }
}

bool Cache::load_policy(const std::string& path, std::string& err) {
  {
    ms::Json j;
    if (ms::json_load(path, j, err) && j.get("features")) return load_ranker(path, err);
    err.clear();
  }
  // the JSON's "w" (7 + NL numbers) and "b"
  FILE* f = fopen(path.c_str(), "r");
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  std::string js;
  char buf[4096];
  size_t r;
  while ((r = fread(buf, 1, sizeof buf, f)) > 0) js.append(buf, r);
  fclose(f);
  auto nums = [&](const std::string& key) {
    std::vector<float> v;
    size_t p = js.find("\"" + key + "\"");
    if (p == std::string::npos) return v;
    p = js.find_first_of("[-0123456789", js.find(':', p) + 1);
    if (p == std::string::npos) return v;
    if (js[p] == '[') {
      std::string body = js.substr(p + 1, js.find(']', p) - p - 1);
      const char* q = body.c_str();
      while (*q) {
        char* end;
        float x = strtof(q, &end);
        if (end == q) {
          q++;
          continue;
        }
        v.push_back(x);
        q = end;
      }
    } else {
      v.push_back(strtof(js.c_str() + p, nullptr));
    }
    return v;
  };
  lev_w = nums("w");
  std::vector<float> b = nums("b");
  if ((int)lev_w.size() != 7 + NL || b.size() != 1) {
    err = path + ": expected w[7 + layers] and b (got " + std::to_string(lev_w.size()) + " weights)";
    return false;
  }
  lev_b = b[0];
  const size_t N = (size_t)NL * NE;
  for (auto& a : lc) a.assign(N, 0.0);
  lc_t.assign(N, 0.0);
  l_last.assign(N, -1e9);
  lp_now.assign(N, 0.f);
  lp_e4.assign(N, 0.f);
  lp_e16.assign(N, 0.f);
  learned_evict = true;
  return true;
}

void Cache::enable_hazard(double count_half_life) {
  const size_t N = (size_t)NL * NE;
  hz_fac = std::pow(2.0, -1.0 / count_half_life);
  hz_t = -1;
  hz_bok = false;
  hz_last.assign(N, -1.0);
  hz_dec.assign(N, 0.0);
  hz_b.assign(N, 0);
  hz_kb.assign(N, 0);
  hz_cur.assign(N, 0);
  hz_u.assign((size_t)NL * HZ_NB, 1.0);  // prior 1 / 2
  hz_n.assign((size_t)NL * HZ_NB, 2.0);
  for (size_t k = 0; k < N; k++) hz_kb[k] = (int16_t)hz_bin(-1.0, 0.0, 0);
  pool_layers.assign(npools, {});
  for (int L = 0; L < NL; L++) pool_layers[layer_pool[L]].push_back(L);
  hzm_ok.assign(NL, 0);
  hzm_v.assign(NL, 0.0);
  hzm_id.assign(NL, -1);
  hzm_slot.assign(NL, -1);
  const char* ck = getenv("DMENG_HZ_CHECK");
  hz_check = ck && atoi(ck) > 0;
  hazard_evict = true;
}

bool Cache::caps_rule() const {
  long slots = 0;
  for (int p = 0; p < npools; p++) slots += P[p].cap;
  return (double)slots < CAPS_BELOW * K * NL;
}

void Cache::caps_auto_apply() {
  const bool suspended = !lcap_saved.empty();  // inside a batched prefill pass (caps_suspend): the new caps wait for resume
  if (caps_rule()) {
    uniform_layer_caps();  // (again at a new budget: the caps follow the slots)
  } else {
    lcap.clear();
    lheld.clear();
  }
  if (suspended) {
    lcap_saved = lcap;
    lcap.clear();
    lheld.clear();
  }
}

void Cache::caps_suspend() {
  if (lcap.empty() || !lcap_saved.empty()) return;
  lcap_saved = lcap;
  lcap.clear();
  lheld.clear();
}

void Cache::caps_resume() {
  if (lcap_saved.empty()) return;
  lcap = lcap_saved;
  lcap_saved.clear();
  lheld.assign(NL, 0);
  for (int p = 0; p < npools; p++)
    for (int i = 0; i < P[p].n; i++)  // (the slab's: probation is not under the caps)
      if (P[p].s[i].layer >= 0) lheld[P[p].s[i].layer]++;
}

void Cache::uniform_layer_caps() {
  lcap.assign(NL, 0);
  lheld.assign(NL, 0);
  for (int p = 0; p < npools; p++) {
    std::vector<int> ls;
    for (int L = 0; L < NL; L++)
      if (layer_pool[L] == p) ls.push_back(L);
    if (ls.empty()) continue;
    const int nl = (int)ls.size(), q = P[p].cap / nl, r = P[p].cap % nl;  // the budget at the call (a later resize keeps these caps)
    for (int i = 0; i < nl; i++) lcap[ls[i]] = q + (i < r ? 1 : 0);
  }
  for (int p = 0; p < npools; p++)  // records already held (normally none: called before the first token)
    for (int i = 0; i < P[p].n; i++)
      if (P[p].s[i].layer >= 0) lheld[P[p].s[i].layer]++;
}

void Cache::apart_begin() {
  apart_on = true;
  apart_q.assign(npools, {});
}

void Cache::apart_end(long positions) {
  if (!apart_on) return;
  apart_on = false;
  for (int p = 0; p < npools; p++)
    for (auto& x : P[p].s) x.apart = false;
  apart_q.clear();
  hz_off += std::max(0L, positions);
  hz_dirty_all();
}

// the oldest unpinned, landed slot the running prompt read (and that is still its record); -1 if none, or when a name
// (reserve > 0) would leave no more than `reserve` unpinned slots in the pool
int Cache::apart_victim(int L, int reserve) {
  const int p = layer_pool[L];
  SlotPool& S = P[p];
  std::deque<int>& q = apart_q[p];
  for (auto it = q.begin(); it != q.end();) {
    const int s = *it;
    if (s >= S.cap || !S.s[s].apart) {  // reused since, or past a lowered budget (the ordinary path gives it back)
      it = q.erase(it);
      continue;
    }
    if (S.s[s].pins > 0 || S.pending[s].load(std::memory_order_relaxed) != 0) {
      ++it;
      continue;
    }
    if (reserve > 0) {
      int unpinned = 0;
      for (int i = 0; i < S.cap && unpinned <= reserve; i++)
        if (S.s[i].pins == 0 && S.pending[i].load(std::memory_order_relaxed) == 0) unpinned++;
      if (unpinned <= reserve) return -1;
    }
    q.erase(it);
    drop(L, s, true);
    n_apart_taken++;
    return s;
  }
  return -1;
}

namespace {
// (int)std::log2(x) for x >= 1 from the exponent bits; std::log2 is called only just below a power of two, where it may
// round up to the next integer (so the result is the libm one exactly)
inline int ilog2_as_libm(double x) {
  uint64_t b;
  std::memcpy(&b, &x, 8);
  const int ex = (int)((b >> 52) & 0x7ff) - 1023;
  const uint64_t mb = (b & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
  double m;
  std::memcpy(&m, &mb, 8);
  return m > 1.9999999999995 ? (int)std::log2(x) : ex;
}
}  // namespace

int Cache::hz_bin(double last, double dec, long t) const {
  int ab = std::min(14, HZ_NA - 1);  // never used: age 2^14
  if (last >= 0) {
    const long age = std::max(1L, t - (long)last);
    ab = std::min(63 - __builtin_clzl((unsigned long)age), HZ_NA - 1);  // floor(log2(age))
  }
  const int fb = std::min(ilog2_as_libm(1.0 + 4.0 * dec), HZ_NF - 1);
  return ab * HZ_NF + fb;
}

void Cache::hz_tick(long t) {
  t -= hz_off;            // (a prompt apart did not move the policy's clock)
  if (t <= hz_t) return;  // the clock never goes back (an older position's touch counts at the current token)
  if ((chunk_t >= 0 || apart_on) && !hz_replay) return;  // inside a multi-position pass: the table moves when the pass ends
  // learn: every record's state at this token is one trial of its bucket. For the next token (g = 1) that bucket is
  // hz_kb, which always holds hz_bin(last, dec, hz_t + 1) (set here and at each touch); after a gap it is recomputed.
  const bool learn = hz_t >= 0;
  const long g = hz_t < 0 ? 1 : t - hz_t;
  const bool next = learn && g == 1;
  const double f = g == 1 ? hz_fac : std::pow(hz_fac, (double)g);
  for (int L = 0; L < NL; L++) {
    double* n = hz_n.data() + (size_t)L * HZ_NB;
    const size_t k0 = (size_t)L * NE;
    for (size_t k = k0; k < k0 + (size_t)NE; k++) {
      if (learn) {
        const int b = next ? hz_kb[k] : hz_bin(hz_last[k], hz_dec[k], t);
        hz_b[k] = (int16_t)b;
        n[b] += 1.0;
      }
      hz_dec[k] *= f;
      hz_kb[k] = (int16_t)hz_bin(hz_last[k], hz_dec[k], t + 1);
    }
  }
  if (learn) hz_bok = true;
  std::fill(hz_cur.begin(), hz_cur.end(), 0);
  hz_dirty_all();
  hz_t = t;
}

void Cache::chunk_done() {
  chunk_t = -1;
  if (chunk_touch.empty()) return;
  std::stable_sort(chunk_touch.begin(), chunk_touch.end(), [](const HzTouch& a, const HzTouch& b) { return a.step < b.step; });
  hz_replay = true;
  {
    PolicyTimer pt(st.policy_ms);
    for (const HzTouch& h : chunk_touch) hz_touch(h.L, h.e, h.t);
  }
  hz_replay = false;
  chunk_touch.clear();
}

void Cache::hz_touch(int L, int e, long t) {
  hz_tick(t);
  t -= hz_off;
  if (t < hz_t) t = hz_t;
  const size_t k = (size_t)L * NE + e;
  if (hz_cur[k]) return;
  hz_cur[k] = 1;
  hz_last[k] = (double)t;
  hz_dec[k] += 1.0;
  if (hz_bok) hz_u[(size_t)L * HZ_NB + hz_b[k]] += 1.0;
  hz_kb[k] = (int16_t)hz_bin(hz_last[k], hz_dec[k], t + 1);
  hzm_ok[L] = 0;
  if (rk.on) rk_gen[L]++;
}

double Cache::keep_score(int L, int e, long t) const {
  static const double hl[3] = {2.0, 8.0, 32.0};
  const size_t k = (size_t)L * NE + e;
  if (hazard_evict) {  // the estimate for the next token (the statistics are at t: victim() ticks first)
    const size_t b = (size_t)L * HZ_NB + hz_kb[k];
    const double v = hz_u[b] / hz_n[b] + hz_last[k] * 1e-9;
    // a background read's value until its step has run (read_bg_value)
    return !floor_keys.empty() && floor_until[k] >= now_step ? std::max(v, (double)floor_v[k]) : v;
  }
  double v = lev_b;
  for (int h = 0; h < 3; h++) v += lev_w[h] * std::log2(lc[h][k] * std::exp2(-((double)t - lc_t[k]) / hl[h]) + 1e-3);
  v += lev_w[3] * std::log2(1.0 + std::max(0.0, (double)t - l_last[k]));
  v += lev_w[4] * lp_now[k] + lev_w[5] * lp_e4[k] + lev_w[6] * lp_e16[k];
  v += lev_w[7 + L];
  return v;
}

void Cache::router_probs(int L, const float* z) {
  if (!learned_evict) return;
  float mx = -INFINITY;
  for (int e = 0; e < NE; e++) mx = std::max(mx, z[e]);
  std::vector<float> p(NE);
  double zs = 0;
  for (int e = 0; e < NE; e++) zs += (p[e] = std::exp(z[e] - mx));
  std::vector<int> idx(NE);
  for (int e = 0; e < NE; e++) idx[e] = e;
  const int top = std::min(32, NE);
  std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int a, int b) { return p[a] > p[b]; });
  std::vector<float> row(NE, 0.f);
  for (int j = 0; j < top; j++) row[idx[j]] = (float)(p[idx[j]] / zs);
  const float a4 = std::exp2(-1.f / 4), a16 = std::exp2(-1.f / 16);
  for (int e = 0; e < NE; e++) {
    const size_t k = (size_t)L * NE + e;
    lp_now[k] = row[e];
    lp_e4[k] = lp_e4[k] * a4 + (1 - a4) * row[e];
    lp_e16[k] = lp_e16[k] * a16 + (1 - a16) * row[e];
  }
}

// ------------------------------------------------------------------------------------------------ learned ranker (v3)
bool Cache::is_ranker_file(const std::string& path) {
  ms::Json j;
  std::string err;
  return ms::json_load(path, j, err) && j.get("features");
}

bool Cache::load_ranker(const std::string& path, std::string& err) {
  ms::Json j;
  if (!ms::json_load(path, j, err)) return false;
  Ranker r;
  r.name = j["name"].str_or("ranker");
  r.router = j["router"].str_or("");
  r.router_lag = (int)j["router_lag"].i(-1);
  r.pre = (int)j["prefilter"].i(128);
  const std::string kind = j["kind"].str_or("mlp");
  if (kind != "mlp" && kind != "lin") return err = path + ": kind mlp|lin, got " + kind, false;
  r.mlp = kind == "mlp";
  const ms::Json& fs = j["features"];
  r.F = (int)fs.size();
  auto hl_index = [](std::vector<double>& v, double h) {
    for (size_t i = 0; i < v.size(); i++)
      if (v[i] == h) return (int)i;
    v.push_back(h);
    return (int)v.size() - 1;
  };
  auto num_after = [](const std::string& s, const std::string& pre, double& h) {
    if (s.compare(0, pre.size(), pre) != 0 || s.size() == pre.size()) return false;
    char* end = nullptr;
    h = strtod(s.c_str() + pre.size(), &end);
    return *end == 0 && h > 0;
  };
  for (int f = 0; f < r.F; f++) {
    const std::string n = fs[f].s();
    double h = 0;
    int k = -1, a = 0;
    if (n == "log2_1p_since_use") k = Ranker::SINCE1;
    else if (n == "log2_1p_since_use2") k = Ranker::SINCE2;
    else if (n == "rel_p_now") k = Ranker::REL_NOW, r.rel = true;
    else if (n == "rt_p_now") k = Ranker::RT_NOW, r.rt = true;
    else if (n == "log2_1p_pos") k = Ranker::POS;
    else if (n == "hazard_p") k = Ranker::HZ;
    else if (n == "log2_hazard_p") k = Ranker::LOG2HZ;
    else if (num_after(n, "log2_c", h)) k = Ranker::CNT, a = hl_index(r.hl_c, h);
    else if (num_after(n, "rel_p_ema", h)) k = Ranker::REL_EMA, a = hl_index(r.hl_rel, h), r.rel = true;
    else if (num_after(n, "rt_p_ema", h)) k = Ranker::RT_EMA, a = hl_index(r.hl_rt, h), r.rt = true;
    else if (num_after(n, "rt_p_max", h)) k = Ranker::RT_MAX, a = hl_index(r.hl_max, h), r.rt = true;
    if (k < 0) return err = path + ": unknown feature " + n, false;
    r.kind.push_back(k);
    r.arg.push_back(a);
  }
  auto vec = [](const ms::Json& x) {
    std::vector<double> v;
    for (size_t i = 0; i < x.size(); i++) v.push_back(x[i].f());
    return v;
  };
  r.lb = vec(j["layer_bias"]);
  if ((int)r.lb.size() != NL) return err = path + ": layer_bias has " + std::to_string(r.lb.size()) + " entries, the model " + std::to_string(NL) + " layers", false;
  if (r.mlp) {
    const ms::Json& w1 = j["w1"];
    if ((int)w1.size() != r.F) return err = path + ": w1 needs one row per feature", false;
    r.b1 = vec(j["b1"]);
    r.w2 = vec(j["w2"]);
    r.H = (int)r.b1.size();
    if (r.H == 0 || (int)r.w2.size() != r.H) return err = path + ": b1 / w2 sizes", false;
    for (int f = 0; f < r.F; f++) {
      std::vector<double> row = vec(w1[f]);
      if ((int)row.size() != r.H) return err = path + ": w1 row size", false;
      r.w1.insert(r.w1.end(), row.begin(), row.end());
    }
  } else {
    r.w = vec(j["w"]);
    if ((int)r.w.size() != r.F) return err = path + ": w needs one weight per feature", false;
  }
  if (learned_evict) return err = "two learned keep-scores: pick one policy file", false;
  if (!hazard_evict) enable_hazard();  // the shortlist order and the hazard features
  const size_t N = (size_t)NL * NE;
  rk_cnt.assign(r.hl_c.size() * N, 0.0);
  rk_ct.assign(N, 0.0);
  rk_l1.assign(N, -1e9);
  rk_l2.assign(N, -1e9);
  rk_rel.assign(N, 0.0);
  rk_relE.assign(r.hl_rel.size() * N, 0.0);
  rk_rt.assign(N, 0.0);
  rk_rtE.assign(r.hl_rt.size() * N, 0.0);
  rk_rtM.assign(r.hl_max.size() * N, 0.0);
  rk_a_rel.clear();
  rk_a_rt.clear();
  rk_a_max.clear();
  for (double h : r.hl_rel) rk_a_rel.push_back(std::pow(2.0, -1.0 / h));  // as Python's 2 ** (-1.0 / h)
  for (double h : r.hl_rt) rk_a_rt.push_back(std::pow(2.0, -1.0 / h));
  for (double h : r.hl_max) rk_a_max.push_back(std::pow(2.0, -1.0 / h));
  rk_sc.assign(N, 0.0);
  rk_sc_t.assign(N, -1);
  rk_sc_g.assign(N, 0);
  rk_gen.assign(NL, 1);
  hzs.assign(NL, {});
  hz_dirty_all();
  r.on = true;
  rk = std::move(r);
  return true;
}

// one router row into a layer's state: now = row, ema = ema a + (1 - a) row, max = max(max a_m, row) (numpy's order)
void Cache::rk_row(double* now, double* ema, double* mx, const std::vector<double>& a_e, const std::vector<double>& a_m,
                   int L, const int32_t* idx, const float* p, int n) {
#pragma clang fp contract(off)
  const size_t N = (size_t)NL * NE, k0 = (size_t)L * NE;
  static thread_local std::vector<double> row;
  row.assign(NE, 0.0);
  for (int j = 0; j < n; j++) row[idx[j]] = (double)p[j];
  for (int e = 0; e < NE; e++) now[k0 + e] = row[e];
  for (size_t i = 0; i < a_e.size(); i++) {
    const double a = a_e[i];
    for (int e = 0; e < NE; e++) ema[i * N + k0 + e] = ema[i * N + k0 + e] * a + (1 - a) * row[e];
  }
  for (size_t i = 0; mx && i < a_m.size(); i++) {
    const double a = a_m[i];
    for (int e = 0; e < NE; e++) mx[i * N + k0 + e] = std::max(mx[i * N + k0 + e] * a, row[e]);
  }
  rk_gen[L]++;
}

namespace {
// the top 32 of a row: softmax probabilities of z (key null), or the values z ranked by key; ties to the lower index
int top32(const float* z, const float* key, int NE, int32_t* idx, float* p) {
  static thread_local std::vector<int> ord;
  ord.resize(NE);
  for (int e = 0; e < NE; e++) ord[e] = e;
  const float* kk = key ? key : z;
  const int n = std::min(32, NE);
  std::partial_sort(ord.begin(), ord.begin() + n, ord.end(), [&](int a, int b) { return kk[a] > kk[b] || (kk[a] == kk[b] && a < b); });
  if (key) {
    for (int j = 0; j < n; j++) idx[j] = ord[j], p[j] = z[ord[j]];
    return n;
  }
  const float mx = z[ord[0]];
  double zs = 0;
  for (int e = 0; e < NE; e++) zs += std::exp((double)(z[e] - mx));
  for (int j = 0; j < n; j++) idx[j] = ord[j], p[j] = (float)(std::exp((double)(z[ord[j]] - mx)) / zs);
  return n;
}
}  // namespace

void Cache::trace_init(long npos) {
  tr_on = true;
  tr_npos = 0;
  tr_rel_i.clear();
  tr_rt_i.clear();
  tr_rel_p.clear();
  tr_rt_p.clear();
  if (npos > 0) {
    const size_t n = (size_t)npos * steps * 32;
    tr_rel_i.reserve(n), tr_rt_i.reserve(n), tr_rel_p.reserve(n), tr_rt_p.reserve(n);
  }
}

static void tr_put(std::vector<int16_t>& vi, std::vector<float>& vp, long& npos, long tok, int steps, int step,
                   const int32_t* idx, const float* p, int n) {
  if (tok < 0) return;
  if (tok >= npos) {
    npos = tok + 1;
    vi.resize((size_t)npos * steps * 32, -1);
    vp.resize((size_t)npos * steps * 32, 0.f);
  }
  const size_t o = ((size_t)tok * steps + step) * 32;
  for (int j = 0; j < 32; j++) {
    vi[o + j] = j < n ? (int16_t)idx[j] : -1;
    vp[o + j] = j < n ? p[j] : 0.f;
  }
}

void Cache::released_row(int step, const float* z, const float* key) {
  int32_t idx[32];
  float p[32];
  const int n = top32(z, key, NE, idx, p);
  released_top(step, idx, p, n);
}

void Cache::released_top(int step, const int32_t* idx, const float* p, int n) {
  if (tr_on) {
    long np = tr_npos;
    tr_put(tr_rel_i, tr_rel_p, np, tok_now, steps, step, idx, p, n);
    if (np > tr_npos) {  // keep both arrays the same length
      tr_npos = np;
      tr_rt_i.resize(tr_rel_i.size(), -1);
      tr_rt_p.resize(tr_rel_p.size(), 0.f);
    }
  }
  if (!rk.rel) return;
  rel_step = step;  // applied at the step's end, after its misses (eviction_v3.py Feats.rel_row)
  std::copy(idx, idx + n, rel_idx);
  std::copy(p, p + n, rel_p);
  for (int j = n; j < 32; j++) rel_idx[j] = 0, rel_p[j] = 0.f;
}

void Cache::step_end(int step) {
  if (rel_step != step) return;
  rk_row(rk_rel.data(), rk_relE.data(), nullptr, rk_a_rel, {}, step_layer[step], rel_idx, rel_p, std::min(32, NE));
  rel_step = -1;
}

void Cache::router_row(long tok, int step, const float* z) {
  if (learned_evict) router_probs(step_layer[step], z);
  if (!rk.rt && !tr_on) return;
  int32_t idx[32];
  float p[32];
  const int n = top32(z, nullptr, NE, idx, p);
  router_top(tok, step, idx, p, n);
}

void Cache::router_top(long tok, int step, const int32_t* idx, const float* p, int n) {
  const int L = step_layer[step];
  if (tr_on) {
    long np = tr_npos;
    tr_put(tr_rt_i, tr_rt_p, np, tok, steps, step, idx, p, n);
    if (np > tr_npos) {
      tr_npos = np;
      tr_rel_i.resize(tr_rt_i.size(), -1);
      tr_rel_p.resize(tr_rt_p.size(), 0.f);
    }
  }
  if (rk.rt) rk_row(rk_rt.data(), rk_rtE.data(), rk_rtM.data(), rk_a_rt, rk_a_max, L, idx, p, n);
}

bool Cache::trace_write(const std::string& path, std::string& err) const {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return err = "cannot write " + path, false;
  // DMEVTR1: npos, steps, NE, 32, NL, K, npools; step_layer [steps], layer_pool [NL], per pool record bytes and slots
  // (int64); then rel idx int16 / p float32 and rt idx / p, each [npos, steps, 32]
  const char magic[8] = {'D', 'M', 'E', 'V', 'T', 'R', '1', 0};
  const int32_t hdr[7] = {(int32_t)tr_npos, steps, NE, 32, NL, K, npools};
  fwrite(magic, 1, 8, f);
  fwrite(hdr, 4, 7, f);
  fwrite(step_layer.data(), 4, steps, f);
  fwrite(layer_pool.data(), 4, NL, f);
  for (int p = 0; p < npools; p++) {
    const int64_t pr[2] = {(int64_t)P[p].rec, (int64_t)P[p].n};
    fwrite(pr, 8, 2, f);
  }
  fwrite(tr_rel_i.data(), 2, tr_rel_i.size(), f);
  fwrite(tr_rel_p.data(), 4, tr_rel_p.size(), f);
  fwrite(tr_rt_i.data(), 2, tr_rt_i.size(), f);
  fwrite(tr_rt_p.data(), 4, tr_rt_p.size(), f);
  fclose(f);
  return true;
}

double Cache::rk_score(int L, int e) {
#pragma clang fp contract(off)
  const size_t N = (size_t)NL * NE, k = (size_t)L * NE + e;
  if (rk_sc_t[k] == tok_now && rk_sc_g[k] == rk_gen[L]) return rk_sc[k];
  const double t = (double)tok_now;
  double f[64];
  for (int i = 0; i < rk.F; i++) {
    const int a = rk.arg[i];
    double v = 0;
    switch (rk.kind[i]) {
      case Ranker::CNT: v = std::log2(rk_cnt[a * N + k] * std::exp2(-(t - rk_ct[k]) / rk.hl_c[a]) + 1e-3); break;
      case Ranker::SINCE1: v = std::log2(1 + std::max(0.0, t - rk_l1[k])); break;
      case Ranker::SINCE2: v = std::log2(1 + std::min(1e6, std::max(0.0, t - rk_l2[k]))); break;
      case Ranker::REL_NOW: v = rk_rel[k]; break;
      case Ranker::REL_EMA: v = rk_relE[a * N + k]; break;
      case Ranker::RT_NOW: v = rk_rt[k]; break;
      case Ranker::RT_EMA: v = rk_rtE[a * N + k]; break;
      case Ranker::RT_MAX: v = rk_rtM[a * N + k]; break;
      case Ranker::POS: v = std::log2(1 + t); break;
      case Ranker::HZ: v = keep_score(L, e, tok_now); break;
      case Ranker::LOG2HZ: v = std::log2(keep_score(L, e, tok_now) + 1e-4); break;
    }
    f[i] = v;
  }
  double sc = rk.lb[L];
  if (rk.mlp) {
    double o = 0;
    for (int h = 0; h < rk.H; h++) {
      double z = 0;
      for (int i = 0; i < rk.F; i++) z += f[i] * rk.w1[(size_t)i * rk.H + h];
      z += rk.b1[h];
      if (z > 0) o += z * rk.w2[h];
    }
    sc += o;
  } else {
    double o = 0;
    for (int i = 0; i < rk.F; i++) o += f[i] * rk.w[i];
    sc += o;
  }
  rk_sc[k] = sc;
  rk_sc_t[k] = tok_now;
  rk_sc_g[k] = rk_gen[L];
  rk_evals++;
  return sc;
}

void Cache::rk_sorted(int L) {  // layer L's held records by (hazard score, id)
  const SlotPool& S = pool(L);
  auto& h = hzs[L];
  h.clear();
  const int32_t* tb = tab.data() + (size_t)L * NE;
  for (int e = 0; e < NE; e++) {
    const int s = tb[e];
    if (s < 0 || s >= S.cap) continue;
    h.push_back({keep_score(L, e, tok_now), L * NE + e, s});
  }
  std::sort(h.begin(), h.end(), [](const HzE& a, const HzE& b) { return a.v < b.v || (a.v == b.v && a.id < b.id); });
  hzm_ok[L] = 1;
}

int Cache::rk_victim(int L, bool capped, int reserve) {
  SlotPool& S = pool(L);
  const std::vector<int>& ls = pool_layers[layer_pool[L]];
  const int n = capped ? 1 : (int)ls.size();
  for (int i = 0; i < n; i++) {
    const int l = capped ? L : ls[i];
    if (!hzm_ok[l]) rk_sorted(l);
  }
  // the shortlist: a merge of the layers' lists, skipping entries whose record has left and records pinned or in flight
  struct Head {
    double v;
    int id, l, pos;
  };
  static thread_local std::vector<Head> heap;
  static thread_local std::vector<HzE> cand;
  heap.clear();
  cand.clear();
  auto later = [](const Head& a, const Head& b) { return a.v > b.v || (a.v == b.v && a.id > b.id); };
  for (int i = 0; i < n; i++) {
    const int l = capped ? L : ls[i];
    if (!hzs[l].empty()) heap.push_back({hzs[l][0].v, hzs[l][0].id, l, 0});
  }
  std::make_heap(heap.begin(), heap.end(), later);
  while (!heap.empty() && (int)cand.size() < rk.pre) {
    std::pop_heap(heap.begin(), heap.end(), later);
    Head hd = heap.back();
    heap.pop_back();
    const HzE& x = hzs[hd.l][hd.pos];
    if (tab[x.id] == x.slot && x.slot < S.cap && S.s[x.slot].pins == 0 && S.pending[x.slot].load(std::memory_order_relaxed) == 0)
      cand.push_back(x);
    if (++hd.pos < (int)hzs[hd.l].size()) {
      hd.v = hzs[hd.l][hd.pos].v;
      hd.id = hzs[hd.l][hd.pos].id;
      heap.push_back(hd);
      std::push_heap(heap.begin(), heap.end(), later);
    }
  }
  if (cand.empty()) return -1;
  int best = -1;
  double bs = 0;
  for (size_t i = 0; i < cand.size(); i++) {
    const double v = rk_score(cand[i].id / NE, cand[i].id % NE);
    if (best < 0 || v < bs) {
      best = (int)i;
      bs = v;
    }
  }
  rk_choices++;
  rk_cands += cand.size();
  if (reserve > 0) {  // as the scan: more than `reserve` unpinned, landed slots (free ones count, caps: own layer)
    int unpinned = 0;
    if (capped) {
      const int32_t* tb = tab.data() + (size_t)L * NE;
      for (int e = 0; e < NE && unpinned <= reserve; e++) {
        const int s = tb[e];
        if (s >= 0 && s < S.cap && S.s[s].pins == 0 && S.pending[s].load(std::memory_order_relaxed) == 0) unpinned++;
      }
    } else {
      for (int i = 0; i < S.cap && unpinned <= reserve; i++)
        if (S.s[i].pins == 0 && S.pending[i].load(std::memory_order_relaxed) == 0) unpinned++;
    }
    if (unpinned <= reserve) return -1;
  }
  return cand[best].slot;
}

}  // namespace dm
