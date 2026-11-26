// The expert store: record formats, slot pools under one byte budget, the eviction policy, and the reader threads.
//
// Records of physical layer L live in store/layer_LL.bin, expert e at byte e * record_bytes. Slots are keyed by physical
// layer, so the passes of a looped layer share them. There is one slot pool per record format (one for the Qwen
// families, two for Inkling: NVFP4 and MXFP8), sized by a byte share of the budget.
//
// Policy: decayed use count (half-life in tokens) with a recency tie-break; free slots first. Pins come from the step
// running now, from early names, and from the loop hold. Reads go through pread + F_NOCACHE on dedicated threads, an
// urgent queue (a pick the released router made that is not in memory) ahead of a speculative one (names).
//
// Mem::FILE (the phone's tier, results/ios/p0/P0.md section 2): each layer file is mapped read-only once and a held
// record is a view into the mapping (records are page aligned), not a copy in a slot. A slot is then bookkeeping only:
// which records the policy believes resident, sized by a byte count that may exceed the app's memory, because clean
// file pages are not charged to it. A miss is filled by a buffered pread of the record into the reader thread's
// scratch buffer (it lands in the OS file cache, which the mapping shares; the copy is discarded), skipped when mincore
// says the record is resident already. The OS may drop pages behind the policy's back: a held record is checked on use
// (mincore of its first and last page) and re-read when its pages are gone (a late miss, counted as a refault); any page
// the check misses is a page fault, never wrong bytes. Eviction drops the bookkeeping entry (optionally a
// MADV_DONTNEED hint on the range). A dirty tier in front of the file tier can be added as a second pool later.
//
// Spare memory under the slab (Cache::file_under, CoreOpts::spare_mem; a locked or plain slab only): the same file
// tier placed below the slab instead of in its place. The layer files are opened buffered (no F_NOCACHE) and mapped
// read-only as in Mem::FILE (for mincore only; the compute never reads the mapping), so a record read into the slab
// also stays in the OS file cache after the slab evicts it, as clean file pages the OS may drop at any time and does
// not charge to the process. A read into the slab whose pages mincore finds resident is a copy from the file cache,
// not a drive read: it is counted in Readers::file_resident_skips / file_bytes_skipped, not in Readers::bytes. The
// bytes in the slab are the file's bytes either way (lossless). spare_pause() (a memory warning) sets F_NOCACHE on the
// files again: reads stop adding pages to the file cache (the slab path as before).
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "dmeng/weights.h"

namespace dm {

// One record layout (FORMAT.md section 2.3): the offsets of its fields and the kernels that read them.
struct ExpertFormat {
  Fmt fmt = Fmt::AFF4;      // AFF4 | NVFP4 | MXFP8
  size_t rec = 0;           // record bytes
  int H = 0, I = 0;         // hidden size, expert intermediate size
  bool fused = false;       // one gate_up matrix [2I, H], row 2j = gate j, row 2j + 1 = up j
  size_t gu_c = 0, gu_s = 0, gu_b = 0;  // gate (or the fused gate_up): codes, scales, biases
  size_t up_c = 0, up_s = 0, up_b = 0;  // up (not fused)
  size_t dn_c = 0, dn_s = 0, dn_b = 0;  // down
  long trailer = -1;        // NVFP4: f32 [gate_up amax, down amax, gate_up denominator, down denominator]
  size_t touched = 0;       // bytes a compute pass reads (statistics)
  // the expert's matrices over a record in memory
  void mats(const uint8_t* rec, Weight& gu, Weight& up, Weight& dn) const;
};

struct Readers {
  struct Job {
    int fd;
    long off;
    uint8_t* dst;       // nullptr: Mem::FILE, read into the thread's scratch buffer (the file cache keeps the pages)
    size_t n;
    int pool, slot;
    const uint8_t* view = nullptr;  // Mem::FILE: the piece's address in the file mapping
    long step = 0;      // speculative reads: the step (token * steps + step index) they serve
    float pp = 0.f;     // ... and the probability a planner gave them
    int64_t t_sub = 0;  // submit time (ns, steady clock)
  };
  void start(int n, std::atomic<int>** pending, bool skip);  // pending[pool][slot]: parts of a record still unread
  void stop();
  void submit(const Job& j, bool urgent) { submit_cls(j, urgent ? 0 : 1); }
  // class 0 urgent, 1 speculative, 2 background: a background read starts only while no urgent or speculative read is
  // queued or in flight (names for a later token that must not delay this token's reads), at most max_bg at a time
  void submit_cls(const Job& j, int cls);
  bool promote(int pool, int slot);  // queued speculative parts of this slot go to the urgent queue; true if any
  bool cancel(int pool, int slot);   // remove a speculative read none of whose parts has started; true if removed
  // remove every speculative read serving a step <= `step` none of whose parts has started; their (pool, slot) out
  void cancel_stale(long step, std::vector<std::pair<int, int>>& out);
  bool prio = false;         // speculative reads by (step, -pp) instead of first in, first out
  bool excl = false;         // start no speculative read while an urgent read is in flight
  int max_spec = 1 << 30;    // at most this many speculative reads in flight
  int max_bg = 1;            // at most this many background reads in flight
  int bg_spec_gate = 0;      // a background read also starts while at most this many speculative reads are in flight
                             // (never while an urgent read is queued or in flight, or a speculative one is queued)
  bool bg_open();            // a background read queued now would start at once
  int bg_queued();           // background reads queued or in flight (jobs)
  bool file_skip_resident = true;  // Mem::FILE: no pread for a piece whose pages are all resident (mincore); spare
                                   // memory under the slab: such a piece is counted as served by the file cache
  std::atomic<bool> spare_on{false};  // spare memory under the slab in use (Cache::file_under, not paused)
  bool file_prefault = true;       // Mem::FILE: after the pread, touch one byte per page of the view (the reader takes
                                   // the soft faults that map the cached pages, not the compute threads)
  int qos = -1;                    // QoS class of the reader threads (qos_class_t); -1: user-initiated
  bool io_low = false;             // the readers' disk I/O one tier below the threads' own (IOPOL_STANDARD); CoreOpts::reader_io_low
  // statistics (written by the readers)
  std::atomic<uint64_t> bytes{0}, jobs{0}, busy_ns{0};
  std::atomic<uint64_t> bytes_cls[3] = {{0}, {0}, {0}};  // bytes read by class: urgent, speculative, background
  std::atomic<uint64_t> file_resident_skips{0}, file_bytes_skipped{0};  // Mem::FILE pieces found resident, not read
  std::atomic<uint64_t> q_ns[3], svc_ns[3], njobs[3], spec_inflight_at_urgent{0};  // [0] urgent, [1] speculative, [2] background
  std::atomic<uint64_t> bg_promoted{0};  // background reads a step needed before they started (moved to urgent)
  uint64_t busy_wall();      // wall time with at least one read in flight
  // wall time by state: [0] no read in flight, nothing queued; [1] none in flight, reads queued (held back by the
  // speculative limits); [2..5] 1, 2, 3, 4+ reads in flight. [6] = bursts (no read in flight -> one)
  void depth_times(uint64_t out[7]);
  bool saturated();  // every reader has a read in flight and reads wait behind them
  bool urgent_backlog();  // every reader has a read in flight and urgent reads wait behind them (the drive is behind now)
  ~Readers() { stop(); }

 private:
  void loop();
  bool can_spec() const;
  bool can_bg() const;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> urgent_, normal_, bg_;
  std::vector<std::thread> th_;
  std::atomic<int>** pending_ = nullptr;
  bool stop_ = false, skip_ = false;
  int inflight_[3] = {0, 0, 0};
  int64_t busy_t0_ = 0;
  uint64_t busy_wall_ns_ = 0;
  int64_t dt_last_ = 0;
  int dt_state_ = 0;
  uint64_t dt_ns_[7] = {0};
  void tick_(int64_t t);  // (mu_ held) after a change of the in-flight count or the queues
};

struct Slot {
  int layer = -1, e = -1;
  double key = -1e300;     // log2(decayed use count) + t / half-life: larger = keep
  long last = -1;          // last use (step): the recency tie-break
  int pins = 0;
  int pf_pins = 0;         // how many of the pins are early names (a miss may take such a slot when nothing else is free)
  bool pf_unused = false;  // landed by a name and not used since
  bool bg_unused = false;  // landed by a background read and not used since
  bool apart = false;      // read by the prompt running now (Cache::apart_on), not used by a decode step since
};

struct SlotPool {
  int n = 0;
  int cap = 0;                 // slots in use now (set_budget lowers it; slots >= cap are given back once unpinned)
  int min_cap = 0;             // the least cap a run-time budget may set (pins of one step / one prefill layer)
  size_t rec = 0;
  uint8_t* slab = nullptr;
  size_t slab_bytes = 0;
  std::vector<Slot> s;
  std::unique_ptr<std::atomic<int>[]> pending;
  std::vector<int> free_;
  std::vector<uint8_t*> preg;  // purgeable mode: one VM object per slot
  std::vector<uint8_t> nonvol;
  std::vector<uint8_t> locked;  // Mem::MLOCK: the slot's memory is locked now (mlock succeeded, not unlocked since)
  uint8_t* row(int i) const { return preg.empty() ? slab + (size_t)i * rec : preg[i]; }
};

struct CacheStats {
  uint64_t n_hit = 0, n_late = 0, n_miss = 0, n_pf_issued = 0, n_pf_dropped = 0, n_pf_unused_evicted = 0, n_purged = 0;
  uint64_t n_pf_gated = 0;  // names that needed a read and were not issued: urgent reads were queued (Cache::name_gate)
  uint64_t n_pf_stolen = 0, n_pf_cancelled = 0, n_late_queued = 0, n_miss_evicted = 0, n_miss_pf_evicted = 0;
  uint64_t miss_pbin[6] = {0};          // misses by a planner's probability for them: none, <.1, <.3, <.5, <.7, >=.7
  uint64_t n_hit_r[2] = {0}, n_pick_r[2] = {0};  // [looped steps, other steps]
  uint64_t n_reuse_hit = 0;             // looped layers: a pick already used by an earlier pass of the same token
  uint64_t n_reuse_reread = 0;          // ... of those, not in memory any more: read again within the token
  uint64_t bytes_urgent = 0, bytes_pf = 0;
  double wait_ms = 0, wait_ms_miss = 0, wait_ms_late = 0;
  double wait_ms_runnable = 0;  // lossless step: waits for a pick while another pick of the same layer had landed, not yet run
  uint64_t file_checks = 0, file_refaults = 0, bytes_refault = 0;  // Mem::FILE: held records checked on use / found gone
  double policy_ms = 0;                 // victim() and pre_use(): the eviction policy's own time
  uint64_t n_cap_borrow = 0;            // urgent slots a layer at its cap took from the pool (its own were all pinned)
  // background reads (Cache::read_bg_value): issued, refused (the victim was valued higher), used by a pick, evicted unused
  uint64_t n_bg_issued = 0, n_bg_refused = 0, n_bg_used = 0, n_bg_unused_evicted = 0, bytes_bg = 0;
};

struct Cache {
  static constexpr int MAXP = 4;
  int NL = 0, NE = 0, K = 0, steps = 0;
  std::vector<int> layer_pool;   // [NL] pool (= record format) of each physical layer
  std::vector<int> step_layer;   // [steps]
  SlotPool P[MAXP];
  int npools = 0;
  std::vector<int32_t> tab;      // [NL * NE] -> slot in the layer's pool, -1 when not held
  std::vector<int> fds;
  Readers rd;
  Mem mem = Mem::PLAIN;
  // Mem::FILE
  std::vector<uint8_t*> fmap;    // [NL] the layer files mapped PROT_READ, MAP_SHARED
  std::vector<size_t> fmap_bytes;
  int file_check = 1;            // check a held record on use: 0 never, 1 first and last page, 2 every page
  // on a dropped record's range: 0 nothing (default: pages are left to the OS), 1 MADV_DONTNEED (measured: no effect on
  // iOS or macOS), 2 msync(MS_INVALIDATE) (iOS, P2: the clean pages leave the file cache at once, so the policy, not the
  // OS's LRU, decides what stays)
  int file_evict_hint = 0;
  // spare memory under the slab (header comment): set before init; spare_pause() any thread, once per session
  bool file_under = false;
  std::atomic<bool> spare_paused{false};
  void spare_pause();
  void file_hit_check(int L, int slot);  // a held, landed record: re-read it now (urgent) if its pages are gone
  uint64_t file_resident_bytes() const;  // mincore over every layer file (a sample: costs ~ms, not per token)
  uint64_t held_bytes() const;           // bytes of the records the policy holds now (all pools)
  // Mem::MLOCK: the bytes of the cache's slots the OS has actually locked, and the locks it refused. A slot whose lock is
  // refused is not taken into use: the budget stops below it (grow, lock_start), so the cache in use is what is locked.
  int64_t locked_bytes = 0;
  uint64_t lock_refused = 0;
  std::string lock_json() const;
  // run-time budget: request from any thread (budget_req), applied by the compute thread at a token or prefill-layer
  // boundary (apply_budget). Lower: entries dropped, their memory given back (munlock + MADV_FREE_REUSABLE, purgeable
  // EMPTY; FILE: bookkeeping only); higher: up to the slots allocated at init.
  std::atomic<int64_t> budget_req{-1};
  void set_budget(int64_t bytes) { budget_req.store(bytes, std::memory_order_release); }
  // a budget that may grow past its start: init maps the slabs (Mem::MLOCK) without locking them (defer_lock), then
  // lock_start(bytes) locks only the start budget's slots; apply_budget locks more as the budget grows (serving: the
  // cache size from the machine, src/serving/memwatch.h)
  bool defer_lock = false;
  void lock_start(int64_t bytes);
  void apply_budget();
  // a loan of memory for a prompt's transient buffers (its scratch, the GPU path's buffers, its cancel snapshot): the
  // budget falls by `bytes` at once (call on the compute thread between passes; slots past the lowered budget are given
  // back as at any budget change, and the prompt's own reads refill what is left) and comes back with lend(0)
  void lend(int64_t bytes);
  int64_t lent = 0, lend_base = 0;
  // while lent: the slots past the budget given back again (measured on macOS: the GPU's first use of a wrapped buffer
  // faults its given-back pages in and charges them to the process again); idempotent, cheap (call between layers)
  void lend_hold();
  std::vector<double> share_;    // the budget split init was given
  int64_t budget_bytes() const;  // the budget in force (slots in use x record bytes)
  // Mem::FILE adaptive budget: the believed set follows what the OS keeps. Each token a rotating sample of held records
  // is checked (first + last page) for the resident fraction, and the refaults per token are averaged; every `every`
  // tokens the budget shrinks toward (resident fraction x held bytes) while refaults stay above refault_hi, and grows
  // by `grow` x max while they stay below refault_lo. Off unless `on`; a caller's set_budget is the new max.
  struct FileAdapt {
    bool on = false;
    int64_t min_bytes = 1000000000, max_bytes = 0;  // 0 max: the budget given at init
    double refault_hi = 1.0, refault_lo = 0.1;      // refaults per token (EMA over ~8 tokens)
    int every = 8, sample = 64;
    double grow = 0.02, shrink_min = 0.05;          // grow step (of max); a shrink takes at least this fraction
    // state and statistics
    double res_frac = 1.0, refault_ema = 0.0;
    uint64_t last_refaults = 0, shrinks = 0, grows = 0;
    long tokens = 0;
    int cursor = 0;
    std::vector<std::pair<long, int64_t>> trace;    // (token, budget) at every change
  } fa;
  // Mem::FILE locked core: the hottest records (decayed use count) copied into a small mlocked slab and served from
  // it (pinned in the policy, never checked or re-read); refreshed every core_every tokens, core_swaps at most per
  // refresh. Off unless core_init() was called with bytes > 0.
  uint8_t* core_slab = nullptr;
  size_t core_slab_bytes = 0;
  int core_n = 0, core_every = 16, core_swaps = 32;
  std::vector<int32_t> core_tab;                    // [NL * NE] -> core slot, -1
  std::vector<int32_t> core_of;                     // [core_n] -> L * NE + e, -1
  uint64_t core_swapped = 0;
  bool core_init(int64_t bytes, std::string& err);
  int core_held() const;
  void token_end_file(long t);                      // Mem::FILE per-token work (adaptive budget, core refresh)
  double half_life = 8.0;
  bool skip_reads = false;
  int split = 1;                 // a record is read as this many pieces (on as many reader threads)
  int extra_reserve[MAXP] = {0}; // slots a name must leave unpinned (the loop hold's worst case)
  std::vector<std::string> layer_paths;  // set before init: per-layer files relative to dir (empty: the layer_file pattern)
  long now_step = 0;             // token * steps + step index of the MoE block running now
  bool evict_pf_ctx = false;     // the eviction being made is for a speculative read
  bool read_bg = false;          // a speculative read issued now goes to the readers' background class
  std::vector<long> ev_step;     // [NL * NE] step at which (L, e) was last evicted (-1: never)
  std::vector<uint32_t> n_reads;  // [NL * NE] reads issued per record since start (the prompt path's re-read count)
  std::vector<uint8_t> ev_pf;    // ... and whether a speculative read evicted it
  // estimate-aware eviction (Stage 3): a slot's value = a planner's probability that its layer's next pass uses it
  // (pred >= 0), else P(used at its layer's next pass | decayed use count), learned online
  bool pred_evict = false, learn_on = false;
  std::vector<float> pred;       // [NL * NE], -1 = none
  static constexpr int VB = 24;
  double use_cnt[VB] = {0}, use_pos[VB] = {0};
  float use_tab[VB] = {0};
  CacheStats st;
  std::atomic<int>* pend_[MAXP] = {nullptr};  // per pool: the readers' view of the pending counts
  // learned keep-score (a logistic next-use predictor trained offline, e.g. the 35B design study's
  // q36_d_evict_policy.json): features per (layer, expert) = log2 decayed use counts at half-lives 2 / 8 / 32 tokens,
  // log2(1 + tokens since last use), a router's latest softmax probability (0 outside its top 32) and its EMAs at
  // half-lives 4 / 16 tokens, the physical layer one-hot; score = w . f + b, the lowest is evicted. keep_score() is the
  // one place the policy lives; off unless load_policy() succeeds.
  bool learned_evict = false;
  std::vector<float> lev_w;  // [7 + NL]
  float lev_b = 0;
  std::vector<double> lc[3], lc_t, l_last;
  std::vector<float> lp_now, lp_e4, lp_e16;
  long tok_now = 0;  // the token the running step belongs to (the policy's clock)
  // a multi-position pass (batched prefill, a verify pass, an Inkling span) runs layer by layer over its positions, so
  // its uses arrive with clocks that go back at every layer. While such a pass runs (ChunkClock, outermost scope wins):
  // the hazard table does not tick and its touches are kept (step, layer, expert, position), then replayed in time order
  // when the pass ends, so the table after the pass is the one single steps leave (it depends on the routing only);
  // the learned scores take the pass as one token at its last position (chunk_t, also tok_now). The decayed use count
  // (freq) is order-free and keeps each use's own position.
  long chunk_t = -1;
  bool hz_replay = false;
  struct HzTouch {
    long step;
    int L, e;
    long t;
  };
  std::vector<HzTouch> chunk_touch;
  void chunk_done();
  // A prompt apart from the decode's cache (PrefillOpts::apart; hazard only): while a prompt runs (prefill_spans), its
  // uses are not counted in the hazard statistics and its positions do not move the policy's clock (hz_off: the clock
  // the statistics see is tok_now - hz_off, so the decode step after the prompt is one token after the one before it),
  // and a slot the prompt needs is taken first from the records the prompt itself read and no longer pins (oldest read
  // first), before any record held when it started; at its end those records are ordinary ones, valued by the decode's
  // statistics like any other. A prompt's reads then displace at most what it pins at once (about one layer's union),
  // not the decode's working set, and the decode's statistics are the decode's own. Placement only: exact outputs never
  // depend on it.
  bool apart_on = false;
  long hz_off = 0;
  std::vector<std::deque<int>> apart_q;  // [npools] slots the running prompt read, in read order
  uint64_t n_apart_taken = 0;            // slots a prompt took from its own earlier reads
  void apart_begin();
  void apart_end(long positions);
  int apart_victim(int L, int reserve);
  struct ChunkClock {
    Cache& c;
    long prev_t;
    ChunkClock(Cache& c_, long t_last) : c(c_), prev_t(c_.chunk_t) {
      if (prev_t < 0) c.chunk_t = c.tok_now = t_last;
    }
    ~ChunkClock() {
      if (prev_t < 0) c.chunk_done();
    }
  };
  bool load_policy(const std::string& path, std::string& err);
  double keep_score(int L, int e, long t) const;
  void router_probs(int L, const float* z);  // a router's logits for physical layer L this token (NE values)
  // hazard keep-score (phone-store Store.swift Evict(kind: .hazard) = docs/evict_runs/evict_sim.py hazard_h128; the
  // phone-size study results/ios/evict/EVICT.md): per physical layer an online table of P(used at the next token |
  // floor(log2 tokens since last use), floor(log2(1 + 4 x use count decayed with a 128-token half-life))), learned from
  // this session's own uses; keep_score = the estimate for the next token (+ 1e-9 x last use), the lowest is evicted
  // (ties: the lower record id). The statistics move in pre_use() / use() (one touch per record per
  // token); the clock ticks at the first touch or eviction of a new token. Off unless enable_hazard(); no router needed.
  bool hazard_evict = false;
  void enable_hazard(double count_half_life = 128.0);
  // per-layer caps (any policy): a layer at its cap evicts one of its own records, below it takes a free slot. With
  // caps summing to the pool's slots a free slot then always exists. The caps are soft for a slot a step needs now
  // (victim with no reserve): when every record of a layer at its cap is pinned or in flight (a multi-position step's
  // pins, a prefill chunk's union), the slot comes from the whole pool; names (a reserve) are refused instead. uniform_layer_caps(): each pool's slots split
  // evenly over its layers (the remainder to the first layers). Off (one pool over all layers) unless set.
  std::vector<int> lcap, lheld;  // [NL]
  void uniform_layer_caps();
  // the reserve a name leaves in a layer at its cap (layer_caps "...:fit"): the fixed rule leaves 2K + the loop holds
  // unpinned, which a cap smaller than that plus the step's own names cannot give (Flash-Next, 24 slots, K = 10: 3 of 10
  // names admitted); fit leaves min(2K + holds, cap - the names a step makes for that layer - the layer's own loop hold),
  // so a layer whose cap holds its names and its hold admits them all. Unchanged
  // wherever the cap is at least 2K + holds + the names per step + the layer's hold. Names only.
  bool cap_fit = false;
  std::vector<int> layer_hold;  // [NL] slots a layer's loop holds can pin (K per holding step of the layer, as extra_reserve)
  // P2: a batched prefill pass names the next layer's whole union, which a layer at its cap refuses (names never
  // borrow), so the reads run in series with compute. suspend lifts the caps for the pass; resume puts them back and
  // recounts what each layer holds (a layer over its cap then evicts its own records first). Placement only.
  void caps_suspend();
  void caps_resume();
  std::vector<int> lcap_saved;
  // the caps off for one multi-position pass (batched prefill, verify, Inkling's span); nested passes keep the outer one's
  struct CapsOff {
    Cache& c;
    bool mine;
    explicit CapsOff(Cache& x) : c(x), mine(!x.lcap.empty() && x.lcap_saved.empty()) {
      if (mine) c.caps_suspend();
    }
    ~CapsOff() {
      if (mine) c.caps_resume();
    }
  };
  // the default's rule: uniform caps while the slots (all pools) are fewer than CAPS_BELOW x K x layers (35B: below
  // 1,280 slots, 2.26 GB; Inkling: 768, 3.8 GB), else one pool per record format; caps_auto re-applies it whenever the
  // budget changes (apply_budget)
  static constexpr double CAPS_BELOW = 4.0;
  bool caps_auto = false;
  bool caps_rule() const;
  void caps_auto_apply();
  static constexpr int HZ_NA = 13, HZ_NF = 10, HZ_NB = HZ_NA * HZ_NF;
  double hz_fac = 0;
  long hz_t = -1;           // the token the statistics are at
  bool hz_bok = false;      // the buckets of the previous tick exist (learning on)
  std::vector<double> hz_last, hz_dec, hz_u, hz_n;  // [NL * NE] last use (token, -1 never), decayed count; [NL * HZ_NB]
  std::vector<int16_t> hz_b, hz_kb;                 // [NL * NE] bucket at the tick (learning), bucket for the next token
  std::vector<uint8_t> hz_cur;                      // [NL * NE] touched this token
  void hz_tick(long t);
  void hz_touch(int L, int e, long t);
  // the released picks of a step, before its misses take slots (Core::lossless / approx): a later pick of the step
  // is then never the lowest score of an earlier miss (the phone's touch-first order); use() does not touch them again
  void pre_use(int L, const int32_t* picks, int n, long t);
  int hz_bin(double last, double dec, long t) const;
  // victim choice without the slot scan (the same choice as the scan, eviction for eviction): per layer the lowest
  // (keep score, record id) among its held records, kept until the layer's statistics change (its touches, the tick)
  // or that record goes; a victim is the lowest of the pool's layers (the layer's own with caps). Pins and reads in
  // flight are checked on the winner only: an ineligible winner makes its layer search its eligible records.
  // DMENG_HZ_CHECK=1 also runs the scan at every choice and aborts on a difference.
  std::vector<std::vector<int>> pool_layers;  // [npools] the physical layers of each pool
  std::vector<uint8_t> hzm_ok;                // [NL] the layer's minimum below is current
  std::vector<double> hzm_v;
  std::vector<int> hzm_id, hzm_slot;          // slot -1: the layer holds nothing
  bool hz_check = false;
  void hz_layer_min(int L, bool eligible, double& v, int& id, int& slot) const;
  int hz_victim(int L, bool capped, int reserve);
  int hz_scan(int L, bool capped, int reserve) const;  // the reference: every slot of the pool
  void hz_gone(int L, int slot) {                      // a held record leaves its slot
    if (!hzm_ok.empty() && hzm_slot[L] == slot) hzm_ok[L] = 0;
  }
  void hz_dirty_all() {
    if (!hzm_ok.empty()) std::fill(hzm_ok.begin(), hzm_ok.end(), 0);
  }

  // learned ranker over hazard's shortlist (the design study's "v3": its eviction_v3.py Ranked, policy files
  // q36_d_evict3_policy.json / ink_d_evict3_policy.json): at an eviction the `prefilter` unpinned, landed records with
  // the lowest hazard keep score (of the pool; with caps, of the layer) are scored by a small model on named features of
  // each record, and the lowest score goes (ties: the lower hazard order). Model-independent: the features are taken by
  // name, the weights and the per-physical-layer bias from the JSON; kind "mlp" (relu(f W1 + b1) . w2 + bias[layer])
  // or "lin" (f . w + bias[layer]). Feature names: log2_c<h> (log2(use count decayed with half-life h tokens + 1e-3)),
  // log2_1p_since_use, log2_1p_since_use2 (tokens since the last / the one before, the latter capped at 1e6),
  // rel_p_now, rel_p_ema<h> (the released router's row of the step: softmax top 32; sigmoid routers: the top 32 of
  // score + bias valued by the sigmoid score; at the step's end), rt_p_now, rt_p_ema<h>, rt_p_max<h> (an early
  // router's row, top 32 of its softmax, when it finishes; max = decayed max), log2_1p_pos (the token index), hazard_p
  // (hazard's keep score), log2_hazard_p (log2(hazard_p + 1e-4)). Uses count at every step's end (Cache::use). Off
  // unless load_policy() reads a file with "features"; it turns the hazard keep-score on.
  struct Ranker {
    enum Kind { CNT, SINCE1, SINCE2, REL_NOW, REL_EMA, RT_NOW, RT_EMA, RT_MAX, POS, HZ, LOG2HZ };
    bool on = false, mlp = true;
    std::string name, router;
    int router_lag = -1, pre = 128, F = 0, H = 0;
    std::vector<int> kind, arg;              // [F] feature kind, index into its half-life list
    std::vector<double> hl_c, hl_rel, hl_rt, hl_max;
    std::vector<double> w1, b1, w2, w, lb;   // w1 [F * H] (row f), b1 / w2 [H], w [F] (lin), lb [NL]
    bool rel = false, rt = false;            // which router rows the features read
  } rk;
  std::vector<double> rk_cnt, rk_ct, rk_l1, rk_l2;        // [hl_c x N], [N] counts and their time, last uses
  std::vector<double> rk_rel, rk_relE, rk_rt, rk_rtE, rk_rtM;  // [N], [hl x N] router rows
  std::vector<double> rk_a_rel, rk_a_rt, rk_a_max;        // per half-life 2^(-1/h)
  std::vector<double> rk_sc;                               // [N] cached model score of the record
  std::vector<long> rk_sc_t;                               // [N] ... at this token
  std::vector<uint32_t> rk_sc_g, rk_gen;                   // [N] ... at this generation of its layer's features; [NL]
  struct HzE {
    double v;
    int id, slot;
  };
  std::vector<std::vector<HzE>> hzs;                       // [NL] held records by (hazard score, id), when rk.on
  int rel_step = -1;                                       // the released row waiting for its step's end
  int32_t rel_idx[32];
  float rel_p[32];
  uint64_t rk_evals = 0, rk_choices = 0, rk_cands = 0;
  bool load_ranker(const std::string& path, std::string& err);
  static bool is_ranker_file(const std::string& path);  // a policy file with "features" (else a keep-score)
  double rk_score(int L, int e);
  int rk_victim(int L, bool capped, int reserve);
  void rk_sorted(int L);
  void rk_row(double* now, double* ema, double* mx, const std::vector<double>& a_e, const std::vector<double>& a_m, int L,
              const int32_t* idx, const float* p, int n);
  // the router rows (any policy may read them; also the trace below): the released router of `step` (softmax logits,
  // or sigmoid scores + their biased keys), and an early router's logits for target `step` of token `tok`
  bool wants_rel() const { return rk.rel || tr_on; }
  bool wants_rt() const { return learned_evict || rk.rt || tr_on; }
  void released_row(int step, const float* z, const float* key);  // key null: softmax over z
  void router_row(long tok, int step, const float* z);
  void released_top(int step, const int32_t* idx, const float* p, int n);  // the same from a row's top entries
  void router_top(long tok, int step, const int32_t* idx, const float* p, int n);
  void step_end(int step);                                        // Core::finish_step: the step's statistics are in
  // eviction trace (dmeng --evict-trace F): per (token, step) the released row and the early router's row, top 32 each,
  // for replaying this run in the design study's simulator (eviction_v3.py) with the features the engine saw
  bool tr_on = false;
  long tr_npos = 0;
  std::vector<int16_t> tr_rel_i, tr_rt_i;                  // [npos, steps, 32], -1: none
  std::vector<float> tr_rel_p, tr_rt_p;
  void trace_init(long npos);
  bool trace_write(const std::string& path, std::string& err) const;

  // budget split over the pools by `share` (one entry per pool, summing to 1)
  bool init(int NL_, int NE_, int K_, const std::vector<int>& step_layer_, const std::vector<int>& layer_pool_,
            const std::vector<size_t>& pool_rec, const std::vector<double>& share, const std::string& dir,
            const std::string& layer_file, double budget, int readers, int split_, Mem mode, bool skip, std::string& err);
  ~Cache();
  SlotPool& pool(int L) { return P[layer_pool[L]]; }
  const SlotPool& pool(int L) const { return P[layer_pool[L]]; }
  uint8_t* rec(int L, int slot) {
    SlotPool& S = pool(L);
    if (mem != Mem::FILE) return S.row(slot);
    const size_t k = (size_t)L * NE + S.s[slot].e;
    if (core_n && core_tab[k] >= 0) return core_slab + (size_t)core_tab[k] * S.rec;
    return fmap[L] + (size_t)S.s[slot].e * S.rec;
  }
  int slot_of(int L, int e) const { return tab[(size_t)L * NE + e]; }
  bool ready(int L, int slot) const { return pool(L).pending[slot].load(std::memory_order_acquire) == 0; }
  const std::atomic<int>* pending_of(int L, int slot) const { return &pool(L).pending[slot]; }
  Slot& slot(int L, int i) { return pool(L).s[i]; }
  // smallest key among unpinned landed slots (free first); fit_names: the names a step makes for layer L (cap_fit)
  int victim(int L, int reserve, int fit_names = -1);
  int victim_in(int L, int reserve, bool capped);             // ... the layer's own records only when capped
  int pick_victim(int L, int reserve, double* val) const;     // value-ordered choice, no side effects; -1 if none
  void evict(int L, int slot);                                // take a slot chosen by pick_victim
  int steal(int L);                                           // a miss with every slot pinned: a slot held only by names
  bool wait_name_read(int L);  // none to steal either: wait for a name's read in flight (false: no such read)
  void issue(int L, int slot, int e, bool urgent, long step = 0, float pp = 0.f);
  void wait(int L, int slot);
  // a wait longer than park_us (DMENG_WAIT_PARK_US; < 0 never) calls on_long_wait once (the core parks its pool's idle
  // workers: they sleep through the read instead of spinning beside the readers)
  long park_us = -1;
  void (*on_long_wait)(void*) = nullptr;
  void* long_wait_ctx = nullptr;
  void use(int L, int slot, long t, long step);
  // an early name: pins; the slot, or -1 if none could be had; fit_names: the naming hook's names per step (cap_fit)
  int name(int L, int e, long step, int reserve = -1, int fit_names = -1);
  // name_gate: a name that needs a read is not issued while urgent reads wait for a reader (Readers::urgent_backlog):
  // the drive cannot serve it before them, and a read it starts now delays them (live sessions at small caches)
  bool name_gate = false;
                                                              // (reserve: slots left free for misses; -1: 2K + holds)
  void unpin(int L, int slot) {
    Slot& x = pool(L).s[slot];
    if (x.pins > 0) x.pins--;
  }
  void unpin_pf(int L, int slot, int e);                      // a name's pin, if the slot still holds that expert
  bool cancel(int L, int slot);                               // undo a speculative read that has not started
  // a name no longer wanted (e.g. for a position a verify rolled back): its pin goes, and its read too if it has not
  // started and nothing else pins the slot; true if the read was cancelled
  bool drop_name(int L, int slot, int e);
  int cancel_stale(long step);                                // drop queued speculative reads serving steps <= step
  // A background read for the drive's idle time: record (L, e), judged likely to be used by step `until` with probability
  // `value`. Read only if a free slot exists or the policy's victim has a keep-score below `value` (hazard: P(used at the
  // next token)); then the record keeps max(its keep-score, value) as its keep-score until step `until` has run (or its
  // first use), and is not pinned. Reads go to the readers' background class in bg_split pieces. Returns true if issued.
  // With per-layer caps, a layer at its cap gives the slot from its own records (as a name does), else the pool.
  // Names only schedule reads: the outputs cannot change.
  bool read_bg_value(int L, int e, double value, long until);
  int bg_split = 2;  // pieces of a background read (one in flight: an urgent read shares the drive with at most half a record)
  bool bg_issue = false;          // (issue: the read being issued is a background value read)
  std::vector<float> floor_v;     // [NL * NE] the value a background read gave the record, while floor_until >= now_step
  std::vector<long> floor_until;  // [NL * NE] (-1: none)
  std::vector<int> floor_keys;    // records with a floor (expired at expire_floors)
  void expire_floors();           // at a step's start: floors whose step has run go (their layers' minima recomputed)
  void learn_use(int L, const int32_t* picks, long t);
  void refresh_use_tab();
  int vbin(double key, long t) const {
    if (key < -1e299) return 0;
    double d = key - (double)t / half_life;
    int b = 1 + (int)std::floor((d + 8.0) * 1.5);
    return b < 1 ? 1 : (b >= VB ? VB - 1 : b);
  }
  void touch_nonvolatile(int L, int slot);  // purgeable mode
  void token_end_volatile(long t);
  int held(int pool) const;
  int slots(int pool) const { return P[pool].n; }

 private:
  void file_compact(int p, int want);                // FILE shrink: drop the lowest-value entries, pack below `want`
  void core_refresh();
  void drop(int L, int slot, bool count_pf_unused);  // the slot forgets its record
  bool slot_memory(int p, int i, bool keep);         // give a slot's memory back (keep = false) or take it again (false: the lock was refused)
};

}  // namespace dm
