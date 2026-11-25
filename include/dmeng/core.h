// The model-independent core: the expert store, the MoE step (released router, early-address hooks, slots and reads,
// expert compute, fixed-order combination) in lossless or approximate mode, and the per-token bookkeeping.
//
// A model's forward pass calls core.moe_step(step, x, y, view) at each MoE application ("step"): x is the MoE block's
// input in fp32 (what the early-address routers call h), y receives the block's output. Steps are numbered per token:
// = the layer for a model with one MoE block per layer, = the application for a looped model (Inkling: 48).
//
// Lossless (the default): the released router decides at every step. Named experts only schedule reads; a pick that has
// not landed is waited for and a pick that was not named is read on the spot. Every output row is computed by one thread
// and the experts are combined in pick order, so token ids never depend on the cache size, the routers or the timing.
#pragma once
#include <TargetConditionals.h>

#include <atomic>
#include <functional>

#include "dmeng/expert_step.h"
#include <memory>
#include <string>
#include <vector>

#include "dmeng/store.h"
#include "dmeng/weights.h"

namespace dm {

// What the core needs to know about a model, filled by the model from its resident file (later: the manifest).
struct ModelInfo {
  std::string family;
  int hidden = 0, vocab = 0, vocab_out = 0;  // vocab_out: logits actually produced (Inkling: the unpadded vocabulary)
  int layers = 0, steps = 0;                 // physical layers, MoE applications per token
  std::vector<int> step_layer;               // [steps] physical layer of each step
  std::vector<int> step_pass;                // [steps] loop pass of each step, -1 outside a loop
  int experts = 0, top_k = 0;
  int shared_inter = 0;                      // intermediate size of the shared expert (0: none)
  std::vector<ExpertFormat> formats;         // record formats (one slot pool each)
  std::vector<int> layer_fmt;                // [layers] index into formats
  std::vector<double> pool_share;            // default budget share per format
  std::string layer_file;                    // printf pattern relative to the model folder, e.g. store/layer_%02d.bin
  std::vector<std::string> layer_paths;      // per layer, relative to the folder (a profile's own files); empty = layer_file
  int eos = -1;
  bool has_ref_path = false;                 // supports the exact fp32-activation reference path
  bool has_verify = false;                   // a multi-position verify pass (Model::verify): drafting; without one every front end refuses it
  // what this family's code supports, in every front end's JSON (dmeng's line, dmserve's READY, dm_stats_json)
  std::string caps_json() const {
    return std::string("{\"verify\":") + (has_verify ? "true" : "false") + ",\"ref_path\":" + (has_ref_path ? "true" : "false") + "}";
  }
  // every multi-position MoE of this model goes through Core::moe_chunk, which runs a union larger than the cache in
  // batches (phone sizes): the pools' run-time floor (min_cap) is then one step's pins (2K + 2), not a prefill layer's
  // union (Inkling; the 35B's batched prefill has its own MoE with no batches yet)
  bool union_batches = false;
  // Core::moe_chunk computes this model's 4-bit affine experts with the packed batched kernels (gemm_aff4_i16, inputs a
  // multiple of 256) from PACK_MIN users and gives the GPU its share of them (moe_gpu): the same bits as the multi-vector
  // path. Off: the multi-vector path, as before (the 35B's verify)
  bool chunk_aff4 = false;
};

// Read-only description of one MoE application, so the core can run the block itself.
struct MoeView {
  enum Rule { SOFTMAX_TOPK_RENORM, SIGMOID_BIAS_TOPK_SCALED } rule = SOFTMAX_TOPK_RENORM;
  bool renorm = true;                  // softmax rule: renormalise the top-K weights
  float route_scale = 1.f;             // sigmoid rule: weights x this after renormalising
  const Weight* router = nullptr;      // bf16 or f32 [experts, hidden], computed on the fp32 input
  const float* router_bias = nullptr;  // sigmoid rule: added to the scores for the selection only
  enum Act { SWIGLU, SOFTCAP_SWIGLU } act = SWIGLU;  // softcap: softcap(g, gate_cap) sigmoid(g) softcap(u, up_cap)
  float gate_cap = 0.f, up_cap = 0.f;
  bool shared = false;                 // a shared expert (8-bit resident) with a sigmoid scalar gate
  const Weight *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr, *sh_scalar_gate = nullptr;
};

// Stage times (ms) and weight bytes traversed, accumulated; the models add theirs to the same counters.
struct Times {
  double embed = 0, attn_proj = 0, attn_core = 0, attn_out = 0, depth = 0, msg = 0, router = 0, experts = 0, wait = 0,
         cache = 0, pf = 0, head = 0, misc = 0;
  double ra = 0;  // run-ahead (Core::ra_predict) inside the MoE steps' expert sections: out of `experts`, which it used to inflate
  double bytes_res = 0, bytes_exp = 0;
  void zero() { *this = Times(); }
};

// Approximate mode (opt-in): a released pick that is not in memory may be replaced by the released router's next-ranked
// expert that is (its top K among the experts in memory, weighted as the release weights its top K), except where a rule
// says to wait for it or read it now. With nothing replaced, set, weights and order are the lossless path's, bit for bit.
struct Approx {
  bool on = false;
  double late_ms = 1000;   // a named pick still in flight: < 0 replace at once; 0 use it if it landed while the step's
                           // other experts ran; T wait up to T ms after the router; >= 1000 always wait
  double read_w = 2;       // a missing pick whose released weight > read_w is read on the spot
  double read_mass = 2;    // ... and the largest missing picks until the step's missing released weight <= read_mass
  bool bg = true;          // a replaced miss is fetched in the background for the layer's next use
  int cold_floor = -1;     // replacing needs at least this many other landed experts of the layer, else the step runs
                           // lossless (-1: 2K; 0: never, the old MLX runner's released top-k among whatever is held)
  bool drop = false;       // leave a missing pick out instead of replacing it; the kept picks keep their released weights
                           // (the release's top-K weighting, no re-weighting over the kept set)
  bool prio = true, cancel = true;  // reader settings that go with it: nearest step first, stale reads dropped
  // statistics
  double steps = 0, w_replaced = 0, n_replaced = 0, n_read_now = 0, w_read_now = 0, n_late_waited = 0, n_late_got = 0,
         n_late_dropped = 0, n_bg = 0, steps_replaced = 0, w_replaced_max = 0, late_wait_ms = 0, cold = 0;
  double w_hist[6] = {0};  // replaced picks by released weight: <.02, <.05, <.1, <.2, <.4, >=.4
  double w_tok_cur = 0;          // released weight replaced / dropped in the current token (summed over its steps)
  std::vector<float> w_tok;      // ... per finished token since the last reset (mean and p90 in the stats)
  void reset_stats();
  // named rule sets, optionally followed by :w=W:mass=M:late=MS:bg=0|1; false if the name or a key is unknown
  bool preset(const std::string& spec);
  bool preset_base(const std::string& name);
  static const char* presets_help();
};

struct GpuGemm;  // src/core/gpu_gemm.h

struct CoreOpts {
  int threads = 8, readers = 4, split = 1;
  int qos = -1, reader_qos = -1;  // QoS class (qos_class_t) of the compute workers / the readers; -1: user-interactive /
                                  // user-initiated (the defaults before these options)
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  bool reader_io_low = true;      // the readers' disk I/O one tier below the app's (iOS: the UI's page-ins go first)
#else
  bool reader_io_low = false;     // macOS: off (the readers are the critical path; a lower tier is throttled by others' I/O)
#endif
  Prec prec = Prec::I16;
  bool ref = false;          // exact reference path (fp32 activations, scalar kernels)
  double cache_gb = 5.0, half_life = 8.0;
  double cache_start_gb = 0;  // > 0 and < cache_gb (mlock): slots for cache_gb reserved, only this much locked at start
  Mem mem = Mem::MLOCK;
  bool skip_reads = false;   // timing without reads (the slots keep whatever bytes they have)
  bool loop_hold = true;     // looped layers: what an earlier pass of the token used stays pinned until its last pass
  std::vector<double> pool_share;  // empty: the model's default
  // the eviction policy (results/cache-policy/POLICY.md): auto = the hazard keep-score, with uniform per-layer caps
  // while the cache holds fewer than 4 records per layer per pick (Cache::caps_rule); freq = the decayed use count of
  // the engines before (one pool), pred = Stage 3's estimate-aware order, hazard = hazard with caps as layer_caps says
  std::string evict = "auto";
  std::string layer_caps = "auto";  // auto (the rule above; hazard only) | uniform | none
};

struct Hook;

class Core {
 public:
  ModelInfo mi;
  CoreOpts o;
  std::unique_ptr<Pool> pool;
  Cache cache;
  Hook* hook = nullptr;
  Approx ns;
  Times t;
  int H = 0, NE = 0, K = 0, steps = 0;
  long tok_clock = 0;  // forward passes so far (the cache's time)
  int pos = 0;         // position of the token being computed
  // the routing of the last forward pass, [steps][K]: released picks / weights, and what was used (= released when
  // nothing was replaced)
  std::vector<int32_t> picks, used;
  std::vector<float> pw, uw;
  std::vector<double> wait_step;      // [steps] wait ms (decode statistics)
  std::vector<uint64_t> miss_step;    // [steps] picks read on the spot
  std::vector<int> next_same;         // [steps] the next step of the same physical layer in the token, -1 if none
  const int16_t* force = nullptr;     // forced routing [npos, steps, K] (timing with skipped reads)
  int force_n = 0;
  XBuf xin;                           // the MoE input of the step running now, quantised (the hooks' rows read it)
  bool cancel_stale = false;          // drop queued speculative reads whose step has run
  std::vector<const MoeView*> views;  // [steps] the model's MoE views (set by the model at bind), for hooks that need them
  // released picks by position, [position][steps][K] (null: not kept; -1: not run). moe_step and moe_chunk write the
  // picks of the positions they ran, so a position rolled back by a verify is overwritten when it runs again (a model's
  // own batched prefill, Qwen3.6's, does not write here).
  std::vector<int16_t>* pick_log = nullptr;
  // RUN-AHEAD (measurement version): when a lossless step is about to wait for a pick's read, the model continues one step
  // ahead from an approximate state (this step's output from the experts in memory only: ypart) and returns the next
  // step's predicted picks (best first, at most maxk); the core issues reads for the absent ones (as early names, below
  // the confirmed reads) and counts, when the real pass reaches that step, how many of its reads were already issued.
  // Only reads change: the model must leave its state as it was. Off unless the model sets it.
  std::function<int(int step, const float* ypart, int32_t* picks, int maxk)> run_ahead;
  std::vector<float> ra_lgv;  // the run-ahead's router logits (the model fills NE): fast mode names by their routing weights
  std::atomic<int64_t> exp_spin_ns{0};
  // the expert step's fork-joins in place: per number of jobs in the call (0..MAXK + 1), and split by whether a wait on a
  // read came between the call and the one before it: calls, ns, weight bytes
  double xc_calls[2][18] = {}, xc_ns[2][18] = {}, xc_bytes[2][18] = {};  // [after a wait][jobs, 0 .. 17]
  double xc_wait_seen = 0;
  // measurement (DMENG_RERUN_AB=1): STEP_RERUN off / on call by call: ns, weight bytes, calls; chunks re-run
  double rr_ns[2] = {0, 0}, rr_bytes[2] = {0, 0}, rr_calls[2] = {0, 0};
  std::atomic<int64_t> rr_count{0};
  std::string xc_json() const;
  int ra_k = 8;  // predicted picks named per run-ahead
  // MEASUREMENT of deeper run-ahead (no reads issued): after the run-ahead's step + 1, the model continues to step + d
  // (d = 2 .. ra_depth) from its approximate state with that step's MoE left out, and returns its top picks; each is scored
  // when the real pass reaches step + d (top-8 match, the share of its on-demand reads it would have issued)
  std::function<int(int depth, int32_t* picks, int maxk)> run_ahead_deeper;
  int ra_depth = 1;
  bool ra_off = false;
  // run-ahead limits (engine options a profile sets): speculative ms per token at most ra_budget x the token's compute
  // (running average; 0 = no limit), at most ra_max_tok run-aheads per token (0 = no limit); ra_sat_skip: none while
  // every reader is busy and reads are queued behind them (the drive has work: earlier names add no lead). (The next
  // layer's attention is the run-ahead's cost, and it is needed: the 35B's router on the residual without it matched
  // 0.79-0.83 of the top 8 and wasted two thirds of its names; PARITY.md.)
  // the experts' exact CPU + GPU split in moe_chunk (a model's batched prefill sets it for its passes): the GPU computes
  // the first moe_gpu_frac of each packed NVFP4 expert's gate / up chunks and down rows (gpu_gemm.h add_nvfp4: the CPU
  // kernel's bits); null = CPU only
  GpuGemm* moe_gpu = nullptr;
  double moe_gpu_frac = 0;
  double moe_gpu_wait_ms = 0, moe_gpu_stage_ms = 0, moe_gpu_macs = 0, moe_cpu_macs = 0;
  uint64_t moe_gpu_sets = 0, moe_gpu_failed = 0;
  double ra_budget = 0;
  int ra_max_tok = 0;
  bool ra_sat_skip = false;
  double ra_tok_ms = 0, ra_comp_ema = 0, ra_tok_wait0 = 0;
  int ra_tok_events = 0;
  std::chrono::steady_clock::time_point ra_tok_t0{};  // measurement (DMENG_RA_AB=1): run-ahead skipped for this token
  struct Deep {
    int step = -1;
    std::vector<int32_t> pred;
    std::vector<uint8_t> absent;
  };
  std::vector<Deep> ra_deep[4];  // [depth - 1][target step]
  double deep_top8[4] = {0}, deep_n[4] = {0}, deep_would[4] = {0}, deep_found[4] = {0}, deep_events[4] = {0}, deep_issued[4] = {0}, deep_wasted[4] = {0};
  // where the remaining wait sits (run-ahead on): at steps the previous step named reads for, or not (a chain's first miss);
  // steps that never waited (where a run-ahead with no miss would have to start)
  bool ra_cov = false;
  double wait_cov_ms = 0, wait_uncov_ms = 0, steps_wait_cov = 0, steps_wait_uncov = 0, steps_no_wait = 0, steps_total = 0;
  std::string ra_json() const;
  struct RunAhead {
    int step = -1;                 // the step predicted (-1: none pending)
    std::vector<uint8_t> absent;   // [NE] absent from memory when the prediction was made
    std::vector<int32_t> pred;     // predicted picks
    std::vector<int> pslot;        // their pinned slots (-1: none had)
    clk::time_point t;
    const std::atomic<int>* watch = nullptr;  // the read the step was about to wait for (0: landed)
    clk::time_point seen;                     // when the model first saw it landed (the overrun starts)
    bool landed = false;
    double events = 0, ms = 0, would_read = 0, found = 0, lead_ms = 0, issued = 0, wasted = 0, top8 = 0, top8_n = 0;
    double aborted = 0, over_ms = 0, refused = 0, withheld = 0, skip_budget = 0, skip_sat = 0;  // withheld: absent, below fast mode's read rule
  } ra;
  int ra_reserve = 0;  // slots a run-ahead name leaves free (as a miss: 0; the early routers' names leave 2K + holds)
  // the model, between its run-ahead sub-stages: true once the awaited read has landed (stop: the work is thrown away)
  bool ra_stopped() {
    if (ra.landed) return true;
    if (ra.watch && ra.watch->load(std::memory_order_acquire) == 0) {
      ra.landed = true;
      ra.seen = clk::now();
    }
    return ra.landed;
  }

  bool init(const ModelInfo& info, const CoreOpts& opts, const std::string& dir, std::string& err);
  // per token: the model calls token_start before its first MoE step and token_end after its last
  void token_start(int position, int tok, const float* router_emb);
  void moe_step(int step, const float* x, float* y, const MoeView& v);
  void token_end();
  // EXTENSION (batched prefill, multi-position verify): one MoE application for n positions (core.pos + i, the cache's
  // time tok_clock + i), lossless. x[i] / y[i]: position i's input / output; position i's released picks / weights go to
  // pk[i * K ..] / w[i * K ..] (null: not kept). Each position's routing, expert arithmetic and combination order are the
  // single step's (the multi-vector kernels are bit-identical to the one-vector ones), so the outputs equal n single
  // steps'; the union of the positions' experts is pinned and read once and each expert computed for every position that
  // picked it (NVFP4 from 6 users through the packed i8mm kernel). No hooks, forced routing or approximate mode inside.
  // The caller advances tok_clock / pos by n afterwards (chunk_end). off: the chunk is positions [off, off + n) of a longer
  // span run application by application (Inkling's span prefill): position i's clock is then tok_clock + off + i.
  void moe_chunk(int step, int n, const float* const* x, float* const* y, const MoeView& v, int32_t* pk = nullptr, float* w = nullptr,
                 int off = 0);
  // a span run application by application: no loop holds (a layer's next pass comes after every other looped layer of the
  // span, so holding its experts would only pin the pool; outputs never depend on it)
  bool span_mode = false;
  // In span mode a step's union stays pinned from one sub-chunk to the next (the later sub-chunks hit it, and the next
  // application's names cannot evict it) until span_unname(step) after its last sub-chunk.
  // A span's next application named early: every expert of `step`'s layer, read behind the running step's own misses and
  // pinned until span_unname(step). Names only: the outputs cannot change.
  void span_name(int step);
  void span_unname(int step);
  std::vector<std::vector<std::pair<int, int>>> span_names;  // [steps] (slot, expert) named
  std::vector<std::vector<int>> span_hold;                   // [steps] the step's union pins kept across its sub-chunks
  void chunk_end(int n) {
    for (int i = 0; i < n; i++) token_end();
    chunk_named = false;
  }
  // Names inside a multi-position verify (n <= 8 positions from core.pos): the model calls chunk_start with the
  // positions' token ids and router embeddings before the pass's first MoE step; while the pass runs, the hooks name
  // every position's experts of later steps (Hook::chunk_*), as single steps do. Names only schedule reads (deadline: the
  // chunk's own step, so they are served in the order the pass needs them); the output cannot change. Off with
  // chunk_names = false (dmeng --verify-names 0).
  bool chunk_names = true;
  bool chunk_named = false;  // the running chunk is named (set by chunk_start)
  void chunk_start(int n, const int32_t* toks, const float* const* embs);
  clk::time_point step_t0;  // the running MoE step's start (Naming's lead statistics)
  // A multi-position verify kept the first a positions of its last chunk: the hooks see the MoE inputs of the last kept
  // position (Hook::observe), as a single step there would have shown them, so trained routers name the next token from
  // the right previous hidden states. Names only: nothing computed changes. moe_chunk keeps the inputs while a hook is set.
  void chunk_commit(int a);
  std::vector<float> chunk_x;  // [steps][chunk_xn][H] the last chunk's MoE inputs (kept only while a hook is set)
  int chunk_xn = 0;
  // released picks of position p at `step` into pick_log (when it is kept); a model's own batched prefill calls it
  void log_picks(int p, int step, const int32_t* pk);
  long n_hold_release = 0;  // chunks whose union needed the loop holds' slots (holds released early: no effect on outputs)
  long n_union_batches = 0; // chunk applications whose union did not fit and ran in batches (Core::moe_chunk; phone sizes)
  struct Chunk;                // the chunk step's buffers (moe_chunk.cpp)
  std::shared_ptr<Chunk> ch_;
  long now(int step) const { return tok_clock * steps + step; }
  long next_abs(int step) const;      // the next occurrence of this step index after the step running now
  long next_use(int step) const;      // the next application of the step's physical layer after this one
  bool looped(int step) const { return mi.step_pass[step] >= 0; }
  void reset_stats();                 // cache, approximate mode, per-step counters, times

 private:
  using ExpJob = ExpertJob;
  std::vector<XBuf> act;              // [K + 1] expert activations, quantised (K = shared expert)
  std::vector<float> eout, actf, lg, sc, bz;
  std::vector<int> ord;
  std::vector<std::vector<int>> holds, used_tok;
  void route(const MoeView& v, int step);
  void weights(const MoeView& v, const int* set, int n, float* w) const;
  void run_experts(ExpJob* jobs, int n, const MoeView& v, const float* x);
  void combine(const MoeView& v, const float* w, const int* rows, int n, float shw, float* y) const;
  void finish_step(int step, const int* slots, int n);
  void lossless(int step, const float* x, float* y, const MoeView& v);
  // run-ahead from a step about to wait for the read whose pending count is `watch`
  void ra_predict(int step, const MoeView& v, const bool* done, const float* w, const std::atomic<int>* watch);
  void ra_where(double wait_ms);  // where this step's wait sits
  void ra_score(int step, const int32_t* pk);                                     // the real picks of a predicted step
  void approx(int step, const float* x, float* y, const MoeView& v);
  int take_slot(int L, int e, int step);  // a slot for a miss read now (urgent): pins it
  ExpJob shared_job(const MoeView& v) const;
  float shw_ = 0.f, mx_ = 0.f;
  double zsum_ = 0;
  int sh_inter_ = 0, I_ = 0;
};

// Hooks that name a step's experts early (oracle replay, trained routers, the read planner). Naming only schedules reads
// and pins what is named until its step has run; the output cannot change.
struct Hook {
  virtual ~Hook() {}
  virtual void token_start(Core&, int tok, const float* emb) {}
  virtual void before_moe(Core&, int step, const float* x) {}
  // the MoE input of `step` of a position that ran outside the hooks (the last kept position of a multi-position verify),
  // for hooks that keep the previous token's inputs; called once per step after the chunk is committed
  virtual void observe(Core&, int step, const float* x) {}
  // rows to compute in the released router's own fork-join (input: this step's MoE input, x fp32 / core.xin quantised)
  virtual void router_rows(Core&, int step, const float* x, std::vector<Job>& jobs) {}
  virtual void after_router(Core&, int step) {}  // the step's own misses have been issued
  virtual void after_moe(Core&, int step, const int32_t* picks, const float* w) {}
  virtual void release(Core&, int step) {}       // the step has run
  virtual void reset_stats() {}
  virtual std::string stats_json(Core&, int ntok) { return "{}"; }
  virtual const int* names() const { return nullptr; }  // [steps, 16] this token's top-16 names per step (routers)
  // names inside a multi-position verify (Core::chunk_start): positions core.pos + i, i < n
  virtual void chunk_start(Core&, int n, const int32_t* toks, const float* const* embs) {}
  // rows to run in the chunk's released-router fork-join (inputs: every position's MoE input at `step`)
  virtual void chunk_rows(Core&, int step, int n, const XQ* const* xq, const float* const* x, std::vector<JobN>& jobs) {}
  virtual void chunk_after_router(Core&, int step, int n) {}            // the chunk's own misses have been issued
  virtual void chunk_after_moe(Core&, int step, int n, const int32_t* pk) {}  // [n][K] released picks
  virtual void chunk_release(Core&, int step, int n) {}                 // the step has run for every position
};

}  // namespace dm
