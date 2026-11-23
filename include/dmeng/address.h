// The early address: hooks that name a step's experts before the step runs, so their reads overlap the steps between.
// Naming only schedules reads and pins what is named until its step has run; at the step the released router decides,
// a pick that has not arrived is waited for, and a pick that was not named is read on the spot. The output cannot change.
//
// Indexing is by step (= layer for a model with one MoE block per layer, = application for a looped model), the index
// the trained routers are trained on.
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dmeng/core.h"

namespace dm {

// Common part of the hooks that name single experts: pins per target step, by the parity of the target token.
// Pins per target position (a ring of 16: a verify's positions and the token after them) and step.
struct Naming : Hook {
  static constexpr int PR = 16;
  struct Pin {
    int slot, e;
    clk::time_point t;  // when it was named (the lead statistics)
  };
  std::vector<std::vector<Pin>> pins[PR];  // [position & 15][step]
  void ensure(int steps);
  // name expert e of `step` of the token at position p (c.pos <= p < c.pos + PR); deadline: the cache's step time the read
  // serves (default: that token's own step time; a verify's names pass the chunk's step, the order the pass runs them)
  void name(Core& c, int p, int step, int e, long deadline = -1);
  void release_at(Core& c, int p, int step, bool chunk);
  void release(Core& c, int step) override { release_at(c, c.pos, step, false); }
  void chunk_release(Core& c, int step, int n) override {
    for (int i = 0; i < n; i++) release_at(c, c.pos + i, step, true);
  }
  // lead of the names a step used: ms from naming to the step's start, [0] one-position steps, [1] verify passes
  double lead_ms[2] = {0, 0}, lead_n[2] = {0, 0};
  std::string lead_json() const;
};

// Replays a recorded routing (dmeng --record): after step s's router it names step s + lead's experts of the same token
// (lead < steps), or, with lead = token, the next token's experts of step s (one whole token early). Steps s < lead of a
// token are named at the token start.
struct Oracle : Naming {
  std::vector<int16_t> rec;  // [npos, steps, K]
  int npos = 0, ns = 0, k = 0, lead = 4;
  bool token_lead = false;
  bool load(const std::string& path, int steps, int K, std::string& err);
  void name_step(Core& c, int p, int s, long deadline = -1);
  void token_start(Core& c, int tok, const float* emb) override;
  void after_router(Core& c, int step) override;
  // inside a verify: steps < lead of every position at the chunk's start, step s + lead after step s (lead < steps)
  void chunk_start(Core& c, int n, const int32_t* toks, const float* const* embs) override;
  void chunk_after_router(Core& c, int step, int n) override;
  int cn = 0;
};

// The free baseline ("transplant k1"): step s + 1's released router, run on step s's MoE input in step s's router
// fork-join, names step s + 1's top m experts. No trained weights; step 0 is not named.
struct Transplant : Naming {
  int m = 8;
  int pending = -1;
  std::vector<float> lg;
  std::vector<int> nm;  // [steps, 8] this token's names
  double t_ms = 0, calls = 0, hit8 = 0;
  void router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) override;
  void after_router(Core& c, int step) override;
  void after_moe(Core& c, int step, const int32_t* picks, const float* w) override;
  void reset_stats() override { t_ms = calls = hit8 = 0; }
  std::string stats_json(Core& c, int ntok) override;
};

// Trained lag-k routers, one per target step, on x = [rmsnorm_w(emb), h_prev, h_lag] (h_lag absent for t < lag):
//   z = lin(x) [+ m2 gelu(m1 x + b1) + b2]   (the "lin" rows and "m1" rows are one matrix `pre` over the first two blocks
//   plus `lagm` over the third), named experts = the top m of z, or (tau >= 0) every one of the top `cand` whose
//   calibrated P(in the released top-K | rank, softmax q) clears tau (counted online from this run's own released picks).
// The first two blocks' rows of every router run at the token start; the lag block's rows of router t ride in step
// t - lag's released-router fork-join. Engine file: DMENGRES1, 8- or 4-bit matrices, f32 vectors.
struct Routers : Naming {
  struct Lay {
    const float* emb_norm = nullptr;
    Weight pre, lagm, m2;
    Weight pre_e, pre_p;  // split files: pre's embedding columns and previous-token columns as two matrices
    bool has_lag = false;
    const float *b_pre = nullptr, *m2_b = nullptr;
  };
  ResFile R;
  Mem mem = Mem::MLOCK;  // how the router file is held (set before load; iOS: PLAIN)
  int lag = 0, nt = 0, hid = 0, H = 0, NE = 0, rows = 0, bits = 8;
  int m = 8, cand = 16, ecap = 0;
  double tau = -1;
  bool active = true, split = false;
  bool feed = false;  // give the cache this router's rows even when it names nothing (a learned eviction ranker)
  std::vector<Lay> lay;
  std::vector<float> prev, pre_out, lag_out, e_out, e_pool;
  std::vector<XBuf> xpre, xe, xp;
  XBuf xh;
  int pending = -1;          // the target whose lag rows run in the current router phase
  std::vector<int> nm;       // [targets, 16] this token's top-16 per target (-1: not computed)
  std::vector<float> nq;     // [targets, 16] their softmax q
  std::vector<int> nbin;     // [targets, 16] calibration bin
  // per-token cache of the embedding columns' output (split files), least recently used goes
  std::vector<int> e_tok;
  std::vector<long> e_age;
  std::unordered_map<int, int> e_map;
  long e_clock = 0;
  // statistics: [group 0 targets < lag (named at the token start), 1 lagged][m 8, 12, 16]
  double cnt[2][3] = {{0}}, msum[2][3] = {{0}}, calls[2] = {0, 0}, t_ms = 0, e_hits = 0, e_calls = 0, bytes_tok = 0;
  double cal_n[16][16] = {{0}}, cal_h[16][16] = {{0}};
  const float* emb = nullptr;
  // a heavy-pick head (objective d of the 35B design study: a second router of the same file format, run beside this
  // one and naming nothing itself): this router names the head's positives (logit > 0) first, then the rest by its own
  // logits. zlast: this token's logits per target.
  Routers* heavy = nullptr;
  std::vector<float> zlast;
  bool load(const std::string& path, Core& c, std::string& err);
  void finish(Core& c, int t, bool with_lag);
  // z = pre + biases (+ lag) of target t -> names for position p (deadline: see Naming::name); writes n16 / q16 / b16
  void finish_z(Core& c, int t, const float* pre, const float* lagr, int p, long deadline, int* n16, float* q16, int* b16);
  const float* e_lookup(int tok);
  // names inside a verify (Core::chunk_start): per position i: the first two blocks' rows with [emb_i, prev] (prev = the
  // MoE inputs of the last committed token: exact for position 0, older by i tokens for position i), the targets < lag
  // named at the chunk's start; then at every step s each position's lag rows of target s + lag (the chunk's own
  // fork-join) and its names. Not for split router files (chunk naming off there).
  int cn = 0;
  std::vector<float> cpre, clag;   // [n][nt][rows], [n][rows]
  std::vector<int> cnm, cbin;      // [n][nt][16]
  std::vector<float> cnq;
  std::vector<XBuf> cx;            // [n * nt] the pre inputs
  std::vector<float*> clag_p;
  double ccnt[8][3] = {{0}}, ccalls[8] = {0};  // per position index: released top-K inside the top 8 / 12 / 16 names
  void chunk_start(Core& c, int n, const int32_t* toks, const float* const* embs) override;
  void chunk_rows(Core& c, int step, int n, const XQ* const* xq, const float* const* x, std::vector<JobN>& jobs) override;
  void chunk_after_router(Core& c, int step, int n) override;
  void chunk_after_moe(Core& c, int step, int n, const int32_t* pk) override;
  void token_start(Core& c, int tok, const float* emb) override;
  void before_moe(Core& c, int step, const float* x) override;
  void observe(Core& c, int step, const float* x) override;
  void router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) override;
  void after_router(Core& c, int step) override;
  void after_moe(Core& c, int step, const int32_t* picks, const float* w) override;
  void reset_stats() override;
  std::string stats_json(Core& c, int ntok) override;
  const int* names() const override { return nm.data(); }
};

// Cross-token routers (trained, e.g. Inkling's ink_x_main xtok_s41): after the source step S of token t has issued its own
// misses, the router of target step a (a < S) on [h_t@S, h_t@a] (both this token's MoE inputs) names token t + 1's experts
// of step a, with the calibrated threshold rule (its own calibration, from the released picks of the next token). A whole
// token of lead for the steps that start a token. Wraps another hook (the in-token routers), which sees every call.
// Engine file (python converter: kind ink_xtok): per target a: a.pre [NE + hid, 2H] (rows [lin; mlp.0]), a.b_pre, a.m2 [NE,
// hid], a.m2_b; cfg source_app, targets, hidden, experts, hidden_mlp, apps.
struct Xtok : Naming {
  struct Lay {
    Weight pre, m2;
    const float *b_pre = nullptr, *m2_b = nullptr;
  };
  Hook* inner = nullptr;
  ResFile R;
  int src = 0, H = 0, NE = 0, hid = 0, rows = 0, cand = 16;
  double tau = 0.5;
  bool background = false;  // its reads go to the readers' background class (start only while this token's reads are done)
  std::vector<int> tg, slot_of;
  std::vector<Lay> lay;
  std::vector<float> hin, xsrc, out;
  std::vector<XBuf> xin;
  XBuf xh;
  std::vector<int> nm_next, nm_cur, bin_next, bin_cur;  // [targets, 16]
  double cal_n[16][16] = {{0}}, cal_h[16][16] = {{0}};
  double cnt[2] = {0, 0}, calls = 0, t_ms = 0, named = 0;
  bool load(const std::string& path, Core& c, std::string& err);
  void token_start(Core& c, int tok, const float* emb) override;
  void before_moe(Core& c, int step, const float* x) override;
  void observe(Core& c, int step, const float* x) override;
  void router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) override {
    if (inner) inner->router_rows(c, step, x, jobs);
  }
  // inside a verify the in-token routers name (forwarded); this router names nothing there (its next token is not known
  // until the pass is committed)
  void chunk_start(Core& c, int n, const int32_t* toks, const float* const* embs) override {
    if (inner) inner->chunk_start(c, n, toks, embs);
  }
  void chunk_rows(Core& c, int step, int n, const XQ* const* xq, const float* const* x, std::vector<JobN>& jobs) override {
    if (inner) inner->chunk_rows(c, step, n, xq, x, jobs);
  }
  void chunk_after_router(Core& c, int step, int n) override {
    if (inner) inner->chunk_after_router(c, step, n);
  }
  void chunk_after_moe(Core& c, int step, int n, const int32_t* pk) override {
    if (inner) inner->chunk_after_moe(c, step, n, pk);
  }
  void chunk_release(Core& c, int step, int n) override {
    Naming::chunk_release(c, step, n);
    if (inner) inner->chunk_release(c, step, n);
  }
  void after_router(Core& c, int step) override;
  void after_moe(Core& c, int step, const int32_t* picks, const float* w) override;
  void release(Core& c, int step) override;
  void reset_stats() override;
  std::string stats_json(Core& c, int ntok) override;
  const int* names() const override { return inner ? inner->names() : nullptr; }
};

// The read planner (Stage 3). One or more trained router sets (a cascade: every set names every step it can, the longest
// lag first, shorter lags refine); scores calibrated online to P(in the released top-K); a speculative read is issued
// only for a record not held, in order of probability, while the marginal-cost threshold from the law tokens/s =
// bandwidth / bytes per token allows (or a per-token byte budget), and (with estimate-aware eviction) only if it is
// likelier to be used than the slot it would take. A shorter-lag set can revoke a longer one's read that has not started.
struct Planner : Hook {
  struct Net {
    ResFile R;
    int lag = 0, hid = 0, rows = 0, id = 0;
    std::vector<Routers::Lay> lay;
    std::vector<float> pre_out, lag_out, q;  // q: [steps * NE] this token's softmax per target
    std::vector<uint8_t> rk;                 // [steps * NE] its rank (capped at 255)
    std::vector<int> qg;                     // [steps] calibration group of that row (-1: none)
    std::vector<XBuf> xpre;
    XBuf xh;
    int pending = -1;
    double issued = 0, pinned_cached = 0, revoked = 0, skip_budget = 0, skip_evict = 0, skip_tau = 0;
  };
  std::vector<std::unique_ptr<Net>> nets;  // longest lag first
  int NS = 0, NE = 0, H = 0, maxlag = 0;
  std::vector<float> prev;
  static constexpr int NR = 20, NQ = 12, NB = NR * NQ;  // rank bins x 1.5-octave bins of q (2^-18 .. 1)
  bool calib_q_only = false;
  std::vector<double> c_n, c_pos;
  std::vector<float> c_tab;
  std::vector<std::vector<int>> pins;  // per target step
  std::vector<uint8_t> state;          // [steps * NE] 0 none, 1 pinned (held), 2 read by us (pinned)
  std::vector<int> iss_slot;           // [steps * NE]
  bool budget_on = true, revoke_on = true, active = true, stats_on = false;
  int budget_mode = 1;  // 1: marginal-cost threshold from the law (default); 2: byte bucket
  double u = 0.9, tau_long = 0.5, tau_fixed = -1, pin_p = 0.2, revoke_p = 0.1;
  int maxcand = 32;
  double bw = 3.0e9, comp_ms = 40, urg_b = 0, allow = 0, credit = 0, tot_b = 0, t_miss = 0.6, rho = 0, thr_law = 0;
  uint64_t bp_last = 0, nm_last = 0, bu_last = 0, rdb_last = 0, rdw_last = 0, rdbusy_last = 0;
  double wm_last = 0, wait_last = 0;
  bool first = true;
  clk::time_point t_last;
  double t_ms = 0, ntok = 0, sum_allow = 0, sum_bw = 0, sum_comp = 0, sum_urg = 0, sum_thr = 0, sum_rho = 0, sum_tmiss = 0;
  double cal_sum_p[8] = {0}, cal_pick_p[8] = {0}, cal_calls[8] = {0};
  const float* emb = nullptr;
  bool load(const std::vector<std::string>& paths, Core& c, std::string& err);
  int qbin(float q) const {
    if (!(q > 0.f)) return 0;
    int b = (int)std::floor((std::log2(q) + 18.f) / 1.5f);
    return b < 0 ? 0 : (b >= NQ ? NQ - 1 : b);
  }
  static int rbin(int r) { return r < 16 ? r : r < 20 ? 16 : r < 24 ? 17 : r < 32 ? 18 : 19; }
  int cell(int r, float q) const { return (calib_q_only ? 0 : rbin(r)) * NQ + qbin(q); }
  void refresh_calib();
  void controller(Core& c);
  void stage(Core& c, Net& n, int T, bool with_lag);
  void token_start(Core& c, int tok, const float* emb) override;
  void before_moe(Core& c, int step, const float* x) override;
  void observe(Core& c, int step, const float* x) override;
  void router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) override;
  void after_router(Core& c, int step) override;
  void after_moe(Core& c, int step, const int32_t* picks, const float* w) override;
  void release(Core& c, int step) override;
  void reset_stats() override;
  std::string stats_json(Core& c, int ntok) override;
};

}  // namespace dm
