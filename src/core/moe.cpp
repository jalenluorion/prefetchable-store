// The MoE step: the released router, the early-address hook points, the slots and reads, the expert compute in two
// passes (experts already in memory first, then the rest as they land), the fixed-order combination, and the
// approximate mode. One routine for every model: the router rule, the activation, the shared expert and the record
// format come from the step's MoeView and the store's record formats.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>

#include <arm_neon.h>

#include "dmeng/core.h"

namespace dm {

constexpr int MAXK = 16;

// ------------------------------------------------------------------------------------------------ approximate presets
void Approx::reset_stats() {
  steps = w_replaced = n_replaced = n_read_now = w_read_now = n_late_waited = n_late_got = n_late_dropped = n_bg = 0;
  steps_replaced = w_replaced_max = late_wait_ms = cold = 0;
  for (auto& h : w_hist) h = 0;
  w_tok_cur = 0;
  w_tok.clear();
}

bool Approx::preset(const std::string& name) {
  if (name == "bounded" || name.empty()) {  // wait for named picks in flight, read on the spot any pick with weight > 0.1
    late_ms = 1000;
    read_w = 0.1;
    read_mass = 2;
    bg = true;
    drop = false;
  } else if (name == "drop") {  // P2 (phone sizes): as bounded, but a missing pick of weight <= 0.1 is left out (no
                                // substitute, no renormalisation of the kept weights), so no floor of experts in memory
                                // is needed; no background fetch of the dropped picks
    late_ms = 1000;
    read_w = 0.1;
    read_mass = 2;
    bg = false;
    drop = true;
  } else if (name == "never") {  // never wait: replace whatever has not landed (a speed bound, not for use)
    late_ms = -1;
    read_w = 2;
    read_mass = 2;
    bg = true;
    drop = false;
  } else {
    return false;
  }
  on = true;
  prio = cancel = true;
  return true;
}

const char* Approx::presets_help() {
  return "bounded (wait for named picks in flight, read any missing pick with released weight > 0.1, replace the rest), "
         "drop (as bounded, but the rest is left out: no substitute, released weights kept, no floor of experts in memory), "
         "never (replace whatever has not landed: a speed bound)";
}

// ------------------------------------------------------------------------------------------------ setup
bool Core::init(const ModelInfo& info, const CoreOpts& opts, const std::string& dir, std::string& err) {
  mi = info;
  o = opts;
  H = mi.hidden;
  NE = mi.experts;
  K = mi.top_k;
  steps = mi.steps;
  if (K > MAXK) {
    err = "top-k above 16";
    return false;
  }
  if (o.ref && !mi.has_ref_path) {
    err = "this model has no fp32 reference path";
    return false;
  }
  pool.reset(new Pool(o.threads, -1, o.qos));  // spin: DMENG_SPIN_US (default 2000)
  if (const char* e = getenv("DMENG_WAIT_PARK_US")) {  // idle workers sleep through expert reads longer than this
    cache.park_us = atol(e);
    cache.on_long_wait = [](void* p) { ((Pool*)p)->park(); };
    cache.long_wait_ctx = pool.get();
  }
  cache.rd.qos = o.reader_qos;
  std::vector<size_t> rec;
  for (auto& f : mi.formats) rec.push_back(f.rec);
  std::vector<double> share = o.pool_share.empty() ? mi.pool_share : o.pool_share;
  if (share.empty()) share.assign(rec.size(), 1.0 / rec.size());
  cache.defer_lock = o.cache_start_gb > 0 && o.cache_start_gb < o.cache_gb && o.mem == Mem::MLOCK;
  if (!cache.init(mi.layers, NE, K, mi.step_layer, mi.layer_fmt, rec, share, dir, mi.layer_file, o.cache_gb * 1e9, o.readers,
                  o.split, o.mem, o.skip_reads, err))
    return false;
  if (cache.defer_lock) cache.lock_start((int64_t)(o.cache_start_gb * 1e9));
  cache.half_life = o.half_life;
  if (o.evict == "pred") {
    cache.pred_evict = cache.learn_on = true;  // measured worse than freq (Stage 3)
  } else if (o.evict == "hazard" || o.evict == "auto") {
    cache.enable_hazard();
  } else if (o.evict != "freq") {
    err = "evict: auto|freq|pred|hazard, got " + o.evict;
    return false;
  }
  if (o.layer_caps == "uniform") {
    cache.uniform_layer_caps();
  } else if (o.layer_caps == "auto") {
    cache.caps_auto = cache.hazard_evict;
    if (cache.caps_auto) cache.caps_auto_apply();
  } else if (o.layer_caps != "none") {
    err = "layer caps: auto|uniform|none, got " + o.layer_caps;
    return false;
  }
  next_same.assign(steps, -1);
  for (int s = 0; s < steps; s++)
    for (int s2 = s + 1; s2 < steps; s2++)
      if (mi.step_layer[s2] == mi.step_layer[s]) {
        next_same[s] = s2;
        break;
      }
  if (o.loop_hold)  // the loop hold's worst case: K slots per step that holds, in that step's pool
    for (int s = 0; s < steps; s++)
      if (next_same[s] >= 0) cache.extra_reserve[mi.layer_fmt[mi.step_layer[s]]] += K;
  I_ = 0;
  for (auto& f : mi.formats) I_ = std::max(I_, f.I);
  sh_inter_ = mi.shared_inter;
  const int IA = std::max(I_, sh_inter_);
  xin.init(H, o.prec);
  act.resize(K + 1);
  for (auto& a : act) a.init(IA, o.prec);
  eout.assign((size_t)(K + 1) * H, 0.f);
  actf.assign((size_t)(K + 1) * IA, 0.f);
  lg.assign(NE + 1, 0.f);
  sc.assign(NE, 0.f);
  bz.assign(NE, 0.f);
  ord.resize(NE);
  picks.assign((size_t)steps * K, -1);
  used.assign((size_t)steps * K, -1);
  pw.assign((size_t)steps * K, 0.f);
  uw.assign((size_t)steps * K, 0.f);
  wait_step.assign(steps, 0.0);
  miss_step.assign(steps, 0);
  holds.assign(mi.layers, {});
  used_tok.assign(mi.layers, {});
  return true;
}

void Core::reset_stats() {
  cache.st = CacheStats();
  ns.reset_stats();
  std::fill(wait_step.begin(), wait_step.end(), 0.0);
  std::fill(miss_step.begin(), miss_step.end(), 0);
  t.zero();
  if (hook) hook->reset_stats();
}

long Core::next_abs(int step) const {
  const long n = cache.now_step;
  long b = (n / steps) * steps + step;
  return b <= n ? b + steps : b;
}

long Core::next_use(int step) const {
  if (next_same[step] >= 0) return now(next_same[step]);
  const int L = mi.step_layer[step];
  int first = step;
  for (int s = 0; s < steps; s++)
    if (mi.step_layer[s] == L) {
      first = s;
      break;
    }
  return (tok_clock + 1) * steps + first;
}

// ------------------------------------------------------------------------------------------------ per token
void Core::token_start(int position, int tok, const float* router_emb) {
  pos = position;
  if (cache.budget_req.load(std::memory_order_relaxed) >= 0) cache.apply_budget();  // a run-time budget (any thread)
  if (hook) {
    auto tp = clk::now();
    hook->token_start(*this, tok, router_emb);
    t.pf += ms_since(tp);
  }
}

void Core::token_end() {
  tok_clock++;
  if (cache.mem == Mem::FILE && (cache.fa.on || cache.core_n)) {  // adaptive budget, locked core
    auto tc = clk::now();
    cache.token_end_file(tok_clock);
    t.cache += ms_since(tc);
  }
  if (cache.mem == Mem::PURGEABLE) {
    auto tc = clk::now();
    cache.token_end_volatile(tok_clock);
    t.cache += ms_since(tc);
  }
}

// ------------------------------------------------------------------------------------------------ routing
void Core::route(const MoeView& v, int step) {
  int32_t* pk = picks.data() + (size_t)step * K;
  float* w = pw.data() + (size_t)step * K;
  for (int e = 0; e < NE; e++) ord[e] = e;
  const bool forced = force && pos < force_n;
  if (v.rule == MoeView::SOFTMAX_TOPK_RENORM) {
    // softmax over all experts (sum in double), top-K (ties: lower index), renormalised
    float mx = -INFINITY;
    for (int e = 0; e < NE; e++) mx = std::max(mx, lg[e]);
    double zsum = 0;
    for (int e = 0; e < NE; e++) zsum += std::exp((double)(lg[e] - mx));
    mx_ = mx;
    zsum_ = zsum;
    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                      [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
    shw_ = v.shared ? sigm(lg[NE]) : 0.f;
  } else {
    // sigmoid scores (the stable form), top-K of score + bias (ties: lower index), renormalised x route scale
    for (int e = 0; e < NE; e++) {
      float yv = 1.f / (1.f + std::exp(std::fabs(lg[e])));
      sc[e] = lg[e] < 0 ? yv : 1.f - yv;
      bz[e] = sc[e] + v.router_bias[e];
    }
    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                      [&](int a, int b) { return bz[a] > bz[b] || (bz[a] == bz[b] && a < b); });
    shw_ = 0.f;
  }
  for (int j = 0; j < K; j++) pk[j] = forced ? force[((size_t)pos * steps + step) * K + j] : ord[j];
  weights(v, pk, K, w);
  if (cache.wants_rel()) cache.released_row(step, v.rule == MoeView::SOFTMAX_TOPK_RENORM ? lg.data() : sc.data(),
                                            v.rule == MoeView::SOFTMAX_TOPK_RENORM ? nullptr : bz.data());
}

void Core::weights(const MoeView& v, const int* set, int n, float* w) const {
  if (v.rule == MoeView::SOFTMAX_TOPK_RENORM) {
    double s = 0;
    for (int i = 0; i < n; i++) {
      w[i] = (float)(std::exp((double)(lg[set[i]] - mx_)) / zsum_);
      s += w[i];
    }
    if (v.renorm)
      for (int i = 0; i < n; i++) w[i] = (float)(w[i] / s);
  } else {
    float ssum = 0;
    for (int i = 0; i < n; i++) ssum += sc[set[i]];
    for (int i = 0; i < n; i++) w[i] = sc[set[i]] / (ssum + 1e-20f) * v.route_scale;
  }
}

// ------------------------------------------------------------------------------------------------ expert compute
namespace {
struct ExpCtx {
  const Core* C;
  struct J {
    const Weight *gu, *up, *dn;
    int inter, so;
  } j[MAXK + 1];
  int n, IA, H;
  bool fused, ref, softcap_act;
  float gcap, ucap;
  const XQ* xq;
  const float* xf;
  float* actf;
  float* eout;
  XBuf* act;
  int start[MAXK + 2];
  std::atomic<int> left[MAXK + 1];  // gate-up chunks of each expert not yet done
  int dn_ch = 32;                    // down rows per chunk
  alignas(128) std::atomic<int> next_gu{0};
};

// gate-up chunk ch (64 activations of one expert) and down chunk (32 output rows of one expert), as separate phases did
inline void gu_chunk(ExpCtx& X, int ch, float* g, float* u) {
  int e = 0;
  while (ch >= X.start[e + 1]) e++;
  const auto& J = X.j[e];
  const int j0 = (ch - X.start[e]) * XB;
  float* a = X.actf + (size_t)J.so * X.IA;
  const bool fused = J.up == nullptr;
  if (fused) {
    mv(*J.gu, 2 * j0, 2 * j0 + 2 * XB, X.xq, X.xf, g, X.ref);  // row 2j = gate j, 2j + 1 = up j
  } else {
    mv(*J.gu, j0, j0 + XB, X.xq, X.xf, g, X.ref);
    mv(*J.up, j0, j0 + XB, X.xq, X.xf, u, X.ref);
  }
  if (fused && X.softcap_act) {  // the activation loops are kept apart so each one vectorises as the originals did
    for (int j = 0; j < XB; j++) {
      const float gg = g[2 * j], uu = g[2 * j + 1];
      a[j0 + j] = softcap(gg, X.gcap) * sigm(gg) * softcap(uu, X.ucap);
    }
  } else if (fused) {
    for (int j = 0; j < XB; j++) a[j0 + j] = silu(g[2 * j]) * g[2 * j + 1];
  } else if (X.softcap_act) {
    for (int j = 0; j < XB; j++) a[j0 + j] = softcap(g[j], X.gcap) * sigm(g[j]) * softcap(u[j], X.ucap);
  } else {
    for (int j = 0; j < XB; j++) a[j0 + j] = silu(g[j]) * u[j];
  }
  if (!X.ref) xq_block(X.act[J.so].q, j0 / XB, a + j0);
  X.left[e].fetch_sub(1, std::memory_order_release);
}

inline void dn_chunk(ExpCtx& X, int e, int a) {
  const int b = std::min(X.H, a + X.dn_ch);
  const auto& J = X.j[e];
  mv(*J.dn, a, b, &X.act[J.so].q, X.actf + (size_t)J.so * X.IA, X.eout + (size_t)J.so * X.H + a, X.ref);
}

// One fork-join for both projections: the gate-up chunks first (all claimed before any down chunk), then the down
// chunks, expert by expert; a down chunk starts once its expert's gate-up chunks are all done (a counter per expert,
// release / acquire). The barrier between the two phases is gone: a thread past the gate-up work starts on the first
// expert whose activations are complete while slower cores finish the rest.
void phase_gudn(void* p, int, int) {
  ExpCtx& X = *(ExpCtx*)p;
  const int tgu = X.start[X.n], nch = (X.H + X.dn_ch - 1) / X.dn_ch, total = tgu + nch * X.n;
  float g[2 * XB], u[XB];
  for (int ch; (ch = X.next_gu.fetch_add(1, std::memory_order_relaxed)) < total;) {
    if (ch < tgu) {
      gu_chunk(X, ch, g, u);
      continue;
    }
    const int d = ch - tgu, e = d / nch;
    while (X.left[e].load(std::memory_order_acquire) > 0) __builtin_arm_yield();
    dn_chunk(X, e, (d % nch) * X.dn_ch);
  }
}
}  // namespace

Core::ExpJob Core::shared_job(const MoeView& v) const {
  ExpJob j;
  j.gu = *v.sh_gate;
  j.up = *v.sh_up;
  j.dn = *v.sh_down;
  j.inter = sh_inter_;
  j.slot_out = K;
  return j;
}

void Core::run_experts(ExpJob* jobs, int n, const MoeView& v, const float* x) {
  ExpCtx X;
  X.C = this;
  X.n = n;
  X.IA = std::max(I_, sh_inter_);
  X.H = H;
  X.ref = o.ref;
  X.softcap_act = v.act == MoeView::SOFTCAP_SWIGLU;
  X.gcap = v.gate_cap;
  X.ucap = v.up_cap;
  X.xq = &xin.q;
  X.xf = x;
  X.actf = actf.data();
  X.eout = eout.data();
  X.act = act.data();
  X.start[0] = 0;
  for (int e = 0; e < n; e++) {
    X.j[e].gu = &jobs[e].gu;
    X.j[e].up = jobs[e].up.kind == Weight::NONE ? nullptr : &jobs[e].up;
    X.j[e].dn = &jobs[e].dn;
    X.j[e].inter = jobs[e].inter;
    X.j[e].so = jobs[e].slot_out;
    X.start[e + 1] = X.start[e] + jobs[e].inter / XB;
    X.left[e].store(jobs[e].inter / XB, std::memory_order_relaxed);
    // down rows per chunk: about 64 KB of weights (32 rows of a small down matrix were a few KB: claim overhead)
    const size_t rb = jobs[e].dn.bytes() / std::max(1, jobs[e].dn.rows);
    X.dn_ch = std::max(32, std::min(256, (int)((65536 / std::max<size_t>(1, rb)) / 32 * 32)));
    act[jobs[e].slot_out].reset(jobs[e].inter);
  }
  static const int dn_env = getenv("DMENG_DN_ROWS") ? std::max(1, atoi(getenv("DMENG_DN_ROWS"))) : 0;  // experiments
  if (dn_env) X.dn_ch = dn_env;
  pool->run(phase_gudn, &X);
}

void Core::combine(const MoeView& v, const float* w, const int* rows, int n, float shw, float* y) const {
  // fixed order: routed experts in pick (rank) order, then the shared expert. Every output element is the chain
  // fma(w[n-1], e[n-1], ... fma(w[0], e[0], 0)), then fma(shw, shared, .), so four elements at a time give the same bits
  const float* e[MAXK + 1];
  for (int i = 0; i < n; i++) e[i] = eout.data() + (size_t)rows[i] * H;
  const float* es = eout.data() + (size_t)K * H;
  int r = 0;
  for (; r + 16 <= H; r += 16) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i++) {
      const float* p = e[i] + r;
      a0 = vfmaq_n_f32(a0, vld1q_f32(p), w[i]);
      a1 = vfmaq_n_f32(a1, vld1q_f32(p + 4), w[i]);
      a2 = vfmaq_n_f32(a2, vld1q_f32(p + 8), w[i]);
      a3 = vfmaq_n_f32(a3, vld1q_f32(p + 12), w[i]);
    }
    if (v.shared) {
      a0 = vfmaq_n_f32(a0, vld1q_f32(es + r), shw);
      a1 = vfmaq_n_f32(a1, vld1q_f32(es + r + 4), shw);
      a2 = vfmaq_n_f32(a2, vld1q_f32(es + r + 8), shw);
      a3 = vfmaq_n_f32(a3, vld1q_f32(es + r + 12), shw);
    }
    vst1q_f32(y + r, a0);
    vst1q_f32(y + r + 4, a1);
    vst1q_f32(y + r + 8, a2);
    vst1q_f32(y + r + 12, a3);
  }
  for (; r < H; r++) {
    float acc = 0;
    for (int i = 0; i < n; i++) acc = __builtin_fmaf(w[i], e[i][r], acc);
    y[r] = v.shared ? __builtin_fmaf(shw, es[r], acc) : acc;
  }
}

int Core::take_slot(int L, int e, int step) {
  int x;
  if (cache.pred_evict) {
    x = cache.pick_victim(L, 0, nullptr);
    if (x >= 0) cache.evict(L, x);
  } else {
    x = cache.victim(L, 0);
  }
  if (x < 0) x = cache.steal(L);
  if (x < 0) {
    fprintf(stderr, "no expert slot free in layer %d's pool (budget too small for the pins)\n", L);
    abort();
  }
  if (cache.mem == Mem::PURGEABLE) cache.touch_nonvolatile(L, x);
  cache.issue(L, x, e, true, now(step));
  cache.slot(L, x).pins = 1;
  cache.st.n_miss++;
  miss_step[step]++;
  return x;
}

void Core::finish_step(int step, const int* slots, int n) {
  // decayed use counts; pins: released now, or (a layer that runs again in this token) held for its later passes
  const int L = mi.step_layer[step];
  const long st = now(step);
  const bool hold = o.loop_hold && next_same[step] >= 0;
  for (int i = 0; i < n; i++) {
    cache.use(L, slots[i], tok_clock, st);
    if (hold) holds[L].push_back(slots[i]);
    else cache.unpin(L, slots[i]);
  }
  if (next_same[step] < 0 && !holds[L].empty()) {
    for (int s : holds[L]) cache.unpin(L, s);
    holds[L].clear();
  }
  cache.step_end(step);
}

// ------------------------------------------------------------------------------------------------ the step
void Core::log_picks(int p, int step, const int32_t* pk) {
  if (p < 0) return;
  const size_t row = (size_t)steps * K, need = (size_t)(p + 1) * row;
  if (pick_log->size() < need) pick_log->resize(std::max(need, 2 * pick_log->size()), (int16_t)-1);
  int16_t* d = pick_log->data() + (size_t)p * row + (size_t)step * K;
  for (int j = 0; j < K; j++) d[j] = (int16_t)pk[j];
}

void Core::moe_step(int step, const float* x, float* y, const MoeView& v) {
  cache.now_step = now(step);
  step_t0 = clk::now();
  cache.tok_now = tok_clock;
  if (hook) {
    auto tp = clk::now();
    hook->before_moe(*this, step, x);
    t.pf += ms_since(tp);
  }
  auto t0 = clk::now();
  static thread_local std::vector<Job> jobs;
  jobs.clear();
  jobs.push_back({v.router, &xin.q, x, lg.data()});
  if (v.shared) jobs.push_back({v.sh_scalar_gate, &xin.q, x, lg.data() + NE});
  if (hook) hook->router_rows(*this, step, x, jobs);
  {  // the MoE input's quantisation (the experts' input) inside the router's fork-join
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {x};
    const int ks[1] = {H};
    run_rows_q(*pool, jobs.data(), (int)jobs.size(), 16, o.ref, &t.bytes_res, qs, xs, ks, 1);
  }
  route(v, step);
  if (pick_log) log_picks(pos, step, picks.data() + (size_t)step * K);
  t.router += ms_since(t0);
  if (cancel_stale) cache.cancel_stale(now(step) - 1);
  if (ns.on && !(force && pos < force_n)) approx(step, x, y, v);
  else lossless(step, x, y, v);
}

void Core::lossless(int step, const float* x, float* y, const MoeView& v) {
  const int L = mi.step_layer[step], pass = mi.step_pass[step];
  const ExpertFormat& F = mi.formats[mi.layer_fmt[L]];
  const long st = now(step);
  const int reg = looped(step) ? 0 : 1;
  const int32_t* pk = picks.data() + (size_t)step * K;
  const float* w = pw.data() + (size_t)step * K;
  auto t1 = clk::now();
  if (cache.learn_on) cache.learn_use(L, pk, tok_clock);
  if (cache.hazard_evict) cache.pre_use(L, pk, K, tok_clock);  // the step's picks count as used before its misses
  // the slots: hits pinned, misses read now (urgent) while the hits are computed
  int sl[MAXK];
  bool missed[MAXK] = {false};
  auto& ut = used_tok[L];
  auto& S = cache.st;
  for (int j = 0; j < K; j++) {
    const int e = pk[j];
    int s = cache.slot_of(L, e);
    S.n_pick_r[reg]++;
    if (pass > 0 && std::find(ut.begin(), ut.end(), e) != ut.end()) S.n_reuse_hit++;
    if (s >= 0 && cache.mem == Mem::PURGEABLE) {
      cache.touch_nonvolatile(L, s);
      s = cache.slot_of(L, e);
    }
    if (s >= 0) {
      if (cache.mem == Mem::FILE && cache.ready(L, s)) cache.file_hit_check(L, s);  // pages dropped: read again (late)
      if (cache.ready(L, s)) {
        S.n_hit++;
        S.n_hit_r[reg]++;
      } else {
        S.n_late++;
      }
      cache.slot(L, s).pins++;
    } else {
      const size_t ke = (size_t)L * NE + e;
      if (cache.ev_step[ke] >= 0 && cache.ev_step[ke] > st - steps) {
        S.n_miss_evicted++;
        if (cache.ev_pf[ke]) S.n_miss_pf_evicted++;
      }
      const float pr = cache.pred[ke];
      S.miss_pbin[pr < 0 ? 0 : pr < 0.1f ? 1 : pr < 0.3f ? 2 : pr < 0.5f ? 3 : pr < 0.7f ? 4 : 5]++;
      s = take_slot(L, e, step);
      missed[j] = true;
    }
    sl[j] = s;
  }
  if (looped(step)) {
    if (pass == 0) ut.clear();
    for (int j = 0; j < K; j++) ut.push_back(pk[j]);
  }
  auto t2a = clk::now();
  t.cache += ms_between(t1, t2a);
  if (hook) hook->after_router(*this, step);  // names for a later step: queued behind this step's own misses
  auto t2 = clk::now();
  t.pf += ms_between(t2a, t2);

  // pass A: the shared expert + every routed expert already in memory; pass B: the rest, each once landed
  bool done[MAXK] = {false};
  const double wait0 = S.wait_ms;
  for (int ps = 0; ps < 2; ps++) {
    ExpJob jb[MAXK + 1];
    int n = 0;
    if (ps == 0 && v.shared) jb[n++] = shared_job(v);
    for (int j = 0; j < K; j++) {
      if (done[j]) continue;
      if (ps == 0 && !cache.ready(L, sl[j])) continue;
      if (ps == 1) {
        const double w0 = S.wait_ms;
        cache.wait(L, sl[j]);
        (missed[j] ? S.wait_ms_miss : S.wait_ms_late) += S.wait_ms - w0;
      }
      F.mats(cache.rec(L, sl[j]), jb[n].gu, jb[n].up, jb[n].dn);
      jb[n].inter = F.I;
      jb[n].slot_out = j;
      n++;
      done[j] = true;
    }
    if (n) run_experts(jb, n, v, x);
  }
  t.bytes_exp += (double)K * F.touched;
  int rows[MAXK];
  for (int j = 0; j < K; j++) rows[j] = j;
  combine(v, w, rows, K, shw_, y);
  std::copy(pk, pk + K, used.data() + (size_t)step * K);
  std::copy(w, w + K, uw.data() + (size_t)step * K);
  auto t3 = clk::now();
  const double wt = S.wait_ms - wait0;
  t.wait += wt;
  wait_step[step] += wt;
  t.experts += ms_between(t2, t3) - wt;
  finish_step(step, sl, K);
  if (hook) {
    auto tp = clk::now();
    hook->after_moe(*this, step, pk, w);
    hook->release(*this, step);
    t.pf += ms_since(tp);
  }
  t.cache += ms_since(t3);
}

// The approximate MoE. The released router has decided (picks / pw hold its top K and their weights). Each released
// pick is landed, late (named, still in flight) or missing. Guaranteed: missing picks with released weight > read_w,
// then the largest missing until the missing weight <= read_mass, every late pick when late_ms >= 1000, and every
// missing pick when fewer than 2K other experts of the layer have landed (a cold cache). The rest are replaced by the
// released ranking's next experts that have landed; the final set is weighted as the release weights its top K and
// combined in rank order.
void Core::approx(int step, const float* x, float* y, const MoeView& v) {
  const int L = mi.step_layer[step], pass = mi.step_pass[step];
  const ExpertFormat& F = mi.formats[mi.layer_fmt[L]];
  const long st = now(step);
  const int reg = looped(step) ? 0 : 1;
  const auto t_dec = clk::now();
  const int32_t* pk = picks.data() + (size_t)step * K;
  const float* pwl = pw.data() + (size_t)step * K;
  const float* key = v.rule == MoeView::SOFTMAX_TOPK_RENORM ? lg.data() : bz.data();
  const float wscale = v.rule == MoeView::SOFTMAX_TOPK_RENORM ? 1.f : v.route_scale;
  auto& S = cache.st;
  if (cache.learn_on) cache.learn_use(L, pk, tok_clock);
  if (cache.hazard_evict) cache.pre_use(L, pk, K, tok_clock);
  // the released ranking beyond its top K (where replacements come from); the first K equal pk
  for (int e = 0; e < NE; e++) ord[e] = e;
  auto by_key = [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a < b); };
  int nord = std::min(NE, 64);
  std::partial_sort(ord.begin(), ord.begin() + nord, ord.end(), by_key);
  auto ensure = [&](int need) {
    if (need > nord && nord < NE) {
      std::sort(ord.begin() + nord, ord.end(), by_key);
      nord = NE;
    }
  };
  struct Ch {
    int e, rank, slot, row;
    int kind;  // 0 landed, 1 in flight and waited for, 2 in flight and tentative (late rule), 3 read now
    bool done;
  };
  Ch ch[2 * MAXK];
  int stt[MAXK];
  bool must[MAXK] = {false}, dropped[MAXK] = {false};
  double frac[MAXK];
  for (int j = 0; j < K; j++) {
    const int e = pk[j];
    frac[j] = pwl[j] / wscale;
    int s = cache.slot_of(L, e);
    S.n_pick_r[reg]++;
    if (s >= 0 && cache.mem == Mem::PURGEABLE) {
      cache.touch_nonvolatile(L, s);
      s = cache.slot_of(L, e);
    }
    if (s >= 0) {
      stt[j] = cache.ready(L, s) ? 0 : 1;
      if (stt[j]) S.n_late++;
      else {
        S.n_hit++;
        S.n_hit_r[reg]++;
      }
      cache.slot(L, s).pins++;  // every pick held is pinned before any miss takes a slot
    } else {
      stt[j] = 2;
      S.n_miss++;
      const float pr = cache.pred[(size_t)L * NE + e];
      S.miss_pbin[pr < 0 ? 0 : pr < 0.1f ? 1 : pr < 0.3f ? 2 : pr < 0.5f ? 3 : pr < 0.7f ? 4 : 5]++;
    }
    ch[j] = {e, j, s, j, 0, false};
  }
  // the guarantees
  double miss_mass = 0;
  for (int j = 0; j < K; j++)
    if (stt[j]) {
      if (frac[j] > ns.read_w) must[j] = true;
      else miss_mass += frac[j];
    }
  while (miss_mass > ns.read_mass) {
    int b = -1;
    for (int j = 0; j < K; j++)
      if (stt[j] && !must[j] && (b < 0 || frac[j] > frac[b])) b = j;
    if (b < 0) break;
    must[b] = true;
    miss_mass -= frac[b];
  }
  if (!ns.drop) {  // too few other experts of this layer in memory to replace from (a cold cache): this step runs lossless
                   // (the drop rule replaces nothing, so it needs no floor)
    int nd = 0;
    for (int j = 0; j < K; j++) nd += stt[j] != 0 && !must[j] && !(stt[j] == 1 && ns.late_ms >= 1000);
    if (nd > 0) {
      int n = 0;
      for (int e = 0; e < NE; e++) {
        const int s = cache.slot_of(L, e);
        if (s < 0 || !cache.ready(L, s) || std::find(pk, pk + K, e) != pk + K) continue;
        n++;
      }
      if (n < std::min(2 * K, NE - K)) {
        for (int j = 0; j < K; j++)
          if (stt[j]) must[j] = true;
        ns.cold++;
      }
    }
  }
  // decide what is replaced now; read the guaranteed misses
  int bg[MAXK], nbg = 0, need = 0;
  for (int j = 0; j < K; j++) {
    if (stt[j] == 0) {
      ch[j].kind = 0;
    } else if (stt[j] == 1) {
      if (must[j] || ns.late_ms >= 1000) {
        ch[j].kind = 1;
      } else if (ns.late_ms < 0) {
        dropped[j] = true;
        cache.unpin(L, ch[j].slot);
        ns.n_late_dropped++;
        need++;
      } else {
        ch[j].kind = 2;
      }
    } else if (must[j]) {
      ch[j].slot = take_slot(L, pk[j], step);
      ch[j].kind = 3;
      ns.n_read_now++;
      ns.w_read_now += frac[j];
    } else {
      dropped[j] = true;
      need++;
      if (ns.bg) bg[nbg++] = pk[j];
    }
  }
  int rpos = K, nch = K;
  int freerow[MAXK], nfree = 0;
  for (int j = 0; j < K; j++)
    if (dropped[j]) freerow[nfree++] = j;
  auto replace = [&](int n) {  // n more experts from the ranking, landed now, not chosen yet
    while (n > 0) {
      ensure(rpos + 1);
      if (rpos >= NE) {
        fprintf(stderr, "approximate mode: fewer than %d experts of layer %d in memory\n", K, L);
        abort();
      }
      const int e = ord[rpos++];
      const int s = cache.slot_of(L, e);
      if (s < 0 || !cache.ready(L, s)) continue;
      cache.slot(L, s).pins++;
      ch[nch++] = {e, rpos - 1, s, freerow[--nfree], 0, false};
      n--;
    }
  };
  if (!ns.drop) replace(need);
  auto t1 = clk::now();
  t.cache += ms_between(t_dec, t1);
  if (hook) hook->after_router(*this, step);
  for (int i = 0; i < nbg; i++) {  // behind the names for later steps: these are for this layer's next use
    const int e = bg[i];
    if (cache.slot_of(L, e) >= 0) continue;
    cache.evict_pf_ctx = true;
    const int s = cache.victim(L, 2 * K + cache.extra_reserve[cache.layer_pool[L]]);
    cache.evict_pf_ctx = false;
    if (s < 0) continue;
    if (cache.mem == Mem::PURGEABLE) cache.touch_nonvolatile(L, s);
    cache.issue(L, s, e, false, next_use(step));
    cache.use(L, s, tok_clock, st);  // as the lossless engine would have used it
    ns.n_bg++;
  }
  auto t2 = clk::now();
  t.pf += ms_between(t1, t2);

  // pass A: shared expert + everything landed; then the waits; pass B: the rest
  const double wait0 = S.wait_ms;
  auto run_pass = [&](bool first) {
    ExpJob jb[MAXK + 1];
    int n = 0;
    if (first && v.shared) jb[n++] = shared_job(v);
    for (int i = 0; i < nch; i++) {
      Ch& q = ch[i];
      if (q.done || (q.rank < K && dropped[q.rank])) continue;
      if (!cache.ready(L, q.slot)) continue;
      F.mats(cache.rec(L, q.slot), jb[n].gu, jb[n].up, jb[n].dn);
      jb[n].inter = F.I;
      jb[n].slot_out = q.row;
      n++;
      q.done = true;
    }
    if (n) run_experts(jb, n, v, x);
  };
  run_pass(true);
  int need2 = 0;
  const auto deadline = t_dec + std::chrono::microseconds((long)(std::max(0.0, ns.late_ms) * 1000.0));
  for (int j = 0; j < K; j++) {
    Ch& q = ch[j];
    if (dropped[j] || q.done) continue;
    if (q.kind == 1 || q.kind == 3) {
      if (q.kind == 1 && !cache.ready(L, q.slot)) ns.n_late_waited++;
      const double w0 = S.wait_ms;
      cache.wait(L, q.slot);
      (q.kind == 3 ? S.wait_ms_miss : S.wait_ms_late) += S.wait_ms - w0;
    } else if (q.kind == 2) {
      if (!cache.ready(L, q.slot) && ns.late_ms > 0 && clk::now() < deadline) {
        ns.n_late_waited++;
        if (cache.rd.promote(cache.layer_pool[L], q.slot)) S.n_late_queued++;
        auto tw = clk::now();
        while (!cache.ready(L, q.slot) && clk::now() < deadline) __builtin_arm_yield();
        const double dw = ms_since(tw);
        S.wait_ms += dw;
        S.wait_ms_late += dw;
        ns.late_wait_ms += dw;
      }
      if (cache.ready(L, q.slot)) {
        ns.n_late_got++;
      } else {
        dropped[j] = true;
        cache.unpin(L, q.slot);
        ns.n_late_dropped++;
        freerow[nfree++] = q.row;
        need2++;
      }
    }
  }
  if (!ns.drop) replace(need2);
  run_pass(false);
  // the final set in the released ranking's order, the release's weighting over it, fixed-order combination
  int fin[2 * MAXK], nf = 0;
  for (int i = 0; i < nch; i++)
    if (!(ch[i].rank < K && dropped[ch[i].rank])) fin[nf++] = i;
  std::sort(fin, fin + nf, [&](int a, int b) { return ch[a].rank < ch[b].rank; });
  int set[2 * MAXK], rows[2 * MAXK], slots[2 * MAXK];
  float wf[2 * MAXK];
  for (int i = 0; i < nf; i++) {
    set[i] = ch[fin[i]].e;
    rows[i] = ch[fin[i]].row;
    slots[i] = ch[fin[i]].slot;
  }
  weights(v, set, nf, wf);
  if (ns.drop)  // the kept released picks keep their released weights (equal to weights() when nothing was left out)
    for (int i = 0; i < nf; i++) wf[i] = pwl[ch[fin[i]].rank];
  t.bytes_exp += (double)nf * F.touched;
  combine(v, wf, rows, nf, shw_, y);
  for (int i = 0; i < nf && i < K; i++) {
    used[(size_t)step * K + i] = set[i];
    uw[(size_t)step * K + i] = wf[i];
  }
  auto t3 = clk::now();
  const double wt = S.wait_ms - wait0;
  t.wait += wt;
  wait_step[step] += wt;
  t.experts += ms_between(t2, t3) - wt;
  auto& ut = used_tok[L];
  if (looped(step)) {
    if (pass == 0) ut.clear();
    for (int i = 0; i < nf; i++) ut.push_back(set[i]);
  }
  finish_step(step, slots, nf);
  ns.steps++;
  int nrep = 0;
  for (int j = 0; j < K; j++)
    if (dropped[j]) {
      nrep++;
      ns.w_replaced += frac[j];
      ns.w_replaced_max = std::max(ns.w_replaced_max, frac[j]);
      const double p = frac[j];
      ns.w_hist[p < 0.02 ? 0 : p < 0.05 ? 1 : p < 0.1 ? 2 : p < 0.2 ? 3 : p < 0.4 ? 4 : 5]++;
    }
  ns.n_replaced += nrep;
  ns.steps_replaced += nrep > 0;
  for (int j = 0; j < K; j++)
    if (dropped[j]) ns.w_tok_cur += frac[j];
  if (step == steps - 1) {  // the token's last step: one per-token sample
    ns.w_tok.push_back((float)ns.w_tok_cur);
    ns.w_tok_cur = 0;
  }
  if (hook) {
    auto tp = clk::now();
    hook->after_moe(*this, step, pk, pwl);
    hook->release(*this, step);
    t.pf += ms_since(tp);
  }
  t.cache += ms_since(t3);
}

}  // namespace dm
