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
#include "dmeng/expert_step.h"

#include <sys/sysctl.h>

namespace dm {

std::string Times::stages_json(double n) const {
  n = n > 0 ? n : 1;
  char b[640];
  snprintf(b, sizeof b,
           "\"embed\":%.4f,\"depth_attn\":%.4f,\"message_gates_norms\":%.4f,\"attn_proj\":%.4f,\"attn_core\":%.4f,\"attn_out\":%.4f,"
           "\"moe_router\":%.4f,\"experts\":%.4f,\"run_ahead\":%.4f,\"wait\":%.4f,\"cache_mgmt\":%.4f,\"prefetch_hooks\":%.4f,\"post_mlp\":%.4f,"
           "\"head\":%.4f",
           embed / n, depth / n, msg / n, attn_proj / n, attn_core / n, attn_out / n, router / n, experts / n, ra / n, wait / n, cache / n, pf / n,
           misc / n, head / n);
  return b;
}

constexpr int MAXK = 16;

// ------------------------------------------------------------------------------------------------ approximate presets
void Approx::reset_stats() {
  steps = w_replaced = n_replaced = n_read_now = w_read_now = n_late_waited = n_late_got = n_late_dropped = n_bg = 0;
  steps_replaced = w_replaced_max = late_wait_ms = cold = 0;
  for (auto& h : w_hist) h = 0;
  w_tok_cur = 0;
  w_tok.clear();
}

bool Approx::preset(const std::string& spec) {
  // PRESET[:k=v...]: a preset, then its rule set by key (a model's manifest tunes fast mode this way): w = read_w, mass =
  // read_mass, late = late_ms (< 0: never wait for a light pick in flight), bg = background fetch of left-out picks
  const size_t c = spec.find(':');
  const std::string name = spec.substr(0, c);
  if (!preset_base(name)) return false;
  cold_floor = -1;
  for (size_t i = c == std::string::npos ? spec.size() : c + 1; i < spec.size();) {
    size_t j = spec.find(':', i);
    if (j == std::string::npos) j = spec.size();
    const std::string kv = spec.substr(i, j - i);
    const size_t q = kv.find('=');
    if (q == std::string::npos) return false;
    const std::string k = kv.substr(0, q);
    char* end = nullptr;
    const double v = std::strtod(kv.c_str() + q + 1, &end);
    if (end == kv.c_str() + q + 1 || *end) return false;
    if (k == "w") read_w = v;
    else if (k == "mass") read_mass = v;
    else if (k == "late") late_ms = v;
    else if (k == "bg") bg = v != 0;
    else if (k == "floor") cold_floor = (int)v;  // the cold-cache floor (-1: 2K; 0: never: substitute from whatever is held)
    else return false;
    i = j + 1;
  }
  return true;
}

bool Approx::preset_base(const std::string& name) {
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
         "never (replace whatever has not landed: a speed bound); each may be followed by :w=W:mass=M:late=MS:bg=0|1:floor=N "
         "(the rule set; floor: the cold-cache floor of the replacing rules, 0 = never read on the spot for it)";
}

// ------------------------------------------------------------------------------------------------ setup
// options of the engine itself, the same for every family (a model's set_option passes on the names it does not know)
bool Core::set_option(const std::string& name, const std::string& value, std::string& err) {
  if (name == "chunk_aff4") return (mi.chunk_aff4 = std::stoi(value) != 0), true;  // moe_chunk's packed affine kernels
  err = "unknown option " + name + " for " + mi.family;
  return false;
}

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
  cache.rd.io_low = o.reader_io_low;
  cache.rd.drive_bps = o.drive_gbs > 0 ? o.drive_gbs * 1e9 : 0;
  std::vector<size_t> rec;
  for (auto& f : mi.formats) rec.push_back(f.rec);
  std::vector<double> share = o.pool_share.empty() ? mi.pool_share : o.pool_share;
  if (share.empty()) share.assign(rec.size(), 1.0 / rec.size());
  cache.layer_paths = mi.layer_paths;
  cache.file_under = o.spare_mem && o.mem != Mem::FILE;
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
  // CAPS[:fit]: the caps, and with :fit the reserve a name leaves in a capped layer (Cache::cap_fit)
  const size_t colon = o.layer_caps.find(':');
  const std::string caps = o.layer_caps.substr(0, colon), rsv = colon == std::string::npos ? "" : o.layer_caps.substr(colon + 1);
  if (rsv != "" && rsv != "fit") {
    err = "layer caps: auto|uniform|none, optionally :fit, got " + o.layer_caps;
    return false;
  }
  cache.cap_fit = rsv == "fit";
  if (caps == "uniform") {
    cache.uniform_layer_caps();
  } else if (caps == "auto") {
    cache.caps_auto = cache.hazard_evict;
    if (cache.caps_auto) cache.caps_auto_apply();
  } else if (caps != "none") {
    err = "layer caps: auto|uniform|none, optionally :fit, got " + o.layer_caps;
    return false;
  }
  next_same.assign(steps, -1);
  for (int s = 0; s < steps; s++)
    for (int s2 = s + 1; s2 < steps; s2++)
      if (mi.step_layer[s2] == mi.step_layer[s]) {
        next_same[s] = s2;
        break;
      }
  cache.layer_hold.assign(mi.layers, 0);
  if (o.loop_hold)  // the loop hold's worst case: K slots per step that holds, in that step's pool
    for (int s = 0; s < steps; s++)
      if (next_same[s] >= 0) {
        cache.extra_reserve[mi.layer_fmt[mi.step_layer[s]]] += K;
        cache.layer_hold[mi.step_layer[s]] += K;  // the same worst case, per layer: K per holding step of the layer
      }
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
  wait_unnamed_step.assign(steps, 0.0);
  nwait_unnamed_step.assign(steps, 0);
  holds.assign(mi.layers, {});
  used_tok.assign(mi.layers, {});
  return true;
}

void Core::reset_stats() {
  cache.st = CacheStats();
  ns.reset_stats();
  std::fill(wait_step.begin(), wait_step.end(), 0.0);
  std::fill(miss_step.begin(), miss_step.end(), 0);
  std::fill(wait_unnamed_step.begin(), wait_unnamed_step.end(), 0.0);
  std::fill(nwait_unnamed_step.begin(), nwait_unnamed_step.end(), 0);
  // the run-ahead's counters and where the wait sits: from here, as every decode statistic (they used to count the
  // prompt's single steps too); its pending names (ra.step, pslot) are state, not statistics, and stay
  ra.events = ra.ms = ra.would_read = ra.found = ra.lead_ms = ra.issued = ra.wasted = ra.top8 = ra.top8_n = 0;
  ra.aborted = ra.over_ms = ra.refused = ra.withheld = ra.skip_budget = ra.skip_sat = 0;
  for (int d = 0; d < 4; d++) deep_top8[d] = deep_n[d] = deep_would[d] = deep_found[d] = deep_events[d] = deep_issued[d] = deep_wasted[d] = 0;
  wait_cov_ms = wait_uncov_ms = steps_wait_cov = steps_wait_uncov = steps_no_wait = steps_total = 0;
  t.zero();
  exp_spin_ns.store(0);
  for (int a = 0; a < 2; a++)
    for (int i = 0; i < 18; i++) xc_calls[a][i] = xc_ns[a][i] = xc_bytes[a][i] = 0;
  for (int a = 0; a < 2; a++) rr_ns[a] = rr_bytes[a] = rr_calls[a] = 0;
  rr_count.store(0);
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
namespace {
// DMENG_AB (weights.h): per arm, decode tokens, their ms and each stage's ms (Times' fields), printed at exit
constexpr int NT = sizeof(Times) / sizeof(double);
struct AbAcc {
  double n[2] = {0, 0}, ms[2] = {0, 0}, st[2][NT] = {}, mb[2] = {0, 0};
  uint64_t b0 = 0;
  clk::time_point t0;
  double s0[NT];
  bool in = false;
  ~AbAcc() {
    if (!g_ab) return;
    static const char* nm[] = {"embed", "attn_proj", "attn_core", "attn_out", "depth", "msg", "router", "experts", "wait", "cache", "pf", "head", "misc", "ra"};
    for (int a = 0; a < 2; a++) {
      const double k = std::max(1.0, n[a]);
      fprintf(stderr, "DMAB %d arm %d tokens %.0f ms %.3f mb_read %.1f |", g_ab, a, n[a], ms[a] / k, mb[a] / k);
      for (int i = 0; i < 14 && i < NT; i++) fprintf(stderr, " %s %.3f", nm[i], st[a][i] / k);
      fprintf(stderr, "\n");
    }
  }
} g_abacc;
}  // namespace

void Core::token_start(int position, int tok, const float* router_emb) {
  if (g_ab) {
    g_ab_arm = (int)(tok_clock & 1);
    g_abacc.t0 = clk::now();
    std::memcpy(g_abacc.s0, &t, sizeof g_abacc.s0);
    g_abacc.b0 = cache.st.bytes_urgent + cache.st.bytes_pf;
    g_abacc.in = true;
  }
  pos = position;
  if (cache.budget_req.load(std::memory_order_relaxed) >= 0) cache.apply_budget();  // a run-time budget (any thread)
  if (hook) {
    auto tp = clk::now();
    stage_mark(ST_PREFETCH);
    hook->token_start(*this, tok, router_emb);
    t.pf += ms_since(tp);
    stage_mark(ST_OTHER);
  }
}

void Core::token_end() {
  if (g_ab && g_abacc.in) {
    const int a = g_ab_arm;
    g_abacc.n[a] += 1;
    g_abacc.ms[a] += ms_since(g_abacc.t0);
    g_abacc.mb[a] += (cache.st.bytes_urgent + cache.st.bytes_pf - g_abacc.b0) / 1e6;
    const double* now_t = (const double*)&t;
    for (int i = 0; i < NT; i++) g_abacc.st[a][i] += now_t[i] - g_abacc.s0[i];
    g_abacc.in = false;
  }
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
static int g_ecores = -1;
int cpu_now() {
  uint64_t v;
  asm volatile("mrs %0, TPIDR_EL0" : "=r"(v));
  return (int)(v & 0xFF);
}
bool on_ecore() {
  if (g_ecores < 0) {
    int n = 0;
    size_t sz = sizeof n;
    g_ecores = sysctlbyname("hw.perflevel1.logicalcpu", &n, &sz, nullptr, 0) == 0 ? n : 0;
  }
  return cpu_now() < g_ecores;
}

namespace {
struct ExpCtx {
  const ExpertStep* S;
  struct J {
    const Weight *gu, *up, *dn;
    int inter, so;
  } j[MAXK + 1];
  int n, IA, H;
  bool ref, softcap_act;
  float gcap, ucap;
  unsigned arms;
  const XQ* xq;
  const float* xf;
  float* actf;
  float* eout;
  XBuf* act;
  int start[MAXK + 2];
  std::atomic<int> left[MAXK + 1];  // gate-up chunks of each expert not yet done
  int dn_ch = 32;                    // down rows per chunk
  alignas(128) std::atomic<int> next_gu{0};
  alignas(128) std::atomic<int> dn_rem{0};  // down quarters not yet claimed (all experts)
  struct alignas(128) Next {                 // per expert, a cache line of its own: its next unclaimed down quarter
    std::atomic<int> v{0};
  } dn_next[MAXK + 1];
  // STEP_RERUN: per gate-up chunk its state (0 free, 1 claimed, 3 claimed and being re-run, 2 done) and claim time
  static constexpr int MAXCH = (MAXK + 1) * 40;
  std::atomic<uint8_t> cst[MAXCH];
  std::atomic<int64_t> ct[MAXCH];
};
std::atomic<int64_t> g_gu_ns{0};  // a running estimate of a gate-up chunk's time (STEP_RERUN's trigger: 4x it)
inline int64_t now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count(); }

// STEP_RERUN's gate-up chunk: the rows and the activation into the thread's own scratch (av), then published once: the
// thread that moves the chunk's state to done copies av into the activations, quantises it and counts the chunk; any
// other thread that computed it discards its copy. The same function of the same inputs: the bits do not depend on
// which thread publishes. Returns the chunk's expert.
inline int gu_chunk_pub(ExpCtx& X, int ch, float* g, float* u, float* av) {
  int e = 0;
  while (ch >= X.start[e + 1]) e++;
  const auto& J = X.j[e];
  const int j0 = (ch - X.start[e]) * XB;
  const bool fused = J.up == nullptr;
  if (fused) {
    mv(*J.gu, 2 * j0, 2 * j0 + 2 * XB, X.xq, X.xf, g, X.ref);
  } else {
    mv(*J.gu, j0, j0 + XB, X.xq, X.xf, g, X.ref);
    mv(*J.up, j0, j0 + XB, X.xq, X.xf, u, X.ref);
  }
  if (fused && X.softcap_act) {
    for (int j = 0; j < XB; j++) {
      const float gg = g[2 * j], uu = g[2 * j + 1];
      av[j] = softcap(gg, X.gcap) * sigm(gg) * softcap(uu, X.ucap);
    }
  } else if (fused) {
    for (int j = 0; j < XB; j++) av[j] = silu(g[2 * j]) * g[2 * j + 1];
  } else if (X.softcap_act) {
    for (int j = 0; j < XB; j++) av[j] = softcap(g[j], X.gcap) * sigm(g[j]) * softcap(u[j], X.ucap);
  } else {
    for (int j = 0; j < XB; j++) av[j] = silu(g[j]) * u[j];
  }
  uint8_t v = X.cst[ch].load(std::memory_order_acquire);
  while (v != 2)
    if (X.cst[ch].compare_exchange_weak(v, 2, std::memory_order_acq_rel)) {
      std::memcpy(X.actf + (size_t)J.so * X.IA + j0, av, XB * sizeof(float));
      if (!X.ref) xq_block(X.act[J.so].q, j0 / XB, av);
      X.left[e].fetch_sub(1, std::memory_order_release);
      break;
    }
  return e;
}

// gate-up chunk ch (64 activations of one expert), as a separate phase did
inline int gu_chunk(ExpCtx& X, int ch, float* g, float* u) {  // returns the chunk's expert
  int e = 0;
  while (ch >= X.start[e + 1]) e++;
  const auto& J = X.j[e];
  const int j0 = (ch - X.start[e]) * XB;
  float* a = X.actf + (size_t)J.so * X.IA;
  const bool fused = J.up == nullptr;
  if (fused) {
    mv(*J.gu, 2 * j0, 2 * j0 + 2 * XB, X.xq, X.xf, g, X.ref);  // row 2j = gate j, row 2j + 1 = up j
  } else {
    mv(*J.gu, j0, j0 + XB, X.xq, X.xf, g, X.ref);
    mv(*J.up, j0, j0 + XB, X.xq, X.xf, u, X.ref);
  }
  if (!(X.arms & STEP_NOACT)) {
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
  }
  X.left[e].fetch_sub(1, std::memory_order_release);
  return e;
}

// One fork-join for both projections: the gate-up chunks first (all claimed before any down chunk), then the down
// chunks, expert by expert; a down chunk starts once its expert's gate-up chunks are all done (a counter per expert,
// release / acquire). The barrier between the two phases is gone: a thread past the gate-up work starts on the first
// expert whose activations are complete while slower cores finish the rest.
// The down rows of each expert are cut into quarter chunks (q = dn_ch / 4 rows, its last one shorter), claimed per expert
// (each expert's counter on its own cache line). A thread past the gate-up chunks takes down rows of the expert of its last
// gate-up chunk while that one is ready and has rows left (its activations are in this core's cache), else of the next
// ready expert after it, round the experts; it waits only when no expert with down rows left is ready. 4 quarters per
// claim while more than 4 per thread remain in the step, then single quarters. Every row is still computed once by one
// thread into its expert's own output row: the bits do not depend on who claims what.
void phase_gudn(void* p, int tid, int nt) {
  ExpCtx& X = *(ExpCtx*)p;
  const unsigned arms = X.arms;
  if ((arms & STEP_PONLY) && on_ecore()) return;
  const int tgu = X.start[X.n], q = std::max(1, X.dn_ch / 4), uq = (X.H + q - 1) / q;
  const int tail = (arms & STEP_SMALLTAIL) ? 8 * nt : 4 * nt;
  float g[2 * XB], u[XB], av[XB];
  int own = -1;
  const bool ecore = (arms & (STEP_ENOGU | STEP_ENODN)) && on_ecore();
  const bool rerun = arms & STEP_RERUN;
  if (!(ecore && (arms & STEP_ENOGU)))
    for (int ch; (ch = X.next_gu.fetch_add(1, std::memory_order_relaxed)) < tgu;) {
      if (!rerun) {
        own = gu_chunk(X, ch, g, u);
        continue;
      }
      const int64_t t0 = now_ns();
      X.ct[ch].store(t0, std::memory_order_relaxed);
      X.cst[ch].store(1, std::memory_order_release);
      own = gu_chunk_pub(X, ch, g, u, av);
      const int64_t d = now_ns() - t0, m = g_gu_ns.load(std::memory_order_relaxed);
      g_gu_ns.store(m ? m + (d - m) / 16 : d, std::memory_order_relaxed);
    }
  if (ecore && (arms & STEP_ENODN)) return;
  if (own < 0) own = tid % X.n;
  int64_t spin = 0;
  clk::time_point ts;
  bool spinning = false;
  for (;;) {
    bool any = false, did = false;
    if ((arms & STEP_PTAIL) && X.dn_rem.load(std::memory_order_relaxed) <= 2 * nt && on_ecore()) break;
    for (int k = 0; k < X.n && !did; k++) {
      const int e = (own + k) % X.n;
      if (X.dn_next[e].v.load(std::memory_order_relaxed) >= uq) continue;  // this expert's down rows are all claimed
      any = true;
      if (!(arms & STEP_NODEP) && X.left[e].load(std::memory_order_acquire) > 0) continue;  // its gate-up not done yet
      const int sz = X.dn_rem.load(std::memory_order_relaxed) > tail ? 4 : 1;
      const int u0 = X.dn_next[e].v.fetch_add(sz, std::memory_order_relaxed);
      if (u0 >= uq) continue;
      const int u1 = std::min(uq, u0 + sz);
      X.dn_rem.fetch_sub(u1 - u0, std::memory_order_relaxed);
      if (spinning) {
        spin += std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - ts).count();
        spinning = false;
      }
      const int a = u0 * q, b = std::min(X.H, u1 * q);
      mv(*X.j[e].dn, a, b, &X.act[X.j[e].so].q, X.actf + (size_t)X.j[e].so * X.IA, X.eout + (size_t)X.j[e].so * X.H + a, X.ref);
      own = e;  // stay on it while it has rows
      did = true;
    }
    if (!any) break;
    if (!did && rerun) {  // a chunk claimed more than 4x a chunk's time ago and not done: compute it too
      static const int64_t lim_env = getenv("DMENG_RERUN_LIM_NS") ? atol(getenv("DMENG_RERUN_LIM_NS")) : -1;  // tests
      const int64_t now = now_ns(), lim = lim_env >= 0 ? lim_env : 4 * std::max<int64_t>(g_gu_ns.load(std::memory_order_relaxed), 1000);
      for (int ch = 0; ch < tgu && !did; ch++) {
        uint8_t v = 1;
        if (X.cst[ch].load(std::memory_order_relaxed) != 1 || now - X.ct[ch].load(std::memory_order_relaxed) < lim) continue;
        if (!X.cst[ch].compare_exchange_strong(v, 3, std::memory_order_acq_rel)) continue;  // one re-runner per chunk
        if (spinning) {
          spin += std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - ts).count();
          spinning = false;
        }
        gu_chunk_pub(X, ch, g, u, av);
        if (X.S->reruns) X.S->reruns->fetch_add(1, std::memory_order_relaxed);
        did = true;
      }
    }
    if (!did) {
      if (!spinning) ts = clk::now(), spinning = true;
      __builtin_arm_yield();
    }
  }
  if (spinning) spin += std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - ts).count();
  if (spin && X.S->spin_ns) X.S->spin_ns->fetch_add(spin, std::memory_order_relaxed);
}
}  // namespace

void expert_step(const ExpertStep& s, const ExpertJob* jobs, int n) {
  ExpCtx X;
  X.S = &s;
  X.n = n;
  X.IA = s.IA;
  X.H = s.H;
  X.ref = s.ref;
  X.softcap_act = s.softcap_act;
  X.gcap = s.gcap;
  X.ucap = s.ucap;
  X.arms = s.arms;
  X.xq = s.xq;
  X.xf = s.xf;
  X.actf = s.actf;
  X.eout = s.eout;
  X.act = s.act;
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
    if (!(s.arms & STEP_NOACT)) s.act[jobs[e].slot_out].reset(jobs[e].inter);
  }
  static const int dn_env = getenv("DMENG_DN_ROWS") ? std::max(1, atoi(getenv("DMENG_DN_ROWS"))) : 0;  // experiments
  if (dn_env) X.dn_ch = dn_env;
  if (X.start[n] > ExpCtx::MAXCH) X.arms &= ~STEP_RERUN;  // (more gate-up chunks than the states hold)
  if (X.arms & STEP_RERUN)
    for (int c = 0; c < X.start[n]; c++) X.cst[c].store(0, std::memory_order_relaxed);
  {  // the down quarters per expert
    const int q = std::max(1, X.dn_ch / 4), uq = (s.H + q - 1) / q;
    for (int e = 0; e < n; e++) X.dn_next[e].v.store(0, std::memory_order_relaxed);
    X.dn_rem.store(uq * n, std::memory_order_relaxed);
  }
  s.pool->run_any(phase_gudn, &X);
}

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
  ExpertStep s;
  s.pool = pool.get();
  s.IA = std::max(I_, sh_inter_);
  s.H = H;
  s.ref = o.ref;
  s.softcap_act = v.act == MoeView::SOFTCAP_SWIGLU;
  s.gcap = v.gate_cap;
  s.ucap = v.up_cap;
  s.xq = &xin.q;
  s.xf = x;
  s.actf = actf.data();
  s.eout = eout.data();
  s.act = act.data();
  s.spin_ns = &exp_spin_ns;
  s.reruns = &rr_count;
  static const bool rr_ab = getenv("DMENG_RERUN_AB") && atoi(getenv("DMENG_RERUN_AB")) > 0;  // measurement
  static uint32_t rr_lcg = 777;
  int rr = 0;
  s.arms = STEP_RERUN;  // a stalled claimed gate-up chunk is re-run (co-runners: PARITY.md); DMENG_RERUN_AB alternates it
  if (rr_ab) rr_lcg = rr_lcg * 1664525u + 1013904223u, rr = (rr_lcg >> 16) & 1, s.arms = rr ? STEP_RERUN : 0;
  double by = 0;
  for (int e = 0; e < n; e++) by += (double)jobs[e].gu.bytes() + (jobs[e].up.kind == Weight::NONE ? 0 : jobs[e].up.bytes()) + jobs[e].dn.bytes();
  const int aw = cache.st.wait_ms != xc_wait_seen;
  const auto t0 = clk::now();
  expert_step(s, jobs, n);
  const int k = std::min(n, 17);
  xc_ns[aw][k] += std::chrono::duration<double, std::nano>(clk::now() - t0).count();
  xc_bytes[aw][k] += by;
  xc_calls[aw][k] += 1;
  if (rr_ab) rr_ns[rr] += std::chrono::duration<double, std::nano>(clk::now() - t0).count(), rr_bytes[rr] += by, rr_calls[rr] += 1;
  xc_wait_seen = cache.st.wait_ms;
}

std::string Core::xc_json() const {
  std::string o = "{";
  for (int a = 0; a < 2; a++) {
    o += a ? ",\"after_wait\":[" : "\"no_wait\":[";
    bool first = true;
    for (int i = 0; i < 18; i++) {
      if (!xc_calls[a][i]) continue;
      char b[128];
      snprintf(b, sizeof b, "%s[%d,%.0f,%.1f]", first ? "" : ",", i, xc_calls[a][i], xc_bytes[a][i] / std::max(1.0, xc_ns[a][i]));
      o += b;
      first = false;
    }
    o += "]";
  }
  return o + "}";
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
  auto grab = [&]() {
    int x2;
    if (cache.pred_evict) {
      x2 = cache.pick_victim(L, 0, nullptr);
      if (x2 >= 0) cache.evict(L, x2);
    } else {
      x2 = cache.victim(L, 0);
    }
    if (x2 < 0) x2 = cache.steal(L);
    return x2;
  };
  int x = grab();
  while (x < 0 && cache.wait_name_read(L)) x = grab();  // names' reads in flight: one lands, its slot is taken
  if (x < 0) {  // phone sizes: the loop hold's pins go before an abort (pins only: the outputs cannot change), as in
                // Core::moe_chunk; never reached while the pool holds the hold's worst case (12 x 2 x 8 + 8 on Inkling)
    bool any = false;
    for (int L2 = 0; L2 < (int)holds.size(); L2++) {
      for (int hs : holds[L2]) cache.unpin(L2, hs);
      any |= !holds[L2].empty();
      holds[L2].clear();
    }
    if (any) {
      n_hold_release++;
      x = grab();
    }
  }
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
  if (!cache.floor_keys.empty()) cache.expire_floors();  // background reads' values whose step has run
  step_t0 = clk::now();
  if (step == 0 && run_ahead) {  // a token starts: the last one's compute (wall - wait) into the budget's average
    if (ra_tok_t0.time_since_epoch().count()) {
      const double comp = std::max(0.0, ms_between(ra_tok_t0, step_t0) - (t.wait - ra_tok_wait0));
      ra_comp_ema = ra_comp_ema > 0 ? 0.8 * ra_comp_ema + 0.2 * comp : comp;
    }
    ra_tok_t0 = step_t0;
    ra_tok_wait0 = t.wait;
    ra_tok_ms = 0;
    ra_tok_events = 0;
  }
  cache.tok_now = tok_clock;
  if (hook) {
    auto tp = clk::now();
    stage_mark(ST_PREFETCH);
    hook->before_moe(*this, step, x);
    t.pf += ms_since(tp);
  }
  auto t0 = clk::now();
  stage_mark(ST_MOE_ROUTER);
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
  // the run-ahead's names for this step: scored against the real picks, unpinned (no slot is taken before this step's own
  // pins: nothing between here and them evicts)
  ra_cov = ra.step == step;
  if (ra.step >= 0) ra_score(step, picks.data() + (size_t)step * K);
  for (int d = 1; d < 4; d++) {  // the deeper predictions (measurement)
    if ((int)ra_deep[d].size() <= step) continue;
    Deep& D = ra_deep[d][step];
    if (D.step != step) continue;
    const int32_t* pk = picks.data() + (size_t)step * K;
    for (int j = 0; j < K; j++) {
      const bool in = std::find(D.pred.begin(), D.pred.end(), pk[j]) != D.pred.end();
      deep_top8[d] += in;
      if (D.absent[pk[j]]) deep_would[d] += 1, deep_found[d] += in;
    }
    for (int e : D.pred)
      if (D.absent[e]) deep_issued[d] += 1, deep_wasted[d] += std::find(pk, pk + K, e) == pk + K;
    deep_n[d] += K;
    D.step = -1;
  }
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
  stage_mark(ST_MOE_CACHE);
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
    const bool reused = pass > 0 && std::find(ut.begin(), ut.end(), e) != ut.end();
    if (reused) S.n_reuse_hit++;
    if (reused && s < 0) S.n_reuse_reread++;
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
  stage_mark(ST_PREFETCH);
  if (hook) hook->after_router(*this, step);  // names for a later step: queued behind this step's own misses
  auto t2 = clk::now();
  t.pf += ms_between(t2a, t2);
  stage_mark(ST_MOE_EXPERTS);

  // waves: the shared expert + every routed expert already in memory; then, while picks remain, whatever has landed, or
  // (none has) a wait for the first one left, in pick order, and whatever landed with it. Each expert's output goes to
  // its own row and the combination below adds them in pick order, so the waves change when an expert runs, not the sum.
  bool done[MAXK] = {false};
  bool ra_done = false;
  const double wait0 = S.wait_ms, ra0 = ra.ms;
  for (int left = K, first = 1; left > 0 || first; first = 0) {
    ExpJob jb[MAXK + 1];
    int n = 0;
    if (first && v.shared) jb[n++] = shared_job(v);
    for (int pass = 0; pass < 2 && n == (first && v.shared ? 1 : 0); pass++) {  // pass 1: nothing landed, wait for one
      for (int j = 0; j < K; j++) {
        if (done[j]) continue;
        if (pass == 1 && n == (first && v.shared ? 1 : 0)) {
          if (first) break;  // the first wave never waits (the shared expert runs meanwhile)
          if (run_ahead && !ra_done && !cache.ready(L, sl[j])) {  // about to block: run ahead one step instead
            ra_done = true;
            ra_predict(step, v, done, w, cache.pending_of(L, sl[j]));
          }
          bool runnable = false;
          for (int j2 = j + 1; j2 < K && !runnable; j2++) runnable = !done[j2] && cache.ready(L, sl[j2]);
          const double w0 = S.wait_ms;
          cache.wait(L, sl[j]);
          (missed[j] ? S.wait_ms_miss : S.wait_ms_late) += S.wait_ms - w0;
          if (runnable) S.wait_ms_runnable += S.wait_ms - w0;
        } else if (!cache.ready(L, sl[j])) {
          continue;
        }
        F.mats(cache.rec(L, sl[j]), jb[n].gu, jb[n].up, jb[n].dn);
        jb[n].inter = F.I;
        jb[n].slot_out = j;
        n++;
        done[j] = true;
        left--;
      }
    }
    if (n) run_experts(jb, n, v, x);
  }
  t.bytes_exp += (double)K * F.touched;
  int rows[MAXK];
  for (int j = 0; j < K; j++) rows[j] = j;
  stage_mark(ST_MOE_COMBINE);
  combine(v, w, rows, K, shw_, y);
  std::copy(pk, pk + K, used.data() + (size_t)step * K);
  std::copy(w, w + K, uw.data() + (size_t)step * K);
  auto t3 = clk::now();
  const double wt = S.wait_ms - wait0, rat = ra.ms - ra0;
  t.wait += wt;
  wait_step[step] += wt;
  t.experts += ms_between(t2, t3) - wt - rat;
  t.ra += rat;
  ra_where(step, wt);
  stage_mark(ST_MOE_CACHE);
  finish_step(step, sl, K);
  if (hook) {
    auto tp = clk::now();
    stage_mark(ST_PREFETCH);
    hook->after_moe(*this, step, pk, w);
    hook->release(*this, step);
    t.pf += ms_since(tp);
  }
  t.cache += ms_since(t3);
}

// ------------------------------------------------------------------------------------------------ run-ahead
void Core::ra_predict(int step, const MoeView& v, const bool* done, const float* w, const std::atomic<int>* watch) {
  if (step + 1 >= steps || ra_off) return;
  if ((ra_max_tok > 0 && ra_tok_events >= ra_max_tok) || (ra_budget > 0 && ra_comp_ema > 0 && ra_tok_ms >= ra_budget * ra_comp_ema)) {
    ra.skip_budget += 1;
    return;
  }
  if (ra_sat_skip && cache.rd.saturated()) {
    ra.skip_sat += 1;
    return;
  }
  auto t0 = clk::now();
  ra.watch = watch;
  ra.landed = false;
  // this step's output from what has run: the experts done so far and the shared expert, in pick order
  static thread_local std::vector<float> ypart;
  ypart.assign(H, 0.f);
  int rows[MAXK], n = 0;
  float wd[MAXK];
  for (int j = 0; j < K; j++)
    if (done[j]) rows[n] = j, wd[n++] = w[j];
  combine(v, wd, rows, n, shw_, ypart.data());
  ra.pred.assign(std::max(K, ra_k), -1);
  g_rows_stop.store(watch, std::memory_order_relaxed);  // the row phases stop claiming once it lands
  int np = ra_stopped() ? 0 : run_ahead(step, ypart.data(), ra.pred.data(), std::max(K, ra_k));
  g_rows_stop.store(nullptr, std::memory_order_relaxed);
  if (ra_stopped()) {  // landed: the speculative work is thrown away (its rows may be incomplete)
    np = 0;
    ra.aborted += 1;
    ra.over_ms += ms_since(ra.seen);
  }
  ra.watch = nullptr;
  ra.pred.resize(std::max(0, np));
  for (int d = 2; np > 0 && d <= std::min(ra_depth, 4) && run_ahead_deeper && step + d < steps; d++) {  // measurement only
    if ((int)ra_deep[d - 1].size() < steps) ra_deep[d - 1].resize(steps);
    Deep& D = ra_deep[d - 1][step + d];
    D.pred.assign(K, -1);
    const int nd = run_ahead_deeper(d, D.pred.data(), K);
    if (nd <= 0) break;
    D.pred.resize(nd);
    const int Ld = mi.step_layer[step + d];
    D.absent.assign(NE, 0);
    for (int e = 0; e < NE; e++) D.absent[e] = cache.slot_of(Ld, e) < 0;
    D.step = step + d;
    deep_events[d - 1] += 1;
  }
  const int s1 = step + 1, L1 = mi.step_layer[s1];
  ra.absent.assign(NE, 0);
  for (int e = 0; e < NE; e++) ra.absent[e] = cache.slot_of(L1, e) < 0;
  ra.pslot.assign(ra.pred.size(), -1);
  // fast mode: name only what the real pass would read, by the mode's own rule on the speculative routing weights (a
  // missing pick of weight > read_w, then the largest missing until the missing weight <= read_mass); exact: all
  bool nm[2 * MAXK + 64];
  const int npn = std::min<int>(std::min<int>((int)ra.pred.size(), ra_k), 2 * MAXK + 64);
  for (int i = 0; i < npn; i++) nm[i] = true;
  if (ns.on && npn > 0 && (int)ra_lgv.size() >= NE) {
    const int kk = std::min(K, npn);
    double fr[MAXK];
    if (v.rule == MoeView::SOFTMAX_TOPK_RENORM) {
      float mx = -INFINITY;
      for (int e = 0; e < NE; e++) mx = std::max(mx, ra_lgv[e]);
      double zs = 0, s = 0;
      for (int e = 0; e < NE; e++) zs += std::exp((double)(ra_lgv[e] - mx));
      for (int i = 0; i < kk; i++) s += fr[i] = std::exp((double)(ra_lgv[ra.pred[i]] - mx)) / zs;
      if (v.renorm)
        for (int i = 0; i < kk; i++) fr[i] /= s;
    } else {
      double s = 0;
      for (int i = 0; i < kk; i++) s += fr[i] = 1.0 / (1.0 + std::exp(-(double)ra_lgv[ra.pred[i]]));
      for (int i = 0; i < kk; i++) fr[i] /= s + 1e-20;
    }
    bool must[MAXK] = {false}, miss[MAXK];
    double mass = 0;
    for (int i = 0; i < kk; i++) {
      const int sl1 = cache.slot_of(L1, ra.pred[i]);
      miss[i] = sl1 < 0 || !cache.ready(L1, sl1);
      if (!miss[i]) continue;
      if (fr[i] > ns.read_w) must[i] = true;
      else mass += fr[i];
    }
    while (mass > ns.read_mass) {
      int b = -1;
      for (int i = 0; i < kk; i++)
        if (miss[i] && !must[i] && (b < 0 || fr[i] > fr[b])) b = i;
      if (b < 0) break;
      must[b] = true;
      mass -= fr[b];
    }
    for (int i = 0; i < npn; i++) {
      nm[i] = i < kk && must[i];
      if (!nm[i] && ra.absent[ra.pred[i]]) ra.withheld += 1;
    }
  }
  const long abs = now(s1);
  for (size_t i = 0; i < ra.pred.size() && (int)i < ra_k; i++)
    if (ra.absent[ra.pred[i]] && nm[i]) {
      ra.pslot[i] = cache.name(L1, ra.pred[i], abs, ra_reserve);
      ra.issued += ra.pslot[i] >= 0;
      ra.refused += ra.pslot[i] < 0;
    }
  ra.step = np > 0 ? s1 : -1;
  ra.t = clk::now();
  ra.events += 1;
  ra.ms += ms_between(t0, ra.t);
  ra_tok_ms += ms_between(t0, ra.t);
  ra_tok_events += 1;
}

void Core::ra_where(int step, double wt) {
  steps_total += 1;
  if (wt <= 0.01) {
    steps_no_wait += 1;
  } else if (ra_cov) {
    wait_cov_ms += wt, steps_wait_cov += 1;
  } else {
    wait_uncov_ms += wt, steps_wait_uncov += 1;
    wait_unnamed_step[step] += wt;
    nwait_unnamed_step[step]++;
  }
  ra_cov = false;
}

void Core::ra_score(int step, const int32_t* pk) {
  if (ra.step != step) {  // (not the step predicted: release the names unscored)
    const int L1 = mi.step_layer[ra.step];
    for (size_t i = 0; i < ra.pslot.size(); i++)
      if (ra.pslot[i] >= 0) cache.unpin_pf(L1, ra.pslot[i], ra.pred[i]);
    ra.step = -1;
    return;
  }
  const int L = mi.step_layer[step];
  for (int j = 0; j < K; j++) {
    const int e = pk[j];
    const bool predicted = std::find(ra.pred.begin(), ra.pred.begin() + std::min<size_t>(ra.pred.size(), K), e) != ra.pred.begin() + std::min<size_t>(ra.pred.size(), K);
    ra.top8 += predicted;
    if (!ra.absent[e]) continue;
    ra.would_read += 1;  // absent when run-ahead looked: an on-demand read without it
    bool named = false;
    for (size_t i = 0; i < ra.pred.size() && (int)i < ra_k; i++) named = named || (ra.pred[i] == e && ra.pslot[i] >= 0);
    if (named) {
      ra.found += 1;
      ra.lead_ms += ms_between(ra.t, step_t0);
    }
  }
  ra.top8_n += K;
  for (size_t i = 0; i < ra.pred.size() && (int)i < ra_k; i++) {
    if (ra.pslot[i] < 0) continue;
    if (std::find(pk, pk + K, ra.pred[i]) == pk + K) ra.wasted += 1;
    cache.unpin_pf(L, ra.pslot[i], ra.pred[i]);
  }
  ra.step = -1;
}

std::string Core::ra_json() const {
  char b[1024];  // (the deeper predictions appended)
  const double ev = std::max(1.0, ra.events);
  snprintf(b, sizeof b,
           "{\"events\":%.0f,\"ms_per_event\":%.3f,\"ms_total\":%.1f,\"aborted\":%.0f,\"overrun_ms_total\":%.2f,\"top8_overlap\":%.4f,"
           "\"would_read\":%.0f,\"found\":%.0f,\"found_share\":%.4f,\"lead_ms\":%.3f,\"issued\":%.0f,\"refused\":%.0f,\"wasted\":%.0f,\"withheld\":%.0f,\"skip_budget\":%.0f,\"skip_saturated\":%.0f}",
           ra.events, ra.ms / ev, ra.ms, ra.aborted, ra.over_ms, ra.top8 / std::max(1.0, ra.top8_n), ra.would_read, ra.found,
           ra.found / std::max(1.0, ra.would_read), ra.lead_ms / std::max(1.0, ra.found), ra.issued, ra.refused, ra.wasted, ra.withheld, ra.skip_budget, ra.skip_sat);
  std::string out = b;
  {
    out.pop_back();
    char c[384];
    snprintf(c, sizeof c, ",\"steps\":%.0f,\"steps_no_wait\":%.0f,\"steps_wait_named\":%.0f,\"steps_wait_not_named\":%.0f,"
             "\"wait_named_ms\":%.1f,\"wait_not_named_ms\":%.1f}", steps_total, steps_no_wait, steps_wait_cov, steps_wait_uncov,
             wait_cov_ms, wait_uncov_ms);
    out += c;
  }
  if (ra_depth > 1) {
    out.pop_back();
    for (int d = 2; d <= std::min(ra_depth, 4); d++) {
      char c[256];
      snprintf(c, sizeof c, ",\"depth%d\":{\"events\":%.0f,\"top8_overlap\":%.4f,\"would_read\":%.0f,\"found_share\":%.4f,\"issued\":%.0f,\"wasted\":%.0f}", d,
               deep_events[d - 1], deep_top8[d - 1] / std::max(1.0, deep_n[d - 1]), deep_would[d - 1],
               deep_found[d - 1] / std::max(1.0, deep_would[d - 1]), deep_issued[d - 1], deep_wasted[d - 1]);
      out += c;
    }
    out += "}";
  }
  return out;
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
  stage_mark(ST_MOE_CACHE);
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
      if (n < std::min(ns.cold_floor >= 0 ? ns.cold_floor : 2 * K, NE - K)) {
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
  stage_mark(ST_PREFETCH);
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
  stage_mark(ST_MOE_EXPERTS);

  // pass A: shared expert + everything landed; then the waits; pass B: the rest
  const double wait0 = S.wait_ms, ra0 = ra.ms;
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
  if (run_ahead) {  // run ahead one step if a pick this step keeps is still in flight (it is about to be waited for)
    bool dn[MAXK] = {false};
    int wj = -1;
    for (int j = 0; j < K; j++) {
      dn[j] = ch[j].done;
      if (wj < 0 && !dropped[j] && !ch[j].done && !cache.ready(L, ch[j].slot)) wj = j;
    }
    if (wj >= 0) ra_predict(step, v, dn, pwl, cache.pending_of(L, ch[wj].slot));
  }
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
  stage_mark(ST_MOE_COMBINE);
  combine(v, wf, rows, nf, shw_, y);
  for (int i = 0; i < nf && i < K; i++) {
    used[(size_t)step * K + i] = set[i];
    uw[(size_t)step * K + i] = wf[i];
  }
  auto t3 = clk::now();
  const double wt = S.wait_ms - wait0, rat = ra.ms - ra0;
  t.wait += wt;
  wait_step[step] += wt;
  t.experts += ms_between(t2, t3) - wt - rat;
  t.ra += rat;
  ra_where(step, wt);
  auto& ut = used_tok[L];
  if (looped(step)) {
    if (pass == 0) ut.clear();
    for (int i = 0; i < nf; i++) ut.push_back(set[i]);
  }
  stage_mark(ST_MOE_CACHE);
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
    stage_mark(ST_PREFETCH);
    hook->after_moe(*this, step, pk, pwl);
    hook->release(*this, step);
    t.pf += ms_since(tp);
  }
  t.cache += ms_since(t3);
}

}  // namespace dm
