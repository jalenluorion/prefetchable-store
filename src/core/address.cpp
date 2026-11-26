// The early address: naming, recorded-routing replay and the trained routers.
#include "dmeng/address.h"
#include <cmath>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <sstream>

namespace dm {

// The routers normalise the embedding with torch RMSNorm's default eps in float32.
constexpr float ROUTER_EPS = 1.1920929e-07f;

// ------------------------------------------------------------------------------------------------ naming
void Naming::ensure(int steps) {
  if (pins[0].empty())
    for (auto& p : pins) p.resize(steps);
}

void Naming::name(Core& c, int p, int step, int e, long deadline) {
  const int L = c.mi.step_layer[step];
  const long abs = deadline >= 0 ? deadline : (c.tok_clock + (p - c.pos)) * c.steps + step;
  const int sl = c.cache.name(L, e, abs, -1, per_step);
  if (sl >= 0) pins[p & (PR - 1)][step].push_back({sl, e, clk::now()});
}

void Naming::release_at(Core& c, int p, int step, bool chunk) {
  if (pins[0].empty()) return;
  const int L = c.mi.step_layer[step];
  auto& x = pins[p & (PR - 1)][step];
  for (auto& q : x) {
    c.cache.unpin_pf(L, q.slot, q.e);
    const double ld = std::chrono::duration<double, std::milli>(c.step_t0 - q.t).count();
    if (ld >= 0) lead_ms[chunk] += ld, lead_n[chunk] += 1;
  }
  x.clear();
}

std::string Naming::lead_json() const {
  std::ostringstream o;
  o << "{\"one_position_names\":" << lead_n[0] << ",\"one_position_lead_ms\":" << lead_ms[0] / std::max(1.0, lead_n[0])
    << ",\"verify_names\":" << lead_n[1] << ",\"verify_lead_ms\":" << lead_ms[1] / std::max(1.0, lead_n[1]) << "}";
  return o.str();
}

// ------------------------------------------------------------------------------------------------ oracle replay
bool Oracle::load(const std::string& path, int steps, int K, std::string& err) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    err = "cannot open " + path;
    return false;
  }
  int32_t hdr[3];
  if (fread(hdr, 4, 3, f) != 3 || hdr[1] != steps || hdr[2] != K) {
    err = path + ": routing record does not match the model";
    fclose(f);
    return false;
  }
  npos = hdr[0];
  rec.resize((size_t)npos * steps * K);
  const size_t got = fread(rec.data(), 2, rec.size(), f);
  fclose(f);
  if (got != rec.size()) {
    err = path + ": short routing record";
    return false;
  }
  ns = steps;
  k = K;
  return true;
}

void Oracle::name_step(Core& c, int p, int s, long deadline) {
  if (p >= npos || s >= ns) return;
  const int16_t* r = rec.data() + ((size_t)p * ns + s) * k;
  per_step = k;
  for (int j = 0; j < k; j++) name(c, p, s, r[j], deadline);
}

void Oracle::chunk_start(Core& c, int n, const int32_t*, const float* const*) {
  ensure(ns);
  cn = n;
  const int l = token_lead ? ns : std::min(lead, ns);
  for (int s = 0; s < l; s++)
    for (int i = 0; i < n; i++) name_step(c, c.pos + i, s, c.now(s));
}

void Oracle::chunk_after_router(Core& c, int step, int n) {
  if (token_lead || step + lead >= ns) return;
  for (int i = 0; i < n; i++) name_step(c, c.pos + i, step + lead, c.now(step + lead));
}

void Oracle::token_start(Core& c, int, const float*) {
  ensure(ns);
  const int p = c.pos;
  if (!token_lead) {
    for (int s = 0; s < std::min(lead, ns); s++) name_step(c, p, s);
  } else if (p == 0) {
    for (int s = 0; s < ns; s++) name_step(c, p, s);  // nothing earlier to name the first token from
  }
}

void Oracle::after_router(Core& c, int step) {
  const int p = c.pos;
  if (!token_lead) {
    if (step + lead < ns) name_step(c, p, step + lead);
  } else {
    name_step(c, p + 1, step);  // a whole token early: next token, same step
  }
}

// ------------------------------------------------------------------------------------------------ transplant k1
void Transplant::router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) {
  if (lg.empty()) {
    ensure(c.steps);
    lg.assign(c.NE, 0.f);
    nm.assign((size_t)c.steps * 8, -1);
  }
  pending = step + 1 < c.steps && c.views.size() == (size_t)c.steps ? step + 1 : -1;
  if (pending < 0) return;
  jobs.push_back({c.views[pending]->router, &c.xin.q, x, lg.data()});  // on the fp32 MoE input, as step + 1's router would run
}

void Transplant::after_router(Core& c, int) {
  if (pending < 0) return;
  auto t0 = clk::now();
  const MoeView& v = *c.views[pending];
  const int NE = c.NE, top = std::min(m, NE);
  static thread_local std::vector<float> key;
  static thread_local std::vector<int> idx;
  key.resize(NE);
  idx.resize(NE);
  for (int e = 0; e < NE; e++) {
    idx[e] = e;
    if (v.rule == MoeView::SOFTMAX_TOPK_RENORM) key[e] = lg[e];
    else {  // the sigmoid rule ranks by score + bias
      const float yv = 1.f / (1.f + std::exp(std::fabs(lg[e])));
      key[e] = (lg[e] < 0 ? yv : 1.f - yv) + v.router_bias[e];
    }
  }
  std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a < b); });
  for (int j = 0; j < std::min(top, 8); j++) nm[(size_t)pending * 8 + j] = idx[j];
  for (int j = 0; j < top; j++) name(c, c.pos, pending, idx[j]);
  pending = -1;
  t_ms += ms_since(t0);
}

void Transplant::after_moe(Core& c, int step, const int32_t* pk, const float*) {
  if (step == 0 || nm.empty()) return;
  const int* n = nm.data() + (size_t)step * 8;
  for (int j = 0; j < c.K; j++)
    for (int i = 0; i < 8; i++) hit8 += n[i] == pk[j];
  calls += 1;
}

std::string Transplant::stats_json(Core& c, int ntok) {
  std::ostringstream o;
  o << "{\"m\":" << m << ",\"top_k_in_named8_steps1plus\":" << hit8 / std::max(1.0, calls * c.K) << ",\"naming_ms_per_token\":" << t_ms / std::max(1, ntok) << "}";
  return o.str();
}

// ------------------------------------------------------------------------------------------------ trained routers
bool Routers::load(const std::string& path, Core& c, std::string& err) {
  if (!R.load(path, mem, err)) return false;
  lag = (int)R.geti("lag");
  nt = (int)(R.has_cfg("apps") ? R.geti("apps") : R.geti("layers"));
  H = (int)R.geti("hidden");
  hid = (int)R.geti("hidden_mlp");
  bits = R.has_cfg("bits") ? (int)R.geti("bits") : 8;
  split = R.has_cfg("split_pre") && R.geti("split_pre") == 1;
  if (nt != c.steps || H != c.H || (R.has_cfg("experts") && R.geti("experts") != c.NE)) {
    err = path + ": router file does not match the model (" + std::to_string(nt) + " targets for " + std::to_string(c.steps) + " steps)";
    return false;
  }
  NE = c.NE;
  rows = NE + hid;
  lay.resize(nt);
  bytes_tok = 0;
  for (int t = 0; t < nt; t++) {
    const std::string s = std::to_string(t) + ".";
    Lay& y = lay[t];
    y.emb_norm = (const float*)R.ptr(s + "emb_norm");
    if (split) {
      y.pre_e = R.weight(s + "pre_e");
      y.pre_p = R.weight(s + "pre_p");
      y.pre.rows = y.pre_e.rows;
      y.pre.cols = y.pre_e.cols + y.pre_p.cols;
    } else {
      y.pre = R.weight(s + "pre");
    }
    y.has_lag = R.has(s + "lagm.q") || R.has(s + "lagm.q4");
    if (y.has_lag) y.lagm = R.weight(s + "lagm");
    y.b_pre = (const float*)R.ptr(s + "b_pre");
    if (hid > 0) {
      y.m2 = R.weight(s + "m2");
      y.m2_b = (const float*)R.ptr(s + "m2_b");
    }
    if (y.pre.rows != rows || y.pre.cols != 2 * H || (t >= lag) != y.has_lag) {
      err = path + ": router " + std::to_string(t) + " has an unexpected shape";
      return false;
    }
    bytes_tok += (split ? y.pre_e.bytes() + y.pre_p.bytes() : y.pre.bytes()) + (y.has_lag ? y.lagm.bytes() : 0) + y.m2.bytes();
  }
  prev.assign((size_t)nt * H, 0.f);
  pre_out.assign((size_t)nt * rows, 0.f);
  lag_out.assign(rows, 0.f);
  xpre.resize(nt);
  for (auto& b : xpre) b.init(2 * H, c.o.prec);
  if (hid > 0) xh.init(hid, c.o.prec);
  nm.assign((size_t)nt * 16, -1);
  nq.assign((size_t)nt * 16, 0.f);
  nbin.assign((size_t)nt * 16, 0);
  if (split) {
    xe.resize(nt);
    xp.resize(nt);
    for (auto& b : xe) b.init(H, c.o.prec);
    for (auto& b : xp) b.init(H, c.o.prec);
    e_out.assign((size_t)nt * rows, 0.f);
    if (ecap > 0) {
      e_pool.assign((size_t)ecap * nt * rows, 0.f);
      e_tok.assign(ecap, -1);
      e_age.assign(ecap, -1);
    }
  }
  ensure(nt);
  return true;
}

const float* Routers::e_lookup(int tok) {
  if (ecap <= 0) return nullptr;
  auto it = e_map.find(tok);
  if (it == e_map.end()) return nullptr;
  e_age[it->second] = ++e_clock;
  return e_pool.data() + (size_t)it->second * nt * rows;
}

namespace {
struct PreCtx {
  Routers* P;
  bool need_e;
  alignas(128) std::atomic<int> next_x{0};
  alignas(128) std::atomic<int> next_r{0};
};
float emb_scale(const float* e, int H) {
  double ss = 0;
  for (int i = 0; i < H; i++) ss += (double)e[i] * e[i];
  return 1.f / std::sqrt((float)(ss / H) + ROUTER_EPS);
}
void phase_pre_x(void* p, int, int) {  // per target: [emb_norm_t(emb), prev_t] quantised
  PreCtx& C = *(PreCtx*)p;
  Routers& P = *C.P;
  const int H = P.H;
  std::vector<float> v(2 * H);
  const float* e = P.emb;
  const float r = emb_scale(e, H);
  for (int t; (t = C.next_x.fetch_add(1, std::memory_order_relaxed)) < P.nt;) {
    const float* w = P.lay[t].emb_norm;
    for (int i = 0; i < H; i++) v[i] = e[i] * r * w[i];
    std::memcpy(v.data() + H, P.prev.data() + (size_t)t * H, sizeof(float) * H);
    P.xpre[t].prep(v.data(), 2 * H);
  }
}
void phase_pre_rows(void* p, int, int) {  // every target's pre rows, in 32-row chunks
  PreCtx& C = *(PreCtx*)p;
  Routers& P = *C.P;
  const int per = (P.rows + 31) / 32, total = per * P.nt;
  for (int ch; (ch = C.next_r.fetch_add(1, std::memory_order_relaxed)) < total;) {
    const int t = ch / per, r0 = (ch % per) * 32, r1 = std::min(P.rows, r0 + 32);
    mv(P.lay[t].pre, r0, r1, &P.xpre[t].q, nullptr, P.pre_out.data() + (size_t)t * P.rows + r0);
  }
}
void phase_split_x(void* p, int, int) {  // per target: emb_norm_t(emb) and prev_t, quantised apart
  PreCtx& C = *(PreCtx*)p;
  Routers& P = *C.P;
  const int H = P.H;
  std::vector<float> v(H);
  const float* e = P.emb;
  const float r = emb_scale(e, H);
  for (int t; (t = C.next_x.fetch_add(1, std::memory_order_relaxed)) < P.nt;) {
    if (C.need_e) {
      const float* w = P.lay[t].emb_norm;
      for (int i = 0; i < H; i++) v[i] = e[i] * r * w[i];
      P.xe[t].prep(v.data(), H);
    }
    P.xp[t].prep(P.prev.data() + (size_t)t * H, H);
  }
}
void phase_split_rows(void* p, int, int) {  // previous-token columns of every target, and the embedding columns on a miss
  PreCtx& C = *(PreCtx*)p;
  Routers& P = *C.P;
  const int per = (P.rows + 31) / 32, half = per * P.nt, total = C.need_e ? 2 * half : half;
  for (int ch; (ch = C.next_r.fetch_add(1, std::memory_order_relaxed)) < total;) {
    const bool ep = ch >= half;
    const int c2 = ep ? ch - half : ch;
    const int t = c2 / per, r0 = (c2 % per) * 32, r1 = std::min(P.rows, r0 + 32);
    if (ep) mv(P.lay[t].pre_e, r0, r1, &P.xe[t].q, nullptr, P.e_out.data() + (size_t)t * P.rows + r0);
    else mv(P.lay[t].pre_p, r0, r1, &P.xp[t].q, nullptr, P.pre_out.data() + (size_t)t * P.rows + r0);
  }
}
}  // namespace

namespace {
// a router's MLP after its first layer, on the pool: the gelu and quantisation of the hidden values block by block, then
// the second layer's rows (16 per claim) once every block is done. Each value is the serial path's (gelu_erf, xq_block,
// mv per row).
struct MlpCtx {
  const float* z;
  float* hv;
  XQ* xq;
  const Weight* m2;
  float* out;
  int nb, nr, NE;
  alignas(128) std::atomic<int> next{0};
  alignas(128) std::atomic<int> left{0};
};
void phase_mlp(void* p, int, int) {
  MlpCtx& X = *(MlpCtx*)p;
  for (int ch; (ch = X.next.fetch_add(1, std::memory_order_relaxed)) < X.nb + X.nr;) {
    if (ch < X.nb) {
      float* h = X.hv + (size_t)ch * XB;
      for (int i = 0; i < XB; i++) h[i] = gelu_erf(X.z[(size_t)ch * XB + i]);
      xq_block(*X.xq, ch, h);
      X.left.fetch_sub(1, std::memory_order_release);
      continue;
    }
    while (X.left.load(std::memory_order_acquire) > 0) __builtin_arm_yield();
    const int r0 = (ch - X.nb) * 16, r1 = std::min(X.NE, r0 + 16);
    mv(*X.m2, r0, r1, X.xq, nullptr, X.out + r0);
  }
}
}  // namespace

void Routers::finish(Core& c, int t, bool with_lag) {
  finish_z(c, t, pre_out.data() + (size_t)t * rows, with_lag ? lag_out.data() : nullptr, c.pos, -1, nm.data() + (size_t)t * 16,
           nq.data() + (size_t)t * 16, nbin.data() + (size_t)t * 16);
}

void Routers::finish_z(Core& c, int t, const float* pre, const float* lagr, int p, long deadline, int* n, float* q16, int* b16) {
  // z = pre + lag part + biases; first NE rows: lin, next hid rows: mlp.0 -> gelu -> mlp.2
  const Lay& y = lay[t];
  const bool single = deadline < 0;  // a one-position step (the keep-score's features and the heavy head go with it)
  static thread_local std::vector<float> z, z2, zz, hv;
  static thread_local std::vector<int> idx;
  z.resize(rows);
  z2.resize(NE);
  zz.resize(NE);
  idx.resize(NE);
  for (int r = 0; r < rows; r++) z[r] = pre[r] + y.b_pre[r] + (lagr ? lagr[r] : 0.f);
  // the MLP path in one fork-join (MlpCtx) when its second layer is large enough for the pool to beat one thread on it
  // (Flash-Next's: 512 x 1024 8-bit, 0.54 MB, an estimated 25 us on one core; token by token, DMENG_AB, the token was
  // faster in three runs of three); smaller ones (the other families' published MLP routers, hidden 512) stay on the
  // calling thread
  if (hid > 0 && hid % XB == 0 && y.m2.bytes() >= (256u << 10) && c.pool->size() > 1) {
    hv.resize(hid);
    MlpCtx X;
    X.z = z.data() + NE, X.hv = hv.data(), X.xq = &xh.q, X.m2 = &y.m2, X.out = z2.data();
    X.nb = hid / XB, X.nr = (NE + 15) / 16, X.NE = NE;
    xh.reset(hid);
    X.left.store(X.nb, std::memory_order_relaxed);
    c.pool->run_any(phase_mlp, &X);
  } else if (hid > 0) {  // the MLP path (a linear router has none)
    hv.resize(hid);
    for (int i = 0; i < hid; i++) hv[i] = gelu_erf(z[NE + i]);
    xh.prep(hv.data(), hid);
    mv(y.m2, 0, NE, &xh.q, nullptr, z2.data());
  }
  for (int e = 0; e < NE; e++) {
    zz[e] = z[e] + (hid > 0 ? z2[e] + y.m2_b[e] : 0.f);
    idx[e] = e;
  }
  if (single) {
    if (zlast.empty()) zlast.assign((size_t)nt * NE, 0.f);
    std::copy(zz.begin(), zz.begin() + NE, zlast.begin() + (size_t)t * NE);
    if ((active || feed) && c.cache.wants_rt()) c.cache.router_row(c.tok_clock, t, zz.data());  // a keep-score's router features
  }
  static thread_local std::vector<float> key;
  key.assign(zz.begin(), zz.begin() + NE);
  if (single && heavy && !heavy->zlast.empty())  // the head's positives first
    for (int e = 0; e < NE; e++) key[e] += heavy->zlast[(size_t)t * NE + e] > 0.f ? 1e4f : 0.f;
  const int keep = std::min(16, NE);
  std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a < b); });
  for (int j = 0; j < keep; j++) n[j] = idx[j];
  if (tau >= 0) {  // softmax q of the top 16 and their calibration bins
    float mx = zz[idx[0]];
    double zs = 0;
    for (int e = 0; e < NE; e++) zs += std::exp((double)(zz[e] - mx));
    for (int j = 0; j < keep; j++) {
      const float q = (float)(std::exp((double)(zz[idx[j]] - mx)) / zs);
      q16[j] = q;
      b16[j] = std::min(15, std::max(0, (int)std::floor(-std::log2(std::max(q, 1e-12f)) * 1.5f)));
    }
  }
  if (!active) return;
  per_step = tau < 0 ? std::min(m, keep) : std::min(cand, keep);
  for (int j = 0; j < keep; j++) {
    bool take;
    if (tau < 0) {
      take = j < m;
    } else {
      if (j >= cand) break;
      const int b = b16[j];
      const double p0 = std::min(0.95, 8.0 * q16[j]);
      take = (cal_h[j][b] + 2.0 * p0) / (cal_n[j][b] + 2.0) >= tau;
    }
    if (take) name(c, p, t, n[j], deadline);
  }
}

// ------------------------------------------------------------------------------------------------ routers inside a verify
namespace {
struct CPreCtx {
  Routers* P;
  int n;
  const float* const* embs;
  alignas(128) std::atomic<int> next{0};
};
void phase_cpre_x(void* p, int, int) {  // per (position, target): [emb_norm_t(emb_i), prev_t] quantised
  CPreCtx& C = *(CPreCtx*)p;
  Routers& P = *C.P;
  const int H = P.H, tot = C.n * P.nt;
  std::vector<float> v(2 * H);
  for (int k; (k = C.next.fetch_add(1, std::memory_order_relaxed)) < tot;) {
    const int i = k / P.nt, t = k % P.nt;
    const float* e = C.embs[i];
    const float r = emb_scale(e, H), *w = P.lay[t].emb_norm;
    for (int d = 0; d < H; d++) v[d] = e[d] * r * w[d];
    std::memcpy(v.data() + H, P.prev.data() + (size_t)t * H, sizeof(float) * H);
    P.cx[k].prep(v.data(), 2 * H);
  }
}
}  // namespace

void Routers::chunk_start(Core& c, int n, const int32_t*, const float* const* embs) {
  cn = 0;
  if (split || n > 8) return;  // (split files: not named inside a verify)
  if (heavy) heavy->chunk_start(c, n, nullptr, embs);
  auto t0 = clk::now();
  cn = n;
  if ((int)cx.size() < n * nt) {
    const size_t o = cx.size();
    cx.resize((size_t)n * nt);
    for (size_t k = o; k < cx.size(); k++) cx[k].init(2 * H, c.o.prec);
  }
  cpre.resize((size_t)n * nt * rows);
  clag.resize((size_t)n * rows);
  cnm.assign((size_t)n * nt * 16, -1);
  cnq.assign((size_t)n * nt * 16, 0.f);
  cbin.assign((size_t)n * nt * 16, 0);
  CPreCtx C;
  C.P = this;
  C.n = n;
  C.embs = embs;
  c.pool->run(phase_cpre_x, &C);
  // every target's first two blocks for every position: one pass over each router's rows (multi-vector)
  std::vector<JobN> jb(nt);
  std::vector<const XQ*> xq((size_t)n * nt);
  std::vector<float*> out((size_t)n * nt);
  for (int t = 0; t < nt; t++)
    for (int i = 0; i < n; i++) {
      xq[(size_t)t * n + i] = &cx[(size_t)i * nt + t].q;
      out[(size_t)t * n + i] = cpre.data() + ((size_t)i * nt + t) * rows;
    }
  for (int t = 0; t < nt; t++) jb[t] = {&lay[t].pre, n, xq.data() + (size_t)t * n, nullptr, out.data() + (size_t)t * n};
  run_rows_n(*c.pool, jb.data(), nt, 32);
  // targets < lag: named now, from the token ids (their embeddings) and the committed token's states
  for (int t = 0; t < std::min(lag, nt); t++)
    for (int i = 0; i < n; i++) {
      const size_t o = ((size_t)i * nt + t);
      finish_z(c, t, cpre.data() + o * rows, nullptr, c.pos + i, c.now(t), cnm.data() + o * 16, cnq.data() + o * 16, cbin.data() + o * 16);
    }
  t_ms += ms_since(t0);
}

void Routers::chunk_rows(Core& c, int step, int n, const XQ* const* xq, const float* const* x, std::vector<JobN>& jobs) {
  pending = cn == n && step + lag < nt ? step + lag : -1;
  if (pending < 0) return;
  clag_p.resize(n);
  for (int i = 0; i < n; i++) clag_p[i] = clag.data() + (size_t)i * rows;
  jobs.push_back({&lay[pending].lagm, n, xq, x, clag_p.data()});
}

void Routers::chunk_after_router(Core& c, int step, int n) {
  if (pending < 0 || cn != n) return;
  auto t0 = clk::now();
  const int t = pending;
  for (int i = 0; i < n; i++) {
    const size_t o = ((size_t)i * nt + t);
    finish_z(c, t, cpre.data() + o * rows, clag.data() + (size_t)i * rows, c.pos + i, c.now(t), cnm.data() + o * 16, cnq.data() + o * 16,
             cbin.data() + o * 16);
  }
  pending = -1;
  t_ms += ms_since(t0);
}

void Routers::chunk_after_moe(Core& c, int step, int n, const int32_t* pk) {
  if (cn != n) return;
  const int K = c.K, ms[3] = {8, 12, 16};
  for (int i = 0; i < n && i < 8; i++) {
    const int* nn = cnm.data() + ((size_t)i * nt + step) * 16;
    if (nn[0] < 0) continue;
    const int32_t* p = pk + (size_t)i * K;
    for (int a = 0; a < 3; a++)
      for (int j = 0; j < K; j++) ccnt[i][a] += std::find(nn, nn + ms[a], p[j]) != nn + ms[a];
    ccalls[i] += 1;
    if (tau >= 0) {
      const int* bb = cbin.data() + ((size_t)i * nt + step) * 16;
      for (int j = 0; j < 16; j++) {
        cal_n[j][bb[j]] += 1;
        cal_h[j][bb[j]] += std::find(p, p + K, nn[j]) != p + K;
      }
    }
  }
}

void Routers::token_start(Core& c, int tok, const float* e) {
  if (heavy) heavy->token_start(c, tok, e);  // the head's targets < lag first (this router's naming reads them)
  auto t0 = clk::now();
  emb = e;
  std::fill(nm.begin(), nm.end(), -1);
  PreCtx C;
  C.P = this;
  if (!split) {
    C.need_e = false;
    c.pool->run(phase_pre_x, &C);
    c.pool->run(phase_pre_rows, &C);
  } else {
    const float* ec = e_lookup(tok);
    e_calls += 1;
    e_hits += ec != nullptr;
    C.need_e = ec == nullptr;
    c.pool->run(phase_split_x, &C);
    c.pool->run(phase_split_rows, &C);
    const size_t n = (size_t)nt * rows;
    if (!ec) {
      ec = e_out.data();
      if (ecap > 0) {  // keep it: the least recently used entry goes
        int sl = 0;
        for (int i = 1; i < ecap; i++)
          if (e_age[i] < e_age[sl]) sl = i;
        if (e_tok[sl] >= 0) e_map.erase(e_tok[sl]);
        std::memcpy(e_pool.data() + (size_t)sl * n, e_out.data(), n * sizeof(float));
        e_tok[sl] = tok;
        e_age[sl] = ++e_clock;
        e_map[tok] = sl;
      }
    }
    for (size_t i = 0; i < n; i++) pre_out[i] += ec[i];
  }
  for (int t = 0; t < std::min(lag, nt); t++) finish(c, t, false);
  t_ms += ms_since(t0);
  fill_bg(c, 0, std::min(lag, nt) - 1);
}

void Routers::fill_bg(Core& c, int from, int to) {
  // the ranks past the naming rule's: past the top m, or (the confidence rule) every one of the top bg_ranks the rule left
  // unnamed (a record held already, named or not, is skipped below)
  const int r0 = tau < 0 ? m : 0;
  if (bg_ranks <= r0 || !active) return;
  Readers& rd = c.cache.rd;
  const int per = std::max(1, c.cache.bg_split);
  if (rd.bg_queued() >= bg_queue * per || !rd.bg_open()) return;
  const int D = std::min(bg_ranks, 16);
  to = std::min(to, nt - 1);
  int tries = 0;
  for (int t = std::max(0, from); t <= to; t++) {
    const int* n = nm.data() + (size_t)t * 16;
    if (n[0] < 0) continue;  // not named yet
    const int L = c.mi.step_layer[t];
    for (int r = r0; r < D; r++) {
      if (c.cache.slot_of(L, n[r]) >= 0) continue;
      const double v = (rk_h[r] + 1.0) / (rk_n[r] + 2.0);
      if (++tries > 32) return;
      if (c.cache.read_bg_value(L, n[r], v - bg_margin, c.now(t)) && rd.bg_queued() >= bg_queue * per) return;
    }
  }
}

void Routers::before_moe(Core& c, int step, const float* x) {
  if (heavy) heavy->before_moe(c, step, x);
  std::memcpy(prev.data() + (size_t)step * H, x, sizeof(float) * H);  // for the next token (this token's pre rows are done)
  fill_bg(c, step + 1, step + lag - 1);
}

void Routers::observe(Core& c, int step, const float* x) {
  if (heavy) heavy->observe(c, step, x);
  std::memcpy(prev.data() + (size_t)step * H, x, sizeof(float) * H);
}

void Routers::router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) {
  if (heavy) heavy->router_rows(c, step, x, jobs);
  pending = step + lag < nt ? step + lag : -1;
  if (pending < 0) return;
  jobs.push_back({&lay[pending].lagm, &c.xin.q, x, lag_out.data()});
}

void Routers::after_router(Core& c, int step) {
  if (heavy) heavy->after_router(c, step);  // the head's logits for the same target, before this router names
  if (pending < 0) return;
  auto t0 = clk::now();
  finish(c, pending, true);
  pending = -1;
  t_ms += ms_since(t0);
  fill_bg(c, step + 1, step + lag);
}

void Routers::after_moe(Core& c, int step, const int32_t* pk, const float* w) {
  const int* n = nm.data() + (size_t)step * 16;
  if (n[0] < 0) return;
  const int K = c.K, g = step < lag ? 0 : 1;
  for (int j = 0; j < 16; j++) {  // P(a pick | rank): the background reads' values
    rk_n[j] += 1;
    rk_h[j] += std::find(pk, pk + K, n[j]) != pk + K;
  }
  fill_bg(c, step + 1, step + lag);
  const int ms[3] = {8, 12, 16};
  for (int a = 0; a < 3; a++) {
    int hit = 0;
    double mass = 0, tot = 0;
    for (int j = 0; j < K; j++) {
      bool in = std::find(n, n + ms[a], pk[j]) != n + ms[a];
      hit += in;
      mass += in ? w[j] : 0.0;
      tot += w[j];
    }
    cnt[g][a] += hit;
    msum[g][a] += mass / std::max(tot, 1e-30);
  }
  calls[g] += 1;
  if (tau >= 0) {
    const int* bb = nbin.data() + (size_t)step * 16;
    for (int j = 0; j < 16; j++) {
      cal_n[j][bb[j]] += 1;
      cal_h[j][bb[j]] += std::find(pk, pk + K, n[j]) != pk + K;
    }
  }
}

void Routers::reset_stats() {
  std::memset(cnt, 0, sizeof cnt);
  std::memset(msum, 0, sizeof msum);
  calls[0] = calls[1] = 0;
  t_ms = e_hits = e_calls = 0;
  std::memset(ccnt, 0, sizeof ccnt);
  std::memset(ccalls, 0, sizeof ccalls);
  lead_ms[0] = lead_ms[1] = lead_n[0] = lead_n[1] = 0;
}

std::string Routers::stats_json(Core& c, int ntok) {
  std::ostringstream o;
  const int K = c.K;
  auto grp = [&](const double* cn, const double* ms, double n) {
    std::ostringstream s;
    n = std::max(1.0, n);
    s << "{\"steps\":" << n << ",\"top_k_in_named8\":" << cn[0] / (n * K) << ",\"top_k_in_named12\":" << cn[1] / (n * K)
      << ",\"top_k_in_named16\":" << cn[2] / (n * K) << ",\"weight_in_named8\":" << ms[0] / n << ",\"weight_in_named12\":" << ms[1] / n
      << ",\"weight_in_named16\":" << ms[2] / n << "}";
    return s.str();
  };
  double ca[3], ma[3];
  for (int a = 0; a < 3; a++) {
    ca[a] = cnt[0][a] + cnt[1][a];
    ma[a] = msum[0][a] + msum[1][a];
  }
  o << "{\"lag\":" << lag << ",\"m\":" << m << ",\"tau\":" << tau << ",\"bits\":" << bits << ",\"split\":" << (split ? "true" : "false")
    << ",\"ecache\":" << ecap << ",\"ecache_hit_rate\":" << e_hits / std::max(1.0, e_calls) << ",\"active\":" << (active ? "true" : "false")
    << ",\"router_ms_per_token\":" << t_ms / std::max(1, ntok) << ",\"router_weight_mb_per_token\":" << bytes_tok / 1e6
    << ",\"first\":" << grp(cnt[0], msum[0], calls[0]) << ",\"lagged\":" << grp(cnt[1], msum[1], calls[1])
    << ",\"all\":" << grp(ca, ma, calls[0] + calls[1]) << ",\"lead\":" << lead_json() << ",\"verify_top_k_in_named8_by_position\":[";
  for (int i = 0; i < 8; i++) o << (i ? "," : "") << ccnt[i][0] / std::max(1.0, ccalls[i] * K);
  o << "],\"verify_top_k_in_named16_by_position\":[";
  for (int i = 0; i < 8; i++) o << (i ? "," : "") << ccnt[i][2] / std::max(1.0, ccalls[i] * K);
  o << "],\"verify_steps_by_position\":[";
  for (int i = 0; i < 8; i++) o << (i ? "," : "") << ccalls[i];
  o << "]}";
  return o.str();
}

// ------------------------------------------------------------------------------------------------ cross-token routers
bool Xtok::load(const std::string& path, Core& c, std::string& err) {
  if (!R.load(path, Mem::MLOCK, err)) return false;
  if (R.gets("kind") != "ink_xtok") {
    err = path + ": not a cross-token router file";
    return false;
  }
  src = (int)R.geti("source_app");
  H = (int)R.geti("hidden");
  NE = (int)R.geti("experts");
  hid = (int)R.geti("hidden_mlp");
  if (H != c.H || NE != c.NE || (int)R.geti("apps") != c.steps) {
    err = "cross-token router file does not match the model";
    return false;
  }
  rows = NE + hid;
  const std::string t = R.gets("targets");
  for (size_t i = 0; i < t.size();) {
    size_t j = t.find(',', i);
    if (j == std::string::npos) j = t.size();
    tg.push_back(std::stoi(t.substr(i, j - i)));
    i = j + 1;
  }
  slot_of.assign(c.steps, -1);
  for (size_t i = 0; i < tg.size(); i++) {
    if (tg[i] >= src || tg[i] < 0) {
      err = "a cross-token target must come before the source step";
      return false;
    }
    slot_of[tg[i]] = (int)i;
    const std::string s = std::to_string(tg[i]) + ".";
    Lay y;
    y.pre = R.weight(s + "pre");
    y.b_pre = R.f32(s + "b_pre");
    y.m2 = R.weight(s + "m2");
    y.m2_b = R.f32(s + "m2_b");
    if (y.pre.rows != rows || y.pre.cols != 2 * H) {
      err = "cross-token router " + s + " has an unexpected shape";
      return false;
    }
    lay.push_back(y);
  }
  const int n = (int)tg.size();
  hin.assign((size_t)n * 2 * H, 0.f);
  xsrc.assign(H, 0.f);
  out.assign((size_t)n * rows, 0.f);
  xin.resize(n);
  for (auto& b : xin) b.init(2 * H, c.o.prec);
  xh.init(hid, c.o.prec);
  nm_next.assign((size_t)n * 16, -1);
  nm_cur = nm_next;
  bin_next.assign((size_t)n * 16, 0);
  bin_cur = bin_next;
  return true;
}

void Xtok::token_start(Core& c, int tok, const float* emb) {
  ensure(c.steps);
  std::swap(nm_cur, nm_next);
  std::swap(bin_cur, bin_next);
  std::fill(nm_next.begin(), nm_next.end(), -1);
  if (inner) inner->token_start(c, tok, emb);
}

void Xtok::observe(Core& c, int step, const float* x) {
  if (inner) inner->observe(c, step, x);  // its own inputs are this token's, set again by before_moe before they are read
}

void Xtok::before_moe(Core& c, int step, const float* x) {
  if (inner) inner->before_moe(c, step, x);
  if (slot_of[step] >= 0) std::memcpy(hin.data() + (size_t)slot_of[step] * 2 * H + H, x, sizeof(float) * H);
  if (step == src) std::memcpy(xsrc.data(), x, sizeof(float) * H);
}

void Xtok::after_router(Core& c, int step) {
  if (inner) inner->after_router(c, step);
  if (step != src) return;
  auto t0 = clk::now();
  const int n = (int)tg.size();
  std::vector<Job> jb(n);
  for (int i = 0; i < n; i++) {
    float* in = hin.data() + (size_t)i * 2 * H;
    std::memcpy(in, xsrc.data(), sizeof(float) * H);
    xin[i].prep(in, 2 * H);
    jb[i] = {&lay[i].pre, &xin[i].q, in, out.data() + (size_t)i * rows};
  }
  run_rows(*c.pool, jb.data(), n, 32, false, nullptr);  // router bytes are not the model's
  std::vector<float> hv(hid), z2(NE), zz(NE);
  std::vector<int> idx(NE);
  for (int i = 0; i < n; i++) {
    const Lay& y = lay[i];
    const float* o = out.data() + (size_t)i * rows;
    for (int k = 0; k < hid; k++) hv[k] = gelu_erf(o[NE + k] + y.b_pre[NE + k]);
    xh.prep(hv.data(), hid);
    mv(y.m2, 0, NE, &xh.q, hv.data(), z2.data());
    float mx = -INFINITY;
    for (int e = 0; e < NE; e++) {
      zz[e] = o[e] + y.b_pre[e] + z2[e] + y.m2_b[e];
      idx[e] = e;
      mx = std::max(mx, zz[e]);
    }
    double zs = 0;
    for (int e = 0; e < NE; e++) zs += std::exp((double)(zz[e] - mx));
    std::partial_sort(idx.begin(), idx.begin() + 16, idx.end(), [&](int a, int b) { return zz[a] > zz[b] || (zz[a] == zz[b] && a < b); });
    int* nm = nm_next.data() + (size_t)i * 16;
    int* bb = bin_next.data() + (size_t)i * 16;
    for (int j = 0; j < 16; j++) {
      nm[j] = idx[j];
      const float q = (float)(std::exp((double)(zz[idx[j]] - mx)) / zs);
      bb[j] = std::min(15, std::max(0, (int)std::floor(-std::log2(std::max(q, 1e-12f)) * 1.5f)));
      if (j >= cand) continue;
      const double p0 = std::min(0.95, 8.0 * q);
      const double p = (cal_h[j][bb[j]] + 2.0 * p0) / (cal_n[j][bb[j]] + 2.0);
      if (p >= tau) {
        c.cache.read_bg = background;
        name(c, c.pos + 1, tg[i], idx[j]);
        c.cache.read_bg = false;
        named++;
      }
    }
  }
  t_ms += ms_since(t0);
}

void Xtok::after_moe(Core& c, int step, const int32_t* picks, const float* w) {
  if (inner) inner->after_moe(c, step, picks, w);
  const int i = slot_of[step];
  if (i < 0) return;
  const int* nm = nm_cur.data() + (size_t)i * 16;
  if (nm[0] < 0) return;
  const int K = c.K;
  for (int a = 0; a < 2; a++) {
    const int m = a ? 16 : 8;
    for (int j = 0; j < K; j++) cnt[a] += std::find(nm, nm + m, picks[j]) != nm + m;
  }
  calls++;
  const int* bb = bin_cur.data() + (size_t)i * 16;
  for (int j = 0; j < 16; j++) {
    cal_n[j][bb[j]] += 1;
    cal_h[j][bb[j]] += std::find(picks, picks + K, nm[j]) != picks + K;
  }
}

void Xtok::release(Core& c, int step) {
  Naming::release(c, step);
  if (inner) inner->release(c, step);
}

void Xtok::reset_stats() {
  cnt[0] = cnt[1] = calls = t_ms = named = 0;
  if (inner) inner->reset_stats();
}

std::string Xtok::stats_json(Core& c, int ntok) {
  std::ostringstream o;
  const double n = std::max(1.0, calls) * c.K;
  o << "{\"source_step\":" << src << ",\"targets\":" << tg.size() << ",\"tau\":" << tau << ",\"top_k_in_named8\":" << cnt[0] / n
    << ",\"top_k_in_named16\":" << cnt[1] / n << ",\"named_per_token\":" << named / std::max(1, ntok) << ",\"ms_per_token\":" << t_ms / std::max(1, ntok)
    << ",\"inner\":" << (inner ? inner->stats_json(c, ntok) : std::string("{}")) << "}";
  return o.str();
}

}  // namespace dm
