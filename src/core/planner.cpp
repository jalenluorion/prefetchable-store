// The read planner: a cascade of trained router sets, online calibration, a read budget from the law, revocation.
#include <algorithm>
#include <atomic>
#include <cstring>
#include <sstream>

#include "dmeng/address.h"

namespace dm {

bool Planner::load(const std::vector<std::string>& paths, Core& c, std::string& err) {
  NS = c.steps;
  NE = c.NE;
  H = c.H;
  for (auto& path : paths) {
    std::unique_ptr<Net> np(new Net());
    Net& n = *np;
    if (!n.R.load(path, Mem::MLOCK, err)) return false;
    n.lag = (int)n.R.geti("lag");
    n.hid = (int)n.R.geti("hidden_mlp");
    const int nt = (int)(n.R.has_cfg("apps") ? n.R.geti("apps") : n.R.geti("layers"));
    if (nt != NS || (int)n.R.geti("hidden") != H) {
      err = path + ": router file does not match the model";
      return false;
    }
    n.rows = NE + n.hid;
    n.lay.resize(NS);
    for (int T = 0; T < NS; T++) {
      const std::string s = std::to_string(T) + ".";
      Routers::Lay& y = n.lay[T];
      y.emb_norm = (const float*)n.R.ptr(s + "emb_norm");
      y.pre = n.R.weight(s + "pre");
      y.has_lag = n.R.has(s + "lagm.q") || n.R.has(s + "lagm.q4");
      if (y.has_lag) y.lagm = n.R.weight(s + "lagm");
      y.b_pre = (const float*)n.R.ptr(s + "b_pre");
      if (n.hid > 0) {
        y.m2 = n.R.weight(s + "m2");
        y.m2_b = (const float*)n.R.ptr(s + "m2_b");
      }
      if (y.pre.rows != n.rows || y.pre.cols != 2 * H || (T >= n.lag) != y.has_lag) {
        err = path + ": router " + std::to_string(T) + " has an unexpected shape";
        return false;
      }
    }
    n.pre_out.assign((size_t)NS * n.rows, 0.f);
    n.lag_out.assign(n.rows, 0.f);
    n.q.assign((size_t)NS * NE, 0.f);
    n.rk.assign((size_t)NS * NE, 255);
    n.qg.assign(NS, -1);
    n.xpre.resize(NS);
    for (auto& b : n.xpre) b.init(2 * H, c.o.prec);
    if (n.hid > 0) n.xh.init(n.hid, c.o.prec);
    nets.push_back(std::move(np));
  }
  if (nets.empty() || nets.size() > 4) {
    err = "planner: 1 to 4 router sets";
    return false;
  }
  std::stable_sort(nets.begin(), nets.end(), [](const std::unique_ptr<Net>& a, const std::unique_ptr<Net>& b) { return a->lag > b->lag; });
  for (size_t i = 0; i < nets.size(); i++) nets[i]->id = (int)i;
  maxlag = nets[0]->lag;
  prev.assign((size_t)NS * H, 0.f);
  const int G = 2 * (int)nets.size();
  c_n.assign((size_t)G * NB, 0.0);
  c_pos.assign((size_t)G * NB, 0.0);
  c_tab.assign((size_t)G * NB, 0.f);
  refresh_calib();
  pins.resize(NS);
  state.assign((size_t)NS * NE, 0);
  iss_slot.assign((size_t)NS * NE, -1);
  return true;
}

void Planner::refresh_calib() {
  // per cell: rate with a weak prior p0 = min(0.95, 6 q); within each rank bin made non-decreasing in q (pool adjacent
  // violators)
  const int G = 2 * (int)nets.size();
  for (int g = 0; g < G; g++)
    for (int rb = 0; rb < NR; rb++) {
      double r[NQ], w[NQ];
      for (int b = 0; b < NQ; b++) {
        double qm = std::exp2(b * 1.5 + 0.75 - 18.0);
        double p0 = std::min(0.95, 6.0 * qm);
        const size_t cc = (size_t)g * NB + rb * NQ + b;
        w[b] = c_n[cc] + 3.0;
        r[b] = (c_pos[cc] + 3.0 * p0) / w[b];
      }
      int lo[NQ], hi[NQ], nb = 0;
      double rv[NQ], wv[NQ];
      for (int b = 0; b < NQ; b++) {
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
        for (int b = lo[k]; b <= hi[k]; b++) c_tab[(size_t)g * NB + rb * NQ + b] = (float)rv[k];
    }
}

void Planner::controller(Core& c) {
  // measured over the last token: compute time (wall minus wait), the drive's rate while busy, the on-the-spot bytes
  auto& C = c.cache;
  const auto now = clk::now();
  const uint64_t bu = C.st.bytes_urgent, bp = C.st.bytes_pf, nm = C.st.n_miss, rdb = C.rd.bytes.load(), rdw = C.rd.busy_wall(),
                 rdbusy = C.rd.busy_ns.load();
  const double wm = C.st.wait_ms_miss;
  if (!first) {
    double dt = std::chrono::duration<double, std::milli>(now - t_last).count();
    double w = c.t.wait >= wait_last ? c.t.wait - wait_last : c.t.wait;  // the token loop zeroes the timers after prefill
    double ub = bu >= bu_last ? (double)(bu - bu_last) : (double)bu;
    double comp = std::max(1.0, dt - w);
    comp_ms = 0.8 * comp_ms + 0.2 * comp;
    urg_b = 0.8 * urg_b + 0.2 * ub;
    double pb = bp >= bp_last ? (double)(bp - bp_last) : (double)bp;
    tot_b = 0.8 * tot_b + 0.2 * (ub + pb);
    double dm = nm >= nm_last ? (double)(nm - nm_last) : (double)nm, dwm = wm >= wm_last ? wm - wm_last : wm;
    if (dm >= 4) t_miss = 0.8 * t_miss + 0.2 * std::max(0.05, dwm / dm);
    const double db = (double)(rdb - rdb_last), dw = (double)(rdw - rdw_last), dbusy = (double)(rdbusy - rdbusy_last);
    if (dw > 2e6 && db > 0) {
      double inst = db / (dw * 1e-9);
      if (dbusy / dw >= 2.0) bw = 0.8 * bw + 0.2 * inst;  // the drive's rate at depth: only from well-filled tokens
      else bw = std::max(bw, inst);
    }
  }
  first = false;
  t_last = now;
  wait_last = c.t.wait;
  bu_last = bu;
  rdb_last = rdb;
  rdw_last = rdw;
  rdbusy_last = rdbusy;
  bp_last = bp;
  nm_last = nm;
  wm_last = wm;
  allow = std::max(0.0, bw * comp_ms * 1e-3 * u - urg_b);
  // the law online: a speculative read costs its drive time (1 - p) of the time it is wrong, weighted by how loaded
  // the drive is (bytes per token over what it moves in a token's compute); it saves p x the stall of a miss
  const double t_rec = (double)C.P[0].rec / bw * 1e3;
  rho = tot_b / std::max(1.0, bw * comp_ms * 1e-3 * u);
  thr_law = rho * t_rec / (t_miss + rho * t_rec);
  sum_thr += thr_law;
  sum_rho += rho;
  sum_tmiss += t_miss;
  ntok += 1;
  sum_allow += allow;
  sum_bw += bw;
  sum_comp += comp_ms;
  sum_urg += urg_b;
}

namespace {
struct PlanPreCtx {
  Planner* P;
  int nx, nr;
  int roff[5];  // chunk offsets per net
  alignas(128) std::atomic<int> next_x{0};
  alignas(128) std::atomic<int> next_r{0};
};
void plan_pre_x(void* p, int, int) {
  PlanPreCtx& C = *(PlanPreCtx*)p;
  Planner& P = *C.P;
  const int H = P.H;
  std::vector<float> v(2 * H);
  const float* e = P.emb;
  double ss = 0;
  for (int i = 0; i < H; i++) ss += (double)e[i] * e[i];
  const float r = 1.f / std::sqrt((float)(ss / H) + 1.1920929e-07f);
  for (int t; (t = C.next_x.fetch_add(1, std::memory_order_relaxed)) < C.nx;) {
    Planner::Net& n = *P.nets[t / P.NS];
    const int T = t % P.NS;
    const float* w = n.lay[T].emb_norm;
    for (int i = 0; i < H; i++) v[i] = e[i] * r * w[i];
    std::memcpy(v.data() + H, P.prev.data() + (size_t)T * H, sizeof(float) * H);
    n.xpre[T].prep(v.data(), 2 * H);
  }
}
void plan_pre_rows(void* p, int, int) {
  PlanPreCtx& C = *(PlanPreCtx*)p;
  Planner& P = *C.P;
  for (int ch; (ch = C.next_r.fetch_add(1, std::memory_order_relaxed)) < C.nr;) {
    int k = 0;
    while (ch >= C.roff[k + 1]) k++;
    Planner::Net& n = *P.nets[k];
    const int per = (n.rows + 31) / 32, c2 = ch - C.roff[k];
    const int T = c2 / per, r0 = (c2 % per) * 32, r1 = std::min(n.rows, r0 + 32);
    mv(n.lay[T].pre, r0, r1, &n.xpre[T].q, nullptr, n.pre_out.data() + (size_t)T * n.rows + r0);
  }
}
}  // namespace

void Planner::stage(Core& c, Net& n, int T, bool with_lag) {
  Routers::Lay& y = n.lay[T];
  const int rows = n.rows, hid = n.hid, L = c.mi.step_layer[T];
  static thread_local std::vector<float> z, z2, hv, pv;
  static thread_local std::vector<int> ord, idx;
  z.resize(rows);
  z2.resize(NE);
  hv.resize(std::max(1, hid));
  pv.resize(NE);
  ord.resize(NE);
  idx.resize(NE);
  const float* pre = n.pre_out.data() + (size_t)T * rows;
  for (int r = 0; r < rows; r++) z[r] = pre[r] + y.b_pre[r] + (with_lag ? n.lag_out[r] : 0.f);
  if (hid > 0) {
    for (int i = 0; i < hid; i++) hv[i] = gelu_erf(z[NE + i]);
    n.xh.prep(hv.data(), hid);
    mv(y.m2, 0, NE, &n.xh.q, nullptr, z2.data());
  }
  float* q = n.q.data() + (size_t)T * NE;
  float mx = -INFINITY;
  for (int e = 0; e < NE; e++) {
    q[e] = z[e] + (hid > 0 ? z2[e] + y.m2_b[e] : 0.f);
    mx = std::max(mx, q[e]);
  }
  double zs = 0;
  for (int e = 0; e < NE; e++) zs += (q[e] = std::exp(q[e] - mx));
  const float inv = (float)(1.0 / zs);
  for (int e = 0; e < NE; e++) q[e] *= inv;
  const int g = n.id * 2 + (with_lag ? 1 : 0);
  n.qg[T] = g;
  for (int e = 0; e < NE; e++) ord[e] = e;
  const int top = std::min(32, NE);
  std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), [&](int a, int b) { return q[a] > q[b] || (q[a] == q[b] && a < b); });
  uint8_t* rk = n.rk.data() + (size_t)T * NE;
  for (int e = 0; e < NE; e++) rk[e] = 255;
  for (int r = 0; r < top; r++) rk[ord[r]] = (uint8_t)r;
  const float* tab = c_tab.data() + (size_t)g * NB;
  float* pred = c.cache.pred.data() + (size_t)L * NE;
  for (int e = 0; e < NE; e++) pred[e] = pv[e] = tab[cell(rk[e], q[e])];  // the latest (shortest-lag) estimate wins
  if (!active) return;
  auto& C = c.cache;
  const size_t base = (size_t)T * NE;
  if (revoke_on)
    for (int e = 0; e < NE; e++) {
      if (state[base + e] != 2 || pv[e] >= revoke_p) continue;
      const int sl = iss_slot[base + e];
      auto& pl = pins[T];
      auto it = std::find(pl.begin(), pl.end(), sl);
      if (it == pl.end()) continue;
      C.unpin(L, sl);
      if (C.cancel(L, sl)) {
        pl.erase(it);
        state[base + e] = 0;
        iss_slot[base + e] = -1;
        credit += (double)C.pool(L).rec;
        n.revoked++;
      } else {
        C.slot(L, sl).pins++;  // already being read: keep it
      }
    }
  for (int e = 0; e < NE; e++) idx[e] = e;
  const int keep = std::min(maxcand, NE);
  std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(), [&](int a, int b) { return pv[a] > pv[b] || (pv[a] == pv[b] && a < b); });
  const bool longest = nets.size() > 1 && n.id == 0;
  double thr = tau_fixed >= 0 ? tau_fixed : 0.0;
  if (budget_on && budget_mode == 1) thr = std::max(thr, thr_law);
  if (longest) thr = std::max(thr, tau_long);
  const bool bucket = budget_on && budget_mode == 2;
  const long pstep = c.next_abs(T);
  const double rec = (double)C.pool(L).rec;
  for (int j = 0; j < keep; j++) {
    const int e = idx[j];
    const float pe = pv[e];
    if (pe < 0.005f) break;
    if (state[base + e]) continue;
    const int sl = C.slot_of(L, e);
    if (sl >= 0) {
      if (pe >= pin_p) {
        C.slot(L, sl).pins++;
        pins[T].push_back(sl);
        state[base + e] = 1;
        n.pinned_cached++;
      }
      continue;
    }
    if (pe < thr) {
      n.skip_tau++;
      continue;
    }
    if (bucket && credit < rec) {
      n.skip_budget++;
      break;
    }
    int v;
    C.evict_pf_ctx = true;
    if (C.pred_evict) {
      double val = 0;
      v = C.pick_victim(L, 2 * c.K, &val);
      if (v >= 0 && val >= pe + C.use_tab[0]) {  // a fresh slot's value: p now + P(used later | used once)
        C.evict_pf_ctx = false;
        n.skip_evict++;
        continue;
      }
      if (v >= 0) C.evict(L, v);
    } else {
      v = C.victim(L, 2 * c.K);
    }
    C.evict_pf_ctx = false;
    if (v < 0) {
      C.st.n_pf_dropped++;
      break;
    }
    C.issue(L, v, e, false, pstep, pe);
    C.slot(L, v).pins = 1;
    C.slot(L, v).pf_unused = true;
    C.st.n_pf_issued++;
    pins[T].push_back(v);
    state[base + e] = 2;
    iss_slot[base + e] = v;
    if (bucket) credit -= rec;
    n.issued++;
  }
}

void Planner::token_start(Core& c, int, const float* e) {
  auto t0 = clk::now();
  emb = e;
  controller(c);
  refresh_calib();
  if (c.cache.learn_on) c.cache.refresh_use_tab();
  PlanPreCtx C;
  C.P = this;
  C.nx = (int)nets.size() * NS;
  C.roff[0] = 0;
  for (size_t k = 0; k < nets.size(); k++) C.roff[k + 1] = C.roff[k] + ((nets[k]->rows + 31) / 32) * NS;
  C.nr = C.roff[nets.size()];
  c.pool->run(plan_pre_x, &C);
  c.pool->run(plan_pre_rows, &C);
  for (int T = 0; T < std::min(maxlag, NS); T++)
    for (auto& n : nets)
      if (T < n->lag) stage(c, *n, T, false);
  t_ms += ms_since(t0);
}

void Planner::before_moe(Core&, int step, const float* x) { std::memcpy(prev.data() + (size_t)step * H, x, sizeof(float) * H); }
void Planner::observe(Core&, int step, const float* x) { std::memcpy(prev.data() + (size_t)step * H, x, sizeof(float) * H); }

void Planner::router_rows(Core& c, int step, const float* x, std::vector<Job>& jobs) {
  for (auto& n : nets) {
    n->pending = step + n->lag < NS ? step + n->lag : -1;
    if (n->pending < 0) continue;
    jobs.push_back({&n->lay[n->pending].lagm, &c.xin.q, x, n->lag_out.data()});
  }
}

void Planner::after_router(Core& c, int) {
  auto t0 = clk::now();
  if (budget_on && budget_mode == 2) credit = std::min(credit + allow / NS, allow * (maxlag + 1) / NS + (double)c.cache.P[0].rec);
  for (int k = (int)nets.size() - 1; k >= 0; k--) {  // the nearest target (shortest lag) first
    Net& n = *nets[k];
    if (n.pending < 0) continue;
    stage(c, n, n.pending, true);
    n.pending = -1;
  }
  t_ms += ms_since(t0);
}

void Planner::after_moe(Core& c, int step, const int32_t* pk, const float*) {
  static thread_local std::vector<char> in;
  in.assign(NE, 0);
  for (int j = 0; j < c.K; j++) in[pk[j]] = 1;
  for (auto& n : nets) {
    const int g = n->qg[step];
    if (g < 0) continue;
    const float* q = n->q.data() + (size_t)step * NE;
    const uint8_t* rk = n->rk.data() + (size_t)step * NE;
    const float* tab = c_tab.data() + (size_t)g * NB;
    double sp = 0, spk = 0;
    for (int e = 0; e < NE; e++) {
      const int b = cell(rk[e], q[e]);
      c_n[(size_t)g * NB + b] += 1;
      c_pos[(size_t)g * NB + b] += in[e];
      sp += tab[b];
      spk += in[e] ? tab[b] : 0.f;
    }
    if (stats_on && g < 8) {
      cal_sum_p[g] += sp;
      cal_pick_p[g] += spk;
      cal_calls[g] += 1;
    }
    n->qg[step] = -1;
  }
}

void Planner::release(Core& c, int step) {
  const int L = c.mi.step_layer[step];
  for (int sl : pins[step]) c.cache.unpin(L, sl);
  pins[step].clear();
  const size_t base = (size_t)step * NE;
  std::memset(state.data() + base, 0, NE);
  for (int e = 0; e < NE; e++) {
    iss_slot[base + e] = -1;
    c.cache.pred[(size_t)L * NE + e] = -1.f;
  }
}

void Planner::reset_stats() {
  t_ms = ntok = sum_allow = sum_bw = sum_comp = sum_urg = sum_thr = sum_rho = sum_tmiss = 0;
  for (int g = 0; g < 8; g++) cal_sum_p[g] = cal_pick_p[g] = cal_calls[g] = 0;
  for (auto& n : nets) n->issued = n->pinned_cached = n->revoked = n->skip_budget = n->skip_evict = n->skip_tau = 0;
  stats_on = true;
}

std::string Planner::stats_json(Core&, int ntk) {
  std::ostringstream o;
  const double nt = std::max(1, ntk), tk = std::max(1.0, ntok);
  o << "{\"router_ms_per_token\":" << t_ms / nt << ",\"u\":" << u << ",\"budget\":" << (budget_on ? "true" : "false")
    << ",\"tau_long\":" << tau_long << ",\"tau_fixed\":" << tau_fixed << ",\"pin_p\":" << pin_p << ",\"revoke\":" << (revoke_on ? "true" : "false")
    << ",\"revoke_p\":" << revoke_p << ",\"allow_mb_per_token\":" << sum_allow / tk / 1048576.0 << ",\"bw_gb_s\":" << sum_bw / tk / 1e9
    << ",\"comp_ms_est\":" << sum_comp / tk << ",\"urgent_mb_est\":" << sum_urg / tk / 1048576.0 << ",\"budget_mode\":" << budget_mode
    << ",\"thr_law_mean\":" << sum_thr / tk << ",\"drive_load_mean\":" << sum_rho / tk << ",\"t_miss_ms_mean\":" << sum_tmiss / tk << ",\"nets\":[";
  for (size_t k = 0; k < nets.size(); k++) {
    Net& n = *nets[k];
    o << (k ? "," : "") << "{\"lag\":" << n.lag << ",\"issued\":" << n.issued / nt << ",\"pinned_cached\":" << n.pinned_cached / nt
      << ",\"revoked\":" << n.revoked / nt << ",\"skip_budget\":" << n.skip_budget / nt << ",\"skip_evict\":" << n.skip_evict / nt
      << ",\"skip_tau\":" << n.skip_tau / nt;
    for (int lagged = 0; lagged < 2; lagged++) {
      const int g = n.id * 2 + lagged;
      const double cc = std::max(1.0, cal_calls[g]);
      o << ",\"" << (lagged ? "lagged" : "first") << "\":{\"calls\":" << cal_calls[g] << ",\"sum_p\":" << cal_sum_p[g] / cc
        << ",\"sum_p_over_picks\":" << cal_pick_p[g] / cc << "}";
    }
    o << "}";
  }
  o << "]}";
  return o.str();
}

}  // namespace dm
