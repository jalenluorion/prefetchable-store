// The MoE step over a chunk of positions (batched prefill, multi-position verify): every position's released routing and
// combination exactly as the single step (moe.cpp) computes them; the union of the positions' experts pinned and read
// once; each expert computed for all positions that picked it in one pass over its record (multi-vector kernels, packed
// i8mm for NVFP4 from 6 users). Lossless only.
#include <algorithm>
#include <atomic>
#include <cstring>

#include "dmeng/core.h"

namespace dm {

constexpr int PACK_MIN = 6;  // users of an expert from which the packed NVFP4 kernel beats the multi-vector one

struct Core::Chunk {
  int n = 0;
  // per position
  std::vector<XBuf> xin;         // [n] the MoE input, quantised
  std::vector<float> lg, sc, bz;  // [n][NE + 1], [n][NE]
  std::vector<int> ord;           // [NE] scratch
  std::vector<float> mx, shw;
  std::vector<double> zsum;
  std::vector<int32_t> pk;        // [n][K]
  std::vector<float> pw;          // [n][K]
  // per (position, row): K routed rows then the shared expert's
  std::vector<XBuf> act;          // [n * (K + 1)]
  std::vector<float> actf, eout;  // [n * (K + 1)][IA], [n * (K + 1)][H]
  // per entry of the union
  struct Ent {
    int e, slot;
    Weight gu, up, dn;
    int inter;
    std::vector<int> users;  // row index i * (K + 1) + j
    XQB xg, xd;              // packed inputs (NVFP4 with >= PACK_MIN users)
    bool packed = false, done = false;
  };
  std::vector<Ent> ent;
  std::vector<int> ent_of;  // [NE] -> entry, -1
};

namespace {
struct ChCtx {
  Core::Chunk* C;
  const std::vector<int>* run;  // entries of this pass
  int H, IA, R;                 // R = K + 1 rows per position
  const float* const* x;        // [n] the fp32 MoE inputs (the reference path reads them)
  bool ref, softcap_act;
  float gcap, ucap;
  std::vector<int> start;       // gate-up chunks per entry (prefix)
  alignas(128) std::atomic<int> next_gu{0};
  alignas(128) std::atomic<int> next_dn{0};
};
}  // namespace

static void chunk_gu(void* p, int, int) {
  ChCtx& X = *(ChCtx*)p;
  Core::Chunk& C = *X.C;
  const int nr = (int)X.run->size(), total = X.start[nr];
  static thread_local std::vector<float> g, u;
  static thread_local std::vector<const XQ*> xq;
  static thread_local std::vector<const float*> xf;
  static thread_local std::vector<float*> ys, ysu;
  for (int ch; (ch = X.next_gu.fetch_add(1, std::memory_order_relaxed)) < total;) {
    int q = 0;
    while (ch >= X.start[q + 1]) q++;
    Core::Chunk::Ent& E = C.ent[(*X.run)[q]];
    const int j0 = (ch - X.start[q]) * XB, nu = (int)E.users.size();
    const bool fused = E.up.kind == Weight::NONE;
    const int gw = fused ? 2 * XB : XB;
    g.resize((size_t)nu * gw);
    u.resize((size_t)nu * XB);
    xq.resize(nu);
    xf.resize(nu);
    ys.resize(nu);
    ysu.resize(nu);
    for (int k = 0; k < nu; k++) {
      const int pi = E.users[k] / X.R;
      xq[k] = &C.xin[pi].q;
      xf[k] = X.x[pi];
      ys[k] = g.data() + (size_t)k * gw;
      ysu[k] = u.data() + (size_t)k * XB;
    }
    if (E.packed) {
      gemm_nvfp4_i16(E.gu.c, (const uint8_t*)E.gu.s, E.gu.gmul, E.gu.cols, fused ? 2 * j0 : j0, fused ? 2 * j0 + 2 * XB : j0 + XB, E.xg, 0, nu,
                     g.data(), gw);
      if (!fused) gemm_nvfp4_i16(E.up.c, (const uint8_t*)E.up.s, E.up.gmul, E.up.cols, j0, j0 + XB, E.xg, 0, nu, u.data(), XB);
    } else if (fused) {
      mv_multi(E.gu, 2 * j0, 2 * j0 + 2 * XB, xq.data(), xf.data(), nu, ys.data(), X.ref);
    } else {
      mv_multi(E.gu, j0, j0 + XB, xq.data(), xf.data(), nu, ys.data(), X.ref);
      mv_multi(E.up, j0, j0 + XB, xq.data(), xf.data(), nu, ysu.data(), X.ref);
    }
    for (int k = 0; k < nu; k++) {
      const int row = E.users[k];
      float* a = C.actf.data() + (size_t)row * X.IA;
      const float* gg_ = g.data() + (size_t)k * gw;
      const float* uu_ = u.data() + (size_t)k * XB;
      if (fused && X.softcap_act) {  // the four activation loops of the single step (moe.cpp phase_gu), kept apart
        for (int j = 0; j < XB; j++) {
          const float gg = gg_[2 * j], uu = gg_[2 * j + 1];
          a[j0 + j] = softcap(gg, X.gcap) * sigm(gg) * softcap(uu, X.ucap);
        }
      } else if (fused) {
        for (int j = 0; j < XB; j++) a[j0 + j] = silu(gg_[2 * j]) * gg_[2 * j + 1];
      } else if (X.softcap_act) {
        for (int j = 0; j < XB; j++) a[j0 + j] = softcap(gg_[j], X.gcap) * sigm(gg_[j]) * softcap(uu_[j], X.ucap);
      } else {
        for (int j = 0; j < XB; j++) a[j0 + j] = silu(gg_[j]) * uu_[j];
      }
      if (!X.ref) xq_block(C.act[row].q, j0 / XB, a + j0);
    }
  }
}

static void chunk_dn(void* p, int, int) {
  ChCtx& X = *(ChCtx*)p;
  Core::Chunk& C = *X.C;
  constexpr int CH = 32;
  const int nch = (X.H + CH - 1) / CH, nr = (int)X.run->size(), total = nch * nr;
  static thread_local std::vector<const XQ*> xq;
  static thread_local std::vector<const float*> xf;
  static thread_local std::vector<float*> ys;
  static thread_local std::vector<float> tmp;
  for (int ch; (ch = X.next_dn.fetch_add(1, std::memory_order_relaxed)) < total;) {
    Core::Chunk::Ent& E = C.ent[(*X.run)[ch / nch]];
    const int a = (ch % nch) * CH, b = std::min(X.H, a + CH), nu = (int)E.users.size();
    if (E.packed && !(a & 3) && !(b & 3)) {
      tmp.resize((size_t)nu * CH);
      gemm_nvfp4_i16(E.dn.c, (const uint8_t*)E.dn.s, E.dn.gmul, E.dn.cols, a, b, E.xd, 0, nu, tmp.data(), CH);
      for (int k = 0; k < nu; k++)
        std::memcpy(C.eout.data() + (size_t)E.users[k] * X.H + a, tmp.data() + (size_t)k * CH, sizeof(float) * (b - a));
      continue;
    }
    xq.resize(nu);
    xf.resize(nu);
    ys.resize(nu);
    for (int k = 0; k < nu; k++) {
      const int row = E.users[k];
      xq[k] = &C.act[row].q;
      xf[k] = C.actf.data() + (size_t)row * X.IA;
      ys[k] = C.eout.data() + (size_t)row * X.H + a;
    }
    mv_multi(E.dn, a, b, xq.data(), xf.data(), nu, ys.data(), X.ref);
  }
}

void Core::moe_chunk(int step, int n, const float* const* x, float* const* y, const MoeView& v, int32_t* pk_out, float* w_out, int off) {
  // a run-time budget (any thread) at each application of a multi-position pass, as the 35B's prefill does at each
  // layer (a long span is otherwise one boundary); records still pinned go at a later boundary
  if (cache.budget_req.load(std::memory_order_relaxed) >= 0) cache.apply_budget();
  Cache::ChunkClock chunk_clock(cache, tok_clock + off + n - 1);  // the pass is one token to the time-ordered statistics
  if (ns.on || (force && pos < force_n) || n < 1) {
    fprintf(stderr, "moe_chunk: lossless and unforced only\n");
    abort();
  }
  if (!ch_) ch_ = std::make_shared<Chunk>();
  Chunk& C = *ch_;
  const int L = mi.step_layer[step], pass = mi.step_pass[step], R = K + 1;
  const ExpertFormat& F = mi.formats[mi.layer_fmt[L]];
  const int IA = std::max(I_, sh_inter_);
  if (C.n < n) {  // per-position buffers, grown as wider chunks come
    C.xin.resize(n);
    for (int i = C.n; i < n; i++) C.xin[i].init(H, o.prec);
    C.act.resize((size_t)n * R);
    for (size_t i = (size_t)C.n * R; i < (size_t)n * R; i++) C.act[i].init(IA, o.prec);
    C.n = n;
  }
  C.lg.resize((size_t)n * (NE + 1));
  C.sc.resize((size_t)n * NE);
  C.bz.resize((size_t)n * NE);
  C.ord.resize(NE);
  C.mx.resize(n);
  C.shw.resize(n);
  C.zsum.resize(n);
  C.pk.resize((size_t)n * K);
  C.pw.resize((size_t)n * K);
  C.actf.resize((size_t)n * R * IA);
  C.eout.resize((size_t)n * R * H);
  const long now_s = (tok_clock + off) * steps + step;  // = now(step) for a chunk that starts the span
  cache.now_step = now_s;
  step_t0 = clk::now();
  if (cancel_stale) cache.cancel_stale(now_s - 1);
  Hook* nh = chunk_named ? hook : nullptr;  // names inside this verify
  if (hook && n <= 8) {  // a verify's inputs, for Hook::observe after the commit (chunk_commit); not for prefill chunks
    if (chunk_xn != n) {
      chunk_xn = n;
      chunk_x.resize((size_t)steps * n * H);
    }
    for (int i = 0; i < n; i++) std::memcpy(chunk_x.data() + ((size_t)step * n + i) * H, x[i], sizeof(float) * H);
  } else if (step == 0) {
    chunk_xn = 0;
  }
  auto t0 = clk::now();
  stage_mark(ST_MOE_ROUTER);
  // ---- the released router of every position (the single step's rows and rule)
  {
    std::vector<const XQ*> xq(n);
    std::vector<float*> out(n), outs(n);
    parallel_for(*pool, n, [&](int i) { C.xin[i].prep(x[i], H); });
    for (int i = 0; i < n; i++) {
      xq[i] = &C.xin[i].q;
      out[i] = C.lg.data() + (size_t)i * (NE + 1);
      outs[i] = out[i] + NE;
    }
    static thread_local std::vector<JobN> jr;
    jr.clear();
    jr.push_back({v.router, n, xq.data(), x, out.data()});
    if (v.shared) jr.push_back({v.sh_scalar_gate, n, xq.data(), x, outs.data()});
    if (nh) nh->chunk_rows(*this, step, n, xq.data(), x, jr);  // the routers' rows for a later step, same fork-join
    run_rows_n(*pool, jr.data(), (int)jr.size(), 16, o.ref, &t.bytes_res);
  }
  parallel_for(*pool, n, [&](int i) {  // each position's own routing
    float* lg_ = C.lg.data() + (size_t)i * (NE + 1);
    float* sc_ = C.sc.data() + (size_t)i * NE;
    float* bz_ = C.bz.data() + (size_t)i * NE;
    int32_t* pk = C.pk.data() + (size_t)i * K;
    float* w = C.pw.data() + (size_t)i * K;
    static thread_local std::vector<int> ord_;
    ord_.resize(NE);
    for (int e = 0; e < NE; e++) ord_[e] = e;
    if (v.rule == MoeView::SOFTMAX_TOPK_RENORM) {  // = Core::route
      float mx = -INFINITY;
      for (int e = 0; e < NE; e++) mx = std::max(mx, lg_[e]);
      double zsum = 0;
      for (int e = 0; e < NE; e++) zsum += std::exp((double)(lg_[e] - mx));
      std::partial_sort(ord_.begin(), ord_.begin() + K, ord_.end(), [&](int a, int b) { return lg_[a] > lg_[b] || (lg_[a] == lg_[b] && a < b); });
      C.shw[i] = v.shared ? sigm(lg_[NE]) : 0.f;
      for (int j = 0; j < K; j++) pk[j] = ord_[j];
      double s = 0;  // = Core::weights
      for (int j = 0; j < K; j++) {
        w[j] = (float)(std::exp((double)(lg_[pk[j]] - mx)) / zsum);
        s += w[j];
      }
      if (v.renorm)
        for (int j = 0; j < K; j++) w[j] = (float)(w[j] / s);
    } else {
      for (int e = 0; e < NE; e++) {
        float yv = 1.f / (1.f + std::exp(std::fabs(lg_[e])));
        sc_[e] = lg_[e] < 0 ? yv : 1.f - yv;
        bz_[e] = sc_[e] + v.router_bias[e];
      }
      std::partial_sort(ord_.begin(), ord_.begin() + K, ord_.end(), [&](int a, int b) { return bz_[a] > bz_[b] || (bz_[a] == bz_[b] && a < b); });
      C.shw[i] = 0.f;
      for (int j = 0; j < K; j++) pk[j] = ord_[j];
      float ssum = 0;
      for (int j = 0; j < K; j++) ssum += sc_[pk[j]];
      for (int j = 0; j < K; j++) w[j] = sc_[pk[j]] / (ssum + 1e-20f) * v.route_scale;
    }
  });
  if (pick_log)
    for (int i = 0; i < n; i++) log_picks(pos + off + i, step, C.pk.data() + (size_t)i * K);
  auto t1 = clk::now();
  t.router += ms_between(t0, t1);
  stage_mark(ST_MOE_CACHE);
  // ---- the union: slots (hits pinned, misses read now)
  C.ent.clear();
  C.ent_of.assign(NE, -1);
  auto& S = cache.st;
  const int reg = looped(step) ? 0 : 1;
  auto& ut = used_tok[L];
  for (int i = 0; i < n; i++) {
    const int32_t* pk = C.pk.data() + (size_t)i * K;
    for (int j = 0; j < K; j++) {
      const int e = pk[j];
      S.n_pick_r[reg]++;
      if (pass > 0 && !span_mode && std::find(ut.begin(), ut.end(), e) != ut.end()) S.n_reuse_hit++;  // (a token's statistic)
      int q = C.ent_of[e];
      if (q < 0) {
        q = C.ent_of[e] = (int)C.ent.size();
        C.ent.emplace_back();
        Chunk::Ent& E = C.ent.back();
        E.e = e;
        int s = cache.slot_of(L, e);
        if (s >= 0 && cache.mem == Mem::PURGEABLE) {
          cache.touch_nonvolatile(L, s);
          s = cache.slot_of(L, e);
        }
        if (s >= 0) {
          if (cache.ready(L, s)) {
            S.n_hit++;
            S.n_hit_r[reg]++;
          } else {
            S.n_late++;
          }
          cache.slot(L, s).pins++;
        } else {  // = Core::take_slot, with one fallback: a wide union may need the slots the loop holds pin
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
          s = grab();
          if (s < 0) {  // release the holds and early names (pins only: the outputs cannot change) and try again
            for (int L2 = 0; L2 < (int)holds.size(); L2++) {
              for (int hs : holds[L2]) cache.unpin(L2, hs);
              holds[L2].clear();
            }
            for (int s2 = 0; s2 < (int)span_names.size(); s2++) {
              for (auto& se : span_names[s2]) cache.unpin_pf(mi.step_layer[s2], se.first, se.second);
              span_names[s2].clear();
            }
            n_hold_release++;
            s = grab();
          }
          if (s < 0) {
            fprintf(stderr, "moe_chunk: no expert slot free in layer %d's pool (budget too small for a chunk's union)\n", L);
            abort();
          }
          if (cache.mem == Mem::PURGEABLE) cache.touch_nonvolatile(L, s);
          cache.issue(L, s, e, true, now_s);
          cache.slot(L, s).pins = 1;
          S.n_miss++;
          miss_step[step]++;
        }
        E.slot = s;
      } else {
        S.n_hit++;  // a repeat inside the chunk: in memory by the time it is computed
        S.n_hit_r[reg]++;
      }
      C.ent[q].users.push_back(i * R + j);
    }
  }
  if (looped(step) && !span_mode) {
    if (pass == 0) ut.clear();
    for (int i = 0; i < n; i++)
      for (int j = 0; j < K; j++) ut.push_back(C.pk[(size_t)i * K + j]);
  }
  auto t2a = clk::now();
  t.cache += ms_between(t1, t2a);
  if (nh) nh->chunk_after_router(*this, step, n);  // names for later steps: queued behind this step's own misses
  auto t2 = clk::now();
  t.pf += ms_between(t2a, t2);
  stage_mark(ST_MOE_EXPERTS);
  // ---- experts: the shared expert and every routed one in memory, then the rest as they land
  const double wait0 = S.wait_ms;
  const bool packable = !o.ref && o.prec == Prec::I16;
  for (int ps = 0; ps < 2; ps++) {
    std::vector<int> run;
    int shared_q = -1;
    if (ps == 0 && v.shared) {  // the shared expert as one more entry: every position, row K
      shared_q = (int)C.ent.size();
      C.ent.emplace_back();
      Chunk::Ent& E = C.ent.back();
      E.e = -1;
      E.slot = -1;
      E.gu = *v.sh_gate;
      E.up = *v.sh_up;
      E.dn = *v.sh_down;
      E.inter = sh_inter_;
      for (int i = 0; i < n; i++) E.users.push_back(i * R + K);
      E.done = true;
      run.push_back(shared_q);
    }
    for (int q = 0; q < (int)C.ent.size(); q++) {
      Chunk::Ent& E = C.ent[q];
      if (E.done) continue;
      if (ps == 0 && !cache.ready(L, E.slot)) continue;
      if (ps == 1) cache.wait(L, E.slot);
      F.mats(cache.rec(L, E.slot), E.gu, E.up, E.dn);
      E.inter = F.I;
      E.packed = packable && E.gu.kind == Weight::NVFP4 && (int)E.users.size() >= PACK_MIN;
      if (E.packed) E.xg.reserve(H, (int)E.users.size());
      E.done = true;
      run.push_back(q);
    }
    if (run.empty()) continue;
    parallel_for(*pool, (int)run.size(), [&](int k) {  // the packed inputs of each entry
      Chunk::Ent& E = C.ent[run[k]];
      if (!E.packed) return;
      for (size_t u = 0; u < E.users.size(); u++) xqb_set_from(E.xg, (int)u, C.xin[E.users[u] / R].q);
      xqb_pad(E.xg);
    });
    for (int q : run)
      for (int row : C.ent[q].users) C.act[row].reset(C.ent[q].inter);
    ChCtx X;
    X.C = &C;
    X.run = &run;
    X.H = H;
    X.IA = IA;
    X.R = R;
    X.x = x;
    X.ref = o.ref;
    X.softcap_act = v.act == MoeView::SOFTCAP_SWIGLU;
    X.gcap = v.gate_cap;
    X.ucap = v.up_cap;
    X.start.assign(run.size() + 1, 0);
    for (size_t k = 0; k < run.size(); k++) X.start[k + 1] = X.start[k] + C.ent[run[k]].inter / XB;
    pool->run(chunk_gu, &X);
    for (int q : run)
      if (C.ent[q].packed) C.ent[q].xd.reserve(C.ent[q].inter, (int)C.ent[q].users.size());
    parallel_for(*pool, (int)run.size(), [&](int k) {  // the down inputs exist now
      Chunk::Ent& E = C.ent[run[k]];
      if (!E.packed) return;
      for (size_t u = 0; u < E.users.size(); u++) xqb_set_from(E.xd, (int)u, C.act[E.users[u]].q);
      xqb_pad(E.xd);
    });
    pool->run(chunk_dn, &X);
  }
  t.bytes_exp += (double)(C.ent.size() - (v.shared ? 1 : 0)) * F.touched;
  stage_mark(ST_MOE_COMBINE);
  // ---- the combination of every position: routed experts in pick order, then the shared expert (= Core::combine)
  // the chain of Core::combine written out: fma(w[K-1], e[K-1], ... fma(w[0], e[0], 0)), then fma(shw, shared, .), so the
  // bits do not depend on what -ffp-contract does with this loop
  parallel_for(*pool, n, [&](int i) {
    const float* w = C.pw.data() + (size_t)i * K;
    const float* eo = C.eout.data() + (size_t)i * R * H;
    for (int r = 0; r < H; r++) {
      float acc = 0;
      for (int j = 0; j < K; j++) acc = __builtin_fmaf(w[j], eo[(size_t)j * H + r], acc);
      y[i][r] = v.shared ? __builtin_fmaf(C.shw[i], eo[(size_t)K * H + r], acc) : acc;
    }
  });
  if (pk_out) std::copy(C.pk.begin(), C.pk.begin() + (size_t)n * K, pk_out);
  if (w_out) std::copy(C.pw.begin(), C.pw.begin() + (size_t)n * K, w_out);
  std::copy(C.pk.begin() + (size_t)(n - 1) * K, C.pk.begin() + (size_t)n * K, picks.data() + (size_t)step * K);
  std::copy(C.pw.begin() + (size_t)(n - 1) * K, C.pw.begin() + (size_t)n * K, pw.data() + (size_t)step * K);
  std::copy(C.pk.begin() + (size_t)(n - 1) * K, C.pk.begin() + (size_t)n * K, used.data() + (size_t)step * K);
  std::copy(C.pw.begin() + (size_t)(n - 1) * K, C.pw.begin() + (size_t)n * K, uw.data() + (size_t)step * K);
  auto t3 = clk::now();
  const double wt = S.wait_ms - wait0;
  t.wait += wt;
  wait_step[step] += wt;
  t.experts += ms_between(t2, t3) - wt;
  stage_mark(ST_MOE_CACHE);
  // ---- the cache's bookkeeping, position by position as single steps would have done it; one pin per entry
  for (int i = 0; i < n; i++) {
    const int32_t* pk = C.pk.data() + (size_t)i * K;
    const long st = (tok_clock + off + i) * steps + step;
    for (int j = 0; j < K; j++) cache.use(L, C.ent[C.ent_of[pk[j]]].slot, tok_clock + off + i, st);
  }
  const bool hold = o.loop_hold && !span_mode && next_same[step] >= 0;
  if (span_mode && (int)span_hold.size() < steps) span_hold.resize(steps);
  for (auto& E : C.ent) {
    if (E.slot < 0) continue;  // the shared expert
    if (hold) holds[L].push_back(E.slot);
    else if (span_mode) span_hold[step].push_back(E.slot);
    else cache.unpin(L, E.slot);
  }
  if (next_same[step] < 0 && !holds[L].empty()) {
    for (int s : holds[L]) cache.unpin(L, s);
    holds[L].clear();
  }
  if (nh) {
    auto tp = clk::now();
    nh->chunk_after_moe(*this, step, n, C.pk.data());
    nh->chunk_release(*this, step, n);
    t.pf += ms_since(tp);
  }
  t.cache += ms_since(t3);
}

void Core::chunk_start(int n, const int32_t* toks, const float* const* embs) {
  chunk_named = chunk_names && hook && n >= 2 && n <= 8 && !(force && pos < force_n);
  if (!chunk_named) return;
  if (cache.budget_req.load(std::memory_order_relaxed) >= 0) cache.apply_budget();
  auto tp = clk::now();
  hook->chunk_start(*this, n, toks, embs);
  t.pf += ms_since(tp);
}

void Core::span_name(int step) {
  if ((int)span_names.size() < steps) span_names.resize(steps);
  const int L = mi.step_layer[step];
  auto& nm = span_names[step];
  for (int e = 0; e < NE; e++) {
    const int sl = cache.name(L, e, now(step));
    if (sl >= 0) nm.push_back({sl, e});
  }
}

void Core::span_unname(int step) {
  const int L = mi.step_layer[step];
  if (step < (int)span_names.size()) {
    for (auto& se : span_names[step]) cache.unpin_pf(L, se.first, se.second);
    span_names[step].clear();
  }
  if (step < (int)span_hold.size()) {
    for (int sl : span_hold[step]) cache.unpin(L, sl);
    span_hold[step].clear();
  }
}

void Core::chunk_commit(int a) {
  if (!hook || a < 1 || a > chunk_xn || chunk_x.size() < (size_t)steps * chunk_xn * H) return;
  for (int s = 0; s < steps; s++) hook->observe(*this, s, chunk_x.data() + ((size_t)s * chunk_xn + a - 1) * H);
}

}  // namespace dm
