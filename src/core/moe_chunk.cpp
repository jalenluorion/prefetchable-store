// The MoE step over a chunk of positions (batched prefill, multi-position verify): every position's released routing and
// combination exactly as the single step (moe.cpp) computes them; the union of the positions' experts pinned and read
// once; each expert computed for all positions that picked it in one pass over its record (multi-vector kernels, packed
// i8mm for NVFP4 from 6 users). Lossless, or fast mode's leave-out rule (approx:drop) for a verify pass.
#include "dmeng/workcount.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

#include "dmeng/core.h"
#include "gpu_gemm.h"

namespace dm {

constexpr int MAXK = 16;     // = moe.cpp's (Core::init refuses more)
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
    std::vector<int> gpos;   // the users' positions (the GPU reads them from the chunk's packed input: Chunk::xall)
    bool packed = false, done = false;
  };
  std::vector<Ent> ent;
  XQB xall;           // every position's MoE input packed, staged once per GPU set (each entry's rows read through gpos)
  bool xall_ok = false;
  std::vector<int> ent_of;  // [NE] -> entry, -1
  // fast mode (approx:drop): keep[i * K + j] = position i's pick j is computed (0: left out); wtok[i] = the released weight
  // left out of position i over the pass's steps so far
  std::vector<uint8_t> keep;
  std::vector<double> wtok;
};

namespace {
struct ChCtx {
  Core::Chunk* C;
  const std::vector<int>* run;  // entries of this pass
  int H, IA, R;                 // R = K + 1 rows per position
  int dn_rows = 32;             // rows per down-projection job (Core::chunk_dn_rows)
  const float* const* x;        // [n] the fp32 MoE inputs (the reference path reads them)
  bool ref, softcap_act;
  float gcap, ucap;
  std::vector<int> start;       // gate-up chunks per entry (prefix)
  std::vector<int> c0;          // per entry: the first gate-up chunk the CPU computes (the GPU's before it); empty = 0
  std::vector<int> dlo, dhi;    // per entry: the down rows the CPU computes, [dlo, dhi); empty = all
  alignas(128) std::atomic<int> next_gu{0};
  alignas(128) std::atomic<int> next_dn{0};
};
}  // namespace

// the activation of one user's gate / up rows [j0, j0 + XB) (gg_: gate, or gate / up interleaved when fused; uu_: up)
static void gu_act(const ChCtx& X, Core::Chunk& C, int row, int j0, const float* gg_, const float* uu_, bool fused) {
  float* a = C.actf.data() + (size_t)row * X.IA;
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

// the packed batched kernel of an affine record (4-bit routed experts, the 8-bit shared expert): gemm_aff4_i16's /
// gemm_aff8_i16's bits = the one-vector kernels' for inputs a multiple of 256
static bool affine(Weight::Kind k) { return k == Weight::AFF4 || k == Weight::AFF8; }
static void gemm_affine(const Weight& m, int r0, int r1, const XQB& x, int nu, float* y, size_t ldy) {
  if (m.kind == Weight::AFF8) gemm_aff8_i16(m.c, (const _Float16*)m.s, (const _Float16*)m.b, m.cols, r0, r1, x, 0, nu, y, ldy);
  else gemm_aff4_i16(m.c, (const _Float16*)m.s, (const _Float16*)m.b, m.cols, r0, r1, x, 0, nu, y, ldy);
}

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
    const int j0 = ((X.c0.empty() ? 0 : X.c0[q]) + ch - X.start[q]) * XB, nu = (int)E.users.size();
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
    if (E.packed && E.gu.kind == Weight::NVFP4) {
      gemm_nvfp4_i16(E.gu.c, (const uint8_t*)E.gu.s, E.gu.gmul, E.gu.cols, fused ? 2 * j0 : j0, fused ? 2 * j0 + 2 * XB : j0 + XB, E.xg, 0, nu,
                     g.data(), gw);
      if (!fused) gemm_nvfp4_i16(E.up.c, (const uint8_t*)E.up.s, E.up.gmul, E.up.cols, j0, j0 + XB, E.xg, 0, nu, u.data(), XB);
    } else if (E.packed && affine(E.gu.kind) && !fused && E.gu.cols % 256 == 0) {  // ModelInfo::chunk_aff4
      gemm_affine(E.gu, j0, j0 + XB, E.xg, nu, g.data(), gw);
      gemm_affine(E.up, j0, j0 + XB, E.xg, nu, u.data(), XB);
    } else if (fused) {
      mv_multi(E.gu, 2 * j0, 2 * j0 + 2 * XB, xq.data(), xf.data(), nu, ys.data(), X.ref);
    } else {
      mv_multi(E.gu, j0, j0 + XB, xq.data(), xf.data(), nu, ys.data(), X.ref);
      mv_multi(E.up, j0, j0 + XB, xq.data(), xf.data(), nu, ysu.data(), X.ref);
    }
    for (int k = 0; k < nu; k++) gu_act(X, C, E.users[k], j0, g.data() + (size_t)k * gw, u.data() + (size_t)k * XB, fused);
  }
}

static void chunk_dn(void* p, int, int) {
  ChCtx& X = *(ChCtx*)p;
  Core::Chunk& C = *X.C;
  const int CH = X.dn_rows;
  const int nch = (X.H + CH - 1) / CH, nr = (int)X.run->size(), total = nch * nr;
  static thread_local std::vector<const XQ*> xq;
  static thread_local std::vector<const float*> xf;
  static thread_local std::vector<float*> ys;
  static thread_local std::vector<float> tmp;
  for (int ch; (ch = X.next_dn.fetch_add(1, std::memory_order_relaxed)) < total;) {
    Core::Chunk::Ent& E = C.ent[(*X.run)[ch / nch]];
    const int a = (ch % nch) * CH, b = std::min(X.H, a + CH), nu = (int)E.users.size();
    if (!X.dlo.empty() && (a < X.dlo[ch / nch] || a >= X.dhi[ch / nch])) continue;  // the GPU's rows
    const bool aff = E.packed && affine(E.dn.kind);  // ModelInfo::chunk_aff4
    if (E.packed && !(a & 3) && !(b & 3) && (!aff || E.dn.cols % 256 == 0)) {
      tmp.resize((size_t)nu * CH);
      if (aff) gemm_affine(E.dn, a, b, E.xd, nu, tmp.data(), CH);
      else gemm_nvfp4_i16(E.dn.c, (const uint8_t*)E.dn.s, E.dn.gmul, E.dn.cols, a, b, E.xd, 0, nu, tmp.data(), CH);
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
  if ((ns.on && !ns.drop) || (force && pos < force_n) || n < 1) {
    fprintf(stderr, "moe_chunk: lossless or the drop rule, unforced only\n");
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
  if (!cache.floor_keys.empty()) cache.expire_floors();
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
  C.xall_ok = false;
  auto& S = cache.st;
  const int reg = looped(step) ? 0 : 1;
  auto& ut = used_tok[L];
  // fast mode: the single step's leave-out rule (Core::approx with ns.drop) for every position, decided at the step's
  // start. A pick held in memory (landed or in flight) is used; a missing pick is read now when its released weight is
  // > read_w (then the largest missing until the position's missing weight is <= read_mass); a missing pick of lower
  // weight is left out, unless another position of the pass reads that expert now (then it is in memory when computed,
  // at no wait). Kept picks keep their released weights; nothing is substituted. With nothing left out the step is the
  // lossless chunk's, bit for bit.
  const bool drop = ns.on;
  uint8_t* keep = nullptr;
  if (drop) {
    C.keep.assign((size_t)n * K, 0);
    if (step == 0 || (int)C.wtok.size() != n) C.wtok.assign(n, 0.0);
    keep = C.keep.data();
    const float wscale = v.rule == MoeView::SOFTMAX_TOPK_RENORM ? 1.f : v.route_scale;
    std::vector<uint8_t> rd(NE, 0);
    for (int i = 0; i < n; i++) {
      const int32_t* pk = C.pk.data() + (size_t)i * K;
      const float* w = C.pw.data() + (size_t)i * K;
      bool must[MAXK] = {false}, miss[MAXK] = {false};
      double mass = 0;
      for (int j = 0; j < K; j++) {
        if (cache.slot_of(L, pk[j]) >= 0) continue;
        miss[j] = true;
        const double f = w[j] / wscale;
        if (f > ns.read_w) must[j] = true;
        else mass += f;
      }
      while (mass > ns.read_mass) {
        int b = -1;
        for (int j = 0; j < K; j++)
          if (miss[j] && !must[j] && (b < 0 || w[j] > w[b])) b = j;
        if (b < 0) break;
        must[b] = true;
        mass -= w[b] / wscale;
      }
      for (int j = 0; j < K; j++) {
        if (!miss[j] || must[j]) keep[(size_t)i * K + j] = 1;
        if (must[j]) rd[pk[j]] = 1;
      }
    }
    for (int i = 0; i < n; i++)
      for (int j = 0; j < K; j++) {
        const size_t r = (size_t)i * K + j;
        if (!keep[r] && rd[C.pk[r]]) keep[r] = 1;
        if (!keep[r]) {
          const double f = C.pw[r] / wscale;
          S.n_miss++;
          ns.n_replaced++;
          ns.w_replaced += f;
          ns.w_replaced_max = std::max(ns.w_replaced_max, f);
          ns.w_hist[f < 0.02 ? 0 : f < 0.05 ? 1 : f < 0.1 ? 2 : f < 0.2 ? 3 : f < 0.4 ? 4 : 5]++;
          C.wtok[i] += f;
        }
      }
    for (int i = 0; i < n; i++) {
      bool any = false;
      for (int j = 0; j < K; j++) any |= !keep[(size_t)i * K + j];
      ns.steps_replaced += any;
    }
    ns.steps += n;
  }
  auto grab = [&]() {  // = Core::take_slot's choice
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
  // Batches (phone sizes): a union larger than the free slots is not an abort. The misses that find no slot after the
  // fallback below are deferred (slot -2) and run after the first batch in later batches, each batch's pins released
  // before the next takes slots (section "batches" below). Every (position, pick) row of C.eout is computed from the same
  // record bytes in any batch, and the combination reads the rows in pick order after the last batch, so the outputs
  // are bit-identical. With a union that fits (every Mac-size run) nothing is deferred and nothing below changes.
  int deferred = 0;
  for (int i = 0; i < n; i++) {
    const int32_t* pk = C.pk.data() + (size_t)i * K;
    for (int j = 0; j < K; j++) {
      const int e = pk[j];
      S.n_pick_r[reg]++;
      if (keep && !keep[(size_t)i * K + j]) continue;  // left out (fast mode)
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
          s = deferred ? -1 : grab();
          if (s < 0 && !deferred) {  // release the holds, early names and the span's union pins (pins only: the
                                     // outputs cannot change) and try again
            for (int L2 = 0; L2 < (int)holds.size(); L2++) {
              for (int hs : holds[L2]) cache.unpin(L2, hs);
              holds[L2].clear();
            }
            for (int s2 = 0; s2 < (int)span_names.size(); s2++) {
              for (auto& se : span_names[s2]) cache.unpin_pf(mi.step_layer[s2], se.first, se.second);
              span_names[s2].clear();
            }
            for (int s2 = 0; s2 < (int)span_hold.size(); s2++) {  // (phone sizes) earlier sub-chunks' unions of a span
              for (int hs : span_hold[s2]) cache.unpin(mi.step_layer[s2], hs);
              span_hold[s2].clear();
            }
            n_hold_release++;
            s = grab();
          }
          if (s < 0) {  // still none: this entry and every later miss of the union run in a later batch
            E.slot = -2;
            deferred++;
            C.ent[q].users.push_back(i * R + j);
            continue;
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
  if (span_next >= 0 && off == 0) {  // a span's next step named while this one runs (its first sub-chunk)
    stage_mark(ST_PREFETCH);
    if (span_next_router) span_name_router(span_next, n, x);
    else span_name(span_next);
  }
  span_next = -1;
  auto t2 = clk::now();
  t.pf += ms_between(t2a, t2);
  stage_mark(ST_MOE_EXPERTS);
  // ---- experts: the shared expert and every routed one in memory, then the rest as they land
  const double wait0 = S.wait_ms;
  const bool packable = !o.ref && o.prec == Prec::I16;
  auto compute = [&](std::vector<int>& run, bool gpu_ok) {  // gate/up then down of the entries in run, into their C.eout rows
    DM_WC("moe.waves", 1);
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
    X.dn_rows = chunk_dn_rows;
    X.x = x;
    X.ref = o.ref;
    X.softcap_act = v.act == MoeView::SOFTCAP_SWIGLU;
    X.gcap = v.gate_cap;
    X.ucap = v.up_cap;
    // the GPU's share (moe_gpu): the first gate-up chunks and down rows of each packed NVFP4 entry, the CPU the rest at
    // the same time; every row computed once, by one side, with the same bits; a set the GPU does not complete is
    // computed on the CPU
    GpuGemm* const G = gpu_ok && moe_gpu && moe_gpu_frac > 0 && !o.ref ? moe_gpu : nullptr;
    struct GT { int k, cg, tg, tu, rd, td; };  // entry index in run; gate-up chunks and tickets; down rows and ticket
    std::vector<GT> gt;
    X.start.assign(run.size() + 1, 0);
    if (G) {
      X.c0.assign(run.size(), 0);
      auto t0 = clk::now();
      G->begin();
      for (size_t k = 0; k < run.size(); k++) {
        Chunk::Ent& E = C.ent[run[k]];
        const bool nv = E.gu.kind == Weight::NVFP4 && E.dn.kind == Weight::NVFP4;
        // ModelInfo::chunk_aff4: the GPU's affine bits are gemm_aff4_i16's / gemm_aff8_i16's, the one-vector kernel's only for
        // inputs a multiple of 256 (gate / up here; the down rows below likewise)
        const bool af = affine(E.gu.kind) && E.dn.kind == E.gu.kind && E.up.kind == E.gu.kind && E.gu.cols % 256 == 0;
        if (!E.packed || !(nv || af)) continue;
        const int nch = E.inter / XB, cg = (int)(nch * moe_gpu_frac), nu = (int)E.users.size();
        if (cg <= 0) continue;
        const bool fused = E.up.kind == Weight::NONE;
        if (!C.xall_ok) {  // the chunk's inputs once (not one gathered copy per expert): the GPU gathers through gpos
          C.xall.reserve(H, n);
          parallel_for(*pool, n, [&](int i) { xqb_set_from(C.xall, i, C.xin[i].q); });
          xqb_pad(C.xall);
          C.xall_ok = true;
        }
        E.gpos.resize(nu);
        for (int u = 0; u < nu; u++) E.gpos[u] = E.users[u] / R;
        const int s = G->stage(C.xall);
        const int* gm = E.gpos.data();
        GT r{(int)k, cg, -1, -1, 0, -1};
        if (af) {
          const int bits = E.gu.kind == Weight::AFF8 ? 8 : 4;
          r.tg = G->add(bits, E.gu.c, (const _Float16*)E.gu.s, (const _Float16*)E.gu.b, E.gu.cols, cg * XB, nu, s, gm);
          if (r.tg >= 0) r.tu = G->add(bits, E.up.c, (const _Float16*)E.up.s, (const _Float16*)E.up.b, E.up.cols, cg * XB, nu, s, gm);
        } else {
          r.tg = G->add_nvfp4(E.gu.c, (const uint8_t*)E.gu.s, E.gu.gmul, E.gu.cols, fused ? 2 * cg * XB : cg * XB, nu, s, gm);
          if (r.tg >= 0 && !fused) r.tu = G->add_nvfp4(E.up.c, (const uint8_t*)E.up.s, E.up.gmul, E.up.cols, cg * XB, nu, s, gm);
        }
        if (r.tg < 0 || (!fused && r.tu < 0)) continue;  // not wrapped: the CPU computes the whole entry
        X.c0[k] = cg;
        gt.push_back(r);
      }
      const int nc = gt.empty() ? 0 : G->prepare();
      if (nc == 0) {
        gt.clear();
        X.c0.clear();
      } else {
        parallel_for(*pool, nc, [&](int i) { G->copy(i); });
        G->submit();
      }
      moe_gpu_stage_ms += ms_since(t0);
    }
    for (size_t k = 0; k < run.size(); k++)
      X.start[k + 1] = X.start[k] + C.ent[run[k]].inter / XB - (X.c0.empty() ? 0 : X.c0[k]);
    pool->run(chunk_gu, &X);
    if (!gt.empty()) {  // the GPU's gate-up rows: wait, then their activations (or, if it failed, the CPU's rows)
      auto tw = clk::now();
      const bool ok = G->finish();
      moe_gpu_wait_ms += ms_since(tw);
      moe_gpu_sets++;
      if (ok) {
        std::vector<std::pair<int, int>> jobs;  // (gt index, chunk)
        for (size_t i = 0; i < gt.size(); i++)
          for (int c = 0; c < gt[i].cg; c++) jobs.push_back({(int)i, c});
        parallel_for(*pool, (int)jobs.size(), [&](int u) {
          const GT& r = gt[jobs[u].first];
          const int c = jobs[u].second;
          Chunk::Ent& E = C.ent[run[r.k]];
          const bool fused = E.up.kind == Weight::NONE;
          const int Rg = fused ? 2 * r.cg * XB : r.cg * XB;
          const float* yg = G->out(r.tg);
          const float* yu = fused ? nullptr : G->out(r.tu);
          for (size_t k = 0; k < E.users.size(); k++)
            gu_act(X, C, E.users[k], c * XB, yg + k * Rg + (fused ? 2 * c * XB : c * XB), fused ? nullptr : yu + k * Rg + c * XB, fused);
        });
      } else {
        moe_gpu_failed++;
        ChCtx Y;
        Y.C = &C;
        Y.run = &run;
        Y.H = H;
        Y.IA = IA;
        Y.R = R;
        Y.dn_rows = chunk_dn_rows;
        Y.x = x;
        Y.ref = o.ref;
        Y.softcap_act = X.softcap_act;
        Y.gcap = X.gcap;
        Y.ucap = X.ucap;
        Y.start.assign(run.size() + 1, 0);
        Y.c0.assign(run.size(), 0);
        std::vector<int> cgk(run.size(), 0);
        for (const GT& r : gt) cgk[r.k] = r.cg;
        for (size_t k = 0; k < run.size(); k++) Y.start[k + 1] = Y.start[k] + cgk[k];
        pool->run(chunk_gu, &Y);
      }
      for (size_t i = 0; i < gt.size(); i++) {
        const Chunk::Ent& E = C.ent[run[gt[i].k]];
        moe_gpu_macs += (double)gt[i].cg * XB * (E.up.kind == Weight::NONE ? 2 : 2) * E.gu.cols * E.users.size();
      }
    }
    for (int q : run)
      if (C.ent[q].packed) C.ent[q].xd.reserve(C.ent[q].inter, (int)C.ent[q].users.size());
    parallel_for(*pool, (int)run.size(), [&](int k) {  // the down inputs exist now
      Chunk::Ent& E = C.ent[run[k]];
      if (!E.packed) return;
      for (size_t u = 0; u < E.users.size(); u++) xqb_set_from(E.xd, (int)u, C.act[E.users[u]].q);
      xqb_pad(E.xd);
    });
    if (!gt.empty()) {  // the down rows: the GPU the first rd of each such entry, the CPU the rest
      auto t0 = clk::now();
      G->begin();
      X.dlo.assign(run.size(), 0);
      X.dhi.assign(run.size(), H);
      std::vector<GT> gd;
      for (GT r : gt) {
        Chunk::Ent& E = C.ent[run[r.k]];
        r.rd = ((int)(H * moe_gpu_frac) / 64) * 64;
        if (r.rd <= 0 || (affine(E.dn.kind) && E.dn.cols % 256 != 0)) continue;
        r.td = affine(E.dn.kind)
                   ? G->add(E.dn.kind == Weight::AFF8 ? 8 : 4, E.dn.c, (const _Float16*)E.dn.s, (const _Float16*)E.dn.b, E.dn.cols, r.rd, (int)E.users.size(), G->stage(E.xd))
                   : G->add_nvfp4(E.dn.c, (const uint8_t*)E.dn.s, E.dn.gmul, E.dn.cols, r.rd, (int)E.users.size(), G->stage(E.xd));
        if (r.td < 0) continue;
        X.dlo[r.k] = r.rd;
        gd.push_back(r);
      }
      gt.swap(gd);
      const int nc = gt.empty() ? 0 : G->prepare();
      if (nc == 0) {
        gt.clear();
        X.dlo.clear();
        X.dhi.clear();
      } else {
        parallel_for(*pool, nc, [&](int i) { G->copy(i); });
        G->submit();
      }
      moe_gpu_stage_ms += ms_since(t0);
    }
    pool->run(chunk_dn, &X);
    if (!gt.empty()) {
      auto tw = clk::now();
      const bool ok = G->finish();
      moe_gpu_wait_ms += ms_since(tw);
      moe_gpu_sets++;
      if (ok) {
        parallel_for(*pool, (int)gt.size(), [&](int i) {
          const GT& r = gt[i];
          const Chunk::Ent& E = C.ent[run[r.k]];
          const float* y = G->out(r.td);
          for (size_t k = 0; k < E.users.size(); k++)
            std::memcpy(C.eout.data() + (size_t)E.users[k] * H, y + k * r.rd, sizeof(float) * r.rd);
        });
      } else {
        moe_gpu_failed++;
        for (size_t k = 0; k < run.size(); k++) X.dhi[k] = X.dlo[k], X.dlo[k] = 0;
        X.next_dn.store(0);
        pool->run(chunk_dn, &X);
      }
      for (const GT& r : gt) moe_gpu_macs += (double)r.rd * C.ent[run[r.k]].dn.cols * C.ent[run[r.k]].users.size();
    }
  };
  // the kernel choice (one rule for every record): the packed batched kernels from a number of users where the format has
  // one, else the multi-vector kernels; the same bits either way. NVFP4 from PACK_MIN; the affine routed records from
  // chunk_pack_affine (-1: PACK_MIN with ModelInfo::chunk_aff4, else never); the shared expert from chunk_pack_shared
  // (0: never); a pass sets the two for its kind (the 35B's prefill: every expert and the shared one packed, as it was)
  const int pmin_aff = chunk_pack_affine >= 0 ? chunk_pack_affine : mi.chunk_aff4 ? PACK_MIN : 0;
  auto pack = [&](Chunk::Ent& E) {
    const int pm = E.gu.kind == Weight::NVFP4 ? PACK_MIN : !affine(E.gu.kind) ? 0 : E.e < 0 ? chunk_pack_shared : pmin_aff;
    E.packed = packable && pm > 0 && (int)E.users.size() >= pm;
    if (E.packed) E.xg.reserve(H, (int)E.users.size());
    E.done = true;
  };
  auto prepare = [&](Chunk::Ent& E) {  // the record's matrices (landed) and its kernel choice
    F.mats(cache.rec(L, E.slot), E.gu, E.up, E.dn);
    E.inter = F.I;
    pack(E);
  };
  auto users_of = [&](const std::vector<int>& run) {
    size_t u = 0;
    for (int q : run) u += C.ent[q].users.size();
    return u;
  };
  // the experts in memory first (with the shared expert); then the rest: all at once once they have landed, or with
  // chunk_waves in waves as they land (wait for one, compute it with whatever else has landed), a wave using the GPU's
  // share only from chunk_wave_gpu_min (position, expert) rows. The order of computation never changes a bit: every row of
  // C.eout is computed once from its record, and the combination below reads them in pick order.
  for (int ps = 0;; ps++) {
    std::vector<int> run;
    if (ps == 0 && v.shared) {  // the shared expert as one more entry: every position, row K
      C.ent.emplace_back();
      Chunk::Ent& E = C.ent.back();
      E.e = -1;
      E.slot = -1;
      E.gu = *v.sh_gate;
      E.up = *v.sh_up;
      E.dn = *v.sh_down;
      E.inter = sh_inter_;
      for (int i = 0; i < n; i++) E.users.push_back(i * R + K);
      pack(E);
      run.push_back((int)C.ent.size() - 1);
    }
    bool waited = false;
    for (int q = 0; q < (int)C.ent.size(); q++) {
      Chunk::Ent& E = C.ent[q];
      if (E.done || E.slot == -2) continue;  // done, or deferred to a later batch
      if (ps > 0 && (!chunk_waves || !waited)) {  // all at once: wait for each; waves: for the first only
        cache.wait(L, E.slot);
        waited = true;
      } else if (!cache.ready(L, E.slot)) {
        continue;
      }
      prepare(E);
      run.push_back(q);
    }
    if (run.empty()) {
      if (ps > 0) break;
      continue;
    }
    compute(run, !chunk_waves || users_of(run) >= (size_t)chunk_wave_gpu_min);
    if (ps > 0 && !chunk_waves) break;
  }
  // ---- batches (phone sizes only: some misses were deferred). The entries computed so far are counted as used, as
  // the bookkeeping below would, and unpinned (no loop hold or span pin for them); then each later batch takes as many
  // slots as the pool gives, waits, computes, counts and unpins in the same way. Eviction statistics differ from a
  // run with a larger cache (the order of the use calls); outputs do not.
  auto use_unpin = [&](Chunk::Ent& E) {
    for (int row : E.users) {
      const int i = row / R;
      cache.use(L, E.slot, tok_clock + off + i, (tok_clock + off + i) * steps + step);
    }
    cache.unpin(L, E.slot);
    E.slot = -3;  // used and released: skipped by the bookkeeping below
  };
  if (deferred) {
    for (auto& E : C.ent)
      if (E.slot >= 0) use_unpin(E);
    n_union_batches++;
    while (deferred) {
      std::vector<int> run;
      for (int q = 0; q < (int)C.ent.size(); q++) {
        Chunk::Ent& E = C.ent[q];
        if (E.slot != -2) continue;
        int s = cache.slot_of(L, E.e);  // read meanwhile by a name (or still held)
        if (s >= 0) {
          cache.slot(L, s).pins++;
        } else {
          s = grab();
          if (s < 0) break;  // the pool is full of this batch's pins
          if (cache.mem == Mem::PURGEABLE) cache.touch_nonvolatile(L, s);
          cache.issue(L, s, E.e, true, now_s);
          cache.slot(L, s).pins = 1;
          S.n_miss++;
          miss_step[step]++;
        }
        E.slot = s;
        run.push_back(q);
        deferred--;
      }
      if (run.empty()) {
        fprintf(stderr, "moe_chunk: no expert slot free in layer %d's pool (budget below one record)\n", L);
        abort();
      }
      for (int q : run) {
        cache.wait(L, C.ent[q].slot);
        prepare(C.ent[q]);
      }
      compute(run, true);
      for (int q : run) use_unpin(C.ent[q]);
    }
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
      for (int j = 0; j < K; j++)
        if (!keep || keep[(size_t)i * K + j]) acc = __builtin_fmaf(w[j], eo[(size_t)j * H + r], acc);
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
    for (int j = 0; j < K; j++) {
      if (keep && !keep[(size_t)i * K + j]) continue;  // left out (fast mode)
      const int s = C.ent[C.ent_of[pk[j]]].slot;
      if (s >= 0) cache.use(L, s, tok_clock + off + i, st);  // (-3: counted in its batch)
    }
  }
  if (keep && step == steps - 1)  // the pass's last step: one per-token sample per position (Approx::w_tok)
    for (int i = 0; i < n; i++) ns.w_tok.push_back((float)C.wtok[i]);
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

// The next step's released router on this step's MoE inputs (C.xin, quantised), its top K per position named (names only:
// the outputs cannot change; the 35B's early names for spans shorter than whole_from)
void Core::span_name_router(int step, int n, const float* const* x) {
  if ((int)span_names.size() < steps) span_names.resize(steps);
  const MoeView* v = step < (int)views.size() ? views[step] : nullptr;
  if (!v || !v->router) return;
  Chunk& C = *ch_;
  const int L = mi.step_layer[step];
  std::vector<float> lg((size_t)n * NE);
  std::vector<const XQ*> xq(n);
  std::vector<float*> out(n);
  for (int i = 0; i < n; i++) xq[i] = &C.xin[i].q, out[i] = lg.data() + (size_t)i * NE;
  JobN jr = {v->router, n, xq.data(), x, out.data()};
  run_rows_n(*pool, &jr, 1, 16, false, nullptr);
  std::vector<uint8_t> want(NE, 0);
  parallel_for(*pool, n, [&](int i) {
    const float* l = lg.data() + (size_t)i * NE;
    static thread_local std::vector<float> key;
    static thread_local std::vector<int> idx;
    key.resize(NE);
    idx.resize(NE);
    for (int e = 0; e < NE; e++) {
      key[e] = v->rule == MoeView::SOFTMAX_TOPK_RENORM ? l[e] : 1.f / (1.f + std::exp(-l[e])) + v->router_bias[e];
      idx[e] = e;
    }
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a < b); });
    for (int j = 0; j < K; j++) __atomic_store_n(&want[idx[j]], 1, __ATOMIC_RELAXED);
  });
  auto& nm = span_names[step];
  for (int e = 0; e < NE; e++)
    if (want[e]) {
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
