// Multi-position verify of drafted tokens for Qwen3.6-35B-A3B: up to TMAX positions from the current state in one pass,
// every position's logits kept, then commit(a) keeps the first a and rolls the rest back.
//
// Exactness: every operation is one the single step (qwen3_5_moe.cpp) runs, through the same compiled code where the
// arithmetic is not a plain shared kernel:
//   - resident matrices (attention projections, o projections, head): run_rows_n, whose multi-vector and packed i8mm
//     kernels are bit-identical per position to the one-vector mv the single step calls; bf16 rows through dot_bf16 per
//     position (the same function);
//   - the Gated DeltaNet conv and heads: the decode path's own gdn::phase_conv / gdn::phase_heads, run once per position in order;
//   - full attention: attn_full_core, the one body decode calls too (KV write at the position, attention over 0..pos);
//   - the MoE block: Core::moe_chunk (each position's released routing, expert arithmetic and combination order are the
//     single step's; the union of the positions' experts is read once);
//   - norms and residual adds: the same inline helpers, per position.
// The Gated DeltaNet in a pass: two buffers per layer. The committed state (A) is only read; the pass writes the state
// after its positions into the second buffer (B): the convolution channel by channel and the heads head by head, each over
// the pass's positions in order (a head's state block stays in the core's cache across them, so the recurrent state is
// streamed once per pass, not once per position), with the decode path's arithmetic per position (gdn::conv_channel_io /
// gdn::conv_channel and gdn::head_io, the decode path's functions with the state's input and output buffers apart). commit(n) swaps A and B (no copy);
// commit(a < n) rebuilds B from A for positions 0..a-1 (the conv rows are the kept inputs; the heads run again on the
// kept convolved inputs) and swaps. Both reproduce the bits of a single steps. The KV of a rolled-back position is simply
// overwritten later (append-only), the position counter and the cache's clock go back.
// The output head: the screened head over all the pass's positions at once (ScreenHead::run_n: the planes read once) when
// the caller wants only the argmax or the top k (Want::GREEDY / TOPK, as a single step under the same Want), else the
// full head. `verify_old` (set_option): the previous pass (states copied at the start and streamed once per position,
// the packed gemm from 6 positions, the full head), for A/B measurements.
#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "dmeng/kernels_batch.h"
#include "qwen3_5_moe.h"

namespace dm {
namespace qwen35 {

namespace {
constexpr int PACK_MIN = 2;      // positions from which the resident matrices go through the packed i8mm gemm
constexpr int PACK_MIN_OLD = 6;  // (verify_old)

// value head h of the Gated DeltaNet over positions 0..n-1 in order: the decode path's head (gdn::head_io) per position,
// the state read from Sin for the first position and kept in Sout from then on (Sout's block of head h then holds the
// state after the n positions). cv[i]: position i's convolved q, k, v; bb[i], aa[i]: its beta and alpha inputs [NVH];
// z[i]: its gate [NVH * DV]; o[i]: its output [NVH * DV] (null: not computed, a replay of the state only).
void lin_head_n(const gdn::Dims& d, const gdn::Weights& w, int h, int n, const float* const* cv, const float* const* bb, const float* const* aa,
                const float* const* z, const float* Sin, float* Sout, float* const* o) {
  const size_t blk = (size_t)d.DV * d.DK;
  for (int i = 0; i < n; i++)
    gdn::head_io(d, w, (i == 0 ? Sin : Sout) + h * blk, Sout + h * blk, cv[i], z ? z[i] + (size_t)h * d.DV : nullptr, bb[i][h], aa[i][h], h,
                 o ? o[i] + (size_t)h * d.DV : nullptr);
}

struct CopyCtx {
  float* d;
  const float* s;
  size_t n;
  alignas(128) std::atomic<size_t> next{0};
};
void phase_copy(void* p, int, int) {
  CopyCtx& C = *(CopyCtx*)p;
  constexpr size_t CH = 1 << 16;
  for (size_t b; (b = C.next.fetch_add(CH, std::memory_order_relaxed)) < C.n;) std::memcpy(C.d + b, C.s + b, sizeof(float) * std::min(CH, C.n - b));
}
void pcopy(Pool& pool, float* d, const float* s, size_t n) {
  CopyCtx C;
  C.d = d;
  C.s = s;
  C.n = n;
  pool.run(phase_copy, &C);
}
}  // namespace

struct Qwen35::Ver {
  int n = 0, pos0 = 0;
  bool live = false;
  std::vector<XBuf> xa, xo, xh;  // [TMAX] quantised inputs per position: attention projections, o projection, head
  XQB pa, po, ph;                // the same packed (>= PACK_MIN positions)
  std::vector<float> x, h, y, ob, lg, fproj, cv;
  std::vector<std::vector<float>> gp;            // [NL] DeltaNet layers: [TMAX][C + ZD + 2 NVH] projections (rollback)
  std::vector<std::vector<float>> cvp;           // [NL] DeltaNet layers: [TMAX][C] convolved inputs (rollback)
  std::vector<std::vector<float>> altc, alts;    // [NL] the second conv / recurrent state buffers (B)
  std::vector<std::vector<float>> snapc, snaps;  // [NL] verify_old: conv / recurrent state at the step's start
  bool old = false;                              // this pass ran the previous path (verify_old)
  double ms_snap = 0, ms_replay = 0;
  long steps = 0, rollbacks = 0, positions = 0;
  // bytes per pass (statistics): state copied (memcpy), recurrent state streamed by the heads (one pass over a layer's
  // whole state = its bytes), resident matrices and head read
  double b_copy = 0, b_stream = 0, b_res = 0, b_head = 0;
};

const float* Qwen35::logits_at(int i) const { return V ? V->lg.data() + (size_t)i * c.V : nullptr; }

const float* Qwen35::final_hidden(int i) const {
  if (i < 0) return h.data();  // the decode step's final rmsnorm output (left in h when the step computed logits)
  return V && i < TMAX ? V->h.data() + (size_t)i * c.H : nullptr;
}

bool Qwen35::embed_row(int32_t tok, float* out) const {
  if (tok < 0 || tok >= c.V) return false;
  rows.row(tok, out);
  return true;
}

int Qwen35::head_argmax(const float* hn, float* conf) {
  if (dh_x.cap < c.H) dh_x.init(c.H, prec);
  dh_x.prep(hn, c.H);
  dh_lg.resize(c.V);
  const int VO = mi.vocab_out > 0 ? mi.vocab_out : c.V;
  int id = 0;
  // the full head (its confidence needs every logit; the screened head, screen_head.h, gives only the exact top rows)
  Job jh = {&head, &dh_x.q, nullptr, dh_lg.data()};
  run_rows(*core->pool, &jh, 1, 64);
  for (int r = 1; r < VO; r++)
    if (dh_lg[r] > dh_lg[id]) id = r;
  double zs = 0;
  for (int r = 0; r < VO; r++) zs += std::exp((double)(dh_lg[r] - dh_lg[id]));
  if (conf) *conf = (float)(1.0 / zs);
  return id;
}

bool Qwen35::verify(State& s0, const int32_t* ids, int n, Want want) {
  S = (QState*)&s0;
  if (n < 1 || n > TMAX || (core->force && S->pos < core->force_n) || S->pos + n > S->ctx) return false;
  const int H = c.H, C = c.conv_dim(), ZD = c.NVH * c.DV, QD = c.NQ * c.HD, KD = c.NKV * c.HD, G = C + ZD + 2 * c.NVH;
  const int FP = 2 * QD + 2 * KD;
  Pool& pool = *core->pool;
  if (!V) {
    V = std::make_shared<Ver>();
    const int maxk = std::max({H, ZD, QD});
    V->xa.resize(TMAX);
    V->xo.resize(TMAX);
    V->xh.resize(TMAX);
    for (int i = 0; i < TMAX; i++) {
      V->xa[i].init(maxk, prec);
      V->xo[i].init(maxk, prec);
      V->xh[i].init(maxk, prec);
    }
    V->x.resize((size_t)TMAX * H);
    V->h.resize((size_t)TMAX * H);
    V->y.resize((size_t)TMAX * H);
    V->ob.resize((size_t)TMAX * std::max(ZD, QD));
    V->lg.resize((size_t)TMAX * c.V);
    V->fproj.resize((size_t)TMAX * FP);
    V->cv.resize(C);
    V->gp.resize(c.NL);
    V->cvp.resize(c.NL);
    V->altc.resize(c.NL);
    V->alts.resize(c.NL);
    V->snapc.resize(c.NL);
    V->snaps.resize(c.NL);
    for (int L = 0; L < c.NL; L++)
      if (!lw[L].full) V->gp[L].resize((size_t)TMAX * G);
  }
  Ver& W8 = *V;
  W8.old = verify_old;
  for (int L = 0; L < c.NL; L++)  // the buffers of the path this pass runs (the state's sizes)
    if (!lw[L].full) {
      auto& a = W8.old ? W8.snapc[L] : W8.altc[L];
      auto& b = W8.old ? W8.snaps[L] : W8.alts[L];
      if (a.size() != S->ls[L].conv.size()) a.resize(S->ls[L].conv.size());
      if (b.size() != S->ls[L].S.size()) b.resize(S->ls[L].S.size());
      if (!W8.old && W8.cvp[L].size() != (size_t)TMAX * C) W8.cvp[L].resize((size_t)TMAX * C);
    }
  const bool pack = n >= (W8.old ? PACK_MIN_OLD : PACK_MIN) && prec == Prec::I16 && H % 256 == 0 && ZD % 256 == 0 && QD % 256 == 0;
  auto jb_bytes = [&](const JobN* j, int nj) {
    for (int i = 0; i < nj; i++) W8.b_res += (double)j[i].w->bytes();
  };
  const int pos0 = S->pos;
  auto pk = [&](XQB& b, std::vector<XBuf>& xs, int K) -> const XQB* {
    if (!pack) return nullptr;
    b.reserve(K, n);
    for (int i = 0; i < n; i++) xqb_set_from(b, i, xs[i].q);
    xqb_pad(b);
    return &b;
  };
  const XQ* xa[TMAX];
  const XQ* xo[TMAX];
  const XQ* xh[TMAX];
  const float* hf[TMAX];
  float* xs[TMAX];
  float* hs[TMAX];
  float* ys[TMAX];
  for (int i = 0; i < n; i++) {
    xa[i] = &W8.xa[i].q;
    xo[i] = &W8.xo[i].q;
    xh[i] = &W8.xh[i].q;
    xs[i] = W8.x.data() + (size_t)i * H;
    hs[i] = W8.h.data() + (size_t)i * H;
    hf[i] = hs[i];
    ys[i] = W8.y.data() + (size_t)i * H;
  }
  auto t0 = clk::now();
  for (int i = 0; i < n; i++) rows.row(ids[i], xs[i]);
  T->embed += ms_since(t0);
  core->pos = pos0;
  Cache::CapsOff caps_off(core->cache);  // the pass names the next layer's union: per-layer caps would refuse it
  core->chunk_start(n, ids, xs);  // names inside the pass (the routers read the embeddings now, before the layers)
  Cache::ChunkClock chunk_clock(core->cache, core->tok_clock + n - 1);  // the policy's statistics in time order at the end
  for (int L = 0; L < c.NL; L++) {
    const Lw& W = lw[L];
    auto ta = clk::now();
    {  // the positions' norms side by side (rmsnorm_n: each position's rmsnorm bits)
      const float* nw[TMAX];
      for (int i = 0; i < n; i++) nw[i] = W.ln_in;
      rmsnorm_n(xs, nw, n, H, c.eps, hs);
    }
    for (int i = 0; i < n; i++) W8.xa[i].prep(hs[i], H);
    const XQB* pA = pk(W8.pa, W8.xa, H);
    float* ob[TMAX];
    for (int i = 0; i < n; i++) ob[i] = W8.ob.data() + (size_t)i * std::max(ZD, QD);
    if (!W.full) {
      float *qkv[TMAX], *z[TMAX], *bb[TMAX], *aa[TMAX];
      for (int i = 0; i < n; i++) {
        qkv[i] = W8.gp[L].data() + (size_t)i * G;
        z[i] = qkv[i] + C;
        bb[i] = z[i] + ZD;
        aa[i] = bb[i] + c.NVH;
      }
      JobN jb[4] = {{&W.qkv, n, xa, hf, qkv, pA}, {&W.z, n, xa, hf, z, pA}, {&W.wb, n, xa, hf, bb, nullptr}, {&W.wa, n, xa, hf, aa, nullptr}};
      run_rows_n(pool, jb, 4, 32);
      jb_bytes(jb, 4);
      auto tb = clk::now();
      T->attn_proj += ms_between(ta, tb);
      LState& St = S->ls[L];
      const double sb = (double)St.S.size() * sizeof(float), cb = (double)St.conv.size() * sizeof(float);
      auto tc = tb;
      if (W8.old) {
        // the state at the step's start, for a rollback
        pcopy(pool, W8.snapc[L].data(), St.conv.data(), St.conv.size());
        pcopy(pool, W8.snaps[L].data(), St.S.data(), St.S.size());
        tc = clk::now();
        W8.ms_snap += ms_between(tb, tc);
        W8.b_copy += sb + cb;
        for (int i = 0; i < n; i++) {  // positions in order through the decode path's phases
          gdn::Ctx X;
          lin_ctx(X, L, qkv[i], z[i], bb[i], aa[i], W8.cv.data(), ob[i], false);
          pool.run(gdn::phase_conv, &X);
          pool.run(gdn::phase_heads, &X);
          W8.xo[i].prep(ob[i], ZD);
        }
        W8.b_stream += n * sb;
      } else {
        // A = the committed state (read only), B = the state after the pass's positions
        const float* Ac = St.conv.data();
        float* Bc = W8.altc[L].data();
        const float* Asn = St.S.data();
        float* Bs = W8.alts[L].data();
        const float* wcv = W.conv;
        float* cvl = W8.cvp[L].data();
        const int CK = c.CK, CB = 256;
        parallel_for(pool, (C + CB - 1) / CB, [&](int blk) {  // the convolution: channel by channel, positions in order
          const int c0 = blk * CB, c1 = std::min(C, c0 + CB);
          for (int ch = c0; ch < c1; ch++) cvl[ch] = gdn::conv_channel_io(Ac, Bc, wcv + (size_t)ch * CK, qkv[0][ch], C, CK, ch);
          for (int i = 1; i < n; i++)
            for (int ch = c0; ch < c1; ch++) cvl[(size_t)i * C + ch] = gdn::conv_channel(Bc, wcv + (size_t)ch * CK, qkv[i][ch], C, CK, ch);
        });
        const float* cvs[TMAX];
        for (int i = 0; i < n; i++) cvs[i] = cvl + (size_t)i * C;
        const float* const* cz = z;
        const float* const* cb2 = bb;
        const float* const* ca = aa;
        const gdn::Weights gw{W.conv, W.A_log, W.dt_bias, W.gnorm};
        parallel_for(pool, c.NVH, [&](int hh) { lin_head_n(gd, gw, hh, n, cvs, cb2, ca, cz, Asn, Bs, ob); });  // head by head
        for (int i = 0; i < n; i++) W8.xo[i].prep(ob[i], ZD);
        W8.b_stream += sb;
      }
      auto td = clk::now();
      T->attn_core += ms_between(tc, td);
      JobN jo = {&W.o, n, xo, nullptr, ys, pk(W8.po, W8.xo, ZD)};
      run_rows_n(pool, &jo, 1, 32);
      jb_bytes(&jo, 1);
      T->attn_out += ms_since(td);
    } else {
      float *qg[TMAX], *kp[TMAX], *vp[TMAX];
      for (int i = 0; i < n; i++) {
        qg[i] = W8.fproj.data() + (size_t)i * FP;
        kp[i] = qg[i] + 2 * QD;
        vp[i] = kp[i] + KD;
      }
      JobN jb[3] = {{&W.q, n, xa, nullptr, qg, pA}, {&W.k, n, xa, nullptr, kp, pA}, {&W.v, n, xa, nullptr, vp, pA}};
      run_rows_n(pool, jb, 3, 32);
      jb_bytes(jb, 3);
      auto tb = clk::now();
      T->attn_proj += ms_between(ta, tb);
      attn_full_multi(L, pos0, n, qg, kp, vp, ob, W8.xo.data());
      auto tc = clk::now();
      T->attn_core += ms_between(tb, tc);
      JobN jo = {&W.o, n, xo, nullptr, ys, pk(W8.po, W8.xo, QD)};
      run_rows_n(pool, &jo, 1, 32);
      jb_bytes(&jo, 1);
      T->attn_out += ms_since(tc);
    }
    for (int i = 0; i < n; i++) {
      float* x = xs[i];
      const float* y = ys[i];
      for (int j = 0; j < H; j++) x[j] += y[j];
    }
    {
      const float* nw[TMAX];
      for (int i = 0; i < n; i++) nw[i] = W.ln_post;
      rmsnorm_n(xs, nw, n, H, c.eps, hs);
    }
    core->moe_chunk(L, n, hf, ys, views[L]);
    for (int i = 0; i < n; i++) {
      float* x = xs[i];
      const float* y = ys[i];
      for (int j = 0; j < H; j++) x[j] += y[j];
    }
  }
  // every position's logits, as the single step's head computes them
  auto th = clk::now();
  {
    const float* nw[TMAX];
    for (int i = 0; i < n; i++) nw[i] = fnorm;
    rmsnorm_n(xs, nw, n, H, c.eps, hs);
  }
  for (int i = 0; i < n; i++) W8.xh[i].prep(hs[i], H);
  float* lo[TMAX];
  for (int i = 0; i < n; i++) lo[i] = W8.lg.data() + (size_t)i * c.V;
  if (fast_head && !W8.old && (want == Want::GREEDY || (want == Want::TOPK && head_topk > 0))) {
    // the screened head, as a single step under this Want screens it: the exact rows that can be in the top k, -inf for
    // the rest; the planes are read once for every position
    const double hb0 = sh.bytes;
    sh.run_n(pool, xh, n, want == Want::GREEDY ? 1 : head_topk, lo, ver_best, nullptr);
    W8.b_head += sh.bytes - hb0;
  } else {
    JobN jh = {&head, n, xh, nullptr, lo, pk(W8.ph, W8.xh, H)};
    run_rows_n(pool, &jh, 1, 64);
    W8.b_head += (double)head.bytes();
    for (int i = 0; i < n; i++) ver_best[i] = -1;
  }
  T->head += ms_since(th);
  S->pos += n;
  core->chunk_end(n);
  W8.n = n;
  W8.pos0 = pos0;
  W8.live = true;
  W8.steps++;
  W8.positions += n;
  return true;
}

void Qwen35::commit(State& s0, int a) {
  S = (QState*)&s0;
  if (!V || !V->live) return;
  Ver& W8 = *V;
  W8.live = false;
  const int n = W8.n;
  a = std::max(1, std::min(a, n));
  const int C = c.conv_dim(), ZD = c.NVH * c.DV, G = C + ZD + 2 * c.NVH, CK = c.CK;
  Pool& pool = *core->pool;
  if (W8.old && a < n) {
    auto t0 = clk::now();
    for (int L = 0; L < c.NL; L++) {
      if (lw[L].full) continue;
      LState& St = S->ls[L];
      pcopy(pool, St.conv.data(), W8.snapc[L].data(), St.conv.size());
      pcopy(pool, St.S.data(), W8.snaps[L].data(), St.S.size());
      W8.b_copy += (double)(St.conv.size() + St.S.size()) * sizeof(float);
      for (int i = 0; i < a; i++) {
        float* qkv = W8.gp[L].data() + (size_t)i * G;
        float* z = qkv + C;
        float* bb = z + ZD;
        float* aa = bb + c.NVH;
        gdn::Ctx X;
        lin_ctx(X, L, qkv, z, bb, aa, W8.cv.data(), W8.ob.data(), false);
        pool.run(gdn::phase_conv, &X);
        pool.run(gdn::phase_heads, &X);
      }
      W8.b_stream += (double)a * St.S.size() * sizeof(float);
    }
    W8.ms_replay += ms_since(t0);
  } else if (!W8.old) {
    auto t0 = clk::now();
    for (int L = 0; L < c.NL; L++) {
      if (lw[L].full) continue;
      LState& St = S->ls[L];
      if (a < n) {  // B again from A for positions 0..a-1
        // the conv state after a positions: the last CK - 1 rows of [A's rows, inputs 0 .. a-1]
        const int R = CK - 1;
        float* Bc = W8.altc[L].data();
        for (int r = 0; r < R; r++) {
          const int idx = a + r;  // index into [A rows 0 .. R-1, inputs 0 ..]
          if (idx < R) std::memcpy(Bc + (size_t)r * C, St.conv.data() + (size_t)idx * C, sizeof(float) * C);
          else
            for (int ch = 0; ch < C; ch++) Bc[(size_t)r * C + ch] = W8.gp[L][(size_t)(idx - R) * G + ch];
        }
        W8.b_copy += (double)R * C * sizeof(float);
        const float* cvs[TMAX];
        const float* bbs[TMAX];
        const float* aas[TMAX];
        for (int i = 0; i < a; i++) {
          cvs[i] = W8.cvp[L].data() + (size_t)i * C;
          bbs[i] = W8.gp[L].data() + (size_t)i * G + C + ZD;
          aas[i] = bbs[i] + c.NVH;
        }
        const Lw& W = lw[L];
        const float* Asn = St.S.data();
        float* Bs = W8.alts[L].data();
        const gdn::Weights gw{W.conv, W.A_log, W.dt_bias, W.gnorm};
        parallel_for(pool, c.NVH, [&](int hh) { lin_head_n(gd, gw, hh, a, cvs, bbs, aas, nullptr, Asn, Bs, nullptr); });
        W8.b_stream += (double)St.S.size() * sizeof(float);
      }
      St.conv.swap(W8.altc[L]);
      St.S.swap(W8.alts[L]);
    }
    if (a < n) W8.ms_replay += ms_since(t0);
  }
  if (a < n) {
    S->pos = W8.pos0 + a;
    core->tok_clock -= n - a;
    W8.rollbacks++;
  }
  std::memcpy(lgts.data(), W8.lg.data() + (size_t)(a - 1) * c.V, sizeof(float) * c.V);
  greedy_ = fast_head ? ver_best[a - 1] : -1;  // (the screened head's argmax of the last kept position; -1: the full head's logits)
  core->chunk_commit(a);
}

std::string Qwen35::verify_json() const {
  if (!V) return "{}";
  char b[640];
  const double p = std::max<long>(1, V->steps);
  snprintf(b, sizeof b,
           "{\"steps\":%ld,\"positions\":%ld,\"rollbacks\":%ld,\"old_path\":%s,\"ms_state_copy\":%.2f,\"ms_rollback\":%.2f,\"mb_per_pass\":{"
           "\"state_copied\":%.3f,\"state_streamed\":%.3f,\"resident\":%.3f,\"head\":%.3f,\"touched\":%.3f}}",
           V->steps, V->positions, V->rollbacks, verify_old ? "true" : "false", V->ms_snap, V->ms_replay, V->b_copy / 1e6 / p,
           V->b_stream / 1e6 / p, V->b_res / 1e6 / p, V->b_head / 1e6 / p, (V->b_copy + V->b_stream + V->b_res + V->b_head) / 1e6 / p);
  return b;
}

}  // namespace qwen35
}  // namespace dm
