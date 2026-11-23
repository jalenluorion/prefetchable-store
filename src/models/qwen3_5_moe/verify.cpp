// Multi-position verify of drafted tokens for Qwen3.6-35B-A3B: up to TMAX positions from the current state in one pass,
// every position's logits kept, then commit(a) keeps the first a and rolls the rest back.
//
// Exactness: every operation is one the single step (qwen3_5_moe.cpp) runs, through the same compiled code where the
// arithmetic is not a plain shared kernel:
//   - resident matrices (attention projections, o projections, head): run_rows_n, whose multi-vector and packed i8mm
//     kernels are bit-identical per position to the one-vector mv the single step calls; bf16 rows through dot_bf16 per
//     position (the same function);
//   - the Gated DeltaNet conv and heads: the decode path's own phase_conv / phase_lin_heads, run once per position in order;
//   - full attention: attn_full_core, the one body decode calls too (KV write at the position, attention over 0..pos);
//   - the MoE block: Core::moe_chunk (each position's released routing, expert arithmetic and combination order are the
//     single step's; the union of the positions' experts is read once);
//   - norms and residual adds: the same inline helpers, per position.
// Rollback: the DeltaNet conv and recurrent states of every layer are copied at the step's start, and each position's
// projections (the inputs of the conv and head phases) are kept; commit(a < n) copies the start state back and runs the
// two phases again on positions 0..a-1, which reproduces their bits. The KV of a rolled-back position is simply
// overwritten later (append-only), the position counter and the cache's clock go back.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>

#include "dmeng/kernels_batch.h"
#include "qwen3_5_moe.h"

namespace dm {
namespace qwen35 {

namespace {
constexpr int PACK_MIN = 6;  // positions from which the resident matrices go through the packed i8mm gemm

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
  std::vector<std::vector<float>> snapc, snaps;  // [NL] conv / recurrent state at the step's start
  double ms_snap = 0, ms_replay = 0;
  long steps = 0, rollbacks = 0;
};

const float* Qwen35::logits_at(int i) const { return V ? V->lg.data() + (size_t)i * c.V : nullptr; }

bool Qwen35::verify(State& s0, const int32_t* ids, int n) {
  S = (QState*)&s0;
  if (n < 1 || n > TMAX || core->ns.on || (core->force && S->pos < core->force_n) || S->pos + n > S->ctx) return false;
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
    V->snapc.resize(c.NL);
    V->snaps.resize(c.NL);
    for (int L = 0; L < c.NL; L++)
      if (!lw[L].full) {
        V->gp[L].resize((size_t)TMAX * G);
        V->snapc[L].resize(S->ls[L].conv.size());
        V->snaps[L].resize(S->ls[L].S.size());
      }
  }
  Ver& W8 = *V;
  const bool pack = n >= PACK_MIN && prec == Prec::I16 && H % 256 == 0 && ZD % 256 == 0 && QD % 256 == 0;
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
    for (int i = 0; i < n; i++) {
      rmsnorm(xs[i], W.ln_in, H, c.eps, hs[i]);
      W8.xa[i].prep(hs[i], H);
    }
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
      auto tb = clk::now();
      T->attn_proj += ms_between(ta, tb);
      // the state at the step's start, for a rollback
      LState& St = S->ls[L];
      pcopy(pool, W8.snapc[L].data(), St.conv.data(), St.conv.size());
      pcopy(pool, W8.snaps[L].data(), St.S.data(), St.S.size());
      auto tc = clk::now();
      W8.ms_snap += ms_between(tb, tc);
      for (int i = 0; i < n; i++) {  // positions in order through the decode path's phases
        LinCtx X{this, L, qkv[i], z[i], bb[i], aa[i], W8.cv.data(), ob[i], false};
        pool.run(phase_conv, &X);
        pool.run(phase_lin_heads, &X);
        W8.xo[i].prep(ob[i], ZD);
      }
      auto td = clk::now();
      T->attn_core += ms_between(tc, td);
      JobN jo = {&W.o, n, xo, nullptr, ys, pk(W8.po, W8.xo, ZD)};
      run_rows_n(pool, &jo, 1, 32);
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
      auto tb = clk::now();
      T->attn_proj += ms_between(ta, tb);
      attn_full_multi(L, pos0, n, qg, kp, vp, ob, W8.xo.data());
      auto tc = clk::now();
      T->attn_core += ms_between(tb, tc);
      JobN jo = {&W.o, n, xo, nullptr, ys, pk(W8.po, W8.xo, QD)};
      run_rows_n(pool, &jo, 1, 32);
      T->attn_out += ms_since(tc);
    }
    for (int i = 0; i < n; i++) {
      float* x = xs[i];
      const float* y = ys[i];
      for (int j = 0; j < H; j++) x[j] += y[j];
      rmsnorm(x, W.ln_post, H, c.eps, hs[i]);
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
  for (int i = 0; i < n; i++) {
    rmsnorm(xs[i], fnorm, H, c.eps, hs[i]);
    W8.xh[i].prep(hs[i], H);
  }
  float* lo[TMAX];
  for (int i = 0; i < n; i++) lo[i] = W8.lg.data() + (size_t)i * c.V;
  JobN jh = {&head, n, xh, nullptr, lo, pk(W8.ph, W8.xh, H)};
  run_rows_n(pool, &jh, 1, 64);
  T->head += ms_since(th);
  S->pos += n;
  core->chunk_end(n);
  W8.n = n;
  W8.pos0 = pos0;
  W8.live = true;
  W8.steps++;
  return true;
}

void Qwen35::commit(State& s0, int a) {
  S = (QState*)&s0;
  if (!V || !V->live) return;
  Ver& W8 = *V;
  W8.live = false;
  const int n = W8.n;
  a = std::max(1, std::min(a, n));
  if (a < n) {
    auto t0 = clk::now();
    Pool& pool = *core->pool;
    const int C = c.conv_dim(), ZD = c.NVH * c.DV, G = C + ZD + 2 * c.NVH;
    for (int L = 0; L < c.NL; L++) {
      if (lw[L].full) continue;
      LState& St = S->ls[L];
      pcopy(pool, St.conv.data(), W8.snapc[L].data(), St.conv.size());
      pcopy(pool, St.S.data(), W8.snaps[L].data(), St.S.size());
      for (int i = 0; i < a; i++) {
        float* qkv = W8.gp[L].data() + (size_t)i * G;
        float* z = qkv + C;
        float* bb = z + ZD;
        float* aa = bb + c.NVH;
        LinCtx X{this, L, qkv, z, bb, aa, W8.cv.data(), W8.ob.data(), false};
        pool.run(phase_conv, &X);
        pool.run(phase_lin_heads, &X);
      }
    }
    S->pos = W8.pos0 + a;
    core->tok_clock -= n - a;
    W8.ms_replay += ms_since(t0);
    W8.rollbacks++;
  }
  std::memcpy(lgts.data(), W8.lg.data() + (size_t)(a - 1) * c.V, sizeof(float) * c.V);
  core->chunk_commit(a);
}

std::string Qwen35::verify_json() const {
  if (!V) return "{}";
  char b[256];
  snprintf(b, sizeof b, "{\"steps\":%ld,\"rollbacks\":%ld,\"ms_state_copy\":%.2f,\"ms_rollback\":%.2f}", V->steps, V->rollbacks, V->ms_snap, V->ms_replay);
  return b;
}

}  // namespace qwen35
}  // namespace dm
