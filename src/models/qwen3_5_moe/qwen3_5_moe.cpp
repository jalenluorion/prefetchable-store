// Qwen3.6-35B-A3B (qwen3_5_moe text model), batch 1, on the CPU. Math as mlx-vlm 0.7.12's qwen3_5_moe:
// x + GDN/attention(input_layernorm(x)), then + MoE(post_attention_layernorm(h)).
// 30 Gated DeltaNet layers (conv state, fp32 recurrent state, silu-gated RMSNorm) and 10 gated full-attention layers
// (partial RoPE 64 of 256, split-K above 512 positions, fp16 KV cache); the MoE block (softmax router, top 8,
// renormalised, shared expert with a sigmoid gate) runs in the core. Precision: residual stream, norms, router,
// recurrent state and every intermediate in fp32; weights as stored (8-bit resident, 4-bit experts, bf16 router and
// small gates); matvec activations I16 by default.
#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <sstream>

#include "qwen3_5_moe.h"
#include "attn_split.h"

namespace dm {
namespace qwen35 {

// ------------------------------------------------------------------------------------------------ load
bool Qwen35::load(const LoadOpts& op, std::string& err) {
  const std::string res = op.resident.empty() ? op.dir + "/dmeng_resident.bin" : op.resident;
  if (!R.load(res, op.res_mem, err)) return false;
  for (auto& kv : op.cfg) R.cfg[kv.first] = kv.second;  // the manifest's configuration, where there is one
  c.from(R);
  lw.resize(c.NL);
  for (int L = 0; L < c.NL; L++) {
    Lw& w = lw[L];
    const std::string s = std::to_string(L) + ".";
    w.full = c.types[L] == 'F';
    w.ln_in = (const float*)R.ptr(s + "ln_in");
    w.ln_post = (const float*)R.ptr(s + "ln_post");
    if (!w.full) {
      w.qkv = R.weight(s + "qkv");
      w.z = R.weight(s + "z");
      w.o = R.weight(s + "o");
      w.wa = R.weight(s + "a");
      w.wb = R.weight(s + "b");
      w.conv = (const float*)R.ptr(s + "conv");
      w.A_log = (const float*)R.ptr(s + "A_log");
      w.dt_bias = (const float*)R.ptr(s + "dt_bias");
      w.gnorm = (const float*)R.ptr(s + "gnorm");
    } else {
      w.q = R.weight(s + "q");
      w.k = R.weight(s + "k");
      w.v = R.weight(s + "v");
      w.o = R.weight(s + "o");
      w.qn = (const float*)R.ptr(s + "qn");
      w.kn = (const float*)R.ptr(s + "kn");
    }
    w.router = R.weight(s + "router");
    w.sg = R.weight(s + "sg");
    w.su = R.weight(s + "su");
    w.sd = R.weight(s + "sd");
    w.shgate = R.bf16_row(s + "shgate");
  }
  gd.NKH = c.NKH;
  gd.NVH = c.NVH;
  gd.DK = c.DK;
  gd.DV = c.DV;
  gd.CK = c.CK;
  gd.eps = c.eps;
  gd.sigmoid_gate = false;
  if (!gd.ok(err)) return false;
  fnorm = (const float*)R.ptr("norm");
  head = R.weight("head");
  if (!rows.open(c.emb_file[0] == '/' ? c.emb_file : op.dir + "/" + c.emb_file, c.emb_off, c.H, err)) return false;
  // what the core needs
  mi.family = "qwen3_5_moe";
  mi.has_verify = true;
  mi.hidden = c.H;
  mi.vocab = mi.vocab_out = c.V;
  mi.layers = mi.steps = c.NL;
  mi.step_layer.resize(c.NL);
  for (int L = 0; L < c.NL; L++) mi.step_layer[L] = L;
  mi.step_pass.assign(c.NL, -1);
  mi.experts = c.NE;
  mi.top_k = c.K;
  mi.shared_inter = c.SI;
  ExpertFormat f;
  f.fmt = Fmt::AFF4;
  f.rec = c.rec;
  f.H = c.H;
  f.I = c.I;
  f.fused = false;
  f.gu_c = c.off[0], f.gu_s = c.soff[0], f.gu_b = c.boff[0];
  f.up_c = c.off[1], f.up_s = c.soff[1], f.up_b = c.boff[1];
  f.dn_c = c.off[2], f.dn_s = c.soff[2], f.dn_b = c.boff[2];
  f.touched = c.rec;
  mi.formats = {f};
  mi.layer_fmt.assign(c.NL, 0);
  mi.pool_share = {1.0};
  mi.layer_file = c.layer_file;
  mi.has_ref_path = false;
  views.resize(c.NL);
  for (int L = 0; L < c.NL; L++) {
    MoeView& v = views[L];
    v.rule = MoeView::SOFTMAX_TOPK_RENORM;
    v.renorm = c.norm_topk != 0;
    v.router = &lw[L].router;
    v.act = MoeView::SWIGLU;
    v.shared = true;
    v.sh_gate = &lw[L].sg;
    v.sh_up = &lw[L].su;
    v.sh_down = &lw[L].sd;
    v.sh_scalar_gate = &lw[L].shgate;
  }
  return true;
}

bool Qwen35::bind(Core& cr, std::string& err) {
  core = &cr;
  prec = cr.o.prec;
  cr.views.resize(c.NL);
  for (int L = 0; L < c.NL; L++) cr.views[L] = &views[L];
  T = &cr.t;
  const int C = c.conv_dim();
  const int maxk = std::max({c.H, c.NVH * c.DV, c.NQ * c.HD, c.SI, c.I});
  xin.init(maxk, prec);
  xatt.init(maxk, prec);
  x.resize(c.H);
  h.resize(c.H);
  hm.resize(c.H);
  y.resize(c.H);
  emb.resize(c.H);
  buf.resize(std::max((size_t)2 * c.NQ * c.HD + 2 * c.NKV * c.HD, (size_t)2 * C + c.NVH * c.DV + 2 * c.NVH) + 64);
  o.resize(std::max(c.NVH * c.DV, c.NQ * c.HD));
  lgts.resize(c.V);
  fixed_off.assign(c.NL, 0);  // the fixed blob: each DeltaNet layer's conv state then recurrent state, in layer order
  for (int L = 0, off = 0; L < c.NL; L++)
    if (c.types[L] == 'L') {
      fixed_off[L] = (size_t)off;
      off += (int)(((size_t)(c.CK - 1) * C + (size_t)c.NVH * c.DV * c.DK) * sizeof(float));
    }
  return true;
}

std::unique_ptr<State> Qwen35::new_state(int ctx_max, std::string& err) {
  std::unique_ptr<QState> s(new QState());
  s->ctx = ctx_max;
  s->ls.resize(c.NL);
  const int C = c.conv_dim();
  for (int L = 0; L < c.NL; L++) {
    LState& st = s->ls[L];
    if (c.types[L] == 'L') {
      st.conv.assign((size_t)(c.CK - 1) * C, 0.f);
      st.S.assign((size_t)c.NVH * c.DV * c.DK, 0.f);
    } else {
      st.kv_bytes = (size_t)c.NKV * ctx_max * c.HD * 2;
      // plain anonymous memory: only the pages a context touches become resident
      st.K = (_Float16*)alloc_region(st.kv_bytes, Mem::PLAIN, &err);
      st.V = (_Float16*)alloc_region(st.kv_bytes, Mem::PLAIN, &err);
      if (!st.K || !st.V) return nullptr;
    }
  }
  return s;
}

void Qwen35::reset(State& s0) {
  QState& s = (QState&)s0;
  for (auto& st : s.ls) {
    std::fill(st.conv.begin(), st.conv.end(), 0.f);
    std::fill(st.S.begin(), st.S.end(), 0.f);
  }
  s.pos = 0;
}

bool Qwen35::seek(State& s0, int p) {
  QState& s = (QState&)s0;
  if (p >= s.ctx) return false;
  s.pos = p;
  for (auto& st : s.ls)  // real pages for the skipped positions (an untouched page would read the shared zero page)
    if (st.K)
      for (int j = 0; j < c.NKV; j++) {
        std::memset(st.K + (size_t)j * s.ctx * c.HD, 0, (size_t)p * c.HD * 2);
        std::memset(st.V + (size_t)j * s.ctx * c.HD, 0, (size_t)p * c.HD * 2);
      }
  return true;
}

// The state for the serving layer (include/dmeng/lm.h): the fixed part = every DeltaNet layer's conv state and fp32
// recurrent state, as computed; the positional part = the fp16 K and V of the full-attention layers, per position.
size_t Qwen35::fixed_bytes(const State&) const {
  size_t n = 0;
  for (int L = 0; L < c.NL; L++)
    if (c.types[L] == 'L') n += ((size_t)(c.CK - 1) * c.conv_dim() + (size_t)c.NVH * c.DV * c.DK) * 4;
  return n;
}
size_t Qwen35::pos_bytes() const {
  int nf = 0;
  for (int L = 0; L < c.NL; L++) nf += c.types[L] == 'F';
  return (size_t)nf * c.NKV * c.HD * 2 * 2;
}
bool Qwen35::save_fixed(const State& s0, void* dst) const {
  const QState& s = (const QState&)s0;
  uint8_t* d = (uint8_t*)dst;
  for (auto& st : s.ls) {
    std::memcpy(d, st.conv.data(), st.conv.size() * 4);
    d += st.conv.size() * 4;
    std::memcpy(d, st.S.data(), st.S.size() * 4);
    d += st.S.size() * 4;
  }
  return true;
}
bool Qwen35::load_fixed(State& s0, const void* src, int p) {
  QState& s = (QState&)s0;
  const uint8_t* d = (const uint8_t*)src;
  for (auto& st : s.ls) {
    std::memcpy(st.conv.data(), d, st.conv.size() * 4);
    d += st.conv.size() * 4;
    std::memcpy(st.S.data(), d, st.S.size() * 4);
    d += st.S.size() * 4;
  }
  s.pos = p;
  return true;
}
bool Qwen35::save_pos(const State& s0, int p0, int p1, void* dst) const {
  const QState& s = (const QState&)s0;
  if (p0 < 0 || p1 > s.ctx || p0 > p1) return false;
  uint8_t* d = (uint8_t*)dst;
  const size_t row = (size_t)c.HD * 2;
  for (int p = p0; p < p1; p++)
    for (auto& st : s.ls)
      if (st.K)
        for (int j = 0; j < c.NKV; j++) {
          std::memcpy(d, st.K + ((size_t)j * s.ctx + p) * c.HD, row);
          std::memcpy(d + row, st.V + ((size_t)j * s.ctx + p) * c.HD, row);
          d += 2 * row;
        }
  return true;
}
bool Qwen35::load_pos(State& s0, int p0, int p1, const void* src) {
  QState& s = (QState&)s0;
  if (p0 < 0 || p1 > s.ctx || p0 > p1) return false;
  const uint8_t* d = (const uint8_t*)src;
  const size_t row = (size_t)c.HD * 2;
  for (int p = p0; p < p1; p++)
    for (auto& st : s.ls)
      if (st.K)
        for (int j = 0; j < c.NKV; j++) {
          std::memcpy(st.K + ((size_t)j * s.ctx + p) * c.HD, d, row);
          std::memcpy(st.V + ((size_t)j * s.ctx + p) * c.HD, d + row, row);
          d += 2 * row;
        }
  return true;
}

// ------------------------------------------------------------------------------------------------ Gated DeltaNet
// (the convolution channel, the value head and the fork-joins: src/models/common/gdn.h, shared with qwen4_exp)
void Qwen35::lin_ctx(gdn::Ctx& X, int L, float* qkv, float* z, float* bb, float* aa, float* cv, float* o, bool quant_heads) {
  const Lw& W = lw[L];
  X.d = &gd;
  X.w = {W.conv, W.A_log, W.dt_bias, W.gnorm};
  X.conv_st = S->ls[L].conv.data();
  X.S = S->ls[L].S.data();
  X.qkv = qkv;
  X.z = z;
  X.bb = bb;
  X.aa = aa;
  X.cv = cv;
  X.o = o;
  X.xq = &xatt.q;
  X.quant_heads = quant_heads;
}

void Qwen35::attn_linear(int L, const float* hin, float* rout) {
  const Lw& W = lw[L];
  const int C = c.conv_dim(), ZD = c.NVH * c.DV;
  auto t0 = clk::now();
  float* qkv = buf.data();
  float* z = qkv + C;
  float* bb = z + ZD;
  float* aa = bb + c.NVH;
  float* cv = aa + c.NVH;
  stage_mark(ST_ATTN_PROJ);
  Job jb[4] = {{&W.qkv, &xin.q, hin, qkv}, {&W.z, &xin.q, hin, z}, {&W.wb, &xin.q, hin, bb}, {&W.wa, &xin.q, hin, aa}};
  {  // the input's quantisation inside the projections' fork-join
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {hin};
    const int ks[1] = {c.H};
    run_rows_q(*core->pool, jb, 4, 32, false, nullptr, qs, xs, ks, 1);
  }
  auto t1 = clk::now();
  T->attn_proj += ms_between(t0, t1);
  stage_mark(ST_ATTN_CORE);
  gdn::Ctx X;
  lin_ctx(X, L, qkv, z, bb, aa, cv, o.data(), c.DV % XB == 0);
  if (c.NKH > 64) {
    fprintf(stderr, "more than 64 key heads\n");
    abort();
  }
  for (int j = 0; j < c.NKH; j++) X.qk_done[j].store(0, std::memory_order_relaxed);
  xatt.reset(ZD);
  if (X.quant_heads) {  // the output projection in the heads' fork-join
    X.wo = &W.o;
    X.rout = rout;
    const int U = (W.o.rows + 7) / 8, tail = 4 * core->pool->size();
    X.o_nb = U > tail ? (U - tail) / 4 : 0;
    X.o_chunks = X.o_nb + (U - 4 * X.o_nb);
    X.heads_left.store(c.NVH, std::memory_order_relaxed);
  }
  core->pool->run(gdn::phase_conv_heads, &X);
  auto t2 = clk::now();
  if (X.o_chunks) {  // the output projection ran in the heads' fork-join: its share (after the last head) is attn_out's
    const auto th = clk::time_point(std::chrono::duration_cast<clk::duration>(std::chrono::nanoseconds(X.heads_done_ns.load())));
    const double core_ms = std::min(ms_between(t1, t2), std::max(0.0, ms_between(t1, th)));
    T->attn_core += core_ms;
    T->attn_out += ms_between(t1, t2) - core_ms;
  } else {
    T->attn_core += ms_between(t1, t2);
  }
  if (!X.quant_heads) {
    xatt.prep(o.data(), ZD);
    Job jo = {&W.o, &xatt.q, nullptr, rout};
    run_rows(*core->pool, &jo, 1, 32);
  }
  T->attn_out += ms_since(t2);
}

// ------------------------------------------------------------------------------------------------ full attention
struct FullCtx {
  Qwen35* M;
  int L, T;  // T = number of positions including the new one
  const float* qg;
  float *cosv, *sinv;
  float* o;
  XQ* xq;  // the quantised output (the o projection's input)
  alignas(128) std::atomic<int> next{0};
};
void rope(float* x, const float* cs, const float* sn, int rot) {
  const int half = rot / 2;
  for (int i = 0; i < half; i++) {
    const float a = x[i], b = x[i + half];
    x[i] = a * cs[i] - b * sn[i];
    x[i + half] = b * cs[i] + a * sn[i];
  }
}
void phase_full_heads(void* p, int, int) {
  FullCtx& X = *(FullCtx*)p;
  Qwen35& M = *X.M;
  const QCfg& c = M.c;
  const Lw& W = M.lw[X.L];
  const int HD = c.HD, Tn = X.T, rep = c.NQ / c.NKV, ctx = M.S->ctx;
  static thread_local std::vector<float> sc;
  if ((int)sc.size() < Tn) sc.resize(Tn + 1024);
  const float scale = 1.f / std::sqrt((float)HD);
  for (int h; (h = X.next.fetch_add(1, std::memory_order_relaxed)) < c.NQ;) {
    float q[1024];
    _Float16 q16[1024];
    const float* qs = X.qg + (size_t)h * 2 * HD;
    float ss = 0;
    for (int i = 0; i < HD; i++) ss += qs[i] * qs[i];
    const float r = 1.f / std::sqrt(ss / HD + c.eps);
    for (int i = 0; i < HD; i++) q[i] = qs[i] * r * W.qn[i];
    rope(q, X.cosv, X.sinv, c.ROT);
    for (int i = 0; i < HD; i++) q16[i] = (_Float16)q[i];  // products against fp16 keys (FMLAL), fp32 sums
    const int kvh = h / rep;
    const _Float16* K = M.S->ls[X.L].K + (size_t)kvh * ctx * HD;
    const _Float16* V = M.S->ls[X.L].V + (size_t)kvh * ctx * HD;
    float mx = -INFINITY;
    for (int t = 0; t < Tn; t++) {
      const _Float16* kr = K + (size_t)t * HD;
      float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
      for (int i = 0; i < HD; i += 8) {
        float16x8_t kv = vld1q_f16((const float16_t*)kr + i), qv = vld1q_f16((const float16_t*)q16 + i);
        s0 = vfmlalq_low_f16(s0, kv, qv);
        s1 = vfmlalq_high_f16(s1, kv, qv);
      }
      sc[t] = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
      mx = std::max(mx, sc[t]);
    }
    double zs = 0;
    for (int t = 0; t < Tn; t++) {
      sc[t] = std::exp(sc[t] - mx);
      zs += sc[t];
    }
    const float iz = (float)(1.0 / zs);
    float32x4_t acc[256];
    const int NA = HD / 4;
    for (int i = 0; i < NA; i++) acc[i] = vdupq_n_f32(0);
    for (int t = 0; t < Tn; t++) {
      const _Float16* vr = V + (size_t)t * HD;
      float32x4_t pv = vdupq_n_f32(sc[t] * iz);
      for (int i = 0; i < HD; i += 8) {
        float16x8_t vv = vld1q_f16((const float16_t*)vr + i);
        acc[i / 4] = vfmaq_f32(acc[i / 4], vcvt_f32_f16(vget_low_f16(vv)), pv);
        acc[i / 4 + 1] = vfmaq_f32(acc[i / 4 + 1], vcvt_high_f32_f16(vv), pv);
      }
    }
    float* o = X.o + (size_t)h * HD;
    const float* gate = qs + HD;
    for (int i = 0; i < HD; i += 4) vst1q_f32(o + i, acc[i / 4]);
    for (int i = 0; i < HD; i++) o[i] *= sigm(gate[i]);
    for (int j = 0; j < HD / XB; j++) xq_block(*X.xq, (h * HD) / XB + j, o + j * XB);
  }
}

// Long contexts: split-K decode attention. The positions are cut into fixed chunks of CHK; a chunk is read once for all
// query heads of its KV head, giving per chunk and head a max, a sum of exp and an unnormalised output; the chunks are
// then combined in chunk order. Fixed chunks + a fixed combination order: the result does not depend on the thread
// count or on scheduling.
constexpr int CHK = 256;
static_assert(CHK == SPLIT_CHK, "phase_split jobs are attn_split.h chunks");
// one split-K job: one compiled body for decode and the verify's batched queries
__attribute__((noinline)) void split_job(const float* q, const _Float16* K, const _Float16* V, int ctx, int T, int ch, int kvh, int NQ, int NKV,
                                         int HD, float* part) {
  split_chunk(q, K, V, ctx, T, ch, kvh, NQ, NKV, HD, part);
}
struct SplitCtx {
  Qwen35* M;
  int L, T, nch;
  const float* q;  // [NQ, HD] normed + rotated queries (fp32)
  float* part;     // [nch, NQ, HD + 2]: max, sum, output
  alignas(128) std::atomic<int> next{0};
};
void phase_split(void* p, int, int) {
  // the job body is attn_split.h's split_chunk: the same arithmetic per element as the loop that was here, with four
  // heads' score dots interleaved and the value sums kept in registers (bit-identical; the serving agent's attn_bench)
  SplitCtx& X = *(SplitCtx*)p;
  Qwen35& M = *X.M;
  const QCfg& c = M.c;
  const int jobs = X.nch * c.NKV;
  for (int jb; (jb = X.next.fetch_add(1, std::memory_order_relaxed)) < jobs;)
    split_job(X.q, M.S->ls[X.L].K, M.S->ls[X.L].V, M.S->ctx, X.T, jb / c.NKV, jb % c.NKV, c.NQ, c.NKV, c.HD, X.part);
}

// the split-K combination, one head per job (each head's arithmetic as before; heads are independent)
struct CombCtx {
  Qwen35* M;
  const SplitCtx* S;
  const float* qg;
  float* o;
  alignas(128) std::atomic<int> next{0};
};
void phase_combine(void* p, int, int) {
  CombCtx& C = *(CombCtx*)p;
  const QCfg& c = C.M->c;
  const int HD = c.HD;
  const float* part = C.S->part;
  for (int hh; (hh = C.next.fetch_add(1, std::memory_order_relaxed)) < c.NQ;) {
    float mx = -INFINITY;
    for (int ch = 0; ch < C.S->nch; ch++) mx = std::max(mx, part[((size_t)ch * c.NQ + hh) * (HD + 2)]);
    std::vector<double> acc(HD, 0.0);
    double z = 0;
    for (int ch = 0; ch < C.S->nch; ch++) {
      const float* pp = part + ((size_t)ch * c.NQ + hh) * (HD + 2);
      const double f = std::exp((double)pp[0] - mx);
      z += f * pp[1];
      for (int i = 0; i < HD; i++) acc[i] += f * pp[2 + i];
    }
    float* oh = C.o + (size_t)hh * HD;
    const float* gate = C.qg + (size_t)hh * 2 * HD + HD;
    for (int i = 0; i < HD; i++) oh[i] = (float)(acc[i] / z) * sigm(gate[i]);
  }
}

void Qwen35::attn_full(int L, const float* hin, float* rout) {
  const Lw& W = lw[L];
  const int HD = c.HD;
  if (S->pos >= S->ctx) {
    fprintf(stderr, "context full (%d tokens)\n", S->ctx);
    abort();
  }
  auto t0 = clk::now();
  float* qg = buf.data();
  float* kp = qg + 2 * c.NQ * HD;
  float* vp = kp + c.NKV * HD;
  stage_mark(ST_ATTN_PROJ);
  Job jb[3] = {{&W.q, &xin.q, nullptr, qg}, {&W.k, &xin.q, nullptr, kp}, {&W.v, &xin.q, nullptr, vp}};
  {  // the input's quantisation inside the projections' fork-join
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {hin};
    const int ks[1] = {c.H};
    run_rows_q(*core->pool, jb, 3, 32, false, nullptr, qs, xs, ks, 1);
  }
  auto t1 = clk::now();
  T->attn_proj += ms_between(t0, t1);
  stage_mark(ST_ATTN_CORE);
  attn_full_core(L, S->pos, qg, kp, vp, o.data(), xatt);
  auto t2 = clk::now();
  T->attn_core += ms_between(t1, t2);
  stage_mark(ST_ATTN_OUT);
  Job jo = {&W.o, &xatt.q, nullptr, rout};
  run_rows(*core->pool, &jo, 1, 32);
  T->attn_out += ms_since(t2);
}

// Everything of a full-attention layer between its projections and its output projection for the position `pos`: K (normed,
// rotated) and V into the cache at pos, the attention over positions 0..pos (one head per job up to 512 positions, split-K
// above), the gate; the output into ob and quantised into xo. One compiled body for decode and the multi-position verify.
// rope tables of a position, and its K (normed, rotated) and V into the cache
__attribute__((noinline)) void Qwen35::attn_full_kv(int L, int pos, const float* kp, const float* vp, float* cosv, float* sinv) {
  const Lw& W = lw[L];
  const int HD = c.HD, ctx = S->ctx;
  if (pos >= ctx) {
    fprintf(stderr, "context full (%d tokens)\n", ctx);
    abort();
  }
  const int half = c.ROT / 2;
  for (int i = 0; i < half; i++) {
    const double inv = std::pow(c.theta, -2.0 * i / c.ROT);
    const float ang = (float)pos * (float)inv;  // mlx: position (fp32) x inv_freq (fp32)
    cosv[i] = std::cos(ang);
    sinv[i] = std::sin(ang);
  }
  LState& St = S->ls[L];
  for (int j = 0; j < c.NKV; j++) {
    float k[1024], ss = 0;
    for (int i = 0; i < HD; i++) {
      k[i] = kp[j * HD + i];
      ss += k[i] * k[i];
    }
    const float r = 1.f / std::sqrt(ss / HD + c.eps);
    for (int i = 0; i < HD; i++) k[i] = k[i] * r * W.kn[i];
    rope(k, cosv, sinv, c.ROT);
    _Float16* kd = St.K + ((size_t)j * ctx + pos) * HD;
    _Float16* vd = St.V + ((size_t)j * ctx + pos) * HD;
    for (int i = 0; i < HD; i++) {
      kd[i] = (_Float16)k[i];
      vd[i] = (_Float16)vp[j * HD + i];
    }
  }
}

// the queries of the long-context path: q_norm, rope, every head
__attribute__((noinline)) void Qwen35::attn_full_q(int L, const float* qg, const float* cosv, const float* sinv, float* qn) {
  const Lw& W = lw[L];
  const int HD = c.HD;
  for (int hh = 0; hh < c.NQ; hh++) {
    const float* qs = qg + (size_t)hh * 2 * HD;
    float ss = 0;
    for (int i = 0; i < HD; i++) ss += qs[i] * qs[i];
    const float r = 1.f / std::sqrt(ss / HD + c.eps);
    float* q = qn + (size_t)hh * HD;
    for (int i = 0; i < HD; i++) q[i] = qs[i] * r * W.qn[i];
    rope(q, cosv, sinv, c.ROT);
  }
}

// the split-K combination of one query's chunks, its gate, the output quantised
void Qwen35::attn_full_combine(int L, int T, const float* qn, float* part, const float* qg, float* ob, XBuf& xo) {
  SplitCtx S2;
  S2.M = this;
  S2.L = L;
  S2.T = T;
  S2.nch = (T + CHK - 1) / CHK;
  S2.q = qn;
  S2.part = part;
  CombCtx CC;
  CC.M = this;
  CC.S = &S2;
  CC.qg = qg;
  CC.o = ob;
  core->pool->run(phase_combine, &CC);
  xo.prep(ob, c.NQ * c.HD);
}

// Everything of a full-attention layer between its projections and its output projection for the position `pos`: K and V
// into the cache at pos, the attention over positions 0..pos (one head per job up to 512 positions, split-K above), the
// gate; the output into ob and quantised into xo. One compiled body (and its parts) for decode and the verify.
__attribute__((noinline)) void Qwen35::attn_full_core(int L, int pos, const float* qg, const float* kp, const float* vp, float* ob, XBuf& xo) {
  const int HD = c.HD;
  float cosv[512], sinv[512];
  attn_full_kv(L, pos, kp, vp, cosv, sinv);
  FullCtx X;
  X.M = this;
  X.L = L;
  X.T = pos + 1;
  X.qg = qg;
  X.cosv = cosv;
  X.sinv = sinv;
  X.o = ob;
  X.xq = &xo.q;
  xo.reset(c.NQ * HD);
  if (X.T <= split_min()) {
    core->pool->run(phase_full_heads, &X);  // short: one head per job
  } else {
    // queries as the short path builds them (q_norm, rope), then the split-K pass and the combination
    std::vector<float> qn((size_t)c.NQ * HD);
    attn_full_q(L, qg, cosv, sinv, qn.data());
    SplitCtx S2;
    S2.M = this;
    S2.L = L;
    S2.T = X.T;
    S2.nch = (X.T + CHK - 1) / CHK;
    S2.q = qn.data();
    static thread_local std::vector<float> part;
    part.resize((size_t)S2.nch * c.NQ * (HD + 2));
    S2.part = part.data();
    core->pool->run(phase_split, &S2);
    attn_full_combine(L, X.T, qn.data(), part.data(), qg, ob, xo);
  }
}

int Qwen35::split_min() const {
  static const int v = getenv("DMENG_SPLIT_MIN") ? atoi(getenv("DMENG_SPLIT_MIN")) : 2 * CHK;  // test switch
  return v;
}

// The verify's positions pos0 .. pos0 + n - 1 of a full-attention layer. Positions within the short path go one by one
// through attn_full_core; the long-context ones write their K and V, build their queries, then every (chunk, KV head,
// query) split-K job runs in one pass (a chunk's keys read once for all the queries that need it, from the cache) and each
// query's chunks are combined as decode combines them. The same functions as decode, so the same bits.
struct MultiSplitCtx {
  Qwen35* M;
  int L, n, maxch;
  const int* T;
  float* const* q;
  float* const* part;
  alignas(128) std::atomic<int> next{0};
};
void phase_split_multi(void* p, int, int) {
  MultiSplitCtx& X = *(MultiSplitCtx*)p;
  Qwen35& M = *X.M;
  const QCfg& c = M.c;
  const int jobs = X.maxch * c.NKV * X.n;
  for (int jb; (jb = X.next.fetch_add(1, std::memory_order_relaxed)) < jobs;) {
    const int i = jb % X.n, rest = jb / X.n, kvh = rest % c.NKV, ch = rest / c.NKV;
    if (ch * CHK >= X.T[i]) continue;
    split_job(X.q[i], M.S->ls[X.L].K, M.S->ls[X.L].V, M.S->ctx, X.T[i], ch, kvh, c.NQ, c.NKV, c.HD, X.part[i]);
  }
}

void Qwen35::attn_full_multi(int L, int pos0, int n, float* const* qg, float* const* kp, float* const* vp, float* const* ob, XBuf* xo) {
  const int HD = c.HD;
  int i0 = 0;
  while (i0 < n && pos0 + i0 + 1 <= split_min()) {
    attn_full_core(L, pos0 + i0, qg[i0], kp[i0], vp[i0], ob[i0], xo[i0]);
    i0++;
  }
  if (i0 == n) return;
  const int m = n - i0;
  static thread_local std::vector<float> qbuf, pbuf;
  const int maxch = (pos0 + n + CHK - 1) / CHK;
  const size_t qs = (size_t)c.NQ * HD, ps = (size_t)maxch * c.NQ * (HD + 2);
  qbuf.resize(qs * m);
  pbuf.resize(ps * m);
  int T[8];
  float* q[8];
  float* part[8];
  for (int k = 0; k < m; k++) {
    float cosv[512], sinv[512];
    const int i = i0 + k;
    attn_full_kv(L, pos0 + i, kp[i], vp[i], cosv, sinv);
    q[k] = qbuf.data() + qs * k;
    part[k] = pbuf.data() + ps * k;
    attn_full_q(L, qg[i], cosv, sinv, q[k]);
    T[k] = pos0 + i + 1;
  }
  MultiSplitCtx X;
  X.M = this;
  X.L = L;
  X.n = m;
  X.maxch = maxch;
  X.T = T;
  X.q = q;
  X.part = part;
  core->pool->run(phase_split_multi, &X);
  for (int k = 0; k < m; k++) {
    const int i = i0 + k;
    xo[i].reset(c.NQ * HD);
    attn_full_combine(L, T[k], q[k], part[k], qg[i], ob[i], xo[i]);
  }
}

// ------------------------------------------------------------------------------------------------ one token
void Qwen35::step(State& s0, int32_t tok, Want want) {
  S = (QState*)&s0;
  greedy_ = -1;
  auto t0 = clk::now();
  stage_mark(ST_EMBED);
  rows.row(tok, x.data());
  std::copy(x.begin(), x.end(), emb.begin());
  T->embed += ms_since(t0);
  core->token_start(S->pos, tok, emb.data());
  for (int L = 0; L < c.NL; L++) {
    const Lw& W = lw[L];
    stage_mark(ST_NORMQ);
    rmsnorm(x.data(), W.ln_in, c.H, c.eps, h.data());
    if (W.full) attn_full(L, h.data(), y.data());
    else attn_linear(L, h.data(), y.data());
    stage_mark(ST_NORMQ);
    for (int i = 0; i < c.H; i++) x[i] += y[i];
    rmsnorm(x.data(), W.ln_post, c.H, c.eps, hm.data());
    core->moe_step(L, hm.data(), y.data(), views[L]);
    for (int i = 0; i < c.H; i++) x[i] += y[i];
  }
  if (want != Want::NONE) {
    auto th = clk::now();
    stage_mark(ST_HEAD);
    rmsnorm(x.data(), fnorm, c.H, c.eps, h.data());
    xin.prep(h.data(), c.H);
    if (fast_head && (want == Want::GREEDY || (want == Want::TOPK && head_topk > 0))) {
      const int k = want == Want::GREEDY ? 1 : head_topk;
      greedy_ = sh.run(*core->pool, xin.q, k, lgts.data());
      if (fast_head_check) {  // the full head as well: the same argmax, the same top-k rows with the same bits
        static std::vector<float> full;
        full.resize(c.V);
        Job jh = {&head, &xin.q, nullptr, full.data()};
        run_rows(*core->pool, &jh, 1, 64);
        std::vector<int> idx(c.V);
        for (int i = 0; i < c.V; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return full[a] > full[b] || (full[a] == full[b] && a < b); });
        bool ok = idx[0] == greedy_;
        for (int j = 0; j < k; j++) ok = ok && std::memcmp(&full[idx[j]], &lgts[idx[j]], 4) == 0;
        if (!ok) hd_mismatch += 1;
      }
    } else {
      Job jh = {&head, &xin.q, nullptr, lgts.data()};
      run_rows(*core->pool, &jh, 1, 64);
    }
    T->head += ms_since(th);
  }
  S->pos++;
  core->token_end();
  stage_mark(ST_OTHER);
}

// Run-ahead (Core::run_ahead): from layer L's approximate output (its experts in memory only), layer L + 1's attention and
// router on the approximate state, its top picks. Nothing of the state changes: a DeltaNet layer's convolution and
// recurrence are evaluated without writing their state (attn_linear_ro); a full-attention layer writes its K and V at this
// position, which the real pass writes again before anything reads them. Between sub-stages (and inside the row phases,
// rows_stop) it gives up as soon as the read the step waits for lands: the picks of an unfinished run are not used. The
// stage timers do not count it. Approximate by design (it only decides reads): its arithmetic need not be the real pass's.
bool Qwen35::attn_linear_ro(int L, const float* hin, float* rout) {
  const Lw& W = lw[L];
  const int C = c.conv_dim(), ZD = c.NVH * c.DV, CK = c.CK, DK = c.DK, DV = c.DV, rep = c.NVH / c.NKH;
  float* qkv = buf.data();
  float* z = qkv + C;
  float* bb = z + ZD;
  float* aa = bb + c.NVH;
  float* cv = aa + c.NVH;
  Job jb[4] = {{&W.qkv, &xin.q, hin, qkv}, {&W.z, &xin.q, hin, z}, {&W.wb, &xin.q, hin, bb}, {&W.wa, &xin.q, hin, aa}};
  {
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {hin};
    const int ks[1] = {c.H};
    run_rows_q(*core->pool, jb, 4, 32, false, nullptr, qs, xs, ks, 1);
  }
  if (core->ra_stopped()) return false;
  const float* st = S->ls[L].conv.data();
  for (int ch = 0; ch < C; ch++) {  // the convolution, the state read only
    float v = qkv[ch] * W.conv[(size_t)ch * CK + CK - 1];
    for (int k = 0; k < CK - 1; k++) v += st[(size_t)k * C + ch] * W.conv[(size_t)ch * CK + k];
    cv[ch] = silu(v);
  }
  float* ob = o.data();
  parallel_for(*core->pool, c.NVH, [&](int h) {  // the heads: S_new = g S + k (x) delta, evaluated, not stored
    const int kh = h / rep;
    const float* qs = cv + kh * DK;
    const float* ks = cv + c.NKH * DK + kh * DK;
    const float* v = cv + 2 * c.NKH * DK + h * DV;
    float qq = 0, kk = 0;
    for (int i = 0; i < DK; i++) qq += qs[i] * qs[i], kk += ks[i] * ks[i];
    const float rq = 1.f / std::sqrt(qq + 1e-6f) / std::sqrt((float)DK), rk = 1.f / std::sqrt(kk + 1e-6f);
    float q[256], k[256];
    for (int i = 0; i < DK; i++) q[i] = qs[i] * rq, k[i] = ks[i] * rk;
    const float beta = sigm(bb[h]), xa = aa[h] + W.dt_bias[h];
    const float g = std::exp(-std::exp(W.A_log[h]) * (xa > 20.f ? xa : std::log1p(std::exp(xa))));
    const float* Sh = S->ls[L].S.data() + (size_t)h * DV * DK;
    float* oh = ob + h * DV;
    for (int vi = 0; vi < DV; vi++) {
      const float* s = Sh + (size_t)vi * DK;
      float32x4_t acc = vdupq_n_f32(0), gv = vdupq_n_f32(g);
      for (int i = 0; i < DK; i += 4) acc = vfmaq_f32(acc, vmulq_f32(vld1q_f32(s + i), gv), vld1q_f32(k + i));
      const float32x4_t dv = vdupq_n_f32((v[vi] - vaddvq_f32(acc)) * beta);
      float32x4_t oa = vdupq_n_f32(0);
      for (int i = 0; i < DK; i += 4)
        oa = vfmaq_f32(oa, vfmaq_f32(vmulq_f32(vld1q_f32(s + i), gv), vld1q_f32(k + i), dv), vld1q_f32(q + i));
      oh[vi] = vaddvq_f32(oa);
    }
    float ss = 0;
    for (int i = 0; i < DV; i++) ss += oh[i] * oh[i];
    const float r = 1.f / std::sqrt(ss / DV + c.eps);
    for (int i = 0; i < DV; i++) oh[i] = oh[i] * r * W.gnorm[i] * silu(z[h * DV + i]);
  });
  if (core->ra_stopped()) return false;
  xatt.reset(ZD);
  xatt.prep(ob, ZD);
  Job jo = {&W.o, &xatt.q, nullptr, rout};
  run_rows(*core->pool, &jo, 1, 32);
  return !core->ra_stopped();
}

int Qwen35::run_ahead_step(int L, const float* ypart, int32_t* picks, int maxk) {
  const int L1 = L + 1;
  if (L1 >= c.NL) return 0;
  const Times keep = *T;
  struct Restore {
    Times* T;
    Times keep;
    ~Restore() { *T = keep; }
  } restore{T, keep};
  ra_L = -1;
  ra_x.resize(c.H);
  ra_h.resize(c.H);
  ra_y.resize(c.H);
  for (int i = 0; i < c.H; i++) ra_x[i] = x[i] + ypart[i];
  rmsnorm(ra_x.data(), lw[L1].ln_in, c.H, c.eps, ra_h.data());
  if (!lw[L1].full) {
    if (!attn_linear_ro(L1, ra_h.data(), ra_y.data())) return 0;
  } else {
    attn_full(L1, ra_h.data(), ra_y.data());
    if (core->ra_stopped()) return 0;
  }
  for (int i = 0; i < c.H; i++) ra_x[i] += ra_y[i];
  rmsnorm(ra_x.data(), lw[L1].ln_post, c.H, c.eps, ra_h.data());
  if (ra_xq.cap < c.H) ra_xq.init(c.H, prec);
  ra_xq.prep(ra_h.data(), c.H);
  ra_lg.resize(lw[L1].router.rows);
  Job jr = {&lw[L1].router, &ra_xq.q, ra_h.data(), ra_lg.data()};
  run_rows(*core->pool, &jr, 1, 16);
  if (core->ra_stopped()) return 0;
  const int NE = c.NE;
  std::vector<int> idx(NE);
  for (int e = 0; e < NE; e++) idx[e] = e;
  const int k = std::min(maxk, NE);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return ra_lg[a] > ra_lg[b] || (ra_lg[a] == ra_lg[b] && a < b); });
  for (int i = 0; i < k; i++) picks[i] = idx[i];
  core->ra_lgv.assign(ra_lg.begin(), ra_lg.begin() + NE);
  ra_L = L1;
  return k;
}

// deeper run-ahead (measurement): the layer after ra_L from the run-ahead's state with ra_L's MoE left out
int Qwen35::run_ahead_deeper_step(int, int32_t* picks, int maxk) {
  const int L2 = ra_L + 1;
  if (ra_L < 0 || L2 >= c.NL) return 0;
  const Times keep = *T;
  rmsnorm(ra_x.data(), lw[L2].ln_in, c.H, c.eps, ra_h.data());
  if (!lw[L2].full) attn_linear_ro(L2, ra_h.data(), ra_y.data());
  else attn_full(L2, ra_h.data(), ra_y.data());
  for (int i = 0; i < c.H; i++) ra_x[i] += ra_y[i];
  rmsnorm(ra_x.data(), lw[L2].ln_post, c.H, c.eps, ra_h.data());
  ra_xq.prep(ra_h.data(), c.H);
  ra_lg.resize(lw[L2].router.rows);
  Job jr = {&lw[L2].router, &ra_xq.q, ra_h.data(), ra_lg.data()};
  run_rows(*core->pool, &jr, 1, 16);
  std::vector<int> idx(c.NE);
  for (int e = 0; e < c.NE; e++) idx[e] = e;
  const int k = std::min(maxk, c.NE);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return ra_lg[a] > ra_lg[b] || (ra_lg[a] == ra_lg[b] && a < b); });
  for (int i = 0; i < k; i++) picks[i] = idx[i];
  ra_L = L2;
  *T = keep;
  return k;
}

bool Qwen35::set_option(const std::string& name, const std::string& value, std::string& err) {
  if (name == "run_ahead") {  // >0: run ahead one layer while a step waits on reads, naming this many predicted picks
    const int k = std::stoi(value);
    if (k <= 0) return (core->run_ahead = nullptr), true;
    core->ra_k = k;
    core->run_ahead = [this](int step, const float* yp, int32_t* pk, int maxk) { return run_ahead_step(step, yp, pk, maxk); };
    core->run_ahead_deeper = [this](int d, int32_t* pk, int maxk) { return run_ahead_deeper_step(d, pk, maxk); };
    return true;
  }
  if (name == "run_ahead_depth") return (core->ra_depth = std::max(1, std::min(4, std::stoi(value)))), true;  // measurement
  if (name == "fast_head") {  // 1: exact greedy head (temperature 0); 2: also check it against the full head
    const int v = std::stoi(value);
    if (v <= 0) return true;
    if (fast_head) return (fast_head_check = v == 2), true;
    if (prec != Prec::I16 || core->o.ref) return (err = "fast_head: needs I16 activations"), false;
    if (R.mem == Mem::FILE) return (err = "fast_head: the resident weights must be loaded (not mapped) to split the head"), false;
    if (!sh.build(*core->pool, head, c.V, c.V, err)) return false;
    fast_head = true;
    fast_head_check = v == 2;
    return true;
  }
  if (name == "head_topk") return (head_topk = std::min(std::stoi(value), ScreenHead::KMAX)), true;  // Want::TOPK's k
  if (name == "verify_old") return (verify_old = std::stoi(value) != 0), true;  // the previous verify pass (A/B measurements)
  if (name == "prefill_chunk") return (po.chunk = std::stoi(value)), true;  // 1: the decode path, one position at a time
  if (name == "prefill_span") return (po.span = std::stoi(value)), true;
  if (name == "prefill_stream") return (po.stream = std::stoi(value)), true;
  if (name == "prefill_whole_from") return (po.whole_from = std::stoi(value)), true;
  if (name == "prefill_record") {  // every position's picks of the batched prefill, int16 [pos][NL][K]
    if (po.record) fclose(po.record);
    po.record = value.empty() ? nullptr : fopen(value.c_str(), "wb");
    return value.empty() || po.record;
  }
  if (name == "prefill_batched_always") return (po.batched_always = std::stoi(value) != 0), true;
  if (name == "prefill_gpu_frac") return set_gpu_frac(std::stod(value), false, err);  // 0: CPU only (the default)
  if (name == "prefill_gpu_experts") return set_gpu_frac(std::stod(value), true, err);  // 0: CPU only (the default)
  if (name == "prefill_gpu_min") return (po.gpu_min = std::stoi(value)), true;
  if (name == "prefill_gpu_min_wave") return (po.gpu_min_wave = std::stoi(value)), true;
  if (name == "prefill_gpu_kernel") return (po.gpu_kernel = std::stoi(value) != 0), true;  // 1 matrix units, 0 integer
  if (name == "prefill_release") return (po.release = std::stoi(value) != 0), true;  // 1: free the scratch after each prefill
  if (name == "prefill_pack_min") return (po.pack_min = std::max(0, std::stoi(value))), true;  // 0: the multi-vector kernels
  if (name == "prefill_dn_rows") return (po.dn_rows = std::max(4, std::stoi(value) / 4 * 4)), true;
  if (!core) return (err = "unknown option " + name + " (or bind the model first)"), false;
  return core->set_option(name, value, err);  // the engine's own options (Core::set_option)
}

std::string Qwen35::stats_json(int) const {
  if (!fast_head) return V ? "{\"verify\":" + verify_json() + "}" : "{}";
  std::ostringstream o;
  const double n = std::max(1.0, sh.n_calls);
  o << "{\"fast_head\":{\"candidates_per_token\":" << sh.n_cand / n << ",\"coarse_ms\":" << sh.ms_coarse / n << ",\"exact_ms\":" << sh.ms_exact / n
    << ",\"check\":" << (fast_head_check ? "true" : "false") << ",\"mismatches\":" << hd_mismatch << "}";
  if (V) o << ",\"verify\":" << verify_json();
  o << "}";
  return o.str();
}

}  // namespace qwen35

std::unique_ptr<Model> make_qwen3_5_moe() { return std::unique_ptr<Model>(new qwen35::Qwen35()); }

}  // namespace dm
