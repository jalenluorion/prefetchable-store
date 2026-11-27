// Inkling 2 Mini (inkling2_loop), batch 1, on the CPU. Math as mlx-lm PR 2086 (mlx_lm/models/inkling2/{model,layers}.py):
// model.py _run_loop, layers.py DecoderLayer / SlidingDecoderLayer / FullDecoderLayer / Attention / ShortConvolution /
// DynamicShortConvolution / MessageGate / BlockAttentionResidual / TopKRouter / Experts.
//
// 48 layer applications per token: prelude 0-5, layers 6-17 three times each restarting from the prelude output with
// message gates from the previous pass, tail 18-23. Depth attention ("attn_res"), sliding (ring of 1,024) / global
// attention with the learned relative bias, sinks, L2 clips, short convolutions, and the final depth attention, norm,
// head and softcap. The MoE block (sigmoid router + gate bias, top 8, renormalised x 8, softcapped SwiGLU, NVFP4 /
// MXFP8 records) runs in the core; its steps are the applications, so the 3 passes of a looped layer share its slots.
// Precision: residual stream, norms, depth attention, router, message gates, attention scores and every intermediate
// fp32; weights as stored (8-bit resident or bf16 as released); matvec activations I16 by default (MXFP8 always fp16);
// KV cache fp16. The reference path (--prec f32): fp32 activations everywhere and scalar kernels.
#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <cstring>
#include <sstream>
#include <sys/stat.h>

#include "dmeng/screen_head.h"
#include "dmeng/model.h"
#include "dmeng/pagevec.h"
#include "dmeng/prefill.h"
#include "dmeng/sampler.h"
#include "ink_draft.h"
#include "../../core/gpu_split.h"

namespace dm {
namespace {

struct ICfg {
  int H, NL, NE, K, I, NH, NKV, HD, DREL, REXT, WIN, RANK, CS, CE, NP, BLOCK, V, VU, SC, eos;
  float softcap, gcap, ucap, rscale, eps, qkclip, relclip, logfloor, logalpha;
  std::string kind, fmt;  // per layer: 'L' sliding / 'G' global; 'N' nvfp4 / 'M' mxfp8
  size_t rec[2], gu_c[2], gu_s[2], dn_c[2], dn_s[2];  // [0] nvfp4, [1] mxfp8
  long trailer = -1, emb_off = 0;
  std::string emb_file, layer_file;
  std::vector<std::string> layer_paths;
  std::vector<std::pair<int, int>> visits;  // (layer, pass) per application
  int NV = 0;
  bool core(int L) const { return L >= CS && L < CE; }
  void from(const ResFile& R) {
    H = R.geti("hidden");
    NL = R.geti("layers");
    NE = R.geti("experts");
    K = R.geti("topk");
    I = R.geti("inter");
    NH = R.geti("heads");
    NKV = R.geti("kv_heads");
    HD = R.geti("head_dim");
    DREL = R.geti("d_rel");
    REXT = R.geti("rel_extent");
    WIN = R.geti("window");
    RANK = R.geti("gate_rank");
    CS = R.geti("core_start");
    CE = R.geti("core_end");
    NP = R.geti("passes");
    BLOCK = R.geti("block");
    V = R.geti("vocab");
    VU = R.geti("unpadded_vocab");
    SC = R.geti("sconv");
    eos = R.geti("eos");
    softcap = (float)R.getf("softcap");
    gcap = (float)R.getf("gate_cap");
    ucap = (float)R.getf("up_cap");
    rscale = (float)R.getf("route_scale");
    eps = (float)R.getf("eps");
    qkclip = (float)R.getf("qk_clip");
    relclip = (float)R.getf("rel_clip");
    logfloor = (float)R.getf("log_floor");
    logalpha = (float)R.getf("log_alpha");
    kind = R.gets("layer_kind");
    fmt = R.gets("layer_format");
    const char* tag[2] = {"nv_", "mx_"};
    for (int i = 0; i < 2; i++) {
      const std::string t = tag[i];
      rec[i] = R.geti(t + "rec");
      gu_c[i] = R.geti(t + "gu_c");
      gu_s[i] = R.geti(t + "gu_s");
      dn_c[i] = R.geti(t + "dn_c");
      dn_s[i] = R.geti(t + "dn_s");
    }
    trailer = R.geti("nv_trailer");
    emb_off = R.geti("emb_off");
    emb_file = R.gets("emb_file");
    layer_file = R.gets("layer_file");
    layer_paths.clear();
    if (R.has_cfg("layer_paths"))  // a profile's own record files (Manifest::apply_profile_experts)
      for (size_t a = 0, b; a <= R.gets("layer_paths").size(); a = b + 1) {
        const std::string s = R.gets("layer_paths");
        b = s.find(';', a);
        if (b == std::string::npos) b = s.size();
        layer_paths.push_back(s.substr(a, b - a));
      }
    visits.clear();
    for (int L = 0; L < CS; L++) visits.push_back({L, 0});
    for (int p = 0; p < NP; p++)
      for (int L = CS; L < CE; L++) visits.push_back({L, p});
    for (int L = CE; L < NL; L++) visits.push_back({L, 0});
    NV = (int)visits.size();
  }
};

struct Gate {  // a message gate: down [RANK, 2H] bf16, forget / gain [H, RANK] f32
  Weight down, forget, gain;
};

struct Lw {
  bool global = false, core = false;
  const float *aq = nullptr, *mq = nullptr;  // depth-attention queries: attention input [Q, H] (L > 0), mlp input [H]
  const float *ln_in = nullptr, *lnq = nullptr, *lnk = nullptr, *lnv = nullptr, *mlnorm = nullptr;
  Weight q, k, v, r, o, g, sink, kwp, vwp, router;
  const float *qclip = nullptr, *rclip = nullptr, *rel = nullptr;
  const float *kconv = nullptr, *vconv = nullptr, *kstat = nullptr, *vstat = nullptr, *aconv = nullptr, *mconv = nullptr;
  const float* rbias = nullptr;
  Gate mg[4];  // q, k, v, mlp
};

struct VState {  // what one layer application carries between tokens
  std::vector<float> conv[4];  // [SC - 1, width], oldest first: k sconv, v sconv, post-attention, post-MLP
  _Float16 *K = nullptr, *V = nullptr;  // [NKV, cap, HD]; sliding: a ring of cap positions
  float *Kf = nullptr, *Vf = nullptr;    // the same in fp32 (reference path only)
  size_t bytes = 0;
  int cap = 0;
  bool ring = false;
};

struct IState : State {
  std::vector<VState> vs;
  int ctx = 0;
  ~IState() override {
    for (auto& s : vs) {
      free_region((uint8_t*)s.K, s.bytes);
      free_region((uint8_t*)s.V, s.bytes);
      free_region((uint8_t*)s.Kf, s.bytes);
      free_region((uint8_t*)s.Vf, s.bytes);
    }
  }
};

// a multi-position step writes ahead of its reads: the rings keep this many positions more than the window
constexpr int RING_SLACK = 7;

struct Inkling : Model {
  ICfg c;
  ResFile R;
  ModelInfo mi;
  Mem res_mem = Mem::MLOCK;
  Core* core = nullptr;
  Times* T = nullptr;
  bool ref = false;
  Prec prec = Prec::I16;
  std::vector<Lw> lw;
  std::vector<MoeView> views;
  Weight head;
  const float *embed_norm = nullptr, *final_q = nullptr, *fnorm = nullptr, *msgq = nullptr;
  EmbedRows rows;
  IState* S = nullptr;
  XBuf xa, xb, xc, xe;
  std::vector<float> h, blk, bank, msg, tmp, lgts, emb, embn, full;
  std::vector<float> h_pre;  // step(): the hidden state entering the loop, restored at each pass (kept, not allocated per token)
  std::vector<float> taps_;  // [4, H] the draft head's features of the last step (empty: not kept)
  std::vector<float> att_sc;  // [NH, n] the grouped decode attention's scores
  ScreenHead sh;             // the screened head (core): Want::GREEDY / Want::TOPK with the fast_head option
  bool fast_head = false;
  int head_topk = 0;         // Want::TOPK's k
  // run-ahead (Core::run_ahead): the application after the one waiting, up to its router, on the approximate state
  bool ra_mode = false;            // layer() stops before the MoE step and leaves its input in ra_xm
  float* ra_xm = nullptr;
  float* cur_res = nullptr;        // the residual of the application in its MoE step (layer() sets it)
  const float* cur_m4 = nullptr;
  std::vector<float> ra_tmp, ra_h, ra_conv, ra_blk, ra_lg, ra_c[4];
  XBuf ra_xq;
  int run_ahead_step(int v, const float* ypart, int32_t* picks, int maxk);
  int greedy_ = -1;
  double gh_fallback = 0;
  int greedy() const override { return fast_head ? greedy_ : -1; }
  int nblk = 0;
  // the draft head (set_option draft_head): fed the features of every committed position
  std::unique_ptr<InkDraft> dh;
  std::string dir;
  bool dh_exact = false;
  bool dh_q8 = true;  // the draft head's matrices in AFF8 (ink_draft.h InkDraftOpts::q8)
  int dh_ctx = 16384;
  std::vector<float> ctaps;  // [n, 4, H] a chunk's features
  std::vector<float> dh_emb, dh_embn;
  int draft_head(int32_t tok, int max, float thr, int32_t* out, float* prob, float* logits1) override;
  bool draft_head_feed(const float* rows, int n, int p0) override {
    if (!dh) return false;
    dh->add_taps(rows, n, p0);
    return true;
  }
  size_t draft_head_bytes() const override { return dh ? dh->weight_bytes() + dh->state_bytes() : 0; }
  bool load_draft_head(std::string& err);
  std::string stats_json(int) const override {  // the exact greedy head's and the draft head's statistics
    std::ostringstream o;
    o << "{";
    if (fast_head) {
      const double n = std::max(1.0, sh.n_calls);
      o << "\"fast_head\":{\"candidates_per_token\":" << sh.n_cand / n << ",\"coarse_ms\":" << sh.ms_coarse / n << ",\"exact_ms\":"
        << sh.ms_exact / n << ",\"full_head_fallbacks\":" << gh_fallback << "}";
    }
    if (dh) o << (fast_head ? "," : "") << "\"draft_head\":" << dh->stats_json();
    if (ver_passes > 0) {
      const double p = ver_passes;
      o << (fast_head || dh ? "," : "") << "\"verify\":{\"steps\":" << ver_passes << ",\"positions\":" << ver_positions
        << ",\"mb_per_pass\":{\"resident\":" << ver_res_bytes / 1e6 / p << ",\"head\":"
        << ver_head_bytes / 1e6 / p << ",\"touched\":" << (ver_res_bytes + ver_head_bytes) / 1e6 / p << "}}";
    }
    o << "}";
    return o.str();
  }
  void reset_stats() override {
    sh.reset_stats();
    gh_fallback = 0;
    ver_passes = ver_positions = ver_head_bytes = ver_res_bytes = 0;
    if (dh) dh->reset_stats();
  }

  const ModelInfo& info() const override { return mi; }
  bool load(const LoadOpts& o, std::string& err) override;
  bool bind(Core& core, std::string& err) override;
  std::unique_ptr<State> new_state(int ctx_max, std::string& err) override;
  void reset(State& s) override;
  size_t fixed_bytes(const State& s) const override;
  size_t pos_bytes() const override;
  bool save_fixed(const State& s, void* dst) const override;
  bool load_fixed(State& s, const void* src, int pos) override;
  bool save_pos(const State& s, int p0, int p1, void* dst) const override;
  bool load_pos(State& s, int p0, int p1, const void* src) override;
  void step(State& s, int32_t id, Want want) override;
  const std::vector<float>& logits() const override { return lgts; }
  const float* router_emb() const override { return embn.data(); }
  bool set_option(const std::string& name, const std::string& value, std::string& err) override;
  const std::vector<float>* taps() const override { return taps_.empty() ? nullptr : &taps_; }
  // light draft heads (draft.h LightHeadSource): the post-norm final hidden of the last step (i < 0) or of verify position
  // i, the raw embedding row, the output head's argmax and the top-1 probability after the softcap
  const float* final_hidden(int i) const override {
    if (i < 0) return tmp.size() >= (size_t)3 * c.H ? tmp.data() + 2 * (size_t)c.H : nullptr;
    return (size_t)(i + 1) * c.H <= vhn.size() ? vhn.data() + (size_t)i * c.H : nullptr;
  }
  bool embed_row(int32_t tok, float* out) const override {
    if (tok < 0 || tok >= c.VU) return false;
    rows.row(tok, out);
    return true;
  }
  int head_argmax(const float* hn, float* conf) override;
  std::vector<float> vhn;  // the last verify's post-norm final hidden per position
  XBuf dh_x;
  std::vector<float> dh_lg;
  void layer(int v, const float* m4);  // one application; m4 = 4 messages [4, H] or null
  // ---- a chunk of positions in one pass (batched prefill; multi-position verify): every application runs for all the
  // chunk's positions before the next; per position the single step's arithmetic (chunk.cpp section below)
  // A span runs application-major: every application over all its positions (in sub-chunks that bound the scratch) before
  // the next, so each application's experts are read once per span, not once per chunk. What a position carries from one
  // application to the next lives in the span arrays; the sub-chunk's scratch in pp.
  struct PP {  // per-position scratch of a sub-chunk
    std::vector<float> tmp, m4;  // layer()'s layout; this application's 4 messages
    XBuf xa, xb, xc, xe;
  };
  std::vector<std::unique_ptr<PP>> pp;
  // per span position: the residual, its value at the loop's start, the block states, the message sources (the previous
  // pass's blocks after its first, then its bank); the block count is the same for every position
  PageVec<float> sp_h, sp_hpre, sp_blk, sp_msrc;  // (PageVec: given back to the system when freed, prefill_release)
  int sp_n = 0, sp_nblk = 0, NBMAX = 0, NMS = 0;
  float* sph(int i) { return sp_h.data() + (size_t)i * c.H; }
  float* spblk(int i) { return sp_blk.data() + (size_t)i * NBMAX * c.H; }
  float* spmsrc(int i) { return sp_msrc.data() + (size_t)i * NMS * c.H; }
  XQB xqb_a, xqb_b, xqb_c, xqb_e, xqb_h;  // the sub-chunk's inputs packed for the batch kernels (>= 6 positions)
  // the batched prompt path's options (dmeng/prefill.h; this model's values: its manifest's "prefill"): the projections'
  // GPU share through run_rows_n_split, the experts' (the NVFP4 layers; MXFP8 stays on the CPU) through moe_chunk
  PrefillOpts po;
  GpuShare gpu;
  GpuSplitAcct gacct;
  // the batched path's wall ms by part, accumulated (the prefill JSON's ms_by_stage reports a prompt's share): the MoE
  // call per expert format [0] NVFP4 / [1] MXFP8 and its experts' compute part, the attention core of sliding / global
  // applications, the message gates, the projections of the attention input / output
  double pf_moe_ms[2] = {0, 0}, pf_exp_ms[2] = {0, 0}, pf_exp_mb[2] = {0, 0}, pf_core_ms[2] = {0, 0}, pf_gates_ms = 0, pf_proj_in_ms = 0,
         pf_proj_out_ms = 0;
  long n_chunks = 0;
  PageVec<uint8_t> span_snap;
  std::vector<uint8_t> ring_tmp;  // the fixed state at a span's start (a cancel inside it); a ring slice
  struct SpanRun {                            // what a prefill span reports and watches
    const std::vector<int>* capture = nullptr;  // positions relative to the span start, in (0, n], whose state goes to sink
    const LM::StateSink* sink = nullptr;
    const std::atomic<bool>* cancel = nullptr;
    const PrefillProgress* progress = nullptr;
    int prog_base = 0, prog_total = 0;
  };
  void chunk_alloc(int n);
  void span_alloc(int n);
  void release_scratch();
  void span_blocks();  // NBMAX, NMS
  // the prompt's bytes per position (prompt_plan): a sub-chunk's scratch (chunk_alloc's per-position buffers and the
  // packed inputs), the span's arrays (span_alloc's, the taps); what the scratch holds now
  double sub_bytes_per_pos() const {
    const int QD = c.NH * c.HD, mk = std::max({c.H, QD, 2 * c.H});
    return 4.0 * (28.0 * c.H + 4.0 * QD + 256) + 4.0 * xq_bytes(mk) + 3.0 * xq_bytes(c.H) + xq_bytes(QD);
  }
  double span_bytes_per_pos() const { return 4.0 * c.H * (2 + NBMAX + NMS) + (dh ? 16.0 * c.H : 0.0); }
  double scratch_held() const {
    double b = 0;
    for (auto& p : pp) b += 4.0 * (p->tmp.capacity() + p->m4.capacity()) + 4.0 * p->xa.cap;
    b += 4.0 * ((double)sp_h.capacity() + sp_hpre.capacity() + sp_blk.capacity() + sp_msrc.capacity() + ctaps.capacity()) + span_snap.capacity();
    return b + xqb_a.cap + xqb_b.cap + xqb_c.cap + xqb_e.cap;
  }
  long n_scratch_releases = 0;
  void layer_chunk(int v, int i0, int n, bool msgs);
  // false: cancelled inside the span (the state is back at its start)
  bool forward_chunk(const int32_t* ids, int n, bool want_last, bool all_logits = false, bool rollback = false, int sub = 0,
                     const SpanRun* run = nullptr);
  void sink_app(int v, int pos, const LM::StateSink& sink);  // application v's slice of the fixed state at pos
  // verify: rollback state of the last chunk (conv states at its start + each position's conv inputs); the rings need no
  // copy (RING_SLACK: a slot a verify overwrites holds a position no later one reads)
  bool rb = false;
  int rb_n = 0, rb_pos0 = 0;
  std::vector<std::vector<float>> rb_snap, rb_in;  // [NV * 4]: conv state, inputs [n, width]
  std::vector<float> mlg;                           // [n, VU] every verified position's logits
  bool verify(State& s, const int32_t* ids, int n, Want want = Want::LAST) override;
  void commit(State& s, int a) override;
  const float* logits_at(int i) const override { return mlg.data() + (size_t)i * c.VU; }
  Want ver_want = Want::LAST;  // what the running verify's caller reads of each position's logits
  int ver_best[RING_SLACK + 1];
  double ver_passes = 0, ver_positions = 0, ver_head_bytes = 0, ver_res_bytes = 0, ver_hb_full_ = 0;
  bool prefill(State& s, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture, const LM::StateSink& sink,
               PrefillStats* st, const std::atomic<bool>* cancel) override;
  ModelMem mem(const State& s) const override;
  void rows_run(const Job* j, int n, int chunk) { run_rows(*core->pool, j, n, chunk, ref, &T->bytes_res); }
};

// ------------------------------------------------------------------------------------------------ helpers
inline float dotf(const float* a, const float* b, int n) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  float t = vaddvq_f32(vaddq_f32(s0, s1));
  for (; i < n; i++) t += a[i] * b[i];
  return t;
}

// depth attention (BlockAttentionResidual): sources src[0..n) (the block states, then the partial state), learned
// queries q [Q, H]; weights = softmax over sources of src . q / sqrt(H); out [Q, H] = weighted sum of the sources.
// Per output the arithmetic of the single loops this replaced, bit for bit: each source's score is dotf's two chains
// (four sources at once, the query loaded once), each output element the chain t = fma(w[i], src[i][d], t) over i in
// order, 16 elements at a time.
void depth_attn(const float* const* src, int n, const float* q, int Q, int H, float* out) {
  const float scale = 1.f / std::sqrt((float)H);
  for (int j = 0; j < Q; j++) {
    float w[16], mx = -INFINITY;
    const float* qj = q + (size_t)j * H;
    int i = 0;
    const int H8 = H & ~7;
    for (; i + 4 <= n; i += 4) {
      float32x4_t a0 = vdupq_n_f32(0), a1 = a0, b0 = a0, b1 = a0, c0 = a0, c1 = a0, d0 = a0, d1 = a0;
      const float *s0 = src[i], *s1 = src[i + 1], *s2 = src[i + 2], *s3 = src[i + 3];
      for (int k = 0; k < H8; k += 8) {
        const float32x4_t q0 = vld1q_f32(qj + k), q1 = vld1q_f32(qj + k + 4);
        a0 = vfmaq_f32(a0, vld1q_f32(s0 + k), q0);
        a1 = vfmaq_f32(a1, vld1q_f32(s0 + k + 4), q1);
        b0 = vfmaq_f32(b0, vld1q_f32(s1 + k), q0);
        b1 = vfmaq_f32(b1, vld1q_f32(s1 + k + 4), q1);
        c0 = vfmaq_f32(c0, vld1q_f32(s2 + k), q0);
        c1 = vfmaq_f32(c1, vld1q_f32(s2 + k + 4), q1);
        d0 = vfmaq_f32(d0, vld1q_f32(s3 + k), q0);
        d1 = vfmaq_f32(d1, vld1q_f32(s3 + k + 4), q1);
      }
      float t[4] = {vaddvq_f32(vaddq_f32(a0, a1)), vaddvq_f32(vaddq_f32(b0, b1)), vaddvq_f32(vaddq_f32(c0, c1)), vaddvq_f32(vaddq_f32(d0, d1))};
      for (int u = 0; u < 4; u++) {
        for (int k = H8; k < H; k++) t[u] += src[i + u][k] * qj[k];
        w[i + u] = t[u] * scale;
      }
    }
    for (; i < n; i++) w[i] = dotf(src[i], qj, H) * scale;
    for (i = 0; i < n; i++) mx = std::max(mx, w[i]);
    float z = 0;
    for (i = 0; i < n; i++) z += (w[i] = std::exp(w[i] - mx));
    for (i = 0; i < n; i++) w[i] /= z;
    float* o = out + (size_t)j * H;
    int d = 0;
    for (; d + 16 <= H; d += 16) {
      float32x4_t o0 = vdupq_n_f32(0), o1 = o0, o2 = o0, o3 = o0;
      for (i = 0; i < n; i++) {
        const float* s = src[i] + d;
        o0 = vfmaq_n_f32(o0, vld1q_f32(s), w[i]);
        o1 = vfmaq_n_f32(o1, vld1q_f32(s + 4), w[i]);
        o2 = vfmaq_n_f32(o2, vld1q_f32(s + 8), w[i]);
        o3 = vfmaq_n_f32(o3, vld1q_f32(s + 12), w[i]);
      }
      vst1q_f32(o + d, o0);
      vst1q_f32(o + d + 4, o1);
      vst1q_f32(o + d + 8, o2);
      vst1q_f32(o + d + 12, o3);
    }
    for (; d < H; d++) {
      float t = 0;
      for (i = 0; i < n; i++) t = __builtin_fmaf(w[i], src[i][d], t);
      o[d] = t;
    }
  }
}

// ShortConvolution: out = (sum_k w[c, k] window[k] + x) / sqrt(SC + 1), window = [state (oldest first), x]
void sconv(std::vector<float>& st, const float* w, const float* x, int C, int SC, float* out) {
  const float inv = 1.f / std::sqrt((float)(SC + 1));
  float* s = st.data();
  for (int ch = 0; ch < C; ch++) {
    float t = w[(size_t)ch * SC + SC - 1] * x[ch];
    for (int k = 0; k < SC - 1; k++) t += w[(size_t)ch * SC + k] * s[(size_t)k * C + ch];
    const float xc = x[ch];
    for (int k = 0; k + 1 < SC - 1; k++) s[(size_t)k * C + ch] = s[(size_t)(k + 1) * C + ch];
    s[(size_t)(SC - 2) * C + ch] = xc;
    out[ch] = (t + xc) * inv;
  }
}

// DynamicShortConvolution: weight[c, lag] = static[lag, c] + dyn[head(c), lag]; lag 0 = this token
void dconv(std::vector<float>& st, const float* stat, const float* dyn, const float* x, int C, int SC, int HD, float* out) {
  const float inv = 1.f / std::sqrt((float)(2 * SC + 1));
  float* s = st.data();
  for (int ch = 0; ch < C; ch++) {
    const float* dh = dyn + (size_t)(ch / HD) * SC;
    float t = (stat[ch] + dh[0]) * x[ch];
    for (int lag = 1; lag < SC; lag++) t += (stat[(size_t)lag * C + ch] + dh[lag]) * s[(size_t)(SC - 1 - lag) * C + ch];
    const float xc = x[ch];
    for (int k = 0; k + 1 < SC - 1; k++) s[(size_t)k * C + ch] = s[(size_t)(k + 1) * C + ch];
    s[(size_t)(SC - 2) * C + ch] = xc;
    out[ch] = (t + xc) * inv;
  }
}

// the same convolutions on channels [c0, c1) only (each channel's arithmetic as above)
void sconv_rng(std::vector<float>& st, const float* w, const float* x, int C, int SC, float* out, int c0, int c1) {
  const float inv = 1.f / std::sqrt((float)(SC + 1));
  float* s = st.data();
  for (int ch = c0; ch < c1; ch++) {
    float t = w[(size_t)ch * SC + SC - 1] * x[ch];
    for (int k = 0; k < SC - 1; k++) t += w[(size_t)ch * SC + k] * s[(size_t)k * C + ch];
    const float xc = x[ch];
    for (int k = 0; k + 1 < SC - 1; k++) s[(size_t)k * C + ch] = s[(size_t)(k + 1) * C + ch];
    s[(size_t)(SC - 2) * C + ch] = xc;
    out[ch] = (t + xc) * inv;
  }
}
void dconv_rng(std::vector<float>& st, const float* stat, const float* dyn, const float* x, int C, int SC, int HD, float* out, int c0, int c1) {
  const float inv = 1.f / std::sqrt((float)(2 * SC + 1));
  float* s = st.data();
  for (int ch = c0; ch < c1; ch++) {
    const float* dh = dyn + (size_t)(ch / HD) * SC;
    float t = (stat[ch] + dh[0]) * x[ch];
    for (int lag = 1; lag < SC; lag++) t += (stat[(size_t)lag * C + ch] + dh[lag]) * s[(size_t)(SC - 1 - lag) * C + ch];
    const float xc = x[ch];
    for (int k = 0; k + 1 < SC - 1; k++) s[(size_t)k * C + ch] = s[(size_t)(k + 1) * C + ch];
    s[(size_t)(SC - 2) * C + ch] = xc;
    out[ch] = (t + xc) * inv;
  }
}

void l2clip(float* x, const float* scale, int n, float thr) {
  double ss = 0;
  for (int i = 0; i < n; i++) {
    if (scale) x[i] *= scale[i];
    ss += (double)x[i] * x[i];
  }
  const float f = std::min(thr / ((float)std::sqrt(ss) + 1e-6f), 1.f);
  for (int i = 0; i < n; i++) x[i] *= f;
}

// ------------------------------------------------------------------------------------------------ load
bool Inkling::load(const LoadOpts& op, std::string& err) {
  if (!R.load(op.resident, op.res_mem, err)) return false;  // (resolve_model names the file)
  dir = op.dir;
  res_mem = op.res_mem;
  for (auto& kv : op.cfg) R.cfg[kv.first] = kv.second;  // the manifest's configuration, where there is one
  c.from(R);
  lw.resize(c.NL);
  for (int L = 0; L < c.NL; L++) {
    Lw& w = lw[L];
    const std::string s = std::to_string(L) + ".";
    w.global = c.kind[L] == 'G';
    w.core = c.core(L);
    if (L > 0) w.aq = R.f32(s + "aq");
    w.mq = R.f32(s + "mq");
    if (w.global) {
      w.lnq = R.f32(s + "lnq");
      w.lnk = R.f32(s + "lnk");
      w.lnv = R.f32(s + "lnv");
      w.kstat = R.f32(s + "kstat");
      w.vstat = R.f32(s + "vstat");
      w.kwp = R.weight(s + "kwp");
      w.vwp = R.weight(s + "vwp");
    } else {
      w.ln_in = R.f32(s + "ln_in");
      w.kconv = R.f32(s + "kconv");
      w.vconv = R.f32(s + "vconv");
    }
    w.q = R.weight(s + "q");
    w.k = R.weight(s + "k");
    w.v = R.weight(s + "v");
    w.r = R.weight(s + "r");
    w.o = R.weight(s + "o");
    w.g = R.weight(s + "g");
    w.sink = R.weight(s + "sink");
    w.qclip = R.f32(s + "qclip");
    w.rclip = R.f32(s + "rclip");
    w.rel = R.f32(s + "rel");
    w.aconv = R.f32(s + "aconv");
    w.mconv = R.f32(s + "mconv");
    w.mlnorm = R.f32(s + "mlnorm");
    w.router = R.weight(s + "router");
    w.rbias = R.f32(s + "rbias");
    if (w.core) {
      const char* nm = "qkvm";
      for (int i = 0; i < 4; i++) {
        const std::string g = s + "mg" + nm[i] + ".";
        w.mg[i].down = R.weight(g + "down");
        w.mg[i].forget = R.weight(g + "forget");
        w.mg[i].gain = R.weight(g + "gain");
      }
    }
  }
  embed_norm = R.f32("embed_norm");
  final_q = R.f32("final_q");
  fnorm = R.f32("norm");
  msgq = R.f32("msgq");
  head = R.weight("head");
  if (!rows.open(c.emb_file[0] == '/' ? c.emb_file : op.dir + "/" + c.emb_file, c.emb_off, c.H, err)) return false;
  // what the core needs
  mi.family = "inkling2_loop";
  mi.has_verify = true;
  mi.hidden = c.H;
  mi.vocab = c.V;
  mi.vocab_out = c.VU;
  mi.layers = c.NL;
  mi.steps = c.NV;
  mi.step_layer.resize(c.NV);
  mi.step_pass.resize(c.NV);
  for (int v = 0; v < c.NV; v++) {
    mi.step_layer[v] = c.visits[v].first;
    mi.step_pass[v] = c.core(c.visits[v].first) ? c.visits[v].second : -1;
  }
  mi.experts = c.NE;
  mi.top_k = c.K;
  mi.shared_inter = 0;
  ExpertFormat f[2];
  for (int i = 0; i < 2; i++) {
    f[i].fmt = i == 0 ? Fmt::NVFP4 : Fmt::MXFP8;
    f[i].rec = c.rec[i];
    f[i].H = c.H;
    f[i].I = c.I;
    f[i].fused = true;
    f[i].gu_c = c.gu_c[i];
    f[i].gu_s = c.gu_s[i];
    f[i].dn_c = c.dn_c[i];
    f[i].dn_s = c.dn_s[i];
  }
  f[0].trailer = c.trailer;
  f[0].touched = (size_t)c.trailer;
  f[1].touched = c.dn_s[1] + (size_t)c.H * (c.I / 32);
  mi.formats = {f[0], f[1]};
  mi.layer_fmt.resize(c.NL);
  bool any_mx = false;
  for (int L = 0; L < c.NL; L++) {
    mi.layer_fmt[L] = c.fmt[L] == 'M' ? 1 : 0;
    any_mx |= c.fmt[L] == 'M';
  }
  mi.pool_share = any_mx ? std::vector<double>{0.8, 0.2} : std::vector<double>{1.0, 0.0};
  mi.layer_file = c.layer_file;
  mi.layer_paths = c.layer_paths;
  mi.eos = c.eos;
  mi.has_ref_path = true;
  views.resize(c.NL);
  for (int L = 0; L < c.NL; L++) {
    MoeView& v = views[L];
    v.rule = MoeView::SIGMOID_BIAS_TOPK_SCALED;
    v.route_scale = c.rscale;
    v.router = &lw[L].router;
    v.router_bias = lw[L].rbias;
    v.act = MoeView::SOFTCAP_SWIGLU;
    v.gate_cap = c.gcap;
    v.up_cap = c.ucap;
    v.shared = false;
  }
  return prefill_declared(op.prefill, po, err);  // the prompt path's values the manifest declares
}

bool Inkling::bind(Core& cr, std::string& err) {
  core = &cr;
  T = &cr.t;
  cr.views.resize(c.NV);
  for (int v = 0; v < c.NV; v++) cr.views[v] = &views[c.visits[v].first];
  ref = cr.o.ref;
  prec = cr.o.prec;
  const int QD = c.NH * c.HD, mk = std::max({c.H, QD, 2 * c.H});
  xa.init(mk, prec);
  xb.init(mk, prec);
  xc.init(mk, prec);
  xe.init(mk, prec);
  h.resize(c.H);
  blk.assign((size_t)8 * c.H, 0.f);
  bank.assign((size_t)8 * c.H, 0.f);
  msg.assign((size_t)(c.CE - c.CS) * 4 * c.H, 0.f);
  tmp.assign((size_t)40 * c.H + 4 * QD, 0.f);
  lgts.assign(c.VU, 0.f);
  emb.assign(c.H, 0.f);
  embn.assign(c.H, 0.f);
  full.assign(c.V, 0.f);
  return true;
}

std::unique_ptr<State> Inkling::new_state(int ctx_max, std::string& err) {
  std::unique_ptr<IState> s(new IState());
  s->ctx = ctx_max;
  s->vs.resize(c.NV);
  const int KVD = c.NKV * c.HD;
  for (int v = 0; v < c.NV; v++) {
    const int L = c.visits[v].first;
    VState& st = s->vs[v];
    st.conv[0].assign((size_t)(c.SC - 1) * KVD, 0.f);
    st.conv[1].assign((size_t)(c.SC - 1) * KVD, 0.f);
    st.conv[2].assign((size_t)(c.SC - 1) * c.H, 0.f);
    st.conv[3].assign((size_t)(c.SC - 1) * c.H, 0.f);
    st.ring = c.kind[L] == 'L';
    st.cap = st.ring ? std::min(c.WIN + RING_SLACK, ctx_max) : ctx_max;
    const size_t n = (size_t)KVD * st.cap;
    if (ref) {  // the reference path keeps keys and values in fp32
      st.bytes = n * 4;
      st.Kf = (float*)alloc_region(st.bytes, Mem::PLAIN, &err);
      st.Vf = (float*)alloc_region(st.bytes, Mem::PLAIN, &err);
      if (!st.Kf || !st.Vf) return nullptr;
      continue;
    }
    st.bytes = n * 2;
    st.K = (_Float16*)alloc_region(st.bytes, Mem::PLAIN, &err);  // touched only as the context grows
    st.V = (_Float16*)alloc_region(st.bytes, Mem::PLAIN, &err);
    if (!st.K || !st.V) return nullptr;
  }
  return s;
}

void Inkling::reset(State& s0) {
  IState& s = (IState&)s0;
  for (auto& st : s.vs)
    for (auto& cv : st.conv) std::fill(cv.begin(), cv.end(), 0.f);
  s.pos = 0;
  if (dh) dh->truncate(0);
}

// The state for the serving layer (include/dmeng/lm.h): the fixed part = every application's four convolution states
// and the sliding layers' rings (fixed size, copied whole); the positional part = the global layers' K and V per position.
size_t Inkling::fixed_bytes(const State& s0) const {
  const IState& s = (const IState&)s0;
  size_t n = 0;
  for (auto& st : s.vs) {
    for (auto& cv : st.conv) n += cv.size() * 4;
    if (st.ring) n += 2 * st.bytes;
  }
  return n;
}
size_t Inkling::pos_bytes() const {
  int ng = 0;
  for (int v = 0; v < c.NV; v++) ng += c.kind[c.visits[v].first] != 'L';
  return (size_t)ng * c.NKV * c.HD * 2 * (ref ? 4 : 2);
}
ModelMem Inkling::mem(const State& s0) const {
  const IState& s = (const IState&)s0;
  ModelMem m;
  m.resident = R.size;
  for (auto& st : s.vs)
    if (!st.ring) m.kv_reserved += 2 * st.bytes;
  m.recurrent = fixed_bytes(s0);  // the convolution states and the sliding layers' rings
  auto add = [&](const auto& v) { m.prefill_scratch += v.capacity() * sizeof(v[0]); };
  for (auto& p : pp) {
    add(p->tmp); add(p->m4);
  }
  add(sp_h); add(sp_hpre); add(sp_blk); add(sp_msrc); add(ctaps);
  for (auto& v : rb_snap) add(v);
  for (auto& v : rb_in) add(v);
  add(mlg);
  m.prefill_xqb = xqb_a.cap + xqb_b.cap + xqb_c.cap + xqb_e.cap + xqb_h.cap;
  return m;
}
bool Inkling::save_fixed(const State& s0, void* dst) const {
  const IState& s = (const IState&)s0;
  uint8_t* d = (uint8_t*)dst;
  for (auto& st : s.vs) {
    for (auto& cv : st.conv) {
      std::memcpy(d, cv.data(), cv.size() * 4);
      d += cv.size() * 4;
    }
    if (st.ring) {
      std::memcpy(d, st.K ? (const void*)st.K : (const void*)st.Kf, st.bytes);
      std::memcpy(d + st.bytes, st.V ? (const void*)st.V : (const void*)st.Vf, st.bytes);
      if (s.pos < st.cap) {  // slots no position has reached yet hold whatever an earlier sequence left: save them as zeros
        const size_t es = st.K ? 2 : 4, row = (size_t)c.HD * es;
        for (int kv = 0; kv < 2; kv++)
          for (int j = 0; j < c.NKV; j++)
            std::memset(d + kv * st.bytes + ((size_t)j * st.cap + s.pos) * row, 0, (size_t)(st.cap - s.pos) * row);
      }
      d += 2 * st.bytes;
    }
  }
  return true;
}
bool Inkling::load_fixed(State& s0, const void* src, int p) {
  IState& s = (IState&)s0;
  const uint8_t* d = (const uint8_t*)src;
  for (auto& st : s.vs) {
    for (auto& cv : st.conv) {
      std::memcpy(cv.data(), d, cv.size() * 4);
      d += cv.size() * 4;
    }
    if (st.ring) {
      std::memcpy(st.K ? (void*)st.K : (void*)st.Kf, d, st.bytes);
      std::memcpy(st.V ? (void*)st.V : (void*)st.Vf, d + st.bytes, st.bytes);
      d += 2 * st.bytes;
    }
  }
  s.pos = p;
  return true;
}
bool Inkling::save_pos(const State& s0, int p0, int p1, void* dst) const {
  const IState& s = (const IState&)s0;
  if (p0 < 0 || p1 > s.ctx || p0 > p1) return false;
  uint8_t* d = (uint8_t*)dst;
  const size_t es = ref ? 4 : 2, row = (size_t)c.HD * es;
  for (int p = p0; p < p1; p++)
    for (auto& st : s.vs)
      if (!st.ring)
        for (int j = 0; j < c.NKV; j++) {
          const size_t o = (((size_t)j * st.cap + p) * c.HD) * es;
          std::memcpy(d, (const uint8_t*)(st.K ? (const void*)st.K : (const void*)st.Kf) + o, row);
          std::memcpy(d + row, (const uint8_t*)(st.V ? (const void*)st.V : (const void*)st.Vf) + o, row);
          d += 2 * row;
        }
  return true;
}
bool Inkling::load_pos(State& s0, int p0, int p1, const void* src) {
  IState& s = (IState&)s0;
  if (p0 < 0 || p1 > s.ctx || p0 > p1) return false;
  const uint8_t* d = (const uint8_t*)src;
  const size_t es = ref ? 4 : 2, row = (size_t)c.HD * es;
  for (int p = p0; p < p1; p++)
    for (auto& st : s.vs)
      if (!st.ring)
        for (int j = 0; j < c.NKV; j++) {
          const size_t o = (((size_t)j * st.cap + p) * c.HD) * es;
          std::memcpy((uint8_t*)(st.K ? (void*)st.K : (void*)st.Kf) + o, d, row);
          std::memcpy((uint8_t*)(st.V ? (void*)st.V : (void*)st.Vf) + o, d + row, row);
          d += 2 * row;
        }
  return true;
}

// ------------------------------------------------------------------------------------------------ attention core
struct AttnCtx {
  Inkling* M;
  int v, T;  // T = this token's position + 1
  bool ring;
  float *q, *r;
  const float *g, *sink, *rel;
  float* o;
  // the KV preparation (one job per KV head before the attention heads that read it): convolutions of its channels,
  // clips, the store into the cache; the attention heads clip their own q and r rows (and scale them, global layers)
  const Lw* W = nullptr;
  VState* St = nullptr;
  float *k, *vv, *dk, *dv, *kc, *vc;
  float tau = 1.f;
  int slot = 0;
  std::atomic<int> kv_done[16];
  // the output projection and its short convolution after the heads, in the same fork-join (row chunks of 32 once every
  // head is done): o_proj's rows as run_rows computes them, then each chunk's channels of the convolution
  const Weight* wo = nullptr;
  float* att = nullptr;
  int o_chunks = 0;
  std::atomic<int> heads_left{0};
  std::atomic<int64_t> heads_done_ns{0};  // steady-clock ns when the last head finished (the fused output projection's start)
  // the grouped form (phase_attn_kv): the 3 heads' scores and 1 / z of each KV head, written by its score job
  float* scv = nullptr;  // [NH, n]
  float iz[64];
  std::atomic<int> sc_done[16];
  alignas(128) std::atomic<int> next{0};
};
void phase_attn(void* p, int, int) {
  AttnCtx& X = *(AttnCtx*)p;
  Inkling& M = *X.M;
  const ICfg& c = M.c;
  const int HD = c.HD, rep = c.NH / c.NKV, pos = X.T - 1;
  const VState& S = M.S->vs[X.v];
  const int t0 = X.ring ? std::max(0, pos - c.WIN + 1) : 0, n = pos - t0 + 1;
  static thread_local std::vector<float> sc, bias, acc;
  if ((int)sc.size() < n) sc.resize(n + 1024);
  if ((int)bias.size() < c.REXT) bias.resize(c.REXT);
  acc.resize(HD);
  const float scale = 1.f / std::sqrt((float)HD);
  const int NKV = c.NKV, QD = c.NH * HD, RD = c.NH * c.DREL;
  (void)QD;
  (void)RD;
  for (int job; (job = X.next.fetch_add(1, std::memory_order_relaxed)) < NKV + c.NH + X.o_chunks;) {
    if (job >= NKV + c.NH) {  // an output-projection chunk: rows [r0, r1), then the convolution of those channels
      while (X.heads_left.load(std::memory_order_acquire) > 0) __builtin_arm_yield();
      const int r0 = (job - NKV - c.NH) * 32, r1 = std::min(X.wo->rows, r0 + 32);
      mv(*X.wo, r0, r1, &M.xe.q, X.o, X.att + r0, M.ref);
      sconv_rng(X.St->conv[2], X.W->aconv, X.att, c.H, c.SC, X.att, r0, r1);
      continue;
    }
    if (job < NKV) {  // KV head j: the serial section's arithmetic on its channels
      const int j = job, c0 = j * HD, c1 = c0 + HD, KVD = NKV * HD;
      const Lw& W = *X.W;
      VState& St = *X.St;
      if (W.global) {
        dconv_rng(St.conv[0], W.kstat, X.dk, X.k, KVD, c.SC, c.HD, X.kc, c0, c1);
        dconv_rng(St.conv[1], W.vstat, X.dv, X.vv, KVD, c.SC, c.HD, X.vc, c0, c1);
      } else {
        sconv_rng(St.conv[0], W.kconv, X.k, KVD, c.SC, X.kc, c0, c1);
        sconv_rng(St.conv[1], W.vconv, X.vv, KVD, c.SC, X.vc, c0, c1);
      }
      float* kh = X.kc + (size_t)j * HD;
      float* vh = X.vc + (size_t)j * HD;
      l2clip(kh, nullptr, HD, c.qkclip);
      rmsnorm(vh, nullptr, HD, c.eps, vh);
      if (St.Kf) {
        std::memcpy(St.Kf + ((size_t)j * St.cap + X.slot) * HD, kh, sizeof(float) * HD);
        std::memcpy(St.Vf + ((size_t)j * St.cap + X.slot) * HD, vh, sizeof(float) * HD);
      } else {
        _Float16* kd = St.K + ((size_t)j * St.cap + X.slot) * HD;
        _Float16* vd = St.V + ((size_t)j * St.cap + X.slot) * HD;
        for (int i = 0; i < HD; i++) {
          kd[i] = (_Float16)kh[i];
          vd[i] = (_Float16)vh[i];
        }
      }
      X.kv_done[j].store(1, std::memory_order_release);
      continue;
    }
    const int hh = job - NKV;
    const int kvh = hh / rep;
    {  // this head's q and r rows: the clips, then the length scaling (global layers), as the serial section did
      const Lw& W = *X.W;
      l2clip(X.q + (size_t)hh * HD, W.qclip + (size_t)hh * HD, HD, c.qkclip);
      l2clip(X.r + (size_t)hh * c.DREL, W.rclip + (size_t)hh * c.DREL, c.DREL, c.relclip);
      if (W.global) {
        for (int i = hh * HD; i < (hh + 1) * HD; i++) X.q[i] *= X.tau;
        for (int i = hh * c.DREL; i < (hh + 1) * c.DREL; i++) X.r[i] *= X.tau;
      }
    }
    while (X.kv_done[kvh].load(std::memory_order_acquire) == 0) __builtin_arm_yield();
    const float* q = X.q + (size_t)hh * HD;
    const float* r = X.r + (size_t)hh * c.DREL;
    const int nb = std::min(n, c.REXT);  // the learned relative bias, distances 0 .. nb-1
    for (int d = 0; d < nb; d++) bias[d] = 0.f;
    for (int j = 0; j < c.DREL; j++) {  // in j order for every d (vectorises over d)
      const float rj = r[j], *rel = X.rel + (size_t)j * c.REXT;
      for (int d = 0; d < nb; d++) bias[d] += rj * rel[d];
    }
    float mx = X.sink[hh];
    for (int i = 0; i < n; i++) {
      const int t = t0 + i, d = pos - t;
      const int slot = X.ring ? t % S.cap : t;
      float dot;
      if (S.Kf) {
        dot = 0;
        const float* kr = S.Kf + ((size_t)kvh * S.cap + slot) * HD;
        for (int k = 0; k < HD; k++) dot += kr[k] * q[k];
      } else {
        const _Float16* kr = S.K + ((size_t)kvh * S.cap + slot) * HD;
        float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
        for (int k = 0; k < HD; k += 8) {
          float16x8_t kv = vld1q_f16((const float16_t*)kr + k);
          s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(kv)), vld1q_f32(q + k));
          s1 = vfmaq_f32(s1, vcvt_high_f32_f16(kv), vld1q_f32(q + k + 4));
        }
        dot = vaddvq_f32(vaddq_f32(s0, s1));
      }
      const float s = dot * scale + (d < c.REXT ? bias[d] : 0.f);
      sc[i] = s;
      mx = std::max(mx, s);
    }
    double z = std::exp((double)X.sink[hh] - mx);
    for (int i = 0; i < n; i++) z += (sc[i] = std::exp(sc[i] - mx));
    const float iz = (float)(1.0 / z);
    std::fill(acc.begin(), acc.end(), 0.f);
    for (int i = 0; i < n; i++) {
      const int t = t0 + i;
      const int slot = X.ring ? t % S.cap : t;
      if (S.Vf) {
        const float* vr = S.Vf + ((size_t)kvh * S.cap + slot) * HD;
        const float pw = sc[i] * iz;
        for (int k = 0; k < HD; k++) acc[k] += pw * vr[k];
        continue;
      }
      const _Float16* vr = S.V + ((size_t)kvh * S.cap + slot) * HD;
      float32x4_t pv = vdupq_n_f32(sc[i] * iz);
      for (int k = 0; k < HD; k += 8) {
        float16x8_t vv = vld1q_f16((const float16_t*)vr + k);
        vst1q_f32(&acc[k], vfmaq_f32(vld1q_f32(&acc[k]), vcvt_f32_f16(vget_low_f16(vv)), pv));
        vst1q_f32(&acc[k + 4], vfmaq_f32(vld1q_f32(&acc[k + 4]), vcvt_high_f32_f16(vv), pv));
      }
    }
    float* o = X.o + (size_t)hh * HD;
    const float* g = X.g + (size_t)hh * HD;
    for (int k = 0; k < HD; k++) o[k] = acc[k] * sigm(g[k]);
    if (!M.ref && HD % XB == 0)
      for (int j = 0; j < HD / XB; j++) xq_block(M.xe.q, (hh * HD) / XB + j, o + j * XB);
    if (X.o_chunks && X.heads_left.fetch_sub(1, std::memory_order_release) == 1)
      X.heads_done_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
                            std::memory_order_relaxed);
  }
}

// The same attention with each key and value row loaded once for the 3 query heads that share it (the prefill's
// phase_attn_g scheme, here for one position). Jobs, in dependency order: the KV preparation of each KV head (as
// phase_attn); a score job per KV head (its 3 heads' clips and scaling, relative bias, scores, max, exp and 1 / z); a
// value job per (KV head, 64 dimensions) once its score job is done (the 3 heads' sums over the keys kept in
// registers, 32 dimensions per pass, the gate, the activation block's quantisation); then the output projection's chunks
// once every value job is done. Each head's arithmetic is phase_attn's: the bias in j order, the score as two FMA
// accumulators over the dimensions then their sum, multiplied by the scale then the bias added (attn_score: not fused,
// as phase_attn compiles it), max / exp / double sum in key order, each output element one FMA chain over the keys
// from 0. fp16 K/V, 3 heads per KV head, HD a multiple of 64 (else phase_attn).
static inline float attn_score1(float dot, float scale, float b) {
  float m = dot * scale;
  __asm__("" : "+w"(m));
  return m + b;
}
void phase_attn_kv(void* p, int, int) {
  constexpr int REP = 3;
  AttnCtx& X = *(AttnCtx*)p;
  Inkling& M = *X.M;
  const ICfg& c = M.c;
  const int HD = c.HD, pos = X.T - 1, NKV = c.NKV, NHALF = HD / 64;
  const VState& S = M.S->vs[X.v];
  const int t0 = X.ring ? std::max(0, pos - c.WIN + 1) : 0, n = pos - t0 + 1;
  static thread_local std::vector<float> biasv;
  if ((int)biasv.size() < REP * c.REXT) biasv.resize((size_t)REP * c.REXT);
  const float scale = 1.f / std::sqrt((float)HD);
  const int nval = NKV * NHALF, total = 2 * NKV + nval + X.o_chunks;
  for (int job; (job = X.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
    if (job >= 2 * NKV + nval) {  // an output-projection chunk: rows [r0, r1), then the convolution of those channels
      while (X.heads_left.load(std::memory_order_acquire) > 0) __builtin_arm_yield();
      const int r0 = (job - 2 * NKV - nval) * 32, r1 = std::min(X.wo->rows, r0 + 32);
      mv(*X.wo, r0, r1, &M.xe.q, X.o, X.att + r0, M.ref);
      sconv_rng(X.St->conv[2], X.W->aconv, X.att, c.H, c.SC, X.att, r0, r1);
      continue;
    }
    if (job < NKV) {  // KV head j: as phase_attn
      const int j = job, c0 = j * HD, c1 = c0 + HD, KVD = NKV * HD;
      const Lw& W = *X.W;
      VState& St = *X.St;
      if (W.global) {
        dconv_rng(St.conv[0], W.kstat, X.dk, X.k, KVD, c.SC, c.HD, X.kc, c0, c1);
        dconv_rng(St.conv[1], W.vstat, X.dv, X.vv, KVD, c.SC, c.HD, X.vc, c0, c1);
      } else {
        sconv_rng(St.conv[0], W.kconv, X.k, KVD, c.SC, X.kc, c0, c1);
        sconv_rng(St.conv[1], W.vconv, X.vv, KVD, c.SC, X.vc, c0, c1);
      }
      float* kh = X.kc + (size_t)j * HD;
      float* vh = X.vc + (size_t)j * HD;
      l2clip(kh, nullptr, HD, c.qkclip);
      rmsnorm(vh, nullptr, HD, c.eps, vh);
      _Float16* kd = St.K + ((size_t)j * St.cap + X.slot) * HD;
      _Float16* vd = St.V + ((size_t)j * St.cap + X.slot) * HD;
      for (int i = 0; i < HD; i++) {
        kd[i] = (_Float16)kh[i];
        vd[i] = (_Float16)vh[i];
      }
      X.kv_done[j].store(1, std::memory_order_release);
      continue;
    }
    if (job < 2 * NKV) {  // the scores of KV head kvh's 3 heads
      const int kvh = job - NKV;
      const Lw& W = *X.W;
      float* sc[REP];
      const float* q[REP];
      float mx[REP];
      const int nb = std::min(n, c.REXT);
      for (int g = 0; g < REP; g++) {
        const int hh = kvh * REP + g;
        l2clip(X.q + (size_t)hh * HD, W.qclip + (size_t)hh * HD, HD, c.qkclip);
        l2clip(X.r + (size_t)hh * c.DREL, W.rclip + (size_t)hh * c.DREL, c.DREL, c.relclip);
        if (W.global) {
          for (int i = hh * HD; i < (hh + 1) * HD; i++) X.q[i] *= X.tau;
          for (int i = hh * c.DREL; i < (hh + 1) * c.DREL; i++) X.r[i] *= X.tau;
        }
        sc[g] = X.scv + (size_t)hh * n;
        q[g] = X.q + (size_t)hh * HD;
        const float* r = X.r + (size_t)hh * c.DREL;
        float* bs = biasv.data() + (size_t)g * c.REXT;
        for (int d = 0; d < nb; d++) bs[d] = 0.f;
        for (int j = 0; j < c.DREL; j++) {
          const float rj = r[j], *rel = X.rel + (size_t)j * c.REXT;
          for (int d = 0; d < nb; d++) bs[d] += rj * rel[d];
        }
        mx[g] = X.sink[hh];
      }
      while (X.kv_done[kvh].load(std::memory_order_acquire) == 0) __builtin_arm_yield();
      const _Float16* Kb = S.K + (size_t)kvh * S.cap * HD;
      for (int i = 0; i < n; i++) {
        const int t = t0 + i, d = pos - t;
        const int slot = X.ring ? t % S.cap : t;
        const _Float16* kr = Kb + (size_t)slot * HD;
        float32x4_t s0[REP], s1[REP];
        for (int g = 0; g < REP; g++) s0[g] = s1[g] = vdupq_n_f32(0);
        for (int k = 0; k < HD; k += 8) {
          float16x8_t kv = vld1q_f16((const float16_t*)kr + k);
          const float32x4_t lo = vcvt_f32_f16(vget_low_f16(kv)), hi = vcvt_high_f32_f16(kv);
          for (int g = 0; g < REP; g++) {
            s0[g] = vfmaq_f32(s0[g], lo, vld1q_f32(q[g] + k));
            s1[g] = vfmaq_f32(s1[g], hi, vld1q_f32(q[g] + k + 4));
          }
        }
        for (int g = 0; g < REP; g++) {
          const float dot = vaddvq_f32(vaddq_f32(s0[g], s1[g]));
          const float sv = attn_score1(dot, scale, d < c.REXT ? biasv[(size_t)g * c.REXT + d] : 0.f);
          sc[g][i] = sv;
          mx[g] = std::max(mx[g], sv);
        }
      }
      for (int g = 0; g < REP; g++) {
        const int hh = kvh * REP + g;
        double z = std::exp((double)X.sink[hh] - mx[g]);
        for (int i = 0; i < n; i++) z += (sc[g][i] = std::exp(sc[g][i] - mx[g]));
        X.iz[hh] = (float)(1.0 / z);
      }
      X.sc_done[kvh].store(1, std::memory_order_release);
      continue;
    }
    // values: KV head kvh, dimensions [k0, k0 + 64) of its 3 heads
    const int vj = job - 2 * NKV, kvh = vj / NHALF, k0 = (vj % NHALF) * 64;
    while (X.sc_done[kvh].load(std::memory_order_acquire) == 0) __builtin_arm_yield();
    const _Float16* Vb = S.V + (size_t)kvh * S.cap * HD;
    float accs[REP][64];
    const float* sc[REP];
    float iz[REP];
    for (int g = 0; g < REP; g++) {
      sc[g] = X.scv + (size_t)(kvh * REP + g) * n;
      iz[g] = X.iz[kvh * REP + g];
    }
    for (int k1 = k0; k1 < k0 + 64; k1 += 32) {
      float32x4_t a[REP][8];
      for (int g = 0; g < REP; g++)
        for (int u = 0; u < 8; u++) a[g][u] = vdupq_n_f32(0);
      for (int i = 0; i < n; i++) {
        const int t = t0 + i;
        const int slot = X.ring ? t % S.cap : t;
        const float16_t* vr = (const float16_t*)(Vb + (size_t)slot * HD + k1);
        float32x4_t vf[8];
        for (int u = 0; u < 4; u++) {
          float16x8_t vv = vld1q_f16(vr + 8 * u);
          vf[2 * u] = vcvt_f32_f16(vget_low_f16(vv));
          vf[2 * u + 1] = vcvt_high_f32_f16(vv);
        }
        for (int g = 0; g < REP; g++) {
          const float32x4_t pv = vdupq_n_f32(sc[g][i] * iz[g]);
          for (int u = 0; u < 8; u++) a[g][u] = vfmaq_f32(a[g][u], vf[u], pv);
        }
      }
      for (int g = 0; g < REP; g++)
        for (int u = 0; u < 8; u++) vst1q_f32(accs[g] + (k1 - k0) + 4 * u, a[g][u]);
    }
    for (int g = 0; g < REP; g++) {
      const int hh = kvh * REP + g;
      float* o = X.o + (size_t)hh * HD + k0;
      const float* gt = X.g + (size_t)hh * HD + k0;
      for (int k = 0; k < 64; k++) o[k] = accs[g][k] * sigm(gt[k]);
      if (!M.ref && 64 % XB == 0) xq_block(M.xe.q, (hh * HD + k0) / XB, o);
    }
    if (X.o_chunks && X.heads_left.fetch_sub(1, std::memory_order_release) == 1)
      X.heads_done_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(),
                            std::memory_order_relaxed);
  }
}

// message gates: out_i = (1 - tanh(F_i s_i)) * hs_i + tanh(G_i s_i) * msg_i, s_i = D_i [rms(hs_i), rms(msg_i)]
void gates(Inkling& M, const Gate* const* G, const float* const* hs, const float* const* mg, float* const* out, int n) {
  // One fork-join for the whole gate call, jobs in dependency order: the 2n input norms; gate i's down rows (chunks of 16,
  // once its two norms are done); gate i's forget and gain rows of each 256-element chunk and their combination (2 tanh per
  // element; once gate i's down rows are done). Each norm, row and element is computed as the serial norms, run_rows and
  // the combination loop computed it.
  const ICfg& c = M.c;
  const int H = c.H, RK = c.RANK;
  std::vector<float> cat((size_t)n * 2 * H), s((size_t)n * RK), fg((size_t)n * 2 * H);
  struct Comb {
    const Gate* const* G;
    float *cat, *s, *fg;
    const float* const* hs;
    const float* const* mg;
    float* const* out;
    int n, H, RK, dch;
    bool ref;
    std::atomic<int> norms_left[8], down_left[8];
    alignas(128) std::atomic<int> next{0};
  } C;
  C.G = G;
  C.cat = cat.data();
  C.s = s.data();
  C.fg = fg.data();
  C.hs = hs;
  C.mg = mg;
  C.out = out;
  C.n = n;
  C.H = H;
  C.RK = RK;
  C.ref = M.ref;
  C.dch = (RK + 15) / 16;
  for (int i = 0; i < n; i++) {
    C.norms_left[i].store(2, std::memory_order_relaxed);
    C.down_left[i].store(C.dch, std::memory_order_relaxed);
    M.T->bytes_res += (double)G[i]->down.bytes() + (double)G[i]->forget.bytes() + (double)G[i]->gain.bytes();
  }
  M.core->pool->run_any(
      [](void* p, int, int) {
        Comb& C = *(Comb*)p;
        const int per = (C.H + 255) / 256, nn = 2 * C.n, nd = C.dch * C.n, total = nn + nd + per * C.n;
        for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
          if (ch < nn) {  // input norm: rms(hs_i) or rms(msg_i) into cat
            const int i = ch / 2, w = ch % 2;
            rmsnorm(w ? C.mg[i] : C.hs[i], nullptr, C.H, 1e-6f, C.cat + (size_t)i * 2 * C.H + (size_t)w * C.H);
            C.norms_left[i].fetch_sub(1, std::memory_order_release);
            continue;
          }
          if (ch < nn + nd) {  // down rows of gate i
            const int i = (ch - nn) / C.dch, r0 = ((ch - nn) % C.dch) * 16, r1 = std::min(C.RK, r0 + 16);
            while (C.norms_left[i].load(std::memory_order_acquire) > 0) __builtin_arm_yield();
            mv(C.G[i]->down, r0, r1, nullptr, C.cat + (size_t)i * 2 * C.H, C.s + (size_t)i * C.RK + r0, C.ref);
            C.down_left[i].fetch_sub(1, std::memory_order_release);
            continue;
          }
          const int c2 = ch - nn - nd, i = c2 / per, d0 = (c2 % per) * 256, d1 = std::min(C.H, d0 + 256);
          while (C.down_left[i].load(std::memory_order_acquire) > 0) __builtin_arm_yield();
          float* f = C.fg + (size_t)i * 2 * C.H;
          float* g = f + C.H;
          const float* sv = C.s + (size_t)i * C.RK;
          for (int r0 = d0; r0 < d1; r0 += 128) {  // run_rows' chunk of 128 rows
            const int r1 = std::min(d1, r0 + 128);
            mv(C.G[i]->forget, r0, r1, nullptr, sv, f + r0, C.ref);
            mv(C.G[i]->gain, r0, r1, nullptr, sv, g + r0, C.ref);
          }
          for (int d = d0; d < d1; d++) C.out[i][d] = (1.f - std::tanh(f[d])) * C.hs[i][d] + std::tanh(g[d]) * C.mg[i][d];
        }
      },
      &C);
}

// ------------------------------------------------------------------------------------------------ one application
void Inkling::layer(int v, const float* m4) {
  const int L = c.visits[v].first, pos = S->pos;
  const Lw& W = lw[L];
  const int H = c.H, QD = c.NH * c.HD, KVD = c.NKV * c.HD, RD = c.NH * c.DREL;
  VState& St = S->vs[v];
  float* hres = tmp.data();  // block residual
  float* nres = hres + H;    // rms(h)
  float* ain = nres + H;     // attention inputs [3, H]
  float* gin = ain + 3 * H;  // message-gated inputs [3, H]
  float* qn = gin + 3 * H;   // normed q / k / v inputs [3, H]
  float* att = qn + 3 * H;   // attention output [H]
  float* res = att + H;      // residual after attention
  float* mlp = res + H;      // mlp input [H]
  float* y = mlp + H;        // MoE output
  float* y2 = y + H;
  float* nst = y2 + H;
  float* xm = nst + H;       // MoE input
  float* q = xm + H;         // projections
  float* k = q + QD;
  float* vv = k + KVD;
  float* r = vv + KVD;
  float* g = r + RD;
  float* sk = g + QD;
  float* dk = sk + 64;
  float* dv = dk + 64;
  float* kc = dv + 64;
  float* vc = kc + KVD;
  float* o = vc + KVD;
  auto t0 = clk::now();
  stage_mark(ST_DEPTH);
  std::memcpy(hres, h.data(), sizeof(float) * H);
  rmsnorm(h.data(), nullptr, H, 1e-6f, nres);
  const int nq = W.global ? 3 : 1;
  if (L > 0) {
    const float* src[16];
    for (int i = 0; i < nblk; i++) src[i] = blk.data() + (size_t)i * H;
    src[nblk] = nres;
    depth_attn(src, nblk + 1, W.aq, nq, H, ain);
  } else {
    for (int i = 0; i < nq; i++) std::memcpy(ain + (size_t)i * H, h.data(), sizeof(float) * H);
  }
  auto t1 = clk::now();
  T->depth += ms_between(t0, t1);
  // the three attention inputs (sliding layers: one input; messages make three)
  const float* qkv_in[3];
  for (int i = 0; i < 3; i++) qkv_in[i] = ain + (size_t)(W.global ? i : 0) * H;
  if (m4) {
    const Gate* G[3] = {&W.mg[0], &W.mg[1], &W.mg[2]};
    const float* mg[3] = {m4, m4 + H, m4 + 2 * H};
    float* out[3] = {gin, gin + H, gin + 2 * H};
    stage_mark(ST_GATES);
    gates(*this, G, qkv_in, mg, out, 3);
    for (int i = 0; i < 3; i++) qkv_in[i] = gin + (size_t)i * H;
  }
  const bool three = W.global || m4;
  stage_mark(ST_NORMQ);
  {  // the three inputs' norms side by side (rmsnorm_n: each rmsnorm's bits)
    const float* nw[3] = {W.global ? W.lnq : W.ln_in, W.global ? W.lnk : W.ln_in, W.global ? W.lnv : W.ln_in};
    float* no[3] = {qn, qn + H, qn + 2 * H};
    rmsnorm_n(qkv_in, nw, three ? 3 : 1, H, c.eps, no);
  }
  const float* qi = qn;
  const float* ki = three ? qn + H : qn;
  const float* vi = three ? qn + 2 * H : qn;
  auto t2 = clk::now();
  T->msg += ms_between(t1, t2);
  // projections, one fork-join (their inputs quantised inside it)
  const XQ* xq_q = &xa.q;
  const XQ* xq_k = three ? &xb.q : &xa.q;
  const XQ* xq_v = three ? &xc.q : &xa.q;
  Job jb[10];
  int nj = 0;
  jb[nj++] = {&W.q, xq_q, qi, q};
  jb[nj++] = {&W.g, xq_q, qi, g};
  jb[nj++] = {&W.k, xq_k, ki, k};
  jb[nj++] = {&W.v, xq_v, vi, vv};
  jb[nj++] = {&W.r, xq_q, qi, r};
  jb[nj++] = {&W.sink, nullptr, qi, sk};
  stage_mark(ST_ATTN_PROJ);
  if (W.global) {
    jb[nj++] = {&W.kwp, nullptr, ki, dk};
    jb[nj++] = {&W.vwp, nullptr, vi, dv};
  }
  if (ref) {
    rows_run(jb, nj, 32);
  } else {
    XBuf* qs[3] = {&xa, &xb, &xc};
    const float* xs[3] = {qi, ki, vi};
    const int ks[3] = {H, H, H};
    run_rows_q(*core->pool, jb, nj, 32, ref, &T->bytes_res, qs, xs, ks, three ? 3 : 1);
  }
  auto t3 = clk::now();
  T->attn_proj += ms_between(t2, t3);
  // short convolutions, clips, the KV cache: inside the attention's fork-join (a job per KV head, before its heads)
  if (!St.ring && pos >= St.cap) {
    fprintf(stderr, "context full (%d tokens)\n", St.cap);
    abort();
  }
  AttnCtx X;
  X.M = this;
  X.v = v;
  X.T = pos + 1;
  X.ring = St.ring;
  X.q = q;
  X.r = r;
  X.g = g;
  X.sink = sk;
  X.rel = W.rel;
  X.o = o;
  X.W = &W;
  X.St = &St;
  X.k = k;
  X.vv = vv;
  X.dk = dk;
  X.dv = dv;
  X.kc = kc;
  X.vc = vc;
  X.tau = W.global ? 1.f + c.logalpha * std::log(std::max(((float)pos + 1.f) / c.logfloor, 1.f)) : 1.f;
  X.slot = St.ring ? pos % St.cap : pos;
  if (c.NKV > 16) {
    fprintf(stderr, "more than 16 KV heads\n");
    abort();
  }
  for (int j = 0; j < c.NKV; j++) X.kv_done[j].store(0, std::memory_order_relaxed);
  xe.reset(QD);
  const bool fuse_o = ref || c.HD % XB == 0;  // the heads quantise their own outputs: the output projection can follow them
  if (fuse_o) {
    X.wo = &W.o;
    X.att = att;
    X.o_chunks = (W.o.rows + 31) / 32;
    X.heads_left.store(c.NH, std::memory_order_relaxed);
    T->bytes_res += (double)W.o.bytes();
  }
  // the grouped form: fp16 K/V, 3 heads per KV head, HD a multiple of 64 (the activation blocks), the output projection fused
  const bool grouped = !St.Kf && c.NH == 3 * c.NKV && c.HD % 64 == 0 && fuse_o && !ref;
  if (grouped) {
    const int n = pos - (St.ring ? std::max(0, pos - c.WIN + 1) : 0) + 1;
    if (att_sc.size() < (size_t)c.NH * n) att_sc.resize((size_t)c.NH * (n + 1024));
    X.scv = att_sc.data();
    for (int j = 0; j < c.NKV; j++) X.sc_done[j].store(0, std::memory_order_relaxed);
    X.heads_left.store(c.NKV * (c.HD / 64), std::memory_order_relaxed);
  }
  stage_mark(ST_ATTN_CORE);
  core->pool->run_any(grouped ? phase_attn_kv : phase_attn, &X);
  auto t4 = clk::now();
  T->split_heads_out(t3, t4, fuse_o ? X.heads_done_ns.load() : 0);  // the fused output projection's share: attn_out
  if (!fuse_o) {
    if (!ref && c.HD % XB != 0) xe.prep(o, QD);
    Job jo = {&W.o, &xe.q, o, att};
    rows_run(&jo, 1, 32);
    sconv(St.conv[2], W.aconv, att, H, c.SC, att);
  }
  auto t5 = clk::now();
  T->attn_out += ms_between(t4, t5);
  // the attention residual, the mlp input
  stage_mark(ST_DEPTH);
  if (L % c.BLOCK == 0) {
    std::memcpy(blk.data() + (size_t)nblk * H, nres, sizeof(float) * H);
    nblk++;
    std::memcpy(res, att, sizeof(float) * H);
  } else {
    for (int i = 0; i < H; i++) res[i] = hres[i] + att[i];
  }
  rmsnorm(res, nullptr, H, 1e-6f, nst);
  {
    const float* src[16];
    for (int i = 0; i < nblk; i++) src[i] = blk.data() + (size_t)i * H;
    src[nblk] = nst;
    depth_attn(src, nblk + 1, W.mq, 1, H, mlp);
  }
  auto t6 = clk::now();
  T->depth += ms_between(t5, t6);
  if (m4) {
    const Gate* G[1] = {&W.mg[3]};
    const float* hs[1] = {mlp};
    const float* mg[1] = {m4 + 3 * H};
    float* out[1] = {mlp};
    stage_mark(ST_GATES);
    gates(*this, G, hs, mg, out, 1);
  }
  stage_mark(ST_NORMQ);
  rmsnorm(mlp, W.mlnorm, H, c.eps, xm);
  T->msg += ms_since(t6);
  if (ra_mode) {  // run-ahead: the MoE input is all it needs
    ra_xm = xm;
    return;
  }
  cur_res = res;
  core->moe_step(v, xm, y, views[L]);
  auto t8 = clk::now();
  stage_mark(ST_CONV);
  sconv(St.conv[3], W.mconv, y, H, c.SC, y2);
  for (int i = 0; i < H; i++) h[i] = res[i] + y2[i];
  T->misc += ms_since(t8);
}

// ------------------------------------------------------------------------------------------------ one token
#if DM_COUNTERS
// Counting builds only: DMENG_PROJ_REPLAY=N runs the decode's 48 attention-projection fork-joins N times before the
// first step, on the model's own pool, weights and input buffers (stage "span_copies"), so their cycles per byte can be
// set beside the same calls inside decode (stage "attn_proj"). Passes alternate between the decode's form (inputs
// quantised inside the fork-join, run_rows_q; stage "span_copies") and inputs quantised by the caller before it
// (run_rows; stage "embed").
static void proj_replay(Inkling& M) {
  static const int N = getenv("DMENG_PROJ_REPLAY") ? atoi(getenv("DMENG_PROJ_REPLAY")) : 0;
  static bool done = false;
  if (!N || done) return;
  done = true;
  const ICfg& c = M.c;
  const int H = c.H, QD = c.NH * c.HD, KVD = c.NKV * c.HD, RD = c.NH * c.DREL;
  std::vector<float> in((size_t)3 * H), out((size_t)2 * QD + 2 * KVD + RD + 256);
  for (size_t i = 0; i < in.size(); i++) in[i] = std::sin(0.37f * (float)i);
  for (int rep = 0; rep < 2 * N; rep++) {
    const bool preq = rep & 1;
    stage_mark(preq ? ST_EMBED : ST_SPAN);
    for (int v = 0; v < c.NV; v++) {
      const int L = c.visits[v].first;
      const Lw& W = M.lw[L];
      const bool three = W.global || c.visits[v].second > 0;
      const float* qi = in.data();
      const float* ki = three ? qi + H : qi;
      const float* vi = three ? qi + 2 * H : qi;
      float* q = out.data();
      float* g = q + QD;
      float* k = g + QD;
      float* vv = k + KVD;
      float* r = vv + KVD;
      float* sk = r + RD;
      Job jb[10];
      int nj = 0;
      const XQ* xq_k = three ? &M.xb.q : &M.xa.q;
      const XQ* xq_v = three ? &M.xc.q : &M.xa.q;
      jb[nj++] = {&W.q, &M.xa.q, qi, q};
      jb[nj++] = {&W.g, &M.xa.q, qi, g};
      jb[nj++] = {&W.k, xq_k, ki, k};
      jb[nj++] = {&W.v, xq_v, vi, vv};
      jb[nj++] = {&W.r, &M.xa.q, qi, r};
      jb[nj++] = {&W.sink, nullptr, qi, sk};
      XBuf* qs[3] = {&M.xa, &M.xb, &M.xc};
      const float* xs[3] = {qi, ki, vi};
      const int ks[3] = {H, H, H};
      if (preq) {
        for (int i = 0; i < (three ? 3 : 1); i++) qs[i]->prep(xs[i], H);
        run_rows(*M.core->pool, jb, nj, 32, false, nullptr);
      } else {
        run_rows_q(*M.core->pool, jb, nj, 32, false, nullptr, qs, xs, ks, three ? 3 : 1);
      }
    }
  }
  stage_mark(ST_OTHER);
}
#endif

void Inkling::step(State& s0, int32_t tok, Want want) {
  S = (IState*)&s0;
#if DM_COUNTERS
  proj_replay(*this);
#endif
  const int H = c.H;
  auto t0 = clk::now();
  stage_mark(ST_EMBED);
  rows.row(tok, emb.data());
  rmsnorm(emb.data(), embed_norm, H, c.eps, h.data());
  embn = h;  // the transformers embed_tokens output: the routers' `emb`
  T->embed += ms_since(t0);
  core->token_start(S->pos, tok, embn.data());
  nblk = 0;
  int v = 0;
  // the draft head's features: the hidden state entering applications 5, 23, 41, and the post-final-norm hidden state
  auto tap = [&](int a) {
    if (!taps_.empty() && (a == 5 || a == 23 || a == 41)) std::memcpy(taps_.data() + (size_t)(a == 5 ? 0 : a == 23 ? 1 : 2) * H, h.data(), sizeof(float) * H);
  };
  for (int L = 0; L < c.CS; L++) {
    tap(v);
    layer(v++, nullptr);
  }
  const int nblk_pre = nblk;
  {
    auto tm = clk::now();  // (the loop's bookkeeping: depth-attention's stage, as the restore at each pass)
    h_pre.assign(h.begin(), h.end());
    T->depth += ms_since(tm);
  }
  const int NC = c.CE - c.CS;
  for (int p = 0; p < c.NP; p++) {
    if (p > 0) {
      auto tm = clk::now();
      stage_mark(ST_DEPTH);
      // the bank: the blocks the previous pass appended after its first one, and rms of its final hidden state
      const float* src[16];
      int nb = 0;
      for (int i = nblk_pre + 1; i < nblk; i++) src[nb++] = blk.data() + (size_t)i * H;
      rmsnorm(h.data(), nullptr, H, 1e-6f, bank.data());
      src[nb++] = bank.data();
      // the 4 messages of each looped layer, every query row its own job (depth_attn computes the rows independently,
      // so each row's arithmetic is the serial loop's): one fork-join instead of 12 serial calls
      struct Mc {
        const float* const* src;
        int nb, H, rows;
        const float* q;
        float* out;
        alignas(128) std::atomic<int> next{0};
      } mc{src, nb, H, NC * 4, msgq, msg.data()};
      core->pool->run_any(
          [](void* p, int, int) {
            Mc& M = *(Mc*)p;
            for (int r; (r = M.next.fetch_add(1, std::memory_order_relaxed)) < M.rows;)
              depth_attn(M.src, M.nb, M.q + (size_t)r * M.H, 1, M.H, M.out + (size_t)r * M.H);
          },
          &mc);
      h = h_pre;
      nblk = nblk_pre;
      T->depth += ms_since(tm);
    }
    for (int i = 0; i < NC; i++) {
      tap(v);
      layer(v++, p > 0 ? msg.data() + (size_t)i * 4 * H : nullptr);
    }
  }
  for (int L = c.CE; L < c.NL; L++) {
    tap(v);
    layer(v++, nullptr);
  }
  if (want != Want::NONE || !taps_.empty()) {  // the head's input (and, with taps kept, the draft head's fourth feature)
    auto th = clk::now();
    stage_mark(ST_HEAD);
    float* nf = tmp.data();
    float* fin = nf + H;
    float* hn = fin + H;
    rmsnorm(h.data(), nullptr, H, 1e-6f, nf);
    const float* src[16];
    for (int i = 0; i < nblk; i++) src[i] = blk.data() + (size_t)i * H;
    src[nblk] = nf;
    depth_attn(src, nblk + 1, final_q, 1, H, fin);
    rmsnorm(fin, fnorm, H, c.eps, hn);
    if (!taps_.empty()) std::memcpy(taps_.data() + 3 * (size_t)H, hn, sizeof(float) * H);
    if (dh) dh->add_taps(taps_.data(), 1, S->pos);
    T->head += ms_since(th);
  }
  if (want != Want::NONE) {
    auto th = clk::now();
    stage_mark(ST_HEAD);
    float* hn = tmp.data() + 2 * H;
    if (!ref) xa.prep(hn, H);
    if (fast_head && (want == Want::GREEDY || (want == Want::TOPK && head_topk > 0)) && !ref) {
      // the screened head with the softcap: the capped logits of every row that can be in the top k (k = 1: the argmax),
      // -inf for the rest; when the cap makes a row left out equal to the k-th kept one (sh.ok false), the full head
      greedy_ = sh.run(*core->pool, xa.q, want == Want::GREEDY ? 1 : head_topk, lgts.data(), c.softcap);
      if (sh.ok) {
        T->head += ms_since(th);
        S->pos++;
        core->token_end();
        stage_mark(ST_OTHER);
        return;
      }
      gh_fallback++;
    }
    Job jh = {&head, &xa.q, hn, full.data()};
    struct Sc {
      const float* full;
      float* lg;
      int vu;
      float cap;
    } sc{full.data(), lgts.data(), c.VU, c.softcap};
    // the softcap of each chunk's rows by the thread that computed them (was a serial loop over the 200K logits)
    run_rows_post(*core->pool, jh, 64, ref, &T->bytes_res, [](void* p, int r0, int r1) {
      const Sc& S = *(const Sc*)p;
      for (int i = r0; i < std::min(r1, S.vu); i++) S.lg[i] = softcap(S.full[i], S.cap);
    }, &sc);
    if (fast_head && want == Want::GREEDY) greedy_ = argmax(lgts.data(), c.VU);  // the fallback's answer
    T->head += ms_since(th);
  }
  S->pos++;
  core->token_end();
  stage_mark(ST_OTHER);
}

// ------------------------------------------------------------------------------------------------ a chunk of positions
// Application-major over n positions (batched prefill): each application's resident matrices are read once for the chunk
// (the packed i8mm gemm from 6 positions, else the multi-vector kernels: bit-identical to the one-vector ones), its MoE
// through core.moe_chunk (each expert read once for the chunk), and per position exactly the single step's arithmetic:
// depth attention, message gates, norms, convolutions (in position order: each updates the state), clips, KV writes,
// attention. Keys and values are written and attended in groups of RING_SLACK + 1 positions, so a sliding ring never
// overwrites a key an earlier position of the group still reads. The state and logits equal n single steps'.
constexpr int CHUNK_MAX = 512;

void Inkling::chunk_alloc(int n) {
  const int QD = c.NH * c.HD, mk = std::max({c.H, QD, 2 * c.H});
  while ((int)pp.size() < n) {
    std::unique_ptr<PP> p(new PP());
    p->tmp.assign((size_t)24 * c.H + 4 * QD + 256, 0.f);  // layer()'s layout ends below 23 H + 4 QD + 192
    p->m4.assign((size_t)4 * c.H, 0.f);
    p->xa.init(mk, prec);
    p->xb.init(mk, prec);
    p->xc.init(mk, prec);
    p->xe.init(mk, prec);
    pp.push_back(std::move(p));
  }
}

// the batched prefill's scratch back to the system after a prefill (prefill_release; the server's default): the span
// arrays (every prompt position's residual, block states and message sources: 0.74 GB after an 8K-token prompt on the
// real model), the sub-chunk scratch past what a verify pass uses, the packed inputs and the draft-head taps. The next
// prefill or pass allocates what it needs again; nothing computed depends on it.
void Inkling::release_scratch() {
  PageVec<float>().swap(sp_h);
  PageVec<float>().swap(sp_hpre);
  PageVec<float>().swap(sp_blk);
  PageVec<float>().swap(sp_msrc);
  PageVec<uint8_t>().swap(span_snap);
  std::vector<float>().swap(ctaps);
  sp_n = 0;
  if (pp.size() > (size_t)RING_SLACK + 1) pp.resize(RING_SLACK + 1);
  for (XQB* q : {&xqb_a, &xqb_b, &xqb_c, &xqb_e, &xqb_h}) {
    std::free(q->mem);
    q->mem = nullptr;
    q->cap = 0;
  }
  n_scratch_releases++;
}

void Inkling::span_blocks() {
  if (!NBMAX) {
    for (int L = 0; L < c.NL; L++) NBMAX += L % c.BLOCK == 0;
    for (int L = c.CS; L < c.CE; L++) NMS += L % c.BLOCK == 0;  // the pass's blocks after its first, plus the bank
    if (NBMAX + 1 > 16 || NMS < 1) {
      fprintf(stderr, "inkling: %d block states (at most 15)\n", NBMAX);
      abort();
    }
  }
}

void Inkling::span_alloc(int n) {
  span_blocks();
  if (n <= sp_n) return;
  sp_h.resize((size_t)n * c.H);
  sp_hpre.resize((size_t)n * c.H);
  sp_blk.resize((size_t)n * NBMAX * c.H);
  sp_msrc.resize((size_t)n * NMS * c.H);
  sp_n = n;
}

// message gates of n inputs per gate, every input its own position: the single step's gates() with the rows of all
// positions in each fork-join
static void gates_n(Inkling& M, const Gate* const* G, int ng, const float* const* hs, const float* const* mg, float* const* out, int n) {
  const ICfg& c = M.c;
  const int H = c.H, RK = c.RANK;
  const int N = ng * n;  // input (gate g, position i) at g * n + i
  std::vector<float> cat((size_t)N * 2 * H), sv((size_t)N * RK), fg((size_t)N * 2 * H);
  parallel_for(*M.core->pool, N, [&](int k) {  // the two norms side by side
    const float* x2[2] = {hs[k], mg[k]};
    float* o2[2] = {cat.data() + (size_t)k * 2 * H, cat.data() + (size_t)k * 2 * H + H};
    rmsnorm_n(x2, nullptr, 2, H, 1e-6f, o2);
  });
  std::vector<const float*> xin(N), sin(N);
  std::vector<float*> so(N), fo(N), go(N);
  for (int k = 0; k < N; k++) {
    xin[k] = cat.data() + (size_t)k * 2 * H;
    so[k] = sv.data() + (size_t)k * RK;
    sin[k] = so[k];
    fo[k] = fg.data() + (size_t)k * 2 * H;
    go[k] = fo[k] + H;
  }
  std::vector<JobN> j1(ng), j2(2 * ng);
  for (int g = 0; g < ng; g++) j1[g] = {&G[g]->down, n, nullptr, xin.data() + (size_t)g * n, so.data() + (size_t)g * n};
  run_rows_n(*M.core->pool, j1.data(), ng, 16, M.ref, &M.T->bytes_res);
  for (int g = 0; g < ng; g++) {
    j2[2 * g] = {&G[g]->forget, n, nullptr, sin.data() + (size_t)g * n, fo.data() + (size_t)g * n};
    j2[2 * g + 1] = {&G[g]->gain, n, nullptr, sin.data() + (size_t)g * n, go.data() + (size_t)g * n};
  }
  run_rows_n(*M.core->pool, j2.data(), 2 * ng, 128, M.ref, &M.T->bytes_res);
  struct Comb {
    const float* fg;
    const float* const* hs;
    const float* const* mg;
    float* const* out;
    int n, H;
    alignas(128) std::atomic<int> next{0};
  } C{fg.data(), hs, mg, out, N, H};
  M.core->pool->run(
      [](void* p, int, int) {
        Comb& C = *(Comb*)p;
        const int per = (C.H + 255) / 256;
        for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < per * C.n;) {
          const int i = ch / per, d0 = (ch % per) * 256, d1 = std::min(C.H, d0 + 256);
          const float *f = C.fg + (size_t)i * 2 * C.H, *g = f + C.H;
          for (int d = d0; d < d1; d++) C.out[i][d] = (1.f - std::tanh(f[d])) * C.hs[i][d] + std::tanh(g[d]) * C.mg[i][d];
        }
      },
      &C);
}

namespace {
struct AttnNCtx {  // attention of positions [i0, i1) of a chunk (each at its own position, causal)
  Inkling* M;
  int v, pos0, i0, i1;
  bool ring;
  const float *rel;
  const float *const *q, *const *r, *const *g, *const *sink;
  float* const* o;
  XQ* const* xe;
  alignas(128) std::atomic<int> next{0};
};
void phase_attn_n(void* p, int, int) {
  AttnNCtx& X = *(AttnNCtx*)p;
  Inkling& M = *X.M;
  const ICfg& c = M.c;
  const int HD = c.HD, rep = c.NH / c.NKV, np = X.i1 - X.i0;
  const VState& S = M.S->vs[X.v];
  static thread_local std::vector<float> sc, bias, acc;
  if ((int)bias.size() < c.REXT) bias.resize(c.REXT);
  acc.resize(HD);
  const float scale = 1.f / std::sqrt((float)HD);
  for (int job; (job = X.next.fetch_add(1, std::memory_order_relaxed)) < np * c.NH;) {
    const int ip = X.i0 + job / c.NH, hh = job % c.NH, pos = X.pos0 + ip;
    const int t0 = X.ring ? std::max(0, pos - c.WIN + 1) : 0, n = pos - t0 + 1;
    if ((int)sc.size() < n) sc.resize(n + 1024);
    const int kvh = hh / rep;
    const float* q = X.q[ip] + (size_t)hh * HD;
    const float* r = X.r[ip] + (size_t)hh * c.DREL;
    const int nb = std::min(n, c.REXT);
    for (int d = 0; d < nb; d++) bias[d] = 0.f;
    for (int j = 0; j < c.DREL; j++) {
      const float rj = r[j], *rel = X.rel + (size_t)j * c.REXT;
      for (int d = 0; d < nb; d++) bias[d] += rj * rel[d];
    }
    float mx = X.sink[ip][hh];
    for (int i = 0; i < n; i++) {
      const int t = t0 + i, d = pos - t;
      const int slot = X.ring ? t % S.cap : t;
      float dot;
      if (S.Kf) {
        dot = 0;
        const float* kr = S.Kf + ((size_t)kvh * S.cap + slot) * HD;
        for (int k = 0; k < HD; k++) dot += kr[k] * q[k];
      } else {
        const _Float16* kr = S.K + ((size_t)kvh * S.cap + slot) * HD;
        float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
        for (int k = 0; k < HD; k += 8) {
          float16x8_t kv = vld1q_f16((const float16_t*)kr + k);
          s0 = vfmaq_f32(s0, vcvt_f32_f16(vget_low_f16(kv)), vld1q_f32(q + k));
          s1 = vfmaq_f32(s1, vcvt_high_f32_f16(kv), vld1q_f32(q + k + 4));
        }
        dot = vaddvq_f32(vaddq_f32(s0, s1));
      }
      const float s = dot * scale + (d < c.REXT ? bias[d] : 0.f);
      sc[i] = s;
      mx = std::max(mx, s);
    }
    double z = std::exp((double)X.sink[ip][hh] - mx);
    for (int i = 0; i < n; i++) z += (sc[i] = std::exp(sc[i] - mx));
    const float iz = (float)(1.0 / z);
    std::fill(acc.begin(), acc.end(), 0.f);
    for (int i = 0; i < n; i++) {
      const int t = t0 + i;
      const int slot = X.ring ? t % S.cap : t;
      if (S.Vf) {
        const float* vr = S.Vf + ((size_t)kvh * S.cap + slot) * HD;
        const float pw = sc[i] * iz;
        for (int k = 0; k < HD; k++) acc[k] += pw * vr[k];
        continue;
      }
      const _Float16* vr = S.V + ((size_t)kvh * S.cap + slot) * HD;
      float32x4_t pv = vdupq_n_f32(sc[i] * iz);
      for (int k = 0; k < HD; k += 8) {
        float16x8_t vv = vld1q_f16((const float16_t*)vr + k);
        vst1q_f32(&acc[k], vfmaq_f32(vld1q_f32(&acc[k]), vcvt_f32_f16(vget_low_f16(vv)), pv));
        vst1q_f32(&acc[k + 4], vfmaq_f32(vld1q_f32(&acc[k + 4]), vcvt_high_f32_f16(vv), pv));
      }
    }
    float* o = X.o[ip] + (size_t)hh * HD;
    const float* g = X.g[ip] + (size_t)hh * HD;
    for (int k = 0; k < HD; k++) o[k] = acc[k] * sigm(g[k]);
    if (!M.ref && HD % XB == 0)
      for (int j = 0; j < HD / XB; j++) xq_block(*X.xe[ip], (hh * HD) / XB + j, o + j * XB);
  }
}
// The same attention, one job per (position, KV head) for its REP query heads: every key and value row is loaded once
// for the REP heads, and the value sums are kept in registers (32 dimensions x REP heads per pass over the keys) instead
// of a load and store per key. Each head's arithmetic is phase_attn_n's (the single step's): the relative bias in j
// order, the score as two FMA accumulators over the dimensions in order then their sum, max / exp / double sum in key
// order, each output element one FMA chain over the keys from 0, then the gate. fp16 K/V only (the reference path and
// other head ratios use phase_attn_n).
// the score as the single step's code compiles it: a multiply, then an add (not fused: checked against phase_attn_n).
// -ffp-contract=fast lets the backend fuse any multiply-add, so an empty asm keeps the product a separate rounding.
static inline float attn_score(float dot, float scale, float b) {
  float m = dot * scale;
  __asm__("" : "+w"(m));
  return m + b;
}
template <int REP>
void phase_attn_g(void* p, int, int) {
  AttnNCtx& X = *(AttnNCtx*)p;
  Inkling& M = *X.M;
  const ICfg& c = M.c;
  const int HD = c.HD, np = X.i1 - X.i0, NKV = c.NKV;
  const VState& S = M.S->vs[X.v];
  static thread_local std::vector<float> scv, biasv, bfw;
  if ((int)biasv.size() < REP * c.REXT) biasv.resize((size_t)REP * c.REXT);
  const float scale = 1.f / std::sqrt((float)HD);
  for (int job; (job = X.next.fetch_add(1, std::memory_order_relaxed)) < np * NKV;) {
    // KV-head-major: the threads work on neighbouring positions of one head at a time, whose keys and values they share
    // in the caches (a global layer's whole cache does not fit them); the order of the jobs changes no result
    const int kvh = job / np, ip = X.i0 + job % np, pos = X.pos0 + ip;
    const int t0 = X.ring ? std::max(0, pos - c.WIN + 1) : 0, n = pos - t0 + 1;
    if ((int)scv.size() < REP * n) scv.resize((size_t)REP * (n + 1024));
    float* sc[REP];
    float* bias[REP];
    const float* q[REP];
    float mx[REP];
    const int nb = std::min(n, c.REXT);
    for (int g = 0; g < REP; g++) {
      const int hh = kvh * REP + g;
      sc[g] = scv.data() + (size_t)g * n;
      bias[g] = biasv.data() + (size_t)g * c.REXT;
      q[g] = X.q[ip] + (size_t)hh * HD;
      const float* r = X.r[ip] + (size_t)hh * c.DREL;
      float* bs = bias[g];
      for (int d = 0; d < nb; d++) bs[d] = 0.f;
      for (int j = 0; j < c.DREL; j++) {
        const float rj = r[j], *rel = X.rel + (size_t)j * c.REXT;
        for (int d = 0; d < nb; d++) bs[d] += rj * rel[d];
      }
      mx[g] = X.sink[ip][hh];
    }
    const _Float16* Kb = S.K + (size_t)kvh * S.cap * HD;
    const _Float16* Vb = S.V + (size_t)kvh * S.cap * HD;
    // each key's relative bias in key order (bias[g][pos - t], 0 past REXT), so four keys' biases are one load
    if ((int)bfw.size() < REP * (n + 4)) bfw.resize((size_t)REP * (n + 1028));
    float* bf[REP];
    for (int g = 0; g < REP; g++) {
      bf[g] = bfw.data() + (size_t)g * (n + 4);
      for (int i = 0; i < n; i++) {
        const int d = pos - (t0 + i);
        bf[g][i] = d < c.REXT ? bias[g][d] : 0.f;
      }
    }
    // the scores four keys at a time (each query's 8 dimensions loaded once for the four): per key and head the same
    // two accumulators over the dimensions in order; the four keys' sums (l0 + l1) + (l2 + l3), as vaddvq gives each, by
    // two pairwise adds; the score a multiply then an add (not fused), as attn_score; the maximum in key order
    const float32x4_t vsc = vdupq_n_f32(scale);
    int i4 = 0;
    for (; i4 + 4 <= n; i4 += 4) {
      const _Float16* kr4[4];
      for (int u = 0; u < 4; u++) {
        const int t = t0 + i4 + u;
        kr4[u] = Kb + (size_t)(X.ring ? t % S.cap : t) * HD;
      }
      float32x4_t s0[4][REP], s1[4][REP];
      for (int u = 0; u < 4; u++)
        for (int g = 0; g < REP; g++) s0[u][g] = s1[u][g] = vdupq_n_f32(0);
      for (int k = 0; k < HD; k += 8) {
        float32x4_t ql[REP], qh[REP];
        for (int g = 0; g < REP; g++) ql[g] = vld1q_f32(q[g] + k), qh[g] = vld1q_f32(q[g] + k + 4);
        for (int u = 0; u < 4; u++) {
          float16x8_t kv = vld1q_f16((const float16_t*)kr4[u] + k);
          const float32x4_t lo = vcvt_f32_f16(vget_low_f16(kv)), hi = vcvt_high_f32_f16(kv);
          for (int g = 0; g < REP; g++) {
            s0[u][g] = vfmaq_f32(s0[u][g], lo, ql[g]);
            s1[u][g] = vfmaq_f32(s1[u][g], hi, qh[g]);
          }
        }
      }
      for (int g = 0; g < REP; g++) {
        const float32x4_t d4 = vpaddq_f32(vpaddq_f32(vaddq_f32(s0[0][g], s1[0][g]), vaddq_f32(s0[1][g], s1[1][g])),
                                          vpaddq_f32(vaddq_f32(s0[2][g], s1[2][g]), vaddq_f32(s0[3][g], s1[3][g])));
        float32x4_t m4 = vmulq_f32(d4, vsc);
        __asm__("" : "+w"(m4));
        const float32x4_t s4 = vaddq_f32(m4, vld1q_f32(bf[g] + i4));
        vst1q_f32(sc[g] + i4, s4);
        mx[g] = std::max(mx[g], vgetq_lane_f32(s4, 0));
        mx[g] = std::max(mx[g], vgetq_lane_f32(s4, 1));
        mx[g] = std::max(mx[g], vgetq_lane_f32(s4, 2));
        mx[g] = std::max(mx[g], vgetq_lane_f32(s4, 3));
      }
    }
    for (int i = i4; i < n; i++) {
      const int t = t0 + i;
      const int slot = X.ring ? t % S.cap : t;
      const _Float16* kr = Kb + (size_t)slot * HD;
      float32x4_t s0[REP], s1[REP];
      for (int g = 0; g < REP; g++) s0[g] = s1[g] = vdupq_n_f32(0);
      for (int k = 0; k < HD; k += 8) {
        float16x8_t kv = vld1q_f16((const float16_t*)kr + k);
        const float32x4_t lo = vcvt_f32_f16(vget_low_f16(kv)), hi = vcvt_high_f32_f16(kv);
        for (int g = 0; g < REP; g++) {
          s0[g] = vfmaq_f32(s0[g], lo, vld1q_f32(q[g] + k));
          s1[g] = vfmaq_f32(s1[g], hi, vld1q_f32(q[g] + k + 4));
        }
      }
      for (int g = 0; g < REP; g++) {
        const float dot = vaddvq_f32(vaddq_f32(s0[g], s1[g]));
        const float s = attn_score(dot, scale, bf[g][i]);
        sc[g][i] = s;
        mx[g] = std::max(mx[g], s);
      }
    }
    float iz[REP];
    for (int g = 0; g < REP; g++) {
      const int hh = kvh * REP + g;
      double z = std::exp((double)X.sink[ip][hh] - mx[g]);
      for (int i = 0; i < n; i++) z += (sc[g][i] = std::exp(sc[g][i] - mx[g]));
      iz[g] = (float)(1.0 / z);
      for (int i = 0; i < n; i++) sc[g][i] = sc[g][i] * iz[g];  // each key's weight once (the same product as per pass)
    }
    float accs[REP][256];
    for (int k0 = 0; k0 < HD; k0 += 32) {  // 32 dimensions of every head per pass over the keys
      float32x4_t a[REP][8];
      for (int g = 0; g < REP; g++)
        for (int u = 0; u < 8; u++) a[g][u] = vdupq_n_f32(0);
      for (int i = 0; i < n; i++) {
        const int t = t0 + i;
        const int slot = X.ring ? t % S.cap : t;
        const float16_t* vr = (const float16_t*)(Vb + (size_t)slot * HD + k0);
        float32x4_t vf[8];
        for (int u = 0; u < 4; u++) {
          float16x8_t vv = vld1q_f16(vr + 8 * u);
          vf[2 * u] = vcvt_f32_f16(vget_low_f16(vv));
          vf[2 * u + 1] = vcvt_high_f32_f16(vv);
        }
        for (int g = 0; g < REP; g++) {
          const float32x4_t pv = vld1q_dup_f32(sc[g] + i);
          for (int u = 0; u < 8; u++) a[g][u] = vfmaq_f32(a[g][u], vf[u], pv);
        }
      }
      for (int g = 0; g < REP; g++)
        for (int u = 0; u < 8; u++) vst1q_f32(accs[g] + k0 + 4 * u, a[g][u]);
    }
    for (int g = 0; g < REP; g++) {
      const int hh = kvh * REP + g;
      float* o = X.o[ip] + (size_t)hh * HD;
      const float* gt = X.g[ip] + (size_t)hh * HD;
      for (int k = 0; k < HD; k++) o[k] = accs[g][k] * sigm(gt[k]);
      if (!M.ref && HD % XB == 0)
        for (int j = 0; j < HD / XB; j++) xq_block(*X.xe[ip], (hh * HD) / XB + j, o + j * XB);
    }
  }
}
}  // namespace

void Inkling::layer_chunk(int v, int i0, int n, bool msgs) {
  const int L = c.visits[v].first, pos0 = S->pos + i0;
  const Lw& W = lw[L];
  const int H = c.H, QD = c.NH * c.HD, KVD = c.NKV * c.HD, RD = c.NH * c.DREL, NC = c.CE - c.CS;
  VState& St = S->vs[v];
  Pool& pool = *core->pool;
  struct P {  // = layer()'s scratch layout, per position
    float *hres, *nres, *ain, *gin, *qn, *att, *res, *mlp, *y, *y2, *nst, *xm, *q, *k, *vv, *r, *g, *sk, *dk, *dv, *kc, *vc, *o;
    const float* m4;
  };
  std::vector<P> P_(n);
  for (int i = 0; i < n; i++) {
    float* t = pp[i]->tmp.data();
    P& p = P_[i];
    p.hres = t; p.nres = t + H; p.ain = t + 2 * H; p.gin = t + 5 * H; p.qn = t + 8 * H; p.att = t + 11 * H; p.res = t + 12 * H;
    p.mlp = t + 13 * H; p.y = t + 14 * H; p.y2 = t + 15 * H; p.nst = t + 16 * H; p.xm = t + 17 * H; p.q = t + 18 * H;
    p.k = p.q + QD; p.vv = p.k + KVD; p.r = p.vv + KVD; p.g = p.r + RD; p.sk = p.g + QD; p.dk = p.sk + 64; p.dv = p.dk + 64;
    p.kc = p.dv + 64; p.vc = p.kc + KVD; p.o = p.vc + KVD;
    p.m4 = msgs ? pp[i]->m4.data() : nullptr;
  }
  // Every loop over positions below runs each position's own arithmetic (the single step's) on the pool; the
  // convolutions, whose state moves position by position, run channel blocks in parallel with the positions in order.
  auto conv_par = [&](int C, const std::function<void(int, int, int)>& f) {  // f(i, c0, c1): position i, channels [c0, c1)
    const int CB = 256;
    parallel_for(pool, (C + CB - 1) / CB, [&](int blk) {
      const int c0 = blk * CB, c1 = std::min(C, c0 + CB);
      for (int i = 0; i < n; i++) f(i, c0, c1);
    });
  };
  auto t0 = clk::now();
  stage_mark(ST_DEPTH);
  const int nq = W.global ? 3 : 1;
  const bool newblk = L % c.BLOCK == 0;
  const int nb0 = sp_nblk, nb1 = nb0 + (newblk ? 1 : 0);  // block states before / after this application
  parallel_for(pool, n, [&](int i) {
    P& p = P_[i];
    float* hh = sph(i0 + i);
    const float* bk = spblk(i0 + i);
    if (msgs) {  // this layer's 4 messages from the previous pass's sources (the single step's rows, computed here)
      const float* src[16];
      for (int b = 0; b < NMS; b++) src[b] = spmsrc(i0 + i) + (size_t)b * H;
      depth_attn(src, NMS, msgq + (size_t)((v - c.CS) % NC) * 4 * H, 4, H, (float*)p.m4);
    }
    std::memcpy(p.hres, hh, sizeof(float) * H);
    rmsnorm(hh, nullptr, H, 1e-6f, p.nres);
    if (L > 0) {
      const float* src[16];
      for (int b = 0; b < nb0; b++) src[b] = bk + (size_t)b * H;
      src[nb0] = p.nres;
      depth_attn(src, nb0 + 1, W.aq, nq, H, p.ain);
    } else {
      for (int b = 0; b < nq; b++) std::memcpy(p.ain + (size_t)b * H, hh, sizeof(float) * H);
    }
  });
  auto t1 = clk::now();
  T->depth += ms_between(t0, t1);
  const bool three = W.global || msgs;
  std::vector<const float*> qi(n), ki(n), vi(n);
  std::vector<const float*> qkv_in((size_t)3 * n);
  for (int i = 0; i < n; i++)
    for (int b = 0; b < 3; b++) qkv_in[(size_t)b * n + i] = P_[i].ain + (size_t)(W.global ? b : 0) * H;
  if (msgs) {
    auto tg = clk::now();
    stage_mark(ST_GATES);
    const Gate* G[3] = {&W.mg[0], &W.mg[1], &W.mg[2]};
    std::vector<const float*> mg((size_t)3 * n);
    std::vector<float*> out((size_t)3 * n);
    for (int b = 0; b < 3; b++)
      for (int i = 0; i < n; i++) {
        mg[(size_t)b * n + i] = P_[i].m4 + (size_t)b * H;
        out[(size_t)b * n + i] = P_[i].gin + (size_t)b * H;
      }
    gates_n(*this, G, 3, qkv_in.data(), mg.data(), out.data(), n);
    for (int b = 0; b < 3; b++)
      for (int i = 0; i < n; i++) qkv_in[(size_t)b * n + i] = P_[i].gin + (size_t)b * H;
    pf_gates_ms += ms_since(tg);
  }
  stage_mark(ST_NORMQ);
  for (int i = 0; i < n; i++) {
    P& p = P_[i];
    qi[i] = p.qn;
    ki[i] = three ? p.qn + H : p.qn;
    vi[i] = three ? p.qn + 2 * H : p.qn;
  }
  const bool packed = !ref && prec == Prec::I16 && n >= (rb ? 2 : 6);  // (a verify pass: from 2 positions)
  if (packed) {
    xqb_a.reserve(H, n);
    if (three) {
      xqb_b.reserve(H, n);
      xqb_c.reserve(H, n);
    }
  }
  parallel_for(pool, n, [&](int i) {
    P& p = P_[i];
    const float* in0 = qkv_in[i];
    const float* in1 = qkv_in[(size_t)n + i];
    const float* in2 = qkv_in[(size_t)2 * n + i];
    {  // as the single step: the three norms side by side
      const float* ins[3] = {in0, in1, in2};
      const float* nw[3] = {W.global ? W.lnq : W.ln_in, W.global ? W.lnk : W.ln_in, W.global ? W.lnv : W.ln_in};
      float* no[3] = {p.qn, p.qn + H, p.qn + 2 * H};
      rmsnorm_n(ins, nw, three ? 3 : 1, H, c.eps, no);
    }
    PP& m = *pp[i];
    if (!ref) {
      m.xa.prep(qi[i], H);
      if (three) {
        m.xb.prep(ki[i], H);
        m.xc.prep(vi[i], H);
      }
    }
    if (packed) {
      xqb_set_from(xqb_a, i, m.xa.q);
      if (three) {
        xqb_set_from(xqb_b, i, m.xb.q);
        xqb_set_from(xqb_c, i, m.xc.q);
      }
    }
  });
  if (packed) {
    xqb_pad(xqb_a);
    if (three) {
      xqb_pad(xqb_b);
      xqb_pad(xqb_c);
    }
  }
  auto t2 = clk::now();
  T->msg += ms_between(t1, t2);
  // projections, one fork-join for all positions
  std::vector<const XQ*> xqa(n), xqk(n), xqv(n);
  for (int i = 0; i < n; i++) {
    xqa[i] = &pp[i]->xa.q;
    xqk[i] = three ? &pp[i]->xb.q : &pp[i]->xa.q;
    xqv[i] = three ? &pp[i]->xc.q : &pp[i]->xa.q;
  }
  std::vector<float*> oq(n), og(n), ok(n), ov(n), orr(n), osk(n), odk(n), odv(n);
  for (int i = 0; i < n; i++) {
    oq[i] = P_[i].q;
    og[i] = P_[i].g;
    ok[i] = P_[i].k;
    ov[i] = P_[i].vv;
    orr[i] = P_[i].r;
    osk[i] = P_[i].sk;
    odk[i] = P_[i].dk;
    odv[i] = P_[i].dv;
  }
  const XQB* pa = packed ? &xqb_a : nullptr;
  const XQB* pk_ = packed ? (three ? &xqb_b : &xqb_a) : nullptr;
  const XQB* pv_ = packed ? (three ? &xqb_c : &xqb_a) : nullptr;
  JobN jb[8];
  int nj = 0;
  stage_mark(ST_ATTN_PROJ);
  jb[nj++] = {&W.q, n, xqa.data(), qi.data(), oq.data(), pa};
  jb[nj++] = {&W.g, n, xqa.data(), qi.data(), og.data(), pa};
  jb[nj++] = {&W.k, n, xqk.data(), ki.data(), ok.data(), pk_};
  jb[nj++] = {&W.v, n, xqv.data(), vi.data(), ov.data(), pv_};
  jb[nj++] = {&W.r, n, xqa.data(), qi.data(), orr.data(), pa};
  jb[nj++] = {&W.sink, n, nullptr, qi.data(), osk.data(), nullptr};
  if (W.global) {
    jb[nj++] = {&W.kwp, n, nullptr, ki.data(), odk.data(), nullptr};
    jb[nj++] = {&W.vwp, n, nullptr, vi.data(), odv.data(), nullptr};
  }
  GpuGemm* const G = gpu && po.gpu_frac > 0 && n >= po.gpu_min ? gpu.get() : nullptr;
  run_rows_n_split(pool, G, po.gpu_frac, jb, nj, 32, ref, &T->bytes_res, &gacct);
  auto t3 = clk::now();
  T->attn_proj += ms_between(t2, t3);
  pf_proj_in_ms += ms_between(t2, t3);
  // the KV preparation of every position (the convolutions in position order per channel block; clips and norms per
  // position and head), then the cache writes and the attention in groups that a sliding ring can hold (a global
  // layer's cache holds every position: one group)
  stage_mark(ST_KV_PREP);
  if (rb)
    parallel_for(pool, n, [&](int i) {
      std::memcpy(rb_in[(size_t)v * 4 + 0].data() + (size_t)(i0 + i) * KVD, P_[i].k, sizeof(float) * KVD);
      std::memcpy(rb_in[(size_t)v * 4 + 1].data() + (size_t)(i0 + i) * KVD, P_[i].vv, sizeof(float) * KVD);
    });
  conv_par(KVD, [&](int i, int c0, int c1) {
    P& p = P_[i];
    if (W.global) {
      dconv_rng(St.conv[0], W.kstat, p.dk, p.k, KVD, c.SC, c.HD, p.kc, c0, c1);
      dconv_rng(St.conv[1], W.vstat, p.dv, p.vv, KVD, c.SC, c.HD, p.vc, c0, c1);
    } else {
      sconv_rng(St.conv[0], W.kconv, p.k, KVD, c.SC, p.kc, c0, c1);
      sconv_rng(St.conv[1], W.vconv, p.vv, KVD, c.SC, p.vc, c0, c1);
    }
  });
  for (int i = 0; i < n; i++)
    if (!St.ring && pos0 + i >= St.cap) {
      fprintf(stderr, "context full (%d tokens)\n", St.cap);
      abort();
    }
  parallel_for(pool, n, [&](int i) {
    P& p = P_[i];
    const int ps = pos0 + i;
    for (int hh = 0; hh < c.NH; hh++) {
      l2clip(p.q + (size_t)hh * c.HD, W.qclip + (size_t)hh * c.HD, c.HD, c.qkclip);
      l2clip(p.r + (size_t)hh * c.DREL, W.rclip + (size_t)hh * c.DREL, c.DREL, c.relclip);
    }
    if (W.global) {
      const float tau = 1.f + c.logalpha * std::log(std::max(((float)ps + 1.f) / c.logfloor, 1.f));
      for (int x = 0; x < QD; x++) p.q[x] *= tau;
      for (int x = 0; x < RD; x++) p.r[x] *= tau;
    }
    for (int j = 0; j < c.NKV; j++) {
      float* kh = p.kc + (size_t)j * c.HD;
      float* vh = p.vc + (size_t)j * c.HD;
      l2clip(kh, nullptr, c.HD, c.qkclip);
      rmsnorm(vh, nullptr, c.HD, c.eps, vh);
    }
  });
  std::vector<const float*> aq(n), ar(n), ag(n), as(n);
  std::vector<float*> ao(n);
  std::vector<XQ*> axe(n);
  for (int i = 0; i < n; i++) {
    aq[i] = P_[i].q;
    ar[i] = P_[i].r;
    ag[i] = P_[i].g;
    as[i] = P_[i].sk;
    ao[i] = P_[i].o;
    axe[i] = &pp[i]->xe.q;
    pp[i]->xe.reset(QD);
  }
  const int rep = c.NH / c.NKV;
  const bool grouped = !St.Kf && rep == 3 && c.HD % 32 == 0 && c.HD <= 256;
  const int GRP = St.ring ? RING_SLACK + 1 : n;
  for (int g0 = 0; g0 < n; g0 += GRP) {
    const int g1 = std::min(n, g0 + GRP);
    stage_mark(ST_KV_PREP);
    for (int i = g0; i < g1; i++) {
      const P& p = P_[i];
      const int ps = pos0 + i, slot = St.ring ? ps % St.cap : ps;
      for (int j = 0; j < c.NKV; j++) {
        const float* kh = p.kc + (size_t)j * c.HD;
        const float* vh = p.vc + (size_t)j * c.HD;
        if (St.Kf) {
          std::memcpy(St.Kf + ((size_t)j * St.cap + slot) * c.HD, kh, sizeof(float) * c.HD);
          std::memcpy(St.Vf + ((size_t)j * St.cap + slot) * c.HD, vh, sizeof(float) * c.HD);
          continue;
        }
        _Float16* kd = St.K + ((size_t)j * St.cap + slot) * c.HD;
        _Float16* vd = St.V + ((size_t)j * St.cap + slot) * c.HD;
        for (int x = 0; x < c.HD; x++) {
          kd[x] = (_Float16)kh[x];
          vd[x] = (_Float16)vh[x];
        }
      }
    }
    AttnNCtx X;
    X.M = this;
    X.v = v;
    X.pos0 = pos0;
    X.i0 = g0;
    X.i1 = g1;
    X.ring = St.ring;
    X.rel = W.rel;
    X.q = aq.data();
    X.r = ar.data();
    X.g = ag.data();
    X.sink = as.data();
    X.o = ao.data();
    X.xe = axe.data();
    stage_mark(ST_ATTN_CORE);
    pool.run(grouped ? phase_attn_g<3> : phase_attn_n, &X);
  }
  stage_mark(ST_ATTN_OUT);
  if (!ref && c.HD % XB != 0) parallel_for(pool, n, [&](int i) { pp[i]->xe.prep(P_[i].o, QD); });
  auto t4 = clk::now();
  T->attn_core += ms_between(t3, t4);
  pf_core_ms[W.global ? 1 : 0] += ms_between(t3, t4);
  {
    std::vector<const XQ*> xo(n);
    std::vector<const float*> fo(n);
    std::vector<float*> oa(n);
    for (int i = 0; i < n; i++) {
      xo[i] = &pp[i]->xe.q;
      fo[i] = P_[i].o;
      oa[i] = P_[i].att;
    }
    const XQB* pe = nullptr;
    if (packed) {
      xqb_e.reserve(QD, n);
      parallel_for(pool, n, [&](int i) { xqb_set_from(xqb_e, i, pp[i]->xe.q); });
      xqb_pad(xqb_e);
      pe = &xqb_e;
    }
    JobN jo = {&W.o, n, xo.data(), fo.data(), oa.data(), pe};
    auto to = clk::now();
    run_rows_n_split(pool, gpu && po.gpu_frac > 0 && n >= po.gpu_min ? gpu.get() : nullptr, po.gpu_frac, &jo, 1, 32, ref, &T->bytes_res, &gacct);
    pf_proj_out_ms += ms_since(to);
  }
  stage_mark(ST_CONV);
  if (rb)
    parallel_for(pool, n, [&](int i) { std::memcpy(rb_in[(size_t)v * 4 + 2].data() + (size_t)(i0 + i) * H, P_[i].att, sizeof(float) * H); });
  conv_par(H, [&](int i, int c0, int c1) { sconv_rng(St.conv[2], W.aconv, P_[i].att, H, c.SC, P_[i].att, c0, c1); });
  auto t5 = clk::now();
  T->attn_out += ms_between(t4, t5);
  stage_mark(ST_DEPTH);
  parallel_for(pool, n, [&](int i) {  // the block count moves when the whole span has run this application (forward_chunk)
    P& p = P_[i];
    float* bk = spblk(i0 + i);
    if (newblk) {
      std::memcpy(bk + (size_t)nb0 * H, p.nres, sizeof(float) * H);
      std::memcpy(p.res, p.att, sizeof(float) * H);
    } else {
      for (int x = 0; x < H; x++) p.res[x] = p.hres[x] + p.att[x];
    }
    rmsnorm(p.res, nullptr, H, 1e-6f, p.nst);
    const float* src[16];
    for (int b = 0; b < nb1; b++) src[b] = bk + (size_t)b * H;
    src[nb1] = p.nst;
    depth_attn(src, nb1 + 1, W.mq, 1, H, p.mlp);
  });
  auto t6 = clk::now();
  T->depth += ms_between(t5, t6);
  if (msgs) {
    stage_mark(ST_GATES);
    const Gate* G[1] = {&W.mg[3]};
    std::vector<const float*> hs(n), mg(n);
    std::vector<float*> out(n);
    for (int i = 0; i < n; i++) {
      hs[i] = P_[i].mlp;
      mg[i] = P_[i].m4 + 3 * H;
      out[i] = P_[i].mlp;
    }
    gates_n(*this, G, 1, hs.data(), mg.data(), out.data(), n);
  }
  std::vector<const float*> xm(n);
  std::vector<float*> ym(n);
  stage_mark(ST_NORMQ);
  for (int i = 0; i < n; i++) {
    xm[i] = P_[i].xm;
    ym[i] = P_[i].y;
  }
  parallel_for(pool, n, [&](int i) { rmsnorm(P_[i].mlp, W.mlnorm, H, c.eps, P_[i].xm); });
  T->msg += ms_since(t6);
  core->moe_gpu = gpu && po.gpu_experts > 0 && n >= po.gpu_min ? gpu.get() : nullptr;  // the experts' share (prefill sub-chunks)
  core->moe_gpu_frac = po.gpu_experts;
  const double ex0 = T->experts, eb0 = T->bytes_exp;
  auto tm0 = clk::now();
  core->moe_chunk(v, n, xm.data(), ym.data(), views[L], nullptr, nullptr, i0);
  core->moe_gpu = nullptr;
  auto t8 = clk::now();
  pf_moe_ms[c.fmt[L] == 'M'] += ms_between(tm0, t8);
  pf_exp_ms[c.fmt[L] == 'M'] += T->experts - ex0;
  pf_exp_mb[c.fmt[L] == 'M'] += (T->bytes_exp - eb0) / MB;
  stage_mark(ST_CONV);
  if (rb)
    parallel_for(pool, n, [&](int i) { std::memcpy(rb_in[(size_t)v * 4 + 3].data() + (size_t)(i0 + i) * H, P_[i].y, sizeof(float) * H); });
  conv_par(H, [&](int i, int c0, int c1) { sconv_rng(St.conv[3], W.mconv, P_[i].y, H, c.SC, P_[i].y2, c0, c1); });
  parallel_for(pool, n, [&](int i) {
    const P& p = P_[i];
    float* hh = sph(i0 + i);
    for (int x = 0; x < H; x++) hh[x] = p.res[x] + p.y2[x];
  });
  T->misc += ms_since(t8);
}

// application v's slice of the fixed state (save_fixed's layout) at position pos, as save_fixed would write it there
void Inkling::sink_app(int v, int pos, const LM::StateSink& sink) {
  size_t off = 0;
  for (int u = 0; u < v; u++) {
    const VState& st = S->vs[u];
    for (auto& cv : st.conv) off += cv.size() * 4;
    if (st.ring) off += 2 * st.bytes;
  }
  const VState& st = S->vs[v];
  for (auto& cv : st.conv) {
    sink(pos, off, cv.data(), cv.size() * 4);
    off += cv.size() * 4;
  }
  if (!st.ring) return;
  const void* kv[2] = {st.K ? (const void*)st.K : (const void*)st.Kf, st.V ? (const void*)st.V : (const void*)st.Vf};
  for (int k = 0; k < 2; k++) {
    if (pos >= st.cap) {
      sink(pos, off, kv[k], st.bytes);
    } else {  // slots no position has reached yet: zeros (save_fixed)
      ring_tmp.assign((const uint8_t*)kv[k], (const uint8_t*)kv[k] + st.bytes);
      const size_t es = st.K ? 2 : 4, row = (size_t)c.HD * es;
      for (int j = 0; j < c.NKV; j++) std::memset(ring_tmp.data() + ((size_t)j * st.cap + pos) * row, 0, (size_t)(st.cap - pos) * row);
      sink(pos, off, ring_tmp.data(), st.bytes);
    }
    off += st.bytes;
  }
}

bool Inkling::forward_chunk(const int32_t* ids, int n, bool want_last, bool all_logits, bool rollback, int sub, const SpanRun* run) {
  const int H = c.H;
  if (sub <= 0 || sub > n) sub = n;
  // sub-chunk boundaries: every sub positions, and at each capture position (an application's state is delivered there)
  std::vector<int> bnd;
  for (int p = sub; p < n; p += sub) bnd.push_back(p);
  if (run && run->capture)
    for (int p : *run->capture)
      if (p > 0 && p < n) bnd.push_back(p);
  bnd.push_back(n);
  std::sort(bnd.begin(), bnd.end());
  bnd.erase(std::unique(bnd.begin(), bnd.end()), bnd.end());
  int wmax = 0;
  for (size_t b = 0, s0 = 0; b < bnd.size(); s0 = bnd[b++]) wmax = std::max(wmax, bnd[b] - (int)s0);
  chunk_alloc(wmax);
  span_alloc(n);
  const bool multi = bnd.size() > 1;
  auto captured = [&](int e) { return run && run->sink && *run->sink && run->capture && std::find(run->capture->begin(), run->capture->end(), e) != run->capture->end(); };
  const std::atomic<bool>* cancel = run ? run->cancel : nullptr;
  if (cancel && multi) {  // a cancel inside the span puts the fixed state back here
    span_snap.resize(fixed_bytes(*S));
    save_fixed(*S, span_snap.data());
  }
  rb = rollback;
  if (rb) {
    const int KVD = c.NKV * c.HD;
    rb_snap.resize((size_t)c.NV * 4);
    rb_in.resize((size_t)c.NV * 4);
    for (int v = 0; v < c.NV; v++)
      for (int cc = 0; cc < 4; cc++) {
        rb_snap[(size_t)v * 4 + cc] = S->vs[v].conv[cc];
        rb_in[(size_t)v * 4 + cc].resize((size_t)n * (cc < 2 ? KVD : H));
      }
    rb_n = n;
    rb_pos0 = S->pos;
  }
  auto t0 = clk::now();
  stage_mark(ST_EMBED);
  for (int i = 0; i < n; i++) {
    rows.row(ids[i], emb.data());
    rmsnorm(emb.data(), embed_norm, H, c.eps, sph(i));
  }
  sp_nblk = 0;
  T->embed += ms_since(t0);
  core->pos = S->pos;
  Cache::CapsOff caps_off(core->cache);  // prefill and verify spans name the next application's union: caps off for the pass
  if (rollback) {  // a verify: names inside the pass (the routers' `emb` = the normed embedding, as a single step gives it)
    std::vector<const float*> e(n);
    for (int i = 0; i < n; i++) e[i] = sph(i);
    core->chunk_start(n, ids, e.data());
  }
  core->span_mode = multi;
  Cache::ChunkClock span_clock(core->cache, core->tok_clock + n - 1);  // the whole span is one token to the policy
  const int names = po.names_for(n, multi);  // the next application's experts named while this one runs (prefill.h)
  MoePass moe_pass(*core, po);
  const bool tapping = dh != nullptr;
  if (tapping) ctaps.resize((size_t)n * 4 * H);
  auto ctap = [&](int a) {  // the hidden state entering applications 5, 23, 41, per position
    if (!tapping || !(a == 5 || a == 23 || a == 41)) return;
    const int t = a == 5 ? 0 : a == 23 ? 1 : 2;
    for (int i = 0; i < n; i++) std::memcpy(ctaps.data() + ((size_t)i * 4 + t) * H, sph(i), sizeof(float) * H);
  };
  int nblk_pre = 0;
  for (int v = 0; v < c.NV; v++) {
    const int L = c.visits[v].first, pass = c.visits[v].second;
    const bool loop_start = c.core(L) && L == c.CS;
    if (multi) core->cache.lend_hold();  // (a prompt's loan: what the GPU's first use of the slab faulted back in goes back again)
    if (loop_start && pass > 0) {  // the previous pass's message sources, then the loop starts again from its input
      auto tm = clk::now();
      stage_mark(ST_SPAN);
      parallel_for(*core->pool, n, [&](int i) {
        float* ms = spmsrc(i);
        const float* bk = spblk(i);
        int nb = 0;
        for (int b = nblk_pre + 1; b < sp_nblk; b++) std::memcpy(ms + (size_t)(nb++) * H, bk + (size_t)b * H, sizeof(float) * H);
        rmsnorm(sph(i), nullptr, H, 1e-6f, ms + (size_t)nb * H);  // the bank
        std::memcpy(sph(i), sp_hpre.data() + (size_t)i * H, sizeof(float) * H);
      });
      sp_nblk = nblk_pre;
      T->depth += ms_since(tm);
    }
    ctap(v);
    const bool msgs = c.core(L) && pass > 0;
    for (size_t b = 0, s0 = 0; b < bnd.size(); s0 = bnd[b++]) {
      if (b == 0 && names && v + 1 < c.NV) {  // named inside moe_chunk, behind this application's own reads
        core->span_next = v + 1;
        core->span_next_router = names == 2;
      }
      layer_chunk(v, (int)s0, bnd[b] - (int)s0, msgs);
      if (captured(bnd[b])) sink_app(v, S->pos + bnd[b], *run->sink);
    }
    core->span_unname(v);  // its union pins and its names
    if (L % c.BLOCK == 0) sp_nblk++;
    if (v + 1 < c.NV && c.visits[v + 1].first == c.CS && c.visits[v + 1].second == 0) {  // the loop's input
      nblk_pre = sp_nblk;
      stage_mark(ST_SPAN);
      parallel_for(*core->pool, n, [&](int i) { std::memcpy(sp_hpre.data() + (size_t)i * H, sph(i), sizeof(float) * H); });
    }
    if (multi && run) {
      if (run->progress && *run->progress) (*run->progress)(run->prog_base + (int)((long)n * (v + 1) / c.NV), run->prog_total, 0.0);
      if (cancel && cancel->load() && v + 1 < c.NV) {  // stop: the fixed state back at the span's start, nothing else moved
        load_fixed(*S, span_snap.data(), S->pos);
        core->span_unname(v + 1);
        core->span_mode = false;
        return false;
      }
    }
  }
  core->span_mode = false;
  stage_mark(ST_HEAD);
  if (tapping) {  // the post-final-norm hidden state of every position (the head reads the same values below)
    auto th = clk::now();
    PP& m = *pp[0];
    float* nf = m.tmp.data();
    float* fin = nf + H;
    for (int i = 0; i < n; i++) {
      rmsnorm(sph(i), nullptr, H, 1e-6f, nf);
      const float* src[16];
      for (int b = 0; b < sp_nblk; b++) src[b] = spblk(i) + (size_t)b * H;
      src[sp_nblk] = nf;
      depth_attn(src, sp_nblk + 1, final_q, 1, H, fin);
      rmsnorm(fin, fnorm, H, c.eps, ctaps.data() + ((size_t)i * 4 + 3) * H);
    }
    dh->add_taps(ctaps.data(), n, S->pos);
    T->head += ms_since(th);
  }
  if (want_last || all_logits) {  // the last position's logits (all_logits: every position's), as step() computes them
    auto th = clk::now();
    const int i0 = all_logits ? 0 : n - 1, nh = n - i0;
    chunk_alloc(nh);
    std::vector<float> hn((size_t)nh * H), fl((size_t)nh * c.V);
    std::vector<const XQ*> xh(nh);
    std::vector<const float*> fh(nh);
    std::vector<float*> oh(nh);
    for (int k = 0; k < nh; k++) {
      PP& m = *pp[k];
      float* nf = m.tmp.data();
      float* fin = nf + H;
      rmsnorm(sph(i0 + k), nullptr, H, 1e-6f, nf);
      const float* src[16];
      for (int i = 0; i < sp_nblk; i++) src[i] = spblk(i0 + k) + (size_t)i * H;
      src[sp_nblk] = nf;
      depth_attn(src, sp_nblk + 1, final_q, 1, H, fin);
      rmsnorm(fin, fnorm, H, c.eps, hn.data() + (size_t)k * H);
      if (!ref) m.xa.prep(hn.data() + (size_t)k * H, H);
      xh[k] = &m.xa.q;
      fh[k] = hn.data() + (size_t)k * H;
      oh[k] = fl.data() + (size_t)k * c.V;
    }
    const bool scr = all_logits && rb && fast_head && !ref && (ver_want == Want::GREEDY || (ver_want == Want::TOPK && head_topk > 0));
    std::vector<int> full_pos;  // positions whose logits come from the full head (all, or the screen's fallbacks)
    if (scr) {
      // the screened head with the softcap, as a single step under this Want: every position's capped logits of the rows
      // that can be in the top k, -inf for the rest (the planes read once for all positions); a position whose cap makes
      // a row left out tie the k-th kept one (ok false) takes the full head, as the single step does
      mlg.resize((size_t)nh * c.VU);
      std::vector<float*> ml(nh);
      for (int k = 0; k < nh; k++) ml[k] = mlg.data() + (size_t)k * c.VU;
      bool oks[RING_SLACK + 1];
      const double hb0 = sh.bytes;
      sh.run_n(*core->pool, xh.data(), nh, ver_want == Want::GREEDY ? 1 : head_topk, ml.data(), ver_best, oks, c.softcap);
      ver_head_bytes += sh.bytes - hb0;
      for (int k = 0; k < nh; k++)
        if (!oks[k]) full_pos.push_back(k), gh_fallback++;
    } else {
      for (int k = 0; k < nh; k++) full_pos.push_back(k);
      if (rb) std::fill(ver_best, ver_best + RING_SLACK + 1, -1);
    }
    if (!full_pos.empty()) {
      const int nf = (int)full_pos.size();
      std::vector<const XQ*> xf(nf);
      std::vector<const float*> ff(nf);
      std::vector<float*> of(nf);
      for (int j = 0; j < nf; j++) xf[j] = xh[full_pos[j]], ff[j] = fh[full_pos[j]], of[j] = oh[full_pos[j]];
      const XQB* ph = nullptr;
      if (!ref && prec == Prec::I16 && nf >= (rb ? 2 : 6)) {
        xqb_h.reserve(H, nf);
        for (int j = 0; j < nf; j++) xqb_set_from(xqb_h, j, pp[full_pos[j]]->xa.q);
        xqb_pad(xqb_h);
        ph = &xqb_h;
      }
      JobN jh = {&head, nf, xf.data(), ff.data(), of.data(), ph};
      const double rb0 = T->bytes_res;
      run_rows_n(*core->pool, &jh, 1, 64, ref, &T->bytes_res);
      if (rb) ver_head_bytes += T->bytes_res - rb0, ver_hb_full_ = T->bytes_res - rb0;
    }
    if (all_logits) vhn.assign(hn.begin(), hn.end());  // (a verify: kept for light draft heads)
    // the softcaps (a tanh per logit: about 6 M cycles per position on one thread) on the pool, 4,096 logits per job
    constexpr int SCB = 4096;
    const int nbk = (c.VU + SCB - 1) / SCB;
    if (all_logits) {
      mlg.resize((size_t)nh * c.VU);
      const int nf = (int)full_pos.size();
      parallel_for(*core->pool, nf * nbk, [&](int j) {
        const int k = full_pos[j / nbk], i0 = (j % nbk) * SCB, i1 = std::min(c.VU, i0 + SCB);
        for (int i = i0; i < i1; i++) mlg[(size_t)k * c.VU + i] = softcap(fl[(size_t)k * c.V + i], c.softcap);
      });
    }
    if (scr) {
      for (int k : full_pos) ver_best[k] = argmax(mlg.data() + (size_t)k * c.VU, c.VU);  // the fallback's answer, as in step()
      std::memcpy(lgts.data(), mlg.data() + (size_t)(nh - 1) * c.VU, sizeof(float) * c.VU);
    } else {
      parallel_for(*core->pool, nbk, [&](int j) {
        const int i0 = j * SCB, i1 = std::min(c.VU, i0 + SCB);
        for (int i = i0; i < i1; i++) lgts[i] = softcap(fl[(size_t)(nh - 1) * c.V + i], c.softcap);
      });
    }
    T->head += ms_since(th);
  }
  S->pos += n;
  core->chunk_end(n);
  n_chunks += (long)bnd.size();
  stage_mark(ST_OTHER);
  return true;
}

int Inkling::head_argmax(const float* hn, float* conf) {
  if (dh_x.cap < c.H) dh_x.init(c.H, prec);
  dh_x.prep(hn, c.H);
  dh_lg.resize(c.V);
  Job jh = {&head, &dh_x.q, hn, dh_lg.data()};
  run_rows(*core->pool, &jh, 1, 64, ref);
  int id = 0;
  for (int r = 1; r < c.VU; r++)
    if (dh_lg[r] > dh_lg[id]) id = r;  // softcap is increasing: the same argmax
  const float top = softcap(dh_lg[id], c.softcap);
  double zs = 0;
  for (int r = 0; r < c.VU; r++) zs += std::exp((double)(softcap(dh_lg[r], c.softcap) - top));
  if (conf) *conf = (float)(1.0 / zs);
  return id;
}

bool Inkling::verify(State& s0, const int32_t* ids, int n, Want want) {
  S = (IState*)&s0;
  if (n < 1 || n > RING_SLACK + 1 || (core->force && S->pos < core->force_n)) return false;
  ver_want = want;
  const double r0 = T->bytes_res;
  ver_hb_full_ = 0;
  forward_chunk(ids, n, false, true, true);
  ver_res_bytes += T->bytes_res - r0 - ver_hb_full_;  // (the full head's bytes are in bytes_res too)
  ver_passes += 1;
  ver_positions += n;
  return true;
}

void Inkling::commit(State& s0, int a) {
  S = (IState*)&s0;
  if (rb && fast_head) greedy_ = ver_best[std::max(1, std::min(a, rb_n)) - 1];  // -1 when the pass used the full head
  if (!rb || a >= rb_n) {
    if (rb) core->chunk_commit(rb_n);  // the hooks see the last position's MoE inputs (names only)
    rb = false;
    return;
  }
  const int KVD = c.NKV * c.HD, R = c.SC - 1;
  for (int v = 0; v < c.NV; v++)
    for (int cc = 0; cc < 4; cc++) {  // the state after position a - 1: the last SC - 1 rows of [snapshot, inputs 0 .. a-1]
      const int C = cc < 2 ? KVD : c.H;
      const std::vector<float>& snap = rb_snap[(size_t)v * 4 + cc];
      const float* in = rb_in[(size_t)v * 4 + cc].data();
      for (int r = 0; r < R; r++) {
        const int src = a - R + r;
        const float* row = src < 0 ? snap.data() + (size_t)(R + src) * C : in + (size_t)src * C;
        std::memcpy(S->vs[v].conv[cc].data() + (size_t)r * C, row, sizeof(float) * C);
      }
    }
  S->pos = rb_pos0 + a;
  core->tok_clock -= rb_n - a;
  rb = false;
  core->chunk_commit(a);
  if (dh) dh->truncate(S->pos);
}

bool Inkling::prefill(State& s0, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture, const LM::StateSink& sink,
                      PrefillStats* st, const std::atomic<bool>* cancel) {
  S = (IState*)&s0;
  // with the exact greedy head on, the next token is greedy(): the prompt's last logits are the full head's, so their
  // argmax (lowest index on a tie) is it (it was left at the previous step's value: -1 after a cold prefill)
  // fast mode (approx:drop): the prompt runs the batched pass losslessly, as the 35B's and Flash-Next's prefills do
  // (prefill_spans turns the leave-out rule off for the prompt, short prompts under drop_upto aside): a span reads each
  // layer's union once, where single fast steps read every position's misses (462 positions in 230 s on the iPhone 15 Plus)
  if (po.chunk <= 1 || (!taps_.empty() && !dh) || (core->force && S->pos < core->force_n)) {
    const bool r = Model::prefill(s0, ids, n, want_logits, capture, sink, st, cancel);  // single steps
    if (r && fast_head && want_logits) greedy_ = argmax(lgts.data(), c.VU);
    return r;
  }
  const double be0 = T->bytes_exp, br0 = T->bytes_res;
  const Times tt0 = *T;  // the stage times before the prompt (ms_by_stage below)
  const double moe0[2] = {pf_moe_ms[0], pf_moe_ms[1]}, exm0[2] = {pf_exp_ms[0], pf_exp_ms[1]}, exb0[2] = {pf_exp_mb[0], pf_exp_mb[1]};
  const double cor0[2] = {pf_core_ms[0], pf_core_ms[1]}, gat0 = pf_gates_ms, pin0 = pf_proj_in_ms, pout0 = pf_proj_out_ms;
  const long c0 = n_chunks;
  const std::vector<uint32_t> nr0 = st ? core->cache.n_reads : std::vector<uint32_t>();  // reads per record before
  if (gpu) gpu->set_kernel(po.gpu_kernel);
  const GpuSplitAcct ga0 = gacct;
  const double me_w0 = core->moe_gpu_wait_ms, me_s0 = core->moe_gpu_stage_ms, me_m0 = core->moe_gpu_macs, me_c0 = core->moe_cpu_macs;
  const uint64_t me_n0 = core->moe_gpu_sets, me_f0 = core->moe_gpu_failed;
  // the prompt's memory (prompt_plan, every family's rule): the sub-chunk's buffers, the span's arrays (residual, its
  // value at the loop's start, block states, message sources: 0.74 GB at 8K positions), the cancel snapshot
  span_blocks();
  PromptBytes pb;
  pb.sub = sub_bytes_per_pos();
  pb.span = span_bytes_per_pos();
  pb.snap = cancel ? (double)fixed_bytes(*S) : 0.0;
  pb.held = scratch_held();
  PrefillOpts oc = po;
  oc.chunk = std::min(po.chunk, CHUNK_MAX);
  const PromptPlan plan = prompt_plan(oc, n, pb);
  const int sub = plan.chunk, span = std::max(sub, plan.span);
  PrefillOpts o = po;
  o.span = span;
  if (plan.lend && gpu) gpu->touch();  // (the GPU's first use of its wrapped buffers before the loan)
  PromptLoan loan(*core, plan.lend);
  int spans = 0;
  const bool ok = prefill_spans(*core, n, capture, cancel, o, [&](int done, int m, bool last, const std::vector<int>& cap_rel) {
    if (m == 1) {
      step(s0, ids[done], last && want_logits ? Want::LAST : Want::NONE);
      n_chunks++;
      if (sink && !cap_rel.empty()) {
        std::vector<uint8_t> blob(fixed_bytes(s0));
        save_fixed(s0, blob.data());
        sink(s0.pos, 0, blob.data(), blob.size());
      }
      return true;
    }
    SpanRun r;
    r.capture = &cap_rel;
    r.sink = &sink;
    r.cancel = cancel;
    r.progress = st ? &st->progress : nullptr;
    r.prog_base = done;
    r.prog_total = n;
    return forward_chunk(ids + done, m, last && want_logits, false, false, sub, &r);
  }, st, &spans);
  if (st) {
    const int done = st->positions;
    st->chunks = (int)(n_chunks - c0);
    std::ostringstream o;
    // bytes per prompt position: expert records read from the store, expert records and resident matrices the kernels
    // went through (once per sub-chunk each)
    const double np_ = std::max(1, done);
    o << "{\"chunk\":" << sub << ",\"chunk_asked\":" << po.chunk << ",\"span\":" << span << ",\"span_asked\":" << po.span << ",\"lent_mb\":" << plan.lend / 1e6 << ",\"spans\":" << spans << ",\"hold_releases\":" << core->n_hold_release << ",\"union_batches\":" << core->n_union_batches
      << ",\"mb_read\":" << st->mb_read << ",\"expert_mb_touched\":" << (T->bytes_exp - be0) / MB
      << ",\"resident_mb_touched\":" << (T->bytes_res - br0) / MB << ",\"read_mb_per_pos\":" << st->mb_read / np_
      << ",\"expert_touched_mb_per_pos\":" << (T->bytes_exp - be0) / MB / np_
      << ",\"resident_touched_mb_per_pos\":" << (T->bytes_res - br0) / MB / np_ << ",\"wait_ms\":" << st->ms_wait;
    {  // records read 0 / 1 / 2 / 3 / 4+ times in this prompt, and per physical layer: records read, reads, MB read and
       // the MB of the records once each (the re-reads' size: reads - records)
      const Cache& Ca = core->cache;
      o << ",\"read_hist\":[" << st->read_hist[0] << "," << st->read_hist[1] << "," << st->read_hist[2] << "," << st->read_hist[3] << "," << st->read_hist[4]
        << "],\"layers\":[";
      double once = 0, all = 0;
      for (int L = 0; L < Ca.NL; L++) {
        long rec = 0, rd = 0;
        for (int e = 0; e < Ca.NE; e++) {
          const size_t k = (size_t)L * Ca.NE + e;
          const uint32_t d = k < nr0.size() && k < Ca.n_reads.size() ? Ca.n_reads[k] - nr0[k] : 0;
          rec += d > 0;
          rd += d;
        }
        const double rb = (double)Ca.P[Ca.layer_pool[L]].rec;
        once += rec * rb;
        all += rd * rb;
        o << (L ? "," : "") << "[" << rec << "," << rd << "]";
      }
      o << "],\"records_once_mb\":" << once / 1e6 << ",\"reads_mb\":" << all / 1e6;
    }
    {  // wall ms by stage over the prompt (the caller thread's time; a stage's GPU wait inside it), MB of experts traversed
      const Times& t = *T;
      o << ",\"ms_by_stage\":{\"embed\":" << t.embed - tt0.embed << ",\"depth_attn\":" << t.depth - tt0.depth << ",\"gates_norms_quant\":" << t.msg - tt0.msg
        << ",\"of_which_gates\":" << pf_gates_ms - gat0 << ",\"attn_proj\":" << t.attn_proj - tt0.attn_proj << ",\"attn_core\":" << t.attn_core - tt0.attn_core
        << ",\"of_core_sliding\":" << pf_core_ms[0] - cor0[0] << ",\"of_core_global\":" << pf_core_ms[1] - cor0[1]
        << ",\"attn_out\":" << t.attn_out - tt0.attn_out << ",\"of_out_proj\":" << pf_proj_out_ms - pout0 << ",\"moe_router\":" << t.router - tt0.router
        << ",\"experts\":" << t.experts - tt0.experts << ",\"wait\":" << t.wait - tt0.wait << ",\"cache\":" << t.cache - tt0.cache
        << ",\"names\":" << t.pf - tt0.pf << ",\"post_mlp\":" << t.misc - tt0.misc << ",\"head\":" << t.head - tt0.head
        << ",\"moe_nvfp4\":" << pf_moe_ms[0] - moe0[0] << ",\"moe_mxfp8\":" << pf_moe_ms[1] - moe0[1]
        << ",\"experts_nvfp4\":" << pf_exp_ms[0] - exm0[0] << ",\"experts_mxfp8\":" << pf_exp_ms[1] - exm0[1]
        << ",\"proj_in\":" << pf_proj_in_ms - pin0 << "},\"expert_mb_nvfp4\":" << pf_exp_mb[0] - exb0[0] << ",\"expert_mb_mxfp8\":" << pf_exp_mb[1] - exb0[1];
    }
    if (gpu && (po.gpu_frac > 0 || po.gpu_experts > 0))  // the GPU shares (the 35B's "gpu" account; experts: the NVFP4 layers)
      o << ",\"gpu\":{\"kernel\":\"" << (gpu->kernel() == 1 ? "matrix" : "integer") << "\",\"attn\":{\"frac\":" << po.gpu_frac
        << ",\"gmac_gpu\":" << (gacct.macs_gpu - ga0.macs_gpu) / 1e9 << ",\"gmac_cpu\":" << (gacct.macs_cpu - ga0.macs_cpu) / 1e9
        << ",\"wait_ms\":" << gacct.ms_wait - ga0.ms_wait << ",\"stage_ms\":" << gacct.ms_stage - ga0.ms_stage << ",\"sets\":" << gacct.sets - ga0.sets
        << ",\"failed\":" << gacct.failed - ga0.failed << "},\"experts\":{\"frac\":" << po.gpu_experts << ",\"gmac_gpu\":" << (core->moe_gpu_macs - me_m0) / 1e9
        << ",\"gmac_cpu\":" << (core->moe_cpu_macs - me_c0) / 1e9 << ",\"wait_ms\":" << core->moe_gpu_wait_ms - me_w0 << ",\"stage_ms\":" << core->moe_gpu_stage_ms - me_s0 << ",\"sets\":" << core->moe_gpu_sets - me_n0
        << ",\"failed\":" << core->moe_gpu_failed - me_f0 << "}}";
    o << "}";
    st->json = o.str();
  }
  if (gpu && po.release) gpu->trim();
  if (po.release) release_scratch();
  if (ok && fast_head && want_logits) greedy_ = argmax(lgts.data(), c.VU);
  return ok;
}

// Run-ahead (Core::run_ahead): from application v's approximate MoE output (its experts in memory only), application
// v + 1 up to its router on the approximate state (layer() in ra_mode), its top picks by the release rule (sigmoid score +
// bias). Nothing of the state changes: v + 1's convolution states, the block list and the hidden state are put back; its K
// and V at this position's slot are written again by the real pass before anything reads them; the scratch is its own.
// Not across a pass restart (v + 1 the first application of a looped pass: its input is rebuilt from the blocks there).
// Gives up when the awaited read lands (Core::ra_stopped, rows_stop). Approximate by design: it only decides reads.
int Inkling::run_ahead_step(int v, const float* ypart, int32_t* picks, int maxk) {
  const int v1 = v + 1, NC = c.CE - c.CS, H = c.H;
  if (v1 >= c.NV || !cur_res) return 0;
  if (v1 >= c.CS && v1 < c.CS + c.NP * NC && v1 > c.CS && (v1 - c.CS) % NC == 0) return 0;  // a looped pass restarts
  const Lw& W = lw[c.visits[v].first];
  const Times keep = *T;
  // h for v + 1: the residual + the post-MLP convolution of the approximate output (on a copy of its state)
  ra_conv = S->vs[v].conv[3];
  ra_h.resize(H);
  std::vector<float> y2(H);
  sconv(ra_conv, W.mconv, ypart, H, c.SC, y2.data());
  for (int i = 0; i < H; i++) ra_h[i] = cur_res[i] + y2[i];
  // what layer(v + 1) changes: h, the block list, its convolution states; its scratch swapped for our own
  std::vector<float> h_keep(h);
  const int nblk_keep = nblk;
  ra_blk.assign(blk.begin() + (size_t)nblk * H, blk.begin() + (size_t)std::min<size_t>(blk.size() / H, nblk + 1) * H);
  VState& St = S->vs[v1];
  for (int k = 0; k < 4; k++) ra_c[k] = St.conv[k];
  h = ra_h;
  if (ra_tmp.size() != tmp.size()) ra_tmp.resize(tmp.size());
  tmp.swap(ra_tmp);
  const float* m4 = nullptr;
  if (v1 >= c.CS + NC && v1 < c.CS + c.NP * NC) m4 = msg.data() + (size_t)((v1 - c.CS) % NC) * 4 * H;
  ra_mode = true;
  ra_xm = nullptr;
  layer(v1, m4);
  ra_mode = false;
  int np = 0;
  if (ra_xm && !core->ra_stopped()) {
    const MoeView& mv1 = views[c.visits[v1].first];
    if (ra_xq.cap < H) ra_xq.init(H, prec);
    ra_xq.prep(ra_xm, H);
    ra_lg.resize(mv1.router->rows);
    Job jr = {mv1.router, &ra_xq.q, ra_xm, ra_lg.data()};
    run_rows(*core->pool, &jr, 1, 16);
    if (!core->ra_stopped()) {
      const int NE = c.NE;
      std::vector<float> key(NE);
      std::vector<int> idx(NE);
      for (int e = 0; e < NE; e++) {
        const float yv = 1.f / (1.f + std::exp(std::fabs(ra_lg[e])));
        key[e] = (ra_lg[e] < 0 ? yv : 1.f - yv) + mv1.router_bias[e];
        idx[e] = e;
      }
      np = std::min(maxk, NE);
      std::partial_sort(idx.begin(), idx.begin() + np, idx.end(), [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a < b); });
      for (int i = 0; i < np; i++) picks[i] = idx[i];
      core->ra_lgv.assign(ra_lg.begin(), ra_lg.begin() + NE);
    }
  }
  tmp.swap(ra_tmp);
  h.swap(h_keep);
  nblk = nblk_keep;
  std::copy(ra_blk.begin(), ra_blk.end(), blk.begin() + (size_t)nblk * H);
  for (int k = 0; k < 4; k++) St.conv[k].swap(ra_c[k]);
  *T = keep;
  return np;
}

bool Inkling::set_option(const std::string& name, const std::string& value, std::string& err) {
  if (name == "run_ahead") {  // >0: run ahead one application while a step waits on reads, naming this many picks
    const int k = std::stoi(value);
    if (k <= 0) return (core->run_ahead = nullptr), true;
    core->ra_k = k;
    core->run_ahead = [this](int step, const float* yp, int32_t* pk, int maxk) { return run_ahead_step(step, yp, pk, maxk); };
    return true;
  }
  if (name == "fast_head") {  // 1: the screened head (greedy, and top-k with head_topk)
    if (std::stoi(value) <= 0 || fast_head) return true;
    if (prec != Prec::I16 || ref) return (err = "fast_head: needs I16 activations"), false;
    if (res_mem == Mem::FILE) return (err = "fast_head: the resident weights must be loaded (not mapped) to split the head"), false;
    if (!sh.build(*core->pool, head, c.V, c.VU, err)) return false;
    fast_head = true;
    return true;
  }
  if (name == "head_topk") return (head_topk = std::min(std::stoi(value), ScreenHead::KMAX)), true;  // Want::TOPK's k
  if (name == "prefill_batched_always" && std::stoi(value) == 0) return true;      // off
  if (name == "prefill_batched_always")  // refused, not ignored
    return (err = name + ": not implemented for inkling2_loop's prefill"), false;
  bool handled = false;  // the shared prefill_* options (dmeng/prefill.h)
  const bool ok = prefill_option(name, value, po, gpu, R.base, R.size, core, handled, err);
  if (handled) return ok;
  if (name == "draft_ctx") return (dh_ctx = std::max(16, std::stoi(value))), true;
  if (name == "draft_exact") return (dh_exact = std::stoi(value) != 0), true;
  if (name == "draft_q8") return (dh_q8 = std::stoi(value) != 0), true;  // the head in AFF8 (default) or bf16 as released  // the port check's output layer (ink_draft.h)
  if (name == "draft_head") {
    if (!std::stoi(value)) return dh.reset(), true;
    return load_draft_head(err);
  }
  if (name == "taps") {  // keep the draft head's features of every step
    if (std::stoi(value)) taps_.assign((size_t)4 * c.H, 0.f);
    else taps_.clear();
    return true;
  }
  if (!core) return (err = "unknown option " + name + " (or bind the model first)"), false;
  return core->set_option(name, value, err);  // the engine's own options (Core::set_option)
}

bool Inkling::load_draft_head(std::string& err) {
  if (!core) return (err = "draft_head: bind the model first"), false;
  InkDraftOpts o;
  o.mtp = dir + "/mtp.safetensors";
  o.resident = dir + "/resident.safetensors";
  const std::string cache = dir + "/.dmeng";
  mkdir(cache.c_str(), 0755);
  o.fold = cache + "/ink_mtp_fold_bf16.bin";
  o.head = &head;
  o.H = c.H;
  o.VU = c.VU;
  o.ctx = dh_ctx;
  o.softcap = c.softcap;
  o.exact = dh_exact;
  o.q8 = dh_q8 && !dh_exact;
  o.prec = prec;
  o.mem = res_mem;
  std::unique_ptr<InkDraft> d(new InkDraft());
  if (!d->load(o, core->pool.get(), err)) return false;
  dh = std::move(d);
  if (taps_.empty()) taps_.assign((size_t)4 * c.H, 0.f);
  dh_emb.resize(c.H);
  dh_embn.resize(c.H);
  return true;
}

int Inkling::draft_head(int32_t tok, int max, float thr, int32_t* out, float* prob, float* logits1) {
  if (!dh) return 0;
  rows.row(tok, dh_emb.data());
  rmsnorm(dh_emb.data(), embed_norm, c.H, c.eps, dh_embn.data());
  return dh->draft(tok, dh_embn.data(), max, thr, out, prob, logits1);
}

}  // namespace

std::unique_ptr<Model> make_inkling2_loop() { return std::unique_ptr<Model>(new Inkling()); }

}  // namespace dm
