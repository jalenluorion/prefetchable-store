// Qwen3.8-Flash-Next (qwen4_exp): the model's private declarations, shared by the decode path (qwen4_exp.cpp) and the
// batched prefill (prefill.cpp). Every per-position piece of arithmetic is one function both paths call, and the matrix
// products go through kernels that give the same bits for one position or many (weights.h mv / mv_multi, the GPU split),
// so a prefill leaves the state a run of single steps leaves, bit for bit.
//
// The architecture (reference/flash/qwen4_exp/language.py; transformers' modeling_qwen4_exp):
//   h = 4 copies of the token's embedding row: the hyper-connection streams [HC, H]
//   per layer L:  (the PLE layer only) h += PLE(h, the token's n-gram rows)
//                 x, inj = GatedResidual_attn(h);  h_s += inj_s * mixer(x)     mixer: Gated DeltaNet or QSA attention
//                 x, inj = GatedResidual_mlp(h);   h_s += inj_s * MoE(x)       the MoE runs in the core (Core::moe_step)
//   logits = lm_head(GatedResidual_final(h))
// GatedResidual: normed = per-stream RMSNorm (weight + 1); mix = sigmoid(up(silu(down(normed) / HC))); x = mean over the
// streams of mix * normed; inj = 2 sigmoid(inject(normed) / HC).
// QSA attention: gated attention (q / k RMSNorm, partial RoPE, sigmoid output gate) over every visible position while
// they are within the indexer budget, else over the indexer's top budget / ratio blocks of `ratio` keys plus the incomplete
// tail block (block scores: relu(q_h . pooled block key) summed over the indexer heads / sqrt(head dim); the pooled key is
// the RMSNormed mean of the block's raw indexer keys, rotated at the block's first position; ties: the earlier block).
// PLE: the n-gram rows (16 = 2 n-gram sizes x 8 heads, hashed from the token and its two predecessors) give a value and
// one key per stream; a signed-square-root gate per stream; plus a dilated depthwise convolution of the gated values.
//
// Precision: the streams, norms, recurrent states and every intermediate in fp32; weights as stored (8-bit resident,
// 4-bit experts, bf16 router and small gates, int8 n-gram rows exactly widened); matvec activations I16 by default. K and
// V and the raw indexer keys in fp16 (fp32 in the reference path, --prec f32).
#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "../common/gdn.h"
#include "dmeng/model.h"
#include "dmeng/rowstore.h"
#include "dmeng/screen_head.h"
#include "../../core/gpu_split.h"

namespace dm {
struct GpuGemm;
namespace qwen4 {

struct FCfg {
  int H = 0, HC = 4, LOW = 0, NL = 0, NE = 0, K = 0, I = 0, SI = 0, norm_topk = 1;
  int NKH = 0, NVH = 0, DK = 0, DV = 0, CK = 0;     // Gated DeltaNet
  int NQ = 0, NKV = 0, HD = 0, ROT = 0;             // QSA attention (ROT: rotary dims, half-split)
  int IXH = 0, IXKV = 1, IXD = 0, budget = 0, ratio = 0;
  int ple = -1, ple_dim = 0, ple_ck = 4, ng_size = 3, hpn = 0, ng_div = 128, eos = 0;
  int64_t ng_base = 0, ng_seed = 0, ng_rows = 0, ng_rpp = 0;
  int ng_parts = 0, ng_dim = 0, ng_bpr = 0, V = 0;
  float eps = 1e-6f;
  double theta = 1e7;
  bool sigmoid_gate = true;
  std::string types, ng_files, emb_file, layer_file;  // types: 'L' Gated DeltaNet, 'Q' QSA attention
  size_t rec = 0, off[3] = {0}, soff[3] = {0}, boff[3] = {0};
  long emb_off = 0;
  int N() const { return HC * H; }
  int conv_dim() const { return 2 * NKH * DK + NVH * DV; }
  int heads() const { return (ng_size - 1) * hpn; }  // n-gram rows per token
  int pstate() const { return (ple_ck - 1) * ng_size; }  // PLE convolution state (positions)
  bool from(const ResFile& R, std::string& err);
};

struct HCw {  // a GatedResidual: grouped norm weight (+1) [HC H], low-rank mix [LOW, HC H] and [HC H, LOW], injection [HC, HC H]
  const float* norm = nullptr;
  Weight down, up, inj;  // inj.kind NONE: the final mixer (no injection)
};

struct Lw {
  bool lin = true, ple = false;
  HCw ahc, mhc;
  Weight qkv, z, wa, wb, o;  // Gated DeltaNet (o: also the QSA output projection)
  gdn::Weights g;
  Weight q, k, v, ix;        // QSA
  const float *qn = nullptr, *kn = nullptr, *ixqn = nullptr, *ixkn = nullptr;
  Weight router, sg, su, sd, shgate;
  Weight pk, pv;             // PLE
  const float *pnk = nullptr, *pnq = nullptr, *pnc = nullptr, *pconv = nullptr;
};

struct LState {
  std::vector<float> conv, S;          // Gated DeltaNet: [CK - 1, C] oldest first, [NVH, DV, DK]
  _Float16 *K = nullptr, *V = nullptr; // QSA: [NKV, ctx, HD]
  float *Kf = nullptr, *Vf = nullptr;  // ... fp32 (reference path)
  _Float16* ixk = nullptr;             // raw indexer keys [ctx, IXD]
  float* ixkf = nullptr;               // ... fp32 (reference path)
  float* ixp = nullptr;                // pooled block keys [ctx / ratio, IXD], normed and rotated (fp32)
  size_t kv_bytes = 0, ix_bytes = 0, ixp_bytes = 0;
};

struct FState : State {
  std::vector<LState> ls;
  std::vector<float> pconv;  // the PLE convolution state [pstate, HC H], oldest first
  int32_t ng_ctx[2] = {0, 0};  // id(t - 2), id(t - 1): the n-gram history (EOS before the first token)
  int ctx = 0;
  ~FState() override {
    for (auto& s : ls) {
      free_region((uint8_t*)s.K, s.kv_bytes);
      free_region((uint8_t*)s.V, s.kv_bytes);
      free_region((uint8_t*)s.Kf, s.kv_bytes);
      free_region((uint8_t*)s.Vf, s.kv_bytes);
      free_region((uint8_t*)s.ixk, s.ix_bytes);
      free_region((uint8_t*)s.ixkf, s.ix_bytes);
      free_region((uint8_t*)s.ixp, s.ixp_bytes);
    }
  }
};

// the n-gram hash (reference/flash/qwen4_exp/language.py NGramEmbedding, transformers' Qwen4ExpTextNGramEmbedding)
struct NGramHash {
  int ngram_size = 3, hpn = 8, heads = 16, eos = 0;
  std::vector<uint64_t> mult;        // [ngram_size]
  std::vector<int64_t> sizes, offs;  // [heads]
  int64_t padded = 0;                // rows of the table
  void build(int vocab, int ngram_size, int hpn, int64_t base, int divisor, int ple_index, uint64_t seed, int eos);
  // rows [heads] of token `cur` after ctx[0] = id(t - 2), ctx[1] = id(t - 1)
  void rows(const int32_t* ctx, int32_t cur, int64_t* out) const;
};

// prefill options (set_option prefill_*)
struct PrefillOpts {
  int chunk = 256;        // positions per sub-chunk of a layer (bounds the scratch); 1 = single steps
  int span = 2048;        // positions per layer-major pass (each layer's experts read once per span)
  int whole_from = 256;   // spans of at least this many positions name every expert of the next layer while a layer runs
  bool names = true;      // prefill_stream 0: no early names in a span
  double gpu_frac = 0, gpu_experts = 0;  // the GPU's share of the projections' rows / the experts' (prefill_gpu_*)
  int gpu_min = 64;       // sub-chunks of at least this many positions use the GPU
  int gpu_kernel = 1;
  bool release = false;   // give the span's scratch and the GPU split's buffers back after each prefill
  double scratch_mb = 320;  // the prefill's scratch bound: sub-chunk buffers and the span's streams, half each (prefill.cpp)
};

struct Flash : Model {
  FCfg c;
  ResFile R;
  ModelInfo mi;
  Core* core = nullptr;
  Prec prec = Prec::I16;
  bool ref = false;
  Mem res_mem = Mem::MLOCK;
  std::string dir;
  std::vector<Lw> lw;
  HCw fin;
  Weight head;
  std::vector<MoeView> views;
  gdn::Dims gd;
  EmbedRows rows;
  NGramHash ng;
  RowStore ngs;
  std::shared_ptr<RowStore::Batch> ng_cur;  // the current step's rows (issued at its start)
  std::vector<float> rope_inv;              // [ROT / 2] inverse frequencies
  FState* S = nullptr;
  Times* T = nullptr;
  // decode scratch
  XBuf xin, xlo, xatt, xe;
  std::vector<float> hyp, nrm, mix, blk, lo, up, emb, lgts, buf, att, ple_e, ple_out, x;
  std::vector<int> sel_;  // the keys the current QSA query attends to
  int greedy_ = -1;
  bool fast_head = false;
  int ab_on = 0, ab_arm = 0;  // DMENG_AB: alternate an arm token by token (measurement)
  double ab_ms[2] = {0, 0}, ab_msg[2] = {0, 0}, ab_n[2] = {0, 0};  // the hyper-connection mixer of a decode step in one fork-join (hc_step)
  int head_topk = 0;
  ScreenHead sh;
  // run-ahead (Core::run_ahead)
  std::vector<float> ra_hyp, ra_mix, ra_blk, ra_lg, ra_inj2;
  XBuf ra_xq;
  const float* cur_inj2 = nullptr;  // the MLP injection weights of the layer whose MoE runs now
  int run_ahead_step(int L, const float* ypart, int32_t* picks, int maxk);
  // prefill
  PrefillOpts po;
  std::unique_ptr<GpuGemm> gpu;
  bool slab_wrapped = false;
  GpuSplitAcct gacct;
  long n_chunks = 0;
  std::vector<size_t> fixed_off;  // [NL] offset of a Gated DeltaNet layer's states in the fixed blob
  size_t fixed_ple = 0, fixed_ng = 0;
  struct Span;                    // prefill.cpp
  std::shared_ptr<Span> sp;
  // multi-position verify of drafted tokens (prefill.cpp): the batched pass over n <= TMAX positions, every position's
  // logits kept, the Gated DeltaNet's states written to second buffers so commit(a) keeps a positions without a copy
  static constexpr int TMAX = 8;
  struct Ver;
  std::shared_ptr<Ver> V;
  int ver_best[TMAX] = {-1, -1, -1, -1, -1, -1, -1, -1};  // the screened head's argmax per position of the last verify

  Flash();
  ~Flash() override;
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
  bool prefill(State& s, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture, const LM::StateSink& sink,
               PrefillStats* st, const std::atomic<bool>* cancel) override;
  bool verify(State& s, const int32_t* ids, int n, Want want) override;
  void commit(State& s, int a) override;
  const float* logits_at(int i) const override;
  const std::vector<float>& logits() const override { return lgts; }
  int greedy() const override { return fast_head ? greedy_ : -1; }
  const float* router_emb() const override { return emb.data(); }
  bool set_option(const std::string& name, const std::string& value, std::string& err) override;
  std::string stats_json(int ntok) const override;
  void reset_stats() override;
  ModelMem mem(const State& s) const override;
  const float* final_hidden(int i) const override { return nullptr; }
  bool embed_row(int32_t tok, float* out) const override;

  // ---- per-position pieces, shared by decode and prefill
  // the n-gram rows of token cur after the context: ids into out[heads]
  void ng_rows(const int32_t* ctx, int32_t cur, int64_t* out) const { ng.rows(ctx, cur, out); }
  // a row's bytes -> its values (int8 code x fp16 scale, exact in fp32)
  void ng_row(const uint8_t* rec, float* out) const;
  // GatedResidual, after its products: normed (per-stream norms) -> mixed [H] and inj [HC] from down / up / inject rows
  void hc_norm(const HCw& w, const float* hyper, float* normed) const;
  void hc_lo(float* lo) const;  // silu(lo / HC), in place
  void hc_mix(const float* normed, const float* up, const float* inj_raw, float* mixed, float* inj) const;
  // the PLE after its products: key [HC H] (raw), value [H], hyper -> gated and normed [HC H]; then per channel the
  // convolution over [state, normed] (silu), the state moving by one (commit) or not
  void ple_gate(const Lw& w, const float* key, const float* value, const float* hyper, float* gated, float* normed) const;
  float ple_conv(const Lw& w, float* st, const float* normed, int ch, bool commit) const;
  // QSA, one position after its projections: the indexer query (normed, rotated) and the raw key; K and V into the cache.
  // qp: the q projection [NQ, 2 HD] (normed + rotated into q [NQ, HD], its second halves copied into gate [NQ, HD]),
  // kp, vp [NKV HD], ixq [(IXH + 1) IXD] (the indexer queries normed + rotated in place, then the raw key)
  void qsa_kv(int L, int pos, float* qp, const float* kp, const float* vp, float* ixq, float* q, float* gate);
  // the pooled key of block b (once its last position's raw key is in the cache)
  void pool_key(int L, int b) const;
  // the selected keys of position pos (ascending); par: the block scores on the pool (the decode path)
  void qsa_select(int L, int pos, const float* ixq, std::vector<int>& sel, bool par) const;
  // the attention of one query head over sel; o: [HD] (times the gate)
  void qsa_head(int L, int h, const float* q, const float* gate, const std::vector<int>& sel, float* o) const;
  void rope(float* x, int pos) const;
  // the per-layer decode blocks
  void hc_step(const HCw& w, const float* hyper, float* mixed, float* inj);
  void gdn_step(int L, const float* x, float* y);
  void qsa_step(int L, const float* x, float* y);
  void ple_step(int L, const float* hyper, float* out, bool commit);
  void head_step(const float* x, Want want);
  // the batched prefill (prefill.cpp)
  // ver (a verify pass): every position's logits into it, the Gated DeltaNet states into its second buffers
  bool span_run(const int32_t* ids, int n, bool want_last, const std::vector<int>* capture, const LM::StateSink* sink,
                const std::atomic<bool>* cancel, const PrefillProgress* progress, int prog_base, int prog_total, Ver* ver = nullptr);
  std::string verify_json() const;
  bool set_gpu_frac(double f, bool experts, std::string& err);
  size_t span_bytes() const;
  double sub_bytes_per_pos() const;
};

}  // namespace qwen4
}  // namespace dm
