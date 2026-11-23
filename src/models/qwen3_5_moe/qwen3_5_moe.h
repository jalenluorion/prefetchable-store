// Qwen3.6-35B-A3B (qwen3_5_moe): the model's private declarations, shared by the decode path (qwen3_5_moe.cpp) and the
// batched prefill (prefill.cpp). The per-position arithmetic both rely on is in the inline functions at the end, so the
// two paths compute the same bits by construction.
#pragma once
#include <atomic>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "dmeng/model.h"
#include "dmeng/greedy_head.h"

namespace dm {
struct GpuGemm;  // src/core/gpu_gemm.h
namespace qwen35 {

struct PrefillOpts {
  int chunk = 512;   // positions per sub-chunk (bounds the scratch memory); 1 = the decode path, one position at a time
  int span = 8192;   // positions per layer-major pass: every layer runs over the whole span, so its experts are read once
  // how a layer's experts reach memory: 0 on demand (read once the layer's router has run, hits computed meanwhile);
  // 1 whole next layer streamed (at layer L, every expert of L+1 not held is named); 2 named early: layer L+1's released
  // router run on layer L's MoE input names L+1's experts at layer L; 3 automatic: spans of at least whole_from
  // positions stream whole layers, shorter ones name early
  int stream = 3;
  int whole_from = 1000;
  bool batched_always = false;  // chunk 1 through the batched code (one position per pass): a debugging reference
  FILE* record = nullptr;       // if set: every position's picks, int16 [pos][NL][K], appended per span
  // share of each 8-bit attention projection's rows computed on the GPU (src/core/gpu_gemm.h: the same bits) beside the
  // CPU, for sub-chunks of at least gpu_min positions; 0 = CPU only. Needs a build with DM_METAL.
  double gpu_frac = 0;
  int gpu_min = 256;
  // free the prefill's scratch (activations, expert inputs, packed activations: ~0.5 GB at 512-position sub-chunks) at
  // the end of every prefill; the next one allocates it again. Placement of memory only: no computed bit changes
  bool release = false;
};
struct Prefill;

struct QCfg {
  int H, NL, NE, K, I, SI, NKH, NVH, DK, DV, CK, NQ, NKV, HD, ROT, V, norm_topk;
  float eps;
  double theta;
  std::string types;  // 'L' Gated DeltaNet, 'F' full attention
  size_t rec;
  size_t off[3], soff[3], boff[3];  // gate, up, down inside a record
  long emb_off;
  std::string emb_file, layer_file;
  int conv_dim() const { return 2 * NKH * DK + NVH * DV; }
  void from(const ResFile& R) {
    H = R.geti("hidden");
    NL = R.geti("layers");
    NE = R.geti("experts");
    K = R.geti("topk");
    I = R.geti("moe_inter");
    SI = R.geti("shared_inter");
    norm_topk = R.geti("norm_topk");
    NKH = R.geti("lin_nk");
    NVH = R.geti("lin_nv");
    DK = R.geti("lin_dk");
    DV = R.geti("lin_dv");
    CK = R.geti("conv_k");
    NQ = R.geti("n_heads");
    NKV = R.geti("n_kv");
    HD = R.geti("head_dim");
    ROT = R.geti("rot_dim");
    V = R.geti("vocab");
    eps = (float)R.getf("eps");
    theta = R.getf("rope_theta");
    types = R.gets("layer_types");
    rec = R.geti("rec_bytes");
    const char* kinds[3] = {"gate_proj", "up_proj", "down_proj"};
    for (int i = 0; i < 3; i++) {
      off[i] = R.geti(std::string("off_") + kinds[i]);
      soff[i] = R.geti(std::string("soff_") + kinds[i]);
      boff[i] = R.geti(std::string("boff_") + kinds[i]);
    }
    emb_off = R.geti("emb_off");
    emb_file = R.gets("emb_file");
    layer_file = R.gets("store_layer_file");
  }
};

struct Lw {
  bool full = false;
  const float *ln_in = nullptr, *ln_post = nullptr;
  Weight qkv, z, o, q, k, v, wa, wb;  // wa / wb: bf16 [NVH, H]
  const float *conv = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr, *qn = nullptr, *kn = nullptr;
  Weight router, sg, su, sd, shgate;
};

struct LState {
  std::vector<float> conv;  // [CK - 1, C]: oldest first
  std::vector<float> S;     // [NVH, DV, DK]
  _Float16 *K = nullptr, *V = nullptr;  // [NKV, ctx, HD]
  size_t kv_bytes = 0;
};

struct QState : State {
  std::vector<LState> ls;
  int ctx = 0;
  ~QState() override {
    for (auto& s : ls) {
      free_region((uint8_t*)s.K, s.kv_bytes);
      free_region((uint8_t*)s.V, s.kv_bytes);
    }
  }
};

struct Qwen35 : Model {
  QCfg c;
  ResFile R;
  ModelInfo mi;
  Core* core = nullptr;
  Prec prec = Prec::I16;
  std::vector<Lw> lw;
  std::vector<MoeView> views;
  Weight head;
  const float* fnorm = nullptr;
  EmbedRows rows;
  QState* S = nullptr;  // the state of the step running now
  XBuf xin, xatt;
  std::vector<float> x, h, hm, y, buf, o, lgts, emb;
  Times* T = nullptr;
  // exact greedy head: a 4-bit copy of the 8-bit head (the top nibble of each code) gives every row's logit within a
  // proven bound; only the rows that can still hold the maximum are computed with the exact 8-bit head, so the argmax
  // (lowest index on ties) is the full head's, bit for bit
  bool fast_head = false, fast_head_check = false;
  double hd_mismatch = 0;
  GreedyHead gh;  // the exact greedy head (core)
  int greedy_ = -1;

  Qwen35();  // both defined in prefill.cpp, where Prefill is complete
  ~Qwen35() override;
  const ModelInfo& info() const override { return mi; }
  bool load(const LoadOpts& o, std::string& err) override;
  bool bind(Core& core, std::string& err) override;
  std::unique_ptr<State> new_state(int ctx_max, std::string& err) override;
  void reset(State& s) override;
  bool seek(State& s, int pos) override;
  size_t fixed_bytes(const State& s) const override;
  size_t pos_bytes() const override;
  bool save_fixed(const State& s, void* dst) const override;
  bool load_fixed(State& s, const void* src, int pos) override;
  bool save_pos(const State& s, int p0, int p1, void* dst) const override;
  bool load_pos(State& s, int p0, int p1, const void* src) override;
  void step(State& s, int32_t id, Want want) override;
  const std::vector<float>& logits() const override { return lgts; }
  int greedy() const override { return fast_head ? greedy_ : -1; }
  const float* router_emb() const override { return emb.data(); }
  bool set_option(const std::string& name, const std::string& value, std::string& err) override;
  std::string stats_json(int ntok) const override;
  void reset_stats() override {
    gh.reset_stats();
    hd_mismatch = 0;
  }
  ModelMem mem(const State& s) const override;  // prefill.cpp (the batched-prefill scratch sizes)
  // batched prefill (prefill.cpp): layer-major over a span, bit-identical to single steps
  bool prefill(State& s, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture, const LM::StateSink& sink,
               PrefillStats* st, const std::atomic<bool>* cancel) override;
  PrefillOpts po;
  std::unique_ptr<Prefill> P;
  std::unique_ptr<GpuGemm> gpu;  // the Metal half of the prefill's 8-bit products (po.gpu_frac > 0), kept across prefills
  bool set_gpu_frac(double f, std::string& err);  // prefill.cpp
  std::vector<size_t> fixed_off;  // [NL] byte offset of a DeltaNet layer's conv state in the fixed blob
  void attn_linear(int L, const float* hin, float* rout);
  void attn_full(int L, const float* hin, float* rout);
  void attn_full_core(int L, int pos, const float* qg, const float* kp, const float* vp, float* ob, XBuf& xo);
  void attn_full_kv(int L, int pos, const float* kp, const float* vp, float* cosv, float* sinv);
  void attn_full_q(int L, const float* qg, const float* cosv, const float* sinv, float* qn);
  void attn_full_combine(int L, int T, const float* qn, float* part, const float* qg, float* ob, XBuf& xo);
  void attn_full_multi(int L, int pos0, int n, float* const* qg, float* const* kp, float* const* vp, float* const* ob, XBuf* xo);
  int split_min() const;
  // multi-position verify of drafted tokens (verify.cpp): up to TMAX positions in one pass, every position's logits kept;
  // commit(a) keeps the first a, the DeltaNet states rebuilt exactly from the step-start copy and the kept positions'
  // stored projections (the decode path's own conv and head phases run again on them); KV needs nothing (append-only)
  static constexpr int TMAX = 8;
  struct Ver;
  std::shared_ptr<Ver> V;
  bool verify(State& s, const int32_t* ids, int n) override;
  void commit(State& s, int a) override;
  const float* logits_at(int i) const override;
  std::string verify_json() const;  // verify statistics (steps, rollbacks, ms of state copies and replays)
};

// the decode path's Gated DeltaNet phases (qwen3_5_moe.cpp), also run by the verify for each of its positions
struct LinCtx {
  Qwen35* M;
  int L;
  float *qkv, *z, *bb, *aa, *cv, *o;
  bool quant_heads;
  std::atomic<int> qk_done[64];  // phase_lin_conv_heads: the conv of key head kh's q and k channels is done
  // phase_lin_conv_heads: the output projection after the heads in the same fork-join (row chunks of 32 once every head
  // is done; the heads quantise their own outputs), its rows as run_rows computes them
  const Weight* wo = nullptr;
  float* rout = nullptr;
  int o_chunks = 0;
  std::atomic<int> heads_left{0};
  alignas(128) std::atomic<int> next{0};
};
void phase_conv(void* p, int tid, int nt);
void phase_lin_heads(void* p, int, int);
void phase_lin_conv_heads(void* p, int, int);  // the decode path: both in one fork-join

// one channel of the causal convolution for one position, as the decode path computes it: the taps over the state
// (oldest first) as a fused multiply-add chain, the newest input's product rounded on its own and added; then the state
// shifts by one
inline float conv_channel(float* st, const float* wc, float x, int C, int CK, int ch) {
  float v = 0;
  for (int k = 0; k < CK - 1; k++) v = __builtin_fmaf(st[(size_t)k * C + ch], wc[k], v);
  v = add_unfused(v, x, wc[CK - 1]);
  for (int k = 0; k + 1 < CK - 1; k++) st[(size_t)k * C + ch] = st[(size_t)(k + 1) * C + ch];
  st[(size_t)(CK - 2) * C + ch] = x;
  return silu(v);
}

}  // namespace qwen35
}  // namespace dm
