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
#include "dmeng/prefill.h"
#include "dmeng/screen_head.h"
#include "../common/gdn.h"

namespace dm {
struct GpuGemm;  // src/core/gpu_gemm.h
namespace qwen35 {

// the 35B's own prefill switches (the shared ones: dmeng/prefill.h PrefillOpts)
struct PrefillExtra {
  bool batched_always = false;  // chunk 1 through the batched code (one position per pass): a debugging reference
  FILE* record = nullptr;       // if set: every position's picks, int16 [pos][NL][K], appended per span
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
  // the screened head (core, screen_head.h): the head split in place into nibble planes; a pass over the high planes
  // bounds every row, the rows that can be in the top k are computed exactly. The argmax (Want::GREEDY) and the top-k
  // logits (Want::TOPK, k = head_topk) are the full head's, bit for bit
  bool fast_head = false, fast_head_check = false;
  int head_topk = 0;
  double hd_mismatch = 0;
  ScreenHead sh;
  int greedy_ = -1;
  // run-ahead (Core::run_ahead): one layer ahead on an approximate state, for its picks
  std::vector<float> ra_x, ra_h, ra_y, ra_lg;
  XBuf ra_xq;
  int run_ahead_step(int L, const float* ypart, int32_t* picks, int maxk);
  bool attn_linear_ro(int L, const float* hin, float* rout);
  int run_ahead_deeper_step(int d, int32_t* picks, int maxk);
  int ra_L = -1;  // the last layer the run-ahead state holds (its MoE left out)

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
    sh.reset_stats();
    hd_mismatch = 0;
  }
  ModelMem mem(const State& s) const override;  // prefill.cpp (the batched-prefill scratch sizes)
  // batched prefill (prefill.cpp): layer-major over a span, bit-identical to single steps
  bool prefill(State& s, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture, const LM::StateSink& sink,
               PrefillStats* st, const std::atomic<bool>* cancel) override;
  PrefillOpts po;   // the batched prompt path's options (dmeng/prefill.h; this model's values: its manifest's "prefill")
  PrefillExtra px;
  std::unique_ptr<Prefill> P;
  GpuShare gpu;     // the Metal share of the prefill's products (po.gpu_frac / gpu_experts > 0), kept across prefills
  std::vector<size_t> fixed_off;  // [NL] byte offset of a DeltaNet layer's conv state in the fixed blob
  gdn::Dims gd;                   // the Gated DeltaNet's shape (silu gate)
  // a DeltaNet layer's fork-join context for one position (gdn.h): the state of S, the layer's weights, the buffers
  void lin_ctx(gdn::Ctx& X, int L, float* qkv, float* z, float* bb, float* aa, float* cv, float* o, bool quant_heads);
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
  bool verify(State& s, const int32_t* ids, int n, Want want = Want::LAST) override;
  void commit(State& s, int a) override;
  int ver_best[TMAX] = {-1, -1, -1, -1, -1, -1, -1, -1};  // the screened head's argmax per position of the last verify
  const float* logits_at(int i) const override;
  // draft heads trained outside (verify.cpp): post-norm final hidden of the last step (i < 0, kept when the step wanted
  // logits) or of verify position i; the raw embedding row; the output head's argmax (screened when built) + top-1 prob
  const float* final_hidden(int i) const override;
  bool embed_row(int32_t tok, float* out) const override;
  int head_argmax(const float* hn, float* conf) override;
  XBuf dh_x;
  std::vector<float> dh_lg;
  std::string verify_json() const;  // verify statistics (steps, rollbacks, ms of state copies and replays)
};

// the Gated DeltaNet's convolution channel and value head are the shared ones (src/models/common/gdn.h): the decode path,
// the verify and the batched prefill all compute through them
using gdn::conv_channel;

}  // namespace qwen35
}  // namespace dm
