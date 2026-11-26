// Inkling 2 Mini's own draft head (mtp.safetensors, the "DSpark" block drafter of sglang PR 51926: models/inkling2_mtp.py on
// models/dflash.py + models/dspark.py), batch 1, on the CPU, weights bf16 as released.
//
//   features of a committed position p: the target's hidden state entering applications 5, 23, 41 and the post-final-norm
//   hidden (taps, taken in the target's own step / chunk). Draft layer i's context K/V at p: rmsnorm_w(sum_t
//   softmax(fuse_weight[i])_t feature_t) -> k_proj_ctx / v_proj_ctx, k_norm_ctx, RoPE (theta 1e6) at p.
//   After the target commits p and produces t[p+1], a block of 8 slots at positions p+1 .. p+8: slot 0 = embed_norm(embed(
//   t[p+1])), slots 1-7 = mask_embed; 3 pre-norm layers (q / k norms, RoPE, attention over the context and the whole block,
//   no mask inside the block; layers 0 and 2 see the last 1,023 context positions), SwiGLU 9216, final norm.
//   Draft j (j = 1..7, the token at p+1+j) = argmax over the unpadded vocab of 30 tanh(head(h_j + hidden_out(latent_j)) / 30),
//   latent_j from the Markov RNN over [state, markov_w1[previous token], h_j, f_rnn(p)]; previous token of draft 1 = t[p+1].
//
// The arithmetic follows the MLX reference (tests/inkling2_loop/draft_ref_mlx_run.py): every op's output rounded to bf16 where MLX's is,
// fp32 inside norms, RoPE, softmax and the RNN gate. Matrix products: bf16 x bf16 with fp32 sums (BFMMLA, ink_draft.cpp).
// The output layer is the target's own. Two ways to apply it (option `exact`):
//   exact = false (decoding): head(h_j) + F latent_j + c, F = unembed . hidden_out.W [VU, 256] and c = unembed .
//     hidden_out.b folded once (bf16 F, cached under <model>/.dmeng/), head(h_j) the engine's head (its 8-bit resident),
//     computed for draft 1 alone and then for the rest in one batch; one 0.1 GB pass per draft instead of a 0.46 GB one.
//   exact = true (the port check): z_j = h_j + hidden_out(latent_j) in bf16, logits = bf16(z_j . unembed) with the
//     released bf16 unembedding, as the reference computes them.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "dmeng/pool.h"
#include "dmeng/kernels.h"
#include "dmeng/kernels_batch.h"
#include "dmeng/weights.h"

namespace dm {

struct InkDraftOpts {
  std::string mtp;        // mtp.safetensors
  std::string resident;   // resident.safetensors (the released bf16 unembedding: the fold, and the exact mode)
  std::string fold;       // the fold cache file (derived on first use)
  const Weight* head = nullptr;  // the engine's head (quantised: an XQ input); exact mode: unused
  int H = 0, VU = 0, ctx = 16384;
  float softcap = 30.f;
  bool exact = false;
  // decoding: hold the block's matrices and the fold F in the engine's 8-bit in-memory format (AFF8, groups of 64,
  // fp16 scale and offset), as every other in-memory weight; drafts only propose, so their precision cannot change the
  // output (a worse draft is rejected). false: bf16 as released (the port check's exact mode always is)
  bool q8 = true;
  Prec prec = Prec::I16;  // the head's activation precision (the target's)
  Mem mem = Mem::MLOCK;
};

class InkDraft {
 public:
  ~InkDraft();
  bool load(const InkDraftOpts& o, Pool* pool, std::string& err);
  // features of committed positions, in order: rows [n, 4, H] for positions p0 .. p0 + n - 1 (p0 = the next position)
  void add_taps(const float* rows, int n, int p0);
  // positions >= pos are gone (rolled back, or a new sequence: pos = 0)
  void truncate(int pos);
  int next_pos() const { return nproj_ + npend_; }
  // drafts after committing position p = next_pos() - 1: tok = t[p+1] (the target's newest token, not yet fed), emb =
  // embed_norm(embed(tok)) as the target computes it [H]. Up to max (<= 7) drafts, stopping after the first whose top-1
  // probability is below thr (that one is not returned). out / prob: [max]; returns the number of drafts.
  int draft(int32_t tok, const float* emb, int max, float thr, int32_t* out, float* prob, float* logits1 = nullptr);
  // the exact mode's logits of draft 1 for the check tool (VU floats, after the softcap), when logits1 is given
  size_t weight_bytes() const { return wbytes_; }
  size_t state_bytes() const;
  std::string stats_json() const;
  void reset_stats();

  // statistics
  long n_calls = 0, n_proj = 0, n_drafts = 0, n_head_batched = 0;
  double ms_proj = 0, ms_block = 0, ms_head = 0, ms_total = 0;

 private:
  struct Layer {
    const uint16_t *q, *k, *v, *o, *g, *u, *d, *kc, *vc;  // bf16 [out, in] (null once quantised)
    Weight wq[9];                                         // q8: the same nine, AFF8
    std::vector<float> ln_in, ln_post, qn, kn, kcn, dfl;    // norms (bf16 values)
  };
  bool read_all(std::string& err);
  bool make_fold(std::string& err);
  bool layout_q8(std::string& err);    // q8: the AFF8 region for the block's matrices and F (filled as they are read)
  bool q8_ = false;
  uint8_t* qreg_ = nullptr;
  size_t qreg_bytes_ = 0;
  std::vector<uint16_t> rnn_wv_, ho_wv_;
  size_t rnn_n_ = 0, ho_n_ = 0;  // q8: the RNN and hidden_out matrices (bf16, small) once wreg_ is gone
  XBuf latq_;
  std::vector<float> xfp_;  // q8: the context projections' inputs (bf16 values) as fp32
  void project(int n);                 // the oldest n pending positions -> context K/V
  void block(int p, const float* emb);  // the 8-slot block after committed position p -> hfin_
  void rnn_step(int prev, int j, float* latent);
  // lg = softcap(a + b + c) over VU (a / c may be null; b may be lg); returns the softmax probability of the argmax (*id)
  float pick(float* lg, const float* a, const float* b, const float* c, int32_t* id);

  InkDraftOpts o_;
  Pool* pool_ = nullptr;
  int H_ = 0, NH_ = 24, NKV_ = 8, HD_ = 128, I_ = 9216, WIN_ = 1024, BS_ = 8, MR_ = 256, VU_ = 0;
  float eps_ = 1e-6f, theta_ = 1e6f;
  uint8_t* wreg_ = nullptr;  // every weight but markov_w1, one region
  size_t wbytes_ = 0, wreg_bytes_ = 0;
  Layer L_[3];
  std::vector<float> norm_, mask_, alpha_, mk_dfl_, rnn_b_, ho_b_;
  const uint16_t *rnn_w_ = nullptr, *ho_w_ = nullptr;
  int mtp_fd_ = -1;
  uint64_t w1_off_ = 0;  // markov_w1 rows read from the file (one 512-byte row per draft)
  // fold: F [VU, 256] bf16, c [VU]
  uint8_t* freg_ = nullptr;
  size_t freg_bytes_ = 0;
  Weight F_;
  std::vector<float> fc_;
  // exact mode: the bf16 unembedding (mapped)
  void* umap_ = nullptr;
  size_t umap_bytes_ = 0;
  Weight U_;
  // context K / V per layer (bf16): layers 0 / 2 rings of WIN_ positions, layer 1 every position
  uint16_t *ck_[3] = {nullptr, nullptr, nullptr}, *cv_[3] = {nullptr, nullptr, nullptr};
  size_t cbytes_[3] = {0, 0, 0};
  int ccap_[3] = {0, 0, 0};
  int nproj_ = 0, npend_ = 0;
  std::vector<float> pend_;   // [npend, 4, H] (bf16 values)
  std::vector<float> last_;   // [4, H] the features of the last projected position
  // scratch
  std::vector<float> x_, resid_, hn_, q_, k_, v_, att_, ao_, gg_, uu_, act_, hfin_, f_rnn_, state_;
  std::vector<uint16_t> xb_, pk_;  // bf16 inputs [n, K] and their packed form
  XBuf hx_[8];
  XQB hxb_;
  std::vector<float> hl_;      // head(h_j) [7, VU]
  std::vector<float> lg_;      // logits of one draft [VU]
};

std::string ink_draft_bench(Pool& pool, int reps);  // the block's bf16 products on synthetic weights (JSON)

}  // namespace dm
