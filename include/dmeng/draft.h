// Draft and verify, shared by dmeng (the command line) and dmserve (the server worker), for every model with a
// multi-position verify (Model::verify / commit). Greedy only: a drafted token is kept only while it equals the model's
// own greedy token, and the rest of the pass is rolled back, so the ids are exactly one-position decoding's whatever the
// drafts were. Drafts only decide how many positions a pass carries (and which expert reads start early).
//
//   DraftSource   proposes candidate tokens to follow the sequence, each with a confidence (the source's estimate that it
//                 is right when the ones before it are) and a bucket (the source's own class of proposal). Sources here:
//                 LookupSource (the longest earlier match of the sequence's last 2-16 tokens, over the whole session:
//                 prompt + everything generated, plus an archive of earlier sequences) and SkeletonSource (fragments of
//                 the tool-call format of a request's tools, e.g. "<tool_call>\n<function=NAME>\n<parameter=KEY>\n").
//                 HeadSource: a model's own draft head (Model::draft_head, Inkling's MTP), one more source.
//   DraftCtl      the controller: acceptance per (source, bucket, draft index), learned from the steps run; the measured
//                 cost of a pass of n positions against one position; the draft length that maximises expected tokens
//                 per unit of cost (0 = a plain step); a probe every `probe_every` declined offers.
//   Drafter       one decode step on a DraftTarget: asks the sources, lets the controller choose, runs a plain step or a
//                 verify, keeps the accepted tokens; optionally names (early reads) the experts the copied source
//                 positions used, for the step's positions and `early` positions past them.
#pragma once
#include <array>
#include <cstdio>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dmeng/weights.h"

namespace dm {

class Core;
class Model;
struct DraftCtl;

// A proposal. toks: candidates to follow the sequence; conf[k]: the source's estimate that toks[k] is right given that
// toks[0..k) were (used as the controller's prior for a cell it has not seen); bucket in [0, buckets()); src: the sequence
// position toks[0] was copied from (-1: none), for early names.
struct Draft {
  std::vector<int32_t> toks;
  std::vector<float> conf;
  int bucket = 0;
  long src = -1;
  void clear() {
    toks.clear();
    conf.clear();
    bucket = 0;
    src = -1;
  }
};

class DraftSource {
 public:
  virtual ~DraftSource() {}
  virtual const char* name() const = 0;
  virtual int buckets() const { return 1; }
  // hist: the whole sequence so far (its last token is the one the next step feeds). Up to max tokens to follow it.
  virtual bool propose(const std::vector<int32_t>& hist, int max, Draft& d) = 0;
  // the last proposal of this source that ran: `offered` tokens verified, the first `accepted` right
  virtual void feedback(int offered, int accepted) {}
  // asked before propose (false: not asked this step); best_gain = the best expected gain (tokens per unit of cost) of
  // the drafts the sources before it offered this step, 0 if none was worth a pass
  virtual bool ask(const std::vector<int32_t>& hist, double best_gain) { return true; }
  // after a step in which this source proposed: l = the draft length the controller chose for it (0: none), gain = its
  // expected gain, chosen = its draft ran, accepted = how many of it were kept, step_ms = the whole step (proposals in)
  virtual void outcome(const DraftCtl& ctl, int l, double gain, bool chosen, int accepted, double step_ms) {}
  bool timed = false;  // its proposal time is left out of the controller's pass costs (a source that runs a model part)
};

// Longest earlier match: for the sequence's last n tokens (n = 16 down to 2), the most recent earlier occurrence; the
// tokens that followed it are the draft. Indexed by bigram (incremental as the sequence grows; rebuilt when it diverges),
// at most `max_cand` most recent occurrences examined. Buckets by match length: 2, 3, 4-7, 8+. An archive (earlier
// sequences of the session that the current one no longer contains, e.g. a previous turn's reasoning the chat template
// dropped) is searched too; a match there has no source position.
class LookupSource : public DraftSource {
 public:
  int min_n = 2, max_n = 16, max_cand = 64;
  size_t archive_cap = 1 << 16;  // tokens kept in the archive
  const char* name() const override { return "lookup"; }
  int buckets() const override { return 4; }
  bool propose(const std::vector<int32_t>& hist, int max, Draft& d) override;
  void reset() { idx_.seq.clear(), idx_.pos.clear(); }
  void add_archive(const std::vector<int32_t>& s);  // a sequence the session produced and may repeat
  int last_n = 0;                                   // match length of the last proposal (0: none)
  static int bucket_of(int n) { return n >= 8 ? 3 : n >= 4 ? 2 : n - 2; }

 private:
  struct Index {
    std::vector<int32_t> seq;
    std::unordered_map<uint64_t, std::vector<int32_t>> pos;  // bigram -> positions of its second token, ascending
    void add(size_t from);
  };
  Index idx_, arch_;  // the sequence (what is indexed of it) and the archive
  bool sync(const std::vector<int32_t>& hist);
  static int match(const std::vector<int32_t>& a, long ia, const std::vector<int32_t>& b, long ib, int cap);
};

// Fragments of a request's tool-call format (token ids, from the front end's tokenizer and chat format): the sequence's
// last tokens matched inside the fragments (at least 1 token, the longest match); the draft is the continuation every
// best-matching fragment position agrees on (so a key or a closing is drafted only once it is determined).
class SkeletonSource : public DraftSource {
 public:
  int max_back = 16;
  const char* name() const override { return "skeleton"; }
  int buckets() const override { return 3; }
  void set(const std::vector<std::vector<int32_t>>& frags) { frags_ = frags; }
  bool empty() const { return frags_.empty(); }
  bool propose(const std::vector<int32_t>& hist, int max, Draft& d) override;
  static int bucket_of(int m) { return m >= 4 ? 2 : m >= 2 ? 1 : 0; }

 private:
  std::vector<std::vector<int32_t>> frags_;
};

// The model's own draft head (Model::draft_head; the model keeps the head's features from its own steps): up to max
// drafts while the head's top-1 probability stays >= thr. Bucket: the bin of the first draft's probability; conf: the
// running acceptance per probability bin. Not run while backed off (after a lost try: no draft, its draft not chosen or
// none of it kept, or the step plus the head slower per token than a plain step; backoff0 tokens, doubling up to
// backoff_max), nor while an earlier source's offer is worth more than the head has been (its running value, its own
// cost included).
class HeadSource : public DraftSource {
 public:
  static constexpr int NB = 6;
  Model* M = nullptr;
  float thr = 0.9f;
  int backoff0 = 32, backoff_max = 256;
  HeadSource() { timed = true; }
  const char* name() const override { return "head"; }
  int buckets() const override { return NB; }
  bool ask(const std::vector<int32_t>& hist, double best_gain) override;
  bool propose(const std::vector<int32_t>& hist, int max, Draft& d) override;
  void outcome(const DraftCtl& ctl, int l, double gain, bool chosen, int accepted, double step_ms) override;
  std::string stats_json() const;
  long runs = 0, offers = 0, used = 0, acc = 0, lost = 0, skipped_backoff = 0, skipped_lookup = 0;
  double ms_total = 0;
  // acceptance per probability bin (priors: the share kept per bin on the Pittsburgh generation and the coding answer,
  // dmeng --draft-check records, results/ink_draft/DRAFT-HEAD.md)
  double cal[NB] = {0.40, 0.55, 0.65, 0.78, 0.88, 0.96};
  static int bin(double p);

 private:
  double h_ms = 0, h_val = 1.5, last_ms = 0;  // head ms per run (running), value of running it (running, cost units)
  long backoff = -1, backoff_until = 0, now = 0;
  std::vector<float> prob_;  // the last proposal's probabilities
};

// A light draft head trained outside the engine (the 35B's: jobs/babel/q36_draft_head.py; model-agnostic). From the
// post-norm final hidden state h_t of the last committed position and the token just chosen from it:
//   z_0 = h_t; for draft offset j = 1..ND: u = [rmsnorm(z_{j-1}) * norm_z_j, rmsnorm(e_j) * norm_e_j] (fp32, eps 1e-6),
//   z_j = z_{j-1} + down_j(silu(up_j(u))), e_1 = the embedding row of the token chosen from h_t, e_j = that of draft j-1;
//   logits_j = vocab . (proj . z_j) (low rank: proj [R, D], vocab [V, R]) or the model's own output head over z_j
//   ("full": Model::head_argmax, the screened greedy head); draft j = argmax (first index on ties), its confidence =
//   max softmax. Drafts stop at the first whose confidence is below thr.
// File: safetensors, bf16, names mlp.{j}.{norm_z,norm_e}.weight [D], mlp.{j}.up.{weight [H, 2D], bias [H]},
// mlp.{j}.down.{weight [D, H], bias [D]}, proj.weight [R, D] and vocab.weight [V, R] (low rank only). Held in the
// engine's 8-bit format (q8, default) or as bf16 values in fp32 (the reference check); drafts only propose.
class LightHeadSource : public DraftSource {
 public:
  static constexpr int NB = 6;
  LightHeadSource() { timed = true; }
  // path: the safetensors file; D must be the model's hidden size
  bool load(const std::string& path, int D, Pool* pool, bool q8, std::string& err);
  std::function<const float*()> hidden;             // h_t of the last committed position (null: no proposal)
  std::function<bool(int32_t, float*)> embed;       // the embedding row of a token
  std::function<int(const float*, float*)> head;    // "full" variants: the model's head over z -> argmax, *conf
  float thr = 0.f;
  int vocab_out = 0;                                // rows that count for the argmax (0: all)
  const char* name() const override { return "light"; }
  int buckets() const override { return NB; }
  bool propose(const std::vector<int32_t>& hist, int max, Draft& d) override;
  void feedback(int offered, int accepted) override;
  // its own cost (ms per call, every draft through the output layer) is paid whether or not a pass runs: it is not run
  // while an earlier source's offer is chosen, nor while backed off (after a call whose draft did not run, kept nothing,
  // or made the step slower per token than a plain step: backoff0 steps, doubling to backoff_max)
  bool ask(const std::vector<int32_t>& hist, double best_gain) override;
  void outcome(const DraftCtl& ctl, int l, double gain, bool chosen, int accepted, double step_ms) override;
  int backoff0 = 32, backoff_max = 256;
  long lost = 0, skipped_backoff = 0, skipped_earlier = 0;
  bool gate = true;                       // confidence-gated width (more()); false: every draft up to the max
  std::function<double(int)> est;         // the controller's estimated ms of a pass of n positions (Drafter::est)
  // the drafts of one call, for checks: from h and the chosen token, up to max (no threshold)
  int run(const float* h, int32_t tok, int max, int32_t* out, float* conf);
  int ND = 0, D = 0, H = 0, R = 0, V = 0;
  bool lowrank = false, q8 = true;
  FILE* dump = nullptr;  // every run: pos i32 (hist index of tok), h [D] f32, tok i32, n i32, drafts i32 [ND], confidences f32 [ND]
  size_t weight_bytes() const;
  std::string stats_json() const;
  long runs = 0, offers = 0, drafts = 0, kept = 0, n_drafts_run = 0, dump_pos = -1;
  double ms = 0, ms_embed = 0, ms_mlp = 0, ms_out = 0;  // per call / per draft: the embedding row, the MLP, the output layer
  double cal[NB] = {0.40, 0.55, 0.65, 0.78, 0.88, 0.96};  // acceptance per confidence bin (learned)
  static int bin(double p);

 private:
  struct Step {
    std::vector<float> nz, ne, ub, db;
    std::vector<uint16_t> up16, dn16;  // bf16 (q8 off)
    OwnedAff8 up, dn;
  };
  std::vector<Step> st_;
  std::vector<uint16_t> proj16_, vocab16_;
  OwnedAff8 proj_, vocab_;
  Pool* pool_ = nullptr;
  XBuf xu_, xa_, xz_, xr_;
  std::vector<float> z_, e_, u_, a_, r_, lg_;
  std::vector<float> last_conf_;
  long now_ = 0, backoff_ = -1, until_ = 0, first_declined_ = 0, declined_seen_ = 0, stopped_early_ = 0;
  double acc_n_[8][NB] = {{0}}, acc_k_[8][NB] = {{0}};  // decayed counts of drafts verified / kept, per index and bin
  double q_of(int j, float c) const;
  bool more(int nd, const float* conf);
  void vocab_rows(int vo, int* id, float* conf);
  void mat(const OwnedAff8& q, const std::vector<uint16_t>& b16, int rows, int cols, XBuf& xb, const float* x, float* y);
};

// The model the drafter runs on (dmeng: the Model and its State; dmserve: the prefix cache's live sequence).
struct DraftTarget {
  virtual ~DraftTarget() {}
  virtual int pos() const = 0;
  virtual void step(int id) = 0;                    // one position; logits() after it
  virtual const float* logits() const = 0;
  virtual bool verify(const int* ids, int n) = 0;   // n positions in one pass; logits_at(i) after position i
  virtual const float* logits_at(int i) const = 0;
  virtual void commit(int a) = 0;                   // keep the first a positions of the last verify
  virtual const float* hidden() const { return nullptr; }  // post-norm final hidden of the last committed position
};

struct DraftCtl {
  static constexpr int DMAX = 8;  // positions per pass (the fed token + up to 7 drafts)
  int probe_every = 48;
  bool fixed = false;  // the whole draft every step (a test of the verify, not a policy)
  double decay = 0.9;
  // acceptance per [source][bucket][draft index 1..7] (< 0: unseen, the proposal's confidence is used)
  std::vector<std::vector<std::array<double, DMAX>>> pa;
  double rr[DMAX + 1];  // cost of a pass of n positions / cost of one position
  double c1 = 0;        // ms of a one-position step (running)
  int seen[DMAX + 1] = {0};
  long offered = 0, declined = 0, probes = 0, since_probe = 0, copy_runs = 0;
  void init(const std::vector<int>& buckets_per_source, const double* cost_prior);
  // the draft length to run for a proposal of source s (0: a plain step)
  int choose(int s, const Draft& d, int max_len, double* gain = nullptr);
  double expected(int s, const Draft& d, int l) const;  // expected tokens of a pass with l drafts
  void accepted(int s, const Draft& d, int l, int acc);
  void cost(int n, double ms);
  static const double* cost_prior(const std::string& family);
};

struct DraftOpts {
  std::string mode = "none";  // none | lookup (controller) | fixed (the whole draft every step)
  int max = 7;                // drafts per pass (<= DMAX - 1)
  int probe_every = 48;
  int early = 0;              // early names for the step's positions and this many past them (0: off)
  bool lookup = true;         // the lookup source (off: only the skeleton and the extra sources draft)
  bool gate = true;           // the cost model from per-pass counters (WidthStat); off: the cost ratio table alone (the earlier rule)
  double margin = 0.05;       // a draft runs only when its estimated time per token beats a plain step's by this share
  bool copy_free = true;      // copy drafts (the lookup) are not rationed once they are being kept (Drafter::step)
  double copy_keep = 0.75;    // ... share of its recent drafts kept from which they count as being kept
};

// Per-pass counters for passes of one width (n positions; n = 1: plain steps). The controller's estimate of a pass of n
// positions is c1 x rho(n) + wait(n) + next(n): c1 = a plain step's compute (wall minus waits, the timed sources' own time
// left out), rho(n) = the pass's compute over c1 at the time (a ratio, so a busy machine slows both alike), wait(n) = the
// pass's wait for reads (measured per width; for a width not measured yet, its urgent MB from the measured widths' slope
// times the passes' wait per urgent MB), next(n) = the wait of the plain step after it beyond a plain step's (what the
// pass left the next step to read). A plain step is c1 + wait(1), wait(1) from plain steps that follow plain steps.
struct WidthStat {
  long n = 0, tokens = 0;
  double rho = -1, wait = -1, urg = -1;  // plain steps: decayed means (wait ms, urgent MB); passes: the last pass's
  double s_cmp = 0, s_wait = 0, s_urg = 0, s_tot = 0;  // sums (statistics): compute ms, wait ms, urgent MB, total MB (urgent + named)
  double d_n = 0, d_wait = 0, d_urg = 0, d_cmp = 0, d_ref = 0, d_tot = 0;  // decayed sums (plain steps: wait, urgent MB; passes: compute, c1 at the time, wait)
};

class Drafter {
 public:
  DraftOpts o;
  LookupSource lookup;
  SkeletonSource skeleton;
  std::vector<DraftSource*> extra;  // more sources (a draft head), asked after the two above
  DraftCtl ctl;
  Core* core = nullptr;                  // for early names (null: none)
  Core* reads = nullptr;                 // the read counters per kind of step (null: not kept)
  // the token the model produces from a position's logits; i = its index among this step's outputs (0 .. accepted).
  // Unset: greedy (argmax). Sampling: the release sampler with that token's own draw (sample_at), so a draft is kept
  // exactly when it equals the token sampled decoding would give there.
  std::function<int(const float* lg, int i)> pick;
  std::vector<int16_t> pick_log;         // every position's released picks (Core::pick_log points here with early on)
  // init after the options are set: family = the model family (the cost prior)
  void init(const std::string& family, Core* c);
  bool on() const { return o.mode != "none"; }
  // One decode step. hist = the whole sequence, its last token the one to feed now (not yet in the model's state).
  // Produces 1 + accepted tokens into out (appended), stopping early at a stop token or when `budget` tokens are out;
  // the state then holds hist plus every produced token but the last (as after that many single steps).
  int step(DraftTarget& T, const std::vector<int32_t>& hist, int vocab, int budget, const std::vector<int>& stop, std::vector<int>& out);
  void end();  // release early names (end of a generation)
  // statistics
  long steps = 0, tokens = 0, verifies = 0, drafted = 0, acc_total = 0;
  long by_n[DraftCtl::DMAX + 1] = {0}, by_src_steps[4] = {0}, by_src_acc[4] = {0};
  long early_names = 0, early_cancelled = 0, prenamed = 0, explored = 0, probes_held = 0, copy_free_runs = 0;
  // reads by kind of step, [0] plain steps, [1] verify passes: steps, tokens produced, on-the-spot reads (misses), picks
  // late (named, still in flight), bytes read on the spot / by names
  double rd_steps[2] = {0, 0}, rd_tok[2] = {0, 0}, rd_miss[2] = {0, 0}, rd_late[2] = {0, 0}, rd_urg[2] = {0, 0}, rd_pf[2] = {0, 0},
         rd_pos[2] = {0, 0};
  int last_n = 1;  // positions of the last step's pass
  double ms_steps = 0;
  // the cost model
  WidthStat ws[DraftCtl::DMAX + 1];
  double c1 = 0;                // a plain step's compute ms (running)
  long plain_n = 0;
  double est(int n) const;      // estimated ms of a pass of n positions (1: a plain step)
  // the compute ratios of passes measured by an earlier run (a line "draft_cost FAMILY r1 .. r8"), in place of the
  // family's prior (DraftCtl::cost_prior); after init. save_cost writes this run's (false: too few passes, or not
  // written): the measured prior of the next run
  bool load_cost(const std::string& path, std::string& err);
  bool save_cost(const std::string& path) const;
  std::string cost_loaded;      // the file the prior came from (empty: the family's table)
  struct Fit {                  // decayed least squares of y = a + b x
    double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    void add(double x, double y, double dec) { sw = dec * sw + 1, sx = dec * sx + x, sy = dec * sy + y, sxx = dec * sxx + x * x, sxy = dec * sxy + x * y; }
    bool solve(double a0, double b0, double pw, double& a, double& b) const;
  };
  Fit f_rho, f_urg, f_wait;     // passes: compute / c1 - 1, urgent MB and wait ms against n - 2
  double k_wait = 0, k_urg = 0; // passes: decayed wait ms and urgent MB
  double nx_s = 0, nx_w = 0;    // the plain step after a pass: decayed excess wait
  void reset_stats();
  std::string stats_json() const;

 private:
  std::vector<DraftSource*> srcs_;
  std::map<std::pair<long, long>, std::vector<std::array<int, 3>>> ea_held_;
  void early(const Draft& d, int L, long p);
  double rho0_[DraftCtl::DMAX + 1] = {0};  // the family's prior compute ratios
  std::string family_;
  double cp_n_ = 0, cp_k_ = 0;  // the copy source's drafts verified / kept (decayed per pass)
  int pend_n_ = 0;
  bool last_full_ = false;  // the last pass kept every draft
  // the probe budget (ms, MB): 1% of every step's cost, minus what probes / explorations / names ahead cost beyond plain
  // steps, plus chosen passes' gains; a probe runs only when it can afford losing all but its first token
  double pb_ms = 0, pb_mb = 0, pb_xb = -1;
  bool gamble_ = false;
  bool probe_ok(int l) const;
  long explore_ = 0;      // the previous step's pass width (> 1: its next-step excess is measured now)
  bool prev_pass_ = false;
  long gen_steps_ = 0;  // steps run (never reset; the first two are not measured)
  void measure(int n, int tok, double cmp, double wait, double urg, double tot);
};

}  // namespace dm
