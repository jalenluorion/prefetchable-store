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
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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

// The model the drafter runs on (dmeng: the Model and its State; dmserve: the prefix cache's live sequence).
struct DraftTarget {
  virtual ~DraftTarget() {}
  virtual int pos() const = 0;
  virtual void step(int id) = 0;                    // one position; logits() after it
  virtual const float* logits() const = 0;
  virtual bool verify(const int* ids, int n) = 0;   // n positions in one pass; logits_at(i) after position i
  virtual const float* logits_at(int i) const = 0;
  virtual void commit(int a) = 0;                   // keep the first a positions of the last verify
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
  long offered = 0, declined = 0, probes = 0, since_probe = 0;
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
  long early_names = 0, early_cancelled = 0;
  // reads by kind of step, [0] plain steps, [1] verify passes: steps, tokens produced, on-the-spot reads (misses), picks
  // late (named, still in flight), bytes read on the spot / by names
  double rd_steps[2] = {0, 0}, rd_tok[2] = {0, 0}, rd_miss[2] = {0, 0}, rd_late[2] = {0, 0}, rd_urg[2] = {0, 0}, rd_pf[2] = {0, 0},
         rd_pos[2] = {0, 0};
  int last_n = 1;  // positions of the last step's pass
  double ms_steps = 0;
  void reset_stats();
  std::string stats_json() const;

 private:
  std::vector<DraftSource*> srcs_;
  std::map<std::pair<long, long>, std::vector<std::array<int, 3>>> ea_held_;
  void early(const Draft& d, int L, long p);
};

}  // namespace dm
