// dmeng: batch-1 decoding of a mixture-of-experts model from its expert store; token ids in, token ids and one JSON line
// of statistics out. The model family is read from the resident file.
//
//   dmeng --model DIR --prompt-file ids.txt                          (the profile's defaults: docs/MANIFEST-PRESETS.md)
//   dmeng --model DIR --mode fast --prompt "text"                    (text through python/run.py)
//   dmeng --model DIR --teacher ids.txt --dump-logits out.f32        (teacher-forced logits at every position)
//   dmeng --help | --help-advanced
#include "dmeng/workcount.h"
#include <limits.h>
#include <sys/stat.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <unistd.h>

#include <cerrno>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <random>
#include <set>
#include <sstream>

#include "dmeng/address.h"
#include "dmeng/draft.h"
#include "dmeng/model.h"
#include "dmeng/presets.h"
#include "dmeng/sampler.h"
#include "../src/models/inkling2_loop/ink_draft.h"
#include "../src/serving/memwatch.h"

using namespace dm;

static const char* HELP_SHORT = R"(usage: dmeng --model DIR (--prompt TEXT | --prompt-file IDS | --prompt-ids IDS) [options]

  --model DIR            the model folder, as downloaded from the Hub
  --prompt TEXT          text in, text out (runs python/run.py of this repo: needs the tokenizers package);
                         --chat renders the model's chat template around it
  --prompt-file F        token ids from a file (commas or spaces); --prompt-ids 1,2,3 on the command line
  --mode exact|fast      exact: generated tokens never depend on the cache, the routers or timing (the Mac's default);
                         fast: a picked expert that is not in memory and has a small routing weight is left out
  --profile mac|phone|phone-small   default: chosen from the device (macOS: mac)
  --n N                  tokens to generate (default 120)
  --temp T --top-p P --top-k K --seed S   sampling (default: greedy)
  --cache-gb G|auto      expert cache size in GB (default auto: from the memory free at start)
  --drafts 0|1           draft and verify, greedy or sampled (the ids are plain decoding's for the seed), in exact and fast
                         mode (default 0)
  --json F               write the statistics line to F as well
  --help-advanced        every other option, grouped
)";

static const char* HELP = R"(usage: dmeng --model DIR (--prompt TEXT | --prompt-ids IDS | --prompt-file F | --teacher F) [options]

profiles and modes (docs/MANIFEST-PRESETS.md)
  --profile auto|mac|phone|phone-small|none   the defaults below come from the profile (manifest keys `profiles`,
                         `defaults`, built-in fallbacks); auto (default): from the device; none: no profile, the
                         engine's built-in values (no names, --cache-gb 5, single-step prompt, ...). A default fills only
                         what the command line leaves unset; any of --prefetch, --routers, --xtok, --oracle,
                         --plan-routers makes the early address entirely the command line's
  --mode exact|fast|lossless|approx[:PRESET]   exact = lossless, fast = approx:drop (default: the profile's)
  --drafts 0|1           1 = --draft lookup --early 4, as in dmserve (a model's draft head: --draft head|both); default:
                         the profile's (off)
  --reserve-gb G         --cache-gb auto: memory left to the rest of the system (default 5.5)
  --print-defaults       print the resolved profile defaults as JSON and exit
  --model-opt NAME=V     a model or engine option (Model::set_option, then Core::set_option: chunk_aff4, prefill_*)
  --chat --reasoning-effort E   with --prompt: the chat template (python/run.py)

model and memory
  --model DIR            the model folder (store/, resident.safetensors, tokenizer files)
  --resident F           the 8-bit resident file (default: derived/ or .dmeng/ of the model folder, derived on first use)
  --embeddings F         embedding rows from this file (the table alone; default: the manifest's, else
                         DIR/embeddings.safetensors when that is absent)
  --cache-gb G|auto      byte budget of the expert slots (default auto under a profile: physical memory - --reserve-gb
                         - the process's other memory, capped by what is free at start; 5 with --profile none)
  --cache-start-gb G     with --mem mlock: slots for --cache-gb reserved, only G locked at start (the serving worker's
                         automatic rule does this); a lock the OS refuses stops the budget there (JSON "cache_lock")
  --pool-share A,B       budget share per record format (default: the model's; Inkling 0.8,0.2)
  --mem plain|mlock|purgeable|file   expert slots (default mlock); --res-mem plain|mlock resident weights (default mlock)
                         file: held records are views of the store files' pages in the OS file cache (--cache-gb is then
                         the policy's bookkeeping, not memory); --file-check 0|1|2 (a held record checked on use: off,
                         first + last page, every page; default 1), --file-hint 0|1|2 (on eviction: 1 MADV_DONTNEED, 2 msync MS_INVALIDATE; default 0),
                         --file-skip-resident 0|1 (default 1), --file-prefault 0|1 (default 1),
                         --file-adapt 0|1 (the budget follows what the OS keeps; --cache-gb is the start and the max),
                         --file-min-gb G (adaptive floor, default 1), --file-core-gb G (a locked core of the hottest records)
  --threads N --readers N --split N    compute threads (8), reader threads (4), pieces per record read (1)
  --reader-io-low 0|1    the readers' disk I/O one tier below the process's own (default 1 on iOS, 0 on macOS)
  --prec i16|i8|f16|f32  activation precision (default i16; f32 = the exact reference path, where the model has one)
  --ctx N                context positions (default 16384)
  --half-life T          eviction: half-life of the decayed use count, in tokens (default 8)
  --no-loop-hold         looped layers: do not keep an earlier pass's experts pinned for the later passes

generation
  --n N --temp T --top-k K --top-p P --seed S --stop-eos ID
  --fast-head 0|1|2      screened head (Qwen3.6, Inkling; default 1): the exact argmax (temperature 0, no drafting) or the exact
                         top-k logits (--top-k 1..64); 0 = the full head; 2 also checks it against the full head
  --start-pos P          timing of long contexts: the first token sits at position P

early address (lossless: names only schedule reads; the released router decides)
  --prefetch none|oracle|router|transplant|plan
  --prefetch transplant --m M      no trained weights: step s + 1's released router on step s's input names its top M
  --oracle F --lead N|token        replay a routing recorded with --record, N steps ahead or one token ahead
  --routers F --m M                trained routers (engine file), name the top M per step (default 8)
  --xtok-bg F                      a second cross-token source whose reads start only while no read of the current
                                   token is queued or in flight (the readers' background class, one at a time)
  --xtok F --xtok-tau T            cross-token routers (engine file, kind ink_xtok): name the next token's first steps
                                   from a late step of this one (threshold T, default 0.5); wraps --prefetch router
  --rt-tau P --rt-cand C           or name the top-C names whose calibrated probability >= P
  --router-ecache N                split router files: cache the embedding columns' output of N token ids
  --router-stats                   run the routers and report how well they name, without naming
  --plan-routers F1,F2,..          read planner over a cascade of router sets (longest lag first); --plan-u,
                                   --plan-budget 0|1|2, --plan-tau-long, --plan-tau, --plan-pin-p, --plan-revoke,
                                   --plan-revoke-p, --plan-maxcand, --plan-calib rank|q
  --evict auto|freq|pred|hazard  --rd-prio 0|1  --rd-excl 0|1  --spec-max N  --cancel 0|1
  --bg-ranks D --bg-gate G --bg-split P --bg-queue Q --bg-max N   background reads for the drive's idle time: the routers' ranks
                         m+1..D of the steps already named, while no urgent read and at most G speculative ones are in
                         flight, in P pieces (N in flight), at most Q records queued (default: the profile's; Flash-Next mac 8, 2, 2, 2, 1;
                         --bg-ranks 0 = off)
  --name-gate 0|1        1: a name that needs a read is not issued while urgent reads wait for a reader (default 0)
                         auto (default): the hazard keep-score (the phone's online per-layer table, results/ios/evict/
                         EVICT.md), with uniform per-layer caps while the cache holds fewer than 4 records per layer per
                         pick (35B below 2.26 GB), else one pool (results/cache-policy/POLICY.md); freq: the decayed use
                         count of the engines before (one pool); hazard: the table, caps as --layer-caps says
  --layer-caps auto|uniform|none   per-layer slot caps (each layer evicts its own records at its cap); auto: the rule
                         (CAPS:fit: a name leaves min(2K + holds, cap - names per step) free in a capped layer, not 2K + holds)
  --router-heavy F       a heavy-pick head (router file of the same format) beside --routers: its positives are named first
  --drop 0|1             approx mode: leave missing light picks out (kept picks keep their released weights) instead of replacing them
  --evict-policy F.json  learned keep-score (logistic next-use predictor; router features from --prefetch router), or a
                         ranker over hazard's shortlist (a file with "features", e.g. the design study's v3:
                         q36_d_evict3_policy.json; turns hazard on; its early-router features need --routers F with
                         --prefetch router or --router-stats, the router of the file's lag)
  --evict-trace F        per (token, step) the released router's and the early router's top-32 rows (for the simulator)

modes
  --mode lossless|exact  (default) token ids never depend on the cache, the routers or the timing
  --mode approx[:PRESET] opt-in (fast = approx:drop): a released pick that has not landed may be replaced by the released router's
                         next-ranked expert in memory. Presets: PRESETS
  --late none|always|MS  --read-w W  --read-mass M  --bg 0|1     the approximate rule set, set by hand

outputs
  --json F --out-ids F --steps-out F --record F --dump-names F
  --teacher F --dump-logits F --dump-from P --nll-out F --ref-nll F --qual-out F --ref-qual F --taps-out F
  --skip-reads --force-routing F   timing without reads, with a recorded routing forced
  --state-check          after the decode, restore the state saved after the prompt (fixed + positional parts) and
                         decode again: the ids must be the same
  --prefill-chunk N      the prompt through the model's batched prefill in chunks of N (1: single steps); exact
                         (default 512 under a profile unless --record is given, else 1)
  --prefill-gpu-frac F   with --prefill-chunk: that share of each attention projection's rows on the GPU beside
                         the CPU (builds with DM_METAL; the same bits); 0 = CPU only (default: the profile's; the
                         mac profile 0.4 for both models, else 0)
  --prefill-gpu-experts F  the same for the experts' products (routed experts in the slab, the shared expert;
                         the mac profile 0.4; Inkling: its NVFP4 layers)
  --prefill-gpu off      the batched prefill on the CPU only (both shares 0)
  --draft lookup|fixed|head|both   draft and verify: drafts verified up to --draft-max 7 at a time in one pass;
                         accepted while each equals the token the pass itself produces (greedy: its argmax; sampled: the
                         sampler with that token's own draw, so the seed's tokens are unchanged; fast mode: the pass applies
                         the single step's leave-out rule), the rest rolled back, so the ids are one-position decoding's. Sources: the lookup (the longest earlier match of the sequence's
                         last 2-16 tokens) and the model's draft head (Inkling's MTP, mtp.safetensors; its features come
                         from the model's own steps). lookup / head / both: a controller picks the source and the draft
                         length from running acceptance and measured step costs (or a plain step); fixed: the whole
                         lookup draft every step
  --draft-q8 0|1         the draft head's matrices in the engine's 8-bit format (AFF8; default 1) or bf16 as released (0);
                         drafts only propose, so the ids do not depend on it
  --draft-thr P          head drafts are used only while their top-1 probability is >= P (default 0.9: the best measured on the
                         Pittsburgh generation, within 0.1% of the best on the coding answer; results/ink_draft/DRAFT-HEAD.md)
  --head-backoff B,M     after a lost head try (no draft over the threshold, its draft not chosen, none accepted, or the
                         step plus the head slower per token than a plain step) the
                         head is not run for B tokens, doubling on each further loss up to M (default 32,256)
  --draft-exact 0|1      the head's output layer as the MLX reference computes it (bf16 unembedding; slow, checks only)
  --draft-check TAPS --prompt-file IDS --draft-check-start P --draft-check-out PREFIX [--draft-check-logits N]
                         the head's port check on recorded features (TAPS: per position int32 + float [4, H], as
                         --taps-out writes): for every committed position p >= P, the 7 drafts after IDS[p+1] without a
                         threshold and their probabilities (PREFIX.drafts: int32 p, int32 [7], float [7]) and draft 1's
                         logits for the first N positions (PREFIX.logits; default 200); no decoding. Compare with
                         tests/inkling2_loop/draft_ref_mlx_run.py + draft_compare.py
  --early N              with --draft: before each step, name the released picks recorded at the lookup match's source
                         positions for the step's own positions and N positions past them, so their reads start before
                         the step (0 = off, the default); names only schedule reads, the ids do not change
  --verify-names 0|1     with --draft and an early-address hook: inside a verify pass, the hook names every position's
                         experts of later steps as one-position steps do (routers: the first two blocks per position at
                         the pass's start, the lag rows in each step's router fork-join; oracle: the recorded picks); names
                         only schedule reads (default 1)
  --verify-old 0|1       1: the previous verify pass (recurrent states copied at the pass's start and streamed once per
                         position, the packed gemm from 6 positions, the full head), for A/B measurements (default 0)
  --copy-free 0|1        with --draft lookup: copy drafts are not rationed by the probe interval and budget once their
                         drafts are being kept (default 1)
  --draft-cost F         with --draft: the pass costs a run measured, read at the start in place of the family's prior when
                         F exists, and this run's written to F at the end (a line "draft_cost FAMILY r1 .. r8")
  --draft-gate 0|1       with --draft lookup: the controller's cost model from per-pass counters per pass width (compute as a
                         ratio to a plain step's, wait for reads, urgent MB, the next step's excess wait): a draft runs when
                         its expected tokens per estimated ms beat a plain step's by --draft-margin (default 0.05)
                         (default 1; 0 = the cost ratio table alone)
  --draft-light F        with --draft: a light draft head trained outside (draft.h LightHeadSource; jobs/babel/q36_draft_head.py's
                         heads_*.safetensors) as one more source; --draft-light-q8 0|1 (default 1: the engine's 8-bit
                         format), --draft-light-thr P (offer drafts while their confidence >= P, default 0),
                         --draft-light-dump F (every run's input and drafts, for the reference check),
                         --draft-light-backoff 0 (run it every step: acceptance measurements)
  --skeleton F           with --draft: tool-call skeleton fragments (one line of token ids each) as a second draft source
  --probe-every N        with --draft lookup: one short draft after N declined offers (default 48)
  --verify-check N       after the prompt: N greedy single steps, then the same positions again through verify passes of
                         2-8 positions (every third with a wrong last draft, rolled back); every kept position's logits and
                         the final state must be bit-identical (prints verify_check in the JSON; exit 3 if not)
)";

struct Args {
  std::string model, resident, embeddings, family, prompt_ids, prompt_file, teacher, dump_logits, record, oracle, routers, out_ids, json,
      steps_out, force, pool_share, nll_out, ref_nll, qual_out, ref_qual, dump_names, taps_out, plan_routers, evict_trace;
  std::string prefetch = "none", mem = "mlock", res_mem = "mlock", prec = "i16", lead = "4", mode = "lossless", evict = "auto",
              plan_calib = "rank", late, evict_policy, router_heavy, layer_caps;
  int reader_io_low = -1;  // -1: CoreOpts' platform default
  int n = 120, start_pos = 0, ctx = 16384, threads = 8, readers = 4, split = 1, m = 8, top_k = 0, dump_from = 0, stop_eos = -1, drop = -1;
  int rt_cand = 16, router_ecache = 0, fast_head = -1, run_ahead = 0, run_ahead_depth = 1, ra_max = 0, ra_sat = 0, rd_prio = -1, rd_excl = 0, spec_max = 0, cancel = -1, bg = -1, name_gate = 0;
  int plan_budget = 1, plan_revoke = 1, plan_maxcand = 32;
  int cold_floor = -1;  // approximate mode's cold-cache floor (Approx::cold_floor)
  int bg_ranks = 0, bg_gate = 0, bg_split = 2, bg_queue = 2, bg_max = 1;
  double bg_margin = 0;  // background reads (the profile's; --bg-ranks 0 = off)
  int prefill_chunk = 1, draft_max = 7, probe_every = 48, early = 0, verify_names = 1, draft_gate = 1, verify_old = 0, copy_free = 1;
  std::string draft_cost;
  double draft_margin = 0.05;
  double prefill_gpu_frac = 0, prefill_gpu_experts = 0;
  std::string xtok, xtok_bg, skeleton, draft_light, draft_light_dump;
  int draft_light_q8 = 1, draft_light_backoff = 1;
  double draft_light_thr = 0;
  int verify_check = 0;
  double xtok_tau = 0.5;
  std::string draft = "none";  // none | lookup | head | both (controller) | fixed (the whole lookup draft every step)
  double draft_thr = 0.9;
  int backoff0 = 32, backoff_max = 256, draft_exact = 0, draft_q8 = 1, check_start = 0, check_logits = 200;
  std::string draft_check, check_out;
  int draft_bench = 0;
  int file_check = 1, file_hint = 0, file_skip_resident = 1, file_prefault = 1, file_adapt = 0;
  double file_min_gb = 1.0, file_core_gb = 0;
  double ra_budget = 0;
  double cache_start_gb = 0;
  double temp = 0, top_p = 1.0, cache_gb = 5.0, half_life = 8.0, rt_tau = -1, read_w = -1, read_mass = -1;
  double plan_u = 0.9, plan_tau_long = 0.5, plan_tau = -1, plan_pin_p = 0.2, plan_revoke_p = 0.1;
  uint64_t seed = 0;
  bool skip_reads = false, router_stats = false, loop_hold = true, state_check = false;
  // the user-facing surface (include/dmeng/presets.h)
  std::string profile = "auto", prompt_text, reasoning_effort;
  int drafts = -1;
  bool cache_auto = false, print_defaults = false, chat = false;
  MemRule mr;
  std::vector<std::pair<std::string, std::string>> model_opts;  // --model-opt NAME=VALUE (Model::set_option, then the engine's)
  std::set<std::string> given;  // the flags on the command line (a profile default fills only the others)
  Given g;                      // the run settings given (presets.h run_flag), resolved by resolve_run
};

// --prompt TEXT: text in, text out through python/run.py of this repo (the tokenizer and chat template are Python's);
// it calls this binary back with --prompt-ids and every other flag unchanged
static int run_text(const Args& a, int argc, char** argv) {
  char self[PATH_MAX];
  uint32_t sz = sizeof self;
  std::string exe = _NSGetExecutablePath(self, &sz) == 0 ? std::string(self) : std::string(argv[0]);
  char real[PATH_MAX];
  if (realpath(exe.c_str(), real)) exe = real;
  const std::string dir = exe.substr(0, exe.find_last_of('/'));
  std::string py = getenv("DMENG_RUN_PY") ? getenv("DMENG_RUN_PY") : "";
  for (const std::string& c : std::vector<std::string>{dir + "/../python/run.py", dir + "/python/run.py"})
    if (py.empty() && access(c.c_str(), R_OK) == 0) py = c;
  if (py.empty()) {
    fprintf(stderr, "--prompt: python/run.py not found beside %s (set DMENG_RUN_PY, or pass token ids with --prompt-file)\n", exe.c_str());
    return 2;
  }
  const char* python = getenv("DMENG_PYTHON") ? getenv("DMENG_PYTHON") : "python3";
  std::vector<std::string> v = {python, py, "--model", a.model, "--bin", exe, "--prompt", a.prompt_text};
  if (a.chat) v.push_back("--chat");
  if (!a.reasoning_effort.empty()) v.insert(v.end(), {"--reasoning-effort", a.reasoning_effort});
  for (int i = 1; i < argc; i++) {
    const std::string k = argv[i];
    if (k == "--prompt" || k == "--model" || k == "--reasoning-effort") i++;
    else if (k != "--chat") v.push_back(k);
  }
  std::vector<char*> cv;
  for (auto& x : v) cv.push_back((char*)x.c_str());
  cv.push_back(nullptr);
  execvp(python, cv.data());
  fprintf(stderr, "--prompt: cannot run %s (%s)\n", python, strerror(errno));
  return 2;
}

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::string t;
  for (char ch : s + ",") {
    if (ch == ',' || ch == ' ' || ch == '\n' || ch == '\t') {
      if (!t.empty()) v.push_back(std::stoi(t));
      t.clear();
    } else {
      t += ch;
    }
  }
  return v;
}

static std::vector<std::string> split_list(const std::string& s) {
  std::vector<std::string> v;
  std::string t;
  for (char ch : s + ",") {
    if (ch == ',') {
      if (!t.empty()) v.push_back(t);
      t.clear();
    } else {
      t += ch;
    }
  }
  return v;
}

static std::string slurp(const std::string& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct MemStat {
  double footprint_gb, footprint_peak_gb, compressed_gb, resident_gb;
};
static MemStat memstat() {
  task_vm_info_data_t vi;
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  MemStat m{0, 0, 0, 0};
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &cnt) == KERN_SUCCESS) {
    m.footprint_gb = vi.phys_footprint / 1e9;
    m.footprint_peak_gb = vi.ledger_phys_footprint_peak / 1e9;
    m.compressed_gb = vi.compressed / 1e9;
    m.resident_gb = vi.resident_size / 1e9;
  }
  return m;
}

// per position of a teacher-forced text: argmax, NLL of the true next token, this run's top 32 (ids, log-probs), and the
// log-probs of a reference run's top 32 ids under this run (--ref-qual)
constexpr int TK = 32;
struct QRec {
  int32_t argmax;
  float nll;
  int32_t ids[TK];
  float lp[TK];
  float ref_lp[TK];
};

static int die(const std::string& what, const std::string& err) {
  fprintf(stderr, "%s: %s\n", what.c_str(), err.c_str());
  return 1;
}

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto v = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", k.c_str());
        exit(2);
      }
      return argv[++i];
    };
    a.given.insert(k);
    if (const std::string rf = run_flag_name(k); !rf.empty()) {  // a run setting: one spelling in every front end (presets.h)
      std::string e;
      if (!run_flag(k, v(), a.g, e)) return die("flags", e);
      a.given.insert(rf);
      continue;
    }
    if (k == "--help" || k == "-h") {
      printf("%s", HELP_SHORT);
      return 0;
    } else if (k == "--help-advanced") {
      std::string h = HELP;
      h.replace(h.find("PRESETS"), 7, Approx::presets_help());
      printf("%s", h.c_str());
      return 0;
    } else if (k == "--model") a.model = v();
    else if (k == "--embeddings") a.embeddings = v();
    else if (k == "--family") a.family = v();
    else if (k == "--prompt-ids") a.prompt_ids = v();
    else if (k == "--prompt-file") a.prompt_file = v();
    else if (k == "--teacher") a.teacher = v();
    else if (k == "--dump-logits") a.dump_logits = v();
    else if (k == "--dump-from") a.dump_from = std::stoi(v());
    else if (k == "--record") a.record = v();
    else if (k == "--oracle") a.oracle = v();
    else if (k == "--router-stats") a.router_stats = true;
    else if (k == "--out-ids") a.out_ids = v();
    else if (k == "--json") a.json = v();
    else if (k == "--steps-out") a.steps_out = v();
    else if (k == "--prefetch") a.prefetch = v();
    else if (k == "--lead") a.lead = v();
    else if (k == "--router-ecache") a.router_ecache = std::stoi(v());
    else if (k == "--mem") a.mem = v();
    else if (k == "--res-mem") a.res_mem = v();
    else if (k == "--prec") a.prec = v();
    else if (k == "--n") a.n = std::stoi(v());
    else if (k == "--ctx") a.ctx = std::stoi(v());
    else if (k == "--start-pos") a.start_pos = std::stoi(v());
    else if (k == "--threads") a.threads = std::stoi(v());
    else if (k == "--split") a.split = std::stoi(v());
    else if (k == "--reader-io-low") a.reader_io_low = std::stoi(v());
    else if (k == "--temp") a.temp = std::stod(v());
    else if (k == "--top-k") a.top_k = std::stoi(v());
    else if (k == "--top-p") a.top_p = std::stod(v());
    else if (k == "--seed") a.seed = std::stoull(v());
    else if (k == "--cache-start-gb") a.cache_start_gb = std::stod(v());
    else if (k == "--profile") a.profile = v();
    else if (k == "--model-opt") {
      const std::string kv = v();
      const size_t e = kv.find('=');
      a.model_opts.push_back({kv.substr(0, e), e == std::string::npos ? "1" : kv.substr(e + 1)});
    }
    else if (k == "--prompt") a.prompt_text = v();
    else if (k == "--chat") a.chat = true;
    else if (k == "--reasoning-effort") a.reasoning_effort = v();
    else if (k == "--reserve-gb") a.mr.reserve_gb = std::stod(v());
    else if (k == "--print-defaults") a.print_defaults = true;
    else if (k == "--pool-share") a.pool_share = v();
    else if (k == "--half-life") a.half_life = std::stod(v());
    else if (k == "--stop-eos") a.stop_eos = std::stoi(v());
    else if (k == "--skip-reads") a.skip_reads = true;
    else if (k == "--no-loop-hold") a.loop_hold = false;
    else if (k == "--force-routing") a.force = v();
    else if (k == "--fast-head") a.fast_head = std::stoi(v());
    else if (k == "--late") a.late = v();
    else if (k == "--read-w") a.read_w = std::stod(v());
    else if (k == "--read-mass") a.read_mass = std::stod(v());
    else if (k == "--bg") a.bg = std::stoi(v());
    else if (k == "--evict-policy") a.evict_policy = v();
    else if (k == "--evict-trace") a.evict_trace = v();
    else if (k == "--drop") a.drop = std::stoi(v());
    else if (k == "--cold-floor") a.cold_floor = std::stoi(v());
    else if (k == "--router-heavy") a.router_heavy = v();
    else if (k == "--rd-excl") a.rd_excl = std::stoi(v());
    else if (k == "--name-gate") a.name_gate = std::stoi(v());
    else if (k == "--spec-max") a.spec_max = std::stoi(v());
    else if (k == "--bg-split") a.bg_split = std::stoi(v());
    else if (k == "--bg-queue") a.bg_queue = std::stoi(v());
    else if (k == "--bg-max") a.bg_max = std::stoi(v());
    else if (k == "--bg-margin") a.bg_margin = std::stod(v());
    else if (k == "--cancel") a.cancel = std::stoi(v());
    else if (k == "--plan-routers") a.plan_routers = v();
    else if (k == "--plan-u") a.plan_u = std::stod(v());
    else if (k == "--plan-budget") a.plan_budget = std::stoi(v());
    else if (k == "--plan-tau-long") a.plan_tau_long = std::stod(v());
    else if (k == "--plan-tau") a.plan_tau = std::stod(v());
    else if (k == "--plan-pin-p") a.plan_pin_p = std::stod(v());
    else if (k == "--plan-revoke") a.plan_revoke = std::stoi(v());
    else if (k == "--plan-revoke-p") a.plan_revoke_p = std::stod(v());
    else if (k == "--plan-maxcand") a.plan_maxcand = std::stoi(v());
    else if (k == "--plan-calib") a.plan_calib = v();
    else if (k == "--nll-out") a.nll_out = v();
    else if (k == "--ref-nll") a.ref_nll = v();
    else if (k == "--qual-out") a.qual_out = v();
    else if (k == "--ref-qual") a.ref_qual = v();
    else if (k == "--dump-names") a.dump_names = v();
    else if (k == "--taps-out") a.taps_out = v();
    else if (k == "--state-check") a.state_check = true;
    else if (k == "--xtok-bg") a.xtok_bg = v();
    else if (k == "--draft-max") a.draft_max = std::stoi(v());
    else if (k == "--draft-thr") a.draft_thr = std::stod(v());
    else if (k == "--draft-exact") a.draft_exact = std::stoi(v());
    else if (k == "--draft-q8") a.draft_q8 = std::stoi(v());
    else if (k == "--draft-check") a.draft_check = v();
    else if (k == "--draft-bench") a.draft_bench = std::stoi(v());
    else if (k == "--draft-check-start") a.check_start = std::stoi(v());
    else if (k == "--draft-check-out") a.check_out = v();
    else if (k == "--draft-check-logits") a.check_logits = std::stoi(v());
    else if (k == "--head-backoff") {
      auto b = split_list(v());
      a.backoff0 = std::stoi(b.at(0));
      a.backoff_max = b.size() > 1 ? std::stoi(b[1]) : a.backoff0;
    }
    else if (k == "--probe-every") a.probe_every = std::stoi(v());
    else if (k == "--verify-names") a.verify_names = std::stoi(v());
    else if (k == "--draft-gate") a.draft_gate = std::stoi(v());
    else if (k == "--verify-old") a.verify_old = std::stoi(v());
    else if (k == "--copy-free") a.copy_free = std::stoi(v());
    else if (k == "--draft-cost") a.draft_cost = v();
    else if (k == "--draft-margin") a.draft_margin = std::stod(v());
    else if (k == "--skeleton") a.skeleton = v();
    else if (k == "--draft-light") a.draft_light = v();
    else if (k == "--draft-light-q8") a.draft_light_q8 = std::stoi(v());
    else if (k == "--draft-light-thr") a.draft_light_thr = std::stod(v());
    else if (k == "--draft-light-dump") a.draft_light_dump = v();
    else if (k == "--draft-light-backoff") a.draft_light_backoff = std::stoi(v());
    else if (k == "--verify-check") a.verify_check = std::stoi(v());
    else if (k == "--file-check") a.file_check = std::stoi(v());
    else if (k == "--file-hint") a.file_hint = std::stoi(v());
    else if (k == "--file-skip-resident") a.file_skip_resident = std::stoi(v());
    else if (k == "--file-prefault") a.file_prefault = std::stoi(v());
    else if (k == "--file-adapt") a.file_adapt = std::stoi(v());
    else if (k == "--file-min-gb") a.file_min_gb = std::stod(v());
    else if (k == "--file-core-gb") a.file_core_gb = std::stod(v());
    else if (k == "--draft-n") {
      fprintf(stderr, "--draft-n: the draft length is --draft-max N (with --draft lookup|fixed)\n");
      return 2;
    } else {
      fprintf(stderr, "unknown flag %s (dmeng --help)\n", k.c_str());
      return 2;
    }
  }
  if (a.draft_bench > 0) {  // the draft head's bf16 products on synthetic weights (no model)
    Pool pool(a.threads);
    printf("%s\n", ink_draft_bench(pool, a.draft_bench).c_str());
    return 0;
  }
  if (a.model.empty()) {
    fprintf(stderr, "usage: dmeng --model DIR (--prompt-ids IDS | --prompt-file F | --teacher F) [options]  (dmeng --help)\n");
    return 2;
  }
  if (!a.prompt_text.empty()) return run_text(a, argc, argv);
  std::string err;
  // ---- the settings: explicit options > the profile > the manifest's defaults > dmeng's own values (resolve_run, the
  // resolver dmserve and the C API use too)
  auto has = [&](const char* f) { return a.given.count(f) > 0; };
  Given g = a.g;
  g.other_address = has("--prefetch") || has("--oracle") || has("--plan-routers");
  g.recording = !a.record.empty();
  RunOpts ro;  // dmeng's own values where nothing else sets one: RunOpts' (single steps for the prompt, a 5 GB cache)
  if (!resolve_run(a.model, a.profile, g, "", -1, ro, err)) return die("profile", err);
  const Preset& P = ro.preset;
  const bool profiled = ro.profiled;
  for (auto& n : ro.notes) fprintf(stderr, "dmeng: %s\n", n.c_str());
  if (a.print_defaults) {
    printf("%s\n", profiled ? P.json().c_str() : "null");
    return 0;
  }
  a.mode = ro.mode;
  a.resident = ro.resident;
  a.routers = ro.routers;
  a.xtok = ro.xtok;
  if (ro.routers_from_profile || (has("--routers") && !has("--prefetch") && !ro.routers.empty())) a.prefetch = "router";  // named routers name
  a.m = ro.m, a.rt_tau = ro.tau, a.rt_cand = ro.cand, a.xtok_tau = ro.xtok_tau, a.rd_prio = ro.rd_prio;
  a.evict = ro.evict, a.layer_caps = ro.layer_caps;
  a.cache_auto = ro.cache != "fixed", a.cache_gb = ro.cache_gb;  // (dmeng is a macOS tool: slab, the iPhone rule, is the automatic one here)
  if (!set_start_rule(a.mr, ro.cache_start, err)) return die("cache", err);
  a.prefill_gpu_frac = ro.gpu_frac, a.prefill_gpu_experts = ro.gpu_experts, a.prefill_chunk = ro.prefill_chunk;
  a.run_ahead = ro.run_ahead, a.run_ahead_depth = ro.run_ahead_depth, a.ra_budget = ro.ra_budget, a.ra_max = ro.ra_max_tok, a.ra_sat = ro.ra_sat_skip;
  a.readers = ro.readers, a.bg_ranks = ro.bg_ranks, a.bg_gate = ro.bg_gate, a.drafts = ro.drafts, a.draft = ro.draft, a.early = ro.early;
  LoadOpts lo;
  std::string family;
  if (!resolve_model(a.model, a.resident, mem_of(a.res_mem), lo, family, err, "", a.embeddings, profiled ? P.profile : ""))
    return die("model", err);
  if (!a.family.empty()) family = a.family;
  CoreOpts co;
  co.threads = a.threads;
  co.readers = a.readers;
  co.split = a.split;
  if (a.reader_io_low >= 0) co.reader_io_low = a.reader_io_low != 0;
  co.prec = a.prec == "i8" ? Prec::I8 : a.prec == "f16" ? Prec::F16 : Prec::I16;
  co.ref = a.prec == "f32";
  co.cache_gb = a.cache_gb;
  co.cache_start_gb = a.cache_start_gb;
  co.half_life = a.half_life;
  co.mem = mem_of(a.mem);
  co.skip_reads = a.skip_reads;
  co.loop_hold = a.loop_hold;
  co.evict = a.evict;
  co.layer_caps = a.layer_caps.empty() ? "auto" : a.layer_caps;
  if (!a.evict_policy.empty() && a.evict == "auto" && !Cache::is_ranker_file(a.evict_policy)) co.evict = "freq";  // a keep-score file replaces the default
  for (auto& s : split_list(a.pool_share)) co.pool_share.push_back(std::stod(s));

  set_qos_interactive();
  auto t_load = clk::now();
  std::unique_ptr<Model> M = make_model(family, err);
  if (!M) return die("model", err);
  if (!M->load(lo, err)) return die("load", err);
  std::string auto_why = "null";
  if (a.cache_auto) {  // the serving worker's rule (src/serving/memwatch.h), the resident weights loaded: here the start size
    const ModelInfo& mi0 = M->info();
    double start = 0;
    // the per-position state is reserved for the context (virtual) and its pages become resident as positions fill: the
    // cache plans for the positions this run holds (the prompt + the tokens to generate), not the whole context
    a.mr.kv_positions = a.ctx;
    {
      const std::string src = !a.prompt_ids.empty() ? a.prompt_ids : !a.prompt_file.empty() ? slurp(a.prompt_file) : !a.teacher.empty() ? slurp(a.teacher) : "";
      const int np = src.empty() ? -1 : (int)parse_ids(src).size();
      if (np >= 0) a.mr.kv_positions = std::min(a.ctx, np + std::max(0, a.n) + 64);
      struct stat rs;  // the router file is loaded after the cache is sized
      if (a.prefetch == "router" && !a.routers.empty() && stat(a.routers.c_str(), &rs) == 0) a.mr.later_gb = rs.st_size / 1e9;
    }
    a.cache_gb = auto_cache_gb(a.mr, M->pos_bytes(), 0, store_bytes_gb(a.model, mi0.layer_file, mi0.layers), start, auto_why);
    a.cache_gb = std::min(a.cache_gb, start);
    co.cache_gb = a.cache_gb;
  }
  Core C;
  if (!C.init(M->info(), co, a.model, err)) return die("core", err);
  C.cache.file_check = a.file_check;
  C.cache.file_evict_hint = a.file_hint;
  C.cache.rd.file_skip_resident = a.file_skip_resident != 0;
  C.cache.rd.file_prefault = a.file_prefault != 0;
  C.cache.fa.on = a.file_adapt != 0 && C.cache.mem == Mem::FILE;
  C.cache.fa.min_bytes = (int64_t)(a.file_min_gb * 1e9);
  if (!C.cache.core_init((int64_t)(a.file_core_gb * 1e9), err)) return die("core", err);
  const double file_res0_gb = C.cache.mem == Mem::FILE ? C.cache.file_resident_bytes() / 1e9 : 0;
  if (!M->bind(C, err)) return die("model", err);
  for (auto& kv : a.model_opts)
    if (!M->set_option(kv.first, kv.second, err)) return die("--model-opt " + kv.first, err);
  std::unique_ptr<State> S = M->new_state(a.ctx, err);
  if (!S) return die("state", err);
  const ModelInfo& mi = M->info();
  const int VO = mi.vocab_out;
  const double load_s = ms_since(t_load) / 1000;

  // ---- the early address
  Oracle orc;
  Routers rts, hrts;
  Transplant tpl;
  Planner pln;
  const bool plan = a.prefetch == "plan";
  if (a.prefetch == "oracle") {
    if (!orc.load(a.oracle, C.steps, C.K, err)) return die("oracle", err);
    orc.token_lead = a.lead == "token";
    orc.lead = orc.token_lead ? C.steps : std::stoi(a.lead);
    C.hook = &orc;
  } else if (a.prefetch == "router" || a.router_stats) {
    rts.ecap = a.router_ecache;
    if (!rts.load(a.routers, C, err)) return die("routers", err);
    rts.m = a.m;
    rts.tau = a.rt_tau;
    rts.cand = a.rt_cand;
    rts.active = a.prefetch == "router";
    if (!a.router_heavy.empty()) {  // a heavy-pick head beside the router (names nothing itself)
      if (!hrts.load(a.router_heavy, C, err)) return die("heavy head", err);
      hrts.active = false;
      rts.heavy = &hrts;
    }
    C.hook = &rts;
  } else if (a.prefetch == "transplant") {
    tpl.m = a.m;
    C.hook = &tpl;
  } else if (plan) {
    if (!pln.load(split_list(a.plan_routers), C, err)) return die("planner", err);
    pln.u = a.plan_u;
    pln.budget_on = a.plan_budget != 0;
    pln.budget_mode = a.plan_budget == 2 ? 2 : 1;
    pln.tau_long = a.plan_tau_long;
    pln.tau_fixed = a.plan_tau;
    pln.pin_p = a.plan_pin_p;
    pln.revoke_on = a.plan_revoke != 0;
    pln.revoke_p = a.plan_revoke_p;
    pln.maxcand = a.plan_maxcand;
    pln.calib_q_only = a.plan_calib == "q";
    C.hook = &pln;
  } else if (a.prefetch != "none") {
    fprintf(stderr, "--prefetch none|oracle|router|transplant|plan\n");
    return 2;
  }

  Xtok xt;  // cross-token routers, wrapping the hook chosen above (if any)
  if (!a.xtok.empty()) {
    if (!xt.load(a.xtok, C, err)) return die("xtok", err);
    xt.tau = a.xtok_tau;
    xt.inner = C.hook;
    C.hook = &xt;
  }
  Xtok xb;  // a second cross-token source whose names use the readers' background class
  if (!a.xtok_bg.empty()) {
    if (!xb.load(a.xtok_bg, C, err)) return die("xtok-bg", err);
    xb.tau = a.xtok_tau;
    xb.background = true;
    xb.inner = C.hook;
    C.hook = &xb;
  }

  // ---- the mode
  if (a.mode != "lossless") {
    const std::string preset = a.mode == "approx" ? "" : a.mode.rfind("approx:", 0) == 0 ? a.mode.substr(7) : "?";
    if (!C.ns.preset(preset)) {
      fprintf(stderr, "--mode lossless|approx[:PRESET]; presets: %s\n", Approx::presets_help());
      return 2;
    }
  }
  if (!a.late.empty()) C.ns.late_ms = a.late == "none" ? -1 : a.late == "always" ? 1000 : std::stod(a.late);
  if (a.read_w >= 0) C.ns.read_w = a.read_w;
  if (a.read_mass >= 0) C.ns.read_mass = a.read_mass;
  if (a.bg >= 0) C.ns.bg = a.bg != 0;
  if (a.drop >= 0) C.ns.drop = a.drop != 0;
  if (a.cold_floor != -1) C.ns.cold_floor = a.cold_floor;  // --cold-floor N: the mode string's floor=N (every front end)
  C.cache.rd.prio = a.rd_prio < 0 ? (plan || (C.ns.on && C.ns.prio)) : a.rd_prio != 0;
  C.cache.rd.excl = a.rd_excl != 0;
  C.cache.name_gate = a.name_gate != 0;
  if (a.spec_max > 0) C.cache.rd.max_spec = a.spec_max;
  if (a.bg_ranks > 0 && C.hook == &rts) {  // background reads: the routers' next ranks for the drive's idle time
    rts.bg_ranks = a.bg_ranks;
    rts.bg_queue = std::max(1, a.bg_queue);
    rts.bg_margin = a.bg_margin;
    C.cache.rd.bg_spec_gate = std::max(0, a.bg_gate);
    C.cache.bg_split = std::max(1, a.bg_split);
    C.cache.rd.max_bg = std::max(1, a.bg_max);
  }
  C.cancel_stale = a.cancel < 0 ? (plan || (C.ns.on && C.ns.cancel)) : a.cancel != 0;
  C.chunk_names = a.verify_names != 0;
  if (!a.evict_policy.empty()) {  // the learned keep-score (router features from --prefetch router), or a ranker
    std::string perr;
    const bool hz = C.cache.hazard_evict;
    if (!C.cache.load_policy(a.evict_policy, perr)) {
      fprintf(stderr, "%s\n", perr.c_str());
      return 1;
    }
    if (hz && C.cache.learned_evict) {
      fprintf(stderr, "--evict hazard and a keep-score --evict-policy are two keep-scores: pick one\n");
      return 1;
    }
    if (C.cache.rk.rt) {  // the ranker's early-router rows
      if (C.hook != &rts || rts.lay.empty()) {
        fprintf(stderr, "%s reads an early router (%s): --routers F with --prefetch router or --router-stats\n",
                a.evict_policy.c_str(), C.cache.rk.router.c_str());
        return 1;
      }
      if (C.cache.rk.router_lag >= 0 && rts.lag != C.cache.rk.router_lag) {
        fprintf(stderr, "%s was trained with a lag-%d router, --routers has lag %d\n", a.evict_policy.c_str(),
                C.cache.rk.router_lag, rts.lag);
        return 1;
      }
      rts.feed = true;
    }
  }
  if (!a.evict_trace.empty()) {
    if (C.hook == &rts) rts.feed = true;
    C.cache.trace_init(0);
  }

  std::vector<int16_t> forced;
  if (!a.force.empty()) {
    FILE* f = fopen(a.force.c_str(), "rb");
    int32_t hdr[3];
    if (!f || fread(hdr, 4, 3, f) != 3 || hdr[1] != C.steps || hdr[2] != C.K) return die(a.force, "not a routing record of this model");
    forced.resize((size_t)hdr[0] * C.steps * C.K);
    if (fread(forced.data(), 2, forced.size(), f) != forced.size()) return die(a.force, "short routing record");
    fclose(f);
    C.force = forced.data();
    C.force_n = hdr[0];
  }
  // the screened head: greedy (temperature 0), or a top-k sampler (top-k 1..64: only the exact top-k logits are needed)
  // the screened head: on by default (Qwen3.6) where it is exact and nothing needs every logit: greedy decoding without
  // drafting or the verify check, or top-k sampling with k <= 64; --fast-head 0 = the full head, 1 / 2 asked explicitly
  // (2 also checks every token against the full head)
  const bool fh_auto = a.fast_head < 0;
  if (fh_auto) a.fast_head = 1;  // every family's own set_option says whether it can (a model without one keeps the full head)
  // (drafting too: plain steps and verify passes then screen the same way; the verify check compares the full logits
  // unless --fast-head is given, then the screened ones)
  const bool fh_greedy_ok = a.temp <= 0 && a.draft_check.empty() && (a.verify_check <= 0 || !fh_auto);
  bool head_topk = a.fast_head > 0 && a.temp > 0 && a.top_k > 0 && a.top_k <= 64;
  if (a.fast_head > 0 && (fh_greedy_ok || head_topk) && a.teacher.empty()) {
    std::string e2;
    const bool ok = M->set_option("fast_head", std::to_string(a.fast_head), e2) &&
                    (!head_topk || M->set_option("head_topk", std::to_string(a.top_k), e2));
    if (!ok && !fh_auto) return die("model", e2);
    if (!ok) a.fast_head = 0, head_topk = false;  // (default on: the full head where it cannot run, e.g. a mapped resident)
  } else if (fh_auto) {
    a.fast_head = 0;
  }
  if (!a.taps_out.empty() && !M->set_option("taps", "1", err)) return die("model", err);
  if (a.run_ahead > 0 && !M->set_option("run_ahead", std::to_string(a.run_ahead), err)) return die("model", err);
  if (a.verify_old && !M->set_option("verify_old", "1", err)) return die("model", err);
  C.ra_budget = a.ra_budget;
  C.ra_max_tok = a.ra_max;
  C.ra_sat_skip = a.ra_sat != 0;
  if (a.run_ahead > 0 && a.run_ahead_depth > 1 && !M->set_option("run_ahead_depth", std::to_string(a.run_ahead_depth), err)) return die("model", err);
  if (a.start_pos > 0 && !M->seek(*S, a.start_pos)) return die("model", "--start-pos is not supported by this model");
  const bool use_lookup = a.draft == "lookup" || a.draft == "fixed" || a.draft == "both";
  const bool use_head = a.draft == "head" || a.draft == "both" || !a.draft_check.empty();
  if (a.draft != "none" && !use_lookup && !use_head) return die("draft", "--draft lookup|fixed|head|both");
  if ((a.draft != "none" || a.verify_check > 0) && !mi.has_verify)  // never silently plain decoding
    return die("draft", "drafting is not supported for this family (" + family + "): the model has no multi-position verify pass");
  double head_load_s = 0;
  if (use_head) {  // before the prompt: the head's features are taken from every committed position
    auto th = clk::now();
    if (!M->set_option("draft_ctx", std::to_string(a.ctx), err) || !M->set_option("draft_exact", std::to_string(a.draft_exact), err) ||
        !M->set_option("draft_q8", std::to_string(a.draft_q8), err) ||
        !M->set_option("draft_head", "1", err))
      return die("draft head", err);
    head_load_s = ms_since(th) / 1000;
  }

  FILE* recf = nullptr;
  int nrec = 0;
  if (!a.record.empty()) {
    recf = fopen(a.record.c_str(), "wb");
    int32_t hdr[3] = {0, C.steps, C.K};
    fwrite(hdr, 4, 3, recf);
  }
  auto record = [&]() {  // the released routing of the last step, [steps, K]
    if (!recf) return;
    std::vector<int16_t> r(C.picks.begin(), C.picks.end());
    fwrite(r.data(), 2, r.size(), recf);
    nrec++;
  };
  FILE* namesf = a.dump_names.empty() ? nullptr : fopen(a.dump_names.c_str(), "wb");
  auto dump_names = [&]() {  // per step: the routers' top-16 names per step, int16 [steps, 16]
    if (!namesf || !C.hook || !C.hook->names()) return;
    std::vector<int16_t> r(C.hook->names(), C.hook->names() + (size_t)C.steps * 16);
    fwrite(r.data(), 2, r.size(), namesf);
  };

  std::ostringstream js;
  js.setf(std::ios::fixed);
  js.precision(4);
  const MemStat m0 = memstat();
  auto& CS = C.cache.st;
  auto& RD = C.cache.rd;
  uint64_t dt0[7] = {0};
  uint64_t rd_busy0 = 0, rd_wall0 = 0, rd_bytes0 = 0, rd_bg0 = 0, q0[3] = {0, 0, 0}, s0[3] = {0, 0, 0}, j0[3] = {0, 0, 0}, sau0 = 0, fskip0 = 0, fskipb0 = 0;
  double fe_ms = 0;  // the front end's per-token work outside the model's step (sampling, records): stage "sample"
  auto reset_stats = [&]() {
    fe_ms = 0;
    C.reset_stats();
    M->reset_stats();
    rd_busy0 = RD.busy_ns.load();
    rd_wall0 = RD.busy_wall();
    RD.depth_times(dt0);
    rd_bytes0 = RD.bytes.load();
    rd_bg0 = RD.bytes_cls[2].load();
    sau0 = RD.spec_inflight_at_urgent.load();
    fskip0 = RD.file_resident_skips.load();
    fskipb0 = RD.file_bytes_skipped.load();
    for (int i = 0; i < 3; i++) {
      q0[i] = RD.q_ns[i].load();
      s0[i] = RD.svc_ns[i].load();
      j0[i] = RD.njobs[i].load();
    }
  };
  // statistics of a timed loop (decode, or the whole teacher-forced text)
  auto emit_stats = [&](int steps, double dec_s, const std::vector<double>& step_ms) {
    const MemStat m1 = memstat();
    std::vector<double> sm = step_ms;
    std::sort(sm.begin(), sm.end());
    const double med = sm.empty() ? 0 : sm[sm.size() / 2];
    double tot_ms = 0;
    for (double v : step_ms) tot_ms += v;
    auto per = [&](double v) { return v / std::max(1, steps); };
    const Times& T = C.t;
    const uint64_t picks = CS.n_hit + CS.n_late + CS.n_miss;
    const double comp = tot_ms - T.wait - T.ra;  // compute: the token's wall time less the wait on reads and the run-ahead
    js << ",\"decode_steps\":" << steps << ",\"decode_s\":" << dec_s << ",\"tok_s\":" << steps / std::max(1e-9, dec_s)
       << ",\"ms_per_token\":" << per(tot_ms) << ",\"ms_per_token_median\":" << med << ",\"compute_ms_per_token\":" << per(comp)
       << ",\"wait_ms_per_token\":" << per(T.wait) << ",\"policy_ms_per_token\":" << per(CS.policy_ms)
       << ",\"by_stage_ms_per_token\":{" << T.stages_json(std::max(1, steps)) << ",\"sample\":" << per(fe_ms) << "}"
       << ",\"weight_mb_touched_per_token\":{\"resident\":" << per(T.bytes_res / 1e6) << ",\"experts\":" << per(T.bytes_exp / 1e6) << "}"
       << ",\"mb_read_per_token\":" << per((CS.bytes_urgent + CS.bytes_pf) / 1e6) << ",\"mb_read_urgent_per_token\":" << per(CS.bytes_urgent / 1e6)
       << ",\"mb_read_prefetch_per_token\":" << per(CS.bytes_pf / 1e6)
       << [&] {  // the drive per token: wall time with no read in flight / exactly one; bytes over the loop's wall time
            uint64_t d1[7];
            RD.depth_times(d1);
            const double rb = (double)(RD.bytes.load() - rd_bytes0);
            std::ostringstream o;
            o.setf(std::ios::fixed);
            o.precision(4);
            o << ",\"drive\":{\"idle_ms_per_token\":" << per((d1[0] - dt0[0] + d1[1] - dt0[1]) / 1e6)
              << ",\"idle_nothing_queued_ms_per_token\":" << per((d1[0] - dt0[0]) / 1e6)
              << ",\"one_read_ms_per_token\":" << per((d1[2] - dt0[2]) / 1e6) << ",\"effective_gb_s\":" << rb / 1e9 / std::max(1e-9, dec_s)
              << ",\"gb_s_while_busy\":" << rb / 1e9 / std::max(1e-9, (RD.busy_wall() - rd_wall0) / 1e9)
              << ",\"mb_background_per_token\":" << per((RD.bytes_cls[2].load() - rd_bg0) / 1e6)
              << ",\"background\":{\"ranks\":" << rts.bg_ranks << ",\"gate\":" << RD.bg_spec_gate << ",\"split\":" << C.cache.bg_split
              << ",\"issued_per_token\":" << per(CS.n_bg_issued) << ",\"refused_per_token\":" << per(CS.n_bg_refused)
              << ",\"used_per_token\":" << per(CS.n_bg_used) << ",\"evicted_unused_per_token\":" << per(CS.n_bg_unused_evicted) << "}}";
            return o.str();
          }()
       << ",\"picks\":" << picks << ",\"hit_rate\":" << (double)CS.n_hit / std::max<uint64_t>(1, picks)
       << ",\"late_rate\":" << (double)CS.n_late / std::max<uint64_t>(1, picks) << ",\"miss_rate\":" << (double)CS.n_miss / std::max<uint64_t>(1, picks)
       << ",\"hit_rate_looped\":" << (double)CS.n_hit_r[0] / std::max<uint64_t>(1, CS.n_pick_r[0])
       << ",\"hit_rate_other\":" << (double)CS.n_hit_r[1] / std::max<uint64_t>(1, CS.n_pick_r[1])
       << ",\"looped_picks_used_by_an_earlier_pass\":" << (double)CS.n_reuse_hit / std::max<uint64_t>(1, CS.n_pick_r[0])
       << ",\"cache_lock\":" << C.cache.lock_json() << ",\"pf_issued_per_token\":" << per(CS.n_pf_issued) << ",\"pf_dropped_per_token\":" << per(CS.n_pf_dropped) << ",\"pf_gated_per_token\":" << per(CS.n_pf_gated)
       << ",\"pf_evicted_unused_per_token\":" << per(CS.n_pf_unused_evicted) << ",\"pf_stolen_per_token\":" << per(CS.n_pf_stolen)
       << ",\"purged\":" << CS.n_purged << ",\"held_at_end\":[";
    for (int p = 0; p < C.cache.npools; p++) js << (p ? "," : "") << C.cache.held(p);
    js << "],\"held_gb\":" << C.cache.held_bytes() / 1e9 << ",\"mb_preads_per_token\":" << per((RD.bytes.load() - rd_bytes0) / 1e6);
    if (C.cache.mem == Mem::FILE) {
      js << ",\"file\":{\"check\":" << C.cache.file_check << ",\"evict_hint\":" << C.cache.file_evict_hint << ",\"skip_resident\":" << RD.file_skip_resident
         << ",\"prefault\":" << RD.file_prefault << ",\"checks_per_token\":" << per(CS.file_checks) << ",\"file_refaults\":" << CS.file_refaults
         << ",\"refaults_per_token\":" << per(CS.file_refaults) << ",\"resident_skips_per_token\":" << per(RD.file_resident_skips.load() - fskip0)
         << ",\"mb_skipped_resident_per_token\":" << per((RD.file_bytes_skipped.load() - fskipb0) / 1e6)
         << ",\"file_cache_resident_gb_at_load\":" << file_res0_gb << ",\"file_cache_resident_gb_at_end\":" << C.cache.file_resident_bytes() / 1e9
         << ",\"budget_gb_at_end\":" << C.cache.budget_bytes() / 1e9 << ",\"adapt\":{\"on\":" << C.cache.fa.on << ",\"resident_fraction\":" << C.cache.fa.res_frac
         << ",\"refaults_per_token_ema\":" << C.cache.fa.refault_ema << ",\"shrinks\":" << C.cache.fa.shrinks << ",\"grows\":" << C.cache.fa.grows << ",\"trace\":[";
      for (size_t i = 0; i < C.cache.fa.trace.size(); i++) js << (i ? "," : "") << "[" << C.cache.fa.trace[i].first << "," << C.cache.fa.trace[i].second / 1e9 << "]";
      js << "]},\"core\":{\"gb\":" << C.cache.core_slab_bytes / 1e9 << ",\"records\":" << C.cache.core_held() << ",\"swapped\":" << C.cache.core_swapped << "}}";
    }
    js << ",\"footprint_gb\":" << m1.footprint_gb << ",\"footprint_peak_gb\":" << m1.footprint_peak_gb << ",\"compressed_gb\":" << m1.compressed_gb
       << ",\"resident_gb\":" << m1.resident_gb << ",\"footprint_after_load_gb\":" << m0.footprint_gb;
    {  // wait and on-the-spot misses by step group: before the loop, the looped steps, after it
      double w[3] = {0, 0, 0}, mi3[3] = {0, 0, 0};
      bool seen = false;
      for (int s = 0; s < C.steps; s++) {
        const int g = C.looped(s) ? 1 : seen ? 2 : 0;
        seen |= C.looped(s);
        w[g] += C.wait_step[s];
        mi3[g] += (double)C.miss_step[s];
      }
      js << ",\"wait_ms_by_group\":{\"before_loop\":" << per(w[0]) << ",\"looped\":" << per(w[1]) << ",\"after_loop\":" << per(w[2])
         << "},\"misses_by_group\":{\"before_loop\":" << per(mi3[0]) << ",\"looped\":" << per(mi3[1]) << ",\"after_loop\":" << per(mi3[2]) << "}";
      js << ",\"wait_ms_by_step\":[";  // per step (application), per token
      for (int s = 0; s < C.steps; s++) js << (s ? "," : "") << per(C.wait_step[s]);
      js << "],\"misses_by_step\":[";
      for (int s = 0; s < C.steps; s++) js << (s ? "," : "") << per((double)C.miss_step[s]);
      js << "]";
    }
    {
      auto dj = [&](int i) { return std::max<uint64_t>(1, RD.njobs[i].load() - j0[i]); };
      const double wall = (RD.busy_wall() - rd_wall0) / 1e6;
      js << ",\"read_queue\":{\"prio\":" << RD.prio << ",\"excl\":" << RD.excl << ",\"spec_max\":" << (a.spec_max > 0 ? a.spec_max : 0)
         << ",\"cancel_stale\":" << C.cancel_stale << ",\"evict\":\"" << a.evict << "\""
         << ",\"evict_policy\":\"" << (C.cache.rk.on ? C.cache.rk.name : C.cache.learned_evict ? "keep-score" : C.cache.hazard_evict ? "hazard" : a.evict) << "\""
         << ",\"layer_caps\":" << (C.cache.lcap.empty() ? "false" : "true")
         << ",\"ranker_choices\":" << C.cache.rk_choices << ",\"ranker_shortlist\":" << (double)C.cache.rk_cands / std::max<uint64_t>(1, C.cache.rk_choices)
         << ",\"ranker_evals_per_token\":" << per(C.cache.rk_evals)
         << ",\"wait_miss_ms_per_token\":" << per(CS.wait_ms_miss) << ",\"wait_late_ms_per_token\":" << per(CS.wait_ms_late)
         << ",\"wait_runnable_ms_per_token\":" << per(CS.wait_ms_runnable)
         << ",\"expert_spin_thread_ms_per_token\":" << per(C.exp_spin_ns.load() / 1e6)
         << ",\"expert_calls_jobs_calls_gb_s\":" << C.xc_json()
         << ",\"rerun_ab_calls_gb_s\":[" << C.rr_calls[0] << "," << C.rr_bytes[0] / std::max(1.0, C.rr_ns[0]) << "," << C.rr_calls[1] << ","
         << C.rr_bytes[1] / std::max(1.0, C.rr_ns[1]) << "," << C.rr_count.load() << "]"
         << ",\"misses_per_token\":" << per(CS.n_miss) << ",\"cap_borrow_per_token\":" << per(CS.n_cap_borrow) << ",\"late_per_token\":" << per(CS.n_late) << ",\"late_still_queued_per_token\":" << per(CS.n_late_queued)
         << ",\"miss_after_evict_per_token\":" << per(CS.n_miss_evicted) << ",\"miss_after_pf_evict_per_token\":" << per(CS.n_miss_pf_evicted)
         << ",\"pf_cancelled_per_token\":" << per(CS.n_pf_cancelled) << ",\"misses_by_planner_p_per_token\":[";
      for (int i = 0; i < 6; i++) js << (i ? "," : "") << per(CS.miss_pbin[i]);
      js << "],\"bg_jobs\":" << RD.njobs[2].load() - j0[2] << ",\"bg_promoted\":" << RD.bg_promoted.load()
         << ",\"bg_queue_ms\":" << (RD.q_ns[2].load() - q0[2]) / 1e6 / dj(2)
         << ",\"urgent_jobs\":" << RD.njobs[0].load() - j0[0] << ",\"urgent_queue_ms\":" << (RD.q_ns[0].load() - q0[0]) / 1e6 / dj(0)
         << ",\"urgent_service_ms\":" << (RD.svc_ns[0].load() - s0[0]) / 1e6 / dj(0)
         << ",\"spec_inflight_at_urgent_start\":" << (double)(RD.spec_inflight_at_urgent.load() - sau0) / dj(0)
         << ",\"spec_jobs\":" << RD.njobs[1].load() - j0[1] << ",\"spec_queue_ms\":" << (RD.q_ns[1].load() - q0[1]) / 1e6 / dj(1)
         << ",\"spec_service_ms\":" << (RD.svc_ns[1].load() - s0[1]) / 1e6 / dj(1) << ",\"drive_busy_ms_per_token\":" << per(wall)
         << ",\"drive_depth\":" << (RD.busy_ns.load() - rd_busy0) / 1e6 / std::max(1e-9, wall);
      uint64_t dt1[7];
      RD.depth_times(dt1);
      js << ",\"reader_idle_ms_per_token\":" << per((dt1[0] - dt0[0]) / 1e6) << ",\"reader_idle_queued_ms_per_token\":" << per((dt1[1] - dt0[1]) / 1e6)
         << ",\"in_flight_1_2_3_4plus_ms_per_token\":[" << per((dt1[2] - dt0[2]) / 1e6) << "," << per((dt1[3] - dt0[3]) / 1e6) << ","
         << per((dt1[4] - dt0[4]) / 1e6) << "," << per((dt1[5] - dt0[5]) / 1e6) << "],\"read_bursts_per_token\":" << per((double)(dt1[6] - dt0[6])) << "}";
    }
    {
      const Approx& N = C.ns;
      const double ns = std::max(1.0, N.steps);
      js << ",\"approx\":{\"on\":" << (N.on ? "true" : "false") << ",\"late_ms\":" << N.late_ms << ",\"read_w\":" << N.read_w
         << ",\"read_mass\":" << N.read_mass << ",\"bg\":" << (N.bg ? "true" : "false") << ",\"weight_replaced\":" << N.w_replaced / ns
         << ",\"replaced_per_token\":" << per(N.n_replaced) << ",\"steps_with_a_replacement\":" << N.steps_replaced / ns
         << ",\"max_replaced_weight\":" << N.w_replaced_max << ",\"read_now_per_token\":" << per(N.n_read_now)
         << ",\"weight_read_now\":" << N.w_read_now / ns << ",\"late_waited_per_token\":" << per(N.n_late_waited)
         << ",\"late_got_per_token\":" << per(N.n_late_got) << ",\"late_dropped_per_token\":" << per(N.n_late_dropped)
         << ",\"late_wait_ms_per_token\":" << per(N.late_wait_ms) << ",\"bg_per_token\":" << per(N.n_bg) << ",\"cold_steps\":" << N.cold
         << ",\"replaced_by_weight_per_token\":[";
      for (int i = 0; i < 6; i++) js << (i ? "," : "") << per(N.w_hist[i]);
      js << "]";
      std::vector<float> w = N.w_tok;  // released weight replaced / dropped per token (drop: left out)
      double sw = 0;
      for (float x : w) sw += x;
      std::sort(w.begin(), w.end());
      js << ",\"drop\":" << (N.drop ? "true" : "false") << ",\"weight_left_out_per_token_mean\":" << sw / std::max<double>(1, w.size())
         << ",\"weight_left_out_per_token_p90\":" << (w.empty() ? 0.0 : (double)w[std::min(w.size() - 1, (size_t)(0.9 * w.size()))])
         << ",\"steps_with_a_drop_share\":" << N.steps_replaced / ns << ",\"picks_read_and_waited_per_token\":" << per(N.n_read_now + N.n_late_waited) << "}";
    }
    if (C.hook) js << ",\"" << (plan ? "planner" : a.prefetch == "oracle" ? "oracle" : a.prefetch == "transplant" ? "transplant" : "routers") << "\":" << C.hook->stats_json(C, steps);
    js << ",\"model\":" << M->stats_json(steps);
    if (C.run_ahead) js << ",\"run_ahead\":" << C.ra_json();
  };
  auto head = [&](const char* mode) {
    js << "{\"engine\":\"dmeng\",\"family\":\"" << family << "\",\"capabilities\":" << mi.caps_json() << ",\"mode\":\"" << mode << "\",\"accuracy\":\"" << a.mode
       << "\",\"preset\":" << (profiled ? P.json() : std::string("null")) << ",\"resolved\":" << [&] {
            RunOpts r = ro;  // as applied: names only with the routers' hook, background reads only with the routers
            if (a.prefetch != "router") r.routers.clear();
            r.bg_ranks = rts.bg_ranks;
            r.cache = a.cache_auto ? r.cache : "fixed";
            return r.resolved(a.cache_gb).json();
          }() << ",\"cache_auto\":" << auto_why << ",\"prefetch\":\"" << a.prefetch
       << "\",\"lead\":\"" << (a.prefetch == "oracle" ? a.lead : a.prefetch == "router" ? std::to_string(rts.lag) : plan ? a.plan_routers : "0")
       << "\",\"m\":" << (a.prefetch == "router" ? a.m : 0) << ",\"cache_gb\":" << a.cache_gb << ",\"slots\":[";
    for (int p = 0; p < C.cache.npools; p++) js << (p ? "," : "") << C.cache.slots(p);
    js << "],\"threads\":" << a.threads << ",\"readers\":" << a.readers << ",\"split\":" << a.split << ",\"mem\":\"" << a.mem << "\",\"res_mem\":\""
       << a.res_mem << "\",\"prec\":\"" << a.prec << "\",\"skip_reads\":" << (a.skip_reads ? "true" : "false") << ",\"loop_hold\":"
       << (a.loop_hold ? "true" : "false") << ",\"ctx\":" << a.ctx << ",\"temp\":" << a.temp << ",\"load_s\":" << load_s;
  };
  if (!a.draft_check.empty()) {
    // ---------------------------------------------------------------- the head's port check on recorded features
    std::vector<int> ids = parse_ids(slurp(a.prompt_file));
    const int N = (int)ids.size(), H = mi.hidden;
    const size_t rec = 4 + (size_t)16 * H;
    std::vector<uint8_t> raw(rec * N);
    FILE* tf = fopen(a.draft_check.c_str(), "rb");
    if (!tf || fread(raw.data(), 1, raw.size(), tf) != raw.size()) return die("draft check", "features file shorter than the ids");
    fclose(tf);
    FILE* fd = fopen((a.check_out + ".drafts").c_str(), "wb");
    FILE* fl = fopen((a.check_out + ".logits").c_str(), "wb");
    if (!fd || !fl) return die("draft check", "cannot write " + a.check_out);
    std::vector<float> feats((size_t)4 * H), lg(VO);
    int fed = 0, done = 0;
    auto t0 = clk::now();
    for (int p = a.check_start; p < N - 2; p++) {
      for (; fed <= p; fed++) {
        std::memcpy(feats.data(), raw.data() + (size_t)fed * rec + 4, feats.size() * 4);
        M->draft_head_feed(feats.data(), 1, fed);
      }
      int32_t d[7] = {0};
      float pr[7] = {0};
      const int nd = M->draft_head(ids[p + 1], 7, 0.f, d, pr, done < a.check_logits ? lg.data() : nullptr);
      if (nd != 7) return die("draft check", "position " + std::to_string(p) + ": " + std::to_string(nd) + " drafts");
      const int32_t pp = p;
      fwrite(&pp, 4, 1, fd);
      fwrite(d, 4, 7, fd);
      fwrite(pr, 4, 7, fd);
      if (done < a.check_logits) fwrite(lg.data(), 4, VO, fl);
      done++;
    }
    fclose(fd);
    fclose(fl);
    const MemStat m1 = memstat();
    js << "{\"engine\":\"dmeng\",\"mode\":\"draft_check\",\"positions\":" << done << ",\"exact\":" << a.draft_exact << ",\"s\":" << ms_since(t0) / 1000
       << ",\"head_load_s\":" << head_load_s << ",\"head_memory_gb\":" << M->draft_head_bytes() / 1e9 << ",\"footprint_peak_gb\":" << m1.footprint_peak_gb
       << ",\"model\":" << M->stats_json(done) << "}";
  } else if (!a.teacher.empty()) {
    // ---------------------------------------------------------------- teacher forcing
    std::vector<int> ids = parse_ids(slurp(a.teacher));
    FILE* df = a.dump_logits.empty() ? nullptr : fopen(a.dump_logits.c_str(), "wb");
    FILE* qf = a.qual_out.empty() ? nullptr : fopen(a.qual_out.c_str(), "wb");
    FILE* tf = a.taps_out.empty() ? nullptr : fopen(a.taps_out.c_str(), "wb");  // per position: argmax int32, taps float [4, H]
    std::vector<QRec> ref;
    if (!a.ref_qual.empty()) {
      FILE* f = fopen(a.ref_qual.c_str(), "rb");
      QRec q;
      while (f && fread(&q, sizeof q, 1, f) == 1) ref.push_back(q);
      if (f) fclose(f);
    }
    struct NllRow {
      double nll;
      int am;
    };
    std::vector<NllRow> rows;
    std::vector<int> am;
    std::vector<double> step_ms;
    std::vector<int> idx(VO);
    reset_stats();
    auto t0 = clk::now();
    for (size_t p = 0; p < ids.size(); p++) {
      const bool want = (int)p >= a.dump_from || qf || tf || !a.nll_out.empty() || !a.ref_nll.empty();
      auto ts = clk::now();
      M->step(*S, ids[p], want ? Want::LAST : Want::NONE);
      const double model_ms = ms_since(ts);
      record();
      dump_names();
      if (want) {
        const std::vector<float>& lg = M->logits();
        const int b = argmax(lg.data(), VO);
        if ((int)p >= a.dump_from) am.push_back(b);
        if (df && (int)p >= a.dump_from) fwrite(lg.data(), 4, VO, df);
        if (tf) {
          fwrite(&b, 4, 1, tf);
          fwrite(M->taps()->data(), 4, M->taps()->size(), tf);
        }
        if (p + 1 < ids.size()) {  // NLL of the next text token: log-sum-exp over the vocabulary in double
          double z = 0;
          for (int i = 0; i < VO; i++) z += std::exp((double)lg[i] - lg[b]);
          const double lse = lg[b] + std::log(z);
          rows.push_back({lse - (double)lg[ids[p + 1]], b});
          if (qf) {
            QRec q;
            q.argmax = b;
            q.nll = (float)(lse - lg[ids[p + 1]]);
            for (int i = 0; i < VO; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + TK, idx.end(), [&](int x, int y) { return lg[x] > lg[y] || (lg[x] == lg[y] && x < y); });
            for (int k = 0; k < TK; k++) {
              q.ids[k] = idx[k];
              q.lp[k] = (float)(lg[idx[k]] - lse);
              q.ref_lp[k] = p < ref.size() ? (float)(lg[ref[p].ids[k]] - lse) : NAN;
            }
            fwrite(&q, sizeof q, 1, qf);
          }
        }
      }
      step_ms.push_back(ms_since(ts));
      fe_ms += step_ms.back() - model_ms;
    }
    const double dt = ms_since(t0) / 1000;
    if (df) fclose(df);
    if (qf) fclose(qf);
    if (tf) fclose(tf);
    double nll = 0;
    for (auto& r : rows) nll += r.nll;
    if (!a.nll_out.empty()) {
      FILE* f = fopen(a.nll_out.c_str(), "w");
      for (auto& r : rows) fprintf(f, "%.6f %d\n", r.nll, r.am);
      fclose(f);
    }
    head("teacher");
    js << ",\"positions\":" << ids.size() << ",\"dumped\":" << am.size() << ",\"vocab\":" << VO << ",\"s\":" << dt
       << ",\"reads\":" << RD.jobs.load() << ",\"scored\":" << rows.size() << ",\"nll_mean\":" << std::setprecision(6) << nll / std::max<size_t>(1, rows.size());
    if (!a.ref_nll.empty()) {
      std::vector<NllRow> rr;
      FILE* f = fopen(a.ref_nll.c_str(), "r");
      NllRow r;
      while (f && fscanf(f, "%lf %d", &r.nll, &r.am) == 2) rr.push_back(r);
      if (f) fclose(f);
      const size_t n = std::min(rr.size(), rows.size());
      double d = 0, rn = 0, d2 = 0;
      size_t agree = 0;
      for (size_t i = 0; i < n; i++) {
        d += rows[i].nll - rr[i].nll;
        d2 += (rows[i].nll - rr[i].nll) * (rows[i].nll - rr[i].nll);
        rn += rr[i].nll;
        agree += rows[i].am == rr[i].am;
      }
      const double nn = std::max<size_t>(1, n);
      js << ",\"ref_positions\":" << n << ",\"ref_nll_mean\":" << rn / nn << ",\"nll_delta\":" << d / nn
         << ",\"nll_delta_se\":" << std::sqrt(std::max(0.0, d2 / nn - (d / nn) * (d / nn)) / nn) << ",\"top1_agree\":" << agree / nn;
    }
    js << std::setprecision(4);
    emit_stats((int)ids.size(), dt, step_ms);
    js << ",\"argmax\":[";
    for (size_t i = 0; i < am.size(); i++) js << (i ? "," : "") << am[i];
    js << "]}";
  } else {
    // ---------------------------------------------------------------- generation
    std::vector<int> prompt = !a.prompt_ids.empty() ? parse_ids(a.prompt_ids) : parse_ids(slurp(a.prompt_file));
    if (prompt.empty()) {
      fprintf(stderr, "empty prompt\n");
      return 2;
    }
    if (a.early < 0 || (a.early > 0 && a.draft == "none")) return die("early", "--early N (N >= 0) goes with --draft lookup|fixed");
    // draft and verify: the shared drafter (include/dmeng/draft.h); with --early it keeps every position's released picks.
    // head / both: the model's draft head as a third source (HeadSource), lookup off for head
    Drafter dr;
    HeadSource hs;
    dr.o.mode = a.draft == "none" || a.draft == "fixed" ? a.draft : "lookup";
    dr.o.lookup = a.draft != "head";
    dr.o.max = a.draft_max;
    dr.o.probe_every = a.probe_every;
    dr.o.gate = a.draft_gate != 0;
    dr.o.margin = a.draft_margin;
    dr.o.early = a.early;
    const bool head_src = a.draft == "head" || a.draft == "both";
    LightHeadSource lh;
    int light_li = -1;  // the last committed position's index in the last pass (-1: a single step)
    if (!a.draft_light.empty()) {
      std::string e2;
      if (!lh.load(a.draft_light, mi.hidden, C.pool.get(), a.draft_light_q8 != 0, e2)) return die("draft light", e2);
      lh.thr = (float)a.draft_light_thr;
      if (!a.draft_light_backoff) lh.backoff0 = lh.backoff_max = 0;  // run every step (acceptance measurements)
      if (!a.draft_light_backoff) lh.gate = false;  // (and every draft: the acceptance per index)
      lh.est = [&](int n) { return dr.est(n); };
      lh.vocab_out = VO;
      lh.hidden = [&]() { return M->final_hidden(light_li); };
      lh.embed = [&](int32_t t, float* o) { return M->embed_row(t, o); };
      lh.head = [&](const float* z, float* c) { return M->head_argmax(z, c); };
      if (!a.draft_light_dump.empty()) lh.dump = fopen(a.draft_light_dump.c_str(), "wb");
      dr.extra.push_back(&lh);
    }
    if (head_src) {
      hs.M = M.get();
      hs.thr = (float)a.draft_thr;
      hs.backoff0 = a.backoff0;
      hs.backoff_max = a.backoff_max;
      dr.extra.push_back(&hs);
    }
    dr.o.copy_free = a.copy_free != 0;
    dr.init(family, a.early > 0 ? &C : nullptr, draft_cost_prior(a.model));
    dr.reads = &C;
    if (!a.draft_cost.empty() && a.draft != "none") {  // a cost file: the prior an earlier run measured (absent: the family's)
      struct stat sb;
      std::string e2;
      if (stat(a.draft_cost.c_str(), &sb) == 0 && !dr.load_cost(a.draft_cost, e2)) return die("draft cost", e2);
    }
    if (!a.skeleton.empty()) {
      std::vector<std::vector<int32_t>> fr;
      std::ifstream sf(a.skeleton);
      for (std::string ln; std::getline(sf, ln);) {
        std::vector<int> v = parse_ids(ln);
        if (!v.empty()) fr.emplace_back(v.begin(), v.end());
      }
      dr.skeleton.set(fr);
    }
    std::mt19937_64 rng(a.seed);
    const bool fast = a.fast_head > 0 && fh_greedy_ok;
    const Want want = fast ? Want::GREEDY : head_topk ? Want::TOPK : Want::LAST;
    auto tp = clk::now();
    if (a.prefill_chunk > 1) {  // the batched prefill (records not kept)
      std::string e2;
      if (!M->set_option("prefill_chunk", std::to_string(a.prefill_chunk), e2)) return die("prefill", e2);
      if (a.prefill_gpu_frac > 0 && !M->set_option("prefill_gpu_frac", std::to_string(a.prefill_gpu_frac), e2)) return die("prefill", e2);
      if (a.prefill_gpu_experts > 0 && !M->set_option("prefill_gpu_experts", std::to_string(a.prefill_gpu_experts), e2)) return die("prefill", e2);
      std::vector<int32_t> pid(prompt.begin(), prompt.end());
      wc::reset();
      M->prefill(*S, pid.data(), (int)pid.size(), true, {}, nullptr, nullptr, nullptr);
      if (wc::on()) fprintf(stderr, "WORK %s\n", wc::json().c_str());
    } else {
      for (size_t i = 0; i < prompt.size(); i++) {
        M->step(*S, prompt[i], i + 1 == prompt.size() ? want : Want::NONE);
        record();
      }
    }
    const double prefill_s = ms_since(tp) / 1000;
    const uint64_t pre_bytes = CS.bytes_urgent + CS.bytes_pf, pre_miss = CS.n_miss, pre_hit = CS.n_hit;
    const double pre_wait_ms = CS.wait_ms;  // the compute thread waiting on reads during the prompt
    // --state-check: the state after the prompt, as the prefix cache would save it
    std::vector<uint8_t> fixed_saved, pos_saved;
    std::vector<float> logits_saved;
    const int pos_saved_at = S->pos;
    if (a.state_check) {
      fixed_saved.resize(M->fixed_bytes(*S));
      pos_saved.resize(M->pos_bytes() * (size_t)S->pos);
      if (!M->save_fixed(*S, fixed_saved.data()) || !M->save_pos(*S, 0, S->pos, pos_saved.data())) return die("state", "save not supported");
      logits_saved = M->logits();
    }
    reset_stats();  // decode statistics from here on
    // (the greedy head's answer; after a batched prefill the logits come from the full head)
    auto next_tok = [&]() { return fast && M->greedy() >= 0 ? M->greedy() : sample(M->logits().data(), VO, a.temp, a.top_k, a.top_p, rng); };
    std::vector<int> out{next_tok()};
    std::vector<double> step_ms, step_wait, step_mb, step_hit;
    double ra_ab_n[2] = {0, 0}, ra_ab_ms[2] = {0, 0}, ra_ab_wait[2] = {0, 0};  // (DMENG_RA_AB) tokens with run-ahead off / on
    // (DMENG_BG_AB=1) background reads off / on, chosen per token at random in one process: tokens, ms, wait, MB read,
    // drive idle ms, ms with one read in flight
    double bg_ab[2][6] = {{0}};
    double ab22[2][2][3] = {{{0}}};  // both: [run-ahead on][background on] tokens, ms, wait
    const int bg_ranks0 = rts.bg_ranks;
    std::vector<int> step_n, step_acc;
    const bool drafting = a.draft != "none";
    if (a.verify_check > 0 && a.temp > 0) return die("verify-check", "greedy only");
    // under sampling, each produced token takes its own draw by index (sample_at): the same tokens as sampled decoding
    Draws draws(a.seed);
    if (drafting && a.temp > 0)
      dr.pick = [&](const float* lg, int i) { return sample_at(lg, VO, a.temp, a.top_k, a.top_p, draws, out.size() + (size_t)i); };
    struct ModelTarget : DraftTarget {
      Model& M;
      State& S;
      Want w;  // the head plain decoding uses (the screened one when it can): plain steps and verify passes alike
      ModelTarget(Model& m, State& s, Want w_) : M(m), S(s), w(w_) {}
      int pos() const override { return S.pos; }
      int* li = nullptr;
      void step(int id) override {
        M.step(S, id, w);
        if (li) *li = -1;
      }
      const float* logits() const override { return M.logits().data(); }
      bool verify(const int* ids, int n) override { return M.verify(S, (const int32_t*)ids, n, w); }
      const float* logits_at(int i) const override { return M.logits_at(i); }
      void commit(int a2) override {
        M.commit(S, a2);
        if (li) *li = a2 - 1;
      }
    } tgt(*M, *S, want);
    tgt.li = &light_li;
    std::string vcheck_json;
    if (a.verify_check > 0) {
      // reference: N single steps from the state after the prompt; then the same tokens through verify passes
      std::vector<uint8_t> f0(M->fixed_bytes(*S)), p0(M->pos_bytes() * (size_t)S->pos);
      const int pp = S->pos;
      M->save_fixed(*S, f0.data());
      M->save_pos(*S, 0, pp, p0.data());
      const std::vector<float> lg0 = M->logits();
      const int N = a.verify_check;
      std::vector<int> t{argmax(lg0.data(), VO)};
      std::vector<float> ref((size_t)N * VO);
      for (int k = 0; k < N; k++) {
        M->step(*S, t[k], want);  // (--fast-head: the screened head, in the passes below too)
        std::memcpy(ref.data() + (size_t)k * VO, M->logits().data(), 4 * (size_t)VO);
        t.push_back(argmax(M->logits().data(), VO));
      }
      std::vector<uint8_t> f1(M->fixed_bytes(*S)), p1(M->pos_bytes() * (size_t)S->pos), f2, p2;
      M->save_fixed(*S, f1.data());
      M->save_pos(*S, 0, S->pos, p1.data());
      if (!M->load_fixed(*S, f0.data(), pp) || !M->load_pos(*S, 0, pp, p0.data())) return die("verify-check", "restore not supported");
      long passes = 0, rolled = 0, bad = 0, cmpd = 0;
      int j = 0;
      while (j < N) {
        const int w = std::min(2 + (int)(passes % 7), N - j);
        std::vector<int32_t> toks(t.begin() + j, t.begin() + j + w);
        const bool wrong = w >= 2 && passes % 3 == 2;
        if (wrong) toks[w - 1] = (toks[w - 1] + 1) % VO;
        if (!M->verify(*S, toks.data(), w, want)) return die("verify-check", "this model has no multi-position verify");
        const int keep = wrong ? w - 1 : w;
        for (int i = 0; i < keep; i++, cmpd++) bad += std::memcmp(M->logits_at(i), ref.data() + (size_t)(j + i) * VO, 4 * (size_t)VO) != 0;
        M->commit(*S, keep);
        rolled += wrong;
        passes++;
        j += keep;
      }
      bad += std::memcmp(M->logits().data(), ref.data() + (size_t)(N - 1) * VO, 4 * (size_t)VO) != 0;
      f2.resize(f1.size());
      p2.resize(p1.size());
      M->save_fixed(*S, f2.data());
      M->save_pos(*S, 0, S->pos, p2.data());
      long fb = 0, pb = 0;
      for (size_t i = 0; i < f1.size(); i++) fb += f1[i] != f2[i];
      for (size_t i = 0; i < p1.size(); i++) pb += p1[i] != p2[i];
      std::ostringstream vc;
      vc << ",\"verify_check\":{\"positions\":" << N << ",\"passes\":" << passes << ",\"rollbacks\":" << rolled << ",\"logits_compared\":" << cmpd
         << ",\"logits_differ\":" << bad << ",\"fixed_bytes_differ\":" << fb << ",\"kv_bytes_differ\":" << pb << ",\"pos_after\":" << S->pos
         << ",\"ok\":" << (bad == 0 && fb == 0 && pb == 0 && S->pos == pp + N ? "true" : "false") << "}";
      vcheck_json = vc.str();
      fprintf(stderr, "verify_check %s\n", vcheck_json.c_str() + 16);
      if (bad || fb || pb) {
        printf("{\"engine\":\"dmeng\"%s}\n", vcheck_json.c_str());
        return 3;
      }
      M->load_fixed(*S, f0.data(), pp);  // back to the state after the prompt for the generation below
      M->load_pos(*S, 0, pp, p0.data());
      reset_stats();
    }
    std::vector<int32_t> hist(prompt.begin(), prompt.end());
    hist.push_back(out.back());
    const std::vector<int> stops = a.stop_eos >= 0 ? std::vector<int>{a.stop_eos} : std::vector<int>{};
    std::vector<int> got;
    auto td = clk::now();
    while ((int)out.size() < a.n) {
      if (a.stop_eos >= 0 && out.back() == a.stop_eos) break;
      const double w0 = C.t.wait;
      const uint64_t b0 = C.cache.st.bytes_urgent + C.cache.st.bytes_pf, h0 = C.cache.st.n_hit;
      static const bool ra_ab = getenv("DMENG_RA_AB") && atoi(getenv("DMENG_RA_AB")) > 0 && C.run_ahead;
      static uint32_t ra_lcg = 12345;
      if (ra_ab) ra_lcg = ra_lcg * 1664525u + 1013904223u, C.ra_off = ((ra_lcg >> 16) & 1) == 0;
      static const bool bgab = getenv("DMENG_BG_AB") && atoi(getenv("DMENG_BG_AB")) > 0 && bg_ranks0 > 0;
      static uint32_t bg_lcg = 54321;
      int bg_arm = -1;
      uint64_t bgd0[7], bgb0 = 0;
      if (bgab) {
        bg_lcg = bg_lcg * 1664525u + 1013904223u;
        bg_arm = (int)((bg_lcg >> 16) & 1);
        rts.bg_ranks = bg_arm ? bg_ranks0 : 0;
        RD.depth_times(bgd0);
        bgb0 = RD.bytes.load();
      }
      auto ts = clk::now();
      int n = 1, acc = 0;
      double model_ms = -1;
      if (!drafting) {
        M->step(*S, out.back(), want);
        model_ms = ms_since(ts);
        record();
        dump_names();
        out.push_back(next_tok());
      } else {
        got.clear();
        acc = dr.step(tgt, hist, VO, a.n - (int)out.size(), stops, got) - 1;
        n = dr.last_n;
        for (int g : got) {
          out.push_back(g);
          hist.push_back(g);
        }
      }
      step_ms.push_back(ms_since(ts));
      if (model_ms >= 0) fe_ms += step_ms.back() - model_ms;  // (drafting: the controller's work is inside the model's passes' time)
      if (ra_ab) ra_ab_n[!C.ra_off] += 1, ra_ab_ms[!C.ra_off] += step_ms.back(), ra_ab_wait[!C.ra_off] += C.t.wait - w0;
      if (bg_arm >= 0) {
        uint64_t d1[7];
        RD.depth_times(d1);
        double* x = bg_ab[bg_arm];
        x[0] += 1, x[1] += step_ms.back(), x[2] += C.t.wait - w0, x[3] += (RD.bytes.load() - bgb0) / 1e6;
        x[4] += (d1[0] - bgd0[0] + d1[1] - bgd0[1]) / 1e6, x[5] += (d1[2] - bgd0[2]) / 1e6;
        if (ra_ab) {
          double* z = ab22[!C.ra_off][bg_arm];
          z[0] += 1, z[1] += step_ms.back(), z[2] += C.t.wait - w0;
        }
      }
      step_wait.push_back(C.t.wait - w0);
      step_mb.push_back((C.cache.st.bytes_urgent + C.cache.st.bytes_pf - b0) / 1e6);
      step_hit.push_back((double)(C.cache.st.n_hit - h0));
      step_n.push_back(n);
      step_acc.push_back(acc);
    }
    if ((int)out.size() > a.n) out.resize(a.n);
    const double dec_s = ms_since(td) / 1000;
    dr.end();
    if (!a.draft_cost.empty() && drafting) dr.save_cost(a.draft_cost);  // this run's measured ratios, the next run's prior
    C.pick_log = nullptr;
    const int steps = (int)out.size() - 1;  // generated tokens (a verify step can produce several)
    int state_same = -1;
    if (a.state_check) {  // restore into a fresh state and decode the same number of tokens again
      std::unique_ptr<State> S2 = M->new_state(a.ctx, err);
      if (!S2 || !M->load_fixed(*S2, fixed_saved.data(), pos_saved_at) || !M->load_pos(*S2, 0, pos_saved_at, pos_saved.data()))
        return die("state", "restore failed");
      std::vector<int> out2{fast ? out[0] : argmax(logits_saved.data(), VO)};
      while (out2.size() < out.size()) {
        M->step(*S2, out2.back(), want);
        out2.push_back(fast ? M->greedy() : argmax(M->logits().data(), VO));
      }
      state_same = out2 == out;
    }
    if (!a.steps_out.empty()) {  // per decode step: wall ms, wait ms, MB read, hits
      FILE* f = fopen(a.steps_out.c_str(), "w");
      for (size_t i = 0; i < step_ms.size(); i++) fprintf(f, "%.3f %.3f %.3f %.0f\n", step_ms[i], step_wait[i], step_mb[i], step_hit[i]);
      fclose(f);
    }
    if (!a.out_ids.empty()) {
      FILE* f = fopen(a.out_ids.c_str(), "w");
      for (size_t i = 0; i < out.size(); i++) fprintf(f, "%s%d", i ? "," : "", out[i]);
      fprintf(f, "\n");
      fclose(f);
    }
    head("generate");
    js << ",\"start_pos\":" << a.start_pos << ",\"prompt_tokens\":" << prompt.size() << ",\"prefill_s\":" << prefill_s
       << ",\"prefill_tok_s\":" << prompt.size() / prefill_s << ",\"prefill_mb_read\":" << pre_bytes / 1e6 << ",\"prefill_hits\":" << pre_hit
       << ",\"prefill_misses\":" << pre_miss << ",\"prefill_wait_ms\":" << pre_wait_ms << ",\"tokens\":" << out.size();
    js << ",\"hold_releases\":" << C.n_hold_release << ",\"union_batches\":" << C.n_union_batches;  // phone sizes: pins released, unions in batches
    if (drafting) {
      std::string dj = dr.stats_json();
      if (head_src) {
        std::string hj = hs.stats_json();
        hj.insert(hj.size() - 1, ",\"load_s\":" + std::to_string(head_load_s));
        dj.insert(dj.size() - 1, ",\"head\":" + hj);
      }
      if (!a.draft_light.empty()) dj.insert(dj.size() - 1, ",\"light\":" + lh.stats_json());
      if (lh.dump) fclose(lh.dump), lh.dump = nullptr;
      js << ",\"draft\":" << dj;
    }
    js << vcheck_json;
    if (ra_ab_n[0] + ra_ab_n[1] > 0)
      js << ",\"ra_ab\":{\"tokens_off\":" << ra_ab_n[0] << ",\"ms_off\":" << ra_ab_ms[0] / std::max(1.0, ra_ab_n[0]) << ",\"wait_off\":"
         << ra_ab_wait[0] / std::max(1.0, ra_ab_n[0]) << ",\"tokens_on\":" << ra_ab_n[1] << ",\"ms_on\":" << ra_ab_ms[1] / std::max(1.0, ra_ab_n[1])
         << ",\"wait_on\":" << ra_ab_wait[1] / std::max(1.0, ra_ab_n[1]) << "}";
    if (bg_ab[0][0] + bg_ab[1][0] > 0) {
      rts.bg_ranks = bg_ranks0;
      js << ",\"bg_ab\":{";
      for (int r = 0; r < 2; r++) {
        const double* x = bg_ab[r];
        const double n = std::max(1.0, x[0]);
        js << (r ? ",\"on\":" : "\"off\":") << "{\"tokens\":" << x[0] << ",\"ms\":" << x[1] / n << ",\"wait\":" << x[2] / n << ",\"mb\":" << x[3] / n
           << ",\"idle_ms\":" << x[4] / n << ",\"one_read_ms\":" << x[5] / n << ",\"effective_gb_s\":" << x[3] / std::max(1e-9, x[1]) << "}";
      }
      js << "}";
    }
    if (ab22[0][0][0] + ab22[1][1][0] > 0) {  // [ra][bg]: tokens, ms, wait
      js << ",\"ra_bg_ab\":[";
      for (int r = 0; r < 2; r++)
        for (int b = 0; b < 2; b++) {
          const double* z = ab22[r][b];
          js << (r + b ? "," : "") << "{\"run_ahead\":" << r << ",\"background\":" << b << ",\"tokens\":" << z[0] << ",\"ms\":" << z[1] / std::max(1.0, z[0])
             << ",\"wait\":" << z[2] / std::max(1.0, z[0]) << "}";
        }
      js << "]";
    }
    if (a.prefill_chunk > 1) js << ",\"prefill_chunk\":" << a.prefill_chunk;
    if (a.prefill_chunk > 1 && (a.prefill_gpu_frac > 0 || a.prefill_gpu_experts > 0))
      js << ",\"prefill_gpu_frac\":" << a.prefill_gpu_frac << ",\"prefill_gpu_experts\":" << a.prefill_gpu_experts;
    if (a.state_check) js << ",\"state_check\":{\"fixed_mb\":" << fixed_saved.size() / 1e6 << ",\"pos_mb\":" << pos_saved.size() / 1e6 << ",\"same_ids\":" << (state_same ? "true" : "false") << "}";
    emit_stats(steps, dec_s, step_ms);
    js << "}";
  }
  if (namesf) fclose(namesf);
  if (!a.evict_trace.empty() && !C.cache.trace_write(a.evict_trace, err)) return die("evict trace", err);
  if (recf) {
    fseek(recf, 0, SEEK_SET);
    fwrite(&nrec, 4, 1, recf);
    fclose(recf);
  }
  const std::string out = js.str();
  if (!a.json.empty()) {
    FILE* f = fopen(a.json.c_str(), "w");
    fprintf(f, "%s\n", out.c_str());
    fclose(f);
  }
  printf("%s\n", out.c_str());
  return 0;
}
