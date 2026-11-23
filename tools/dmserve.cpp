// dmserve: the model worker behind python/serve/dmserve.py (the OpenAI-compatible front end). One model process, one request
// at a time; the front end owns HTTP, the tokenizer, the chat template, stop strings and tool-call parsing, and queues.
//
// Protocol (one line per message; numbers in decimal):
//   in   R <rid> <max_new> <temp> <top_p> <top_k> <seed> <n_ids> <n_marks> <n_stop> [<n_skel>]   then four lines: ids, marks,
//        stop ids, the request's tool schema (JSON array on one line); then n_skel lines (default 0), each the token ids of
//        one fragment of the tool-call format of the request's tools (the front end renders and tokenises them): drafts
//        for the skeleton source when drafting is on (--draft; greedy requests only, temperature 0)
//        C <rid>          cancel that request (prefill stops at the next chunk, decode at the next token)
//        A <id> <n_ids> <n_marks>   then two lines: ids, marks. A predicted prefix of the next prompt, prefilled while no
//                         request runs (src/serving/ahead.h); any request cancels it. out: A <id> {json} when it ends
//        S                statistics       Q   quit
//   out  READY {json}     B <rid> {json}  (prompt state ready: what was reused / computed, the prefill timings)
//        T <rid> <id>     one generated token      E <rid> <stop|length|cancel> {json}
//   The E line's json: tokens and timings, "decode" (decode statistics of the request), "decode_first50" (the same for
//   its first 50 decode steps; {} when it made fewer), "mem" (where the process's memory is, MB: LM::mem_json), the live
//   length and the prefix cache's statistics.
//
// Any model family of this build (the family is read from the resident file); the prefix cache and this loop see only
// the LM interface (include/dmeng/lm.h), implemented once for every family by ModelLM.
//
// Modes for measurement: --check IDS_FILE (batched prefill vs token by token: logits, state and the greedy continuation
// must be bit-identical), --prefill-bench IDS_FILE (one cold prefill, timings), --record-routing F (with --prefill-bench:
// write every position's picks per layer, int16 [pos][NL][K]).
//
// Prefill memory and speed: the batched prefill's scratch (~0.5 GB) is freed after every prefill (as the earlier serving
// worker did; --keep-scratch keeps it between prefills); --gpu-frac F computes that share of each 8-bit attention
// projection's rows on the GPU beside the CPU (macOS builds with DM_METAL; the same bits; default 0 = CPU only).
#include <mach/mach.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dmeng/address.h"
#include "dmeng/draft.h"
#include "dmeng/model_lm.h"
#include "dmeng/presets.h"
#include "dmeng/sampler.h"
#include "dmeng/session.h"
#include "../src/serving/ahead.h"
#include "../src/serving/hotset.h"
#include "../src/serving/memwatch.h"

using namespace dm;
using sclk = std::chrono::steady_clock;
static double ms_since2(sclk::time_point t) { return std::chrono::duration<double, std::milli>(sclk::now() - t).count(); }
static int sample(const std::vector<float>& lg, double temp, int top_k, double top_p, std::mt19937_64& rng) {
  return dm::sample(lg.data(), (int)lg.size(), temp, top_k, top_p, rng);
}

static std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> v;
  const char* p = s.c_str();
  while (*p) {
    while (*p && !(isdigit((unsigned char)*p) || *p == '-')) p++;
    if (!*p) break;
    char* e;
    long x = strtol(p, &e, 10);
    v.push_back((int)x);
    p = e;
  }
  return v;
}
static std::vector<int> read_ids_file(const std::string& f) {
  std::ifstream in(f);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_ints(ss.str());
}

struct Args {
  std::string model, resident, family, routers;
  int threads = 8, readers = 4, ctx = 33792, m = 8;
  double cache_gb = 5;
  std::string mem = "mlock", res_mem = "mlock";
  int chunk = 512, span = 8192, stream = 3;
  SessionOptions s;
  std::string check, bench, record;
  int check_gen = 16, bench_rep = 1;
  int ref_chunk = 1;
  std::string draft = "none";  // draft and verify for greedy requests: none | lookup (controller) | fixed
  int draft_max = 7, early = 0, probe_every = 48, verify_names = 1;
  double gpu_frac = 0;       // prefill: share of each 8-bit attention projection's rows on the GPU (exact); 0 = CPU only
  bool keep_scratch = false;  // prefill: keep the scratch between prefills (default: freed after each)
  int ahead_span = 512;       // idle-time prefill: positions per span (a cancel keeps the finished spans); 0 = --span
  bool ahead_park = true;     // idle-time prefill parks the live conversation it replaces
  std::string check_ahead_a, check_ahead_b;
  std::string hotset;         // saved hot set: written at quit, refilled in the background at start ("" = off)
  int check_ahead_len = 0;
  bool cache_auto = false;    // --cache-gb auto: the size from the machine (src/serving/memwatch.h), then the watcher
  MemRule mr;                 // --reserve-gb, --cache-min-gb
  int mem_watch = -1;         // follow macOS memory pressure (default: on with --cache-gb auto)
  int mem_low = 20, mem_high = 35;  // kern.memorystatus_level thresholds (shrink below low, grow back at or above high)
  double mem_grow_s = 30, mem_retry_s = 600;
  std::string evict = "auto", xtok;
  double rt_tau = -1, xtok_tau = 0.5;
  int rd_prio = -1, rt_cand = 16;
  // the user-facing surface (include/dmeng/presets.h)
  std::string profile = "auto", mode;
  int drafts = -1;
  bool print_defaults = false;
  std::set<std::string> given;  // flags on the command line (a profile default fills only the others)
};

static const char* HELP_SHORT = R"(usage: dmserve --model DIR [options]   (the model worker; python/serve/dmserve.py is the HTTP server)

  --model DIR            the model folder, as downloaded from the Hub
  --mode exact|fast      exact: generated tokens never depend on the cache, the routers or timing (the Mac's default);
                         fast: a picked expert that is not in memory and has a small routing weight is left out
  --profile mac|phone|phone-small   default: chosen from the device (macOS: mac)
  --cache-gb G|auto      expert cache size in GB (default auto: from free memory, follows memory pressure)
  --drafts 0|1           draft and verify for greedy requests, experimental (default 0)
  --help-advanced        every other option, grouped
)";

static const char* HELP = R"(usage: dmserve --model DIR [options]

profiles and modes (docs/MANIFEST-PRESETS.md)
  --profile auto|mac|phone|phone-small|none   defaults from the profile (auto: the device's; none: the worker's
                         built-in values). A default fills only what the command line leaves unset; --routers or --xtok
                         makes the early address entirely the command line's (--routers none: no names)
  --mode exact|fast|lossless|approx[:PRESET]   default: the profile's
  --drafts 0|1           1 = --draft lookup; default: the profile's (off)
  --print-defaults       print the resolved profile defaults as JSON and exit
model and memory
  --resident F --family F --ctx N (33792) --threads N (8) --readers N (4) --mem plain|mlock --res-mem plain|mlock
  --cache-gb G|auto      default auto under a profile (5 with --profile none); --reserve-gb G (5.5) --cache-min-gb G
                         --headroom-gb G; the watcher: --mem-watch 0|1 --mem-low-pct N --mem-high-pct N
                         --mem-grow-after-s S --mem-retry-s S
  --evict auto|hazard|freq
early names (lossless: names only schedule reads)
  --routers F|none --m M --rt-tau T --rt-cand C     the profile's routers and names rule by default
  --xtok F|none --xtok-tau T --rd-prio 0|1   cross-token names (Inkling: source 47 with priority reads by default)
prompt path
  --chunk N (512) --span N (8192) --stream N (3) --gpu-frac F --keep-scratch
  prefix cache: --ckpt-gb G --park-gb G --ckpt-every N --disk DIR|none (default DIR/.dmeng/prefix) --disk-gb G
                --disk-every N (Inkling: 1024) --preload | --no-preload
  --ahead-span N (512) --ahead-park 0|1   idle-time prefill (the front end sends the predicted prefix)
  --hotset F|none        saved hot set (default DIR/.dmeng/hotset.bin)
draft and verify (greedy requests only; the ids are plain decoding's)
  --draft none|lookup|fixed --draft-max N (7) --early N (0) --probe-every N (48) --verify-names 0|1 (default 0)
measurement
  --check IDS --ref-chunk N --gen N | --prefill-bench IDS --rep N --record-routing F | --check-ahead A B --check-ahead-len N
)";

// the process's memory footprint (what the system charges it, as Activity Monitor shows), now and its peak, in MB
static void footprint_mb(double& now, double& peak) {
  task_vm_info_data_t v;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  now = peak = 0;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n) != KERN_SUCCESS) return;
  now = v.phys_footprint / 1048576.0;
  peak = v.ledger_phys_footprint_peak / 1048576.0;
}

static std::mutex out_mu;
static void out(const std::string& s) {
  std::lock_guard<std::mutex> lk(out_mu);
  fwrite(s.data(), 1, s.size(), stdout);
  fputc('\n', stdout);
  fflush(stdout);
}

// ---------------------------------------------------------------------------------------------------- measurement modes
static void set_prefill(Model& m, int chunk, int span, int stream, bool batched_always) {
  std::string e;
  m.set_option("prefill_chunk", std::to_string(chunk), e);
  m.set_option("prefill_span", std::to_string(span), e);
  m.set_option("prefill_stream", std::to_string(stream), e);
  m.set_option("prefill_batched_always", batched_always ? "1" : "0", e);
}

static size_t state_digest(LM& lm, std::vector<uint8_t>& fixed, std::vector<uint8_t>& posd) {
  fixed.resize(lm.info().fixed_bytes);
  lm.save_fixed(fixed.data());
  posd.resize((size_t)lm.pos() * lm.info().pos_bytes);
  lm.save_pos(0, lm.pos(), posd.data());
  return fixed.size() + posd.size();
}

// batched prefill vs the reference (token by token, or the batched code one position at a time with --ref-chunk -1):
// the logits after the prompt bit for bit, every byte of the fixed state and of the KV the prompt left,
// and the greedy continuation itself
static int run_check(ModelLM& lm, const Args& a) {
  std::vector<int> ids = read_ids_file(a.check);
  const int V = lm.info().vocab;
  struct Res {
    std::vector<float> lg;
    std::vector<uint8_t> fx, pd;
    std::vector<int> gen;
    double ms = 0;
    std::string js;
  };
  auto run = [&](int chunk) {
    Res r;
    lm.reset();
    set_prefill(lm.model(), chunk < 0 ? 1 : chunk, chunk > 1 ? a.span : 1, a.stream, chunk < 0);
    r.lg.resize(V);
    PrefillStats st;
    auto t = sclk::now();
    lm.prefill(ids.data(), (int)ids.size(), r.lg.data(), {}, nullptr, &st);
    r.ms = ms_since2(t);
    r.js = st.json.empty() ? "{}" : st.json;
    state_digest(lm, r.fx, r.pd);  // the state the prompt left
    std::vector<float> lg = r.lg;
    std::mt19937_64 rng(0);
    for (int i = 0; i < a.check_gen; i++) {
      int t2 = sample(lg, 0, 0, 1, rng);
      r.gen.push_back(t2);
      lm.step(t2, lg.data());
    }
    return r;
  };
  Res A = run(a.ref_chunk);
  Res B = run(a.chunk);
  size_t lg_bad = 0, fx_bad = 0, pd_bad = 0;
  long fx_first = -1, pd_first_pos = -1;
  double lg_md = 0;
  for (int i = 0; i < V; i++) {
    lg_bad += memcmp(&A.lg[i], &B.lg[i], 4) != 0;
    lg_md = std::max(lg_md, (double)std::fabs(A.lg[i] - B.lg[i]));
  }
  for (size_t i = 0; i < A.fx.size(); i++)
    if (A.fx[i] != B.fx[i]) {
      fx_bad++;
      if (fx_first < 0) fx_first = (long)i;
    }
  const size_t pb = std::max<size_t>(1, lm.info().pos_bytes);
  for (size_t i = 0; i < A.pd.size() && i < B.pd.size(); i++)
    if (A.pd[i] != B.pd[i]) {
      pd_bad++;
      if (pd_first_pos < 0) pd_first_pos = (long)(i / pb);
    }
  printf("{\"mode\":\"check\",\"model\":\"%s\",\"positions\":%zu,\"ref_chunk\":%d,\"chunk\":%d,\"logit_bits_differ\":%zu,\"logit_max_diff\":%g,"
         "\"fixed_state_bytes_differ\":%zu,\"fixed_first_differing_byte\":%ld,\"kv_bytes_differ\":%zu,\"kv_first_differing_position\":%ld,"
         "\"greedy_identical\":%s,\"gen_ref\":\"",
         lm.info().name.c_str(), ids.size(), a.ref_chunk, a.chunk, lg_bad, lg_md, fx_bad, fx_first, pd_bad, pd_first_pos, A.gen == B.gen ? "true" : "false");
  for (size_t i = 0; i < A.gen.size(); i++) printf("%s%d", i ? "," : "", A.gen[i]);
  printf("\",\"gen_batched\":\"");
  for (size_t i = 0; i < B.gen.size(); i++) printf("%s%d", i ? "," : "", B.gen[i]);
  printf("\",\"ref_ms\":%.1f,\"batched_ms\":%.1f,\"ref\":%s,\"batched\":%s}\n", A.ms, B.ms, A.js.c_str(), B.js.c_str());
  return (lg_bad || fx_bad || pd_bad || A.gen != B.gen) ? 1 : 0;
}

static int run_bench(ModelLM& lm, const Args& a) {
  std::vector<int> ids = read_ids_file(a.bench);
  std::string e;
  for (int rep = 0; rep < a.bench_rep; rep++) {
    lm.reset();
    if (rep == 0 && !a.record.empty()) lm.model().set_option("prefill_record", a.record, e);
    std::vector<float> lg(lm.info().vocab);
    PrefillStats st;
    auto t = sclk::now();
    lm.prefill(ids.data(), (int)ids.size(), lg.data(), {}, nullptr, &st);
    double ms = ms_since2(t);
    double fp_now, fp_peak;
    footprint_mb(fp_now, fp_peak);  // after the prefill (its scratch freed unless --keep-scratch)
    if (rep == 0 && !a.record.empty()) lm.model().set_option("prefill_record", "", e);
    // a few decode steps after it (greedy): decode speed at this context, with the cache as prefill left it
    std::mt19937_64 rng(0);
    std::vector<double> sm;
    int tok = sample(lg, 0, 0, 1, rng);
    for (int i = 0; i < a.check_gen; i++) {
      auto ts = sclk::now();
      lm.step(tok, lg.data());
      sm.push_back(ms_since2(ts));
      tok = sample(lg, 0, 0, 1, rng);
    }
    double dsum = 0;
    for (double v : sm) dsum += v;
    printf("{\"mode\":\"prefill_bench\",\"rep\":%d,\"positions\":%zu,\"chunk\":%d,\"stream\":%d,\"cache_gb\":%.2f,\"ms\":%.1f,\"tok_s\":%.2f,\"mb_read\":%.1f,"
           "\"decode_steps\":%zu,\"decode_tok_s\":%.2f,\"gpu_frac\":%.2f,\"keep_scratch\":%s,\"footprint_mb_after_prefill\":%.1f,"
           "\"footprint_peak_mb\":%.1f,\"prefill\":%s}\n",
           rep, ids.size(), a.chunk, a.stream, a.cache_gb, ms, ids.size() / (ms / 1e3), st.mb_read, sm.size(), sm.size() / (dsum / 1e3), a.gpu_frac,
           a.keep_scratch ? "true" : "false", fp_now, fp_peak, st.json.empty() ? "{}" : st.json.c_str());
    fflush(stdout);
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------------- worker
struct Req {
  long rid = 0;
  int max_new = 0, top_k = 0;
  double temp = 0, top_p = 1;
  uint64_t seed = 0;
  std::vector<int> ids, marks, stop;
  // The request's OpenAI tool schema (JSON array, one line, "[]" without tools). For decode-side drafters: the tool-call
  // skeleton a schema forces can be drafted from it; the session's token history is pc.live() (prompt + generated so far).
  std::string tools_json;
  std::vector<std::vector<int32_t>> skel;  // tool-call format fragments (token ids)
  sclk::time_point t_arrive;  // when the reader got it (an idle-time prefill it cancelled delays the start)
};

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto v = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    a.given.insert(k);
    if (k == "--help" || k == "-h") {
      printf("%s", HELP_SHORT);
      return 0;
    } else if (k == "--help-advanced") {
      printf("%s", HELP);
      return 0;
    } else if (k == "--profile") a.profile = v();
    else if (k == "--mode") a.mode = v();
    else if (k == "--drafts") a.drafts = std::stoi(v());
    else if (k == "--print-defaults") a.print_defaults = true;
    else if (k == "--model") a.model = v();
    else if (k == "--resident") a.resident = v();
    else if (k == "--family") a.family = v();
    else if (k == "--cache-gb") {
      const std::string x = v();
      a.cache_auto = x == "auto";
      if (!a.cache_auto) a.cache_gb = std::stod(x);
    } else if (k == "--reserve-gb") a.mr.reserve_gb = std::stod(v());
    else if (k == "--cache-min-gb") a.mr.min_gb = std::stod(v());
    else if (k == "--mem-watch") a.mem_watch = std::stoi(v());
    else if (k == "--mem-low-pct") a.mem_low = std::stoi(v());
    else if (k == "--mem-high-pct") a.mem_high = std::stoi(v());
    else if (k == "--mem-grow-after-s") a.mem_grow_s = std::stod(v());
    else if (k == "--mem-retry-s") a.mem_retry_s = std::stod(v());
    else if (k == "--headroom-gb") a.mr.headroom_gb = std::stod(v());
    else if (k == "--evict") a.evict = v();
    else if (k == "--rt-tau") a.rt_tau = std::stod(v());
    else if (k == "--rt-cand") a.rt_cand = std::stoi(v());
    else if (k == "--xtok") a.xtok = v();
    else if (k == "--xtok-tau") a.xtok_tau = std::stod(v());
    else if (k == "--rd-prio") a.rd_prio = std::stoi(v());
    else if (k == "--ctx") a.ctx = std::stoi(v());
    else if (k == "--threads") a.threads = std::stoi(v());
    else if (k == "--readers") a.readers = std::stoi(v());
    else if (k == "--mem") a.mem = v();
    else if (k == "--res-mem") a.res_mem = v();
    else if (k == "--chunk") a.chunk = std::stoi(v());
    else if (k == "--stream") a.stream = std::stoi(v());
    else if (k == "--span") a.span = std::stoi(v());
    else if (k == "--routers") a.routers = v();
    else if (k == "--m") a.m = std::stoi(v());
    else if (k == "--ckpt-gb") a.s.ckpt_gb = std::stod(v());
    else if (k == "--park-gb") a.s.park_gb = std::stod(v());
    else if (k == "--ckpt-every") a.s.ckpt_every = std::stoi(v());
    else if (k == "--disk-every") a.s.disk_every = std::stoi(v());
    else if (k == "--disk") a.s.disk_dir = v();
    else if (k == "--disk-gb") a.s.disk_gb = std::stod(v());
    else if (k == "--no-preload") a.s.preload_experts = false;
    else if (k == "--preload") a.s.preload_experts = true;
    else if (k == "--check") a.check = v();
    else if (k == "--ref-chunk") a.ref_chunk = std::stoi(v());
    else if (k == "--gen") a.check_gen = std::stoi(v());
    else if (k == "--prefill-bench") a.bench = v();
    else if (k == "--rep") a.bench_rep = std::stoi(v());
    else if (k == "--record-routing") a.record = v();
    else if (k == "--gpu-frac") a.gpu_frac = std::stod(v());
    else if (k == "--keep-scratch") a.keep_scratch = true;
    else if (k == "--draft") a.draft = v();
    else if (k == "--draft-max") a.draft_max = std::stoi(v());
    else if (k == "--early") a.early = std::stoi(v());
    else if (k == "--verify-names") a.verify_names = std::stoi(v());
    else if (k == "--probe-every") a.probe_every = std::stoi(v());
    else if (k == "--ahead-span") a.ahead_span = std::stoi(v());
    else if (k == "--ahead-park") a.ahead_park = std::stoi(v()) != 0;
    else if (k == "--check-ahead") {
      a.check_ahead_a = v();
      a.check_ahead_b = v();
    } else if (k == "--check-ahead-len") a.check_ahead_len = std::stoi(v());
    else if (k == "--hotset") a.hotset = v();
    else {
      fprintf(stderr, "unknown flag %s\n", k.c_str());
      return 2;
    }
  }
  if (a.model.empty()) {
    fprintf(stderr,
            "usage: dmserve --model DIR [--resident F] [options]\n"
            "  prefix cache: --ckpt-gb G (checkpoints in memory, default %.2f) --park-gb G (parked sequences, default %.2f)\n"
            "                --disk DIR --disk-gb G (default %.1f)\n"
            "  draft and verify (greedy requests only; the ids are plain decoding's): --draft none|lookup|fixed (default none),\n"
            "                --draft-max N (7), --early N (name the copied positions' experts N positions ahead; 0 = off),\n"
            "                --probe-every N (48)\n",
            SessionOptions().ckpt_gb, SessionOptions().park_gb, SessionOptions().disk_gb);
    return 2;
  }
  std::string err;
  // the profile's defaults: each fills only what the command line leaves unset
  auto has = [&](const char* f) { return a.given.count(f) > 0; };
  Preset P;
  if (a.profile != "none" && !resolve_preset(a.model, a.profile, a.mode, "", -1, P, err)) {
    fprintf(stderr, "profile: %s\n", err.c_str());
    return 1;
  }
  const bool profiled = a.profile != "none" && P.profile != "none";
  if (a.print_defaults) {
    printf("%s\n", profiled ? P.json().c_str() : "null");
    return 0;
  }
  if (profiled) {
    for (auto& n : P.notes) fprintf(stderr, "dmserve: %s\n", n.c_str());
    a.mode = P.engine_mode;
    if (!has("--resident") && !P.resident.empty()) a.resident = P.resident;
    if (!has("--routers") && !has("--xtok")) {
      a.routers = P.routers;
      if (!has("--m")) a.m = P.m;
      if (!has("--rt-tau")) a.rt_tau = P.tau;
      if (!has("--rt-cand")) a.rt_cand = P.cand;
      a.xtok = P.xtok;
      if (!has("--xtok-tau")) a.xtok_tau = P.xtok_tau;
      if (!has("--rd-prio") && P.rd_prio) a.rd_prio = 1;
    }
    if (!has("--evict")) a.evict = P.evict;
    if (!has("--cache-gb")) a.cache_auto = true;
    if (!has("--disk")) a.s.disk_dir = P.cache_dir + "/prefix";
    if (!has("--disk-every") && P.family == "inkling2_loop") a.s.disk_every = 1024;  // its fixed state is 173 MB
    if (!has("--hotset")) a.hotset = P.cache_dir + "/hotset.bin";
    if (!has("--verify-names")) a.verify_names = 0;
    if (a.drafts < 0) a.drafts = P.drafts ? 1 : 0;
  } else {
    a.mode = engine_mode_of(a.mode.empty() ? "lossless" : a.mode);
  }
  if (a.drafts > 0 && !has("--draft")) a.draft = "lookup";
  if (a.routers == "none") a.routers.clear();
  if (a.xtok == "none") a.xtok.clear();
  if (a.hotset == "none") a.hotset.clear();
  if (a.s.disk_dir == "none") a.s.disk_dir.clear();
  auto tl = sclk::now();
  LoadOpts lo;
  std::string family;
  std::unique_ptr<Model> M;
  if (resolve_model(a.model, a.resident, mem_of(a.res_mem), lo, family, err)) M = make_model(a.family.empty() ? family : a.family, err);
  if (!M || !M->load(lo, err)) {
    fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  std::string auto_why = "null";
  double cache_start = 0;
  if (a.cache_auto) {  // the largest locked cache that leaves the reserve (measured now, the resident weights loaded)
    const ModelInfo& mi0 = M->info();
    a.cache_gb = auto_cache_gb(a.mr, M->pos_bytes(), a.s.ckpt_gb, store_bytes_gb(a.model, mi0.layer_file, mi0.layers), cache_start, auto_why);
  }
  CoreOpts co;
  co.threads = a.threads;
  co.readers = a.readers;
  co.cache_gb = a.cache_gb;
  co.mem = mem_of(a.mem);
  co.evict = a.evict;
  co.cache_start_gb = cache_start;  // auto: slots for the ceiling reserved, the start size locked
  Core C;
  if (!C.init(M->info(), co, a.model, err) || !M->bind(C, err)) {
    fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  Routers rts;  // decode: trained lag-k routers naming m experts early (lossless); none = cache only
  if (!a.routers.empty()) {
    if (!rts.load(a.routers, C, err)) {
      fprintf(stderr, "routers: %s\n", err.c_str());
      return 1;
    }
    rts.m = a.m;
    rts.tau = a.rt_tau;
    rts.cand = a.rt_cand;
    C.hook = &rts;
  }
  Xtok xt;  // cross-token routers (names the next token's first layers), wrapping the hook above
  if (!a.xtok.empty()) {
    if (!xt.load(a.xtok, C, err)) {
      fprintf(stderr, "xtok: %s\n", err.c_str());
      return 1;
    }
    xt.tau = a.xtok_tau;
    xt.inner = C.hook;
    C.hook = &xt;
  }
  if (a.mode != "lossless") {  // opt-in approximate mode, as dmeng --mode
    const std::string preset = a.mode == "approx" ? "" : a.mode.rfind("approx:", 0) == 0 ? a.mode.substr(7) : "?";
    if (!C.ns.preset(preset)) {
      fprintf(stderr, "--mode exact|fast|lossless|approx[:PRESET]; presets: %s\n", Approx::presets_help());
      return 2;
    }
    if (C.ns.prio) C.cache.rd.prio = true;
    if (C.ns.cancel) C.cancel_stale = true;
  }
  if (a.rd_prio >= 0) C.cache.rd.prio = a.rd_prio != 0;  // speculative reads by priority instead of first in, first out
  std::unique_ptr<State> S = M->new_state(a.ctx, err);
  if (!S) {
    fprintf(stderr, "state: %s\n", err.c_str());
    return 1;
  }
  set_prefill(*M, a.chunk, a.span, a.stream, false);
  {
    std::string e;
    if (!a.keep_scratch) M->set_option("prefill_release", "1", e);  // families without a batched prefill have no scratch
    if (a.gpu_frac > 0 && !M->set_option("prefill_gpu_frac", std::to_string(a.gpu_frac), e)) {
      fprintf(stderr, "--gpu-frac: %s\n", e.c_str());
      return 1;
    }
  }
  ModelLM lm(*M, C, *S, a.ctx);
  double load_ms = ms_since2(tl);
  if (!a.check.empty()) return run_check(lm, a);
  if (!a.bench.empty()) return run_bench(lm, a);
  if (!a.check_ahead_a.empty()) {
    std::vector<int> A = read_ids_file(a.check_ahead_a), B = read_ids_file(a.check_ahead_b);
    const int al = a.check_ahead_len > 0 ? a.check_ahead_len : (int)B.size() - 16;
    return check_ahead(lm, a.s, A, B, al, a.chunk, a.span, a.ahead_span > 0 ? a.ahead_span : a.span, a.stream, a.check_gen);
  }

  PrefixCache pc(lm, a.s);
  // draft and verify (greedy requests): the shared drafter over the session (prompt + everything generated), plus the
  // request's tool-call fragments; with --early it keeps every position's released picks for early names
  if (a.draft != "none" && a.draft != "lookup" && a.draft != "fixed") {
    fprintf(stderr, "--draft none|lookup|fixed\n");
    return 2;
  }
  Drafter dr;
  dr.o.mode = a.draft;
  dr.o.max = a.draft_max;
  dr.o.early = a.early;
  dr.o.probe_every = a.probe_every;
  dr.init(M->info().family, a.early > 0 && a.draft != "none" ? &C : nullptr);
  dr.reads = &C;
  C.chunk_names = a.verify_names != 0;
  struct PcTarget : DraftTarget {
    PrefixCache& pc;
    LM& lm;
    PcTarget(PrefixCache& p, LM& l) : pc(p), lm(l) {}
    int pos() const override { return lm.pos(); }
    void step(int id) override { pc.step(id, nullptr); }
    const float* logits() const override { return pc.logits(); }
    bool verify(const int* ids, int n) override { return pc.verify(ids, n); }
    const float* logits_at(int i) const override { return pc.logits_at(i); }
    void commit(int a2) override { pc.commit(a2); }
  } tgt(pc, lm);
  AheadRunner ahead(lm, pc, a.ahead_span > 0 ? a.ahead_span : a.span, a.span, a.ahead_park);
  std::unique_ptr<HotSet> hot;
  if (!a.hotset.empty()) hot.reset(new HotSet(lm, a.hotset));
  const bool hot_loaded = hot && hot->load();
  {
    std::ostringstream s;
    s << "READY {\"model\":\"" << lm.info().name << "\",\"vocab\":" << lm.info().vocab << ",\"ctx\":" << lm.info().max_ctx << ",\"load_ms\":" << load_ms
      << ",\"cache_gb\":" << a.cache_gb << ",\"chunk\":" << a.chunk << ",\"stream\":" << a.stream
      << ",\"fixed_mb\":" << lm.info().fixed_bytes / 1e6 << ",\"pos_kb\":" << lm.info().pos_bytes / 1e3 << ",\"hotset\":" << (hot_loaded ? hot->size() : 0)
      << ",\"cache_auto\":" << auto_why << ",\"mode\":\"" << a.mode << "\",\"preset\":" << (profiled ? P.json() : std::string("null"))
      << ",\"mem\":" << lm.mem_json() << "}";
    out(s.str());
  }
  // stdin reader: requests into a queue, cancels applied at once
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Req> q;
  std::atomic<long> cur{-1}, cancelled{-1};
  std::atomic<bool> quit{false}, want_stats{false};
  std::unique_ptr<MemWatch> mw;  // the cache follows macOS memory pressure (shrinks, grows back to the start size)
  if (a.mem_watch > 0 || (a.mem_watch < 0 && a.cache_auto)) {
    mw.reset(new MemWatch(C.cache, cache_start > 0 ? cache_start : a.cache_gb, a.cache_gb, std::min(a.mr.min_gb, a.cache_gb)));
    mw->low_pct = a.mem_low;
    mw->high_pct = a.mem_high;
    mw->grow_after_s = a.mem_grow_s;
    mw->retry_s = a.mem_retry_s;
    mw->wake = [&] { cv.notify_all(); };
    mw->note = [&](const std::string& j) { out("M 0 " + j.substr(0, j.size() - 1) + ",\"mem\":" + lm.mem_json() + "}"); };
    mw->start();
    out("M 0 {\"watcher\":\"on\",\"floor_gb\":" + std::to_string(mw->floor_gb()) + ",\"start_gb\":" + std::to_string(mw->target() / 1e9) +
        ",\"ceiling_gb\":" + std::to_string(a.cache_gb) + "}");
  }
  std::thread reader([&] {
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      if (line[0] == 'C') {
        long rid = atol(line.c_str() + 2);
        cancelled.store(rid);
        if (cur.load() == rid) lm.cancel();
        continue;
      }
      if (line[0] == 'S') {
        want_stats.store(true);
        cv.notify_all();
        continue;
      }
      if (line[0] == 'Q') break;
      if (line[0] == 'P') {  // P <low_pct> <high_pct>: the watcher's thresholds (tests of the shrink / grow path)
        int lo = 0, hi = 0;
        if (mw && sscanf(line.c_str() + 2, "%d %d", &lo, &hi) == 2) {
          mw->low_pct = lo;
          mw->high_pct = hi;
        }
        continue;
      }
      if (line[0] == 'A') {
        long id = 0;
        int nid = 0, nmk = 0;
        sscanf(line.c_str() + 2, "%ld %d %d", &id, &nid, &nmk);
        std::string l1, l2;
        std::getline(std::cin, l1);
        std::getline(std::cin, l2);
        ahead.submit(id, parse_ints(l1), parse_ints(l2));
        cv.notify_all();
        continue;
      }
      if (line[0] == 'R') {
        ahead.on_request();
        if (hot) hot->stop();  // the refill stops between two batches
        Req r;
        int nid = 0, nmk = 0, nst = 0, nsk = 0;
        r.t_arrive = sclk::now();
        unsigned long long seed = 0;
        sscanf(line.c_str() + 2, "%ld %d %lf %lf %d %llu %d %d %d %d", &r.rid, &r.max_new, &r.temp, &r.top_p, &r.top_k, &seed, &nid, &nmk, &nst, &nsk);
        r.seed = seed;
        std::string l1, l2, l3, l4;
        std::getline(std::cin, l1);
        std::getline(std::cin, l2);
        std::getline(std::cin, l3);
        std::getline(std::cin, l4);
        r.tools_json = l4;
        r.ids = parse_ints(l1);
        r.marks = parse_ints(l2);
        r.stop = parse_ints(l3);
        for (int i = 0; i < nsk; i++) {
          std::string ls;
          std::getline(std::cin, ls);
          std::vector<int> f = parse_ints(ls);
          if (!f.empty()) r.skel.emplace_back(f.begin(), f.end());
        }
        std::lock_guard<std::mutex> lk(mu);
        q.push_back(std::move(r));
        cv.notify_all();
      }
    }
    quit.store(true);
    ahead.on_request();  // stop an idle-time prefill: the front end is gone
    cv.notify_all();
  });
  std::vector<float> lg(lm.info().vocab);
  // a budget change the idle worker could not finish (a slot past the new cap still pinned or in flight) is tried again
  // every 0.5 s, never in the way of a request: what can be applied is applied, the rest waits (a running step also
  // applies it at its boundaries), and the size in force is what the watcher's lines report
  sclk::time_point budget_next = sclk::now();
  auto budget_due = [&] { return C.cache.budget_req.load() >= 0 && sclk::now() >= budget_next; };
  while (true) {
    Req r;
    {
      std::unique_lock<std::mutex> lk(mu);
      cv.wait_for(lk, std::chrono::milliseconds(200), [&] {
        return quit.load() || !q.empty() || want_stats.load() || ahead.pending() || (hot && hot->pending()) || budget_due();
      });
      if (budget_due() && q.empty()) {  // idle: apply a budget change now
        C.cache.apply_budget();
        if (C.cache.budget_req.load() >= 0) budget_next = sclk::now() + std::chrono::milliseconds(500);
      }
      if (q.empty() && !quit.load() && !want_stats.load() && !ahead.pending() && !(hot && hot->pending())) continue;
      if (want_stats.exchange(false)) out("STATS {\"session\":" + pc.stats_json() + ",\"model\":" + lm.stats_json() + "}");
      if (q.empty()) {
        if (quit.load()) break;
        if (hot && hot->pending()) {  // idle at start: refill the saved hot set, one batch at a time
          lk.unlock();
          std::string l = hot->step(false);
          if (!l.empty()) out(l);
          continue;
        }
        if (ahead.pending()) {  // idle: prefill the predicted next prompt (a request arriving cancels it)
          lk.unlock();
          std::string l = ahead.run();
          if (!l.empty()) out(l);
        }
        continue;
      }
      r = std::move(q.front());
      q.pop_front();
    }
    cur.store(r.rid);
    auto t0 = sclk::now();
    if (hot) {  // the refill's end (done, or stopped by this request)
      std::string l = hot->step(true);
      if (!l.empty()) out(l);
    }
    if (cancelled.load() == r.rid) {
      out("E " + std::to_string(r.rid) + " cancel {}");
      continue;
    }
    SyncResult sr = pc.sync(r.ids, r.marks, lg.data());
    double ttp = ms_since2(t0);
    if (dr.core && sr.source != "live" && sr.source != "checkpoint") {  // picks of restored positions never ran here
      const size_t row = (size_t)C.steps * C.K, nr = std::min(dr.pick_log.size(), (size_t)sr.reused * row);
      std::fill(dr.pick_log.begin(), dr.pick_log.begin() + nr, (int16_t)-1);
    }
    {
      std::ostringstream s;
      s << "B " << r.rid << " {\"prompt\":" << r.ids.size() << ",\"reused\":" << sr.reused << ",\"computed\":" << sr.computed << ",\"source\":\"" << sr.source
        << "\",\"lcp_live\":" << sr.lcp_live << ",\"live_len\":" << sr.live_len << ",\"lcp_disk\":" << sr.lcp_disk << ",\"ms_restore\":" << sr.ms_restore << ",\"ms_preload\":" << sr.ms_preload << ",\"preloaded\":" << sr.preloaded << ",\"ms_prefill\":" << sr.ms_prefill
        << ",\"ms_ready\":" << ttp << ",\"ms_wait_worker\":" << std::chrono::duration<double, std::milli>(t0 - r.t_arrive).count() << ",\"prefill\":" << (sr.pf.json.empty() ? "{}" : sr.pf.json) << "}";
      out(s.str());
    }
    if (sr.cancelled) {
      out("E " + std::to_string(r.rid) + " cancel {}");
      cur.store(-1);
      continue;
    }
    std::mt19937_64 rng(r.seed);
    lm.reset_stats();
    double samp_ms = 0, out_ms = 0;
    std::string first50 = "{}";
    int ntok = 0;
    std::string reason = "length";
    auto td = sclk::now();
    double first_ms = 0;
    const bool drafting = dr.on() && r.temp <= 0;
    std::vector<int32_t> hist;
    std::vector<int> got, gen;
    size_t gi = 0;  // got[gi..] are produced and not yet emitted (all but the last already in the state)
    if (drafting) {
      hist.assign(pc.live().begin(), pc.live().end());
      dr.skeleton.set(r.skel);
      dr.reset_stats();
    }
    while (ntok < r.max_new) {
      auto ts0 = sclk::now();
      int t;
      if (gi < got.size()) t = got[gi++];
      else t = sample(lg, r.temp, r.top_k, r.top_p, rng);
      samp_ms += ms_since2(ts0);
      ntok++;
      if (drafting) gen.push_back(t);
      auto to0 = sclk::now();
      out("T " + std::to_string(r.rid) + " " + std::to_string(t));
      out_ms += ms_since2(to0);
      if (ntok == 1) first_ms = ms_since2(t0);
      if (ntok == 51) first50 = lm.stats_json();  // the first 50 decode steps after this request's prefill
      if (std::find(r.stop.begin(), r.stop.end(), t) != r.stop.end()) {
        reason = "stop";
        break;
      }
      if (cancelled.load() == r.rid) {
        reason = "cancel";
        break;
      }
      if (ntok >= r.max_new) break;
      if (gi < got.size()) continue;  // the next produced token is already computed
      if ((int)pc.live().size() + 1 >= lm.info().max_ctx) {
        reason = "length";
        break;
      }
      if (!drafting) {
        pc.step(t, lg.data());
        continue;
      }
      // decode step through the drafter: t fed, plus whatever drafts the controller runs; produces 1 + accepted tokens
      hist.push_back(t);
      got.clear();
      gi = 0;
      const int budget = std::min(r.max_new - ntok, lm.info().max_ctx - (int)pc.live().size() - 2);
      dr.step(tgt, hist, lm.info().vocab, std::max(1, budget), r.stop, got);
      hist.insert(hist.end(), got.begin(), got.end() - 1);
    }
    if (drafting) {
      dr.lookup.add_archive(gen);  // a later prompt may drop it (the template drops earlier reasoning)
      dr.end();
    }
    double dec_ms = ms_since2(td);
    // the disk copy of this prompt's first long marked prefix (the agent's system prompt), off the answer's critical path
    for (int mk : r.marks)
      if (mk >= a.s.disk_min) {
        pc.persist(mk);
        break;
      }
    std::ostringstream s;
    s << "E " << r.rid << " " << reason << " {\"tokens\":" << ntok << ",\"ms_first_token\":" << first_ms << ",\"ms_decode\":" << dec_ms
      << ",\"decode_tok_s\":" << (ntok > 1 ? (ntok - 1) / (dec_ms / 1e3) : 0) << ",\"sample_ms_per_token\":" << samp_ms / std::max(1, ntok) << ",\"emit_ms_per_token\":" << out_ms / std::max(1, ntok)
      << ",\"decode\":" << lm.stats_json() << ",\"draft\":" << (drafting ? dr.stats_json() : std::string("{}")) << ",\"decode_first50\":" << first50 << ",\"mem\":" << lm.mem_json() << ",\"live\":" << pc.live().size() << ",\"session\":" << pc.stats_json() << "}";
    out(s.str());
    cur.store(-1);
  }
  if (mw) mw->stop();
  if (hot) {
    const bool ok = hot->save();
    out(std::string("H 0 {\"saved\":") + (ok ? "true" : "false") + "}");
  }
  reader.detach();
  return 0;
}
