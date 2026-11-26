// The user-facing surface (docs/MANIFEST-PRESETS.md): two modes (exact, fast), three profiles picked from the device
// (mac, phone, phone-small) and the per-model defaults, read from the manifest's `modes` / `profiles` / `defaults` keys
// with built-in fallbacks. dmeng, dmserve and the C API all resolve them here, so they cannot disagree.
#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace dm {

// The iPhone's expert slab (a profile's cache "slab"), sized once at open by a named rule: a profile's "cache_slab"
// declares it, else the built-in "available" rule (every profile of every model):
//   available         slab = the memory iOS makes available at open - the in-memory weights' file - the router file
//                     - state_gb (0.3) - margin_gb (0.9)
//   footprint_target  slab = target_gb (the device class's app footprint) - non_cache_gb (the model's measured footprint
//                     beside the cache): {"rule": "footprint_target", "target_gb": T, "non_cache_gb": N, "calibration": {...}}
// either clamped to [min_gb, max_gb] (0.6, 8). Applied by the C API where iOS reports the memory available (macOS front
// ends size a slab profile's cache by the automatic rule instead).
struct SlabRule {
  std::string rule = "available";
  double state_gb = 0.3, margin_gb = 0.9;   // available
  double target_gb = 0, non_cache_gb = 0;   // footprint_target
  double min_gb = 0.6, max_gb = 8.0;
  // the slab in bytes from what the app has at open, the in-memory weights' file and the router file
  int64_t bytes(int64_t avail, int64_t resident, int64_t router) const;
  std::string json() const;  // the rule and its declared inputs
};

struct Preset {
  std::string family, profile, profile_why;
  std::string cache_dir;           // the engine's cache folder (DIR/.dmeng): prefix-cache disk copies, the saved hot set
  std::string mode, engine_mode;   // "exact" / "fast"; the engine's mode string ("lossless", "approx:drop")
  // where engine_mode comes from: "modes.<mode>" (the model-level declaration: the folder's manifest, the published one,
  // the built-in), "profiles.<profile>.modes.<mode>" (the profile's own declaration over it), "given" (an explicit
  // engine mode string, --mode / the C API's mode, which no declaration replaces)
  std::string engine_mode_from;
  std::string resident_id;         // resident_q8 | resident_q4
  std::string resident;            // the file when it is not the resolver's default (resident_q4); "" = default
  std::string cache = "auto";      // auto (from free memory, then memory pressure) | slab (sized at open, iOS)
  // the automatic cache's start rule (src/serving/memwatch.h MemRule): "headroom:G" leaves G GB of what is free once the
  // model's memory still to come is in (built-in, G = 3); "free_level:P" keeps the kernel's free level at P% (a
  // manifest's profile "cache_start": {"rule": "free_level", "floor_pct": P}, measured per model)
  std::string cache_start = "headroom:3";
  SlabRule slab;                   // the slab's sizing rule (cache "slab")
  std::string router_id, routers;  // early routers (names); routers = "" when none resolved
  // names rule (a profile's or the defaults' "names": one rule, or one per mode {"exact": rule, "fast": rule}, taken
  // for the mode in force; an engine string given as the mode counts as exact when lossless, else fast):
  int m = 8;                       // a fixed top m ...
  double tau = -1;                 // ... or the confidence rule (tau > 0) over `cand` candidates
  int cand = 16;
  std::string xtok_id, xtok;       // cross-token routers ("" = none)
  double xtok_tau = 0.5;
  bool rd_prio = false;            // speculative reads by priority (defaults.priority_reads; with names)
  std::string evict = "auto";
  std::string layer_caps;          // per-layer slot caps (CoreOpts::layer_caps): "" = the core's rule (auto)
  // drafting by default (a profile's "drafts": {"lookup": B, "head": B}; an older manifest's bool: the lookup drafts):
  // the lookup and tool-call drafts, the model's own draft head; off in every built-in profile
  bool draft_lookup = false, draft_head = false;
  // the batched prefill's exact CPU + GPU split (Metal builds): shares of the attention projections' and the experts'
  // rows on the GPU; 0 = CPU only (mac: 0.4 / 0.4 for both models, results/ios/p3-gpu/GPU-PREFILL.md; Inkling's experts: the NVFP4 layers)
  double gpu_frac = 0, gpu_experts = 0;
  // run-ahead while a step waits on reads (exact: names only): the picks it names (0 = off) and its depth in layers
  // (the mac profile of both models: 8 at depth 1, results/parity/PARITY.md; the phone profiles: off)
  int run_ahead = 0, run_ahead_depth = 1;
  // its limits (Core::ra_budget / ra_max_tok / ra_sat_skip; built-in: no limit, no skip): speculative ms per token at
  // most budget x the token's compute, at most max_tok run-aheads per token, none while every reader is busy
  double ra_budget = 0;
  int ra_max_tok = 0;
  bool ra_sat_skip = false;
  // (a profile's run_ahead may list "modes": on only in those; the phones: exact)
  int readers = 0;  // reader threads (0 = the front end's default; the phones: 8)
  // background reads for the drive's idle time (Routers::bg_ranks; 0 = off): the routers' ranks up to `ranks` of the
  // steps already named, started while at most `gate` speculative reads are in flight (off in every built-in profile:
  // on Flash-Next, measured in one process token by token, they raised the drive's effective rate 4-8% and the bytes
  // per token 2-8%, and the tokens did not get faster)
  int bg_ranks = 0, bg_gate = 0;
  // the OS file cache under the locked slab (store.h, Cache::file_under): a profile's "spare_memory":
  // {"min_physical_gb": G} turns it on where the device has at least G GB of physical memory (hw.memsize). Built-in:
  // phone 11 (the 12 GB class); mac and phone-small undeclared (off)
  double spare_min_gb = 0;
  bool spare_mem = false;
  // the context positions the front end reserves (a profile's "ctx": N or {"positions": N, "calibration": {...}});
  // 0 = undeclared, the front end's own (dmeng and the C API 16384, dmserve 33792)
  int ctx = 0;
  std::vector<std::string> notes;  // defaults that did not resolve, fallbacks taken
  std::string json() const;
};

// Precedence, in every front end (dmeng, dmserve, the C API): an option given explicitly > the profile > the manifest's
// defaults > the built-in values. The names rule: an explicit m without an explicit tau is a plain top m (the profile's
// confidence rule does not replace it); an explicit tau > 0 is the confidence rule, tau <= 0 a plain top m. The
// resolved tau is -1 for a plain top m (Routers::tau = 0 would name every candidate).
void names_rule(const Preset& p, bool profiled, bool m_given, int m, bool tau_given, double tau, bool cand_given, int cand, int& m_out,
                double& tau_out, int& cand_out);

// The settings a run actually uses after that precedence, written under "resolved" in every run's JSON (dmeng --json,
// dmserve's READY line, dm_stats_json), so an option silently replaced by a default shows.
struct Resolved {
  std::string profile, mode, routers, xtok, evict;  // mode: the engine's mode string; routers / xtok: "" = none
  int m = 0, cand = 0;
  std::string draft = "none";  // the drafting in force: none | lookup | fixed | head | both (JSON: "drafts": {"lookup", "head"})
  double tau = -1, cache_gb = 0, gpu_frac = 0, gpu_experts = 0;  // tau as the routers use it: >= 0 the confidence rule
  int run_ahead = 0, run_ahead_depth = 1;                         // picks named by the run-ahead (0 = off), its depth
  double ra_budget = 0;                                           // its limits (0 / 0 / false: none)
  int ra_max_tok = 0;
  bool ra_sat_skip = false;
  int readers = 0;
  int bg_ranks = 0, bg_gate = 0;
  bool cache_auto = false;
  std::string cache_start;  // the automatic cache's start rule (Preset::cache_start)
  // the slab's sizing rule and its declared inputs (SlabRule::json), written only when the cache is a slab (a phone
  // profile's, or the C API's slab on iOS): the block is unchanged otherwise
  std::string slab;
  std::string layer_caps;  // "" = the core's rule
  double xtok_tau = 0.5;
  int rd_prio = -1, prefill_chunk = 1;
  bool spare_mem = false;  // written only when on (the block is unchanged with it off)
  int ctx = 0;             // the context positions in force, written only when the profile declares a ctx (else 0)
  double drive_gbs = 0;    // bench only (--drive-gbs): written only when set (the block is unchanged without it)
  std::string json() const;
};

// exact -> lossless, fast -> approx:drop (the built-in names); anything else is returned unchanged (lossless,
// approx[:PRESET] stay accepted everywhere)
std::string engine_mode_of(const std::string& mode);

// the device's physical memory in bytes (hw.memsize; DMENG_TEST_PHYS_MEM overrides it, tests only)
double physical_memory_bytes();

// The profile of this device: macOS -> mac; iOS -> phone when the memory available to the process (avail_bytes, or
// os_proc_available_memory() when avail_bytes < 0) is at least min_gb, else phone-small.
std::string device_profile(int64_t avail_bytes, double min_gb, std::string& why);

// Resolve the defaults of model folder `dir` for profile `profile_req` ("" / "auto" = the device's) and mode
// `mode_req` ("" = the profile's; exact | fast | lossless | approx[:PRESET]). derived_dir: the engine's cache folder
// ("" = DIR/.dmeng). Router files are found (or derived) as docs/MANIFEST-PRESETS.md says. A mode's engine string is the
// model-level modes.<mode>.engine, replaced by the profile's own profiles.<profile>.modes.<mode>.engine when it declares
// one (Preset::engine_mode_from says which); a mode_req that names no declared mode is an engine string and is kept.
bool resolve_preset(const std::string& dir, const std::string& profile_req, const std::string& mode_req, const std::string& derived_dir,
                    int64_t avail_bytes, Preset& p, std::string& err);

// ---------------------------------------------------------------- one resolver for the three front ends
// What a front end was given explicitly: each fills it from its own spelling (dmeng's and dmserve's flags, the C API's
// dm_open_opts); an unset member is left to the profile, then the manifest's defaults, then the built-in values. The
// front ends differ only in that spelling, never in the precedence, so a setting resolves the same way in all three.
struct Given {
  std::optional<std::string> mode;      // exact | fast | lossless | approx[:PRESET]
  std::optional<std::string> resident;  // an explicit resident file
  std::optional<std::string> routers;   // an engine router file, or "none"
  std::optional<std::string> xtok;      // a cross-token router file, or "none"
  bool other_address = false;           // another early address (oracle, planner, transplant, a router id): no profile routers
  std::optional<int> m, cand;
  std::optional<double> tau, xtok_tau;
  std::optional<bool> rd_prio;
  std::optional<bool> spare_mem;        // --spare-mem 0|1: the OS file cache under the slab
  std::optional<double> drive_gbs;      // --drive-gbs X: bench only, the readers' drive rate capped (never from a profile)
  std::optional<int> ctx;               // context positions (dmeng / dmserve --ctx, the C API's ctx > 0)
  std::optional<std::string> evict, layer_caps;
  std::optional<double> cache_gb;       // an explicit size
  bool cache_auto = false;              // the automatic rule asked for explicitly
  std::optional<double> gpu_frac, gpu_experts;
  bool gpu_off = false;
  std::optional<int> run_ahead, run_ahead_depth, ra_max_tok;
  std::optional<double> ra_budget;
  std::optional<bool> ra_sat_skip;
  std::optional<int> readers, bg_ranks, bg_gate, prefill_chunk, drafts, early;
  std::optional<std::string> draft;    // none | lookup | fixed | head | both
  bool recording = false;  // a routing record is written (dmeng --record): single steps unless a chunk is given
};

// The run settings' flags, one spelling for dmeng and dmserve (the C API passes its dm_open_opts fields through the same
// function under these names): --mode --resident --routers FILE|none --xtok FILE|none --m --rt-tau --rt-cand --xtok-tau
// --rd-prio --evict --layer-caps --cache-gb N|auto --prefill-gpu-frac F --prefill-gpu-experts F --prefill-gpu off|on
// --prefill-chunk N --run-ahead --run-ahead-depth --run-ahead-budget --run-ahead-max --run-ahead-sat-skip --readers
// --bg-ranks --bg-gate --drafts 0|1 --draft MODE --early N --spare-mem 0|1 --drive-gbs X; aliases kept: --gpu-frac, --gpu-experts, --chunk (dmserve's).
// run_flag_name: the canonical name of a run-setting flag ("" if `flag` is not one); run_flag: its value into g (false
// and err on a malformed value).
std::string run_flag_name(const std::string& flag);
bool run_flag(const std::string& flag, const std::string& value, Given& g, std::string& err);

// The run options after the precedence. Before resolve_run, a front end puts its own built-in values here (what applies
// when neither the command line, the profile nor the manifest sets a value); resolve_run fills in the rest.
struct RunOpts {
  Preset preset;
  bool profiled = false;
  std::string mode = "lossless";  // the engine's mode string
  std::string resident;           // "" = the model resolver's default (resident_q8)
  std::string routers, xtok;      // engine files; "" = none
  bool routers_from_profile = false;
  int m = 8, cand = 16;
  double tau = -1, xtok_tau = 0.5;  // tau as the routers use it (>= 0: the confidence rule)
  int rd_prio = -1;                 // -1: the front end's rule (on with planner / fast mode), 0 / 1 explicit or the profile's
  std::string evict = "auto";
  std::string layer_caps;           // "" = the core's rule
  std::string cache = "fixed";      // fixed (cache_gb) | auto (from the machine) | slab (sized at open from what iOS gives)
  double cache_gb = 5.0;
  std::string cache_start = "headroom:3";  // the automatic cache's start rule (Preset::cache_start)
  SlabRule slab;                           // the slab's rule (Preset::slab; the built-in "available" without a profile)
  bool slab_on = false;                    // the cache is a slab: the profile's cache "slab" (or the C API's on iOS)
  double gpu_frac = 0, gpu_experts = 0;
  int run_ahead = 0, run_ahead_depth = 1, ra_max_tok = 0;
  double ra_budget = 0;
  bool ra_sat_skip = false;
  int readers = 4, bg_ranks = 0, bg_gate = 0, prefill_chunk = 1, drafts = 0;
  // drafting, once for every front end: an explicit --draft MODE; else --drafts 0|1 (1: the lookup and tool-call drafts);
  // else the profile's declared drafts (Preset::draft_lookup / draft_head: lookup, head or both); drafts = draft != none
  std::string draft = "none";
  int early = 0;
  bool spare_mem = false;  // the OS file cache under the slab (CoreOpts::spare_mem): explicit, else the profile's rule
  double drive_gbs = 0;    // bench only (CoreOpts::drive_gbs): explicit only, 0 = off
  int ctx = 16384;         // context positions: explicit, else the profile's declared ctx, else the front end's own
  bool ctx_declared = false;  // the profile declares one (resolved.ctx is written)
  std::vector<std::string> notes;
  // the "resolved" block of every run's JSON (the cache's size once a front end has computed it)
  Resolved resolved(double cache_gb_in_force) const;
};

// Resolve model folder `dir` for profile `profile_req` ("none": no profile; "" / "auto": the device's) with the explicit
// options `g`; `r` comes in holding the front end's built-in values. derived_dir, avail_bytes: as resolve_preset.
bool resolve_run(const std::string& dir, const std::string& profile_req, const Given& g, const std::string& derived_dir, int64_t avail_bytes,
                 RunOpts& r, std::string& err);

// The batched prompt path's values a model folder declares (its manifest's "prefill" key, else the family's published
// manifest's; family "" = the folder's manifest's), then the profile's own "prefill" block over them key by key (the
// folder's profile, else the published one; "" / "none": none), as key -> value strings for prefill_declared
// (dmeng/prefill.h)
std::vector<std::pair<std::string, std::string>> declared_prefill(const std::string& dir, const std::string& family,
                                                                  const std::string& profile = "");
// those values as a JSON object (every front end's JSON: "prefill_declared")
std::string prefill_json(const std::vector<std::pair<std::string, std::string>>& kv);

// The drafting controller's prior the model's manifest declares (drafting.cost_prior.pass_ms_over_step_ms with its
// provenance, "measured" or "estimated"): the cost of a pass of 0..8 positions over one step; empty when the folder has
// no manifest or the manifest declares none (the controller then starts from its neutral prior)
std::vector<double> draft_cost_prior(const std::string& dir);

}  // namespace dm
