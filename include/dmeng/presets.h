// The user-facing surface (docs/MANIFEST-PRESETS.md): two modes (exact, fast), three profiles picked from the device
// (mac, phone, phone-small) and the per-model defaults, read from the manifest's `modes` / `profiles` / `defaults` keys
// with built-in fallbacks. dmeng, dmserve and the C API all resolve them here, so they cannot disagree.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dm {

struct Preset {
  std::string family, profile, profile_why;
  std::string cache_dir;           // the engine's cache folder (DIR/.dmeng): prefix-cache disk copies, the saved hot set
  std::string mode, engine_mode;   // "exact" / "fast"; the engine's mode string ("lossless", "approx:drop")
  std::string resident_id;         // resident_q8 | resident_q4
  std::string resident;            // the file when it is not the resolver's default (resident_q4); "" = default
  std::string cache = "auto";      // auto (from free memory, then memory pressure) | slab (sized at open, iOS)
  std::string router_id, routers;  // early routers (names); routers = "" when none resolved
  int m = 8;                       // names rule: a fixed top m ...
  double tau = -1;                 // ... or the confidence rule (tau > 0) over `cand` candidates
  int cand = 16;
  std::string xtok_id, xtok;       // cross-token routers ("" = none)
  double xtok_tau = 0.5;
  bool rd_prio = false;            // speculative reads by priority (with the cross-token names)
  std::string evict = "auto";
  std::string layer_caps;          // per-layer slot caps (CoreOpts::layer_caps): "" = the core's rule (auto)
  bool drafts = false;
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
  int m = 0, cand = 0, drafts = 0;
  double tau = -1, cache_gb = 0, gpu_frac = 0, gpu_experts = 0;  // tau as the routers use it: >= 0 the confidence rule
  int run_ahead = 0, run_ahead_depth = 1;                         // picks named by the run-ahead (0 = off), its depth
  double ra_budget = 0;                                           // its limits (0 / 0 / false: none)
  int ra_max_tok = 0;
  bool ra_sat_skip = false;
  int readers = 0;
  int bg_ranks = 0, bg_gate = 0;
  bool cache_auto = false;
  std::string json() const;
};

// exact -> lossless, fast -> approx:drop (the built-in names); anything else is returned unchanged (lossless,
// approx[:PRESET] stay accepted everywhere)
std::string engine_mode_of(const std::string& mode);

// The profile of this device: macOS -> mac; iOS -> phone when the memory available to the process (avail_bytes, or
// os_proc_available_memory() when avail_bytes < 0) is at least min_gb, else phone-small.
std::string device_profile(int64_t avail_bytes, double min_gb, std::string& why);

// Resolve the defaults of model folder `dir` for profile `profile_req` ("" / "auto" = the device's) and mode
// `mode_req` ("" = the profile's; exact | fast | lossless | approx[:PRESET]). derived_dir: the engine's cache folder
// ("" = DIR/.dmeng). Router files are found (or derived) as docs/MANIFEST-PRESETS.md says.
bool resolve_preset(const std::string& dir, const std::string& profile_req, const std::string& mode_req, const std::string& derived_dir,
                    int64_t avail_bytes, Preset& p, std::string& err);

}  // namespace dm
