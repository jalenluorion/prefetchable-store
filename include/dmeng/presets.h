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
  bool drafts = false;
  std::vector<std::string> notes;  // defaults that did not resolve, fallbacks taken
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
