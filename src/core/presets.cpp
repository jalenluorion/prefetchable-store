// Modes, profiles and per-model defaults (include/dmeng/presets.h, docs/MANIFEST-PRESETS.md).
#include "dmeng/presets.h"

#include <sys/stat.h>
#include <unistd.h>

#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
#include <os/proc.h>
#endif

#include <cstdio>
#include <sstream>

#include "../format/derive.h"
#include "../format/files.h"
#include "../format/manifest.h"

namespace dm {

using ms::Json;

// The built-in values: equal to the tables of PRESETS.md, used for every key a manifest leaves out.
static const char* BUILTIN_Q36 = R"({
 "modes": {"exact": {"engine": "lossless"}, "fast": {"engine": "approx:drop"}},
 "profiles": {
  "mac": {"mode": "exact", "resident": "resident_q8", "cache": "auto"},
  "phone": {"mode": "exact", "resident": "resident_q8", "cache": "slab", "min_available_gb": 5.0},
  "phone-small": {"mode": "fast", "resident": "resident_q4", "cache": "slab", "names": {"rule": "confidence", "tau": 0.5, "candidates": 16}}},
 "defaults": {"routers": ["linear_k2"], "names": {"rule": "top", "m": 8}, "cross_token": null, "eviction": "auto", "drafts": false}})";
static const char* BUILTIN_INK = R"({
 "modes": {"exact": {"engine": "lossless"}, "fast": {"engine": "approx:drop"}},
 "profiles": {
  "mac": {"mode": "exact", "resident": "resident_q8", "cache": "auto"},
  "phone": {"mode": "exact", "resident": "resident_q8", "cache": "slab", "min_available_gb": 5.0},
  "phone-small": {"mode": "fast", "resident": "resident_q8", "cache": "slab"}},
 "defaults": {"routers": ["mlp_k2_h512"], "names": {"rule": "confidence", "tau": 0.5, "candidates": 16},
  "cross_token": {"routers": ["xtok_s47_h512"], "source": 47, "tau": 0.5, "priority_reads": true}, "eviction": "auto", "drafts": false}})";

static bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

std::string engine_mode_of(const std::string& mode) {
  if (mode == "exact") return "lossless";
  if (mode == "fast") return "approx:drop";
  return mode;
}

std::string device_profile(int64_t avail, double min_gb, std::string& why) {
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  if (avail < 0) avail = (int64_t)os_proc_available_memory();
  char b[160];
  snprintf(b, sizeof b, "iOS, %.2f GB available at open (phone from %.1f GB)", avail / 1e9, min_gb);
  why = b;
  return avail >= (int64_t)(min_gb * 1e9) ? "phone" : "phone-small";
#else
  (void)avail;
  (void)min_gb;
  why = "macOS";
  return "mac";
#endif
}

// Research folders keep the files the defaults were measured with in their own places (docs/MANIFEST-PRESETS.md step 3)
static std::vector<std::string> research_paths(const std::string& family, const std::string& id) {
  auto num = [&](const std::string& pre, const std::string& post) -> std::string {  // "linear_k2" -> "2"
    if (id.rfind(pre, 0) != 0 || id.size() < pre.size() + post.size() || id.compare(id.size() - post.size(), post.size(), post) != 0) return "";
    return id.substr(pre.size(), id.size() - pre.size() - post.size());
  };
  std::vector<std::string> v;
  std::string n;
  if (family == "qwen3_5_moe") {
    if (!(n = num("linear_k", "")).empty()) v.push_back("routers/lin/q36_router_lin_k" + n + ".dmr");
  } else if (family == "inkling2_loop") {
    if (!(n = num("mlp_k", "_h512")).empty()) {
      v.push_back("routers/ink_r_best/ink_router_full_k" + n + "_h512_b8.dmr");
      v.push_back("routers/ink_r_fast/ink_router_full_k" + n + "_h512_b8.dmr");
    }
    if (!(n = num("xtok_s", "_h512")).empty()) v.push_back("routers/ink_x_main/ink_router_xtok_s" + n + "_t0-5_b8.dmr");
  }
  return v;
}

// The engine file of router id `id` (cross-token: xtok), in the order of docs/MANIFEST-PRESETS.md; "" if none
static std::string router_file(const ms::Manifest* m, const std::string& dir, const std::string& family, const std::string& id, bool xtok,
                               std::vector<std::string>& notes) {
  std::string src = "routers/" + id + ".safetensors";
  if (m && !xtok) {
    if (const ms::RouterEntry* r = m->router(id)) src = r->path;
  } else if (m) {
    for (auto& x : m->raw["cross_token_routers"].arr)
      if (x["id"].s() == id) src = x["path"].str_or(src);
  }
  const std::string did = "router." + id + ".q8";
  const ms::Derived* d = m ? m->derived_of(did) : nullptr;
  ms::Derived dd = d ? *d : ms::Derived{did, "dmr1", "", {src}, ""};
  if (m) {
    const std::string p = m->shipped_path(dd);
    if (!p.empty()) return p;
  }
  for (auto& r : research_paths(family, id))
    if (exists(dir + "/" + r)) return dir + "/" + r;
  if (!m) return "";
  const std::string cached = m->derived_path(dd);
  if (exists(cached)) return cached;
  if (xtok || !exists(dir + "/" + src)) {
    notes.push_back((xtok ? "cross-token router " : "router ") + id + ": no file found, left out");
    return "";
  }
  mkdir(m->cache_dir().c_str(), 0755);
  ms::DeriveStats st;
  std::string err;
  const std::string tmp = cached + ".part";
  fprintf(stderr, "dmeng: deriving %s into %s (first use)\n", did.c_str(), cached.c_str());
  if (!ms::derive_router(dir + "/" + src, tmp, 8, false, st, err)) {
    notes.push_back("router " + id + ": " + err);
    return "";
  }
  if (!dd.sha256.empty()) {
    std::string e2;
    if (ms::sha256_file(tmp, e2, false) != dd.sha256) {
      notes.push_back("router " + id + ": the derived file does not match the manifest's sha256, left out");
      unlink(tmp.c_str());
      return "";
    }
  }
  if (rename(tmp.c_str(), cached.c_str()) != 0) return "";
  return cached;
}

static const Json* member(const Json* o, const std::string& k) { return o && o->kind == Json::OBJ ? o->get(k) : nullptr; }

bool resolve_preset(const std::string& dir, const std::string& profile_req, const std::string& mode_req, const std::string& derived_dir,
                    int64_t avail, Preset& p, std::string& err) {
  p = Preset();
  ms::Manifest man;
  const ms::Manifest* m = nullptr;
  if (exists(dir + "/moved-store.json")) {
    if (!man.load(dir, err)) return false;
    man.cache_override = derived_dir;
    m = &man;
    p.cache_dir = man.cache_dir();
    p.family = man.family;
  }
  if (!m) {  // a folder without a manifest (the tiny test models): no profile, the engine's built-in values
    p.profile = "none";
    p.profile_why = "no moved-store.json";
    p.mode = mode_req.empty() ? "exact" : mode_req;
    p.engine_mode = engine_mode_of(p.mode);
    return true;
  }
  Json bi;
  std::string jerr;
  if (!ms::json_parse(p.family == "inkling2_loop" ? BUILTIN_INK : BUILTIN_Q36, bi, jerr)) {
    err = "built-in presets: " + jerr;
    return false;
  }
  const Json& raw = m->raw;
  // the profile
  const std::string want = profile_req == "auto" ? "" : profile_req;
  if (want.empty()) {
    const Json* mg = member(member(member(&raw, "profiles"), "phone"), "min_available_gb");
    if (!mg) mg = member(member(member(&bi, "profiles"), "phone"), "min_available_gb");
    p.profile = device_profile(avail, mg ? mg->f(5.0) : 5.0, p.profile_why);
  } else {
    p.profile = want;
    p.profile_why = "chosen (--profile)";
  }
  const Json* mp = member(member(&raw, "profiles"), p.profile);
  const Json* bp = member(member(&bi, "profiles"), p.profile);
  if (!bp && !(mp && member(mp, "mode"))) {
    err = "profile " + p.profile + ": not in the manifest's profiles (mac, phone, phone-small)";
    return false;
  }
  const Json* md = member(&raw, "defaults");
  const Json* bd = member(&bi, "defaults");
  auto prof = [&](const char* k) { const Json* v = member(mp, k); return v ? v : member(bp, k); };
  auto dflt = [&](const char* k) {  // a profile's own value, then the manifest's defaults, then the built-in ones
    const Json* v = prof(k);
    if (!v) v = member(md, k);
    return v ? v : member(bd, k);
  };
  // the mode
  const Json* pm = prof("mode");
  p.mode = !mode_req.empty() ? mode_req : pm ? pm->str_or("exact") : "exact";
  const Json* me = member(member(member(&raw, "modes"), p.mode), "engine");
  if (!me) me = member(member(member(&bi, "modes"), p.mode), "engine");
  p.engine_mode = me ? me->str_or(engine_mode_of(p.mode)) : engine_mode_of(p.mode);
  // the in-memory weights
  const Json* pr = prof("resident");
  p.resident_id = pr ? pr->str_or("resident_q8") : "resident_q8";
  if (p.resident_id != "resident_q8") {
    const ms::Derived* d = m->derived_of(p.resident_id);
    const ms::Derived* d8 = m->derived_of("resident_q8");
    ms::Derived dd = d ? *d : ms::Derived{p.resident_id, "dmres1", "", d8 ? d8->from : std::vector<std::string>{"resident.safetensors"}, ""};
    std::string f = m->shipped_path(dd);
    if (f.empty() && exists(m->derived_path(dd))) f = m->derived_path(dd);
    if (f.empty()) {
      p.notes.push_back(p.resident_id + ": not in the folder, resident_q8 used");
      p.resident_id = "resident_q8";
    } else {
      p.resident = f;
    }
  }
  const Json* pc = prof("cache");
  p.cache = pc ? pc->str_or("auto") : "auto";
  // names
  if (const Json* rt = dflt("routers")) {
    for (auto& id : rt->arr) {
      const std::string f = router_file(m, dir, p.family, id.s(), false, p.notes);
      if (!f.empty()) {
        p.router_id = id.s();
        p.routers = f;
        break;
      }
    }
  }
  if (const Json* nm = dflt("names")) {
    if ((*nm)["rule"].s() == "confidence") {
      p.tau = (*nm)["tau"].f(0.5);
      p.cand = (int)(*nm)["candidates"].i(16);
    } else {
      p.m = (int)(*nm)["m"].i(8);
    }
  }
  if (const Json* xt = dflt("cross_token"); xt && xt->kind == Json::OBJ) {
    for (auto& id : (*xt)["routers"].arr) {
      const std::string f = router_file(m, dir, p.family, id.s(), true, p.notes);
      if (!f.empty()) {
        p.xtok_id = id.s();
        p.xtok = f;
        break;
      }
    }
    p.xtok_tau = (*xt)["tau"].f(0.5);
    p.rd_prio = !p.xtok.empty() && (*xt)["priority_reads"].kind == Json::BOOL && (*xt)["priority_reads"].b;
  }
  if (const Json* ev = dflt("eviction")) p.evict = ev->str_or("auto");
  if (const Json* dr = dflt("drafts")) p.drafts = dr->kind == Json::BOOL && dr->b;
  return true;
}

static std::string q(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

std::string Preset::json() const {
  std::ostringstream o;
  o << "{\"profile\":" << q(profile) << ",\"profile_why\":" << q(profile_why) << ",\"mode\":" << q(mode) << ",\"engine_mode\":" << q(engine_mode)
    << ",\"resident\":" << q(resident_id) << ",\"cache\":" << q(cache) << ",\"routers\":" << q(router_id) << ",\"routers_file\":" << q(routers)
    << ",\"names\":" << (tau > 0 ? "{\"rule\":\"confidence\",\"tau\":" + std::to_string(tau) + ",\"candidates\":" + std::to_string(cand) + "}"
                                 : "{\"rule\":\"top\",\"m\":" + std::to_string(m) + "}")
    << ",\"cross_token\":" << q(xtok_id) << ",\"cross_token_file\":" << q(xtok) << ",\"priority_reads\":" << (rd_prio ? "true" : "false")
    << ",\"eviction\":" << q(evict) << ",\"drafts\":" << (drafts ? "true" : "false") << ",\"notes\":[";
  for (size_t i = 0; i < notes.size(); i++) o << (i ? "," : "") << q(notes[i]);
  o << "]}";
  return o.str();
}

}  // namespace dm
