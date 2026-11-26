// Modes, profiles and per-model defaults (include/dmeng/presets.h, docs/MANIFEST-PRESETS.md).
#include "dmeng/presets.h"

#include <sys/stat.h>
#include <unistd.h>

#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
#include <os/proc.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "../format/derive.h"
#include "../format/files.h"
#include "../format/manifest.h"

namespace dm {

using ms::Json;

// Where a key a folder's manifest leaves out comes from, in this order: the published manifest of the same family
// (published_defaults.inc, generated from python/convert/<family>/moved-store.json: an older manifest without
// `profiles`, or a folder without one run with --profile, resolves as the published one does), then the built-in values
// below, one table for every model. What differs between models is declared in the manifests (modes, profiles,
// defaults, drafting); what is left here is a property of the device class each profile stands for, measured with the
// models of the weight repos (results/ios/p3-gpu/GPU-PREFILL.md: the Mac's GPU share of the prefill;
// results/parity/PARITY.md: run-ahead at depth 1; the phones: 8 readers, run-ahead limited to 0.2 of the token's compute,
// off while the readers are saturated, exact mode only; both phones default to fast mode (exact on request);
// phone-small: the 4-bit resident with the confidence names rule, results/ios). The run-ahead names a step's top_k
// picks ("top_k": the manifest's experts.top_k) unless a profile says otherwise.
#include "published_defaults.inc"
static const char* BUILTIN = R"({
 "modes": {"exact": {"engine": "lossless"}, "fast": {"engine": "approx:drop"}},
 "profiles": {
  "mac": {"mode": "exact", "resident": "resident_q8", "cache": "auto", "prefill_gpu": {"projections": 0.4, "experts": 0.4},
          "run_ahead": {"picks": "top_k", "depth": 1}},
  "phone": {"mode": "fast", "resident": "resident_q8", "cache": "slab", "min_available_gb": 5.0, "readers": 8,
            "run_ahead": {"picks": "top_k", "depth": 1, "budget": 0.2, "sat_skip": true, "modes": ["exact"]}},
  "phone-small": {"mode": "fast", "resident": "resident_q4", "cache": "slab", "names": {"rule": "confidence", "tau": 0.5, "candidates": 16},
                  "readers": 8, "run_ahead": {"picks": "top_k", "depth": 1, "budget": 0.2, "sat_skip": true, "modes": ["exact"]}}},
 "defaults": {"routers": [], "names": {"rule": "top", "m": 8}, "cross_token": null, "eviction": "auto", "drafts": {"lookup": false, "head": false}}})";

// the published declarations of `family` (null: none, an unknown family)
static const Json* published(const std::string& family) {
  static Json all;
  static bool parsed = false;
  if (!parsed) {
    std::string e;
    if (!ms::json_parse(PUBLISHED_DEFAULTS, all, e)) fprintf(stderr, "dmeng: published defaults: %s\n", e.c_str());
    parsed = true;
  }
  return all.kind == Json::OBJ ? all.get(family) : nullptr;
}

// the family of a folder without a manifest: config.json's model_type ("" if none)
static std::string config_family(const std::string& dir) {
  Json cfg;
  std::string e;
  return ms::json_load(dir + "/config.json", cfg, e) ? cfg["model_type"].str_or("") : "";
}

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

// The engine file of router id `id` (cross-token: xtok), in the order of docs/MANIFEST-PRESETS.md; "" if none
static std::string router_file(const ms::Manifest* m, const std::string& dir, const std::string& id, bool xtok,
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
  auto checked = [&](const std::string& f) -> std::string {  // the manifest's size and sha256 of the file
    std::string e;
    if (m->check_derived(dd, f, e)) return f;
    notes.push_back((xtok ? "cross-token router " : "router ") + id + ": " + e + "; left out");
    return "";
  };
  if (m) {
    const std::string p = m->shipped_path(dd);
    if (!p.empty()) return checked(p);
  }
  if (!m) return "";
  const std::string cached = m->derived_path(dd);
  if (exists(cached)) return checked(cached);
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

// experts.top_k of the folder: its manifest's, else config.json's (a folder without a manifest)
static int top_k_of(const ms::Manifest* m, const std::string& dir) {
  if (m) return (int)m->top_k;
  Json cfg;
  std::string e;
  if (!ms::json_load(dir + "/config.json", cfg, e)) return 0;
  const Json& tc = cfg["text_config"].kind == Json::OBJ ? cfg["text_config"] : cfg;
  return (int)tc["num_experts_per_tok"].i(0);
}

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
  if (!m && (profile_req.empty() || profile_req == "auto")) {  // a folder without a manifest (the tiny test models): no
    // profile unless one is named, the engine's built-in values
    p.profile = "none";
    p.profile_why = "no moved-store.json";
    p.mode = mode_req.empty() ? "exact" : mode_req;
    p.engine_mode = engine_mode_of(p.mode);
    return true;
  }
  if (!m) {  // a named profile on a folder without a manifest: the family from config.json, its published declarations
    p.family = config_family(dir);
    if (!published(p.family)) {
      err = "profile " + profile_req + ": no moved-store.json and no known model_type in config.json (--profile none runs without one)";
      return false;
    }
    p.cache_dir = dir + "/.dmeng";
  }
  Json bi;
  std::string jerr;
  if (!ms::json_parse(BUILTIN, bi, jerr)) {
    err = "built-in presets: " + jerr;
    return false;
  }
  static const Json none;
  const Json& raw = m ? m->raw : none;
  const Json* pub = published(p.family);  // the family's published declarations (the second source of every key)
  // the profile
  const std::string want = profile_req == "auto" ? "" : profile_req;
  if (want.empty()) {
    const Json* mg = member(member(member(&raw, "profiles"), "phone"), "min_available_gb");
    if (!mg) mg = member(member(member(pub, "profiles"), "phone"), "min_available_gb");
    if (!mg) mg = member(member(member(&bi, "profiles"), "phone"), "min_available_gb");
    p.profile = device_profile(avail, mg ? mg->f(5.0) : 5.0, p.profile_why);
  } else {
    p.profile = want;
    p.profile_why = "chosen (--profile)";
  }
  const Json* mp = member(member(&raw, "profiles"), p.profile);
  const Json* fp = member(member(pub, "profiles"), p.profile);
  const Json* bp = member(member(&bi, "profiles"), p.profile);
  if (!mp && !fp && !bp) {
    err = "profile " + p.profile + ": not declared in the manifest's profiles (mac, phone, phone-small)";
    return false;
  }
  const Json* md = member(&raw, "defaults");
  const Json* fd = member(pub, "defaults");
  const Json* bd = member(&bi, "defaults");
  auto prof = [&](const char* k) {  // the folder's profile, the published one, the built-in one
    const Json* v = member(mp, k);
    if (!v) v = member(fp, k);
    return v ? v : member(bp, k);
  };
  auto dflt = [&](const char* k) {  // a profile's own value, then the defaults (the folder's, the published, the built-in)
    const Json* v = prof(k);
    if (!v) v = member(md, k);
    if (!v) v = member(fd, k);
    return v ? v : member(bd, k);
  };
  // the mode
  const Json* pm = prof("mode");
  p.mode = !mode_req.empty() ? mode_req : pm ? pm->str_or("exact") : "exact";
  const Json* me = member(member(member(&raw, "modes"), p.mode), "engine");
  if (!me) me = member(member(member(pub, "modes"), p.mode), "engine");
  if (!me) me = member(member(member(&bi, "modes"), p.mode), "engine");
  p.engine_mode = me ? me->str_or(engine_mode_of(p.mode)) : engine_mode_of(p.mode);
  // the in-memory weights
  const Json* pr = prof("resident");
  p.resident_id = pr ? pr->str_or("resident_q8") : "resident_q8";
  if (p.resident_id != "resident_q8" && !m) {
    p.notes.push_back(p.resident_id + ": no manifest, resident_q8 used");
    p.resident_id = "resident_q8";
  } else if (p.resident_id != "resident_q8") {
    const ms::Derived* d = m->derived_of(p.resident_id);
    const ms::Derived* d8 = m->derived_of("resident_q8");
    ms::Derived dd = d ? *d : ms::Derived{p.resident_id, "dmres1", "", d8 ? d8->from : std::vector<std::string>{"resident.safetensors"}, ""};
    std::string f = m->shipped_path(dd);
    if (f.empty() && exists(m->derived_path(dd))) f = m->derived_path(dd);
    if (!f.empty() && !m->check_derived(dd, f, err)) return false;
    if (f.empty()) {
      p.notes.push_back(p.resident_id + ": not in the folder, resident_q8 used");
      p.resident_id = "resident_q8";
    } else {
      p.resident = f;
    }
  }
  const Json* pc = prof("cache");
  p.cache = pc ? pc->str_or("auto") : "auto";
  if (const Json* cs = prof("cache_start"); cs && cs->kind == Json::OBJ) {
    const std::string rule = (*cs)["rule"].str_or("");
    char b[64];
    if (rule == "free_level") {
      snprintf(b, sizeof b, "free_level:%d", (int)(*cs)["floor_pct"].i(14));
    } else if (rule == "headroom") {
      snprintf(b, sizeof b, "headroom:%g", (*cs)["headroom_gb"].f(3.0));
    } else {
      err = "profile " + p.profile + ": cache_start.rule must be headroom or free_level";
      return false;
    }
    p.cache_start = b;
  }
  // names
  if (const Json* rt = dflt("routers")) {
    for (auto& id : rt->arr) {
      const std::string f = router_file(m, dir, id.s(), false, p.notes);
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
      const std::string f = router_file(m, dir, id.s(), true, p.notes);
      if (!f.empty()) {
        p.xtok_id = id.s();
        p.xtok = f;
        break;
      }
    }
    p.xtok_tau = (*xt)["tau"].f(0.5);
    p.rd_prio = !p.xtok.empty() && (*xt)["priority_reads"].kind == Json::BOOL && (*xt)["priority_reads"].b;  // (older manifests)
  }
  // speculative reads (router and cross-token names) by priority, nearest step first, instead of first in, first out:
  // defaults.priority_reads (every family's declaration), else the cross-token entry's (an older manifest)
  if (const Json* pr = dflt("priority_reads"); pr && pr->kind == Json::BOOL) p.rd_prio = pr->b && (!p.routers.empty() || !p.xtok.empty());
  if (const Json* ev = dflt("eviction")) p.evict = ev->str_or("auto");
  if (const Json* lc = dflt("layer_caps")) p.layer_caps = lc->str_or("");
  if (const Json* dr = dflt("drafts")) {  // {"lookup": B, "head": B}; a bool (an older manifest): the lookup drafts
    if (dr->kind == Json::BOOL) {
      p.draft_lookup = dr->b;
    } else if (dr->kind == Json::OBJ) {
      for (auto& kv : dr->obj)
        if (kv.first != "lookup" && kv.first != "head") return (err = "profile " + p.profile + ": drafts." + kv.first + ": not a drafting source (lookup, head)"), false;
      p.draft_lookup = (*dr)["lookup"].kind == Json::BOOL && (*dr)["lookup"].b;
      p.draft_head = (*dr)["head"].kind == Json::BOOL && (*dr)["head"].b;
    } else {
      return (err = "profile " + p.profile + ": drafts must be {\"lookup\": bool, \"head\": bool}"), false;
    }
  }
  if (const Json* pg = prof("prefill_gpu"); pg && pg->kind == Json::OBJ) {  // per profile only (the device decides)
    p.gpu_frac = (*pg)["projections"].f(0);
    p.gpu_experts = (*pg)["experts"].f(0);
#ifndef DM_METAL
    if (p.gpu_frac > 0 || p.gpu_experts > 0) p.notes.push_back("prefill_gpu: a build without Metal, CPU only");
    p.gpu_frac = p.gpu_experts = 0;
#endif
  }
  if (const Json* ra = prof("run_ahead"); ra && ra->kind == Json::OBJ) {  // per profile only (measured per device)
    p.run_ahead = (*ra)["picks"].kind == Json::STR && (*ra)["picks"].s() == "top_k" ? top_k_of(m, dir) : (int)(*ra)["picks"].i(0);
    p.run_ahead_depth = (int)(*ra)["depth"].i(1);
    p.ra_budget = (*ra)["budget"].f(0);
    p.ra_max_tok = (int)(*ra)["max_tok"].i(0);
    p.ra_sat_skip = (*ra)["sat_skip"].kind == Json::BOOL && (*ra)["sat_skip"].b;
    if ((*ra)["modes"].kind == Json::ARR) {  // on only in the listed modes (the phones: exact)
      bool in = false;
      for (auto& x : (*ra)["modes"].arr) in |= engine_mode_of(x.s()) == p.engine_mode;
      if (!in) p.run_ahead = 0;
    }
  }
  if (const Json* rd = prof("readers"); rd && rd->kind == Json::NUM) p.readers = (int)rd->i(0);
  if (const Json* bg = prof("background"); bg && bg->kind == Json::OBJ) {  // per profile only (the drive decides)
    p.bg_ranks = (int)(*bg)["ranks"].i(0);
    p.bg_gate = (int)(*bg)["gate"].i(0);
  }
  return true;
}

void names_rule(const Preset& p, bool profiled, bool m_given, int m, bool tau_given, double tau, bool cand_given, int cand, int& m_out,
                double& tau_out, int& cand_out) {
  m_out = m_given ? m : profiled ? p.m : m;
  cand_out = cand_given ? cand : profiled ? p.cand : cand;
  // tau_out < 0 = a plain top m (Routers::tau: 0 would be the confidence rule with a zero threshold, every candidate)
  if (tau_given) tau_out = tau > 0 ? tau : -1;             // the confidence rule, or (<= 0) a plain top m
  else if (m_given) tau_out = -1;                           // an explicit top m is not replaced by the profile's rule
  else tau_out = profiled && p.tau > 0 ? p.tau : (tau > 0 ? tau : -1);
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
    << ",\"resident\":" << q(resident_id) << ",\"cache\":" << q(cache) << ",\"cache_start\":" << q(cache_start) << ",\"routers\":" << q(router_id) << ",\"routers_file\":" << q(routers)
    << ",\"names\":" << (tau > 0 ? "{\"rule\":\"confidence\",\"tau\":" + std::to_string(tau) + ",\"candidates\":" + std::to_string(cand) + "}"
                                 : "{\"rule\":\"top\",\"m\":" + std::to_string(m) + "}")
    << ",\"cross_token\":" << q(xtok_id) << ",\"cross_token_file\":" << q(xtok) << ",\"priority_reads\":" << (rd_prio ? "true" : "false")
    << ",\"eviction\":" << q(evict) << ",\"layer_caps\":" << q(layer_caps) << ",\"drafts\":{\"lookup\":" << (draft_lookup ? "true" : "false") << ",\"head\":" << (draft_head ? "true" : "false")
    << "},\"prefill_gpu\":{\"projections\":" << gpu_frac
    << ",\"experts\":" << gpu_experts << "},\"run_ahead\":{\"picks\":" << run_ahead << ",\"depth\":" << run_ahead_depth << ",\"budget\":" << ra_budget
    << ",\"max_tok\":" << ra_max_tok << ",\"sat_skip\":" << (ra_sat_skip ? "true" : "false") << "},\"readers\":" << readers
    << ",\"background\":{\"ranks\":" << bg_ranks << ",\"gate\":" << bg_gate << "},\"notes\":[";
  for (size_t i = 0; i < notes.size(); i++) o << (i ? "," : "") << q(notes[i]);
  o << "]}";
  return o.str();
}

}  // namespace dm

namespace dm {
std::string Resolved::json() const {
  std::ostringstream o;
  auto qq = [](const std::string& s) {
    std::string r = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\') r += '\\';
      r += c;
    }
    return r + "\"";
  };
  o << "{\"profile\":" << qq(profile) << ",\"mode\":" << qq(mode) << ",\"routers\":" << qq(routers) << ",\"names\":"
    << (routers.empty() ? std::string("null")
                        : tau >= 0 ? "{\"rule\":\"confidence\",\"tau\":" + std::to_string(tau) + ",\"candidates\":" + std::to_string(cand) + "}"
                                  : "{\"rule\":\"top\",\"m\":" + std::to_string(m) + "}")
    << ",\"cross_token\":" << qq(xtok) << ",\"cache_gb\":" << cache_gb << ",\"cache_auto\":" << (cache_auto ? "true" : "false")
    << ",\"cache_start\":" << (cache_auto ? qq(cache_start) : std::string("null"))
    << ",\"eviction\":" << qq(evict) << ",\"drafts\":{\"lookup\":" << (draft == "lookup" || draft == "fixed" || draft == "both" ? "true" : "false")
    << ",\"head\":" << (draft == "head" || draft == "both" ? "true" : "false") << "},\"draft\":" << qq(draft) << ",\"prefill_gpu\":{\"projections\":" << gpu_frac
    << ",\"experts\":" << gpu_experts << "},\"run_ahead\":{\"picks\":" << run_ahead << ",\"depth\":" << run_ahead_depth << ",\"budget\":" << ra_budget
    << ",\"max_tok\":" << ra_max_tok << ",\"sat_skip\":" << (ra_sat_skip ? "true" : "false") << "},\"readers\":" << readers
    << ",\"background\":{\"ranks\":" << bg_ranks << ",\"gate\":" << bg_gate << "},\"layer_caps\":" << qq(layer_caps)
    << ",\"cross_token_tau\":" << xtok_tau << ",\"priority_reads\":" << rd_prio << ",\"prefill_chunk\":" << prefill_chunk << "}";
  return o.str();
}

std::vector<std::pair<std::string, std::string>> declared_prefill(const std::string& dir, const std::string& family, const std::string& profile) {
  std::vector<std::pair<std::string, std::string>> out;
  ms::Manifest m;
  std::string err;
  const bool has = exists(dir + "/moved-store.json") && m.load(dir, err);
  const Json* pub = published(has ? m.family : family);
  auto take = [&](const Json* pf) {  // the block's values, each key replacing an earlier one
    if (!pf || pf->kind != Json::OBJ) return;
    for (auto& kv : pf->obj) {
      const Json& v = kv.second;
      if (kv.first == "notes" || kv.first == "note") continue;
      char b[64];
      if (v.kind == Json::NUM) snprintf(b, sizeof b, "%.17g", v.f(0));
      else if (v.kind == Json::BOOL) snprintf(b, sizeof b, "%d", v.b ? 1 : 0);
      else if (v.kind == Json::STR) snprintf(b, sizeof b, "%s", v.s().c_str());
      else continue;  // notes, calibrations and the like
      bool found = false;
      for (auto& o : out)
        if (o.first == kv.first) o.second = b, found = true;
      if (!found) out.push_back({kv.first, b});
    }
  };
  // the model's: the folder manifest's "prefill", else the family's published one
  const Json* pf = has ? member(&m.raw, "prefill") : nullptr;
  take(pf ? pf : member(pub, "prefill"));
  // then the profile's own "prefill" block (measured per device), the folder's profile, else the published one
  if (!profile.empty() && profile != "none") {
    const Json* pp = has ? member(member(member(&m.raw, "profiles"), profile), "prefill") : nullptr;
    take(pp ? pp : member(member(member(pub, "profiles"), profile), "prefill"));
  }
  return out;
}

std::string prefill_json(const std::vector<std::pair<std::string, std::string>>& kv) {
  std::string o = "{";
  for (size_t i = 0; i < kv.size(); i++) o += (i ? ",\"" : "\"") + kv[i].first + "\":" + q(kv[i].second);
  return o + "}";
}

std::vector<double> draft_cost_prior(const std::string& dir) {
  std::vector<double> v;
  ms::Manifest m;
  std::string err;
  const bool has = exists(dir + "/moved-store.json") && m.load(dir, err);
  // the folder's manifest, else the family's published one (a manifest without the key: the published prior)
  const Json* pub = published(has ? m.family : config_family(dir));
  const Json* cpp = has ? member(member(&m.raw, "drafting"), "cost_prior") : nullptr;
  if (!cpp) cpp = member(member(pub, "drafting"), "cost_prior");
  static const Json none;
  const Json& cp = cpp ? *cpp : none;
  // a prior that states where it comes from: "measured" (step costs of passes) or "estimated" (from measured parts)
  if (cp.kind != Json::OBJ || (cp["measured"].kind != Json::STR && cp["estimated"].kind != Json::STR)) return v;
  for (auto& x : cp["pass_ms_over_step_ms"].arr) v.push_back(x.f(0));
  return v;
}

Resolved RunOpts::resolved(double cache_gb_in_force) const {
  Resolved x;
  x.profile = profiled ? preset.profile : "none";
  x.mode = mode;
  x.routers = routers;
  x.m = m;
  x.tau = tau;
  x.cand = cand;
  x.xtok = xtok;
  x.cache_gb = cache_gb_in_force;
  x.cache_auto = cache != "fixed";
  x.cache_start = cache_start;
  x.evict = evict;
  x.draft = draft;
  x.gpu_frac = gpu_frac;
  x.gpu_experts = gpu_experts;
  x.run_ahead = run_ahead > 0 ? run_ahead : 0;
  x.run_ahead_depth = run_ahead_depth;
  x.ra_budget = ra_budget;
  x.ra_max_tok = ra_max_tok;
  x.ra_sat_skip = ra_sat_skip;
  x.readers = readers;
  x.bg_ranks = bg_ranks;
  x.bg_gate = bg_ranks > 0 ? bg_gate : 0;
  x.layer_caps = layer_caps;
  x.xtok_tau = xtok_tau;
  x.rd_prio = rd_prio;
  x.prefill_chunk = prefill_chunk;
  return x;
}

bool resolve_run(const std::string& dir, const std::string& profile_req, const Given& g, const std::string& derived_dir, int64_t avail,
                 RunOpts& r, std::string& err) {
  Preset& P = r.preset;
  P = Preset();
  if (profile_req != "none" && !resolve_preset(dir, profile_req, g.mode ? *g.mode : "", derived_dir, avail, P, err)) return false;
  r.profiled = profile_req != "none" && P.profile != "none";
  const bool pf = r.profiled;
  r.notes = P.notes;
  // the mode
  r.mode = pf ? P.engine_mode : engine_mode_of(g.mode ? *g.mode : r.mode);
  // the in-memory weights
  if (g.resident) r.resident = *g.resident;
  else if (pf && !P.resident.empty()) r.resident = P.resident;
  // names: the profile's routers (and cross-token routers) only when no early address was given
  const bool addr = g.routers || g.xtok || g.other_address;
  if (g.routers) r.routers = *g.routers == "none" ? "" : *g.routers;
  if (g.xtok) r.xtok = *g.xtok == "none" ? "" : *g.xtok;
  r.routers_from_profile = pf && !addr && !P.routers.empty();
  if (pf && !addr) {
    r.routers = P.routers;
    r.xtok = P.xtok;
    if (!P.xtok.empty()) r.xtok_tau = P.xtok_tau;
    if (P.rd_prio) r.rd_prio = 1;
  }
  if (g.xtok_tau) r.xtok_tau = *g.xtok_tau;
  if (g.rd_prio) r.rd_prio = *g.rd_prio ? 1 : 0;
  names_rule(P, r.routers_from_profile, (bool)g.m, g.m ? *g.m : r.m, (bool)g.tau, g.tau ? *g.tau : r.tau, (bool)g.cand, g.cand ? *g.cand : r.cand, r.m,
             r.tau, r.cand);
  // the cache
  r.evict = g.evict ? *g.evict : pf ? P.evict : r.evict;
  if (g.layer_caps) r.layer_caps = *g.layer_caps;
  else if (pf && !P.layer_caps.empty()) r.layer_caps = P.layer_caps;
  if (g.cache_gb) r.cache = "fixed", r.cache_gb = *g.cache_gb;
  else if (g.cache_auto) r.cache = "auto";
  else if (pf) r.cache = P.cache;
  if (pf) r.cache_start = P.cache_start;
  // the prompt path
  r.gpu_frac = g.gpu_frac ? *g.gpu_frac : pf ? P.gpu_frac : r.gpu_frac;
  r.gpu_experts = g.gpu_experts ? *g.gpu_experts : pf ? P.gpu_experts : r.gpu_experts;
  if (g.gpu_off) r.gpu_frac = r.gpu_experts = 0;
  // the prompt: an explicit --prefill-chunk; under a profile the batched path with the model's declared sub-chunk and
  // span (0: the manifest's "prefill", as the C API); else (or while recording a routing) the front end's own value
  if (g.prefill_chunk) r.prefill_chunk = *g.prefill_chunk;
  else if (pf && !g.recording) r.prefill_chunk = 0;
  // run-ahead, readers, background reads
  r.run_ahead = g.run_ahead ? *g.run_ahead : pf ? P.run_ahead : r.run_ahead;
  r.run_ahead_depth = g.run_ahead_depth ? *g.run_ahead_depth : pf ? P.run_ahead_depth : r.run_ahead_depth;
  r.ra_budget = g.ra_budget ? *g.ra_budget : pf ? P.ra_budget : r.ra_budget;
  r.ra_max_tok = g.ra_max_tok ? *g.ra_max_tok : pf ? P.ra_max_tok : r.ra_max_tok;
  r.ra_sat_skip = g.ra_sat_skip ? *g.ra_sat_skip : pf ? P.ra_sat_skip : r.ra_sat_skip;
  if (g.readers) r.readers = *g.readers;
  else if (pf && P.readers > 0) r.readers = P.readers;
  r.bg_ranks = g.bg_ranks ? *g.bg_ranks : pf ? P.bg_ranks : r.bg_ranks;
  r.bg_gate = g.bg_gate ? *g.bg_gate : pf ? P.bg_gate : r.bg_gate;
  // drafting
  if (g.draft) r.draft = *g.draft;
  else if (g.drafts) r.draft = *g.drafts > 0 ? "lookup" : "none";
  else if (pf) r.draft = P.draft_lookup && P.draft_head ? "both" : P.draft_lookup ? "lookup" : P.draft_head ? "head" : "none";
  r.drafts = r.draft != "none" ? 1 : 0;
  // no early names under --drafts 1: on the real 35B and Inkling (agent turns, 6.6 / 7.8 GB) naming the lookup source's
  // picks 4 positions ahead read 40-70 MB per token more than plain decoding and cancelled the passes' gain (early 0: fewer
  // MB per token than plain decoding at the same tokens per step); --early N still sets it
  r.early = g.early ? *g.early : r.drafts > 0 ? 0 : r.early;
  return true;
}

namespace {
struct RunFlag {
  const char* name;
  const char* alias;
  bool (*set)(const std::string& v, Given& g);
};
template <class T>
bool num(const std::string& v, T& out) {
  char* e = nullptr;
  const double d = std::strtod(v.c_str(), &e);
  if (v.empty() || *e) return false;
  out = (T)d;
  return true;
}
#define RF_INT(field) [](const std::string& v, Given& g) { int x; return num(v, x) && ((g.field = x), true); }
#define RF_DBL(field) [](const std::string& v, Given& g) { double x; return num(v, x) && ((g.field = x), true); }
#define RF_STR(field) [](const std::string& v, Given& g) { return (g.field = v), true; }
#define RF_NONE(field) [](const std::string& v, Given& g) { return (g.field = v.empty() ? std::string("none") : v), true; }
const RunFlag RUN_FLAGS[] = {
    {"--mode", nullptr, RF_STR(mode)},
    {"--resident", nullptr, RF_STR(resident)},
    {"--routers", nullptr, RF_NONE(routers)},
    {"--xtok", nullptr, RF_NONE(xtok)},
    {"--m", nullptr, RF_INT(m)},
    {"--rt-tau", nullptr, RF_DBL(tau)},
    {"--rt-cand", nullptr, RF_INT(cand)},
    {"--xtok-tau", nullptr, RF_DBL(xtok_tau)},
    {"--rd-prio", nullptr, [](const std::string& v, Given& g) { int x; return num(v, x) && ((g.rd_prio = x != 0), true); }},
    {"--evict", nullptr, RF_STR(evict)},
    {"--layer-caps", nullptr, RF_STR(layer_caps)},
    {"--cache-gb", nullptr, [](const std::string& v, Given& g) {
       if (v == "auto") return (g.cache_auto = true, g.cache_gb.reset()), true;
       double x;
       return num(v, x) && ((g.cache_gb = x, g.cache_auto = false), true);
     }},
    {"--prefill-gpu-frac", "--gpu-frac", RF_DBL(gpu_frac)},
    {"--prefill-gpu-experts", "--gpu-experts", RF_DBL(gpu_experts)},
    {"--prefill-gpu", nullptr, [](const std::string& v, Given& g) { return (v == "off" || v == "on") && ((g.gpu_off = v == "off"), true); }},
    {"--prefill-chunk", "--chunk", RF_INT(prefill_chunk)},
    {"--run-ahead", nullptr, RF_INT(run_ahead)},
    {"--run-ahead-depth", nullptr, RF_INT(run_ahead_depth)},
    {"--run-ahead-budget", nullptr, RF_DBL(ra_budget)},
    {"--run-ahead-max", nullptr, RF_INT(ra_max_tok)},
    {"--run-ahead-sat-skip", nullptr, [](const std::string& v, Given& g) { int x; return num(v, x) && ((g.ra_sat_skip = x != 0), true); }},
    {"--readers", nullptr, RF_INT(readers)},
    {"--bg-ranks", nullptr, RF_INT(bg_ranks)},
    {"--bg-gate", nullptr, RF_INT(bg_gate)},
    {"--drafts", nullptr, RF_INT(drafts)},
    {"--draft", nullptr, RF_STR(draft)},
    {"--early", nullptr, RF_INT(early)},
};
#undef RF_INT
#undef RF_DBL
#undef RF_STR
#undef RF_NONE
const RunFlag* find_flag(const std::string& f) {
  for (const RunFlag& r : RUN_FLAGS)
    if (f == r.name || (r.alias && f == r.alias)) return &r;
  return nullptr;
}
}  // namespace

std::string run_flag_name(const std::string& flag) {
  const RunFlag* r = find_flag(flag);
  return r ? r->name : "";
}

bool run_flag(const std::string& flag, const std::string& value, Given& g, std::string& err) {
  const RunFlag* r = find_flag(flag);
  if (!r) return (err = flag + ": not a run setting"), false;
  if (!r->set(value, g)) return (err = flag + ": bad value '" + value + "'"), false;
  return true;
}
}  // namespace dm
