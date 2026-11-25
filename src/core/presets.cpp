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

// The built-in values, used for every key a manifest leaves out: one table for every model. What differs between models
// is declared in each one's manifest (modes, profiles, defaults); what is left here is a property of the device class
// each profile stands for, measured with the models of the weight repos (results/ios/p3-gpu/GPU-PREFILL.md: the Mac's
// GPU share of the prefill; results/parity/PARITY.md: run-ahead at depth 1; the phones: 8 readers, run-ahead limited to
// 0.2 of the token's compute, off while the readers are saturated, exact mode only). The run-ahead names a step's
// top_k picks ("top_k": the manifest's experts.top_k) unless a profile says otherwise.
static const char* BUILTIN = R"({
 "modes": {"exact": {"engine": "lossless"}, "fast": {"engine": "approx:drop"}},
 "profiles": {
  "mac": {"resident": "resident_q8", "cache": "auto", "prefill_gpu": {"projections": 0.4, "experts": 0.4}, "run_ahead": {"picks": "top_k", "depth": 1}},
  "phone": {"resident": "resident_q8", "cache": "slab", "min_available_gb": 5.0, "readers": 8,
            "run_ahead": {"picks": "top_k", "depth": 1, "budget": 0.2, "sat_skip": true, "modes": ["exact"]}},
  "phone-small": {"resident": "resident_q4", "cache": "slab", "readers": 8,
                  "run_ahead": {"picks": "top_k", "depth": 1, "budget": 0.2, "sat_skip": true, "modes": ["exact"]}}},
 "defaults": {"routers": [], "names": {"rule": "top", "m": 8}, "cross_token": null, "eviction": "auto", "drafts": false}})";

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
  if (!m) {  // a profile is declared in the manifest: a folder without one has none
    err = "profile " + profile_req + ": the folder has no moved-store.json (profiles are declared there; --profile none runs without one)";
    return false;
  }
  Json bi;
  std::string jerr;
  if (!ms::json_parse(BUILTIN, bi, jerr)) {
    err = "built-in presets: " + jerr;
    return false;
  }
  static const Json none;
  const Json& raw = m ? m->raw : none;
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
  if (!mp) {
    std::string have;
    for (auto& kv : member(&raw, "profiles") ? member(&raw, "profiles")->obj : std::vector<std::pair<std::string, Json>>{}) have += (have.empty() ? "" : ", ") + kv.first;
    err = "profile " + p.profile + ": not declared in the manifest's profiles (" + have + ")";
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
    p.rd_prio = !p.xtok.empty() && (*xt)["priority_reads"].kind == Json::BOOL && (*xt)["priority_reads"].b;
  }
  if (const Json* ev = dflt("eviction")) p.evict = ev->str_or("auto");
  if (const Json* lc = dflt("layer_caps")) p.layer_caps = lc->str_or("");
  if (const Json* dr = dflt("drafts")) p.drafts = dr->kind == Json::BOOL && dr->b;
  if (const Json* pg = prof("prefill_gpu"); pg && pg->kind == Json::OBJ) {  // per profile only (the device decides)
    p.gpu_frac = (*pg)["projections"].f(0);
    p.gpu_experts = (*pg)["experts"].f(0);
    if (p.family != "qwen3_5_moe" && p.family != "inkling2_loop" && p.family != "qwen4_exp" && (p.gpu_frac > 0 || p.gpu_experts > 0)) {
      p.notes.push_back("prefill_gpu: this model's prefill has no GPU split, CPU only");
      p.gpu_frac = p.gpu_experts = 0;
    }
#ifndef DM_METAL
    if (p.gpu_frac > 0 || p.gpu_experts > 0) p.notes.push_back("prefill_gpu: a build without Metal, CPU only");
    p.gpu_frac = p.gpu_experts = 0;
#endif
  }
  if (const Json* ra = prof("run_ahead"); ra && ra->kind == Json::OBJ) {  // per profile only (measured per device)
    p.run_ahead = (*ra)["picks"].kind == Json::STR && (*ra)["picks"].s() == "top_k" ? (int)m->top_k : (int)(*ra)["picks"].i(0);
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
    << ",\"eviction\":" << q(evict) << ",\"layer_caps\":" << q(layer_caps) << ",\"drafts\":" << (drafts ? "true" : "false") << ",\"prefill_gpu\":{\"projections\":" << gpu_frac
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
    << ",\"eviction\":" << qq(evict) << ",\"drafts\":" << drafts << ",\"prefill_gpu\":{\"projections\":" << gpu_frac
    << ",\"experts\":" << gpu_experts << "},\"run_ahead\":{\"picks\":" << run_ahead << ",\"depth\":" << run_ahead_depth << ",\"budget\":" << ra_budget
    << ",\"max_tok\":" << ra_max_tok << ",\"sat_skip\":" << (ra_sat_skip ? "true" : "false") << "},\"readers\":" << readers
    << ",\"background\":{\"ranks\":" << bg_ranks << ",\"gate\":" << bg_gate << "},\"layer_caps\":" << qq(layer_caps)
    << ",\"cross_token_tau\":" << xtok_tau << ",\"priority_reads\":" << rd_prio << ",\"prefill_chunk\":" << prefill_chunk << "}";
  return o.str();
}

std::vector<double> draft_cost_prior(const std::string& dir) {
  std::vector<double> v;
  if (!exists(dir + "/moved-store.json")) return v;
  ms::Manifest m;
  std::string err;
  if (!m.load(dir, err)) return v;
  const Json& cp = m.raw["drafting"]["cost_prior"];
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
  x.drafts = drafts > 0 ? 1 : 0;
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
    if (!P.xtok.empty()) {
      r.xtok_tau = P.xtok_tau;
      if (P.rd_prio) r.rd_prio = 1;
    }
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
  if (g.prefill_chunk) r.prefill_chunk = *g.prefill_chunk;
  else if (pf && !g.recording) r.prefill_chunk = 512;
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
  if (g.drafts) r.drafts = *g.drafts;
  else if (pf) r.drafts = P.drafts ? 1 : 0;
  return true;
}
}  // namespace dm
