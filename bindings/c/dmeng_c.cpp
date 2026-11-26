// dmeng C API over the core (see dmeng_c.h): the open sequence of tools/dmserve.cpp, the prefix cache, the sampler.
#include "dmeng_c.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <TargetConditionals.h>
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
#include <os/proc.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "derive.h"
#include "files.h"
#include "dmeng/address.h"
#include "dmeng/core.h"
#include "dmeng/model.h"
#include "dmeng/model_lm.h"
#include "dmeng/presets.h"
#include "dmeng/sampler.h"
#include "dmeng/session.h"
#include "dmeng/weights.h"
#include "manifest.h"
#include "../../src/serving/memwatch.h"
#include "../../src/core/gpu_check.h"

using namespace dm;

struct dm_engine {
  std::unique_ptr<Model> M;
  std::unique_ptr<Core> C;
  std::unique_ptr<Routers> R;
  std::unique_ptr<Xtok> X;  // cross-token routers, wrapping R
  std::string preset_json = "null";
  Resolved res;  // the settings after precedence (presets.h), in dm_stats_json under "resolved"
  std::unique_ptr<Planner> PL;
  std::unique_ptr<Pool> prefill_pool;  // dm_sync's compute threads when they differ from decode's
  std::unique_ptr<State> S;
  std::unique_ptr<ModelLM> lm;
  std::unique_ptr<PrefixCache> pc;
  std::string err, stats, routers_path, res_file, last_prefill = "{}";  // the caller's strings are not kept (only copies)
  std::string cache_why = "null";  // the automatic cache rule's numbers (memwatch.h auto_cache_gb), as dmeng and dmserve print them
  dm_open_opts o{};
  qos_class_t qos = QOS_CLASS_USER_INTERACTIVE;
  uint64_t rs_urgent = 0, rs_pf = 0, rs_pf_n = 0, rs_unused = 0, rs_busy = 0, rs_bytes = 0, rs_steps = 0;  // reads at the last stats call
  std::chrono::steady_clock::time_point rs_t = std::chrono::steady_clock::now();
  std::mutex busy;                     // held by dm_sync / dm_step (dm_set_cache_bytes applies at once when free)
  std::atomic<bool> cancel_req{false};
  int64_t avail_open = 0, slab_auto = 0;  // iOS: os_proc_available_memory at open, the slab the sizing rule chose
  int64_t steps = 0;
  int64_t rs_steps_dt = 0;
  uint64_t rs_dt[7] = {0};  // the readers' time by reads in flight at the last stats call (Readers::depth_times)
  uint32_t ra_lcg = 12345;  // run_ahead_ab: the arm choice per step, as dmeng's DMENG_RA_AB
  double ab_n[2] = {0, 0}, ab_ms[2] = {0, 0}, ab_wait[2] = {0, 0}, ab_mb[2] = {0, 0}, ab_urg[2] = {0, 0}, ab_named[2] = {0, 0};
};

static thread_local std::string g_err;

static qos_class_t qos_of(int q) {
  switch (q) {
    case DM_QOS_USER_INITIATED: return QOS_CLASS_USER_INITIATED;
    case DM_QOS_DEFAULT: return QOS_CLASS_DEFAULT;
    case DM_QOS_UTILITY: return QOS_CLASS_UTILITY;
    default: return QOS_CLASS_USER_INTERACTIVE;
  }
}

static int sysctl_int(const char* k) {
  int v = 0;
  size_t n = sizeof v;
  return sysctlbyname(k, &v, &n, nullptr, 0) == 0 ? v : 0;
}

static bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

// The router file of a manifest router id: the derived 8-bit file in the cache folder (derived from
// routers/<id>.safetensors on first use if absent).
static bool router_path(const std::string& dir, const std::string& derived_dir, const std::string& want, std::string& path,
                        std::string& err) {
  ms::Manifest m;
  if (!m.load(dir, err)) return false;
  m.cache_override = derived_dir;
  const ms::RouterEntry* r = m.router(want == "default" ? "" : want);
  if (!r) {
    err = "no router " + want + " in " + dir + "/moved-store.json";
    return false;
  }
  const std::string id = "router." + r->id + ".q8";
  const ms::Derived* d = m.derived_of(id);
  ms::Derived dd = d ? *d : ms::Derived{id, "dmr1", "", {r->path}};
  path = m.derived_path(dd);
  if (exists(path)) return true;
  mkdir(m.cache_dir().c_str(), 0755);
  ms::DeriveStats st;
  const std::string tmp = path + ".part";
  fprintf(stderr, "dmeng: deriving %s into %s (first use)\n", id.c_str(), path.c_str());
  if (!ms::derive_router(dir + "/" + r->path, tmp, 8, false, st, err)) return false;
  if (!dd.sha256.empty()) {
    std::string e2;
    if (ms::sha256_file(tmp, e2, false) != dd.sha256) {
      err = tmp + ": derived router does not match the manifest's sha256";
      return false;
    }
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) {
    err = "cannot move " + tmp + " into place";
    return false;
  }
  return true;
}

static Mem mem_c(int m) {
  return m == DM_MEM_PLAIN ? Mem::PLAIN : m == DM_MEM_PURGEABLE ? Mem::PURGEABLE : m == DM_MEM_FILE ? Mem::FILE : Mem::MLOCK;
}

struct QosScope {  // the caller's QoS for the duration of a call
  explicit QosScope(qos_class_t q) { pthread_set_qos_class_self_np(q, 0); }
};

extern "C" {

const char* dm_gpu_check(int threads, double seconds) {
  static thread_local std::string js;
  std::string text;
  js = gpu_gemm_selftest(threads > 0 ? threads : 4, seconds > 0 ? seconds : 2, &text);
  fprintf(stderr, "%s", text.c_str());
  return js.c_str();
}

void dm_default_opts(dm_open_opts* o) {
  std::memset(o, 0, sizeof *o);
  o->qos = DM_QOS_USER_INTERACTIVE;
  o->core_class = DM_CORES_ANY;
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  o->memory_mode = DM_MEM_MLOCK;  // P2 (results/ios/p2/P2.md section 0b): a locked slab, the resident locked
  o->res_memory_mode = DM_MEM_MLOCK;
#else
  o->memory_mode = DM_MEM_MLOCK;
  o->res_memory_mode = DM_MEM_MLOCK;
#endif
  o->disk_gb = 4.0;
  o->ctx = 16384;
  o->file_check = 1;
}

dm_engine* dm_open(const char* model_dir, const dm_open_opts* opts) {
  if (!model_dir) {
    g_err = "model_dir is NULL";
    return nullptr;
  }
  auto e = std::make_unique<dm_engine>();
  if (opts) e->o = *opts;
  else dm_default_opts(&e->o);
  dm_open_opts& o = e->o;
  e->qos = qos_of(o.qos);
  QosScope qs(e->qos);
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  e->avail_open = (int64_t)os_proc_available_memory();  // before anything is loaded
#else
  // tests only (tests/resolve_compare.py): the iPhone's sizing rules on a Mac, as if iOS gave the app this many bytes
  if (const char* t = getenv("DMENG_TEST_AVAIL_OPEN")) e->avail_open = (int64_t)atoll(t);
#endif
  std::string err, family;
  LoadOpts lo;
  const std::string derived = o.derived_dir ? o.derived_dir : "";
  // the settings: explicit options > the profile > the manifest's defaults > the C API's own values (resolve_run, the
  // resolver dmeng and dmserve use too); an option left at zero / NULL is not given, a negative one is "off"
  const std::string prof = o.profile && *o.profile ? o.profile : "auto";
  // the fields through the front ends' own flag spelling (presets.h run_flag): one reading of every value
  Given g;
  auto F = [&](const char* flag, const std::string& v) {
    std::string e;
    if (!run_flag(flag, v, g, e) && err.empty()) err = e;
  };
  auto num = [](double x) {
    char b[32];
    snprintf(b, sizeof b, "%.17g", x);
    return std::string(b);
  };
  if (o.mode && *o.mode) F("--mode", o.mode);
  if (o.resident) F("--resident", o.resident);
  if (o.routers && *o.routers) F("--routers", o.routers);  // a file or "none"
  else if (o.routers_id && std::strcmp(o.routers_id, "none") == 0) F("--routers", "none");
  else if (o.routers_id && *o.routers_id) {
    std::string rp;
    if (!router_path(model_dir, derived, o.routers_id, rp, err)) {
      g_err = "routers: " + err;
      return nullptr;
    }
    F("--routers", rp);
  }
  if (o.xtok) F("--xtok", std::strcmp(o.xtok, "none") == 0 ? "none" : o.xtok);
  if (o.router_m > 0) F("--m", std::to_string(o.router_m));
  if (o.router_tau != 0) F("--rt-tau", num(o.router_tau));
  if (o.router_cand > 0) F("--rt-cand", std::to_string(o.router_cand));
  if (o.evict && *o.evict) F("--evict", o.evict);
  if (o.layer_caps == 1) F("--layer-caps", "uniform");
  else if (o.layer_caps == 2) F("--layer-caps", "none");
  if (o.cache_bytes > 0) F("--cache-gb", num((double)o.cache_bytes / 1e9));
  if (o.prefill_gpu_frac != 0) F("--prefill-gpu-frac", num(std::max(0.0, o.prefill_gpu_frac)));  // negative: off
  if (o.prefill_gpu_experts != 0) F("--prefill-gpu-experts", num(std::max(0.0, o.prefill_gpu_experts)));
  if (o.run_ahead != 0) F("--run-ahead", std::to_string(std::max(0, o.run_ahead)));
  if (o.run_ahead_depth > 0) F("--run-ahead-depth", std::to_string(o.run_ahead_depth));
  if (o.run_ahead_budget != 0) F("--run-ahead-budget", num(std::max(0.0, o.run_ahead_budget)));
  if (o.run_ahead_max != 0) F("--run-ahead-max", std::to_string(std::max(0, o.run_ahead_max)));
  if (o.run_ahead_sat_skip != 0) F("--run-ahead-sat-skip", o.run_ahead_sat_skip > 0 ? "1" : "0");
  if (o.readers > 0) F("--readers", std::to_string(o.readers));
  // model_options entries "run:--FLAG=VALUE": any dmeng run flag (presets.h run_flag), read after the fields above (so a
  // measurement can set what the struct has no field for: --bg-ranks, --layer-caps auto:fit, --mode fast-eps, --rd-prio)
  std::string model_opts;
  {
    const std::string mo = o.model_options ? o.model_options : "";
    for (size_t i = 0; i < mo.size();) {
      size_t j = mo.find(';', i);
      if (j == std::string::npos) j = mo.size();
      const std::string kv = mo.substr(i, j - i);
      if (kv.rfind("run:", 0) == 0) {
        const size_t q = kv.find('=');
        F(kv.substr(4, q == std::string::npos ? std::string::npos : q - 4).c_str(), q == std::string::npos ? "" : kv.substr(q + 1));
      } else if (!kv.empty()) {
        model_opts += (model_opts.empty() ? "" : ";") + kv;
      }
      i = j + 1;
    }
  }
  if (!err.empty()) {
    g_err = "options: " + err;
    return nullptr;
  }
  RunOpts ro;  // the C API's own values where nothing else sets one
  ro.cache_gb = 3.3;
  ro.readers = CoreOpts().readers;
  ro.evict = CoreOpts().evict;
  if (!resolve_run(model_dir, prof, g, derived, e->avail_open > 0 ? e->avail_open : -1, ro, err)) {
    g_err = "profile: " + err;
    return nullptr;
  }
  const Preset& P = ro.preset;
  const bool profiled = ro.profiled;
  if (profiled) e->preset_json = P.json();
  if (ro.drafts > 0) {  // (no drafting in the C API: a manifest default cannot turn on what this front end does not do)
    ro.notes.push_back("drafts: the C API has no drafting, off");
    ro.drafts = 0;
  }
  // the prompt's sub-chunk is the model's own here (the C API does not set it; dmeng and dmserve use 512): printed as 0
  ro.prefill_chunk = 0;
  for (auto& n : ro.notes) fprintf(stderr, "dmeng: %s\n", n.c_str());
  const std::string mode = ro.mode;
  std::string res_file = o.resident ? o.resident : "";
  if (res_file.empty() && o.resident_bits == 0) res_file = ro.resident;
  if (res_file.empty() && o.resident_bits == 4) {  // the 4-bit derived variant (P2), by the engine's derived naming
    ms::Manifest mf;
    if (!mf.load(model_dir, err)) {
      g_err = "load: " + err;
      return nullptr;
    }
    mf.cache_override = derived;
    const ms::Derived* d8 = mf.derived_of("resident_q8");
    const ms::Derived* d4 = mf.derived_of("resident_q4");
    ms::Derived dd = d4 ? *d4 : ms::Derived{"resident_q4", "dmres1", "", d8 ? d8->from : std::vector<std::string>{"resident.safetensors"}};
    res_file = mf.shipped_path(dd);  // a folder from the Hub ships it in derived/
    if (res_file.empty()) res_file = mf.derived_path(dd);
    if (!exists(res_file)) {
      g_err = "load: no 4-bit resident " + res_file + " (ms_derive resident DIR --bits 4)";
      return nullptr;
    }
    if (!mf.check_derived(dd, res_file, err)) {
      g_err = "load: " + err;
      return nullptr;
    }
  } else if (o.resident_bits != 0 && o.resident_bits != 8 && o.resident_bits != 4) {
    g_err = "resident_bits: 8 or 4";
    return nullptr;
  }
  e->res_file = res_file;
  const Mem res_mem = o.res_memory_mode == DM_MEM_PLAIN ? Mem::PLAIN : o.res_memory_mode == DM_MEM_FILE ? Mem::FILE : Mem::MLOCK;
  if (resolve_model(model_dir, res_file, res_mem, lo, family, err, derived, o.embeddings ? o.embeddings : "", profiled ? P.profile : ""))
    e->M = make_model(family, err);
  if (!e->M || !e->M->load(lo, err)) {
    g_err = "load: " + err;
    return nullptr;
  }
  e->routers_path = ro.routers;
  CoreOpts co;
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  co.threads = 4;  // P0: 4 threads for the 8-bit resident on 2 P + 4 E cores (6 for a 4-bit resident)
#else
  co.threads = 8;
#endif
  if (o.threads > 0) co.threads = o.threads;
  if (o.core_class == DM_CORES_PERF) {
    const int p = sysctl_int("hw.perflevel0.physicalcpu");
    if (p > 0) co.threads = std::min(co.threads, p);
  }
  co.readers = ro.readers;
  co.cache_gb = ro.cache_gb;
  co.mem = mem_c(o.memory_mode);
  // (slab is the iPhone's rule; on macOS a profile's slab is the Mac's automatic rule, as in dmeng and dmserve)
  if (ro.cache == "slab" && e->avail_open <= 0) ro.cache = "auto";
  if (ro.cache == "auto" && (e->avail_open > 0 || co.mem != Mem::MLOCK)) ro.cache = "fixed";
  if (ro.cache == "auto") {  // the Mac rule
    const ModelInfo& mi0 = e->M->info();
    double start = 0;
    std::string why;
    MemRule mr;
    if (!set_start_rule(mr, ro.cache_start, err)) {
      g_err = "cache: " + err;
      return nullptr;
    }
    mr.kv_positions = o.ctx > 0 ? o.ctx : 16384;
    {
      struct stat rs;  // the router file is loaded after the cache is sized (as in dmeng and dmserve)
      if (!e->routers_path.empty() && stat(e->routers_path.c_str(), &rs) == 0) mr.later_gb = rs.st_size / 1e9;
    }
    const double ceil = auto_cache_gb(mr, e->M->pos_bytes(), std::max(o.ckpt_gb > 0 ? o.ckpt_gb : 0.25, SessionOptions().ckpt_min * [&] {  // the checkpoints' memory (a fresh
      std::string e2;                                                                // state's fixed part, only reserved)
      auto tmp = e->M->new_state(o.ctx > 0 ? o.ctx : 16384, e2);
      return tmp ? (double)e->M->fixed_bytes(*tmp) / 1e9 : 0.0;
    }()), store_bytes_gb(model_dir, mi0.layer_file, mi0.layers), start, why);
    co.cache_gb = std::min(ceil, start);
    e->cache_why = why;
  }
  if (o.cache_bytes <= 0 && e->avail_open > 0 && co.mem != Mem::FILE) {  // iOS: the slab from the memory the app has
    ro.cache = "slab";
    struct stat st;
    const int64_t res = stat(lo.resident.c_str(), &st) == 0 ? (int64_t)st.st_size : (int64_t)2.1e9;
    const int64_t rt = !e->routers_path.empty() && stat(e->routers_path.c_str(), &st) == 0 ? (int64_t)st.st_size : 0;
    const int64_t slab = std::min<int64_t>((int64_t)8.0e9, std::max<int64_t>((int64_t)0.6e9, e->avail_open - res - rt - (int64_t)0.3e9 - (int64_t)0.9e9));  // follows what iOS makes available: phones with more memory get a larger cache
    e->slab_auto = slab;
    co.cache_gb = slab / 1e9;
    fprintf(stderr, "dmeng: expert slab %.2f GB = available %.2f - resident %.2f - router %.2f - state 0.30 - margin 0.90 GB\n",
            slab / 1e9, e->avail_open / 1e9, res / 1e9, rt / 1e9);
  }
  if (o.split > 0) co.split = o.split;
  // the eviction policy: the core's own default (auto: hazard, uniform per-layer caps below 4 x K x layers slots)
  // unless the caller names one
  co.evict = ro.evict;
  if (!ro.layer_caps.empty()) co.layer_caps = ro.layer_caps;
  co.qos = (int)e->qos;
  if (o.reader_qos > 0) co.reader_qos = (int)qos_of(o.reader_qos - 1);
  if (o.reader_io_low != 0) co.reader_io_low = o.reader_io_low > 0;  // 0: CoreOpts' platform default
  e->C = std::make_unique<Core>();
  if (!e->C->init(e->M->info(), co, model_dir, err) || !e->M->bind(*e->C, err)) {
    g_err = "core: " + err;
    return nullptr;
  }
  e->C->cache.file_check = o.file_check;
  e->C->cache.file_evict_hint = o.file_evict_hint;
  e->C->cache.fa.on = o.file_adapt != 0 && co.mem == Mem::FILE;
  if (o.file_min_bytes > 0) e->C->cache.fa.min_bytes = o.file_min_bytes;
  if (!e->C->cache.core_init(o.file_core_bytes, err)) {
    g_err = "core: " + err;
    return nullptr;
  }
  if (!e->routers_path.empty() && o.plan > 0) {  // the read planner over the router: names only while they pay
    e->PL = std::make_unique<Planner>();
    if (!e->PL->load({e->routers_path}, *e->C, err)) {
      g_err = "planner: " + err;
      return nullptr;
    }
    e->PL->budget_on = true;
    e->PL->budget_mode = o.plan == 2 ? 2 : 1;
    if (o.plan_u > 0) e->PL->u = o.plan_u;
    e->C->hook = e->PL.get();
    e->C->cache.rd.prio = true;  // as dmeng --prefetch plan
    e->C->cancel_stale = true;
  } else if (!e->routers_path.empty()) {
    e->R = std::make_unique<Routers>();
    e->R->mem = res_mem;
    if (!e->R->load(e->routers_path, *e->C, err)) {
      g_err = "routers: " + err;
      return nullptr;
    }
    // the names rule as resolved (precedence: presets.h)
    e->R->m = ro.m;
    if (ro.tau > 0) e->R->tau = ro.tau;
    e->R->cand = ro.cand;
    e->C->hook = e->R.get();
    if (ro.bg_ranks > 0) {  // background reads: the routers' next ranks for the drive's idle time (as dmeng; its defaults)
      e->R->bg_ranks = ro.bg_ranks;
      e->R->bg_queue = 2;
      e->C->cache.rd.bg_spec_gate = std::max(0, ro.bg_gate);
      e->C->cache.bg_split = 2;
      e->C->cache.rd.max_bg = 1;
    }
  } else if (ro.bg_ranks > 0) {
    fprintf(stderr, "dmeng: background reads need the routers: off\n");
    ro.bg_ranks = 0;
  }
  const std::string xtok = ro.xtok;
  if (!xtok.empty()) {  // cross-token routers: the next token's first steps, from a late step of this one
    e->X = std::make_unique<Xtok>();
    if (!e->X->load(xtok, *e->C, err)) {
      g_err = "xtok: " + err;
      return nullptr;
    }
    e->X->tau = ro.xtok_tau;
    e->X->inner = e->C->hook;
    e->C->hook = e->X.get();
    if (ro.rd_prio > 0) e->C->cache.rd.prio = true;
  }
  if (mode != "lossless") {  // fast or an advanced approximate preset (dmeng --mode)
    const std::string md = mode;
    const std::string preset = md == "approx" ? "" : md.rfind("approx:", 0) == 0 ? md.substr(7) : "?";
    if (!e->C->ns.preset(preset)) {  // PRESET[:w=W:mass=M:late=MS:bg=0|1] (Approx::preset)
      g_err = "mode: exact|fast|lossless|approx[:PRESET[:w=W:mass=M:late=MS:bg=0|1]], got " + md;
      return nullptr;
    }
    if (e->C->ns.prio) e->C->cache.rd.prio = true;
    if (e->C->ns.cancel) e->C->cancel_stale = true;
  }
  // the prefill's GPU split (exact): 0 = the profile's shares (mac: 0.4 / 0.4), < 0 = off
  const double gpu_frac = ro.gpu_frac, gpu_experts = ro.gpu_experts;
  if (gpu_frac > 0 || gpu_experts > 0 || o.prefill_gpu_min > 0 || o.prefill_gpu_min_wave > 0) {
    std::string ge;
    char v[32];
    auto set = [&](const char* k, double x) {
      snprintf(v, sizeof v, "%.6f", x);
      return e->M->set_option(k, v, ge);
    };
    if ((o.prefill_gpu_int_kernel && !set("prefill_gpu_kernel", 0)) || (o.prefill_gpu_min_wave > 0 && !set("prefill_gpu_min_wave", o.prefill_gpu_min_wave)) || (o.prefill_gpu_min > 0 && !set("prefill_gpu_min", o.prefill_gpu_min)) || (gpu_frac > 0 && !set("prefill_gpu_frac", gpu_frac)) ||
        (gpu_experts > 0 && !set("prefill_gpu_experts", gpu_experts))) {
      g_err = "prefill gpu: " + ge;
      return nullptr;
    }
  }
  {  // run-ahead (exact: names only): 0 = the profile's, < 0 = off
    const int ra = ro.run_ahead, rd = ro.run_ahead_depth;
    std::string re;
    if (ra > 0 && (!e->M->set_option("run_ahead", std::to_string(ra), re) || (rd > 1 && !e->M->set_option("run_ahead_depth", std::to_string(rd), re)))) {
      g_err = "run_ahead: " + re;
      return nullptr;
    }
    e->C->ra_budget = ro.ra_budget;
    e->C->ra_max_tok = ro.ra_max_tok;
    e->C->ra_sat_skip = ro.ra_sat_skip;
    e->C->cache.rd.depth_times(e->rs_dt);
  }
  int pt = o.prefill_threads;
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  // (P2: 6 prefill threads with the 4-bit resident measured slower on the phone, 33.3 vs 30.3 s for 472 tokens: off)
#endif
  if (pt > 0 && pt != e->C->o.threads) e->prefill_pool.reset(new Pool(pt, -1, (int)e->qos));
  {  // the model's own options ("name=value;..."); iOS default: free the batched prefill's scratch after each prompt
     // (P2: a 357-token prompt left ~0.35 GB of footprint behind for the rest of the session)
    std::string mo = model_opts;  // (the run: entries were read above)
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
    if (mo.find("prefill_release") == std::string::npos) mo = "prefill_release=1" + std::string(mo.empty() ? "" : ";") + mo;
#endif
    for (size_t i = 0; i < mo.size();) {
      size_t j = mo.find(';', i);
      if (j == std::string::npos) j = mo.size();
      const std::string kv = mo.substr(i, j - i);
      const size_t q = kv.find('=');
      std::string merr;
      if (q != std::string::npos && !e->M->set_option(kv.substr(0, q), kv.substr(q + 1), merr)) {
        g_err = "model_options: " + merr;
        return nullptr;
      }
      i = j + 1;
    }
  }
  const int ctx = o.ctx > 0 ? o.ctx : 16384;
  e->S = e->M->new_state(ctx, err);
  if (!e->S) {
    g_err = "state: " + err;
    return nullptr;
  }
  e->lm = std::make_unique<ModelLM>(*e->M, *e->C, *e->S, ctx);
  SessionOptions so;
  if (o.disk_dir && *o.disk_dir) so.disk_dir = o.disk_dir;
  if (o.disk_gb > 0) so.disk_gb = o.disk_gb;
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  so.ckpt_gb = 0.2;  // the phone's 4.27 GB limit: about 3 checkpoints and one short parked sequence
  so.ckpt_min = 0;   // (the byte budget alone there)
  so.park_gb = 0.3;
#endif
#if TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
  so.disk_min = 16;  // P2: short chats get a disk copy (a chat continued after a relaunch restores it)
  so.disk_gain = 16;  // ... and a restore of it pays from 16 positions (the 256 default never fires for a short chat)
#endif
  if (o.disk_min > 0) so.disk_min = o.disk_min;
  so.preload_experts = o.preload_experts != 0;
  so.refresh_pinned = so.preload_experts;  // P2: the preload of a new chat brings the last turn's experts
  if (o.ckpt_gb > 0) so.ckpt_gb = o.ckpt_gb;
  if (o.park_gb > 0) so.park_gb = o.park_gb;
  e->pc = std::make_unique<PrefixCache>(*e->lm, so);
  ro.routers = e->R || e->PL ? e->routers_path : "";
  e->res = ro.resolved(co.cache_gb);
  return e.release();
}

void dm_close(dm_engine* e) {
  if (!e) return;
  dm_cancel(e);
  {
    std::lock_guard<std::mutex> lk(e->busy);  // a call in flight finishes (within one layer) first
  }
  delete e;
}

const char* dm_last_error(dm_engine* e) { return e ? e->err.c_str() : g_err.c_str(); }

int dm_info(dm_engine* e, dm_info_t* out) {
  if (!e || !out) return DM_EINVAL;
  const LMInfo& i = e->lm->info();
  std::memset(out, 0, sizeof *out);
  out->vocab = i.vocab;
  out->max_ctx = i.max_ctx;
  out->n_layers = i.n_layers;
  out->n_experts = i.n_experts;
  out->pos = e->lm->pos();
  out->fixed_bytes = (int64_t)i.fixed_bytes;
  out->pos_bytes = (int64_t)i.pos_bytes;
  out->threads = e->C->o.threads;
  out->readers = e->C->o.readers;
  out->memory_mode = e->o.memory_mode;
  out->router_m = e->R ? e->R->m : 0;
  int64_t cb = 0;
  for (int p = 0; p < e->C->cache.npools; p++) cb += (int64_t)e->C->cache.P[p].cap * (int64_t)e->C->cache.P[p].rec;
  out->cache_bytes = cb;
  std::strncpy(out->name, i.name.c_str(), sizeof out->name - 1);
  std::strncpy(out->routers, e->routers_path.c_str(), sizeof out->routers - 1);
  return DM_OK;
}

int dm_sync(dm_engine* e, const int32_t* ids, int n, const int32_t* marks, int nmarks, float* logits,
            dm_progress_fn progress, void* user, dm_sync_result* res) {
  if (!e || !ids || n <= 0) return DM_EINVAL;
  std::lock_guard<std::mutex> lk(e->busy);
  QosScope qs(e->qos);
  if (res) std::memset(res, 0, sizeof *res);
  if (e->cancel_req.exchange(false)) {
    if (res) res->cancelled = 1;
    return DM_ECANCELLED;
  }
  auto t0 = std::chrono::steady_clock::now();
  std::vector<int> v(ids, ids + n), mk;
  for (int i = 0; i < nmarks; i++) mk.push_back(marks[i]);
  // the progress hook runs at every layer boundary: it also turns a pending cancel into the model's own cancel
  e->lm->on_progress = [e, progress, user](int done, int total, double mb) {
    if (e->cancel_req.load()) e->lm->cancel();
    if (progress) progress(user, done, total, mb);
  };
  if (e->prefill_pool) std::swap(e->C->pool, e->prefill_pool);  // the prompt on prefill_threads, decode on threads
  SyncResult r = e->pc->sync(v, mk, logits, logits != nullptr);  // NULL logits: state only (an idle-time prefix)
  if (e->prefill_pool) std::swap(e->C->pool, e->prefill_pool);
  e->lm->on_progress = nullptr;
  if (r.cancelled) e->cancel_req.store(false);
  {
    std::ostringstream lp;
    lp << "{\"source\":\"" << r.source << "\",\"reused\":" << r.reused << ",\"computed\":" << r.computed << ",\"ms_restore\":" << r.ms_restore
       << ",\"ms_preload\":" << r.ms_preload << ",\"preloaded\":" << r.preloaded
       << ",\"ms_prefill\":" << r.ms_prefill << ",\"positions\":" << r.pf.positions << ",\"chunks\":" << r.pf.chunks << ",\"ms\":" << r.pf.ms
       << ",\"ms_wait\":" << r.pf.ms_wait << ",\"mb_read\":" << r.pf.mb_read << ",\"detail\":" << (r.pf.json.empty() ? "{}" : r.pf.json) << "}";
    e->last_prefill = lp.str();
  }
  if (res) {
    res->reused = r.reused;
    res->computed = r.computed;
    std::strncpy(res->source, r.source.c_str(), sizeof res->source - 1);
    res->ms_total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    res->ms_restore = r.ms_restore;
    res->ms_prefill = r.ms_prefill;
    res->ms_wait_reads = r.pf.ms_wait;
    res->mb_read = r.pf.mb_read;
    res->cancelled = r.cancelled;
  }
  return r.cancelled ? DM_ECANCELLED : DM_OK;
}

const char* dm_last_prefill_json(dm_engine* e) { return e ? e->last_prefill.c_str() : "{}"; }

int dm_step(dm_engine* e, int32_t id, float* logits) {
  if (!e || !logits) return DM_EINVAL;
  std::lock_guard<std::mutex> lk(e->busy);
  QosScope qs(e->qos);
  if (e->cancel_req.exchange(false)) return DM_ECANCELLED;
  const bool ab = e->o.run_ahead_ab > 0 && e->C->run_ahead;
  if (ab) e->ra_lcg = e->ra_lcg * 1664525u + 1013904223u, e->C->ra_off = ((e->ra_lcg >> 16) & 1) == 0;
  const auto t0 = std::chrono::steady_clock::now();
  const double w0 = e->C->t.wait;
  const uint64_t b0 = e->C->cache.rd.bytes.load(), u0 = e->C->cache.st.bytes_urgent, n0 = e->C->cache.st.bytes_pf;
  e->pc->step(id, logits);
  e->steps++;
  if (ab) {  // per arm (on = 1): reads that land during the next token count there (both arms alike)
    const int a = !e->C->ra_off;
    e->ab_n[a] += 1;
    e->ab_ms[a] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    e->ab_wait[a] += e->C->t.wait - w0;
    e->ab_mb[a] += (e->C->cache.rd.bytes.load() - b0) / 1e6;
    e->ab_urg[a] += (e->C->cache.st.bytes_urgent - u0) / 1e6;  // (a stats call between steps resets these: then 0)
    e->ab_named[a] += (e->C->cache.st.bytes_pf - n0) / 1e6;
  }
  return DM_OK;
}

void dm_cancel(dm_engine* e) {
  if (!e) return;
  e->cancel_req.store(true);
  if (e->lm) e->lm->cancel();
}

void dm_cancel_clear(dm_engine* e) {
  if (e) e->cancel_req.store(false);
}

int dm_persist(dm_engine* e, int pos) {
  if (!e) return DM_EINVAL;
  std::lock_guard<std::mutex> lk(e->busy);
  e->pc->persist(pos);
  return DM_OK;
}

void dm_sampler_default(dm_sampler* s) {
  std::memset(s, 0, sizeof *s);
  s->top_p = 1.0;
  s->repetition_penalty = 1.0;
}

int32_t dm_argmax(const float* logits, int vocab) { return argmax(logits, vocab); }

static uint64_t splitmix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

int32_t dm_sample(const float* logits, int V, const dm_sampler* s, const int32_t* history, int nh) {
  dm_sampler d;
  if (!s) {
    dm_sampler_default(&d);
    s = &d;
  }
  const bool pen = s->presence_penalty != 0 || s->repetition_penalty != 1 || s->frequency_penalty != 0;
  if (s->temp <= 0 && !pen) return argmax(logits, V);
  std::vector<float> lg(logits, logits + V);
  if (pen && history && nh > 0) {  // penalties on the raw logits over the reply's tokens
    std::vector<int32_t> h(history, history + nh);
    std::sort(h.begin(), h.end());
    for (size_t i = 0; i < h.size();) {
      size_t j = i;
      while (j < h.size() && h[j] == h[i]) j++;
      const int t = h[i];
      if (t >= 0 && t < V) {
        float& x = lg[t];
        if (s->repetition_penalty != 1) x = x > 0 ? (float)(x / s->repetition_penalty) : (float)(x * s->repetition_penalty);
        x -= (float)s->presence_penalty;
        x -= (float)(s->frequency_penalty * (double)(j - i));
      }
      i = j;
    }
  }
  if (s->temp <= 0) return argmax(lg.data(), V);
  // candidates: the top-k by logit (ties: lower index first), or the top 4096 for top-p / min-p only
  std::vector<int> idx(V);
  for (int i = 0; i < V; i++) idx[i] = i;
  int kk = s->top_k > 0 ? std::min(s->top_k, V) : std::min(V, 4096);
  std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b] || (lg[a] == lg[b] && a < b); });
  std::vector<double> p(kk);
  const double mx = (double)lg[idx[0]] / s->temp;
  double z = 0;
  for (int j = 0; j < kk; j++) z += (p[j] = std::exp((double)lg[idx[j]] / s->temp - mx));
  int keep = kk;
  if (s->top_p < 1.0) {
    double cum = 0;
    for (int j = 0; j < kk; j++) {
      if (cum >= s->top_p * z) {
        keep = j;
        break;
      }
      cum += p[j];
    }
  }
  if (s->min_p > 0) {  // relative to the most likely candidate (p[0] = 1 before normalising)
    int k2 = 0;
    while (k2 < keep && p[k2] >= s->min_p) k2++;
    keep = std::max(1, k2);
  }
  double zz = 0;
  for (int j = 0; j < keep; j++) zz += p[j];
  const uint64_t r = splitmix(splitmix(s->seed) ^ (uint64_t)(uint32_t)nh);
  const double u = (double)(r >> 11) * (1.0 / 9007199254740992.0) * zz;  // [0, zz)
  double acc = 0;
  for (int j = 0; j < keep; j++) {
    acc += p[j];
    if (u < acc) return idx[j];
  }
  return idx[keep - 1];
}

int dm_set_cache_bytes(dm_engine* e, int64_t bytes) {
  if (!e || bytes <= 0) return DM_EINVAL;
  if (e->C->cache.fa.on) e->C->cache.fa.max_bytes = bytes;
  e->C->cache.set_budget(bytes);
  if (e->busy.try_lock()) {  // idle: apply now (the call thread would otherwise apply it at its next boundary)
    e->C->cache.apply_budget();
    e->busy.unlock();
  }
  return DM_OK;
}

int dm_counters(dm_engine* e, dm_counters_t* out) {
  if (!e || !out) return DM_EINVAL;
  const CacheStats& S = e->C->cache.st;
  std::memset(out, 0, sizeof *out);
  out->hits = S.n_hit;
  out->late = S.n_late;
  out->misses = S.n_miss;
  out->picks = S.n_hit + S.n_late + S.n_miss;
  out->bytes_issued = S.bytes_urgent + S.bytes_pf;
  out->bytes_read = e->C->cache.rd.bytes.load(std::memory_order_relaxed);
  out->file_refaults = S.file_refaults;
  out->wait_ms = e->C->t.wait;
  out->held_bytes = (int64_t)e->C->cache.held_bytes();
  out->budget_bytes = e->C->cache.budget_bytes();
  out->decode_steps = e->steps;
  return DM_OK;
}

const char* dm_stats_json(dm_engine* e) {
  if (!e) return "{}";
  std::lock_guard<std::mutex> lk(e->busy);
  task_vm_info_data_t vi;
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  double foot = 0, peak = 0;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &cnt) == KERN_SUCCESS) {
    foot = vi.phys_footprint / 1e9;
    peak = vi.ledger_phys_footprint_peak / 1e9;
  }
  const Cache& Ca = e->C->cache;
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(3);
  s << "{\"model\":" << e->lm->stats_json() << ",\"prefix_cache\":" << e->pc->stats_json() << ",\"memory\":{\"footprint_gb\":" << foot
    << ",\"footprint_peak_gb\":" << peak << ",\"held_gb\":" << Ca.held_bytes() / 1e9 << ",\"mode\":" << e->o.memory_mode
    << ",\"reader_gb_read\":" << Ca.rd.bytes.load() / 1e9;
  {
    const ModelMem mm = e->M->mem(*e->S);
    s << ",\"model\":{\"resident_gb\":" << mm.resident / 1e9 << ",\"kv_reserved_gb\":" << mm.kv_reserved / 1e9 << ",\"recurrent_gb\":" << mm.recurrent / 1e9
      << ",\"prefill_scratch_gb\":" << mm.prefill_scratch / 1e9 << ",\"prefill_xqb_gb\":" << mm.prefill_xqb / 1e9 << "}";
  }
  if (Ca.mem == Mem::FILE)
    s << ",\"file_cache_resident_gb\":" << Ca.file_resident_bytes() / 1e9 << ",\"file_refaults\":" << Ca.st.file_refaults
      << ",\"file_resident_skips\":" << Ca.rd.file_resident_skips.load() << ",\"file_gb_skipped\":" << Ca.rd.file_bytes_skipped.load() / 1e9
      << ",\"budget_gb\":" << Ca.budget_bytes() / 1e9 << ",\"adapt\":{\"on\":" << Ca.fa.on << ",\"min_gb\":" << Ca.fa.min_bytes / 1e9
      << ",\"max_gb\":" << Ca.fa.max_bytes / 1e9 << ",\"resident_fraction\":" << Ca.fa.res_frac << ",\"refaults_per_token_ema\":" << Ca.fa.refault_ema
      << ",\"shrinks\":" << Ca.fa.shrinks << ",\"grows\":" << Ca.fa.grows << ",\"changes\":" << Ca.fa.trace.size() << "},\"core_gb\":" << Ca.core_slab_bytes / 1e9
      << ",\"core_records\":" << Ca.core_held();
  s << "}";
  s << ",\"cache_lock\":" << Ca.lock_json();
  s << ",\"policy\":{\"evict\":\"" << e->C->o.evict << "\",\"hazard\":" << Ca.hazard_evict << ",\"layer_caps_opt\":\"" << e->C->o.layer_caps
    << "\",\"caps_auto\":" << Ca.caps_auto << ",\"caps_active\":" << !Ca.lcap.empty() << ",\"slots\":" << Ca.budget_bytes() / std::max<int64_t>(1, (int64_t)Ca.P[0].rec)
    << ",\"plan\":" << e->o.plan << ",\"routers\":\"" << e->routers_path << "\",\"router_m\":" << (e->R ? e->R->m : 0)
    << ",\"mode\":\"" << (e->C->ns.on ? "approx" : "lossless") << "\",\"preset\":" << e->preset_json << ",\"xtok\":" << (e->X ? "true" : "false") << ",\"router_tau\":" << (e->R ? e->R->tau : 0) << ",\"split\":" << e->C->o.split
    << ",\"prefill_threads\":" << (e->prefill_pool ? e->prefill_pool->size() : e->C->o.threads)
    << ",\"resident_bits\":" << (e->o.resident_bits == 4 ? 4 : 8) << ",\"available_at_open_gb\":" << e->avail_open / 1e9 << ",\"slab_auto_gb\":" << e->slab_auto / 1e9
    << ",\"threads\":" << e->C->o.threads << ",\"readers\":" << e->C->o.readers << ",\"reader_io_low\":" << e->C->o.reader_io_low << ",\"res_memory_mode\":" << e->o.res_memory_mode << "}";
  {  // P2 levers: the reads since the last stats call (urgent = read when needed, named = issued ahead by the router;
     // named unused = named records evicted before any use; the cache counters are reset by every stats call, the
     // readers' are not), drive busy (wall time with a read in flight) and rate
    Cache& Cm = e->C->cache;
    const uint64_t bw = Cm.rd.busy_wall(), by = Cm.rd.bytes.load(), stp = e->steps;
    auto dl = [](uint64_t now, uint64_t then) { return now >= then ? (double)(now - then) : 0.0; };
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - e->rs_t).count();
    const double tok = std::max(1.0, dl(stp, e->rs_steps)), busy_s = dl(bw, e->rs_busy) / 1e9;
    const double rec = (double)Cm.P[0].rec;
    s << ",\"reads\":{\"tokens\":" << dl(stp, e->rs_steps) << ",\"mb_urgent_per_token\":" << Cm.st.bytes_urgent / 1e6 / tok
      << ",\"mb_named_per_token\":" << Cm.st.bytes_pf / 1e6 / tok << ",\"named_records_per_token\":" << Cm.st.n_pf_issued / tok
      << ",\"mb_named_unused_per_token\":" << Cm.st.n_pf_unused_evicted * rec / 1e6 / tok << ",\"wall_s\":" << wall << ",\"busy_s\":" << busy_s
      << ",\"busy_pct\":" << 100.0 * busy_s / std::max(1e-9, wall) << ",\"gb_per_s_busy\":" << dl(by, e->rs_bytes) / 1e9 / std::max(1e-9, busy_s)
      << ",\"gb_per_s_wall\":" << dl(by, e->rs_bytes) / 1e9 / std::max(1e-9, wall) << "}";
    e->rs_busy = bw; e->rs_bytes = by; e->rs_steps = stp;
    e->rs_t = std::chrono::steady_clock::now();
  }
  {  // the approximate mode's own account (since the last stats call): raw totals over layer steps
    const Approx& N = e->C->ns;
    s << ",\"approx\":{\"on\":" << N.on << ",\"layer_steps\":" << N.steps << ",\"weight_replaced_sum\":" << N.w_replaced
      << ",\"weight_replaced_per_step\":" << N.w_replaced / std::max(1.0, N.steps) << ",\"replaced\":" << N.n_replaced
      << ",\"steps_with_a_replacement\":" << N.steps_replaced << ",\"max_replaced_weight\":" << N.w_replaced_max
      << ",\"read_now\":" << N.n_read_now << ",\"weight_read_now_sum\":" << N.w_read_now << ",\"late_waited\":" << N.n_late_waited
      << ",\"late_got\":" << N.n_late_got << ",\"late_dropped\":" << N.n_late_dropped << ",\"late_wait_ms\":" << N.late_wait_ms
      << ",\"bg\":" << N.n_bg << ",\"bg_on\":" << N.bg << ",\"read_w\":" << N.read_w << ",\"late_ms\":" << N.late_ms << ",\"drop\":" << N.drop;
    std::vector<float> w = N.w_tok;  // released weight left out per token
    double sum = 0;
    for (float x : w) sum += x;
    std::sort(w.begin(), w.end());
    const double nt = std::max<double>(1, w.size());
    s << ",\"tokens\":" << w.size() << ",\"weight_left_out_per_token_mean\":" << sum / nt
      << ",\"weight_left_out_per_token_p90\":" << (w.empty() ? 0.0 : (double)w[std::min(w.size() - 1, (size_t)(0.9 * w.size()))])
      << ",\"steps_with_a_drop_share\":" << N.steps_replaced / std::max(1.0, N.steps) << ",\"picks_left_out_per_token\":" << N.n_replaced / nt
      << ",\"picks_read_and_waited_per_token\":" << (N.n_read_now + N.n_late_waited) / nt << "}";
  }
  if (e->C->run_ahead) s << ",\"run_ahead\":" << e->C->ra_json();  // cumulative since open (dmeng's counters)
  {  // the readers' wall time by reads in flight since the last stats call (per token: dmeng's read_queue counters)
    uint64_t d[7];
    e->C->cache.rd.depth_times(d);
    const double tk = std::max<double>(1, e->steps - e->rs_steps_dt);
    auto per = [&](int i) { return (d[i] - e->rs_dt[i]) / 1e6 / tk; };
    s << ",\"read_queue\":{\"tokens\":" << e->steps - e->rs_steps_dt << ",\"reader_idle_ms_per_token\":" << per(0)
      << ",\"reader_idle_queued_ms_per_token\":" << per(1) << ",\"in_flight_1_2_3_4plus_ms_per_token\":[" << per(2) << "," << per(3)
      << "," << per(4) << "," << per(5) << "],\"read_bursts_per_token\":" << (d[6] - e->rs_dt[6]) / tk << "}";
    std::memcpy(e->rs_dt, d, sizeof d);
    e->rs_steps_dt = e->steps;
  }
  if (e->ab_n[0] + e->ab_n[1] > 0) {
    s << ",\"ra_ab\":{";
    for (int a = 0; a < 2; a++) {
      const double n = std::max(1.0, e->ab_n[a]);
      s << (a ? ",\"on\":" : "\"off\":") << "{\"tokens\":" << e->ab_n[a] << ",\"ms\":" << e->ab_ms[a] / n << ",\"wait_ms\":" << e->ab_wait[a] / n
        << ",\"mb_read\":" << e->ab_mb[a] / n << ",\"mb_urgent\":" << e->ab_urg[a] / n << ",\"mb_named\":" << e->ab_named[a] / n << "}";
    }
    s << "}";
  }
  if (e->C->hook) s << ",\"hook\":" << e->C->hook->stats_json(*e->C, (int)std::max<int64_t>(1, e->steps));
  s << ",\"cache_auto\":" << e->cache_why << ",\"resolved\":" << e->res.json() << ",\"capabilities\":" << e->M->info().caps_json() << "}";
  e->stats = s.str();
  e->lm->reset_stats();
  return e->stats.c_str();
}

int dm_bench(dm_engine* e, const char* kind, const char* spec, char** json_out) {
  if (json_out) *json_out = nullptr;
  if (!e || !kind || !json_out) return DM_EINVAL;
  if (std::strcmp(kind, "filecache") != 0) return DM_ENOTIMPL;
  int n = 24;
  if (spec && std::strncmp(spec, "n=", 2) == 0) n = std::max(1, atoi(spec + 2));
  std::lock_guard<std::mutex> lk(e->busy);
  Cache& Ca = e->C->cache;
  const ModelInfo& mi = e->M->info();
  const size_t R = mi.formats.empty() ? 0 : mi.formats[0].rec, pg = (size_t)getpagesize();
  if (!R) return DM_EINVAL;
  std::mt19937 rng(1122);
  std::vector<uint8_t> scratch(R);
  std::vector<char> v((R + pg - 1) / pg);
  double t_read = 0, t_touch = 0, res_frac = 0;
  long faults = 0;
  int done = 0;
  auto faults_now = []() {
    task_events_info_data_t ev;
    mach_msg_type_number_t c = TASK_EVENTS_INFO_COUNT;
    task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&ev, &c);
    return (long)ev.faults;
  };
  for (int t = 0; t < 20000 && done < n; t++) {
    const int L = (int)(rng() % (unsigned)mi.layers), x = (int)(rng() % (unsigned)mi.experts);
    if (mi.layer_fmt[L] != 0) continue;
    const off_t off = (off_t)((size_t)x * R);
    const int fd = Ca.fds[L];
    const size_t sz = (size_t)lseek(fd, 0, SEEK_END);
    uint8_t* m = (uint8_t*)mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) continue;
    mincore(m + off, R, v.data());
    bool cold = true;
    for (char c : v) cold &= !(c & MINCORE_INCORE);
    if (!cold) {
      munmap(m, sz);
      continue;
    }
    // a buffered read on a descriptor of its own (the cache's may carry F_NOCACHE in the slot modes)
    char path[MAXPATHLEN] = {0};
    int fd2 = fcntl(fd, F_GETPATH, path) == 0 ? open(path, O_RDONLY) : -1;
    if (fd2 < 0) fd2 = fd;
    else fcntl(fd2, F_RDAHEAD, 0);
    auto a = std::chrono::steady_clock::now();
    size_t got = 0;
    while (got < R) {
      ssize_t r = pread(fd2, scratch.data() + got, R - got, off + (off_t)got);
      if (r <= 0) break;
      got += (size_t)r;
    }
    auto b = std::chrono::steady_clock::now();
    if (fd2 != fd) close(fd2);
    mincore(m + off, R, v.data());
    int in = 0;
    for (char c : v) in += (c & MINCORE_INCORE) != 0;
    const long f0 = faults_now();
    auto c0 = std::chrono::steady_clock::now();
    uint64_t acc = 0;
    for (size_t i = 0; i < R / 8; i++) acc += ((const uint64_t*)(m + off))[i];
    auto c1 = std::chrono::steady_clock::now();
    faults += faults_now() - f0;
    (void)acc;
    munmap(m, sz);
    t_read += std::chrono::duration<double, std::milli>(b - a).count();
    t_touch += std::chrono::duration<double, std::milli>(c1 - c0).count();
    res_frac += (double)in / v.size();
    done++;
  }
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(4);
  const double d = std::max(1, done);
  s << "{\"kind\":\"filecache\",\"records\":" << done << ",\"record_bytes\":" << R << ",\"pread_ms\":" << t_read / d
    << ",\"pread_gb_s\":" << (done * (double)R / 1e9) / std::max(1e-9, t_read / 1e3) << ",\"resident_after_pread\":" << res_frac / d
    << ",\"first_touch_ms\":" << t_touch / d << ",\"first_touch_faults\":" << faults / d << "}";
  *json_out = strdup(s.str().c_str());
  return DM_OK;
}

}  // extern "C"
