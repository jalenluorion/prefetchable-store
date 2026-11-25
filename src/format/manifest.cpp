#include "manifest.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

#include "files.h"

namespace ms {

const MatrixLayout* RecordFormat::matrix(const std::string& n) const {
  for (auto& m : matrices)
    if (m.name == n) return &m;
  return nullptr;
}

static std::string join(const std::string& a, const std::string& b) {
  if (!b.empty() && b[0] == '/') return b;  // an absolute path (e.g. a standalone embedding file) stays as it is
  return a.empty() || a.back() == '/' ? a + b : a + "/" + b;
}

bool Manifest::load(const std::string& model_dir, std::string& err) {
  dir = model_dir;
  if (!json_load(join(dir, "moved-store.json"), raw, err)) return false;
  if (raw["format"].s() != "moved-store") {
    err = "moved-store.json: format is not \"moved-store\"";
    return false;
  }
  format_version = raw["format_version"].s();
  if (atoi(format_version.c_str()) != 1) {
    err = "moved-store.json: format_version " + format_version + " (this reader knows major version 1)";
    return false;
  }
  family = raw["family"].s();
  name = raw["name"].s();
  for (auto& p : raw["parts"].arr) {
    Part q;
    q.path = p["path"].s();
    q.sha256 = p["sha256"].str_or("");
    q.role = p["role"].s();
    q.precision = p["precision"].str_or("");
    q.bytes = p["bytes"].kind == Json::NUM ? p["bytes"].i() : -1;
    q.optional = p["optional"].kind == Json::BOOL && p["optional"].b;
    parts.push_back(q);
  }
  const Json& md = raw["model"];
  hidden = md["hidden"].i();
  vocab = md["vocab"].i();
  vocab_unpadded = md["vocab_unpadded"].i(vocab);
  physical_layers = md["physical_layers"].i();
  steps = md["steps"].i(physical_layers);
  for (auto& x : md["step_layer"].arr) step_layer.push_back((int)x.i());
  for (auto& x : md["step_pass"].arr) step_pass.push_back((int)x.i());
  layer_types = md["layer_types"].s();
  emb_file = md["embedding"]["file"].s();
  emb_tensor = md["embedding"]["tensor"].s();
  emb_dtype = md["embedding"]["dtype"].s();
  const Json& ex = raw["experts"];
  layer_file = ex["layer_file"].s();
  experts = ex["experts_per_layer"].i();
  top_k = ex["top_k"].i();
  page = ex["page"].i();
  router_rule = ex["router"].s();
  activation = ex["activation"].s();
  route_scale = ex["route_scale"].f(1.0);
  if (ex["layer_format"].kind == Json::STR) {
    layer_format.assign(physical_layers, ex["layer_format"].s());
  } else {
    for (auto& x : ex["layer_format"].arr) layer_format.push_back(x.s());
  }
  for (auto& kv : ex["formats"].obj) {
    RecordFormat f;
    f.id = kv.first;
    const Json& j = kv.second;
    f.kind = j["kind"].s();
    f.bits = (int)j["bits"].i();
    f.group = (int)j["group"].i();
    f.record_bytes = j["record_bytes"].i();
    f.record_bytes_used = j["record_bytes_used"].i(f.record_bytes);
    f.align = j["align"].i();
    for (auto& m : j["matrices"].arr) {
      MatrixLayout L;
      L.name = m["name"].s();
      L.rows = m["shape"][0].i();
      L.cols = m["shape"][1].i();
      L.codes = m["codes"].i(-1);
      L.scales = m["scales"].i(-1);
      L.biases = m["biases"].i(-1);
      L.interleave = m["interleave"].str_or("");
      f.matrices.push_back(L);
    }
    f.trailer = j["trailer"].kind == Json::OBJ ? j["trailer"]["offset"].i(-1) : -1;
    formats.push_back(f);
  }
  for (auto& r : raw["routers"].arr) {
    RouterEntry e;
    e.id = r["id"].s();
    e.path = r["path"].s();
    e.form = r["form"].s();
    e.index = r["index"].s();
    e.lag = (int)r["lag"].i();
    e.hidden = (int)r["hidden"].i();
    e.targets = (int)r["targets"].i();
    e.is_default = r["default"].kind == Json::BOOL && r["default"].b;
    routers.push_back(e);
  }
  for (auto& d : raw["derived"].arr) {
    Derived x;
    x.id = d["id"].s();
    x.engine_format = d["engine_format"].s();
    x.sha256 = d["sha256"].str_or("");
    x.bytes = d["bytes"].kind == Json::NUM ? d["bytes"].i() : -1;
    x.path = d["path"].str_or("");
    for (auto& f : d["from"].arr) x.from.push_back(f.s());
    derived.push_back(x);
  }
  // config.json is a part like any other; a manifest kept apart from its weights (Flash-Next on the Mac) has none beside it
  std::string cerr;
  if (!json_load(join(dir, "config.json"), config, cerr)) {
    if (part("config.json") && access(join(dir, "config.json").c_str(), F_OK) == 0) {
      err = cerr;
      return false;
    }
    config = Json();
  }
  return true;
}

const Part* Manifest::part(const std::string& p) const {
  for (auto& q : parts)
    if (q.path == p) return &q;
  return nullptr;
}
const RecordFormat* Manifest::format(const std::string& id) const {
  for (auto& f : formats)
    if (f.id == id) return &f;
  return nullptr;
}
const RouterEntry* Manifest::router(const std::string& id) const {
  for (auto& r : routers)
    if (id.empty() ? r.is_default : r.id == id) return &r;
  return nullptr;
}
const Derived* Manifest::derived_of(const std::string& id) const {
  for (auto& d : derived)
    if (d.id == id) return &d;
  return nullptr;
}

std::string Manifest::layer_file_printf() const {
  std::string s = layer_file;
  size_t p = s.find("{L:02d}");
  if (p != std::string::npos) s.replace(p, 7, "%02d");
  return s;
}
std::string Manifest::layer_path(int L) const {
  char b[512];
  snprintf(b, sizeof b, layer_file_printf().c_str(), L);
  return join(dir, b);
}

std::string Manifest::cache_dir() const {
  if (!cache_override.empty()) return cache_override;
  std::string d = join(dir, ".dmeng");
  struct stat st;
  if ((stat(d.c_str(), &st) == 0 && S_ISDIR(st.st_mode) && access(d.c_str(), W_OK) == 0) ||
      (stat(d.c_str(), &st) != 0 && access(dir.c_str(), W_OK) == 0))
    return d;
  std::string err, h = sha256_file(join(dir, "moved-store.json"), err, false);
  const char* x = getenv("XDG_CACHE_HOME");
  std::string base = x && *x ? std::string(x) : std::string(getenv("HOME") ? getenv("HOME") : ".") + "/.cache";
  return base + "/dmeng/" + name + "-" + h.substr(0, 12);
}

std::string Manifest::derived_path(const Derived& d) const {
  const Part* src = d.from.empty() ? nullptr : part(d.from[0]);
  std::string h = src && src->sha256.size() >= 12 ? src->sha256.substr(0, 12) : "unknown";
  return join(cache_dir(), d.id + "." + d.engine_format + "." + h + ".bin");
}

std::string Manifest::shipped_path(const Derived& d) const {
  struct stat st;
  if (!d.path.empty() && stat(join(dir, d.path).c_str(), &st) == 0) return join(dir, d.path);
  const std::string full = derived_path(d);
  const std::string p = join(join(dir, "derived"), full.substr(full.find_last_of('/') + 1));
  return stat(p.c_str(), &st) == 0 ? p : "";
}

// ---------------------------------------------------------------------------------------------- engine cfg
static bool emb_offset(const Manifest& m, int64_t& off, std::string& err) {
  SafeTensors st;
  if (!st.open(join(m.dir, m.emb_file), err)) return false;
  const StTensor* t = st.find(m.emb_tensor);
  if (!t) {
    err = m.emb_file + ": no tensor " + m.emb_tensor;
    return false;
  }
  off = (int64_t)st.data_offset(*t);
  return true;
}

static std::string I(int64_t v) { return std::to_string(v); }

bool Manifest::apply_profile_experts(const std::string& profile, std::string& err) {
  if (profile.empty()) return true;
  const Json& ex = raw["profiles"][profile]["experts"];
  if (ex.is_null()) return true;
  if (!ex["layer_format"].is_null()) {
    if ((int64_t)ex["layer_format"].size() != physical_layers) {
      err = "profiles." + profile + ".experts.layer_format: one id per physical layer";
      return false;
    }
    layer_format.clear();
    for (auto& x : ex["layer_format"].arr) {
      if (!format(x.s())) {
        err = "profiles." + profile + ".experts.layer_format: unknown record format " + x.s();
        return false;
      }
      layer_format.push_back(x.s());
    }
  }
  if (!ex["layer_files"].is_null()) {
    layer_paths.clear();
    for (int L = 0; L < physical_layers; L++) {
      char b[512];
      snprintf(b, sizeof b, layer_file_printf().c_str(), L);
      layer_paths.push_back(b);
    }
    for (auto& kv : ex["layer_files"].obj) {
      const int L = atoi(kv.first.c_str());
      if (L < 0 || L >= physical_layers) {
        err = "profiles." + profile + ".experts.layer_files: no layer " + kv.first;
        return false;
      }
      layer_paths[L] = kv.second.s();
    }
  }
  return true;
}

bool Manifest::engine_cfg(Cfg& out, std::string& err) const {
  out.clear();
  int64_t eoff;
  if (!emb_offset(*this, eoff, err)) return false;
  auto need = [&](const Json& j, const char* k) -> const Json& {
    const Json* p = j.get(k);
    if (!p && err.empty()) err = std::string("config.json: missing ") + k;
    static const Json z;
    return p ? *p : z;
  };
  if (family == "qwen3_5_moe") {
    const Json& tc = config.get("text_config") ? config["text_config"] : config;
    if ((int64_t)need(tc, "hidden_size").i() != hidden || need(tc, "vocab_size").i() != vocab ||
        need(tc, "num_experts").i() != experts || need(tc, "num_experts_per_tok").i() != top_k ||
        need(tc, "num_hidden_layers").i() != physical_layers) {
      if (err.empty()) err = "manifest model / experts disagree with config.json";
      return false;
    }
    const RecordFormat* f = format(layer_format.at(0));
    if (!f || f->kind != "affine") {
      err = "qwen3_5_moe expects one affine record format";
      return false;
    }
    const Json& rp = tc["rope_parameters"];
    int64_t hd = need(tc, "head_dim").i();
    double prf = rp.get("partial_rotary_factor") ? rp["partial_rotary_factor"].f() : tc["partial_rotary_factor"].f(0.25);
    std::string theta = rp.get("rope_theta") ? py_str(rp["rope_theta"]) : "10000000";
    out = {{"hidden", I(hidden)},
           {"layers", I(physical_layers)},
           {"layer_types", layer_types},
           {"experts", I(experts)},
           {"topk", I(top_k)},
           {"moe_inter", py_str(need(tc, "moe_intermediate_size"))},
           {"shared_inter", py_str(need(tc, "shared_expert_intermediate_size"))},
           {"norm_topk", I(tc.get("norm_topk_prob") ? (int64_t)tc["norm_topk_prob"].i() : 1)},
           {"lin_nk", py_str(need(tc, "linear_num_key_heads"))},
           {"lin_nv", py_str(need(tc, "linear_num_value_heads"))},
           {"lin_dk", py_str(need(tc, "linear_key_head_dim"))},
           {"lin_dv", py_str(need(tc, "linear_value_head_dim"))},
           {"conv_k", py_str(need(tc, "linear_conv_kernel_dim"))},
           {"n_heads", py_str(need(tc, "num_attention_heads"))},
           {"n_kv", py_str(need(tc, "num_key_value_heads"))},
           {"head_dim", I(hd)},
           {"rot_dim", I((int64_t)(hd * prf))},
           {"rope_theta", theta},
           {"eps", py_str(need(tc, "rms_norm_eps"))},
           {"vocab", I(vocab)},
           {"emb_file", emb_file},
           {"emb_off", I(eoff)},
           {"rec_bytes", I(f->record_bytes)},
           {"store_layer_file", layer_file_printf()}};
    const char* kinds[3] = {"gate_proj", "up_proj", "down_proj"};
    const char* mats[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; i++) {
      const MatrixLayout* L = f->matrix(mats[i]);
      if (!L) {
        err = std::string("record format has no matrix ") + mats[i];
        return false;
      }
      out.push_back({std::string("off_") + kinds[i], I(L->codes)});
      out.push_back({std::string("soff_") + kinds[i], I(L->scales)});
      out.push_back({std::string("boff_") + kinds[i], I(L->biases)});
    }
    return err.empty();
  }
  if (family == "inkling2_loop") {
    const Json& tc = config["text_config"];
    const Json& loop = tc["loop_config"];
    if (need(tc, "hidden_size").i() != hidden || need(tc, "vocab_size").i() != vocab ||
        need(tc, "n_routed_experts").i() != experts || need(tc, "num_experts_per_tok").i() != top_k ||
        need(tc, "num_hidden_layers").i() != physical_layers) {
      if (err.empty()) err = "manifest model / experts disagree with config.json";
      return false;
    }
    std::string kind, fmt;
    for (char c : layer_types) kind += c == 'G' ? 'G' : 'L';
    for (auto& id : layer_format) {
      const RecordFormat* f = format(id);
      if (!f) {
        err = "unknown record format " + id;
        return false;
      }
      fmt += f->kind == "nvfp4" ? 'N' : 'M';
    }
    const Json& caps = raw["experts"]["activation_caps"];
    out = {{"model", family},
           {"bits", "8"},
           {"source", "resident_q8.safetensors_(mlx_run.py_--resident-bits_8_cache)"},
           {"hidden", I(hidden)},
           {"layers", I(physical_layers)},
           {"experts", I(experts)},
           {"topk", I(top_k)},
           {"inter", py_str(need(tc, "intermediate_size"))},
           {"heads", py_str(need(tc, "num_attention_heads"))},
           {"kv_heads", py_str(need(tc, "num_key_value_heads"))},
           {"head_dim", py_str(need(tc, "head_dim"))},
           {"d_rel", py_str(need(tc, "d_rel"))},
           {"rel_extent", py_str(need(tc, "rel_extent"))},
           {"window", py_str(need(tc, "sliding_window_size"))},
           {"gate_rank", py_str(need(loop, "gate_rank"))},
           {"core_start", py_str(need(loop, "core_start"))},
           {"core_end", py_str(need(loop, "core_end"))},
           {"passes", py_str(need(loop, "num_passes"))},
           {"block", py_str(need(tc, "n_layers_per_attn_res_block"))},
           {"vocab", I(vocab)},
           {"unpadded_vocab", I(vocab_unpadded)},
           {"softcap", py_str(need(tc, "final_logit_softcapping"))},
           {"gate_cap", caps.get("gate") ? py_str(caps["gate"]) : py_str(need(tc, "mlp_gate_softcap"))},
           {"up_cap", caps.get("up") ? py_str(caps["up"]) : py_str(need(tc, "mlp_up_softcap"))},
           {"route_scale", py_str(raw["experts"]["route_scale"])},
           {"eps", py_str(need(tc, "rms_norm_eps"))},
           {"qk_clip", py_str(need(tc, "qk_l2_clip_threshold"))},
           {"rel_clip", py_str(need(tc, "rel_l2_clip_threshold"))},
           {"log_floor", py_str(need(tc, "log_scaling_n_floor"))},
           {"log_alpha", py_str(need(tc, "log_scaling_alpha"))},
           {"sconv", py_str(need(tc, "sconv_kernel_size"))},
           {"layer_kind", kind},
           {"layer_format", fmt},
           {"emb_file", emb_file},
           {"emb_off", I(eoff)},
           {"layer_file", layer_file_printf()},
           {"eos", config.get("eos_token_id") ? py_str(config["eos_token_id"]) : "-1"}};
    if (!layer_paths.empty()) {  // a profile's own record files (apply_profile_experts): ';'-joined, one per layer
      std::string lp;
      for (auto& p : layer_paths) lp += (lp.empty() ? "" : ";") + p;
      out.push_back({"layer_paths", lp});
    }
    for (auto& f : formats) {
      std::string tag = f.kind == "nvfp4" ? "nv" : "mx";
      const MatrixLayout *gu = f.matrix("gate_up"), *dn = f.matrix("down");
      if (!gu || !dn) {
        err = "record format " + f.id + " lacks gate_up / down";
        return false;
      }
      out.push_back({tag + "_rec", I(f.record_bytes)});
      out.push_back({tag + "_gu_c", I(gu->codes)});
      out.push_back({tag + "_gu_s", I(gu->scales)});
      out.push_back({tag + "_dn_c", I(dn->codes)});
      out.push_back({tag + "_dn_s", I(dn->scales)});
      if (f.trailer >= 0) out.push_back({tag + "_trailer", I(f.trailer)});
    }
    return err.empty();
  }
  if (family == "qwen4_exp") {  // Qwen3.8-Flash-Next: config.json's text_config, store/meta.json's record, extras.ngram
    const Json& tc = config.get("text_config") ? config["text_config"] : config;
    if (need(tc, "hidden_size").i() != hidden || need(tc, "vocab_size").i() != vocab || need(tc, "num_experts").i() != experts ||
        need(tc, "num_experts_per_tok").i() != top_k || need(tc, "num_hidden_layers").i() != physical_layers) {
      if (err.empty()) err = "manifest model / experts disagree with config.json";
      return false;
    }
    const RecordFormat* f = format(layer_format.at(0));
    if (!f || f->kind != "affine" || f->bits != 4 || f->group != 64) {
      err = "qwen4_exp expects one affine 4-bit group-64 record format";
      return false;
    }
    const Json& rp = tc["rope_parameters"];
    const std::string rt = rp.get("rope_type") ? rp["rope_type"].s() : rp.get("type") ? rp["type"].s() : "default";
    if (rt != "default") {
      err = "qwen4_exp: rope type " + rt + " (default only)";
      return false;
    }
    int ple = -1, nple = 0;
    for (auto& x : tc["ple_layer_ids"].arr) ple = (int)x.i() - 1, nple++;
    if (nple != 1) {
      err = "qwen4_exp: one PLE layer expected (the n-gram store holds one table), config lists " + I(nple);
      return false;
    }
    const Json& ng = raw["extras"]["ngram"];
    if (ng.is_null()) {
      err = "qwen4_exp: the manifest has no extras.ngram (the n-gram rows are part of the model)";
      return false;
    }
    const Json& eos = tc["eos_token_id"];
    const int64_t hd = need(tc, "head_dim").i();
    const double prf = rp.get("partial_rotary_factor") ? rp["partial_rotary_factor"].f() : tc["partial_rotary_factor"].f(0.25);
    std::string ngf = ng["files"].s();
    const size_t q = ngf.find("{P:02d}");
    if (q != std::string::npos) ngf.replace(q, 7, "%02d");
    const std::string gate = tc["output_gate_type"].str_or("sigmoid");
    out = {{"model", family},
           {"hidden", I(hidden)},
           {"hc", I(tc["hc_count"].i(4))},
           {"hc_low", I(tc["hc_lowrank"].i(320))},
           {"layers", I(physical_layers)},
           {"layer_types", layer_types},
           {"experts", I(experts)},
           {"topk", I(top_k)},
           {"moe_inter", py_str(need(tc, "moe_intermediate_size"))},
           {"shared_inter", py_str(need(tc, "shared_expert_intermediate_size"))},
           {"norm_topk", I(tc.get("norm_topk_prob") ? (int64_t)tc["norm_topk_prob"].i() : 1)},
           {"lin_nk", py_str(need(tc, "linear_num_key_heads"))},
           {"lin_nv", py_str(need(tc, "linear_num_value_heads"))},
           {"lin_dk", py_str(need(tc, "linear_key_head_dim"))},
           {"lin_dv", py_str(need(tc, "linear_value_head_dim"))},
           {"conv_k", py_str(need(tc, "linear_conv_kernel_dim"))},
           {"gate", gate},
           {"n_heads", py_str(need(tc, "num_attention_heads"))},
           {"n_kv", py_str(need(tc, "num_key_value_heads"))},
           {"head_dim", I(hd)},
           {"rot_dim", I((int64_t)(hd * prf))},
           {"rope_theta", rp.get("rope_theta") ? py_str(rp["rope_theta"]) : "10000000"},
           {"ix_heads", I(tc["indexer_n_heads"].i(4))},
           {"ix_kv", I(tc["indexer_kv_heads"].i(1))},
           {"ix_dim", I(tc["indexer_head_dim"].i(128))},
           {"ix_budget", I(tc["indexer_budget"].i(2048))},
           {"ix_ratio", I(tc["indexer_compress_ratio"].i(4))},
           {"ple_layer", I(ple)},
           {"ple_dim", I(tc["ple_embed_dim"].i(hidden))},
           {"ple_conv_k", I(tc["ple_conv_kernel_size"].i(4))},
           {"ng_size", I(tc["ngram_size"].i(3))},
           {"ng_hpn", I(tc["heads_per_ngram"].i(8))},
           {"ng_base", I(tc["ngram_vocab_size_base"].i(20000000))},
           {"ng_div", I(tc["make_ngram_vocab_size_divisible_by"].i(128))},
           {"ng_seed", I(tc["seed"].i(1234))},
           {"eos", eos.kind == Json::ARR ? py_str(eos[0]) : py_str(need(tc, "eos_token_id"))},
           {"ng_files", ngf},
           {"ng_parts", I(ng["parts"].i())},
           {"ng_rows", I(ng["rows"].i())},
           {"ng_rows_per_part", I(ng["rows_per_part"].i())},
           {"ng_dim", I(ng["dim"].i())},
           {"ng_bpr", I(ng["bytes_per_row"].i())},
           {"eps", py_str(need(tc, "rms_norm_eps"))},
           {"vocab", I(vocab)},
           {"emb_file", emb_file},
           {"emb_off", I(eoff)},
           {"rec_bytes", I(f->record_bytes)},
           {"store_layer_file", layer_file_printf()}};
    const char* kinds[3] = {"gate_proj", "up_proj", "down_proj"};
    const char* mats[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; i++) {
      const MatrixLayout* L = f->matrix(mats[i]);
      if (!L) {
        err = std::string("record format has no matrix ") + mats[i];
        return false;
      }
      out.push_back({std::string("off_") + kinds[i], I(L->codes)});
      out.push_back({std::string("soff_") + kinds[i], I(L->scales)});
      out.push_back({std::string("boff_") + kinds[i], I(L->biases)});
    }
    return err.empty();
  }
  err = "engine_cfg: no engine reads family " + family + " yet";
  return false;
}

const std::vector<std::pair<std::string, char>>& cfg_keys(const std::string& family) {
  static const std::vector<std::pair<std::string, char>> q36 = {
      {"hidden", 'i'}, {"layers", 'i'}, {"experts", 'i'}, {"topk", 'i'}, {"moe_inter", 'i'}, {"shared_inter", 'i'},
      {"norm_topk", 'i'}, {"lin_nk", 'i'}, {"lin_nv", 'i'}, {"lin_dk", 'i'}, {"lin_dv", 'i'}, {"conv_k", 'i'},
      {"n_heads", 'i'}, {"n_kv", 'i'}, {"head_dim", 'i'}, {"rot_dim", 'i'}, {"vocab", 'i'}, {"eps", 'f'},
      {"rope_theta", 'd'}, {"layer_types", 's'}, {"rec_bytes", 'i'}, {"off_gate_proj", 'i'}, {"soff_gate_proj", 'i'},
      {"boff_gate_proj", 'i'}, {"off_up_proj", 'i'}, {"soff_up_proj", 'i'}, {"boff_up_proj", 'i'},
      {"off_down_proj", 'i'}, {"soff_down_proj", 'i'}, {"boff_down_proj", 'i'}, {"emb_off", 'i'}, {"emb_file", 's'},
      {"store_layer_file", 's'}};
  static const std::vector<std::pair<std::string, char>> ink = {
      {"hidden", 'i'}, {"layers", 'i'}, {"experts", 'i'}, {"topk", 'i'}, {"inter", 'i'}, {"heads", 'i'},
      {"kv_heads", 'i'}, {"head_dim", 'i'}, {"d_rel", 'i'}, {"rel_extent", 'i'}, {"window", 'i'}, {"gate_rank", 'i'},
      {"core_start", 'i'}, {"core_end", 'i'}, {"passes", 'i'}, {"block", 'i'}, {"vocab", 'i'}, {"unpadded_vocab", 'i'},
      {"sconv", 'i'}, {"eos", 'i'}, {"softcap", 'f'}, {"gate_cap", 'f'}, {"up_cap", 'f'}, {"route_scale", 'f'},
      {"eps", 'f'}, {"qk_clip", 'f'}, {"rel_clip", 'f'}, {"log_floor", 'f'}, {"log_alpha", 'f'}, {"layer_kind", 's'},
      {"layer_format", 's'}, {"nv_rec", 'i'}, {"nv_gu_c", 'i'}, {"nv_gu_s", 'i'}, {"nv_dn_c", 'i'}, {"nv_dn_s", 'i'},
      {"mx_rec", 'i'}, {"mx_gu_c", 'i'}, {"mx_gu_s", 'i'}, {"mx_dn_c", 'i'}, {"mx_dn_s", 'i'}, {"nv_trailer", 'i'},
      {"emb_off", 'i'}, {"emb_file", 's'}, {"layer_file", 's'}};
  static const std::vector<std::pair<std::string, char>> flash = {
      {"hidden", 'i'}, {"hc", 'i'}, {"hc_low", 'i'}, {"layers", 'i'}, {"experts", 'i'}, {"topk", 'i'}, {"moe_inter", 'i'},
      {"shared_inter", 'i'}, {"norm_topk", 'i'}, {"lin_nk", 'i'}, {"lin_nv", 'i'}, {"lin_dk", 'i'}, {"lin_dv", 'i'},
      {"conv_k", 'i'}, {"n_heads", 'i'}, {"n_kv", 'i'}, {"head_dim", 'i'}, {"rot_dim", 'i'}, {"ix_heads", 'i'}, {"ix_kv", 'i'},
      {"ix_dim", 'i'}, {"ix_budget", 'i'}, {"ix_ratio", 'i'}, {"ple_layer", 'i'}, {"ple_dim", 'i'}, {"ple_conv_k", 'i'},
      {"ng_size", 'i'}, {"ng_hpn", 'i'}, {"ng_base", 'i'}, {"ng_div", 'i'}, {"ng_seed", 'i'}, {"eos", 'i'}, {"ng_parts", 'i'},
      {"ng_rows", 'i'}, {"ng_rows_per_part", 'i'}, {"ng_dim", 'i'}, {"ng_bpr", 'i'}, {"vocab", 'i'}, {"eps", 'f'},
      {"rope_theta", 'd'}, {"layer_types", 's'}, {"gate", 's'}, {"ng_files", 's'}, {"rec_bytes", 'i'}, {"off_gate_proj", 'i'},
      {"soff_gate_proj", 'i'}, {"boff_gate_proj", 'i'}, {"off_up_proj", 'i'}, {"soff_up_proj", 'i'}, {"boff_up_proj", 'i'},
      {"off_down_proj", 'i'}, {"soff_down_proj", 'i'}, {"boff_down_proj", 'i'}, {"emb_off", 'i'}, {"emb_file", 's'},
      {"store_layer_file", 's'}};
  static const std::vector<std::pair<std::string, char>> none;
  if (family == "qwen3_5_moe") return q36;
  if (family == "inkling2_loop") return ink;
  if (family == "qwen4_exp") return flash;
  return none;
}

bool manifest_validate(const Manifest& m, std::vector<std::string>& problems) {
  if ((int64_t)m.layer_format.size() != m.physical_layers) problems.push_back("layer_format has the wrong length");
  if ((int64_t)m.layer_types.size() != m.physical_layers) problems.push_back("layer_types has the wrong length");
  if (!m.step_layer.empty() && (int64_t)m.step_layer.size() != m.steps) problems.push_back("step_layer has the wrong length");
  for (auto& f : m.formats) {
    if (m.page && f.record_bytes % m.page) problems.push_back(f.id + ": record_bytes is not a whole number of pages");
    for (auto& L : f.matrices) {
      int64_t n = L.rows * L.cols;
      int64_t code_bytes = f.kind == "nvfp4" || (f.kind == "affine" && f.bits == 4) ? n / 2 : n;
      int64_t scale_bytes = (n / f.group) * (f.kind == "affine" ? 2 : 1);
      if (L.codes < 0 || L.codes + code_bytes > f.record_bytes_used) problems.push_back(f.id + "." + L.name + ": codes outside the record");
      if (L.scales < 0 || L.scales + scale_bytes > f.record_bytes_used) problems.push_back(f.id + "." + L.name + ": scales outside the record");
      if (L.biases >= 0 && L.biases + scale_bytes > f.record_bytes_used) problems.push_back(f.id + "." + L.name + ": biases outside the record");
    }
    if (f.trailer >= 0 && f.trailer + 16 > f.record_bytes_used) problems.push_back(f.id + ": trailer outside the record");
  }
  for (auto& id : m.layer_format)
    if (!m.format(id)) problems.push_back("layer_format names unknown format " + id);
  int ndef = 0;
  for (auto& r : m.routers) {
    ndef += r.is_default;
    if (!m.part(r.path)) problems.push_back("router " + r.id + ": " + r.path + " is not a part");
    if (r.targets != m.steps) problems.push_back("router " + r.id + ": targets != steps");
  }
  if (!m.routers.empty() && ndef != 1) problems.push_back("routers: exactly one default expected");
  for (auto& d : m.derived)
    for (auto& f : d.from)
      if (!m.part(f)) problems.push_back("derived " + d.id + ": source " + f + " is not a part");
  for (int L = 0; L < m.physical_layers; L++) {
    char b[512];
    snprintf(b, sizeof b, m.layer_file_printf().c_str(), L);
    const Part* p = m.part(b);
    const RecordFormat* f = L < (int)m.layer_format.size() ? m.format(m.layer_format[L]) : nullptr;
    if (!p) problems.push_back(std::string(b) + " is not a part");
    else if (f && p->bytes >= 0 && p->bytes != f->record_bytes * m.experts) problems.push_back(std::string(b) + ": bytes != experts x record_bytes");
  }
  return problems.empty();
}

bool Manifest::check_derived(const Derived& d, const std::string& path, std::string& err) const {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    err = path + ": missing";
    return false;
  }
  if (d.bytes >= 0 && (int64_t)st.st_size != d.bytes) {
    err = path + ": " + std::to_string((long long)st.st_size) + " bytes, the manifest's " + d.id + " lists " + std::to_string((long long)d.bytes);
    return false;
  }
  if (d.sha256.empty()) return true;
  const long long mt = (long long)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
  const std::string base = path.substr(path.rfind('/') + 1), stamps = cache_dir() + "/checked", stamp = stamps + "/" + base + ".sha256";
  char want[256];
  snprintf(want, sizeof want, "%lld %lld %s", (long long)st.st_size, mt, d.sha256.c_str());
  if (FILE* f = fopen(stamp.c_str(), "r")) {  // checked before, and the file has not changed since
    char got[256] = {0};
    const bool same = fgets(got, sizeof got, f) && std::string(got).rfind(want, 0) == 0;
    fclose(f);
    if (same) return true;
  }
  std::string e2;
  const std::string h = sha256_file(path, e2, false);
  if (h != d.sha256) {
    err = path + ": sha256 " + (h.empty() ? e2 : h) + " is not the manifest's " + d.id + " (" + d.sha256 + "): derive it again or fetch it again";
    return false;
  }
  mkdir(cache_dir().c_str(), 0755);
  mkdir(stamps.c_str(), 0755);
  if (FILE* f = fopen(stamp.c_str(), "w")) {  // (a read-only folder: checked again next time)
    fprintf(f, "%s\n", want);
    fclose(f);
  }
  return true;
}

}  // namespace ms
