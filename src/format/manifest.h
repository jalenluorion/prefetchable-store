// The weight-repo manifest (moved-store.json, FORMAT.md section 2), read without third-party code. The engine takes its
// configuration from it (resolve_model in src/models/registry.cpp); `manifest_dump` prints and checks it against the cfg
// lines of a resident file.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "json.h"

namespace ms {

struct Part {
  std::string path, sha256, role, precision;
  int64_t bytes = -1;  // -1: not recorded
  bool optional = false;
};

struct MatrixLayout {
  std::string name, interleave;
  int64_t rows = 0, cols = 0;
  int64_t codes = -1, scales = -1, biases = -1;
};

struct RecordFormat {
  std::string id, kind;
  int bits = 0, group = 0;
  int64_t record_bytes = 0, record_bytes_used = 0, align = 0;
  std::vector<MatrixLayout> matrices;
  int64_t trailer = -1;
  const MatrixLayout* matrix(const std::string& n) const;
};

struct RouterEntry {
  std::string id, path, form, index;
  int lag = 0, hidden = 0, targets = 0;
  bool is_default = false;
};

struct Derived {
  std::string id, engine_format, sha256;
  std::vector<std::string> from;
  std::string path;  // where the weight repo ships the engine-ready file (e.g. derived/<name>), "" = not shipped
  int64_t bytes = -1;
};

using Cfg = std::vector<std::pair<std::string, std::string>>;

struct Manifest {
  std::string dir;  // the model folder
  Json raw, config;
  std::string format_version, family, name;
  std::vector<Part> parts;
  // model
  int64_t hidden = 0, vocab = 0, vocab_unpadded = 0, physical_layers = 0, steps = 0;
  std::vector<int> step_layer, step_pass;
  std::string layer_types, emb_file, emb_tensor, emb_dtype;
  // experts
  std::string layer_file;  // "store/layer_{L:02d}.bin"
  int64_t experts = 0, top_k = 0, page = 0;
  std::string router_rule, activation;
  double route_scale = 1.0;
  std::vector<std::string> layer_format;  // per physical layer
  // per physical layer, relative to the folder: a profile's own record files (profiles.<name>.experts.layer_files);
  // empty = layer_file for every layer
  std::vector<std::string> layer_paths;
  std::vector<RecordFormat> formats;      // manifest order
  std::vector<RouterEntry> routers;
  std::vector<Derived> derived;

  bool load(const std::string& model_dir, std::string& err);  // moved-store.json + config.json
  // profiles.<name>.experts, when present: its layer_format (one id per physical layer) replaces the folder's, and its
  // layer_files ({"20": "store_nv4/layer_20.bin", ...}) name that profile's own files for those layers (e.g. Inkling's
  // phone-small: layers 20-23 rewritten as NVFP4, the originals untouched). No experts entry: nothing changes.
  bool apply_profile_experts(const std::string& profile, std::string& err);
  const Part* part(const std::string& path) const;
  const RecordFormat* format(const std::string& id) const;
  const RouterEntry* router(const std::string& id) const;  // "" = the default router
  const Derived* derived_of(const std::string& id) const;
  std::string layer_path(int L) const;  // absolute path of store/layer_XX.bin
  std::string layer_file_printf() const;  // "store/layer_%02d.bin" (the engines' cfg form)

  // The cache folder of FORMAT.md section 6 and the name a derived file takes in it. cache_override (set by the
  // caller, e.g. an app's Application Support folder) replaces the folder.
  std::string cache_override;
  std::string cache_dir() const;
  std::string derived_path(const Derived& d) const;
  // An engine-ready file the model folder already holds: the entry's `path`, else derived/<the derived_path name>
  // (a folder downloaded from the Hub); "" if neither exists. The cache folder is not looked at here.
  std::string shipped_path(const Derived& d) const;
  // An engine-ready file about to be used against the manifest's entry: its size (when listed) and sha256 (when
  // listed); the hash is computed once per file version and remembered in the cache folder (checked/<name>.sha256: size,
  // modification time, hash). False with err when it does not match; true when it matches or the entry lists no hash.
  bool check_derived(const Derived& d, const std::string& path, std::string& err) const;

  // The configuration today's engines read from the cfg lines of their resident file, built from the manifest,
  // config.json and the resident safetensors header (the embedding offset), in the order the converters write it.
  bool engine_cfg(Cfg& out, std::string& err) const;
};

// The cfg keys each family's Cfg::from reads, and how it reads them ('i' geti / strtol, 'f' getf / strtod -> float,
// 'd' getf -> double, 's' gets).
const std::vector<std::pair<std::string, char>>& cfg_keys(const std::string& family);

// validates the manifest's own consistency (format version, record layouts fit their records, routers point at parts)
bool manifest_validate(const Manifest& m, std::vector<std::string>& problems);

}  // namespace ms
