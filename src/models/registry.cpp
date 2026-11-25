// The model families of this build.
#include "dmeng/model.h"

#include <sys/mount.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../format/derive.h"
#include "../format/files.h"
#include "../format/manifest.h"

namespace dm {

std::unique_ptr<Model> make_qwen3_5_moe();
std::unique_ptr<Model> make_inkling2_loop();
std::unique_ptr<Model> make_qwen4_exp();

std::unique_ptr<Model> make_model(const std::string& family, std::string& err) {
  if (family == "qwen3_5_moe") return make_qwen3_5_moe();
  if (family == "inkling2_loop") return make_inkling2_loop();
  if (family == "qwen4_exp") return make_qwen4_exp();
  err = "unknown model family '" + family + "' (this build: qwen3_5_moe, inkling2_loop, qwen4_exp)";
  return nullptr;
}

std::string detect_family(const std::string& resident, std::string& err) {
  std::map<std::string, std::string> cfg;
  if (!ResFile::read_cfg(resident, cfg, err)) return "";
  auto it = cfg.find("model");
  if (it != cfg.end()) return it->second;
  if (cfg.count("layer_types") && cfg.count("lin_nk")) return "qwen3_5_moe";
  err = resident + ": cannot tell the model family (no cfg `model`)";
  return "";
}

static bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

bool resolve_model(const std::string& dir, const std::string& resident, Mem res_mem, LoadOpts& lo, std::string& family,
                   std::string& err, const std::string& derived_dir, const std::string& emb_file, const std::string& profile) {
  lo.dir = dir;
  lo.res_mem = res_mem;
  lo.cfg.clear();
  if (!exists(dir + "/moved-store.json")) {  // no manifest: the resident file says everything
    lo.resident = resident.empty() ? dir + "/dmeng_resident.bin" : resident;
    family = detect_family(lo.resident, err);
    return !family.empty();
  }
  ms::Manifest m;
  if (!m.load(dir, err)) return false;
  m.cache_override = derived_dir;
  if (!m.apply_profile_experts(profile, err)) return false;  // a profile's own record formats / files (phone-small)
  if (!emb_file.empty()) {
    m.emb_file = emb_file;
  } else if (!exists(dir + "/" + m.emb_file) && exists(dir + "/embeddings.safetensors")) {
    fprintf(stderr, "dmeng: %s absent: embedding rows from embeddings.safetensors\n", m.emb_file.c_str());
    m.emb_file = "embeddings.safetensors";
    // an engine-ready file like the others: checked against the manifest (its derived entry, else its part entry)
    ms::Derived de;
    if (const ms::Derived* d = m.derived_of("embeddings")) de = *d;
    else if (const ms::Part* q = m.part("embeddings.safetensors")) de.id = "embeddings", de.sha256 = q->sha256, de.bytes = q->bytes;
    if (!m.check_derived(de, dir + "/embeddings.safetensors", err)) return false;
  }
  family = m.family;
  if (!m.engine_cfg(lo.cfg, err)) return false;
  if (!resident.empty()) {
    lo.resident = resident;
    return true;
  }
  const ms::Derived* d = m.derived_of("resident_q8");
  if (!d) {
    err = "moved-store.json: no derived resident_q8 entry";
    return false;
  }
  lo.resident = m.shipped_path(*d);  // a folder from the Hub ships it in derived/
  if (!lo.resident.empty()) return m.check_derived(*d, lo.resident, err);
  lo.resident = m.derived_path(*d);
  if (exists(lo.resident)) return m.check_derived(*d, lo.resident, err);
  mkdir(m.cache_dir().c_str(), 0755);
  const std::string legacy = dir + "/dmeng_resident.bin";
  if (exists(legacy) && !d->sha256.empty()) {  // the file today's converter wrote: link it if it is the derivation's
    std::string e2;
    if (ms::sha256_file(legacy, e2, false) == d->sha256 && symlink(legacy.c_str(), lo.resident.c_str()) == 0) return true;
  }
  {  // the derived file is about the size of the resident safetensors: leave room for the system and other work
    const ms::Part* src = d->from.empty() ? nullptr : m.part(d->from[0]);
    struct statfs fs;
    const double need = (src && src->bytes > 0 ? (double)src->bytes : 5e9) + 5e9;
    if (statfs(m.cache_dir().c_str(), &fs) == 0 && (double)fs.f_bavail * fs.f_bsize < need) {
      err = "not enough free disk to derive the resident file into " + m.cache_dir() + " (need " + std::to_string((int)(need / 1e9)) +
            " GB); free space, or derive it elsewhere with ms_derive and pass --resident";
      return false;
    }
  }
  fprintf(stderr, "deriving the 8-bit resident file into %s (first run)\n", lo.resident.c_str());
  ms::DeriveStats st;
  const std::string tmp = lo.resident + ".part";
  if (!ms::derive_resident(m, tmp, st, err)) return false;
  if (!d->sha256.empty()) {
    std::string e2;
    if (ms::sha256_file(tmp, e2, false) != d->sha256) {
      err = tmp + ": derived file does not match the manifest's sha256";
      return false;
    }
  }
  if (rename(tmp.c_str(), lo.resident.c_str()) != 0) {
    err = "cannot move " + tmp + " into place";
    return false;
  }
  return true;
}

}  // namespace dm
