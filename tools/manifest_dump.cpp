// manifest_dump: reads a model folder's moved-store.json (FORMAT.md) with the standalone manifest library and prints
// what the engine needs; checks the folder against the manifest; compares the engine configuration it derives with
// the cfg lines of a DMENGRES1 resident file (the keys today's Cfg::from reads).
//
//   manifest_dump DIR                    summary + the engine cfg lines built from the manifest
//   manifest_dump DIR --compare FILE     every key Cfg::from reads: manifest value vs FILE's value (typed as the engine reads it)
//   manifest_dump DIR --check [--sha]    manifest consistency, every part's size (and sha256), cache files present
//   manifest_dump DIR --check --no-files manifest consistency only (a manifest kept apart from its weights)
//   manifest_dump DIR --check --profile NAME [--sha]   a profile-only download: the parts the profile lists (its
//                                        `parts` objects or `files` paths) are present with their sizes (and sha256);
//                                        parts outside the profile may be absent
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

#include "files.h"
#include "manifest.h"

using namespace ms;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: manifest_dump DIR [--compare RESIDENT.bin] [--check [--sha | --no-files] [--profile NAME]]\n");
    return 2;
  }
  std::string dir = argv[1], cmp, profile;
  bool check = false, sha = false, nofiles = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--compare" && i + 1 < argc) cmp = argv[++i];
    else if (a == "--check") check = true;
    else if (a == "--sha") sha = check = true;
    else if (a == "--no-files") nofiles = check = true;
    else if (a == "--profile" && i + 1 < argc) profile = argv[++i];
  }
  Manifest m;
  std::string err;
  if (!m.load(dir, err)) {
    fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  {  // the engine reads the embedding rows from embeddings.safetensors when the bf16 resident is absent (resolve_model)
    struct stat st;
    if (stat((dir + "/" + m.emb_file).c_str(), &st) != 0 && stat((dir + "/embeddings.safetensors").c_str(), &st) == 0) {
      printf("%s absent: embedding rows from embeddings.safetensors\n", m.emb_file.c_str());
      m.emb_file = "embeddings.safetensors";
    }
  }
  int bad = 0;
  printf("%s: format %s, family %s, %zu parts, %zu record formats, %zu routers, %zu derived\n", m.name.c_str(),
         m.format_version.c_str(), m.family.c_str(), m.parts.size(), m.formats.size(), m.routers.size(), m.derived.size());
  printf("model: hidden %lld, vocab %lld, physical layers %lld, steps %lld, layer types %s\n", (long long)m.hidden,
         (long long)m.vocab, (long long)m.physical_layers, (long long)m.steps, m.layer_types.c_str());
  printf("experts: %lld per layer, top %lld, router %s x %g, activation %s, layer file %s\n", (long long)m.experts,
         (long long)m.top_k, m.router_rule.c_str(), m.route_scale, m.activation.c_str(), m.layer_file.c_str());
  for (auto& f : m.formats) {
    printf("  format %s: %s %d-bit g%d, record %lld B (%lld used)", f.id.c_str(), f.kind.c_str(), f.bits, f.group,
           (long long)f.record_bytes, (long long)f.record_bytes_used);
    for (auto& L : f.matrices) printf("; %s [%lld, %lld] @%lld", L.name.c_str(), (long long)L.rows, (long long)L.cols, (long long)L.codes);
    printf("\n");
  }
  for (auto& r : m.routers)
    printf("  router %s: %s lag %d hidden %d, %s x %d%s\n", r.id.c_str(), r.form.c_str(), r.lag, r.hidden, r.index.c_str(),
           r.targets, r.is_default ? " (default)" : "");
  printf("cache: %s\n", m.cache_dir().c_str());
  for (auto& d : m.derived) printf("  derived %s -> %s\n", d.id.c_str(), m.derived_path(d).c_str());

  Cfg cfg;
  if (!m.engine_cfg(cfg, err)) {
    printf("engine cfg: %s\n", err.c_str());
    if (!cmp.empty()) bad++;
  } else if (cmp.empty() && !check) {
    printf("engine cfg (%zu lines):\n", cfg.size());
    for (auto& kv : cfg) printf("cfg %s %s\n", kv.first.c_str(), kv.second.c_str());
  }

  if (!cmp.empty() && err.empty()) {
    DmFile f;
    if (!f.open(cmp, err)) {
      fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    auto ours = [&](const std::string& k) -> const std::string* {
      for (auto& kv : cfg)
        if (kv.first == k) return &kv.second;
      return nullptr;
    };
    int n = 0, ok = 0, same_text = 0;
    for (auto& kc : cfg_keys(m.family)) {
      const std::string *a = ours(kc.first), *b = f.get(kc.first);
      bool eq = false;
      if (a && b) {
        if (kc.second == 'i') eq = strtol(a->c_str(), nullptr, 10) == strtol(b->c_str(), nullptr, 10);
        else if (kc.second == 'f') eq = (float)strtod(a->c_str(), nullptr) == (float)strtod(b->c_str(), nullptr);
        else if (kc.second == 'd') eq = strtod(a->c_str(), nullptr) == strtod(b->c_str(), nullptr);
        else eq = *a == *b;
        same_text += *a == *b;
      }
      n++;
      ok += eq;
      printf("  %-18s %c manifest %-42s file %-42s %s\n", kc.first.c_str(), kc.second, a ? a->c_str() : "(missing)",
             b ? b->c_str() : "(missing)", eq ? "ok" : "DIFFERENT");
    }
    int all_same = 0;
    for (auto& kv : f.cfg) {
      const std::string* a = ours(kv.first);
      all_same += a && *a == kv.second;
    }
    printf("compare %s: %d of %d keys read by Cfg::from agree (%d also as text); %d of the file's %zu cfg lines identical\n",
           cmp.c_str(), ok, n, same_text, all_same, f.cfg.size());
    if (ok != n) bad++;
  }

  if (check) {
    std::vector<std::string> problems;
    manifest_validate(m, problems);
    for (auto& p : problems) printf("  manifest: %s\n", p.c_str());
    bad += (int)problems.size();
    int present = 0, missing = 0, size_bad = 0, sha_bad = 0;
    std::vector<Part> want = m.parts;  // what must be there: every required part, or the profile's own list
    if (!profile.empty()) {
      const Json* pj = m.raw["profiles"].get(profile);
      if (!pj) {
        printf("  profile %s: not in the manifest's profiles\n", profile.c_str());
        bad++;
      } else {
        want.clear();
        auto add = [&](const std::string& path, int64_t bytes, const std::string& sha) {
          Part q;
          if (const Part* mp = m.part(path)) q = *mp;
          q.path = path;
          if (bytes >= 0) q.bytes = bytes;
          if (!sha.empty()) q.sha256 = sha;
          q.optional = false;
          want.push_back(q);
        };
        for (auto& x : (*pj)["parts"].arr)
          if (x.kind == Json::OBJ) add(x["path"].s(), x["bytes"].kind == Json::NUM ? x["bytes"].i() : -1, x["sha256"].str_or(""));
          else add(x.s(), -1, "");
        if (want.empty())
          for (auto& x : (*pj)["files"].arr) add(x.s(), -1, "");
        printf("profile %s: %zu parts listed\n", profile.c_str(), want.size());
      }
    }
    for (auto& p : want) {
      if (nofiles) break;
      if (p.path == "moved-store.json") continue;
      uint64_t n = 0;
      std::string full = m.dir + "/" + p.path;
      if (!file_size(full, n)) {
        if (!p.optional) {
          printf("  missing: %s\n", p.path.c_str());
          missing++;
        }
        continue;
      }
      present++;
      if (p.bytes >= 0 && (int64_t)n != p.bytes) {
        printf("  size: %s is %llu B, manifest %lld\n", p.path.c_str(), (unsigned long long)n, (long long)p.bytes);
        size_bad++;
      }
      if (sha && !p.sha256.empty()) {
        std::string e, h = sha256_file(full, e);
        if (h != p.sha256) {
          printf("  sha256: %s is %s, manifest %s\n", p.path.c_str(), h.c_str(), p.sha256.c_str());
          sha_bad++;
        }
      }
    }
    int cached = 0;
    for (auto& d : m.derived) {
      uint64_t n;
      if (!m.shipped_path(d).empty() || file_size(m.derived_path(d), n)) cached++;
    }
    printf("check: %d parts present, %d missing, %d wrong size%s, %d of %zu derived files in derived/ or the cache\n", present, missing, size_bad,
           sha ? (", " + std::to_string(sha_bad) + " wrong sha256").c_str() : " (sha256 not checked; --sha)", cached, m.derived.size());
    bad += missing + size_bad + sha_bad;
  }
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
