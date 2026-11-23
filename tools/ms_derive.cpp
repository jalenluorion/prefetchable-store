// ms_derive: makes the engine's derived files (FORMAT.md section 6) in C++ from a model folder in the moved-store format,
// writes them into the cache folder (<model>/.dmeng/), and checks them against the manifest's derived[].sha256 and,
// optionally, against a file made by today's Python converters.
//
//   ms_derive resident DIR [--bits 8|4] [--out FILE] [--ref FILE]   (--bits 4: resident_q4, the 4-bit variant)
//   ms_derive router   DIR ROUTER_ID [--bits 8|4] [--split] [--out FILE] [--ref FILE]
//   ms_derive router-file ROUTER.safetensors OUT [--bits 8|4] [--split] [--ref FILE]
//
// --ref: whole-file comparison, and on a difference a per-tensor report (which payloads differ, how many bytes).
#include <sys/stat.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "derive.h"
#include "files.h"
#include "manifest.h"

using namespace ms;

static int usage() {
  fprintf(stderr,
          "usage: ms_derive resident DIR [--bits 8|4] [--out FILE] [--ref FILE]\n"
          "       ms_derive router DIR ROUTER_ID [--bits 8|4] [--split] [--out FILE] [--ref FILE]\n"
          "       ms_derive router-file ROUTER.safetensors OUT [--bits 8|4] [--split] [--ref FILE]\n");
  return 2;
}

static bool compare_files(const std::string& a, const std::string& ref) {
  std::string why;
  if (files_identical(a, ref, why)) {
    printf("  identical to %s (whole file)\n", ref.c_str());
    return true;
  }
  printf("  DIFFERS from %s: %s\n", ref.c_str(), why.c_str());
  DmFile A, B;
  std::string err;
  if (!A.open(a, err) || !B.open(ref, err)) {
    printf("  %s\n", err.c_str());
    return false;
  }
  int same = 0, diff = 0, missing = 0;
  for (auto& kv : B.cfg) {
    const std::string* v = A.get(kv.first);
    if (!v || *v != kv.second) printf("  cfg %s: ours '%s', ref '%s'\n", kv.first.c_str(), v ? v->c_str() : "(none)", kv.second.c_str());
  }
  for (auto& t : B.tensors) {
    const DmTensor* u = A.find(t.name);
    if (!u) {
      missing++;
      continue;
    }
    std::vector<uint8_t> x, y;
    if (!A.read(*u, x, err) || !B.read(t, y, err)) {
      printf("  %s\n", err.c_str());
      return false;
    }
    if (x == y) {
      same++;
    } else {
      size_t nd = 0;
      for (size_t i = 0; i < std::min(x.size(), y.size()); i++) nd += x[i] != y[i];
      if (diff < 20) printf("  tensor %s differs: %zu of %zu bytes\n", t.name.c_str(), nd, y.size());
      diff++;
    }
  }
  printf("  tensors: %d identical, %d differ, %d missing (of %zu)\n", same, diff, missing, B.tensors.size());
  return false;
}

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  std::string cmd = argv[1];
  std::vector<std::string> pos;
  std::string out, ref;
  int bits = 8;
  bool split = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--ref" && i + 1 < argc) ref = argv[++i];
    else if (a == "--bits" && i + 1 < argc) bits = atoi(argv[++i]);
    else if (a == "--split") split = true;
    else pos.push_back(a);
  }
  if (bits != 8 && bits != 4) return usage();
  std::string err, expect_sha;
  DeriveStats st;
  if (cmd == "resident" || cmd == "router") {
    if (pos.empty() || (cmd == "router" && pos.size() < 2)) return usage();
    Manifest m;
    if (!m.load(pos[0], err)) {
      fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    Derived d;
    std::string src;
    if (cmd == "resident") {
      const std::string rid = bits == 4 ? "resident_q4" : "resident_q8";
      const Derived* x = m.derived_of(rid);
      d = x ? *x : Derived{rid, "dmres1", "", {"resident.safetensors"}};
    } else {
      const RouterEntry* r = m.router(pos[1]);
      if (!r) {
        fprintf(stderr, "no router %s in the manifest\n", pos[1].c_str());
        return 1;
      }
      std::string id = "router." + r->id + (bits == 8 ? ".q8" : ".q4") + (split ? "_split" : "");
      const Derived* x = m.derived_of(id);
      d = x ? *x : Derived{id, "dmr1", "", {r->path}};
      src = m.dir + "/" + r->path;
    }
    expect_sha = d.sha256;
    if (out.empty()) {
      out = m.derived_path(d);
      mkdir(m.cache_dir().c_str(), 0755);
    }
    printf("%s -> %s\n", d.id.c_str(), out.c_str());
    bool ok = cmd == "resident" ? derive_resident(m, out, st, err, bits) : derive_router(src, out, bits, split, st, err);
    if (!ok) {
      fprintf(stderr, "failed: %s\n", err.c_str());
      return 1;
    }
  } else if (cmd == "router-file") {
    if (pos.size() < 2) return usage();
    out = pos[1];
    printf("%s -> %s\n", pos[0].c_str(), out.c_str());
    if (!derive_router(pos[0], out, bits, split, st, err)) {
      fprintf(stderr, "failed: %s\n", err.c_str());
      return 1;
    }
  } else {
    return usage();
  }
  std::string sha = sha256_file(out, err);
  uint64_t n = 0;
  file_size(out, n);
  printf("  %d matrices, %d tensors, %.3f GB, %.1f s, sha256 %s\n", st.matrices, st.tensors, n / 1e9, st.seconds, sha.c_str());
  bool ok = true;
  if (!expect_sha.empty()) {
    bool m = sha == expect_sha;
    printf("  manifest derived sha256 %s: %s\n", expect_sha.c_str(), m ? "MATCH" : "MISMATCH");
    ok &= m;
  }
  if (!ref.empty()) ok &= compare_files(out, ref);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
