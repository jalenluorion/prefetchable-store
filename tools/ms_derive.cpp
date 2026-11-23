// ms_derive: makes the engine's derived files (FORMAT.md section 6) in C++ from a model folder in the moved-store format,
// writes them into the cache folder (<model>/.dmeng/), and checks them against the manifest's derived[].sha256 and,
// optionally, against a file made by today's Python converters.
//
//   ms_derive resident DIR [--bits 8|4] [--out FILE] [--ref FILE]   (--bits 4: resident_q4, the 4-bit variant)
//   ms_derive resident DIR --bits 4 --from-calibrated FILE --out FILE [--ref FILE]
//       resident_q4 with a calibrated export's codes / scales / biases (derive.h); --out is required (never the cache path)
//   ms_derive router   DIR ROUTER_ID [--bits 8|4] [--split] [--out FILE] [--ref FILE]
//   ms_derive router-file ROUTER.safetensors OUT [--bits 8|4] [--split] [--ref FILE]
//   ms_derive embeddings DIR --out FILE
//       the manifest's embedding tensor alone (model.embedding: file / tensor / dtype), bytes copied unchanged into a
//       one-tensor safetensors file: the phone layout's embeddings.safetensors (no bf16 resident on the device)
//
// --ref: whole-file comparison, and on a difference a per-tensor report (which payloads differ, how many bytes).
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <algorithm>
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
          "       ms_derive resident DIR --bits 4 --from-calibrated FILE --out FILE [--ref FILE]\n"
          "       ms_derive router DIR ROUTER_ID [--bits 8|4] [--split] [--out FILE] [--ref FILE]\n"
          "       ms_derive router-file ROUTER.safetensors OUT [--bits 8|4] [--split] [--ref FILE]\n"
          "       ms_derive embeddings DIR --out FILE\n");
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
  std::string out, ref, cal;
  int bits = 8;
  bool split = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--ref" && i + 1 < argc) ref = argv[++i];
    else if (a == "--from-calibrated" && i + 1 < argc) cal = argv[++i];
    else if (a == "--bits" && i + 1 < argc) bits = atoi(argv[++i]);
    else if (a == "--split") split = true;
    else pos.push_back(a);
  }
  if (bits != 8 && bits != 4) return usage();
  if (!cal.empty() && (cmd != "resident" || bits != 4 || out.empty())) {
    fprintf(stderr, "--from-calibrated: resident --bits 4 with an explicit --out only\n");
    return 2;
  }
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
    expect_sha = cal.empty() ? d.sha256 : "";  // a calibrated file has its own sha256, not the manifest's
    if (out.empty()) {
      out = m.derived_path(d);
      mkdir(m.cache_dir().c_str(), 0755);
    }
    printf("%s -> %s\n", d.id.c_str(), out.c_str());
    if (!cal.empty()) printf("  calibrated 4-bit matrices from %s\n", cal.c_str());
    bool ok = cmd == "resident" ? derive_resident(m, out, st, err, bits, cal) : derive_router(src, out, bits, split, st, err);
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
  } else if (cmd == "embeddings") {  // the embedding table alone, bytes unchanged
    if (pos.empty() || out.empty()) return usage();
    Manifest m;
    if (!m.load(pos[0], err)) {
      fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    const std::string src = m.emb_file.empty() || m.emb_file[0] == '/' ? m.emb_file : m.dir + "/" + m.emb_file;
    SafeTensors S;
    if (!S.open(src, err)) {
      fprintf(stderr, "%s\n", err.c_str());
      return 1;
    }
    const StTensor* t = S.find(m.emb_tensor);
    if (!t) {
      fprintf(stderr, "no tensor %s in %s\n", m.emb_tensor.c_str(), src.c_str());
      return 1;
    }
    printf("%s:%s (%s) -> %s\n", src.c_str(), m.emb_tensor.c_str(), t->dtype.c_str(), out.c_str());
    // the header of the Hub staging tool (local/hub_staging/tools/stage.py embeddings()), byte for byte, so a derived
    // file and the Hub's part have the same sha256: compact JSON, __metadata__ first, padded with spaces to 8 bytes
    const Part* sp = m.part(m.emb_file);
    std::string hdr = "{\"__metadata__\":{\"format\":\"moved-store-embedding\",\"format_version\":\"1.0\",\"tensor\":\"" +
                      t->name + "\",\"source\":\"" + m.emb_file + "\",\"source_sha256\":\"" + (sp ? sp->sha256 : "") +
                      "\",\"note\":\"the embedding rows alone, byte for byte the source tensor's data; the engine reads rows "
                      "from it when the source file is absent\"},\"" + t->name + "\":{\"dtype\":\"" + t->dtype + "\",\"shape\":[";
    for (size_t i = 0; i < t->shape.size(); i++) hdr += (i ? "," : "") + std::to_string(t->shape[i]);
    hdr += "],\"data_offsets\":[0," + std::to_string(t->end - t->begin) + "]}}";
    while (hdr.size() % 8) hdr += ' ';
    FILE* f = fopen(out.c_str(), "wb");
    if (!f) {
      fprintf(stderr, "cannot write %s\n", out.c_str());
      return 1;
    }
    const uint64_t hl = hdr.size();
    fwrite(&hl, 8, 1, f);
    fwrite(hdr.data(), 1, hdr.size(), f);
    std::vector<uint8_t> buf(64 << 20);
    for (uint64_t o = 0, n = t->end - t->begin; o < n;) {
      const size_t k = (size_t)std::min<uint64_t>(buf.size(), n - o);
      if (pread(S.fd, buf.data(), k, (off_t)(S.data_offset(*t) + o)) != (ssize_t)k || fwrite(buf.data(), 1, k, f) != k) {
        fprintf(stderr, "copy failed at %llu\n", (unsigned long long)o);
        fclose(f);
        return 1;
      }
      o += k;
    }
    if (fclose(f) != 0) return 1;
    st.tensors = 1;
  } else {
    return usage();
  }
  std::string sha = sha256_file(out, err);
  uint64_t n = 0;
  file_size(out, n);
  printf("  %d matrices, %d tensors, %.3f GB, %.1f s, sha256 %s\n", st.matrices, st.tensors, n / 1e9, st.seconds, sha.c_str());
  if (!cal.empty()) printf("  %d matrices from the calibrated export\n", st.calibrated);
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
