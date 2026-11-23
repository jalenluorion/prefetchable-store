// The C++ eviction decisions against the offline simulator (results/ios/evict/evict_sim.py): replays a recorded routing
// file (int32 positions, layers, K header + int16 [positions, layers, K]) through dm::Cache the way Core::lossless and
// Core::finish_step drive it in cache-only mode (picks in order: a hit pins its slot, a miss takes victim(L, 0) and is
// issued urgent, after pre_use() of the step's picks; then use() and unpin for every pick), reads skipped, and writes one line per eviction:
// "step victim_record" (record = layer * 256 + expert), then "misses M picks P".
//   evict_check RECORD.bin SLOTS freq|hazard OUT.txt [uniform]   (uniform: Cache::uniform_layer_caps)
// Build: clang++ -std=c++17 -O2 -Iinclude tests/evict_check/evict_check.cpp src/core/store.cpp src/core/weights.cpp
// Check: results/ios/evict/check_cpp.sh (the simulator writes the same lines with --log; the files must be identical).
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "dmeng/store.h"

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    fprintf(stderr, "usage: evict_check RECORD.bin SLOTS freq|hazard OUT.txt [uniform]\n");
    return 2;
  }
  FILE* f = fopen(argv[1], "rb");
  if (!f) return fprintf(stderr, "cannot open %s\n", argv[1]), 1;
  int hdr[3];
  if (fread(hdr, 4, 3, f) != 3) return 1;
  const int npos = hdr[0], NL = hdr[1], K = hdr[2], NE = 256;
  std::vector<int16_t> rec((size_t)npos * NL * K);
  if (fread(rec.data(), 2, rec.size(), f) != rec.size()) return fprintf(stderr, "short record\n"), 1;
  fclose(f);
  const int slots = atoi(argv[2]);
  const std::string pol = argv[3];
  const size_t R = 1769472;

  const char* td = getenv("TMPDIR");  // the cache opens one (empty) file per layer; reads are skipped
  std::string tmpl = std::string(td && *td ? td : "/tmp") + "/evict_check_XXXXXX";
  const char* dir = mkdtemp(&tmpl[0]);
  if (!dir) return 1;
  for (int L = 0; L < NL; L++) {
    char p[512];
    snprintf(p, sizeof p, "%s/layer_%02d.bin", dir, L);
    FILE* g = fopen(p, "wb");
    if (!g) return 1;
    fclose(g);
  }
  std::vector<int> step_layer(NL), layer_pool(NL, 0);
  for (int L = 0; L < NL; L++) step_layer[L] = L;
  dm::Cache c;
  std::string err;
  if (!c.init(NL, NE, K, step_layer, layer_pool, {R}, {1.0}, dir, "layer_%02d.bin", (double)slots * (double)R, 1, 1,
              dm::Mem::PLAIN, true, err))
    return fprintf(stderr, "init: %s\n", err.c_str()), 1;
  if (c.slots(0) != slots) return fprintf(stderr, "slots %d != %d\n", c.slots(0), slots), 1;
  if (pol == "hazard") c.enable_hazard();
  else if (pol != "freq") return fprintf(stderr, "policy freq|hazard\n"), 2;
  if (argc == 6) c.uniform_layer_caps();

  FILE* out = fopen(argv[4], "w");
  std::vector<int> owner(slots, -1);
  long misses = 0, picks = 0;
  for (int t = 0; t < npos; t++) {
    c.tok_now = t;
    for (int L = 0; L < NL; L++) {
      const long step = (long)t * NL + L;
      c.now_step = step;
      int sl[16];
      int32_t pk[16];
      for (int j = 0; j < K; j++) pk[j] = rec[((size_t)t * NL + L) * K + j];
      c.pre_use(L, pk, K, t);  // as Core::lossless
      for (int j = 0; j < K; j++) {
        const int e = rec[((size_t)t * NL + L) * K + j];
        int s = c.slot_of(L, e);
        picks++;
        if (s >= 0) {
          c.slot(L, s).pins++;
        } else {
          s = c.victim(L, 0);
          if (s < 0) return fprintf(stderr, "no slot at step %ld\n", step), 1;
          if (owner[s] >= 0) fprintf(out, "%ld %d\n", step, owner[s]);
          c.issue(L, s, e, true, step);
          c.slot(L, s).pins = 1;
          owner[s] = L * NE + e;
          misses++;
        }
        sl[j] = s;
      }
      for (int j = 0; j < K; j++) {
        c.use(L, sl[j], t, step);
        c.unpin(L, sl[j]);
      }
    }
  }
  fprintf(out, "misses %ld picks %ld\n", misses, picks);
  fclose(out);
  for (int L = 0; L < NL; L++) {
    char p[512];
    snprintf(p, sizeof p, "%s/layer_%02d.bin", dir, L);
    unlink(p);
  }
  rmdir(dir);
  printf("%s %s slots %d: misses %ld of %ld picks, policy %.4f ms/token\n", argv[1], pol.c_str(), slots, misses, picks,
         c.st.policy_ms / npos);
  return 0;
}
