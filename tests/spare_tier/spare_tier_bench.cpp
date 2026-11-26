// The spare tier's cost to the compute, on a synthetic store (no model): records of real size (default 4 MB) used by a
// Core-like step (slot_for_use, a miss read and waited for, the step's use and unpin, tier_step at each step) and
// "computed" by a bandwidth-bound pass over every byte on 4 threads, while the readers bring in the next layer's picks
// (named before the compute, as the routers' names are: the readers' work overlaps the kernels). The compute is timed
// once the step's own reads have landed; the process's faults (TASK_EVENTS_INFO: soft + hard) and page-ins over that
// time, less those the readers counted as theirs, are the compute's. Tier off and on, one process each.
//
//   spare_tier_bench DIR [--rec-mb 4] [--layers 6] [--experts 24] [--k 4] [--slab 24] [--prob 48] [--tokens 60] [--tier 0|1]
//                    [--uses N] [--margin M]
// prints one JSON line: compute_ms_per_token, check_ms_per_token (slot_for_use), faults / pageins per token during the
// compute, the tier's counters.
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dmeng/pool.h"
#include "dmeng/store.h"

using namespace dm;
using clk = std::chrono::steady_clock;
#ifdef SPARE_BENCH_OLD  // (built against the tree before the readers counted their own faults)
#define READER_FAULTS(c) 0
#define READER_PAGEINS(c) 0
#else
#define READER_FAULTS(c) (c).rd.reader_faults.load()
#define READER_PAGEINS(c) (c).rd.reader_pageins.load()
#endif

static uint64_t task_faults() {
  task_events_info_data_t e;
  mach_msg_type_number_t c = TASK_EVENTS_INFO_COUNT;
  task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&e, &c);
  return (uint64_t)e.faults;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: spare_tier_bench DIR [options]\n");
    return 2;
  }
  std::string dir = argv[1];
  double rec_mb = 4;
  int NL = 6, NE = 24, K = 4, slab = 24, prob = 48, tokens = 60, tier = 1, uses = 3;
  double margin = 0.1;
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--rec-mb") rec_mb = atof(v.c_str());
    else if (k == "--layers") NL = atoi(v.c_str());
    else if (k == "--experts") NE = atoi(v.c_str());
    else if (k == "--k") K = atoi(v.c_str());
    else if (k == "--slab") slab = atoi(v.c_str());
    else if (k == "--prob") prob = atoi(v.c_str());
    else if (k == "--tokens") tokens = atoi(v.c_str());
    else if (k == "--tier") tier = atoi(v.c_str());
    else if (k == "--uses") uses = atoi(v.c_str());
    else if (k == "--margin") margin = atof(v.c_str());
  }
  const size_t R = ((size_t)(rec_mb * 1e6) + 16383) & ~size_t(16383);
  for (int L = 0; L < NL; L++) {  // the store (written once; reused by later runs of the same shape)
    char b[64];
    snprintf(b, sizeof b, "/bench_%02d.bin", L);
    const std::string p = dir + b;
    if (access(p.c_str(), F_OK) == 0) continue;
    std::vector<uint8_t> v(R);
    const int fd = open(p.c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    for (int e = 0; e < NE; e++) {
      for (size_t i = 0; i < R; i++) v[i] = (uint8_t)(L * 7 + e * 13 + i);
      if (write(fd, v.data(), R) != (ssize_t)R) return 2;
    }
    fsync(fd);
    close(fd);
  }
  Cache c;
  c.file_under = tier != 0;
  c.prob_bytes = (int64_t)prob * (int64_t)R;
  std::vector<int> step_layer(NL), layer_pool(NL, 0);
  for (int L = 0; L < NL; L++) step_layer[L] = L;
  std::string err;
  if (!c.init(NL, NE, K, step_layer, layer_pool, {R}, {1.0}, dir, "bench_%02d.bin", (double)slab * R, 4, 1, Mem::PLAIN, false, err)) {
    fprintf(stderr, "init: %s\n", err.c_str());
    return 1;
  }
  c.enable_hazard();
  c.prom_uses = uses;
  c.prom_margin = margin;
  for (auto& m : c.fmap) msync(m, c.fmap_bytes[0], MS_INVALIDATE);  // cold
  Pool pool(4);
  std::mt19937 rng(5);
  // a skewed routing: expert e with weight 1 / (1 + rank)^1.1 under a per-layer permutation
  std::vector<std::vector<int>> perm(NL, std::vector<int>(NE));
  for (int L = 0; L < NL; L++) {
    for (int e = 0; e < NE; e++) perm[L][e] = e;
    std::shuffle(perm[L].begin(), perm[L].end(), rng);
  }
  std::vector<double> w(NE);
  for (int r = 0; r < NE; r++) w[r] = 1.0 / std::pow(1.0 + r, 1.1);
  std::discrete_distribution<int> pick(w.begin(), w.end());
  // every token's picks, drawn up front (the same in every run of a shape)
  std::vector<int> picks((size_t)tokens * NL * K);
  for (int t = 0; t < tokens; t++)
    for (int L = 0; L < NL; L++) {
      int* pk = picks.data() + ((size_t)t * NL + L) * K;
      for (int n = 0; n < K;) {
        const int e = perm[L][pick(rng)];
        bool dup = false;
        for (int j = 0; j < n; j++) dup |= pk[j] == e;
        if (!dup) pk[n++] = e;
      }
    }
  std::vector<int> named(NL * K, -1);  // the names of each layer's picks (made one step ahead), -1 none
  double comp_ms = 0, check_ms = 0;
  uint64_t faults = 0, pageins = 0, sink = 0;
  const int warm = tokens / 4;
  for (int t = 0; t < tokens; t++) {
    if (t == warm) {
      comp_ms = check_ms = 0;
      faults = pageins = 0;
      c.st = CacheStats();
    }
    for (int L = 0; L < NL; L++) {
      c.tok_now = t;
      c.now_step = (long)t * NL + L;
      if (c.tier_due()) c.tier_step();
      const int* pk = picks.data() + ((size_t)t * NL + L) * K;
      int sl[16];
      c.pre_use(L, pk, K, t);
      const auto c0 = clk::now();
      for (int j = 0; j < K; j++) {
        int s = c.slot_for_use(L, pk[j]);
        if (s >= 0) {
          c.slot(L, s).pins++;
        } else {
          s = c.victim(L, 0);
          if (s < 0) s = c.steal(L);
          c.issue(L, s, pk[j], true, c.now_step);
          c.slot(L, s).pins = 1;
        }
        sl[j] = s;
      }
      check_ms += std::chrono::duration<double, std::milli>(clk::now() - c0).count();
      if (t * NL + L + 1 < tokens * NL) {  // the next step's picks named now (their reads overlap this step's compute)
        const int t2 = L + 1 < NL ? t : t + 1, L2 = (L + 1) % NL;
        const int* pk2 = picks.data() + ((size_t)t2 * NL + L2) * K;
        for (int j = 0; j < K; j++) named[L2 * K + j] = c.name(L2, pk2[j], (long)t2 * NL + L2);
      }
      for (int j = 0; j < K; j++) c.wait(L, sl[j]);
      // the compute: every byte of the K records, 4 threads (no read of this step outstanding)
      const uint64_t f0 = task_faults(), p0 = Cache::task_pageins(), rf0 = READER_FAULTS(c), rp0 = READER_PAGEINS(c);
      const auto t0 = clk::now();
      std::vector<uint64_t> part(4 * K, 0);
      parallel_for(pool, 4 * K, [&](int i) {
        const uint64_t* p = (const uint64_t*)(c.rec(L, sl[i / 4]) + (size_t)(i % 4) * (R / 4));
        uint64_t a = 0;
        for (size_t q = 0; q < R / 4 / 8; q++) a += p[q];
        part[i] = a;
      });
      comp_ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
      faults += (task_faults() - f0) - (READER_FAULTS(c) - rf0);
      pageins += (Cache::task_pageins() - p0) - (READER_PAGEINS(c) - rp0);
      for (uint64_t a : part) sink += a;
      for (int j = 0; j < K; j++) {
        c.use(L, sl[j], t, c.now_step);
        c.unpin(L, sl[j]);
        if (named[L * K + j] >= 0) c.unpin_pf(L, named[L * K + j], pk[j]);
        named[L * K + j] = -1;
      }
    }
  }
  const double nt = tokens - warm;
  const CacheStats& S = c.st;
  printf("{\"tier\":%d,\"rec_mb\":%.2f,\"compute_ms_per_token\":%.3f,\"check_ms_per_token\":%.4f,\"compute_faults_per_token\":%.1f,"
         "\"compute_pageins_per_token\":%.2f,\"prob_reuses_per_token\":%.2f,\"promotions_per_token\":%.2f,"
         "\"reclaimed_per_token\":%.2f,\"budget_drops_per_token\":%.2f,\"mb_issued_per_token\":%.1f,\"sink\":%llu}\n",
         tier, R / 1e6, comp_ms / nt, check_ms / nt, (double)(int64_t)faults / nt, (double)(int64_t)pageins / nt, S.n_prob_hit / nt, S.n_promoted / nt,
         S.n_prob_reclaimed / nt, S.n_prob_budget_drops / nt, (S.bytes_urgent + S.bytes_pf) / 1e6 / nt, (unsigned long long)(sink & 1));
  return 0;
}
