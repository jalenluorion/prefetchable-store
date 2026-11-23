// An engine run's routing and router rows replayed through dm::Cache in cache-only mode, for any model and policy:
// RECORD.bin from dmeng --record, TRACE.bin from dmeng --evict-trace (the model's shape, the released router's and the
// early router's top-32 rows per token and step). Per token: the early router's rows for the steps < lag (as
// Routers::token_start); per step: the released row (Core::route), pre_use, the picks in order (a hit pins its slot, a
// miss takes victim(L, 0) and is issued urgent), the early router's row for step + lag (Routers::after_router), then
// use() and unpin, a looped layer's picks held until its last pass of the token (Core::finish_step), step_end().
// Writes one line per eviction ("step victim_record", record = layer * NE + expert) and prints hits, misses and MB per
// counted token (positions >= START), and the policy's time.
//   evict_replay RECORD.bin TRACE.bin SLOTS[,SLOTS2..] freq|hazard|POLICY.json OUT.txt [uniform] [START] [LAG]
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <memory>
#include <cstdlib>
#include <string>
#include <vector>

#include "dmeng/store.h"

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: evict_replay RECORD.bin TRACE.bin SLOTS[,..] freq|hazard|POLICY.json OUT.txt [uniform] [START] [LAG]\n");
    return 2;
  }
  // the trace
  FILE* f = fopen(argv[2], "rb");
  if (!f) return fprintf(stderr, "cannot open %s\n", argv[2]), 1;
  char magic[8];
  int32_t h[7];
  if (fread(magic, 1, 8, f) != 8 || std::string(magic, 7) != "DMEVTR1" || fread(h, 4, 7, f) != 7) return fprintf(stderr, "not a trace\n"), 1;
  const int npos_t = h[0], steps = h[1], NE = h[2], W = h[3], NL = h[4], K = h[5], npools = h[6];
  std::vector<int> step_layer(steps), layer_pool(NL);
  if (fread(step_layer.data(), 4, steps, f) != (size_t)steps || fread(layer_pool.data(), 4, NL, f) != (size_t)NL) return 1;
  std::vector<int64_t> prec(2 * npools);
  if (fread(prec.data(), 8, 2 * npools, f) != (size_t)(2 * npools)) return 1;
  const size_t nr = (size_t)npos_t * steps * W;
  std::vector<int16_t> rli(nr), rti(nr);
  std::vector<float> rlp(nr), rtp(nr);
  if (fread(rli.data(), 2, nr, f) != nr || fread(rlp.data(), 4, nr, f) != nr || fread(rti.data(), 2, nr, f) != nr ||
      fread(rtp.data(), 4, nr, f) != nr)
    return fprintf(stderr, "short trace\n"), 1;
  fclose(f);
  // the routing
  f = fopen(argv[1], "rb");
  if (!f) return fprintf(stderr, "cannot open %s\n", argv[1]), 1;
  int hdr[3];
  if (fread(hdr, 4, 3, f) != 3 || hdr[1] != steps || hdr[2] != K) return fprintf(stderr, "record does not match the trace\n"), 1;
  const int npos = std::min(hdr[0], npos_t);
  std::vector<int16_t> rec((size_t)hdr[0] * steps * K);
  if (fread(rec.data(), 2, rec.size(), f) != rec.size()) return fprintf(stderr, "short record\n"), 1;
  fclose(f);

  std::vector<int> want;
  for (const char* q = argv[3]; *q;) {
    char* end;
    want.push_back((int)strtol(q, &end, 10));
    q = *end ? end + 1 : end;
  }
  if ((int)want.size() != npools) return fprintf(stderr, "%d pools: give %d slot counts\n", npools, npools), 2;
  const std::string pol = argv[4];
  bool caps = false;
  int start = 0, lag = 2;
  for (int i = 6, k = 0; i < argc; i++) {
    if (std::string(argv[i]) == "uniform") caps = true;
    else if (k++ == 0) start = atoi(argv[i]);
    else lag = atoi(argv[i]);
  }

  const char* td = getenv("TMPDIR");
  std::string tmpl = std::string(td && *td ? td : "/tmp") + "/evict_replay_XXXXXX";
  const char* dir = mkdtemp(&tmpl[0]);
  if (!dir) return 1;
  for (int L = 0; L < NL; L++) {
    char p[512];
    snprintf(p, sizeof p, "%s/layer_%02d.bin", dir, L);
    FILE* g = fopen(p, "wb");
    if (!g) return 1;
    fclose(g);
  }
  std::vector<size_t> rb(npools);
  std::vector<double> share(npools);
  double budget = 0;
  for (int p = 0; p < npools; p++) rb[p] = (size_t)prec[2 * p], budget += ((double)want[p] + 0.5) * (double)rb[p];
  for (int p = 0; p < npools; p++) share[p] = ((double)want[p] + 0.5) * (double)rb[p] / budget;
  dm::Cache c;
  std::string err;
  if (!c.init(NL, NE, K, step_layer, layer_pool, rb, share, dir, "layer_%02d.bin", budget, 1, 1, dm::Mem::PLAIN, true, err))
    return fprintf(stderr, "init: %s\n", err.c_str()), 1;
  for (int p = 0; p < npools; p++)
    if (c.slots(p) != want[p]) return fprintf(stderr, "pool %d: slots %d != %d\n", p, c.slots(p), want[p]), 1;
  if (pol == "hazard") c.enable_hazard();
  else if (pol != "freq") {
    if (!c.load_policy(pol, err)) return fprintf(stderr, "%s\n", err.c_str()), 1;
    if (c.rk.router_lag >= 0) lag = c.rk.router_lag;
  }
  if (caps) c.uniform_layer_caps();

  std::vector<int> next_same(steps, -1);  // the next step of the token with the same layer (loop hold)
  for (int a = 0; a < steps; a++)
    for (int b = a + 1; b < steps; b++)
      if (step_layer[b] == step_layer[a]) {
        next_same[a] = b;
        break;
      }
  std::vector<std::vector<int>> holds(NL);
  for (int a = 0; a < steps; a++)  // as Core::init: the loop hold's worst case is left to the misses
    if (next_same[a] >= 0) c.extra_reserve[layer_pool[step_layer[a]]] += K;
  // optional naming (lossless router prefetch as Routers does it, from the trace's early-router rows): NAME_M = names per
  // step (the row's top M; 0: cache only), issued at the token's start for steps < lag and after step a's misses for
  // step a + lag; a held record is pinned, else a slot is taken with victim(L, 2K + loop reserve) and read; pins go at
  // the target's end. NAME_BUDGET_MB: a name that needs a read is skipped once this token's reads (urgent + named) would
  // pass the budget (within a step the lower-ranked names go first).
  const long count_from = getenv("COUNT_FROM") ? atol(getenv("COUNT_FROM")) : start;  // counted positions (default: START on)
  const int name_m = getenv("NAME_M") ? atoi(getenv("NAME_M")) : 0;
  const double name_budget = getenv("NAME_BUDGET_MB") ? atof(getenv("NAME_BUDGET_MB")) * 1e6 : 0.0;
  std::vector<std::vector<std::pair<int, int>>> npins(steps);
  std::vector<std::vector<int>> nissued(steps);
  double tok_bytes = 0, named_c = 0, unused_c = 0, urgent_c = 0;
  long names_c = 0, names_read_c = 0, names_skipped_c = 0;
  // optional: a run's names (dmeng --dump-names, int16 [positions, steps, 16]) scored against this cache-only replay:
  // NAMES=F; MISSES_OUT=F writes per miss "token step expert rank" (rank in the names, -1 none) and per step and token
  // "W token step wasted8 wasted12 wasted16" (named, not held at the step's start, not picked)
  std::vector<int16_t> nm;
  const char* nf = getenv("NAMES");
  if (nf) {
    FILE* g = fopen(nf, "rb");
    if (!g) return fprintf(stderr, "cannot open %s\n", nf), 1;
    nm.resize((size_t)npos * steps * 16, -1);
    const size_t got = fread(nm.data(), 2, nm.size(), g);
    fclose(g);
    if (got < nm.size()) nm.resize(got);
  }
  const long names_start = getenv("NAMES_START") ? atol(getenv("NAMES_START")) : 0;
  FILE* mo = getenv("MISSES_OUT") ? fopen(getenv("MISSES_OUT"), "w") : nullptr;
  FILE* out = fopen(argv[5], "w");
  std::vector<std::vector<int>> owner(npools);
  for (int p = 0; p < npools; p++) owner[p].assign(want[p], -1);
  long misses = 0, picks = 0, hits_c = 0, picks_c = 0;
  double bytes_c = 0;
  int32_t idx[32];
  float pv[32];
  auto row = [&](const std::vector<int16_t>& vi, const std::vector<float>& vp, int t, int a) {
    const size_t o = ((size_t)t * steps + a) * W;
    if (vi[o] < 0) return 0;
    for (int j = 0; j < W; j++) idx[j] = vi[o + j], pv[j] = vp[o + j];
    return W;
  };
  auto name_target = [&](int t, int b) {
    if (!name_m) return;
    const int n = row(rti, rtp, t, b);
    if (!n) return;
    const int Lb = step_layer[b], pb = layer_pool[Lb];
    for (int j = 0; j < std::min(name_m, n); j++) {
      const int e = idx[j];
      if (t >= count_from) names_c++;
      const int s0 = c.slot_of(Lb, e);
      if (s0 >= 0) {
        c.slot(Lb, s0).pins++;
        c.slot(Lb, s0).pf_pins++;
        npins[b].push_back({s0, e});
        continue;
      }
      if (name_budget > 0 && tok_bytes + (double)rb[pb] > name_budget) {
        if (t >= count_from) names_skipped_c++;
        continue;
      }
      c.evict_pf_ctx = true;
      const int v = c.victim(Lb, 2 * K + c.extra_reserve[pb]);
      c.evict_pf_ctx = false;
      if (v < 0) continue;
      if (owner[pb][v] >= 0) fprintf(out, "%ld %d\n", (long)t * steps + b, owner[pb][v]);
      c.issue(Lb, v, e, false, (long)t * steps + b);
      c.slot(Lb, v).pins = 1;
      c.slot(Lb, v).pf_pins = 1;
      c.slot(Lb, v).pf_unused = true;
      owner[pb][v] = Lb * NE + e;
      npins[b].push_back({v, e});
      nissued[b].push_back(e);
      tok_bytes += (double)rb[pb];
      if (t >= count_from) named_c += (double)rb[pb], bytes_c += (double)rb[pb], names_read_c++;
    }
  };
  // PROMPT_CHUNK=C: the positions before START go through the cache as dmeng's batched prefill does it (Prefill::
  // forward_chunk: per C positions, layer by layer, the layer's union of picks pinned (hits) or read (misses), then
  // use() for every pick, expert by expert, and the unpins), cache only; the generated positions as single steps.
  // NO_CHUNK_CLOCK=1 leaves out the pass clock (the behaviour before it).
  const int pchunk = getenv("PROMPT_CHUNK") ? atoi(getenv("PROMPT_CHUNK")) : 0;
  int t_first = 0;
  if (pchunk > 0) {
    const bool clock_on = !(getenv("NO_CHUNK_CLOCK") && atoi(getenv("NO_CHUNK_CLOCK")));
    for (int c0 = 0; c0 < start; c0 += pchunk) {
      const int n = std::min(pchunk, start - c0);
#ifndef OLD_STORE
      std::unique_ptr<dm::Cache::ChunkClock> cc;
      if (clock_on) cc.reset(new dm::Cache::ChunkClock(c, c0 + n - 1));
#else
      (void)clock_on;
#endif
      for (int L = 0; L < NL; L++) {  // steps = layers (the 35B)
        const int p = layer_pool[L];
        c.now_step = (long)(c0 + n - 1) * steps + L;
        std::vector<std::vector<int>> epos(NE);
        for (int i = 0; i < n; i++)
          for (int j = 0; j < K; j++) epos[rec[((size_t)(c0 + i) * steps + L) * K + j]].push_back(i);
        std::vector<int> slot(NE, -1);
        for (int e = 0; e < NE; e++) {
          if (epos[e].empty()) continue;
          int x = c.slot_of(L, e);
          if (x >= 0) {
            c.slot(L, x).pins++;
          } else {
            x = c.victim(L, 0);
            if (x < 0) x = c.steal(L);
            if (x < 0) return fprintf(stderr, "no slot in the prompt chunk\n"), 1;
            c.issue(L, x, e, true, (long)(c0 + n - 1) * steps + L);
            c.slot(L, x).pins = 1;
            owner[p][x] = L * NE + e;
          }
          slot[e] = x;
        }
        for (int e = 0; e < NE; e++) {
          if (slot[e] < 0) continue;
          for (int i : epos[e]) c.use(L, slot[e], c0 + i, (long)(c0 + i) * steps + L);
          c.unpin(L, slot[e]);
        }
      }
    }
    t_first = start;
  }
  for (int t = t_first; t < npos; t++) {
    c.tok_now = t;
    tok_bytes = 0;
    for (int a = 0; a < std::min(lag, steps); a++)
      if (int n = row(rti, rtp, t, a)) c.router_top(t, a, idx, pv, n);
    for (int a = 0; a < std::min(lag, steps); a++) name_target(t, a);
    for (int a = 0; a < steps; a++) {
      const int L = step_layer[a], p = layer_pool[L];
      const long step = (long)t * steps + a;
      c.now_step = step;
      c.tok_now = t;
      if (int n = row(rli, rlp, t, a)) c.released_top(a, idx, pv, n);
      int sl[16];
      int32_t pk[16];
      for (int j = 0; j < K; j++) pk[j] = rec[((size_t)t * steps + a) * K + j];
      const int16_t* nn = nullptr;
      const long tn = t - names_start;  // dmeng writes names for the generated positions only (NAMES_START = prompt length)
      if (!nm.empty() && tn >= 0 && ((size_t)tn * steps + a + 1) * 16 <= nm.size()) nn = nm.data() + ((size_t)tn * steps + a) * 16;
      if (mo && nn) {  // wasted names: not held now, not picked
        int wst[3] = {0, 0, 0};
        for (int r = 0; r < 16; r++) {
          const int e = nn[r];
          if (e < 0 || c.slot_of(L, e) >= 0) continue;
          bool pkd = false;
          for (int j = 0; j < K; j++) pkd |= pk[j] == e;
          if (pkd) continue;
          for (int q = 0; q < 3; q++) wst[q] += r < 8 + 4 * q;
        }
        fprintf(mo, "W %d %d %d %d %d\n", t, a, wst[0], wst[1], wst[2]);
      }
      c.pre_use(L, pk, K, t);
      for (int j = 0; j < K; j++) {
        const int e = pk[j];
        int s = c.slot_of(L, e);
        picks++;
        if (t >= count_from) picks_c++;
        if (s >= 0) {
          c.slot(L, s).pins++;
          if (t >= count_from) hits_c++;
        } else {
          s = c.victim(L, 0);
          if (s < 0) s = c.steal(L);
          if (s < 0) return fprintf(stderr, "no slot at step %ld\n", step), 1;
          if (owner[p][s] >= 0) fprintf(out, "%ld %d\n", step, owner[p][s]);
          c.issue(L, s, e, true, step);
          c.slot(L, s).pins = 1;
          owner[p][s] = L * NE + e;
          misses++;
          if (t >= count_from) bytes_c += (double)rb[p], urgent_c += (double)rb[p];
          tok_bytes += (double)rb[p];
          if (mo) {
            int rk = -1;
            for (int r = 0; nn && r < 16; r++)
              if (nn[r] == e) rk = r;
            fprintf(mo, "M %d %d %d %d\n", t, a, e, rk);
          }
        }
        sl[j] = s;
      }
      if (a + lag < steps) {
        if (int n = row(rti, rtp, t, a + lag)) c.router_top(t, a + lag, idx, pv, n);
        name_target(t, a + lag);
      }
      const bool hold = next_same[a] >= 0;
      for (int j = 0; j < K; j++) {
        c.use(L, sl[j], t, step);
        if (hold) holds[L].push_back(sl[j]);
        else c.unpin(L, sl[j]);
      }
      if (next_same[a] < 0 && !holds[L].empty()) {
        for (int s : holds[L]) c.unpin(L, s);
        holds[L].clear();
      }
      c.step_end(a);
      for (auto& se : npins[a]) c.unpin_pf(L, se.first, se.second);  // Naming::release
      npins[a].clear();
      for (int e : nissued[a]) {
        bool used = false;
        for (int j = 0; j < K; j++) used |= pk[j] == e;
        if (!used && t >= count_from) unused_c += (double)rb[p];
      }
      nissued[a].clear();
    }
  }
  fprintf(out, "misses %ld picks %ld\n", misses, picks);
  if (mo) fclose(mo);
  fclose(out);
  for (int L = 0; L < NL; L++) {
    char p[512];
    snprintf(p, sizeof p, "%s/layer_%02d.bin", dir, L);
    unlink(p);
  }
  rmdir(dir);
  const int counted = npos - (int)count_from;
  printf("{\"policy\":\"%s\",\"caps\":%s,\"positions\":%d,\"counted\":%d,\"hit_rate\":%.6f,\"mb_per_token\":%.4f,\"misses\":%ld,"
         "\"policy_ms_per_token\":%.4f,\"ranker_shortlist\":%.1f,\"ranker_evals_per_token\":%.1f,\"urgent_mb_per_token\":%.4f,"
         "\"named_mb_per_token\":%.4f,\"named_unused_mb_per_token\":%.4f,\"names_per_token\":%.2f,\"names_read_per_token\":%.2f,"
         "\"names_skipped_per_token\":%.2f}\n",
         pol.c_str(), caps ? "true" : "false", npos, counted, (double)hits_c / std::max(1L, picks_c), bytes_c / 1e6 / std::max(1, counted),
         misses, c.st.policy_ms / npos, (double)c.rk_cands / std::max<uint64_t>(1, c.rk_choices), (double)c.rk_evals / npos,
         urgent_c / 1e6 / std::max(1, counted), named_c / 1e6 / std::max(1, counted), unused_c / 1e6 / std::max(1, counted),
         (double)names_c / std::max(1, counted), (double)names_read_c / std::max(1, counted), (double)names_skipped_c / std::max(1, counted));
  return 0;
}
