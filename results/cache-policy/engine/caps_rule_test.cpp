// the auto caps rule and its re-application at a run-time budget change (dm::Cache only, reads skipped)
#include <cstdio>
#include <string>
#include <unistd.h>
#include "dmeng/store.h"
int main() {
  const int NL = 40, NE = 256, K = 8;
  const size_t R = 1769472;
  char tmpl[] = "/tmp/capsrule_XXXXXX";
  const char* dir = mkdtemp(tmpl);
  for (int L = 0; L < NL; L++) { char p[256]; snprintf(p, sizeof p, "%s/layer_%02d.bin", dir, L); fclose(fopen(p, "wb")); }
  std::vector<int> sl(NL), lp(NL, 0);
  for (int L = 0; L < NL; L++) sl[L] = L;
  for (double gb : {0.8, 2.0, 2.26, 2.27, 3.3}) {
    dm::Cache c; std::string err;
    if (!c.init(NL, NE, K, sl, lp, {R}, {1.0}, dir, "layer_%02d.bin", gb * 1e9, 1, 1, dm::Mem::PLAIN, true, err)) return printf("%s\n", err.c_str()), 1;
    c.enable_hazard(); c.caps_auto = true; c.caps_auto_apply();
    int sum = 0; for (int v : c.lcap) sum += v;
    printf("gb %.2f slots %d caps %s (sum %d)", gb, c.slots(0), c.lcap.empty() ? "off" : "on", sum);
    c.set_budget((int64_t)(1.0e9)); c.apply_budget();
    sum = 0; for (int v : c.lcap) sum += v;
    printf(" -> budget 1.0 GB: cap %d caps %s (sum %d)", c.P[0].cap, c.lcap.empty() ? "off" : "on", sum);
    c.set_budget((int64_t)(gb * 1e9)); c.apply_budget();
    printf(" -> back: cap %d caps %s\n", c.P[0].cap, c.lcap.empty() ? "off" : "on");
  }
  return 0;
}
