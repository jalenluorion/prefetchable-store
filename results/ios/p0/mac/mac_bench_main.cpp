// The P0 bench on the Mac (calibration): the same sources the phone app builds (phone-store native/Sources/DMBench).
//   mac_bench "SPEC" > out.json      (SPEC as dmb_kernel_bench)
#include <cstdio>
#include "include/dmbench.h"
static void lg(const char* s) { fprintf(stderr, "%s\n", s); }
int main(int argc, char** argv) {
  char* r = dmb_kernel_bench(argc > 1 ? argv[1] : "", lg);
  fputs(r, stdout);
  dmb_free(r);
  return 0;
}
