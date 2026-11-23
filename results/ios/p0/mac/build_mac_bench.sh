#!/bin/bash
# Build the P0 bench for the Mac from the phone app's sources (calibration). 3 compile jobs at a time (shared Mac).
set -euo pipefail
S=${S:-$(cd "$(dirname "$0")/../../../../apps/ios/p0bench/DMBench" && pwd)}  # the copy of the phone app's DMBench sources
O=$(cd "$(dirname "$0")" && pwd)
B=$(mktemp -d)
FL=(-O3 -std=c++17 -ffp-contract=fast -mcpu=apple-m4 -I"$S" -fobjc-arc -Wall -Wno-unused-function)
printf '%s\n' kernels.cpp kernels_batch.cpp pool.cpp bench_cpu.cpp util.mm metal.mm filecache.mm | \
  xargs -P 3 -I{} sh -c 's="$0"; b="$1"; shift; clang++ "$@" -c "$s/{}" -o "$b/$(basename {}).o"' "$S" "$B" "${FL[@]}"
clang++ "${FL[@]}" -c "$O/mac_bench_main.cpp" -o "$B/main.o"
clang++ "$B"/*.o -framework Metal -framework Foundation -o "$O/mac_bench"
rm -rf "$B"
echo "built $O/mac_bench"
