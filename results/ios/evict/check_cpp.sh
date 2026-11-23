#!/bin/bash
# C++ Cache decisions (tests/evict_check) vs the simulator, eviction by eviction, on recorded traces.
#   results/ios/evict/check_cpp.sh  (from the repo root or anywhere; builds cpp/evict_check first)
set -e
cd "$(dirname "$0")"
ROOT=$(cd ../../.. && pwd)
ENG=<work>/engine
PY=$ENG/.venv/bin/python
clang++ -std=c++17 -O2 -mcpu=apple-m4 -ffp-contract=fast -Wall -Wno-unused-function -I$ROOT/include \
  $ROOT/tests/evict_check/evict_check.cpp $ROOT/src/core/{store,weights,kernels,pool}.cpp -o cpp/evict_check
mkdir -p cpp/logs
ok=1
for cell in "pitt results/stage2/runs/record_400.bin" "coding results/stage3/runs/record_code_400.bin" "agent results/stage3c/runs/record_tf_agent.bin"; do
  set -- $cell
  for slots in ${SLOTS:-452 720 1130 1864}; do
    for caps in pooled uniform; do
      tag=${1}_${slots}_hazard_${caps}; extra=$([ $caps = uniform ] && echo uniform || true)
      cpp/evict_check $ENG/$2 $slots hazard cpp/logs/cpp_$tag.txt $extra > /dev/null
      $PY evict_sim.py log $1 $slots hazard:128 cpp/logs/py_$tag.txt $extra > /dev/null
      if cmp -s cpp/logs/cpp_$tag.txt cpp/logs/py_$tag.txt; then
        echo "$1 $slots hazard $caps: identical ($(wc -l < cpp/logs/cpp_$tag.txt | tr -d ' ') evictions; $(tail -1 cpp/logs/cpp_$tag.txt))"
      else
        ok=0; echo "$1 $slots hazard $caps: DIFFER (first: $(diff cpp/logs/cpp_$tag.txt cpp/logs/py_$tag.txt | head -3 | tr '\n' ' '))"
      fi
    done
  done
done
[ $ok = 1 ] && echo "PASS" || { echo "FAIL"; exit 1; }
