#!/bin/zsh
# the expert-step replica (tests/kernels/bench_kmulti_step.cpp) on both models, through the guard queue
W=${0:A:h}/../..
DM=${W:A}/../../..
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
TAG=${1:-a}
BUSY=${2:-0}
$G $O/step_${TAG}_ink.log 600 -- $W/build/bench_kmulti_step ink ${DM:A}/inkling/model 8000 2.0 $BUSY || echo "FAIL ink"
$G $O/step_${TAG}_q36.log 600 -- $W/build/bench_kmulti_step q36 ${DM:A}/q36/model 8000 1.5 $BUSY || echo "FAIL q36"
echo STEP DONE
