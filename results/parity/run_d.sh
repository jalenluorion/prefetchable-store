#!/bin/zsh
# run-ahead depth (35B: depth 1 and depth 1-3 measured), and where the remaining wait sits (named / not named by the
# previous step's run-ahead, steps that never waited), both models, Pittsburgh and code, default caches, counting build
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for t in pitt code; do
  run d1_q36_$t env DM_COUNTS=1 $W/bin-d/dmeng-counters $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids --run-ahead 8
  run d3_q36_$t env DM_COUNTS=1 $W/bin-d/dmeng-counters $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids --run-ahead 8 --run-ahead-depth 3
  run d1_ink_$t env DM_COUNTS=1 $W/bin-d/dmeng-counters $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/$t.ids --run-ahead 8
done
# the two down-claim rules alternated call by call inside one process (same load for both): wall ns per weight byte of the
# expert fork-join and spin per rule, both models, Pittsburgh and code
for t in pitt code; do
  run ab_q36_$t env DMENG_DN_AB=1 $W/bin-ab/dmeng $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids
  run ab_ink_$t env DMENG_DN_AB=1 $W/bin-ab/dmeng $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/$t.ids
done
echo D DONE
