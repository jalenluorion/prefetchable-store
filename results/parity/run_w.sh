#!/bin/zsh
# waves: the lossless expert step before (w0: one wait for every remaining pick, then one run) and after (w1: waves), both
# models at their default caches (wait, wait while another pick had landed, per token), and the light gate (64 greedy ids
# at 2 GB against v1's) for both models on w1
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for b in w0 w1; do
  run ${b}_q36 $W/bin-$b/dmeng $FQ --cache-gb 6.6 --n 128
  run ${b}_ink $W/bin-$b/dmeng $FI --cache-gb 7.8 --n 128
done
run w1_gate_q36 $W/bin-w1/dmeng $FQ --cache-gb 2 --n 64
run w1_gate_ink $W/bin-w1/dmeng $FI --cache-gb 2 --n 64
echo W DONE
