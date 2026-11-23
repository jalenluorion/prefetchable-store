#!/bin/zsh
# the expert step's down claims: before (s0: in claim order, waiting on the claimed expert) and after (s1: any ready expert),
# counting runs of both models (expert stage cycles, expert_spin_thread_ms_per_token), and the light gate on s1
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for b in s1 s0; do
  run ${b}b_q36_decode env DM_COUNTS=1 $W/bin-$b/dmeng-counters $FQ --cache-gb 6.6 --n 128
done
echo S2 DONE
