#!/bin/zsh
# p1: the 4-bit kernel in row pairs (35B experts): the light gate (64 greedy ids at 2 GB against v1's), the 35B's counting
# run, and both models' counting runs on the 4 performance cores alone (--threads 4) against 8 threads (p0 / p1)
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
B=$W/bin-p1
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run p1_gate_q36 $B/dmeng $FQ --cache-gb 2 --n 64
run p1_q36_decode env DM_COUNTS=1 $B/dmeng-counters $FQ --cache-gb 6.6 --n 128
run p1t4_q36_decode env DM_COUNTS=1 $B/dmeng-counters $FQ --cache-gb 6.6 --n 128 --threads 4
run p1t4_ink_decode env DM_COUNTS=1 $B/dmeng-counters $FI --cache-gb 7.8 --n 128 --threads 4
echo P1 DONE
