#!/bin/zsh
# batch 1: the counting profile of decode before any change, and the reference ids
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --temp 0 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
$G $O/cnt_before.log 600 -- env DM_COUNTS=1 $W/bin-base/dmeng-counters $F --n 128 --json $O/cnt_before.json --out-ids $O/cnt_before.ids || exit 1
$G $O/ref_exact.log 600 -- $W/bin-base/dmeng $F --n 200 --json $O/ref_exact.json --out-ids $O/ref_exact.ids || exit 1
echo BATCH1 DONE
