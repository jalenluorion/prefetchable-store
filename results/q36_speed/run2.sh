#!/bin/zsh
# batch 2: the screened head on the real model (checked against the full head every token; sampled ids vs the release
# binary for three seeds), and counting runs of the new build with the head off and on
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
B=$W/bin-h1
$G $O/h1_check_greedy.log 600 -- $B/dmeng $F --temp 0 --n 200 --fast-head 2 --json $O/h1_check_greedy.json --out-ids $O/h1_check_greedy.ids || exit 1
for s in 1 2 3; do
  $G $O/samp_base_s$s.log 600 -- $W/bin-base/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100 --json $O/samp_base_s$s.json --out-ids $O/samp_base_s$s.ids || exit 1
  $G $O/samp_h1_s$s.log 600 -- $B/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100 --fast-head 2 --json $O/samp_h1_s$s.json --out-ids $O/samp_h1_s$s.ids || exit 1
done
$G $O/cnt_h0.log 600 -- env DM_COUNTS=1 $B/dmeng-counters $F --temp 0 --n 128 --json $O/cnt_h0.json --out-ids $O/cnt_h0.ids || exit 1
$G $O/cnt_h1.log 600 -- env DM_COUNTS=1 $B/dmeng-counters $F --temp 0 --n 128 --fast-head 1 --json $O/cnt_h1.json --out-ids $O/cnt_h1.ids || exit 1
echo BATCH2 DONE
