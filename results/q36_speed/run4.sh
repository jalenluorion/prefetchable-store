#!/bin/zsh
# batch 4: the two-level screened head (2-bit plane, then 4-bit, then exact) on the real model, gates and counting runs
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
N=$W/bin-n2; B=$W/bin-base
run() { local name=$1; shift; $G $O/$name.log 600 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run n2_check_greedy $N/dmeng $F --temp 0 --n 200 --fast-head 2
for s in 1 2 3; do
  [[ -f $O/samp_base_s$s.ids ]] || run samp_base_s$s $B/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100
  run n2_samp_s$s $N/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100 --fast-head 2
done
run cnt_h0_before env DM_COUNTS=1 $W/bin-h1/dmeng-counters $F --temp 0 --n 128
run cnt_n2_h0 env DM_COUNTS=1 $N/dmeng-counters $F --temp 0 --n 128
run cnt_n2_h1 env DM_COUNTS=1 $N/dmeng-counters $F --temp 0 --n 128 --fast-head 1
run cnt_n2_h1s env DM_COUNTS=1 $N/dmeng-counters $F --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 128 --fast-head 1
echo BATCH4 DONE
