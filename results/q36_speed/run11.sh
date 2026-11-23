#!/bin/zsh
# batch 11: the high-activation-plane pass of the screened head (n6) against the two-plane pass (n5): counting runs side by
# side, the 35B greedy check and three sampled seeds, Inkling greedy and one seed
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
FI=(--model $I --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids --cache-gb 7.8 --ctx 16384 --mem mlock)
FQ=(--model $M --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
N=$W/bin-n6
run() { local name=$1; shift; $G $O/$name.log 900 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run cnt_n5_q env DM_COUNTS=1 $W/bin-n5/dmeng-counters $FQ --temp 0 --n 128
run cnt_n6_q env DM_COUNTS=1 $N/dmeng-counters $FQ --temp 0 --n 128
run cnt_n5_qs env DM_COUNTS=1 $W/bin-n5/dmeng-counters $FQ --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 128
run cnt_n6_qs env DM_COUNTS=1 $N/dmeng-counters $FQ --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 128
run n6_check_greedy $N/dmeng $FQ --temp 0 --n 200 --fast-head 2
for s in 1 2 3; do run n6_samp_s$s $N/dmeng $FQ --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100 --fast-head 2; done
run n6_i_greedy $N/dmeng $FI --temp 0 --n 200
run n6_i_samp_s1 $N/dmeng $FI --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 100
echo BATCH11 DONE
