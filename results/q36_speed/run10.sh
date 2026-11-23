#!/bin/zsh
# batch 10: the screened head on Inkling (default on in the new build) against v1, both prefill checks on the split head,
# and the 35B ids on this build
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
FI=(--model $I --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids --cache-gb 7.8 --ctx 16384 --mem mlock)
FQ=(--model $M --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
N=$W/bin-n5; B=$W/bin-base
run() { local name=$1; shift; $G $O/$name.log 900 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run i_base_greedy $B/dmeng $FI --temp 0 --n 200
run i_new_greedy $N/dmeng $FI --temp 0 --n 200
for s in 1 2 3; do
  run i_base_samp_s$s $B/dmeng $FI --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100
  run i_new_samp_s$s $N/dmeng $FI --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100
done
run n5_q_greedy $N/dmeng $FQ --temp 0 --n 200
run n5_q_samp_s1 $N/dmeng $FQ --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 100
$G $O/prefill_check_ink_n5.log 1800 -- env DMSERVE_BIN=$N/dmserve zsh $W/tests/prefill_check.sh $I --fast-head 1 || { echo "FAIL prefill_check ink"; exit 1; }
$G $O/prefill_check_q36_n5.log 1800 -- env DMSERVE_BIN=$N/dmserve zsh $W/tests/prefill_check.sh $M --fast-head 1 || { echo "FAIL prefill_check q36"; exit 1; }
echo BATCH10 DONE
