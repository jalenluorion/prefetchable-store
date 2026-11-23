#!/bin/zsh
# batch 6: the final build (rebased on ink-speed): ids gates, a counting run with the screened head, the prefill check
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
N=$W/bin-n3
run() { local name=$1; shift; $G $O/$name.log 900 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run n3_exact $N/dmeng $F --temp 0 --n 200
run n3_fast $N/dmeng $F --temp 0 --n 200 --mode fast
run n3_check_greedy $N/dmeng $F --temp 0 --n 200 --fast-head 2
run n3_samp_s2 $N/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed 2 --n 100 --fast-head 2
run cnt_n3_h1 env DM_COUNTS=1 $N/dmeng-counters $F --temp 0 --n 128 --fast-head 1
$G $O/prefill_check_n3.log 1800 -- env DMSERVE_BIN=$N/dmserve zsh $W/tests/prefill_check.sh $M || { echo "FAIL prefill_check"; exit 1; }
echo BATCH6 DONE
