#!/bin/zsh
# batch 7: wall-clock before / after with the final build, 6.6 GB, 400 tokens, exact and fast mode, interleaved
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
N=$W/bin-n3; B=$W/bin-base
run() { local name=$1; shift; $G $O/$name.log 900 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run x_base_exact $B/dmeng $F --temp 0 --n 400
run x_new_exact $N/dmeng $F --temp 0 --n 400
run x_new_exact_fh $N/dmeng $F --temp 0 --n 400 --fast-head 1
run x_base_fast $B/dmeng $F --temp 0 --n 400 --mode fast
run x_new_fast $N/dmeng $F --temp 0 --n 400 --mode fast
run x_new_fast_fh $N/dmeng $F --temp 0 --n 400 --mode fast --fast-head 1
run x_new_samp_fh $N/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 400 --fast-head 1
echo BATCH7 DONE
