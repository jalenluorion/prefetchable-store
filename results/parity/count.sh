#!/bin/zsh
# Counting runs for the parity table (counting build, DM_COUNTS=1): per model, decode (128 tokens after the Pittsburgh
# prompt, default settings) and prefill (1,024 ids, batched, one token out). Usage: count.sh TAG COUNTING_DMENG
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
T=$1; B=$2
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
FI=(--model $I --cache-gb 7.8 --ctx 16384 --mem mlock)
run() { local name=$1; shift; $G $O/$name.log 1200 -- env DM_COUNTS=1 "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run ${T}_q36_decode $B $FQ --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --temp 0 --n 128
run ${T}_ink_decode $B $FI --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids --temp 0 --n 128
run ${T}_q36_prefill $B $FQ --prompt-file $W/results/parity/prefill1024.ids --temp 0 --n 1
run ${T}_ink_prefill $B $FI --prompt-file $W/results/parity/prefill1024.ids --temp 0 --n 1
echo COUNT DONE
