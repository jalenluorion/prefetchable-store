#!/bin/zsh
# batch 5: wall-clock before / after at 6.6 GB (exact and fast mode, 400 tokens, interleaved), 10 threads, the sampler
# with and without the screened head, and the 35B prefill exactness check on the new build
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
N=$W/bin-n2; B=$W/bin-base
run() { local name=$1; shift; $G $O/$name.log 900 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run w_base_exact $B/dmeng $F --temp 0 --n 400
run w_new_exact $N/dmeng $F --temp 0 --n 400
run w_new_exact_fh $N/dmeng $F --temp 0 --n 400 --fast-head 1
run w_base_fast $B/dmeng $F --temp 0 --n 400 --mode fast
run w_new_fast $N/dmeng $F --temp 0 --n 400 --mode fast
run w_new_fast_fh $N/dmeng $F --temp 0 --n 400 --mode fast --fast-head 1
run w_new_exact_fh_t10 $N/dmeng $F --temp 0 --n 400 --fast-head 1 --threads 10
run w_base_samp $B/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 400
run w_new_samp_fh $N/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed 1 --n 400 --fast-head 1
$G $O/prefill_check.log 1800 -- env DMSERVE_BIN=$N/dmserve zsh $W/tests/prefill_check.sh $M || { echo "FAIL prefill_check"; exit 1; }
echo BATCH5 DONE
