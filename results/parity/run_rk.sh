#!/bin/zsh
# the light gate with the re-run on by default: 64 greedy tokens at 2 GB, both models (vs v1's ids)
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || echo "FAIL $name"; }
run rk_gate_q36 $W/bin-rk/dmeng $FQ --cache-gb 2 --n 64 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids
run rk_gate_ink $W/bin-rk/dmeng $FI --cache-gb 2 --n 64 --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids
echo RK DONE
