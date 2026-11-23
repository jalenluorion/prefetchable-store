#!/bin/zsh
# run-ahead's cheap path on the 35B (the next layer's attention left out): top-8 match, reads named early, cost per
# event, and off / on token by token in one process (exact mode, 6.6 GB, 256 tokens)
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1500 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for t in pitt code; do
  run c_q36_$t env DMENG_RA_AB=1 $W/bin-b/dmeng $FQ --cache-gb 6.6 --n 256 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids --run-ahead 8 --run-ahead-cheap 1
done
echo B DONE
