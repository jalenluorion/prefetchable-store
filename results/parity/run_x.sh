#!/bin/zsh
# the expert step in place: its fork-joins by number of jobs and by whether a read wait came just before (exact, default
# caches, 128 tokens), both models, Pittsburgh and code
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for t in pitt code; do
  run x_q36_$t $W/bin-x/dmeng $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids
  run x_ink_$t $W/bin-x/dmeng $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/$t.ids
done
echo X DONE
