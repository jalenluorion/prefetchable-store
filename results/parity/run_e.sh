#!/bin/zsh
# run-ahead off / on alternated token by token inside one process (same load for both arms; per-token wall ms and wait
# per arm), both models at their default caches, Pittsburgh and code, 256 tokens
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1500 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for t in pitt code; do
  run e_q36_$t env DMENG_RA_AB=1 $W/bin-rab/dmeng $FQ --cache-gb 6.6 --n 256 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids --run-ahead 8
  run e_ink_$t env DMENG_RA_AB=1 $W/bin-rab/dmeng $FI --cache-gb 7.8 --n 256 --prompt-file $W/bench/prompts/inkling2_loop/$t.ids --run-ahead 8
done
echo E DONE
