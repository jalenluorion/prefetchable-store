#!/bin/zsh
# one counting session: (1) the expert step's down claims, s0 (claim order) / s1 (any ready expert) / s2 (own expert
# first, padded counters), both models, interleaved; (3) run-ahead off / on with s2, both models at their default caches,
# Pittsburgh and code; the light gate on s2 for both models
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for b in s0 s1 s2; do
  run t_${b}_q36 env DM_COUNTS=1 $W/bin-$b/dmeng-counters $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids
  run t_${b}_ink env DM_COUNTS=1 $W/bin-$b/dmeng-counters $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids
done
for t in pitt code; do
  for ra in 0 8; do
    run t_ra${ra}_q36_$t env DM_COUNTS=1 $W/bin-s2/dmeng-counters $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/$t.ids --run-ahead $ra
    run t_ra${ra}_ink_$t env DM_COUNTS=1 $W/bin-s2/dmeng-counters $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/$t.ids --run-ahead $ra
  done
done
run t_gate_q36 $W/bin-s2/dmeng $FQ --cache-gb 2 --n 64 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids
run t_gate_ink $W/bin-s2/dmeng $FI --cache-gb 2 --n 64 --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids
echo T DONE
