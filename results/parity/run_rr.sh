#!/bin/zsh
# STEP_RERUN (re-run of a stalled claimed gate-up chunk): the replica under 0, 1, 2 spinning co-runner threads; the engine,
# rerun off / on call by call (DMENG_RERUN_AB=1), quiet and beside a co-runner process (spinner, 2 threads), both models
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
B=$W/bin-rr
for busy in 0 1 2; do
  $G $O/rr_step_ink_b$busy.log 600 -- $B/bench_kmulti_step ink $I 16000 2.0 $busy || echo "FAIL step ink $busy"
  $G $O/rr_step_q36_b$busy.log 600 -- $B/bench_kmulti_step q36 $M 16000 1.5 $busy || echo "FAIL step q36 $busy"
done
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || echo "FAIL $name"; }
# the co-runner starts inside the guarded command (only while this run holds the slot) and stops with it
for load in 0 2; do
  run rr_q36_l$load env DMENG_RERUN_AB=1 DM_SPIN=$load $W/results/parity/with_spinner.sh $B/dmeng $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids
  run rr_ink_l$load env DMENG_RERUN_AB=1 DM_SPIN=$load $W/results/parity/with_spinner.sh $B/dmeng $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/pitt.ids
done
echo RR DONE
