#!/bin/zsh
# a second engine pair beside the co-runner process (both models), and the Inkling replica with 2 co-runner threads again
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model; I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs
B=$W/bin-rr
FQ=(--model $M --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8 --temp 0)
FI=(--model $I --ctx 16384 --mem mlock --temp 0)
run() { local name=$1; shift; $G $O/$name.log 1200 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || echo "FAIL $name"; }
run rr_q36_l2b env DMENG_RERUN_AB=1 DM_SPIN=2 $W/results/parity/with_spinner.sh $B/dmeng $FQ --cache-gb 6.6 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/code.ids
run rr_ink_l2b env DMENG_RERUN_AB=1 DM_SPIN=2 $W/results/parity/with_spinner.sh $B/dmeng $FI --cache-gb 7.8 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/code.ids
$G $O/rr_step_ink_b2.log 1200 -- $B/bench_kmulti_step ink $I 8000 2.0 2 || echo "FAIL step ink 2"
echo RR2 DONE
