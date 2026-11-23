#!/bin/zsh
# run-ahead counters, abortable version (bin-r2): as run_ra.sh
# the phone's configuration on the Mac (4-bit resident, 1.40 GB, approx:drop read-w 0.08, linear k2 names; prose and code,
# 400 teacher-forced positions), each with run-ahead off and on
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
B=$W/bin-r2/dmeng
LIN=(--prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr)
R4=${DM:A}/q36/q4c/derived/resident_q4.dmres1.a603e898ddee.bin
run() { local name=$1; shift; $G $O/$name.log 1800 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for ra in 0 8; do
  run rb${ra}_exact_pitt $B --model $M --ctx 16384 --mem mlock $LIN --m 8 --cache-gb 6.6 --temp 0 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/pitt.ids --run-ahead $ra
  run rb${ra}_exact_code $B --model $M --ctx 16384 --mem mlock $LIN --m 8 --cache-gb 6.6 --temp 0 --n 128 --prompt-file $W/bench/prompts/qwen3_5_moe/code.ids --run-ahead $ra
  for t in prose code; do
    run rb${ra}_phone_$t $B --model $M --ctx 16384 --mem mlock --temp 0 --resident $R4 --cache-gb 1.40 --mode approx:drop --read-w 0.08 $LIN --rt-tau 0.5 --rt-cand 16 --teacher $W/results/parity/${t}400.ids --run-ahead $ra
  done
done
echo RB DONE
