#!/bin/zsh
# batch 9: the screened head on by default (dmeng and dmserve): ids and sampled seeds against v1 through both, the
# OpenCode replay through the HTTP front end with the v1 and the new worker, the prefill check
W=${0:A:h}/../..
DM=${W:A}/../../..
M=${DM:A}/q36/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/q36_speed/logs; mkdir -p $O
P=$W/bench/prompts/qwen3_5_moe/pitt.ids
F=(--model $M --prompt-file $P --cache-gb 6.6 --ctx 16384 --mem mlock --prefetch router --routers $M/routers/lin/q36_router_lin_k2.dmr --m 8)
SW=(--model $M --cache-gb 6.6 --ctx 16384 --mem mlock)
N=$W/bin-n4; B=$W/bin-base
run() { local name=$1; shift; $G $O/$name.log 900 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
run d4_greedy $N/dmeng $F --temp 0 --n 200
for s in 1 2 3; do run d4_samp_s$s $N/dmeng $F --temp 1 --top-k 20 --top-p 0.95 --seed $s --n 100; done
for b in base n4; do
  BIN=$B; [[ $b = n4 ]] && BIN=$N
  $G $O/srv_$b.log 1200 -- python3 $W/results/q36_speed/drive.py $O/srv_$b.json $W/results/q36_speed/srv_reqs.json -- $BIN/dmserve $SW || { echo "FAIL srv_$b"; exit 1; }
done
for b in base n4; do
  BIN=$B; [[ $b = n4 ]] && BIN=$N
  $G $O/replay_$b.log 2400 -- zsh $W/bench/drafts/replay.sh $W $BIN/dmserve $O/replay_$b 128 6 --cache-gb 6.6 --mem mlock || { echo "FAIL replay_$b"; exit 1; }
done
$G $O/prefill_check_n4.log 1800 -- env DMSERVE_BIN=$N/dmserve zsh $W/tests/prefill_check.sh $M || { echo "FAIL prefill_check"; exit 1; }
echo BATCH9 DONE
