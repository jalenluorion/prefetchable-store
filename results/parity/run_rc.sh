#!/bin/zsh
# run-ahead on Inkling (bin-r3): exact mode at 7.8 GB, Pittsburgh and code, 128 tokens, run-ahead off and on
W=${0:A:h}/../..
DM=${W:A}/../../..
I=${DM:A}/inkling/model
G=(python3 ${DM:A}/../engine/bench/guard.py)
O=$W/results/parity/logs; mkdir -p $O
B=$W/bin-r3/dmeng
run() { local name=$1; shift; $G $O/$name.log 1800 -- "$@" --json $O/$name.json --out-ids $O/$name.ids || { echo "FAIL $name"; exit 1; }; }
for ra in 0 8; do
  for t in pitt code; do run rc${ra}_ink_$t $B --model $I --ctx 16384 --mem mlock --cache-gb 7.8 --temp 0 --n 128 --prompt-file $W/bench/prompts/inkling2_loop/$t.ids --run-ahead $ra; done
done
echo RC DONE
