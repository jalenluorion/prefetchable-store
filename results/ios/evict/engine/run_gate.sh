#!/bin/bash
# Lossless identical-ids gate of the eviction options in the full engine, one model run at a time through guard.py.
#   results/ios/evict/engine/run_gate.sh   -> <name>.json / .ids / .log here
cd "$(dirname "$0")"
ENG=<work>/engine
DM=$(cd ../../../.. && pwd)/build/dmeng
MODEL=<work>/local/q36/model
PITT=$(tr -d '\n ' < $ENG/results/stage2/pitt_ids.txt)
CODE=$(tr -d '\n ' < $ENG/results/stage3/code_ids.txt)
run() {  # name gb prompt extra...
  local name=$1 gb=$2 ids=$3; shift 3
  while [ -f $ENG/bench/QUIET ] && [ $(( $(date +%s) - $(stat -f %m $ENG/bench/QUIET) )) -lt 720 ]; do sleep 30; done
  [ -f $name.json ] && return
  POLL_S=2 python3 $ENG/bench/guard.py $name.log 1200 -- $DM --model $MODEL --cache-gb $gb --ctx 16384 --temp 0 --mem mlock --prec i16 \
    --prompt-ids $ids --n 400 --json $name.json --out-ids $name.ids "$@"
}
run pitt_def_1.27 1.27402 $PITT
run pitt_hzcaps_1.27 1.27402 $PITT --evict hazard --layer-caps uniform
run code_def_1.27 1.27402 $CODE
run code_hzcaps_1.27 1.27402 $CODE --evict hazard --layer-caps uniform
run pitt_def_3.3 3.3 $PITT
run pitt_hz_3.3 3.3 $PITT --evict hazard
run code_def_3.3 3.3 $CODE
run code_hz_3.3 3.3 $CODE --evict hazard
