#!/bin/bash
# one guarded 35B run: run_q36.sh NAME GB PROMPT(pitt|code) [extra dmeng args...]  -> runs/NAME.{json,ids,log,rec,trace}
W=${CP_WORK:-.}
ENG=${DEEP_MEMORY}/engine
DM=${CP_REPO}/build-cp/dmeng
MODEL=${DEEP_MEMORY}/local/inkling/model
name=$1 gb=$2 pr=$3; shift 3
IDS=$(tr -d "\n " < $ENG/results/inkling_draft/pitt_ids.txt)
N=${N:-160}
cd $W/runs
[ -f $name.json ] && exit 0
while [ -f $ENG/bench/QUIET ] && [ $(( $(date +%s) - $(stat -f %m $ENG/bench/QUIET) )) -lt 720 ]; do sleep 15; done
POLL_S=5 python3 $ENG/bench/guard.py $name.log 1500 -- $DM --model $MODEL --cache-gb $gb --ctx 16384 --temp 0 --mem mlock \
  --prompt-ids $IDS --n $N --json $name.json --out-ids $name.ids --record $name.rec "$@"
