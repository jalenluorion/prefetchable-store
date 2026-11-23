#!/bin/bash
# replay (C++ dm::Cache) vs simulator (eviction_v3 core, deterministic shortlist) on one engine trace
#   cmp_replay_sim.sh REC TRACE START POLICY_JSON "SLOTS MODE" ...   (MODE pooled|uniform)
W=${CP_WORK:-.}
PY=${DEEP_MEMORY}/engine/.venv/bin/python
REC=$1 TR=$2 START=$3 PJ=$4
shift 4
for cfg in "$@"; do
  set -- $cfg
  ex=""; cx=""
  if [ "$2" = uniform ]; then ex=uniform; cx=caps; fi
  for pol in v3 hazard decay; do
    case $pol in v3) cp=$PJ;; hazard) cp=hazard;; decay) cp=freq;; esac
    a=$($W/bin/evict_replay $REC $TR $1 $cp $W/logs/replay.txt $ex $START | python3 -c "import json,sys;j=json.load(sys.stdin);print(j['hit_rate'],j['mb_per_token'],j['policy_ms_per_token'])")
    b=$($PY $W/sim_trace.py $REC $TR $START $1 $pol $cx --policy-json $PJ --det | python3 -c "import json,sys;j=json.load(sys.stdin);print(j['hit_rate'],j['mb_per_token'])")
    echo "$1 $2 $pol | replay hit MB ms: $a | sim hit MB: $b"
  done
done
