#!/bin/zsh
# Requests through the model worker (dmserve) for several builds, alternating builds rep by rep (A B A B), one fresh
# worker per run with an empty prefix-cache folder (the first request is cold). Every run through the memory guard.
#   zsh serve_ab.sh OUT GUARD_PY REQS.json REPS "name=DMSERVE ..." -- [worker flags]
# Writes OUT/NAME_rK.json (worker_drive.py's output) and NAME_rK.log (guard log).
set -u
O=$1 G=$2 RQ=$3 REPS=$4; shift 4
arms=(); while [ "$1" != "--" ]; do arms+=$1; shift; done; shift
H=${0:A:h}; WD=$H/../drafts/worker_drive.py; mkdir -p $O
for r in $(seq 1 $REPS); do
  ord=($arms); [ $((r % 2)) = 0 ] && ord=(${(Oa)arms})
  for a in $ord; do
    n=${a%%=*}_r$r; b=${a#*=}
    [ -f $O/$n.json ] && continue
    rm -rf $O/disk_$n; mkdir -p $O/disk_$n
    DISK_GB=8 FOOT_GB=13.5 SWAP_MB=2000 POLL_S=2 python3 $G $O/$n.log 2400 -- python3 $WD $O/$n.json $RQ -- $b --disk $O/disk_$n "$@" > /dev/null
    rm -rf $O/disk_$n
    echo "$n: $(grep -o '"killed_by_guard": [^,]*' $O/$n.log | tail -1), others $(grep -o '"others_cpu_pct_mean": [^,]*' $O/$n.log | tail -1)"
  done
done
