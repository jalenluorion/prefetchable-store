#!/bin/zsh
# The first request after a restart, with and without the saved hot set. Run under engine/bench/guard.py.
#   hotset_restart.sh OUT_PREFIX SESSION_DIR FIRST_N HOTSET_FILE ORDER [worker args...]
# ORDER: e.g. "save off on off on": "save" serves the session's first user turn (forced answers) with --hotset and stops
# (the worker writes the hot set); "on" / "off" start a fresh server with / without --hotset, wait REFILL_WAIT s (the
# refill runs meanwhile), send recorded request FIRST_N (free, fixed seed, MAX_NEW tokens), stop.
cd ${0:A:h}/../..
O=$1; S=$2; N=$3; HS=$4; ORDER=$5; shift 5
PY=${PY:-../../.venv/bin/python}
PORT=${PORT:-8092}
MODEL=${MODEL:-../../../q36/model}
i=0
for st in ${=ORDER}; do
  i=$((i+1)); P=$O.$i.$st
  HA=(); [ $st != off ] && HA=(--hotset $HS)
  $PY python/serve/dmserve.py --model-dir $MODEL --port $PORT --allow-force --ahead 0 --seed 1 --idle-s 0 --log-ids --log $P.server.jsonl \
    --worker build/dmserve -- "$@" $HA > $P.server.out 2>&1 &
  SP=$!
  for k in $(seq 1 600); do grep -q '"event": "serving"' $P.server.out 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; done
  if [ $st = save ]; then
    $PY bench/serving/ahead_replay.py $S $P.replay.jsonl --url http://127.0.0.1:$PORT/v1 --turn2 $((N+5)) --gaps 0
  else
    sleep ${REFILL_WAIT:-8}
    $PY bench/serving/ahead_replay.py $S $P.replay.jsonl --url http://127.0.0.1:$PORT/v1 --single $N --max-new ${MAX_NEW:-128}
  fi
  sleep 1; kill $SP; wait $SP 2>/dev/null; for w in $(seq 1 120); do pgrep -f "^build/dmserve --model" > /dev/null || break; sleep 0.5; done
done
echo done
