#!/bin/zsh
# One server process (ahead on or off), the forced replay with idle gaps, then stop. Run under engine/bench/guard.py.
#   ahead_replay.sh ON(0|1) OUT_PREFIX SESSION_DIR TURN2 GAPS [worker args...]
cd ${0:A:h}/../..
ON=$1; O=$2; S=$3; T2=$4; G=$5; shift 5
PY=${PY:-../../.venv/bin/python}
PORT=${PORT:-8092}
MODEL=${MODEL:-../../../q36/model}
$PY python/serve/dmserve.py --model-dir $MODEL --port $PORT --allow-force --ahead $ON --seed 1 --idle-s 0 --log-ids --log $O.server.jsonl \
  --worker build/dmserve -- "$@" > $O.server.out 2>&1 &
SP=$!
for i in $(seq 1 600); do grep -q '"event": "serving"' $O.server.out 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; done
grep -q '"event": "serving"' $O.server.out || { echo "server did not start"; tail -5 $O.server.out; kill $SP 2>/dev/null; exit 1; }
$PY bench/serving/ahead_replay.py $S $O.replay.jsonl --url http://127.0.0.1:$PORT/v1 --turn2 $T2 --gaps $G
sleep 2
kill $SP; wait $SP 2>/dev/null; for w in $(seq 1 120); do pgrep -f "^build/dmserve --model" > /dev/null || break; sleep 0.5; done
echo done
