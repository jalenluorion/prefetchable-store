#!/bin/zsh
# The recorded OpenCode replay (sampling as OpenCode sends it, seed 1) through a front end and worker, token ids logged
# (--log-ids), so two builds can be compared, with the repo of the front end as an argument.
#   replay.sh REPO WORKER OUT_PREFIX MAX_NEW TO [worker args...]
#   MODEL  the model folder; PY  a python with the front end's packages (default python3); REPLAY  the replay script and
#   its recorded session (not in this repository: REPLAY="agentic_replay.py SESSION_DIR")
R=$1; WK=$2; O=$3; MX=$4; TO=$5; shift 5
PY=${PY:-python3}
PORT=8094
$PY $R/python/serve/dmserve.py --model-dir ${MODEL:?set MODEL} --worker $WK --port $PORT --max-new-cap $MX --seed 1 --idle-s 0 --log-ids --log $O.server.jsonl -- "$@" > $O.server.out 2>&1 &
SP=$!
for i in $(seq 1 600); do grep -q '"event": "serving"' $O.server.out 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; done
grep -q '"event": "serving"' $O.server.out || { echo "server did not start"; tail -5 $O.server.out; kill $SP 2>/dev/null; exit 1; }
$PY ${=REPLAY:?set REPLAY} $O.replay.jsonl --url http://127.0.0.1:$PORT/v1 --only main --to $TO
kill $SP; wait $SP 2>/dev/null
echo done
