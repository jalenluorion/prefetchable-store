#!/bin/zsh
# One live stock-OpenCode task on the agent profile (the server given only the model folder), cold (the disk prefix
# moved aside first), then a warm start: a new server process, the task's first request again. Run under guard.py.
#   live_profile.sh MODEL_DIR MODEL_ID OPENCODE_CONFIG OUT_PREFIX
cd ${0:A:h}/../..
MODEL=$1; MID=$2; CFG=$3; O=${4:A}
PY=${PY:-../../.venv/bin/python}
[ -d $MODEL/.dmeng/prefix ] && mv $MODEL/.dmeng/prefix $MODEL/.dmeng/prefix.prev-$(date +%Y%m%d%H%M%S)
SERVE_ARGS="--log-bodies --log-ids" bench/serving/live_opencode.sh $MODEL $MID $CFG $O
$PY python/serve/dmserve.py --model-dir $MODEL --model-id $MID --port 8090 --idle-s 0 --log $O.warm.server.jsonl --worker build/dmserve > $O.warm.server.out 2>&1 &
SP=$!
for i in $(seq 1 600); do grep -q '"event": "serving"' $O.warm.server.out 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; done
$PY bench/serving/warm_request.py $O.server.jsonl http://127.0.0.1:8090/v1 $O.warm.json
sleep 1; kill $SP; wait $SP 2>/dev/null; for w in $(seq 1 120); do pgrep -f "^build/dmserve --model" > /dev/null || break; sleep 0.5; done
echo done
