#!/bin/zsh
# A live stock OpenCode task against the server (the task of engine RESULTS-AGENTIC.md: a fresh tiny repo, two user
# turns: opencode run, then run --continue). Run under engine/bench/guard.py.
#   live_opencode.sh MODEL_DIR MODEL_ID OPENCODE_CONFIG OUT_PREFIX [worker args...]
# OpenCode gets only the provider entry (OPENCODE_CONFIG) plus the clean-install switches of the recorded replay.
cd ${0:A:h}/../..
MODEL=$1; MID=$2; CFG=${3:A}; O=${4:A}; shift 4
E=$PWD
PY=${PY:-../../.venv/bin/python}
REPO=${LIVE_REPO_BASE:-/tmp}/oc_live_$(date +%H%M%S)
mkdir -p $REPO && cd $REPO
printf 'def mean(xs):\n    return sum(xs) / len(xs)\n' > stats.py
printf 'import unittest\n\nfrom stats import mean\n\n\nclass TestStats(unittest.TestCase):\n    def test_mean(self):\n        self.assertEqual(mean([1, 2, 3]), 2)\n\n\nif __name__ == "__main__":\n    unittest.main()\n' > test_stats.py
git init -q . && git add . && git -c user.name=t -c user.email=t@t commit -qm init
cd $E
$PY python/serve/dmserve.py --model-dir $MODEL --model-id $MID --port 8090 --idle-s 0 --log $O.server.jsonl --worker build/dmserve ${=SERVE_ARGS} -- "$@" > $O.server.out 2>&1 &
SP=$!
for i in $(seq 1 600); do grep -q '"event": "serving"' $O.server.out 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; done
grep -q '"event": "serving"' $O.server.out || { echo "server did not start"; tail -5 $O.server.out; kill $SP; exit 1; }
export OPENCODE_CONFIG=$CFG OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_EXTERNAL_SKILLS=1 OPENCODE_DISABLE_CLAUDE_CODE=1
cd $REPO
t0=$(date +%s); ~/.opencode/bin/opencode run --format json "Add a median(xs) function to stats.py and a test for it in test_stats.py, then run the tests." > $O.turn1.jsonl 2>&1; t1=$(date +%s)
sleep ${TURN_GAP:-0}
t1b=$(date +%s); ~/.opencode/bin/opencode run --continue --format json "Make median raise ValueError on an empty list, and test that too." > $O.turn2.jsonl 2>&1; t2=$(date +%s)
echo "{\"turn1_s\": $((t1-t0)), \"turn2_s\": $((t2-t1b)), \"gap_s\": $((t1b-t1)), \"repo\": \"$REPO\"}" > $O.walls.json
python3 -m unittest -q > $O.tests.txt 2>&1; git -C $REPO diff > $O.diff.txt
cd $E; sleep 2; kill $SP; wait $SP 2>/dev/null; for w in $(seq 1 120); do pgrep -f "^build/dmserve --model" > /dev/null || break; sleep 0.5; done
cat $O.walls.json
