#!/bin/zsh
# A bug-fix task for Qwen3.8-Flash-Next (bench/serving/demo_task: intervals.py has a one-line bug, check.py fails on it) in
# stock OpenCode against this engine's server, timed from the prompt to the end of OpenCode's run, then check.py run
# again here. Run one model process at a time (it starts and stops the server).
#   demo_task.sh MODEL_DIR OUT_DIR [worker args...]      (e.g. -- --cache-gb 3.0)
# The task folder holds the task's files (check.py, intervals.py, opencode.json with this server's port) as a git
# repository at OUT_DIR/demo, the path OpenCode's system prompt names. The server runs with --demo (fast mode, effort
# low, canned titles, warm-up), OpenCode with its stock prompt and tools and the three clean-install variables. A first
# request ("hi") in the folder saves the system prompt (a new thread then finds it); the task is the next thread, 65 s
# later (OpenCode 1.19.0 can stall when started again within a minute). OC_MCP_OFF="name ...": MCP servers of the machine's global OpenCode config to switch off.
set -u
cd ${0:A:h}/../..
MODEL=${1:A}; O=${2:A}; shift 2
[ "${1:-}" = "--" ] && shift
PY=${PY:-python3}
PORT=${PORT:-8098}
OC=${OC:-$HOME/.opencode/bin/opencode}
D=$O/demo
mkdir -p $O && rm -rf $D $O/pfx && mkdir -p $D $O/pfx
cp bench/serving/demo_task/{check.py,intervals.py} $D/
$PY - $D/opencode.json $PORT ${=OC_MCP_OFF:-} <<'PY'
import json, sys
c = json.load(open("bench/serving/demo_task/opencode.json"))
c["provider"]["store"]["options"]["baseURL"] = f"http://127.0.0.1:{sys.argv[2]}/v1"
for name in sys.argv[3:]:  # MCP servers of this machine's global OpenCode config, switched off (they add tools to the prompt)
    c.setdefault("mcp", {})[name] = {"type": "local", "command": ["true"], "enabled": False}
json.dump(c, open(sys.argv[1], "w"), indent=2)
PY
(cd $D && git init -q . && git add . && git -c user.name=t -c user.email=t@t commit -qm init)
$PY python/serve/dmserve.py --model-dir $MODEL --port $PORT --mode fast --demo --log $O/server.jsonl -- --disk $O/pfx --disk-gb 2.5 "$@" > $O/server.out 2>&1 &
SP=$!
for i in $(seq 1 1200); do grep -q '"event": "serving"' $O/server.jsonl 2>/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 0.5; done
grep -q '"event": "serving"' $O/server.jsonl || { echo "server did not start"; tail -5 $O/server.out; kill $SP; exit 1; }
export OPENCODE_CONFIG=$D/opencode.json OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_EXTERNAL_SKILLS=1 OPENCODE_DISABLE_CLAUDE_CODE=1
now() { $PY -c 'import time; print(time.time())'; }
cd $D
t0=$(now); $OC run --format json "hi" > $O/warm.jsonl 2> $O/warm.err; t1=$(now)
git checkout -q . ; git clean -qfd -e opencode.json
sleep 65
t2=$(now); $OC run --format json "check.py isnt working. find the issue and fix it" > $O/task.jsonl 2> $O/task.err; t3=$(now)
$PY check.py > $O/check.txt 2>&1; echo "rc=$?" >> $O/check.txt
git diff > $O/diff.txt
cd - > /dev/null
kill $SP; wait $SP 2>/dev/null
rm -rf $O/pfx
$PY -c "import json, sys; t = list(map(float, sys.argv[1:5])); print(json.dumps({'warm_s': round(t[1] - t[0], 1), 'task_s': round(t[3] - t[2], 1), 'check': open(sys.argv[5]).read().strip()}))" $t0 $t1 $t2 $t3 $O/check.txt
