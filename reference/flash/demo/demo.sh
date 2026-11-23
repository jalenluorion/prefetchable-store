#!/bin/zsh
# One-command demo: start the store server (warm, prompt cache on) if it is not already up, then open stock OpenCode in this folder.
#   ./demo.sh            # prefetch mode, 10 named / keep 3 / 24 slots per layer, thinking effort low
#   ./demo.sh -v         # extra flags go to mlx_run.py (e.g. --reasoning-effort xhigh, the model's own default)
# Expects the weights in ../model (hf download jalenluorion/Qwen3.8-Flash-Next-moved-store --local-dir model --exclude "routers3/*").
# OpenCode runs with its stock prompt and tools. The three environment variables only keep this machine's personal setup out
# of the prompt (the machine's personal skills and instruction files) and stop OpenCode from updating itself.
# Stop the server afterwards with: pkill -f "mlx_run.py --dir model"
set -e
HERE=${0:A:h}
ROOT=${HERE:h}
[ -f "$ROOT/model/store/meta.json" ] || { echo "no weights in $ROOT/model (see the README's quick start)"; exit 1; }
# The interpreter with mlx, mlx_vlm, tokenizers, jinja2 and numpy: $PY if set, else the first candidate that imports them.
if [ -z "$PY" ]; then
  for c in "$ROOT/.venv/bin/python" python3 python; do
    if "$c" -c "import mlx.core, mlx_vlm, tokenizers, jinja2, numpy" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with mlx, mlx_vlm, tokenizers, jinja2 and numpy found; set PY=/path/to/python"; exit 1; }
OC=$(command -v opencode || echo "$HOME/.opencode/bin/opencode")
[ -x "$OC" ] || { echo "OpenCode not found (looked on the PATH and in ~/.opencode/bin)"; exit 1; }
cd "$HERE"
if ! curl -s -m 2 http://127.0.0.1:8080/v1/models >/dev/null; then
  echo "starting the server (log: demo/server.log)"
  [ -f server.log ] && mv server.log server.prev.log  # keep the last run's per-turn timings
  (cd "$ROOT" && nohup $PY mlx_run.py --dir model --serve 8080 --prompt-cache demo/cache --warm --reasoning-effort low "$@" > demo/server.log 2>&1 &)
  # "warm" takes seconds when today's snapshot of OpenCode's prompt exists, and is a cold prefill of ~7,900 tokens on the
  # first start of a day (OpenCode's prompt carries the date), on a new machine, or after a config change
  until grep -q " serving " server.log 2>/dev/null; do
    grep -q "Traceback\|Error" server.log 2>/dev/null && { tail -20 server.log; exit 1; }
    grep -q " loaded in " server.log 2>/dev/null && [ -z "$said" ] && { grep " loaded in " server.log | cut -c1-140; echo "warming OpenCode's prompt (seconds from today's snapshot, a few minutes if cold)..."; said=1; }
    sleep 1
  done
  grep " warm (\| serving " server.log | cut -c1-200
fi
export OPENCODE_DISABLE_EXTERNAL_SKILLS=1 OPENCODE_DISABLE_CLAUDE_CODE=1 OPENCODE_DISABLE_AUTOUPDATE=1
exec "$OC"
