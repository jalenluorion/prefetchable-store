"""The OpenAI-compatible server on the tiny Qwen3.8-Flash-Next: python/serve/dmserve.py (the folder's tokenizer and template
through python/serve/qwen4_exp/chat_format.py) in front of the engine's worker (build/dmserve) with its prefix cache.
Checks /v1/models, a chat completion with tools, the same request streamed (the same text), and a follow-up turn whose
prompt extends the first (its prefix comes from the cache: the log's reused positions).

  FLASH_TINY=<OUT of make_tiny.py> python tests/qwen4_exp/server_check.py     (python: tokenizers, jinja2)
"""
import json, os, signal, socket, subprocess, sys, tempfile, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MOD = os.path.join(os.environ["FLASH_TINY"], "model")
TMP = tempfile.mkdtemp()
s = socket.socket()
s.bind(("127.0.0.1", 0))
PORT = s.getsockname()[1]
s.close()
cmd = [sys.executable, os.path.join(REPO, "python/serve/dmserve.py"), "--model-dir", MOD, "--port", str(PORT), "--log", os.path.join(TMP, "server.jsonl"),
       "--ahead", "0", "--idle-s", "0", "--", "--cache-gb", "0.05", "--mem", "plain", "--res-mem", "plain", "--threads", "2", "--readers", "2",
       "--ctx", "4096", "--disk", os.path.join(TMP, "prefix")]
if os.environ.get("DMSERVE_BIN"):
    cmd[cmd.index("--") : cmd.index("--")] = ["--worker", os.environ["DMSERVE_BIN"]]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
fails = []


def say(ok, m):
    print(("PASS  " if ok else "FAIL  ") + m, flush=True)
    if not ok:
        fails.append(m)


def req(path, body=None):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode() if body else None,
                               headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=120).read().decode()


try:
    t0 = time.time()
    for line in p.stdout:
        if '"event": "serving"' in line:
            break
        if time.time() - t0 > 120:
            raise SystemExit("the server did not start")
    m = json.loads(req("/v1/models"))
    say(m["data"][0]["id"] == "qwen3.8-flash-next", f"/v1/models: {m['data'][0]['id']}")
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    msgs = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Hello, how are you today? Show me the file."}]
    body = {"model": "x", "messages": msgs, "tools": tools, "max_tokens": 24, "temperature": 0}
    r1 = json.loads(req("/v1/chat/completions", body))
    c1 = r1["choices"][0]
    say(r1["usage"]["completion_tokens"] > 0 and c1["finish_reason"] in ("stop", "length", "tool_calls"),
        f"chat completion: {r1['usage']['completion_tokens']} tokens, finish {c1['finish_reason']}, content {c1['message'].get('content', '')[:40]!r}")
    raw = req("/v1/chat/completions", {**body, "stream": True})
    text = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content") or "" for l in raw.splitlines()
                   if l.startswith("data: ") and l != "data: [DONE]" and json.loads(l[6:])["choices"])
    say(text == (c1["message"].get("content") or ""), f"streamed: the same text ({len(text)} characters)")
    follow = {**body, "messages": msgs + [{"role": "assistant", "content": c1["message"].get("content") or ""}, {"role": "user", "content": "Thanks."}]}
    json.loads(req("/v1/chat/completions", follow))
    time.sleep(0.5)
    recs = [json.loads(l) for l in open(os.path.join(TMP, "server.jsonl")) if l.strip().startswith("{")]
    last = [r for r in recs if r.get("event") == "request"][-1]
    wr = last.get("worker_ready") or {}
    say(wr.get("reused", 0) > 0, f"follow-up turn: {wr.get('reused')} of {last.get('prompt')} prompt positions from the prefix cache ({wr.get('source')})")
finally:
    os.killpg(p.pid, signal.SIGTERM)
    p.wait(timeout=30)
print("SERVER CHECK " + ("PASSED" if not fails else "FAILED"))
sys.exit(1 if fails else 0)
