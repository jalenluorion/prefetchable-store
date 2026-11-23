#!/usr/bin/env python3
"""Send the first main request of a logged live session (front end --log-bodies) to a server again: the warm start
(a new process, the system prompt from the disk prefix). usage: warm_request.py SERVER_LOG URL OUT.json"""
import json, sys, time, urllib.request

log, url, out = sys.argv[1:4]
body = next(json.loads(l)["body"] for l in open(log) if '"body"' in l and json.loads(l).get("kind") == "main")
body = dict(body, stream=True, stream_options={"include_usage": True}, seed=1)
t0 = time.time()
first = None
req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        l = raw.decode(errors="replace").strip()
        if l.startswith("data:") and l[5:].strip() != "[DONE]":
            ch = json.loads(l[5:])
            for c in ch.get("choices", []):
                if first is None and any(c.get("delta", {}).get(k) for k in ("reasoning_content", "content", "tool_calls")):
                    first = time.time()
json.dump({"ttft_s": round(first - t0, 3) if first else None, "total_s": round(time.time() - t0, 3)}, open(out, "w"))
print(open(out).read())
