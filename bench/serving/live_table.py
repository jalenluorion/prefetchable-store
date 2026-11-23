#!/usr/bin/env python3
"""One row per request of a live server log (bench/serving/live_opencode.sh): kind, prompt, time to first token, what
was reused / recomputed and from where, the answer's tokens, decode tokens/s, tool calls, the process footprint at the
request's end; plus the idle-time prefills and the walls.   usage: live_table.py PREFIX"""
import json, os, sys

P = sys.argv[1]
srv = [json.loads(l) for l in open(P + ".server.jsonl")]
rows = []
for x in srv:
    if x.get("event") == "request":
        wr, we = x.get("worker_ready", {}), x.get("worker_end", {})
        rows.append({"kind": x["kind"], "prompt": x["prompt"], "ttft_s": x.get("ttft_s"), "queued_s": x.get("queued_s"), "reused": wr.get("reused"),
                     "computed": wr.get("computed"), "source": wr.get("source"), "lcp_live": wr.get("lcp_live"), "live_len": wr.get("live_len"),
                     "prefill_s": round(wr.get("ms_prefill", 0) / 1e3, 2), "completion": x.get("completion"), "decode_tok_s": round(we.get("decode_tok_s", 0), 2),
                     "tool_calls": x.get("tool_calls"), "finish": x.get("finish"), "footprint_mb": (we.get("mem") or {}).get("footprint_mb"),
                     "total_s": x.get("total_s"), "lcp_prev_main": x.get("lcp_prev_main"), "diverged_prev": x.get("diverged_prev"), "diverged_this": x.get("diverged_this")})
    elif x.get("event") in ("ahead", "worker_exit", "worker_restart", "hotset"):
        rows.append({"event": x["event"], **{k: x.get(k) for k in ("predicted", "computed", "cancelled", "ms_prefill", "rc", "refilled", "ms")}})
walls = json.load(open(P + ".walls.json")) if os.path.exists(P + ".walls.json") else {}
tests = open(P + ".tests.txt").read()[-300:] if os.path.exists(P + ".tests.txt") else ""
print(json.dumps({"walls": walls, "tests_tail": tests, "rows": rows}, indent=1))
