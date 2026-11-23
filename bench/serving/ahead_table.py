#!/usr/bin/env python3
"""Tables for the idle-time prefill runs (bench/serving/ahead_replay.sh): per run and gap, the new user turn's time to
first token, what its prompt reused / recomputed, the idle-time prefill that ran before it (positions, seconds of
prefill, cancelled or not, the delay a request saw), the power state; and whether every request's generated tokens are
the same in the off and on runs (fixed seed).   usage: ahead_table.py DIR RUN...   (RUN: off1 on1 ...)"""
import json, os, sys

d, runs = sys.argv[1], sys.argv[2:]
out = {"runs": {}, "tokens_identical": None}
gens = {}
for r in runs:
    rep = [json.loads(l) for l in open(os.path.join(d, r + ".replay.jsonl"))]
    srv = [json.loads(l) for l in open(os.path.join(d, r + ".server.jsonl"))]
    power = open(os.path.join(d, r + ".log.power")).read().strip() if os.path.exists(os.path.join(d, r + ".log.power")) else ""
    reqs = [x for x in srv if x.get("event") == "request" and x.get("kind") == "main"]
    gens[r] = [x.get("gen_ids") for x in reqs]
    rows, i_req, aheads = [], 0, []
    for x in srv:
        if x.get("event") == "ahead":
            aheads.append(x)
        if x.get("event") == "request" and x.get("kind") == "main":
            i_req += 1
            if not x.get("forced"):  # the new user turn
                wr = x.get("worker_ready", {})
                rows.append({"ttft_s": x.get("ttft_s"), "prompt": x.get("prompt"), "reused": wr.get("reused"), "computed": wr.get("computed"),
                             "source": wr.get("source"), "ms_wait_worker": round(wr.get("ms_wait_worker", 0), 1), "ms_prefill": round(wr.get("ms_prefill", 0)),
                             "ahead_before": [{k: a.get(k) for k in ("predicted", "computed", "cancelled", "ms_prefill", "ms_cancel_to_stop", "mb_read", "source")} for a in aheads]})
                aheads = []
    nut = [x for x in rep if x["tag"] == "new_user_turn"]
    for row, x in zip(rows, nut):
        row["gap_s"] = x["gap_s"]
        row["client_ttft_s"] = x["ttft_s"]
    foot = max((x.get("worker_end", {}).get("mem", {}).get("footprint_mb", 0) for x in reqs), default=0)
    out["runs"][r] = {"power": power, "new_user_turn": rows, "max_footprint_mb_at_request_end": foot}
names = list(gens)
if len(names) > 1:
    base = gens[names[0]]
    out["tokens_identical"] = {n: gens[n] == base for n in names[1:]}
    out["requests_compared"] = len(base)
print(json.dumps(out, indent=1))
