#!/usr/bin/env python3
"""Replay a recorded OpenCode session (stock OpenCode 1.19.0 bodies, engine/results/agentic/replay/<session>/) to measure
a new user turn's time to first token after an idle gap, with the server's idle-time prefill on or off.

Each recorded answer of the first user turn is sent as dm_force (the server, started with --allow-force, feeds it as the
model's own turn), so every next request finds the live state a live session would have; the new-user-turn request is
then sent after GAP seconds and generates freely (fixed seed). Repeated for each gap in one server process.

  ahead_replay.py SESSION_DIR OUT.jsonl --url URL --turn2 7 --gaps 5,15,30 [--upto 6]"""
import argparse, glob, json, os, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8092/v1")
ap.add_argument("--turn2", type=int, default=0, help="number of the recorded new-user-turn request")
ap.add_argument("--gaps", default="5,15,30")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--single", type=int, default=0, help="send only this recorded request, free (no dm_force), then exit")
a = ap.parse_args()
recs = {r["n"]: r for r in (json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.session, "[0-9][0-9][0-9].json"))))}
mains = [n for n in sorted(recs) if recs[n]["body"].get("tools")]
turn1 = [n for n in mains if n < a.turn2]
OUT = open(a.out, "a")


def send(r, force, tag, gap):
    body = dict(r["body"])
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    body["seed"] = 1
    body["max_tokens"] = a.max_new
    if force:
        body["dm_force"] = r["answer"]
    req = urllib.request.Request(a.url + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, usage = None, None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            ch = json.loads(d)
            if ch.get("usage"):
                usage = ch["usage"]
            for c in ch.get("choices", []):
                dl = c.get("delta", {})
                if first is None and any(dl.get(k) for k in ("reasoning_content", "content", "tool_calls")):
                    first = time.time()
    rec = {"tag": tag, "gap_s": gap, "n": r["n"], "forced": force, "ttft_s": round(first - t0, 3) if first else None, "total_s": round(time.time() - t0, 3),
           "usage": usage, "t": round(t0, 3)}
    OUT.write(json.dumps(rec) + "\n")
    OUT.flush()
    print(json.dumps(rec), flush=True)


if a.single:
    send(recs[a.single], False, "single", 0)
    raise SystemExit(0)
for gap in [float(g) for g in a.gaps.split(",")]:
    for n in turn1:
        send(recs[n], True, "turn1", gap)
    time.sleep(gap)
    send(recs[a.turn2], False, "new_user_turn", gap)
