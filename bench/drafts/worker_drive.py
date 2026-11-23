#!/usr/bin/env python3
"""Drive the model worker (build/dmserve) by token ids, without the HTTP front end: a list of requests (each: prompt ids,
max_new, temperature, optional tool-call fragments), the generated ids and the worker's E-line JSON of each, to a JSON file.
Used to compare drafting off / on in the server's own decode loop (ids must be identical).

  worker_drive.py OUT.json REQS.json -- WORKER --model DIR [worker flags...]
REQS.json: [{"ids": [...] | "ids_file": F, "max_new": N, "temp": 0, "top_p": 1, "top_k": 0, "seed": 1, "marks": [...], "skel": [[...], ...],
             "continue": true (prompt = the previous request's prompt + its output + "ids")}, ...]"""
import json, subprocess, sys, time

out_path, reqs_path = sys.argv[1], sys.argv[2]
cmd = sys.argv[sys.argv.index("--") + 1:]
reqs = json.load(open(reqs_path))
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
ready = p.stdout.readline()
assert ready.startswith("READY"), ready
print(ready.strip(), file=sys.stderr, flush=True)  # the worker's resolved settings and cache size, for the run's log
res, prev = [], None
for k, rq in enumerate(reqs):
    ids = rq.get("ids") or [int(x) for x in open(rq["ids_file"]).read().replace(",", " ").split()]
    if rq.get("continue") and prev is not None:
        ids = prev["prompt"] + prev["gen"] + ids
    skel = rq.get("skel", [])
    marks = rq.get("marks", [])
    stop = rq.get("stop", [])
    rid = k + 1
    t0 = time.time()
    p.stdin.write(f"R {rid} {rq.get('max_new', 64)} {rq.get('temp', 0)} {rq.get('top_p', 1)} {rq.get('top_k', 0)} {rq.get('seed', 1)} {len(ids)} {len(marks)} {len(stop)} {len(skel)}\n"
                  f"{' '.join(map(str, ids))}\n{' '.join(map(str, marks))}\n{' '.join(map(str, stop))}\n[]\n")
    for f in skel:
        p.stdin.write(" ".join(map(str, f)) + "\n")
    p.stdin.flush()
    gen, end, b = [], None, None
    for line in p.stdout:
        line = line.rstrip("\n")
        if line.startswith("T "):
            gen.append(int(line.split()[2]))
        elif line.startswith("B "):
            b = json.loads(line.split(" ", 2)[2])
        elif line.startswith("E "):
            parts = line.split(" ", 3)
            end = {"reason": parts[2], "json": json.loads(parts[3]) if len(parts) > 3 else {}}
            break
    prev = {"prompt": ids, "gen": gen}
    res.append({"prompt_len": len(ids), "gen": gen, "wall_s": time.time() - t0, "B": b, "E": end})
    print(f"req {rid}: prompt {len(ids)}, {len(gen)} tokens, {end['reason']}, draft {json.dumps(end['json'].get('draft', {}))[:200]}", flush=True)
p.stdin.write("Q\n")
p.stdin.flush()
p.wait(60)
json.dump(res, open(out_path, "w"))
