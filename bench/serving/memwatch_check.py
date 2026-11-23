#!/usr/bin/env python3
"""The auto cache size and the memory-pressure watcher on the real worker (no front end): start with --cache-gb auto,
generate greedily from a fixed prompt, force the watcher to shrink the cache to its minimum (thresholds raised through
the worker's P line) down to the floor the worker reports (its watcher line: floor_gb), generate again, let it grow
back, generate again. The three generations must be the same tokens (only the budget moved); every budget change is
logged with the process footprint. Every wait has a hard limit and the whole check a deadline (DEADLINE_S, 290 s): a
worker that stops answering fails the check instead of hanging it.
  memwatch_check.py OUT.jsonl IDS_FILE N_GEN -- build/dmserve --model M --cache-gb auto ...   (run under guard.py)"""
import json, os, subprocess, sys, time, threading, queue

out, ids_file, ngen = sys.argv[1], sys.argv[2], int(sys.argv[3])
cmd = sys.argv[sys.argv.index("--") + 1:]
ids = [int(x) for x in open(ids_file).read().replace(",", " ").split()]
p = subprocess.Popen(cmd + ["--mem-grow-after-s", "3", "--mem-retry-s", "0"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
O = open(out, "w")
lines = queue.Queue()


def rd():
    for l in p.stdout:
        lines.put(l.rstrip("\n"))


threading.Thread(target=rd, daemon=True).start()
try:
    ready = lines.get(timeout=180)
except queue.Empty:
    print("FAILED no READY")
    p.kill()
    sys.exit(1)
O.write(json.dumps({"ready": json.loads(ready[6:])}) + "\n")
budget = [None]
floor = [2.0]
T_END = time.time() + float(os.environ.get("DEADLINE_S", 290))


def fail(why):
    O.write(json.dumps({"failed": why, "budget_gb": budget[0]}) + "\n")
    O.flush()
    print("FAILED", why)
    p.kill()
    sys.exit(1)


def pump(until, timeout):
    t_stop = min(time.time() + timeout, T_END)
    while time.time() < t_stop:
        try:
            l = lines.get(timeout=0.5)
        except queue.Empty:
            l = ""
        if l.startswith("M "):
            j = json.loads(l[4:])
            if "floor_gb" in j:
                floor[0] = j["floor_gb"]
                budget[0] = j["start_gb"]
            else:
                budget[0] = j["budget_gb"]
            O.write(json.dumps({"t": round(time.time(), 2), "budget_change": j}) + "\n")
            O.flush()
        r = until(l)  # also on a quiet tick: the condition may already hold
        if r is not None:
            return r
    return None


def gen(rid, tag):
    p.stdin.write(f"R {rid} {ngen} 0 1 0 1 {len(ids)} 0 1\n{' '.join(map(str, ids))}\n\n{os.environ.get('STOP', '248046')}\n[]\n")
    p.stdin.flush()
    toks = []

    def u(l):
        if l.startswith(f"T {rid} "):
            toks.append(int(l.split()[2]))
        if l.startswith(f"E {rid} "):
            return json.loads(l.split(" ", 3)[3])
    e = pump(u, 150)
    if e is None:
        fail(f"no end of generation {tag} within its limit")
    O.write(json.dumps({"tag": tag, "budget_gb": budget[0], "tokens": toks, "decode_tok_s": e.get("decode_tok_s"), "mem": e.get("mem"),
                        "hit": (e.get("decode") or {}).get("hit"), "mb_per_step": (e.get("decode") or {}).get("mb_read_per_step")}) + "\n")
    O.flush()
    return toks


a = gen(1, "start")
start = max(json.loads(ready[6:])["cache_auto"]["start_gb"], floor[0])
p.stdin.write("P 101 200\n"); p.stdin.flush()  # shrink: the free level is always below 101%
pump(lambda l: True if budget[0] is not None and budget[0] <= floor[0] + 0.01 else None, 60)
b = gen(2, "shrunk")
p.stdin.write("P -1 0\n"); p.stdin.flush()  # grow back: always calm
pump(lambda l: True if budget[0] is not None and budget[0] >= start - 0.01 else None, 40)  # real pressure may hold it lower
c = gen(3, "grown")
O.write(json.dumps({"tokens_identical": a == b == c, "n": len(a), "floor_gb": floor[0], "wall_s": round(time.time() - (T_END - float(os.environ.get("DEADLINE_S", 290))), 1)}) + "\n")
p.stdin.write("Q\n"); p.stdin.flush()
try:
    p.wait(30)
except subprocess.TimeoutExpired:
    p.kill()
print("identical", a == b == c, len(a))
sys.exit(0 if a == b == c else 1)
