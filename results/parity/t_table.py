#!/usr/bin/env python3
"""Tables for run_t.sh / run_d.sh (counting runs). usage: t_table.py LOGDIR
(1) the expert step's down claims s0 / s1 / s2: expert-stage cycles and instructions per weight byte, spin thread-ms per token
(3) run-ahead off / on: compute + wait (+ the run-ahead's share of compute) per token, and the tok/s they add up to
(2) run-ahead depth 1-3 (35B) and where the remaining wait sits"""
import json, os, re, sys

d = sys.argv[1]
MB = {"q36": 700, "ink": 1839}  # expert weight MB per token (bench/parity.py)


def find(x, key):
    if isinstance(x, dict):
        if key in x:
            return x[key]
        for v in x.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def load(n):
    p = os.path.join(d, n)
    if not os.path.exists(p + ".json"):
        return None, None, "?"
    js = json.load(open(p + ".json"))
    st = {}
    txt = open(p + ".log").read()
    for l in txt.splitlines():
        if l.startswith("DMSTAGE"):
            f = l.split()
            st[f[1]] = [float(x) for x in f[2:6]]
    o = re.search(r'"others_cpu_pct_mean": ([0-9.]+)', txt)
    return js, st, o.group(1) if o else "?"


print("(1) down claims: experts (+ combine) stage")
print(f"{'run':14s} {'others%':>7s} {'cyc/B':>6s} {'ins/B':>6s} {'spin thr-ms/tok':>15s} {'experts ms':>10s} {'wait ms':>7s}")
for m in ("q36", "ink"):
    for b in ("s0", "s1", "s2"):
        js, st, o = load(f"t_{b}_{m}")
        if js is None:
            print(f"t_{b}_{m:9s} missing")
            continue
        n = max(1, js["decode_steps"])
        cyc = sum(st.get(k, [0] * 4)[1] + st.get(k, [0] * 4)[3] for k in ("moe_experts", "moe_combine")) / n
        ins = sum(st.get(k, [0] * 4)[0] + st.get(k, [0] * 4)[2] for k in ("moe_experts", "moe_combine")) / n
        B = MB[m] * 1e6
        print(f"t_{b}_{m:9s} {o:>7s} {cyc / B:6.3f} {ins / B:6.3f} {find(js, 'expert_spin_thread_ms_per_token') or 0:15.2f} "
              f"{js['by_stage_ms_per_token'].get('experts', 0):10.2f} {js['wait_ms_per_token']:7.2f}")

print("\n(3) run-ahead off / on (compute includes the run-ahead's own time)")
print(f"{'run':18s} {'others%':>7s} {'compute':>7s} {'wait':>6s} {'RA ms':>6s} {'sum':>6s} {'tok/s eq':>8s} {'ids':>5s}")
for m in ("q36", "ink"):
    for t in ("pitt", "code"):
        r = {}
        for ra in (0, 8):
            n = f"t_ra{ra}_{m}_{t}"
            js, st, o = load(n)
            if js is None:
                print(f"{n:18s} missing")
                continue
            k = max(1, js["decode_steps"])
            ram = (js.get("run_ahead") or {}).get("ms_total", 0) / k
            c, w = js["compute_ms_per_token"], js["wait_ms_per_token"]
            r[ra] = c + w
            ids = open(os.path.join(d, n + ".ids")).read().split()
            r[f"ids{ra}"] = ids
            print(f"{n:18s} {o:>7s} {c:7.2f} {w:6.2f} {ram:6.2f} {c + w:6.2f} {1000 / (c + w):8.2f} {len(ids):5d}")
        if 0 in r and 8 in r:
            print(f"{'  net':18s} {'':7s} on/off tok/s eq {r[0] / r[8] - 1:+.1%}   ids same: {r['ids0'] == r['ids8']}")

print("\n(2) run-ahead depth and where the wait sits (run-ahead on)")
for n in ("d1_q36_pitt", "d3_q36_pitt", "d1_q36_code", "d3_q36_code", "d1_ink_pitt", "d1_ink_code"):
    js, st, o = load(n)
    if js is None:
        print(f"{n:14s} missing")
        continue
    ra = js.get("run_ahead") or {}
    k = max(1, js["decode_steps"])
    print(f"{n:14s} others {o}%  compute {js['compute_ms_per_token']:.2f} wait {js['wait_ms_per_token']:.2f}  RA ms/tok {ra.get('ms_total', 0) / k:.2f} "
          f"ev/tok {ra.get('events', 0) / k:.1f} aborted {ra.get('aborted', 0):.0f}")
    print(f"   depth1: top8 {ra.get('top8_overlap', 0):.3f} would_read {ra.get('would_read', 0):.0f} found {ra.get('found_share', 0):.3f} "
          f"issued {ra.get('issued', 0):.0f} wasted {ra.get('wasted', 0):.0f} lead {ra.get('lead_ms', 0):.2f} ms")
    for dd in (2, 3):
        x = ra.get(f"depth{dd}")
        if x:
            print(f"   depth{dd}: events {x['events']:.0f} top8 {x['top8_overlap']:.3f} would_read {x['would_read']:.0f} found {x['found_share']:.3f} "
                  f"issued {x['issued']:.0f} wasted {x['wasted']:.0f}")
    if "steps" in ra:
        print(f"   steps {ra['steps']:.0f}: no wait {ra['steps_no_wait']:.0f}, waited after a named step {ra['steps_wait_named']:.0f} "
              f"({ra['wait_named_ms'] / k:.2f} ms/tok), waited with nothing named {ra['steps_wait_not_named']:.0f} ({ra['wait_not_named_ms'] / k:.2f} ms/tok)")
