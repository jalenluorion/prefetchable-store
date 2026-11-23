#!/usr/bin/env python3
"""Per-stage counts per decode token from counting-build logs (DMSTAGE lines, build with -DDM_COUNTERS=ON, DM_COUNTS=1).
usage: stages.py LOG [LOG ...]   (the run's JSON beside the log gives the decode steps)
Columns per stage, in millions per token: pool instructions and cycles (all threads), the main thread's serial
instructions and cycles, its spin-wait cycles (pool joins; expert reads in moe_experts) and its total cycles."""
import json, os, sys

MB = {  # bytes per token each stage reads (weights / state), for cycles per byte; from the weight file and record layout
    "attn_proj": 1001e6, "attn_core": 268e6 + 126e6, "attn_out": 89e6, "moe_router": 82e6, "moe_experts": 694e6, "head": 540e6}

for log in sys.argv[1:]:
    js = json.load(open(log[:-4] + ".json"))
    n = js["decode_steps"]
    rows = {}
    for l in open(log):
        if l.startswith("DMSTAGE"):
            f = l.split()
            v = [float(x) for x in f[2:]]
            while len(v) < 7:
                v.append(float("nan"))
            rows[f[1]] = v
    print(f"== {os.path.basename(log)}  ({n} tokens, {js['tok_s']} tok/s, compute {js['compute_ms_per_token']} ms/token)")
    print(f"{'stage':12s} {'pool Mins':>10s} {'pool Mcyc':>10s} {'ser Mins':>9s} {'ser Mcyc':>9s} {'spin Mcyc':>10s} {'main Mcyc':>10s} {'cyc/B':>6s}")
    tot = [0.0] * 7
    for k, v in rows.items():
        if all(x == 0 for x in v[:4]):
            continue
        sv, sc, pi, pc, spi, spc, mc = v
        cpb = f"{pc / n / MB[k]:.3f}" if k in MB else ""
        print(f"{k:12s} {pi / n / 1e6:10.2f} {pc / n / 1e6:10.2f} {sv / n / 1e6:9.2f} {sc / n / 1e6:9.2f} {spc / n / 1e6:10.2f} {mc / n / 1e6:10.2f} {cpb:>6s}")
        for i, x in enumerate((sv, sc, pi, pc, spi, spc, mc)):
            tot[i] += x
    sv, sc, pi, pc, spi, spc, mc = tot
    print(f"{'total':12s} {pi / n / 1e6:10.2f} {pc / n / 1e6:10.2f} {sv / n / 1e6:9.2f} {sc / n / 1e6:9.2f} {spc / n / 1e6:10.2f} {mc / n / 1e6:10.2f}")
