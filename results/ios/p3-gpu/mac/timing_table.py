#!/usr/bin/env python3
"""The Mac timing cells (run_gates.sh timing) as a markdown table: prefill ms of the second prefill (first in
parentheses), read wait, the attention / expert stage times, and the GPU split's waits."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
cells = [("off", "off", 4), ("attn50", "attn 0.5", 4), ("exp50", "experts 0.5", 4), ("both50", "both 0.5", 4), ("both35", "both 0.35", 4),
         ("both65", "both 0.65", 4), ("both50int", "both 0.5, integer kernel", 4), ("both65_t2", "both 0.65", 2), ("off_t2", "off", 2)]
print("| cell | CPU threads | prefill ms (first) | tok/s | read wait ms | attn proj + out ms | attn core ms | experts ms | GPU wait attn / experts ms | others' CPU % |")
print("|---|---|---|---|---|---|---|---|---|---|")
for key, name, t in cells:
    p = os.path.join(D, f"t_{key}.log")
    if not os.path.exists(p):
        continue
    reps, other = [], ""
    for l in open(p):
        if l.startswith('{"mode":"prefill_bench"'):
            reps.append(json.loads(l))
        if l.startswith("GUARD"):
            other = "%.0f" % json.loads(l[6:]).get("others_cpu_pct_mean", 0)
    if not reps:
        print(f"| {name} | {t} | (no result) | | | | | | | {other} |")
        continue
    r = reps[-1]
    st = r["prefill"]["ms_by_stage"]
    g = r["prefill"].get("gpu")
    gw = f'{g["attn"]["wait_ms"]:.0f} / {g["experts"]["wait_ms"]:.0f}' if g else "-"
    first = f' ({reps[0]["ms"]:.0f})' if len(reps) > 1 else ""
    print(f'| {name} | {t} | {r["ms"]:.0f}{first} | {r["tok_s"]:.1f} | {st["wait"]:.0f} | {st["attn_proj"] + st["attn_out"]:.0f} | {st["attn_core"]:.0f} | {st["experts"]:.0f} | {gw} | {other} |')
