#!/usr/bin/env python3
"""run_oc.sh cells as a markdown table: per cell both prefills (s, tok/s), peak footprint, GPU waits, others' CPU."""
import glob, json, os, re
D = os.path.dirname(os.path.abspath(__file__))
names = {"off": "split off", "off2": "split off (again, later)", "old38": "integer kernel, projections 0.38 (old setting)", "a50": "matrix, projections 0.5", "a65": "matrix, projections 0.65",
         "a75": "matrix, projections 0.75", "a50e50": "matrix, 0.5 / experts 0.5", "a65e65": "matrix, 0.65 / experts 0.65", "a65e50": "matrix, 0.65 / experts 0.5",
         "a75e75": "matrix, 0.75 / experts 0.75", "a75e50": "matrix, 0.75 / experts 0.5", "a50e38": "matrix, 0.5 / experts 0.38", "a40": "matrix, projections 0.4", "a40e40": "matrix, 0.4 / experts 0.4", "a45e45": "matrix, 0.45 / experts 0.45", "a40e30": "matrix, 0.4 / experts 0.3"}
print("| threads | cell | prefill s, run 1 / run 2 | tok/s (best) | peak footprint GB | GPU wait attn / experts s (run 2) | others' CPU % (mean) | swap MB start |")
print("|---|---|---|---|---|---|---|---|")
rows = []
for p in glob.glob(os.path.join(D, "oc_t*_*.log")):
    m = re.match(r"oc_t(\d+)_(\w+)\.log", os.path.basename(p))
    t, c = int(m.group(1)), m.group(2)
    reps, gd = [], {}
    for l in open(p):
        if l.startswith('{"mode":"prefill_bench"'): reps.append(json.loads(l))
        if l.startswith("GUARD"): gd = json.loads(l[6:])
    rows.append((t, list(names).index(c) if c in names else 99, c, reps, gd))
for t, _, c, reps, gd in sorted(rows):
    if not reps:
        print(f"| {t} | {names.get(c, c)} | (no result) | | | | {gd.get('others_cpu_pct_mean', '')} | |"); continue
    s = " / ".join("%.1f" % (r["ms"] / 1e3) for r in reps)
    best = max(r["tok_s"] for r in reps)
    fp = max(r["footprint_peak_mb"] for r in reps) / 1e3
    g = reps[-1]["prefill"].get("gpu")
    gw = "%.1f / %.1f" % (g["attn"]["wait_ms"] / 1e3, g["experts"]["wait_ms"] / 1e3) if g else "-"
    print(f"| {t} | {names.get(c, c)} | {s} | {best:.1f} | {fp:.2f} | {gw} | {gd.get('others_cpu_pct_mean', 0):.0f} | {gd.get('swap_start_mb', '')} |")
