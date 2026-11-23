#!/usr/bin/env python3
"""The P1 tables from results/ios/p1/runs: gates (ids vs references) and FILE vs MLOCK per prompt.
python3 results/ios/p1/table.py [RUNS_DIR]"""
import glob, json, os, re, sys

O = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
rows = []
for j in sorted(glob.glob(os.path.join(O, "*.json"))):
    n = os.path.basename(j)[:-5]
    if not os.path.exists(os.path.join(O, n + ".log")):
        continue
    d = json.load(open(j))
    g = re.search(r"GUARD (.*)", open(os.path.join(O, n + ".log")).read())
    g = json.loads(g.group(1)) if g else {}
    f = d.get("file", {})
    st = d.get("stats", {}).get("memory", {})
    rows.append((n, d, g, f, st))
print("| run | tok/s | compute ms/tok | wait ms/tok | MB issued/tok | MB read/tok (preads) | hit (layer) | refaults/tok | peak footprint GB | file cache resident GB (end) | others' CPU % | power |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for n, d, g, f, st in rows:
    issued = d.get("mb_read_per_token", d.get("mb_issued_per_token", 0))
    fc = f.get("file_cache_resident_gb_at_end", st.get("file_cache_resident_gb", ""))
    rf = f.get("refaults_per_token", "")
    foot = d.get("footprint_peak_gb", g.get("peak_footprint_gb_guard", ""))
    print(f"| {n} | {d['tok_s']:.2f} | {d['compute_ms_per_token']:.1f} | {d['wait_ms_per_token']:.1f} | {issued:.1f} | {d.get('mb_preads_per_token', 0):.1f} | "
          f"{d['hit_rate']:.3f} | {rf if rf == '' else f'{rf:.2f}'} | {foot if foot == '' else f'{foot:.2f}'} | {fc if fc == '' else f'{fc:.2f}'} | "
          f"{g.get('others_cpu_pct_mean', '')} | {g.get('before', {}).get('power', '')} |")
