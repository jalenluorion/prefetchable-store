#!/usr/bin/env python3
"""Run-ahead off / on, one line per run: tok/s, compute, wait, MB read and urgent MB per token, weight left out (fast mode),
the run-ahead counters. usage: ra_table.py LOGDIR NAME..."""
import json, os, re, sys
d = sys.argv[1]
for n in sys.argv[2:]:
    p = os.path.join(d, n)
    if not os.path.exists(p + ".json"):
        print(f"{n:22s} missing")
        continue
    j = json.load(open(p + ".json"))
    o = re.search(r'"others_cpu_pct_mean": ([0-9.]+)', open(p + ".log").read())
    ra = j.get("run_ahead", {})
    ap = j.get("approx") or {}
    print(f"{n:22s} others {o.group(1) if o else '?':>6s}%  {j.get('tok_s', 0):6.2f} tok/s  compute {j.get('compute_ms_per_token', 0):5.1f}  "
          f"wait {j.get('wait_ms_per_token', 0):5.1f}  MB {j.get('mb_read_per_token', 0):6.1f}  urgent {j.get('mb_read_urgent_per_token', 0) or 0:6.1f}  "
          f"left out {ap.get('weight_left_out_per_token_mean', '-')}  "
          + (f"RA ev/tok {ra['events'] / max(1, j.get('decode_steps') or j.get('positions') or 1):.1f} ms/tok {ra['ms_total'] / max(1, j.get('decode_steps') or j.get('positions') or 1):.2f} "
             f"aborted {ra['aborted']:.0f} overrun_ms/tok {ra['overrun_ms_total'] / max(1, j.get('decode_steps') or j.get('positions') or 1):.3f} top8 {ra['top8_overlap']:.3f} "
             f"found {ra['found_share']:.3f} lead {ra['lead_ms']:.2f} issued {ra['issued']:.0f} refused {ra['refused']:.0f} wasted {ra['wasted']:.0f}" if ra else ""))
