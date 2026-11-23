#!/usr/bin/env python3
"""run_rr.sh: STEP_RERUN off / on, replica (engine vs rerun arm) under 0/1/2 co-runners, engine call by call quiet and
beside a 2-thread co-runner process. usage: rr_table.py LOGDIR"""
import json, os, re, sys
d = sys.argv[1]
for m in ("ink", "q36"):
    for b in (0, 1, 2):
        p = os.path.join(d, f"rr_step_{m}_b{b}.log")
        if not os.path.exists(p):
            continue
        t = open(p).read()
        arms = {j["arm"]: j for j in (json.loads(l) for l in t.splitlines() if l.startswith('{"model"'))}
        o = re.search(r'"others_cpu_pct_mean": ([0-9.]+)', t)
        e, r = arms.get("engine"), arms.get("rerun")
        if e and r:
            print(f"replica {m} co-runner threads {b} (others {o.group(1) if o else '?'}%): engine {e['gb_s']:.1f} rerun {r['gb_s']:.1f} GB/s "
                  f"({r['gb_s'] / e['gb_s'] - 1:+.1%}); stream {arms['stream']['gb_s']:.1f}")
    for l in (0, 2, "2b"):
        p = os.path.join(d, f"rr_{m}_l{l}")
        if not os.path.exists(p + ".json"):
            continue
        t = open(p + ".json").read()
        v = json.loads(re.search(r'"rerun_ab_calls_gb_s":(\[[^]]*\])', t).group(1))
        o = re.search(r'"others_cpu_pct_mean": ([0-9.]+)', open(p + ".log").read())
        print(f"engine {m} co-runner {l} threads (others {o.group(1) if o else '?'}%): off {v[1]:.1f} on {v[3]:.1f} GB/s ({v[3] / v[1] - 1:+.1%}), "
              f"calls {v[0]:.0f}/{v[2]:.0f}, chunks re-run {v[4]:.0f}")
