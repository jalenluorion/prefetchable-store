#!/usr/bin/env python3
"""One line per timed run: tok/s, compute and wait per token, the stages, others' CPU (guard), ids against a reference.
usage: wall.py LOGDIR NAME[=REF] ..."""
import json, os, re, sys
d = sys.argv[1]
for a in sys.argv[2:]:
    n, _, ref = a.partition("=")
    p = os.path.join(d, n)
    if not os.path.exists(p + ".json"):
        continue
    j = json.load(open(p + ".json"))
    o = re.search(r'"others_cpu_pct_mean": ([0-9.]+)', open(p + ".log").read())
    st = j["by_stage_ms_per_token"]
    ids = ""
    if ref:
        x, y = open(p + ".ids").read().strip().split(","), open(os.path.join(d, ref + ".ids")).read().strip().split(",")
        m = min(len(x), len(y))
        ids = "ids same" if x[:m] == y[:m] else "ids DIFFER"
    print(f"{n:22s} {j['tok_s']:6.2f} tok/s  compute {j['compute_ms_per_token']:5.1f}  wait {j['wait_ms_per_token']:5.1f}  "
          f"proj {st['attn_proj']:5.2f} core {st['attn_core']:4.2f} out {st['attn_out']:4.2f} router {st['moe_router']:4.2f} "
          f"experts {st['experts']:5.2f} head {st['head']:4.2f} hooks {st['prefetch_hooks']:4.2f}  others {o.group(1) if o else '?'}%  {ids}")
