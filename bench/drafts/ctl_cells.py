#!/usr/bin/env python3
"""Per drafts run of ctl_timing.py: tok/s against its off run, and the gate's cells (passes, tokens per pass, ms per token
against the plain baseline, gated steps, backoffs): which source and bucket drafted and whether it paid."""
import glob, json, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "drafts", "ctl_runs")
for f in sorted(glob.glob(os.path.join(R, "*_on_r[0-9].json"))):
    b = json.load(open(f))
    off = f.replace("_on_", "_off_")
    a = json.load(open(off)) if os.path.exists(off) else None
    d = b["draft"]
    g = d.get("gate", {})
    print(f"{os.path.basename(f)[:-5]}: {b['tok_s']:.2f} vs off {a['tok_s'] if a else float('nan'):.2f}; tokens/step {d['tokens_per_step']:.2f}, verifies {d['verifies']}, "
          f"plain {g.get('plain_ms', 0):.1f} ms ({g.get('plain_samples')} samples, {g.get('plain_probes')} probes)")
    for c in g.get("cells", []):
        print(f"   {c['source']:8s} b{c['bucket']}: {c['passes']} passes, {c['tokens_per_pass']:.2f} tok/pass, {c['ms_per_token']:.1f} ms/tok, gated {c['gated_steps']}, backoffs {c['backoffs']}")
