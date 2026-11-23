#!/usr/bin/env python3
"""The 35B, exact, 6.6 GB: prose and the coding answer with drafts off vs on (lookup + the gated light head), by counters:
ms per produced token (compute + wait, the head's own time in it), MB read per token, tokens per step, passes, head calls,
the head's ms per produced token and per draft, the break-even acceptance it ran at, ids. results/drafts/runs2/q36_g_*."""
import json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "drafts", "runs2")
f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
print("| case | drafts | ms per token (compute + wait) | MB (urgent) per token | tokens / step | passes (kept / drafted) | head calls | head ms per token | head ms per draft (MLP / output layer) | first draft break-even | ids | others' CPU |")
print("|" + "---|" * 12)
for case in ("prose", "code"):
    base = None
    for c in ("off", "on"):
        p = os.path.join(R, f"q36_g_{case}_{c}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        m = os.path.join(R, f"q36_g_{case}_{c}.meta.json")
        g = (json.load(open(m)).get("guard") or {}) if os.path.exists(m) else {}
        d = j.get("draft") or {}
        lt = d.get("light") or {}
        ms = j["compute_ms_per_token"] + j["wait_ms_per_token"]
        rel = "" if base is None else f" ({100 * (ms / base - 1):+.1f}%)"
        if base is None:
            base = ms
        same = "-"
        if c == "on":
            a, b = (open(os.path.join(R, f"q36_g_{case}_{x}.ids")).read() for x in ("off", "on"))
            same = "identical" if a == b else "DIFFER"
        tok = max(1, j.get("tokens", 1) - 1)
        hm = lt.get("ms_per_run", 0) * lt.get("runs", 0) / tok if lt else None
        pd = lt.get("ms_per_draft") or {}
        print(f"| {case} | {c} | {f(j['compute_ms_per_token'])} + {f(j['wait_ms_per_token'])} = {f(ms)}{rel} | {f(j['mb_read_per_token'], 0)} ({f(j['mb_read_urgent_per_token'], 0)}) | "
              f"{f(d.get('tokens_per_step'), 2) if d else '1.00'} | {d.get('verifies', 0)} ({d.get('accepted', 0)} / {d.get('drafted', 0)}) | {lt.get('runs', '-')} | {f(hm, 2)} | "
              f"{f(pd.get('mlp'), 3)} / {f(pd.get('output_layer'), 3)} | {f(lt.get('break_even_first'), 2)} | {same} | {f(g.get('others_cpu_pct_mean'), 0)} |")
