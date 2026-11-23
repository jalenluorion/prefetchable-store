#!/usr/bin/env python3
"""The 35B's light draft head in the engine: counters per cell from results/drafts/runs2/q36_lh_* (drafts2 batch b4).
Per run: tokens per step and per pass, drafts kept by source, compute / wait ms and MB read per produced token (whole run,
plain steps, passes), the pass cost ratio (a pass's compute over a plain step's, in the run), the head's own ms per call
and per draft (embedding row, MLP, output layer), and the head's acceptance per draft index against the run's own tokens
(from its dump: every call's drafts; draft j counted where drafts 1..j-1 were right). Prints markdown tables."""
import glob, json, os, struct, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "drafts", "runs2")
PR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "drafts", "prompts")
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "qwen3_5_moe", "code.ids")
D, ND = 2048, 3


def ids(path):
    return [int(x) for x in open(path).read().replace(",", " ").split()]


def accept(name, case):
    dump = os.path.join(R, name + ".lhdump")
    if not os.path.exists(dump):
        return None
    prompt = ids(CODE if case == "code" else os.path.join(PR, f"q36_{case}_ids.txt"))
    seq = prompt + ids(os.path.join(R, name + ".ids"))
    raw = open(dump, "rb").read()
    rec = 4 * (1 + D + 2 + 2 * ND)
    num, den = [0] * ND, [0] * ND
    conf_right, conf_n = 0.0, 0
    for k in range(len(raw) // rec):
        o = k * rec
        pos = struct.unpack_from("<i", raw, o)[0]
        n = struct.unpack_from("<i", raw, o + 4 + 4 * D + 4)[0]
        d = struct.unpack_from(f"<{ND}i", raw, o + 4 + 4 * D + 8)
        for j in range(n):
            t = pos + 1 + j
            if t >= len(seq):
                break
            den[j] += 1
            if d[j] != seq[t]:
                break
            num[j] += 1
    return [round(a / b, 3) if b else None for a, b in zip(num, den)], den


def row(name, case):
    p = os.path.join(R, name + ".json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    m = os.path.join(R, name + ".meta.json")
    g = (json.load(open(m)).get("guard") or {}) if os.path.exists(m) else {}
    d = j.get("draft") or {}
    W = {w["positions"]: w for w in (d.get("model") or {}).get("widths", [])}
    P = W.get(1, {})
    Q = [w for k, w in W.items() if k > 1]
    pt = sum(w["passes"] * w["tokens_per_pass"] for w in Q)
    pp = sum(w["passes"] for w in Q)
    avg = lambda key: sum(w[key] * w["passes"] * w["tokens_per_pass"] for w in Q) / pt if pt else None
    rho = (sum(w["compute_ms_per_token"] * w["passes"] * w["tokens_per_pass"] for w in Q) / pp) / P["compute_ms_per_token"] if pp and P else None
    widths = sum(k * w["passes"] for k, w in W.items() if k > 1) / pp if pp else None
    lt = d.get("light") or {}
    return {"name": name, "tok_s": j.get("tok_s"), "others": g.get("others_cpu_pct_mean"), "tps": d.get("tokens_per_step"),
            "passes": pp, "tokens_per_pass": pt / pp if pp else None, "mean_width": widths,
            "by_source": (d.get("verifies_by_source"), d.get("accepted_by_source")),
            "cmp": j.get("compute_ms_per_token"), "wait": j.get("wait_ms_per_token"), "mb": j.get("mb_read_per_token"), "urg": j.get("mb_read_urgent_per_token"),
            "plain": (P.get("compute_ms_per_token"), P.get("wait_ms_per_token"), P.get("mb_total_per_token")),
            "pass": (avg("compute_ms_per_token"), avg("wait_ms_per_token"), avg("mb_total_per_token")), "rho": rho,
            "head": (lt.get("ms_per_run"), lt.get("drafts_per_run"), lt.get("ms_per_draft")), "lo": (j.get("approx") or {}).get("replaced_per_token"),
            "accept": accept(name, case)}


f = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"
out = ["| case | mode | cache | tokens / step | passes | tokens / pass | width | compute / wait ms per token (run) | MB (urgent) per token | plain steps: compute / wait / MB per token | passes: compute / wait / MB per token | pass cost ratio | head ms per call (drafts) | others' CPU |",
       "|" + "---|" * 14]
acc = ["| case | mode | cache | draft 1 right | draft 2 right given 1 | draft 3 right given 1, 2 | calls |", "|" + "---|" * 7]
cost = ["| run | head ms per call | drafts per call | per draft: embedding row / MLP / output layer (ms) |", "|---|---|---|---|"]
for case in ("prose", "code", "toolrw"):
    for mode in ("exact", "fast"):
        for gb in (6.6, 5.0):
            r = row(f"q36_lh_{case}_{mode}_gb{gb}", case)
            if not r:
                continue
            out.append(f"| {case} | {mode} | {gb} | {f(r['tps'], 2)} | {r['passes']} | {f(r['tokens_per_pass'], 2)} | {f(r['mean_width'], 1)} | {f(r['cmp'])} / {f(r['wait'])} | {f(r['mb'], 0)} ({f(r['urg'], 0)}) | "
                       f"{' / '.join(f(x) for x in r['plain'])} | {' / '.join(f(x) for x in r['pass'])} | {f(r['rho'], 2)} | {f(r['head'][0], 2)} ({f(r['head'][1], 2)}) | {f(r['others'], 0)} |")
            if r["accept"]:
                a, n = r["accept"]
                acc.append(f"| {case} | {mode} | {gb} | {f(a[0], 3)} | {f(a[1], 3)} | {f(a[2], 3)} | {n[0]} |")
            h = r["head"]
            if h[2]:
                cost.append(f"| {case} {mode} {gb} | {f(h[0], 2)} | {f(h[1], 2)} | {f(h[2]['embedding_row'], 3)} / {f(h[2]['mlp'], 3)} / {f(h[2]['output_layer'], 3)} |")
for case in ("prose", "code"):  # the head every step (--draft-light-backoff 0): every position's drafts
    r = row(f"q36_lh_acc_{case}", case)
    if r and r["accept"]:
        a, n = r["accept"]
        acc.append(f"| {case} | exact, head every step | 6.6 | {f(a[0], 3)} | {f(a[1], 3)} | {f(a[2], 3)} | {n[0]} |")
        h = r["head"]
        if h[2]:
            cost.append(f"| {case} exact 6.6, head every step | {f(h[0], 2)} | {f(h[1], 2)} | {f(h[2]['embedding_row'], 3)} / {f(h[2]['mlp'], 3)} / {f(h[2]['output_layer'], 3)} |")
print("\n".join(out) + "\n\n" + "\n".join(acc) + "\n\n" + "\n".join(cost))
