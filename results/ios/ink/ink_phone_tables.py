"""Tables for INK-PHONE.md from ink_phone_sim.py output (sim.json [+ sim_exact.json]) and the compute model below.

  engine/.venv/bin/python ink_phone_tables.py sim.json [sim_exact.json] > tables.md

Compute model (PREDICTION): Inkling's bytes per token by matrix class (results/ink_speed/INK-SPEED.md) divided by the
A16's measured rate for the nearest 35B kernel class (results/ios/p0/P0.md section 3.1, best cell), plus the M4's
non-matvec remainder scaled by the same phone/M4 matvec ratio (P0's method, calibrated 1.00 on the Mac). NVFP4 and
MXFP8 kernels were never timed on the A16: NVFP4 is taken at the 4-bit expert shapes' rate, MXFP8 at the 8-bit rate.
"""
import json, sys
from collections import defaultdict

MB = {  # MB / token (INK-SPEED.md; experts = 44 x 8 x 4.489 + 4 x 8 x 8.225)
    "experts_nvfp4": 44 * 8 * 4.489216, "experts_mxfp8": 4 * 8 * 8.224768, "attn_proj": 1024, "o_proj": 361,
    "head": 492, "msg_gates": 170, "router": 28}
PHONE = {"experts_nvfp4": 31.5, "experts_mxfp8": 43.4, "attn_proj": 43.4, "o_proj": 43.5, "head": 45.6,
         "msg_gates": 41.6, "router": 41.6}  # GB/s, A16 best cells (a4gu/a4dn, a8qkv, a8out, a8head, a8rt)
MAC_MATVEC_MS = 21.5 + 11.0 + 4.0 + 5.9 + 0.7 + 2.0  # RESULTS-INKLING.md (a): experts, attn proj, o, head, router, gates
MAC_TOTAL_MS = 50.0
SUSTAIN = 0.62  # P0: the A16's sustained power-limit factor (experts' kernel, 60 s)


def compute_ms():
    mv = sum(MB[k] / PHONE[k] for k in MB)  # MB / (GB/s) = ms
    other = (MAC_TOTAL_MS - MAC_MATVEC_MS) * mv / MAC_MATVEC_MS
    return mv + other, (mv + other) / SUSTAIN, {k: round(MB[k] / PHONE[k], 1) for k in MB}, round(other, 1)


def main():
    rows = []
    for p in sys.argv[1:]:
        rows += json.load(open(p))["rows"]
    burst, sust, parts, other = compute_ms()
    print(f"compute ms/token (prediction): burst {burst:.0f}, sustained {sust:.0f}; per class {parts}, other {other}\n")
    key = lambda r: (r["policy"], r["pools"], r["hold"], r["thr"])
    cells = defaultdict(dict)
    for r in rows:
        cells[key(r)][(r["text"], r["gb"])] = r
    texts = ["pitt", "code", "edit", "agent"]
    gbs = sorted({r["gb"] for r in rows})
    print("## Exact: MB read per token (decode), per text; mean; hit looped / outer (mean)\n")
    print("| policy | pools | hold | GB | " + " | ".join(texts) + " | mean | hit looped / outer |")
    print("|---|---|---|---|" + "---|" * (len(texts) + 2))
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3])):
        if k[3] is not None:
            continue
        for gb in gbs:
            c = [cells[k].get((t, gb)) for t in texts]
            if not any(c):
                continue
            ok = [x for x in c if x and "error" not in x]
            vals = [f"{x['mb_per_token']:.0f}" if x and "error" not in x else ("abort" if x else "-") for x in c]
            mean = sum(x["mb_per_token"] for x in ok) / len(ok) if ok and len(ok) == len([x for x in c if x]) else None
            hl = sum(x["hit_loop"] for x in ok) / len(ok) if ok else 0
            ho = sum(x["hit_out"] for x in ok) / len(ok) if ok else 0
            print(f"| {k[0]} | {k[1]} | {k[2]} | {gb} | " + " | ".join(vals) + f" | {'%.0f' % mean if mean else '-'} | {hl:.3f} / {ho:.3f} |")
    print("\n## Fast (drop, weight fraction <= 0.1 left out when missing), agent text only (the one trace with weights)\n")
    print("| policy | pools | GB | MB/token | Exact MB/token | weight left out per token (of 48) | dropped picks per token |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3])):
        if k[3] is None:
            continue
        for gb in gbs:
            x = cells[k].get(("agent", gb))
            if not x:
                continue
            ex = cells.get((k[0], k[1], "off", None), {}).get(("agent", gb))
            print(f"| {k[0]} | {k[1]} | {gb} | {x['mb_per_token']:.0f} | {ex['mb_per_token']:.0f} | {x['w_left_out_per_token']:.2f} |"
                  f" {x['dropped'] / x['tokens']:.1f} |" if ex else f"| {k[0]} | {k[1]} | {gb} | {x['mb_per_token']:.0f} | - | {x['w_left_out_per_token']:.2f} | {x['dropped'] / x['tokens']:.1f} |")
    print("\n## Predicted tok/s = 1 / (compute + MB / drive), burst / sustained compute\n")
    print("| mode | policy | pools | GB | MB/token (mean) | 0.7 GB/s | 1.1 GB/s |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2], -1 if k[3] is None else k[3])):
        if k[2] != "off":
            continue
        for gb in gbs:
            if k[3] is None:
                c = [cells[k].get((t, gb)) for t in texts]
                c = [x for x in c if x and "error" not in x]
                if len(c) < len(texts):
                    continue
                mb = sum(x["mb_per_token"] for x in c) / len(c)
                mode = "Exact"
            else:
                x = cells[k].get(("agent", gb))
                ex = cells.get((k[0], k[1], "off", None), {}).get(("agent", gb))
                ce = [cells[(k[0], k[1], "off", None)].get((t, gb)) for t in texts]
                ce = [y for y in ce if y and "error" not in y]
                if not x or not ex or len(ce) < len(texts):
                    continue
                mb = x["mb_per_token"] / ex["mb_per_token"] * sum(y["mb_per_token"] for y in ce) / len(ce)  # agent's ratio on the mean
                mode = "Fast (agent ratio)"
            f = lambda d, c_: 1000.0 / (c_ + mb / d)
            print(f"| {mode} | {k[0]} | {k[1]} | {gb} | {mb:.0f} | {f(0.7, burst):.2f} / {f(0.7, sust):.2f} | {f(1.1, burst):.2f} / {f(1.1, sust):.2f} |")


if __name__ == "__main__":
    main()
