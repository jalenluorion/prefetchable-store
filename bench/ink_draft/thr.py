"""Choosing the draft head's probability threshold from the port check's records (dmeng --draft-check on greedy
sequences: the ids after the prompt are the model's own greedy tokens, so draft j at p is right when it equals ids[p+1+j]).

  python3 bench/ink_draft/thr.py PREFIX IDS START [--head-ms 25 --step-ms 144]

Per threshold: share of positions with a draft, mean drafts used, mean accepted, and the tokens per unit of time of the
rule "verify every draft whose probability is >= thr" with the measured step costs (one position = 1; 2..8 positions =
the 6 GB table of the engine's verify, RESULTS-STAGE4-INKLING.md) and the head's cost added on every step.
"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("prefix")
ap.add_argument("ids")
ap.add_argument("start", type=int)
ap.add_argument("--head-ms", type=float, default=25)
ap.add_argument("--step-ms", type=float, default=144)
a = ap.parse_args()
R = [0, 1.0, 1.64, 2.07, 2.50, 2.95, 3.41, 3.86, 4.32]
ids = [int(t) for t in open(a.ids).read().split(",")]
r = np.fromfile(a.prefix + ".drafts", dtype=np.int32).reshape(-1, 15)
P, D, PR = r[:, 0], r[:, 1:8], r[:, 8:15].copy().view(np.float32)
# acceptance by probability bin (draft j kept when all before it were kept)
bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995, 1.01]
acc = {}
for i, p in enumerate(P):
    for j in range(7):
        if p + 2 + j >= len(ids):
            break
        ok = D[i, j] == ids[p + 2 + j]
        b = np.searchsorted(bins, PR[i, j], side="right") - 1
        acc.setdefault(b, []).append(ok)
        if not ok:
            break
print("| probability bin | drafts | kept |")
print("|---|---|---|")
for b in sorted(acc):
    print(f"| {bins[b]:.3f}-{bins[b + 1]:.3f} | {len(acc[b])} | {np.mean(acc[b]):.3f} |")
print()
print("| thr | share with a draft | drafts / step | kept / step | tokens per one-position step time |")
print("|---|---|---|---|---|")
for thr in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    tok = tim = nd_tot = kept_tot = with_d = 0
    for i, p in enumerate(P):
        nd = 0
        while nd < 7 and PR[i, nd] >= thr and p + 2 + nd < len(ids):
            nd += 1
        k = 0
        while k < nd and D[i, k] == ids[p + 2 + k]:
            k += 1
        with_d += nd > 0
        nd_tot += nd
        kept_tot += k
        tok += 1 + k
        tim += R[1 + nd] + a.head_ms / a.step_ms
    n = len(P)
    print(f"| {thr} | {with_d / n:.2f} | {nd_tot / n:.2f} | {kept_tot / n:.2f} | {tok / tim:.3f} |")
