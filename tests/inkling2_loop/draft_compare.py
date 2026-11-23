"""Compare the C++ draft head (dmeng --draft-check) with the MLX reference (draft_ref_mlx_run.py) on the same features.

  python draft_compare.py CPP_PREFIX REF_PREFIX [--vocab 200058] [--tie 0.25]

Reports: draft 1 equal (share of positions), draft j equal where drafts 1..j-1 were equal (each side conditions on its own
previous drafts), the largest and mean |logit difference| of draft 1 over the positions with dumped logits, and for every
draft-1 disagreement the reference's own top-2 margin (a margin under --tie is a near tie: MLX rounds the logits to bf16,
spaced ~0.06-0.125 near the top).
"""
import argparse, json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("cpp")
ap.add_argument("ref")
ap.add_argument("--vocab", type=int, default=200058)
ap.add_argument("--tie", type=float, default=0.25)
a = ap.parse_args()


def drafts(prefix):
    r = np.fromfile(prefix + ".drafts", dtype=np.int32).reshape(-1, 15)
    return r[:, 0], r[:, 1:8], r[:, 8:15].copy().view(np.float32)


pc, dc, prc = drafts(a.cpp)
pr_, dr, prr = drafts(a.ref)
n = min(len(pc), len(pr_))
assert (pc[:n] == pr_[:n]).all(), "position lists differ"
dc, dr, prc, prr = dc[:n], dr[:n], prc[:n], prr[:n]
res = {"positions": int(n), "draft1_equal": float((dc[:, 0] == dr[:, 0]).mean())}
eq = np.ones(n, bool)
cond = []
for j in range(7):
    m = eq & (dc[:, j] == dr[:, j])
    cond.append({"j": j + 1, "compared": int(eq.sum()), "equal": float(m.sum() / max(1, eq.sum()))})
    eq = m
res["draft_j_equal_given_previous_equal"] = cond
res["all7_equal"] = float(eq.mean())
same1 = dc[:, 0] == dr[:, 0]
res["prob1_max_abs_diff_where_equal"] = float(np.abs(prc[same1, 0] - prr[same1, 0]).max()) if same1.any() else None
lc = np.fromfile(a.cpp + ".logits", dtype=np.float32).reshape(-1, a.vocab)
lr = np.fromfile(a.ref + ".logits", dtype=np.float32).reshape(-1, a.vocab)
k = min(len(lc), len(lr))
d = np.abs(lc[:k] - lr[:k])
res["logit_positions"] = int(k)
res["draft1_logit_max_abs_diff"] = float(d.max())
res["draft1_logit_mean_abs_diff"] = float(d.mean())
res["draft1_logit_max_abs_diff_at_ref_top"] = float(max(abs(lc[i, int(np.argmax(lr[i]))] - lr[i].max()) for i in range(k)))
mis = []
for i in np.nonzero(~same1)[0]:
    if i < k:
        top = np.sort(lr[i])[-2:]
        mis.append({"p": int(pc[i]), "cpp": int(dc[i, 0]), "ref": int(dr[i, 0]), "ref_margin": float(top[1] - top[0]),
                    "cpp_logit_of_ref": float(lc[i, dr[i, 0]]), "cpp_logit_of_cpp": float(lc[i, dc[i, 0]])})
    else:
        mis.append({"p": int(pc[i]), "cpp": int(dc[i, 0]), "ref": int(dr[i, 0])})
res["draft1_disagreements"] = mis
res["draft1_disagreements_near_tie"] = sum(1 for m in mis if "ref_margin" in m and m["ref_margin"] < a.tie)
print(json.dumps(res, indent=1))
