"""Part 1 (a): dmeng on make_tiny.py's model (local/q36/_tiny/run/model, the release's attention geometry, random weights)
against the references test_run.py built for it:
  R8   transformers' Qwen3_5MoeForCausalLM in float32 from the store's values, every resident linear int8-rounded by the
       runner's rule (MLX affine 8-bit, group 64, of the fp16-cast weight) == exactly the weights dmeng computes with
  R32  the same with the resident linears unrounded (information: the int8 resident's own cost)
  MLX  mlx_run.py --dtype int8, exact mode (fp16 activations), saved by test_run.py (run_int8_resident_vs_R8.npz)
Teacher-forced over the 40 random token ids of the references; dmeng runs in every activation precision and at several
expert budgets (the budget must not change a single bit: misses are read on the spot, results are combined in a fixed order).

  Q36_TINY=<folder with model/ and the ref_*.pt / .npz references> python tests/qwen3_5_moe/tiny_check.py
  (torch only to read the .pt references; DMENG_BIN, TINY_OUT override the binary and the output folder)
"""
import json, os, subprocess, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TINY = os.environ["Q36_TINY"]
MODEL = os.path.join(TINY, "model")
OUT = os.environ.get("TINY_OUT", os.path.join(REPO, "build", "tests", "qwen3_5_moe"))
os.makedirs(OUT, exist_ok=True)
BIN = os.environ.get("DMENG_BIN", os.path.join(REPO, "build", "dmeng"))

refs = {k: torch.load(os.path.join(TINY, f"ref_{k}.pt"), map_location="cpu") for k in ("R8", "R32")}
ids = [int(t) for t in refs["R8"]["ids"]]
V = json.load(open(os.path.join(MODEL, "config.json")))["text_config"]["vocab_size"]
open(os.path.join(OUT, "ids.txt"), "w").write(",".join(map(str, ids)))
mlx = np.load(os.path.join(TINY, "run_int8_resident_vs_R8.npz"))
mlx_lg = dict(zip(mlx["positions"].tolist(), mlx["logits"]))


def run(tag, *extra):
    lg_path = os.path.join(OUT, f"logits_{tag}.f32")
    cmd = [BIN, "--model", MODEL, "--teacher", os.path.join(OUT, "ids.txt"), "--dump-logits", lg_path, "--threads", "4", "--readers", "2", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f"dmeng failed: {tag}")
    return np.fromfile(lg_path, dtype=np.float32).reshape(len(ids), V)


def vs_ref(lg, ref):
    tid, tlg = ref["all_top_ids"].numpy(), ref["all_top_logits"].numpy()
    d = np.abs(np.take_along_axis(lg, tid, 1) - tlg)
    agree = int((lg.argmax(1) == ref["all_argmax"].numpy()).sum())
    return agree, float(d.max()), float(np.median(d.max(1)))


rows, res = [], {}
budget_full = 8 * 16 * 16384 / 1e9  # every expert of every layer
for prec in ("i16", "f16", "i8"):
    for gb, name in ((budget_full, "all experts"), (0.0005, "31 slots"), (0.00017, "10 slots")):
        tag = f"{prec}_{gb}"
        lg = run(tag, "--prec", prec, "--cache-gb", str(gb))
        res[tag] = lg
        a8, d8, m8 = vs_ref(lg, refs["R8"])
        a32, d32, _ = vs_ref(lg, refs["R32"])
        P = sorted(mlx_lg)
        X = np.stack([lg[p] for p in P])
        Y = np.stack([mlx_lg[p] for p in P])
        rows.append({"prec": prec, "budget": name, "vs_R8_agree": f"{a8}/{len(ids)}", "vs_R8_top5_max": round(d8, 5), "vs_R8_top5_median": round(m8, 5),
                     "vs_R32_agree": f"{a32}/{len(ids)}", "vs_R32_top5_max": round(d32, 4),
                     "vs_MLX_int8_agree": f"{int((X.argmax(1) == Y.argmax(1)).sum())}/{len(P)}", "vs_MLX_int8_max_vocab": round(float(np.abs(X - Y).max()), 4)})
same = {prec: max(float(np.abs(res[f"{prec}_{budget_full}"] - res[f"{prec}_{gb}"]).max()) for gb in (0.0005, 0.00017)) for prec in ("i16", "f16", "i8")}
# MLX itself against R8 (test_run.py's figure, recomputed): the half-precision floor of the reference engine
P = sorted(mlx_lg)
ml = np.stack([mlx_lg[p] for p in P])
mlx_vs_r8 = vs_ref(ml, refs["R8"])
summary = {"rows": rows, "max_abs_diff_between_budgets": same, "mlx_int8_vs_R8": {"agree": mlx_vs_r8[0], "top5_max": round(mlx_vs_r8[1], 4), "top5_median": round(mlx_vs_r8[2], 4)},
           "max_abs_logit_R8": float(np.abs(refs["R8"]["all_top_logits"].numpy()).max())}
json.dump(summary, open(os.path.join(OUT, "tiny_check.json"), "w"), indent=1)
for r in rows:
    print(r)
print("max |diff| between expert budgets (must be 0):", same)
ok = all(v == 0.0 for v in same.values())
print("TINY CHECK PASSED" if ok else "TINY CHECK FAILED")
print("MLX int8 vs R8:", summary["mlx_int8_vs_R8"], "| max |logit| R8 top-5:", round(summary["max_abs_logit_R8"], 3))
sys.exit(0 if ok else 1)
