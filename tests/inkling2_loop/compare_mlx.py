"""Optional check on the REAL model: dmeng against an independent Python/MLX runner (an Inkling mlx_run.py with --mode
exact: the released routing, misses read on the spot, 8-bit resident attention + head from the same codes, bf16
activations; not part of this repo), teacher-forced over the same N tokens, position by position.

  python tests/inkling2_loop/compare_mlx.py [all|prepare|mlx|compare] --model MODEL --mlx-run PATH/mlx_run.py [--mlx-python PY]
  (env instead of flags: DM_MODEL, MLX_RUN, MLX_PY; DMENG_BIN another binary; CMP_OUT the output folder, default
  build/tests/inkling2_loop/vs_mlx; N positions, default 60; EXTRA more dmeng flags for the teacher-forced run, TAG a
  suffix for its files; WRAP a command prefix for every process that loads the model, e.g. a guard that waits for a quiet
  machine; TIE the near-tie margin, default 0.5: a logit near 30 x tanh(.) in bf16 is spaced ~0.125, and MLX rounds the
  head's output to bf16)

  prepare  dmeng: the Pittsburgh prompt + dmeng's own greedy continuation, cut to N ids; then teacher-forced logits over
           them (--teacher, --dump-logits)
  mlx      the MLX runner on the same ids (--check-ids SPEC --check-save MLX.npz, token by token)
  compare  top-1 agreement, max / mean |logit diff|; PASSED when every top-1 disagreement is a near tie (MLX's own top-2
           margin < TIE)
Skips (exit 0, a "skipped: ..." line) when the model folder or the MLX runner is absent. Needs numpy (and the tokenizer's
`tokenizers`, through python/run.py).
"""
import argparse, json, os, shlex, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FAMILY = "inkling2_loop"

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("step", nargs="?", default="all", choices=["all", "prepare", "mlx", "compare"])
ap.add_argument("--model", default=os.environ.get("DM_MODEL"))
ap.add_argument("--mlx-run", default=os.environ.get("MLX_RUN"))
ap.add_argument("--mlx-python", default=os.environ.get("MLX_PY", sys.executable))
a = ap.parse_args()
BIN = os.environ.get("DMENG_BIN", os.path.join(REPO, "build", "dmeng"))
OUT = os.environ.get("CMP_OUT", os.path.join(REPO, "build", "tests", FAMILY, "vs_mlx"))
N = int(os.environ.get("N", 60))
TAG = os.environ.get("TAG", "")
EXTRA = shlex.split(os.environ.get("EXTRA", ""))
WRAP = shlex.split(os.environ.get("WRAP", ""))
TIE = float(os.environ.get("TIE", 0.5))
DM = ["--cache-gb", "6"]


def skip(why):
    print(f"skipped: {why}")
    raise SystemExit(0)


steps = ["prepare", "mlx", "compare"] if a.step == "all" else [a.step]
if "compare" not in steps or len(steps) > 1:
    man = os.path.join(a.model or "", "moved-store.json")
    if not a.model or not os.path.exists(man):
        skip(f"no model folder with moved-store.json ({a.model or 'set --model or DM_MODEL'})")
    if json.load(open(man)).get("family") != FAMILY:
        skip(f"{a.model} is not an {FAMILY} model")
if "mlx" in steps and (not a.mlx_run or not os.path.exists(a.mlx_run)):
    skip(f"no MLX runner ({a.mlx_run or 'set --mlx-run or MLX_RUN'})")
if "prepare" in steps and not os.path.exists(BIN):
    raise SystemExit(f"no dmeng binary at {BIN} (build first, or set DMENG_BIN)")
os.makedirs(OUT, exist_ok=True)
ids_path = os.path.join(OUT, "ids.txt")
lg_path = os.path.join(OUT, f"dmeng_logits{TAG}.f32")
mlx_path = os.path.join(OUT, "mlx_logits.npz")


def run(cmd, log):
    r = subprocess.run(WRAP + cmd, capture_output=True, text=True)
    open(os.path.join(OUT, log), "w").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"{os.path.basename(cmd[0])} exited {r.returncode} (log: {os.path.join(OUT, log)})")
    return r.stdout


def ids():
    return [int(t) for t in open(ids_path).read().split(",")]


if "prepare" in steps:
    if not os.path.exists(ids_path):
        r = subprocess.run([sys.executable, os.path.join(REPO, "python", "run.py"), "--model", a.model, "--prompt-ids-only"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("python/run.py --prompt-ids-only failed:\n" + r.stderr)
        prompt = [int(t) for t in r.stdout.strip().split(",")]
        gen_path = os.path.join(OUT, "gen.ids")
        run([BIN, "--model", a.model, "--prompt-ids", ",".join(map(str, prompt)), "--n", str(N - len(prompt) + 1), *DM, "--out-ids", gen_path], "gen.log")
        gen = [int(t) for t in open(gen_path).read().strip().split(",")]
        open(ids_path, "w").write(",".join(map(str, (prompt + gen)[:N])))
    print(run([BIN, "--model", a.model, "--teacher", ids_path, "--dump-logits", lg_path, *DM, *EXTRA], f"teacher{TAG}.log")[-400:])
    print("prepared", len(ids()), "positions in", OUT)

if "mlx" in steps:
    spec = os.path.join(OUT, "check_ids.json")
    json.dump({"ids": ids(), "batch": 0, "step": 0}, open(spec, "w"))
    print(run([a.mlx_python, a.mlx_run, "--dir", a.model, "--mode", "exact", "--core-slots", "64", "--outer-slots", "8", "--check-ids", spec,
               "--check-save", mlx_path], "mlx_check.log")[-600:])

if "compare" in steps:
    for p in (ids_path, lg_path, mlx_path):
        if not os.path.exists(p):
            skip(f"{p} is missing (run prepare and mlx first)")
    n = len(ids())
    lg = np.fromfile(lg_path, dtype=np.float32).reshape(n, -1)
    V = lg.shape[1]  # the unpadded vocabulary (dmeng's vocab_out); MLX's logits are padded
    z = np.load(mlx_path)
    P = z["positions"].tolist()
    X, Y = lg[P], z["logits"].astype(np.float32)[:, :V]
    d = np.abs(X - Y)
    agree = X.argmax(1) == Y.argmax(1)
    ys = np.sort(Y, axis=1)
    margin = ys[:, -1] - ys[:, -2]  # where the two disagree: a near tie, or a real difference
    top5 = np.argsort(-Y, axis=1)[:, :5]
    d5 = np.abs(np.take_along_axis(X, top5, 1) - np.take_along_axis(Y, top5, 1))
    dis = [{"pos": int(p), "dmeng": int(X[i].argmax()), "mlx": int(Y[i].argmax()), "mlx_top2_margin": round(float(margin[i]), 4),
            "dmeng_margin_between_the_two": round(float(X[i].max() - X[i][Y[i].argmax()]), 4)} for i, p in enumerate(P) if not agree[i]]
    res = {"positions": len(P), "top1_agree": int(agree.sum()), "max_abs_logit_diff_vocab": float(d.max()), "mean_abs_logit_diff_vocab": float(d.mean()),
           "median_over_positions_of_max_diff": float(np.median(d.max(1))), "max_abs_diff_on_mlx_top5": float(d5.max()),
           "mean_abs_diff_on_mlx_top5": float(d5.mean()), "logit_scale_max_abs": float(np.abs(Y).max()), "logit_std_mean": float(Y.std(1).mean()),
           "tie": TIE, "disagreements": dis, "per_position_max_diff": [round(float(v), 3) for v in d.max(1)]}
    json.dump(res, open(os.path.join(OUT, f"compare{TAG}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_position_max_diff"}, indent=1))
    ok = all(x["mlx_top2_margin"] < TIE for x in dis)
    print(f"COMPARE MLX ({FAMILY})", "PASSED" if ok else "FAILED", f"top-1 {res['top1_agree']}/{res['positions']}")
    raise SystemExit(0 if ok else 1)
