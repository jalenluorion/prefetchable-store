"""Check 1: dmeng on the tiny random Inkling (INK_TINY: 12 layers, loop 4-7 x 3, NVFP4 0-9 / MXFP8 10-11, 16
experts top-4) against the PR's MLX model (mlx-lm PR 2086, the tiny RELEASE through mlx_lm.utils.load_model), teacher-forced
over test_run.py's 40 random tokens (and its 300-token long case), token by token.

  INK_TINY=<folder with release/ and model/> INK_RES0=<bf16 resident> INK_RES8=<8-bit resident> \
      python tests/inkling2_loop/tiny_check.py     (DMENG_BIN, TINY_OUT override the binary and the output folder;
      DMENG_OLD_FLAGS=1 drives a binary with the research tree's flag names)

Tolerances are test_run.py's: on DIFFERENT code paths in float32 the PR's own 1e-4; the same arithmetic must be exact.
  A. dmeng --prec f32 (fp32 activations, scalar reference kernels), resident as released  vs  PR float32         tol 1e-4
  B. dmeng --prec f32, 8-bit resident (the same codes as mlx_run.py's resident_q8)  vs  PR float32 with those same
     8-bit codes loaded (nn.quantize structure, scales/biases from the cache, cast to float32)                     tol 1e-4
  C. the speed path: dmeng --prec i16, 8-bit resident vs B's reference (I16 activations: reported, the q36 analogue)
  D. dmeng logits identical across expert budgets (every expert held / few slots with evictions and re-reads, with and
     without the loop hold) and with the oracle replaying the routing at leads 1, 2, 4 and token                   exact
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INK = os.environ["INK_TINY"]
REL, MOD = os.path.join(INK, "release"), os.path.join(INK, "model")
BIN = os.environ.get("DMENG_BIN", os.path.join(REPO, "build", "dmeng"))
RES0, RES8 = os.environ["INK_RES0"], os.environ["INK_RES8"]
OUT = os.environ.get("TINY_OUT", os.path.join(REPO, "build", "tests", "inkling2_loop"))
OLD = os.environ.get("DMENG_OLD_FLAGS") == "1"
os.makedirs(OUT, exist_ok=True)
TMP = tempfile.mkdtemp()
from mlx_lm.utils import load_model  # noqa: E402

cfg = json.load(open(os.path.join(MOD, "config.json")))["text_config"]
V = cfg["unpadded_vocab_size"]
QPATHS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.r_proj", "self_attn.o_proj", "self_attn.gate_proj")
lines, fails, res = [], [], {}


def say(s):
    print(s, flush=True)
    lines.append(s)


def pr_model(q8=False):
    m, _ = load_model(Path(REL))
    m.set_dtype(mx.float32)
    if q8:  # the 8-bit codes of mlx_run.py's cache, computed in float32
        nn.quantize(m, group_size=64, bits=8, class_predicate=lambda p, x: isinstance(x, nn.Linear) and (p == "lm_head" or (p.startswith("model.language_model.") and p.endswith(QPATHS))))
        w = mx.load(os.path.join(MOD, "resident_q8.safetensors"))
        w = {k: (v.astype(mx.float32) if k.endswith((".scales", ".biases")) else v) for k, v in w.items() if k.endswith((".scales", ".biases", ".weight")) and (k.startswith("lm_head") or any(k.startswith(f"model.language_model.layers.{L}.{q}.") for L in range(cfg["num_hidden_layers"]) for q in QPATHS))}
        m.load_weights(list(w.items()), strict=False)
    mx.eval(m.parameters())
    return m


def pr_logits(m, ids):
    c, out = m.make_cache(), []
    tok = mx.array([ids])
    for p in range(len(ids)):
        out.append(np.array(m(tok[:, p : p + 1], cache=c)[0, -1]))
    return np.stack(out)


def new_flags(flags):
    """the merged binary's names: --mx-frac F -> --pool-share 1-F,F; --ns 1 -> --mode approx; --ns-late -> --late; ..."""
    out, f, i = [], list(flags), 0
    while i < len(f):
        k = f[i]
        if k == "--mx-frac":
            x = float(f[i + 1]); out += ["--pool-share", f"{1 - x},{x}"]; i += 2
        elif k == "--ns":
            out += ["--mode", "approx", "--late", "always", "--read-w", "2", "--read-mass", "2", "--bg", "1", "--rd-prio", "1", "--cancel", "1"] if f[i + 1] != "0" else []; i += 2
        elif k in ("--ns-late", "--ns-read-w", "--ns-read-mass", "--ns-bg"):
            out += [{"--ns-late": "--late", "--ns-read-w": "--read-w", "--ns-read-mass": "--read-mass", "--ns-bg": "--bg"}[k], f[i + 1]]; i += 2
        else:
            out.append(k); i += 1
    return out


def dmeng(ids, *flags, name="run"):
    flags = flags if OLD else new_flags(flags)
    ip, lp = os.path.join(TMP, "ids.txt"), os.path.join(TMP, name + ".f32")
    open(ip, "w").write(",".join(map(str, ids)))
    cmd = [BIN, "--model", MOD, "--teacher", ip, "--dump-logits", lp, "--ctx", "1024", "--threads", "2", "--readers", "2", "--mem", "plain", "--res-mem", "plain", *flags]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"dmeng failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return np.fromfile(lp, dtype=np.float32).reshape(len(ids), V), info


def compare(name, got, want, tol):
    d = np.abs(got - want)
    agree = int((got.argmax(1) == want.argmax(1)).sum())
    ok = tol is None or float(d.max()) <= tol
    res[name] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "argmax_agree": agree, "positions": len(got), "tol": tol, "pass": ok}
    say(f"{'PASS' if ok else ('FAIL' if tol is not None else 'INFO')}  {name:78s} max|diff| {d.max():.2e} mean {d.mean():.1e} (tol {tol})  argmax {agree}/{len(got)}  max|logit| {np.abs(want).max():.2f}")
    res[name]["per_position_max"] = [round(float(x), 6) for x in d.max(1)]
    if tol is None:
        say("      per position: " + " ".join(f"{x:.1e}" for x in d.max(1)[:48]))
    if not ok:
        fails.append(name)
        say("      per position: " + " ".join(f"{x:.1e}" for x in d.max(1)[:48]))


rng = np.random.default_rng(0)
ids40 = rng.integers(0, V, 40).tolist()
ids300 = rng.integers(0, V, 300).tolist()
ALL = ["--cache-gb", "0.01", "--mx-frac", "0.2"]           # every expert held (160 nvfp4 + 32 mxfp8 records)
FEW = ["--cache-gb", "0.00105", "--mx-frac", "0.375"]      # 40 nvfp4 + 12 mxfp8 slots: evictions and re-reads
FEWNH = ["--cache-gb", "0.0006", "--mx-frac", "0.55", "--no-loop-hold"]  # 16 nvfp4 + 10 mxfp8, no loop hold

say("== A. exact reference path (fp32 activations), resident as released, vs the PR in float32")
m32 = pr_model()
ref40 = pr_logits(m32, ids40)
got, _ = dmeng(ids40, "--resident", RES0, "--prec", "f32", *ALL, name="a40")
compare("A f32, bf16 resident, all experts held, 40 tokens", got, ref40, 1e-4)
ref300 = pr_logits(m32, ids300)
got, info = dmeng(ids300, "--resident", RES0, "--prec", "f32", *FEW, name="a300")
compare(f"A f32, bf16 resident, 40/12 slots, 300 tokens (window 16, reads {info['reads']})", got, ref300, 1e-4)
got_i, _ = dmeng(ids40, "--resident", RES0, "--prec", "i16", *ALL, name="a40i")
compare("A' i16 activations, bf16 resident (info)", got_i, ref40, None)
del m32

say("== B. 8-bit resident (mlx_run.py's codes), fp32 activations, vs the PR in float32 with the same codes")
mq = pr_model(q8=True)
refq = pr_logits(mq, ids40)
got_q, _ = dmeng(ids40, "--resident", RES8, "--prec", "f32", *ALL, name="b40")
compare("B f32, 8-bit resident, all held", got_q, refq, 1e-4)
say("== C. the speed path: I16 activations (MXFP8 fp16), 8-bit resident")
got_c, _ = dmeng(ids40, "--resident", RES8, *ALL, name="c40")
compare("C i16, 8-bit resident vs B's reference (info: the activation rounding)", got_c, refq, None)
got_c8, _ = dmeng(ids40, "--resident", RES8, "--prec", "f32", *ALL, name="c40f")
compare("C i16, 8-bit resident vs dmeng f32 8-bit (same weights)", got_c, got_c8, None)
for p in ("i8", "f16"):
    g, _ = dmeng(ids40, "--resident", RES8, "--prec", p, *ALL, name="c40" + p)
    compare(f"C {p}, 8-bit resident vs B's reference (info)", g, refq, None)

say("== D. identical logits across budgets and replay modes (speed path)")
rec = os.path.join(TMP, "rec.bin")
base, _ = dmeng(ids40, "--resident", RES8, *ALL, "--record", rec, name="d0")
for nm, fl in (("40/12 slots", FEW), ("16/10 slots, no loop hold", FEWNH)):
    g, info = dmeng(ids40, "--resident", RES8, *fl, name="d1")
    compare(f"D {nm} (reads {info['reads']}) vs all held", g, base, 0.0)
for k in ("1", "2", "4", "token"):
    g, info = dmeng(ids40, "--resident", RES8, *FEW, "--prefetch", "oracle", "--oracle", rec, "--lead", k, name="d2")
    compare(f"D oracle lead {k}, 40/12 slots (reads {info['reads']}) vs all held", g, base, 0.0)
for nsf in (["--ns", "1", "--ns-late", "always"], ["--ns", "1", "--ns-late", "none", "--ns-read-w", "0.1"], ["--ns", "1", "--ns-late", "0", "--ns-read-mass", "0.1"]):
    g, _ = dmeng(ids40, "--resident", RES8, *ALL, *nsf, name="dns")
    # with every pick's weight > 0.1 (top-4 of 16) the read-w / mass rules guarantee every pick: nothing is replaced and the
    # result must equal lossless bit for bit; "late always" replaces never-named misses even before the cache fills (info)
    exact = "always" not in nsf
    compare(f"D approximate mode {' '.join(nsf[2:])}, budget for every expert" + ("" if exact else " (info: early misses replaced)"), g, base, 0.0 if exact else None)
    g, info = dmeng(ids40, "--resident", RES8, *FEW, *nsf, "--prefetch", "oracle", "--oracle", rec, "--lead", "1", name="dns2")
    compare(f"E approximate mode {' '.join(nsf[2:])}, 40/12 slots, oracle lead 1 (info: replacements happen)", g, base, None)
g, _ = dmeng(ids40, "--resident", RES8, *ALL, "--threads", "5", name="d3")
compare("D 5 compute threads vs 2", g, base, 0.0)

say("TINY CHECK FAILED: " + ", ".join(fails) if fails else "TINY CHECK PASSED")
open(os.path.join(OUT, "tiny_check.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, "tiny_check.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
