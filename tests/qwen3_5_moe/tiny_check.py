"""dmeng on the tiny random Qwen3.6-35B-A3B (tests/qwen3_5_moe/make_tiny.py) against an independent reference:
transformers' own PyTorch model of the release (Qwen3_5MoeForCausalLM, float32, CPU), given exactly the weights the engine
computes with: the store's 4-bit expert records dequantised (code x scale + bias) and every in-memory matrix the engine
holds at 8 bits as its codes dequantised (read from the engine's own derived resident file); every other weight as
released (bf16 widened). Teacher-forced over random tokens, 40 and 300.

  TINY=<OUT of make_tiny.py> python tests/qwen3_5_moe/tiny_check.py      (python: numpy, torch, transformers)
  (DMENG_BIN overrides the binary, TINY_OUT the output folder)

  A. the exact reference path, dmeng --prec f32 (fp32 activations, fp32 K / V, scalar kernels) vs the reference:
     max |logit diff| <= 1e-4 x max |logit|, every argmax equal
  B. the speed path (I16 activations, the default; F16 and I8 too) vs the reference: argmax agreement and |diff| (info)
"""
import json, os, subprocess, sys, tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TINY = os.environ.get("TINY") or os.environ["Q36_TINY"]
REL, MOD = os.path.join(TINY, "release"), os.path.join(TINY, "model")
BIN = os.environ.get("DMENG_BIN", os.path.join(REPO, "build", "dmeng"))
OUT = os.environ.get("TINY_OUT", os.path.join(REPO, "build", "tests", "qwen3_5_moe"))
os.makedirs(OUT, exist_ok=True)
TMP = tempfile.mkdtemp()
cfg_full = json.load(open(os.path.join(MOD, "config.json")))
cfg = cfg_full["text_config"]
V, NL, NE = cfg["vocab_size"], cfg["num_hidden_layers"], cfg["num_experts"]
lines, fails, res = [], [], {}


def say(s):
    print(s, flush=True)
    lines.append(s)


def dmeng(ids, *flags, name="run"):
    ip, lp = os.path.join(TMP, name + ".ids"), os.path.join(TMP, name + ".f32")
    open(ip, "w").write(",".join(map(str, ids)))
    cmd = [BIN, "--model", MOD, "--profile", "none", "--teacher", ip, "--dump-logits", lp, "--threads", "4", "--readers", "2", "--mem", "plain",
           "--res-mem", "plain", "--cache-gb", "0.05", *flags]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return None, (r.stdout + r.stderr).strip()
    return np.fromfile(lp, dtype=np.float32).reshape(len(ids), V), json.loads(r.stdout.strip().splitlines()[-1])


def compare(name, got, want, tol, rel=False):
    d = np.abs(got - want)
    lim = tol * (float(np.abs(want).max()) if rel else 1.0) if tol is not None else None
    agree = int((got.argmax(1) == want.argmax(1)).sum())
    ok = lim is None or (float(d.max()) <= lim and agree == len(got))
    res[name] = {"max_abs_diff": float(d.max()), "argmax_agree": agree, "positions": len(got), "limit": lim, "pass": ok}
    tag = "PASS" if ok and lim is not None else ("INFO" if lim is None else "FAIL")
    say(f"{tag}  {name:70s} max|diff| {d.max():.2e} (limit {'-' if lim is None else f'{lim:.1e}'})  argmax {agree}/{len(got)}  max|logit| {np.abs(want).max():.2f}")
    if not ok:
        fails.append(name)


rng = np.random.default_rng(1)
ids40 = [int(t) for t in rng.integers(0, V, 40)]
ids300 = [int(t) for t in rng.integers(0, V, 300)]
base40, info = dmeng(ids40, name="base40")
if base40 is None:
    raise SystemExit(f"dmeng failed: {info}")
resf = sorted(f for f in os.listdir(os.path.join(MOD, ".dmeng")) if f.startswith("resident_q8.") and f.endswith(".bin"))[-1]
resf = os.path.join(MOD, ".dmeng", resf)


def dmres(path):
    with open(path, "rb") as f:
        first = f.readline().decode().split()
        hdr = int(first[1])
        head = f.read(hdr - f.tell()).decode().splitlines()
    tab = {}
    for line in head:
        p = line.split()
        if p and p[0] == "t":
            tab[p[1]] = (p[2], int(p[3]), int(p[4]), [int(x) for x in p[6:6 + int(p[5])]])
    return hdr, tab


def deq(codes, s, b, bits):
    R = codes.shape[0]
    if bits == 4:
        c = np.empty((R, codes.shape[1] * 2), dtype=np.uint8)
        c[:, 0::2], c[:, 1::2] = codes & 15, codes >> 4
    else:
        c = codes
    C = c.shape[1]
    w = c.reshape(R, C // 64, 64).astype(np.float32) * s.astype(np.float32)[:, :, None] + b.astype(np.float32)[:, :, None]
    return w.reshape(R, C)


hdr, tab = dmres(resf)
raw = np.fromfile(resf, dtype=np.uint8)


def tensor(n):
    dt, off, nb, shp = tab[n]
    a = raw[hdr + off: hdr + off + nb]
    return a.view({"u8": np.uint8, "f16": np.float16, "f32": np.float32, "bf16": np.uint16}[dt]).reshape(shp)


def hf_name(n):
    if n == "head":
        return "lm_head.weight"
    L, k = n.split(".", 1)
    lin = cfg["layer_types"][int(L)] == "linear_attention"
    m = {"qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "o": "linear_attn.out_proj" if lin else "self_attn.o_proj",
         "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "sg": "mlp.shared_expert.gate_proj",
         "su": "mlp.shared_expert.up_proj", "sd": "mlp.shared_expert.down_proj"}[k]
    return f"model.layers.{L}.{m}.weight"


def reference(q8):
    import warnings

    warnings.filterwarnings("ignore")
    from safetensors.torch import load_file
    from transformers import Qwen3_5MoeForCausalLM
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    m = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**cfg)).float()
    sd = m.state_dict()
    rel = {}
    for f in sorted(os.listdir(REL)):
        if f.endswith(".safetensors"):
            rel.update(load_file(os.path.join(REL, f)))
    with torch.no_grad():
        n = 0
        for k, v in rel.items():  # the release's weights (bf16 widened); the language model's names without the prefix
            kk = k.replace("model.language_model.", "model.")
            if kk in sd:
                sd[kk].copy_(v.float())
                n += 1
        assert n == len(sd), (n, len(sd))
        meta = json.load(open(os.path.join(MOD, "store", "meta.json")))
        REC = meta["record_bytes"]
        for L in range(NL):  # the store's expert records
            recs = np.fromfile(os.path.join(MOD, "store", f"layer_{L:02d}.bin"), dtype=np.uint8).reshape(NE, REC)
            mats = {}
            for kind in ("gate_proj", "up_proj", "down_proj"):
                r, c = meta["shapes"][kind]
                o, so, bo = meta["offsets"][kind], meta["scale_offsets"][kind], meta["bias_offsets"][kind]
                mats[kind] = np.stack([deq(recs[e, o: o + r * c // 2].reshape(r, c // 2), recs[e, so: so + r * c // 32].view(np.float16).reshape(r, c // 64),
                                           recs[e, bo: bo + r * c // 32].view(np.float16).reshape(r, c // 64), 4) for e in range(NE)])
            sd[f"model.layers.{L}.mlp.experts.gate_up_proj"].copy_(torch.from_numpy(np.concatenate([mats["gate_proj"], mats["up_proj"]], axis=1)))
            sd[f"model.layers.{L}.mlp.experts.down_proj"].copy_(torch.from_numpy(mats["down_proj"]))
        n8 = 0
        if q8:  # the engine's 8-bit matrices, dequantised
            for t in tab:
                if t.endswith(".q"):
                    base = t[:-2]
                    sd[hf_name(base)].copy_(torch.from_numpy(deq(tensor(t), tensor(base + ".s"), tensor(base + ".b"), 8)))
                    n8 += 1
    m.eval()
    return m, n8


def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()


say("== A. the exact reference path vs transformers' model with the engine's weights")
m8, n8 = reference(True)
say(f"      reference: transformers Qwen3_5MoeForCausalLM, float32; {n8} matrices at the engine's 8-bit codes, experts from the store")
r40, r300 = ref_logits(m8, ids40), ref_logits(m8, ids300)
del m8
f40, info = dmeng(ids40, "--prec", "f32", name="f40")
if f40 is None and "no fp32 reference path" in info:
    say("NOT SUPPORTED the exact fp32 reference path (dmeng --prec f32): " + info.splitlines()[-1])
elif f40 is None:
    say("FAIL  A dmeng --prec f32: " + info[-300:])
    fails.append("A f32")
else:
    compare("A f32 reference path, 40 tokens", f40, r40, 1e-4, rel=True)
    f300, _ = dmeng(ids300, "--prec", "f32", name="f300")
    compare("A f32 reference path, 300 tokens", f300, r300, 1e-4, rel=True)
say("== B. the speed path vs the reference (info)")
compare("B i16 (default), 40 tokens", base40, r40, None)
i300, _ = dmeng(ids300, name="i300")
compare("B i16, 300 tokens", i300, r300, None)
for p in ("f16", "i8"):
    g, _ = dmeng(ids40, "--prec", p, name="b" + p)
    compare(f"B {p}, 40 tokens", g, r40, None)
m32, _ = reference(False)
compare("B (info) the 8-bit resident's own cost: reference with 8-bit vs bf16 matrices, 40 tokens", r40, ref_logits(m32, ids40), None)
del m32

say("TINY CHECK FAILED: " + ", ".join(fails) if fails else "TINY CHECK PASSED")
open(os.path.join(OUT, "tiny_check.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, "tiny_check.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
