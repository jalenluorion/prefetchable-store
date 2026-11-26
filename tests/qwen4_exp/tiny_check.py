"""dmeng on the tiny random Qwen3.8-Flash-Next (tests/qwen4_exp/make_tiny.py) against an independent reference:
transformers' own PyTorch model of the release (Qwen4ExpForCausalLM, float32, CPU), given exactly the weights the engine
computes with: the store's 4-bit expert records dequantised (code x scale + bias), the n-gram rows as stored (int8 code x
fp16 scale), and every in-memory matrix the engine holds at 8 bits as its codes dequantised (read from the engine's own
derived resident file); every other weight as released (bf16 widened). Teacher-forced over random tokens with the n-gram
EOS inside (the hash's EOS rule), 40 tokens (the QSA budget of 16 crossed: the indexer's selection runs) and 300.

  TINY=<OUT of make_tiny.py> python tests/qwen4_exp/tiny_check.py      (python: numpy, torch, transformers)
  (DMENG_BIN, DMSERVE_BIN override the binaries, TINY_OUT the output folder)

  A. the exact reference path, dmeng --prec f32 (fp32 activations, fp32 K / V / indexer keys, scalar kernels) vs the
     reference: max |logit diff| <= 1e-4 x max |logit|, every argmax equal
  B. the speed path (I16 activations, the default; F16 and I8 too) vs the reference: argmax agreement and |diff| (info)
  C. exact mode is exact: the I16 logits identical, bit for bit, at every expert budget (all held; slots for a few
     steps, with evictions and re-reads), with 1 or 5 compute threads, with the trained routers naming (mlp_k4, lag 4,
     derived in place from routers_k4.npz), with run-ahead, with the oracle replaying the routing at leads 1, 2, 4 and one
     token, with the frequency eviction instead of the keep-score, and in fast mode while every expert is held
  D. fast mode (approx:drop) runs at a budget that drops picks: argmax agreement with exact (info)
  E. the batched prefill: dmserve's prefill check (1,100 tokens: many sub-chunks, the sparse attention far past its
     budget, state captured mid-span) equal to single steps bit for bit (logits, fixed state, KV, greedy continuation),
     on the CPU and with the GPU's share of the projections and the experts; dmeng's generated ids equal with the
     batched and the token-by-token prompt
  F. the prefix cache's checkpoints (dmserve --check-ckpt) and a state saved and restored into a fresh one (dmeng
     --state-check), bit for bit / the same ids
"""
import glob, json, os, subprocess, sys, tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TINY = os.environ.get("TINY") or os.environ["FLASH_TINY"]
REL, MOD = os.path.join(TINY, "release"), os.path.join(TINY, "model")
BIN = os.environ.get("DMENG_BIN", os.path.join(REPO, "build", "dmeng"))
SRV = os.environ.get("DMSERVE_BIN", os.path.join(REPO, "build", "dmserve"))
OUT = os.environ.get("TINY_OUT", os.path.join(REPO, "build", "tests", "qwen4_exp"))
os.makedirs(OUT, exist_ok=True)
TMP = tempfile.mkdtemp()
cfg = json.load(open(os.path.join(MOD, "config.json")))["text_config"]
V, NL, NE, K = cfg["vocab_size"], cfg["num_hidden_layers"], cfg["num_experts"], cfg["num_experts_per_tok"]
lines, fails, res = [], [], {}


def say(s):
    print(s, flush=True)
    lines.append(s)


def dmeng(ids, *flags, name="run", n_out=None):
    ip, lp = os.path.join(TMP, name + ".ids"), os.path.join(TMP, name + ".f32")
    open(ip, "w").write(",".join(map(str, ids)))
    cmd = [BIN, "--model", MOD, "--profile", "none", "--teacher", ip, "--dump-logits", lp, "--threads", "4", "--readers", "2", "--mem", "plain",
           "--res-mem", "plain", "--cache-gb", "0.05", *flags]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"dmeng failed: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return np.fromfile(lp, dtype=np.float32).reshape(len(ids), V), json.loads(r.stdout.strip().splitlines()[-1])


def compare(name, got, want, tol, rel=False):
    d = np.abs(got - want)
    lim = tol * (float(np.abs(want).max()) if rel else 1.0) if tol is not None else None
    agree = int((got.argmax(1) == want.argmax(1)).sum())
    ok = lim is None or (float(d.max()) <= lim and agree == len(got))
    res[name] = {"max_abs_diff": float(d.max()), "argmax_agree": agree, "positions": len(got), "limit": lim, "pass": ok}
    tag = "PASS" if ok and lim is not None else ("INFO" if lim is None else "FAIL")
    say(f"{tag}  {name:86s} max|diff| {d.max():.2e} (limit {'-' if lim is None else f'{lim:.1e}'})  argmax {agree}/{len(got)}  max|logit| {np.abs(want).max():.2f}")
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------------------------------------------- the reference
rng = np.random.default_rng(1)
EOS = cfg["eos_token_id"] if not isinstance(cfg["eos_token_id"], list) else cfg["eos_token_id"][0]
ids40 = [int(t) for t in rng.integers(10, V, 40)]
ids40[13] = ids40[27] = EOS
ids300 = [int(t) for t in rng.integers(10, V, 300)]
for p in (5, 6, 100, 201):
    ids300[p] = EOS
base40, info = dmeng(ids40, name="base40")  # (also derives the engine's resident file on first use)
resf = sorted(glob.glob(os.path.join(MOD, ".dmeng", "resident_q8.dmres1.*.bin")))[-1]


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
    return a.view({"u8": np.uint8, "f16": np.float16, "f32": np.float32, "bf16": np.uint16, "i64": np.int64}[dt]).reshape(shp)


def hf_name(n):
    if n == "head":
        return "lm_head.weight"
    if n.startswith("fhc_"):
        return "model.hyper_connection_mixer." + {"fhc_d": "input_mix_weight_down", "fhc_u": "input_mix_weight_up"}[n] + ".weight"
    L, k = n.split(".", 1)
    lin = cfg["layer_types"][int(L)] == "linear_attention"
    m = {"ahc_d": "attn_hyper_connection.input_mix_weight_down", "ahc_u": "attn_hyper_connection.input_mix_weight_up",
         "ahc_i": "attn_hyper_connection.block_inject_weight", "mhc_d": "mlp_hyper_connection.input_mix_weight_down",
         "mhc_u": "mlp_hyper_connection.input_mix_weight_up", "mhc_i": "mlp_hyper_connection.block_inject_weight",
         "qkv": "linear_attn.in_proj_qkv", "z": "linear_attn.in_proj_z", "a": "linear_attn.in_proj_a", "b": "linear_attn.in_proj_b",
         "o": "linear_attn.out_proj" if lin else "self_attn.o_proj", "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
         "ix": "self_attn.indexer.index_qk_proj", "sg": "mlp.shared_expert.gate_proj", "su": "mlp.shared_expert.up_proj",
         "sd": "mlp.shared_expert.down_proj", "shgate": "mlp.shared_expert_gate", "router": "mlp.gate", "pk": "ple.key_proj", "pv": "ple.value_proj"}[k]
    return f"model.layers.{L}.{m}.weight"


TIES = [0]


def stable_indexer():
    """The indexer's block choice with equal scores broken toward the earlier block, the engine's rule (blocks whose heads
    all score below zero tie at 0 after the relu; the release leaves the order of equal scores unspecified, and torch's
    topk takes the later one): torch.topk inside the indexer only, as a stable descending sort. Ties at the cut counted."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as Q

    fwd = Q.Qwen4ExpTextQSAIndexer.forward
    if getattr(fwd, "_stable", False):
        return

    def forward(self, *a, **kw):
        orig = torch.Tensor.topk

        def topk(x, k, dim=-1, *aa, **kk):
            v, i = torch.sort(-x, dim=dim, stable=True)
            if k < x.shape[dim] and bool((-v.narrow(dim, k - 1, 1) == -v.narrow(dim, k, 1)).any()):
                TIES[0] += 1
            return torch.return_types.topk((-v.narrow(dim, 0, k), i.narrow(dim, 0, k)))

        torch.Tensor.topk = topk
        try:
            return fwd(self, *a, **kw)
        finally:
            torch.Tensor.topk = orig

    forward._stable = True
    Q.Qwen4ExpTextQSAIndexer.forward = forward


def reference(q8):
    import warnings

    warnings.filterwarnings("ignore")
    from transformers import Qwen4ExpForCausalLM

    stable_indexer()

    m, li = Qwen4ExpForCausalLM.from_pretrained(REL, dtype=torch.float32, output_loading_info=True)
    assert not any(v for v in li.values()), li
    sd = m.state_dict()
    meta = json.load(open(os.path.join(MOD, "store", "meta.json")))
    REC, (I, H) = meta["record_bytes"], meta["shapes"]["gate_proj"]
    with torch.no_grad():
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
        ngm = json.load(open(os.path.join(MOD, "ngram", "meta.json")))  # the n-gram rows as stored
        rows = np.concatenate([np.fromfile(os.path.join(MOD, f), dtype=np.uint8).reshape(-1, ngm["bytes_per_row"]) for f in ngm["files"]])
        table = rows[:, 2:].view(np.int8).astype(np.float32) * rows[:, :2].copy().view(np.float16).astype(np.float32)
        key = [k for k in sd if k.endswith("ngram_embedding.weight")][0]
        sd[key].copy_(torch.from_numpy(table))
        n8 = 0
        if q8:  # the engine's 8-bit matrices, dequantised
            for n in tab:
                if n.endswith(".q"):
                    base = n[:-2]
                    sd[hf_name(base)].copy_(torch.from_numpy(deq(tensor(n), tensor(base + ".s"), tensor(base + ".b"), 8)))
                    n8 += 1
    m.eval()
    return m, n8


def ref_logits(m, ids):
    with torch.no_grad():
        return m(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()


say("== A. the exact reference path vs transformers' model with the engine's weights")
m8, n8 = reference(True)
say(f"      reference: transformers Qwen4ExpForCausalLM, float32; {n8} matrices at the engine's 8-bit codes, experts and n-gram rows from the store")
r40, r300 = ref_logits(m8, ids40), ref_logits(m8, ids300)
del m8
say(f"      the indexer's block choice had a tie at its cut {TIES[0]} times (broken toward the earlier block on both sides)")
f40, _ = dmeng(ids40, "--prec", "f32", name="f40")
compare("A f32 reference path, 40 tokens", f40, r40, 1e-4, rel=True)
f300, _ = dmeng(ids300, "--prec", "f32", name="f300")
compare("A f32 reference path, 300 tokens (the indexer's top 4 of up to 75 blocks)", f300, r300, 1e-4, rel=True)
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

say("== C. exact mode: identical logits whatever the cache, routers, run-ahead, threads and timing")
rec = os.path.join(TMP, "rec.bin")
base, _ = dmeng(ids300, "--record", rec, name="c0")
router = sorted(glob.glob(os.path.join(MOD, ".dmeng", "router.mlp_k4.q8.*.bin")))
if not router:  # the engine derives it on first use with the profile; here explicitly
    subprocess.run([os.path.join(os.path.dirname(BIN), "ms_derive"), "router", MOD, "mlp_k4"], check=True, capture_output=True)
    router = sorted(glob.glob(os.path.join(MOD, ".dmeng", "router.mlp_k4.q8.*.bin")))
FEW = ["--cache-gb", "0.004"]  # 34 slots for 8 layers x 10 picks: evictions and re-reads every token
for nm, fl in (("few slots", FEW), ("few slots, 5 threads", FEW + ["--threads", "5"]), ("1 thread", ["--threads", "1"]),
               ("few slots, routers mlp_k4 (lag 4) naming 10", FEW + ["--routers", router[0], "--m", "10"]),
               ("few slots, routers, confidence rule", FEW + ["--routers", router[0], "--rt-tau", "0.5", "--rt-cand", "16"]),
               ("few slots, run-ahead 10", FEW + ["--run-ahead", "10"]),
               ("few slots, routers + run-ahead", FEW + ["--routers", router[0], "--m", "10", "--run-ahead", "10"]),
               ("few slots, frequency eviction", FEW + ["--evict", "freq"]),
               ("fast mode with every missing pick read on the spot (read rule 0)", FEW + ["--mode", "approx:drop:w=0"])):
    g, info = dmeng(ids300, *fl, name="c1")
    compare(f"C {nm} (reads {info.get('reads', '?')})", g, base, 0.0)
for lead in ("1", "2", "4", "token"):  # a whole token of names (80 pins) needs more slots than FEW
    g, _ = dmeng(ids300, *(FEW if lead != "token" else ["--cache-gb", "0.012"]), "--prefetch", "oracle", "--oracle", rec, "--lead", lead, name="c2")
    compare(f"C {'few' if lead != 'token' else '104'} slots, oracle replay at lead {lead}", g, base, 0.0)
say("== D. fast mode at a budget that drops picks (info)")
g, info = dmeng(ids300, *FEW, "--mode", "fast", name="d1")
compare(f"D fast mode, few slots (released weight left out per token {info['approx']['weight_left_out_per_token_mean']:.3f})", g, base, None)
g, info = dmeng(ids300, *FEW, "--mode", "fast", "--routers", router[0], "--m", "10", "--run-ahead", "10", name="d2")
compare(f"D fast mode, few slots, routers + run-ahead (left out per token {info['approx']['weight_left_out_per_token_mean']:.3f})", g, base, None)

say("== E. the batched prefill equals single steps, bit for bit")
ip = os.path.join(TMP, "ids1100.txt")
rng2 = np.random.default_rng(11)
long_ids = [int(t) for t in rng2.integers(10, V, 1100)]
for p in range(0, 1100, 97):
    long_ids[p] = EOS
open(ip, "w").write(",".join(map(str, long_ids)))
for nm, extra in (("CPU", []), ("CPU, sub-chunks of 64", ["--chunk", "64"]), ("GPU share 0.4 / 0.4", ["--gpu-frac", "0.4", "--gpu-experts", "0.4"]),
                  ("few slots, union in batches", ["--cache-gb", "0.004"])):
    cmd = [SRV, "--model", MOD, "--profile", "none", "--cache-gb", "0.05", "--mem", "plain", "--ctx", "2048", "--chunk", "256", "--span", "8192",
           "--check", ip, "--gen", "16", "--threads", "4", "--readers", "2", *extra]
    # DMENG_QSA_PAR_MIN: the single steps score the indexer's blocks on the pool from 16 blocks (the prefill scores them
    # serially per position): the same bits either way
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "DMENG_QSA_PAR_MIN": "16"})
    try:
        js = json.loads(r.stdout.strip().splitlines()[-1])
        ok = js["logit_bits_differ"] == 0 and js["fixed_state_bytes_differ"] == 0 and js["kv_bytes_differ"] == 0 and js["greedy_identical"]
    except Exception:
        js, ok = {"error": (r.stdout + r.stderr)[-1500:]}, False
    say(f"{'PASS' if ok else 'FAIL'}  E dmserve --check, 1,100 tokens, {nm}: " + json.dumps({k: js.get(k) for k in ("logit_bits_differ", "fixed_state_bytes_differ", "kv_bytes_differ", "greedy_identical", "error") if k in js}))
    if not ok:
        fails.append("E " + nm)
gen = {}
for nm, extra in (("batched", ["--prefill-chunk", "64"]), ("token by token", ["--prefill-chunk", "1"])):
    op = os.path.join(TMP, f"gen_{nm[:3]}.ids")
    r = subprocess.run([BIN, "--model", MOD, "--profile", "none", "--prompt-file", ip, "--n", "32", "--out-ids", op, "--threads", "4", "--readers", "2",
                        "--mem", "plain", "--res-mem", "plain", "--cache-gb", "0.05", "--ctx", "2048", *extra], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    gen[nm] = open(op).read().strip()
ok = gen["batched"] == gen["token by token"]
say(f"{'PASS' if ok else 'FAIL'}  E dmeng: 32 generated ids after a 1,100-token prompt, batched prefill == token by token")
if not ok:
    fails.append("E dmeng ids")

say("== F. the prefix cache and the saved state")
ck = os.path.join(TMP, "ck.txt")
rng3 = np.random.default_rng(5)
open(ck, "w").write(",".join(str(int(t)) for t in rng3.integers(1, V, 1200)))
r = subprocess.run([SRV, "--model", MOD, "--profile", "none", "--cache-gb", "0.05", "--mem", "plain", "--ctx", "2048", "--chunk", "128", "--span", "8192",
                    "--ckpt-every", "256", "--ckpt-gb", "0", "--check-ckpt", ck, "--gen", "8", "--threads", "4", "--readers", "2"], capture_output=True, text=True)
try:
    js = json.loads(r.stdout.strip().splitlines()[-1])
    ok = bool(js["passed"])
    rows = ", ".join(f"{k} {js[k]['source']} reused {js[k]['reused']}" for k in ("long", "identical", "diverging"))
except Exception:
    ok, rows = False, (r.stdout + r.stderr)[-800:]
say(f"{'PASS' if ok else 'FAIL'}  F dmserve --check-ckpt (a long request, the same again, one diverging at 60%; each equal to a cold computation bit for bit): {rows}")
if not ok:
    fails.append("F check-ckpt")
for pc in ("1", "64"):
    r = subprocess.run([BIN, "--model", MOD, "--profile", "none", "--prompt-file", ip, "--n", "24", "--state-check", "--prefill-chunk", pc, "--threads", "4",
                        "--readers", "2", "--mem", "plain", "--res-mem", "plain", "--cache-gb", "0.05", "--ctx", "2048"], capture_output=True, text=True)
    sc = json.loads(r.stdout.strip().splitlines()[-1]).get("state_check", {}) if r.returncode == 0 else {}
    ok = sc.get("same_ids") is True
    say(f"{'PASS' if ok else 'FAIL'}  F dmeng --state-check after a 1,100-token prompt ({'batched' if pc != '1' else 'token by token'}): the fixed and positional "
        f"state saved, restored into a fresh state (the pooled index keys rebuilt from the raw keys), the same {24} ids ({sc})")
    if not ok:
        fails.append("F state-check " + pc)

say("TINY CHECK FAILED: " + ", ".join(fails) if fails else "TINY CHECK PASSED")
open(os.path.join(OUT, "tiny_check.txt"), "w").write("\n".join(lines) + "\n")
json.dump(res, open(os.path.join(OUT, "tiny_check.json"), "w"), indent=1)
sys.exit(1 if fails else 0)
