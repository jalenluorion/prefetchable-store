"""The tiny random Qwen3.8-Flash-Next of the engine's checks, built the way the real weight repo was: a synthetic release
(reference/flash/make_tiny.py, the release's tensor names and layouts), the published converter (reference/flash/convert.py:
4-bit expert records, int8 n-gram rows in part files, resident.safetensors), a random lag-4 router set in the trainer's npz
layout (routers_k4.npz, flash_router3 keys, float32), and the manifest (tests/tiny_manifest.py, from the weight repo's
python/convert/qwen4_exp/moved-store.json).

  python tests/qwen4_exp/make_tiny.py OUT      (python: torch, safetensors, tokenizers, numpy, huggingface_hub)

Sizes: hidden 256 (the batched kernels' 256-input blocks run), 4 streams with the real low rank 320, 8 layers (LLLQ
twice), 32 experts top 10 (the real top-k), expert and shared intermediate 320 (as the real 640, not a multiple of 256: the
down projections take the kernels' other path), the PLE at layer 1 with 4 n-gram heads x 64 (4,096 rows in 5 parts of
1,000, not aligned with the release's 4 shards of 1,024), vocab 512, QSA budget 16 (a 40-token prompt crosses it).
OUT/release is the synthetic release, OUT/model the weight repo the engine runs.
"""
import json, os, shutil, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.abspath(sys.argv[1])
REL, MOD, ST = os.path.join(OUT, "release"), os.path.join(OUT, "model"), os.path.join(OUT, "convert_state")
SIZES = {"TINY_H": "256", "TINY_LOW": "320", "TINY_NE": "32", "TINY_TOPK": "10", "TINY_I": "320", "TINY_SH": "320"}
env = {**os.environ, **SIZES}


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout


if os.path.exists(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)
info = run([sys.executable, os.path.join(REPO, "reference/flash/make_tiny.py"), REL], env=env)
ng = json.loads(info.strip().splitlines()[-1])
run([sys.executable, os.path.join(REPO, "reference/flash/convert.py")],
    env={**env, "REPO": REL, "OUT": MOD, "STATE_DIR": ST, "NG_SHARD_ROWS": str(ng["NG_SHARD_ROWS"]), "NG_DIM": str(ng["NG_DIM"]),
         "NG_PART_ROWS": "1000", "THREADS": "2"})
for f in ("_marker", "model.safetensors.index.json"):
    if os.path.exists(os.path.join(MOD, f)):
        os.remove(os.path.join(MOD, f))

# the response format of the tiny template (ChatML, JSON tool calls), in tokenizer_config.json as the release ships its own
tc_path = os.path.join(MOD, "tokenizer_config.json")
tcfg = json.load(open(tc_path))
tcfg["response_template"] = {
    "defaults": {"role": "assistant"}, "start_anchor": "<|im_start|>assistant\n",
    "fields": {"tool_calls": {"open": "<tool_call>", "close": "</tool_call>", "repeats": True, "content": "json",
                              "transform": {"type": "function", "function": {"name": "{content.name}", "arguments": "{content.arguments}"}}},
               "content": {"close": ["<|im_end|>", "<|endoftext|>"], "content": "text", "repeats": True, "join": ""}}}
json.dump(tcfg, open(tc_path, "w"), indent=1)

# a random lag-4 router set in the trainer's export layout: per target L, x = [rmsnorm(emb) * emb_norm, h_prev(, h_lag)]
cfg = json.load(open(os.path.join(MOD, "config.json")))["text_config"]
H, NL, NE, LAG, RH = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_experts"], 4, 64
rng = np.random.default_rng(4)
z = {"__lag__": np.array(LAG, dtype=np.int64)}
for L in range(NL):
    W = (3 if L >= LAG else 2) * H
    z[f"{L}.emb_norm.weight"] = (1 + 0.1 * rng.standard_normal(H)).astype(np.float32)
    z[f"{L}.lin.weight"] = (rng.standard_normal((NE, W)) / np.sqrt(W)).astype(np.float32)
    z[f"{L}.lin.bias"] = (0.1 * rng.standard_normal(NE)).astype(np.float32)
    z[f"{L}.mlp.0.weight"] = (rng.standard_normal((RH, W)) / np.sqrt(W)).astype(np.float32)
    z[f"{L}.mlp.0.bias"] = (0.1 * rng.standard_normal(RH)).astype(np.float32)
    z[f"{L}.mlp.2.weight"] = (rng.standard_normal((NE, RH)) / np.sqrt(RH)).astype(np.float32)
    z[f"{L}.mlp.2.bias"] = (0.1 * rng.standard_normal(NE)).astype(np.float32)
np.savez(os.path.join(MOD, "routers_k4.npz"), **z)
run([sys.executable, os.path.join(REPO, "tests/tiny_manifest.py"), MOD])
print(f"{MOD}: tiny Qwen3.8-Flash-Next ({H} hidden, {NL} layers, {NE} experts top {cfg['num_experts_per_tok']}, n-gram {ng})")
