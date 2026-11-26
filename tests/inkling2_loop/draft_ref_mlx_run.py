"""The MLX reference of Inkling's draft head, for the C++ port check (dmeng --draft-check). The arithmetic is that of
sglang PR 51926's DSpark drafter, unchanged; only the output differs:
per committed position p, the 7 drafts after t[p+1] (each conditioned on the previous draft), their top-1 probabilities, and
draft 1's logits for the first N positions, in dmeng --draft-check's record format.

  python draft_ref_mlx_run.py --model MODEL --taps F --ids F --start P --out PREFIX [--logits 200] [--check-accept accept1.json]

--check-accept: mtp_accept.py's saved rows (p, draft 1, greedy, prob) for the same taps; prints how many of them this copy
reproduces (draft 1 and probability), to show the copy is the reference.
"""
import argparse, json, os, time

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--taps", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--logits", type=int, default=200)
ap.add_argument("--check-accept", default=None)
a = ap.parse_args()

MODEL = a.model
cfg = json.load(open(os.path.join(MODEL, "config.json")))
mc = cfg["mtp_config"]
H, NH, NKV, HD = mc["hidden_size"], mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"]
EPS, THETA, WIN, BS = mc["rms_norm_eps"], mc["rope_theta"], mc["sliding_window_size"], mc["block_size"]
LOCAL = set(mc["local_layer_ids"])
VU = cfg["text_config"]["unpadded_vocab_size"]
SOFTCAP = cfg["text_config"]["final_logit_softcapping"]
BF = mx.bfloat16

W = mx.load(os.path.join(MODEL, "mtp.safetensors"))
R = mx.load(os.path.join(MODEL, "resident.safetensors"))
emb_w, emb_norm_w, head_w = R["model.llm.embed.weight"], R["model.llm.embed_norm.weight"], R["model.llm.unembed.weight"]
alpha = mx.softmax(W["fuse_weight"].astype(mx.float32), axis=-1).astype(BF)

ids = [int(t) for t in open(a.ids).read().split(",")]
N = len(ids)
raw = np.fromfile(a.taps, dtype=np.uint8)
rec = 4 + 4 * 4 * H
assert raw.size == N * rec, (raw.size, N, rec)
raw = raw.reshape(N, rec)
feats = mx.array(raw[:, 4:].copy().view(np.float32).reshape(N, 4, H)).astype(BF)


def rms(x, w, eps=EPS):
    x32 = x.astype(mx.float32)
    return (x32 * mx.rsqrt((x32 * x32).mean(-1, keepdims=True) + eps) * w.astype(mx.float32)).astype(BF)


def lin(x, w, b=None):
    y = x.astype(BF) @ w.T
    return y if b is None else y + b


def mix_norm(f, row, w):
    m = (f.astype(mx.float32) * alpha[row].astype(mx.float32)[None, :, None]).sum(1)
    return (m * mx.rsqrt((m * m).mean(-1, keepdims=True) + EPS) * w.astype(mx.float32)).astype(BF)


def rope(x, pos):
    half = HD // 2
    inv = 1.0 / (THETA ** (np.arange(half, dtype=np.float64) * 2.0 / HD))
    ang = mx.array((np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]).astype(np.float32))
    c, s = mx.cos(ang)[:, None, :], mx.sin(ang)[:, None, :]
    x32 = x.astype(mx.float32)
    x1, x2 = x32[..., :half], x32[..., half:]
    return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(BF)


CK, CV = [], []
for i in range(3):
    p_ = f"layers.{i}.self_attn."
    ctx = mix_norm(feats, i, W[f"layers.{i}.dflare_norm.weight"])
    k = lin(ctx, W[p_ + "k_proj_ctx.weight"]).reshape(N, NKV, HD)
    k = rope(rms(k, W[p_ + "k_norm_ctx.weight"]), np.arange(N))
    v = lin(ctx, W[p_ + "v_proj_ctx.weight"]).reshape(N, NKV, HD)
    mx.eval(k, v)
    CK.append(k)
    CV.append(v)
f_rnn_all = mix_norm(feats, 3, W["markov_head.dflare_norm.weight"])
mask_e = W["mask_embed"]


def embed(t):
    return rms(emb_w[t][None, :], emb_norm_w)


def draft(p, nsteps):
    pos = np.arange(p + 1, p + 1 + BS)
    x = mx.concatenate([embed(ids[p + 1]), mx.broadcast_to(mask_e[None, :], (BS - 1, H))], axis=0)
    resid = None
    for i in range(3):
        p_ = f"layers.{i}."
        resid = x if resid is None else resid + x
        hn = rms(resid, W[p_ + "input_layernorm.weight"])
        q = rope(rms(lin(hn, W[p_ + "self_attn.q_proj.weight"]).reshape(BS, NH, HD), W[p_ + "self_attn.q_norm.weight"]), pos)
        k = rope(rms(lin(hn, W[p_ + "self_attn.k_proj.weight"]).reshape(BS, NKV, HD), W[p_ + "self_attn.k_norm.weight"]), pos)
        v = lin(hn, W[p_ + "self_attn.v_proj.weight"]).reshape(BS, NKV, HD)
        lo = max(0, (p + 1) - (WIN - 1)) if i in LOCAL else 0
        K = mx.concatenate([CK[i][lo : p + 1], k], axis=0)
        V = mx.concatenate([CV[i][lo : p + 1], v], axis=0)
        qh = q.astype(mx.float32).reshape(BS, NKV, NH // NKV, HD).transpose(1, 2, 0, 3)
        kh = K.astype(mx.float32).transpose(1, 0, 2)
        vh = V.astype(mx.float32).transpose(1, 0, 2)
        sc = (qh @ kh[:, None].swapaxes(-1, -2)) * (HD ** -0.5)
        pr = mx.softmax(sc, axis=-1)
        o = (pr @ vh[:, None]).transpose(2, 0, 1, 3).reshape(BS, NH * HD).astype(BF)
        attn = lin(o, W[p_ + "self_attn.o_proj.weight"])
        resid = resid + attn
        hn = rms(resid, W[p_ + "post_attention_layernorm.weight"])
        g = lin(hn, W[p_ + "mlp.gate_proj.weight"]).astype(mx.float32)
        u = lin(hn, W[p_ + "mlp.up_proj.weight"]).astype(mx.float32)
        x = lin((g * mx.sigmoid(g) * u).astype(BF), W[p_ + "mlp.down_proj.weight"])
    hfin = rms(resid + x, W["norm.weight"])
    state = mx.zeros((1, 256), dtype=BF)
    f_rnn = f_rnn_all[p][None, :]
    prev = ids[p + 1]
    out, probs, lg1 = [], [], None
    for j in range(1, 1 + nsteps):
        hj = hfin[j][None, :]
        inp = mx.concatenate([state, W["markov_head.markov_w1.weight"][prev][None, :], hj, f_rnn], axis=-1)
        pr_ = lin(inp, W["markov_head.rnn_proj.weight"], W["markov_head.rnn_proj.bias"]).astype(mx.float32)
        gate = mx.sigmoid(pr_[:, :256])
        state = (gate * state.astype(mx.float32) + (1 - gate) * mx.tanh(pr_[:, 256:512])).astype(BF)
        latent = mx.tanh(pr_[:, 512:]).astype(BF)
        z = hj + lin(latent, W["markov_head.hidden_out.weight"], W["markov_head.hidden_out.bias"])
        logits = lin(z, head_w).astype(mx.float32)[0, :VU]
        logits = SOFTCAP * mx.tanh(logits / SOFTCAP)
        t = int(mx.argmax(logits).item())
        probs.append(float(mx.softmax(logits).max().item()))
        if j == 1:
            lg1 = np.array(logits, dtype=np.float32)
        out.append(t)
        prev = t
    return out, probs, lg1


t0 = time.time()
fd = open(a.out + ".drafts", "wb")
fl = open(a.out + ".logits", "wb")
done = 0
mine = {}
for p in range(a.start, N - 2):
    d, pr, lg1 = draft(p, 7)
    fd.write(np.array([p] + d, dtype=np.int32).tobytes() + np.array(pr, dtype=np.float32).tobytes())
    if done < a.logits:
        fl.write(lg1.tobytes())
    mine[p] = (d[0], pr[0])
    done += 1
fd.close()
fl.close()
res = {"positions": done, "s": round(time.time() - t0, 1)}
if a.check_accept:
    rows = json.load(open(a.check_accept))["rows"]
    same = sum(1 for (p, d0, g, prob) in rows if p in mine and mine[p][0] == d0)
    dp = max(abs(mine[p][1] - prob) for (p, d0, g, prob) in rows if p in mine)
    res["vs_mtp_accept"] = {"rows": len(rows), "same_draft1": same, "max_prob_diff": dp}
print(json.dumps(res))
