#!/usr/bin/env python3
"""The parity table: every decode and prefill stage of both models side by side, from counting-build runs.

usage: parity.py LOGDIR TAG [--base TAG0] [--md OUT.md]   (--base: runs missing under TAG come from TAG0)
  reads LOGDIR/TAG_{q36,ink}_{decode,prefill}.{log,json} (results/parity/count.sh TAG BIN makes them: dmeng from a
  -DDM_COUNTERS=ON build with DM_COUNTS=1; decode = 128 tokens after the Pittsburgh prompt, prefill = 1,024 ids batched)

Per decode stage and model: ms per token (the engine's stage timer), weight MB per token (fixed by the model and the
settings, table BYTES below), GB/s (MB over ms; the machine reads 102-107 GB/s at 8-10 threads), cycles and instructions
per weight byte (all threads' work in the stage's pool runs plus the main thread's serial part, from DMSTAGE), the main
thread's serial share, and the code path (shared core function or model-specific code, table PATH). Flags:
  CYC  the stage runs the same weight format in both models and one costs > 10% more cycles per weight byte
  PATH one model runs model-specific code where the other runs a shared core function
  READ the stage is below 75% of the read rate (105 GB/s) while weight-bound
Prefill stages: cycles and instructions per position, ms per position (the stage's share of the prefill's cycles times
the prefill's ms per position)."""
import json, os, re, sys

READ = 105.0  # GB/s, the base M4 at 8-10 threads

# decode stages: (label, DMSTAGE names summed, timer keys summed)
STAGES = [
    ("embed", ["embed"], ["embed"]),
    ("routers: the model's + ours (lag rows, token start, naming)", ["moe_router", "prefetch"], ["moe_router", "prefetch_hooks"]),
    ("depth attention", ["depth_attn"], ["depth_attn"]),
    ("message gates + norms / quantisation / convs / KV prep", ["msg_gates", "norms_quant", "convs", "kv_prep"], ["message_gates_norms"]),
    ("attention projections", ["attn_proj"], ["attn_proj"]),
    ("attention core (+ fused output projection)", ["attn_core"], ["attn_core"]),
    ("attention output projection", ["attn_out"], ["attn_out"]),
    ("expert slots / cache", ["moe_cache"], ["cache_mgmt"]),
    ("experts (+ combine)", ["moe_experts", "moe_combine"], ["experts"]),
    ("output head", ["head"], ["head"]),
    ("other", ["other", "span_copies"], ["post_mlp"]),
]
# weight MB per decode token and format, per model (35B: results/q36_speed; Inkling:
# results/ink_speed/INK-SPEED.md "Bytes per token by stage"); "routers" adds the run's router_weight_mb_per_token
BYTES = {
    "q36": {"attention projections": (1007, "AFF8"), "attention core (+ fused output projection)": (267 + 126, "AFF8 + fp32 state"),
            "attention output projection": (89, "AFF8"), "routers: the model's + ours (lag rows, token start, naming)": (42, "bf16 + ours 8-bit"),
            "experts (+ combine)": (700, "AFF4 routed + AFF8 shared"), "output head": (540, "AFF8 (full pass)")},
    "ink": {"attention projections": (1024, "AFF8"), "attention core (+ fused output projection)": (361, "AFF8"),
            "message gates + norms / quantisation / convs / KV prep": (170, "bf16 + f32 (gates)"), "routers: the model's + ours (lag rows, token start, naming)": (28, "bf16 + ours 8-bit"),
            "experts (+ combine)": (1839, "NVFP4 + MXFP8"), "output head": (492, "AFF8 (full pass)")},
}
SAME_FORMAT = {"attention projections", "attention output projection", "output head"}  # compared for CYC (attn core: o only)
# code path per stage (file: function); "shared" = the same core function for both models
PATH = {
    "attention projections": ("shared: src/core/weights.cpp run_rows_q + kernels.cpp mv_aff8", "shared: same"),
    "attention core (+ fused output projection)": ("model: qwen3_5_moe.cpp phase_lin_conv_heads (DeltaNet) / phase_full_heads, split_job",
                                                   "model: inkling2_loop.cpp phase_attn / phase_attn_kv (grouped)"),
    "attention output projection": ("shared: weights.cpp run_rows (full layers)", "fused into the core (inkling2_loop.cpp)"),
    "routers: the model's + ours (lag rows, token start, naming)": ("shared: moe.cpp moe_step run_rows_q (bf16 dot_bf16) + address.cpp Routers (lag rows in the router's fork-join; token start: phase_pre_x / phase_pre_rows; naming serial)",
                                                                    "shared: same"),
    "experts (+ combine)": ("shared: moe.cpp phase_gudn + kernels.cpp mv_aff4", "shared: moe.cpp phase_gudn + kernels.cpp mv_nvfp4 / mv_mxfp8"),
    "output head": ("shared: screen_head.cpp ScreenHead (default on)", "shared: screen_head.cpp ScreenHead with softcap (default on)"),
    "depth attention": ("-", "model: inkling2_loop.cpp depth_attn (main thread)"),
    "message gates + norms / quantisation / convs / KV prep": ("model: qwen3_5_moe.cpp step rmsnorm (weights.h), serial",
                                                               "model: inkling2_loop.cpp gates (pool) + rmsnorm, convs, KV prep (serial)"),
    "expert slots / cache": ("shared: moe.cpp lossless + store.cpp", "shared: same"),
    "embed": ("shared: weights.cpp EmbedRows (pread of one bf16 row)", "model: inkling2_loop.cpp embedding + its norm"),
}


USED = []  # (model, kind, tag) of every run read


def load(logdir, tag, model, kind):
    base = os.path.join(logdir, f"{tag}_{model}_{kind}")
    if not os.path.exists(base + ".json") and "--base" in sys.argv:  # a model / kind the change did not touch: the base run
        tag = sys.argv[sys.argv.index("--base") + 1]
        base = os.path.join(logdir, f"{tag}_{model}_{kind}")
    if not os.path.exists(base + ".json"):
        return None
    USED.append((model, kind, tag))
    js = json.load(open(base + ".json"))
    st = {}
    for l in open(base + ".log"):
        if l.startswith("DMSTAGE"):
            f = l.split()
            st[f[1]] = [float(x) for x in f[2:6]]  # serial ins, serial cyc, pool ins, pool cyc
    o = re.search(r'"others_cpu_pct_mean": ([0-9.]+)', open(base + ".log").read())
    return js, st, o.group(1) if o else "?"


def find(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def decode_rows(js, st, model):
    n = max(1, js["decode_steps"])
    t = js["by_stage_ms_per_token"]
    rows = {}
    for label, names, timers in STAGES:
        si = sum(st.get(k, [0] * 4)[0] for k in names) / n
        sc = sum(st.get(k, [0] * 4)[1] for k in names) / n
        pi = sum(st.get(k, [0] * 4)[2] for k in names) / n
        pc = sum(st.get(k, [0] * 4)[3] for k in names) / n
        ms = sum(t.get(k, 0) for k in timers) if timers else None
        mb, fmt = BYTES[model].get(label, (None, ""))
        if mb is not None and label.startswith("routers"):
            mb += find(js, "router_weight_mb_per_token") or 0
        if label == "output head" and find(js, "fast_head"):
            fmt += ", screened: reads ~53%"
        rows[label] = dict(ms=ms, mb=mb, fmt=fmt, ins=pi + si, cyc=pc + sc, ser=sc)
    return rows


def fmt(x, f="{:.2f}"):
    return "-" if x is None else f.format(x)


def main():
    logdir, tag = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    L = []
    D = {m: load(logdir, tag, m, "decode") for m in ("q36", "ink")}
    P = {m: load(logdir, tag, m, "prefill") for m in ("q36", "ink")}
    R = {m: decode_rows(D[m][0], D[m][1], m) for m in D if D[m]}
    L.append(f"## Decode, per token (counting build; others' CPU: 35B {D['q36'][2] if D['q36'] else '?'}%, Inkling {D['ink'][2] if D['ink'] else '?'}%)\n")
    for m in ("q36", "ink"):
        if D[m]:
            js = D[m][0]
            L.append(f"- {'35B' if m == 'q36' else 'Inkling'}: compute {js['compute_ms_per_token']:.1f} ms/token, wait {js['wait_ms_per_token']:.1f}, "
                     f"{js['tok_s']:.1f} tok/s (loaded machine: compare cycles, not ms)")
    L.append("")
    L.append("| stage | model | ms | weight MB | format | GB/s | cyc/B | ins/B | M cyc | main serial M cyc | code path | flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, _, _ in STAGES:
        rs = {m: R[m][label] for m in R}
        if all((r["ms"] in (None, 0) or r["ms"] < 0.005) and r["cyc"] < 5e4 for r in rs.values()):
            continue
        cpb = {m: (r["cyc"] / (r["mb"] * 1e6) if r["mb"] else None) for m, r in rs.items()}
        for m, r in rs.items():
            flags = []
            if label in SAME_FORMAT and all(cpb.get(k) for k in ("q36", "ink")):
                o = "ink" if m == "q36" else "q36"
                if cpb[m] > 1.10 * cpb[o]:
                    flags.append(f"CYC (+{100 * (cpb[m] / cpb[o] - 1):.0f}% vs {'Inkling' if o == 'ink' else '35B'})")
            pq, pi = PATH.get(label, ("", ""))
            mine, other = (pq, pi) if m == "q36" else (pi, pq)
            if mine.startswith("model:") and other.startswith("shared"):
                flags.append("PATH")
            gbs = r["mb"] / r["ms"] if r["mb"] and r["ms"] else None
            if gbs and label != "output head" and gbs < 0.75 * READ:
                flags.append(f"READ ({100 * gbs / READ:.0f}%)")
            L.append(f"| {label} | {'35B' if m == 'q36' else 'Inkling'} | {fmt(r['ms'])} | {fmt(r['mb'], '{:.0f}')} | {r['fmt']} | {fmt(gbs, '{:.0f}')} | "
                     f"{fmt(cpb[m], '{:.3f}')} | {fmt(r['ins'] / (r['mb'] * 1e6) if r['mb'] else None, '{:.3f}')} | {r['cyc'] / 1e6:.1f} | {r['ser'] / 1e6:.2f} | {mine or '-'} | {' '.join(flags)} |")
    L.append("")
    L.append("## Prefill, per position (1,024 ids, batched; counting build)\n")
    L.append("Cycles per weight byte per position (cyc/B) for the stages whose weights are known: the compute cost of one position's")
    L.append("use of each weight byte (prefill reads a weight once per chunk; what limits it is arithmetic).\n")
    L.append("| stage | 35B M cyc / pos | 35B M ins / pos | 35B cyc/B | 35B main serial M cyc / pos | Inkling M cyc / pos | Inkling M ins / pos | Inkling cyc/B | Inkling main serial M cyc / pos | flags |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    PB = {"attn_proj": ("attention projections", "AFF8"), "moe_experts": ("experts (+ combine)", None), "attn_core": ("attention core (+ fused output projection)", None)}
    pr = {}
    for m in ("q36", "ink"):
        if not P[m]:
            continue
        js, st, _ = P[m]
        npos = js.get("prompt_tokens") or 1024
        pr[m] = {k: [x / npos for x in v] for k, v in st.items()}
        pr[m]["_ms"] = js.get("prefill_s", 0) * 1000 / npos
    order = ["embed", "depth_attn", "msg_gates", "norms_quant", "convs", "kv_prep", "attn_proj", "attn_core", "attn_out", "moe_router",
             "moe_cache", "moe_experts", "moe_combine", "head", "prefetch", "span_copies", "other"]
    for k in order:
        vals, cb = [], {}
        for m in ("q36", "ink"):
            v = pr.get(m, {}).get(k)
            mb = BYTES[m].get(PB[k][0], (None,))[0] if k in PB else None
            if v:
                cyc = v[3] + v[1]
                cb[m] = cyc / (mb * 1e6) if mb else None
                vals += [fmt(cyc / 1e6), fmt((v[2] + v[0]) / 1e6), fmt(cb[m], "{:.3f}"), fmt(v[1] / 1e6, "{:.3f}")]
            else:
                vals += ["-"] * 4
        if all(x in ("-", "0.00", "0.000") for x in vals):
            continue
        flags = []
        if k in PB and PB[k][1] and cb.get("q36") and cb.get("ink"):
            hi, lo = ("ink", "q36") if cb["ink"] > cb["q36"] else ("q36", "ink")
            if cb[hi] > 1.10 * cb[lo]:
                flags.append(f"CYC {'Inkling' if hi == 'ink' else '35B'} +{100 * (cb[hi] / cb[lo] - 1):.0f}%")
        if k == "moe_experts" and cb.get("q36") and cb.get("ink"):
            flags.append(f"formats differ (AFF4 vs NVFP4/MXFP8); Inkling / 35B = {cb['ink'] / cb['q36']:.2f}x per byte; paths differ: "
                         "35B prefill.cpp (its own MoE, gemm_aff4_i16), Inkling moe_chunk.cpp (shared)")
        L.append(f"| {k} | " + " | ".join(vals) + f" | {' '.join(flags)} |")
    L.append("")
    for m in ("q36", "ink"):
        if m in pr:
            L.append(f"- {'35B' if m == 'q36' else 'Inkling'} prefill: {pr[m]['_ms']:.2f} ms per position (others' CPU {P[m][2]}%; "
                     f"waits on expert reads {P[m][0].get('prefill_wait_ms', 0) / 1024:.2f} ms per position)")
    L.append("")
    L.append("Runs: " + ", ".join(f"{'35B' if m == 'q36' else 'Inkling'} {k} = {t}" for m, k, t in USED))
    text = "\n".join(L)
    print(text)
    if out:
        open(out, "w").write(text + "\n")


if __name__ == "__main__":
    main()
