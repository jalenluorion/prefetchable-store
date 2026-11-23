"""Frontier table of the router design program (2026-11-21): reads every result JSON that exists and prints one row per
(design cell, replacement rule), grouped by model and cache budget, marking the Pareto set within each group.
  python3 jobs/babel/design_table.py [--dir ~/deep-memory/out_babel] [--md]      (babel login node; plain python3, no numpy)
Sources: q36 runs (out_babel/q36/*.json, key "design_cells", from q36_router.py DCELLS) and Inkling runs
(out_babel/inkling/*.json, keys "design_cells" from ink_router.py DCELLS and "engine_cells" from ENG_CELLS).
Columns: stall reads/token (picks read on the spot: the waits the engine pays), MB/token (all reads: names, on-the-spot,
background), ms/token from the late-read proxy's clock (compute D_LAYER_MS per layer + on-the-spot reads + waits for late
named reads at D_DRIVE_GBPS; a speed proxy, not a measurement), share of the released routing weight replaced, late-read proxy (named picks whose read would end after their
layer starts), KL to the released model over its top 32 tokens per domain (nats/token; mean = the ranking measure), top-1
agreement (mean), NLL delta (mean), router bytes touched per token (bf16 router weights; 0 = no router).
Pareto (marked *): no other row of the same model and budget is at least as good on stall reads, MB/token and mean KL and
better on one. Lossless rows have KL 0 by construction."""
import glob, json, os, sys

args = sys.argv[1:]
root = os.path.expanduser(args[args.index("--dir") + 1] if "--dir" in args else "~/deep-memory/out_babel")
md = "--md" in args
rows = []


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


for path in sorted(glob.glob(os.path.join(root, "q36", "*.json")) + glob.glob(os.path.join(root, "inkling", "*.json"))):
    try:
        r = json.load(open(path))
    except Exception:  # noqa: BLE001
        continue
    model = "q36" if "/q36/" in path else "inkling"
    run = os.path.basename(path)[:-5]
    for c in r.get("design_cells", []):
        doms = c.get("domains", {})
        if not doms:
            continue
        rules = sorted({k for x in doms.values() for k in x.get("rules", {})}) or ["-"]
        for rule in rules:
            kl = {d: (x["rules"][rule]["kl_top32"] if rule in x.get("rules", {}) else (0.0 if c["mode"] == "lossless" else None)) for d, x in doms.items()}
            rows.append({"model": model, "run": run, "gb": c["gb"], "design": c["label"], "rule": rule,
                         "stall": mean([x["stall_reads_per_token"] for x in doms.values()]), "mb": mean([x["mb_per_token"] for x in doms.values()]),
                         "w_repl": mean([x["weight_replaced"] for x in doms.values()]), "late": mean([x.get("late_proxy_share_of_named") for x in doms.values()]),
                         "hit": mean([x.get("hit_rate") for x in doms.values()]), "ms": mean([x.get("ms_per_token_proxy") for x in doms.values()]), "kl": kl, "kl_mean": mean(list(kl.values())),
                         "top1": mean([x["rules"][rule]["top1_agree"] if rule in x.get("rules", {}) else (1.0 if c["mode"] == "lossless" else None) for x in doms.values()]),
                         "dnll": mean([x["rules"][rule]["nll_delta"] if rule in x.get("rules", {}) else (0.0 if c["mode"] == "lossless" else None) for x in doms.values()]),
                         "rbytes": c.get("router_bytes_per_token", 0), "ndom": len(doms)})
    for c in r.get("engine_cells", []):  # ink_router.py ENG_CELLS (no-stall, loss delta only)
        doms = c.get("domains", {})
        rows.append({"model": model, "run": run, "gb": c["gb"], "design": f"{c['var']}:nostall:{c['gb']}:{c['rule']}", "rule": "next",
                     "stall": mean([x.get("fallback_reads_per_token") for x in doms.values()]), "mb": mean([x["mb_per_token"] for x in doms.values()]),
                     "w_repl": mean([1 - x["weight_kept"] for x in doms.values()]), "late": None, "hit": None, "ms": None,
                     "kl": {d: None for d in doms}, "kl_mean": None, "top1": None, "dnll": mean([x["delta"] for x in doms.values()]),
                     "rbytes": None, "ndom": len(doms)})


def better_eq(a, b):
    ks = ("stall", "mb", "kl_mean")
    if any(a[k] is None or b[k] is None for k in ks):
        return False
    return all(a[k] <= b[k] for k in ks) and any(a[k] < b[k] for k in ks)


def f(v, p=3):
    return "-" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


groups = {}
for x in rows:
    groups.setdefault((x["model"], x["gb"]), []).append(x)
for (model, gb), g in sorted(groups.items()):
    for x in g:
        x["pareto"] = x["kl_mean"] is not None and not any(better_eq(y, x) for y in g if y is not x)
    g.sort(key=lambda x: (x["kl_mean"] if x["kl_mean"] is not None else 9, x["stall"] or 0))
    doms = sorted({d for x in g for d in x["kl"]})
    head = ["", "design", "rule", "stall rd/tok", "MB/tok", "ms/tok proxy", "w repl", "late", "hit"] + [f"KL {d}" for d in doms] + ["KL mean", "top1", "dNLL", "router B/tok", "run"]
    print(f"\n## {model}, {gb} GB ({len(g)} rows)")
    if md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
    else:
        print("  ".join(head))
    for x in g:
        cells = ["*" if x["pareto"] else "", x["design"], x["rule"], f(x["stall"], 2), f(x["mb"], 1), f(x["ms"], 1), f(x["w_repl"], 4), f(x["late"], 3), f(x["hit"], 3)] + \
                [f(x["kl"].get(d), 4) for d in doms] + [f(x["kl_mean"], 4), f(x["top1"], 3), f(x["dnll"], 4), f(x["rbytes"]), x["run"]]
        print(("| " + " | ".join(cells) + " |") if md else "  ".join(cells))
# E6: trace-driven eviction (jobs/babel/eviction_study.py -> out_babel/q36/q36_d_evict.json): hit rate / MB per token / share
# of the decay -> Belady gap recovered, per domain and budget, and the policy's scoring work per token
for path in sorted(glob.glob(os.path.join(root, "*", "*_evict*.json"))):
    if path.endswith("_policy.json"):
        continue
    try:
        ev = json.load(open(path))
    except Exception:  # noqa: BLE001
        continue
    if "choice" in ev:  # eviction_v3.py: modes pool / caps, the default per budget, then every cell as MB/token / gap
        print(f"\n## E6 v3 eviction ({os.path.basename(path)}, {ev['model']}): the engine's default per cache size, mean MB/token over "
              f"{', '.join(ev['windows'])} (gap = share of decay-one-pool -> Belady-one-pool); validation: "
              + ("; ".join(f"{v['policy']}{' caps' if v['caps'] else ''} {v['gb']}: {v['phone_core_mb']} vs {v['this_core_mb']}" for v in ev.get("validate", [])) or "-"))
        print("  GB  | decay (pool) | best hazard | best learned | best online | Belady (pool)")
        for gb, c in ev["choice"].items():
            bl = c.get("best_learned")
            print(f"  {gb} | {c['decay_pool_mean_mb']:.1f} | {c['best_hazard']['mean_mb']:.1f} ({c['best_hazard']['mode']}) | "
                  + (f"{bl['mean_mb']:.1f} ({bl['policy']}, {bl['mode']})" if bl else "-")
                  + f" | {c['best']['policy']} {c['best']['mode']} {c['best']['mean_mb']:.1f} | {c['belady_pool_mean_mb']:.1f}")
        for mode, ds in ev["results"].items():
            for d, bs in ds.items():
                for gb, ps in bs.items():
                    print(f"  {mode} {d} {gb} GB | " + " | ".join(f"{p} {x['mb_per_token']:.1f} / {'-' if x['gap_recovered'] is None else round(x['gap_recovered'], 3)}"
                                                         + (f" ({x['model_macs_per_token'] / 1e6:.2f}M MAC)" if x.get("model_macs_per_token") else "") for p, x in ps.items()))
        rows.append(None)
        continue
    print(f"\n## E6 eviction ({os.path.basename(path)}): hit rate / MB per token / gap recovered (decay -> Belady); learned: {ev.get('learned', {}) and {r: round(v['val_acc'], 3) for r, v in ev['learned'].items()}}")
    for d, bs in ev.get("results", {}).items():
        for gb, ps in bs.items():
            cells = [f"{p}: {x['hit_rate']:.4f} / {x['mb_per_token']:.1f} / {'-' if x['gap_recovered'] is None else round(x['gap_recovered'], 3)}"
                     + (f" ({x['policy_macs_per_token'] / 1e6:.2f}M MAC/tok)" if x.get("policy_macs_per_token") else "") for p, x in ps.items()]
            print(f"{d} {gb} GB | " + " | ".join(cells))
    rows.append(None)
if not [x for x in rows if x]:
    print(f"no design-cell results under {root} yet")
