"""P0 tables: compute ms/token from the kernel bench (Mac calibration + phone), and the tier table.
  python3 p0_report.py --mac mac/kernelbench-mac.json [--phone phone/kernelbench-*.json] [--filecache-gb X] > tables.md

Compute model (decode, one token, Qwen3.6-35B-A3B; MACs from config.json):
  attention projections  30 linear-attention layers x (qkv 8192x2048 + z 4096x2048 + out 2048x4096)
                         + 10 full-attention layers x (q+gate 8192x2048 + k 512x2048 + v 512x2048 + o 2048x4096)
  shared expert          40 x 3 x (512 x 2048)            router 40 x 256x2048        LM head 248320 x 2048
  routed experts         40 x 8 x 3 x (512 x 2048), aff4 (the store's records)
  bytes = MACs x (1 + 4/64) for aff8 (codes + fp16 scale and bias per 64) and MACs x (0.5 + 4/64) for aff4.
Each class is timed at the bench's measured DRAM-fed GB/s for the matching shape (best thread count, QoS
user-interactive); the shared expert uses the out_proj rate (no 512x2048 aff8 cell). Non-matvec work (attention core,
norms, conv, recurrent update, sampling, cache bookkeeping) is the M4 stage split's 3.1 + 0.5 ms, scaled on the phone by
the phone/Mac ratio of the matvec total (an assumption, labelled). Calibration: the same model on the Mac's bench rates
against the measured M4 compute ceiling 31.5-33.6 ms/token (RESULTS.md "Compute ceiling", 8 threads); the phone
estimate is reported raw and times that calibration factor.
"""
import argparse, json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
M = 1e6
ATT = 30 * (8192 * 2048 + 4096 * 2048 + 2048 * 4096) + 10 * (8192 * 2048 + 512 * 2048 + 512 * 2048 + 2048 * 4096)
SHARED = 40 * 3 * 512 * 2048
ROUTER = 40 * 256 * 2048
HEAD = 248320 * 2048
EXPERTS = 40 * 8 * 3 * 512 * 2048
B8, B4 = 1 + 4 / 64, 0.5 + 4 / 64
M4_OTHER_MS = 3.1 + 0.5
M4_CEIL = (31.5, 33.6)
REC = 1769472


def load(p):
    d = json.load(open(p))
    return d.get("result", d)


def rates(kb, qos="ui"):
    """best GB/s per case over thread counts (QoS qos), and the thread count"""
    best = {}
    for c in kb["matvec"]:
        if c["qos"] != qos:
            continue
        k = c["case"]
        if k not in best or c["gbps"] > best[k][0]:
            best[k] = (c["gbps"], c["threads"])
    return best


def compute_ms(r, resident_bits):
    a = "a8" if resident_bits == 8 else "a4"
    bb = B8 if resident_bits == 8 else B4
    g = lambda k: r[k][0] * 1e9
    att_rate = (g(a + "qkv") * 2 + g(a + "out")) / 3  # 2/3 of the attention bytes are [*, 2048] inputs, 1/3 out_proj
    parts = {
        "attention_proj": ATT * bb / att_rate,
        "shared_expert": SHARED * bb / g(a + "out"),
        "router": ROUTER * B8 / g("a8rt"),
        "head": HEAD * bb / g(a + "head"),
        "experts": EXPERTS * B4 / ((g("a4gu") * 2 + g("a4dn")) / 3),
    }
    parts = {k: v * 1e3 for k, v in parts.items()}
    parts["matvec_total"] = sum(parts.values())
    return parts


def metal_ms(kb, resident_bits):
    mt = kb.get("metal", {})
    if "matvec" not in mt:
        return None
    r = {c["case"]: (c["gbps_wall"], 0) for c in mt["matvec"]}
    return compute_ms(r, resident_bits)


def bytes_per_token(resident_bits):
    bb = B8 if resident_bits == 8 else B4
    return {"resident_gb": (ATT + SHARED + HEAD) * bb / 1e9 + ROUTER * B8 / 1e9, "experts_gb": EXPERTS * B4 / 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mac", required=True)
    ap.add_argument("--phone", nargs="*", help="phone kernelbench JSONs: per-case best over all of them (the first one carries the Metal part)")
    ap.add_argument("--filecache-gb", type=float, default=None)
    ap.add_argument("--sim", default=os.path.join(HERE, "tier_sim.json"))
    a = ap.parse_args()
    out = {}
    mac = load(a.mac)
    rm = rates(mac)
    mac8 = compute_ms(rm, 8)
    mac_pred = mac8["matvec_total"] + M4_OTHER_MS
    cal = (sum(M4_CEIL) / 2) / mac_pred
    print("## Compute model calibration (Mac M4, the same bench)\n")
    print("| class | Mac GB/s (threads) | ms/token at aff8 resident |")
    print("|---|---|---|")
    for k in ["a8qkv", "a8out", "a8rt", "a8head", "a4gu", "a4dn"]:
        print(f"| {k} | {rm[k][0]:.1f} ({rm[k][1]}) | |")
    for k, v in mac8.items():
        print(f"| {k} | | {v:.1f} |")
    print(f"\nMac predicted {mac_pred:.1f} ms/token (matvec {mac8['matvec_total']:.1f} + other {M4_OTHER_MS}) vs measured "
          f"{M4_CEIL[0]}-{M4_CEIL[1]}: calibration factor {cal:.2f}\n")
    out["mac"] = {"rates": rm, "ms_aff8": mac8, "predicted_ms": mac_pred, "calibration": cal}
    for bits in (8, 4):
        bt = bytes_per_token(bits)
        print(f"bytes/token with aff{bits} resident: resident {bt['resident_gb']:.2f} GB + experts {bt['experts_gb']:.2f} GB")
    if a.phone:
        phs = [load(p) for p in a.phone]
        ph = phs[0]
        rp = {}
        for x in phs:
            for k, v in rates(x).items():
                if k not in rp or v[0] > rp[k][0]:
                    rp[k] = v
        print("\n## Phone compute (A16), CPU\n")
        print("| class | phone GB/s (threads) | Mac GB/s | ratio |")
        print("|---|---|---|---|")
        for k in ["a8qkv", "a8out", "a8rt", "a8head", "a4qkv", "a4out", "a4head", "a4gu", "a4dn"]:
            if k in rp:
                print(f"| {k} | {rp[k][0]:.1f} ({rp[k][1]}) | {rm.get(k, (float('nan'),))[0]:.1f} | {rp[k][0] / rm[k][0] if k in rm else float('nan'):.2f} |")
        out["phone"] = {"rates": rp}
        for bits in (4, 8):
            p = compute_ms(rp, bits)
            mv = p["matvec_total"]
            other = M4_OTHER_MS * mv / mac8["matvec_total"]
            raw = mv + other
            print(f"\naff{bits} resident: " + ", ".join(f"{k} {v:.1f}" for k, v in p.items()) +
                  f"; other (scaled) {other:.1f}; **CPU compute {raw:.1f} ms/token raw, {raw * cal:.1f} calibrated**")
            out["phone"][f"cpu_ms_aff{bits}"] = {"parts": p, "other": other, "raw": raw, "calibrated": raw * cal}
            mtl = metal_ms(ph, bits)
            if mtl:
                mraw = mtl["matvec_total"] + other
                print(f"  Metal (exact kernels, wall incl. dispatch): " + ", ".join(f"{k} {v:.1f}" for k, v in mtl.items()) +
                      f"; with other {mraw:.1f} ms/token")
                out["phone"][f"metal_ms_aff{bits}"] = {"parts": mtl, "raw": mraw}
        mt = ph.get("metal", {})
        if "together" in mt:
            t = mt["together"]
            print(f"\nCPU+GPU on the experts' kernel: GPU alone {t['gpu_alone_gbps']:.1f}, CPU alone {t['cpu_alone_gbps']:.1f}, "
                  f"together {t['gpu_together_gbps']:.1f} + {t['cpu_together_gbps']:.1f} = {t['sum_together_gbps']:.1f} GB/s")
    # tier table
    sim = json.load(open(a.sim))
    cpu_ms = out.get("phone", {}).get("cpu_ms_aff4", {}).get("calibrated")
    cpu_ms8 = out.get("phone", {}).get("cpu_ms_aff8", {}).get("calibrated")
    if cpu_ms:
        print("\n## Tier table (lossless decode; reads of 1,769,472-byte records; policy = the engine's decayed count / Belady)\n")
        print("| tier | resident | cache GB (slots) | text | policy | hit | misses/token | MB/token | drive GB/s | read ms | compute ms |"
              " tok/s cache-only (sum) | tok/s named early (max) | limit |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        tiers = [("6 GB today", 1.27, [0.8, 1.1], "aff4 1.10 GB", cpu_ms)]
        if a.filecache_gb:
            fc = min(sim["sizes_gb"], key=lambda g: abs(g - a.filecache_gb))
            tiers.append((f"6 GB file cache ({a.filecache_gb:.2f} GB measured; row {fc})", fc, [0.8, 1.1], "aff4 1.10 GB", cpu_ms))
        tiers += [("8 GB (assumed +2 GB cache)", 3.3, [1.1, 2.0], "aff4 1.10 GB", cpu_ms),
                  ("12 GB (assumed +6 GB cache, row 6.6)", 6.6, [1.1, 2.0], "aff8 2.09 GB", cpu_ms8 or cpu_ms)]
        rows = []
        for name, gb, drives, res, cms in tiers:
            for text in ("pitt", "coding", "agent_text"):
                r = sim["texts"][text]["rows"][str(gb)]
                for pol in ("ours", "belady"):
                    x = r[pol]
                    for dr in drives:
                        rd = x["mb_per_token"] / 1e3 / dr * 1e3
                        s = 1e3 / (rd + cms)
                        m = 1e3 / max(rd, cms)
                        lim = "drive (hardware rate x implementation's miss count)" if rd > cms else "compute"
                        rows.append(dict(tier=name, res=res, gb=gb, slots=r["slots"], text=text, pol=pol, hit=x["hit"],
                                         mpt=x["misses_per_token"], mb=x["mb_per_token"], dr=dr, rd=rd, cms=cms, s=s, m=m, lim=lim))
                        if text != "agent_text" or True:
                            print(f"| {name} | {res} | {gb} ({r['slots']}) | {text} | {pol} | {x['hit']:.3f} | {x['misses_per_token']:.0f} | "
                                  f"{x['mb_per_token']:.0f} | {dr} | {rd:.0f} | {cms:.0f} | {s:.2f} | {m:.2f} | {lim} |")
        out["tiers"] = rows
    json.dump(out, open(os.path.join(HERE, "compute_model.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
