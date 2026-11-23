"""Tokens/s on the iPhone 15 Plus from the simulated MB/token and P0's measurements (results/ios/p0/P0.md on port-ios):
drive 0.8 GB/s (buffered pread, 8 threads: 0.69-0.76 measured; 0.8 = P0's table), compute 55 ms/token burst and 89 ms
sustained with the 4-bit resident (CPU, section 3.2), a record copied out of the OS file cache 0.15 ms (buffered pread of
a resident record, 0.13-0.20 ms, section 2.2).
  cache only    a layer's misses are read after its router; compute waits: compute + reads
  named early   every read named before its layer runs and overlapped with compute: max(compute, reads) (the ceiling
                with exact names)
  router lag 2  the trained k2 router's named records read early (overlapped), the released picks it missed read on the
                spot: max(compute, early) + spot (simulated in the named-early mode, wasted reads included)
  python3 predict.py > predict.md   (reads results.json written by tables.py)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
DRIVE = 0.8  # GB/s = MB/ms
COMPUTE = {"burst": 55.0, "sustained": 89.0}
COPY_MS = 0.15
REC_MB = 1.769472


def get(tn, gb, lab):
    return R.get(tn, {}).get(gb, {}).get(lab)


def tok_s(read_ms, c, mode, extra_ms=0.0, spot_ms=0.0):
    if mode == "cache":
        return 1000.0 / (c + read_ms + extra_ms)
    if mode == "early":
        return 1000.0 / (max(c, read_ms) + extra_ms)
    return 1000.0 / (max(c, read_ms) + spot_ms + extra_ms)


def row(label, mbt, extra_ms=0.0):
    ms = mbt / DRIVE
    cells = []
    for c in COMPUTE.values():
        cells.append(f"{tok_s(ms, c, 'cache', extra_ms):.2f}")
    for c in COMPUTE.values():
        cells.append(f"{tok_s(ms, c, 'early', extra_ms):.2f}")
    return f"| {label} | {mbt:.0f} | {ms:.0f} | " + " | ".join(cells) + " |"


if __name__ == "__main__":
    hdr = ("| arrangement / policy | MB/token (drive) | read ms | tok/s cache only, burst | sustained | "
           "tok/s named early (ceiling), burst | sustained |\n|---|---|---|---|---|---|---|")
    for tn in ("pitt", "coding", "agent"):
        print(f"\n**{tn}**\n")
        print(hdr)
        for lab, name in (("decay:8", "1.27 GB, decay (today's default)"), ("lru", "1.27 GB, LRU"),
                          ("hazard", "1.27 GB, hazard pooled"), ("hazard|caps_uniform", "1.27 GB, hazard + uniform per-layer caps (the port)"),
                          ("phone_hazard_nodrop", "1.27 GB, the phone engine's policy (table caps, hazard, no drop)"),
                          ("belady", "1.27 GB, Belady (offline bound)")):
            v = get(tn, "1.27", lab)
            if v:
                print(row(name, v["mb_per_token"]))
        for lab, name in (("A_excl|decay:8", "1.27 decay + 2.1 GB LRU tier (exclusive)"),
                          ("A_excl|hazard", "1.27 hazard + 2.1 GB LRU tier (exclusive)"),
                          ("A_incl|decay:8", "1.27 decay + 2.1 GB OS cache (inclusive, buffered reads)"),
                          ("A_incl|hazard", "1.27 hazard + 2.1 GB OS cache (inclusive, buffered reads)")):
            v = get(tn, "1.27", lab)
            if v:
                print(row(name + f" [+{v['tier2_hits_per_token']:.0f} copies/token]", v["drive_mb_per_token"],
                          v["tier2_hits_per_token"] * COPY_MS))
        for lab, name in (("lru", "3.3 GB one tier, OS LRU decides (= LRU at 3.3)"),
                          ("decay:8", "3.3 GB one tier, decay (needs page drop)"),
                          ("hazard", "3.3 GB one tier, hazard (needs page drop)"),
                          ("hazard|caps_uniform", "3.3 GB one tier, hazard + uniform caps (needs page drop)"),
                          ("belady", "3.3 GB, Belady (offline bound)")):
            v = get(tn, "3.3", lab)
            if v:
                print(row(name, v["mb_per_token"]))
    print("\nRouter lag 2 named early (trained k2 router, top N read two layers ahead; agent transcript):\n")
    print("| cache | policy | N | early MB | spot MB | wasted rec/token | tok/s burst | sustained |\n|---|---|---|---|---|---|---|---|")
    for gb in ("1.27", "2.0", "3.3"):
        for p in ("decay:8", "lru", "hazard"):
            for N in ("8", "12"):
                v = get("agent", gb, f"pf_r8_{N}|{p}")
                if not v:
                    continue
                e, s = v["mb_early"] / DRIVE, v["mb_spot"] / DRIVE
                ts = [1000.0 / (max(c, e) + s) for c in COMPUTE.values()]
                print(f"| {gb} | {p} | {N} | {v['mb_early']:.0f} | {v['mb_spot']:.0f} | {v['wasted_per_token']:.1f} | "
                      f"{ts[0]:.2f} | {ts[1]:.2f} |")
