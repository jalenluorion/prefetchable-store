"""Merge res_*.json into results.json and print the markdown tables of EVICT.md (MB/token, decimal MB, per counted token).
  python3 tables.py > tables.md
"""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for p in sorted(glob.glob(os.path.join(HERE, "res_*.json"))):
    for tn, by in json.load(open(p)).items():
        for gb, cells in by.items():
            R.setdefault(tn, {}).setdefault(gb, {}).update(cells)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
SIZES = ["0.8", "1.0", "1.27", "1.6", "2.0", "2.5", "3.3", "5.0"]
SLOTS = {"0.8": 452, "1.0": 565, "1.27": 720, "1.6": 904, "2.0": 1130, "2.5": 1412, "3.3": 1864, "5.0": 2825}
NAMES = {"pitt": "Pittsburgh (399 generated)", "coding": "coding (399 generated)", "agent": "agent transcript (2,873 TF)",
         "code_text": "code text (1,800 TF)", "prose_text": "prose (1,546 TF)", "ph_chat": "phone chat dump (1,541)",
         "ph_wiki": "phone wiki dump (2,082)", "ph_code": "phone code dump (2,340)"}


def mb(tn, gb, lab, key="mb_per_token"):
    v = R.get(tn, {}).get(gb, {}).get(lab)
    return None if v is None or key not in v else v[key]


def f(x, d=0):
    return "" if x is None else (f"{x:.{d}f}" if d else f"{x:.0f}")


def table(texts, cols, title, key="mb_per_token", d=0, rel=None, sizes=SIZES):
    print(f"\n### {title}\n")
    for tn in texts:
        if tn not in R:
            continue
        print(f"\n**{NAMES.get(tn, tn)}**\n")
        print("| GB (slots) | " + " | ".join(c[0] for c in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for gb in sizes:
            if gb not in R[tn]:
                continue
            row = []
            base = mb(tn, gb, rel) if rel else None
            for _, lab in cols:
                v = mb(tn, gb, lab, key)
                s = f(v, d)
                if rel and v is not None and base and lab != rel:
                    s += f" ({100 * (v / base - 1):+.0f}%)"
                row.append(s)
            print(f"| {gb} ({SLOTS[gb]}) | " + " | ".join(row) + " |")


MAIN = ["pitt", "coding", "agent"]
OTHER = ["code_text", "prose_text", "ph_chat", "ph_wiki", "ph_code"]
ALL = MAIN + OTHER

def headline(texts=MAIN, sizes=SIZES):
    cols = [("default: decay hl 8", "decay:8"), ("best decay hl (which)", None), ("LRU", "lru"), ("ARC", "arc"),
            ("learned (cache-only)", "learned"), ("hazard, one pool", "hazard"), ("**hazard + uniform caps**", "hazard|caps_uniform"),
            ("la 40 (1 token exact, bound)", "la:40|decay"), ("Belady (bound)", "belady")]
    print("| text | GB | " + " | ".join(c[0] for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for tn in texts:
        for gb in sizes:
            d = R.get(tn, {}).get(gb, {})
            if "decay:8" not in d:
                continue
            base = d["decay:8"]["mb_per_token"]
            row = []
            for _, lab in cols:
                if lab is None:
                    hs = [(d[k]["mb_per_token"], k.split(":")[1]) for k in d if k.startswith("decay:") and "|" not in k]
                    v, h = min(hs)
                    row.append(f"{v:.0f} (hl {h}, {100 * (v / base - 1):+.0f}%)")
                    continue
                v = d.get(lab, {}).get("mb_per_token")
                row.append("" if v is None else (f"{v:.0f}" if lab == "decay:8" else f"{v:.0f} ({100 * (v / base - 1):+.0f}%)"))
            print(f"| {tn} | {gb} | " + " | ".join(row) + " |")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "headline":
    headline(MAIN)
    print()
    headline(OTHER, ["0.8", "1.27", "2.0", "3.3"])
elif __name__ == "__main__":
    base_cols = [("decay hl 8 (default)", "decay:8"), ("LRU", "lru"), ("LFU", "lfu"), ("LFU+hist", "lfuh"),
                 ("decay+hist", "decayh:8"), ("ARC", "arc"), ("2Q", "2q"), ("S3-FIFO", "s3fifo"),
                 ("hazard (pooled)", "hazard"), ("hazard + keep 31", "hazard_drop31"), ("learned (router feats 0)", "learned"),
                 ("learned v2 (router feats 0)", "learned2"), ("Belady", "belady")]
    table(ALL, base_cols, "Online policies, one pool (engine step semantics); MB/token, % vs the default", rel="decay:8")
    hl_cols = [(f"hl {h}", f"decay:{h}") for h in (1, 2, 4, 8, 16, 32, 64)] + \
              [(f"hist hl {h}", f"decayh:{h}") for h in (2, 8, 32, 128)]
    table(ALL, hl_cols, "Decayed count: half-life sweep (tokens); MB/token")
    caps_cols = [("decay pooled", "decay:8"), ("decay uniform", "decay:8|caps_uniform"), ("decay table", "decay:8|caps_table"),
                 ("LRU pooled", "lru"), ("LRU uniform", "lru|caps_uniform"), ("LRU table", "lru|caps_table"),
                 ("hazard pooled", "hazard"), ("hazard uniform", "hazard|caps_uniform"), ("hazard table", "hazard|caps_table"),
                 ("phone: hazard, table, no drop", "phone_hazard_nodrop"), ("phone: hazard, table, keep 31", "phone_hazard_keep31"),
                 ("phone: LRU, table, keep 31", "phone_lru_keep31"), ("Belady pooled", "belady")]
    table(ALL, caps_cols, "One pool vs per-layer caps (uniform; the phone's LayerCapsTable shape); MB/token, % vs the default",
          rel="decay:8")
    la_cols = [("default", "decay:8")] + [(f"la {k}", f"la:{k}|decay") for k in (2, 4, 8, 40, 80, 160)] + \
              [("la 2 / LRU", "la:2|lru"), ("la 40 / LRU", "la:40|lru"),
               ("names exact lag 2", "nm:exact:2|decay"), ("names exact lag 4", "nm:exact:4|decay"),
               ("names r8 lag 2 top 8", "nm:r8:2:8|decay"), ("names r8 top 12", "nm:r8:2:12|decay"),
               ("names r8 top 16", "nm:r8:2:16|decay"),
               ("la 2 / hazard", "la:2|hazard"), ("la 40 / hazard", "la:40|hazard"), ("names exact 2 / hazard", "nm:exact:2|hazard"),
               ("names r8 12 / hazard", "nm:r8:2:12|hazard"), ("Belady", "belady")]
    table(ALL, la_cols, "Knowledge of the future: Belady restricted to k layers (la k, exact: upper bound) and names inside the "
          "token (exact = upper bound; r8 = the trained k2 router's recorded top N); base = decay unless named; MB/token",
          rel="decay:8")
    es_cols = [(c, l) for c, l in [("default", "decay:8"), ("LRU", "lru"), ("hazard", "hazard"), ("hazard table", "hazard|caps_table"),
                                    ("names r8 12", "nm:r8:2:12|decay"), ("Belady", "belady")]]
    table(["agent", "code_text", "prose_text"], es_cols,
          "Cache only: share of the misses the trained router named in its top 8 (could be read early)", key="miss_in_named8", d=3)
    table(["agent", "code_text", "prose_text"], es_cols, "... and in its top 12", key="miss_in_named12", d=3)
    pf_cols = []
    for p in ("decay:8", "lru", "hazard"):
        for N in ("8", "12"):
            pf_cols.append((f"{p} r8 top {N}: total", f"pf_r8_{N}|{p}"))
        pf_cols.append((f"{p} exact 8: total", f"pf_exact_8|{p}"))
    table(MAIN + ["code_text", "prose_text"], pf_cols, "Named early (lag 2): total MB/token = early + spot", key="mb_per_token")
    table(MAIN + ["code_text", "prose_text"], pf_cols, "Named early: of which read on the spot (MB/token)", key="mb_spot")
    table(MAIN + ["code_text", "prose_text"], pf_cols, "Named early: named records never used before eviction (records/token)",
          key="wasted_per_token", d=1)
    tier_cols = []
    for p in ("decay:8", "lru", "hazard", "arc", "belady"):
        tier_cols.append((f"{p} excl", f"A_excl|{p}"))
        tier_cols.append((f"{p} incl", f"A_incl|{p}"))
    table(ALL, tier_cols, "Two tiers: 1.27 GB policy tier + 2.1 GB LRU tier; drive MB/token", key="drive_mb_per_token",
          sizes=["1.27"])
