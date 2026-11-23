"""The engine gate runs (run_gate.sh) against the references and the simulator: python3 summary.py > summary.md"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ENG = "<work>/engine"
REF = {"pitt": ENG + "/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids", "code": ENG + "/results/stage4/runs/dc_off_code.ids"}
SIM = json.load(open(os.path.join(HERE, "..", "results.json")))
CELLS = [("pitt_def_1.27", "pitt", "1.27", "decay:8"), ("pitt_hzcaps_1.27", "pitt", "1.27", "hazard|caps_uniform"),
         ("code_def_1.27", "coding", "1.27", "decay:8"), ("code_hzcaps_1.27", "coding", "1.27", "hazard|caps_uniform"),
         ("pitt_def_3.3", "pitt", "3.3", "decay:8"), ("pitt_hz_3.3", "pitt", "3.3", "hazard"),
         ("code_def_3.3", "coding", "3.3", "decay:8"), ("code_hz_3.3", "coding", "3.3", "hazard")]


def ids(p):
    return [int(x) for x in open(p).read().replace("\n", ",").split(",") if x.strip()]


print("| run | slots | ids vs reference | hit rate engine / sim | MB/token engine / sim | compute ms/token | policy ms/token | "
      "wait ms/token | tok/s | others' CPU % (guard) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, tn, gb, lab in CELLS:
    jp = os.path.join(HERE, name + ".json")
    if not os.path.exists(jp):
        continue
    j = json.load(open(jp))
    a, b = ids(os.path.join(HERE, name + ".ids")), ids(REF["pitt" if tn == "pitt" else "code"])
    n = min(len(a), len(b))
    diff = next((i for i in range(n) if a[i] != b[i]), None)
    idv = f"identical ({n})" if diff is None and len(a) == len(b) else f"first differs at {diff} (of {n})"
    s = SIM[tn][gb][lab]
    g = re.search(r"^GUARD (.*)$", open(os.path.join(HERE, name + ".log")).read(), re.M)
    oth = json.loads(g.group(1)).get("others_cpu_pct_mean") if g else None
    print(f"| {name} | {j['slots'][0] if isinstance(j['slots'], list) else j['slots']} | {idv} | {j['hit_rate']:.4f} / {s['hit']:.4f} | {j['mb_read_per_token']:.1f} / "
          f"{s['mb_per_token']:.1f} | {j['compute_ms_per_token']:.1f} | {j.get('policy_ms_per_token', 0):.2f} | "
          f"{j['wait_ms_per_token']:.1f} | {j['tok_s']:.2f} | {oth} |")
