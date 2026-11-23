"""The three phone slab sizes of INK-PHONE.md section 1 (1.15 footprint-equal, 1.70 C API rule, 2.35 zero margin):
hazard, two pools at 0.2, loop hold off; Exact on four texts, Fast (drop 0.1) on the agent text.
  engine/.venv/bin/python run_phone_budgets.py sim_exact.json"""
import json, sys
from multiprocessing import Pool
import ink_phone_sim as S

if __name__ == "__main__":
    specs = [(t, gb, "hazard", "share0.2", "off", False, None) for t in ("pitt", "code", "edit", "agent") for gb in (1.15, 1.7, 2.35)]
    specs += [("agent", gb, "hazard", "share0.2", "off", False, 0.1) for gb in (1.15, 1.7, 2.35)]
    with Pool(3) as p:
        rows = p.map(S.job, specs)
    json.dump({"rows": rows}, open(sys.argv[1], "w"), indent=1)
