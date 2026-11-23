"""(c) of the phone-small plan: 4-bit in-memory weights (0.713 GB, frees 0.554 GB) and layers 20-23 as NVFP4 records, so
the slab at the same footprint is ~1.70 GB (1.15 + 0.55; 1.8 also run) and every record is
4.49 MB: hazard, one pool, loop hold off; Exact on four texts, Fast (drop 0.1) on the agent text.
  engine/.venv/bin/python run_nv4.py sim_nv4.json"""
import json, sys
from multiprocessing import Pool
import ink_phone_sim as S

if __name__ == "__main__":
    gbs = (1.15, 1.7, 1.8, 2.25, 2.9)
    specs = [(t, gb, "hazard", "nv", "off", False, None) for t in ("pitt", "code", "edit", "agent") for gb in gbs]
    specs += [("agent", gb, "hazard", "nv", "off", False, 0.1) for gb in gbs]
    with Pool(2) as p:
        rows = p.map(S.job, specs)
    json.dump({"rows": rows}, open(sys.argv[1], "w"), indent=1)
