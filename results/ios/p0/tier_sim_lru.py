"""LRU at record granularity (a stand-in for the OS file cache's own replacement; the OS works on 16 KB pages and its
policy is not documented), same inputs as tier_sim.py -> tier_sim_lru.json"""
import json, os, sys
from collections import OrderedDict
sys.argv = sys.argv[:1]
__file__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tier_sim.py")
src = open(__file__).read().split("SIZES =")[0]
exec(src)
SIZES = [0.8, 1.27, 1.6, 2.0, 2.5, 3.3, 4.0, 5.0, 6.6, 8.0]
out = {}
for name, (rel, start) in RECS.items():
    rec = load_rec(os.path.join(ENG, rel))
    out[name] = {}
    for gb in SIZES:
        slots = 720 if gb == 1.27 else int(gb * 1e9 // REC_BYTES)
        c = OrderedDict(); hit = tot = 0
        for t in range(rec.shape[0]):
            for L in range(NL):
                for e in rec[t, L].tolist():
                    k = (L, e)
                    if t >= start:
                        tot += 1; hit += k in c
                    if k in c: c.move_to_end(k)
                    else:
                        c[k] = 1
                        if len(c) > slots: c.popitem(last=False)
        h = hit / tot
        out[name][str(gb)] = {"hit": round(h, 4), "mb_per_token": round((1 - h) * 320 * REC_BYTES / 1e6, 1)}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(HERE, "tier_sim_lru.json"), "w"), indent=1)
