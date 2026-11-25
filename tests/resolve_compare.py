"""Two builds resolve a family's published manifest to the same run settings, in every front end.

  python3 tests/resolve_compare.py TINY_MODEL PUBLISHED_MANIFEST OUT --a BIN_A --b BIN_B      (the standard library only)

TINY_MODEL: a tiny model folder of the family (tests/tiny_all.sh's OUT/<family>/model, engine-ready files derived);
PUBLISHED_MANIFEST: the family's moved-store.json as the weight repo publishes it. Three folders are made beside each
other in OUT (hard links to the tiny model's files): "published" (the tiny manifest with the published `modes`,
`profiles`, `defaults`, `drafting` and cross-token routers), "no_profiles" (the tiny manifest without those keys, as an
older manifest) and "no_manifest" (no moved-store.json at all; the run asks for a profile). For every folder, profile
(mac, phone, phone-small) and mode (the profile's, exact, fast), each front end of each build (dmeng, the C API through
dm_gate, dmserve) prints its "resolved" settings; the C API also runs as on an iPhone (DMENG_TEST_AVAIL_OPEN: the memory
iOS gives the app) at 6 and 3.5 GB, where the expert slab is sized from it. The two builds' settings are compared key by
key (keys only one build prints are listed apart; an automatic cache's size, which follows free memory, is compared as
"auto"). Prints one line per difference and a summary; exit 0 when the builds agree.
"""
import argparse, json, os, shutil, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("tiny")
ap.add_argument("published")
ap.add_argument("out")
ap.add_argument("--a", required=True, help="the first build's folder (dmeng, dmserve, dm_gate)")
ap.add_argument("--b", required=True, help="the second build's folder")
ap.add_argument("--ignore", default="", help="comma-separated keys left out of the comparison (listed as such)")
a = ap.parse_args()
TINY, OUT = os.path.abspath(a.tiny), os.path.abspath(a.out)
os.makedirs(OUT, exist_ok=True)
man = json.load(open(os.path.join(TINY, "moved-store.json")))
pub = json.load(open(a.published))
DROP = {"parts", "bytes", "bytes_rule", "optional", "bytes_with_optional", "optional_note", "experts", "status"}


def folder(name, m):
    d = os.path.join(OUT, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(TINY, d, copy_function=os.link, ignore=shutil.ignore_patterns("prefix", "checked", "moved-store.json"))
    if m is not None:
        json.dump(m, open(os.path.join(d, "moved-store.json"), "w"), indent=1)
    else:  # no manifest: the engine reads the resident file the folder holds under its old name
        res = [f for f in os.listdir(os.path.join(d, ".dmeng")) if f.startswith("resident_q8.")][0]
        os.link(os.path.join(d, ".dmeng", res), os.path.join(d, "dmeng_resident.bin"))
    return d


p = dict(man)
for k in ("modes", "defaults", "drafting", "cross_token_routers"):
    p.pop(k, None)
    if k in pub:
        p[k] = pub[k]
p["profiles"] = {n: {k: v for k, v in pv.items() if k not in DROP} for n, pv in pub["profiles"].items()}
o = {k: v for k, v in man.items() if k not in ("modes", "profiles", "defaults", "drafting", "cross_token_routers")}
dirs = {"published": folder("published", p), "no_profiles": folder("no_profiles", o), "no_manifest": folder("no_manifest", None)}
ids = os.path.join(OUT, "ids.txt")
open(ids, "w").write(",".join(str(10 + i) for i in range(8)))


def run(cmd, env=None, inp=None):
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})}, input=inp, timeout=300)


def resolved(bin_dir, front, d, prof, mode, avail=None):
    j = os.path.join(OUT, "r.json")
    if os.path.exists(j):
        os.remove(j)
    md = ["--mode", mode] if mode else []
    if front == "dmeng":
        r = run([f"{bin_dir}/dmeng", "--model", d, "--profile", prof, *md, "--prompt-file", ids, "--n", "4", "--threads", "2",
                 "--draft", "lookup", "--json", j])
        if r.returncode:
            return {"error": (r.stdout + r.stderr)[-200:]}
        js = json.load(open(j))
        out = dict(js["resolved"])
        cp = (js.get("draft") or {}).get("cost_prior")
        if cp is not None:
            out["draft_cost_prior"] = [round(x, 4) for x in cp]
        return out
    if front == "capi":
        r = run([f"{bin_dir}/dm_gate", "--model", d, "--profile", prof, *md, "--prompt-ids", ids, "--n", "4", "--threads", "2",
                 "--cache-gb", "auto", "--json", j], env={"DMENG_TEST_AVAIL_OPEN": str(int(avail))} if avail else None)
        if r.returncode:
            return {"error": (r.stdout + r.stderr)[-200:]}
        st = json.load(open(j))["stats"]
        out = dict(st["resolved"])
        if avail:
            out["slab_auto_gb"] = round(st.get("policy", {}).get("slab_auto_gb", -1), 6)
        return out
    r = run([f"{bin_dir}/dmserve", "--model", d, "--profile", prof, *md, "--threads", "2"], inp="")
    ready = [l for l in r.stdout.splitlines() if l.startswith("READY ")]
    return json.loads(ready[0][6:])["resolved"] if ready else {"error": (r.stdout + r.stderr)[-200:]}


ignore = set(x for x in a.ignore.split(",") if x)
only_keys = ("draft_cost_prior", "cache_start", "slab_auto_gb", "profile", "routers", "names", "mode", "cross_token")


def norm(g):
    g = dict(g)
    if g.get("cache_auto") and "cache_gb" in g:
        g["cache_gb"] = "auto"
    return g


diffs, only, cells, errs, seen = [], {}, 0, [], {}
for fname, d in dirs.items():
    for prof in ("mac", "phone", "phone-small"):
        for mode in ("", "exact", "fast"):
            fronts = [("dmeng", None), ("capi", None), ("dmserve", None), ("capi", 6.0e9), ("capi", 3.5e9)]
            for fe, av in fronts:
                ga, gb = norm(resolved(a.a, fe, d, prof, mode, av)), norm(resolved(a.b, fe, d, prof, mode, av))
                cells += 1
                tag = f"{fname} {prof} mode={mode or 'profile'} {fe}{' iOS %.1f GB' % (av / 1e9) if av else ''}"
                for nm, g in (("a", ga), ("b", gb)):
                    if "error" in g:
                        errs.append(f"{tag} build {nm}: {g['error'][-160:]!r}")
                    for k in only_keys:
                        if k in g:
                            seen.setdefault(k, set()).add(json.dumps(g[k]))
                for k in sorted(set(ga) | set(gb)):
                    if k in ignore:
                        continue
                    if k not in ga or k not in gb:
                        only.setdefault(k, set()).add("a" if k in ga else "b")
                        continue
                    if json.dumps(ga[k], sort_keys=True) != json.dumps(gb[k], sort_keys=True):
                        diffs.append(f"{tag}: {k}: a {json.dumps(ga[k])} | b {json.dumps(gb[k])}")
for l in diffs:
    print("DIFF", l)
for l in errs:
    print("ERROR", l)
for k, v in sorted(seen.items()):
    print(f"VALUES {k}: {' | '.join(sorted(v))}")
for k, s in sorted(only.items()):
    print(f"ONLY {'/'.join(sorted(s))}: {k}")
print(f"RESOLVE COMPARE {'SAME' if not diffs else 'DIFFERENT'}: {cells} cells, {len(diffs)} differences, {len(errs)} runs refused, {len(only)} keys printed by one build only"
      + (f", ignored: {','.join(sorted(ignore))}" if ignore else ""))
sys.exit(1 if diffs else 0)
