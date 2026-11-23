"""The Inkling phone-small set (4-bit in-memory weights, NVFP4 layers 20-23): the phone's moved-store.json and the file list.
  python make_phone_set.py STAGED_MANIFEST PHONE_DIR OUT_MANIFEST OUT_LIST
STAGED_MANIFEST: the Hub-staged moved-store.json (profiles with parts); PHONE_DIR: the phone-layout folder holding
derived/resident_q4..., store_nv4/layer_20..23.bin and (links to) the rest. The phone-small profile is rewritten:
resident_q4, the experts override (layer_format all nvfp4_g16, layer_files 20-23 -> store_nv4/), parts with the new files
(sha256 computed here) in place of store/layer_20..23 and the 8-bit resident; bytes = the sum of the parts."""
import hashlib, json, os, sys

staged, pdir, out_m, out_l = sys.argv[1:5]
m = json.load(open(staged))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest()


ps = m["profiles"]["phone-small"]
q4 = "derived/resident_q4.dmres1.15a39fa8aa48.bin"
parts = []
for p in ps["parts"]:
    if p["path"] in ("LICENSE", "NOTICE.md"):
        parts.append(p)  # licence files: listed, small
        continue
    if p["path"] == "derived/resident_q8.dmres1.15a39fa8aa48.bin":
        f = os.path.join(pdir, q4)
        parts.append({"path": q4, "bytes": os.path.getsize(f), "sha256": sha(f)})
        continue
    if p["path"] in ["store/layer_%02d.bin" % L for L in (20, 21, 22, 23)]:
        n = "store_nv4/" + os.path.basename(p["path"])
        f = os.path.join(pdir, n)
        parts.append({"path": n, "bytes": os.path.getsize(f), "sha256": sha(f)})
        continue
    parts.append(p)
ps.update({"resident": "resident_q4", "precision": "4-bit", "precision_detail": "4-bit affine, groups of 64 (plain; a calibrated file replaces it later under the same name)",
           "experts": {"layer_format": ["nvfp4_g16"] * 24, "layer_files": {str(L): "store_nv4/layer_%02d.bin" % L for L in (20, 21, 22, 23)},
                       "note": "layers 20-23 re-quantised from the release's MXFP8 to NVFP4 (python/derive/ink_mx_to_nv.py: least-squares block scales); the release's files stay as they are for the other profiles"},
           "parts": parts, "bytes": sum(p["bytes"] for p in parts),
           "status": "approved for memory-limited phones (2026-11-23); agreement with the exact 8-bit engine prose / agent: Exact 0.923 / 0.940, Fast at a 1.7 GB slab 0.909 / 0.933"})
ps.pop("bytes_rule", None)
json.dump(m, open(out_m, "w"), indent=1)
with open(out_l, "w") as g:
    g.write("path\tbytes\tsha256\n")
    for p in parts:
        g.write(f"{p['path']}\t{p['bytes']}\t{p['sha256']}\n")
    g.write(f"moved-store.json\t{os.path.getsize(out_m)}\t{sha(out_m)}\n")
print(len(parts) + 1, "files,", sum(p["bytes"] for p in parts) + os.path.getsize(out_m), "bytes")
