#!/usr/bin/env python3
"""Download a model for this machine from the Hugging Face Hub: the files its manifest (moved-store.json) lists for one
profile, and nothing else.

  python3 python/fetch.py MODEL [DIR] [--profile mac] [--list]

MODEL    35b, inkling or 180b, or a weight repo id (owner/name)
DIR      where the files go (default: model)
--list   print the files and their total size, download nothing

Needs huggingface_hub (pip install huggingface_hub). A download that stops can be run again: files already there are
kept.
"""
import argparse, json, os, sys

REPOS = {
    "35b": "jalenluorion/Qwen3.6-35B-A3B-prefetchable-store",
    "inkling": "jalenluorion/Inkling-2-Mini-prefetchable-store",
    "180b": "jalenluorion/Qwen3.8-Flash-Next-prefetchable-store",
}


def main():
    ap = argparse.ArgumentParser(prog="python/fetch.py", description=__doc__.split("\n")[0])
    ap.add_argument("model", help="35b | inkling | 180b, or a weight repo id (owner/name)")
    ap.add_argument("dir", nargs="?", default="model", help="where the files go (default: model)")
    ap.add_argument("--profile", default="mac", help="the manifest profile whose files to fetch (default: mac)")
    ap.add_argument("--list", action="store_true", help="print the files and their total size, download nothing")
    a = ap.parse_args()
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        sys.exit("python/fetch.py needs huggingface_hub: pip install huggingface_hub")
    repo = REPOS.get(a.model.lower(), a.model)
    if "/" not in repo:
        sys.exit(f"unknown model '{a.model}': use one of {', '.join(REPOS)} or a weight repo id (owner/name)")
    manifest = json.load(open(hf_hub_download(repo, "moved-store.json", local_dir=None if a.list else a.dir)))
    profiles = manifest.get("profiles") or {}
    if a.profile not in profiles:
        sys.exit(f"{repo} has no profile '{a.profile}' (it has: {', '.join(profiles) or 'none'})")
    parts = profiles[a.profile].get("parts") or []
    total = sum(p["bytes"] for p in parts)
    print(f"{repo}, profile {a.profile}: {len(parts)} files, {total / 1e9:.2f} GB")
    if a.list:
        for p in parts:
            print(f"  {p['bytes'] / 1e9:9.3f} GB  {p['path']}")
        return
    snapshot_download(repo, local_dir=a.dir, allow_patterns=[p["path"] for p in parts] + ["moved-store.json"])
    short = [p["path"] for p in parts if not os.path.exists(os.path.join(a.dir, p["path"])) or os.path.getsize(os.path.join(a.dir, p["path"])) != p["bytes"]]
    if short:
        sys.exit("incomplete: " + ", ".join(short[:8]) + (" ..." if len(short) > 8 else "") + "\nrun the same command again")
    print(f"done: {a.dir}\n  build/dmeng --model {a.dir} --chat --prompt \"Write a haiku about rivers.\" --n 200")


if __name__ == "__main__":
    main()
