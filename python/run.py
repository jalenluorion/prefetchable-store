"""Text in, text out for either model family: tokenise (optionally render the release's chat template), run the compiled
engine (tools/dmeng: the whole token loop, prefill included, is C++), print the text and dmeng's statistics line. Every
flag after the known ones goes to dmeng unchanged.

  python python/run.py --model MODEL --prompt "The history of the city of Pittsburgh begins with" --n 120 --cache-gb 5
  python python/run.py --model MODEL --chat --prompt "Explain in a short paragraph why the sky is blue." --n 200
  python python/run.py --model MODEL --prompt-ids-only [--chat] --prompt "..."      # print the ids and exit
  python python/run.py --model MODEL --prompt "..." --n 200 --draft lookup --early 2  # any dmeng flag passes through

The family comes from MODEL/moved-store.json ("family"), else from the resident file's cfg `model` (--resident F, else
MODEL/dmeng_resident.bin), else --family. --chat renders the template as the server does (python/serve/dmserve.py): the
family's format module python/serve/<family>/chat_format.py if there is one, else python/serve/generic/chat_format.py
(MODEL/chat_template.jinja and the release's response format); one user turn with the generation prompt;
--reasoning-effort goes in as a request's reasoning_effort (template_kwargs of that module: none turns thinking off).
--out PREFIX writes PREFIX.json (dmeng's statistics + the text) and PREFIX.ids. --bin (or DMENG_BIN) picks the binary,
default build/dmeng of this repo. MODEL may also come from DM_MODEL.

Python dependencies: tokenizers (the release's tokenizer.json); jinja2 for --chat. Nothing else (no torch, no MLX).
"""
import argparse, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FAMILIES = ("qwen3_5_moe", "inkling2_loop", "qwen4_exp")

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
ap.add_argument("--model", default=os.environ.get("DM_MODEL"), help="the model folder (default: $DM_MODEL)")
ap.add_argument("--family", default=None, choices=FAMILIES, help="override the detected family (also passed to dmeng)")
ap.add_argument("--prompt", default="The history of the city of Pittsburgh begins with")
ap.add_argument("--chat", action="store_true", help="render the release's chat template (one user turn; needs jinja2)")
ap.add_argument("--reasoning-effort", default=None, help="the template's thinking switch (see above)")
ap.add_argument("--prompt-ids-only", action="store_true", help="print the prompt's token ids and exit")
ap.add_argument("--out", default=None, help="write PREFIX.json and PREFIX.ids")
ap.add_argument("--bin", default=os.environ.get("DMENG_BIN", os.path.join(REPO, "build", "dmeng")))
a, rest = ap.parse_known_args()
if not a.model:
    ap.error("--model MODEL (or DM_MODEL) is required")
a.model = os.path.abspath(a.model)


def resident_family(path):
    """cfg `model` of a DMENGRES1 resident file (its text header), or None."""
    try:
        with open(path, "rb") as f:
            first = f.readline().decode("ascii", "replace").split()
            if len(first) != 2 or first[0] != "DMENGRES1":
                return None
            head = f.read(int(first[1]) - f.tell()).decode("utf-8", "replace")
    except (OSError, ValueError):
        return None
    for line in head.splitlines():
        p = line.split()
        if p == ["end"]:
            break
        if len(p) >= 3 and p[0] == "cfg" and p[1] == "model":
            return p[2]
    return None


def detect_family():
    if a.family:
        return a.family
    man = os.path.join(a.model, "moved-store.json")
    if os.path.exists(man):
        fam = json.load(open(man)).get("family")
        if fam:
            return fam
    res = rest[rest.index("--resident") + 1] if "--resident" in rest[:-1] else os.path.join(a.model, "dmeng_resident.bin")
    fam = resident_family(res)
    if fam:
        return fam
    raise SystemExit(f"cannot tell the model family of {a.model} (no moved-store.json, no resident cfg `model`): pass --family")


def render(family, prompt):
    """the server's rendering of one user turn (dmserve.py: the format module by folder, its template_kwargs)"""
    serve = os.path.join(REPO, "python", "serve")
    fmt = os.path.join(serve, family) if os.path.exists(os.path.join(serve, family, "chat_format.py")) else os.path.join(serve, "generic")
    sys.path.insert(0, fmt)
    import chat_format as CF  # noqa: E402

    kw, _budget = CF.template_kwargs({"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {})
    return CF.Tok(a.model).render([{"role": "user", "content": prompt}], **kw)


family = detect_family()
from tokenizers import Tokenizer  # noqa: E402

tk = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
text = render(family, a.prompt) if a.chat else a.prompt
ids = tk.encode(text, add_special_tokens=False).ids
if a.prompt_ids_only:
    print(",".join(map(str, ids)))
    raise SystemExit(0)
if a.out:
    out_ids = a.out + ".ids"
else:
    fd, out_ids = tempfile.mkstemp(prefix="dmeng_", suffix=".ids")
    os.close(fd)
cmd = [a.bin, "--model", a.model, "--prompt-ids", ",".join(map(str, ids)), "--out-ids", out_ids, *rest]
if a.family:
    cmd += ["--family", a.family]
r = subprocess.run(cmd, capture_output=True, text=True)
sys.stderr.write(r.stderr)
if r.returncode:
    raise SystemExit(f"dmeng exited {r.returncode}")
stats = json.loads(r.stdout.strip().splitlines()[-1])
gen = [int(t) for t in open(out_ids).read().strip().split(",") if t]
stats["text"] = tk.decode(gen, skip_special_tokens=False)
stats["prompt"] = text
print(text + stats["text"])
print(json.dumps({k: v for k, v in stats.items() if k not in ("text", "prompt")}))
if a.out:
    json.dump(stats, open(a.out + ".json", "w"), indent=1)
elif os.path.exists(out_ids):
    os.remove(out_ids)
