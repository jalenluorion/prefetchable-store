"""Text in, text out for either model family: tokenise (optionally render the release's chat template), run the compiled
engine (tools/dmeng: the whole token loop, prefill included, is C++), print the text and dmeng's statistics line. Every
flag after the known ones goes to dmeng unchanged.

  python python/run.py --model MODEL --prompt "The history of the city of Pittsburgh begins with" --n 120 --cache-gb 5
  python python/run.py --model MODEL --chat --prompt "Explain in a short paragraph why the sky is blue." --n 200
  python python/run.py --model MODEL --prompt-ids-only [--chat] --prompt "..."      # print the ids and exit
  python python/run.py --model MODEL --prompt "..." --n 200 --draft lookup --early 2  # any dmeng flag passes through

The family comes from MODEL/moved-store.json ("family"), else from the resident file's cfg `model` (--resident F, else
MODEL/dmeng_resident.bin), else --family. --chat renders the template the release ships:
  qwen3_5_moe    python/serve/qwen3_5_moe/chat_format.py (the server's), one user turn with the generation prompt;
                 --reasoning-effort none turns thinking off (enable_thinking=false), anything else keeps the release's default
  inkling2_loop  MODEL/chat_template.jinja, one user turn, rendered as the MLX runner renders it (plain jinja2, the
                 {% generation %} block renders its body, no generation prompt); --reasoning-effort is the template's
                 reasoning_effort (default none)
  qwen4_exp      python/serve/qwen4_exp/chat_format.py (the server's: MODEL/chat_template.jinja), one user turn with the
                 generation prompt; --reasoning-effort xhigh (the release's default) | medium | low, none turns thinking off
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


def render_inkling(model_dir, messages, **kw):
    """The MLX runner's render_chat: plain jinja2, the {% generation %} block renders its body; no generation prompt."""
    import jinja2
    from jinja2.ext import Extension

    class Generation(Extension):
        tags = {"generation"}

        def parse(self, parser):
            next(parser.stream)
            return parser.parse_statements(["name:endgeneration"], drop_needle=True)

    def raise_exception(msg):
        raise ValueError(msg)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=[Generation, "jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda x, sort_keys=False, separators=None, **_: json.dumps(x, sort_keys=sort_keys, separators=separators, ensure_ascii=False)
    tpl = env.from_string(open(os.path.join(model_dir, "chat_template.jinja")).read())
    return tpl.render(messages=messages, **{k: v for k, v in kw.items() if v is not None})


def render(family, prompt):
    msgs = [{"role": "user", "content": prompt}]
    if family == "qwen3_5_moe":
        sys.path.insert(0, os.path.join(REPO, "python", "serve", "qwen3_5_moe"))
        import chat_format as CF  # noqa: E402

        kw, _budget = CF.effort(a.reasoning_effort)  # a thinking budget is the server's; here only the switch
        return CF.Tok(a.model).render(msgs, **kw)
    if family == "inkling2_loop":
        return render_inkling(a.model, msgs, reasoning_effort=a.reasoning_effort or "none")
    if family == "qwen4_exp":
        sys.path.insert(0, os.path.join(REPO, "python", "serve", "qwen4_exp"))
        import chat_format as CF  # noqa: E402

        return CF.Tok(a.model).render(msgs, **CF.effort_kwargs(a.reasoning_effort))
    raise SystemExit(f"--chat: no chat format for family {family}")


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
