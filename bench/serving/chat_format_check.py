#!/usr/bin/env python3
"""Gate for a model folder's chat format in the server front end (python/serve/generic/chat_format.py), no model run:
  1. render: every recorded OpenCode request rendered by the front end == transformers' apply_chat_template on the same
     folder (token ids), and the message boundaries are token positions of that rendering;
  2. parse: every recorded assistant turn (model-format text) parses to reasoning / content / tool calls that the
     template renders back to the same text; OpenCode's recorded tool calls survive render -> parse;
  3. streaming: the incremental split, fed token by token as the server decodes, sends exactly the final parse.
Run with a Python that has transformers (local/inkling/.venv):
  python bench/serving/chat_format_check.py --model-dir MODEL --sessions S1 S2 --turns FILE... > out.json"""
import argparse, glob, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--sessions", nargs="*", default=[])
ap.add_argument("--turns", nargs="*", default=[], help="text files in the model's own format with assistant turns in them")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "serve", "generic"))
import chat_format as CF  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = CF.Tok(a.model_dir)
CF.bind(tok)
ref = AutoTokenizer.from_pretrained(a.model_dir)
EOS = tok.eos_token
res = {"render": [], "parse_recorded": [], "parse_opencode_calls": [], "stream": []}
ok = True


def ref_ids(msgs, tools, kw):
    r = ref.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=True, **kw)
    if isinstance(r, dict) or hasattr(r, "keys"):
        r = r["input_ids"]
    return list(r)


def stream_check(turn_text, th=False):
    ids = tok.encode(turn_text)
    got = {"reasoning_content": "", "content": ""}
    sp = CF.StreamSplitter(th, lambda f, t: got.__setitem__(f, got[f] + t))
    text = ""
    for i in range(1, len(ids) + 1):
        full = tok.decode(ids[:i], skip_special_tokens=False)
        sp.feed(full[len(text):])
        text = full
    r, c, _ = CF.parse_turn(text, th, ())
    return got["reasoning_content"] == (r or "") and got["content"] == (c or "")


# 1. render, on the recorded OpenCode requests
for d in a.sessions:
    for f in sorted(glob.glob(os.path.join(d, "[0-9][0-9][0-9].json"))):
        b = json.load(open(f))["body"]
        kw, _ = CF.template_kwargs(b)
        msgs = CF.to_template(b["messages"])
        ids, marks = tok.prompt_ids(msgs, b.get("tools"), **kw)
        rids = ref_ids(msgs, b.get("tools"), kw)
        same = ids == rids
        ok &= same
        res["render"].append({"session": os.path.basename(d.rstrip("/")), "file": os.path.basename(f), "tokens": len(ids), "ref_tokens": len(rids),
                              "identical": same, "marks": len(marks), "messages": len(msgs)})
        # OpenCode's own tool calls: rendered as the model would write them, parsed back
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant" or not m.get("tool_calls"):
                continue
            pre = tok.render(msgs[:i], b.get("tools"), add_generation_prompt=True, **kw)
            full = tok.render(msgs[: i + 1], b.get("tools"), add_generation_prompt=False, **kw)
            if not full.startswith(pre):
                res["parse_opencode_calls"].append({"file": os.path.basename(f), "msg": i, "error": "prefix"})
                ok = False
                continue
            turn = full[len(pre):]
            turn = turn[: turn.rfind(EOS)] if EOS in turn else turn
            th = tok.thinking(tok.encode(pre))  # the prompt opened a field (Qwen3.6: reasoning)
            r, c, calls = CF.parse_turn(turn, th, b.get("tools") or [])
            want = [(tc["function"]["name"], tc["function"]["arguments"]) for tc in m["tool_calls"]]
            have = [(tc["function"]["name"], json.loads(tc["function"]["arguments"])) for tc in calls]
            same = want == have and (r or "") == (m.get("reasoning_content") or "") and c == (m.get("content") or "")
            st = stream_check(turn, th)
            ok &= same and st
            res["parse_opencode_calls"].append({"file": os.path.basename(f), "msg": i, "calls": len(calls), "identical": same, "stream_identical": st})

# 2. parse, on turns written in the model's own format
for f in a.turns:
    t = open(f).read()
    # an assistant turn: from the first <|message_model|> after the last non-model message, to the end-of-turn token
    pos = 0
    for chunk in t.split(EOS)[:-1]:
        start = pos
        pos += len(chunk) + len(EOS)
        last = max(chunk.rfind(h) for h in ("<|message_user|>", "<|message_tool|>", "<|message_system|>"))
        k = chunk.find("<|message_model|>", max(0, last))
        if k < 0:
            continue
        turn = chunk[k:] + EOS
        m = type("M", (), {"start": lambda self, g=0, v=start + k: v})()
        body = turn[: -len(EOS)]
        r, c, calls = CF.parse_turn(body, False, ())
        msg = CF.to_template([CF.to_message(r, c, calls)])[0]
        rer = tok.render([{"role": "user", "content": "x"}, msg], None, add_generation_prompt=False)
        u = tok.render([{"role": "user", "content": "x"}], None, add_generation_prompt=False)
        back = rer[len(u):]
        same = back == turn
        st = stream_check(body)
        ok &= same and st
        res["parse_recorded"].append({"file": os.path.basename(f), "at": m.start(1), "calls": [tc["function"]["name"] for tc in calls], "reasoning_chars": len(r or ""),
                                      "content_chars": len(c or ""), "rerender_identical": same, "stream_identical": st,
                                      **({} if same else {"orig": turn[:300], "back": back[:300]})})
res["passed"] = ok
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
