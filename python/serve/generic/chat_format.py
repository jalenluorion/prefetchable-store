"""A model folder's chat format, read from the folder itself (no model-specific text): the tokenizer (tokenizer.json), the
chat template (tokenizer_config.json's chat_template, else chat_template.jinja), the stop tokens (generation_config.json)
and the response format (tokenizer_config.json's response_template: the fields of an assistant turn, their open / close
markers, repeats and joins, content types and the transform to OpenAI tool calls). Same interface as
python/serve/qwen3_5_moe/chat_format.py, so python/serve/dmserve.py serves any folder that ships these files; written for
Inkling 2 Mini (local/inkling/model), whose template renders reasoning, text and tool calls as special-token-delimited
segments and has no generation prompt (the model's turn starts with its own <|message_model|>).

Checked against the release's own rendering (transformers' apply_chat_template on the same folder) by
bench/serving/chat_format_check.py: token ids equal on recorded OpenCode requests; recorded turns parse and re-render to
the same text.

Template switches: a request's chat_template_kwargs pass through; reasoning_effort (or reasoning.effort) is passed as the
template variable of that name when the template uses one (Inkling: "Thinking effort level", default 0.9).
Message boundaries (checkpoint candidates) come from rendering each prefix of the message list: a boundary is where the
prefix's rendering stops agreeing with the whole conversation's, if a token starts there."""
import json, os, re, uuid

from tokenizers import Tokenizer

PARSE_KEEP_SPECIAL = True  # the turn's structure is in special tokens: parse the text with them kept


class Tok:
    def __init__(self, path, defaults=None):
        import jinja2
        import jinja2.ext

        self.tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        tc = json.load(open(os.path.join(path, "tokenizer_config.json")))
        self.chat_template = tc.get("chat_template")
        if not self.chat_template and os.path.exists(os.path.join(path, "chat_template.jinja")):
            self.chat_template = open(os.path.join(path, "chat_template.jinja")).read()
        if isinstance(self.chat_template, list):  # named templates: the default one
            self.chat_template = next((t["template"] for t in self.chat_template if t.get("name") == "default"), self.chat_template[0]["template"])
        tok_of = lambda t: t.get("content") if isinstance(t, dict) else t
        self.bos_token, self.eos_token = tok_of(tc.get("bos_token")), tok_of(tc.get("eos_token"))
        self.response_template = tc.get("response_template")
        if isinstance(self.response_template, str):
            self.response_template = json.loads(self.response_template)
        self.defaults = dict(defaults or {})

        class Generation(jinja2.ext.Extension):  # transformers' {% generation %}: the body renders unchanged
            tags = {"generation"}

            def parse(self, parser):
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
                return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)

            def _body(self, caller):
                return caller()

        env = jinja2.Environment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", Generation])

        def raise_exception(m):
            raise jinja2.TemplateError(m)

        env.globals["raise_exception"] = raise_exception
        # transformers' tojson
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self.tpl = env.from_string(self.chat_template)
        self.uses_effort = "reasoning_effort" in self.chat_template
        self.parser = Schema(self.response_template or {})
        # the generation prompt's text after the response's start anchor (Qwen3.6: "<think>\n", the reasoning opened by the
        # prompt): a turn generated after it is parsed as if that text came first. Inkling: none (no generation prompt).
        u = [{"role": "user", "content": "x"}]
        try:
            self.gp = self.render(u, None, add_generation_prompt=True)[len(self.render(u, None, add_generation_prompt=False)):]
        except Exception:
            self.gp = ""
        rt = self.response_template or {}
        anc = rt.get("start_anchor_pattern") or (re.escape(rt["start_anchor"]) if rt.get("start_anchor") else None)
        ms = list(re.finditer(anc, self.gp)) if anc and self.gp else []
        self.parser.lead = self.gp[ms[-1].end():] if ms else ""

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        kw = {**self.defaults, **{k: v for k, v in kw.items() if v is not None}}
        return self.tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, bos_token=self.bos_token, eos_token=self.eos_token, **kw)

    def prompt_ids(self, messages, tools=None, **kw):
        text = self.render(messages, tools, **kw)
        enc = self.tk.encode(text, add_special_tokens=False)
        ids, starts = enc.ids, {s: i for i, (s, e) in enumerate(enc.offsets)}
        marks = []
        for i in range(1, len(messages)):
            try:
                part = self.render(messages[:i], tools, add_generation_prompt=False, **{k: v for k, v in kw.items() if k != "add_generation_prompt"})
            except Exception:
                continue
            n = min(len(part), len(text))
            c = 0
            while c < n and part[c] == text[c]:
                c += 1
            if c == len(part) and c in starts and starts[c] > 0 and (not marks or starts[c] > marks[-1]):
                marks.append(starts[c])
        return ids, marks

    def thinking(self, ids):
        """does this prompt end with the generation prompt's opened field (parse_turn then reads the turn after it)?"""
        return bool(self.parser.lead) and self.decode(ids[-32:]).endswith(self.gp)


def template_kwargs(req, default_effort=None, tok=None):
    """(template variables, thinking budget = 0) from a request: chat_template_kwargs as given, plus reasoning_effort."""
    kw = dict(req.get("chat_template_kwargs") or {})
    eff = req.get("reasoning_effort")
    if not eff and isinstance(req.get("reasoning"), dict):
        eff = req["reasoning"].get("effort")
    eff = eff or default_effort
    if eff is not None and "reasoning_effort" not in kw:
        kw["reasoning_effort"] = eff
    return kw, 0


# -- the response format (tokenizer_config.json response_template)
def _alt(x):
    if x is None:
        return None
    xs = x if isinstance(x, list) else [x]
    return "|".join(re.escape(s) for s in xs)


class Field:
    def __init__(self, name, spec):
        self.name = name
        self.open = spec.get("open_pattern") or _alt(spec.get("open"))
        self.close = spec.get("close_pattern") or _alt(spec.get("close"))
        self.open_re = re.compile(self.open, re.S) if self.open else None
        self.close_re = re.compile(self.close, re.S) if self.close else None
        self.repeats = bool(spec.get("repeats"))
        self.join = spec.get("join", "")
        self.kind = spec.get("content", "text")
        self.args = spec.get("content_args") or {}
        self.transform = spec.get("transform")
        self.tag_re = re.compile(self.args["tag_pattern"], re.S) if self.args.get("tag_pattern") else None


class Schema:
    """parse(text) -> [(field name, segment text, open-match groups, closed?)] in order, from the response_template."""

    def __init__(self, rt):
        self.fields = [Field(k, v) for k, v in (rt.get("fields") or {}).items()]
        self.default = next((f for f in self.fields if f.open_re is None), None)  # text outside any opened field
        self.lead = ""
        self.open_lits = [l for l in (_literal_prefix(f.open) for f in self.fields if f.open) if l]

    def hold_from(self, text):
        """streaming: where the text outside any field must stop being sent, because what follows could still become
        another field's opening marker (e.g. "<tool_call>\n<function=rea" before its ">")"""
        if self.default is None:
            return len(text)
        segs = self.segments(text)
        if not segs or segs[-1][0] is not self.default or segs[-1][3]:
            return len(text)
        start = len(text) - len(segs[-1][1])
        cut = len(text)
        for lit in self.open_lits:
            k = text.find(lit, start)
            if k >= 0:
                cut = min(cut, k)
            for j in range(1, len(lit)):  # the text ends with the start of a marker
                if text.endswith(lit[:j]):
                    cut = min(cut, len(text) - j)
        return cut

    def segments(self, text):
        out, pos = [], 0
        opened = [f for f in self.fields if f.open_re is not None]
        while pos < len(text):
            best = None
            for f in opened:
                m = f.open_re.search(text, pos)
                if m and (best is None or m.start() < best[1].start()):
                    best = (f, m)
            if self.default is not None:
                end = best[1].start() if best else len(text)
                cm = self.default.close_re.search(text, pos, end) if self.default.close_re else None
                seg_end = cm.start() if cm else end
                if seg_end > pos:
                    out.append((self.default, text[pos:seg_end], {}, cm is not None))
                if cm and not best:
                    break
            if not best:
                break
            f, m = best
            cm = f.close_re.search(text, m.end()) if f.close_re else None
            seg = text[m.end(): cm.start() if cm else len(text)]
            out.append((f, seg, m.groupdict(), cm is not None))
            if not cm:
                break
            pos = max(cm.end() if cm.end() > cm.start() else cm.start(), m.end())
        return out


def _literal_prefix(pat):
    """the literal characters a regex starts with ("<tool_call>\\s*..." -> "<tool_call>")"""
    out, i = "", 0
    while i < len(pat):
        ch = pat[i]
        if ch == "\\" and i + 1 < len(pat) and not pat[i + 1].isalnum():
            out += pat[i + 1]
            i += 2
            continue
        if ch in ".^$*+?{}[]|()\\":
            if ch in "*?{" and out:  # the last char was optional
                out = out[:-1]
            break
        out += ch
        i += 1
    return out


def _lookup(path, env):
    v = env
    for p in path.split("."):
        v = v.get(p) if isinstance(v, dict) else None
    return v


def _fill(t, env):
    if isinstance(t, dict):
        return {k: _fill(v, env) for k, v in t.items()}
    if isinstance(t, str):
        m = re.fullmatch(r"\{([\w.]+)\}", t)
        if m:
            return _lookup(m.group(1), env)
        return re.sub(r"\{([\w.]+)\}", lambda mm: str(_lookup(mm.group(1), env)), t)
    return t


def call_id():
    return "call_" + uuid.uuid4().hex[:24]


def _xml_args(f, seg, specs, name):
    args = {}
    for pm in f.tag_re.finditer(seg):
        k, v = pm.group("key").strip(), pm.group("value")
        typ = (specs.get(name, {}).get(k) or {}).get("type")
        if typ in ("integer", "number", "boolean", "object", "array", "null") or (typ is None and v[:1] in "[{"):
            try:
                v = json.loads(v)
            except Exception:
                pass
        args[k] = v
    return args


PARSER = [None]


def parse_turn(text, thinking=False, tools=(), parser=None):
    """A generated turn (decoded with special tokens kept) -> (reasoning or None, content, OpenAI tool calls)."""
    sch = parser or PARSER[0]
    if thinking and sch.lead:
        text = sch.lead + text
    specs = {t.get("function", {}).get("name"): (t.get("function", {}).get("parameters") or {}).get("properties") or {} for t in tools or ()}
    vals, calls = {}, []
    for f, seg, groups, closed in sch.segments(text):
        if f.kind == "text":
            s = seg.strip() if f.args.get("strip", True) else seg  # the schema's default; Inkling's fields say strip: false
            if f.repeats or f.name not in vals:
                vals[f.name] = (vals[f.name] + f.join + s) if f.name in vals else s
            continue
        if f.kind == "json":
            try:
                content = json.loads(seg)
            except Exception:
                continue  # an unfinished call (cut by max_tokens): not a call
            env = {"content": content, **groups}
        elif f.kind == "xml-inline":
            env = {"content": _xml_args(f, seg, specs, (groups.get("name") or "").strip()), **{k: (v or "").strip() for k, v in groups.items()}}
        else:
            continue
        obj = _fill(f.transform, env) if f.transform else env["content"]
        if f.name == "tool_calls":
            fn = obj.get("function", {}) if isinstance(obj, dict) else {}
            args = fn.get("arguments")
            calls.append({"id": call_id(), "type": "function",
                          "function": {"name": fn.get("name") or "", "arguments": args if isinstance(args, str) else json.dumps(args if args is not None else {}, ensure_ascii=False)}})
    return vals.get("reasoning_content"), vals.get("content", ""), calls


def to_message(reasoning, content, calls):
    msg = {"role": "assistant", "content": content or ""}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg


def to_template(messages):
    """OpenAI messages -> what a chat template takes: tool-call arguments parsed to objects (ids kept: a template may name
    a tool result by the call it answers), text content parts joined, 'reasoning' accepted for reasoning_content."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                    except Exception:
                        pass
                tc["function"] = fn
                tc.setdefault("type", "function")
                tcs.append(tc)
            m["tool_calls"] = tcs
            m["content"] = m.get("content") or ""
        if m.get("role") == "assistant" and m.get("reasoning") and not m.get("reasoning_content"):
            m["reasoning_content"] = m.pop("reasoning")
        if isinstance(m.get("content"), list) and all(isinstance(p, dict) and p.get("type", "text") in ("text", "input_text") for p in m["content"]):
            m["content"] = "".join(p.get("text", "") for p in m["content"])
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


class StreamSplitter:
    """parse_turn incrementally: reasoning_content and content are sent as they grow (each a prefix of the final parse);
    tool calls are sent at the end from parse_turn on the whole text."""

    def __init__(self, thinking, send, parser=None):
        self.text, self.sent, self.send, self.parser = "", {"reasoning_content": "", "content": ""}, send, parser or PARSER[0]
        self.thinking = thinking

    def feed(self, piece):
        self.text += piece
        t = (self.parser.lead + self.text) if self.thinking and self.parser.lead else self.text
        cut = self.parser.hold_from(t) - (len(t) - len(self.text))
        r, c, _ = parse_turn(self.text[: max(0, cut)], self.thinking, (), self.parser)
        for field, v in (("reasoning_content", r or ""), ("content", c or "")):
            s = self.sent[field]
            if len(v) > len(s) and v.startswith(s):
                self.send(field, v[len(s):])
                self.sent[field] = v


def bind(tok):
    """the module-level parser for parse_turn / StreamSplitter (the front end calls this once with its Tok)"""
    PARSER[0] = tok.parser
    TOK[0] = tok


TOK = [None]


def tool_skeletons(tools):
    """Text fragments of the tool-call format for a request's tools, rendered by the folder's own template: one assistant
    turn calling each tool with its parameters set to a placeholder, in the schema's order and required keys first when
    that differs (a template may reorder them itself), cut after the generation prompt. The engine's skeleton draft source
    drafts the fixed parts between the placeholders. Drafts only: never forced."""
    tk = TOK[0]
    if tk is None:
        return []
    u = [{"role": "user", "content": "x"}]
    try:
        pre = tk.render(u, tools, add_generation_prompt=True)
    except Exception:
        return []
    out = []
    for t in tools or ():
        f = t.get("function", t)
        name = f.get("name")
        if not name:
            continue
        params = f.get("parameters") or {}
        keys = list((params.get("properties") or {}).keys())
        req = [k for k in params.get("required") or [] if k in keys]
        orders = [keys] + ([req + [k for k in keys if k not in req]] if req + [k for k in keys if k not in req] != keys else [])
        for ks in orders:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": name, "arguments": {k: "X" for k in ks}}}]}
            try:
                full = tk.render(u + [msg], tools, add_generation_prompt=False)
            except Exception:
                continue
            if full.startswith(pre) and full[len(pre):] not in out:
                out.append(full[len(pre):])
    return out
