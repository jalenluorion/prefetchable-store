"""The release's chat format for Qwen/Qwen3.6-35B-A3B, without transformers: tokenizer + chat template (Tok), the reasoning
and tool-call split of a generated turn (parse_turn), the same split done incrementally while streaming (StreamSplitter),
OpenAI messages -> what the template expects (to_template), and the thinking switches (effort). Imported by mlx_run.py and
by test_run.py, which checks it against the real release files (release_small/: tokenizer and template, no weights).

What the release's files say (chat_template.jinja and tokenizer_config.json, revision 345d467b, read 2026-11-16):
- ChatML. The generation prompt is "<|im_start|>assistant\\n<think>\\n" (thinking on, the default), or
  "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n" with enable_thinking=false. There is NO reasoning-effort switch in this
  template (Qwen3.8's had xhigh/medium/low); see effort() for what --reasoning-effort means here.
- Tool calls are the Qwen3-Coder XML form, after the content (separated by "\\n\\n" when there is content):
      <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>
  several calls joined by "\\n"; string values are written raw, every other value as JSON (tojson). Tool results come back
  as role "tool", rendered inside a user turn as "<tool_response>\\n...\\n</tool_response>".
- tokenizer_config.json's response_template (the release's own parsing rules): reasoning ends at "</think>" OR where a
  "<tool_call>" begins; a call is "<tool_call>\\s*<function=NAME>" up to "</tool_call>" or the next "<tool_call>"; a
  parameter is "<parameter=KEY>\\n?VALUE\\n?" up to "</parameter>", the next "<parameter=" or "</function>"; content ends at
  "<|im_end|>" / "<|endoftext|>". parse_turn implements exactly these (plus: an unterminated last call runs to the end).
- History: an assistant turn's reasoning is re-rendered only for turns after the last real user message (an agent's tool
  loop keeps its reasoning), unless preserve_thinking=true (then every turn keeps it, and the rendered history stays the
  token sequence the model produced: the session's live prefix survives a new user message).
- "<think>", "</think>", "<tool_call>", "</tool_call>" are single, non-special tokens (decode keeps them with
  skip_special_tokens); "<function=...>" and "<parameter=...>" are ordinary text.
"""
import json, os, re, uuid

from tokenizers import Tokenizer


class Tok:
    """The release's tokenizer and chat template with the small libraries only: `tokenizers` for the pieces, jinja2 for the
    template, set up as transformers sets it up (trim_blocks, lstrip_blocks, loopcontrols, its tojson; {% generation %}
    renders its body). test_run.py checks the rendering byte for byte against transformers' apply_chat_template."""

    def __init__(self, path, defaults=None):
        import jinja2
        import jinja2.ext

        self.tk = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        tc = json.load(open(os.path.join(path, "tokenizer_config.json")))
        self.chat_template, self.bos_token, self.eos_token = tc.get("chat_template"), tc.get("bos_token"), tc.get("eos_token")
        if not self.chat_template and os.path.exists(os.path.join(path, "chat_template.jinja")):  # newer releases ship the template as its own file
            self.chat_template = open(os.path.join(path, "chat_template.jinja")).read()
        self.bos_token, self.eos_token = [t.get("content") if isinstance(t, dict) else t for t in (self.bos_token, self.eos_token)]
        self.eos_token_id = self.tk.token_to_id(self.eos_token) if self.eos_token else None
        self.defaults = dict(defaults or {})  # template switches applied when a call does not set them (--preserve-thinking)

        class Generation(jinja2.ext.Extension):  # transformers' {% generation %} ... {% endgeneration %} (it marks the assistant's tokens): the body renders unchanged
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
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self.tpl = env.from_string(self.chat_template) if self.chat_template else None
        self.think_open, self.think_close = self.tk.token_to_id("<think>"), self.tk.token_to_id("</think>")
        self.think_prompt = self.encode("<think>\n")  # how the generation prompt ends when thinking is on

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        kw = {**self.defaults, **{k: v for k, v in kw.items() if v is not None}}
        return self.tpl.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, bos_token=self.bos_token, eos_token=self.eos_token, **kw)

    def prompt_ids(self, messages, tools=None, **kw):
        """Token ids of a conversation, and the positions where each message ends (the prefixes worth snapshotting). The
        boundaries are found in the one rendering, after each "<|im_end|>\\n" (ChatML; the template refuses to render a
        conversation without a user message, so the system prompt cannot be rendered alone), at a token boundary."""
        text = self.render(messages, tools, **kw)
        enc = self.tk.encode(text, add_special_tokens=False)
        ids, starts = enc.ids, {s: i for i, (s, e) in enumerate(enc.offsets)}
        marks, sep = [], "<|im_end|>\n"
        p = text.find(sep)
        while p >= 0:
            q = p + len(sep)
            if q in starts:
                marks.append(starts[q])
            p = text.find(sep, q)
        return ids, marks

    def thinking(self, ids):
        """Does this prompt end inside an open reasoning block (the generation prompt with thinking on)?"""
        return list(ids[-len(self.think_prompt):]) == self.think_prompt


# -- thinking switches. The template has enable_thinking (and preserve_thinking) only. --reasoning-effort / a request's
# reasoning_effort map onto it plus a thinking BUDGET: after that many reasoning tokens the runner writes Qwen's documented
# early-exit sentence and "</think>" into the turn itself (the tokens are fed like sampled ones, so the turn stays the
# model's own token sequence and re-renders identically), and the answer follows. high/xhigh: no budget (the release as
# shipped); none: thinking off (the template's empty block).
THINK_BUDGET = {"high": 0, "medium": 4096, "low": 1024}
EFFORT_ALIASES = {"xhigh": "high", "high": "high", "medium": "medium", "low": "low", "minimal": "low", "none": "none", "off": "none"}
FORCE_CLOSE = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"


def effort(name):
    """(template kwargs, thinking budget in tokens, 0 = none) for an effort name (None: the release's default)."""
    if not name:
        return {}, 0
    e = EFFORT_ALIASES.get(str(name).lower())
    if e is None:
        raise ValueError(f"unknown reasoning effort {name!r} (use one of {sorted(EFFORT_ALIASES)})")
    if e == "none":
        return {"enable_thinking": False}, 0
    return {}, THINK_BUDGET[e]


def template_kwargs(req, default_effort=None):
    """The chat template's switches from a request: vLLM's chat_template_kwargs (enable_thinking, preserve_thinking, ...)
    and OpenAI's reasoning_effort / reasoning.effort (what OpenCode sends for a reasoning variant); returns (template
    kwargs, thinking budget). Nothing set: --reasoning-effort, else the template's defaults (thinking on, no budget)."""
    kw = dict(req.get("chat_template_kwargs") or {})
    eff = req.get("reasoning_effort")
    if not eff and isinstance(req.get("reasoning"), dict):
        eff = req["reasoning"].get("effort")
    eff = kw.pop("reasoning_effort", None) or eff or default_effort
    budget = int(kw.pop("thinking_budget", 0) or 0)
    if eff:
        ekw, b = effort(eff)
        kw = {**ekw, **kw}
        budget = budget or b
    return kw, budget


# -- a generated turn -> reasoning, content, tool calls (the release's response_template rules, see the module docstring)
CALL_OPEN = re.compile(r"<tool_call>\s*<function=([^>\n]+)>")
PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?(?:</parameter>|(?=<parameter=)|(?=</function>)|\Z)", re.S)
JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)  # Qwen3-Instruct's JSON form, accepted too
BARE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*\n(\{.*\})\s*$", re.S)


def _value(v, typ):
    """A parameter's text as the template would have written it: strings raw, everything else JSON. With a schema type,
    typed values are parsed; without one, only what looks like a JSON object or array (a bare "123" stays a string, which
    re-renders as the same text)."""
    if typ in ("integer", "number", "boolean", "object", "array", "null") or (typ is None and v[:1] in "[{"):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def tool_skeletons(tools):
    """Text fragments of the tool-call format for a request's tools, as the chat template writes a call (the engine's
    skeleton draft source matches the generation's last tokens inside them and drafts what follows: the call's opening,
    the rest of a tool name, a key, the closings). One fragment per tool and parameter order (the schema's, and required
    keys first when that differs); values are a placeholder the drafts never cross. Drafts only: never forced."""
    out = []
    for t in tools or ():
        f = t.get("function", t)
        name = f.get("name")
        if not name:
            continue
        params = f.get("parameters") or {}
        keys = list((params.get("properties") or {}).keys())
        req = [k for k in params.get("required") or [] if k in keys]
        orders = [keys]
        alt = req + [k for k in keys if k not in req]
        if alt != keys:
            orders.append(alt)
        for ks in orders:
            body = "".join(f"<parameter={k}>\nX\n</parameter>\n" for k in ks)
            out.append(f"\n\n<tool_call>\n<function={name}>\n{body}</function>\n</tool_call><|im_end|>")
    return out


def call_id():
    return "call_" + uuid.uuid4().hex[:24]


def parse_tool_calls(text, tools=()):
    """The tool calls in a turn's answer part, as OpenAI tool_calls, and the content around them."""
    names = {t.get("function", {}).get("name") for t in tools or ()}
    specs = {t.get("function", {}).get("name"): (t.get("function", {}).get("parameters") or {}).get("properties") or {} for t in tools or ()}
    calls, spans = [], []

    def call(name, args):
        return {"id": call_id(), "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}

    opens = list(CALL_OPEN.finditer(text))
    for i, m in enumerate(opens):
        end = opens[i + 1].start() if i + 1 < len(opens) else len(text)  # a call runs to "</tool_call>" or the next "<tool_call>"
        close = text.find("</tool_call>", m.end(), end)
        body_end, span_end = (close, close + len("</tool_call>")) if close >= 0 else (end, end)
        body = text[m.end():body_end]
        fe = body.find("</function>")
        body = body if fe < 0 else body[:fe]
        name, args = m.group(1).strip(), {}
        for pm in PARAM.finditer(body):
            k = pm.group(1).strip()
            args[k] = _value(pm.group(2), (specs.get(name, {}).get(k) or {}).get("type"))
        calls.append(call(name, args))
        spans.append((m.start(), span_end))
    for m in JSON_CALL.finditer(text):
        if any(a <= m.start() < b for a, b in spans):
            continue
        try:
            d = json.loads(m.group(1))
            calls.append(call(d.get("name", ""), d.get("arguments", {})))
            spans.append((m.start(), m.end()))
        except Exception:
            pass
    content = text
    for a, b in sorted(spans, reverse=True):
        content = content[:a] + content[b:]
    content = content.strip()
    if not calls:  # a JSON call written without the tag wrapper ("bash\n{...}"), seen occasionally: accepted when the name is a known tool
        m = BARE.match(content)
        if m and m.group(1) in names:
            try:
                args = json.loads(m.group(2))
                if isinstance(args, dict):
                    calls.append(call(m.group(1), args.get("arguments", args) if set(args) == {"name", "arguments"} else args))
                    content = ""
            except Exception:
                pass
    return content, calls


def parse_turn(text, thinking, tools=()):
    """A generated turn (decoded with skip_special_tokens, after the generation prompt) -> (reasoning or None, content,
    tool calls). With thinking on, the turn starts inside the reasoning, which ends at "</think>" or where a tool call
    begins (no "</think>": the reasoning ran to the end, or straight into a call)."""
    reasoning = None
    if thinking:
        a, b = text.find("</think>"), text.find("<tool_call>")
        if a >= 0 and (b < 0 or a < b):
            reasoning, text = text[:a], text[a + len("</think>"):]
        elif b >= 0:
            reasoning, text = text[:b], text[b:]
        else:
            reasoning, text = text, ""
        reasoning = reasoning.strip()
    content, calls = parse_tool_calls(text, tools)
    return reasoning, content, calls


def to_message(reasoning, content, calls):
    """A parsed turn as the assistant message a client sends back (and the chat loop keeps)."""
    msg = {"role": "assistant", "content": content or ""}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = calls
    return msg


def to_template(messages):
    """OpenAI message dicts -> what the chat template expects (tool arguments as dicts, content parts as text)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except Exception:
                        pass
                tcs.append({"type": "function", "function": fn})
            m["tool_calls"] = tcs
            m["content"] = m.get("content") or ""
        if m.get("role") == "assistant" and m.get("reasoning") and not m.get("reasoning_content"):  # some clients send it back under this name
            m["reasoning_content"] = m.pop("reasoning")
        if isinstance(m.get("content"), list):
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        out.append(m)
    return out


class StreamSplitter:
    """parse_turn, incrementally: feed(piece) with each new piece of decoded text; send(field, text) receives reasoning
    (field "reasoning_content", what vLLM's qwen3 reasoning parser sends and OpenCode shows as thinking) and content as
    they become final; tool-call blocks are held back (emitted as tool_calls at the end, from parse_turn on the whole
    text). Whitespace at the edges of each part is held: the template trims both parts and writes its own separators when
    the turn comes back, and the reused prefix must match."""

    def __init__(self, thinking, send):
        self.mode, self.held, self.lead, self.send = ("reasoning" if thinking else "content"), "", True, send

    def feed(self, piece):
        if self.mode == "tool":
            return
        buf = self.held + piece
        if self.mode == "reasoning":
            a, b = buf.find("</think>"), buf.find("<tool_call>")
            if a >= 0 or b >= 0:
                cut = a if a >= 0 and (b < 0 or a < b) else b
                head = buf[:cut].rstrip()
                if self.lead:
                    head = head.lstrip()
                if head:
                    self.send("reasoning_content", head)
                self.mode, self.held, self.lead = "content", "", True
                rest = buf[cut + len("</think>"):] if cut == a else buf[cut:]
                return self.feed(rest) if rest else None
            field = "reasoning_content"
        else:
            if "<tool_call>" in buf:
                head = buf[: buf.index("<tool_call>")].rstrip()
                if self.lead:
                    head = head.lstrip()
                if head:
                    self.send("content", head)
                self.mode, self.held = "tool", ""
                return
            field = "content"
        if self.lead:  # nothing sent in this part yet: leading whitespace is the template's separator
            buf = buf.lstrip()
        keep = len(buf) - len(buf.rstrip())  # trailing whitespace waits until something follows it
        if len(buf) - keep > 0:
            self.send(field, buf[: len(buf) - keep])
            self.lead = False
        self.held = buf[len(buf) - keep:]


def title_of(messages):
    """A thread title from the last user message's first words."""
    text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
            break
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    return (" ".join(words[:7]) + ("..." if len(words) > 7 else "")) or "New thread"
