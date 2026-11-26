"""Qwen3.8-Flash-Next's chat format: the generic module (python/serve/generic/chat_format.py: the folder's own tokenizer,
chat_template.jinja and tokenizer_config.json response_template, nothing model-specific in code), with the release's
reasoning switches. The template (a Qwen3.6 ChatML template with Qwen3-Coder XML tool calls) takes reasoning_effort
xhigh (its default), medium or low and states it at the start of the system prompt; enable_thinking=false turns thinking
off. A request's reasoning_effort (or reasoning.effort, what OpenCode sends for a reasoning variant) maps onto those as
the MLX reference implementation mapped it: high -> xhigh, minimal -> low, none -> thinking
off. Checked against transformers' apply_chat_template on the release's files by tests/qwen4_exp/chat_check.py."""
import importlib.util
import os

_g = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generic", "chat_format.py")
_spec = importlib.util.spec_from_file_location("generic_chat_format", _g)
_G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_G)
for _k in dir(_G):
    if not _k.startswith("__"):
        globals()[_k] = getattr(_G, _k)

EFFORT = {"xhigh": "xhigh", "high": "xhigh", "medium": "medium", "low": "low", "minimal": "low"}  # the template knows xhigh / medium / low


def effort_kwargs(eff):
    """template variables for an effort name (None: the template's defaults)"""
    if not eff:
        return {}
    e = str(eff).lower()
    if e in ("none", "off"):
        return {"enable_thinking": False}
    return {"reasoning_effort": EFFORT.get(e, e)}


def template_kwargs(req, default_effort=None, tok=None):
    """(template variables, thinking budget = 0) from a request: chat_template_kwargs as given, the effort mapped"""
    kw = dict(req.get("chat_template_kwargs") or {})
    eff = req.get("reasoning_effort")
    if not eff and isinstance(req.get("reasoning"), dict):
        eff = req["reasoning"].get("effort")
    eff = kw.pop("reasoning_effort", None) or eff or default_effort
    return {**effort_kwargs(eff), **kw}, 0
