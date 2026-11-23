#!/usr/bin/env python3
"""dmserve: an OpenAI-compatible server (/v1/chat/completions with streaming and tool calls, /v1/models) in front of the
engine's model worker (build/dmserve: one model process, one request at a time, prefix cache inside).

  python python/serve/dmserve.py --model-dir MODEL                  # the profile's defaults (docs/MANIFEST-PRESETS.md)
  python python/serve/dmserve.py --model-dir MODEL --mode fast --port 8090
  python python/serve/dmserve.py --model-dir MODEL -- --cache-gb 5  # everything after "--" goes to the worker
  (tools/dmserve.cpp; a worker flag overrides that one default, `--profile none` drops the profile)

This process owns what is specific to a model folder and to HTTP: the release's tokenizer and chat template (through the
model's format module, python/serve/qwen3_5_moe/chat_format.py for Qwen3.6), the reasoning / tool-call split of the generated text, stop
strings, the request queue. Nothing in the prompt is changed: the template renders the client's messages as they come.

Scheduling: requests with tools (the agent's main thread) go first. A request without tools (a client's title request) runs
only when no main request waits; if a main request arrives while it runs, it is cancelled and run again later (its output
is held until it is complete, so the client never sees a restart). The prefix cache parks the main conversation's state
while a side request runs, so the side request does not cost the main thread its prefix.
Each request's log line says where its prompt diverged from the previous main prompt and what was there."""
import argparse, heapq, http.server, itertools, json, os, queue, subprocess, sys, threading, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ADV = "--help-advanced" in sys.argv[1:]
if ADV:
    sys.argv = [x if x != "--help-advanced" else "--help" for x in sys.argv]


def adv(h):
    """advanced options are listed by --help-advanced only"""
    return h if ADV else argparse.SUPPRESS


ap = argparse.ArgumentParser(description=__doc__ if ADV else __doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                             epilog=None if ADV else "--help-advanced lists every other option.")
ap.add_argument("--model-dir", "--model", required=True, help="the model folder, as downloaded from the Hub")
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--mode", default=None, choices=["exact", "fast"], help="exact: generated tokens never depend on the cache, the routers or timing (the Mac's default); fast: a picked expert that is not in memory and has a small routing weight is left out (default: the profile's)")
ap.add_argument("--profile", default="auto", help="mac | phone | phone-small (default: from the device); none: the worker's built-in values")
ap.add_argument("--drafts", type=int, default=0, help="1: draft and verify for greedy requests (lookup + tool-call skeletons), experimental (default 0)")
ap.add_argument("--format", default=None, help=adv("folder holding the model's chat_format.py (default: python/serve/<family> if it exists, else python/serve/generic, which reads the model folder's own template and response format)"))
ap.add_argument("--model-id", default="qwen3.6-35b-a3b", help=adv("the model name /v1/models reports"))
ap.add_argument("--worker", default=os.path.join(REPO, "build", "dmserve"), help=adv("the worker binary"))
ap.add_argument("--log", default="server.jsonl", help=adv("request log (JSON lines)"))
ap.add_argument("--max-new-cap", type=int, default=0, help=adv("cap on max_tokens (0: the client's value); the replay uses it to bound decode"))
ap.add_argument("--seed", type=int, default=None, help=adv("seed for requests that send none (benchmarks: deterministic sampling); default: random per request"))
ap.add_argument("--side-grace", type=float, default=3.0, help=adv("seconds the main thread must be quiet before a side request (title) runs"))
ap.add_argument("--log-ids", action="store_true", help=adv("log every request's prompt and generated token ids (comparisons of two builds)"))
ap.add_argument("--ahead", type=int, default=1, help=adv("idle-time prefill: when a turn ends without tool calls, prefill the next prompt's predicted prefix while no request runs (python/serve/ahead.py; 0: off)"))
ap.add_argument("--reserve-gb", type=float, default=None, help=adv("memory left to the rest of the system by the automatic cache size (worker default 5.5)"))
ap.add_argument("--log-bodies", action="store_true", help=adv("log every request's body (a later warm-start replay sends one again)"))
ap.add_argument("--allow-force", action="store_true", help=adv("benchmarks only: honour a request's dm_force (a recorded answer fed as the model's turn, so a replay's next request matches the live state)"))
ap.add_argument("--idle-s", type=float, default=300, help=adv("stop the model worker after this long without a request (frees its locked memory); the next request starts it again and the prefix cache resumes from disk (0: never)"))
argv = sys.argv[1:]
wargs = argv[argv.index("--") + 1:] if "--" in argv else []
a = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
if a.format is None:
    try:
        fam = json.load(open(os.path.join(a.model_dir, "moved-store.json"))).get("family", "")
    except Exception:
        fam = ""
    a.format = os.path.join(HERE, fam) if fam and os.path.exists(os.path.join(HERE, fam, "chat_format.py")) else os.path.join(HERE, "generic")
sys.path.insert(0, a.format)
import chat_format as CF  # noqa: E402
sys.path.insert(0, HERE)
import agent_profile as PR  # noqa: E402
# the profile, mode and drafts go to the worker, which resolves the defaults (a worker flag after "--" wins)
_prof = "auto" if a.profile == "agent" else a.profile
_extra = PR.agent_profile(a.model_dir, drafts=bool(a.drafts), reserve_gb=a.reserve_gb, profile=_prof, mode=a.mode)
wargs = [x for i, x in enumerate(_extra) if not (x.startswith("--") and x in wargs) and not (i and _extra[i - 1].startswith("--") and _extra[i - 1] in wargs)] + wargs
sys.path.insert(0, HERE)
import ahead as AH  # noqa: E402

tok = CF.Tok(a.model_dir)
if hasattr(CF, "bind"):
    CF.bind(tok)
KEEP_SPECIAL = getattr(CF, "PARSE_KEEP_SPECIAL", False)  # formats whose turn structure is special tokens parse with them kept
GEN = json.load(open(os.path.join(a.model_dir, "generation_config.json")))
EOS = GEN.get("eos_token_id", [])
EOS = EOS if isinstance(EOS, list) else [EOS]
LOGF = open(a.log, "a")
LOCK = threading.Lock()


def log(rec):
    rec["t"] = round(time.time(), 3)
    with LOCK:
        LOGF.write(json.dumps(rec) + "\n")
        LOGF.flush()
    print(json.dumps(rec)[:400], flush=True)


_SKEL = {}


def skeleton_ids(tools):
    """Token ids of the format module's tool-call fragments for these tools (cached per schema); none if the module has none"""
    if not tools or not hasattr(CF, "tool_skeletons"):
        return []
    key = json.dumps(tools, sort_keys=True)
    if key not in _SKEL:
        _SKEL[key] = [tok.encode(t) for t in CF.tool_skeletons(tools)]
    return _SKEL[key]


# ---------------------------------------------------------------------------------------------------- the worker
class Worker:
    def __init__(self):
        self.stopping = False
        self.last = time.time()
        cmd = [a.worker, "--model", a.model_dir] + wargs
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.wlock = threading.Lock()
        self.subs = {}  # rid -> queue of (kind, payload)
        line = self.p.stdout.readline()
        if not line.startswith("READY"):
            raise SystemExit(f"worker failed to start: {line!r}")
        self.info = json.loads(line[6:])
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            kind, rest = line[0], line[2:]
            if kind == "S":  # STATS
                continue
            if kind in "AHM":  # an idle-time prefill ended (done or cancelled) / the saved hot set's refill or save
                _, _, js = rest.partition(" ")
                try:
                    log({"event": {"A": "ahead", "H": "hotset", "M": "cache_budget"}[kind], **json.loads(js)})
                except Exception:
                    pass
                continue
            rid_s, _, payload = rest.partition(" ")
            q = self.subs.get(int(rid_s))
            if q:
                q.put((kind, payload))
        rc = self.p.wait()
        log({"event": "worker_exit", "rc": rc, "idle_stop": self.stopping})
        if not self.stopping:  # a crash: end the requests in flight with an error; the next request starts a new worker
            global W
            with WLOCK:
                if W is self:
                    W = None
            for rid, q in list(self.subs.items()):
                q.put(("E", "error " + json.dumps({"error": f"model worker exited (rc {rc})"})))

    def stop(self):
        self.stopping = True
        with self.wlock:
            try:
                self.p.stdin.write("Q\n")
                self.p.stdin.flush()
            except Exception:
                pass
        try:
            self.p.wait(30)
        except Exception:
            self.p.kill()

    def submit(self, rid, ids, marks, max_new, temp, top_p, top_k, seed, stop_ids, tools=None):
        q = queue.Queue()
        self.subs[rid] = q
        tj = json.dumps(tools or [], separators=(",", ":"), ensure_ascii=True)  # one line: the request's tool schema
        sk = skeleton_ids(tools)  # the tool-call format's fragments, token ids: drafts for the worker (greedy requests)
        with self.wlock:
            self.p.stdin.write(f"R {rid} {max_new} {temp} {top_p} {top_k} {seed} {len(ids)} {len(marks)} {len(stop_ids)} {len(sk)}\n"
                               f"{' '.join(map(str, ids))}\n{' '.join(map(str, marks))}\n{' '.join(map(str, stop_ids))}\n{tj}\n"
                               + "".join(" ".join(map(str, f)) + "\n" for f in sk))
            self.p.stdin.flush()
        return q

    def ahead(self, aid, ids, marks):
        with self.wlock:
            self.p.stdin.write(f"A {aid} {len(ids)} {len(marks)}\n{' '.join(map(str, ids))}\n{' '.join(map(str, marks))}\n")
            self.p.stdin.flush()

    def cancel(self, rid):
        with self.wlock:
            try:
                self.p.stdin.write(f"C {rid}\n")
                self.p.stdin.flush()
            except Exception:  # the worker is gone (its reader ends the request)
                pass

    def done(self, rid):
        self.subs.pop(rid, None)


W = Worker()
RID = itertools.count(1)
WLOCK = threading.Lock()


def worker():
    """the live worker, started again if the idle timer stopped it"""
    global W
    with WLOCK:
        if W is None:
            W = Worker()
            log({"event": "worker_restart", "worker": W.info})
        W.last = time.time()
        return W


def idle_watch():
    global W
    while a.idle_s > 0:
        time.sleep(5)
        with WLOCK, QLOCK:
            if W is not None and RUNNING[0] is None and not WAITING and time.time() - W.last > a.idle_s:
                log({"event": "idle_stop", "idle_s": a.idle_s})
                W.stop()
                W = None


# ---------------------------------------------------------------------------------------------------- the queue
class Job:
    def __init__(self, prio, req):
        self.prio, self.req = prio, req
        self.go = threading.Event()  # set when this job may run on the worker
        self.preempt = threading.Event()
        self.t_arrive = time.time()


QLOCK = threading.Condition()
WAITING = []  # heap of (prio, seq, job)
SEQ = itertools.count()
RUNNING = [None]


LAST_MAIN_END = [0.0]


def schedule():
    """start the best waiting job if the worker is free. A side request (no tools: a title) starts only when no main
    request waits and the main thread has been quiet for --side-grace seconds: an agent sends its next turn within a
    fraction of a second of a tool result, and a side request started in that gap would be preempted and wasted"""
    with QLOCK:
        if RUNNING[0] is None and WAITING:
            prio, _, j0 = WAITING[0]
            now = time.time()
            if prio > 0 and (now - LAST_MAIN_END[0] < a.side_grace or now - j0.t_arrive < a.side_grace):
                return  # a side request waits until it has waited and the main thread has been quiet for side_grace
            _, _, j = heapq.heappop(WAITING)
            RUNNING[0] = j
            j.go.set()


def ticker():
    while True:
        time.sleep(0.2)
        schedule()


AHEAD = {"pred": None, "seq": 0}  # the predicted next main prompt (ids, marks), until a main request arrives


def send_ahead():
    """hand the pending prediction to the worker (it runs it while idle; any request cancels it)"""
    with QLOCK:
        pred = AHEAD["pred"]
        # a waiting side request (a title, which waits for a quiet main thread and may never run) does not hold it back:
        # when the side request starts, the worker cancels the prediction, and it is handed over again after
        if pred is None or W is None or RUNNING[0] is not None or any(w[2].prio == 0 for w in WAITING):
            return
        AHEAD["seq"] += 1
        aid = AHEAD["seq"]
    try:
        W.ahead(aid, *pred)
    except Exception:
        pass


def enqueue(job):
    with QLOCK:
        if job.prio == 0:
            AHEAD["pred"] = None  # the next main request is here: the prediction is used or obsolete
        heapq.heappush(WAITING, (job.prio, next(SEQ), job))
        r = RUNNING[0]
        if r is not None and r.prio > job.prio:  # a main request arrived while a side request runs: preempt it
            r.preempt.set()
    schedule()


def release(job):
    with QLOCK:
        if RUNNING[0] is job:
            RUNNING[0] = None
        if job.prio == 0:
            LAST_MAIN_END[0] = time.time()
    schedule()
    if job.prio > 0:
        send_ahead()  # a side request cancelled a running prediction: hand it over again


LAST_MAIN = {"ids": None}


def divergence(ids):
    """where this prompt leaves the previous main prompt, and the text on both sides there"""
    prev = LAST_MAIN["ids"]
    if prev is None:
        return {"lcp_prev_main": 0}
    n = min(len(prev), len(ids))
    k = 0
    while k < n and prev[k] == ids[k]:
        k += 1
    d = {"lcp_prev_main": k, "prev_len": len(prev)}
    if k < len(prev):
        d["diverged_prev"] = tok.decode(prev[max(0, k - 6):k + 18])
        d["diverged_this"] = tok.decode(ids[max(0, k - 6):k + 18])
    return d


def sse(rid, created, model, delta, finish=None):
    return ("data: " + json.dumps({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *x):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": a.model_id, "object": "model", "owned_by": "local"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": "not found"})
        t0 = time.time()
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        tools = req.get("tools") or None
        try:
            kw, budget = CF.template_kwargs(req)
            ids, marks = tok.prompt_ids(CF.to_template(req.get("messages", [])), tools, **kw)
        except Exception as e:
            return self._json(400, {"error": {"message": f"chat template: {e}", "type": "invalid_request_error"}})
        thinking = tok.thinking(ids)
        forced, prompt_n = None, len(ids)
        if a.allow_force and req.get("dm_force"):
            fa = req["dm_force"]
            fmsg = CF.to_template([CF.to_message(fa.get("reasoning"), fa.get("content"), fa.get("tool_calls") or [])])[0]
            full, _ = tok.prompt_ids(CF.to_template(req.get("messages", [])) + [fmsg], tools, add_generation_prompt=False, **kw)
            if full[: len(ids)] != ids:
                return self._json(400, {"error": {"message": "dm_force: the answer does not extend the prompt"}})
            forced = full[len(ids):]
            cut = [i for i, t in enumerate(forced) if t in EOS]
            forced = forced[: cut[0]] if cut else forced
            ids, marks = ids + forced, marks + [len(ids)]
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 4096)
        if a.max_new_cap:
            max_new = min(max_new, a.max_new_cap)
        max_new = max(1, min(max_new, CTX - len(ids) - 1))
        temp = float(req["temperature"]) if req.get("temperature") is not None else float(GEN.get("temperature", 1.0))
        top_p = float(req["top_p"]) if req.get("top_p") is not None else float(GEN.get("top_p", 1.0))
        top_k = int(req["top_k"]) if req.get("top_k") is not None else int(GEN.get("top_k", 0))
        seed = int(req["seed"]) if req.get("seed") is not None else (a.seed if a.seed is not None else int.from_bytes(os.urandom(6), "little"))
        stops = req.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        stream = bool(req.get("stream"))
        side = tools is None
        job = Job(1 if side else 0, req)
        rid_s, created, model = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time()), str(req.get("model") or a.model_id)
        div = divergence(ids) if not side else {}
        if not side:
            LAST_MAIN["ids"] = ids
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        alive = [True]

        def write(b):
            if not alive[0]:
                return False
            try:
                self.wfile.write(b)
                self.wfile.flush()
                return True
            except Exception:
                alive[0] = False
                return False

        enqueue(job)
        while not job.go.wait(5):  # keep the stream alive while queued
            if stream and not write(b": queued\n\n"):
                with QLOCK:
                    WAITING[:] = [w for w in WAITING if w[2] is not job]
                    heapq.heapify(WAITING)
                return
        t_start = time.time()
        attempts = 0
        try:
            while True:
                attempts += 1
                res = self.run(job, ids, marks, max_new, temp, top_p, top_k, seed, stops, thinking and forced is None, stream and not side, write, rid_s, created, model, tools)
                if res is None:  # preempted: back in the queue, run again later
                    job.preempt.clear()
                    job.go.clear()
                    release(job)
                    enqueue(job)
                    while not job.go.wait(5):
                        if stream and not write(b": queued\n\n"):
                            return
                    continue
                break
        finally:
            release(job)
        text, new_ids, finish, binfo, einfo, t_first = res
        reasoning, content, calls = CF.parse_turn(text, thinking and forced is None, tools or [])
        if calls:
            finish = "tool_calls"
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(new_ids), "total_tokens": len(ids) + len(new_ids),
                 "prompt_tokens_details": {"cached_tokens": binfo.get("reused", 0)}}
        if stream:
            if side:  # held until complete (a preempted side request restarts invisibly)
                write(sse(rid_s, created, model, {"role": "assistant", "content": ""}))
                if reasoning:
                    write(sse(rid_s, created, model, {"reasoning_content": reasoning}))
                if content:
                    write(sse(rid_s, created, model, {"content": content}))
            if calls:
                write(sse(rid_s, created, model, {"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]}))
            write(sse(rid_s, created, model, {}, finish))
            if (req.get("stream_options") or {}).get("include_usage"):
                write(("data: " + json.dumps({"id": rid_s, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage}) + "\n\n").encode())
            write(b"data: [DONE]\n\n")
        else:
            msg = {"role": "assistant", "content": content or None}
            if reasoning:
                msg["reasoning_content"] = reasoning
            if calls:
                msg["tool_calls"] = calls
            write(json.dumps({"id": rid_s, "object": "chat.completion", "created": created, "model": model,
                              "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}).encode())
        t_end = time.time()
        log({"event": "request", "kind": "side" if side else "main", "prompt": len(ids), "marks": len(marks), "max_new": max_new, "temp": temp, "top_p": top_p, "top_k": top_k,
             "stream": stream, "queued_s": round(t_start - t0, 3), "ttft_s": round(t_first - t0, 3) if t_first else None, "total_s": round(t_end - t0, 3),
             "attempts": attempts, "completion": len(new_ids), "finish": finish, "tool_calls": len(calls), "reasoning_chars": len(reasoning or ""),
             "client_alive": alive[0], "forced": len(forced) if forced is not None else 0, **div, "worker_ready": binfo, "worker_end": einfo, **({"prompt_ids": ids, "gen_ids": new_ids} if a.log_ids else {}), **({"body": req} if a.log_bodies else {})})
        if not side and a.ahead and not einfo.get("error"):
            try:  # the next main prompt's predicted prefix, if this turn ended the agent's loop
                if forced is not None:
                    fa = req["dm_force"]
                    t_r, t_c, t_calls = fa.get("reasoning"), fa.get("content"), fa.get("tool_calls") or []
                else:
                    t_r, t_c, t_calls = reasoning, content, calls
                if not t_calls and finish != "length":
                    t_a = time.time()
                    S, smarks = AH.predict(tok, CF.to_template(req.get("messages", [])), tools, kw, CF.to_message(t_r, t_c, []))
                    if S:
                        with QLOCK:
                            if not any(w[2].prio == 0 for w in WAITING):
                                AHEAD["pred"] = (S, smarks)
                        log({"event": "ahead_predicted", "predicted": len(S), "ms_render": round(1e3 * (time.time() - t_a), 1)})
            except Exception as e:
                log({"event": "ahead_predict_failed", "error": str(e)[:200]})
        if not side:
            send_ahead()

    def run(self, job, ids, marks, max_new, temp, top_p, top_k, seed, stops, thinking, live_stream, write, rid_s, created, model, tools):
        rid = next(RID)
        W = worker()
        q = W.submit(rid, ids, marks, max_new, temp, top_p, top_k, seed, EOS, tools)
        new, text, sent_text = [], "", 0
        splitter = CF.StreamSplitter(thinking, lambda field, t: write(sse(rid_s, created, model, {field: t}))) if live_stream else None
        if live_stream:
            write(sse(rid_s, created, model, {"role": "assistant", "content": ""}))
        binfo, einfo, finish, t_first, cancelled_by = {}, {}, "length", None, None
        last_ping = time.time()
        try:
            while True:
                try:
                    kind, payload = q.get(timeout=2)
                except queue.Empty:
                    if live_stream and time.time() - last_ping > 5:
                        if not write(b": prefill\n\n") and cancelled_by is None:
                            cancelled_by = "disconnect"
                            W.cancel(rid)
                        last_ping = time.time()
                    if job.preempt.is_set() and cancelled_by is None:
                        cancelled_by = "preempt"
                        W.cancel(rid)
                    continue
                if kind == "B":
                    binfo = json.loads(payload)
                elif kind == "T":
                    t = int(payload)
                    if t in EOS:
                        continue
                    if t_first is None:
                        t_first = time.time()
                    new.append(t)
                    full = tok.decode(new, skip_special_tokens=False)
                    piece, text = full[len(text):], full
                    if any(s and s in text for s in stops) and cancelled_by is None:  # client stop strings
                        cut = min(text.find(s) for s in stops if s and s in text)
                        piece = piece[: max(0, len(piece) - (len(text) - cut))]
                        text = text[:cut]
                        cancelled_by = "stop"
                        W.cancel(rid)
                    if splitter and piece:
                        splitter.feed(piece)
                    if live_stream and not alive_check(write) and cancelled_by is None:
                        cancelled_by = "disconnect"
                        W.cancel(rid)
                    if job.preempt.is_set() and cancelled_by is None:
                        cancelled_by = "preempt"
                        W.cancel(rid)
                elif kind == "E":
                    reason, _, js = payload.partition(" ")
                    einfo = json.loads(js) if js.strip().startswith("{") else {}
                    if reason == "error":
                        cancelled_by = cancelled_by or "error"
                    finish = "stop" if reason == "stop" or cancelled_by == "stop" else ("length" if reason == "length" else "stop")
                    break
        finally:
            W.done(rid)
            W.last = time.time()
        if cancelled_by == "preempt":
            return None
        text = tok.decode(new, skip_special_tokens=not KEEP_SPECIAL) if cancelled_by != "stop" else text
        return text, new, finish, binfo, einfo, t_first


def alive_check(write):
    return write(b"") is not False


CTX = W.info["ctx"]
threading.Thread(target=idle_watch, daemon=True).start()
threading.Thread(target=ticker, daemon=True).start()
http.server.ThreadingHTTPServer.allow_reuse_address = True
log({"event": "serving", "port": a.port, "worker": W.info, "worker_args": wargs})
http.server.ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
