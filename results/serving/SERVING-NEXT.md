# Serving, next steps: idle-time prefill, Inkling under stock OpenCode, saved hot set, worker restart

Branch `serve-next` (from `unified-wip`). Machine: base M4 MacBook Pro 16 GB, internal SSD, macOS on battery unless a
row says otherwise. Every model run went through `engine/bench/guard.py` via `bench/serving/slot.sh` (power rule, no
fresh QUIET of another job, one model process, PRIORITY during serving tests, QUIET held for timed runs); each run's
power state is in its `*.log.power` file and in the guard log's `STATE_BEFORE`. Raw data: this folder.

## What was built (new files unless marked)

| piece | files |
|---|---|
| idle-time prefill in the prefix cache: `PrefixCache::ahead()` = the request path's `sync()` on predicted ids; the live conversation is parked whole first, checkpoints on the shared prefix are protected, the parked copy is dropped once a request reaches the prediction | `include/dmeng/session.h`, `src/serving/session.cpp` (edited) |
| the worker's runner: `A` protocol lines, runs while no request waits, spans of `--ahead-span` positions (a cancel keeps the finished spans), any request cancels it within one layer (progress hook closes the start-of-call race), `A <id> {json}` out with positions, seconds, MB read, cancel delay | `src/serving/ahead.{h,cpp}` |
| `dmserve --check-ahead A B`: B token by token vs (A served, 8 tokens generated, ahead() of B's prefix cancelled once inside its 2nd span, resumed, B synced) and vs the plain request path | `src/serving/ahead.cpp` |
| the prediction in the front end: render the conversation + the finished turn + two different placeholder user messages with the request path's template; keep the common prefix up to the last message boundary | `python/serve/ahead.py`, `python/serve/dmserve.py` (edited) |
| generic chat format, read from the model folder: tokenizer.json, chat template, `response_template` (fields, open / close markers, repeats, joins, json / xml-inline content, transforms), generation-prompt lead after the start anchor, message boundaries from prefix renders; streaming of reasoning / content with partial markers held | `python/serve/generic/chat_format.py`; the front end picks `python/serve/<family>` if it exists, else `generic` |
| saved hot set: `--hotset FILE`, written at quit (also the front end's idle release), refilled at start in batches of 64 records while no request waits (a request stops it between batches) | `src/serving/hotset.{h,cpp}` |
| worker restart after a crash: requests in flight end with an error, the next request starts a new worker (was: the server exited) | `python/serve/dmserve.py` (edited) |
| benchmarks only: `--allow-force` + a request's `dm_force` (a recorded answer fed as the model's turn, so a replay's next request finds the live state a live session has) | `python/serve/dmserve.py` |
| gates and runs | `bench/serving/` (`ahead_predict_check.py`, `chat_format_check.py`, `ahead_replay.{py,sh}`, `ahead_table.py`, `hotset_restart.sh`, `live_opencode.sh`, `live_table.py`, `opencode_inkling.json`, `slot.sh`, `queue_serve_next.sh`) |

## 1. Inkling 2 Mini under stock OpenCode

The front end serves Inkling from its folder alone (`python/serve/generic/chat_format.py`, picked automatically: there is
no `python/serve/inkling2_loop/`): the folder's `chat_template.jinja`, `tokenizer.json`, `generation_config.json` (stop
token `<|content_model_end_sampling|>`) and `tokenizer_config.json`'s `response_template` (reasoning
`<|content_thinking|>`, text `<|content_text|>`, tool calls `<|content_invoke_tool_json|>` JSON `{"name","args"}`, each
closed by `<|end_message|>`, optional `<|message_model|>NAME` prefix). No Inkling text is written in the code. Tool-call
arguments are kept as objects for the template, call ids are kept (Inkling's template names a tool result by the call it
answers), `reasoning_effort` is passed to the template's own variable. Streaming sends `reasoning_content` / `content` as
they grow; tool calls at the end.

Gates (no model run; `bench/serving/chat_format_check.py`, reference = transformers 5.23 `apply_chat_template` on the same
folder, in `local/inkling/.venv`; output `chat_format_inkling.json`):

| gate | result |
|---|---|
| token ids of the front end's rendering == the reference, recorded OpenCode 1.19.0 requests (s2_clean 11, s4_clean_delays 6, s1_personal 11; 615-19,899 tokens) | 28 / 28 identical |
| OpenCode's recorded assistant turns (128 tool calls in 90 turns) rendered in Inkling's format, parsed back: calls, reasoning, content | 90 / 90 identical; streaming split identical in all |
| assistant turns written in Inkling's format (`engine/results/inkling_approx/texts/agent.txt`, `inkling_draft/tool_prompt.txt`: 9 turns, 8 calls) parsed, re-rendered by the template | 9 / 9 byte-identical; streaming identical |
| the server's batched prefill on Inkling vs token by token (`dmserve --check`, 1,100 ids, chunk 512; `check/check_ink.log`) | 0 differing logit bits, 0 fixed-state bytes, 0 KV bytes, greedy 16/16 identical |
| the same module on the Qwen3.6 folder (cross-check of the generic reading: its schema has a prompt-opened reasoning field, an unopened content field and XML-inline calls) | 17 / 17 renders identical, 50 / 50 turns (72 calls) identical, streaming identical (`chat_format_q36_generic.json`); the 35B keeps its own validated module |

Live task (`bench/serving/live_opencode.sh`, `inkling/live1.*`): stock OpenCode 1.19.0, provider entry only
(`bench/serving/opencode_inkling.json`), the clean-install switches of the recorded replay, a fresh repo, the two user
messages of RESULTS-AGENTIC.md ("Add a median(xs) function ...", then `run --continue` "Make median raise ValueError on an
empty list, and test that too."). Server: 6 GB expert cache, mlock, disk prefix under `inkling/model/.dmeng/prefix`
(checkpoint every 1,024 positions: Inkling's fixed state is 172.8 MB, its positional state 32.8 KB per position).
**Result: the model added `median` with the empty-list check and two tests; all 3 tests pass** (`live1.tests.txt`,
`live1.diff.txt`). AC power, the machine shared with other queues (others' CPU 42.7% mean, swap 8.7 GB in use): times
are indicative only.

| request | prompt | TTFT s | recomputed (reused from) | answer tokens | decode tok/s | |
|---|---|---|---|---|---|---|
| 1 task (cold) | 8,362 | 358.9 | 8,362 (cold) | 40 | 8.1 | 1 call |
| 2 | 8,557 | 8.2 | 155 (live) | 62 | 7.9 | 2 calls |
| 3 | 8,949 | 16.8 | 330 (live) | 136 | 9.0 | 1 call |
| 4 | 9,094 | 1.0 | 9 (live) | 278 | 8.4 | 1 call |
| 5 answer | 9,436 | 4.6 | 64 (live) | 102 | 8.3 | stop |
| 6 second user message | 9,559 | 2.4 | 21 (live) | 167 | 7.8 | 2 calls |
| 7 | 10,217 | 24.2 | 491 (live) | 271 | 7.9 | 2 calls |
| 8 | 10,505 | 1.9 | 17 (live) | 37 | 8.1 | 1 call |
| 9 answer | 10,628 | 5.3 | 86 (live) | 27 | 6.9 | stop |

Walls: turn 1 466 s (of which 359 s the cold prefill), turn 2 100 s. In every request the longest common prefix with
the live state was the whole live sequence (`lcp_live == live_len`): the model's own tokens re-render exactly, tool
rounds recompute only the tool results. Inkling's template keeps every turn's reasoning (OpenCode sends
`reasoning_content` back), so the new user turn recomputed 21 tokens, not the previous turn. Footprint: 8.3-9.2 GB at
request ends, peak 9.69 GB (guard; during the cold prefill): slightly above the 9.5 GB target at a 6 GB cache, so
`--cache-gb 5.8` is the setting to use for a 9.5 GB cap [est]. Memory at the end: resident 1,207 MB, expert slab 5,714 MB,
KV touched 333 MB (1,056 reserved), recurrent + rings 165 MB, prefill scratch 215 MB, checkpoints 2 x 173 MB.
What dominates Inkling's turns here: prefill (~23 tok/s cold, ~20 tok/s for tool results, on a shared machine). Inkling's
batched prefill runs in chunks of 512 positions (`Inkling::prefill`), not layer-major over a whole span as the 35B's
does; whether that is the limit was not isolated in this run.
Not exercised: a warm first request from the disk prefix in a new process; the title request (dropped by `opencode run`
as for the 35B).

## 2. Idle-time prefill of the next turn (the 35B)

When a main request ends without tool calls, the front end predicts the next main prompt's prefix (`python/serve/ahead.py`:
the conversation + the finished turn + a placeholder user message, rendered twice with different placeholders by the
request path's own template, cut at the last message boundary both share) and hands it to the worker (`A` line). The
worker runs it only while its queue is empty, as `PrefixCache::ahead()` = the request path's `sync()` (same kernels, same
checkpoints, same state bits), in spans of `--ahead-span` positions (default 512; a cancel keeps every finished span).
Any request cancels it within one layer. The live conversation is parked whole before the prediction overwrites its tail, and the checkpoints on the shared prefix are protected; a request that reaches the
prediction's end drops that parked copy, one that returns to the old tail resumes from it. Switch: front end `--ahead 1|0`
(default 1), worker `--ahead-span N`, `--ahead-park 0|1`.

Gates:

| gate | result |
|---|---|
| prediction is a prefix of the real next request, all recorded sessions (`bench/serving/ahead_predict_check.py`, `ahead_predict_q36.json`) | s2_clean (clean install): predicted 8,767 tokens, a prefix of the next request; the next request then has 25 new tokens instead of 1,222. s1 / s3 (this machine's personal skills): not a prefix, because the new OpenCode process reorders its skills list at token 8,209 / 7,660, before the template's own change at 16,189: a client change nothing can predict; the prediction's 1,197 tokens are wasted there |
| exactness, tiny Qwen3.6-geometry model (`--check-ahead`, A 600 / B 630 ids, ahead span 64, cancelled after its first span and resumed) | logits 0 differing bits, fixed state 0 bytes, KV 0 bytes, greedy 16/16 identical, both with and without ahead (`check/check_ahead_tiny_q36.log`) |
| exactness, the 35B (`--check-ahead check/ahead_A.txt check/ahead_B.txt --check-ahead-len 2267`: the replay's request 006 / 007 from token 6,500 on, 2,281 / 2,292 ids, diverging at 1,070 as the template makes them; ahead cancelled inside its 2nd span of 512, kept 512 positions, resumed, then B synced: 25 positions) vs B token by token from an empty state | **0 differing logit bits, 0 fixed-state bytes, 0 KV bytes (46.9 MB compared), greedy 16/16 identical**; the request path without ahead the same (`check/check_ahead_q36.log`) |
| generated tokens with the feature off vs on, in serving (`bench/serving/ahead_replay.sh`, forced replay of s2_clean, fixed seed 1, new user turn after 5 s and after 30 s, 64 free tokens each; 12 requests per run) | **identical in all 12 requests** (`ahead/tokens_off_on.json`), including the new user turn whose state came from a completed prediction (30 s) and the one after a cancelled prediction (5 s) |
| a request cancels it | 5 s gap: the prediction was inside its first span; the request waited 91.7 ms (`ms_wait_worker`) for it to stop; nothing of that span was kept (spans are kept only when finished) |
| live conversation kept | in both on-run new user turns: the old tail was parked (`ahead.parks` 1, 2) and dropped once the request reached the prediction (`ahead.hits` 1, 2); the pinned system checkpoint (7,537) was still in memory after each prediction (`checkpoints` [7537, 8787, 8792]) |

Indicative numbers from those two runs (one run each, machine shared with other queues, AC; not a timing result): with a
30 s gap the prediction ran 20.8 s of prefill (1,230 positions, 3 spans of 512, 46.0 GB read) and the new user turn then
computed 25 positions (TTFT 1.1 s vs 17.3 s off); with a 5 s gap nothing was kept (TTFT 18.1 vs 17.8 s). Energy facts:
per finished turn the prediction is one prefill of the dropped turn (~1.2K positions here, ~21 s at full load), which the
next user turn would otherwise compute itself; it is extra work only when no next turn comes or the client changes the
prompt earlier (the personal-skills case above), and smaller spans read more (each span reads its layer's experts again).
Default kept on (lossless, cancel delay under 0.1 s measured); the timing to settle it is below.

## 3. Saved hot set

`--hotset FILE` (worker): at quit the cache's expert ids are written to FILE (most valuable first; the front end's idle
release quits the worker too), and at start they are read back in batches of 64 records while no request waits; a request
stops the refill between two batches and the worker reports `H {"hotset","refilled","ms","stopped"}`.

| gate | result |
|---|---|
| the first request after a restart, without / with the saved hot set (`bench/serving/hotset_restart.sh ... "save off on"`: s2_clean's first request, disk prefix warm, fixed seed, 128 tokens) | **88 generated tokens identical** (`hotset/q36.*`) |
| refill | 2,599 records refilled in 1.32 s before the request (started 8 s after the server) |

Indicative (one run each, shared machine): TTFT 1.86 s off / 1.56 s on; the request's prefill read 5.76 / 4.77 GB; decode
hit rate over the first 50 steps 0.80 / 0.80 (106 MB per step both): the saved set helps the prefill of the first message,
not the first decode steps, in this one run.

## 4. Worker restart after a crash (built, not part of the trimmed gates)

The front end no longer exits when the worker dies: requests in flight end, the next request starts a new worker (the
prefix cache resumes from the disk prefix). Tested with a stand-in worker that exits mid-request (front-end logic only).
Python dependence left: the whole front end (`python/serve/dmserve.py`: HTTP, the tokenizer via `tokenizers`, the chat
template via `jinja2`, the response parsing); the C++ worker needs no Python. Removing it needs a C++ HTTP layer, a BPE
tokenizer and a Jinja subset in C++: not done.

## Commands (from the repository root; every model command goes through `bench/serving/slot.sh LOG TIMEOUT cmd...`)

Build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4`. Python: `../../.venv/bin/python`
(local/flash/.venv: `tokenizers`, `jinja2`); the format gate needs transformers (`local/inkling/.venv`).

Gates:
```
# no model run
../../.venv/bin/python bench/serving/ahead_predict_check.py --model-dir ../../../q36/model --format python/serve/qwen3_5_moe REPLAY/s1_personal REPLAY/s2_clean REPLAY/s3_personal REPLAY/s4_clean_delays
../../../inkling/.venv/bin/python bench/serving/chat_format_check.py --model-dir ../../../inkling/model --sessions REPLAY/s2_clean REPLAY/s4_clean_delays REPLAY/s1_personal --turns ENGINE/results/inkling_approx/texts/agent.txt ENGINE/results/inkling_draft/tool_prompt.txt
# model runs (REPLAY = engine/results/agentic/replay, ENGINE = engine)
zsh bench/serving/queue_serve_next.sh liveink check35 tok35 hot35 checkink   # resumable; each step skipped once its log has a good GUARD line
```
The steps: `check35` = `build/dmserve --model ../../../q36/model --cache-gb 4.6 --mem mlock --ctx 4096 --chunk 512 --span 8192
--ahead-span 512 --check-ahead results/serving/check/ahead_A.txt results/serving/check/ahead_B.txt --check-ahead-len 2267 --gen 16`
(prints `"passed":true`); `tok35` = `bench/serving/ahead_replay.sh 0|1 OUT REPLAY/s2_clean 7 5,30 WORKER_ARGS` twice, then
`bench/serving/ahead_table.py results/serving/ahead toff ton` (`tokens_identical`); `hot35` = `bench/serving/hotset_restart.sh
OUT REPLAY/s2_clean 2 ../../../q36/model/.dmeng/hotset.bin "save off on" WORKER_ARGS`; `liveink` = `bench/serving/live_opencode.sh
../../../inkling/model inkling-2-mini bench/serving/opencode_inkling.json OUT --cache-gb 6 --mem mlock --chunk 512 --span 8192
--disk ../../../inkling/model/.dmeng/prefix --disk-every 1024`. WORKER_ARGS for the 35B: `--cache-gb 4.6 --mem mlock --chunk 512
--span 8192 --stream 3 --disk ../../../q36/model/.dmeng/prefix --ahead-span 512`.

Timing (to run on a quiet machine; `TIMED=1` makes slot.sh hold QUIET, `SERVE=1` PRIORITY):
```
# new user turn TTFT, idle-time prefill off / on, gaps 5, 15, 30 s, interleaved twice
zsh bench/serving/queue_serve_next.sh ahead35        # off1 on1 off2 on2 -> results/serving/ahead/{off,on}{1,2}.*
../../.venv/bin/python bench/serving/ahead_table.py results/serving/ahead off1 on1 off2 on2
# the first request after a restart, without / with the saved hot set, interleaved twice
zsh bench/serving/queue_serve_next.sh hotset35       # "save off on off on" -> results/serving/hotset/q36.*
# a span-size choice for the prediction: rerun ahead35 with --ahead-span 256 / 1024 / 8192 in W35 (fewer, longer spans read less; a cancel keeps less)
```

## Edits inside `tools/dmserve.cpp` (all small and local; the generation loop is untouched)

1. header comment: the `A` protocol line;
2. includes `src/serving/ahead.h`, `src/serving/hotset.h`;
3. `Args`: `ahead_span`, `ahead_park`, `check_ahead_a/_b`, `check_ahead_len`, `hotset`; their flags `--ahead-span`,
   `--ahead-park`, `--check-ahead A B`, `--check-ahead-len`, `--hotset`;
4. `Req`: `t_arrive` (set by the reader);
5. after `run_bench`: the `--check-ahead` mode (calls `check_ahead()`);
6. after `PrefixCache pc`: `AheadRunner ahead(...)`, the optional `HotSet`, its load; READY gets `"hotset"`;
7. reader thread: parse `A` lines (`ahead.submit`); on `R`: `ahead.on_request()`, `hot->stop()`, `t_arrive`; at EOF / `Q`:
   `ahead.on_request()`;
8. main loop wait predicate: `|| ahead.pending() || hot->pending()`; idle branch: one hot-set batch, else `ahead.run()`;
9. at a request's start: the hot-set refill's end line; the `B` line gets `ms_wait_worker`;
10. at exit: `hot->save()` and an `H` line.

Other edited shared files: `include/dmeng/session.h` + `src/serving/session.cpp` (`ahead()`, `protect_upto_`, `no_park_`,
`from_ahead` parked entries, stats), `CMakeLists.txt` (two sources in `dmengserve`), `python/serve/dmserve.py`.

## What did not work / what remains

- The first slot runs lost races with other queues (`mlock failed`, another model started in the same second);
  `slot.sh` now retries those.
- A waiting title request held the prediction back in the first Inkling run (fixed: only a waiting main request does).
- A 5 s gap keeps nothing with 512-position spans (the first span takes ~5-7 s here); whether smaller first spans pay
  is a timing question (above).
- Not built: persisting the prediction's state to disk (it is lost when the worker's idle release stops it after 5 min).
- Inkling: no warm-start (new process, disk prefix) request measured; the peak footprint at a 6 GB cache was 9.69 GB.
- Worker restart: tested with a stand-in worker only. Python front end remains (see 4).
