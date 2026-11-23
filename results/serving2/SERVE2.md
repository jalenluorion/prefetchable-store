# serve2: cache size from the machine, agent profiles, live tasks with the eviction fix

Branch `serve2` from `rc` 5f7687a. Base M4 MacBook Pro 16 GB, AC power for every run (`*.log.power`, guard
`STATE_BEFORE`). Other agents' processes were running: others' CPU 112-337% mean in the guard logs, swap 9-13 GB in use.

## 1. `--cache-gb auto` and the memory-pressure watcher (`src/serving/memwatch.{h,cpp}`)

Measured on this Mac with no model process (vm_stat, 16 KB pages): free 1.3 GB, active 5.4 GB, inactive 5.3 GB, wired
2.3 GB, compressor 2.1 GB; free + inactive + speculative + purgeable 6.9-9.7 GB over the session; kern.memorystatus_level
48-73%. Other applications open on the machine hold the active memory.

Rule (all from the machine at start, the resident weights already loaded):
- ceiling = physical (17.18 GB) - reserve (`--reserve-gb`, default 5.5) - the process's other memory (its footprint
  now + KV for 16K positions + checkpoints + prefill scratch 0.5 + margin 0.3), clamped to [2 GB, the store's size].
  35B: 17.18 - 5.5 - 3.48 = **8.2 GB**; Inkling: **8.8 GB**. Slots for the ceiling are reserved, not locked.
- start = min(ceiling, free now - the process's memory still to come - headroom 3.0 GB): only this much is locked
  (`Cache::defer_lock` + `lock_start`; growth locks more slots).
- watcher (every 2 s, woken at once by the system's memory-pressure notification): pressure warn/critical or a free level
  under 20% shrinks the budget by 1 GB per check down to 2 GB; normal pressure with a free level at or above 35% for 30 s
  grows it by 1 GB, up to the ceiling but not back into a size that met pressure until 10 min of calm. Only the budget
  moves (`Cache::set_budget`, applied at a token, a prefill layer, an application of a multi-position pass, or by the
  idle worker).

The first version locked the whole start size from physical memory alone (8.2 GB): the free level fell to 4% (critical)
within 2 s of start and the watcher took it to 5.2 GB before the first token (`memwatch_q36_rule1.jsonl`). Hence the
start from free memory and the 3 GB headroom (warn pressure / a free level under 20% came with 2.5-3.4 GB left).

Footprint at each size (35B, `memwatch_q36.jsonl`; process footprint at the moment of each change):

| budget GB | 6.28 | 5.28 | 4.28 | 3.28 | 2.28 | 2.0 |
|---|---|---|---|---|---|---|
| footprint GB | 8.68 | 7.85 | 6.90 | 5.95 | 4.99 | 4.76 |

(about budget + 2.5-2.7 GB). Inkling (`memwatch_ink.jsonl`): 5 GB 7.92, 4 GB 7.16, 3 GB 6.40, 2 GB 5.63 GB on a short
prompt; but its cache cannot go under ~3.6 GB in force: each of its two record pools keeps the slots one step pins
(`min_cap`), so a 2 GB request leaves 3.6 GB. The watcher now takes that floor as its minimum (after the runs below).

Gates: the same 48 greedy tokens at 5.28, 2.0 and 5.0 GB in one process (35B, `tokens_identical: true`); Inkling: the same 48 tokens at
2.0 (in force 3.6), 2.0 and 4.0 GB (`tokens_identical: true`). Two bugs found and fixed on the way (both lossless):
- a slot past a lowered budget that was in use when the budget fell stayed held, and the hazard (and ranker) eviction
  then reused it, so the memory never came back (live 35B run 1: footprint 8.3 GB at a 2 GB budget). Victims past the cap
  are now given back and the choice made again (`Cache::victim_in`).
- with the first fix alone, Inkling stalled at its floor (a step's pins filled the budget and nothing past it could be
  reused: 0% CPU for 13 min, `memwatch_ink_stalled.jsonl`); the last past-cap slot is now reused when nothing within
  the budget can go, as before the fix.
- Inkling's span prefill never reached a budget boundary (a whole prompt is one span): the budget now applies at each
  application of a multi-position pass (`Core::moe_chunk`).

What this machine gave the cache, live: the 35B started at 3.56 GB, grew to 4.56 GB, met pressure, settled at 2.0-2.56
GB; Inkling started at 5.31 GB and was at its 3.6 GB floor before the end of the cold prefill (footprint 8.2-8.8 GB,
of which 0.74 GB is Inkling's span-prefill scratch, kept between prefills). The 8 GB the clean timing used is not available here without pushing the desktop's memory into
compression and swap.

## 2. Agent profiles (`python/serve/agent_profile.py`)

The front end applies them when no worker arguments follow `--` (`--profile none` turns that off):
- 35B: `--routers routers/lin/q36_router_lin_k2.dmr --m 8` if present, `--evict auto`, `--stream 3`;
- Inkling: `--routers routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5` (ink_r_fast if absent),
  `--xtok routers/ink_x_main/ink_router_xtok_s47_t0-5_b8.dmr --rd-prio 1`, `--disk-every 1024`;
- both: `--cache-gb auto`, the disk prefix under `<model>/.dmeng/prefix`, idle-time prefill on (front end), drafts
  `--draft lookup` with the request's tool-call skeletons (Inkling's now come from its own template:
  `generic/chat_format.tool_skeletons`). One switch: front end `--drafts 0|1`.
- dmserve gained the flags the profile needs: `--evict`, `--rt-tau`, `--xtok`, `--xtok-tau`, `--rd-prio`.

Not in the profile: Inkling's draft head. It costs ~0.75 GB (0.64 GB weights + ~0.1 GB state at a 16K context, from
`results/ink_draft/runs`), i.e. 0.75 GB less cache under the auto rule, and dmserve does not feed it the features of
positions the prefix cache restores (disk, parked, checkpoints), so it is not wired into the server.

Drafts apply to greedy requests only (temperature 0). Stock OpenCode sends no temperature, so the release's sampling
defaults apply (temperature 1.0) and the drafts were inactive in both live tasks. Turning them on for agent turns means
serving OpenCode greedily: a behaviour choice, not made here.

## 3. Live stock-OpenCode tasks on the profile (`bench/serving/live_profile.sh`; `live_*`)

Server given only the model folder; OpenCode 1.19.0 with the endpoint only; the task of RESULTS-AGENTIC.md; cold = the
disk prefix moved aside first; warm = a new server process, the task's first request again. Times count as measured
on this machine, but the cache was what the watcher allowed (above), not 8 GB.

Inkling (`live_ink.*`): **tests pass** (3: mean, median, the empty-list ValueError). 22 main requests (it used more
tool rounds than the 35B: reads, edits, test runs, one call per round).

| | value |
|---|---|
| request 1, cold (8,362 tokens, all computed) | TTFT 226.9 s |
| warm start (new process, request 1 again: 33 tokens from the disk prefix) | TTFT 4.3 s (client), 3.9 s (server) |
| tool rounds (20 requests, 7-294 tokens recomputed each) | TTFT 1.4-13.8 s, median 6.4 s |
| second user message (template keeps reasoning: 20 tokens recomputed) | TTFT 3.8 s |
| decode, all requests | 1.9-4.6 tok/s (median 4.0); hit rate 0.68-0.77; 477-650 MB read per token |
| footprint at request ends / guard peak | 8.2-8.8 GB / 10.19 GB |

Walls: turn 1 797 s, turn 2 305 s; others' CPU 183% mean. Decode is read-bound at the floor cache: ~550 MB per token at
the drive's ~3.3 GB/s alone is ~170 ms.


35B (`live_q36.*`): **tests pass** (3: mean, median, the empty-list ValueError). Cache 3.56 -> 4.56 -> 2.56 -> 2.0 GB.

| request | prompt | TTFT s | recomputed | answer tokens | decode tok/s | hit | MB read / token | footprint GB |
|---|---|---|---|---|---|---|---|---|
| 1 task, cold | 7,586 | 95.3 | 7,586 | 56 | 9.4 | 0.82 | 238 | 6.06 |
| 2-6 tool rounds | 7,796-9,088 | 4.3, 7.0, 1.4, 1.2, 1.6 | 154, 328, 29, 20, 31 | 41-375 | 7.9-10.1 | 0.74-0.77 | 277-312 | 6.08-6.16 |
| 7 answer | 9,193 | 2.9 | 59 | 37 | 10.4 | 0.76 | 262 | 6.17 |
| 8 second user message | 9,117 | 25.2 | 1,566 (the template drops turn 1's reasoning; the idle prefill was cancelled: `run --continue` came at once) | 505 | 7.7 | 0.69 | 369 | 5.19 |
| 9-10 | 9,651-9,765 | 1.8, 3.5 | 29, 74 | 38-40 | 8.3-9.3 | 0.69-0.71 | 313-357 | 5.22-5.25 |
| warm start (new process, request 1 again) | 7,586 | 2.1 | 35 (disk prefix) | | 10.1 | | | 5.46 |

Walls: turn 1 231 s, turn 2 109 s. Guard peak footprint 7.96 GB, others' CPU 112% mean.

## Commands
```
zsh bench/serving/queue_serve2.sh mem35 live35 memink liveink    # resumable; each step through slot.sh / guard.py
python python/serve/dmserve.py --model-dir MODEL --port 8090      # the agent profile; --drafts 0, --reserve-gb G, --profile none
```

## Edits inside shared files
`tools/dmserve.cpp` (flags `--cache-gb auto`, `--reserve-gb`, `--cache-min-gb`, `--headroom-gb`, `--mem-watch`,
`--mem-low-pct`, `--mem-high-pct`, `--mem-grow-after-s`, `--mem-retry-s`, `--evict`, `--rt-tau`, `--xtok`, `--xtok-tau`,
`--rd-prio`; the auto size before `Core::init`; the cross-token hook; the watcher; the idle loop applies a budget; a `P`
line sets the watcher's thresholds for tests; READY carries the rule's numbers and `mem`); `include/dmeng/core.h`
(`cache_start_gb`), `src/core/moe.cpp` (deferred lock), `include/dmeng/store.h` + `src/core/store.cpp` (`defer_lock`,
`lock_start`, past-cap victims), `src/core/moe_chunk.cpp` (budget per application), `python/serve/dmserve.py`
(profile, `--drafts`, `--log-bodies`, `cache_budget` events), `python/serve/generic/chat_format.py` (`tool_skeletons`).
