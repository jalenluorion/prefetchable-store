# Cache policy by cache size, 35B and Inkling (branch cache-policy, 2026-11-22)

Lossless decode in the unified engine (dmeng). Everything below is reads, not time: MB read per token and hit rate
do not depend on machine load. The Mac was shared by several queues (and swapping) when these ran, so no timing is
reported; the timing cells for the clean campaign are listed at the end of each section.

## Answer

**Default: `--evict auto`** (now the default of dmeng and of `CoreOpts`): the hazard keep-score (the phone's online
per-layer table; no training, no router), with uniform per-layer caps while the cache holds fewer than 4 records per
layer per pick (`slots < 4 x K x layers`), else one pool per record format. As switch values:

| model | cache | switch | what it is |
|---|---|---|---|
| 35B | below 2.26 GB (1,280 slots): 0.8 / 1.27 / 2.0 | `--evict auto` = `--evict hazard --layer-caps uniform` | hazard, per-layer caps |
| 35B | 2.26 GB and up: 3.3 / 5 / 6.6 | `--evict auto` = `--evict hazard --layer-caps none` | hazard, one pool |
| Inkling | 3.8 GB and up (768 slots): 4 / 6 / 7.8 | `--evict auto` = `--evict hazard --layer-caps none` | hazard, one pool per format |

`--evict freq --layer-caps none` gives the old default exactly. The learned ranker (`--evict-policy` with
`babel/q36_d_evict3_policy.json` or `babel/ink_d_evict3_policy.json`) is optional, not a default: it needs its lag-2
router running every token (2.2-4.1 ms/token of router compute and 200-230 MB/token of router weights read from
memory, measured on a loaded Mac) and costs 0.7-1.4 ms/token of policy time as built, for 1-2.3 % fewer bytes than hazard
on 35B and 2.2-4.9 % on Inkling (study means). At the Mac's 3.4 GB/s that is about 0.4-0.6 ms/token of reads on 35B
and 1.4-2.0 ms on Inkling, less than the router alone. It can pay only where the same router already runs for naming,
and then only on Inkling; the timing cells below decide that.

Where the rule comes from (the design study, babel `design_table.py`, mean MB/token over five domains, 32 windows of
1,024 tokens each; `babel/q36_d_evict3.json`, `babel/ink_d_evict3.json`):

| model | GB | decay (old default) | hazard, one pool | hazard, caps | ranker v3, one pool | ranker v3, caps | Belady | rule picks | rule vs old default | best v3 vs rule |
|---|---|---|---|---|---|---|---|---|---|---|
| 35B | 0.8 | 401.7 | 334.0 | **308.9** | 336.5 | 311.0 | 216.2 | caps | -23.1 % | +0.7 % |
| 35B | 1.27 | 296.2 | 266.1 | **253.4** | 269.9 | 255.5 | 167.5 | caps | -14.4 % | +0.8 % |
| 35B | 2.0 | 214.9 | 206.0 | **201.3** | 206.3 | 201.9 | 122.9 | caps | -6.3 % | +0.3 % |
| 35B | 3.3 | 150.5 | **140.7** | 141.1 | 139.2 | 139.8 | 77.8 | pool | -6.5 % | -1.1 % |
| 35B | 5.0 | 97.6 | **89.2** | 93.3 | 87.4 | 91.7 | 47.3 | pool | -8.6 % | -2.0 % |
| 35B | 6.6 | 65.1 | **59.9** | 65.8 | 58.5 | 64.3 | 32.1 | pool | -8.0 % | -2.3 % |
| Inkling | 4.0 | 342.5 | **306.7** | 312.2 | 299.9 | 305.6 | 169.6 | pool | -10.5 % | -2.2 % |
| Inkling | 6.0 | 180.9 | **166.3** | 174.3 | 160.3 | 167.0 | 86.4 | pool | -8.1 % | -3.6 % |
| Inkling | 7.8 | 104.8 | **94.3** | 101.5 | 89.7 | 96.1 | 48.1 | pool | -10.0 % | -4.9 % |

The rule picks the better hazard mode in all nine cells (35B crossover between 2.0 and 3.3 GB, where the two modes are
within 1-2 %: the phone study's pitt 2.5 GB 155 vs 156). Inkling has no study cell below 4 GB, so its 3.8 GB switch
point is the rule's, not a measurement.

## 1. In the engine: reads match the simulator, ids identical

Each run records its routing (`--record`) and the released and early router rows (`--evict-trace`); the study's
simulator core (`babel/eviction_v3.py simulate()`, unchanged, driven by `engine/sim_trace.py`) replays the same
routing. The ranker's shortlist is taken in (hazard score, record id) order there (`--det`), as the engine does (numpy's
argpartition breaks ties arbitrarily; without `--det` the simulator differs by 5 hits of 100,844 at 35B 3.3 GB). Text: the
Pittsburgh prompt, 400 generated tokens on 35B (399 counted), 160 on Inkling (159 counted), lossless, `--temp 0`.

| model | GB | slots | policy | ids vs reference | hit rate engine / sim | MB/token engine / sim |
|---|---|---|---|---|---|---|
| 35B | 0.8 | 452 | old default (freq) | identical (400) | 0.3295 / 0.3318 | 379.66 / 378.36 |
| 35B | 0.8 | 452 | hazard + caps (auto) | identical | 0.5105 / 0.5105 | 277.20 / 277.20 |
| 35B | 1.27 | 720 | old default | identical | 0.5326 / 0.5324 | 264.64 / 264.75 |
| 35B | 1.27 | 720 | hazard + caps (auto) | identical | 0.6015 / 0.6015 | 225.64 / 225.64 |
| 35B | 2.0 | 1130 | old default | identical | 0.6618 / 0.6618 | 191.51 / 191.52 |
| 35B | 2.0 | 1130 | **auto (no flag)** | identical | 0.6819 / 0.6819 | 180.14 / 180.14 |
| 35B | 3.3 | 1864 | old default | identical | 0.7639 / 0.7639 | 133.69 / 133.69 |
| 35B | 3.3 | 1864 | hazard, one pool (auto) | identical | 0.7844 / 0.7844 | 122.08 / 122.08 |
| 35B | 3.3 | 1864 | ranker v3 | identical | 0.7898 / 0.7898 | 119.01 / 119.01 |
| 35B | 5.0 | 2825 | old default | identical | 0.8606 / 0.8606 | 78.94 / 78.94 |
| 35B | 5.0 | 2825 | hazard, one pool (auto) | identical | 0.8718 / 0.8718 | 72.57 / 72.57 |
| 35B | 5.0 | 2825 | ranker v3 | identical | 0.8783 / 0.8783 | 68.89 / 68.89 |
| 35B | 6.6 | 3729 | old default | identical | 0.9146 / 0.9146 | 48.33 / 48.33 |
| 35B | 6.6 | 3729 | hazard, one pool (auto) | identical | 0.9196 / 0.9196 | 45.53 / 45.53 |
| 35B | 6.6 | 3729 | ranker v3 | identical | 0.9231 / 0.9231 | 43.57 / 43.57 |
| Inkling | 4.0 | 712 + 97 | old default | identical (160) | 0.8514 / 0.8514 | 283.37 / 283.42 |
| Inkling | 4.0 | 712 + 97 | hazard, one pool (auto) | identical | 0.8570 / 0.8570 | 274.15 / 274.15 |
| Inkling | 4.0 | 712 + 97 | ranker v3 | identical | 0.8664 / 0.8664 | 256.55 / 256.55 |
| Inkling | 6.0 | 1069 + 145 | old default | identical | 0.9279 / 0.9279 | 140.29 / 140.37 |
| Inkling | 6.0 | 1069 + 145 | hazard, one pool (auto) | identical | 0.9306 / 0.9306 | 135.19 / 135.19 |
| Inkling | 6.0 | 1069 + 145 | ranker v3 | identical | 0.9363 / 0.9363 | 124.19 / 124.19 |
| Inkling | 7.8 | 1389 + 189 | old default | replay * | 0.9623 / 0.9623 | 74.75 / 74.75 |
| Inkling | 7.8 | 1389 + 189 | hazard, one pool (auto) | replay * | 0.9640 / 0.9640 | 71.51 / 71.51 |
| Inkling | 7.8 | 1389 + 189 | ranker v3 | replay * | 0.9670 / 0.9670 | 65.21 / 65.21 |

References: 35B `engine/results/stage4/runs/r_r8_pitt_gb3.3_r1.ids`, Inkling `engine/results/inkling_draft/runs/
pitt_one_gb6_r1.ids` (the old engine); the routing of every run of a model is byte-identical to its first run's.
\* Inkling 7.8 GB: no model run (memory); the 6 GB run's routing and router rows replayed through `dm::Cache` at the
7.8 GB slot counts (`tests/evict_check/evict_replay`, which reproduces every engine run above exactly) and through the
simulator. The old default differs from the simulator only by its tie-break (engine: lowest slot; simulator: lowest
record id; the phone study's 0.8 GB 379.7 vs 378.4). Hazard and the ranker agree to every digit in every cell, and the
replay agrees eviction for eviction with the old full-scan hazard (36 cells, below). The same replay with caps at
Inkling 4 / 6 / 7.8 GB: hazard 279.1 / 142.6 / 77.5 MB/token against one pool's 274.2 / 135.2 / 71.5, as the rule says.

On this one text the ranker's cut is larger than the study's mean (35B -2.5 / -5.1 / -4.3 %, Inkling -6.4 / -8.1 /
-8.8 % against hazard); the study's five domains are the number to plan with.

### Cheap hazard

`Cache::hz_victim`: per layer the lowest (keep score, record id) among its held records, kept until the layer's
statistics change (its touches, the token's tick) or that record leaves its slot; a victim is the lowest of the pool's
layers (the layer's own with caps), pins and reads in flight checked on the winner only. The tick reuses each record's
bucket for the next token and takes floor(log2) from the exponent bits (libm's log2 only just below a power of two).
`tests/evict_check` replays three recorded texts (pitt 408, coding 756, agent 2,873 positions) x 452 / 720 / 1130 /
1864 / 2825 / 3729 slots x one pool / caps: **identical evictions to the old scan in all 36 cells**. Policy time in
those replays 0.06-0.17 ms/token in one pool (0.04-0.11 with caps) against 0.39-1.63 ms for the scan in one pool (same machine, same minute, loaded);
in the engine runs above `policy_ms_per_token` was 0.10-0.33 for hazard (the phone study measured 1.12-1.27 ms for the
scan at 3.3 GB). `DMENG_HZ_CHECK=1` runs the scan beside it and aborts on any difference.

### The ranker loader

`Cache::load_policy` reads any file with `"features"` as a ranker (`load_ranker`): features by name (`log2_c<h>`,
`log2_1p_since_use`, `log2_1p_since_use2`, `rel_p_now`, `rel_p_ema<h>`, `rt_p_now`, `rt_p_ema<h>`, `rt_p_max<h>`,
`log2_1p_pos`, `hazard_p`, `log2_hazard_p`), per-physical-layer bias and weights from the JSON, kind `mlp` or `lin`,
the prefilter (128) by hazard order. Released-router rows (softmax top 32; Inkling: top 32 by sigmoid + bias, valued by
the sigmoid) are applied at each step's end, early-router rows when the router finishes (also with `--router-stats`,
naming nothing), applications indexed as Inkling's 48 over 24 layers, two pools, loop hold as the engine. dmeng checks
the router's lag against the file.

### Timing cells (for the clean campaign)

Common: `dmeng --ctx 16384 --temp 0 --mem mlock --prompt-ids $(cat pitt_ids) --n 400` (35B adds `--prec i16`; Inkling
uses its own pitt ids, `engine/results/inkling_draft/pitt_ids.txt`). Per cell, three arms:

    35B   G in 3.3 5 6.6:   --model $Q36 --cache-gb G --evict freq --layer-caps none            # old default
                            --model $Q36 --cache-gb G                                           # auto (hazard)
                            --model $Q36 --cache-gb G --evict-policy results/cache-policy/babel/q36_d_evict3_policy.json \
                              --routers $Q36/routers/q36_router_full_k2_h512.dmr --router-stats --layer-caps none
    Inkling G in 4 6 7.8:   --model $INK --cache-gb G --evict freq --layer-caps none
                            --model $INK --cache-gb G
                            --model $INK --cache-gb G --evict-policy results/cache-policy/babel/ink_d_evict3_policy.json \
                              --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --router-stats --layer-caps none
    with naming (where the ranker can pay): the same three with --prefetch router --routers <the k2 file> in all arms

## 2. Background read class for cross-token names

`Readers` has a third class (2, background): a read of that class starts only while no urgent or speculative read is
queued or in flight, at most one at a time (`max_bg`); a step that needs a background read before it started promotes
it to urgent (`bg_promoted`), and cancel / cancel_stale see the class. `Xtok::background` sends a cross-token source's
names there; `dmeng --xtok-bg F` adds such a source beside `--xtok`, so source 47 (speculative, `--rd-prio 1`) and source
41 (background) run together. Names only schedule reads; the released router decides.

Gate runs (Inkling, pitt ids, 200 tokens, lossless, default eviction `auto`, `--prefetch router` with
`ink_r_fast/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5`, one run per cell, AC power; reads only, no timing). ids: the
first 160 equal the old engine's `pitt_one_gb6_r1.ids`, all 200 equal the k2-only run's. Misses = reads issued on the
spot (not named), per token, by application group; late = named but not landed when its step ran, per token; jobs over
the 199 decode tokens.

| GB | names | ids | MB/token | misses 0-5 / 6-41 / 42-47 | late / token | jobs urgent / speculative / background | background promoted | hit rate |
|---|---|---|---|---|---|---|---|---|
| 6 | k2 only | identical | 155.6 | 3.26 / 6.01 / 2.49 | 5.92 | 2348 / 3796 / 0 | 0 | 0.9539 |
| 6 | + source 47, `--rd-prio 1` | identical | 157.5 | 3.09 / 6.04 / 2.51 | 4.32 | 2324 / 3903 / 0 | 0 | 0.9585 |
| 6 | + source 41, background | identical | 157.5 | 3.10 / 6.03 / 2.52 | 4.82 | 2335 / 3387 / 502 | 25 | 0.9571 |
| 6 | + both | identical | 158.8 | 3.02 / 6.04 / 2.53 | 4.51 | 2325 / 3472 / 489 | 31 | 0.9581 |
| 7.8 | k2 only | identical | 77.6 | 1.48 / 3.54 / 1.58 | 2.80 | 1318 / 1684 / 0 | 0 | 0.9755 |
| 7.8 | + source 47, `--rd-prio 1` | identical | 78.5 | 1.42 / 3.58 / 1.58 | 2.09 | 1312 / 1732 / 0 | 0 | 0.9774 |
| 7.8 | + source 41, background | identical | 78.1 | 1.41 / 3.56 / 1.58 | 2.16 | 1303 / 1538 / 183 | 16 | 0.9773 |
| 7.8 | + both | identical | 78.8 | 1.38 / 3.58 / 1.58 | 2.03 | 1301 / 1576 / 180 | 15 | 0.9777 |

The cross-token names add 1.9-3.2 MB/token at 6 GB and 0.5-1.2 at 7.8; the first six applications' on-the-spot misses
fall by 0.16-0.24 (6 GB) and 0.06-0.10 (7.8 GB) per token; the last six's misses do not change (the background class
only moves when the drive is used, which only timing shows). About 5 % (6 GB) to 9 % (7.8 GB) of the background reads
had not started when their step came and were promoted.

### Timing cells

Common: `dmeng --model $INK --cache-gb G --ctx 16384 --temp 0 --mem mlock --prompt-ids $(cat
engine/results/inkling_draft/pitt_ids.txt) --n 200 --steps-out F.steps --json F.json`, G in 6 and 7.8, with
`A = --prefetch router --routers $INK/routers/ink_r_fast/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5`:

    routers alone:            A
    source 47, priority:      A --xtok $INK/routers/ink_x_main/ink_router_xtok_s47_t0-5_b8.dmr --rd-prio 1
    source 41, background:    A --xtok-bg $INK/routers/ink_x_main/ink_router_xtok_s41_t0-5_b8.dmr
    both:                     A --xtok $INK/routers/ink_x_main/ink_router_xtok_s47_t0-5_b8.dmr --rd-prio 1 \
                                --xtok-bg $INK/routers/ink_x_main/ink_router_xtok_s41_t0-5_b8.dmr
    (control, source 41 in the normal class: A --xtok $INK/routers/ink_x_main/ink_router_xtok_s41_t0-5_b8.dmr)

Wait by group is in the JSON (`wait_ms_by_group`: before_loop = applications 0-5, looped = 6-41, after_loop = 42-47;
per application in `wait_ms_by_step`), with `misses_by_step` and `read_queue.bg_jobs / bg_promoted / bg_queue_ms`.

## 3. Longer-lead names for Inkling's last six applications (offline sizing)

**The sizing says it does not pay; nothing was built.** From the 6 GB run's recorded routing (hazard, one pool, cache
only: 5.9 misses per token in applications 42-47) and each lead's names, dumped from the routers in
`local/inkling/model/routers/ink_r_best` (`--router-stats --dump-names`, the same text, ids identical), by
`engine/names_size.py` (output `engine/names_size.txt`), per token:

| app | misses | recall@8 k1 / k2 / k4 / k8 / k16 | misses named in the top 8, k1 / k2 / k4 / k8 / k16 | wasted name reads top 8, MB, k2 / k16 |
|---|---|---|---|---|
| 42 | 0.78 | 0.82 / 0.82 / 0.81 / 0.80 / 0.79 | 0.65 / 0.59 / 0.65 / 0.56 / 0.59 | 1.2 / 1.2 |
| 43 | 0.94 | 0.86 / 0.80 / 0.82 / 0.81 / 0.81 | 0.65 / 0.51 / 0.60 / 0.54 / 0.56 | 1.1 / 1.2 |
| 44 | 1.14 | 0.88 / 0.85 / 0.82 / 0.81 / 0.81 | 0.73 / 0.67 / 0.60 / 0.60 / 0.62 | 2.3 / 2.7 |
| 45 | 1.08 | 0.89 / 0.87 / 0.84 / 0.83 / 0.83 | 0.77 / 0.69 / 0.67 / 0.63 / 0.59 | 2.3 / 2.7 |
| 46 | 0.87 | 0.90 / 0.89 / 0.88 / 0.86 / 0.86 | 0.77 / 0.72 / 0.69 / 0.70 / 0.64 | 1.3 / 2.0 |
| 47 | 1.07 | 0.93 / 0.91 / 0.89 / 0.88 / 0.87 | 0.85 / 0.82 / 0.77 / 0.75 / 0.76 | 1.2 / 1.5 |

A longer lead names no more of the misses (k16 names 0.56-0.76 of them against k2's 0.51-0.82; recall@8 changes by
-0.04 to +0.004 from k2 to k16), so all it could buy is earlier reads of the same names. A drive-queue model (reads first in,
first out at 3.4 GB/s, a name read issued when its router runs, i.e. at the start of application a - lead, a miss not
named read at its own step, 1.25 ms of compute per application = the cleanest k2 timing run, 60 ms/token)
gives, wait ms/token in the last six / all applications: k2 8.6 / 20.1, k2 + k4 on 42-47 8.2 / 20.5, k2 + k8 3.4 / 21.1,
k2 + k16 2.3 / 19.4. The last six's wait falls, but the 8.2 MB records named 8-16 applications earlier now share the
drive with the looped layers' reads, and the total moves by +1.0 to -0.7 ms; the model under-states the looped wait
(2.8 ms against 9.7 measured), so that contention is larger in the engine than modelled. Against it: the extra router
for six targets (the full 48-target sets measured 2.2-4.1 ms/token and 204-227 MB/token of router weights here, so about
0.3-0.5 ms and 26 MB for six) and +2.1 MB/token of wasted name reads (k16, included in the model). Net at best
+0.2 ms/token, inside the model's error: not worth a router set. (The per-application wait measured in the timing runs,
8.0 ms in 42-47 with a perfect 16-ahead router at 0.6, is mostly lateness of named reads behind other reads, which is
what the background class in section 2 addresses.)

## Files

`babel/`: the study's policy files, results and simulator (copied from babel, personal paths removed).
`engine/`: `sim_trace.py` (an engine run through the simulator), `summary.py` (the table of section 1),
`cmp_replay_sim.sh`, `run_ident.sh` (old scan vs incremental), `caps_rule_test.cpp`, `names_size.py`, the run scripts.
Code: `src/core/store.cpp` (`hz_victim`, `load_ranker`, `rk_victim`, `released_top` / `router_top` / `step_end`,
`caps_rule`, the readers' background class), `src/core/moe.cpp` (`CoreOpts::evict` / `layer_caps`, released rows),
`src/core/address.cpp` (router rows, `Xtok::background`), `tools/dmeng.cpp` (`--evict auto`, `--evict-trace`,
`--xtok-bg`, per-step misses / wait in the JSON), `tests/evict_check/evict_replay.cpp`.
