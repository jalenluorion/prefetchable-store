# Inkling decode: compute per token, stage by stage

Branch `ink-speed` (from `v1` 3f71bdf). Question: can Inkling 2 Mini decode at 20+ (target 25) tokens per second on the
base M4 MacBook Pro 16 GB, output bit-identical? This file: where the decode's compute goes, what changed, the gates,
and how far the result is from 40 ms per token. The machine was shared all night (others' CPU 50-250% during every run,
a timing campaign in the same queue), so the per-stage evidence is the counting build (instructions and cycles per
thread, summed over threads); wall-clock numbers are labelled with others' CPU and are not comparable run to run.

## Bytes per token by stage (fixed by the model and the settings; MB = 10^6 bytes)

| stage (engine timer) | Inkling MB / token | what | 35B MB / token |
|---|---:|---|---:|
| experts | 1,839 | 44 NVFP4 applications x 8 x 4.49 MB + 4 MXFP8 x 8 x 8.22 MB | 700 (566 AFF4 routed + 134 8-bit shared) |
| attention projections (`attn_proj`) | 1,024 | q, g (3072 rows), k, v (1024), r (384) at 8 bits + the f32 sink rows, x 48 applications (layers 6-17 three times) | 1,007 (30 linear + 10 full layers) |
| attention core + fused output projection (`attn_core`) | 361 (o) | o (2304 x 3072, 8 bits) x 48; the heads read the KV cache | 267 (linear layers' o) + 89 (full layers' o, `attn_out`) |
| output head | 492 | 201,024 x 2304 at 8 bits | 524 |
| message gates | 170 | 24 message applications x 4 gates (bf16 down, f32 forget / gain) | - |
| model's router | 28 | 128 x 2304 bf16 x 48 | 42 |
| our routers (`prefetch`) | 226 (8-bit file), 120 (4-bit file) | `router_weight_mb_per_token` | - |
| **total** | 4,140 (3,914 without our routers) | | 2,629 + ours |

At the machine's read rate (105 GB/s at 8 threads, 107 at 10) Inkling's 3.91 GB are 37.3 ms, 4.14 GB with our 8-bit
routers 39.4 ms, with the 4-bit router file 38.4 ms. That is the floor for exact decoding of this model on this machine
with every weight read once per use: 25 tok/s (40 ms) leaves 0.6-2.7 ms for everything that is not a weight read.

## Per-stage counts, before and after (counting build, default settings: exact, our routers, 7.8 GB, 8 threads)

Pittsburgh prompt (10 ids) + 200 tokens, single steps (`--prefill-chunk 1`, so the prompt positions are decode steps
too; per token = total / 210). Instructions and cycles are summed over the threads that ran the stage (pool work
includes the in-phase spin-waits); "serial" is the main thread alone while the pool waits.

| stage | M ins before | after | M cyc before | after | serial M cyc before | after |
|---|---:|---:|---:|---:|---:|---:|
| embed | 0.1 | 0.1 | 0.1 | 0.1 | 0.11 | 0.09 |
| depth_attn | 14.5 | 14.5 | 3.9 | 3.8 | 2.80 | 2.77 |
| msg_gates | 145.6 | 135.5 | 51.3 | 48.3 | 0.68 | 0.63 |
| norms_quant | 1.5 | 1.5 | 1.2 | 1.2 | 1.18 | 1.18 |
| attn_proj | 390.9 | 386.3 | 370.5 | 356.4 | 0.21 | 0.21 |
| attn_core | 295.8 | 275.9 | 145.0 | 142.6 | 0.15 | 0.14 |
| convs | 6.9 | 6.9 | 1.3 | 1.3 | 1.32 | 1.32 |
| moe_router | 50.7 | 49.7 | 35.3 | 34.0 | 0.48 | 0.46 |
| moe_cache | 0.6 | 0.6 | 0.6 | 0.6 | 0.61 | 0.57 |
| moe_experts | 2257.4 | 1846.0 | 608.3 | 571.2 | 0.51 | 0.48 |
| moe_combine | 0.9 | 0.9 | 0.4 | 0.4 | 0.44 | 0.43 |
| head | 178.6 | 178.7 | 124.4 | 121.9 | 0.06 | 0.06 |
| prefetch | 76.1 | 76.2 | 50.8 | 49.9 | 3.14 | 3.02 |
| **total** | 3419.5 | 2972.7 | 1393.2 | 1331.8 | 11.69 | 11.34 |

(M = 10^6 per token. `attn_core` holds the KV preparation, the heads and the fused output projection; `moe_router` the
model's router rows plus our routers' lag rows in the same fork-join; `prefetch` the rest of our routers. Others' CPU
138% before, 127% after.)

Cycles per weight byte, the efficiency that matters for a weight-streaming stage (the head, one large fork-join of the
same 8-bit kernel, is the reference: it runs at the read rate in a clean run):

| stage | Inkling before | after | 35B (same night) |
|---|---:|---:|---:|
| experts | 0.331 (1.23 ins/B) | 0.311 (1.00 ins/B) | 0.316 (0.82 ins/B) |
| attention projections | 0.362 (0.38 ins/B) | 0.348 | 0.285 (0.37 ins/B) |
| attention + output projection | 0.40 | 0.395 | 0.46 (linear attention, its state update included) |
| output head | 0.253 | 0.248 | 0.263 |
| message gates | 0.30 | 0.28 | - |
| our routers (`moe_router` lag rows + `prefetch`) | 0.30 (68.6 M cycles for 226 MB) | | - |

What the counts say, stage by stage (the brief's candidates):

- **Experts (1.84 GB): the NVFP4 kernel's arithmetic.** 1.23 instructions per weight byte at IPC 3.7: the most
  instruction-heavy stage, and the E-cores (4 of the 8 threads) cannot issue it at the read rate. The no-model streaming
  bench (8 threads, 1 GB of rows, the night's load): the v1 NVFP4 kernel 73-88 GB/s, the new one 81-91, plain reads and
  the 8-bit and MXFP8 kernels 98-109. Changed (below): instructions -18%, cycles -6%. The rest of the gap to the read
  rate is still the kernel (TBL decode + two SDOT planes per weight are fixed by the format and the I16 activations).
- **Attention projections (1.02 GB): our code, not the kernel.** The same 8-bit kernel runs the head at 0.25 cycles per
  byte and the projections at 0.36. In isolation the projections' exact call pattern (48 calls of q, g, k, v, r from
  distinct memory, chunks of 32 rows, the input quantised inside the call) runs at 91-99 GB/s, 211-232 us per call
  (no-model bench through `run_rows_q`, others' CPU 96%); in the model the stage takes about 15.8 ms = 329 us per call
  (the clean run), with the threads busy (370 M cycles per token = 275 us per call per thread at 3.5 GHz). So about
  100 us per call is lost inside the model, not to the kernel, the chunking, the joins or the input quantisation (the
  three-input quantisation costs 16-20 us per call of that). Not found tonight; the next experiment is the same stage
  with the readers idle (`--skip-reads --force-routing` on a recorded routing) against the bench.
- **Attention core (short context): the output projection.** `attn_core` = KV preparation + heads + the fused output
  projection (361 MB); at the Pittsburgh lengths (< 210 positions) the heads themselves are a small part. The decode
  attention had the prefill's first inefficiency (every key and value row loaded once per query head, the value sums
  through memory); fixed (below), it matters with context length.
- **Serial work on the main thread: 11.3 M cycles per token (about 3 ms at P-core clocks).** The largest pieces: our
  routers' hooks 3.0, depth attention with its two norms 2.8, the post-MLP convolution and residual 1.3, norms 1.2. The
  norms are a sequential double sum (2304 dependent adds, about 2 us each) that cannot be reordered without changing
  bits; no single piece is worth a fork-join of its own.
- **Three passes over the looped layers' projections**: the model's bytes (36 of the 48 applications are layers 6-17
  again); nothing to remove exactly.
- **The routers**: below (bytes without touching the weights).
- **Thread imbalance / 10 threads**: below.

## What changed (all bit-identical)

1. **The NVFP4 one-vector kernel with I16 activations** (`src/core/kernels.cpp`, `nv_rows_i16`; the expert kernel of 44 of
   48 applications). Two rows at a time share the activation planes and the block scales; the low plane's SDOTs
   accumulate into the high plane's sums shifted by 8 (the same integer (hi << 8) + lo); the E4M3 scales are converted by
   a sign-extending shift (exact, as before) eight at a time. Every float operation, operand and order is the v1
   kernel's. Instructions per weight byte 1.11 -> 0.89 single-thread (`tests/kernels/nv_i16_check.cpp`, which also
   compares every output bit with a copy of the v1 kernel: 25,555 rows, every E4M3 code, K = 2304, 1152, 3072, 320, 192,
   odd row ranges: 0 differ). In the decode: experts 2,257 -> 1,846 M instructions per token, cycles 608 -> 571 M.
2. **Decode attention with each key and value row loaded once for its 3 query heads** (`phase_attn_kv` in
   `src/models/inkling2_loop/inkling2_loop.cpp`, the prefill's `phase_attn_g` scheme for one position): a score job per
   KV head (its 3 heads' clips, bias, scores, softmax), value jobs per (KV head, 64 dimensions) with the 3 heads' sums in
   registers, then the output projection's chunks as before. Each head's arithmetic is the single step's (score product
   not fused, as `phase_attn` compiles it). At short context the attention itself is small next to the output
   projection's 361 MB; the change is for long contexts: after a 4,000-id prompt, 64 decode
   tokens (positions 4,000-4,063; counting build, the phase function's own counters, output projection included in
   both), `phase_attn` (v1) 1,622 M instructions / 485 M cycles per token, `phase_attn_kv` 747 M / 303 M: -54% / -37%,
   about 180 M cycles per token (some 6 ms of the 8 threads) at 4K; ids identical (others' CPU 152% / 184%).
3. **Counting build only**: stage marks in Inkling's and the 35B's single step and in the core's MoE step, the prefetch
   hooks as their own stage (`ST_PREFETCH`). No effect on the default build.

Core files touched (for the 35B branch): `src/core/kernels.cpp` (new I16 NVFP4 rows; `mv_nvfp4`'s F16 / I8 paths
unchanged), `src/core/moe.cpp` and `src/core/pool.cpp` / `include/dmeng/pool.h` (stage marks, a new stage name),
`CMakeLists.txt` (the check target). Tried and dropped: an exact top-k head (below).

## Gates

| gate | result |
|---|---|
| `nv_i16_check` (new kernel vs the v1 kernel, bit for bit) | PASSED, 0 / 25,555 rows differ |
| tiny models: `tests/draft_check.sh` (verify passes with rollbacks in every precision, drafting ids in dmeng and the server) | DRAFT CHECK PASSED (also after the top-k head was dropped) |
| tiny Inkling, v1 build vs this build: 260 teacher-forced positions' logits (window 16: the rings wrap) and 200 greedy ids | identical |
| real model, v1 build vs this build: teacher-forced logits of positions 1000-1100 of a 1,100-id sequence (past the 1,024 window: every sliding ring wraps) | identical (80 MB of logits, `cmp`) |
| real model, 200 greedy ids, default settings (exact, routers), counting build and timing build, prompt by single steps and by the batched prefill | identical to the v1 ids |
| real model, 200 ids with drafting on (`--drafts 1`) | identical to the v1 ids |
| real model, `tests/prefill_check.sh` (1,100 ids: the batched prefill's grouped attention vs single decode steps, which now run `phase_attn_kv`) | PREFILL CHECK PASSED: 0 logit bits, 0 state bytes, 0 KV bytes differ, greedy identical |
| real model, `--fast-head 1` (existing exact greedy head) and the 4-bit router file | ids identical to the v1 ids |

## Wall clock (one pair, 7.8 GB, 200 tokens, others' CPU in brackets: not comparable)

| run | tok/s | ms / token | compute | wait | others' CPU |
|---|---:|---:|---:|---:|---:|
| v1, exact | 13.13 | 76.1 | 62.7 | 13.4 | 93% |
| this branch, exact | 11.35 | 88.1 | 74.2 | 13.9 | 178% |
| v1, fast | 15.72 | 63.6 | 58.7 | 4.9 | 120% |
| this branch, fast | 12.17 | 82.2 | 77.6 | 4.6 | 153% |

The pair measures the machine's load, not the code: the branch's runs met the heavier load (every stage slower,
the shortest fork-joins most). For a load-robust wall-clock comparison the same process ran the v1 code on odd
positions and this branch's on even ones (a scratch build, `DMENG_AB`; both bit-identical, so the decode is the same),
fast mode, 300 tokens, per-step compute excluding read waits:

| alternated per token | this branch | v1 | others' CPU |
|---|---:|---:|---:|
| kernel + attention | 80.3 ms | 77.5 ms | 206% |
| attention only | 69.7 | 71.6 | 191% |
| NVFP4 kernel only | 68.1 | 68.0 | 170% |
| attention only, a 4,000-id prompt by single steps (contexts 0-4,080) | 82.5 | 83.9 | 210% |

The differences are within the run-to-run spread of this night (the "both" row and the two single rows disagree by
more than either effect), so: **no wall-clock gain is resolvable tonight**; the counts say -6% cycles in the experts and
-4.4% in total.

## Bytes per token without touching the weights

- **Exact greedy head (`--fast-head 1`, exists, off by default)**: a 4-bit copy derived at load gives coarse logits and a
  certified bound; 6,658 rows per token (3.3%) are then computed exactly; ids identical. Head bytes 492 -> about 275 MB
  (the 4-bit copy 231 MB + its scales and bounds 44 MB, + 16 MB of exact rows), head cycles 124 -> 94 M per token.
  Cost: +0.27 GB of resident memory (the copy), which comes out of the cache on a 16 GB machine.
- **Exact top-k head for sampling (OpenCode: top-k 20)**: built and measured, then dropped. The same bound with the
  20th-largest lower bound as the threshold left 125,121 of 200,058 rows as candidates per token (temperature 0.8,
  ids identical to the full head, 100 tokens): slower than the full head (258 M vs 114 M head cycles). The 4-bit copy's
  error bound (row error norm x input norm) is too loose for anything past the argmax. A tighter certified bound
  (a 6-bit copy, or per-group error norms) would be needed; not built.
- **Our routers**: 225.8 MB per token with the 8-bit file (`router_ms_per_token` 2.6 in the counting run; 68.6 M cycles
  = the routers-on run minus the routers-off run); the 4-bit file (`ink_r_fast/ink_router_full_k2_h512_b4.dmr`) 119.6 MB,
  1.7 ms, 24 M fewer cycles per token, ids identical, naming slightly weaker (top-8 picks among the 16 named: 0.979 ->
  0.971). A linear router per step (2304 inputs x 128 experts) would be 0.30 MB at 8 bits, 14 MB per token for 48 steps
  (28 MB in bf16): about 1/16 of the 8-bit MLP routers. The choice is not made here.
- **10 threads against 8**: worse, clearly. Fast mode, the same alternated run at 10 threads: 224 ms per token against 80 ms at 8 (others'
  CPU 196%); the plain fast run at 10 threads: 83.9 ms compute against 77.6 at 8. With 10 compute threads on 10 cores
  plus the readers and anything else running, one preempted thread stalls every join (the short fork-joins grew most:
  message gates 4.7 -> 26 ms, router 3.5 -> 21). The streaming bench alone gains about 5% at 10 (NVFP4 85-95 GB/s); the
  engine does not. Keep 8.

## How far from 40 ms, and what is left

Per-stage milliseconds per token, exact decode at 7.8 GB, 8 threads. "Now" is the cleanest earlier run; "after"
applies this branch's measured change (cycles, counting build) where the code changed and is "now" elsewhere; "at the
machine's rate" is the stage's bytes at 105 GB/s, or 0 for work that moves no weights.

| stage | now | after | at the machine's rate | what limits it |
|---|---:|---:|---:|---|
| experts (1.84 GB) | 20.7 | about 19.5 (-6% cycles; not resolvable in wall clock tonight) | 17.5 | our code: the NVFP4 kernel's arithmetic on the E-cores |
| attention projections (1.02 GB) | 15.8 | 15.8 | 9.8 | our code: about 100 us per call lost in the model, cause not found (the kernel and the call pattern reach 91-99 GB/s alone) |
| attention core + output projection (0.36 GB) | 5.2 | 5.2 short context; at 4K context about 6 ms less per token | 3.4 | the machine (o) + our code (the heads) |
| output head (0.49 GB) | 4.7 | 4.7 (2.7 with `--fast-head 1`, greedy only, +0.27 GB resident) | 4.7 (2.7) | the machine; the model's bytes unless the screened head is on |
| our routers (0.23 GB) | 2.5 | 2.5 (1.7 with the 4-bit file) | 2.2 (1.1) | the machine; bytes are a router choice |
| message gates and norms (0.17 GB) | 2.1 | 2.0 | 1.6 | the machine + serial norms |
| model's router (28 MB) | 1.4 | 1.4 | 0.3 | our code: a fork-join and a serial top-k per step for 0.6 MB |
| depth attention | 0.8 | 0.8 | 0 | our code: serial on the main thread |
| **compute** | **53.2** | **about 52** | **39.5** | |
| + waits (fast / exact) | 3.4 / 10.7 | same | | the drive |

**How far from 40 ms.** This branch does not reach it: about 52 ms of compute at short context (about 19 tok/s with no
waits, 18 in `fast`), with the gains tonight only visible in the counts (-4.4% cycles per token at short context, -37%
in the attention at 4K context). The floor for this model at the machine's read rate is 39.5 ms of compute with our
8-bit routers (37.3 without them), before any wait: **25 tok/s in `fast` (40 ms including 3.4 ms of waits) is below
what the machine's read rate allows for the model's bytes** unless the bytes fall. With the exact greedy head and the
4-bit router file the floor is 36.4 ms of compute, 39.8 ms with fast mode's waits: 25 tok/s only if every stage ran at
the read rate and nothing else cost anything. **20 tok/s (50 ms) is in reach**: closing the projections' unexplained
gap (about 6 ms, our code) alone gives about 46 ms of compute, about 49.5 ms in `fast`.

- The model's bytes: experts 1.84 GB, the projections' three passes over layers 6-17, the head 0.49 GB.
- The machine: 105 GB/s; the head and the o projection already run near it.
- Still our code: the projections' in-model loss (the largest, about 6 ms), the NVFP4 kernel's last 10-15% (about 2-3
  ms), the model's router and depth attention (about 2 ms of fork-joins and serial work), the serial hooks of our
  routers (0.8 ms).

**The next biggest stage** is the attention projections: 15.8 ms for 1.02 GB where the same kernel, matrices and call
pattern run in 10.3-11.0 ms alone. Estimate: 4-6 ms per token if the in-model loss is found (first experiment: the
stage with the readers idle on a recorded routing, against the no-model bench, then per-call cycles with and without
the preceding expert stage's traffic).

## Stage 2: the joins, the projections' in-model loss, 10 threads

The question: do the decode's fork-joins wait for their slowest or descheduled thread (8 threads on 4 P + 6 E cores,
beside 4 reader threads and the main thread), and is that what the attention projections lose (about 100 us per call
in the model against the no-model bench)?

**The phases were already dynamically scheduled.** Every row phase (`run_rows`, `run_rows_q`, `run_rows_post`,
`run_rows_n`), the expert step (`phase_gudn`), the attention and the gates claim 32-row (or 64-activation) chunks
from a shared atomic counter; nothing splits rows statically by thread. What did wait for every thread was the join:
`Pool::run` returned only when each of the n - 1 workers had entered and left the phase function, even a worker that
woke after all the work was claimed and found none.

### The join trace (`cmake -DDM_POOLTRACE=ON`, new diagnostic build option)

Per phase function: each thread's entry, exit and clock (cycles / busy time: P-cores about 3.3-4 GHz, E-cores below
3), the join's wait after the median thread's exit and after the caller's, the latest entry after the dispatch, and
how often a worker entered only after every other thread had left. `DMJOIN` lines at exit. Exact decode, 7.8 GB, 100
tokens, Pittsburgh prompt:

Projection and router fork-join (`phase_rows_q`; 96 per token: 48 projection calls and 48 router calls):

| run | join | threads | others' CPU | compute ms / token | mean run (us) | join wait after the median thread (us) | latest entry (us) | runs with a worker entering after all others left |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| a: default (readers on) | old | 8 | 200% | 93.1 | 272 | 83 | 69 | 777 / 10,464 |
| b: readers at background QoS | old | 8 | 204% | 76.9 (+36 ms of read waits) | 239 | 51 | 35 | 399 / 9,504 |
| c: readers idle (recorded routing, `--skip-reads`) | old | 8 | 90% | 52.3 | 179 | 10 | 4 | 149 / 10,464 |
| d: default | `run_any` | 8 | 102% | 57.5 | 180 | 13 | 4 | 10 / 9,504 |
| f: default | old | 10 | 108% | 136.6 | 442 | 243 | 178 | 1,252 / 9,504 |
| e: default | `run_any` | 10 | 95% | 90.6 | 275 | 94 | 17 | 10 / 9,504 |

The expert step (`phase_gudn`, 36 MB per call) shows the same pattern: join wait after the median thread 90 us (a),
12 us (c), 15 us (d), 285 us (f), 121 us (e).

What it says:

- **Readers idle does not remove the projections' loss.** With the drive readers idle (recorded routing,
  `--skip-reads`) the projection fork-join still runs about 300 us per call, and the threads are busy for nearly all of
  it (join waits about 10 us). So the loss is in the threads' own time, not in the join.
- **The join waits are real under load and dominant at 10 threads.** At others' CPU 200% a projection-and-router
  fork-join waited about 83 us after its median thread (30% of the run) and the last worker entered 69 us after the
  dispatch; 777 of 10,464 runs had a worker that entered after every other thread had left (it had nothing to do, the
  join waited for it anyway). Readers at background QoS halve that (51 us) but triple the read waits (36 ms per
  token): not an option. At 10 threads the latest entry is 178 us and the join waits 243 us per projection call.
- **Every thread spends about 40% of its phases on a P-core**: the scheduler moves the 8 workers between the clusters,
  so there is no fixed slow thread to give fewer rows to; dynamic claiming already balances them.

### The change: `Pool::run_any` (shared pool, both models)

`include/dmeng/pool.h`, `src/core/pool.cpp`. Interface:

```c++
void Pool::run(TaskFn fn, void* ctx);      // unchanged: fn on every thread, returns when every thread has left fn
void Pool::run_any(TaskFn fn, void* ctx);  // fn's work is claimed from shared counters (any thread may do any part,
                                           // a thread may do none, a part only waits for parts claimed before it):
                                           // returns once the caller has left fn and every worker that entered has
                                           // left; a worker arriving after that skips the run
```

A run's state word carries its generation tag, its kind, a closed bit and the count of workers inside; a worker enters
an any-run by a tagged compare-and-swap only while it is open, so a late worker never touches the context of a run its
caller has left. Phases that split work by `tid` (the 35B's `phase_conv`, the prefill's `for_phase`, the draft head's
`phase_pick`) keep `run`. Moved to `run_any`: `run_rows`, `run_rows_q`, `run_rows_post`, `run_rows_n` (the
projections, routers and heads of both models, the 35B's linear-attention projections), `Core::run_experts` (the
expert step of both models), `parallel_for`, and Inkling's decode attention, message gates and messages. Who computes
a row changes, not how: every row is the same kernel call on the same inputs, so outputs are bit-identical.

**For the 35B branch**: rebase onto `ink-speed`; do not edit `pool.cpp` / `pool.h`; a new claimed-work phase should
call `run_any`, a phase that uses `tid` must keep `run`.

### Measured (paired per token in one process: odd positions the old join, even ones `run_any`)

Alternate-token build (scratch, not committed): the old join on odd positions, `run_any` on even ones, per-token
compute excluding read waits, paired per adjacent token:

| setting | `run_any` - old join, per token | others' CPU |
|---|---:|---:|
| fast, 8 threads, 400 tokens (204 pairs) | mean -0.67 ms, median -0.12, sd 10.5, 95% interval -2.10 .. +0.77 | 116% |
| fast, 8 threads, 300 tokens (means only) | 55.23 vs 55.40 ms | 85% |
| exact, 8 threads, 300 tokens (means only) | 65.84 vs 66.12 ms | 143% |
| fast, 10 threads, 300 tokens (means only) | 82.77 vs 106.57 ms | 93% |

At 8 threads the gain is not resolvable (under 1 ms, inside the spread); at 10 threads `run_any` removes most of the
penalty (-24 ms per token) but **10 threads stay worse than 8** (83 against 55 ms): with 10 compute threads plus the
readers and the rest of the system on 10 cores, a worker is preempted in the middle of a claimed chunk and the join
waits for that chunk (94 us after the median thread, trace e). Keep 8.

### Step 3: cache and TLB state at the start of a call

Not the cause. No-model bench of the projections' call pattern (48 calls of q, g, k, v, r, chunk 32, 8 threads, others'
CPU 97%): 208-224 us per call back to back, 214-225 us with an expert step's 42 MB of other traffic between calls,
210-227 us when the caller first touches one line of every page of the next call. In the model, a touch-ahead of the
next chunk's pages inside the projections' phase, paired per token: mean -0.11 ms (95% interval -2.21 .. +1.99). A
smaller footprint (1.5 GB cache against 7.8, readers idle) did not shorten the stage either (16.8 against 15.7 ms, at
higher load). The loss stays unexplained: in the quietest run of the night (others' CPU 25%, readers idle) the stage
took 15.7 ms per token, about 327 us per call against 210-230 us for the same kernel, sizes and call pattern alone.
The weights themselves are not it either. A no-model bench on the real resident file's q, g, k, v, r (48 calls in decode
order, anonymous memory as the engine loads it, counting build): 0.27 cycles per weight byte, the head's rate; the same
matrices in the model's projection stage: 0.36 (counting build, default settings). Random against real codes and scales,
alternated pass by pass in one process: no difference beyond the load's spread. Per call in the model (join trace keyed
by stage): the eight threads are each busy for about 76% of the call, so the extra cycles are stalls inside the
threads' own chunks, not waits.

**Next step (counts only):** replay the 48 projection calls inside the model process right after load, on the model's
own pool and resident region (no decode before them), and compare cycles per byte with the same calls inside decode.
0.27 there means the decode state left by the stages before each call is the cause (what ran last: the message gates'
f32 rows, the expert slots, the KV ring); 0.36 there means the model's memory or pool setup is.

### Step 4: the main thread's solo work

Not changed. The main thread works alone for 10.8 M cycles per token (counting build, after): our routers' hooks 2.9,
depth attention with its two norms 2.7, the post-MLP convolution and residual 1.2, norms 1.2, the rest under 0.6 each.
The norms are a sequential double sum over 2304 elements that cannot be split without changing bits (about 9 K
cycles each, 5 per application); the vectorisable rest is 3-10 us per call, about what a fork-join costs. Moving it
onto the pool would gain well under 1 ms; not started under the freeze.

### Gates (stage 2)

| gate | result |
|---|---|
| tiny: `draft_check.sh`, `nv_i16_check`, Inkling 2 / 12 / 16 threads and the 35B 2 / 16 threads against the v1 ids | all passed / identical |
| real, 200 ids against the v1 reference, run_any build: Inkling default, drafting on, 4-bit routers, 10 threads; the 35B default and 10 threads | identical |
| real, Inkling `--fast-head 1`, default prefill | was -1, 0, 0, ... from the first token in v1 and in this branch (the token after a prefill was never set); fixed, identical |
| full gate on the merged tree | the merged tree's gate |

### Where it stands

Counting build, M cycles per token summed over threads, v1 against the committed tree (cycles from runs at different
loads: Inkling 138% / 106% others' CPU, the 35B 95% / 109%); ms = cycles / 26.2 G per second (the rate that maps
Inkling's 1,393 M cycles to its clean 53.2 ms and the 35B's 835 M to its 31-32 ms); "rate" = the stage's bytes at 105 GB/s:

| stage | Inkling v1 | after | rate | 35B v1 | after | rate |
|---|---:|---:|---:|---:|---:|---:|
| experts | 608 (23.2 ms) | 578 (22.0) | 17.5 | 221 (8.4) | 204 (7.8) | 6.7 |
| attention projections | 371 (14.1) | 366 (14.0) | 9.8 | 288 (11.0) | 269 (10.3) | 9.6 |
| attention core + output projection | 145 (5.5) | 127 (4.8) | 3.4 | 147 (5.6) | 134 (5.1) | 3.4 |
| head | 124 (4.7) | 120 (4.6) | 4.7 | 138 (5.3) | 131 (5.0) | 5.0 |
| routers (model's + ours) | 86 (3.3) | 81 (3.1) | 2.4 | 40 (1.5) | 38 (1.5) | 0.4 + ours |
| gates, norms, depth, convs, rest | 59 (2.3) | 55 (2.1) | 1.6 | 2 (0.1) | 2 (0.1) | - |
| **total** | **1,393 (53.2)** | **1,326 (50.6)** | **39.4** | **835 (31.9)** | **778 (29.7)** | **25.1** |

The 35B's numbers moved only through the pool (its expert kernel is AFF4, untouched), so most of its -7% is the load
difference, not the change. Inkling at about 50.6 ms of compute is 4.6 ms from 46 and 10.6 from 40; the largest piece
of our code left is the attention projections' in-model loss (about 4-5 ms, cause not found), then the NVFP4 kernel's
last 10-15% (2-3 ms).

## Stage 3: the projections' lost time found (the activation quantisation's barrier)

The bisection, counts only (counting + join-trace build, 64 tokens, 2 GB, Pittsburgh prompt):

- **Replay in the model process** (`DMENG_PROJ_REPLAY=N`, counting builds: the 48 projection fork-joins N times
  before the first step, on the model's own pool, weights and input buffers): 319 us per call, the same as the same
  calls inside decode (325). So neither the decode state nor what ran before each call is the cause.
- **No-model bench on the real resident weights** (copied into one mlocked region as the engine loads them, the f32
  sink job included): 231-245 us per call, 0.27 cycles per byte. The weights, the memory, the pool and the sink job
  are not it either.
- **The input quantisation**: in the replay, quantising the inputs before the fork-join (`run_rows`) instead of inside
  it (`run_rows_q`, decode's form) took 288 against 319 us per call. Inside `run_rows_q` the 36-108 activation blocks
  are claimed first and every row waits until all of them are done; `xq_block` was scalar code (2,550 instructions,
  360 cycles per 64-element block), so every thread spun at that barrier for the slowest block.

**Changes (f85724d):** `xq_block` (I16) with NEON, 327 instructions and 83 cycles per block, every field equal to the
scalar definition (sum in order, products rounded once, ties to even, the perm order by unzipping), checked on 4,000
vectors with zeros, ties, huge and tiny values (`tests/kernels/nv_i16_check.cpp`); and in `run_rows_q` a job's rows wait
only for their own input's blocks (neutral within the spread). After: replay 233-241 us per call in the decode's form,
209-214 with the inputs quantised before; decode 246-251 us per call.

Counts per token, same settings and the same build otherwise (64 tokens, 2 GB, others' CPU 87% / 95%; the replay's
stages excluded), and the 35B's (one input of 32 blocks per call, a shorter barrier):

| stage | Inkling before | after | rate (ms) | 35B before | after | rate (ms) |
|---|---:|---:|---:|---:|---:|---:|
| attention projections | 377 M (14.4 ms) | 289 M (11.0) | 9.8 | 269 M (10.3) | 276 M (10.5) | 9.6 |
| experts (activation blocks quantised inside the gate-up chunks) | 573 M (21.9) | 529 M (20.2) | 17.5 | 204 M (7.8) | 206 M (7.9) | 6.7 |
| attention core + output projection | 115 M (4.4) | 107 M (4.1) | 3.4 | 134 M (5.1) | 134 M (5.1) | 3.4 |
| head | 116 M (4.4) | 111 M (4.2) | 4.7 | 131 M (5.0) | 122 M (4.7) | 5.0 |
| **total (all stages)** | **1,318 M (50.3 ms)** | **1,162 M (44.3 ms)** | **39.4** | **778 M (29.7)** | **777 M (29.7)** | **25.1** |

(ms = cycles / 26.2 G per second as before. The 35B's before is the run_any build at 200 tokens / 6.6 GB, its after 64
tokens / 2 GB: per-token stage cycles, not the same settings.) Inkling: -11.9% cycles per token, about 44 ms of compute,
under the 46 ms mark; 4.3 ms from 40. Gates (light rule): tiny `draft_check`, the kernel check, tiny logits and ids of
both models against v1; real models, 64 greedy ids at 2 GB against the v1 ids: identical for both.

## Stage 3b: the expert kernels' instructions (single thread, matrix hot in L2)

| kernel | instructions per weight byte | cycles per byte | instructions per 64 weights |
|---|---:|---:|---:|
| AFF4 (35B experts, before the parity branch's row pairing) | 0.770 | 0.133 | 27.7 |
| NVFP4, I16 (Inkling layers 0-19; two rows already share activations and scales since 990ad3e) | 0.890 | 0.185 | 32.0 |
| MXFP8 (Inkling layers 20-23) | 0.789 | 0.204 | 52.1 |
| AFF8 (resident) | 0.361 | 0.151 | 24.5 |

NVFP4 already is "a 16-entry table lookup into int8, then the integer dot product": per 64 weights and row, 8 SDOT
(two activation planes), 4 TBL (one per 16 codes), 2 AND / 2 USHR (nibbles), 2 SHL (the high plane), 1 ADDP, 1 SCVTF,
2 FMUL / FMLA, the codes' load, half of the shared activation loads, about 2.5 for the E4M3 scales and the rest address
arithmetic. Over AFF4 that is the 4 TBL and, because NVFP4 has a scale per 16 weights (four per 64, against AFF4's one
scale and one bias per 64), about 2 more for the scales; the cycles are 39% higher than AFF4's at 16% more
instructions (the TBL and SDOT issue). MXFP8 with two rows sharing the activations: bit-identical but 0.880 instructions
per byte against 0.789 (register pressure), cycles equal: not kept. In decode the experts run at about 0.29 cycles per
byte (after), the kernel's own cost 0.185 hot: the rest is memory stalls and the efficiency cores.

## Stage 4: a flat profile of the token path, both models

`sample` (12 s, every thread) on the counting build during decode, default settings, 6 GB; Inkling at others' CPU 272%,
the 35B at 133% (the waits and joins are inflated by the load; the kernels and the serial pieces are not). Thread-ms
per token summed over all threads, and the main thread's own time (its self samples):

| entry | Inkling, all threads | Inkling, main thread | 35B, all threads | 35B, main thread | kind |
|---|---:|---:|---:|---:|---|
| 8-bit resident kernel (projections, head, shared expert) | 159 | 20.6 | 148 | 18.4 | kernel |
| expert kernels (NVFP4 + MXFP8 / AFF4) | 152 | 19.3 | 46 | 5.9 | kernel |
| waiting for expert reads (`Cache::wait`) | 20 | 20.3 | 9.4 | 9.4 | waits on the drive |
| the caller at the join (`Pool::dispatch`) | 17.5 | 17.5 | 3.6 | 3.6 | waits on the slowest worker (load) |
| expert step's own code (`phase_gudn` self: the spin of a down chunk on its expert's gate-up chunks) | 53.6 | 7.3 | 3.7 | 0.6 | waits inside the phase |
| attention / linear attention phases (self) | 16.9 | 2.3 | 10.4 | 1.4 | arithmetic, on the pool |
| message gates (self) | 8.4 | 1.3 | - | - | arithmetic, on the pool |
| activation softcaps and SiLU (`tanhf`, `expf`) | 7.3 | 0.9 | 1.1 | 0.1 | libm calls, on the pool |
| greedy token choice (`sample` -> `argmax` over the vocabulary) | 0.31 | 0.31 | 0.43 | 0.43 | **scalar and serial** |
| depth attention + its norms, the convolutions (self) | 1.1 | 0.9 | - | - | serial; already vectorised |
| quantisation (`xq_block`) | 1.1 | 0.2 | 0.4 | 0.04 | fixed in f85724d |
| everything else on the main thread (route, combine, cache, hooks, timers) | | under 0.3 each | | under 0.2 each | |

What was plainly scalar or needlessly serial, fixed bit-identically:

- **`argmax` (`src/core/sampler.cpp`)**: the greedy choice was a scalar loop over 200-248 K logits on the main thread,
  1.45 M cycles per call; now the maximum with NEON, then the first index holding it: 117 K cycles, the same index on
  20,000 vectors (ties, +0 / -0, infinities, short and odd lengths). Every greedy token of both models and every
  verified position of the draft path calls it: about 0.35 ms per token on each model.
- **Inkling's chunk head (`forward_chunk`)**: the softcap of every logit (a tanh each, about 6 M cycles per position)
  ran on the main thread for each verified position (drafting) and for the prompt's last position; now on the pool
  in blocks of 4,096 logits (each element the same call).

Checked and left: depth attention's weighted sum and the short convolutions are already vectorised by the compiler
(NEON rewrites gave the same cycles); the norms are sequential double sums by definition; the activations' tanh / exp
are libm calls whose exact values a vector form would not reproduce. **Not mine to change, the largest waits**: the
expert step's down chunks spin on their expert's gate-up chunks in claim order (`phase_gudn` in `src/core/moe.cpp`,
53.6 thread-ms per token for Inkling under this load, 3.7 for the 35B; claiming down chunks from experts whose gate-up
is complete would cut it), and our routers' hooks on the main thread (0.6-0.8 ms per token for Inkling).

## Stage 5: the prefill expert kernels (branch `ink-prefill-kernels`, from `v2` 4369522)

Single thread, weights hot in L2 (a 1,152 x 2,304 slice), cycles counted (`thread_selfcounts`), 16 and 64 positions
(the same to 1%):

| kernel | cycles per (weight byte x position) | cycles per MAC | instructions per MAC |
|---|---:|---:|---:|
| AFF4 packed gemm (the 35B's experts, SMMLA) | 0.0864 | 0.049 | 0.178 |
| NVFP4 packed gemm (Inkling layers 0-19, SMMLA), v2 | 0.1226 | 0.069 | 0.374 |
| NVFP4 packed gemm, this branch | 0.1204-0.1209 | 0.068 | 0.344 |
| MXFP8 multi-vector (Inkling layers 20-23, fp16 FMLAL) | 0.0907 | 0.094 | 0.582 |

In the model (counting build, a 2,048-id prompt, chunks of 512, others' CPU 193%): Inkling's prefill experts 305 M
cycles per position (gate-up 207, down 98) for 3.06 G MACs, 0.100 cycles per MAC.

Where NVFP4's 1.4x over AFF4 goes, per 2 rows x 2 positions x 64 inputs (256 MACs, 16 SMMLA): AFF4 has one scale and one
bias per 64 inputs, so its float epilogue (convert, scale product, two FMAs, one 4x4 transpose) runs once per 256
inputs and (row, position); NVFP4 has an E4M3 scale per 16 inputs, so the same epilogue (the 4x4 transpose of the four
16-input integer sums, convert, scale x activation scale, multiply, add, in the decode kernel's order) runs once per 64
inputs: four times as often, about +0.07 instructions per MAC. Plus the table lookup that turns E2M1 codes into int8
(4 TBL per 64 inputs per row pair, shared by the tile's 8 positions). Both are the format's, and the float order is
fixed by the decode kernel the prefill must equal. Change made: the low plane's SMMLAs accumulate into the high plane's
sums shifted by 8 (the same integers): -8% instructions, -2% cycles (bit-identical: `expert_users` compares every
batched output with the one-vector kernel at 1-512 positions). Tiles of 2 or 3 position pairs instead of 4 were slower
(more table lookups per MAC).

MXFP8: its exact definition is an fp16 FMLAL chain per (row, position) (products of two fp16 values, exact in fp32, one
rounding per step); 8 FMLAL per 32 inputs and position are the floor, about 0.25 of its 0.58 instructions per MAC. The
code conversion is already shared by up to 8 positions; the remaining per-position work is the activation loads (4 per
32 inputs) and the pair sums. An fp32 FMA on pre-converted operands is the same arithmetic at the same 4 MACs per
instruction, so no packed form is faster; not changed. It is 8% of Inkling's expert MACs.
