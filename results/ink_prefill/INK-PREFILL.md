# Inkling span prefill: each application's experts read once per span

Branch `ink-prefill` (from `rc` 0adc611). Inkling's cold first token under stock OpenCode was 359 s for an 8,362-token
first request (results/serving/inkling/live1, 23 tok/s, chunk 256), against 85-97 s for the 35B. The 35B's prefill is
layer-major over a whole span (8,192 positions), so a layer's experts are read from the store once per span; Inkling's
ran chunk by chunk (256 positions, every one of its 48 applications per chunk), so every application's experts were read
again for every chunk. This branch gives Inkling the 35B's schedule.

## What changed

| piece | where |
|---|---|
| `Core::moe_chunk(..., off)`: a chunk that is positions `[off, off + n)` of a longer span (cache clocks and logged picks at `tok_clock + off + i`); `span_mode`: no loop holds while a span runs application by application (a looped layer's next pass comes after all the other looped layers of the span, so a hold would only pin the pool); in span mode a step's union also stays pinned from one sub-chunk to the next until the application ends (without it the next application's names evicted it and a small MXFP8 pool ran out of slots: the first real run aborted in layer 21); `span_name(step)` / `span_unname(step)`: every expert of the next application's layer named (read behind the running application's own misses, pinned until that application has run; a union with no slot free releases them first) | `include/dmeng/core.h`, `src/core/moe_chunk.cpp` |
| Inkling `forward_chunk(ids, n, ..., sub, run)`: application-major over the span, each application over all positions in sub-chunks of `sub` before the next one | `src/models/inkling2_loop/inkling2_loop.cpp` |
| per span position: the residual, its value at the loop's start, the block states, the message sources (the previous pass's blocks after its first, then its bank): 8 x H floats = 72 KB a position (590 MB at 8,192) | same |
| messages: each looped layer's 4 message rows computed when that layer runs, from the stored sources (the single step computes the same rows from the same sources; `depth_attn` computes each row on its own), so the 12 x 4 x H messages per position are never stored | same |
| capture positions (prefix-cache checkpoints): sub-chunks end there, and each application delivers its own slice of the fixed state (`save_fixed`'s layout: four conv states, the sliding ring with not-yet-reached slots as zeros) as it passes the position, like the 35B's per-layer sink | `Inkling::sink_app` |
| cancel inside a span: checked after every application; the fixed state goes back to the span's start (snapshot), the position does not move, the next application's names are released | `forward_chunk` |
| `prefill()`: spans of `prefill_span` (default 8,192; `dmserve --span`), sub-chunks of `prefill_chunk` (now up to 512), no short last span (a rest under a quarter span joins the one before: an 8,365-token request is one span, not 8,192 + 173 that would read every application's experts again for 173 positions), one position = the single step; `prefill_stream 0` turns the early names off; the prefill JSON reports `mb_read`, `expert_mb_touched`, `resident_mb_touched` and the same per prompt position | same |
| a span of one sub-chunk (`span <= chunk`, every verify) runs as before: loop holds on, no names | same |

The 35B's prefill (`src/models/qwen3_5_moe/prefill.cpp`) is not touched; it keeps its own layer-major MoE (its AFF4 gemm on
gathered inputs, waves, stream names). The shared piece both now use for the span schedule is the core: Inkling's sub-
chunk MoE is `Core::moe_chunk` (union pinned and read once, every expert applied to all its users of the sub-chunk with
the multi-position kernels, the single step's combination order), with the offset, span mode and names above.

## The expert kernels are not the limit (short kernel bench, no model)

`tests/kernels/expert_users.cpp` (target `expert_users`): Inkling's expert shapes (gate-up 2304 x 2304 fused, down 2304 x
1152), random records, 40 NVFP4 / 24 MXFP8 experts (larger than the caches), 10 threads, median of 5; every output
compared bit for bit with the one-vector kernel (all equal). AC power, others' CPU 15.6% (log:
`engine/unify/work/ip/kb/expert_users.log`).

| users per expert | NVFP4 packed `gemm_nvfp4_i16` GMAC/s | NVFP4 multi-vector `mv_nvfp4_multi` | MXFP8 `mv_mxfp8_multi` |
|---:|---:|---:|---:|
| 1 | 115 | 160 | 98 |
| 4 | 312 | 303 | 233 |
| 8 | 351 | 326 | 245 |
| 16 | 347 | 316 | 240 |
| 64 | 337 | 310 | 236 |
| 256 | 333 | 294 | 237 |
| 512 | 329 | 291 | 231 |

From 8 users on, both formats run at a flat rate: the kernels are bound by their own arithmetic (NVFP4: the two int8
planes of the I16 activations through SMMLA; MXFP8: the fp16 FMLAL chain that the exact reference `mv_mxfp8` defines), not
by weight traffic. A 256-position sub-chunk already gives every expert about 16 users, so neither a wider MoE call nor a
new multi-position kernel would move this; no kernel was added. At these rates the expert arithmetic of one prompt token
(44 NVFP4 + 4 MXFP8 applications x 8 picks) takes about 8.0 + 1.0 ms on all cores, about 110 prompt tokens/s for the
experts alone; the attention and the rest come on top. The MXFP8 layers are 8% of the expert work at 70% of the NVFP4
rate: not the limit either.

## Gates

All with the final code (5add4bc) unless noted; every model run through `engine/bench/guard.py` (ticket queue), AC power.
Logs in `logs/` (paths shortened to DM/).

| gate | result |
|---|---|
| Inkling, recorded OpenCode request: `engine/results/agentic/replay/s2_clean/002.json` (OpenCode's own system prompt and 10 tools, its first main request) tokenised by Inkling's front end (`oc_ids.py` -> `oc_002.ids`, 8,365 ids; the live run's request was 8,362), 6 GB, `dmserve --check --ref-chunk 1` (token by token) vs chunk 256, span 8,192 (one span, 33 sub-chunks; stage 1 code, see stage 2 below for the final code) | 0 logit bits, 0 fixed-state bytes, 0 KV bytes differ; greedy 16 identical (`inkoc.log`) |
| Inkling, `tests/prefill_check.sh` (1,100 pseudo-random ids, 5 GB, chunk 256, span 8,192: one span of 5 sub-chunks) | PREFILL CHECK PASSED: 0 / 0 / 0, greedy identical (`ink1100.log`; built before the last-span rule, which does not touch a 1,100-id span) |
| 35B, `tests/prefill_check.sh` (1,100 ids; its prefill code is unchanged) | PREFILL CHECK PASSED: 0 / 0 / 0, greedy identical (`q36.log`) |
| tiny Inkling, `dmserve --check` (1,100 ids, window 16 so the rings wrap often), both resident builds x (chunk, span) = (256, 8192), (64, 8192), (5, 8192: odd sub-chunks across the 6-position packed-kernel switch), (64, 300: four spans), (256, 100) | 10 / 10 PASS, 0 / 0 / 0, greedy identical (`tiny_batch.log`) |
| `tests/draft_check.sh` (both tiny models: verify-check with rollbacks in every precision, drafting ids in dmeng and the server worker) | DRAFT CHECK PASSED (the verify path now computes its messages per layer) |
| `dmserve --check-ahead` (idle-time prefill cancelled inside a span, resumed, the request synced from a checkpoint), tiny Inkling and tiny 35B | PASSED, both |
| the same on the tiny Inkling with checkpoints every 48 positions inside multi-sub-chunk spans (chunk 16 / span 256 and chunk 64 / span 512): the per-application sinks build the checkpoints, the request path resumes from one (480 positions reused) and the state equals token by token; the cancelled run stops inside its first span with the state back at its start | PASSED, both shapes (`logs/tiny2.log`) |


## Bytes per prompt token (do not depend on machine load)

MB here = 2^20 bytes (the JSON's unit). The recorded OpenCode request (8,365 ids), one cold `dmserve --prefill-bench` per row, 6 GB cache (mlock). "Read" =
expert records read from the store; "touched" = bytes of expert records and resident matrices the kernels went through
(each once per sub-chunk). The first row is today's schedule (span = chunk = 256: every chunk runs all 48 applications;
loop holds on), i.e. the rc path (run with a6b7244; the two later commits do not touch a span of one sub-chunk).

| schedule | read MB / prompt token | expert MB touched / token | resident MB touched / token | read total | waiting on reads | prefill tok/s (indicative) |
|---|---:|---:|---:|---:|---:|---:|
| before: chunks of 256 | 61.8 | 90.7 | 5.80 | 505 GiB | 139.1 s of 365.4 s | 22.9 (others' CPU 20%) |
| span 8,192, sub-chunk 256 | **2.98** | 90.7 | 5.80 | 24.3 GB | 0.7 s of 237.9 s | 35.2 (others 19%) |
| span 8,192, sub-chunk 512 | 2.97 | 50.4 | 3.01 | 24.3 GiB | 0.8 s of 254.5 s | 32.9 (others 83%: not comparable) |

- Reads fall 20.7x (61.8 -> 2.98 MB per prompt token): every application's union, about all 128 experts, is read once
  for the span (44 x 128 x 4.49 MB + 4 x 128 x 8.2 MB = 27.5 GiB if nothing were held; 24.3 GiB measured, the rest hit
  the 6 GB cache). The waits on the store go from 139 s to under 1 s.
- Weight bytes touched do not depend on the span: they are once per sub-chunk. Sub-chunks of 512 halve them (90.7 -> 50.4
  MB expert, 5.8 -> 3.0 MB resident per token), but the kernel bench says the kernels run at the same rate from 8 users
  on, so this is not what limits the prefill; the 512 row ran under 83% others' CPU and its time says nothing.
- The prefill is now compute-bound: 237 s for 8,365 positions is 28 ms per prompt token, of which the expert arithmetic
  is about 9 ms at the bench's rates. The next step for TTFT is the non-expert part (attention projections and core over
  an 8K context, depth attention, message gates), measured stage by stage in the clean campaign.
- Extra memory: the span arrays, 72 KB per position (590 MB for 8,192 positions); peak footprint 8.43 GB vs 7.69 GB
  before at 6 GB cache.
- Indicative only (shared machine): 22.9 -> 35.2 tok/s on this request, a cold 8,365-token prefill from 365 s to 238 s.


## Stage 2: the non-expert work (per-stage counts; branch commits d3fff2b, e2e7640, 151123c)

After stage 1 the prefill waited under a second on the store and ran at 28 ms per prompt token, experts about 9 ms. Where
the rest went, from the counting build (`cmake -DDM_COUNTERS=ON`; the main thread marks the stage that runs,
`stage_mark`, and the build sums per stage the main thread's serial work and all threads' pool work; `DMSTAGE` lines at
exit, `logs/stages.py` prints them per prompt token). Instructions and cycles are per thread and summed over the threads,
so they do not depend on the load; "serial" is work only the main thread does while the pool waits. The recorded
OpenCode request, 8,365 positions, 6 GB, span 8,192, sub-chunk 256 (`logs/counts_before.log`, `logs/counts_after.log`):

| stage | M instructions / token, before | after | M cycles / token, before | after | of which serial (main thread) cycles, before | after |
|---|---:|---:|---:|---:|---:|---:|
| attention projections (q, g, k, v, r, sink, dynamic conv) | 170.4 | 170.4 | 58.7 | 59.5 | 0.01 | 0.00 |
| attention core (scores, softmax, values over the context) | 913.6 | 420.7 | 168.5 | 92.7 | 0.02 | 0.01 |
| output projection | 60.2 | 60.3 | 21.5 | 21.8 | 0.24 | 0.00 |
| depth attention (incl. the messages) | 14.4 | 14.3 | 2.9 | 4.0 | 2.94 | 0.00 |
| message gates | 86.7 | 86.7 | 21.6 | 21.9 | 1.29 | 0.12 |
| norms + activation quantisation | 11.7 | 11.7 | 2.9 | 3.6 | 2.86 | 0.01 |
| KV prep (clips, norms, cache writes) | 7.2 | 7.5 | 1.6 | 1.7 | 1.58 | 0.09 |
| short convolutions + residual | 14.0 | 13.7 | 1.9 | 2.6 | 1.86 | 0.02 |
| embedding, head, span copies | 0.1 | 0.1 | 0.1 | 0.1 | 0.04 | 0.01 |
| MoE router | 18.3 | 18.3 | 4.5 | 4.7 | 0.88 | 0.00 |
| MoE union / cache bookkeeping | 7.4 | 6.6 | 1.8 | 1.6 | 1.82 | 1.64 |
| MoE combination | 6.3 | 6.3 | 1.3 | 1.3 | 1.27 | 0.00 |
| MoE experts | 1356.3 | 1356.3 | 307.8 | 307.9 | 2.41 | 0.03 |
| **all but the experts** | 1310.2 | 816.5 | 287.1 | 215.5 | 14.81 | 1.91 |
| **total** | 2666.5 | 2172.8 | 594.9 | 523.4 | 17.22 | 1.94 |

What the 'before' column said, and what changed:

- **Serial per-position loops: 17.2 M main-thread cycles per prompt token (about 4 ms of every 28, the pool idle).** The
  span prefill ran depth attention, norms and quantisation, the convolutions, the clips, the routing, the combination and
  the expert-input packing position by position on the main thread. The 35B's prefill runs all of these on the pool
  (`pfor`). Now Inkling's do too: `parallel_for` (moved into `include/dmeng/pool.h`) over positions for every loop whose
  iterations are independent, and the convolutions, whose state moves position by position, as channel blocks in
  parallel with the positions in order inside each block (`sconv_rng` / `dconv_rng`, the single step's functions). The
  KV preparation runs for the whole sub-chunk first, then the cache writes and attention go in groups the sliding ring
  can hold (a global layer, one group). Serial: 17.2 -> 1.9 M cycles. The rest is the cache bookkeeping, which has to
  run in order. A per-token statistic (looped-layer reuse) that scanned all of a span's picks once per pick is now off
  in span mode.
- **Attention core: 914 M instructions per token, 518 per (key, head).** One job per (position, head) loaded every key
  and value row for each of the 3 query heads that share it, and accumulated the values through memory (a load and a
  store of the sums per key). The 35B's decode rewrite uses the same scheme. Now a job is one (position, KV head) for its
  3 query heads: each key row is loaded and converted once for the three score dots, and the value sums stay in
  registers (32 dimensions x 3 heads per pass over the keys). Each head's arithmetic stays the single step's: the bias
  in j order, two FMA accumulators over the dimensions then their sum, max / exp / double sum in key order, each output
  element one FMA chain over the keys from 0. Result: 914 -> 421 M instructions, 168 -> 93 M cycles.
- **Exactness catch.** The first version of that kernel differed in the last bit. The single step's score
  `dot * scale + bias` compiles as a multiply then an add, and in the new loop the compiler fused it under
  `-ffp-contract=fast`. A `#pragma clang fp contract(off)` does not stop that: with `fast`, the backend fuses any
  multiply-add. An empty `asm` on the product does (`attn_score`). Found by running both kernels on the same inputs (the
  differences were 1 ulp from position 0), then confirmed by the gates below.
- **The resident projections** already go through the packed i8mm gemm at sub-chunk width (positions >= 6). They come to
  59.5 + 21.8 M cycles, 16% of the total. The 35B's CPU+GPU split of the 8-bit products (`--gpu-frac`) is the next
  candidate if that share matters; not done here.
- **Unchanged and now the largest: the experts, 308 of 523 M cycles (59%).** Inkling's expert arithmetic per token is
  inherent to the model: 48 applications x 8 picks x 8 M weights = 3.1 G MACs. The 35B has 40 x 8 x 3.1 M + shared = 1.1
  G. On top of that, Inkling has 48 attention applications against the 35B's 10 full-attention layers plus 30 linear
  ones. That is most of the gap to the 35B's 11 ms per prompt token.

Gates on the final code (151123c), all through the guard:

| gate | result |
|---|---|
| Inkling, recorded OpenCode request (8,365 ids, 6 GB, token by token vs span 8,192 / sub-chunk 256) | 0 logit bits, 0 fixed-state bytes, 0 KV bytes differ; greedy identical (`logs/inkoc_stage2.log`) |
| Inkling, `tests/prefill_check.sh` (1,100 ids) | PASSED, 0 / 0 / 0 (`logs/ink1100_stage2.log`) |
| 35B, `tests/prefill_check.sh` (its prefill unchanged) | PASSED, 0 / 0 / 0 (`logs/q36_stage2.log`) |
| tiny: 10 prefill shapes x 2 resident builds, `draft_check.sh`, check-ahead both tiny models, checkpoints + cancel inside spans | all PASSED (`logs/tiny_batch_stage2.log`, `logs/tiny2_stage2.log`) |

Indicative only (shared machine, others' CPU 17%): the 8,365-token cold prefill went from 238 s (35.2 tok/s) to 186 s
(45.0 tok/s, 22.2 ms per prompt token; `logs/bytes_span_stage2.log`). Against today's rc schedule (365 s, 22.9 tok/s)
it is 2.0x. Reads and bytes touched are unchanged from stage 1.

Counting commands (cold, through the guard; load-independent):

```zsh
cmake -S . -B build-cnt -DCMAKE_BUILD_TYPE=Release -DDM_COUNTERS=ON && cmake --build build-cnt -j4 --target dmserve
DM_COUNTS=1 build-cnt/dmserve --model $M --cache-gb 6 --mem mlock --ctx 9000 --chunk 256 --span 8192 --prefill-bench $IDS --rep 1 --gen 0 2> counts.log
python3 results/ink_prefill/logs/stages.py counts.log
```

## Timing command lines for the clean campaign

Each a cold process, one per run, under the guard, inside a QUIET window (write `engine/bench/QUIET`, wait 10 s, run,
delete), power rule checked before each; `IDS` = `results/ink_prefill/oc_002.ids`, `M` = the Inkling model folder, `B` =
this branch's `build/`:

```zsh
# before (today's schedule) / after (span) / after without early names / after with sub-chunks of 512
$B/dmserve --model $M --cache-gb 6 --mem mlock --ctx 9000 --chunk 256 --span 256  --prefill-bench $IDS --rep 1 --gen 4
$B/dmserve --model $M --cache-gb 6 --mem mlock --ctx 9000 --chunk 256 --span 8192 --prefill-bench $IDS --rep 1 --gen 4
$B/dmserve --model $M --cache-gb 6 --mem mlock --ctx 9000 --chunk 256 --span 8192 --stream 0 --prefill-bench $IDS --rep 1 --gen 4
$B/dmserve --model $M --cache-gb 6 --mem mlock --ctx 9000 --chunk 512 --span 8192 --prefill-bench $IDS --rep 1 --gen 4
# the expert kernels by users per expert (no model; reps, threads)
$B/expert_users 5 10
# the live check: stock OpenCode's cold first token, as results/serving/inkling/live1 (worker args after --)
env LIVE_REPO_BASE=SCRATCH bench/serving/live_opencode.sh ../../../inkling/model inkling-2-mini bench/serving/opencode_inkling.json \
  results/serving/inkling/live_span --cache-gb 6 --mem mlock --chunk 256 --span 8192 --disk ../../../inkling/model/.dmeng/prefix --disk-every 1024
```

Read from each: `ms`, `tok_s`, `mb_read`, `prefill.read_mb_per_pos`, `prefill.expert_touched_mb_per_pos`,
`prefill.wait_ms`, `footprint_peak_mb`; for the live run the request's `ttft_s` and `worker_ready.ms_prefill`. The
server's `--chunk 512` was clamped to 256 before this branch; it now means sub-chunks of 512.

