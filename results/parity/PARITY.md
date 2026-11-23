# Parity: both models' decode and prefill, stage by stage

Made by `bench/parity.py results/parity/logs p1 --base p0` from the counting runs of `results/parity/count.sh p0 BIN` (p0) and `results/parity/run_p1.sh` (p1: the 35B's decode after the row-pair 4-bit kernel) (branch
`parity` = `q36-speed` on `ink-speed` 9bbeabb, with stage markers added to the 35B's batched prefill; counting build,
default settings: exact mode, our routers, the screened head; 35B 6.6 GB, Inkling 7.8 GB; decode = 128 tokens after the
Pittsburgh prompt, prefill = 1,024 ids). Regenerate after any change: `zsh results/parity/count.sh TAG BIN` then
`python3 bench/parity.py results/parity/logs TAG --md results/parity/PARITY.md`.

How to read it. The machine was loaded during these runs (others' CPU 96-404%), so the ms and GB/s columns are slower
than a quiet machine; the cycles and instructions per weight byte are the comparable columns (pool work of all threads
in the stage's runs plus the main thread's serial part; in-phase spin-waits included). Weight MB per token are fixed by
model and settings (sources in the script). Flags: CYC = same weight format, one model > 10% more cycles per weight byte;
PATH = model-specific code where the other model runs a shared function; READ = a weight-bound stage below 75% of the
machine's read rate (105 GB/s).

## Decode, per token (counting build; others' CPU: 35B 98.0%, Inkling 123.1%)

- 35B: compute 32.9 ms/token, wait 11.4, 22.6 tok/s (loaded machine: compare cycles, not ms)
- Inkling: compute 60.3 ms/token, wait 14.1, 13.4 tok/s (loaded machine: compare cycles, not ms)

| stage | model | ms | weight MB | format | GB/s | cyc/B | ins/B | M cyc | main serial M cyc | code path | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| embed | 35B | 0.11 | - |  | - | - | - | 0.1 | 0.09 | shared: weights.cpp EmbedRows (pread of one bf16 row) |  |
| embed | Inkling | 0.34 | - |  | - | - | - | 0.1 | 0.11 | model: inkling2_loop.cpp embedding + its norm | PATH |
| routers: the model's + ours (lag rows, token start, naming) | 35B | 2.26 | 108 | bf16 + ours 8-bit | 48 | 0.373 | 0.583 | 40.2 | 1.71 | shared: moe.cpp moe_step run_rows_q (bf16 dot_bf16) + address.cpp Routers (lag rows in the router's fork-join; token start: phase_pre_x / phase_pre_rows; naming serial) | READ (45%) |
| routers: the model's + ours (lag rows, token start, naming) | Inkling | 4.86 | 254 | bf16 + ours 8-bit | 52 | 0.335 | 0.494 | 85.1 | 3.36 | shared: same | READ (50%) |
| depth attention | 35B | 0.00 | - |  | - | - | - | 0.0 | 0.00 | - |  |
| depth attention | Inkling | 0.95 | - |  | - | - | - | 4.1 | 2.71 | model: inkling2_loop.cpp depth_attn (main thread) |  |
| message gates + norms / quantisation / convs / KV prep | 35B | 0.00 | - |  | - | - | - | 1.3 | 0.65 | model: qwen3_5_moe.cpp step rmsnorm (weights.h), serial |  |
| message gates + norms / quantisation / convs / KV prep | Inkling | 2.84 | 170 | bf16 + f32 (gates) | 60 | 0.301 | 0.846 | 51.2 | 3.17 | model: inkling2_loop.cpp gates (pool) + rmsnorm, convs, KV prep (serial) | READ (57%) |
| attention projections | 35B | 11.45 | 1007 | AFF8 | 88 | 0.283 | 0.394 | 284.6 | 0.18 | shared: src/core/weights.cpp run_rows_q + kernels.cpp mv_aff8 |  |
| attention projections | Inkling | 16.87 | 1024 | AFF8 | 61 | 0.381 | 0.398 | 389.9 | 0.20 | shared: same | CYC (+35% vs 35B) READ (58%) |
| attention core (+ fused output projection) | 35B | 5.18 | 393 | AFF8 + fp32 state | 76 | 0.300 | 0.659 | 117.7 | 0.30 | model: qwen3_5_moe.cpp phase_lin_conv_heads (DeltaNet) / phase_full_heads, split_job | READ (72%) |
| attention core (+ fused output projection) | Inkling | 5.80 | 361 | AFF8 | 62 | 0.337 | 0.583 | 121.5 | 0.15 | model: inkling2_loop.cpp phase_attn / phase_attn_kv (grouped) | READ (59%) |
| attention output projection | 35B | 0.99 | 89 | AFF8 | 89 | 0.295 | 0.443 | 26.3 | 0.04 | shared: weights.cpp run_rows (full layers) |  |
| attention output projection | Inkling | 0.00 | - |  | - | - | - | 3.1 | 0.05 | fused into the core (inkling2_loop.cpp) |  |
| expert slots / cache | 35B | 0.23 | - |  | - | - | - | 0.6 | 0.59 | shared: moe.cpp lossless + store.cpp |  |
| expert slots / cache | Inkling | 0.26 | - |  | - | - | - | 0.6 | 0.57 | shared: same |  |
| experts (+ combine) | 35B | 8.88 | 700 | AFF4 routed + AFF8 shared | 79 | 0.333 | 0.792 | 233.3 | 0.80 | shared: moe.cpp phase_gudn + kernels.cpp mv_aff4 |  |
| experts (+ combine) | Inkling | 24.55 | 1839 | NVFP4 + MXFP8 | 75 | 0.328 | 1.071 | 602.9 | 0.92 | shared: moe.cpp phase_gudn + kernels.cpp mv_nvfp4 / mv_mxfp8 | READ (71%) |
| output head | 35B | 3.58 | 540 | AFF8 (full pass), screened: reads ~53% | 151 | 0.169 | 0.509 | 91.1 | 0.05 | shared: screen_head.cpp ScreenHead (default on) |  |
| output head | Inkling | 3.44 | 492 | AFF8 (full pass), screened: reads ~53% | 143 | 0.171 | 0.506 | 84.1 | 0.11 | shared: screen_head.cpp ScreenHead with softcap (default on) |  |
| other | 35B | 0.00 | - |  | - | - | - | 0.0 | 0.00 | - |  |
| other | Inkling | 0.39 | - |  | - | - | - | 0.0 | 0.00 | - |  |

## Prefill, per position (1,024 ids, batched; counting build)

Cycles per weight byte per position (cyc/B) for the stages whose weights are known: the compute cost of one position's
use of each weight byte (prefill reads a weight once per chunk; what limits it is arithmetic).

| stage | 35B M cyc / pos | 35B M ins / pos | 35B cyc/B | 35B main serial M cyc / pos | Inkling M cyc / pos | Inkling M ins / pos | Inkling cyc/B | Inkling main serial M cyc / pos | flags |
|---|---|---|---|---|---|---|---|---|---|
| embed | 0.08 | 0.07 | - | 0.001 | 0.06 | 0.08 | - | 0.061 |  |
| depth_attn | 0.00 | 0.00 | - | 0.000 | 9.79 | 17.57 | - | 0.005 |  |
| msg_gates | 0.00 | 0.00 | - | 0.000 | 25.81 | 88.84 | - | 0.768 |  |
| norms_quant | 0.00 | 0.00 | - | 0.000 | 10.28 | 14.83 | - | 0.043 |  |
| convs | 4.73 | 25.72 | - | 0.000 | 3.86 | 14.30 | - | 0.038 |  |
| kv_prep | 0.00 | 0.00 | - | 0.000 | 2.09 | 7.63 | - | 0.188 |  |
| attn_proj | 63.40 | 177.26 | 0.063 | 0.002 | 70.72 | 172.79 | 0.069 | 0.002 |  |
| attn_core | 35.72 | 138.56 | 0.091 | 0.006 | 41.20 | 158.02 | 0.114 | 0.052 |  |
| attn_out | 22.92 | 58.83 | - | 0.001 | 26.43 | 61.27 | - | 0.007 |  |
| moe_router | 6.97 | 27.47 | - | 0.002 | 6.33 | 18.90 | - | 0.006 |  |
| moe_cache | 0.16 | 0.38 | - | 0.158 | 0.07 | 0.15 | - | 0.075 |  |
| moe_experts | 96.63 | 267.39 | 0.138 | 0.219 | 351.79 | 1315.73 | 0.191 | 0.099 | formats differ (AFF4 vs NVFP4/MXFP8); Inkling / 35B = 1.39x per byte; paths differ: 35B prefill.cpp (its own MoE, gemm_aff4_i16), Inkling moe_chunk.cpp (shared) |
| moe_combine | 1.51 | 5.34 | - | 0.000 | 2.41 | 6.95 | - | 0.003 |  |
| head | 0.15 | 0.24 | - | 0.000 | 0.14 | 0.30 | - | 0.004 |  |
| prefetch | 0.02 | 0.08 | - | 0.021 | 0.00 | 0.00 | - | 0.000 |  |
| span_copies | 0.00 | 0.00 | - | 0.000 | 0.27 | 0.14 | - | 0.001 |  |

- 35B prefill: 12.47 ms per position (others' CPU 179.8%; waits on expert reads 0.02 ms per position)
- Inkling prefill: 58.48 ms per position (others' CPU 404.5%; waits on expert reads 0.16 ms per position)

Runs: 35B decode = p1, Inkling decode = p0, 35B prefill = p0, Inkling prefill = p0


## What the table says (ranked by milliseconds to gain; "clean" = quiet-machine figures from the branch reports)

| # | stage | model(s) | gap | where the time goes | owner / path |
|---|---|---|---|---|---|
| 1 | experts, decode | Inkling (and 35B) | Inkling 1,839 MB at ~75 GB/s here (~89 clean): 3-7 ms over the read rate; 35B 700 MB: ~1.2 ms clean | per byte the cycles match (0.33 vs 0.38), instructions per byte are 1.07 / 0.99: the 4-bit kernels issue twice the 8-bit ones', which the efficiency cores cannot sustain | shared `moe.cpp` phase_gudn + `kernels.cpp`; tails and run_any already in |
| 2 | experts, prefill | Inkling | 1.39x the 35B's cycles and 1.9x its instructions per weight byte per position | different paths: the 35B's batched prefill has its own MoE (`prefill.cpp`, gemm_aff4_i16 / i8mm), Inkling the shared `moe_chunk.cpp` (packed NVFP4 gemm from 6 users, multi-vector below) | move the faster structure into the shared path |
| 3 | attention projections, decode | Inkling | +23% cycles per byte vs the 35B at the same format and the same shared code | stalls inside the model (measured on Inkling) | Inkling |
| 4 | routers (model's + ours) | both | serial main-thread 2.05 / 3.36 M cycles per token (35B / Inkling), ~0.5 / 0.8 ms; Inkling's file is 3.4x the 35B's bytes | naming (z, sort, name calls) on the main thread after the router's fork-join; the token-start rows on the pool | shared `address.cpp` Routers: C5 (move off the critical path) |
| 5 | message gates + norms / convs / KV prep | Inkling | 2.8 ms here for 170 MB of gates (1.6 ms at the read rate) + 3.2 M cycles of serial main-thread work | serial rmsnorms, convs and KV prep between the fork-joins | Inkling model code; the 35B has no counterpart |
| 6 | depth attention | Inkling | 0.95 ms, 2.7 M cycles serial | `depth_attn` on the main thread | Inkling-only stage |
| 7 | attention core | 35B | ~0.7 ms clean over its bytes (268 MB o + 126 MB state) | per-head serial chains, model-specific DeltaNet code | `qwen3_5_moe.cpp` |
| - | head | both | at parity (0.18 / 0.17 cycles per byte, screened) | | shared `screen_head.cpp` |

## Changes and measurements since p0

**The 35B's 4-bit expert kernel (`mv_aff4`, I16) in row pairs** (commit ef73ec1, core `src/core/kernels.cpp`). Where its
instructions went, per 32 code bytes (64 weights) of one row: 2 code loads, 4 nibble unpacks (and / shift), 8
activation-plane loads (4 high-byte, 4 low-byte), 8 SDOTs, 2 to combine the planes, and ~3 for the per-group affine
float part (scale x activation scale, bias x group sum) shared over 4 groups; no tail work at the 35B's shapes. The
activation loads are the same for every row, so two rows now share them: 8 of 24 per row become 4. The group totals are
the same exact integers and each row's float operations the same, in the same order (108 shapes checked bit for bit
against the single-row loop; the 35B's 64 greedy ids at 2 GB identical to v1's).

| 35B experts (decode, per token) | p0 (others 96%) | p1 (others 98%) |
|---|---|---|
| instructions per weight byte | 0.987 | **0.791** (-20%) |
| cycles per weight byte | 0.377 | **0.333** (-12%) |
| stage ms / GB/s | 11.31 / 62 | 8.88 / 79 |

Inkling's experts run the NVFP4 / MXFP8 kernels (Inkling's own), unchanged here.

**Which cores take expert work** (counting runs, 4 threads = the performance cores alone; loads differ, so the cycles
are the comparable column): 35B 8 threads 0.333 cycles per byte, 8.88 ms (others 98%) / 4 threads 0.242, 10.7 ms
(others 180%); Inkling 8 threads 0.328, 24.6 ms (others 123%) / 4 threads 0.249, 29.7 ms (others 163%). The performance
cores alone spend fewer cycles per byte but the stage takes longer: the efficiency cores add throughput (the kernel alone
reads 68 GB/s on 4 threads and 89 on 8 or 10, below); the rows are claimed by pull, so a slower core already takes fewer
of them. Keep 8 threads.

**SME for decode** (`tests/kernels/bench_kmulti_sme4.cpp`, no model: 384 MB of a store layer file as 2048-input 4-bit
rows, I16 activations, one position; SME rows checked bit for bit against mv_aff4; others' CPU 109%):

| cell | GB/s of expert bytes |
|---|---|
| NEON, 4 / 8 / 10 threads | 68.0 / 88.8 / 89.5 |
| streaming SVE SDOT (512-bit) on the P cluster's SME unit / the E cluster's / both | 1.8 / 1.5 / 3.2 |
| SME P + NEON x4 / SME P + E + NEON x8 | 67.2 / 82.0 |

Streaming SVE vector arithmetic (not the ZA outer products) runs at a few GB/s on these units, and the core that issues it
is lost to NEON: adding SME lowers the total. The outer-product form (SMOPA) wastes 15 of 16 positions at batch 1 and
needs the nibbles unpacked and interleaved by NEON first, about the NEON kernel's own unpack cost. SME for decode: no,
by these numbers. The kernel alone reaches 89 GB/s on 8 threads against the stage's 79 inside the engine: the remaining
gap of the 35B's expert stage is its structure (a fork-join per layer, the gate / up -> down dependency), about 1 ms.

**Prefill experts** (Inkling 1.39x the 35B's cycles per weight byte per position): both run the same structure, each
expert computed for all positions that picked it in one pass over its record; the 35B's batched prefill (`prefill.cpp`)
uses the i8mm gemm for every expert, the shared `moe_chunk.cpp` uses the packed NVFP4 gemm from 6 users and the
multi-vector kernels below that, and has no packed MXFP8 kernel (Inkling's layers 20-23, 14% of its expert bytes). The
difference is in the NVFP4 / MXFP8 kernels (Inkling's own), not in a structure the 35B could lend: not moved.

**C5 (our routers' naming off the critical path): parked.** It is on the path for both models: after the router's
fork-join the main thread finishes the pending target's logits, sorts them, applies the calibration and calls
Cache::name (2.05 / 3.36 M cycles per token of serial work in the routers' stage, part of it the model's own softmax
and sort). Moving it to a helper thread is not the one-pass change: the naming touches the cache (names, the keep-score
features), Inkling's cross-token router wraps it with its own fork-join at its source step, and the names would be issued
after the experts instead of before them (later reads). Estimated gain 0.3-0.5 ms per model; needs its own design.

## Run-ahead (one layer ahead while a step waits on reads), measured

`dmeng --run-ahead 8` (commits 6f09744, 91709f0, 8baf842): when a step is about to wait on a read, the model runs the next
layer (application) up to its router on the approximate state (this step's experts in memory only), the core names that
router's top 8 absent picks as reads (taking a slot as a miss does), and the work is dropped the moment the awaited read
lands. Exact mode: ids identical (by construction: only reads change; tiny-model logits checked with it on). Fast mode:
more picks are in memory when a layer runs, so fewer are left out (the output changes, towards exact). Counters per token,
from `results/parity/ra_table.py` (machine loaded: others' CPU in each row).

| run | others | wait ms | urgent MB | MB read | weight left out | run-ahead ms | top-8 overlap | on-demand reads issued early | lead ms | wasted names |
|---|---|---|---|---|---|---|---|---|---|---|
| 35B exact 6.6 GB, Pittsburgh, off / on | 83 / 127% | 11.3 / 6.0 | 20.0 / 15.2 | 71.3 / 71.5 | - | 4.5 | 0.985 | 0.931 | 1.31 | 21 of 371 |
| 35B exact 6.6 GB, code, off / on | 56 / 92% | 16.7 / 9.9 | 27.6 / 18.6 | 96.7 / 97.5 | - | 5.7 | 0.978 | 0.917 | 1.59 | 69 of 713 |
| Inkling exact 7.8 GB, Pittsburgh, off / on | 149 / 148% | 14.4 / 10.6 | 31.6 / 24.2 | 82.9 / 83.4 | - | 3.0 | 0.979 | 0.895 | 3.56 | 19 of 181 |
| Inkling exact 7.8 GB, code, off / on | 175 / 169% | 32.3 / 25.0 | 65.2 / 44.4 | 168.3 / 170.9 | - | 6.4 | 0.967 | 0.898 | 4.30 | 61 of 518 |
| 35B phone config (q4 resident, 1.40 GB, approx:drop 0.08), prose, off / on | 178 / 172% | 39.0 / 34.3 | 79.6 / 38.9 | 159.9 / 193.1 | 2.30 / 1.26 | 8.0 | 0.959 | 0.899 | 1.70 | 1,625 of 16,757 |
| same, code, off / on | 95 / 149% | 46.0 / 41.4 | 106.2 / 48.7 | 176.0 / 216.1 | 2.44 / 1.36 | 8.1 | 0.951 | 0.892 | 1.91 | 2,266 of 21,987 |

The run-ahead's own time is spent while the awaited read is outstanding (overrun after it lands, as first seen: about 0
ms per token; a third of the runs are given up when the read lands); the engine counts it as compute, not wait, so on the
Mac "wait" falls by about the run-ahead time plus what the earlier reads saved (35B code: 16.7 -> 9.9 + 5.7 run-ahead,
about 1 ms net at 6.6 GB, where the drive is not saturated). The phone (drive saturated, CPU idle three quarters of the
time) is where the earlier reads should pay; there the cost is the extra bytes of fast mode reading more of its picks.

## Same-load A/B inside one process (2026-11-23)

The machine was shared during every run (others' CPU 54-228%), and separate runs of one binary differed more than the
changes under test: the same down-claim binary gave 17.9 and 5.2 spin thread-ms per token in two runs, and the run-ahead
off / on pairs in separate processes swung between -16% and +6.5%. So both comparisons below alternate the two arms
inside one process (a fixed pseudo-random choice per call or per token), so both arms see the same load.

**Down claims** (`DMENG_DN_AB=1`, `results/parity/run_d.sh`, logs `ab_*`): the expert fork-join alternates call by call
between the earlier rule (one claim counter, down chunks in expert order, a claim waits on its expert's gate-up) and the
current one (02785f7: each thread starts on the expert of its last gate-up chunk, then the next ready expert; per-expert
counters on their own cache lines). Exact mode, default caches, 128 tokens.

| model, prompt | others | calls (claim order / own expert) | ns per weight byte | GB/s | spin thread-ms (all) |
|---|---|---|---|---|---|
| 35B, Pittsburgh | 127% | 3,547 / 3,465 | 0.0127 / 0.0127 | 79 / 79 | 243 / 242 |
| 35B, code | 68% | 3,880 / 3,808 | 0.0122 / 0.0119 | 82 / 84 | 157 / 124 |
| Inkling, Pittsburgh | 167% | 3,631 / 3,531 | 0.0151 / 0.0143 | 66 / 70 | 1,553 / 997 |
| Inkling, code | 170% | 4,017 / 3,930 | 0.0140 / 0.0144 | 71 / 69 | 928 / 1,079 |

The current rule does not slow the 35B (equal on Pittsburgh, 2.5% faster on code). On Inkling it is +5.6% on Pittsburgh
and -2.8% on code, which is within this machine's noise. The earlier figures (the 35B worse, Inkling spin 93.6 -> 11.8)
came from separate runs at different loads and are withdrawn. It is one rule for both models. Gates on it: 64 greedy
tokens at 2 GB equal v1's ids on both models (`t_gate_*`).

**Run-ahead, honest Mac number** (`DMENG_RA_AB=1`, `results/parity/run_e.sh`, logs `e_*`): run-ahead is skipped or
allowed token by token in one process (exact mode, `--run-ahead 8`, default caches 6.6 / 7.8 GB, 256 tokens; the wall ms
per token is everything: compute, wait, run-ahead, head).

| model, prompt | others | tokens off / on | ms per token off / on | wait off / on | tok/s on vs off |
|---|---|---|---|---|---|
| 35B, Pittsburgh | 177% | 121 / 134 | 53.1 / 47.6 | 11.2 / 5.1 | +11.6% |
| 35B, code | 125% | 121 / 134 | 49.6 / 48.4 | 16.5 / 10.6 | +2.5% |
| Inkling, Pittsburgh | 180% | 121 / 134 | 77.9 / 73.2 | 13.0 / 9.2 | +6.4% |
| Inkling, code | 190% | 121 / 134 | 86.4 / 86.6 | 27.1 / 21.4 | -0.3% |

Prose gains on both models; code is neutral on both, where the wait it removes is about what it costs. Two biases go
against "on": names issued on an on token can also help the next off token, and the cores the run-ahead uses during a
wait were contended by other processes here.

**Depth and where the remaining wait sits** (`results/parity/run_d.sh`, logs `d1_*`, `d3_*`; counting build; 35B
`--run-ahead-depth 3` continues from the run-ahead's state with each layer's MoE left out, measurement only, not
abortable).

- Today the run-ahead goes one layer ahead (it starts only when a step is about to wait).
- Depth 1: top-8 match 0.985 / 0.978 (35B Pittsburgh / code), 0.93 / 0.92 of the on-demand reads issued early, 25 / 67
  wasted names.
- Depth 2: top-8 match 0.87 / 0.86. Our lag-2 routers have already named most of layer L+2's picks by then; of the 328 /
  612 picks still absent, 44% / 49% are found, and 160 of 304 / 271 of 571 names are wasted.
- Depth 3: top-8 match 0.82 / 0.81; 65% / 64% of the absent picks found, 318 of 907 / 476 of 1,447 names wasted. Our routers
  name those picks one step later anyway.
- Cost: depths 2 and 3 add 8-9 ms per token of serial work against a remaining wait of 6-10 ms. Not worth it.
- Where the wait sits, depth 1 on:
  - Steps that never waited: 79% / 72% (35B), 88% / 82% (Inkling).
  - Waits at steps the previous step named reads for: 1.3 / 3.5 ms per token (35B), 2.7 / 5.6 (Inkling).
  - Waits at steps nothing was named for (a chain's first miss, or a run-ahead given up when its read landed): 5.0 /
    6.8 ms (35B), 8.2 / 16.3 (Inkling).
- A run-ahead at steps with no miss: no. It would run at about 31 (35B) / 42 (Inkling) no-wait steps per token, at 0.44 /
  0.53 ms each, which is 14 / 22 ms of serial work. The cores are not spare at those steps: the step's own work
  occupies all 8 threads (spin is about 1% of thread time; the experts stream at 79 GB/s). The most it could recover is
  the 5.0 / 8.2 ms of not-named wait.

**Run-ahead limits and the cheap path** (c70c1fc; `results/parity/run_b.sh`, logs `c_*`). Engine options a profile sets:
`ra_budget` (speculative ms per token at most this fraction of the running average compute per token), `ra_max_tok`
(run-aheads per token), `ra_sat_skip` (none while every reader has a read in flight and reads are queued behind them);
dmeng `--run-ahead-budget F --run-ahead-max N --run-ahead-sat-skip 1`; counters `skip_budget`, `skip_saturated`. The
cheapest input to the next router was tried on the 35B and removed: its router on the residual with its attention left
out costs 0.03-0.05 ms per event instead of 0.44, but matched 0.83 / 0.79 of the top 8 (Pittsburgh / code; full path
0.985 / 0.978), named 0.59 / 0.52 of the on-demand reads early, and 507 of 790 / 970 of 1,470 names were wasted; off / on
token by token: 43.5 / 46.2 and 84.5 / 84.9 ms per token (no gain). The next layer's attention is what the prediction
needs; the budget is the lever.

## Inkling's decode expert kernel: candidates, streaming, and the libm activation (2026-11-23)

`tests/kernels/bench_kmulti_nv.cpp` (log `results/q36_speed/nv/bench1.log`, through guard): real records (40 NVFP4 experts of
layer 0, 24 MXFP8 of layer 21; 180 / 197 MB), one thread in cache (least cycles of 9 trials) and 8 threads streaming over
the set (rows claimed 64 at a time), outputs compared bit for bit with the engine's kernel and against a float64 reference
on 36,864 real rows.

| kernel | ins/B | cyc/B (1 thread) | GB/s (8 threads, best / median) | bits vs engine |
|---|---|---|---|---|
| NVFP4 K0 (engine `mv_nvfp4`) | 0.890 | 0.168 | 89.7 / 88.3 | - |
| K1: four rows share the activation loads | 0.837 | 0.168 | 76.6 / 75.5 | equal |
| K0U: scales by one unsigned shift (every real scale is positive) | 0.877 | 0.165 | 84.3 / 75.7 | equal |
| G2: group-lane code order (reordered at load) | 0.849 | 0.158 | 91.8 / 90.2 | equal |
| G4: G2, four rows | 0.820 | 0.158 | 81.1 / 80.5 | equal |
| G2I: integer fold (mantissa multiply, exponent shift) | 1.114 | 0.197 | 89.3 / 88.8 | differ (error 1.07e-7 vs 0.91e-7) |
| MXFP8 M0 (engine `mv_mxfp8`) | 0.789 | 0.143 | 103.8 / 102.6 | - |
| M1: four accumulators | 0.900 | 0.159 | 106.3 / 104.5 | 210 of 9,216 rows differ |
| AFF4 (the 35B's kernel, same shape, codes only) | 0.653 | 0.116 | 102.2 / 101.4 | - |

- The fused-FMA epilogue is already what K0 compiles to (`-ffp-contract=fast`).
- Four-row variants stream slower (more concurrent row streams per thread).
- Scale exponents per row span at most 5 values (median 2) and are never subnormal. Grouping by exponent buys nothing: the per-group scale is already applied lane-parallel.

**Verdict.** The kernel streams at 88-90 GB/s on 8 threads, but the engine's expert step runs Inkling at 66-71 GB/s (same-load A/B
above). The 35B shows the same pattern: 102 streaming, 79-84 in the engine. So both models lose 20-25% inside the engine's
expert step, and Inkling's extra gap to the 35B is mostly the format: 88 vs 102 streaming (a lookup per 16 weights, a scale
per 16). The best bit-identical rewrite (G2) gains 2-5% streaming and needs NVFP4 codes reordered at load, which the
batched kernels would also have to read. Not worth it. What the expert step spends beyond streaming, per Inkling token:
- spin while down rows wait on their expert's gate-up: 8-14 thread-ms, which is 1-1.7 ms of the ~24 ms step;
- the activation's libm calls (below): about 0.3-0.6 ms;
- about 56 fork-joins per token (1.17 per application);
- the rest, not attributed here: reads landing in parallel, and E-core tails.

**The activation's libm calls** (`tests/kernels/vforce_check.cpp`, `libm_cr_check.cpp`, every one of the 2^32 inputs):
- Accelerate's `vvtanhf` differs from `tanhf` on 170,432,080 inputs, over all |x| < 9.01. `vvexpf` differs from `expf` on
  89,499,432 inputs, over -98.8 .. 88.7.
- libm itself is not correctly rounded: `tanhf` differs from the float rounding of the double `tanh` on 4,421,346 inputs (|x| < 6.19), and `expf` differs on 2,041,856. So no correctly rounded vector routine reproduces its bits either.
- Restricting the input range does not help. The softcap's input is the gate-up output divided by the cap, which nothing in the code bounds, and the differences cover the whole range where tanh is not saturated.
- Left as is. The cost at stake is small: `tanhf` 3.6 ns and `expf` 3.5 ns per call, against 1.9 ns for `vvtanhf` (measured at background priority), which is under 3% of the expert step.

## The expert step against streaming: replica and in place (2026-11-23)

`expert_step` (include/dmeng/expert_step.h) is now the engine's expert fork-join as a free function: `Core::run_experts`
calls it, and so does `tests/kernels/bench_kmulti_step.cpp`. The bench runs the same chunking, claim rules and pool on real
records (Inkling 48 NVFP4 records, 215 MB; the 35B 120 AFF4 records, 212 MB, plus a synthetic AFF8 shared expert). Each
application is 8 experts, with its arm drawn at random (same load for every arm). The arms that keep the arithmetic give
the engine arm's outputs bit for bit. Logs: `logs/step_{a,b,c}_*.log` (`results/parity/run_step.sh TAG BUSY`).

GB/s of weight bytes through the call(s):

| arm | 35B quiet (b) | 35B, 2 co-runner threads (c) | Inkling quiet (b) | Inkling, 2 co-runners (c) |
|---|---|---|---|---|
| engine (one call) | 106.0 | 83.1 | 91.2 | 71.2 |
| stream (rows claimed 64 at a time, no activation, no dependency) | 99.9 | 82.7 | 88.6 | 73.5 |
| nodep: down rows do not wait for gate-up | 105.0 | 86.7 | 89.6 | 71.3 |
| noact: no activation or its quantisation | 106.3 | 85.9 | 90.6 | 76.0 |
| twojoins: the step as two calls | 104.7 | 82.6 | 88.2 | 66.1 |
| wave7+1: 7 experts, then 1 | 101.4 | 72.9 | 86.1 | 57.6 |
| parked: workers asleep before the call (as after a long read wait) | 96.7 | 68.0 | 84.6 | 67.6 |
| reader: F_NOCACHE preads at 1.5 / 2 GB/s alongside | 105.0 | 93.8 | 91.5 | 81.1 |
| warm: the previous application's records again | 123.7 | 95.4 | 95.0 | 73.8 |
| ptail / smalltail (efficiency cores stop at the down tail / smaller tail claims) | 105.6 / 105.2 | 85.5 / 74.7 | 89.8 / 92.4 | 71.3 / 72.1 |
| ponly: efficiency-core threads claim nothing | 93.0 | 64.0 | 73.4 | 58.0 |
| e_no_gateup / e_no_down | 98.6 / 92.6 | 73.0 / 77.2 | 77.8 / 77.7 | 56.1 / 61.2 |

In place (`expert_calls_jobs_calls_gb_s` in dmeng's JSON; logs `x_*`, exact, default caches, 128 tokens, others' CPU
20-38%), by jobs per call:
- Inkling Pittsburgh: 8 jobs 94.6 GB/s (5,179 calls), 7 jobs 92.3. Single-expert calls after a read wait: 67.8 (924 calls).
- Inkling code: 8 jobs 84.6, 7 jobs 85.1. Single-expert calls after a wait: 64.5 (1,623 calls).
- 35B Pittsburgh: 9 jobs 91.3 (3,586 calls), 8 jobs 89.7. Single-expert calls after a wait: 63.8 (1,593 calls).
- 35B code: 9 jobs 91.4, 8 jobs 89.2. Single-expert calls after a wait: 65.7 (2,180 calls).

Ids: the first 64 equal v1's on both models.

**Attribution.**
- **The step is not the loss.** On a quiet machine it runs at or above the kernels' streaming rate in this process (35B
  106 vs 100, Inkling 91 vs 89), and in place its full calls run at 89-95 GB/s on both models.
- **The 20-25% gap is other processes' load.** The 66-71 / 79-84 GB/s figures came from runs with 1.3-1.9 cores of other
  processes. Two spinning co-runner threads reproduce them in the replica (71 / 83). Plain streaming drops the same amount
  (74 / 83), so the cores are taken away from the step, not lost inside it.
- **Arm by arm, quiet machine:**
  - each change is 5% or less: the dependency, the activation, a second join, cold records, a concurrent reader;
  - parking after a long wait costs 7-9%, and a 7+1 wave split 4-6%;
  - under load these two grow (to 18% and 12-19%), because a co-runner can preempt a thread that holds a claimed chunk.
- **Efficiency cores help.** Without them the step is 12-20% slower quiet and 19-23% slower loaded. Run a (others' CPU
  129%) gave the opposite once on the 35B (149 vs 104), so where co-runners sit decides it.
- **What is left to fix in place is small.** The single-expert waves after a read wait run at 64-68 GB/s, on 2-5% of the
  bytes: at most about 1.5% of the step.

## Re-run of a stalled claimed gate-up chunk (STEP_RERUN, on by default)

A thread that would spin (no down rows of a ready expert left) looks for a gate-up chunk that was claimed more than 4x a
chunk's running time ago and is not done. It re-runs that chunk into its own scratch. Every gate-up chunk is now computed
into scratch and published once: a per-chunk state goes claimed (1) or re-running (3) to done (2) by compare-and-swap, and
only the winner copies the activations, quantises them and counts the chunk. Any other copy is discarded, so no two threads
write the same bytes. It is the same function of the same inputs, so the outputs are bit-identical. Checked:
- the replica's bit test, 300 forced re-run steps per model, 13-33 chunks re-run, outputs equal;
- the tiny model with every stalled chunk re-run (`DMENG_RERUN_LIM_NS=0`), ids equal;
- 64 greedy tokens at 2 GB on both models equal v1's ids (603 and 32 chunks re-run).

Measured with `results/parity/run_rr.sh` and `run_rr2.sh` (`rr_table.py`). In the engine, off and on alternate call by
call (`DMENG_RERUN_AB=1`, exact, default caches, 128 tokens); the co-runner is a second process spinning 2 threads, started
inside the guarded command:

| | quiet | beside the co-runner (Pittsburgh) | beside the co-runner (code) |
|---|---|---|---|
| 35B in place, GB/s off / on | 91.4 / 91.4 (0.0%) | 70.8 / 76.5 (+8.1%) | 53.0 / 65.9 (+24.4%) |
| Inkling in place | 88.9 / 90.1 (+1.3%) | 76.6 / 78.4 (+2.3%) | 69.7 / 73.6 (+5.7%) |
| replica 35B (0 / 1 / 2 co-runner threads in process) | +0.1% / +0.3% / -1.5% | | |
| replica Inkling (0 / 1 / 2) | -0.3% / -1.7% / +4.1% | | |

Kept: no cost when the machine is quiet (0.0% and +1.3%). Beside a real co-runner process it gains 8-24% on the 35B and 2-6% on
Inkling. In the replica the co-runners are threads of the same process and it moves little (-1.7% to +4.1%).
