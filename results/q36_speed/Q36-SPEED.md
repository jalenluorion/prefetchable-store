# Qwen3.6-35B-A3B decode compute: what was cut, exactly (2026-11-23)

Branch `q36-speed`, rebased on `ink-speed` (which carries the shared stage markers of the counting build). Every change
keeps the output bit-identical: the same token ids, the same logits wherever logits are produced, the same sampled ids
for a seed. Setting of every real-model run: base M4 MacBook Pro 16 GB, 8 compute threads, 4 readers, the Pittsburgh prompt
(`bench/prompts/qwen3_5_moe/pitt.ids`), linear k2 routers with m 8, 6.6 GB expert cache, mlock, one token at a time,
each run through `engine/bench/guard.py`. The machine was shared with a timing campaign, Inkling measurements and a phone
build all session (one of them ran at 650% CPU outside the guard's queue for the last hour): every number below carries
the others' CPU of its run ("others"), and only runs at similar others' CPU are compared.

## 1. Where decode compute goes (before), and what limits each stage

Counting build (`-DDM_COUNTERS=ON`, `DM_COUNTS=1`), 128 tokens, per decode token. Bytes are what the stage reads per
token (weight file and record layout); GB/s is bytes over the stage's time; the machine reads 102-107 GB/s at 8-10 threads.

| stage | MB / token | ms (v1, others 53%) | GB/s | instructions (M) | cycles per byte (all threads) | limit |
|---|---|---|---|---|---|---|
| attention + DeltaNet projections (8-bit) | 1,001 | 10.98 | 91 | 374 | 0.28 | read rate within 10-15%; the rest is fork-join tails |
| attention core (DeltaNet conv + recurrence + its fused output projection; full-attention heads) | 268 + 126 state | 4.71 | 84 | 240 | 0.29 | bytes: 394 MB need 3.8 ms at 105 GB/s; ~0.9 ms is our code (tails, the per-head serial chain) |
| attention output (full layers, 8-bit) | 89 | 0.97 | 92 | 32 | 0.26 | read rate |
| router + lag-2 router rows (bf16) | 82 | 1.22 | 67 | 34 | 0.29 | our code: 2 MB per layer per fork-join, tails |
| routed experts (4-bit) + shared expert (8-bit) | 566 + 128 | 8.41 | 83 | 530 | 0.30 | our code + the efficiency cores: the I16 4-bit kernel needs 2.1x the instructions per byte of the 8-bit one |
| output head (8-bit) | 540 | 5.42 | 100 | 197 | 0.26 | the model's bytes per token at the read rate (nothing left to gain without reading fewer bytes) |
| main-thread serial (norms, route, cache, combine, hooks, embed) | - | ~1.0 | - | 6 | - | sequential double sums and libm exp that fix the bits |

Gaps named: (a) the expert stage is ~1.5 ms off the read rate. Per byte it costs the same cycles as the 8-bit stages
(0.30 vs 0.28), so it is not the dequantisation arithmetic on the performance cores; the time goes to the efficiency
cores (the 4-bit I16 kernel issues 0.76 instructions per byte, which they cannot sustain at their share of the read
rate) and to the join of each layer's fork-join (8 picks over 8 threads, a 64 KB down chunk last on a slow core). (b) The
head is at the read rate already: the only way to make it cheaper is to read fewer bytes. (c) Attention core: no repeated
work; it is bytes (the DeltaNet output projection fused into the heads' fork-join, 2.6 ms; the fp32 state read and
written, 1.2 ms) plus ~0.9 ms of tails and per-head serial chains. (d) Joins: under load, the main thread's spin at the
joins was 14 / 11 / 6 / 59 Mcycles per token in projections / core / router / experts (the experts' figure includes the
waits on expert reads), i.e. one slow or interrupted thread held every fork-join for a whole 32-64 row chunk. (e) Serial:
~4.2 Mcycles per token, most of it bit-fixing sequential arithmetic (rmsnorm's double sum, the router's 256 libm exp in
double); left as is.

## 2. What changed (two commits on top of `ink-speed`; core files listed)

1. **Short claim tails** (core: `src/core/weights.cpp`, `src/core/moe.cpp`; model: `qwen3_5_moe.{h,cpp}`). The row
   phases (`run_rows`, `run_rows_q`: projections, router, attention output, head), the expert step's down rows and the
   DeltaNet output projection now claim whole chunks while more than four quarter chunks per thread remain, then quarter
   chunks. Every row is still computed by one kernel call on one thread, so no bit can change (checked: teacher-forced
   logits of the tiny model bit-identical to v1 at i16 / i8 / f16 and 3 / 8 threads; real-model ids identical).
   Overlap note: this is the same lever as its planned dynamic row-block scheduling in the pool;
   it lives in `weights.cpp` / `moe.cpp`, not in `pool.cpp`, and either can replace the other.
2. **Screened head** (core: new `src/core/screen_head.cpp`, `include/dmeng/screen_head.h`; `include/dmeng/weights.h`,
   `src/core/weights.cpp` (Weight::AFF8S), `include/dmeng/kernels.h`, `src/core/kernels.cpp` (mv_aff8s_i16),
   `include/dmeng/model.h` (Want::TOPK), `CMakeLists.txt`; model: `qwen3_5_moe.{h,cpp}`, `prefill.cpp`; `tools/dmeng.cpp`).
   - The 8-bit head is split **in place** into nibble planes per 64-row block (no copy of the weights; +8 MB of bounds,
     +1 MB per-row scratch). Every kernel that reads it (decode, prefill's last position, the multi-vector and packed-gemm
     verify paths, other precisions) gives the AFF8 bits: the decode kernel rebuilds the bytes with vsli / vsri before
     the same USDOTs and float steps; the others rebuild rows and call the AFF8 kernels. At load the split is checked
     against the original bytes on a test input (one-position kernel, multi-vector, packed gemm, the screen at k 1 and
     20), bit for bit; a mismatch refuses the option.
   - One pass over the high nibbles (half the code bytes + scales and biases: 286 of 540 MB) gives each row an interval
     that holds its exact logit: the low nibbles' reach per group is min(7.5 sum|x|, ||lo - 7.5||_2 ||x||_2) (the 2-norm
     per row and group stored rounded up), plus a float slack ~40x above both kernels' rounding. The floor is the higher
     of the k-th largest lower end and the k-th largest exact logit of the k rows with the best midpoints. Rows above the
     floor are computed with the full head's row kernel; the rest get -inf.
   - Exact for the argmax (lowest index on ties) and for the exact top-k rows, order and logits, so a top-k sampler
     (the release sampler: temperature, then top-k, then top-p within the k) gives the same ids and probabilities.
   - `dmeng --fast-head 1`: greedy, or sampling with `--top-k 1..64`; `--fast-head 2` also runs the full head every token
     and counts mismatches. Opt-in; drafting still uses the full head (it is refused with the option, as before); the
     serving worker (`dmserve`) is not plumbed yet.
   - Also fixed: with a batched prefill, dmeng's first greedy token read a stale id (-1) from the fast head.
   - Bound tightness, measured on the real model (candidates per token, of 248,320 rows):

     | bound | greedy | top-k 20, top-p 0.95, temperature 1 |
     |---|---|---|
     | L1 only (7.5 sum|x| per group) | 36,754 | 243,237 (useless) |
     | + per-group Cauchy-Schwarz + exact-midpoint floor (kept) | 75 | 4,343-6,756 (three seeds) |
     | a 2-bit first level (bits 7-6), then 4-bit, then exact (tried, dropped) | 229,966 rows reached level 2 | - |

## 3. Per stage, before and after (counting runs, per decode token)

Cells: stage timer ms / instructions (M) / main-thread cycles (M) / main-thread spin at the joins (M cycles). The main-
thread and spin columns come from a counting variant built before the rebase (two extra DMSTAGE columns in `pool.cpp`,
dropped so the branch stays out of the pool, which is changed separately); the rebased build's counting run with the
head (`cnt_n3_h1`, others 69%) gives compute 32.3 ms, head 3.74 ms, ids identical.

| stage | before: v1 behaviour, others 130% | after, head off, others 130% | after, `--fast-head 1` greedy, others 75% | after, `--fast-head 1` top-20 sampling, others 150% |
|---|---|---|---|---|
| projections | 17.28 / 393 / 50.7 / 14.1 | 11.90 / 377 / 38.7 / 2.5 | 11.44 / 375 / 37.1 / 1.3 | 13.60 / 378 / 40.7 / 4.3 |
| attention core | 11.13 / 411 / 30.3 / 10.6 | 5.30 / 251 / 17.1 / 1.8 | 5.05 / 246 / 16.2 / 1.2 | 6.73 / 274 / 19.4 / 3.3 |
| attention output | 1.78 / 32 / 5.1 / 2.1 | 1.06 / 32 / 3.4 / 0.3 | 1.05 / 32 / 3.4 / 0.3 | 1.34 / 32 / 3.9 / 0.8 |
| router | 4.02 / 50 / 10.1 / 6.1 | 1.62 / 35 / 4.9 / 1.2 | 1.35 / 35 / 4.4 / 0.7 | 2.22 / 37 / 6.2 / 2.4 |
| experts (spin includes read waits) | 19.56 / 913 / 94.2 / 58.9 | 9.58 / 563 / 68.4 / 40.7 | 8.89 / 545 / 64.5 / 37.4 | 12.14 / 621 / 71.0 / 41.6 |
| head | 6.21 / 197 / 17.9 / 1.2 | 5.56 / 197 / 17.6 / 0.1 | **3.60** / 274 / 11.8 / 0.1 | **4.56** / 282 / 13.3 / 0.4 |
| compute ms / token | 63.0 | 37.2 | 32.9 | 42.6 |

The before / after pair at the same others' CPU (130%): compute 63.0 -> 37.2 ms per token (ids identical); the joins'
spin fell 5-9x in every stage. Under load the tails were the dominant loss (an interrupted worker held each join for a
whole chunk); on a quiet machine the same change is worth less (v1 measured 33.7 ms at others 53%, the brief's clean
figure is 31.1). The head: 5.4-5.6 -> 3.6 ms greedy and 4.6 ms for top-20 sampling (under heavier load), from reading
286 instead of 540 MB; its instructions rose (274 vs 197 M: the high-nibble pass issues 2 SDOT per element pair over half
the bytes), its cycles fell by a third.

## 4. Gates

| gate | result |
|---|---|
| tiny model: teacher-forced logits vs v1 binary, i16 / i8 / f16, 3 and 8 threads | bit-identical (all 6) |
| tiny model: ids with `--fast-head 2`, greedy (prefill chunk 1 and 512) and top-20 sampling, 3 seeds | identical, 0 mismatches |
| `tests/draft_check.sh` (both tiny models) | DRAFT CHECK PASSED (v1-based and rebased builds) |
| real model, 200 tokens, default settings, vs the release campaign's ids | identical (`n1_exact`; rebased build `n3_exact`) |
| real model, `--mode fast` vs the v1 binary | identical over 200 tokens (`n1_fast`, `n3_fast`) and over 400 (`w_new_fast`, `w_new_fast_fh`, `x_new_fast` all equal the v1 run `w_base_fast`). The fast mode is not run-to-run deterministic: two runs of the v1 binary itself (`w_base_fast`, `x_base_fast`) part at token 353 (which picks have landed decides what is dropped) |
| real model, `--fast-head 2` greedy, 200 tokens | ids identical to the reference, 0 mismatches against the full head (`n2_check_greedy`) |
| real model, top-k 20 / top-p 0.95 / temperature 1, seeds 1-3, 100 tokens, `--fast-head 2` vs the v1 binary | sampled ids identical for all three seeds, 0 mismatches (`n2_samp_s1..3`; 128 tokens of seed 1 in `cnt_n2_h1s`) |
| 35B prefill exactness (`tests/prefill_check.sh`, 1,100 positions) | PREFILL CHECK PASSED on both builds: 0 logit bits, 0 state bytes, 0 KV bytes differ, greedy identical |
| rebased build (on `ink-speed`): 200 exact ids, `--fast-head 2` greedy 200, seed-2 sampling 100 | identical, 0 mismatches (`n3_*`) |

## 5. Wall clock, before / after, 6.6 GB, 400 tokens

Exact = `--temp 0`, fast = `--mode fast`; "+head" = `--fast-head 1`. Compute = time per token minus waiting on reads.
Pairs are back to back in the guard's queue; others' CPU differed between members of most pairs (it ranged 63-316% over
the session), so the pairs below are the best-matched ones; the rest are in `logs/` (`w_*`, `x_*`, `y_*`).

| run | others | tok/s | compute ms | wait ms | head ms | ids |
|---|---|---|---|---|---|---|
| v1, exact (`y_base_exact`) | 93% | 22.47 | 34.5 | 10.0 | 5.33 | reference |
| new, exact + head (`y_new_exact_fh`) | 74% | **25.15** | **30.2** | 9.6 | 3.38 | same |
| new, exact, head off (`x_new_exact`) | 92% | 22.11 | 35.2 | 10.0 | 5.42 | same |
| v1, fast (`x_base_fast`) | 96% | 27.16 | 33.5 | 3.3 | 5.25 | reference |
| new, fast + head (`x_new_fast_fh`) | 69% | **29.90** | **30.0** | 3.5 | 3.41 | same as `w_base_fast` |
| new, fast, head off (`x_new_fast`) | 64% | 27.85 | 32.5 | 3.4 | 5.24 | same as `w_base_fast` |
| v1, sampling top-20 (`w_base_samp`) | 70% | 23.12 | 32.5 | 10.7 | 5.24 | reference |
| new, sampling top-20 + head (`x_new_samp_fh`) | 63% | 24.11 | 30.9 | 10.6 | 3.76 | same |
| new, exact + head, 10 threads (`w_new_exact_fh_t10`) | 139% | 9.49 | 92.5 | 12.8 | 4.49 | same |

At moderate load the head is the durable gain: 5.2-5.4 -> 3.4 ms greedy, 3.8 ms sampling, ~1.9 / 1.5 ms of compute per
token, +10-12% tok/s at 6.6 GB. The claim tails are worth ~1 ms at others ~90% and much more under heavy load (section 3:
63 -> 37 ms at 130%); they are not resolved on a quiet machine, which these runs never had. 10 threads are far worse
than 8 under load (a stalled thread holds every join), as the Inkling measurements found.

## 6. Next biggest stage

The projections (1.0 GB per token, ~10.3 ms clean) are within 10-15% of the read rate: ~1 ms is reachable only by the
straggler work (the pool scheduling being built, or these tails). The **routed experts** are the
largest gap to a limit: ~8.4 ms clean for 694 MB where the read rate allows ~6.6-6.9 ms, so ~1.5 ms. The cause is
compute on the efficiency cores (0.76 instructions per byte for the I16 4-bit kernel against 0.36 for 8-bit), not the
arithmetic on the performance cores; what would move it: give the efficiency cores a smaller share of the expert rows
(or none) and the performance cores the rest, or an expert kernel with fewer instructions per byte at the same bits.
Second, the head: the high-nibble pass now issues more instructions than the full head; computing its bound on the high
activation plane only (one SDOT per pair, a second bound term for the low plane) would cut ~25% of its instructions,
worth up to ~0.5 ms if the pass is compute-bound on the efficiency cores (not measured). The serving worker would get the
head's gain (1.8-2 ms per token at greedy, ~1 ms at top-20) once `Want::TOPK` is plumbed through the session.

Never a ceiling: the head at 5.4 ms was the model's bytes per token at the read rate; it moved by reading fewer bytes,
exactly. The stages left are set by the read rate (projections, attention output), by our code (experts, router,
the joins) or by the state's bytes (DeltaNet recurrence: 126 MB per token).

## 7. Follow-up (same day): the head on by default, on Inkling, and a cheaper first pass tried

**Default on where exact** (commit "Screened head on by default where it is exact"; serving files `include/dmeng/lm.h`,
`include/dmeng/model_lm.h`, `src/core/model_lm.cpp`, `include/dmeng/session.h`, `src/serving/session.cpp`; tools
`dmserve.cpp`, `dmeng.cpp`). `LM::set_head_k(k)` tells the model what a step must deliver (0 every logit, 1 the argmax, k
<= 64 the exact top k); the prefix cache records the k its stored logits came from and reuses them only for a request
they cover (a repeated prompt whose last logits were screened is recomputed for a request that needs every logit).
dmserve builds the head at start and asks per request: greedy without drafting -> 1, top-k sampling with k <= 64 -> k,
anything else (top-p only, k > 64, drafting's verify passes) -> the full head; `--fast-head 0` keeps the full head always.
dmeng defaults to it (Qwen3.6, Inkling) for greedy without drafting or the verify check and for top-k <= 64, and falls back
to the full head where the head cannot be split (a mapped resident, other precisions).

| gate (real 35B unless noted) | result |
|---|---|
| dmeng defaults, 200 greedy ids vs v1 | identical (`d4_greedy`; 75.5 rows per token) |
| dmeng defaults, top-20 sampling seeds 1-3 vs v1 | identical (`d4_samp_s1..3`) |
| dmserve, 200 greedy + seeds 1-3 (one worker, 4 requests) vs v1's dmserve | identical (`srv_base` / `srv_n4`); head 5.57 -> 3.53 ms greedy, 5.5 -> 4.1-5.4 ms sampling (loaded) |
| OpenCode replay (`bench/drafts/replay.sh`: the recorded session through the HTTP front end, top-k 20, seed 1, 5 main requests, 128 tokens cap) | prompt ids and generated ids identical in all 5 requests (`replay_base` / `replay_n4`); head 5.4-6.4 -> 4.2-4.5 ms per step, 2.9-10K rows per token |
| prefill check (1,100 positions), the worker as served and with the head split (`--fast-head 1`) | PASSED both: 0 logit bits, 0 state bytes, 0 KV bytes |
| tiny: dmserve requests mixing greedy, top-20, top-p only, top-64, top-100, a repeat of the live prompt needing every logit after a screened step | identical to v1's dmserve, the stored logits recomputed where they did not cover the request; `draft_check.sh` PASSED |

**Inkling** (commit "Screened head for Inkling"; core `screen_head.{h,cpp}`, `weights.{h,cpp}` (a `srows` field: the
draft head views the head with VU rows, the split's layout keeps its own row count); Inkling's model file: the member
declarations (lines 22, 153-157 of the old file: `GreedyHead gh` -> `ScreenHead sh`, `int head_topk`), `stats_json` /
`reset_stats` (178-188), the head call site in `step()` (1309-1329: the old greedy block replaced by one `sh.run` with the
softcap), and `set_option` (2283-2288: `fast_head` builds the screen, `head_topk` added)). The softcap 30 tanh(x / 30) is
monotone but can round distinct logits to the same float, so the screen applies it to the kept rows and checks that the
largest upper end of every row left out, capped, is below the k-th kept capped value; if not, the full head runs (0
fallbacks in every run so far).

| gate (real Inkling, default settings, 7.8 GB) | result |
|---|---|
| 200 greedy ids vs v1 | identical; 22.6 rows per token of 200K; head 4.82 -> 3.52 ms (others 51% / 187%) |
| top-20 sampling seeds 1-3, 100 tokens vs v1 | identical; 2.9-5.5K rows per token; head 4.8-5.0 -> 3.35-3.8 ms |
| Inkling prefill check with the split head (1,100 positions) | PASSED: 0 / 0 / 0, greedy identical |
| tiny: dmeng greedy and top-k 20 / 40 (8-bit resident) and the bf16 resident (falls back), dmserve with drafting lookup / fixed, `draft_check.sh` | identical to v1, PASSED |

At low load the 35B now decodes at 24.8 tok/s exact (compute 29.5 ms, head 3.39 ms, others 29%: `n5_q_greedy`) and 22.6
tok/s with OpenCode's sampler (compute 31.1-31.3 ms, head 3.8-4.1 ms).

**Tried and dropped: the first pass from the high activation byte only** (exact: the low byte's share is bounded instead,
with a second stored norm per row and group). Head instructions fell 273 -> 228 M per token, its time did not (3.6 vs
3.6-3.9 ms, the pass is bound by the 286 MB it reads), and the looser bound kept more rows (top-20: 6.4K -> 10.3K, exact
part 0.5 -> 0.9 ms). Reverted.

**Experts: not changed.** The pull-based pool change had not landed then (it
touches `weights.cpp` and `moe.cpp` where the quarter-chunk claims live: whichever lands second should take the other's
place). The rows are already claimed by pull, so the efficiency cores already take fewer rows; what remains is the 4-bit
I16 kernel's 0.76 instructions per byte, and no exact formulation with fewer instructions per byte was found (SMLAL on
int16 activations and table lookups both cost more than two SDOTs per pair).

## Files

`logs/` holds every run's JSON, ids and guard log (the CMD lines carry this machine's paths: kept out of the commit);
`stages.py` prints the per-stage counts of a counting log, `wall.py` the timed runs; `run1.sh` .. `run6.sh` are the
batches as run.
