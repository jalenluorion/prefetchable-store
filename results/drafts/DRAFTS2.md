# Drafting, second pass: a controller from per-pass counters, sampled requests, fast mode (branch drafts2)

Base M4 16 GB, both models larger than memory. Branch `drafts2` = the release branch `v1` plus the draft-controller branch's
commits (the measured gate, sampling-safe drafting, `--verify-names` 0) and the work below. `--drafts` stays the switch and
stays off by default. Numbers are from `bench/drafts/drafts2.py` (every run through `engine/bench/guard.py`, the shared
first-come-first-served queue); the machine was shared with a timing campaign and two agents building and timing kernels,
so every row carries the others' CPU, and only ratios inside one interleaved pair are read.

Summary. Commits 6eaa70a..4acab53 (the draft-ctl commits, then this work). Every real-model pair is ids-identical: greedy
drafts vs exact on 8 cells, sampled drafts vs sampled plain for three seeds per model in dmeng and four (35B) / one
(Inkling) requests through dmserve, fast mode with nothing left out vs exact on both models. Drafting now runs in fast
mode. Counters (section 4): on copied text a pass halves compute per token and keeps the wait per token near a plain
step's; Inkling's coding answer in fast mode reaches 5.75 tokens per step.

## 1. The controller (`src/core/draft.cpp`, `Drafter::est`, `Drafter::measure`)

The rule. For every pass width n (n positions: the fed token and n - 1 drafts; n = 1 is a plain step) the drafter keeps
per-pass counters: passes, tokens produced, compute ms (wall minus the wait for reads, the draft head's own time left out),
wait ms, urgent MB and total MB (urgent + named), and the excess wait of the plain step after a pass. A pass of width n is
priced at `c1 * rho(n) + wait(n) + next`: `c1` = a plain step's compute; `rho(n)` = the pass's compute over `c1` at the
time (a ratio, so a busy machine slows both sides alike); `wait(n)` = the pass's wait; `next` = the excess wait it leaves
the following step. A width with three or more recent passes is priced by its own decayed sums; any other width by lines
in the width fitted over every pass (compute ratio, urgent MB and wait; priors: the family's cost table and two plain
steps' urgent MB per drafted position). A plain step is `c1 + wait(1)` from plain steps that follow plain steps. For each
proposal the drafter takes the draft length l that maximises expected tokens (acceptance learned per source, bucket and
draft index) times `est(1) / est(l + 1)`, and drafts only if that beats 1 by 5%; otherwise it runs a plain step. After 48
declined offers it probes with the most drafts the learned acceptance expects all kept with probability >= 0.5 (a
long copied span is probed wide, an uncertain or unseen proposal with one draft), names the copied span's experts one step before the
probe, and every 6th pass after a fully kept one runs two drafts more than chosen (so more than one width is measured).

What the counters showed on the way (each a controller version; all ids-identical):

- **Rejected tries were paid at full width.** The draft-ctl gate tried a confident source at the controller's full length;
  on the 35B at 5 GB those were 5-7 position passes reading 420-440 MB urgently each (plain steps 45-68 MB) for 2-3
  tokens: 468 ms per token on the new-file call, 215-234 ms on the coding answer, against 49-86 ms plain steps (draft-ctl
  `ctl_runs`, gate cells). The extra compute per token of those "on" runs was the machine: their prefill, which drafting
  does not touch, was 23-38% slower than the paired off runs' (others' CPU 140-186% against 62-110%).
- **A pass without names waits for all its drafted positions.** Two-position probes on the 35B file rewrite waited 118 ms
  (plain 30) and were never repeated although 8 / 8 were kept; naming the span one step ahead of a probe fixed that cell
  (4.2 tokens per step, 343 / 362 kept). Naming at every near-paying plain step read 85 MB per token more on Inkling's
  file rewrite for one pass, so names ahead go only before probes.
- **Narrow probes misprice wide passes.** Probes of 2 positions are latency-bound (wait per urgent MB 1.9 ms against 0.7 in
  8-position passes on Inkling in fast mode) and the line through them priced the 6-8 position passes that pay on copied
  spans out; per-width pricing of a few passes measured in a CPU spike froze another run at 2-position passes. Hence the
  hybrid pricing, the wide probes for confident spans and the exploration step.

`--drafts 1` = `--draft lookup --early 4` (+ `--skeleton F` where the tool-call fragments are given) on both models.
Inkling's draft head stays opt-in (`--draft both`): with it, the Inkling run at a 7.8 GB cache reached the 10.5 GB footprint
limit and was killed (the head is 0.64 GB; the plain run's footprint is 9.86 GB).

## 2. Sampled requests (`Draws` / `sample_at`, commit from draft-ctl)

The sampler takes exactly one 64-bit draw of `mt19937_64(seed)` per sampled token, so the k-th token of a generation always
uses draw k. Under drafting each produced token takes the draw of its own index, and a drafted token is kept exactly when it
equals the token the sampler picks there with that draw; the ids are then plain sampled decoding's for the seed, whatever
was drafted. dmeng and dmserve both draft sampled requests now (`--drafts 1` no longer requires temperature 0).

Gates (release sampler: temperature 1.0, top-p 0.95, top-k 20; the file-rewrite call, 160 tokens, `--drafts 1` vs off,
35B at 6.6 GB, Inkling at 7.8 GB; tiny models first: three seeds x both models x `--draft fixed` / `lookup`, all identical):

| model | seed | ids on = off | passes | drafts kept / run | tokens per step |
|---|---|---|---|---|---|
| 35B | 1 | identical | 23 | 78 / 78 | 1.96 |
| 35B | 2 | identical | 25 | 83 / 83 | 2.09 |
| 35B | 3 | identical | 18 | 42 / 42 | 1.36 |
| Inkling | 1 | identical | 7 | 5 / 10 | 1.03 |
| Inkling | 2 | identical | 2 | 2 / 2 | 1.01 |
| Inkling | 3 (stops at 107) | identical | 5 | 13 / 15 | 1.14 |

Acceptance under sampling: 35B 203 / 203 drafts kept (greedy, same call: 343 / 362), Inkling 20 / 27 (greedy file rewrite:
see the timing table). These sampled runs used an earlier controller version (d706209); the ids rule does not depend on it.
The worker (dmserve, one greedy and three sampled requests on the same call, drafting off vs on):

| model | request | ids on = off | tokens | drafts kept / run | tokens per step |
|---|---|---|---|---|---|
| 35B (6.6 GB) | greedy | identical | 160 | 14 / 14 | 1.10 |
| 35B | seed 1 | identical | 160 | 122 / 137 | 4.30 |
| 35B | seed 2 | identical | 160 | 136 / 140 | 6.91 |
| 35B | seed 3 | identical | 160 | 111 / 123 | 3.31 |
| Inkling (6 GB) | seed 1 | identical | 160 | 5 / 7 | 1.03 |

(The 35B's later requests copy the earlier ones' outputs, which the worker archives. Inkling's worker at 7.8 GB passed
the 10.5 GB footprint limit with the prefix cache, so its gate ran at 6 GB; exact ids do not depend on the cache size.)

## 3. Fast mode (`Core::moe_chunk`, the models' `verify`)

Defined as: a drafted token is accepted when it equals the token the pass itself produces under the fast rule. The verify
pass applies the single step's leave-out rule (`approx:drop`, `Core::approx`) to every position at each MoE step: a pick
held in memory (landed or in flight) is used; a missing pick with released weight > 0.1 is read now (then the largest
missing until the missing weight is <= `read_mass`); a missing pick of lower weight is left out, unless another position of
the same pass reads that expert now (it is then in memory when computed, at no wait). Kept picks keep their released
weights, nothing is substituted, the combination is the same fixed fma chain over the kept picks. With nothing left out
the pass is the lossless chunk's bit for bit. Before this, `verify` returned false in fast mode and drafting silently fell
back to plain steps. Other approximate presets (`bounded`, `never`) still verify nothing.

Gates: `dmeng --verify-check 150` in fast mode with `--read-w 0` (every missing pick read, so nothing is left out) on both
tiny models: 0 logits, 0 state bytes, 0 KV bytes differ, at a large and at a small cache; with the default rule and a cache
small enough that picks are left out, the pass and single steps leave out different picks (as two runs at different
cache sizes do), so they differ, as expected of fast mode. Real models:

`--mode fast --read-w 0 --drafts 1` (the fast path with every missing pick read, so nothing is left out: 0 left out per
token) against exact: 35B file rewrite 160 / 160 tokens identical to exact (82 / 82 drafts kept, 2.06 tokens per step);
Inkling file rewrite 108 / 108 identical (to the stop token). With the default rule, fast + drafts runs (section 4): the
passes leave out about as much as single steps (left out per token in the table).

## 4. Counters (step 5): exact / exact + drafts / fast / fast + drafts

One run per cell (the rule for these runs: no repeats, judge by counters; the wall-clock table is taken separately, once,
on the merged tree). Profile defaults (`--model DIR`), `--prefill-chunk` 512 / 256, greedy, `--cache-gb` 6.6 (35B) and 7.8
(Inkling). Coding answer: `bench/prompts/<family>/code.ids`, 300 tokens. File rewrite: `results/drafts/prompts/*_toolrw_*`,
run with `--n 800` (it stops at 451 tokens on the 35B, 108 on Inkling exact, 229 in Inkling fast mode; `drafts2.py` now
defaults to 300). Drafted runs on controller 4dad02d (the last change, 4acab53, only narrows probes on proposals never
seen; tiny gates only). The machine was shared (others' CPU 57-387%), so compute per token is compared inside a run:
plain steps against passes of the same run.

| model | case | config | tokens/step | kept / drafted | plain steps: compute / wait / MB per token | passes: compute / wait / MB (urgent) per token | whole run MB (urgent) per token vs the run without drafts | wait per token vs without |
|---|---|---|---|---|---|---|---|---|
| ink | code | exact + drafts | 1.68 | 121 / 128 | 65.8 / 30.1 / 159 | 37.1 / 28.7 / 67 (147 tokens) | 175 (65) vs 145 (58) | 29.4 vs 27.1 |
| ink | code | fast + drafts | 5.75 | 247 / 253 | 151.1 / 15.2 / 136 | 38.9 / 16.5 / 24 (283 tokens) | 134 (23) vs 95 (11) | 16.4 vs 9.2 |
| ink | toolrw | exact + drafts | 1.07 | 7 / 7 | 70.6 / 32.5 / 179 | 50.1 / 16.2 / 80 (8 tokens) | 175 (64) vs 174 (62) | 31.3 vs 31.2 |
| ink | toolrw | fast + drafts | 1.48 | 74 / 110 | 158.5 / 9.2 / 101 | 105.4 / 20.1 / 36 (96 tokens) | 144 (19) vs 98 (10) | 13.8 vs 10.6 |
| q36 | code | exact + drafts | 1.03 | 9 / 26 | 177.5 / 13.2 / 92 | 165.5 / 38.1 / 88 (18 tokens) | 106 (30) vs 94 (27) | 14.7 vs 16.2 |
| q36 | code | fast + drafts | 1.05 | 13 / 39 | 211.2 / 3.7 / 73 | 327.1 / 7.3 / 28 (27 tokens) | 83 (9) vs 73 (8) | 4.0 vs 8.0 |
| q36 | toolrw | exact + drafts | 1.11 | 45 / 45 | 47.8 / 29.2 / 152 | 30.5 / 39.4 / 82 (53 tokens) | 162 (53) vs 154 (49) | 30.4 vs 29.2 |
| q36 | toolrw | fast + drafts | 2.16 | 242 / 250 | 49.1 / 19.0 / 131 | 29.6 / 26.9 / 59 (278 tokens) | 150 (47) vs 135 (30) | 23.9 vs 19.4 |


How to read it: on copied text a pass cuts compute per token to 0.45-0.65 of the same run's plain steps (Inkling coding
answer 37 against 66 ms; 35B file rewrite 30 against 48) while its wait per token stays near a plain step's (it reads
the drafted positions' experts urgently: 24-82 MB per token in passes against 10-49 urgent per plain step), so compute +
wait per produced token is lower in every cell where passes produced many tokens (Inkling code 66 against 96, fast 55
against 166 (that run's plain steps were contended); Inkling rewrite fast 125 against 168; 35B rewrite 70 against 77,
fast 57 against 68). Bytes read per token rise 5-40% over the run without drafts. On the 35B coding answer the text
does not repeat (9 / 26 and 13 / 39 drafts kept) and drafting stays at 1.03-1.05 tokens per step; there its probes cost
more than they return (passes 18-27 tokens; whole run +12-14% MB per token), which 4acab53 targets. The 35B file
rewrite in exact mode drafts 1.11 tokens per step here (the earlier version that named near-paying offers every step
reached 4.2 on the same cell, at 85 MB per token more on Inkling's rewrite); in fast mode 2.16 (8-position passes at 7.8
tokens each).

Wall time, for the record only (one run each, contended): `results/drafts/drafts2_table.md`.

## 5. Recommendation

- **`fast` profile: `--drafts` on for Inkling** (coding answer 5.75 tokens per step, 247 / 253 kept, passes 39 ms compute +
  16.5 ms wait per token against plain steps' 66-70 ms compute + 9-15 ms wait in clean runs; this is the route to the
  25 tok/s target). **Off for the 35B** until the final timing table confirms the file-rewrite gain (counters say
  +17% per drafted token on the rewrite, a small loss on the coding answer).
- **`exact` profile: off for both** for now: drafting keeps the ids and wins per drafted token on copied text, but in exact
  mode the passes' urgent reads leave the gain at 9-30% per drafted token and the 35B rewrite at 1.1 tokens per step;
  turn it on for Inkling's exact profile if the clean table shows the coding-answer gain (1.68 tokens per step here).
- Sampled requests may draft (the ids are the seed's); acceptance on the 35B rewrite stays full under the release sampler,
  so OpenCode traffic is no longer excluded.

## 6. Follow-ups (2026-11-23)

**Probe budget (dba4b46).** Probes, explorations and the names issued one step ahead of a probe may spend at most 1% of
the run's cost so far, in ms and in MB: every step credits 1% of its compute + wait and of its MB read; a probe (or
exploration, or a step that named ahead) is debited what it cost beyond plain steps for the tokens it produced, so kept
drafts pay it back; a chosen pass's gain over plain steps refills the credit. A probe runs only when the credit covers
it producing just its first token (est(l + 1) - est(1) ms, and the measured excess MB per drafted position). One run per
model (exact + drafts, coding answer, 300 tokens, ids identical to exact):

| model | passes / steps | probes run / held | tokens per step | MB per token (urgent) vs exact | compute / wait ms per token vs exact | others' CPU vs exact |
|---|---|---|---|---|---|---|
| 35B, 6.6 GB | 1 / 299 (2 positions) | 1 / 71 | 1.00 | 94.9 (27.1) vs 94.0 (26.9) | 33.4 / 17.0 vs 65.0 / 16.2 | 38% vs 204% |
| Inkling, 7.8 GB | 43 / 74 | 0 / 0 | 4.04 (225 / 229 kept) | 187 (73) vs 145 (58) | 71 / 27 vs 78 / 27 | 247% vs 126% |

On the 35B coding answer drafting now costs one rejected two-position pass and one named-ahead step in 299 steps; the
in-run plain steps read 94.1 MB per token against the exact run's 94.0. The whole-run difference against the exact run
(+1.0% MB per token) is within the run-to-run spread of the named reads (they depend on timing; the two runs had
others' CPU 38% and 204%). Inkling's coding answer is unaffected (8-position passes at 7.85 tokens each).

**Inkling's draft head in the engine's 8-bit format (22331f6, 35e870a).** The head's three blocks and the fold F are
held as AFF8 (groups of 64, fp16 scale and offset), every matrix read through one bf16 buffer and quantised into its
slot (the fold block by block), so the released bf16 weights are never all in memory; the RNN and hidden_out matrices
(small) stay bf16. Drafts only propose: a wrong draft is rejected, so the ids cannot change. The head's port check on
recorded features (`dmeng --draft-check`, no decoding; 399 positions per text, 7 drafts each, chain = drafts kept in a
row against the text):

| text | head | draft 1 kept | mean drafts kept in a row (of 7) | head memory (weights + state) | load peak footprint |
|---|---|---|---|---|---|
| coding answer | bf16 as released | 0.980 | 6.29 | 0.731 GB | 3.25 GB |
| coding answer | AFF8 | 0.980 | 6.29 | 0.437 GB | 3.51 GB (before the streaming load) |
| Pittsburgh | bf16 as released | 0.594 | 1.19 | 0.712 GB | 3.20 GB |
| Pittsburgh | AFF8 (streaming load) | 0.591 | 1.20 | 0.419 GB | 2.96 GB |

Same drafts as bf16: draft 1 on 399 / 399 (coding) and 398 / 399 (Pittsburgh) positions, all 7 on 0.987 and 0.945; draft-1
probability within 0.04 where equal. The bf16 numbers reproduce the MLX reference's (0.980 / 6.29 and 0.591 / 1.20).
Saved: 0.29 GB.

Decoding with it at 7.8 GB (`--drafts 1` = lookup + the head on Inkling now; coding answer, 300 tokens, one run): ids
identical to exact, peak footprint 10.28 GB (the bf16 head was killed at the guard's 10.5 GB on the same cell), 2.30
tokens per step, 169 / 171 drafts kept. The head ran 3 times (78 ms per run; back-off held it 77 steps, the lookup's
offer was worth more on 50), so on this answer the lookup does the drafting; the head's value is on text the lookup
cannot copy (its check on the Pittsburgh text: 0.59 of first drafts kept).

**Light draft head source (22331f6).** `LightHeadSource` (draft.h) implements the trainer's convention
(`jobs/babel/q36_draft_head.py`): h_t = the post-norm final hidden (rounded to bf16), e_1 = the raw embedding row of
the token chosen from h_t, z_j = z_{j-1} + down_j(silu(up_j([rmsnorm(z) * norm_z_j, rmsnorm(e_j) * norm_e_j]))), logits
through the low-rank pair (vocab . proj . z) or the model's own head (`Model::head_argmax`: the screened greedy head when
built, else the 8-bit head; confidence from the coarse logits there); first index on ties; confidence = max softmax; the
controller gets it binned and calibrated per bin. Shapes come from the file (ND offsets, D, H, R, V); matrices held in
AFF8 (`--draft-light-q8 0`: bf16). Model hooks: `final_hidden(i)` (last step, or verify position i), `embed_row`,
`head_argmax`, implemented for the 35B; Inkling can implement the same three. `dmeng --draft lookup --draft-light
heads_<variant>.safetensors`. Gate `tests/draft_light_check.py TINY_MODEL TINY_RELEASE` (random weights in the
trainer's export format on the tiny 35B): the bf16 head's drafts equal a numpy reference of the convention on all 78
runs (confidence within 7e-7), ids with the head on equal plain decoding for the low-rank and full variants with q8 on
and off; the 8-bit head's first draft equals the bf16 one on 77 / 78 (low rank) and 76 / 78 (full). The trained file
had not landed at the time of writing.

## Reproduce

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
tests/draft_check.sh Q36_TINY INK_TINY INK_RES0 INK_RES8          # tiny models: verify bit-checks, ids with drafts = without
python3 bench/drafts/drafts2.py gates timing server table          # the real-model runs above (resumable)
```

## 7. The 35B's light draft head in the engine (2026-11-23)

Head: `heads_lr_h2048.safetensors` (low rank, hidden 2048, 108 MB in AFF8). One run per cell; counters; others' CPU in the tables. Cells below ran on c54cd7a (back-off, head every draft); the acceptance rows ran the head every step (`--draft-light-backoff 0`). Ids: every exact cell identical to the no-draft run; the coding answer under the release sampler, seeds 1-3, head on = off.

| case | mode | cache | tokens / step | passes | tokens / pass | width | compute / wait ms per token (run) | MB (urgent) per token | plain steps: compute / wait / MB per token | passes: compute / wait / MB per token | pass cost ratio | head ms per call (drafts) | others' CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | exact | 6.6 | 1.00 | 0 | - | - | 37.0 / 9.9 | 64 (19) | 36.9 / 9.9 / 63.9 | - / - / - | - | 10.95 (3.00) | 148 |
| prose | exact | 5.0 | 1.00 | 0 | - | - | 37.4 / 15.7 | 98 (27) | 37.3 / 15.7 / 98.4 | - / - / - | - | 7.73 (3.00) | 209 |
| prose | fast | 6.6 | 1.08 | 21 | 2.10 | 2.5 | 32.1 / 6.5 | 64 (9) | 32.5 / 5.1 / 58.2 | 29.0 / 14.4 / 35.5 | 1.87 | 10.79 (3.00) | 119 |
| prose | fast | 5.0 | 1.01 | 4 | 1.50 | 4.2 | 39.4 / 8.0 | 82 (7) | 38.6 / 7.4 / 77.7 | 65.1 / 36.6 / 69.0 | 2.53 | 10.50 (3.00) | 162 |
| code | exact | 6.6 | 1.00 | 1 | 2.00 | 2.0 | 38.0 / 16.4 | 95 (27) | 37.5 / 16.4 / 92.9 | 35.9 / 26.6 / 73.4 | 1.91 | 34.61 (3.00) | 110 |
| code | exact | 5.0 | 1.00 | 1 | 2.00 | 2.0 | 38.5 / 24.5 | 139 (37) | 38.5 / 24.4 / 136.2 | 30.9 / 50.8 / 97.3 | 1.61 | 7.30 (3.00) | 155 |
| code | fast | 6.6 | 1.03 | 13 | 1.62 | 2.4 | 35.6 / 8.7 | 78 (9) | 35.3 / 7.8 / 73.2 | 34.0 / 20.8 / 30.8 | 1.55 | 17.48 (3.00) | 106 |
| code | fast | 5.0 | 1.01 | 2 | 2.00 | 2.0 | 36.4 / 13.0 | 111 (10) | 36.5 / 12.7 / 108.2 | 25.1 / 34.6 / 40.7 | 1.37 | 7.77 (3.00) | 141 |
| toolrw | exact | 6.6 | 1.53 | 33 | 4.15 | 4.2 | 41.1 / 31.0 | 167 (63) | 44.4 / 29.5 / 156.0 | 36.4 / 32.7 / 80.6 | 3.40 | 38.41 (3.00) | 109 |
| toolrw | exact | 5.0 | 1.01 | 2 | 2.00 | 2.0 | 42.1 / 44.5 | 216 (68) | 42.1 / 43.8 / 213.0 | 32.9 / 92.0 / 121.7 | 1.57 | 13.37 (3.00) | 142 |
| toolrw | fast | 6.6 | 1.66 | 19 | 7.26 | 7.3 | 50.7 / 23.2 | 148 (42) | 59.1 / 20.9 / 137.6 | 40.0 / 26.0 / 60.0 | 4.92 | 38.47 (3.00) | 183 |
| toolrw | fast | 5.0 | 1.01 | 2 | 2.00 | 2.0 | 35.3 / 33.6 | 188 (41) | 35.2 / 33.1 / 185.6 | 28.3 / 74.8 / 85.8 | 1.60 | 14.19 (3.00) | 93 |

| case | mode | cache | draft 1 right | draft 2 right given 1 | draft 3 right given 1, 2 | calls |
|---|---|---|---|---|---|---|
| prose | exact | 6.6 | 0.500 | 0.500 | 0.000 | 4 |
| prose | exact | 5.0 | 0.500 | 0.500 | 0.000 | 4 |
| prose | fast | 6.6 | 0.500 | 0.000 | - | 4 |
| prose | fast | 5.0 | 0.167 | 1.000 | 0.000 | 6 |
| code | exact | 6.6 | 0.750 | 0.333 | 0.000 | 4 |
| code | exact | 5.0 | 0.750 | 0.333 | 0.000 | 4 |
| code | fast | 6.6 | 0.571 | 0.250 | 0.000 | 7 |
| code | fast | 5.0 | 0.750 | 0.333 | 0.000 | 4 |
| toolrw | exact | 6.6 | 0.667 | 0.500 | 1.000 | 3 |
| toolrw | exact | 5.0 | 0.750 | 0.333 | 1.000 | 4 |
| toolrw | fast | 6.6 | 0.667 | 0.500 | 1.000 | 3 |
| toolrw | fast | 5.0 | 0.750 | 0.333 | 1.000 | 4 |
| prose | exact, head every step | 6.6 | 0.536 | 0.397 | 0.300 | 237 |
| code | exact, head every step | 6.6 | 0.533 | 0.370 | 0.300 | 255 |

| run | head ms per call | drafts per call | per draft: embedding row / MLP / output layer (ms) |
|---|---|---|---|
| prose exact 6.6 | 10.95 | 3.00 | 0.143 / 0.832 / 2.675 |
| prose exact 5.0 | 7.73 | 3.00 | 0.101 / 0.301 / 2.174 |
| prose fast 6.6 | 10.79 | 3.00 | 0.044 / 0.496 / 3.055 |
| prose fast 5.0 | 10.50 | 3.00 | 0.083 / 0.518 / 2.898 |
| code exact 6.6 | 34.61 | 3.00 | 0.185 / 3.446 / 7.902 |
| code exact 5.0 | 7.30 | 3.00 | 0.238 / 0.270 / 1.923 |
| code fast 6.6 | 17.48 | 3.00 | 0.151 / 1.657 / 4.016 |
| code fast 5.0 | 7.77 | 3.00 | 0.126 / 0.244 / 2.220 |
| toolrw exact 6.6 | 38.41 | 3.00 | 0.193 / 4.211 / 8.394 |
| toolrw exact 5.0 | 13.37 | 3.00 | 0.078 / 1.680 / 2.697 |
| toolrw fast 6.6 | 38.47 | 3.00 | 0.351 / 3.923 / 8.546 |
| toolrw fast 5.0 | 14.19 | 3.00 | 0.074 / 1.134 / 3.520 |
| prose exact 6.6, head every step | 7.03 | 2.99 | 0.066 / 0.269 / 2.016 |
| code exact 6.6, head every step | 7.11 | 2.99 | 0.084 / 0.270 / 2.024 |

Gated (cb0eb83: vocabulary layer on the pool with a per-chunk epilogue; first draft offered and further drafts computed only above the break-even acceptance from the controller's pass costs and the head's own ms per draft), exact, 6.6 GB, off vs on:

| case | drafts | ms per token (compute + wait) | MB (urgent) per token | tokens / step | passes (kept / drafted) | head calls | head ms per token | head ms per draft (MLP / output layer) | first draft break-even | ids | others' CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prose | off | 37.2 + 10.0 = 47.3 | 64 (19) | 1.00 | 0 (0 / 0) | - | - | - / - | - | - | 179 |
| prose | on | 34.7 + 10.0 = 44.7 (-5.4%) | 64 (19) | 1.00 | 0 (0 / 0) | 4 | 0.05 | 0.511 / 2.836 | 1.00 | identical | 154 |
| code | off | 39.7 + 15.6 = 55.3 | 94 (27) | 1.00 | 0 (0 / 0) | - | - | - / - | - | - | 180 |
| code | on | 37.9 + 16.5 = 54.3 (-1.7%) | 96 (27) | 1.00 | 1 (0 / 1) | 4 | 0.20 | 3.166 / 11.518 | 1.27 | identical | 85 |

The break-even acceptance for a first draft came out at 1.00 (prose) and 1.27 (coding answer): a two-position pass costs 1.6-1.9 plain steps here (table above), so one kept draft cannot pay for it at any confidence and the gated head offers nothing; the on-off differences (-5.4%, -1.7%) are the machine (nothing was drafted). The output layer stays at 2.0 ms per draft with the head every step (2.8 in the gated runs): it was already split over the pool; the cost is the AFF8 row kernel at 256 columns (4 groups per row) over 248,320 rows, not the threading. Verdict: the head ships as an option (`--draft-light F`), not a default.
