# Inkling 2 Mini's own draft head in the engine (base M4, 16 GB), 2026-11-22

**Status: built; both correctness gates pass; timing not run (stopped: the Mac was swapping); the
timing cells are written out below to run in one clean campaign on the merged tree.** The C++ head reproduces the MLX
reference on recorded features (first drafts equal on 99.5% / 100% of 399 positions of two texts, largest draft-1 logit
difference 0.19), decoding with it is token-identical to one-position decoding on the four prompts, and it adds 0.64 GB
of weights (0.73 GB with its reserved state at 16K context).

## What was built

- **The draft head forward pass in C++** (`src/models/inkling2_loop/ink_draft.{h,cpp}`): the released `mtp.safetensors`
  ("DSpark" block drafter, sglang PR 51926), weights bf16 as released. Context K/V of every committed position
  (`fuse_weight` mix of the four features, `dflare_norm`, `k/v_proj_ctx`, `k_norm_ctx`, RoPE theta 1e6), the 8-slot block
  after the newest token (slot 0 its normed embedding, slots 1-7 `mask_embed`; 3 layers, no mask inside the block, layers 0
  and 2 over the last 1,023 context positions), the Markov RNN (state, `markov_w1[previous token]`, h_j, f_rnn), the
  target's own output layer, softcap 30. Every op's output is rounded to bf16 where the MLX reference's is.
  Products: a bf16 x bf16 BFMMLA kernel (2 rows x 2 inputs x 4 per instruction, fp32 sums, inputs packed in slot pairs),
  threaded over row chunks. The output layer in decoding: head(h_j) (the engine's own head, as the target uses it) +
  F latent_j + c, with F = unembed . hidden_out.W (200,058 x 256) and c = unembed . hidden_out.b folded once from the
  released bf16 tensors (fp32 sums, stored bf16 in `$MODEL/.dmeng/ink_mtp_fold_bf16.bin`, 103 MB); head(h_1) alone, then
  head(h_2..h_7) in one batched pass if draft 1 passes the threshold; one 0.1 GB pass per draft instead of a 0.46 GB one.
  `--draft-exact 1` applies the output layer exactly as the reference does (bf16 z_j through the released bf16
  unembedding), for the port check only.
- **Feature taps**: the hidden state entering applications 5, 23, 41 and the post-final-norm hidden state, taken inside
  the target's own one-position step, its verify chunk and its batched prefill chunk (the final norm, without the head,
  is computed for every position while the head is loaded; no extra pass). Features of committed positions are queued
  and projected to context K/V in one batch the next time the head runs (so a backed-off head costs nothing per token);
  rolled-back verify positions are dropped at `commit`.
- **A draft source in the controller** (`tools/dmeng.cpp`): `--draft head|both` (`lookup`, `fixed` unchanged).
  `--draft-thr P` (default 0.9, below), `--head-backoff 32,256`. Per step: the lookup offer and its expected tokens per
  step cost (unchanged); the head runs unless backed off or unless the lookup's offer is worth more than the head's
  running value (expected tokens per cost including the head's own measured time); then the controller picks the lookup
  draft, the head draft (drafts used while their top-1 probability >= P; acceptance per probability bin learned online,
  priors from the port records) or a plain step, whichever has the most expected tokens per step cost. A head try is lost
  when no draft passes the threshold, its draft is not chosen, or none of it is kept; then the head is not run for 32
  tokens, doubling to 256 on further losses, reset on a win.
- **Checks**: `dmeng --draft-check TAPS` (the head on recorded features, no decoding), the MLX reference
  `tests/inkling2_loop/draft_ref_mlx_run.py` (sglang PR 51926's arithmetic, unchanged), `draft_compare.py`;
  `dmeng --draft-bench R` (the block's products on synthetic weights). Campaign: `bench/ink_draft/run.py`.

**Memory**: weights 0.533 GB resident (3 layers + RNN + norms, bf16; `markov_w1`, 103 MB, stays in the file: one 512-byte
row read per draft) + the fold 0.103 GB = **0.636 GB**; state: context K/V in bf16, 4 KB per position for the global
layer (reserved for the context, touched as it fills) and two 8.4 MB rings for the sliding layers, plus 36 KB of queued
features per position not yet projected. In a decoding run at 16K context the model reports 0.73 GB for the head
(weights + reserved state); measured footprint 8.45 GB at 6 GB of slots with the k2 routers and the head.

## Gate 1: the C++ head equals the MLX reference on the same features

Features: the engine's recorded taps of the Pittsburgh generation (prompt + 400 greedy tokens) and of the coding chat
answer (462-token prompt + 400 greedy tokens) (`FEATS/taps_*_seq.bin`); every committed position after
the prompt (399 per text), all 7 drafts per position without a threshold. The reference script reproduces mtp_accept.py's
own saved draft-1 rows exactly (399 / 399 on both texts, probability difference 0.0). Draft j is compared where drafts
1..j-1 agreed (each side conditions on its own previous drafts). Logits: draft 1's, over the full vocabulary, first 200
positions.

| text | output layer | draft 1 equal | drafts 2..7 equal (given the previous equal) | all 7 equal | largest / mean draft-1 logit difference | disagreements |
|---|---|---|---|---|---|---|
| Pittsburgh | faithful (`--draft-exact 1`) | **0.995** (397 / 399) | 0.995, 0.997, 1.000, 0.995, 0.997, 0.995 | 0.975 | **0.190** / 0.019 | 2 (positions 246, 355) |
| Pittsburgh | decoding (8-bit head + fold) | 0.992 (396 / 399) | 0.990, 0.990, 0.992, 0.992, 0.987, 0.989 | 0.935 | 0.277 / 0.029 | 3 |
| coding answer | faithful | **1.000** (399 / 399) | 1.000, 1.000, 0.995, 1.000, 0.998, 1.000 | 0.992 | **0.190** / 0.015 | 0 |
| coding answer | decoding | 1.000 | 1.000, 1.000, 0.998, 1.000, 0.995, 1.000 | 0.992 | 0.304 / 0.028 | 0 |

The reference rounds every logit to bf16 (spacing 0.125 between 16 and 32), so a 0.19 difference is under two bf16 steps;
the two faithful-mode disagreements fall outside the 200 positions with dumped logits, so their margins are not known.
Draft-1 probability differs by at most 0.022 where the drafts agree.

## Threshold (from the port records, decoding configuration)

Share of drafts kept by their own top-1 probability (draft j counted when 1..j-1 were kept; greedy sequences):

| probability | Pittsburgh drafts | kept | coding drafts | kept |
|---|---|---|---|---|
| < 0.3 | 68 | 0.16 | 3 | 0.33 |
| 0.3-0.5 | 136 | 0.31 | 25 | 0.20 |
| 0.5-0.6 | 85 | 0.46 | 15 | 0.27 |
| 0.6-0.7 | 65 | 0.48 | 19 | 0.58 |
| 0.7-0.8 | 76 | 0.41 | 13 | 0.62 |
| 0.8-0.9 | 96 | 0.51 | 20 | 0.80 |
| 0.9-0.95 | 73 | 0.60 | 15 | 0.80 |
| 0.95-0.98 | 62 | 0.74 | 30 | 0.83 |
| 0.98-0.995 | 68 | 0.78 | 71 | 0.99 |
| >= 0.995 | 143 | 0.91 | 2365 | 0.997 |

On free text the head is overconfident (0.8-0.9 kept half the time). Projected tokens per one-position step time for
"verify every draft at or above the threshold", with the measured 6 GB verify costs and 30 ms of head per step
(`bench/ink_draft/thr.py`):

| threshold | 0.3 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|---|
| Pittsburgh | 0.614 | 0.783 | 0.855 | 0.909 | 0.950 | **0.957** | 0.953 |
| coding answer | 1.626 | 1.671 | 1.690 | 1.696 | 1.700 | **1.701** | 1.702 |

So the default is 0.9. On Pittsburgh no threshold reaches 1.0 if the head ran every step: free text has to be carried by
the back-off and the controller's choice, not the threshold.

## Kernel (`dmeng --draft-bench 20`, synthetic weights of the block's shapes, 0.50 GB, 8 threads; battery 100%, others' CPU 53%)

| inputs | 1 | 2 | 4 | 8 (the block) |
|---|---|---|---|---|
| ms for the 3 layers' products | 6.07 | 6.54 | 8.53 | **12.71** |
| weight GB/s | 81.7 | 75.7 | 58.1 | 39.0 |

One input runs at the memory's speed; the 8-slot block is bound by the BFMMLA arithmetic (~316 GFLOP/s), 2.1x one input.
In decoding runs the head costs 15-20 ms for the block and 20-40 ms for the output layer and RNN per run (busy machine).

## Gate 2: ids identical to one-position decoding

`python3 bench/ink_draft/run.py exact`: each prompt decoded 200 tokens greedy at 6 GB of slots with the lossless k2
routers (`--prefetch router --routers ink_r_fast/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5`) and `--prefill-chunk 64`,
drafting off, then `--draft head` and `--draft both` (the final controller of this commit); `.ids` files compared.

| prompt | head: ids = off | both: ids = off | head: tokens / step, head runs / steps used / drafts kept / lost tries | both: tokens / step, lookup steps, head steps |
|---|---|---|---|---|
| pitt (prose) | **identical** (200) | **identical** | 1.03; 7 / 4 / 5 / 5 (187 tokens backed off) | 1.08; 3, 4 |
| code (coding chat) | **identical** | **identical** | 1.75; 18 / 18 / 85 / 2 | 4.63; 33, 3 |
| edit (edit a file, print it whole) | **identical** | **identical** | 1.33; 14 / 11 / 49 / 4 | 1.75; 21, 3 |
| tool (tool call rewriting a file in the conversation) | **identical** | **identical** | 2.24; 25 / 24 / 110 / 2 | 2.55; 17, 18 |

Also identical: the 400-token runs with the first controller (pitt head, pitt both, code head; `runs/v1_controller/`).
Every verify step rolls back the rejected positions' features from the head's queue at `commit`, so these runs also
check the taps' bookkeeping. Power: AC, charging 82-100%; others' CPU 17-173% (tok/s of these runs is not quotable).

## Timing: not run here (the Mac was swapping); the cells to run

One campaign on a quiet machine, every run through the guard, two runs per cell, interleaved (t1: off, lookup, head, both;
t2: the reverse), lossless k2 routers on in every cell as in the existing campaign, the prompt through the batched prefill
(`--prefill-chunk 64`, exact). `python3 bench/ink_draft/run.py timing` runs exactly these and `... table` prints the table
(tok/s each run, ratio to off, tokens / step, wait ms and MB read per generated token, footprint, power at start, others'
CPU). Per cell, by hand:

```
MODEL=<model folder>   # $MODEL/routers/ink_r_fast/ink_router_full_k2_h512_b8.dmr, $MODEL/mtp.safetensors
R="--prefetch router --routers $MODEL/routers/ink_r_fast/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5"
P=bench/prompts/inkling2_loop/pitt.ids                 N=400   # prose (raw text)
P=bench/prompts/inkling2_loop/code.ids                 N=400   # coding chat
P=results/ink_draft/prompts/edit_ids.txt               N=400   # edit a file given in the prompt, print it whole
P=results/ink_draft/prompts/tool_ids.txt               N=640   # tool call that REWRITES a file already in the conversation (read_file result -> write_file)
P=results/ink_draft/prompts/newfile_ids.txt            N=640   # tool call that writes a NEW file (~600 tokens; nothing to copy)
G=6 or 7.8
off:     build/dmeng --model $MODEL --cache-gb $G --mem mlock --prompt-file $P --n $N --prefill-chunk 64 $R --json OUT.json --out-ids OUT.ids --steps-out OUT.steps
lookup:  ... same ... --draft lookup
head:    ... same ... --draft head      (at the 7.8 budget: --cache-gb 7.05)
both:    ... same ... --draft both      (at the 7.8 budget: --cache-gb 7.05)
```

Why 7.05 at the 7.8 budget: the head adds 0.73 GB to the process (weights 0.636 + reserved state), and the footprint at
7.8 GB of slots with routers is ~9.5 GB already (9.33-9.38 GB measured by the dmink campaign without routers, +0.23 GB
routers), so head cells at 7.8 GB of slots would be ~10.2 GB, over the 9.5 GB rule; with the slots cut by the head's
memory the process holds the same total as drafting off. Check after the runs: `footprint_peak_gb` <= 9.5, and the
`draft.head` block of each JSON (runs, offers, steps, accepted, lost, skipped_backoff, skipped_lookup, ms_per_run,
ms_per_token, calibration) plus `model.draft_head` (projection / block / output-layer ms per run). The quotable rule for
prose (pitt): head and both within 1% of off.

### What the two gate-era runs showed (single runs on a busy, swapping machine; NOT quotable, the reason for one change)

| run (6 GB, k2 routers, first controller) | tok/s | tokens / step | wait ms / token | MB read / token | footprint GB | head |
|---|---|---|---|---|---|---|
| pitt off, 400 | 10.79 | 1.00 | 29.4 | 147 | 7.62 | - |
| pitt head, 400 | 11.75 | 1.01 | 30.1 | 146 | 8.28 | 12 runs, 382 tokens backed off, 0.96 ms / token |
| pitt both, 400 | 10.84 | 1.07 | 30.2 | 150 | 8.32 | 18 runs, 1.87 ms / token |
| code off, 400 | 7.27 | 1.00 | 54.7 | 272 | 7.81 | - |
| code head, 400 | 6.91 | **3.47** | 48.6 | 236 | 8.51 | 51 runs, 78 ms / run, 9.9 ms / token |

(battery 95-81% discharging; others' CPU 27-101%.) On the coding answer the head kept 284 of 319 drafts and 3.47 tokens per
step, but the run was not faster: the 24 eight-position steps took 988 ms on average (7.2x the run's one-position step,
against 4.3x in the warm step bench of RESULTS-STAGE4-INKLING.md), per generated token the expert compute rose (40.9 vs
34.8 ms) while the wait fell only 6 ms, and the routers' prefetch works only inside one-position steps (5.7 vs 31.6
prefetches per token; the verify chunk reads its experts on the spot: 206 vs 108 MB urgent per token). The first
controller counted a head try as won whenever something was kept, so it kept drafting at break-even plus the head's
10 ms per token. Changed (this commit): a try is also lost when the step plus the head took longer per generated token
than a plain step (measured), which backs the head off there. The timing campaign above measures the changed controller.

## What did not work / limits

- **Verify steps are where the time goes, not the head.** On the coding answer at 6 GB the first controller kept 3.47
  tokens per step from the head and still did not run faster (single runs, busy machine): eight-position verify steps cost
  ~7x a one-position step in the run, the experts' union is read on the spot because the routers' prefetch runs only in
  one-position steps (`Core::moe_chunk` calls no hook), and the multi-position kernels lose bandwidth (RESULTS-STAGE4-INKLING
  kernel table). Naming the verify chunk's experts early (routers or the lookup's early addressing inside chunks) is the
  lever the head cannot replace; not done here.
- The head itself: 0.5 GB of bf16 weights per block pass; the 8-slot block is arithmetic-bound at 12.7 ms on synthetic
  weights (BFMMLA, ~316 GFLOP/s); in runs 15-31 ms for the block and 13-45 ms for the output layer and RNN (one 0.46 GB
  head pass for draft 1, one batched pass for drafts 2-7, one 0.1 GB fold pass per draft). An SME2 (BFMOPA) block kernel
  and a GPU path were not tried.
- The head's probabilities are overconfident on free text (kept 51% at 0.8-0.9 on Pittsburgh); prose relies on the
  back-off (in the gate runs the head ran 7 times in 200 tokens on pitt).
- At the 7.8 GB budget the head does not fit under the 9.5 GB footprint rule with the slots unchanged; its cells take
  the slots down by the head's 0.73 GB.
- One sequence only (the dmeng loop): the serving layer (lm.h, prefix cache) does not save or restore the head's context
  K/V; a restored state would start the head with no context.
- Temperature 0 only (the verify rule is greedy equality).
- Not verified against sglang's own output (no GPU run); the reference is the MLX port of its arithmetic.

## Commands

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j --target dmeng
python3 bench/ink_draft/run.py port     # gate 1 (MLX reference + C++ faithful / decoding, compare): results/ink_draft/runs/port_*
python3 bench/ink_draft/run.py exact    # gate 2: off / head / both, 200 tokens (EXACT_N), pitt, code, edit, tool, 6 GB, k2 routers
python3 bench/ink_draft/run.py timing   # off / lookup / head / both x 5 prompts x 6 and 7.8 GB, two interleaved runs each
python3 bench/ink_draft/run.py table    # the timing table
python3 bench/ink_draft/run.py bench    # the kernel bench
$INK_PY bench/ink_draft/make_prompts.py   # the new-file tool-call prompt (results/ink_draft/prompts/newfile_ids.txt)
build/dmeng --model $MODEL --cache-gb 6 --prompt-file IDS --n 400 --prefill-chunk 64 --draft both \
  --prefetch router --routers $MODEL/routers/ink_r_fast/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5
```
