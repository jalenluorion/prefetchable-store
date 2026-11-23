# Names inside multi-position verify passes (both models, base M4 16 GB), 2026-11-22

**Status: built (shared core, both models); the exactness gate passes: ids identical to drafting off on pitt / code / edit /
tool for both models, drafts with and without names (16 of 16 runs). With names, on-the-spot reads per verify pass fall
by 52-63% on the 35B (edit, tool, code) and 5-39% on Inkling; the 35B's MB read urgently per accepted token on the edit and
tool turns falls below drafting off's (70 vs 97, 67 vs 96). Timing not run (shared machine): the cells are below.**

## What was built (branch `verify-names`, from `rc`)

1. **The lag-k routers run inside a verify pass** (`Hook::chunk_*`, `Core::chunk_start`; `src/core/moe_chunk.cpp`,
   `src/core/address.cpp`). At the pass's start the routers' first two blocks run for every position (one multi-vector
   pass over each router's rows, inputs [embedding of the position's token id, the previous token's MoE inputs]); then at
   each step s every position's lag rows of target s + lag ride in that step's released-router fork-join, and each
   position's names are issued after the step's own misses, as one-position steps do. Both models call
   `core->chunk_start` from their verify (`qwen3_5_moe/verify.cpp`, `inkling2_loop` forward_chunk when it can roll back).
   The previous-token block is exact for the pass's first position; for position i it is the committed token's (i tokens
   older), since position i - 1's inputs at the target step do not exist yet when the pass reaches step s. Recall (released
   top-8 inside the top-8 names) stays 0.62-0.88 at every position index (table).
   **Lead**: names in a pass wait 13-43 ms for their step (Inkling 13-27 ms, 35B 5-43 ms) against 2.5-8 ms in
   one-position steps of the same runs, because a step of an n-position pass takes several times longer.
2. **Names from the token ids before the pass**: the routers' targets < lag (steps 0 and 1 for the k2 routers) are named
   for every drafted position at the pass's start from the position's token embedding (with the committed token's state).
   The embedding-only cross-token router of `ink_x_main` (`ink_router_xemb_fp16.npz`) is not converted to the engine
   format and was not used; the cross-token router (`--xtok`) forwards the chunk calls to the in-token routers and names
   nothing itself inside a pass (its next token is unknown until the commit). `--early N` (recorded picks of the lookup's
   source span) works alongside, unchanged, and also the oracle hook (`--prefetch oracle`, chunk naming added).
3. **Rejected positions**: names inside a pass carry the deadline of the pass's own step (`Naming::name(.., deadline)`),
   so the readers serve them in the order the pass needs them; every position of a pass runs every step, so no name of a
   rejected position outlives the pass. Early names past the pass that the next step no longer wants (e.g. after a
   rejection) are dropped with their read if it has not started (`Cache::drop_name`; `early_reads_cancelled` in the JSON).
   Names never change the outputs (gate below).
4. **Cost model**: the controller measures every pass's cost (as before) and now gives widths it has not measured the
   measured passes' cost per extra position instead of the prior table's (measured without names), so the named passes'
   cost drives the draft length (`DraftCtl::cost`; learned ratios per run under `draft.cost_ratio`).
5. Counters: `draft.reads` per kind of step (plain / verify: steps, positions, tokens, misses and late picks per step,
   MB urgent per step); `routers.lead` (ms from naming to the step, one-position vs verify), per-position recall in verify
   passes. `--verify-names 0|1` (dmeng and dmserve, default 1).

## Gate: ids identical to drafting off (200 tokens, both models, routers on)

Inkling 6 GB (`ink_r_best` k2, tau 0.5), drafts `--draft both --early 4` (lookup + the draft head); the 35B 5 GB
(`q36_router_full_k2_h512 --m 8`), drafts `--draft lookup --early 4`; prompts pitt and code (`bench/prompts/`), edit and
the tool turn rewriting a file (`results/drafts/prompts/*_edit_ids.txt`, `*_toolrw_ids.txt`). **All 16 drafted runs
(names off and on) give ids identical to drafting off** (`ids = off` column). AC, 100%; others' CPU 11-41% during the
names runs (counts below do not depend on load; tok/s is not reported).

## Counts: names off vs on inside verify passes (load-independent)

Per verify pass: on-the-spot reads (misses) and MB read on the spot, measured between the pass's start and its commit.
Per token: over the whole decode, per generated (= accepted) token. "names refused": names the cache could not give a
slot (all of a layer's free slots pinned).

| model | prompt | drafts | ids = off | tokens / step | verify passes | misses / verify pass | MB urgent / verify pass | misses / token | MB urgent / token | MB total / token | names refused / token | verify names: lead ms | named top-8 recall pos 0 / 1 / 3 / 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ink | pitt | off | - | 1.00 | 0 | 0.0 | 0 | 11.5 | 57 | 170 | 0.0 | 0 | - |
| ink | pitt | plain | yes | 1.08 | 5 | 91.2 | 463 | 12.1 | 60 | 234 | 0.7 | 0 | - |
| ink | pitt | names | yes | 1.08 | 5 | 55.8 | 280 | 11.2 | 56 | 236 | 1.2 | 13 | 0.82 / 0.81 / 0.77 / 0.00 |
| ink | code | off | - | 1.00 | 0 | 0.0 | 0 | 20.0 | 105 | 307 | 0.0 | 0 | - |
| ink | code | plain | yes | 5.69 | 27 | 222.9 | 1188 | 30.9 | 164 | 323 | 13.3 | 0 | - |
| ink | code | names | yes | 5.69 | 27 | 208.6 | 1108 | 29.0 | 154 | 324 | 36.0 | 27 | 0.76 / 0.75 / 0.73 / 0.74 |
| ink | edit | off | - | 1.00 | 0 | 0.0 | 0 | 16.9 | 88 | 304 | 0.0 | 0 | - |
| ink | edit | plain | yes | 1.78 | 40 | 122.6 | 648 | 29.3 | 154 | 438 | 2.8 | 0 | - |
| ink | edit | names | yes | 1.79 | 39 | 91.4 | 486 | 22.8 | 121 | 444 | 12.2 | 17 | 0.77 / 0.75 / 0.74 / 0.00 |
| ink | tool | off | - | 1.00 | 0 | 0.0 | 0 | 19.8 | 101 | 319 | 0.0 | 0 | - |
| ink | tool | plain | yes | 1.55 | 21 | 113.0 | 567 | 30.8 | 155 | 524 | 3.2 | 0 | - |
| ink | tool | names | yes | 1.57 | 17 | 107.3 | 536 | 26.3 | 132 | 517 | 15.0 | 22 | 0.76 / 0.74 / 0.73 / 0.00 |
| q36 | pitt | off | - | 1.00 | 0 | 0.0 | 0 | 17.2 | 30 | 131 | 0.0 | 0 | - |
| q36 | pitt | plain | yes | 1.00 | 1 | 62.0 | 110 | 16.5 | 29 | 138 | 0.0 | 0 | - |
| q36 | pitt | names | yes | 1.00 | 1 | 34.0 | 60 | 16.3 | 29 | 138 | 0.0 | 5 | 0.84 / 0.88 / 0.83 / 0.00 |
| q36 | code | off | - | 1.00 | 0 | 0.0 | 0 | 29.3 | 52 | 272 | 0.0 | 0 | - |
| q36 | code | plain | yes | 1.09 | 21 | 111.1 | 197 | 30.7 | 54 | 298 | 0.0 | 0 | - |
| q36 | code | names | yes | 1.09 | 18 | 53.1 | 94 | 24.4 | 43 | 315 | 0.0 | 12 | 0.81 / 0.77 / 0.76 / 0.00 |
| q36 | edit | off | - | 1.00 | 0 | 0.0 | 0 | 55.0 | 97 | 357 | 0.0 | 0 | - |
| q36 | edit | plain | yes | 7.11 | 26 | 806.4 | 1427 | 105.6 | 187 | 326 | 0.0 | 0 | - |
| q36 | edit | names | yes | 7.11 | 26 | 299.9 | 531 | 39.4 | 70 | 395 | 0.0 | 43 | 0.68 / 0.63 / 0.62 / 0.62 |
| q36 | tool | off | - | 1.00 | 0 | 0.0 | 0 | 54.1 | 96 | 409 | 0.0 | 0 | - |
| q36 | tool | plain | yes | 3.49 | 33 | 505.2 | 894 | 87.1 | 154 | 356 | 0.0 | 0 | - |
| q36 | tool | names | yes | 3.43 | 33 | 207.7 | 367 | 37.9 | 67 | 423 | 0.0 | 30 | 0.69 / 0.65 / 0.62 / 0.00 |

Read:
- **The 35B**: names inside the pass remove 52-63% of its on-the-spot reads (edit 806 -> 300 per pass, tool 505 -> 208,
  code 111 -> 53), and MB read urgently per accepted token falls from 187 to 70 (edit) and 154 to 67 (tool), below drafting
  off (97, 96): drafting there no longer pays for its experts on the spot. The price is reads of names that were not
  picked: MB read in total per token rises 326 -> 395 (edit), 356 -> 423 (tool), 298 -> 315 (code). No name was refused.
- **Inkling**: smaller effect (pitt 91 -> 56 per pass, edit 123 -> 91, tool 113 -> 107, code 223 -> 209). The cache refuses
  many names (code 13 -> 36 per token): in a pass, the union, the looped layers' holds of earlier passes and the
  `--early` names (2,161 per step on code: 12 positions x 48 applications x 8 picks of the copied span) pin most of each
  layer's ~53 slots at 6 GB, so the routers' names get no slot. On code the early names already cover the copied positions.
  Whether `--early` should stay on beside the routers' names on Inkling is the one cell left open (the command below,
  `namesx`: names on, early off), cancelled to free the machine.
- Lead: named reads get 13-43 ms per step in a pass against 2.5-8 ms in one-position steps.

## Timing cells for the clean campaign (not run here)

Off / drafts without names / drafts with names; Inkling at 6 GB, the 35B at 5 GB; the coding answer (300 tokens), the
tool call writing a new file (`toolnew`, up to 800) and the tool call rewriting a file the conversation read (`toolrw`, up
to 800); tool cells with the request's skeleton fragments; two runs per cell, interleaved (r1: off, plain, names; r2:
names, plain, off); every run through the guard. `$INK`, `$Q36` = the model folders; from the repo root after
`cmake -S . -B build && cmake --build build -j`:

```
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file bench/prompts/inkling2_loop/code.ids --n 300 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --json ink_code_off_rN.json --out-ids ink_code_off_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file bench/prompts/inkling2_loop/code.ids --n 300 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --draft both --early 4 --verify-names 0 --json ink_code_plain_rN.json --out-ids ink_code_plain_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file bench/prompts/inkling2_loop/code.ids --n 300 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --draft both --early 4 --verify-names 1 --json ink_code_names_rN.json --out-ids ink_code_names_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/ink_toolnew_ids.txt --n 800 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --json ink_toolnew_off_rN.json --out-ids ink_toolnew_off_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/ink_toolnew_ids.txt --n 800 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --draft both --early 4 --verify-names 0 --skeleton results/drafts/prompts/ink_toolnew_skel.txt --json ink_toolnew_plain_rN.json --out-ids ink_toolnew_plain_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/ink_toolnew_ids.txt --n 800 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --draft both --early 4 --verify-names 1 --skeleton results/drafts/prompts/ink_toolnew_skel.txt --json ink_toolnew_names_rN.json --out-ids ink_toolnew_names_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/ink_toolrw_ids.txt --n 800 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --json ink_toolrw_off_rN.json --out-ids ink_toolrw_off_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/ink_toolrw_ids.txt --n 800 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --draft both --early 4 --verify-names 0 --skeleton results/drafts/prompts/ink_toolrw_skel.txt --json ink_toolrw_plain_rN.json --out-ids ink_toolrw_plain_rN.ids
build/dmeng --model $INK --cache-gb 6 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/ink_toolrw_ids.txt --n 800 --prefill-chunk 64 --stop-eos 200006 --prefetch router --routers $INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5 --draft both --early 4 --verify-names 1 --skeleton results/drafts/prompts/ink_toolrw_skel.txt --json ink_toolrw_names_rN.json --out-ids ink_toolrw_names_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file bench/prompts/qwen3_5_moe/code.ids --n 300 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --json q36_code_off_rN.json --out-ids q36_code_off_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file bench/prompts/qwen3_5_moe/code.ids --n 300 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --draft lookup --early 4 --verify-names 0 --json q36_code_plain_rN.json --out-ids q36_code_plain_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file bench/prompts/qwen3_5_moe/code.ids --n 300 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --draft lookup --early 4 --verify-names 1 --json q36_code_names_rN.json --out-ids q36_code_names_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/q36_toolnew_ids.txt --n 800 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --json q36_toolnew_off_rN.json --out-ids q36_toolnew_off_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/q36_toolnew_ids.txt --n 800 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --draft lookup --early 4 --verify-names 0 --skeleton results/drafts/prompts/q36_toolnew_skel.txt --json q36_toolnew_plain_rN.json --out-ids q36_toolnew_plain_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/q36_toolnew_ids.txt --n 800 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --draft lookup --early 4 --verify-names 1 --skeleton results/drafts/prompts/q36_toolnew_skel.txt --json q36_toolnew_names_rN.json --out-ids q36_toolnew_names_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/q36_toolrw_ids.txt --n 800 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --json q36_toolrw_off_rN.json --out-ids q36_toolrw_off_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/q36_toolrw_ids.txt --n 800 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --draft lookup --early 4 --verify-names 0 --skeleton results/drafts/prompts/q36_toolrw_skel.txt --json q36_toolrw_plain_rN.json --out-ids q36_toolrw_plain_rN.ids
build/dmeng --model $Q36 --cache-gb 5 --mem mlock --ctx 16384 --temp 0 --prompt-file results/drafts/prompts/q36_toolrw_ids.txt --n 800 --prefill-chunk 512 --stop-eos 248046 --prefetch router --routers $Q36/routers/q36_router_full_k2_h512.dmr --m 8 --draft lookup --early 4 --verify-names 1 --skeleton results/drafts/prompts/q36_toolrw_skel.txt --json q36_toolrw_names_rN.json --out-ids q36_toolrw_names_rN.ids
```

The open Inkling cell (names on, early off): the `names` line with `--early 4` removed (`bench/verify_names/run.py
noearly` runs it on code and tool for both models). Read per run: `tok_s`, `draft.reads.verify` (misses / MB urgent per
pass), `mb_read_urgent_per_token`, `mb_read_per_token`, `draft.tokens_per_step`, `draft.cost_ratio`, `routers.lead`,
`footprint_peak_gb`; `cmp` the ids of each drafted run against its off run.

## Limits

- Position i > 0 of a pass names from the committed token's state (stale by i tokens); split router files (embedding /
  previous-token columns apart) are not named inside a pass.
- Inkling at 6 GB: names refused for lack of unpinned slots (above); the 35B at 5 GB had none refused.
- The cross-token router names nothing inside a pass; the embedding-only router is not converted.
- Counts only; wall-clock effect to be measured in the clean campaign.

## Commands

```
python3 bench/verify_names/run.py gates    # 24 runs (2 models x 4 prompts x off / plain / names), resumable
python3 bench/verify_names/run.py table    # the table above
python3 bench/verify_names/run.py timing   # prints the timing cells
python3 bench/verify_names/run.py noearly  # the open cell (names on, early off; code and tool, both models)
```
