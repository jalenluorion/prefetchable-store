# Draft and verify for both models, in dmeng and in the server (branch drafts-q36)

Base M4 16 GB, the model larger than memory. Greedy only: a drafted token is kept only while it equals the model's own
greedy token and the rest of the pass is rolled back, so the ids are one-position decoding's whatever the drafts are.
With sampling on (OpenCode's default requests), drafting stays off.

## What was built

1. **Multi-position verify for Qwen3.6-35B-A3B** (`src/models/qwen3_5_moe/verify.cpp`; `Model::verify / commit / logits_at`).
   Up to 8 positions in one pass, every position's logits kept. Exact by construction, not by a copy of the arithmetic:
   - resident matrices (projections, o projections, head) through `run_rows_n` (multi-vector and, from 6 positions, the
     packed i8mm gemm: both bit-identical per position to the one-vector `mv` decode calls), bf16 rows through `dot_bf16`;
   - the Gated DeltaNet conv and heads: decode's own `phase_conv` / `phase_lin_heads`, once per position in order;
   - full attention: `attn_full_core`, now one non-inlined body that decode calls too (KV write, attention, gate);
   - the MoE block: the shared `Core::moe_chunk` (its combination now written as the explicit fma chain of
     `Core::combine`, so it does not depend on contraction).
   Rollback: each DeltaNet layer's conv and recurrent state is copied at the step's start (63 MB per pass, pool-parallel)
   and each position's projections are kept; `commit(a < n)` copies the start state back and runs the two phases again on
   positions 0..a-1 (same inputs, same compiled code, same bits). KV is append-only: only the position goes back, and the
   cache's clock.
2. **Shared draft component** (`include/dmeng/draft.h`, `src/core/draft.cpp`), used by `dmeng` and `dmserve`, both models:
   - `LookupSource`: longest earlier match of the last 2-16 tokens over the whole session (prompt + everything generated;
     in the server the live sequence across requests, plus an archive of each request's generated tokens, since the chat
     template drops earlier reasoning), bigram index, incremental, most recent occurrence wins ties; buckets by match length.
   - `SkeletonSource`: tool-call format fragments; the draft is what every best-matching fragment position agrees on next.
   - `DraftCtl`: acceptance per (source, bucket, draft index) learned online (prior = the source's confidence), pass cost
     per width measured against one position (prior = the measured table; a width's first two passes ignored), draft length
     = argmax expected tokens / cost, 0 = plain step; a probe after 48 declined offers.
   - `Drafter::step(target, history, vocab, budget, stops, out)`: one decode step on a `DraftTarget` (dmeng: Model + State;
     dmserve: the prefix cache's live sequence), never accepting past a stop token or the token budget.
3. **Tool-call skeleton drafts in dmserve.** The front end renders, per request with tools, the format's fragments through
   the model's format module (`chat_format.tool_skeletons`: per tool and parameter order, values a placeholder) and sends
   their token ids on extra protocol lines (R line field 10, backward compatible). They go into the normal verify as one
   more draft source: drafts only, never forced.
4. **Early names (`--early N`) with trained routers on, both models.** The Drafter names the released picks recorded at the
   copied source positions for the step's positions and N past them (Core::pick_log; the 35B's batched prefill now writes
   it too, so prompt positions are sources). New `Hook::observe` + `Core::chunk_commit`: after a verify the routers see the
   last kept position's MoE inputs as their previous-token features (Routers, Planner, Xtok). Names only schedule reads.
5. `dmeng --verify-check N`: N single steps, then the same positions through verify passes of 2-8 positions with every
   third pass's last draft wrong (rolled back); every kept logit and the final state compared bit for bit.

## Draft-source interface (for the Inkling draft head on ink-draft)

```
struct Draft { std::vector<int32_t> toks; std::vector<float> conf; int bucket; long src; };
class DraftSource {
  virtual const char* name() const;
  virtual int buckets() const;                       // the source's own classes of proposal (e.g. confidence bins)
  virtual bool propose(const std::vector<int32_t>& hist, int max, Draft& d);  // hist: the whole sequence, last = token fed next
  virtual void feedback(int offered, int accepted);  // the outcome of its last proposal that ran
};
Drafter dr; dr.extra.push_back(&head); dr.init(family, core);   // the controller chooses among all sources per step
```
conf[k] is the source's estimate that toks[k] is right given toks[0..k) are; the controller uses it as the prior of a
(source, bucket, index) cell it has not seen and learns the cell from there. A head that needs model features reads them
from the Model (`taps()`) inside propose.

## Gates (all passed; real-model runs through engine/bench/guard.py, on AC, others' CPU 23-59%)

| gate | result |
|---|---|
| tiny models, `tests/draft_check.sh` (both models; i16 / i8 / f16 / f32, split-K and multi-chunk attention) | verify-check: every kept logit and the final state bit-identical through passes of 2-8 positions with rollbacks; ids with `--draft fixed` / `lookup` = plain decoding in dmeng and in the server worker: PASSED |
| tiny models, decode logits after the refactor vs the unified-wip build (teacher-forced, both models, all precisions, short and split-K attention) | bit-identical |
| 35B, `dmeng --verify-check 200` (coding prompt, 5 GB, routers k2 on) | 200 positions, 44 passes, 14 rollbacks: 0 logits differ, 0 state bytes, 0 KV bytes |
| Inkling, `dmeng --verify-check 200` (coding prompt, 6 GB, routers k2 on) | 200 positions, 44 passes, 14 rollbacks: 0 differ |
| 35B batched prefill check (`dmserve --check`, 1,100 tokens) after the attention refactor | 0 logit bits, 0 state bytes, 0 KV bytes differ; greedy identical |
| server worker, the rewrite tool-call turn with its tool-call fragments, drafting off vs on (`--draft lookup --early 4`, routers on), 35B 5 GB | 200 / 200 tokens, ids identical; 4.52 tokens per step; lookup 23 verifies (140 accepted), skeleton 4 (15 accepted) |
| same, Inkling 6 GB | 108 / 108 tokens (stop), ids identical; 1.70 tokens per step; lookup 8 (23 accepted), skeleton 4 (21 accepted) |
| 35B fixed-draft stress (`dmeng --draft fixed`, the whole draft every step), the rewrite turn, 200 tokens | ids identical to the server's off run; 28 verifies, 195 drafted, 157 accepted, 27 eight-position passes |

The ids-identical gates per draft source on prose / coding / edit / tool turns for both models and the OpenCode replay
gate run in the integration tree's campaign (rc), not here (split of the work). Not quotable as speed: no timing was taken.

## What remains

- Timing (commands below), including whether the controller stays within 1% on prose; nothing here says drafting pays.
- The 35B's step cost prior is Inkling's table until the controller measures its own (it does online; first two passes per
  width ignored).
- Sampling: drafting stays off (needs the speculative accept / resample rule).

## Timing: not run (2026-11-22: the Mac was swapping under six queues)

The cells, exactly as `bench/drafts/campaign.py timing` runs them (each through `engine/bench/guard.py`, power / QUIET /
PRIORITY rules applied before each run, power state and others' CPU recorded in `runs/NAME.meta.json`; resumable):
the 35B at 5 and 6.6 GB, Inkling at 6 and 7.8 GB; cases prose (200 tokens), coding answer (300), a tool call writing a new
file of ~600 tokens (`toolnew`, up to 800), a tool call rewriting a file the conversation read (`toolrw`, up to 800), a short
tool call (`toolshort`, up to 120); off / on interleaved, two runs per cell (order per cell: off r1, on r1, off r2, on r2).
Lossless routers on in every run (35B: `q36_router_full_k2_h512.dmr --m 8`; Inkling: `ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5`).
"on" = `--draft lookup --early 4`, plus `--skeleton` (the request's tool-call fragments) for the tool cases.

```
# from the repo root, after building build/ (cmake -S . -B build && cmake --build build -j)
L=../../../inkling/.venv/bin/python; F=../../.venv/bin/python   # Pythons with tokenizers (+ jinja2 for the 35B)
$F bench/drafts/make_prompts.py q36 ENGINE_DIR Q36_MODEL_DIR      # results/drafts/prompts/* (committed; rerun only to regenerate)
$L bench/drafts/make_prompts.py ink ENGINE_DIR INK_MODEL_DIR
python3 bench/drafts/campaign.py timing table                   # 80 runs; table: results/drafts/table.md
GBS="q36:5" MODELS=q36 CASES=toolrw REPS=1 python3 bench/drafts/campaign.py timing   # one cell
```

One cell written out (35B, 5 GB, the rewrite; `M` = the model folder, `P` = results/drafts/prompts):

```
build/dmeng --model M --prefill-chunk 512 --prompt-file P/q36_toolrw_ids.txt --n 800 --cache-gb 5 --ctx 16384 --temp 0 \
  --mem mlock --stop-eos 248046 --prefetch router --routers M/routers/q36_router_full_k2_h512.dmr --m 8 --json off.json --out-ids off.ids
build/dmeng ...same... --draft lookup --early 4 --skeleton P/q36_toolrw_skel.txt --json on.json --out-ids on.ids
# Inkling: --model INK --resident engine/build_ink/resident_real_b8.bin --prefill-chunk 256 --stop-eos 200006
#          --prefetch router --routers INK/routers/ink_r_best/ink_router_full_k2_h512_b8.dmr --rt-tau 0.5
```

What to read from each JSON: `tok_s`; `draft.tokens_per_step`, `draft.steps_by_positions`, `draft.verifies_by_source`,
`draft.accepted_by_source`, the learned `draft.cost_ratio` and `draft.one_position_ms`; `wait_ms_per_token` and
`mb_read_per_token` (per generated token, i.e. per accepted token); `footprint_peak_gb`; the 35B's `model.verify`
(state-copy and rollback ms). Gate on every pair: `cmp off.ids on.ids`. The table script prints all of these per run.
The controller's prose budget (lose no more than 1%) is the `prose` cell's on / off ratio.

The server under drafting (greedy requests): `python/serve/dmserve.py ... -- --draft lookup --early 4 [--draft-max 7]`;
the worker-level comparison is `campaign.py server` (bench/drafts/worker_drive.py: the recorded OpenCode requests'
tokens with their tool fragments, off vs on). Stock OpenCode sends no temperature, so its requests sample and draft
nothing; only greedy requests (temperature 0) draft.
