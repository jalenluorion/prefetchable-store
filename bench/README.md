# Benchmarks

`bench/run.py` runs one campaign for one model: a set of cells, crossed with cache sizes, prompts and repetitions.
Every run goes through a guard script, one model process at a time, under `/usr/bin/time -l`. The campaign can be
resumed: a run whose JSON already exists is skipped. Every run is made twice: a timed run with the default binary, and a
counting run with the counting build, which gives the instructions and cycles of the useful work. Those stay readable
when the machine is busy and wall-clock times are not.

## Builds

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j                                  # timed runs
cmake -S . -B build-counters -DCMAKE_BUILD_TYPE=Release -DDM_COUNTERS=ON && cmake --build build-counters -j  # counting runs
```

`DM_COUNTERS` is OFF by default, and the OFF build compiles to the same objects as a tree without the option. When it
is ON, the pool reads the per-thread counters of macOS (`thread_selfcounts`, instructions and cycles) around every
phase. It does this on the main thread and on each worker. It also reads them around the main thread's spin-waits (the
pool's join and the expert-slot wait in `Cache::wait`). With `DM_COUNTS` set in the environment, the process prints
these lines to stderr at exit:

```
DMCOUNTS {"work_ins": .., "work_cyc": .., "main_work_ins": .., "main_serial_ins": .., "main_serial_cyc": .., "spin_ins": .., "spin_cyc": .., "runs": ..}
DMFN <image> <base> <addr> <ins> <cyc> <runs>        (one line per phase function the pool ran)
```

The fields are:

- `work`: instructions and cycles inside the pool's phases, summed over all threads.
- `main_serial`: the main thread's instructions between its first and last phase that are neither phase work nor spinning.
- `spin`: the spin-waits.

Useful work is `work + main_serial`. Spins are left out.

## Usage

```
python bench/run.py --model DIR --out DIR [--cells cache,routers,policy,approx] [--gb 3.3,5] [--prompts pitt,code]
                    [--routers F.dmr] [--router-args "--m 12"] [--evict-policy F.json] [--hazard] [--approx drop]
                    [--cell NAME=FLAGS] [--reps 2] [--bin build/dmeng] [--count-bin build-counters/dmeng | --no-counts]
                    [--guard GUARD | $DM_GUARD] [--resident F] [--extra "..."] [--dry-run] [--report [--symbolize]]
```

- `--model`: the model folder. Its `moved-store.json` gives the family. A folder without a manifest also needs
  `--family`.
- `--guard`: the guard script, called as `GUARD LOG TIMEOUT_S -- CMD...`. It must wait until no other model process is
  running, run the command under `/usr/bin/time -l` with its output in `LOG`, and print one line
  `GUARD {json}` with `rc`, `killed_by_guard`, `peak_footprint_gb_time`, `others_cpu_pct_mean` and the `before` /
  `after` machine state (`power`, `battery`, `mem_free_pct`). The guard is your own script (none ships with this
  repository). Give the guard with `--guard` or with `DM_GUARD`, and its interpreter with `--guard-python` or `DM_GUARD_PYTHON`.
- `--dry-run`: prints every command, with `# done, skipped:` in front of the runs that already have a JSON. It runs
  nothing.
- `--report`: rebuilds `results.jsonl`, `results.csv` and the table from `OUT/runs/` without running anything.
  `--symbolize` adds a table of the counting runs' phase functions, named with `atos`.
- Common flags of every run: `--n 400 --ctx 16384 --temp 0 --mem mlock`. You can change them with `--n`, `--ctx` and
  `--mem`. `--extra` adds flags to every run.
- Before each run, the driver waits while the machine is on battery below `--batt-floor` percent (default 25).
- When the others' CPU mean is above `--redo-others` percent (default 150), the timed run is moved to `runs/contended/`
  and redone once. The second result is kept, and the table flags it.
- When `mlock` is refused because another process holds the wired memory, the driver retries every 60 s, up to 20 times.

## Cells

| cell | flags | needs |
|---|---|---|
| `cache` | the cache alone (`--prefetch none`) | |
| `routers` | `--prefetch router --routers F` + `--router-args` | `--routers` |
| `policy` | the routers + `--evict-policy F` and / or `--evict hazard` + `--policy-args` | `--evict-policy` (with `--routers`) or `--hazard` |
| `approx` | the routers + `--mode approx:RULE` (`--approx`, default `drop`; `drop:w=0.12` for another weight) | `--routers` |
| `--cell NAME=FLAGS` | any dmeng flags; `{routers}` expands to the routers' flags | |

With no `--cells`, the campaign runs every standard cell whose inputs are given, then the `--cell` ones. Each cell runs
at every cache size in `--gb`, for every prompt in `--prompts`, `--reps` times. The loops are nested repetition, then
prompt, then cache size, then cell, so each cell's runs are spread through the campaign. Each counting run follows its
timed run.

Prompts are `pitt` and `code` (`bench/prompts/<family>/`), or paths to token-id files (comma or whitespace separated).

- `pitt` is the raw text "The history of the city of Pittsburgh begins with".
- `code` is a coding question: find the races in a small Python log-batching class. For Inkling 2 Mini, it is in
  the chat template.

Run files are named `CELL_PROMPT_gbG_rR.{log,json,ids,steps}`. Counting runs have the same names with `cnt_` in front.

## Columns

The table puts the instruction and cycle columns first, because they do not depend on what else the machine is doing.
The wall-clock columns follow. A wall-clock figure is marked `!` when the others' CPU mean during the run was above
`--busy` percent (default 50).

| column | meaning | source |
|---|---|---|
| useful G ins / G cyc | `work + main_serial` instructions and cycles, 1e9, whole run (prompt included) | counting run, DMCOUNTS |
| IPC | useful instructions / useful cycles | counting run |
| spin G ins | instructions spent spinning (pool join, slot wait), excluded from useful | counting run |
| mem | `P`: page reclaims above 1.5x the least of the runs at the same cache size. This means memory pressure: pages faulted back in on the run's own threads, so kernel work lands inside the counters. Treat those counts as suspect. | counting run, `time -l` |
| ids | the token ids compared with the prompt's reference. The reference is the first lossless cell at the first cache size, repetition 1, or `--ref-ids PROMPT=FILE`. `same` / `DIFF` / `ref`. `/c!` means the counting run's ids differ. `apx` marks an approximate cell, whose ids are not expected to match. | `.ids` |
| tok/s | decode tokens per second | timed run JSON `tok_s` |
| comp ms/tok | `compute_ms_per_token` (time per token minus waiting on reads and minus the run-ahead) | timed run |
| wait ms/tok | `wait_ms_per_token` (waiting on expert reads) | timed run |
| MB/tok | `mb_read_per_token`, in 1e6 bytes | timed run |
| peak GB | peak memory footprint from `time -l` (the guard's own sample if that is missing), in 1e9 bytes | guard |
| others CPU % | mean CPU of every other process during the run (100 = one core) | guard `others_cpu_pct_mean` |
| power | `AC` when on AC power before and after the run, else the battery charge after it | guard |
| trailing `P` | the timed run's page reclaims are above 1.5x the least at its cache size | timed run, `time -l` |

`results.jsonl` and `results.csv` have one row per cell run, with more fields:

- the timed run's `hit_rate`, `late_rate`, `miss_rate` and `hit_rate_looped` / `hit_rate_other`;
- from `read_queue`: `misses_per_token`, `late_per_token`, `wait_miss_ms_per_token`; from `drive`: `gb_s_while_busy` (the one drive rate: reader bytes / wall time with a read in flight);
- the routers' `top_k_in_named8/12/16` and `router_ms_per_token`;
- the approx cells' `weight_replaced`, `steps_with_a_replacement` and `weight_read_now`;
- from `draft`, when it is used: `tokens_per_step`, `accepted` and `offered`;
- `policy_ms_per_token`, the swap growth, and `time -l`'s whole-process `proc_ins` / `proc_cyc`;
- for the counting run: `work_*`, `main_serial_*`, `spin_*`, `pool_runs`, `useful_*_per_token`, `cnt_reclaims` and
  `cnt_others_cpu_pct`;
- in the JSONL only, the per-phase-function counts (`cnt_fn`).

## Example

```sh
export DM_GUARD=<your guard script>
python bench/run.py --model <model dir> --out <out dir> --gb 3.3,5 --reps 2 \
    --routers <model dir>/routers/<router>.dmr --router-args "--m 12" --hazard --dry-run
```
