# How much expert cache an agent turn needs (offline, 2026-11-22)

The OpenCode file-rewrite tool call (prompts `q36_toolrw_ids.txt`, 869 tokens, and `ink_toolrw_ids.txt`, 839 tokens,
from `results/drafts/prompts` of rc), 451 generated tokens, lossless. One recording run per model (dmeng of this branch,
cache only, 35B at 5 GB, Inkling at 6 GB, the prompt in single steps, k2 routers run without naming:
`--routers <k2> --router-stats --record F.rec --evict-trace F.trace`; AC power; the 35B ids equal the timing run
`engine/results/headline/runs/x_q36_toolrw_gb5_off_r1.ids`, all 451). Every cell below is that routing and those router
rows replayed through dm::Cache (`tests/evict_check/evict_replay`, table by `engine/agent_curve.py`); reads per token are
load-independent, counted over the 451 generated positions (the prompt is replayed first, not counted). Prose (pitt, 400 /
160 generated tokens, the runs of POLICY.md) for contrast.

Arms. (a) cache only, the old default (freq, half-life 8). (b) cache only, hazard. (c) the lag-2 router's top 8 per step
named as dmeng does it (`--prefetch router --m 8`: at the token's start for steps 0-1, after step a's misses for step
a + 2; a held record is pinned, else a slot is taken and read; pins go at the target's end), hazard; every name read
counts, the brackets give the named reads never used. (d) as (c) with a name budget: a name that needs a read is skipped
once this token's reads (urgent so far + named) would pass 120 MB (35B) / 180 MB (Inkling), the drive's 3.5 GB/s over
the token's compute; within a step the lower-ranked names go first. (e) as (c) with the old default (the timing run's
setup before auto). Hit rate = picks held when their step runs (the engine's hits + late).

**The replay is the engine in named mode too**: the 35B at 5 GB with `--prefetch router --routers <k2> --m 8` (hazard,
single-step prompt) measured hit + late 0.891, 205.6 MB/token (61.9 urgent + 143.7 named, 81.2 names read per token),
ids identical; arm (c) at 5 GB replays 0.891 / 206 (62 + 144). The timing campaign's run of the same turn at 5 GB
(build `verify-names`, `--prefill-chunk 512`) read 406 MB/token (98 + 308, 174 names read per token): twice
the names this branch's engine reads on the same ids. That difference is in that build or its batched prefill, not in
the routing; it should be checked before the 406 is used.

Router recall (the released top 8 inside the router's top 8): agent turn 0.685 (35B) / 0.747 (Inkling), prose 0.853 /
0.842.

### 35B, agent turn (file rewrite): hit rate / MB read per token (urgent + named, named-but-unused in brackets)

| GB | slots | (a) cache only, old default (freq) | (b) cache only, hazard | (c) k2 names m 8, hazard | (d) k2 names m 8, hazard, name budget | (e) k2 names m 8, old default (the timing run's setup) |
|---|---|---|---|---|---|---|
| 3.3 | 1864 | 0.605 / 224 | 0.623 / 213 | 0.849 / 295 (85 + 210 [82]) | 0.730 / 243 (153 + 90 [29]), 68 names skipped | 0.844 / 309 (88 + 221 [85]) |
| 5 | 2825 | 0.725 / 156 | 0.742 / 146 | 0.891 / 206 (62 + 144 [59]) | 0.833 / 179 (95 + 85 [33]), 33 names skipped | 0.887 / 218 (64 + 154 [62]) |
| 6.6 | 3729 | 0.800 / 113 | 0.818 / 103 | 0.920 / 147 (45 + 102 [44]) | 0.897 / 137 (58 + 79 [34]), 13 names skipped | 0.917 / 159 (47 + 112 [46]) |
| 8 | 4521 | 0.852 / 84 | 0.864 / 77 | 0.939 / 110 (34 + 75 [33]) | 0.931 / 107 (39 + 68 [30]), 4 names skipped | 0.937 / 119 (36 + 83 [35]) |
| 9.5 | 5368 | 0.895 / 59 | 0.904 / 54 | 0.956 / 78 (25 + 53 [24]) | 0.953 / 78 (26 + 51 [24]), 1 names skipped | 0.955 / 85 (26 + 59 [25]) |
| 11 | 6216 | 0.933 / 38 | 0.937 / 36 | 0.970 / 52 (17 + 35 [17]) | 0.969 / 52 (18 + 35 [17]), 0 names skipped | 0.970 / 55 (17 + 38 [17]) |

### Inkling, agent turn (file rewrite): hit rate / MB read per token (urgent + named, named-but-unused in brackets)

| GB | slots | (a) cache only, old default (freq) | (b) cache only, hazard | (c) k2 names m 8, hazard | (d) k2 names m 8, hazard, name budget | (e) k2 names m 8, old default (the timing run's setup) |
|---|---|---|---|---|---|---|
| 4 | 712+97 | 0.781 / 422 | 0.807 / 374 | 0.926 / 502 (143 + 359 [124]) | 0.865 / 413 (273 + 140 [40]), 40 names skipped | 0.921 / 558 (150 + 408 [135]) |
| 6 | 1069+145 | 0.892 / 212 | 0.900 / 198 | 0.958 / 270 (81 + 188 [71]) | 0.942 / 244 (121 + 123 [46]), 10 names skipped | 0.957 / 288 (85 + 203 [75]) |
| 7.8 | 1389+189 | 0.939 / 123 | 0.945 / 111 | 0.976 / 153 (48 + 104 [41]) | 0.972 / 148 (57 + 91 [37]), 2 names skipped | 0.974 / 169 (52 + 118 [46]) |
| 9 | 1603+218 | 0.960 / 82 | 0.966 / 71 | 0.984 / 98 (32 + 66 [28]) | 0.983 / 97 (35 + 63 [27]), 0 names skipped | 0.983 / 114 (35 + 78 [32]) |
| 10.5 | 1871+255 | 0.979 / 45 | 0.982 / 40 | 0.991 / 55 (19 + 36 [16]) | 0.991 / 55 (19 + 36 [16]), 0 names skipped | 0.990 / 64 (21 + 43 [18]) |

### 35B, prose (pitt): hit rate / MB read per token (urgent + named, named-but-unused in brackets)

| GB | slots | (a) cache only, old default (freq) | (b) cache only, hazard | (c) k2 names m 8, hazard | (d) k2 names m 8, hazard, name budget | (e) k2 names m 8, old default (the timing run's setup) |
|---|---|---|---|---|---|---|
| 3.3 | 1864 | 0.764 / 134 | 0.784 / 122 | 0.941 / 151 (33 + 118 [29]) | 0.899 / 144 (57 + 87 [22]), 17 names skipped | 0.942 / 163 (33 + 130 [30]) |
| 5 | 2825 | 0.861 / 79 | 0.872 / 73 | 0.958 / 92 (24 + 68 [20]) | 0.948 / 90 (29 + 61 [18]), 4 names skipped | 0.958 / 99 (24 + 75 [21]) |
| 6.6 | 3729 | 0.915 / 48 | 0.920 / 46 | 0.971 / 58 (16 + 41 [14]) | 0.968 / 58 (18 + 39 [13]), 1 names skipped | 0.970 / 61 (17 + 45 [14]) |
| 8 | 4521 | 0.941 / 33 | 0.943 / 32 | 0.979 / 40 (12 + 28 [10]) | 0.977 / 40 (13 + 27 [9]), 1 names skipped | 0.979 / 41 (12 + 29 [10]) |
| 9.5 | 5368 | 0.955 / 25 | 0.954 / 26 | 0.984 / 30 (9 + 21 [7]) | 0.982 / 30 (10 + 20 [7]), 1 names skipped | 0.984 / 30 (9 + 21 [7]) |
| 11 | 6216 | 0.959 / 23 | 0.959 / 23 | 0.986 / 25 (8 + 17 [5]) | 0.985 / 25 (9 + 17 [5]), 1 names skipped | 0.986 / 25 (8 + 18 [6]) |

### Inkling, prose (pitt): hit rate / MB read per token (urgent + named, named-but-unused in brackets)

| GB | slots | (a) cache only, old default (freq) | (b) cache only, hazard | (c) k2 names m 8, hazard | (d) k2 names m 8, hazard, name budget | (e) k2 names m 8, old default (the timing run's setup) |
|---|---|---|---|---|---|---|
| 4 | 712+97 | 0.851 / 283 | 0.857 / 274 | 0.956 / 338 (82 + 256 [62]) | 0.917 / 306 (169 + 137 [32]), 21 names skipped | 0.956 / 350 (83 + 267 [65]) |
| 6 | 1069+145 | 0.928 / 140 | 0.931 / 135 | 0.975 / 168 (47 + 121 [34]) | 0.969 / 163 (63 + 100 [29]), 3 names skipped | 0.975 / 174 (47 + 127 [35]) |
| 7.8 | 1389+189 | 0.962 / 75 | 0.964 / 72 | 0.986 / 85 (27 + 58 [17]) | 0.985 / 85 (30 + 56 [17]), 0 names skipped | 0.986 / 90 (27 + 63 [18]) |
| 9 | 1603+218 | 0.974 / 51 | 0.974 / 51 | 0.991 / 57 (18 + 39 [11]) | 0.990 / 58 (20 + 38 [11]), 0 names skipped | 0.991 / 57 (18 + 39 [11]) |
| 10.5 | 1871+255 | 0.979 / 40 | 0.979 / 41 | 0.993 / 41 (13 + 28 [8]) | 0.992 / 43 (15 + 27 [8]), 0 names skipped | 0.993 / 41 (13 + 28 [8]) |

## Readings

- **Cache needed before reads stop being the limit** (reads per token under the drive's budget for the token's compute,
  3.5 GB/s x 35 ms = 120 MB on the 35B, x 52 ms = 180 MB on Inkling): cache only with hazard, the 35B agent turn needs
  about 6 GB (146 MB at 5 GB, 103 at 6.6), Inkling about 7 GB (198 at 6, 111 at 7.8). With k2 names the 35B needs about
  7.5 GB (147 at 6.6, 110 at 8) and Inkling about 7.5 GB (270 at 6, 153 at 7.8). Prose gets there at about 3.3-4 GB on
  the 35B (122 at 3.3 cache only) and 5 GB on Inkling (135 at 6, 274 at 4).
- **Names add bytes on this turn at every size**: +40 % at 5 GB (35B: 206 against 146; 59 MB/token of named reads never
  used), +36 % at 6 GB on Inkling (270 against 198). They raise the share of picks held (0.742 -> 0.891 at 35B 5 GB)
  but only by reading more, so while the drive is saturated (35B below about 7.5 GB, Inkling below about 7.5 GB) they
  cost throughput; they pay only where the drive has slack in the token's compute time to hide the reads (35B 8 GB and
  up: 110 MB with names, under 120; Inkling 7.8 GB and up).
- **The byte budget does not fix it**: it cuts the named reads (35B 5 GB: 85 MB named instead of 144, 33 names skipped
  per token) but the skipped names come back as urgent misses (95 MB instead of 62), so the total stays above cache only
  at every size (179 against 146 at 5 GB, 243 against 213 at 3.3) while the hit rate falls toward cache only's. With
  this router's recall on agent text, cache only with hazard reads the fewest bytes at every size up to 11 GB; the
  budget only bounds the damage.
- Hazard against the old default, cache only: -5 to -13 % of the bytes at 3.3-9.5 GB (35B) and 4-9 GB (Inkling) on the agent turn (35B 213 vs 224
  at 3.3, 146 vs 156 at 5; Inkling 374 vs 422 at 4, 198 vs 212 at 6).

## Commands

    # recording (once per model and text)
    dmeng --model $Q36 --cache-gb 5 --ctx 16384 --temp 0 --mem mlock --prec i16 --prompt-file q36_toolrw_ids.txt --n 451 \
      --routers $Q36/routers/q36_router_full_k2_h512.dmr --router-stats --record ag_q36.rec --evict-trace ag_q36.trace
    dmeng --model $INK --cache-gb 6 --ctx 16384 --temp 0 --mem mlock --prompt-file ink_toolrw_ids.txt --n 451 \
      --routers $INK/routers/ink_r_fast/ink_router_full_k2_h512_b8.dmr --router-stats --record ag_ink.rec --evict-trace ag_ink.trace
    # replay (any slot counts; NAME_M=8 for names, NAME_BUDGET_MB=120 for the budget)
    NAME_M=8 NAME_BUDGET_MB=120 build-tools/evict_replay ag_q36.rec ag_q36.trace 2825 hazard /dev/null 869
    CP_WORK=<folder with bin/evict_replay> python3 results/cache-policy/engine/agent_curve.py <runs folder>
