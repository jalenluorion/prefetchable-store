# Keep or evict at phone sizes (0.8-3.3 GB): offline study on recorded routing (2026-11-22)

Qwen3.6-35B-A3B in the unified engine (dmeng), lossless decode, records of 1,769,472 B. No model ran: a Python
simulator replays recorded top-8 routing under a cache of N records with a pluggable policy (`evict_sim.py`). MB =
10^6 B per counted token, as stage4_sizing and P0. Upper bounds are labelled. Every number below names its text and
cache size; all of them are in `results.json` (merged from `res_*.json` by `tables.py`), full tables in `tables.md`.

## Answer

**At phone sizes use the hazard keep-score with uniform per-layer caps** (`dmeng --evict hazard --layer-caps uniform`,
ported on this branch, not the default). It is the phone engine's own policy (Store.swift's online per-layer table of
P(used at the next token | age bucket, decayed-count bucket)) with each layer holding an equal share of the slots. At
**1.27 GB it reads 226 / 270 / 301 MB/token on Pittsburgh / coding / agent, against 265 / 318 / 349 for the default
(decay, half-life 8; -15 / -15 / -14 %), 254 / 308 / 336 for LRU and 148 / 175 / 197 for Belady**. At 0.8 GB the cut is
-27 / -23 / -21 %, at 2.0 GB -6 / -6 / -6 %; on all eight texts it is the best online policy from 0.8 to 2.0 GB. From
3.3 GB the per-layer caps stop paying: at 3.3 GB hazard with caps and with one pool are within 1-3 % (123 / 153 / 180
and 122 / 154 / 182, default 134 / 157 / 188), at 5 GB the caps cost 1-10 %, so above about 2.5 GB use hazard with
one pool (`--evict hazard`). What remains to Belady (about 1.5x the bytes at every size) is
knowledge of the future beyond a few tokens: even an exact view of the next 4 tokens (160 layers) with the default
behind it reads 187 / 217 MB/token at 1.27 GB (Pittsburgh / coding) against Belady's 148 / 175, and one exact token
(40 layers) only matches what hazard + caps does with no future at all.

## 1. Simulator and validation

`evict_sim.py`: steps (token t, layer L) in order; a miss takes a slot from the free list, else the policy's lowest
score among held, unpinned records (one pool, or the layer's own slots when per-layer caps are on). Two step semantics:
**engine** (Core::lossless: picks taken in order, each pins its slot as it is taken, use statistics at the end of the
step) and **phone** (Store.swift exact mode: the layer's picks touched first and all protected, then the misses;
optional keep-window drop). Policies are score arrays over all 10,240 (layer, expert) records, or own lists (ARC, 2Q,
S3-FIFO). Counting: Pittsburgh and coding count the generated positions (399 / 399; the prompt is replayed first, not counted), the
teacher-forced texts and the phone dumps count every position from a cold start (stage4's convention).

| check (`python evict_sim.py validate` -> `validate.json`) | ours | reference |
|---|---|---|
| stage4 decay hl 8, hit rate, pitt 3.3 / 5 / 6.6 GB | 0.7639 / 0.8606 / 0.9146 | 0.7639 / 0.8606 / 0.9146 |
| stage4 Belady, pitt 3.3 / 5 / 6.6 | 0.8861 / 0.9318 / 0.9515 | 0.8861 / 0.9318 / 0.9515 |
| stage4 decay / Belady, coding 3.3 / 5 / 6.6 | 0.723 / 0.8231 / 0.877; 0.8582 / 0.914 / 0.9467 | identical |
| phone evict_sim.py, cap 48 per layer, excl. 64 tokens, MiB/token: hazard_h128 chat / wiki / code | 123.79 / 115.68 / 138.37 | 123.79 / 115.68 / 138.37 |
| the same, lru_keep (keep 31) / lru | 137.74, 134.34, 156.21 / 137.46, 134.00, 156.17 | identical |
| P0 tier_sim (1.27 GB, pitt): decay / LRU / Belady MB/token | 264.7 / 254.4 / 148.3 | 264.6 / 254 / 148.3 |

P0's decay row at 0.8 GB is 379.7 against our 378.4: the tie-break among equal keys (P0 and the engine: lowest slot;
ours: lowest record id). Belady and LRU agree there too.

Texts. Engine recordings (Qwen3.6-35B-A3B, dmeng): **pitt** (408 positions, 9 prompt), **coding** (756, 357 prompt),
**agent** transcript (2,873, teacher-forced), code text (1,800), prose (1,546). The trained k2 router's top-16 names
(lag 2, 8-bit) exist only for the three teacher-forced texts (`results/stage3c/runs/c_r8_*_gb6.6.names`; checked:
released top-8 inside its named 8 = 0.772 / 0.783 / 0.843, as stage 3c). Phone dumps (phone-store
`docs/evict_runs/{chat,wiki,code}.npz`, the same model through the phone's MLX path, exact mode: 1,541 / 2,082 / 2,340
positions, other texts and another quantisation, so the routing is not the engine's): used for the phone's policy and
caps table (which was fitted on these three dumps, so it is in-sample there). No lag-4 router names are recorded on the
Mac, and the cluster router traces (q36_d_trace) are on compute-node storage; neither was used.

## 2. Online policies (engine semantics): MB/token, % vs the default

`python3 tables.py headline` (the best half-life is chosen per cell, i.e. tuned on the text itself):

| text | GB | default: decay hl 8 | best decay hl (which) | LRU | ARC | learned (cache-only) | hazard, one pool | **hazard + uniform caps** | la 40 (1 token exact, bound) | Belady (bound) |
|---|---|---|---|---|---|---|---|---|---|---|
| pitt | 0.8 | 378 | 302 (hl 2, -20%) | 325 (-14%) | 313 (-17%) | 310 (-18%) | 305 (-19%) | 277 (-27%) | 253 (-33%) | 195 (-48%) |
| pitt | 1.0 | 321 | 272 (hl 2, -15%) | 280 (-13%) | 284 (-12%) | 284 (-12%) | 272 (-15%) | 251 (-22%) | 237 (-26%) | 172 (-46%) |
| pitt | 1.27 | 265 | 240 (hl 4, -9%) | 254 (-4%) | 251 (-5%) | 255 (-4%) | 239 (-10%) | 226 (-15%) | 223 (-16%) | 148 (-44%) |
| pitt | 1.6 | 221 | 216 (hl 4, -2%) | 226 (+2%) | 226 (+2%) | 227 (+3%) | 210 (-5%) | 203 (-8%) | 203 (-8%) | 127 (-42%) |
| pitt | 2.0 | 192 | 192 (hl 8, +0%) | 199 (+4%) | 199 (+4%) | 198 (+3%) | 183 (-4%) | 180 (-6%) | 182 (-5%) | 107 (-44%) |
| pitt | 2.5 | 166 | 166 (hl 8, +0%) | 173 (+4%) | 173 (+4%) | 169 (+1%) | 156 (-6%) | 155 (-7%) | 159 (-5%) | 88 (-47%) |
| pitt | 3.3 | 134 | 130 (hl 16, -3%) | 139 (+4%) | 136 (+2%) | 130 (-3%) | 122 (-9%) | 123 (-8%) | 129 (-4%) | 65 (-52%) |
| pitt | 5.0 | 79 | 74 (hl 32, -7%) | 81 (+3%) | 78 (-2%) | 74 (-6%) | 73 (-8%) | 80 (+1%) | 77 (-3%) | 39 (-51%) |
| coding | 0.8 | 423 | 357 (hl 2, -16%) | 373 (-12%) | 361 (-15%) | 360 (-15%) | 351 (-17%) | 326 (-23%) | 294 (-31%) | 227 (-46%) |
| coding | 1.0 | 374 | 327 (hl 2, -13%) | 337 (-10%) | 332 (-11%) | 336 (-10%) | 319 (-15%) | 299 (-20%) | 277 (-26%) | 202 (-46%) |
| coding | 1.27 | 318 | 291 (hl 4, -9%) | 308 (-3%) | 297 (-7%) | 308 (-3%) | 284 (-11%) | 270 (-15%) | 260 (-18%) | 175 (-45%) |
| coding | 1.6 | 271 | 260 (hl 4, -4%) | 276 (+2%) | 265 (-2%) | 279 (+3%) | 252 (-7%) | 244 (-10%) | 242 (-11%) | 150 (-45%) |
| coding | 2.0 | 231 | 231 (hl 4, -0%) | 240 (+4%) | 234 (+1%) | 251 (+8%) | 220 (-5%) | 216 (-6%) | 216 (-7%) | 126 (-45%) |
| coding | 2.5 | 197 | 197 (hl 8, +0%) | 205 (+4%) | 201 (+2%) | 219 (+11%) | 191 (-3%) | 188 (-5%) | 187 (-5%) | 105 (-47%) |
| coding | 3.3 | 157 | 157 (hl 8, +0%) | 162 (+3%) | 161 (+3%) | 178 (+13%) | 154 (-2%) | 153 (-3%) | 149 (-5%) | 80 (-49%) |
| coding | 5.0 | 100 | 100 (hl 8, +0%) | 104 (+4%) | 108 (+8%) | 114 (+14%) | 99 (-1%) | 104 (+4%) | 98 (-2%) | 49 (-51%) |
| agent | 0.8 | 450 | 384 (hl 2, -15%) | 399 (-11%) | 389 (-14%) | 385 (-14%) | 378 (-16%) | 356 (-21%) | 322 (-28%) | 248 (-45%) |
| agent | 1.27 | 349 | 323 (hl 4, -7%) | 336 (-4%) | 328 (-6%) | 331 (-5%) | 313 (-10%) | 301 (-14%) | 289 (-17%) | 197 (-43%) |
| agent | 2.0 | 262 | 260 (hl 4, -1%) | 271 (+4%) | 267 (+2%) | 271 (+4%) | 251 (-4%) | 246 (-6%) | 243 (-7%) | 149 (-43%) |
| agent | 3.3 | 188 | 188 (hl 8, +0%) | 197 (+5%) | 195 (+4%) | 198 (+6%) | 182 (-3%) | 180 (-4%) | 180 (-4%) | 99 (-47%) |
| agent | 5.0 | 129 | | 134 (+4%) | 135 (+5%) | 136 (+6%) | 124 (-3%) | | | 62 (-52%) |
| code text | 0.8 / 1.27 / 2.0 / 3.3 | 369 / 269 / 205 / 149 | | 360 / 283 / 224 / 158 | 332 / 272 / 215 / 155 | 309 / 255 / 203 / 145 | 314 / 250 / 196 / 139 | **305 / 243 / 193 / 141** | 269 / 238 / 195 / 144 | 212 / 164 / 122 / 80 |
| prose | same | 345 / 248 / 180 / 116 | | 317 / 248 / 191 / 123 | 307 / 245 / 189 / 119 | 298 / 243 / 183 / 112 | 295 / 229 / 168 / 101 | **270 / 218 / 166 / 105** | 245 / 213 / 170 / 112 | 189 / 141 / 98 / 55 |
| phone chat | same | 402 / 297 / 210 / 142 | | 356 / 271 / 212 / 150 | 339 / 273 / 213 / 151 | 337 / 279 / 219 / 148 | 327 / 253 / 196 / 135 | **305 / 244 / 192 / 136** | | 210 / 160 / 118 / 75 |
| phone wiki | same | 375 / 271 / 196 / 138 | | 333 / 263 / 206 / 145 | 322 / 260 / 204 / 143 | 315 / 260 / 204 / 137 | 314 / 247 / 189 / 125 | **284 / 231 / 182 / 126** | | 201 / 154 / 112 / 70 |
| phone code | same | 408 / 305 / 227 / 161 | | 371 / 298 / 234 / 168 | 358 / 295 / 235 / 171 | 344 / 289 / 229 / 158 | 341 / 273 / 214 / 149 | **320 / 261 / 209 / 149** | | 224 / 175 / 131 / 84 |

(The agent transcript's caps / lookahead / half-life cells ran at 0.8, 1.27, 2.0 and 3.3 GB only; the secondary texts
at those four.) Other online policies (`tables.md`): pure LFU is far worse everywhere (+20 to +170 %); LFU with
history +3 to +67 % at >= 1.27 GB; 2Q and S3-FIFO sit with ARC (-9 to -17 % at 0.8 GB, -6 to +10 % from 1.6 GB);
decay with counts kept across evictions (hl 8) is -16 to -20 % at 0.8 GB and within 1 % of the default from 2 GB;
hazard with the keep-window drop (keep 31) equals hazard without it to 0.1 MB except at 5 GB (+13 % on pitt: the drop
empties slots the cache could keep). The learned keep-score (`learned`, cluster's q36_d_evict_policy.json evaluated as
keep_score() does when no router is loaded, router features 0): -14 to -18 % at 0.8 GB, but +8 to +14 % on coding from
2 GB; its v2 (q36_d_evict2_policy.json, router features 0, not in the engine) is similar. With its router features
(only on cluster's traces) cluster measured -3 to -7 % at 3.3 GB; not reproducible here.

Readings:
- **The default is tuned for 3.3 GB and above**: its 8-token half-life is the best decay half-life only from 2 GB; at
  0.8-1 GB a 2-token half-life reads 13-20 % less, at 1.27-1.6 GB a 4-token one 2-9 %. Hazard learns its own
  age / frequency trade-off online, untuned, and beats the best fixed half-life (chosen per cell, i.e. tuned on the
  text) in every cell but two (pitt and prose at 0.8 GB: 305 vs 302 and 295 vs 292 with hl 2), one pool.
- **Per-layer caps help every policy below 2 GB, and uniform caps get it.** With hazard they are worth a further
  3-10 % at 0.8-1.27 GB; with decay much more (it is the default's worst flaw at small sizes: one pool lets a few
  layers' fresh records push out other layers' reused ones; pitt 0.8 GB 378 -> 307, coding 423 -> 356, agent 450 -> 386;
  at 1.27 GB 265 -> 249, 318 -> 294, 349 -> 324); with LRU 1-3 %. From 2 GB caps are within +-2 % for every policy, and
  from 3.3 GB they cost up to 3 % (5 GB: +1 to +10 %). The phone's LayerCapsTable shape (fitted on the phone dumps) beats
  uniform by 0-1 % on the engine's texts (pitt 1.27 GB hazard: 224.8 vs 225.6) and by about 1 % on its own dumps. The
  phone engine's exact policy (table caps, touch first, hazard, no drop) reads 224.6 / 268.7 / 299 MB/token at 1.27 GB
  (pitt / coding / agent): the same as the port.
- **One pool wins from about 3.3 GB** (hazard pooled 122 vs caps 123 on pitt, 182 vs 180 agent; at 5 GB 73 vs 80 on
  pitt): the cache holds most of each layer's working set and fixed shares start to waste slots on cold layers.

## 3. Knowledge of the future

`la:k` = Belady restricted to k layers of exact lookahead (protect records used within k steps, nearer first; demote
held records of layers that run within k steps without using them; the rest by the base policy). `nm` = names inside
the token as the only future (named for L+1..L+lag protected, held records of those layers not named demoted).
MB/token, base decay hl 8 unless named (`tables.md` has every cell):

| text, GB | default | la 2 | la 4 | la 8 | la 40 (1 token) | la 80 | la 160 (4 tokens) | names exact lag 2 / 4 | router k2 top 8 / 12 / 16 | hazard + caps (no future) | Belady |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pitt 0.8 | 378 | 292 | 285 | 279 | 253 | 234 | 211 | 292 / 288 | (no names) | 277 | 195 |
| pitt 1.27 | 265 | 248 | 235 | 232 | 223 | 205 | 187 | 249 / 237 | | 226 | 148 |
| pitt 2.0 | 192 | 198 | 187 | 185 | 182 | 174 | 160 | 199 / 192 | | 180 | 107 |
| coding 1.27 | 318 | 286 | 277 | 272 | 260 | 241 | 217 | 286 / 279 | | 270 | 175 |
| code text 0.8 | 369 | 317 | 310 | 302 | 269 | 250 | 227 | 319 / 314 | 331 / 323 / 320 | 305 | 212 |
| code text 1.27 | 269 | 263 | 252 | 247 | 238 | 218 | 199 | 263 / 255 | 270 / 265 / 262 | 243 | 164 |
| code text 2.0 | 205 | 208 | 201 | 199 | 195 | 187 | 172 | 207 / 204 | 211 / 208 / 207 | 193 | 122 |
| agent 0.8 | 450 | 363 | 359 | 352 | 322 | 298 | 266 | 365 / 364 | 379 / 370 / 366 | 356 | 248 |
| agent 1.27 | 349 | 312 | 306 | 302 | 289 | 265 | 238 | 312 / 309 | 322 / 315 / 312 | 301 | 197 |
| agent 2.0 | 262 | 256 | 250 | 249 | 243 | 229 | 209 | 255 / 254 | 261 / 257 / 255 | 246 | 149 |
| prose 1.27 | 248 | | | | 213 | | | 239 / - | 245 / 234 / 230 | 218 | 141 |

The same future on top of hazard (one pool) instead of the default: names add nothing (agent 1.27 GB: hazard 313,
+ exact names lag 2 308, + router top 12 311; code text 250 / 257 / 259; the hazard table already ranks by what a lag-2
name says, and demoting un-named records of the next layers hurts it), and even one exact token (la 40 + hazard: pitt
216, coding 258, agent 286 at 1.27 GB) is only 4-5 % below hazard + caps (226 / 270 / 301).

Readings: (i) the trained router's names as eviction input help the default at 0.8 GB (-18 / -12 / -18 % with its top 12 on
agent / code text / prose) and at 1.27 GB (-10 / -1.5 / -6 %), nothing from 2 GB (within +-2 %; stage 3's loss at 3.3 GB
holds), and always less than hazard + caps with no names (1.27 GB: 315 / 265 / 234 vs 301 / 243 / 218). Its top 12-16
protect about as well as exact names of the same lag (the extra names cover its misses).
(ii) Exact names of the next 2 layers on top of the default are worth about what hazard is worth alone; the next 4
tokens (la 160) close only half the gap from hazard + caps to Belady. (iii) Belady's advantage is long-range reuse: the
records it keeps are used again many tokens later, which no lag-k router names.

**Reads that could be issued early.** In cache-only mode, the share of each policy's misses that the trained k2 router
had in its top 8 / 12 (so they could have been read two layers ahead) is 0.69-0.74 / 0.84-0.87 on the agent transcript
at 0.8-1.27 GB, 0.65-0.71 / 0.81-0.85 on code text, 0.76-0.80 / 0.90-0.92 on prose, falling by 0.03-0.08 from 0.8 to
3.3 GB and by 0.01-0.03 from the default to hazard + caps (the better the policy, the more of what is left is what the
router misses). With exact names every miss could be early. The rest (26-31 % on agent at 1.27 GB) is read on the spot.

**Named-early mode** (the engine's lossless router mode: the router's top N for layer L + 2 are read into the cache
while layer L runs and pinned until their step; a name leaves 2K slots): on a drive-bound phone it **reads more in
total than cache only**, because wrong names cost reads and evict useful records. Agent transcript, 1.27 GB, MB/token
early + spot (records named and never used, per token): default 343 + 100 = 443 (53), hazard 308 + 95 = 403 (50), top
12: 522 + 51 = 573 with hazard (145), against 349 / 313 cache only. Same picture on code text and prose and at 2-3.3 GB
(`tables.md`). Exact names (upper bound) read exactly the cache-only bytes, all early.

## 4. Two tiers: the policy tier and the OS file cache (1.27 GB + 2.1 GB, or one 3.3 GB)

Drive MB/token; copies = records taken from the second tier per token (each a 0.13-0.20 ms copy out of the OS cache,
P0 2.2):

| text | A: 1.27 P + 2.1 LRU, exclusive (decay / LRU / hazard / ARC / Belady) | A': 1.27 P, buffered reads, OS cache 2.1 inclusive (decay / LRU / hazard / Belady) | B: one 3.3 GB tier, OS evicts LRU (= LRU 3.3) | B2: one 3.3 GB tier, P decides (needs a page drop): decay / hazard / hazard + caps |
|---|---|---|---|---|
| pitt | 137 / 136 / **131** / 134 / 125 (copies 72 / 67 / 61 / 66 / 13) | 159 / 196 / 170 / 141 | 139 | 134 / **122** / 123 |
| coding | 157 / 158 / **154** / 156 / 147 | 186 / 234 / 204 / 164 | 162 | 157 / 154 / **153** |
| agent | 190 / 194 / **187** / 190 / 177 | 218 / 270 / 238 / 191 | 197 | 188 / 182 / **180** |
| code text | 150 / 156 / 148 / 151 / 144 | 171 / 222 / 177 / 157 | 158 | 149 / **139** / 141 |
| prose | 119 / 120 / 115 / 117 / 112 | 145 / 183 / 153 / 132 | 123 | 116 / **101** / 105 |
| phone chat | 144 / 147 / 142 / 145 / 135 | 166 / 209 / 185 / 151 | 150 | 142 / **135** / 136 |

Which arrangement reads least: **one policy-managed tier over the whole 3.3 GB (B2) with hazard**, 122 / 153-154 / 180-182
MB/token on pitt / coding / agent, provided the engine can make the OS drop the pages of the records it evicts (not
tested on iOS: madvise / msync on clean file pages). If the OS decides what stays (B, Mem::FILE as P0 proposes), the
engine's policy has no say: drive reads are LRU's at 3.3 GB (139 / 162 / 197), whatever P is. The two-tier design as
posed (A, exclusive) reads 131 / 154 / 187 with hazard in the first tier, but exclusivity is not buildable with clean
file pages: a record evicted from the dirty tier only enters the OS cache by being read again. The buildable two-tier
(A': buffered reads fill the OS cache, so it holds copies of the first tier) reads more than B: 159-218 with decay, and
hazard in the first tier makes it worse (170 / 204 / 238), because the OS cache then sees only the first tier's misses,
which hazard has made harder to reuse. Ranking: B2 hazard < A exclusive hazard < B (OS LRU) < A' (inclusive).

## 5. What it predicts for tokens/s on the 15 Plus

`predict.py` with P0's numbers: drive 0.8 GB/s, compute 55 ms/token burst / 89 sustained (4-bit resident, CPU),
0.15 ms per copy from the OS cache. Cache only = compute + reads; named early (ceiling, exact names) = max(compute,
reads). Pittsburgh / coding / agent:

| configuration | MB/token | tok/s cache only, burst ; sustained | tok/s named early (ceiling, burst; = sustained where reads > 89 ms) |
|---|---|---|---|
| 1.27 GB, default (today's core) | 265 / 318 / 349 | 2.59 / 2.21 / 2.04 ; 2.38 / 2.05 / 1.91 | 3.02 / 2.51 / 2.29 |
| 1.27 GB, hazard + uniform caps (the port) | 226 / 270 / 301 | 2.97 / 2.55 / 2.32 ; 2.70 / 2.34 / 2.15 | 3.55 / 2.96 / 2.66 |
| 1.27 GB, Belady (bound) | 148 / 175 / 197 | 4.16 / 3.66 / 3.32 | 5.40 / 4.58 / 4.06 |
| B: 3.3 GB file-backed, OS LRU | 139 / 162 / 197 | 4.37 / 3.88 / 3.32 ; 3.80 / 3.43 / 2.98 | 5.75 / 4.94 / 4.05 |
| B2: 3.3 GB, hazard + caps (needs page drop) | 123 / 153 / 180 | 4.79 / 4.06 / 3.57 ; 4.12 / 3.57 / 3.18 | 6.50 / 5.23 / 4.44 |
| 3.3 GB, Belady (bound) | 65 / 80 / 99 | 7.37 / 6.44 / 5.61 | 12.4 / 9.97 / 8.11 |

So the port is worth +14-15 % tok/s on today's 1.27 GB phone (2.59 -> 2.97 on Pittsburgh, cache only, burst), the
file cache +43-61 % more on top (B with the OS deciding to B2 with hazard + caps), and every configuration stays
drive-bound (reads 150-440 ms against 55-89 ms of compute). With the trained router naming two layers ahead (named-early mode, wasted reads counted) the phone is slower
than cache only at these sizes: agent 1.27 GB, hazard, top 8: max(55, 385) + 119 ms = 1.99 tok/s vs 2.24 cache only.
Naming early pays only once reads fall near compute time (about 44-70 MB/token of misses at 0.8 GB/s), i.e. not on a
6 GB phone with this model.

## 6. The port (branch port-ios-evict)

`src/core/store.cpp`, `include/dmeng/store.h`: `Cache::enable_hazard()` (keep_score() returns the hazard estimate for
the next token + 1e-9 x last use; victim() takes the lowest, ties to the lower record id; statistics in
`pre_use()` / `use()`, the clock ticks lazily per token; per record 2 doubles + 2 bucket indices, per layer 130 x 2
counts; the tick is one pass over 10,240 records per token), `Cache::uniform_layer_caps()` (a layer at its cap evicts
one of its own records; held counts kept in issue / drop / cancel / purge). `src/core/moe.cpp`: lossless() and approx()
call `pre_use()` for the step's released picks before its misses take slots (hazard only; the default path is
unchanged). This order matters with caps: without it a miss can evict a later pick of its own step whose statistics are
a token old (pitt 1.27 GB with caps: 236.3 MB/token with the touch at the end of the step vs 225.6 with it first;
pooled the two differ by 0.3 %). `tools/dmeng.cpp`: `--evict hazard`, `--layer-caps uniform` (defaults unchanged: freq,
one pool; hazard and `--evict-policy` refuse to combine).

**Unit check** (`check_cpp.sh`, `tests/evict_check/evict_check.cpp`): the recorded trace is replayed through dm::Cache
exactly as Core::lossless + finish_step drive it in cache-only mode (reads skipped), and every eviction is written as
(step, record); the simulator writes the same with `evict_sim.py log`. **Identical, eviction for eviction, in all 24
cells**: pitt, coding, agent x 452 / 720 / 1130 / 1864 slots x one pool / uniform caps (e.g. agent 720 slots uniform:
487,366 evictions, 488,085 misses in both; `check_cpp.log`). The full engine was then built and run (section 6a).

## 6a. In the engine (Mac, 2026-11-22 03:55-04:05)

dmeng built from this branch (`cmake --build build --target dmeng -j3`), lossless, `--temp 0 --ctx 16384 --prec i16
--mem mlock`, model MODEL_DIR (the 35B, 8-bit resident), 400 tokens, every run through
a guard script (one model run at a time; it waited for another job's dmeng gates). 1.27 GB is
`--cache-gb 1.27402` (exactly the simulator's 720 slots). References: stage4's `r_r8_pitt_gb3.3_r1.ids` and
`dc_off_code.ids`. `engine/run_gate.sh`, table by `engine/summary.py` (`engine/summary.md`); hit rate and MB/token are
over the 399 decode steps, as the simulator counts. Policy ms = the new `policy_ms_per_token` (a timer around victim()
and pre_use()).

| run | slots | ids vs reference | hit rate engine / sim | MB/token engine / sim | compute ms/token | policy ms/token | wait ms/token | tok/s | others' CPU % |
|---|---|---|---|---|---|---|---|---|---|
| pitt, default, 1.27 | 720 | identical (400) | 0.5326 / 0.5324 | 264.6 / 264.7 | 39.5 | 0.22 | 83.0 | 8.16 | 114 |
| pitt, hazard + uniform caps, 1.27 | 720 | identical (400) | 0.6015 / 0.6015 | 225.6 / 225.6 | 35.9 | 0.29 | 76.6 | 8.89 | 50 |
| coding, default, 1.27 | 720 | identical (400) | 0.4378 / 0.4380 | 318.3 / 318.2 | 35.3 | 0.28 | 104.9 | 7.13 | 37 |
| coding, hazard + uniform caps, 1.27 | 720 | identical (400) | 0.5230 / 0.5230 | 270.1 / 270.1 | 34.5 | 0.29 | 89.6 | 8.05 | 30 |
| pitt, default, 3.3 | 1864 | identical (400) | 0.7639 / 0.7639 | 133.7 / 133.7 | 33.4 | 0.22 | 45.8 | 12.61 | 32 |
| pitt, hazard one pool, 3.3 | 1864 | identical (400) | 0.7844 / 0.7844 | 122.1 / 122.1 | 35.8 | 1.12 | 39.8 | 13.22 | 59 |
| coding, default, 3.3 | 1864 | identical (400) | 0.7230 / 0.7230 | 156.8 / 156.8 | 33.9 | 0.26 | 50.8 | 11.80 | 37 |
| coding, hazard one pool, 3.3 | 1864 | identical (400) | 0.7285 / 0.7285 | 153.7 / 153.7 | 35.0 | 1.27 | 50.0 | 11.77 | 34 |

- **Lossless: 8 of 8 runs give the reference's 400 ids.** The policy changes which records are held, never the output.
- **Reads: the engine equals the simulator** to the 4th digit of the hit rate in all four hazard runs and both 3.3 GB
  default runs; the default at 1.27 GB differs by 0.0002 (0.1 MB/token), the tie-break among equal keys (engine: lowest
  slot; simulator: lowest record id; P0's slot-order simulator gives the engine's 0.5326). The coding prompt goes through
  the engine's batched prefill and the simulator replays it token by token; the decode reads agree anyway.
- **Policy cost: 0.29 ms/token with hazard + caps at 1.27 GB (default 0.22-0.28); 1.1-1.3 ms/token with hazard in one
  pool at 3.3 GB (default 0.22-0.26)**: the pooled hazard scan evaluates every unpinned slot's table lookup per miss
  (1,864 slots x ~40 misses). Not optimised (a per-layer minimum or a heap would remove most of it).
- Timings are indicative only: the Mac was shared (other dmeng gates between ours, Xcode builds; others' CPU
  30-114 % during the runs, load average 11-15; AC power; no swap growth except 1.2 GB in one run). The tok/s and wait
  columns move in the direction of the reads (pitt 1.27 GB: wait 83.0 -> 76.6 ms/token, 8.16 -> 8.89 tok/s) but a single
  run each on a loaded machine is not a speed result; the Mac's drive (3.4 GB/s) is not the phone's.

## 7. Not tested

- No phone run: the gain in tokens/s on the 15 Plus is predicted, not measured; on the Mac one run per cell on a shared
  machine (section 6a), no speed claim.
- Lag-4 router names (none recorded on the Mac) and the trained routers' names on Pittsburgh / coding; the learned
  keep-score with its router features (cluster's traces are on compute-node storage).
- CLOCK-Pro (ARC / 2Q / S3-FIFO cover recency + frequency + ghosts; all lose to hazard).
- Long sessions beyond 2,873 positions, prefill (batched) paths, multi-position steps, the purgeable mode.
- Whether iOS lets an app drop clean file pages it evicts (B2), and the real OS file cache's policy (taken as LRU on
  whole records; it works on 16 KB pages and is undocumented). The 2.1 GB tier size is P0's steady plateau.
- Per-layer caps from a table fitted on the engine's own texts; caps that adapt online.
