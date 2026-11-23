
**pitt**

| arrangement / policy | MB/token (drive) | read ms | tok/s cache only, burst | sustained | tok/s named early (ceiling), burst | sustained |
|---|---|---|---|---|---|---|
| 1.27 GB, decay (today's default) | 265 | 331 | 2.59 | 2.38 | 3.02 | 3.02 |
| 1.27 GB, LRU | 254 | 318 | 2.68 | 2.46 | 3.14 | 3.14 |
| 1.27 GB, hazard pooled | 239 | 298 | 2.83 | 2.58 | 3.35 | 3.35 |
| 1.27 GB, hazard + uniform per-layer caps (the port) | 226 | 282 | 2.97 | 2.70 | 3.55 | 3.55 |
| 1.27 GB, the phone engine's policy (table caps, hazard, no drop) | 225 | 281 | 2.98 | 2.70 | 3.56 | 3.56 |
| 1.27 GB, Belady (offline bound) | 148 | 185 | 4.16 | 3.64 | 5.40 | 5.40 |
| 1.27 decay + 2.1 GB LRU tier (exclusive) [+72 copies/token] | 137 | 171 | 4.21 | 3.69 | 5.49 | 5.49 |
| 1.27 hazard + 2.1 GB LRU tier (exclusive) [+61 copies/token] | 131 | 163 | 4.39 | 3.82 | 5.80 | 5.80 |
| 1.27 decay + 2.1 GB OS cache (inclusive, buffered reads) [+60 copies/token] | 159 | 198 | 3.81 | 3.37 | 4.82 | 4.82 |
| 1.27 hazard + 2.1 GB OS cache (inclusive, buffered reads) [+39 copies/token] | 170 | 213 | 3.66 | 3.25 | 4.58 | 4.58 |
| 3.3 GB one tier, OS LRU decides (= LRU at 3.3) | 139 | 174 | 4.37 | 3.80 | 5.75 | 5.75 |
| 3.3 GB one tier, decay (needs page drop) | 134 | 167 | 4.50 | 3.90 | 5.98 | 5.98 |
| 3.3 GB one tier, hazard (needs page drop) | 122 | 153 | 4.82 | 4.14 | 6.55 | 6.55 |
| 3.3 GB one tier, hazard + uniform caps (needs page drop) | 123 | 154 | 4.79 | 4.12 | 6.50 | 6.50 |
| 3.3 GB, Belady (offline bound) | 65 | 81 | 7.37 | 5.90 | 12.40 | 11.24 |

**coding**

| arrangement / policy | MB/token (drive) | read ms | tok/s cache only, burst | sustained | tok/s named early (ceiling), burst | sustained |
|---|---|---|---|---|---|---|
| 1.27 GB, decay (today's default) | 318 | 398 | 2.21 | 2.05 | 2.51 | 2.51 |
| 1.27 GB, LRU | 308 | 385 | 2.27 | 2.11 | 2.60 | 2.60 |
| 1.27 GB, hazard pooled | 284 | 355 | 2.44 | 2.25 | 2.82 | 2.82 |
| 1.27 GB, hazard + uniform per-layer caps (the port) | 270 | 338 | 2.55 | 2.34 | 2.96 | 2.96 |
| 1.27 GB, the phone engine's policy (table caps, hazard, no drop) | 269 | 336 | 2.56 | 2.35 | 2.98 | 2.98 |
| 1.27 GB, Belady (offline bound) | 175 | 219 | 3.66 | 3.25 | 4.58 | 4.58 |
| 1.27 decay + 2.1 GB LRU tier (exclusive) [+91 copies/token] | 157 | 196 | 3.77 | 3.34 | 4.76 | 4.76 |
| 1.27 hazard + 2.1 GB LRU tier (exclusive) [+73 copies/token] | 154 | 193 | 3.86 | 3.42 | 4.91 | 4.91 |
| 1.27 decay + 2.1 GB OS cache (inclusive, buffered reads) [+75 copies/token] | 186 | 233 | 3.34 | 3.00 | 4.10 | 4.10 |
| 1.27 hazard + 2.1 GB OS cache (inclusive, buffered reads) [+45 copies/token] | 204 | 255 | 3.16 | 2.85 | 3.82 | 3.82 |
| 3.3 GB one tier, OS LRU decides (= LRU at 3.3) | 162 | 202 | 3.88 | 3.43 | 4.94 | 4.94 |
| 3.3 GB one tier, decay (needs page drop) | 157 | 196 | 3.98 | 3.51 | 5.10 | 5.10 |
| 3.3 GB one tier, hazard (needs page drop) | 154 | 192 | 4.05 | 3.56 | 5.20 | 5.20 |
| 3.3 GB one tier, hazard + uniform caps (needs page drop) | 153 | 191 | 4.06 | 3.57 | 5.23 | 5.23 |
| 3.3 GB, Belady (offline bound) | 80 | 100 | 6.44 | 5.28 | 9.97 | 9.97 |

**agent**

| arrangement / policy | MB/token (drive) | read ms | tok/s cache only, burst | sustained | tok/s named early (ceiling), burst | sustained |
|---|---|---|---|---|---|---|
| 1.27 GB, decay (today's default) | 349 | 436 | 2.04 | 1.91 | 2.29 | 2.29 |
| 1.27 GB, LRU | 336 | 420 | 2.10 | 1.96 | 2.38 | 2.38 |
| 1.27 GB, hazard pooled | 313 | 392 | 2.24 | 2.08 | 2.55 | 2.55 |
| 1.27 GB, hazard + uniform per-layer caps (the port) | 301 | 376 | 2.32 | 2.15 | 2.66 | 2.66 |
| 1.27 GB, the phone engine's policy (table caps, hazard, no drop) | 299 | 374 | 2.33 | 2.16 | 2.67 | 2.67 |
| 1.27 GB, Belady (offline bound) | 197 | 246 | 3.32 | 2.98 | 4.06 | 4.06 |
| 1.27 decay + 2.1 GB LRU tier (exclusive) [+90 copies/token] | 190 | 237 | 3.27 | 2.94 | 3.99 | 3.99 |
| 1.27 hazard + 2.1 GB LRU tier (exclusive) [+71 copies/token] | 187 | 234 | 3.34 | 3.00 | 4.09 | 4.09 |
| 1.27 decay + 2.1 GB OS cache (inclusive, buffered reads) [+74 copies/token] | 218 | 272 | 2.96 | 2.69 | 3.53 | 3.53 |
| 1.27 hazard + 2.1 GB OS cache (inclusive, buffered reads) [+43 copies/token] | 238 | 297 | 2.79 | 2.55 | 3.29 | 3.29 |
| 3.3 GB one tier, OS LRU decides (= LRU at 3.3) | 197 | 247 | 3.32 | 2.98 | 4.05 | 4.05 |
| 3.3 GB one tier, decay (needs page drop) | 188 | 234 | 3.45 | 3.09 | 4.26 | 4.26 |
| 3.3 GB one tier, hazard (needs page drop) | 182 | 228 | 3.53 | 3.15 | 4.38 | 4.38 |
| 3.3 GB one tier, hazard + uniform caps (needs page drop) | 180 | 225 | 3.57 | 3.18 | 4.44 | 4.44 |
| 3.3 GB, Belady (offline bound) | 99 | 123 | 5.61 | 4.71 | 8.11 | 8.11 |

Router lag 2 named early (trained k2 router, top N read two layers ahead; agent transcript):

| cache | policy | N | early MB | spot MB | wasted rec/token | tok/s burst | sustained |
|---|---|---|---|---|---|---|---|
| 1.27 | decay:8 | 8 | 343 | 100 | 53.0 | 1.80 | 1.80 |
| 1.27 | decay:8 | 12 | 568 | 53 | 151.9 | 1.29 | 1.29 |
| 1.27 | lru | 8 | 329 | 100 | 41.7 | 1.87 | 1.87 |
| 1.27 | lru | 12 | 567 | 55 | 129.6 | 1.29 | 1.29 |
| 1.27 | hazard | 8 | 308 | 95 | 49.5 | 1.99 | 1.99 |
| 1.27 | hazard | 12 | 522 | 51 | 144.6 | 1.40 | 1.40 |
| 2.0 | decay:8 | 8 | 255 | 83 | 43.0 | 2.37 | 2.37 |
| 2.0 | decay:8 | 12 | 443 | 45 | 126.8 | 1.64 | 1.64 |
| 2.0 | lru | 8 | 262 | 84 | 31.8 | 2.31 | 2.31 |
| 2.0 | lru | 12 | 455 | 47 | 100.5 | 1.59 | 1.59 |
| 2.0 | hazard | 8 | 245 | 81 | 41.6 | 2.46 | 2.46 |
| 2.0 | hazard | 12 | 430 | 44 | 124.1 | 1.69 | 1.69 |
| 3.3 | decay:8 | 8 | 183 | 64 | 33.2 | 3.25 | 3.25 |
| 3.3 | decay:8 | 12 | 331 | 35 | 100.4 | 2.19 | 2.19 |
| 3.3 | lru | 8 | 190 | 66 | 21.8 | 3.13 | 3.13 |
| 3.3 | lru | 12 | 334 | 37 | 69.4 | 2.16 | 2.16 |
| 3.3 | hazard | 8 | 177 | 63 | 32.3 | 3.33 | 3.33 |
| 3.3 | hazard | 12 | 325 | 35 | 98.9 | 2.23 | 2.23 |
