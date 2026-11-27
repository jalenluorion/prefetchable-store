/* dmeng C API.
 *
 * The one boundary an app links against: a C header over include/dmeng/lm.h (dm::LM, dm::ModelLM),
 * include/dmeng/session.h (dm::PrefixCache) and the model loader (include/dmeng/model.h), so Swift (the iPhone app),
 * Kotlin/JNI or Python ctypes need no C++. Token ids in, logits out: the tokenizer and the chat template stay in the
 * app, as in python/serve/dmserve.py.
 *
 * Threads. Every call on an engine is made from one caller thread at a time (the thread becomes thread 0 of the compute
 * pool while dm_sync / dm_step run; their QoS is set to the open options' class for the call). Exceptions, callable
 * from any thread at any time: dm_cancel, dm_set_cache_bytes, dm_last_error(NULL). dm_counters may be called from any
 * thread; its numbers are a snapshot without a lock (they may be one step apart from each other).
 *
 * Profiles and modes (docs/MANIFEST-PRESETS.md, the same as dmeng / dmserve --profile / --mode). dm_default_opts
 * leaves profile NULL: the profile is chosen from the device at dm_open (macOS: mac; iOS: phone with at least the
 * manifest's min_available_gb (5.0) available at open, else phone-small), and its defaults fill every option left at
 * zero / NULL: the mode (exact, or fast on phone-small), the in-memory weights, the routers and their names rule, the
 * cross-token routers, the eviction policy and the cache size. Set profile to "none" for the engine's own values.
 *
 * Exactness. Mode exact (lossless) is the default on mac and phone: the released router decides every expert, so token ids
 * never depend on the cache size, the memory mode, the routers or the timing. dm_sync's restored state is the bits that
 * were saved (session.h), and the logits after dm_sync(ids) equal those of feeding ids one at a time. dm_sample is pure
 * integer / double arithmetic in a fixed order (no platform RNG distribution), so the Mac and the phone draw the same
 * token from the same logits, parameters and draw index.
 */
#ifndef DMENG_C_H
#define DMENG_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dm_engine dm_engine; /* one model + its core (pool, expert cache, readers) + one prefix cache */

enum {
  DM_OK = 0,
  DM_EINVAL = -1,     /* bad argument */
  DM_ELOAD = -2,      /* model folder / resident file / routers could not be loaded (dm_last_error says why) */
  DM_ECANCELLED = -3, /* dm_cancel was called; the state is consistent at the position dm_info reports */
  DM_ENOTIMPL = -4,   /* not in this build */
  DM_ENOMEM = -5
};

/* QoS class of the compute threads (Apple platforms). */
enum { DM_QOS_USER_INTERACTIVE = 0, DM_QOS_USER_INITIATED = 1, DM_QOS_DEFAULT = 2, DM_QOS_UTILITY = 3 };
/* Which cores the compute pool is sized for: any (threads as given), or at most the performance cores. */
enum { DM_CORES_ANY = 0, DM_CORES_PERF = 1 };

/* How the expert cache is backed. */
enum {
  DM_MEM_PLAIN = 0,     /* anonymous dirty memory */
  DM_MEM_MLOCK = 1,     /* locked (the Mac default) */
  DM_MEM_PURGEABLE = 2, /* purgeable VM objects: the OS may take them back under pressure (a miss re-reads) */
  DM_MEM_FILE = 3       /* held records are views of the store files' pages in the OS file cache (the phone's tier):
                           cache_bytes is the policy's bookkeeping, not memory charged to the app */
};

typedef struct dm_open_opts {
  int threads;             /* compute threads; 0 = the platform default (iOS 4, macOS 8) */
  int qos;                 /* DM_QOS_* of the compute workers and of the caller during dm_sync / dm_step */
  int core_class;          /* DM_CORES_*: DM_CORES_PERF caps threads at the performance cores (sysctl hw.perflevel0) */
  int readers;             /* expert reader threads; 0 = 4 */
  int64_t cache_bytes;     /* expert cache budget in bytes; 0 = the profile's: on macOS a profile's automatic cache
                              (mac: cache auto, and a phone profile's slab) with DM_MEM_MLOCK is sized by the Mac's rule
                              (profiles.mac.cache_start and the ceiling, as dmeng and dmserve), else 3.3e9; on iOS (not DM_MEM_FILE) a slab
                              sized once at open by the profile's cache_slab rule (presets.h SlabRule): "available" =
                              os_proc_available_memory() - resident file - router file - 0.3 GB (state, scratch) - 0.9 GB
                              (margin) (iPhone 15 Plus: at 4.27 GB available a 1.0 GB slab drew no warning, 1.3 GB did), or
                              "footprint_target" = the device class's footprint target - the model's non-cache footprint
                              (phone-small: 2.96 GB less 1.54 / 1.26 GB for the 35B / Inkling); clamped to [0.6, 8.0] GB */
  int memory_mode;         /* DM_MEM_* of the expert cache; default DM_MEM_MLOCK on both (on the iPhone 15 Plus the FILE tier
                              read at 0.56 GB/s against 0.74 for a locked slab, and the OS kept 0.85 GB of it) */
  int res_memory_mode;     /* resident weights: DM_MEM_MLOCK, DM_MEM_PLAIN, or DM_MEM_FILE (mapped read-only from the
                              resident file: clean file-backed pages, not charged as dirty memory); default MLOCK
                              (iOS too: an unlocked resident is compressed by the OS under read pressure and
                              every token decompresses it, 0.47 tok/s; mapped it is re-read from flash every token) */
  const char* routers_id;  /* early naming in decode by a trained router of the manifest: NULL or "" = the profile's
                              (profile "none": no routers), "none" = no routers, "default" = the manifest's default, or an id: "linear_k2",
                              "linear_k4", "mlp_k2_h512", "mlp_k4_h512", ... (derived file <derived_dir>/router.<id>.q8...) */
  const char* routers;     /* an explicit engine router file (.dmr); overrides routers_id */
  int router_m;            /* experts named early per step; 0 = 8 */
  const char* derived_dir; /* the engine's derived files (8-bit resident, routers); NULL = <model>/.dmeng */
  const char* resident;    /* explicit resident file; NULL = resolved from the model folder */
  const char* embeddings;  /* embedding rows from this safetensors file (the table alone, e.g. embeddings.safetensors,
                              relative to the model folder or absolute); NULL = the manifest's file, or
                              <model>/embeddings.safetensors when that is absent (the phone: no bf16 resident) */
  const char* disk_dir;    /* prefix-cache disk copies (system prompt states); NULL = none */
  double disk_gb;          /* cap of disk_dir; 0 = 4 */
  double ckpt_gb;          /* prefix cache: memory for checkpoints on the live sequence (one is the fixed state, 66 MB
                              for Qwen3.6, 173 MB for Inkling); 0 = 0.25 on macOS; on iOS the pinned system boundary and the newest, whatever their
                              size */
  double park_gb;          /* prefix cache: memory for parked sequences; 0 = 0.9 on macOS, 0.3 on iOS */
  int ctx;                 /* max positions; 0 = the profile's declared ctx (profiles.<p>.ctx), else 16384 */
  int file_check;          /* DM_MEM_FILE: a held record is checked on use: 0 never, 1 first + last page (default), 2 all */
  int file_evict_hint;     /* DM_MEM_FILE, on eviction: 0 nothing (default: pages are left to the OS), 1 MADV_DONTNEED,
                              2 msync(MS_INVALIDATE): the record's clean pages leave the file cache at once (iOS) */
  int file_adapt;          /* DM_MEM_FILE: the budget follows what the OS keeps (shrinks toward the resident part of
                              the held set while refaults exceed 1 per token, grows back while there are none);
                              cache_bytes is the start and the max. Default 0 (1 on iOS) */
  int64_t file_min_bytes;  /* DM_MEM_FILE adaptive floor; 0 = 1e9 */
  int64_t file_core_bytes; /* DM_MEM_FILE: a locked core of the hottest records (mlock), served from memory; 0 = none */
  /* the eviction policy, the read planner, the reader QoS. Zero keeps the defaults. */
  const char* evict;       /* keep-score: NULL / "" = the core's default "auto" (the hazard table, with layer_caps 0:
                              uniform per-layer caps only while the cache holds fewer than 4 x K x layers slots),
                              "hazard", "freq" (the decayed use count, one pool), "pred" */
  int layer_caps;          /* 0 = the core's default (auto with hazard), 1 = uniform per-layer caps, 2 = one pool */
  int plan;                /* early naming by the read planner over the router (routers / routers_id) instead of a fixed
                              top router_m: 0 = off (default: fixed top router_m), 1 = names allowed by the marginal-cost
                              threshold from the measured drive rate and compute time, 2 = a per-token byte bucket */
  double plan_u;           /* planner: the drive-load target of rule 1; 0 = 0.9 */
  int reader_qos;          /* DM_QOS_* + 1 of the reader threads; 0 = user-initiated (default) */
  double router_tau;       /* fixed routers: > 0 = name the top router_cand whose calibrated probability >= tau (the
                              confidence rule) instead of a fixed top router_m; 0 = the profile's; < 0 = off (a plain top
                              router_m whatever the profile) */
  int router_cand;         /* candidates of the confidence rule; 0 = 16 */
  int split;               /* pieces per record read; 0 = 1 */
  const char* mode;        /* NULL = the profile's; "exact" (= "lossless") or "fast" (the model's declared leave-out rule,
                              approx:drop[:w=W]; approximate: tokens can differ from the exact ones); or that rule as an
                              engine string, "approx:drop:w=0.12" */
  int resident_bits;       /* 0 = the profile's (8-bit; phone-small 35B: 4-bit when the file is there); 8 = the 8-bit
                              resident (resident_q8); 4 = the 4-bit variant resident_q4
                              (<derived_dir>/resident_q4.dmres1.<12 hex of resident.safetensors' sha256>.bin, made by
                              `ms_derive resident DIR --bits 4`; for memory-limited devices: tokens differ from 8-bit).
                              The file opened is reported in dm_stats_json: policy.resident_bits / policy.resident and
                              resolved.resident_bits / resolved.resident (8 | 4; 0 and "" for another file given) */
  int prefill_threads;     /* compute threads during dm_sync (the prompt's prefill); 0 = threads (6 measured slower
                              than 4 on the 15 Plus for a 472-token prompt with the 4-bit resident) */
  int disk_min;            /* prefix cache: write a disk copy for a marked prefix of at least this many positions;
                              0 = 1024 (iOS: 16, so a chat's prefix is kept across launches) */
  int preload_experts;     /* prefix cache: a disk restore also reads the expert set saved with it; 0 = off */
  const char* profile;     /* NULL / "" / "auto" = from the device; "mac", "phone", "phone-small"; "none" = no profile */
  const char* xtok;        /* cross-token routers (engine file): NULL = the profile's (Inkling: source 47, with priority
                              reads), "none" = off */
  const char* model_options; /* "name=value;name=value": the model's own options (Model::set_option), applied after load;
                              NULL = none (iOS: prefill_release=1, the batched prefill's scratch freed after each prompt) */
  /* Batched prefill split with the GPU (Metal; exact: the GPU's rows are the CPU kernels' bits, so tokens do not
   * change). The GPU computes this share of each matrix's rows while the CPU computes the rest, for prefill sub-chunks
   * of at least prefill_gpu_min positions. 0 = the profile's share (profiles.<p>.prefill_gpu: mac 0.4 and 0.4 for the
   * three models; phone and phone-small 0.65 and 0.65 for the 35B, CPU only for Inkling), < 0 = CPU only. Builds without
   * Metal refuse > 0 (the profile's is 0 there). */
  double prefill_gpu_frac;    /* attention projections (8- or 4-bit resident) */
  double prefill_gpu_experts; /* experts' products: routed experts in a locked / plain slab, and the shared expert */
  int prefill_gpu_min;        /* 0 = the model's declared prefill.gpu_min (64 in every published manifest; 256 built in) */
  int prefill_gpu_int_kernel; /* 0 = the matrix-unit kernel (default), 1 = the integer kernel (the same bits; slower) */
  int prefill_gpu_min_wave;   /* a wave of landed experts uses the GPU from this many (position, expert) pairs; 0 = the
                                 model's declared prefill.gpu_min_wave (128 in every published manifest) */
  int run_ahead;              /* run-ahead while a step waits on reads: the picks it names; 0 = the profile's (the 35B and
                                 Inkling: 8 on every profile, on phone and phone-small in exact mode only; Flash-Next's
                                 mac: 10), < 0 = off */
  int run_ahead_depth;        /* its depth in layers; 0 = the profile's (1) */
  double run_ahead_budget;    /* its speculative ms per token at most this x the running-average compute per token;
                                 0 = the profile's (phones, exact: 0.2), < 0 = no limit */
  int run_ahead_max;          /* at most this many run-aheads per token; 0 = the profile's, < 0 = no limit */
  int run_ahead_sat_skip;     /* 1: no run-ahead while every reader has a read in flight and reads wait behind them;
                                 0 = the profile's (phones, exact: on), < 0 = off */
  int run_ahead_ab;           /* measurement: 1 = run-ahead skipped or allowed token by token in this process (a fixed
                                 pseudo-random choice per dm_step, dmeng's DMENG_RA_AB), the two arms in dm_stats_json
                                 "ra_ab" (tokens, ms, wait and MB read per token each) */
  int reader_io_low;          /* the reader threads' disk I/O one tier below the app's (IOPOL_STANDARD; CPU QoS unchanged):
                                 0 = the platform's default (on on iOS: the UI's page-ins go first; off on macOS), 1 = on,
                                 -1 = off */
} dm_open_opts;

void dm_default_opts(dm_open_opts* o);

/* The prefill GPU split's self-test on this device (no model): both GPU kernels against the CPU kernels bit for bit,
 * then rates (GMAC/s) of each GPU kernel alone, the CPU's 4-bit batched kernel alone on `threads`, and both at once.
 * About 8 x seconds. JSON: "exact" true/false, "rates_gmacs" {...}; {"gpu":false} without a GPU. The string is valid
 * until the next call on the same thread. */
const char* dm_gpu_check(int threads, double seconds);

/* Open a moved-store model folder (moved-store.json, store/, resident or derived files, routers/). NULL on failure;
 * the reason is in dm_last_error(NULL). */
dm_engine* dm_open(const char* model_dir, const dm_open_opts* opts);
void dm_close(dm_engine* e);
const char* dm_last_error(dm_engine* e); /* NULL: the calling thread's last dm_open failure */

typedef struct dm_info_t {
  int vocab, max_ctx, n_layers, n_experts;
  int pos;              /* positions in the live state */
  int64_t fixed_bytes;  /* fixed (recurrent) state size */
  int64_t pos_bytes;    /* positional state bytes per position */
  int threads, readers, memory_mode, router_m;
  int64_t cache_bytes;  /* the budget in force */
  char name[64];
  char routers[256];    /* the router file in use ("" = none) */
} dm_info_t;
int dm_info(dm_engine* e, dm_info_t* out);

/* Prefill progress: positions done (moves at least once per layer of a long prompt), positions this call computes,
 * MB (1e6 B) of expert records issued so far. Called on the caller's thread inside dm_sync. */
typedef void (*dm_progress_fn)(void* user, int done, int total, double mb_read);

typedef struct dm_sync_result {
  int reused, computed;
  char source[16];      /* "live" | "checkpoint" | "parked" | "disk" | "cold" */
  double ms_total, ms_restore, ms_prefill, ms_wait_reads;
  double mb_read;       /* MB of expert records the prefill issued */
  int cancelled;
} dm_sync_result;

/* Bring the live state to ids[0..n) with the least recomputation (live prefix, checkpoint, parked sequence, disk copy,
 * else cold), and write the logits after ids[n-1] (vocab floats). logits NULL = state only (an idle-time prefix the next
 * call extends: a saved state at n itself is restored whole, no position recomputed; dm_step must not follow until a
 * dm_sync with logits). marks[0..nmarks): positions (1..n) where a message ends (checkpoint candidates; the first >=
 * disk_min is the disk prefix, its file pinned). Pass the system prompt's boundary first, the same in every call: that
 * checkpoint is pinned in memory (never evicted). At most one checkpoint is pinned: a call whose first mark differs
 * (a typing position, say) pins its own and releases the previous one, which then counts against ckpt_gb like any other
 * checkpoint; dm_stats_json's prefix_cache reports pinned_checkpoints, pinned_mb, pins_released, pins_over_budget (a
 * pin larger than the whole budget: kept, and counted) and checkpoint_mb against checkpoint_budget_mb.
 * dm_persist(pos) at the live end (after decode) checkpoints it first. Returns DM_ECANCELLED if
 * dm_cancel stopped it (within one layer); the state is then consistent at dm_info's pos. */
int dm_sync(dm_engine* e, const int32_t* ids, int n, const int32_t* marks, int nmarks, float* logits,
            dm_progress_fn progress, void* user, dm_sync_result* res);

/* One generated token: append id, logits after it (vocab floats, every one computed by the full output head).
 * DM_ECANCELLED (nothing computed) if a cancel is pending. */
int dm_step(dm_engine* e, int32_t id, float* logits);
/* The last dm_sync's prefill account as JSON (source, reused / computed, wall and read-wait ms, MB, and the model's own
 * per-stage detail: union per layer, names used / unused, ...). Valid until the next dm_sync. */
const char* dm_last_prefill_json(dm_engine* e);
/* Stop the running dm_sync (within one layer) or make the next dm_sync / dm_step return DM_ECANCELLED at once; the
 * request is consumed by the call that returns DM_ECANCELLED. Any thread. */
void dm_cancel(dm_engine* e);
/* Drop a cancel that found no dm_sync / dm_step to stop (dm_cancel is sticky until the next call consumes it): a caller
 * that cancels background work (an idle-time prefix) queues this after the cancel, before its next real call. */
void dm_cancel_clear(dm_engine* e);
/* Write the disk copy of the live sequence's prefix [0, pos) if pos is a checkpointed position. */
int dm_persist(dm_engine* e, int pos);

/* Sampling, in the order of transformers / mlx-lm: penalties on the raw logits over history[0..n_history) (the
 * tokens generated so far in this reply), temperature, top-k, then top-p and min-p on the tempered distribution of the
 * top-k survivors, one draw. The draw uses a generator seeded by (seed, n_history), so a reply is a pure function of its
 * logits, its seed and its tokens. temp <= 0 and no penalties: argmax (lowest index on ties). */
typedef struct dm_sampler {
  double temp;               /* <= 0: greedy */
  int top_k;                 /* 0: off */
  double top_p;              /* 1: off */
  double min_p;              /* 0: off */
  double presence_penalty;   /* 0: off; once per distinct token in history */
  double repetition_penalty; /* 1: off; logit > 0 ? logit / r : logit * r */
  double frequency_penalty;  /* 0: off; times the count in history */
  uint64_t seed;
} dm_sampler;
void dm_sampler_default(dm_sampler* s); /* greedy */
int32_t dm_sample(const float* logits, int vocab, const dm_sampler* s, const int32_t* history, int n_history);
int32_t dm_argmax(const float* logits, int vocab);

/* Drafting (docs/ADVANCED.md, "Drafting"): one pass verifies several tokens and keeps exactly the tokens plain decoding
 * would produce; the controller, its sources and its statistics are dmeng's and dmserve's (include/dmeng/draft.h).
 * Whether it is on is resolved at dm_open as in dmeng and dmserve: model_options "run:draft=MODE" (none | lookup | head |
 * both | fixed; "run:drafts=1" = lookup, "run:drafts=0" = none) > the profile's declared "drafts" {"lookup", "head"} >
 * off. lookup: copy drafts from text already in the sequence (the prompt, the reply so far, earlier replies of the
 * session) plus the fragments of dm_set_draft_fragments; head: the model's own draft head (Inkling's mtp.safetensors);
 * both: the two. "run:draft-max=N" (1..7, default the profile's drafts.max, else 7): drafts per pass at most.
 * "run:draft-cost=FILE|none": the pass costs measured (read at open, written at the end of each generation; default
 * under a profile <derived_dir>/draft_cost.txt, as dmserve). A family without a multi-position verify, or a head the
 * folder does not have, fails dm_open (dm_last_error says why), the declared default too. With drafting off nothing of
 * it is allocated and every call behaves as before. On iOS the slab sizing rule gives drafting's memory (a verify
 * pass's buffers, the head's weights) out of the slab (dm_stats_json policy.slab_rule.draft_reserve_gb).
 *
 * dm_step_n: one decode step that may produce several tokens. Feeds `id` (the token just chosen, as dm_step) and writes
 * the next tokens to out[0 .. *n_out), 1 <= *n_out <= max_out, each exactly the token the app's own loop
 * would choose there:
 *   out[0] = dm_sample(logits after id, s, history, n_history)
 *   out[i] = dm_sample(logits after out[i-1], s, history + out[0 .. i), n_history + i)
 * (history: this reply's tokens so far, id included, as the app passes dm_sample; s NULL = greedy: dm_argmax, as is a
 * sampler with temp <= 0 and no penalties; with penalties dm_sample applies them before its argmax). The state then holds
 * every position up to out[*n_out - 2]: the next call feeds out[*n_out - 1]. Without drafting it is dm_step followed by
 * dm_sample (*n_out = 1). Stops after a token in stop[0 .. n_stop) (it is the last one written). max_out bounds the
 * tokens of this call (out holds max_out; 8 is the most a call produces): pass 1 for a step whose next token the app
 * forces, or the number of tokens up to a token it will force. logits (NULL = not wanted): the logits out[*n_out - 1]
 * was drawn from (every one). DM_ECANCELLED (nothing computed) if a cancel is pending. dm_counters' decode_steps and step_ns count
 * the positions fed and the time inside the call (dm_sample's included).
 *
 * The screened output head (exact): with logits NULL, dm_step_n computes only the output rows its sampler can draw from
 * when that is provable from dm_sample's order of operations: s NULL or temp <= 0 without penalties (the argmax), or
 * temp > 0 with 1 <= top_k <= 64 and no penalties (the top_k largest logits, bit for bit; top-p, min-p and temperature
 * read only those). Any penalty, top_k 0 or above 64, or logits wanted: the full head. The tokens are the full head's
 * in every case. dm_sync and dm_step always compute every logit. It is built at dm_open (a model that cannot split its
 * head, e.g. a mapped resident, keeps the full head: resolved.screened_head false, policy.screened_head_why says why);
 * model_options "run:fast-head=0" (or the environment's DMENG_CAPI_FAST_HEAD=0) leaves it off for measurement.
 * dm_stats_json policy.head_calls counts the decode calls by head (full, screened). */
int dm_step_n(dm_engine* e, int32_t id, const dm_sampler* s, const int32_t* history, int n_history, const int32_t* stop,
              int n_stop, int max_out, int32_t* out, int* n_out, float* logits);
/* The text lookup drafts copy from besides the sequence itself: n fragments of token ids, fragment i = lens[i] ids,
 * concatenated in ids (e.g. the tool-call format of a request's tools, dmserve's skeleton source). Kept for the following
 * dm_step_n calls until replaced; n = 0 clears them. Without drafting they are kept and not used. */
int dm_set_draft_fragments(dm_engine* e, const int32_t* ids, const int32_t* lens, int n);

/* Change the expert cache budget at run time (any thread). Applied at once if the engine is idle, else at the next
 * token or prefill-layer boundary. Lower: entries are dropped and their memory given back (DM_MEM_FILE: bookkeeping
 * only, the lowest-value entries first); higher: up to the budget given at dm_open. With file_adapt it is also the new
 * ceiling of the adaptive budget. */
int dm_set_cache_bytes(dm_engine* e, int64_t bytes);

/* Counters since dm_open or the last dm_stats_json (which starts them again; bytes_read and decode_steps never restart),
 * for per-token deltas inside a reply. */
typedef struct dm_counters_t {
  uint64_t picks, hits, late, misses;   /* released picks: in memory, in flight, read on the spot */
  uint64_t bytes_issued;                /* expert records named or missed (bytes) */
  uint64_t bytes_read;                  /* bytes the readers actually read (DM_MEM_FILE: resident records skip) */
  uint64_t file_refaults;               /* DM_MEM_FILE: held records whose pages the OS had dropped (read again) */
  double wait_ms;                       /* compute thread waiting on reads */
  int64_t held_bytes;                   /* records the policy holds now */
  int64_t budget_bytes;                 /* the budget in force (adaptive in DM_MEM_FILE) */
  int64_t decode_steps;
  uint64_t bytes_spare;                 /* expert bytes the OS file cache served instead of the drive (spare memory: uses
                                           of a probation record after its first, and reads whose pages were resident;
                                           DM_MEM_FILE: resident records not read) */
  uint64_t read_busy_ns;                /* wall time with at least one expert read in flight (drive rate = bytes_read / it) */
  int spare_memory;                     /* the file cache under the slab: 0 off, 1 on, 2 paused (dm_spare_memory_pause) */
  uint64_t step_ns;                     /* wall time inside dm_step (steady clock), summed: decode_steps / it = the rate */
  uint64_t bytes_named_unused;          /* named records evicted before any use (records x the first pool's record bytes):
                                           reads that served nothing */
  uint64_t promotions;                  /* spare memory: probation records moved into the slab at their reuse */
  uint64_t probation_reclaimed;         /* spare memory: probation records the OS reclaimed before their reuse (read again) */
  uint64_t pageins;                     /* the compute's page-ins since dm_open: the process's faults that read the drive,
                                           less those the readers took bringing probation records in */
  int64_t probation_bytes;              /* spare memory: records held in probation now (in the file cache, not the app's) */
} dm_counters_t;
int dm_counters(dm_engine* e, dm_counters_t* out);

/* Spare memory (the model's files that the system keeps in memory the app does not own are used before reading the
 * drive: records the slab has no room for are held in the OS file cache): on by default on the phone profile with at
 * least 11 GB of physical memory, in both modes (the profile's spare_memory rule, resolved at dm_open
 * for the mode of that open; resolved.spare_memory / spare_memory_from / spare_memory_rule in dm_stats_json);
 * model_options "run:spare-mem=0|1" (the app's switch) forces it off or on in either mode, "run:spare-gb=X" its budget.
 * The mode is fixed per engine: a mode change is a new dm_open, and dm_close drops the file-cache pages the engine held.
 * dm_spare_memory_pause: for the rest of the session no read lands in the file cache and the records held there are
 * dropped (at once when the engine is idle, else at its next step); reads go to the slab as without it (a memory
 * warning). Any thread; idempotent. DM_EINVAL if spare memory is not on. */
int dm_spare_memory_pause(dm_engine* e);

/* The store's pages the OS file cache holds now (bytes, mincore over the mapped store files): DM_MEM_FILE or spare
 * memory under the slab (paused included); -1 when the store is not mapped (a locked slab alone reads past the file
 * cache). Any thread. */
int64_t dm_store_resident_bytes(dm_engine* e);

/* JSON of what ran since the last call (decode ms per stage, hit rate, MB read, prefix-cache counters, memory: process
 * footprint, the policy's held GB and, in DM_MEM_FILE, the store pages resident in the file cache by mincore; with
 * drafting, "draft": dmeng's counters (offered, declined, drafted, accepted, verifies, tokens per step, the controller's
 * cost model), dm_step_n's calls and tokens per call, its memory (the first verify pass's footprint rise, the head's
 * weights and state, the slab's reserve), and policy.draft / draft_max). The
 * string is owned by the engine and valid until the next call on it. */
const char* dm_stats_json(dm_engine* e);

/* Built-in benches on the open engine's store; JSON into a malloc'd *json_out (free with free()).
 *   "filecache": cold records read by buffered pread into scratch, then mincore and a full read through the mapping
 *                (spec: "n=24"): pread GB/s, resident fraction after the pread, first-touch ms and faults.
 * DM_ENOTIMPL for other kinds (the kernel benches live in the phone app's DMBench target). */
int dm_bench(dm_engine* e, const char* kind, const char* spec, char** json_out);

#ifdef __cplusplus
}
#endif
#endif /* DMENG_C_H */
