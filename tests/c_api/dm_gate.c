/* The P1 gate through the C API only (bindings/c/dmeng_c.h): open a model folder, dm_sync a prompt (token ids), decode
 * greedily with dm_step + dm_argmax, write the ids. Optional: change the cache budget mid-decode (--resize-at T --resize-gb
 * G; lossless, so the ids must not change), and a cancel check (--cancel-test: a dm_sync stopped from another thread
 * returns DM_ECANCELLED with a consistent state, and the next dm_sync of the same prompt gives the uncancelled logits).
 *
 *   dm_gate --model DIR --prompt-ids FILE --n 400 --out OUT.ids [--mem file|mlock|plain] [--cache-gb 3.3]
 *           [--routers-id mlp_k2_h512] [--threads 8] [--readers 4] [--json OUT.json] [--embeddings F]
 *           [--file-adapt 1 --file-min-gb G] [--file-core-gb G]
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "dmeng_c.h"

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int read_ids(const char* path, int32_t** out) {
  FILE* f = fopen(path, "r");
  if (!f) return -1;
  int cap = 1024, n = 0, v;
  int32_t* a = malloc(cap * sizeof *a);
  while (fscanf(f, "%d", &v) == 1) {
    if (n == cap) a = realloc(a, (cap *= 2) * sizeof *a);
    a[n++] = v;
    int c = fgetc(f);
    while (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t') c = fgetc(f);
    if (c == EOF) break;
    ungetc(c, f);
  }
  fclose(f);
  *out = a;
  return n;
}

static void on_progress(void* user, int done, int total, double mb) {
  int* last = (int*)user;
  if (done - *last >= 64 || done == total) {
    fprintf(stderr, "prefill %d / %d (%.0f MB)\n", done, total, mb);
    *last = done;
  }
}

struct canceller {
  dm_engine* e;
  int after_ms;
};
static void* cancel_later(void* p) {
  struct canceller* c = p;
  usleep(c->after_ms * 1000);
  dm_cancel(c->e);
  return NULL;
}

int main(int argc, char** argv) {
  const char *model = NULL, *prompt = NULL, *out = NULL, *json = NULL, *mem = "mlock", *rid = NULL, *emb = NULL;
  int adapt = 0;
  double min_gb = 0, core_gb = 0;
  int n = 400, threads = 0, readers = 0, resize_at = -1, cancel_test = 0;
  double gb = 3.3, resize_gb = 0;
  const char* policy_probe = NULL;
  const char *resident = NULL, *mode = NULL;  /* --resident FILE (e.g. the 4-bit resident_q4), --mode approx:drop */
  const char* model_options = NULL;           /* --model-options "k=v;run:FLAG=VALUE" (dm_open_opts.model_options) */
  double gpu_frac = 0, gpu_experts = 0;      /* --gpu-frac F --gpu-experts F: the prefill's exact GPU split */
  int gpu_min = 0, prefill_threads = 0;      /* --gpu-min N (sub-chunks of at least N positions), --prefill-threads N */
  int gpu_int = 0;                           /* --gpu-int-kernel: the integer GPU kernel instead of the matrix-unit one */
  int gpu_wave = 0;                          /* --gpu-min-wave N: expert waves of at least N assignments use the GPU */
  const char *profile = NULL, *evict = NULL, *routers = NULL;  /* --profile NAME, --evict P, --routers FILE|none */
  int router_m = 0, router_cand = 0;          /* --m M, --rt-cand C (0: the profile's) */
  double router_tau = 0;                     /* --rt-tau T (0: the profile's; < 0: a plain top m) */
  int run_ahead = 0;                         /* --run-ahead N (0: the profile's; < 0: off) */
  double ra_budget = 0;                      /* --run-ahead-budget X --run-ahead-max N --run-ahead-sat-skip 0|1 */
  int ra_max = 0, ra_sat = 0;
  for (int i = 1; i < argc; i++)  /* --gpu-check [SECONDS]: the GPU kernels' self-test (no model), then exit */
    if (!strcmp(argv[i], "--gpu-check")) {
      printf("%s\n", dm_gpu_check(0, i + 1 < argc ? atof(argv[i + 1]) : 1.0));
      return 0;
    }
  for (int i = 1; i < argc; i++) {
    const char* k = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : NULL;
    if (!strcmp(k, "--model")) model = v, i++;
    else if (!strcmp(k, "--prompt-ids")) prompt = v, i++;
    else if (!strcmp(k, "--out")) out = v, i++;
    else if (!strcmp(k, "--json")) json = v, i++;
    else if (!strcmp(k, "--mem")) mem = v, i++;
    else if (!strcmp(k, "--routers-id")) rid = v, i++;
    else if (!strcmp(k, "--n")) n = atoi(v), i++;
    else if (!strcmp(k, "--threads")) threads = atoi(v), i++;
    else if (!strcmp(k, "--readers")) readers = atoi(v), i++;
    else if (!strcmp(k, "--cache-gb")) gb = !strcmp(v, "auto") ? 0 : atof(v), i++;  /* auto: the profile's rule */
    else if (!strcmp(k, "--profile")) profile = v, i++;
    else if (!strcmp(k, "--evict")) evict = v, i++;
    else if (!strcmp(k, "--routers")) routers = v, i++;
    else if (!strcmp(k, "--m")) router_m = atoi(v), i++;
    else if (!strcmp(k, "--rt-tau")) router_tau = atof(v), i++;
    else if (!strcmp(k, "--rt-cand")) router_cand = atoi(v), i++;
    else if (!strcmp(k, "--run-ahead")) run_ahead = atoi(v), i++;
    else if (!strcmp(k, "--run-ahead-budget")) ra_budget = atof(v), i++;
    else if (!strcmp(k, "--run-ahead-max")) ra_max = atoi(v), i++;
    else if (!strcmp(k, "--run-ahead-sat-skip")) ra_sat = atoi(v) ? 1 : -1, i++;
    else if (!strcmp(k, "--resize-at")) resize_at = atoi(v), i++;
    else if (!strcmp(k, "--resize-gb")) resize_gb = atof(v), i++;
    else if (!strcmp(k, "--cancel-test")) cancel_test = 1;
    else if (!strcmp(k, "--file-adapt")) adapt = atoi(v), i++;
    else if (!strcmp(k, "--file-min-gb")) min_gb = atof(v), i++;
    else if (!strcmp(k, "--file-core-gb")) core_gb = atof(v), i++;
    else if (!strcmp(k, "--embeddings")) emb = v, i++;
    else if (!strcmp(k, "--policy-probe")) policy_probe = v, i++;
    else if (!strcmp(k, "--resident")) resident = v, i++;
    else if (!strcmp(k, "--model-options")) model_options = v, i++;
    else if (!strcmp(k, "--mode")) mode = v, i++;
    else if (!strcmp(k, "--gpu-frac")) gpu_frac = atof(v), i++;
    else if (!strcmp(k, "--gpu-experts")) gpu_experts = atof(v), i++;
    else if (!strcmp(k, "--gpu-min")) gpu_min = atoi(v), i++;
    else if (!strcmp(k, "--gpu-int-kernel")) gpu_int = 1;
    else if (!strcmp(k, "--gpu-min-wave")) gpu_wave = atoi(v), i++;
    else if (!strcmp(k, "--prefill-threads")) prefill_threads = atoi(v), i++;
    else {
      fprintf(stderr, "unknown flag %s\n", k);
      return 2;
    }
  }
  if (!model || !prompt) {
    fprintf(stderr, "usage: dm_gate --model DIR --prompt-ids FILE [--n N] [--out F] [--mem file|mlock|plain] ...\n");
    return 2;
  }
  int32_t* ids = NULL;
  const int np = read_ids(prompt, &ids);
  if (np <= 0) {
    fprintf(stderr, "no prompt ids in %s\n", prompt);
    return 2;
  }
  dm_open_opts o;
  dm_default_opts(&o);
  o.threads = threads;
  o.readers = readers;
  o.cache_bytes = (int64_t)(gb * 1e9);
  o.memory_mode = !strcmp(mem, "file") ? DM_MEM_FILE : !strcmp(mem, "plain") ? DM_MEM_PLAIN : DM_MEM_MLOCK;
  o.routers_id = rid;
  o.routers = routers;
  o.profile = profile;
  o.evict = evict;
  o.router_m = router_m;
  o.router_tau = router_tau;
  o.router_cand = router_cand;
  o.run_ahead = run_ahead;
  o.run_ahead_budget = ra_budget;
  o.run_ahead_max = ra_max;
  o.run_ahead_sat_skip = ra_sat;
  o.embeddings = emb;
  o.resident = resident;
  if (model_options) o.model_options = model_options;
  o.mode = mode;
  o.file_adapt = adapt;
  o.file_min_bytes = (int64_t)(min_gb * 1e9);
  o.file_core_bytes = (int64_t)(core_gb * 1e9);
  o.ctx = 16384;
  o.prefill_gpu_frac = gpu_frac;
  o.prefill_gpu_experts = gpu_experts;
  o.prefill_gpu_min = gpu_min;
  o.prefill_gpu_int_kernel = gpu_int;
  o.prefill_gpu_min_wave = gpu_wave;
  o.prefill_threads = prefill_threads;
  double t0 = now_ms();
  dm_engine* e = dm_open(model, &o);
  if (!e) {
    fprintf(stderr, "dm_open: %s\n", dm_last_error(NULL));
    return 1;
  }
  const double load_ms = now_ms() - t0;
  dm_info_t inf;
  dm_info(e, &inf);
  fprintf(stderr, "open %s: vocab %d, %d layers x %d experts, threads %d, readers %d, cache %.2f GB, mem %d, routers %s (%.1f s)\n", inf.name,
          inf.vocab, inf.n_layers, inf.n_experts, inf.threads, inf.readers, inf.cache_bytes / 1e9, inf.memory_mode,
          inf.routers[0] ? inf.routers : "none", load_ms / 1e3);
  float* lg = malloc(sizeof(float) * inf.vocab);
  int cancel_ok = -1;
  if (cancel_test) {  // stop a cold prefill of the prompt 300 ms in; the state must be consistent
    pthread_t th;
    struct canceller c = {e, 300};
    pthread_create(&th, NULL, cancel_later, &c);
    dm_sync_result cr;
    int rc = dm_sync(e, ids, np, NULL, 0, lg, NULL, NULL, &cr);
    pthread_join(th, NULL);
    dm_info(e, &inf);
    fprintf(stderr, "cancel test: rc %d, computed %d, state at pos %d of %d\n", rc, cr.computed, inf.pos, np);
    cancel_ok = rc == DM_ECANCELLED && inf.pos < np;
  }
  dm_sync_result sr;
  int last = 0;
  double tp = now_ms();
  int rc = dm_sync(e, ids, np, NULL, 0, lg, on_progress, &last, &sr);
  const double prefill_ms = now_ms() - tp;
  if (rc != DM_OK) {
    fprintf(stderr, "dm_sync: %d\n", rc);
    return 1;
  }
  int32_t* gen = malloc(sizeof(int32_t) * (n + 1));
  gen[0] = dm_argmax(lg, inf.vocab);
  dm_counters_t c0, c1;
  dm_stats_json(e);  // decode statistics from here
  dm_counters(e, &c0);
  double td = now_ms();
  int k = 1;
  for (; k < n; k++) {
    if (k == resize_at && resize_gb > 0) {
      dm_set_cache_bytes(e, (int64_t)(resize_gb * 1e9));
      fprintf(stderr, "cache budget -> %.2f GB at token %d\n", resize_gb, k);
    }
    if (dm_step(e, gen[k - 1], lg) != DM_OK) break;
    gen[k] = dm_argmax(lg, inf.vocab);
    if (k % 50 == 0) {
      dm_counters_t c;
      dm_counters(e, &c);
      fprintf(stderr, "token %d: budget %.2f GB, held %.2f GB, refaults so far %llu\n", k, c.budget_bytes / 1e9, c.held_bytes / 1e9,
              (unsigned long long)c.file_refaults);
    }
  }
  const double dec_ms = now_ms() - td;
  dm_counters(e, &c1);
  const char* st = dm_stats_json(e);
  if (out) {
    FILE* f = fopen(out, "w");
    for (int i = 0; i < k; i++) fprintf(f, "%s%d", i ? "," : "", gen[i]);
    fprintf(f, "\n");
    fclose(f);
  }
  const int steps = k - 1;
  const double wait = c1.wait_ms - c0.wait_ms;
  const uint64_t picks = c1.picks - c0.picks;
  char head[2048];
  snprintf(head, sizeof head,
           "{\"engine\":\"dm_gate (C API)\",\"mem\":\"%s\",\"cache_gb\":%.2f,\"routers\":\"%s\",\"threads\":%d,\"readers\":%d,\"prompt_tokens\":%d,"
           "\"load_s\":%.3f,\"prefill_s\":%.3f,\"prefill_source\":\"%s\",\"prefill_mb_read\":%.1f,\"tokens\":%d,\"decode_s\":%.3f,"
           "\"tok_s\":%.3f,\"ms_per_token\":%.2f,\"wait_ms_per_token\":%.2f,\"compute_ms_per_token\":%.2f,\"hit_rate\":%.4f,"
           "\"mb_issued_per_token\":%.2f,\"mb_preads_per_token\":%.2f,\"file_refaults\":%llu,\"cancel_test\":%d,\"stats\":",
           mem, gb, inf.routers, inf.threads, inf.readers, np, load_ms / 1e3, prefill_ms / 1e3, sr.source, sr.mb_read, k, dec_ms / 1e3,
           steps / (dec_ms / 1e3), dec_ms / steps, wait / steps, (dec_ms - wait) / steps, picks ? (double)(c1.hits - c0.hits) / picks : 0,
           (c1.bytes_issued - c0.bytes_issued) / 1e6 / steps, (c1.bytes_read - c0.bytes_read) / 1e6 / steps,
           (unsigned long long)(c1.file_refaults - c0.file_refaults), cancel_ok);
  const char* pj = dm_last_prefill_json(e);  /* the prefill's account (with the GPU split's, when on) */
  printf("%s%s,\"prefill\":%s}\n", head, st, pj && *pj ? pj : "{}");
  if (json) {
    FILE* f = fopen(json, "w");
    fprintf(f, "%s%s,\"prefill\":%s}\n", head, st, pj && *pj ? pj : "{}");
    fclose(f);
  }
  if (policy_probe) {  /* the policy the core chose at each budget (dm_set_cache_bytes applies at once when idle) */
    char buf[256];
    strncpy(buf, policy_probe, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    for (char* t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
      dm_set_cache_bytes(e, (int64_t)(atof(t) * 1e9));
      const char* js = dm_stats_json(e);
      const char* p = strstr(js, "\"policy\":");
      const char* q = p ? strchr(p, '}') : NULL;
      fprintf(stderr, "POLICY at %s GB: %.*s\n", t, p && q ? (int)(q - p + 1) : 0, p ? p : "");
    }
  }
  dm_close(e);
  return cancel_ok == 0 ? 3 : 0;
}
