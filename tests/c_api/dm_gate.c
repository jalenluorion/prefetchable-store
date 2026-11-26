/* The P1 gate through the C API only (bindings/c/dmeng_c.h): open a model folder, dm_sync a prompt (token ids), decode
 * greedily with dm_step + dm_argmax, write the ids. Optional: change the cache budget mid-decode (--resize-at T --resize-gb
 * G; lossless, so the ids must not change), and a cancel check (--cancel-test: a dm_sync stopped from another thread
 * returns DM_ECANCELLED with a consistent state, and the next dm_sync of the same prompt gives the uncancelled logits).
 *
 *   dm_gate --model DIR --prompt-ids FILE --n 400 --out OUT.ids [--mem file|mlock|plain] [--cache-gb 3.3]
 *           [--routers-id mlp_k2_h512] [--threads 8] [--readers 4] [--json OUT.json] [--embeddings F]
 *           [--file-adapt 1 --file-min-gb G] [--file-core-gb G] [--model-options S] [--file-hint 0|1|2]
 *           [--profile P] [--ctx N (0: the profile's ctx, else 16384)]
 *
 * --arms LIST (with --warm-n N): arms of one prompt in ONE process, each its own dm_open .. dm_close, in order, after a
 * warm-up open of N tokens (tier off): "off" (spare memory off: run:spare-mem=0), "on" (run:spare-mem=1), "nothing"
 * (spare memory on, but each record's pages leave the file cache when the slab evicts it: file_evict_hint 2, the OS
 * keeping no spare pages). An arm name followed by "+warm" (on+warm) keeps the file cache the previous arm left; every
 * other arm starts with the store's pages dropped from the file cache (msync MS_INVALIDATE over each store file: no
 * sudo; the resident fraction before and after is printed). Per arm: tok/s, MB per token read by the readers, MB per
 * token served by the file cache, the process's drive bytes per token (proc_pid_rusage diskio, decode only), wait ms
 * per token; the ids are compared across arms (lossless: identical).
 *
 * An arm may carry overrides after its name, each "/KEY=VALUE": mode (exact|fast|...), mem (file|mlock|plain), spare
 * (0|1, replacing the name's), threads, readers, cache (GB), drive (GB/s: run:drive-gbs, the bench-only drive cap), e.g.
 * --arms off/mode=exact,off/mode=fast,on/mode=fast,off/mode=exact/mem=file (an arm without mode= takes --mode: give it).
 * The ids are compared within each mode (an approximate mode's ids legitimately depend on what is in memory); the exit
 * code is 4 when two arms of the exact mode differ. --measure-after N: each arm decodes N tokens before its counters and
 * clock start (n - N measured). --force-exact 1: an arm of another mode is fed the first exact arm's ids (teacher forcing)
 * and reports its top-1 agreement with them over the measured positions (an exact arm should then be the longest).
 * More overrides: /n=N (tokens for this arm), /opt=K:V+K:V (model options K=V), /run=FLAG:V+FLAG:V (run flags, as
 * run:--FLAG=V), /alt=1 (the model folder given by --alt-model instead of --model). Per arm also: the sampled peak
 * footprint (prefill progress and every 10 tokens), the reads split (urgent, named, named and never used, MB per
 * pool), the run-ahead's and the approximate mode's accounts.
 */
#include <dirent.h>
#include <fcntl.h>
#include <libproc.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <mach/mach.h>

#include "dmeng_c.h"

/* the process's physical footprint now (task_vm_info, what iOS charges an app), bytes */
static uint64_t footprint_now(void) {
  task_vm_info_data_t vi;
  mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
  return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &c) == KERN_SUCCESS ? vi.phys_footprint : 0;
}
static uint64_t arm_peak = 0;  /* the arm's sampled peak footprint (prefill progress, every 10 decode tokens) */
static void peak_sample(void) {
  const uint64_t f = footprint_now();
  if (f > arm_peak) arm_peak = f;
}
static void on_progress_peak(void* user, int done, int total, double mb) {
  (void)user, (void)done, (void)total, (void)mb;
  peak_sample();
}

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

/* the store's pages in the file cache (GB, mincore), dropping them first when `drop` (msync MS_INVALIDATE) */
static double store_cache_gb(const char* model, int drop) {
  char dir[4096];
  snprintf(dir, sizeof dir, "%s/store", model);
  DIR* d = opendir(dir);
  if (!d) return -1;
  const size_t pg = (size_t)getpagesize();
  double res = 0;
  struct dirent* de;
  while ((de = readdir(d))) {
    if (de->d_name[0] == '.') continue;
    char p[8192];
    snprintf(p, sizeof p, "%s/%s", dir, de->d_name);
    int fd = open(p, O_RDONLY);
    if (fd < 0) continue;
    struct stat st;
    if (fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
      void* m = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
      if (m != MAP_FAILED) {
        if (drop) msync(m, (size_t)st.st_size, MS_INVALIDATE);
        const size_t np = ((size_t)st.st_size + pg - 1) / pg;
        char* v = malloc(np);
        if (mincore(m, (size_t)st.st_size, v) == 0)
          for (size_t i = 0; i < np; i++) res += (v[i] & MINCORE_INCORE) ? pg : 0;
        free(v);
        munmap(m, (size_t)st.st_size);
      }
    }
    close(fd);
  }
  closedir(d);
  return res / 1e9;
}

static uint64_t disk_read_bytes(void) {
  struct rusage_info_v4 ri;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*)&ri) != 0) return 0;
  return ri.ri_diskio_bytesread;
}

/* the value of "/KEY=" in an arm spec, or NULL */
static const char* arm_opt(const char* arm, const char* key, char* buf, size_t cap) {
  char pat[32];
  snprintf(pat, sizeof pat, "/%s=", key);
  const char* p = strstr(arm, pat);
  if (!p) return NULL;
  p += strlen(pat);
  size_t n = strcspn(p, "/");
  if (n >= cap) n = cap - 1;
  memcpy(buf, p, n);
  buf[n] = 0;
  return buf;
}

static int arm_exact(const char* mode) { return !strcmp(mode, "exact") || !strcmp(mode, "lossless"); }

static uint64_t pageins(void) {
  struct rusage_info_v4 ri;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*)&ri) != 0) return 0;
  return ri.ri_pageins;
}

/* the JSON object after "KEY": in js, copied into out (or "null") */
static void json_obj(const char* js, const char* key, char* out, size_t cap) {
  char pat[64];
  snprintf(pat, sizeof pat, "\"%s\":{", key);
  const char* p = strstr(js, pat);
  snprintf(out, cap, "null");
  if (!p) return;
  p += strlen(pat) - 1;
  int depth = 0;
  const char* q = p;
  for (; *q; q++) {
    if (*q == '{') depth++;
    else if (*q == '}' && --depth == 0) break;
  }
  if (!*q) return;
  const size_t n = (size_t)(q - p + 1) < cap ? (size_t)(q - p + 1) : cap - 1;
  memcpy(out, p, n);
  out[n] = 0;
}

/* one arm: open, prompt, n greedy tokens (the first `after` not measured), close; a JSON line on stdout; the ids into
 * gen (n). force: the ids fed instead of the arm's own (teacher forcing), NULL = free running */
static int run_arm(const char* model, dm_open_opts o, const char* arm, const int32_t* ids, int np, int n, int after,
                   const int32_t* force, int nforce, int32_t* gen, FILE* jf) {
  char mo[1024], v[64], mode_s[64] = "", mem_s[16] = "", drive[64] = "";
  const int nothing = !strncmp(arm, "nothing", 7);
  int on = nothing || !strncmp(arm, "on", 2);
  const int warm = strstr(arm, "+warm") != NULL;
  if (arm_opt(arm, "spare", v, sizeof v)) on = atoi(v) != 0;
  if (arm_opt(arm, "mode", mode_s, sizeof mode_s)) o.mode = mode_s;
  if (arm_opt(arm, "mem", mem_s, sizeof mem_s))
    o.memory_mode = !strcmp(mem_s, "file") ? DM_MEM_FILE : !strcmp(mem_s, "plain") ? DM_MEM_PLAIN : DM_MEM_MLOCK;
  if (arm_opt(arm, "threads", v, sizeof v)) o.threads = atoi(v);
  if (arm_opt(arm, "readers", v, sizeof v)) o.readers = atoi(v);
  if (arm_opt(arm, "cache", v, sizeof v)) o.cache_bytes = (int64_t)(atof(v) * 1e9);
  if (arm_opt(arm, "drive", v, sizeof v)) snprintf(drive, sizeof drive, ";run:drive-gbs=%s", v);
  if (arm_opt(arm, "n", v, sizeof v) && atoi(v) > 1) n = atoi(v);
  char extra[512] = "", ov[256];
  if (arm_opt(arm, "opt", ov, sizeof ov))
    for (char *sv = NULL, *t = strtok_r(ov, "+", &sv); t; t = strtok_r(NULL, "+", &sv)) {
      char* c = strchr(t, ':');
      if (c) *c = '=';
      snprintf(extra + strlen(extra), sizeof extra - strlen(extra), ";%s", t);
    }
  if (arm_opt(arm, "run", ov, sizeof ov))
    for (char *sv = NULL, *t = strtok_r(ov, "+", &sv); t; t = strtok_r(NULL, "+", &sv)) {
      char* c = strchr(t, ':');
      if (c) *c = '=';
      snprintf(extra + strlen(extra), sizeof extra - strlen(extra), ";run:%s", t);
    }
  snprintf(mo, sizeof mo, "%s%srun:spare-mem=%d%s%s", o.model_options ? o.model_options : "", o.model_options && *o.model_options ? ";" : "", on,
           drive, extra);
  o.model_options = mo;
  if (nothing) o.file_evict_hint = 2;
  if (after < 0 || after >= n - 1) after = 0;
  const double before = store_cache_gb(model, 0), at_start = warm ? before : store_cache_gb(model, 1);
  arm_peak = 0;
  double t0 = now_ms();
  dm_engine* e = dm_open(model, &o);
  if (!e) {
    fprintf(stderr, "dm_open (%s): %s\n", arm, dm_last_error(NULL));
    return -1;
  }
  peak_sample();
  const double load_s = (now_ms() - t0) / 1e3;
  dm_info_t inf;
  dm_info(e, &inf);
  float* lg = malloc(sizeof(float) * inf.vocab);
  dm_sync_result sr;
  double tp = now_ms();
  if (dm_sync(e, ids, np, NULL, 0, lg, on_progress_peak, NULL, &sr) != DM_OK) return -1;
  peak_sample();
  const double prefill_s = (now_ms() - tp) / 1e3;
  gen[0] = dm_argmax(lg, inf.vocab);
  dm_counters_t c0, c1;
  int k = 1, agree = 0, forced = 0;
  for (; k <= after; k++) {  /* not measured */
    const int32_t in = force && k - 1 < nforce ? force[k - 1] : gen[k - 1];
    if (dm_step(e, in, lg) != DM_OK) break;
    gen[k] = dm_argmax(lg, inf.vocab);
  }
  const int k0 = k;
  double pool0[4] = {0, 0, 0, 0};
  {
    const char* js0 = dm_stats_json(e);
    const char* p = strstr(js0, "\"gb_read_by_pool_total\":[");
    if (p) sscanf(p + strlen("\"gb_read_by_pool_total\":["), "%lf,%lf,%lf,%lf", &pool0[0], &pool0[1], &pool0[2], &pool0[3]);
  }
  dm_counters(e, &c0);
  const uint64_t d0 = disk_read_bytes(), pi0 = pageins();
  double td = now_ms(), t300 = 0;
  for (; k < n; k++) {
    const int32_t in = force && k - 1 < nforce ? force[k - 1] : gen[k - 1];
    if (dm_step(e, in, lg) != DM_OK) break;
    gen[k] = dm_argmax(lg, inf.vocab);
    if (force && k < nforce) agree += gen[k] == force[k], forced++;
    if (k % 10 == 0) peak_sample();
    if (k - k0 + 1 == 300) t300 = now_ms();  /* the first 300 measured tokens' rate too (arms of different lengths) */
  }
  const double dec_ms = now_ms() - td;
  const uint64_t d1 = disk_read_bytes(), pi1 = pageins();
  dm_counters(e, &c1);
  const char* st = dm_stats_json(e);
  const char* rs = strstr(st, "\"resolved\":");  /* the resolved block: up to ,"capabilities" */
  const char* re = rs ? strstr(rs, ",\"capabilities\"") : NULL;
  if (rs) rs += strlen("\"resolved\":");
  const int rlen = rs && re ? (int)(re - rs) : 4;
  if (!rs || !re) rs = "null";
  char loop[1024], reads[1024], ra[2048], apx[1024];
  json_obj(st, "loop", loop, sizeof loop);
  json_obj(st, "reads", reads, sizeof reads);
  json_obj(st, "run_ahead", ra, sizeof ra);
  json_obj(st, "approx", apx, sizeof apx);
  double pool1[4] = {0, 0, 0, 0};
  {
    const char* p = strstr(st, "\"gb_read_by_pool_total\":[");
    if (p) sscanf(p + strlen("\"gb_read_by_pool_total\":["), "%lf,%lf,%lf,%lf", &pool1[0], &pool1[1], &pool1[2], &pool1[3]);
  }
  const int steps = k - k0 > 0 ? k - k0 : 1;
  const uint64_t picks = c1.picks - c0.picks;
  const double wait = (c1.wait_ms - c0.wait_ms) / steps;
  char line[16384];
  snprintf(line, sizeof line,
           "{\"arm\":\"%s\",\"mode\":\"%s\",\"mem\":%d,\"threads\":%d,\"readers\":%d,\"spare_memory\":%d,\"evict_hint\":%d,\"store_cache_gb_before\":%.2f,"
           "\"store_cache_gb_at_start\":%.2f,\"store_cache_gb_at_end\":%.2f,\"cache_gb\":%.2f,\"load_s\":%.2f,\"prefill_s\":%.2f,\"prefill_mb\":%.0f,"
           "\"tokens\":%d,\"measured_tokens\":%d,\"tok_s\":%.3f,\"ms_per_token\":%.2f,\"wait_ms_per_token\":%.2f,\"compute_ms_per_token\":%.2f,"
           "\"hit_rate\":%.4f,\"mb_drive_counter_per_token\":%.2f,\"mb_spare_per_token\":%.2f,\"mb_disk_rusage_per_token\":%.2f,"
           "\"read_gb_s_busy\":%.3f,\"pageins_per_token\":%.2f,\"file_refaults_per_token\":%.3f,\"teacher_forced\":%d,\"agreement_with_exact\":%.4f,"
           "\"tok_s_first_300\":%.3f,\"peak_footprint_gb\":%.2f,\"mb_by_pool_per_token\":[%.2f,%.2f],\"loop\":%s,\"reads\":%s,\"run_ahead\":%s,\"approx\":%s,\"resolved\":%.*s}",
           arm, o.mode ? o.mode : "", inf.memory_mode, inf.threads, inf.readers, c1.spare_memory, o.file_evict_hint, before, at_start,
           store_cache_gb(model, 0), inf.cache_bytes / 1e9, load_s, prefill_s, sr.mb_read, k, k - k0, (k - k0) / (dec_ms / 1e3),
           dec_ms / steps, wait, dec_ms / steps - wait, picks ? (double)(c1.hits - c0.hits) / picks : 0,
           (c1.bytes_read - c0.bytes_read) / 1e6 / steps, (c1.bytes_spare - c0.bytes_spare) / 1e6 / steps, (d1 - d0) / 1e6 / steps,
           (c1.bytes_read - c0.bytes_read) / 1e9 / ((c1.read_busy_ns - c0.read_busy_ns) / 1e9 + 1e-12), (double)(pi1 - pi0) / steps,
           (double)(c1.file_refaults - c0.file_refaults) / steps, force != NULL, forced ? (double)agree / forced : -1.0,
           t300 > 0 ? 300 / ((t300 - td) / 1e3) : -1.0, arm_peak / 1e9, (pool1[0] - pool0[0]) * 1e3 / steps, (pool1[1] - pool0[1]) * 1e3 / steps, loop, reads, ra, apx, rlen, rs);
  printf("%s\n", line);
  fflush(stdout);
  if (jf) fprintf(jf, "%s\n", line), fflush(jf);
  free(lg);
  dm_close(e);
  return k;
}

int main(int argc, char** argv) {
  const char *model = NULL, *prompt = NULL, *out = NULL, *json = NULL, *mem = "mlock", *rid = NULL, *emb = NULL;
  int adapt = 0;
  double min_gb = 0, core_gb = 0;
  int n = 400, threads = 0, readers = 0, resize_at = -1, cancel_test = 0;
  double gb = 3.3, resize_gb = 0;
  const char* policy_probe = NULL;
  const char *arms = NULL, *alt_model = NULL;  /* --arms off,on,nothing (one process) */
  int warm_n = 50, file_hint = 0, measure_after = 0, force_exact = 0;
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
  int ctx = 0;                               /* --ctx N (0: the profile's, else 16384) */
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
    else if (!strcmp(k, "--ctx")) ctx = atoi(v), i++;
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
    else if (!strcmp(k, "--arms")) arms = v, i++;
    else if (!strcmp(k, "--warm-n")) warm_n = atoi(v), i++;
    else if (!strcmp(k, "--measure-after")) measure_after = atoi(v), i++;
    else if (!strcmp(k, "--alt-model")) alt_model = v, i++;
    else if (!strcmp(k, "--force-exact")) force_exact = atoi(v), i++;
    else if (!strcmp(k, "--file-hint")) file_hint = atoi(v), i++;
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
  o.ctx = ctx;  /* 0: the profile's, else 16384 */
  o.prefill_gpu_frac = gpu_frac;
  o.prefill_gpu_experts = gpu_experts;
  o.prefill_gpu_min = gpu_min;
  o.prefill_gpu_int_kernel = gpu_int;
  o.prefill_gpu_min_wave = gpu_wave;
  o.prefill_threads = prefill_threads;
  o.file_evict_hint = file_hint;
  if (arms) {  /* the arms in one process (header comment) */
    FILE* jf = json ? fopen(json, "w") : NULL;
    /* the ids compared within each mode: the first arm of a mode is that mode's reference */
    enum { MAXA = 32 };
    char mode_of[MAXA][64];
    int32_t* ref[MAXA];
    int nref[MAXA], nmodes = 0, same_exact = 1, same_all = 1;
    int32_t* gen = malloc(sizeof(int32_t) * (n + 1 > 8192 ? n + 1 : 8192));  /* an arm may ask for more tokens (/n=) */
    int32_t* exact_ids = NULL;
    int nexact = 0;
    if (warm_n > 0) {
      fprintf(stderr, "warm-up (off, %d tokens)\n", warm_n);
      if (run_arm(model, o, "warmup-off", ids, np, warm_n, 0, NULL, 0, gen, jf) < 0) return 1;
    }
    char buf[2048];
    strncpy(buf, arms, sizeof buf - 1);
    buf[sizeof buf - 1] = 0;
    char* save = NULL;
    for (char* t = strtok_r(buf, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
      char arm[256], md[64] = "";
      strncpy(arm, t, sizeof arm - 1);
      arm[sizeof arm - 1] = 0;
      if (!arm_opt(arm, "mode", md, sizeof md)) snprintf(md, sizeof md, "%s", mode ? mode : "exact");
      const int exact = arm_exact(md);
      const int forced = force_exact && !exact && exact_ids;
      fprintf(stderr, "arm %s%s\n", arm, forced ? " (fed the first exact arm's ids)" : "");
      char altv[8];
      const char* mdir = arm_opt(arm, "alt", altv, sizeof altv) && atoi(altv) && alt_model ? alt_model : model;
      const int k = run_arm(mdir, o, arm, ids, np, n, measure_after, forced ? exact_ids : NULL, nexact, gen, jf);
      if (k < 0) return 1;
      if (exact && !exact_ids) exact_ids = malloc(sizeof(int32_t) * k), memcpy(exact_ids, gen, sizeof(int32_t) * k), nexact = k;
      int m = 0;
      while (m < nmodes && strcmp(mode_of[m], md)) m++;
      if (forced) {
        fprintf(stderr, "arm %s: teacher-forced on the exact ids (its own ids are not compared)\n", arm);
      } else if (m == nmodes && nmodes < MAXA) {
        snprintf(mode_of[m], sizeof mode_of[m], "%s", md);
        ref[m] = malloc(sizeof(int32_t) * k), memcpy(ref[m], gen, sizeof(int32_t) * k), nref[m] = k, nmodes++;
        fprintf(stderr, "arm %s: %d ids, the first arm of mode '%s'\n", arm, k, md);
      } else if (m < nmodes) {
        const int kk = k < nref[m] ? k : nref[m];  /* arms of different lengths (/n=): their common prefix */
        const int same = !memcmp(ref[m], gen, sizeof(int32_t) * kk);
        same_all &= same;
        if (exact) same_exact &= same;
        fprintf(stderr, "arm %s: %d ids, %s the first '%s' arm's%s\n", arm, k, same ? "identical to" : "DIFFERENT from", md,
                exact || same ? "" : " (an approximate mode: may differ)");
      }
      if (out) {
        char p[4096];
        snprintf(p, sizeof p, "%s.%s", out, arm);
        for (char* c = p + strlen(out) + 1; *c; c++)
          if (*c == '/' || *c == '=') *c = '_';
        FILE* f = fopen(p, "w");
        for (int i = 0; i < k; i++) fprintf(f, "%s%d", i ? "," : "", gen[i]);
        fprintf(f, "\n");
        fclose(f);
      }
    }
    printf("{\"ids_identical_across_arms\":%s,\"ids_identical_across_exact_arms\":%s}\n", same_all ? "true" : "false", same_exact ? "true" : "false");
    if (jf)
      fprintf(jf, "{\"ids_identical_across_arms\":%s,\"ids_identical_across_exact_arms\":%s}\n", same_all ? "true" : "false",
              same_exact ? "true" : "false"),
          fclose(jf);
    return same_exact ? 0 : 4;
  }
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
