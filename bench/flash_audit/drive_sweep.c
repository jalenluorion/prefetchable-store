// Drive rate for Flash-Next's read pattern on this Mac, with every knob of the engine's reader exposed (audit tool, no
// model). Reads go to store/layer_XX.bin (2,768,896 B records, 512 per file, 48 files).
//
// usage: drive_sweep STORE_DIR [key=value ...]
//   threads=N    reader threads, each doing blocking pread (the engine's Readers: queue depth = threads)    default 4
//   split=K      a record is K page-aligned pieces, each its own job taken by any thread (as Cache::split) default 1
//   chunk=B      instead of whole records: contiguous reads of B bytes at random page-aligned offsets       default 0
//   run=R        whole records, R adjacent ones per read (one pread of R x 2.77 MB: a co-access layout)   default 1
//   misalign=B   add B bytes to every offset (alignment test)                                             default 0
//   nocache=0|1  F_NOCACHE on the descriptors (the engine's default 1)                                    default 1
//   rdahead=0|1  F_RDAHEAD (the engine sets 0)                                                            default 0
//   seq=0|1      records in file order instead of random                                                  default 0
//   secs=S       duration                                                                                 default 3
//   layers=A-B   restrict to a layer range                                                                default 0-47
//   burst=b pause_ms=p   decode-shaped load: after b jobs each thread sleeps p ms                         default off
//   seed=S
// Prints one JSON line: GB/s (1e9 B/s, wall), jobs/s, per-job latency mean/p50/p99 ms, and the config.
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define NL 48
#define NE 512
#define REC 2768896L
#define PAGE 16384L

static int fds[NL];
static int L0 = 0, L1 = NL - 1, nthreads = 4, split = 1, run = 1, nocache = 1, rdahead = 0, seq = 0, burst = 0, pause_ms = 0;
static long chunk = 0, misalign = 0, seed = 12345;
static double secs = 3;
static atomic_long job_cursor, bytes_done, jobs_done;
static atomic_int stop_flag;
static double* lat;
static long lat_cap = 1L << 21;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

// a per-job pseudo-random stream (splitmix64), so threads do not share an RNG
static uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static void* worker(void* arg) {
  (void)arg;
  long maxn = chunk ? chunk : REC * run;
  void* buf;
  if (posix_memalign(&buf, PAGE, (size_t)maxn + PAGE)) return NULL;
  memset(buf, 0, (size_t)maxn + PAGE);
  long piece = ((REC / split) + PAGE - 1) & ~(PAGE - 1);
  long mine = 0;
  const int nl = L1 - L0 + 1;
  while (!atomic_load_explicit(&stop_flag, memory_order_relaxed)) {
    long j = atomic_fetch_add(&job_cursor, 1);
    uint64_t r = mix((uint64_t)j ^ (uint64_t)seed * 0x100000001B3ull);
    int L;
    long off, n;
    if (chunk) {
      L = L0 + (int)(r % nl);
      long span = NE * REC - chunk;
      off = (long)((r >> 16) % (uint64_t)(span / PAGE)) * PAGE;
      n = chunk;
    } else {
      long rec_job = split > 1 ? j / split : j;  // the pieces of one record are consecutive jobs
      int k = split > 1 ? (int)(j % split) : 0;
      uint64_t rr = mix((uint64_t)rec_job ^ (uint64_t)seed * 0x100000001B3ull);
      int e;
      if (seq) {
        long g = rec_job * run;
        L = L0 + (int)((g / NE) % nl);
        e = (int)(g % NE);
        if (e + run > NE) e = NE - run;
      } else {
        L = L0 + (int)(rr % nl);
        e = (int)((rr >> 20) % (uint64_t)(NE - run + 1));
      }
      if (split > 1) {
        off = (long)e * REC + k * piece;
        n = k == split - 1 ? REC - k * piece : piece;
        if (n <= 0) continue;
      } else {
        off = (long)e * REC;
        n = REC * run;
      }
    }
    off += misalign;
    if (off + n > NE * REC) n = NE * REC - off;
    double t0 = now();
    long done = 0;
    while (done < n) {
      ssize_t got = pread(fds[L], (char*)buf + done, (size_t)(n - done), (off_t)(off + done));
      if (got <= 0) {
        perror("pread");
        exit(1);
      }
      done += got;
    }
    long q = atomic_fetch_add(&jobs_done, 1);
    if (q < lat_cap) lat[q] = now() - t0;
    atomic_fetch_add(&bytes_done, n);
    if (burst && ++mine % burst == 0) usleep((useconds_t)pause_ms * 1000);
  }
  free(buf);
  return NULL;
}

static int cmp(const void* a, const void* b) {
  double x = *(const double*)a, y = *(const double*)b;
  return x < y ? -1 : x > y;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s STORE_DIR [key=value ...]\n", argv[0]);
    return 2;
  }
  for (int i = 2; i < argc; i++) {
    char k[64];
    char v[64];
    if (sscanf(argv[i], "%63[^=]=%63s", k, v) != 2) continue;
    if (!strcmp(k, "threads")) nthreads = atoi(v);
    else if (!strcmp(k, "split")) split = atoi(v);
    else if (!strcmp(k, "chunk")) chunk = atol(v);
    else if (!strcmp(k, "run")) run = atoi(v);
    else if (!strcmp(k, "misalign")) misalign = atol(v);
    else if (!strcmp(k, "nocache")) nocache = atoi(v);
    else if (!strcmp(k, "rdahead")) rdahead = atoi(v);
    else if (!strcmp(k, "seq")) seq = atoi(v);
    else if (!strcmp(k, "secs")) secs = atof(v);
    else if (!strcmp(k, "burst")) burst = atoi(v);
    else if (!strcmp(k, "pause_ms")) pause_ms = atoi(v);
    else if (!strcmp(k, "seed")) seed = atol(v);
    else if (!strcmp(k, "layers")) sscanf(v, "%d-%d", &L0, &L1);
  }
  char path[4096];
  for (int L = 0; L < NL; L++) {
    snprintf(path, sizeof path, "%s/layer_%02d.bin", argv[1], L);
    fds[L] = open(path, O_RDONLY);
    if (fds[L] < 0) {
      perror(path);
      return 1;
    }
    if (nocache) fcntl(fds[L], F_NOCACHE, 1);
    fcntl(fds[L], F_RDAHEAD, rdahead);
  }
  lat = malloc(sizeof(double) * lat_cap);
  pthread_t th[256];
  double t0 = now();
  for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, worker, NULL);
  usleep((useconds_t)(secs * 1e6));
  atomic_store(&stop_flag, 1);
  for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
  double dt = now() - t0;
  long n = atomic_load(&jobs_done);
  if (n > lat_cap) n = lat_cap;
  double sum = 0;
  for (long i = 0; i < n; i++) sum += lat[i];
  qsort(lat, (size_t)n, sizeof(double), cmp);
  printf("{\"threads\":%d,\"split\":%d,\"chunk\":%ld,\"run\":%d,\"misalign\":%ld,\"nocache\":%d,\"rdahead\":%d,\"seq\":%d,\"burst\":%d,"
         "\"pause_ms\":%d,\"layers\":\"%d-%d\",\"secs\":%.2f,\"gb_s\":%.3f,\"jobs_s\":%.0f,\"lat_ms_mean\":%.3f,\"lat_ms_p50\":%.3f,"
         "\"lat_ms_p99\":%.3f,\"jobs\":%ld}\n",
         nthreads, split, chunk, run, misalign, nocache, rdahead, seq, burst, pause_ms, L0, L1, dt, atomic_load(&bytes_done) / dt / 1e9,
         atomic_load(&jobs_done) / dt, n ? 1000 * sum / n : 0, n ? 1000 * lat[n / 2] : 0, n ? 1000 * lat[(long)(n * 0.99)] : 0,
         atomic_load(&jobs_done));
  return 0;
}
