// Raw drive rate for the store's read pattern: random whole-record preads (2,768,896 B, 16 KB aligned, F_NOCACHE)
// from store/layer_XX.bin, no Python and no model. Usage: drivebench STORE_DIR THREADS SECONDS [SUBREADS] [BURST PAUSE_MS]
//   SUBREADS k   : each record is read as k equal pieces (default 1)
//   BURST b, PAUSE_MS p : decode-shaped load; after every b records each thread sleeps p ms (default: no pause)
// Prints one line: threads, sub-reads, GB/s (1e9), records/s, mean and p99 record latency.
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NL 48
#define NE 512
#define REC 2768896L
#define PAGE 16384L

static int fds[NL];
static int *order;
static atomic_long cursor, bytes_done, recs_done;
static atomic_int stop;
static int subreads = 1, burst = 0, pause_ms = 0;
static double *lat;  // per-record latency, seconds
static long lat_cap;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void *worker(void *arg) {
    (void)arg;
    void *buf;
    if (posix_memalign(&buf, PAGE, REC)) return NULL;
    memset(buf, 0, REC);
    long piece = (REC / subreads / PAGE) * PAGE, mine = 0;
    while (!atomic_load(&stop)) {
        long i = atomic_fetch_add(&cursor, 1) % (NL * NE);
        int L = order[i] / NE, e = order[i] % NE;
        double t0 = now();
        long off = 0;
        for (int k = 0; k < subreads; k++) {
            long n = k == subreads - 1 ? REC - off : piece;
            if (pread(fds[L], (char *)buf + off, n, (off_t)e * REC + off) != n) { perror("pread"); exit(1); }
            off += n;
        }
        long r = atomic_fetch_add(&recs_done, 1);
        if (r < lat_cap) lat[r] = now() - t0;
        atomic_fetch_add(&bytes_done, REC);
        if (burst && ++mine % burst == 0) usleep(pause_ms * 1000);
    }
    free(buf);
    return NULL;
}

static int cmp(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s STORE_DIR THREADS SECONDS [SUBREADS] [BURST PAUSE_MS]\n", argv[0]); return 2; }
    int nt = atoi(argv[2]);
    double secs = atof(argv[3]);
    if (argc > 4) subreads = atoi(argv[4]);
    if (argc > 6) { burst = atoi(argv[5]); pause_ms = atoi(argv[6]); }
    char path[4096];
    for (int L = 0; L < NL; L++) {
        snprintf(path, sizeof path, "%s/layer_%02d.bin", argv[1], L);
        fds[L] = open(path, O_RDONLY);
        if (fds[L] < 0) { perror(path); return 1; }
        fcntl(fds[L], F_NOCACHE, 1);
    }
    order = malloc(sizeof(int) * NL * NE);
    for (int i = 0; i < NL * NE; i++) order[i] = i;
    srandom(12345);
    for (int i = NL * NE - 1; i > 0; i--) { int j = random() % (i + 1), t = order[i]; order[i] = order[j]; order[j] = t; }
    lat_cap = 1L << 20;
    lat = malloc(sizeof(double) * lat_cap);
    pthread_t th[64];
    double t0 = now();
    for (int i = 0; i < nt; i++) pthread_create(&th[i], NULL, worker, NULL);
    usleep((useconds_t)(secs * 1e6));
    atomic_store(&stop, 1);
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    double dt = now() - t0;
    long n = atomic_load(&recs_done);
    if (n > lat_cap) n = lat_cap;
    double sum = 0;
    for (long i = 0; i < n; i++) sum += lat[i];
    qsort(lat, n, sizeof(double), cmp);
    printf("threads %d subreads %d burst %d pause_ms %d | %.2f GB/s | %.0f records/s | latency mean %.2f ms p50 %.2f p99 %.2f | %.1f s, %ld records\n",
           nt, subreads, burst, pause_ms, atomic_load(&bytes_done) / dt / 1e9, atomic_load(&recs_done) / dt, n ? 1000 * sum / n : 0,
           n ? 1000 * lat[n / 2] : 0, n ? 1000 * lat[(long)(n * 0.99)] : 0, dt, atomic_load(&recs_done));
    return 0;
}
