// Read log for an unmodified engine binary: every pread (time, file, offset, bytes) into memory, written at exit to
// $PREADLOG (binary: per read uint64 ns, int32 file id, int32 bytes, int64 offset), with the file names in $PREADLOG.files.
// Built as a dylib and loaded with DYLD_INSERT_LIBRARIES through preadrun (protected binaries such as /usr/bin/time
// drop DYLD_* variables, so preadrun sets it itself and execs the engine).
//   cc -O2 -dynamiclib -o preadlog.dylib preadlog.c && cc -O2 -o preadrun preadrun.c
#include <fcntl.h>
#include <mach/mach_time.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

#define DYLD_INTERPOSE(_r, _o) \
  __attribute__((used)) static struct { const void* r; const void* o; } _interpose_##_o \
      __attribute__((section("__DATA,__interpose"))) = {(const void*)(unsigned long)&_r, (const void*)(unsigned long)&_o};

typedef struct { uint64_t t; int32_t f, n; int64_t off; } Rec;
#define CAP (1 << 21)
#define NFD 4096
static Rec* recs;
static _Atomic long nrec;
static int fid[NFD];  // fd -> file id + 1 (0: not seen yet)
static char names[1024][MAXPATHLEN];
static _Atomic int nnames;
static mach_timebase_info_data_t tb;

static int file_id(int fd) {
  if (fd < 0 || fd >= NFD) return -1;
  int v = __atomic_load_n(&fid[fd], __ATOMIC_ACQUIRE);
  if (v) return v - 1;
  char p[MAXPATHLEN];
  if (fcntl(fd, F_GETPATH, p) != 0) return -1;
  for (int i = 0; i < nnames; i++)
    if (!strcmp(names[i], p)) return __atomic_store_n(&fid[fd], i + 1, __ATOMIC_RELEASE), i;
  int i = atomic_fetch_add(&nnames, 1);
  if (i >= 1024) return -1;
  strncpy(names[i], p, MAXPATHLEN - 1);
  __atomic_store_n(&fid[fd], i + 1, __ATOMIC_RELEASE);
  return i;
}

static ssize_t my_pread(int fd, void* buf, size_t n, off_t off) {
  ssize_t r = pread(fd, buf, n, off);
  if (recs) {
    long k = atomic_fetch_add(&nrec, 1);
    if (k < CAP) recs[k] = (Rec){mach_absolute_time() * tb.numer / tb.denom, file_id(fd), (int32_t)r, (int64_t)off};
  }
  return r;
}
DYLD_INTERPOSE(my_pread, pread)

static int my_close(int fd) {
  if (fd >= 0 && fd < NFD) __atomic_store_n(&fid[fd], 0, __ATOMIC_RELEASE);
  return close(fd);
}
DYLD_INTERPOSE(my_close, close)

static void dump(void) {
  const char* p = getenv("PREADLOG");
  if (!p || !recs) return;
  FILE* f = fopen(p, "wb");
  long n = nrec < CAP ? nrec : CAP;
  fwrite(recs, sizeof(Rec), n, f);
  fclose(f);
  char q[MAXPATHLEN + 8];
  snprintf(q, sizeof q, "%s.files", p);
  f = fopen(q, "w");
  for (int i = 0; i < nnames && i < 1024; i++) fprintf(f, "%s\n", names[i]);
  fprintf(f, "# reads %ld (cap %d)\n", (long)nrec, CAP);
  fclose(f);
}

__attribute__((constructor)) static void init(void) {
  if (!getenv("PREADLOG")) return;
  mach_timebase_info(&tb);
  recs = (Rec*)malloc(sizeof(Rec) * CAP);
  atexit(dump);
}
