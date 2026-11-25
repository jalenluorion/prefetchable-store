// Two facts about the store files that decide read speed, without reading them (audit tool):
//  1. how much of each file is in the OS file cache now (mmap + mincore; maps only, touches no page);
//  2. how fragmented each file is on the drive: physical extents from F_LOG2PHYS_EXT, walked from offset 0 to the end,
//     and how many extents a 2.77 MB record spans (a record split across extents costs more than one command).
// usage: store_state DIR_OF_FILES PATTERN COUNT     e.g. store_state model/store layer_%02d.bin 48
// Prints one JSON line per file and a summary line.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define REC 2768896L

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s DIR PATTERN COUNT\n", argv[0]);
    return 2;
  }
  const int count = atoi(argv[3]);
  const long pg = getpagesize();
  double tot_bytes = 0, tot_res = 0;
  long tot_ext = 0, tot_rec = 0, tot_rec_split = 0;
  long min_ext = 1L << 60, max_ext = 0;
  for (int f = 0; f < count; f++) {
    char name[512], path[4096];
    snprintf(name, sizeof name, argv[2], f);
    snprintf(path, sizeof path, "%s/%s", argv[1], name);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
      perror(path);
      return 1;
    }
    struct stat st;
    fstat(fd, &st);
    const size_t sz = (size_t)st.st_size;
    long resident = 0;
    void* m = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (m != MAP_FAILED) {
      size_t np = (sz + pg - 1) / pg;
      char* v = malloc(np);
      if (mincore(m, sz, v) == 0)
        for (size_t i = 0; i < np; i++) resident += (v[i] & MINCORE_INCORE) != 0;
      free(v);
      munmap(m, sz);
    }
    // extents: F_LOG2PHYS_EXT maps (offset, length) to the physical offset of the contiguous run that starts there
    long ext = 0, off = 0;
    long* ext_end = malloc(sizeof(long) * 1);
    long cap = 1, n_end = 0;
    int64_t prev_phys_end = -1;
    long merged = 0;  // extents physically contiguous with the previous one (an extent boundary that costs nothing)
    while (off < (long)sz) {
      struct log2phys l;
      memset(&l, 0, sizeof l);
      l.l2p_contigbytes = (off_t)(sz - off);
      l.l2p_devoffset = (off_t)off;
      if (fcntl(fd, F_LOG2PHYS_EXT, &l) < 0) {
        perror("F_LOG2PHYS_EXT");
        break;
      }
      long len = (long)l.l2p_contigbytes;
      if (len <= 0) break;
      if (prev_phys_end == (int64_t)l.l2p_devoffset) merged++;
      prev_phys_end = (int64_t)l.l2p_devoffset + len;
      ext++;
      off += len;
      if (n_end == cap) ext_end = realloc(ext_end, sizeof(long) * (cap *= 2));
      ext_end[n_end++] = off;
    }
    // records that cross an extent boundary
    long rec_split = 0, k = 0;
    const long nrec = (long)sz / REC;
    for (long r = 0; r < nrec; r++) {
      long a = r * REC, b = a + REC;
      while (k < n_end && ext_end[k] <= a) k++;
      if (k < n_end && ext_end[k] < b) rec_split++;
    }
    free(ext_end);
    close(fd);
    printf("{\"file\":\"%s\",\"bytes\":%zu,\"resident_frac\":%.4f,\"extents\":%ld,\"phys_contig_boundaries\":%ld,\"records\":%ld,"
           "\"records_crossing_an_extent\":%ld}\n",
           name, sz, resident * (double)pg / (double)sz, ext, merged, nrec, rec_split);
    tot_bytes += sz;
    tot_res += resident * (double)pg;
    tot_ext += ext;
    tot_rec += nrec;
    tot_rec_split += rec_split;
    if (ext < min_ext) min_ext = ext;
    if (ext > max_ext) max_ext = ext;
  }
  printf("{\"summary\":true,\"files\":%d,\"gb\":%.2f,\"resident_gb\":%.3f,\"extents_total\":%ld,\"extents_min\":%ld,\"extents_max\":%ld,"
         "\"records\":%ld,\"records_crossing_an_extent\":%ld}\n",
         count, tot_bytes / 1e9, tot_res / 1e9, tot_ext, min_ext, max_ext, tot_rec, tot_rec_split);
  return 0;
}
