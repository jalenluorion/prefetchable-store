// P1 Mac probe for Mem::FILE (no model): on the common store's layer files, does a buffered pread of a record into a
// scratch buffer make the record's pages resident for a read-only mapping of the same file (mincore), and does reading
// the record through the mapping afterwards still take page faults / time? Also: the cost of mincore per call, the
// effect of F_RDAHEAD 0 (does a buffered pread read ahead into the next record?), and of MADV_DONTNEED on a record.
//   clang++ -O2 -std=c++17 fc_mac_probe.cpp -o /tmp/fc_mac_probe && /tmp/fc_mac_probe STORE_DIR [records]
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const size_t REC = 1769472, PG = 16384, NP = REC / PG;
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int64_t faults() {
  task_events_info_data_t e;
  mach_msg_type_number_t c = TASK_EVENTS_INFO_COUNT;
  task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&e, &c);
  return e.faults;
}
static int64_t pageins() {
  task_events_info_data_t e;
  mach_msg_type_number_t c = TASK_EVENTS_INFO_COUNT;
  task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&e, &c);
  return e.pageins;
}
static int resident_pages(const uint8_t* p, size_t n) {
  std::vector<char> v(n / PG);
  mincore((void*)p, n, v.data());
  int k = 0;
  for (char c : v) k += (c & MINCORE_INCORE) != 0;
  return k;
}
static volatile uint64_t sink;
static double touch_all(const uint8_t* p, size_t n) {  // read every byte (as the kernels do), 8 bytes at a time
  double t = now_ms();
  uint64_t s = 0;
  const uint64_t* q = (const uint64_t*)p;
  for (size_t i = 0; i < n / 8; i++) s += q[i];
  sink = s;
  return now_ms() - t;
}

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  std::string dir = argv[1];
  const int N = argc > 2 ? atoi(argv[2]) : 24;
  std::mt19937 rng(1122);
  struct L {
    int fd;
    uint8_t* map;
    size_t bytes;
  };
  std::vector<L> ls;
  for (int l = 0; l < 40; l++) {
    char b[64];
    snprintf(b, sizeof b, "/layer_%02d.bin", l);
    int fd = open((dir + b).c_str(), O_RDONLY);
    if (fd < 0) return 3;
    size_t sz = (size_t)lseek(fd, 0, SEEK_END);
    void* m = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) return 4;
    ls.push_back({fd, (uint8_t*)m, sz});
  }
  const int NE = (int)(ls[0].bytes / REC);
  std::vector<uint8_t> scratch(REC);
  // pick cold records (no page resident), with a cold next record too (to see read-ahead)
  auto pick_cold = [&](int& l, int& e) {
    for (int t = 0; t < 100000; t++) {
      l = rng() % 40;
      e = rng() % (NE - 1);
      if (resident_pages(ls[l].map + (size_t)e * REC, 2 * REC) == 0) return true;
    }
    return false;
  };
  printf("{\"records\":%d,\"rows\":[\n", N);
  double sum[2][6] = {{0}};
  int cnt[2] = {0, 0};
  const bool rev = getenv("FC_REV") != nullptr;  // FC_REV=1: F_RDAHEAD 0 first (order check)
  for (int ri = 0; ri < 2; ri++) {      // F_RDAHEAD default (on), then 0
    const int ra = rev ? 1 - ri : ri;
    for (int i = 0; i < N; i++) {
      int l, e;
      if (!pick_cold(l, e)) break;
      fcntl(ls[l].fd, F_RDAHEAD, ra ? 0 : 1);
      const uint8_t* v = ls[l].map + (size_t)e * REC;
      int64_t f0 = faults(), p0 = pageins();
      double t0 = now_ms();
      size_t done = 0;
      while (done < REC) done += (size_t)pread(ls[l].fd, scratch.data() + done, REC - done, (off_t)((size_t)e * REC + done));
      double t_read = now_ms() - t0;
      int64_t f1 = faults(), p1 = pageins();
      int res = resident_pages(v, REC), res_next = resident_pages(v + REC, REC);
      double t_touch1 = touch_all(v, REC);
      int64_t f2 = faults(), p2 = pageins();
      double t_touch2 = touch_all(v, REC);
      int64_t f3 = faults();
      // cost of mincore: one page (first) and the whole record
      double tm = now_ms();
      char c1;
      for (int k = 0; k < 1000; k++) mincore((void*)v, PG, &c1);
      double mc1 = (now_ms() - tm);  // ms per 1000 = us per call
      std::vector<char> vv(NP);
      tm = now_ms();
      for (int k = 0; k < 1000; k++) mincore((void*)v, REC, vv.data());
      double mcr = (now_ms() - tm);
      printf("%s{\"rdahead\":%d,\"layer\":%d,\"e\":%d,\"pread_ms\":%.3f,\"pread_faults\":%lld,\"pread_pageins\":%lld,\"resident_after_pread\":%d,"
             "\"next_record_resident\":%d,\"touch1_ms\":%.3f,\"touch1_faults\":%lld,\"touch1_pageins\":%lld,\"touch2_ms\":%.3f,\"touch2_faults\":%lld,"
             "\"mincore_1page_us\":%.3f,\"mincore_record_us\":%.3f}\n",
             (ri || i) ? "," : "", ra ? 0 : 1, l, e, t_read, (long long)(f1 - f0), (long long)(p1 - p0), res, res_next, t_touch1,
             (long long)(f2 - f1), (long long)(p2 - p1), t_touch2, (long long)(f3 - f2), mc1, mcr);
      double row[6] = {t_read, (double)res, (double)res_next, t_touch1, (double)(f2 - f1), t_touch2};
      for (int k = 0; k < 6; k++) sum[ra][k] += row[k];
      cnt[ra]++;
    }
  }
  // MADV_DONTNEED on a resident record: does it leave the cache?
  int l, e;
  pick_cold(l, e);
  const uint8_t* v = ls[l].map + (size_t)e * REC;
  touch_all(v, REC);
  int before = resident_pages(v, REC);
  madvise((void*)v, REC, MADV_DONTNEED);
  int after = resident_pages(v, REC);
  usleep(200000);
  int after2 = resident_pages(v, REC);
  // prefault cost: pread into scratch, then touch one byte per page (what a reader thread would do), then a full read
  pick_cold(l, e);
  v = ls[l].map + (size_t)e * REC;
  size_t done = 0;
  while (done < REC) done += (size_t)pread(ls[l].fd, scratch.data() + done, REC - done, (off_t)((size_t)e * REC + done));
  int64_t f0 = faults();
  double tp = now_ms();
  uint64_t s = 0;
  for (size_t o = 0; o < REC; o += PG) s += v[o];
  sink = s;
  tp = now_ms() - tp;
  int64_t f1 = faults();
  double tfull = touch_all(v, REC);
  int64_t f2 = faults();
  printf("],\"madv_dontneed\":{\"resident_before\":%d,\"after\":%d,\"after_200ms\":%d},", before, after, after2);
  printf("\"prefault_one_byte_per_page\":{\"ms\":%.3f,\"faults\":%lld,\"then_full_read_ms\":%.3f,\"then_faults\":%lld},", tp, (long long)(f1 - f0), tfull,
         (long long)(f2 - f1));
  printf("\"mean\":{");
  const char* nm[6] = {"pread_ms", "resident_after_pread", "next_record_resident", "touch1_ms", "touch1_faults", "touch2_ms"};
  for (int ra = 0; ra < 2; ra++) {
    printf("%s\"%s\":{", ra ? "," : "", ra ? "rdahead_0" : "rdahead_default");
    for (int k = 0; k < 6; k++) printf("%s\"%s\":%.3f", k ? "," : "", nm[k], sum[ra][k] / std::max(1, cnt[ra]));
    printf("}");
  }
  printf("},\"pages_per_record\":%zu}\n", NP);
  return 0;
}
