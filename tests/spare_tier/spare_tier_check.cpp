// The spare tier's mechanics on a small store of known bytes (no model): what the tiny models cannot show.
//
//   spare_tier_check [DIR]      (DIR: a scratch folder, default $TMPDIR; prints PASS / FAIL lines, exit 1 on a failure)
//
// A read takes a free slab slot first (its file pages dropped after it); with the slab full it lands in probation (a
// slot past the slab, the record read through the mapping, its pages resident and mapped by the reader: no page-in
// when the compute reads it); its reuse at a later token promotes it into the slab at the next
// step boundary (the slab's bytes are the file's, the file pages are dropped); a reuse inside one token does not; the
// budget drops the oldest unpinned probation record's pages; a record whose pages the OS reclaimed is a miss at its use
// (unpinned) or read again in place (pinned); a pause drains probation and later reads go to the slab; with the tier off
// nothing is mapped and every read goes to the slab. Two pools (Inkling's record formats) each keep their own probation.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "dmeng/store.h"

using namespace dm;

static int fails = 0;
static void check(bool ok, const char* what) {
  printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) fails++;
}

static const int NL = 2, NE = 16, K = 2;
static const size_t R0 = 64 << 10, R1 = 32 << 10;  // record bytes (whole 16 KB pages)

static uint8_t byte_of(int L, int e, size_t i) { return (uint8_t)(L * 131 + e * 17 + i * 7 + (i >> 12)); }

static bool resident(const uint8_t* p, size_t n) {  // every page
  const size_t pg = (size_t)getpagesize();
  std::vector<char> v((n + pg - 1) / pg);
  if (mincore((void*)p, n, v.data()) != 0) return false;
  for (char c : v)
    if (!(c & MINCORE_INCORE)) return false;
  return true;
}
static bool none_resident(const uint8_t* p, size_t n) {
  const size_t pg = (size_t)getpagesize();
  std::vector<char> v((n + pg - 1) / pg);
  if (mincore((void*)p, n, v.data()) != 0) return false;
  for (char c : v)
    if (c & MINCORE_INCORE) return false;
  return true;
}
static bool same_as_file(const uint8_t* p, int L, int e, size_t R) {
  for (size_t i = 0; i < R; i++)
    if (p[i] != byte_of(L, e, i)) return false;
  return true;
}

static void write_store(const std::string& dir, const size_t* rec) {
  for (int L = 0; L < NL; L++) {
    char b[64];
    snprintf(b, sizeof b, "/layer_%02d.bin", L);
    std::vector<uint8_t> v(rec[L] * NE);
    for (int e = 0; e < NE; e++)
      for (size_t i = 0; i < rec[L]; i++) v[(size_t)e * rec[L] + i] = byte_of(L, e, i);
    const int fd = open((dir + b).c_str(), O_CREAT | O_TRUNC | O_RDWR, 0644);
    if (fd < 0 || write(fd, v.data(), v.size()) != (ssize_t)v.size()) {
      fprintf(stderr, "cannot write %s%s\n", dir.c_str(), b);
      exit(2);
    }
    fsync(fd);
    void* m = mmap(nullptr, v.size(), PROT_READ, MAP_SHARED, fd, 0);  // start cold: the written pages out of the file cache
    msync(m, v.size(), MS_INVALIDATE);
    munmap(m, v.size());
    close(fd);
  }
}

// a read of (L, e) the way a miss takes it (victim, issue, wait); its slot
static int miss(Cache& c, int L, int e, long step) {
  const int s = c.victim(L, 0);
  if (s < 0) return -1;
  c.issue(L, s, e, true, step);
  c.slot(L, s).pins = 1;
  c.wait(L, s);
  return s;
}
static void done(Cache& c, int L, int s, long t) {  // the step's use and unpin (Core::finish_step)
  c.use(L, s, t, t * 2 + L);
  c.unpin(L, s);
}
// reads into the slab's free slots (each used once at token 0) until it is full; true if every one went to the slab with
// the file's bytes and left no file pages behind
static bool fill_slab(Cache& c, int L, const std::vector<int>& es, size_t R) {
  bool ok = true;
  for (int e : es) {
    const int s = miss(c, L, e, 0);
    ok &= s >= 0 && s < c.pool(L).n && same_as_file(c.rec(L, s), L, e, R) && none_resident(c.fmap[L] + (size_t)e * R, R);
    done(c, L, s, 0);
  }
  return ok && c.pool(L).free_.empty();
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : (getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp");
  dir += "/spare_tier_check." + std::to_string(getpid());
  if (system(("mkdir -p '" + dir + "'").c_str()) != 0) return 2;
  std::string err;
  // ---------------------------------------------------------------- one pool, the tier on
  {
    const size_t rec[2] = {R0, R0};
    write_store(dir, rec);
    Cache c;
    c.file_under = true;
    c.prob_bytes = (int64_t)(6 * R0);
    if (!c.init(NL, NE, K, {0, 1}, {0, 0}, {R0}, {1.0}, dir, "layer_%02d.bin", 8.0 * R0, 2, 1, Mem::PLAIN, false, err)) {
      printf("FAIL init: %s\n", err.c_str());
      return 1;
    }
    c.enable_hazard();
    SlotPool& S = c.P[0];
    check(S.n == 8 && S.np == 6 && c.prob_on() && c.fmap.size() == 2, "the tier on: 8 slab slots, 6 probation slots, the files mapped");
    // free slab slots are taken first (nothing displaced; the slab read leaves no file pages: one place)
    check(fill_slab(c, 0, {4, 5, 6, 7, 8, 10, 12, 13}, R0) && c.prob_held(0) == 0, "free slab slots first: the slab's bytes, no file pages left");
    // with the slab full, a miss lands in probation, read through the mapping
    const uint64_t b0 = c.rd.bytes.load();
    int s = miss(c, 0, 3, 0);
    const uint8_t* view = c.fmap[0] + 3 * R0;
    check(s >= S.n && c.rec(0, s) == view && same_as_file(c.rec(0, s), 0, 3, R0), "a miss lands in probation: the record is the mapping's, the file's bytes");
    check(c.rd.bytes.load() - b0 == R0 && resident(view, R0), "the reader read it from the drive into the file cache (pages resident)");
    check(c.held(0) == 8 && c.prob_held(0) == 1 && c.prob_held_bytes() == R0, "the slab's 8 untouched; probation holds one record");
    // the compute reads it: no page-in (the reader mapped every page)
    {
      const uint64_t p0 = Cache::task_pageins();
      uint64_t acc = 0;
      for (size_t i = 0; i < R0; i += 64) acc += ((const volatile uint8_t*)c.rec(0, s))[i];
      check(Cache::task_pageins() - p0 == 0 && acc != 0, "the compute reads a probation record with no page-in");
      const uint8_t* cold = c.fmap[0] + 15 * R0;  // the counter's control: a record never read, touched through the mapping
      const uint64_t p1 = Cache::task_pageins();
      for (size_t i = 0; i < R0; i += 4096) acc += ((const volatile uint8_t*)cold)[i];
      check(Cache::task_pageins() - p1 > 0, "the page-in counter sees a fault on the drive (a cold record touched: the control)");
      msync((void*)cold, R0, MS_INVALIDATE);
    }
    done(c, 0, s, 0);
    // a second use inside the same token: no promotion
    c.slot(0, s).pins = 1;
    done(c, 0, s, 0);
    c.tok_now = 0;
    check(!c.tier_due() && c.slot_of(0, 3) == s, "a reuse inside the token (a looped layer's pass) does not promote");
    // a use in a second token: not yet (prom_uses 3 distinct tokens); a third: a candidate
    c.slot(0, s).pins = 1;
    s = c.slot_for_use(0, 3);
    done(c, 0, s, 1);
    check(c.st.n_prob_hit == 2 && c.st.bytes_prob_hit == 2 * R0, "uses after the first are counted as served by the file cache");
    check(c.prom_uses == 3 && !c.tier_due(), "used in two tokens: not a candidate (the rule's 3 distinct tokens)");
    c.slot(0, s).pins = 1;
    done(c, 0, s, 2);
    check(c.tier_due(), "used in a third token: a candidate for promotion");
    // the gate: the slab is full, so its keep score must beat the record it would displace by the margin
    c.tok_now = 2;
    c.prom_margin = 2.0;  // (a hazard score is a probability: nothing beats another by 2)
    c.tier_step();
    check(c.slot_of(0, 3) == s && c.st.n_promote_score_refused == 1 && c.st.prom_score_n == 1 && !c.tier_due(),
          "refused by the keep-score gate: stays in probation (slab untouched)");
    c.prom_margin = -2.0;  // (every candidate passes)
    c.slot(0, s).pins = 1;
    done(c, 0, s, 3);  // its next reuse: tested again
    c.tok_now = 3;
    c.slot(0, s).pins = 1;  // pinned (a name): not started at this boundary
    c.tier_step();
    check(c.slot_of(0, 3) == s && c.tier_due() && c.st.n_promote_started == 0, "a pinned record waits for a later boundary");
    c.unpin(0, s);
    c.tier_step();
    check(c.st.n_promote_started == 1 && c.slot_of(0, 3) == s && c.rec(0, s) == view && c.held(0) == 7,
          "started: a slab slot reserved (its record evicted), the copy queued, the record still used from the mapping");
    c.rd.drops_flush();  // the copy lands (the readers' tier thread)
    c.tier_step();       // the move, at the next boundary
    c.rd.drops_flush();  // (the pages are dropped by the same thread)
    const int ps = c.slot_of(0, 3);
    check(ps >= 0 && ps < S.n && same_as_file(c.rec(0, ps), 0, 3, R0) && c.rec(0, ps) == S.row(ps), "promoted: a slab slot with the file's bytes");
    check(none_resident(view, R0), "promoted: its file pages dropped (one place)");
    check(c.st.n_promoted == 1 && c.held(0) == 8 && c.prob_held(0) == 0 && !c.tier_due() && c.rd.copies_done.load() == 1,
          "promotions 1 (a slab record evicted for it, one copy on the tier thread); slab 8, probation 0");
    // the budget: 7 more reads into a probation of 6 drop the oldest's pages
    int sl[7];
    for (int i = 0; i < 7; i++) {
      sl[i] = miss(c, 1, i, 2);
      done(c, 1, sl[i], 2);
    }
    c.rd.drops_flush();
    const uint8_t* v10 = c.fmap[1];
    check(c.slot_of(1, 0) < 0 && none_resident(v10, R0) && c.st.n_prob_budget_drops == 1, "the budget: the oldest probation record dropped, its pages too");
    check(c.prob_held(0) == 6 && c.held(0) == 8, "probation holds its budget (6), the slab is untouched");
    for (int i = 1; i < 7; i++) check(c.slot_of(1, i) >= S.n, ("record " + std::to_string(i) + " in probation").c_str());
    // the OS reclaims a probation record: unpinned, its use is a miss; pinned, it is read again in place
    const uint8_t* v11 = c.fmap[1] + 1 * R0;
    msync((void*)v11, R0, MS_INVALIDATE);
    check(c.slot_for_use(1, 1) < 0 && c.slot_of(1, 1) < 0 && c.st.n_prob_reclaimed == 1, "reclaimed by the OS, unpinned: a miss at its use");
    const int s12 = c.slot_of(1, 2);
    c.slot(1, s12).pins = 1;
    const uint8_t* v12 = c.fmap[1] + 2 * R0;
    msync((void*)v12, R0, MS_INVALIDATE);
    const uint64_t b1 = c.rd.bytes.load();
    check(c.slot_for_use(1, 2) == s12 && c.st.n_prob_reclaimed == 2, "reclaimed, pinned: kept in its slot");
    c.wait(1, s12);
    check(resident(v12, R0) && c.rd.bytes.load() - b1 == R0 && same_as_file(c.rec(1, s12), 1, 2, R0), "... read again in place by a reader (urgent)");
    c.unpin(1, s12);
    // a promotion whose pages the OS took: dropped, not copied from the drive on the compute thread
    c.prom_uses = 2;
    {
      const int s13 = c.slot_of(1, 3);
      c.slot(1, s13).pins = 1;
      done(c, 1, s13, 5);  // a later token: marked
      msync(c.fmap[1] + 3 * R0, R0, MS_INVALIDATE);
      c.tok_now = 5;
      c.tier_step();
      check(c.slot_of(1, 3) < 0 && c.st.n_promoted == 1 && c.st.n_prob_reclaimed == 3, "a promotion of reclaimed pages: dropped, no copy");
    }
    // a drop that a read of the same record overtakes is cancelled (the pages are wanted again), never run after it
    {
      const uint64_t c0 = c.rd.drops_cancelled.load(), d0 = c.rd.drops_done.load();
      // record 0 of layer 1 was dropped for the budget: read it again now, and check its pages are resident after it
      const int s10 = miss(c, 1, 0, 3);
      c.rd.drops_flush();
      check(s10 >= 0 && resident(c.fmap[1], R0) && same_as_file(c.rec(1, s10), 1, 0, R0) && c.rd.drops_cancelled.load() + c.rd.drops_done.load() >= c0 + d0,
            "a read after a drop of the same record: its pages resident, the bytes the file's");
      done(c, 1, s10, 3);
    }
    // probation loses a record while its copy is in flight (the budget, the OS): the copy is installed as it is
    {
      c.prom_margin = -2.0;
      const int s14 = c.slot_of(1, 4);
      c.slot(1, s14).pins = 1;
      done(c, 1, s14, 9);
      c.tok_now = 9;
      const uint64_t started = c.st.n_promote_started, promoted = c.st.n_promoted;
      c.tier_step();
      msync(c.fmap[1] + 4 * R0, R0, MS_INVALIDATE);  // the OS takes it before its use
      check(c.slot_for_use(1, 4) < 0 && c.st.n_promote_started == started + 1, "in flight, reclaimed: the use misses");
      c.rd.drops_flush();
      c.tier_step();
      const int p14 = c.slot_of(1, 4);
      check(p14 >= 0 && p14 < S.n && same_as_file(c.rec(1, p14), 1, 4, R0) && c.st.n_promoted == promoted + 1, "... and the landed copy is installed in the slab");
    }
    // names land in probation too
    const int sn = c.name(0, 9, 12);
    check(sn >= S.n && c.slot(0, sn).pf_pins == 1, "a name's read lands in probation");
    c.wait(0, sn);
    c.unpin_pf(0, sn, 9);
    // the pause: probation drained at the next boundary, later reads go to the slab, files F_NOCACHE again
    c.spare_pause();
    check(!c.prob_on() && c.tier_due(), "paused: no read lands in probation");
    c.tier_step();
    c.rd.drops_flush();
    check(c.prob_held(0) == 0 && !c.tier_due(), "the pause drains probation");
    bool gone = true;
    for (int e = 5; e < 7; e++) gone &= none_resident(c.fmap[1] + (size_t)e * R0, R0);
    check(gone && none_resident(c.fmap[0] + 9 * R0, R0), "... and drops its pages");
    const int s2 = miss(c, 0, 11, 20);
    check(s2 >= 0 && s2 < S.n && same_as_file(c.rec(0, s2), 0, 11, R0), "after the pause a miss goes to the slab");
    done(c, 0, s2, 20);
  }
  // ---------------------------------------------------------------- two pools (Inkling's two record formats)
  {
    const size_t rec[2] = {R0, R1};
    write_store(dir, rec);
    Cache c;
    c.file_under = true;
    c.prob_bytes = (int64_t)(4 * R0 + 4 * R1) * 2;  // shares 0.5 / 0.5 (pool 1: at most the 4 records its slab cannot hold)
    if (!c.init(NL, NE, K, {0, 1}, {0, 1}, {R0, R1}, {0.5, 0.5}, dir, "layer_%02d.bin", 12.0 * R0, 2, 1, Mem::PLAIN, false, err)) {
      printf("FAIL init (two pools): %s\n", err.c_str());
      return 1;
    }
    check(c.P[0].np > 0 && c.P[1].np > 0, "two pools: each has probation slots");
    check(fill_slab(c, 0, {8, 9, 10, 11, 12, 13}, R0) && fill_slab(c, 1, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, R1), "two pools: both slabs filled");
    const int a = miss(c, 0, 1, 0), b = miss(c, 1, 2, 0);
    check(a >= c.P[0].n && b >= c.P[1].n && same_as_file(c.rec(1, b), 1, 2, R1), "two pools: reads land in each pool's probation");
    done(c, 0, a, 0);
    done(c, 1, b, 0);
    c.prom_uses = 2;
    c.prom_margin = -2.0;
    c.slot(1, b).pins = 1;
    done(c, 1, b, 1);
    c.tok_now = 1;
    c.tier_step();
    c.rd.drops_flush();
    c.tier_step();
    c.rd.drops_flush();
    const int pb = c.slot_of(1, 2);
    check(pb >= 0 && pb < c.P[1].n && same_as_file(c.rec(1, pb), 1, 2, R1) && none_resident(c.fmap[1] + 2 * R1, R1), "two pools: the second pool's record promoted");
  }
  // ---------------------------------------------------------------- the tier off
  {
    const size_t rec[2] = {R0, R0};
    write_store(dir, rec);
    Cache c;
    if (!c.init(NL, NE, K, {0, 1}, {0, 0}, {R0}, {1.0}, dir, "layer_%02d.bin", 8.0 * R0, 2, 1, Mem::PLAIN, false, err)) {
      printf("FAIL init (off): %s\n", err.c_str());
      return 1;
    }
    const int s = miss(c, 0, 5, 0);
    check(!c.prob_on() && !c.file_under && c.P[0].np == 0 && c.fmap.empty() && s < c.P[0].n && same_as_file(c.rec(0, s), 0, 5, R0),
          "the tier off: nothing mapped, the read goes to the slab");
    done(c, 0, s, 0);
    c.slot(0, s).pins = 1;
    done(c, 0, s, 1);
    check(!c.tier_due() && c.slot_for_use(0, 5) == s, "the tier off: slot_for_use = slot_of, nothing to promote");
  }
  if (system(("rm -rf '" + dir + "'").c_str()) != 0) fails++;
  printf("%s: %d failed\n", fails ? "SPARE TIER CHECK FAILED" : "SPARE TIER CHECK PASSED", fails);
  return fails ? 1 : 0;
}
