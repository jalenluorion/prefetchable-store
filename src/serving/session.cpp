// The prefix cache (include/dmeng/session.h).
#include "dmeng/session.h"
#include "dmeng/pagevec.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

namespace dm {

using sclk = std::chrono::steady_clock;
static double ms_from(sclk::time_point t) { return std::chrono::duration<double, std::milli>(sclk::now() - t).count(); }

struct PrefixCache::Ckpt {  // (state blobs in pages from the system: a dropped one is given back, dmeng/pagevec.h)
  int pos = 0;
  PageVec<uint8_t> fixed;
  uint64_t used = 0;
  bool pinned = false;
};
struct PrefixCache::Parked {
  std::vector<int> toks;
  PageVec<uint8_t> fixed, posd;
  std::vector<float> logits;
  int logits_k = 0;                       // the head k they were computed with (LM::set_head_k; 0 = every logit)
  std::vector<std::unique_ptr<Ckpt>> ck;  // the live sequence's checkpoints, moved with it
  uint64_t used = 0;
  bool from_ahead = false;  // parked by ahead() (the tail a predicted next prompt replaces)
  size_t bytes() const {
    size_t b = fixed.size() + posd.size() + logits.size() * 4;
    for (auto& c : ck) b += c->fixed.size();
    return b;
  }
};

static int lcp(const std::vector<int>& a, const std::vector<int>& b) {
  size_t n = std::min(a.size(), b.size()), i = 0;
  while (i < n && a[i] == b[i]) i++;
  return (int)i;
}

// ------------------------------------------------------------------------------------------------ disk format
// "DMPC" u32 2 | i32 n | u64 fixed_bytes | u64 pos_bytes | i32 nck | i32 ck_pos[nck] | i32 nexp | i32 tokens[n]
// | fixed blobs [nck] (in ck_pos order) | positional [0, n) | i32 experts[nexp]
// The header is written first (sizes known), so the checkpoint blobs can be written while the prefill computes them and
// the positional part / expert set at persist time; the file is renamed into place only when complete.
using DiskHdr = PrefixCache::PendHdr;
static bool write_hdr(int fd, const DiskHdr& h) {
  std::vector<uint8_t> b;
  auto put = [&](const void* p, size_t k) { b.insert(b.end(), (const uint8_t*)p, (const uint8_t*)p + k); };
  uint32_t ver = 2;
  put("DMPC", 4);
  put(&ver, 4);
  put(&h.n, 4);
  put(&h.fb, 8);
  put(&h.pb, 8);
  put(&h.nck, 4);
  put(h.ck.data(), 4 * h.ck.size());
  put(&h.nexp, 4);
  put(h.toks.data(), 4 * h.toks.size());
  return pwrite(fd, b.data(), b.size(), 0) == (ssize_t)b.size();
}
static bool read_hdr(int fd, DiskHdr& h) {
  char mg[4];
  uint32_t ver = 0;
  if (pread(fd, mg, 4, 0) != 4 || memcmp(mg, "DMPC", 4) || pread(fd, &ver, 4, 4) != 4 || ver != 2) return false;
  long o = 8;
  if (pread(fd, &h.n, 4, o) != 4 || h.n <= 0 || h.n > (1 << 24)) return false;
  o += 4;
  if (pread(fd, &h.fb, 8, o) != 8 || pread(fd, &h.pb, 8, o + 8) != 8) return false;
  o += 16;
  if (pread(fd, &h.nck, 4, o) != 4 || h.nck < 0 || h.nck > 100000) return false;
  o += 4;
  h.ck.resize(h.nck);
  if (pread(fd, h.ck.data(), 4L * h.nck, o) != 4L * h.nck) return false;
  o += 4L * h.nck;
  if (pread(fd, &h.nexp, 4, o) != 4) return false;
  o += 4;
  h.toks.resize(h.n);
  return pread(fd, h.toks.data(), 4L * h.n, o) == 4L * h.n;
}
static uint64_t fnv(const int* p, int n) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < n; i++) {
    h ^= (uint32_t)p[i];
    h *= 1099511628211ull;
  }
  return h;
}

PrefixCache::PrefixCache(LM& m, const SessionOptions& o) : m_(m), o_(o) {
  live_logits_.resize(m.info().vocab);
  if (!o_.disk_dir.empty()) mkdir(o_.disk_dir.c_str(), 0755);
}
PrefixCache::~PrefixCache() {
  if (pend_fd_ >= 0) {
    close(pend_fd_);
    unlink(pend_tmp_.c_str());
  }
}

void PrefixCache::add_checkpoint(std::unique_ptr<Ckpt> c) {
  for (auto& x : ck_)
    if (x->pos == c->pos) {
      x->pinned |= c->pinned;
      x->used = ++clock_;
      x->fixed.swap(c->fixed);
      return;
    }
  c->used = ++clock_;
  ck_.push_back(std::move(c));
  std::sort(ck_.begin(), ck_.end(), [](auto& a, auto& b) { return a->pos < b->pos; });
  trim_checkpoints();
}

void PrefixCache::checkpoint(int p, bool pinned) {
  auto c = std::make_unique<Ckpt>();
  c->pos = p;
  c->pinned = pinned;
  c->fixed.resize(m_.info().fixed_bytes);
  m_.save_fixed(c->fixed.data());
  add_checkpoint(std::move(c));
}

void PrefixCache::trim_checkpoints() {
  const size_t each = m_.info().fixed_bytes, cap = std::max((size_t)(o_.ckpt_gb * 1e9), (size_t)std::max(0, o_.ckpt_min) * each);
  while (ck_.size() * each > cap && !ck_.empty()) {
    int best = -1;  // least recently used unpinned, never the newest position (the end of the last prompt)
    for (int i = 0; i < (int)ck_.size(); i++) {
      if (ck_[i]->pinned || i + 1 == (int)ck_.size() || ck_[i]->pos <= protect_upto_) continue;
      if (best < 0 || ck_[i]->used < ck_[best]->used) best = i;
    }
    if (best < 0) break;
    ck_.erase(ck_.begin() + best);
  }
}

void PrefixCache::drop_checkpoints_after(int p) {
  ck_.erase(std::remove_if(ck_.begin(), ck_.end(), [&](auto& c) { return c->pos > p; }), ck_.end());
}

void PrefixCache::park_live(int keep_upto) {
  if (live_.empty()) return;
  auto pk = std::make_unique<Parked>();
  pk->toks = live_;
  pk->fixed.resize(m_.info().fixed_bytes);
  m_.save_fixed(pk->fixed.data());
  pk->posd.resize(live_.size() * m_.info().pos_bytes);
  m_.save_pos(0, (int)live_.size(), pk->posd.data());
  if (live_logits_ok_) pk->logits = live_logits_, pk->logits_k = live_k_;
  // checkpoints past the point the live sequence resumes from go with it (so a return to this conversation can resume
  // inside it, e.g. at the end of its last prompt when its re-rendered last turn differs from what was generated)
  for (auto& c : ck_)
    if (c && c->pos > keep_upto) pk->ck.push_back(std::move(c));
  ck_.erase(std::remove_if(ck_.begin(), ck_.end(), [](auto& c) { return !c; }), ck_.end());
  pk->used = ++clock_;
  const size_t cap = (size_t)(o_.park_gb * 1e9);
  while (pk->bytes() > cap && !pk->ck.empty()) pk->ck.erase(pk->ck.begin());  // oldest positions first
  if (pk->bytes() > cap) return;
  size_t tot = pk->bytes();
  for (auto& p : parked_) tot += p->bytes();
  while (tot > cap && !parked_.empty()) {
    auto it = std::min_element(parked_.begin(), parked_.end(), [](auto& a, auto& b) { return a->used < b->used; });
    tot -= (*it)->bytes();
    parked_.erase(it);
  }
  parked_.push_back(std::move(pk));
  n_parks_++;
}

// ------------------------------------------------------------------------------------------------ disk
std::string PrefixCache::disk_path(const std::vector<int>& ids, int p, bool pin) const {
  char name[64];
  snprintf(name, sizeof name, pin ? "/prefix_%016llx.pin.bin" : "/prefix_%016llx.bin", (unsigned long long)fnv(ids.data(), p));
  return o_.disk_dir + name;
}
std::string PrefixCache::disk_existing(const std::vector<int>& ids, int p) const {
  struct stat sb;
  for (bool pin : {true, false}) {
    std::string q = disk_path(ids, p, pin);
    if (stat(q.c_str(), &sb) == 0) return q;
  }
  return "";
}

int PrefixCache::best_disk(const std::vector<int>& ids, std::string& path, int cut) const {
  if (o_.disk_dir.empty()) return 0;
  DIR* d = opendir(o_.disk_dir.c_str());
  if (!d) return 0;
  int best = 0;
  while (dirent* e = readdir(d)) {
    std::string nm = e->d_name;
    if (nm.size() < 5 || nm.substr(nm.size() - 4) != ".bin") continue;
    std::string p = o_.disk_dir + "/" + nm;
    int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) continue;
    DiskHdr h;
    if (read_hdr(fd, h)) {  // rank a file by the checkpoint it can resume from, not by the raw shared prefix
      const int k = std::min(lcp(h.toks, ids), (int)ids.size() - cut);
      int u = 0;
      for (int q : h.ck)
        if (q <= k) u = std::max(u, q);
      if (u > best) {
        best = u;
        path = p;
      }
    }
    close(fd);
  }
  closedir(d);
  return best;
}

bool PrefixCache::load_disk(const std::vector<int>& ids, int& got, double& ms_preload, int& npre, int cut) {
  std::string path;
  int k = best_disk(ids, path, cut);
  got = 0;
  if (k <= 0) return false;
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) return false;
  DiskHdr h;
  if (!read_hdr(fd, h) || h.fb != m_.info().fixed_bytes || h.pb != m_.info().pos_bytes) {
    close(fd);
    return false;
  }
  const int lim = std::min(k, (int)ids.size() - cut);  // the request's last token must be computed (its logits) unless cut 0
  int ci = -1;
  for (int i = 0; i < h.nck; i++)
    if (h.ck[i] <= lim && (ci < 0 || h.ck[i] > h.ck[ci])) ci = i;
  if (ci < 0) {
    close(fd);
    return false;
  }
  const int use = h.ck[ci];
  fcntl(fd, F_RDAHEAD, 1);
  std::vector<uint8_t> buf(std::max((size_t)h.fb, (size_t)use * (size_t)h.pb));
  if (pread(fd, buf.data(), (size_t)use * h.pb, h.pos_off()) != (ssize_t)((size_t)use * h.pb)) {
    close(fd);
    return false;
  }
  m_.load_pos(0, use, buf.data());
  if (pread(fd, buf.data(), h.fb, h.blobs_off() + (long)ci * (long)h.fb) != (ssize_t)h.fb) {
    close(fd);
    return false;
  }
  m_.load_fixed(buf.data(), use);
  std::vector<int32_t> ex(h.nexp);
  if (h.nexp > 0 && pread(fd, ex.data(), 4L * h.nexp, h.exp_off()) != 4L * h.nexp) ex.clear();
  close(fd);
  utimes(path.c_str(), nullptr);
  live_.assign(ids.begin(), ids.begin() + use);
  ck_.clear();
  live_logits_ok_ = false;
  checkpoint(use, true);
  got = use;
  n_disk_loads_++;
  if (o_.preload_experts && !ex.empty()) {
    auto t = sclk::now();
    m_.preload_experts(ex);
    ms_preload = ms_from(t);
    npre = (int)ex.size();
  }
  return true;
}

// a file for the prefix [0, p) of ids, whose dense checkpoints the coming prefill writes as it computes them
void PrefixCache::begin_disk(const std::vector<int>& ids, int p, int start) {
  if (pend_fd_ >= 0) {
    close(pend_fd_);
    unlink(pend_tmp_.c_str());
    pend_fd_ = -1;
  }
  pend_path_ = disk_path(ids, p, true);  // the first marked prefix: the system boundary, pinned
  if (!disk_existing(ids, p).empty()) return;  // already on disk
  DiskHdr h;
  h.n = p;
  h.fb = m_.info().fixed_bytes;
  h.pb = m_.info().pos_bytes;
  for (int q = (start / o_.disk_every + 1) * o_.disk_every; q < p; q += o_.disk_every) h.ck.push_back(q);
  h.ck.push_back(p);
  h.nck = (int)h.ck.size();
  h.toks.assign(ids.begin(), ids.begin() + p);
  h.nexp = 0;  // filled at persist time
  pend_tmp_ = pend_path_ + ".tmp";
  pend_fd_ = open(pend_tmp_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (pend_fd_ < 0) return;
  if (!write_hdr(pend_fd_, h)) {
    close(pend_fd_);
    unlink(pend_tmp_.c_str());
    pend_fd_ = -1;
    return;
  }
  pend_ = h;
  pend_ids_ = h.toks;
}

void PrefixCache::persist(int p) {
  if (o_.disk_dir.empty() || p < o_.disk_min || p > (int)live_.size()) return;
  std::vector<int> pre(live_.begin(), live_.begin() + p);
  if (pend_fd_ >= 0 && pend_.n == p && pend_ids_ == pre) {  // the dense file the prefill wrote: add KV + experts, finish
    std::vector<int32_t> ex = m_.held_experts();
    pend_.nexp = (int)ex.size();
    std::vector<uint8_t> pd((size_t)p * pend_.pb);
    m_.save_pos(0, p, pd.data());
    bool ok = write_hdr(pend_fd_, pend_) && pwrite(pend_fd_, pd.data(), pd.size(), pend_.pos_off()) == (ssize_t)pd.size() &&
              pwrite(pend_fd_, ex.data(), 4 * ex.size(), pend_.exp_off()) == (ssize_t)(4 * ex.size()) && fsync(pend_fd_) == 0;
    close(pend_fd_);
    pend_fd_ = -1;
    if (ok && pend_ok_) {
      rename(pend_tmp_.c_str(), pend_path_.c_str());
      n_disk_writes_++;
    } else {
      unlink(pend_tmp_.c_str());
    }
    enforce_disk_cap(pend_path_);
    return;
  }
  // otherwise: a file from the in-memory checkpoints at or before p (needs one at p; the live end is checkpointed here)
  {
    std::string have = disk_existing(live_, p);
    if (!have.empty()) {
      const bool pin = have.size() > 8 && have.compare(have.size() - 8, 8, ".pin.bin") == 0;
      if (pin && o_.refresh_pinned) {  // the expert list sits at the end: write the new one, then its count, then the size
        int fd = open(have.c_str(), O_RDWR);
        DiskHdr h;
        if (fd >= 0 && read_hdr(fd, h)) {
          std::vector<int32_t> ex = m_.held_experts();
          if (!ex.empty() && pwrite(fd, ex.data(), 4 * ex.size(), h.exp_off()) == (ssize_t)(4 * ex.size())) {
            h.nexp = (int)ex.size();
            if (write_hdr(fd, h)) ftruncate(fd, h.exp_off() + 4L * h.nexp);
          }
        }
        if (fd >= 0) close(fd);
      }
      utimes(have.c_str(), nullptr);
      return;
    }
  }
  const Ckpt* end = nullptr;
  for (auto& c : ck_)
    if (c->pos == p) end = c.get();
  if (!end && p == (int)live_.size()) {  // the end of a reply (decode leaves no checkpoint): the state now is exactly p's
    checkpoint(p, false);
    for (auto& c : ck_)
      if (c->pos == p) end = c.get();
  }
  if (!end) return;
  const std::string path = disk_path(live_, p, end->pinned);
  DiskHdr h;
  h.n = p;
  h.fb = m_.info().fixed_bytes;
  h.pb = m_.info().pos_bytes;
  // the end and the pinned (system boundary) states only: a restore resumes at the end, an edit near the start from the
  // boundary (P2: every checkpoint <= p was 63 MB each and 3.9 s of flash writes per reply on the phone)
  std::vector<const Ckpt*> cs;
  for (auto& c : ck_)
    if (c->pos <= p && (c->pos == p || c->pinned)) {
      cs.push_back(c.get());
      h.ck.push_back(c->pos);
    }
  h.nck = (int)cs.size();
  h.toks = pre;
  std::vector<int32_t> ex = m_.held_experts();
  h.nexp = (int)ex.size();
  std::string tmp = path + ".tmp";
  int fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return;
  bool ok = write_hdr(fd, h);
  for (int i = 0; ok && i < h.nck; i++) ok = pwrite(fd, cs[i]->fixed.data(), h.fb, h.blobs_off() + (long)i * (long)h.fb) == (ssize_t)h.fb;
  std::vector<uint8_t> pd((size_t)p * h.pb);
  m_.save_pos(0, p, pd.data());
  ok = ok && pwrite(fd, pd.data(), pd.size(), h.pos_off()) == (ssize_t)pd.size() && pwrite(fd, ex.data(), 4 * ex.size(), h.exp_off()) == (ssize_t)(4 * ex.size());
  close(fd);
  if (!ok) {
    unlink(tmp.c_str());
    return;
  }
  rename(tmp.c_str(), path.c_str());
  n_disk_writes_++;
  enforce_disk_cap(path);
}

void PrefixCache::enforce_disk_cap(const std::string& keep) {
  DIR* d = opendir(o_.disk_dir.c_str());
  if (!d) return;
  struct stat sb;
  std::vector<std::pair<long, std::pair<std::string, long>>> files;
  long tot = 0;
  while (dirent* e = readdir(d)) {
    std::string nm = e->d_name;
    if (nm.size() < 5 || nm.substr(nm.size() - 4) != ".bin") continue;
    std::string fp = o_.disk_dir + "/" + nm;
    if (stat(fp.c_str(), &sb) == 0) {
      files.push_back({(long)sb.st_mtime, {fp, (long)sb.st_size}});
      tot += sb.st_size;
    }
  }
  closedir(d);
  std::sort(files.begin(), files.end());
  // pinned files (the system boundary's, "*.pin.bin") stay while they total at most half of the cap
  auto pinned = [](const std::string& f) { return f.size() > 8 && f.compare(f.size() - 8, 8, ".pin.bin") == 0; };
  long pin_tot = 0;
  for (auto& fl : files)
    if (pinned(fl.second.first)) pin_tot += fl.second.second;
  for (int pass = 0; pass < 2; pass++)
    for (auto& fl : files) {
      if (tot <= (long)(o_.disk_gb * 1e9) || fl.second.first == keep || fl.second.second < 0) continue;
      const bool pn = pinned(fl.second.first);
      if (pn && (pass == 0 || pin_tot <= (long)(o_.disk_gb * 0.5e9))) continue;
      unlink(fl.second.first.c_str());
      tot -= fl.second.second;
      if (pn) pin_tot -= fl.second.second;
      fl.second.second = -1;
    }
}

// ------------------------------------------------------------------------------------------------ sync
SyncResult PrefixCache::sync(const std::vector<int>& ids, const std::vector<int>& marks, float* logits, bool want_logits) {
  SyncResult R;
  if (!no_park_) {  // a request (not ahead() itself)
    n_req_++;
    ahead_seen(ids);
  }
  const int n = (int)ids.size();
  const int cut = want_logits ? 1 : 0;  // state only: a checkpoint at n itself is enough
  auto t0 = sclk::now();
  const int k = lcp(live_, ids);
  R.lcp_live = k;
  R.live_len = (int)live_.size();
  int u_live = 0;
  // stored logits serve this request when they were complete, or screened for at least the top k it needs
  const int need_k = m_.head_k();
  auto fits = [&](int lk) { return lk == 0 || (need_k > 0 && lk >= need_k); };
  if (live_logits_ok_ && !fits(live_k_)) live_logits_ok_ = false;
  bool exact_live = k == n && n == (int)live_.size() && live_logits_ok_;
  if (exact_live || (!want_logits && k == n && n == (int)live_.size())) {
    u_live = n;
  } else if (k == (int)live_.size() && k < n) {
    u_live = k;
  } else {
    for (auto& c : ck_)
      if (c->pos <= std::min(k, n - cut)) u_live = std::max(u_live, c->pos);
  }
  int u_park = 0, ip = -1;
  for (int i = 0; i < (int)parked_.size(); i++) {
    const Parked& p = *parked_[i];
    int kp = lcp(p.toks, ids), u = 0;
    if (kp == (int)p.toks.size() && (kp < n || (kp == n && ((!p.logits.empty() && fits(p.logits_k)) || !want_logits)))) u = kp;
    for (auto& c : p.ck)
      if (c->pos <= std::min(kp, n - cut)) u = std::max(u, c->pos);
    if (u > u_park) {
      u_park = u;
      ip = i;
    }
  }
  std::string dpath;
  const int k_disk = best_disk(ids, dpath, cut);
  R.lcp_disk = k_disk;
  int start = u_live;
  R.source = u_live == 0 ? "cold" : (u_live == (int)live_.size() ? "live" : "checkpoint");
  const bool use_park = ip >= 0 && u_park > u_live;
  bool use_disk = !use_park && k_disk > u_live + o_.disk_gain;  // k_disk: the best resumable disk checkpoint; a load costs ~0.1-0.2 s
  std::unique_ptr<Parked> up;
  if (use_park) {  // take it out first: parking the live sequence below must not evict it
    up = std::move(parked_[ip]);
    parked_.erase(parked_.begin() + ip);
  }
  // park the live conversation only for an unrelated request (one sharing less than half of it: a client's title request,
  // another session); a request that rewrote the live conversation's own tail (a new user turn after the template dropped
  // the previous reasoning) will not come back to that tail, so parking it would only hold memory
  if (!no_park_ && (use_park || use_disk || u_live < (int)live_.size()) && (int)live_.size() - std::max(u_live, k) >= o_.park_min &&
      2 * k < (int)live_.size())
    park_live(use_park || use_disk ? 0 : u_live);
  if (use_park) {
    Parked& p = *up;
    const int u = u_park;
    m_.load_pos(0, u, p.posd.data());  // the positional entries [0, u) are a prefix of the parked buffer (layout per range)
    ck_.clear();
    if (u == (int)p.toks.size()) {
      m_.load_fixed(p.fixed.data(), u);
      live_logits_ok_ = !p.logits.empty() && fits(p.logits_k);
      if (live_logits_ok_) live_logits_ = p.logits, live_k_ = p.logits_k;
    } else {
      for (auto& c : p.ck)
        if (c->pos == u) m_.load_fixed(c->fixed.data(), u);
      live_logits_ok_ = false;
    }
    live_.assign(p.toks.begin(), p.toks.begin() + u);
    for (auto& c : p.ck)
      if (c->pos <= u) ck_.push_back(std::move(c));
    std::sort(ck_.begin(), ck_.end(), [](auto& a, auto& b) { return a->pos < b->pos; });
    if (ck_.empty() || ck_.back()->pos != u) checkpoint(u, false);
    start = u_park;
    R.source = "parked";
    exact_live = false;
    n_unparks_++;
  } else if (use_disk) {
    int got = 0;
    if (load_disk(ids, got, R.ms_preload, R.preloaded, cut) && got > u_live) {
      start = got;
      R.source = "disk";
    } else {
      use_disk = false;
    }
  }
  if (!use_park && !use_disk && start < (int)live_.size()) {
    if (start == 0) {
      live_.clear();
      ck_.clear();
      m_.reset();
    } else {
      for (auto& c : ck_)
        if (c->pos == start) {
          m_.load_fixed(c->fixed.data(), start);
          c->used = ++clock_;
        }
      live_.resize(start);
      drop_checkpoints_after(start);
    }
    live_logits_ok_ = false;
  }
  R.ms_restore = ms_from(t0);
  R.reused = start;
  if (start == n && (live_logits_ok_ || !want_logits)) {
    if (logits && live_logits_ok_) std::memcpy(logits, live_logits_.data(), live_logits_.size() * 4);
    n_reused_ += n;
    return R;
  }
  // checkpoints wanted from this prefill (taken inside the run, no split): message boundaries, the prompt's end, every
  // ckpt_every positions in memory; and, if the first long marked prefix (the system block) is not on disk yet, every
  // disk_every positions of it straight into its file
  int first_mark = -1;
  std::map<int, std::unique_ptr<Ckpt>> mem;  // absolute position -> blob being assembled
  std::map<int, int> disk_idx;              // absolute position -> index in the pending file
  for (int mk : marks)
    if (mk > 0 && mk <= n && first_mark < 0 && mk >= o_.disk_min) first_mark = mk;
  auto want_mem = [&](int p, bool pin) {
    if (p <= start || p > n) return;
    auto& c = mem[p];
    if (!c) {
      c = std::make_unique<Ckpt>();
      c->pos = p;  // (its blob is allocated below, once the memory cap's rule has kept it)
    }
    c->pinned |= pin;
  };
  for (int mk : marks) want_mem(mk, mk == first_mark);
  want_mem(n, false);
  if (o_.ckpt_every > 0)
    for (int p = (start / o_.ckpt_every + 1) * o_.ckpt_every; p < n; p += o_.ckpt_every) want_mem(p, false);
  // only the checkpoints that the memory cap keeps once this prefill's are added (trim_checkpoints after each add, run
  // here on positions alone) are filled: the others were evicted right after the prefill, but every one of them was a
  // whole fixed state in memory during it (Flash-Next: 118 MB each, six for a cold agent prompt). The kept set is the
  // same as before; a prefill cancelled after some of its spans keeps fewer of the skipped intermediate ones.
  {
    struct Sim {
      int pos;
      bool pinned;
      uint64_t used;
      bool fresh;
    };
    std::vector<Sim> sim;
    for (auto& c : ck_) sim.push_back({c->pos, c->pinned, c->used, false});
    const size_t each = m_.info().fixed_bytes, cap = std::max((size_t)(o_.ckpt_gb * 1e9), (size_t)std::max(0, o_.ckpt_min) * each);
    uint64_t clk = clock_;
    for (auto& kv : mem) {
      sim.push_back({kv.first, kv.second->pinned, ++clk, true});
      std::sort(sim.begin(), sim.end(), [](const Sim& a, const Sim& b) { return a.pos < b.pos; });
      while (sim.size() * each > cap && !sim.empty()) {
        int best = -1;
        for (int i = 0; i < (int)sim.size(); i++) {
          if (sim[i].pinned || i + 1 == (int)sim.size() || sim[i].pos <= protect_upto_) continue;
          if (best < 0 || sim[i].used < sim[best].used) best = i;
        }
        if (best < 0) break;
        sim.erase(sim.begin() + best);
      }
    }
    for (auto it = mem.begin(); it != mem.end();) {
      const bool kept = std::any_of(sim.begin(), sim.end(), [&](const Sim& s) { return s.fresh && s.pos == it->first; });
      it = kept ? std::next(it) : mem.erase(it);
    }
  }
  // the prompt's end is the state the prefill leaves: saved from it afterwards (not filled during the pass, beside the
  // pass's scratch: one fixed state less at the prefill's peak)
  std::unique_ptr<Ckpt> end_ck;
  if (auto e = mem.find(n); e != mem.end()) {
    end_ck = std::move(e->second);
    mem.erase(e);
  }
  for (auto& kv : mem) kv.second->fixed.resize(m_.info().fixed_bytes);
  pend_ok_ = true;
  if (!o_.disk_dir.empty() && first_mark > start && o_.disk_every > 0) {
    begin_disk(ids, first_mark, start);
    if (pend_fd_ >= 0)
      for (int i = 0; i < pend_.nck; i++) disk_idx[pend_.ck[i]] = i;
  }
  std::vector<int> caps;
  for (auto& kv : mem) caps.push_back(kv.first - start);
  for (auto& kv : disk_idx) caps.push_back(kv.first - start);
  auto tp = sclk::now();
  const int base = start;
  bool ok = m_.prefill(ids.data() + start, n - start, live_logits_.data(), caps,
                       [&](int p, size_t off, const void* data, size_t bytes) {
                         auto it = mem.find(p);
                         if (it != mem.end()) std::memcpy(it->second->fixed.data() + off, data, bytes);
                         auto di = disk_idx.find(p);
                         if (di != disk_idx.end() && pend_fd_ >= 0) {
                           const long o = pend_.blobs_off() + (long)di->second * (long)pend_.fb + (long)off;
                           if (pwrite(pend_fd_, data, bytes, o) != (ssize_t)bytes) pend_ok_ = false;
                         }
                       },
                       &R.pf);
  R.ms_prefill = ms_from(tp);
  live_.assign(ids.begin(), ids.begin() + m_.pos());
  for (auto& kv : mem)
    if (kv.first <= m_.pos()) add_checkpoint(std::move(kv.second));
  if (ok && end_ck && m_.pos() == n) {
    end_ck->fixed.resize(m_.info().fixed_bytes);
    m_.save_fixed(end_ck->fixed.data());
    add_checkpoint(std::move(end_ck));
  }
  if (!ok) {
    R.cancelled = true;
    live_logits_ok_ = false;
    R.computed = m_.pos() - base;
    pend_ok_ = false;
    return R;
  }
  live_logits_ok_ = true;
  live_k_ = 0;
  if (logits) std::memcpy(logits, live_logits_.data(), live_logits_.size() * 4);
  R.computed = n - base;
  n_reused_ += base;
  n_computed_ += n - base;
  return R;
}

void PrefixCache::step(int id, float* logits) {
  m_.step(id, live_logits_.data());
  live_.push_back(id);
  live_logits_ok_ = true;
  live_k_ = m_.head_k();
  if (logits) std::memcpy(logits, live_logits_.data(), live_logits_.size() * 4);
}

bool PrefixCache::verify(const int* ids, int n) {
  if (!m_.verify(ids, n)) return false;
  ver_ids_.assign(ids, ids + n);
  live_logits_ok_ = false;
  return true;
}

void PrefixCache::commit(int a) {
  m_.commit(a);
  live_.insert(live_.end(), ver_ids_.begin(), ver_ids_.begin() + a);
  std::memcpy(live_logits_.data(), m_.logits_at(a - 1), live_logits_.size() * 4);
  live_logits_ok_ = true;
  live_k_ = m_.head_k();  // (the pass screened its positions as a step does)
}

std::string PrefixCache::stats_json() const {
  std::ostringstream s;
  size_t pb = 0;
  for (auto& p : parked_) pb += p->bytes();
  s << "{\"requests\":" << n_req_ << ",\"reused_positions\":" << n_reused_ << ",\"computed_positions\":" << n_computed_ << ",\"live\":" << live_.size()
    << ",\"checkpoints\":[";
  for (size_t i = 0; i < ck_.size(); i++) s << (i ? "," : "") << ck_[i]->pos;
  s << "],\"checkpoint_mb\":" << ck_.size() * m_.info().fixed_bytes / 1e6 << ",\"parked\":" << parked_.size() << ",\"parked_mb\":" << pb / 1e6
    << ",\"parks\":" << n_parks_ << ",\"unparks\":" << n_unparks_ << ",\"disk_loads\":" << n_disk_loads_ << ",\"disk_writes\":" << n_disk_writes_
    << ",\"ahead\":{\"runs\":" << n_ahead_ << ",\"positions\":" << n_ahead_pos_ << ",\"hits\":" << n_ahead_hits_ << ",\"parks\":" << n_ahead_parks_
    << ",\"pending_len\":" << ahead_ids_.size() << "}}";
  return s.str();
}

// ------------------------------------------------------------------------------------------------ ahead
// A request that reaches the end of the last prediction confirms it: the old tail ahead() parked is not coming back.
void PrefixCache::ahead_seen(const std::vector<int>& ids) {
  if (ahead_ids_.empty()) return;
  if (lcp(ids, ahead_ids_) >= (int)ahead_ids_.size()) {
    n_ahead_hits_++;
    parked_.erase(std::remove_if(parked_.begin(), parked_.end(), [](auto& p) { return p->from_ahead; }), parked_.end());
    ahead_ids_.clear();
  }
}

SyncResult PrefixCache::ahead(const std::vector<int>& ids, const std::vector<int>& marks, bool park_old) {
  SyncResult R;
  const int n = (int)ids.size(), k = lcp(live_, ids);
  R.lcp_live = k;
  R.live_len = (int)live_.size();
  ahead_ids_ = ids;
  if (n == 0 || k >= n) {  // already in the live state
    R.source = "live";
    R.reused = n;
    return R;
  }
  // keep the live conversation whole: park it (its checkpoints past k go with it; those on the shared prefix stay)
  bool have_park = false;
  for (auto& p : parked_)
    if (p->from_ahead && p->toks == live_) have_park = true;  // a resumed ahead() after a cancel: parked already
  if (park_old && !have_park && k < (int)live_.size()) {
    const size_t np = parked_.size();
    const uint64_t parks = n_parks_;
    park_live(k);
    if (n_parks_ > parks && !parked_.empty()) {
      parked_.back()->from_ahead = true;
      n_ahead_parks_++;
    }
    (void)np;
  }
  n_ahead_++;
  protect_upto_ = k;
  no_park_ = true;
  R = sync(ids, marks, nullptr);
  no_park_ = false;
  protect_upto_ = -1;
  n_ahead_pos_ += R.computed;
  return R;
}

}  // namespace dm
