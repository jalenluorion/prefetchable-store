// The row store (include/dmeng/rowstore.h).
#include "dmeng/rowstore.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dm {

constexpr int64_t PAGE = 16384;

bool RowStore::open(const std::vector<std::string>& paths, const std::vector<int64_t>& part_rows, int64_t rows_per_part, int bpr, int threads,
                    std::string& err) {
  bpr_ = bpr;
  rpp_ = rows_per_part;
  rows_ = 0;
  for (size_t p = 0; p < paths.size(); p++) {
    const int fd = ::open(paths[p].c_str(), O_RDONLY);
    if (fd < 0) {
      err = "cannot open " + paths[p];
      return false;
    }
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
    struct stat sb;
    if (fstat(fd, &sb) != 0 || (int64_t)sb.st_size != part_rows[p] * bpr) {
      err = paths[p] + ": " + std::to_string((long long)sb.st_size) + " bytes, expected " + std::to_string((long long)(part_rows[p] * bpr)) +
            " (an incomplete part)";
      ::close(fd);
      return false;
    }
    if (p + 1 < paths.size() && part_rows[p] != rows_per_part) {
      err = paths[p] + ": a part before the last with " + std::to_string((long long)part_rows[p]) + " rows";
      ::close(fd);
      return false;
    }
    fds_.push_back(fd);
    size_.push_back((int64_t)sb.st_size);
    rows_ += part_rows[p];
  }
  for (int t = 0; t < std::max(1, threads); t++) th_.emplace_back([this] { loop(); });
  return true;
}

RowStore::~RowStore() {
  {
    std::lock_guard<std::mutex> g(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : th_) t.join();
  for (int fd : fds_) ::close(fd);
}

std::shared_ptr<RowStore::Batch> RowStore::issue(const int64_t* rows, int n) {
  auto b = std::make_shared<Batch>();
  b->n = n;
  b->data.resize((size_t)n * bpr_);
  b->t_issue = std::chrono::steady_clock::now();
  // the rows in file order, then grouped into reads of consecutive pages
  std::vector<std::pair<int64_t, int>> ord(n);  // (part * 2^40 + offset, row index)
  for (int i = 0; i < n; i++) {
    if (rows[i] < 0 || rows[i] >= rows_) {
      fprintf(stderr, "row store: row %lld outside the table (%lld rows)\n", (long long)rows[i], (long long)rows_);
      abort();
    }
    const int64_t p = rows[i] / rpp_;
    ord[i] = {(p << 40) + (rows[i] - p * rpp_) * bpr_, i};
  }
  std::sort(ord.begin(), ord.end());
  std::vector<Job> jobs;
  for (auto& o : ord) {
    const int p = (int)(o.first >> 40);
    const int64_t off = o.first & ((1ll << 40) - 1), pg = off / PAGE, pe = (off + bpr_ - 1) / PAGE;
    if (!jobs.empty() && jobs.back().part == p && pg <= jobs.back().pg1) {
      jobs.back().pg1 = std::max(jobs.back().pg1, pe);
    } else {
      jobs.emplace_back();
      jobs.back().b = b;
      jobs.back().part = p;
      jobs.back().pg0 = pg;
      jobs.back().pg1 = pe;
    }
    jobs.back().items.push_back({o.second, off});
  }
  n_rows += (uint64_t)n;
  n_reads += (uint64_t)jobs.size();
  b->left.store((int)jobs.size(), std::memory_order_relaxed);
  if (jobs.empty()) return b;
  {
    std::lock_guard<std::mutex> g(mu_);
    for (auto& j : jobs) q_.push_back(std::move(j));
  }
  cv_.notify_all();
  return b;
}

void RowStore::loop() {
  std::vector<uint8_t> buf;
  for (;;) {
    Job j;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
      if (stop_ && q_.empty()) return;
      j = std::move(q_.front());
      q_.pop_front();
    }
    const int64_t base = j.pg0 * PAGE, want = std::min((j.pg1 + 1) * PAGE, size_[j.part]) - base;
    if ((int64_t)buf.size() < want) buf.resize(want);
    int64_t got = 0;
    while (got < want) {
      const ssize_t r = pread(fds_[j.part], buf.data() + got, (size_t)(want - got), (off_t)(base + got));
      if (r <= 0) {
        fprintf(stderr, "row store: short read of part %d at %lld\n", j.part, (long long)(base + got));
        abort();
      }
      got += r;
    }
    n_bytes += (uint64_t)want;
    for (const Item& it : j.items) std::memcpy(j.b->data.data() + (size_t)it.i * bpr_, buf.data() + (it.off - base), (size_t)bpr_);
    if (j.b->left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard<std::mutex> g(mu_);
      done_.notify_all();
    }
  }
}

void RowStore::wait(Batch& b) {
  const auto t0 = std::chrono::steady_clock::now();
  if (b.left.load(std::memory_order_acquire) > 0) {
    for (int spin = 0; spin < 2000 && b.left.load(std::memory_order_acquire) > 0; spin++) __builtin_arm_yield();
    if (b.left.load(std::memory_order_acquire) > 0) {
      std::unique_lock<std::mutex> lk(mu_);
      done_.wait(lk, [&] { return b.left.load(std::memory_order_acquire) == 0; });
    }
  }
  const auto t1 = std::chrono::steady_clock::now();
  wait_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  lead_ms += std::chrono::duration<double, std::milli>(t0 - b.t_issue).count();
  uses += 1;
}

void RowStore::reset_stats() {
  n_rows = 0;
  n_reads = 0;
  n_bytes = 0;
  wait_ms = lead_ms = uses = 0;
}

}  // namespace dm
