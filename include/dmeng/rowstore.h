// A second store beside the expert records: fixed-size rows in part files, read by row id on the store's own reader
// threads (Qwen3.8-Flash-Next's n-gram table: 320M rows of 162 bytes in 16 parts, 16 rows per token). A row's id
// follows from token ids alone, so a model issues a batch of rows the moment the ids are known (a token at the start of
// its step, a prompt span before its first layer) and takes it where the rows are used, normally long after they have
// landed. Reads: the 16 KB page(s) holding each row, rows sharing a page share one read, pread with F_NOCACHE (as the
// expert records), grouped in file order and spread over the readers.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace dm {

class RowStore {
 public:
  struct Batch {
    int n = 0;
    std::vector<uint8_t> data;  // [n][bpr], row i of the batch at i * bpr
    std::atomic<int> left{0};   // reads not done
    std::chrono::steady_clock::time_point t_issue;
  };
  // paths[p]: part p, holding part_rows[p] rows; global row r is row r - p * rows_per_part of part r / rows_per_part
  bool open(const std::vector<std::string>& paths, const std::vector<int64_t>& part_rows, int64_t rows_per_part, int bpr, int threads,
            std::string& err);
  ~RowStore();
  int bpr() const { return bpr_; }
  int64_t rows() const { return rows_; }
  // start reading rows[0..n) (ids checked: an id outside the table aborts; the caller's hash makes them)
  std::shared_ptr<Batch> issue(const int64_t* rows, int n);
  // wait until every row of b has landed
  void wait(Batch& b);
  // statistics since reset_stats(): rows, page reads, bytes read, ms the computation waited, ms from issue to use
  std::atomic<uint64_t> n_rows{0}, n_reads{0}, n_bytes{0};
  double wait_ms = 0, lead_ms = 0, uses = 0;
  void reset_stats();

 private:
  struct Item {
    int i;         // row index in the batch
    int64_t off;   // byte offset of the row in its part
  };
  struct Job {
    std::shared_ptr<Batch> b;
    int part = 0;
    int64_t pg0 = 0, pg1 = 0;  // pages [pg0, pg1]
    std::vector<Item> items;
  };
  void loop();
  std::vector<int> fds_;
  std::vector<int64_t> size_;  // bytes per part
  int64_t rows_ = 0, rpp_ = 0;
  int bpr_ = 0;
  std::vector<std::thread> th_;
  std::mutex mu_;
  std::condition_variable cv_, done_;
  std::deque<Job> q_;
  bool stop_ = false;
};

}  // namespace dm
