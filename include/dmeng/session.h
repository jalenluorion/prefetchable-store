// The prefix cache: keeps a model's state across requests and brings it to a new prompt with the least recomputation.
// Model-agnostic: it sees only dm::LM (lm.h) and token ids.
//
//   live       the model's state, valid for the token sequence `live` (positional entries [0, live.size()))
//   checkpoints  the fixed part of the state at positions on the live sequence (message boundaries the front end marks,
//              the end of each prompt, and every `ckpt_every` positions of a long prefill); a request that shares only
//              a prefix of the live sequence resumes from the nearest checkpoint at or before the divergence
//   parked     other sequences kept whole in memory (fixed part + positional range): a request for an unrelated short
//              prompt (a client's title request) parks the live conversation instead of throwing it away
//   disk       the longest marked prefix of a sequence (an agent's system prompt + tools), with checkpoints inside it and
//              the expert set the session held, so a new process / new session starts warm
// Everything restored is the bits that were saved: the cache never changes what the model computes.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dmeng/draft.h"
#include "dmeng/lm.h"

namespace dm {

struct SessionOptions {
  double ckpt_gb = 0.25;       // memory for checkpoints on the live sequence (the system prefix's are on disk too)
  int ckpt_min = 4;            // ... but room for at least this many whatever their size (Inkling's 173 MB fixed state:
                               // 0.25 GB held one, the prompt's end, so a repeated or diverging request restored nothing;
                               // the 35B's 63 MB states fit four); 0 = ckpt_gb alone (iOS)
  double park_gb = 0.9;        // memory for parked sequences
  int ckpt_every = 2048;       // also checkpoint every this many positions inside a long prefill (memory)
  int disk_every = 256;        // the disk copy of the system prefix keeps a checkpoint every this many positions, so a
                               // change near its end (the date, the working directory) recomputes only the tail
  // the server's rule for disk_every from the model's fixed (recurrent) state, one checkpoint's bytes: every 256
  // positions up to 100 MB a checkpoint (the 35B's 66 MB), every 1024 above (Inkling's 173 MB, Flash-Next's 118 MB)
  static int disk_every_for(size_t fixed_bytes) { return fixed_bytes > (size_t)100e6 ? 1024 : 256; }
  int park_min = 1024;         // park the live sequence only if it would lose at least this many positions
  std::string disk_dir;        // "" = no disk copies
  double disk_gb = 4.0;        // size cap of disk_dir (least recently used files go first)
  int disk_min = 1024;         // write a disk copy for a marked prefix of at least this many positions
  int disk_gain = 256;         // restore from disk only when it saves more than this many positions over the live state
  bool preload_experts = false; // a disk restore also reloads the expert set saved with it (measured: no decode gain, +1.4 s)
  bool refresh_pinned = false;  // persist() at a pinned file's position (the system boundary, after each reply) replaces
                                // its saved expert set with the one held now: a new chat's preload then brings in the
                                // experts the last turn used (its generation-prompt tail included), not the system prompt's
};

struct SyncResult {
  int reused = 0;          // positions not recomputed
  int computed = 0;        // positions prefilled
  std::string source;      // "live", "checkpoint", "parked", "disk", "cold"
  int lcp_live = 0, live_len = 0, lcp_disk = 0;  // where the request left the live sequence / the best disk prefix
  double ms_restore = 0, ms_prefill = 0, ms_preload = 0;
  int preloaded = 0;
  PrefillStats pf;
  bool cancelled = false;
};

class PrefixCache {
 public:
  PrefixCache(LM& m, const SessionOptions& o);
  ~PrefixCache();
  // Bring the live state to ids (all of it) and put the logits after the last id in `logits`. marks: positions (1..n)
  // where a message ends (checkpoint candidates; the first mark >= disk_min is the disk prefix).
  // want_logits false (state only: an idle-time sync of a prefix the next request extends): a checkpoint at the end of
  // ids may be restored whole (no position recomputed for its logits); live_logits are then not valid.
  SyncResult sync(const std::vector<int>& ids, const std::vector<int>& marks, float* logits, bool want_logits = true);
  // After sync: one generated token at a time (the live sequence grows with it).
  void step(int id, float* logits);
  // Draft and verify on the live sequence: ids[0..n) in one pass (logits_at(i) after position i); commit(a) keeps the first
  // a, and they join the live sequence. logits(): after the last live position.
  bool verify(const int* ids, int n);
  const float* logits_at(int i) const { return m_.logits_at(i); }
  void commit(int a);
  const float* logits() const { return live_logits_.data(); }
  // Write the disk copy of the live sequence's prefix [0, p) if p is a checkpointed position (or the live end, which is
  // checkpointed first) and no file holds it. A file for the first marked prefix (the system boundary) is pinned: the
  // disk LRU keeps it while pinned files stay under half of disk_gb.
  void persist(int p);
  const std::vector<int>& live() const { return live_; }
  std::string stats_json() const;
  // Idle-time prefill of a predicted prefix of the next prompt (src/serving/ahead.cpp): brings the live state to `ids`
  // with the same sync() a request uses, without losing anything of the live conversation: if ids leave the live
  // sequence, the live sequence is parked whole first (its KV, state, logits and the checkpoints past the shared prefix),
  // and checkpoints on the shared prefix are protected from eviction by the new ones. A later request that reaches the
  // end of ids drops that parked copy; one that returns to the old tail resumes from it. Cancellable like sync().
  SyncResult ahead(const std::vector<int>& ids, const std::vector<int>& marks, bool park_old = true);
  int ahead_len() const { return (int)ahead_ids_.size(); }

 private:
  struct Ckpt;
  struct Parked;
  LM& m_;
  SessionOptions o_;
  std::vector<int> live_;
  std::vector<std::unique_ptr<Ckpt>> ck_;
  std::vector<std::unique_ptr<Parked>> parked_;
  std::vector<float> live_logits_;  // logits after the last live position (a repeated prompt needs no compute)
  std::vector<int> ver_ids_;        // the last verify's positions
  bool live_logits_ok_ = false;
  int live_k_ = 0;  // the head k live_logits_ came from (LM::set_head_k; 0 = every logit)
  uint64_t clock_ = 0;
  // counters
  uint64_t n_req_ = 0, n_reused_ = 0, n_computed_ = 0, n_parks_ = 0, n_unparks_ = 0, n_disk_loads_ = 0, n_disk_writes_ = 0;
  void checkpoint(int p, bool pinned);
  void add_checkpoint(std::unique_ptr<Ckpt> c);
  std::string disk_path(const std::vector<int>& ids, int p, bool pin = false) const;
  std::string disk_existing(const std::vector<int>& ids, int p) const;  // either name, "" if none
  void begin_disk(const std::vector<int>& ids, int p, int start);
  void enforce_disk_cap(const std::string& keep);
  // the disk file being written by the current prefill (dense checkpoints of the system prefix)
 public:
  struct PendHdr {
    int n = 0, nck = 0, nexp = 0;
    uint64_t fb = 0, pb = 0;
    std::vector<int> ck, toks;
    long blobs_off() const { return 4 + 4 + 4 + 8 + 8 + 4 + 4L * nck + 4 + 4L * n; }
    long pos_off() const { return blobs_off() + (long)nck * (long)fb; }
    long exp_off() const { return pos_off() + (long)n * (long)pb; }
  };

 private:
  int pend_fd_ = -1;
  bool pend_ok_ = true;
  std::string pend_path_, pend_tmp_;
  std::vector<int> pend_ids_;
  PendHdr pend_;
 public:
  using DiskHdrT = PendHdr;
 private:
  void trim_checkpoints();
  void drop_checkpoints_after(int p);
  // ahead(): checkpoints at positions <= protect_upto_ are not evicted; no_park_ keeps sync() from parking the live
  // sequence a second time; ahead_ids_ is the last prediction (its parked old tail is dropped once a request reaches it)
  int protect_upto_ = -1;
  bool no_park_ = false;
  std::vector<int> ahead_ids_;
  uint64_t n_ahead_ = 0, n_ahead_pos_ = 0, n_ahead_hits_ = 0, n_ahead_parks_ = 0;
  void ahead_seen(const std::vector<int>& ids);
  void park_live(int keep_upto);
  bool load_disk(const std::vector<int>& ids, int& got, double& ms_preload, int& npre, int cut = 1);
  int best_disk(const std::vector<int>& ids, std::string& path, int cut = 1) const;
};

// The drafter's target over a prefix cache's live sequence (dmserve's worker and the C API draft through it)
struct PrefixTarget : DraftTarget {
  PrefixCache& pc;
  LM& lm;
  PrefixTarget(PrefixCache& p, LM& l) : pc(p), lm(l) {}
  int pos() const override { return lm.pos(); }
  void step(int id) override { pc.step(id, nullptr); }
  const float* logits() const override { return pc.logits(); }
  bool verify(const int* ids, int n) override { return pc.verify(ids, n); }
  const float* logits_at(int i) const override { return pc.logits_at(i); }
  void commit(int a) override { pc.commit(a); }
};

}  // namespace dm
