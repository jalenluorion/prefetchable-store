// The narrow interface between a model and the serving layer (prefix cache, batched-prefill driver, server worker).
//
// Nothing above this header knows which model it runs: the server front end (python/dmserve.py) owns the tokenizer and
// the chat template of the model folder and talks token ids; the worker (tools/dmserve.cpp) owns one LM and one
// PrefixCache (include/dmeng/session.h). A model implements LM once (src/q36_lm.cpp for Qwen3.6-35B-A3B).
//
// State model. A model's state at position p is split in two:
//   - the positional part (attention KV): one entry per position, append-only, so the entries of positions < p stay
//     valid as long as the live token sequence keeps the same prefix. Saved / restored by position range.
//   - the fixed part (recurrent and conv states, anything whose size does not grow with p): saved whole, as computed
//     (fp32 recurrent states stay fp32), so a restore is bit-exact.
// A checkpoint at p on the live sequence is therefore just the fixed part (tens of MB); a sequence parked away from
// the live one (or written to disk) is the fixed part plus its positional range.
//
// Exactness contract: prefill(ids, n) leaves the same state bits and the same logits as n calls of step(); every
// restore reproduces the saved bits. The serving layer never changes what the model computes.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dm {

struct LMInfo {
  std::string name;
  int vocab = 0;
  int max_ctx = 0;                 // positions the positional state can hold
  size_t fixed_bytes = 0;          // size of the fixed part of the state
  size_t pos_bytes = 0;            // positional-state bytes per position (q36: 20,480 = fp16 K and V of 10 layers)
  int n_layers = 0, n_experts = 0; // for expert-set bookkeeping (0 for a dense model)
};

// Progress of a prefill: positions done so far (a layer-major model reports fractions of a span as the layers pass,
// so it moves at least once per layer), positions this call computes, MB read so far. Called on the computing thread.
using PrefillProgress = std::function<void(int done, int total, double mb_read)>;

struct PrefillStats {              // what one prefill call did (for the server's per-request log)
  int positions = 0, chunks = 0;
  double ms = 0, ms_wait = 0;      // wall time, of which waiting on expert reads
  double mb_read = 0;
  std::string json;                // model-specific detail
  PrefillProgress progress;        // input: called as the prefill runs (empty: none)
};

class LM {
 public:
  virtual ~LM() {}
  virtual const LMInfo& info() const = 0;
  virtual int pos() const = 0;  // positions in the live state
  virtual void reset() = 0;     // empty state, position 0

  // Extend the live state by ids[0..n). If logits is non-null it receives the logits after the last position (vocab
  // floats). The model chooses chunking and expert streaming. capture_at: positions (relative to the current pos, in
  // (0, n]) whose fixed state the caller wants (checkpoints) without splitting the run: the model hands it over as slices
  // on_state(absolute position, byte offset in the fixed blob, data, bytes) while it computes (a layer-major model emits a
  // layer's slice as that layer passes the position), so the caller can assemble the blob in memory or write it to disk.
  // Every byte of the blob of each captured position is delivered exactly once before prefill returns. Returns false if
  // cancel() was called (then pos() says where the state is; it is consistent).
  using StateSink = std::function<void(int pos, size_t off, const void* data, size_t bytes)>;
  virtual bool prefill(const int* ids, int n, float* logits, const std::vector<int>& capture_at, const StateSink& on_state,
                       PrefillStats* st) = 0;
  // One decode step: append id, logits after it.
  virtual void step(int id, float* logits) = 0;
  // What step() must deliver from now on: 0 = every logit; k in [1, 64] = only the exact top k (their rows, order and
  // values; the other rows may hold -inf), which a model with a screened head computes without the full head (k = 1:
  // the argmax). Returns the k in force (0 when the model has no screened head or k is out of range).
  virtual int set_head_k(int k) { return 0; }
  virtual int head_k() const { return 0; }
  // Multi-position verify (draft and verify, include/dmeng/draft.h): ids[0..n) in one pass from the live state, the logits
  // after each position at logits_at(i); commit(a) keeps the first a (the state then equals a single steps'). False: the
  // model has none.
  virtual bool verify(const int* ids, int n) { return false; }
  virtual const float* logits_at(int i) const { return nullptr; }
  virtual void commit(int a) {}
  // Stop a running prefill at the next safe point (any thread): within one layer for a layer-major model, which then
  // puts the state back at the start of the span it was in; prefill returns false and pos() says where the state is.
  virtual void cancel() {}
  PrefillProgress on_progress;     // if set, every prefill reports to it (with the MB read filled in)

  // State. save_fixed / load_fixed: the fixed part at the live position (load_fixed sets the live position to p; the
  // caller guarantees the positional entries < p belong to the same tokens). save_pos / load_pos: positional entries
  // [p0, p1) as (p1 - p0) consecutive per-position records of pos_bytes (position-major, so a prefix of a saved buffer
  // is a contiguous slice; the layout inside a record is the model's).
  virtual void save_fixed(void* dst) const = 0;
  virtual void load_fixed(const void* src, int p) = 0;
  virtual void save_pos(int p0, int p1, void* dst) const = 0;
  virtual void load_pos(int p0, int p1, const void* src) = 0;

  // Expert store. held_experts: the (layer * n_experts + expert) records in memory now, most valuable first.
  // preload_experts: read these records into the cache now (before the first token), waiting until they have landed.
  virtual std::vector<int32_t> held_experts() const { return {}; }
  virtual void preload_experts(const std::vector<int32_t>& ids) {}
  virtual std::string stats_json() { return "{}"; }  // what ran since reset_stats() (decode steps, prefill)
  virtual void reset_stats() {}
  // where the process's memory is now (MB): footprint_mb, resident_mb, expert_slab_mb, kv_reserved_mb,
  // kv_touched_mb_at_pos, recurrent_mb, prefill_scratch_mb, prefill_xqb_mb
  virtual std::string mem_json() const { return "{}"; }
};

}  // namespace dm
