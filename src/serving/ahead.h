// Idle-time prefill for the serving worker (tools/dmserve.cpp): while no request runs, bring the prefix cache to a
// predicted prefix of the next prompt (the front end renders it with the request path's own chat template, e.g. the
// conversation as the template will show it once the next user message makes it drop the finished turn's reasoning).
//
// The work is PrefixCache::ahead(): the request path's sync() on the predicted ids, so the kernels, the checkpoints and
// the bits are the request path's. It runs in spans of `span` positions (a cancel keeps every finished span), and any
// request cancels it: the reader thread calls on_request(), which stops the prefill within one layer (the model's own
// cancel; a progress hook closes the race with the start of a prefill call).
#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include "dmeng/model_lm.h"
#include "dmeng/session.h"

namespace dm {

class AheadRunner {
 public:
  // span: positions per prefill span while running ahead; normal_span: the request path's span (restored after)
  AheadRunner(ModelLM& lm, PrefixCache& pc, int span, int normal_span, bool park_old)
      : lm_(lm), pc_(pc), span_(span), normal_span_(normal_span), park_old_(park_old) {}
  // reader thread: a predicted prompt (replaces any pending one)
  void submit(long id, std::vector<int> ids, std::vector<int> marks);
  // reader thread: a request arrived. Drops the pending prediction and stops a running one.
  void on_request();
  bool pending() const { return has_.load(); }
  // main thread, while no request waits: run the pending prediction; returns the worker's "A <id> {json}" line
  std::string run();

 private:
  ModelLM& lm_;
  PrefixCache& pc_;
  int span_, normal_span_;
  bool park_old_;
  std::mutex mu_;
  long id_ = 0;
  std::vector<int> ids_, marks_;
  std::atomic<bool> has_{false}, stop_{false}, running_{false};
  std::chrono::steady_clock::time_point t_stop_;
  uint64_t n_runs_ = 0;
  double ms_total_ = 0;
};

// dmserve --check-ahead A_IDS B_IDS: the state a request finds after an idle-time prefill (A served, a few tokens
// generated, ahead() of B's prefix cancelled once inside a span and resumed, then B synced) against B prefilled token by
// token from an empty state: logits, every fixed-state and KV byte, the greedy continuation. Also the request path
// without ahead(). Returns 0 only if both are bit-identical to the reference.
// A long request L, the identical request, then one that diverges from L at 60% of its length, through one prefix cache
// (each followed by a short greedy reply): logits, fixed state, KV and the reply bit for bit equal to a cold token-by-token
// computation of the same prompt, and the second and third restored from a checkpoint (reused > 0, at most the shared
// part). Prints one JSON line; 0 = passed.
int check_ckpt(ModelLM& lm, const SessionOptions& so, const std::vector<int>& L, int chunk, int span, int stream, int gen);
int check_ahead(ModelLM& lm, const SessionOptions& so, const std::vector<int>& A, const std::vector<int>& B, int ahead_len,
                int chunk, int span, int ahead_span, int stream, int gen);

}  // namespace dm
