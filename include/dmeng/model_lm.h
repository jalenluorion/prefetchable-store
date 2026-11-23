// Any model of this engine as the serving layer's LM (include/dmeng/lm.h): one adapter for every family, so the prefix
// cache (session.h) and the server worker (tools/dmserve.cpp) work unchanged on all of them.
#pragma once
#include <atomic>
#include <string>

#include "dmeng/lm.h"
#include "dmeng/model.h"

namespace dm {

class ModelLM : public LM {
 public:
  ModelLM(Model& m, Core& c, State& s, int max_ctx);
  const LMInfo& info() const override { return inf_; }
  int pos() const override { return s_.pos; }
  void reset() override { m_.reset(s_); }
  bool prefill(const int* ids, int n, float* logits, const std::vector<int>& capture_at, const StateSink& on_state,
               PrefillStats* st) override;
  void step(int id, float* logits) override;
  int set_head_k(int k) override;
  int head_k() const override { return head_k_; }
  // build the model's screened head (option fast_head; once). False: the model has none (the full head stays).
  bool enable_screened_head(std::string& err);
  bool verify(const int* ids, int n) override;
  const float* logits_at(int i) const override { return m_.logits_at(i); }
  void commit(int a) override;
  void cancel() override { cancel_.store(true); }
  void save_fixed(void* dst) const override { m_.save_fixed(s_, dst); }
  void load_fixed(const void* src, int p) override { m_.load_fixed(s_, src, p); }
  void save_pos(int p0, int p1, void* dst) const override { m_.save_pos(s_, p0, p1, dst); }
  void load_pos(int p0, int p1, const void* src) override { m_.load_pos(s_, p0, p1, src); }
  std::vector<int32_t> held_experts() const override;  // (layer * experts + expert), most valuable first
  void preload_experts(const std::vector<int32_t>& ids) override;
  std::string stats_json() override;  // decode since reset_stats(): ms per step by stage, hit rates, MB read
  void reset_stats() override;
  std::string mem_json() const override;  // the process footprint, the model's parts (Model::mem) and the expert slab
  Model& model() { return m_; }
  Core& core() { return c_; }

 private:
  Model& m_;
  Core& c_;
  State& s_;
  LMInfo inf_;
  std::atomic<bool> cancel_{false};
  long dec_steps_ = 0;
  bool screened_ = false;
  int head_k_ = 0;
};

}  // namespace dm
