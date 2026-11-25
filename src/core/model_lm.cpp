#include "dmeng/model_lm.h"

#include <mach/mach.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace dm {

// The default run of positions: single steps, the whole fixed blob handed over at each captured position.
bool Model::prefill(State& s, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture,
                    const LM::StateSink& sink, PrefillStats* st, const std::atomic<bool>* cancel) {
  std::vector<int> caps(capture);
  std::sort(caps.begin(), caps.end());
  std::vector<uint8_t> blob;
  for (int i = 0; i < n; i++) {
    if (cancel && cancel->load()) return false;
    step(s, ids[i], i + 1 == n && want_logits ? Want::LAST : Want::NONE);
    if (st && st->progress) st->progress(i + 1, n, 0.0);
    if (sink && std::binary_search(caps.begin(), caps.end(), i + 1)) {
      blob.resize(fixed_bytes(s));
      save_fixed(s, blob.data());
      sink(s.pos, 0, blob.data(), blob.size());
    }
  }
  if (st) {
    st->positions = n;
    st->chunks = n;
  }
  return true;
}

ModelLM::ModelLM(Model& m, Core& c, State& s, int max_ctx) : m_(m), c_(c), s_(s) {
  const ModelInfo& mi = m.info();
  inf_.name = mi.family;
  inf_.vocab = mi.vocab_out;
  inf_.max_ctx = max_ctx;
  inf_.fixed_bytes = m.fixed_bytes(s);
  inf_.pos_bytes = m.pos_bytes();
  inf_.n_layers = mi.layers;
  inf_.n_experts = mi.experts;
}

bool ModelLM::prefill(const int* ids, int n, float* logits, const std::vector<int>& capture_at, const StateSink& on_state,
                      PrefillStats* st) {
  cancel_.store(false);
  auto t0 = clk::now();
  const double w0 = c_.t.wait;
  const uint64_t b0 = c_.cache.st.bytes_urgent + c_.cache.st.bytes_pf;
  std::vector<int> caps;
  for (int p : capture_at)
    if (p > 0 && p <= n) caps.push_back(p);
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  PrefillStats local;
  PrefillStats* ps = st ? st : &local;
  if (on_progress && !ps->progress)
    ps->progress = [this, b0](int done, int total, double) {
      on_progress(done, total, (c_.cache.st.bytes_urgent + c_.cache.st.bytes_pf - b0) / 1e6);
    };
  const bool ok = m_.prefill(s_, (const int32_t*)ids, n, logits != nullptr, caps, on_state, ps, &cancel_);
  if (logits && ok) std::memcpy(logits, m_.logits().data(), sizeof(float) * inf_.vocab);
  if (st) {
    if (st->ms == 0) st->ms = ms_since(t0);
    if (st->ms_wait == 0) st->ms_wait = c_.t.wait - w0;
    if (st->mb_read == 0) st->mb_read = (c_.cache.st.bytes_urgent + c_.cache.st.bytes_pf - b0) / MB;
  }
  return ok;
}

void ModelLM::step(int id, float* logits) {
  dec_steps_++;
  m_.step(s_, id, !logits ? Want::NONE : head_k_ == 1 ? Want::GREEDY : head_k_ > 1 ? Want::TOPK : Want::LAST);
  if (logits) std::memcpy(logits, m_.logits().data(), sizeof(float) * inf_.vocab);
}

bool ModelLM::enable_screened_head(std::string& err) {
  if (screened_) return true;
  // a family without a top-k screened head refuses head_topk (asked first: its greedy-only fast head is not built)
  if (!m_.set_option("head_topk", "1", err) || !m_.set_option("fast_head", "1", err)) return false;
  screened_ = true;
  return true;
}

int ModelLM::set_head_k(int k) {
  if (!screened_ || k < 1 || k > 64) return head_k_ = 0;
  if (k > 1) {
    std::string e;
    if (!m_.set_option("head_topk", std::to_string(k), e)) return head_k_ = 0;
  }
  return head_k_ = k;
}

bool ModelLM::verify(const int* ids, int n) {  // each position's head as a step's (set_head_k)
  return m_.verify(s_, (const int32_t*)ids, n, head_k_ == 1 ? Want::GREEDY : head_k_ > 1 ? Want::TOPK : Want::LAST);
}
void ModelLM::commit(int a) {
  dec_steps_ += a;
  m_.commit(s_, a);
}

std::vector<int32_t> ModelLM::held_experts() const {
  std::vector<std::pair<double, int32_t>> v;
  const Cache& C = c_.cache;
  for (int p = 0; p < C.npools; p++)
    for (int i = 0; i < C.P[p].n; i++) {
      const Slot& x = C.P[p].s[i];
      if (x.layer >= 0 && C.P[p].pending[i].load(std::memory_order_acquire) == 0) v.push_back({x.key, x.layer * C.NE + x.e});
    }
  std::sort(v.begin(), v.end(), [](const std::pair<double, int32_t>& a, const std::pair<double, int32_t>& b) { return a.first > b.first; });
  std::vector<int32_t> out;
  for (auto& q : v) out.push_back(q.second);
  return out;
}

void ModelLM::preload_experts(const std::vector<int32_t>& ids) {
  Cache& C = c_.cache;
  std::vector<std::pair<int, int>> got;  // (layer, slot), released once everything has landed
  int room[Cache::MAXP];
  for (int p = 0; p < C.npools; p++) room[p] = C.P[p].n - 2 * C.K - 2;
  for (int32_t id : ids) {
    const int L = id / C.NE, e = id % C.NE;
    if (L < 0 || L >= C.NL || C.slot_of(L, e) >= 0 || room[C.layer_pool[L]] <= 0) continue;
    const int x = C.victim(L, 0);
    if (x < 0) continue;
    C.issue(L, x, e, false);
    C.slot(L, x).pins = 1;
    // a preloaded record gets one use at the clock now, so decode's first tokens do not evict it before they look
    C.use(L, x, c_.tok_clock, c_.now(0));
    got.push_back({L, x});
    room[C.layer_pool[L]]--;
  }
  for (auto& ls : got) {
    C.wait(ls.first, ls.second);
    C.unpin(ls.first, ls.second);
  }
}

void ModelLM::reset_stats() {
  c_.reset_stats();
  m_.reset_stats();
  dec_steps_ = 0;
}

std::string ModelLM::stats_json() {
  const CacheStats& S = c_.cache.st;
  const Times& T = c_.t;
  const double n = std::max<long>(1, dec_steps_);
  const uint64_t picks = S.n_hit + S.n_late + S.n_miss;
  std::ostringstream o;
  o.setf(std::ios::fixed);
  o.precision(2);
  o << "{\"steps\":" << dec_steps_ << ",\"ms_per_step\":{" << T.stages_json(n)
    << "},\"hit\":" << (double)S.n_hit / std::max<uint64_t>(1, picks) << ",\"late\":" << (double)S.n_late / std::max<uint64_t>(1, picks)
    << ",\"mb_read_per_step\":" << (S.bytes_urgent + S.bytes_pf) / MB / n << ",\"routers\":" << (c_.hook ? "true" : "false")
    << ",\"pf_issued_per_step\":" << S.n_pf_issued / n << ",\"model\":" << m_.stats_json((int)dec_steps_) << "}";
  return o.str();
}

std::string ModelLM::mem_json() const {
  auto mb = [](size_t b) { return b / MB; };  // MB = 10^6 bytes, as every counter
  const ModelMem mm = m_.mem(s_);
  const Cache& C = c_.cache;
  size_t slab = C.core_slab_bytes;  // Mem::FILE: no slab (the records are the layer files' pages), only the locked core
  if (C.mem != Mem::FILE)
    for (int p = 0; p < C.npools; p++) slab += C.P[p].slab_bytes;
  task_vm_info_data_t vi;
  mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
  const double foot = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &cnt) == KERN_SUCCESS ? vi.phys_footprint / MB : -1;
  char b[768];
  snprintf(b, sizeof b,
           "{\"footprint_mb\":%.0f,\"resident_mb\":%.0f,\"expert_slab_mb\":%.0f,\"kv_reserved_mb\":%.0f,\"kv_touched_mb_at_pos\":%.0f,\"recurrent_mb\":%.0f,"
           "\"prefill_scratch_mb\":%.0f,\"prefill_xqb_mb\":%.0f}",
           foot, mb(mm.resident), mb(slab), mb(mm.kv_reserved), mb((size_t)s_.pos * inf_.pos_bytes), mb(mm.recurrent), mb(mm.prefill_scratch),
           mb(mm.prefill_xqb));
  return b;
}

}  // namespace dm
