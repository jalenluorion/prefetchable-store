// The batched prompt path's shared parts (include/dmeng/prefill.h).
#include "dmeng/prefill.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "dmeng/core.h"
#include "gpu_gemm.h"
#include "gpu_half.h"

namespace dm {

GpuShare::GpuShare() = default;
GpuShare::~GpuShare() = default;

static bool gpu_share(GpuShare& gs, double f, bool experts, const void* res_base, size_t res_size, Core* core, std::string& err) {
  if (!(f >= 0 && f < 1)) return (err = experts ? "prefill_gpu_experts must be in [0, 1)" : "prefill_gpu_frac must be in [0, 1)"), false;
  if (f <= 0) return true;
#if defined(DM_METAL)
  if (!res_base || !core) return (err = "prefill GPU split: load and bind the model first"), false;
  if (!gs.g) {
    std::unique_ptr<GpuGemm> g(new GpuGemm());
    if (!g->init()) return (err = "prefill GPU split: GPU (Metal) init failed"), false;
    if (!g->wrap(res_base, res_size)) return (err = "prefill GPU split: the resident weights could not be wrapped"), false;
    gs.g = std::move(g);
    fprintf(stderr, "dmeng: prefill GPU split on (resident weights wrapped, %.2f GB)\n", gs.g->wrapped_bytes() / 1e9);
  }
  if (experts && !gs.slab_wrapped) {  // the expert slab(s) where they are (locked or plain; mapped records stay on the CPU)
    bool any = false;
    for (int p = 0; p < core->cache.npools; p++)
      if (core->cache.P[p].slab) any = gs.g->wrap(core->cache.P[p].slab, core->cache.P[p].slab_bytes) || any;
    if (!any) return (err = "prefill GPU split: the expert slab could not be wrapped"), false;
    gs.slab_wrapped = true;
    fprintf(stderr, "dmeng: prefill GPU split: the experts too (%.2f GB wrapped)\n", gs.g->wrapped_bytes() / 1e9);
  }
  return true;
#else
  (void)experts, (void)res_base, (void)res_size, (void)core, (void)gs;
  return (err = "prefill GPU split: this build has no Metal (DM_METAL off)"), false;
#endif
}

bool prefill_option(const std::string& name, const std::string& value, PrefillOpts& o, GpuShare& gpu, const void* res_base, size_t res_size,
                    Core* core, bool& handled, std::string& err) {
  handled = true;
  auto i = [&]() { return std::stoi(value); };
  if (name == "prefill_chunk") return (o.chunk = std::max(1, i())), true;  // 1: single steps
  if (name == "prefill_span") return (o.span = std::max(1, i())), true;
  if (name == "prefill_whole_from") return (o.whole_from = std::max(0, i())), true;
  if (name == "prefill_stream") return (o.stream = std::max(0, std::min(3, i()))), true;
  if (name == "prefill_short_names") return (value == "router" || value == "none") && ((o.short_router = value == "router"), true);
  if (name == "prefill_join") return (o.join = std::stod(value)), true;
  if (name == "prefill_waves") return (o.waves = i() != 0), true;
  if (name == "prefill_gpu_min") return (o.gpu_min = std::max(1, i())), true;
  if (name == "prefill_gpu_min_wave") return (o.gpu_min_wave = std::max(0, i())), true;
  if (name == "prefill_pack_min") return (o.pack_min = std::max(-1, i())), true;
  if (name == "prefill_dn_rows") return (o.dn_rows = std::max(4, i() / 4 * 4)), true;
  if (name == "prefill_gpu_kernel") return (o.gpu_kernel = i() != 0), true;  // 1 matrix units, 0 integer
  if (name == "prefill_release") return (o.release = i() != 0), true;
  if (name == "prefill_moe_block") return (o.moe_block = std::max(0, i())), true;
  if (name == "prefill_scratch_mb") return (o.scratch_mb = std::max(0.0, std::stod(value))), true;
  if (name == "prefill_drop_upto") return (o.drop_upto = std::max(0, i())), true;
  if (name == "prefill_apart") return (o.apart = i() != 0), true;
  if (name == "prefill_lend") return (o.lend = i() != 0), true;
  if (name == "prefill_fast") {
    o.fast = i() != 0;
    if (!o.fast || !res_base || !core) return true;  // (a manifest's declaration: set up when the model is bound)
#if defined(DM_METAL)
    if (!gpu.h) {
      std::unique_ptr<GpuHalf> h(new GpuHalf());
      if (!h->init()) return (err = "prefill_fast: GPU (Metal) init failed"), false;
      if (!h->wrap(res_base, res_size)) return (err = "prefill_fast: the resident weights could not be wrapped"), false;
      bool any = false;
      for (int p = 0; p < core->cache.npools; p++)
        if (core->cache.P[p].slab) any = h->wrap(core->cache.P[p].slab, core->cache.P[p].slab_bytes) || any;
      if (!any) return (err = "prefill_fast: the expert slab could not be wrapped (plain or locked slots only)"), false;
      gpu.h = std::move(h);
      fprintf(stderr, "dmeng: fast prompt path on (half-precision products on the GPU, %.2f GB wrapped)\n", gpu.h->wrapped_bytes() / 1e9);
    }
    return true;
#else
    o.fast = false;
    return (err = "prefill_fast: this build has no Metal (DM_METAL off)"), false;
#endif
  }
  if (name == "prefill_gpu_frac" || name == "prefill_gpu_experts") {
    const bool ex = name == "prefill_gpu_experts";
    const double f = std::stod(value);
    if (!gpu_share(gpu, f, ex, res_base, res_size, core, err)) return false;
    (ex ? o.gpu_experts : o.gpu_frac) = f;
    return true;
  }
  handled = false;
  return false;
}

MoePass::MoePass(Core& core, const PrefillOpts& o) : c(core) {
  c.chunk_waves = o.waves;
  c.chunk_wave_gpu_min = o.gpu_min_wave;
  c.chunk_pack_affine = o.pack_min;
  c.chunk_pack_shared = o.pack_min;
  c.chunk_dn_rows = o.dn_rows;
}
MoePass::~MoePass() {
  c.chunk_waves = false;
  c.chunk_wave_gpu_min = 0;
  c.chunk_pack_affine = -1;
  c.chunk_pack_shared = 0;
  c.chunk_dn_rows = 32;
}

bool prefill_declared(const std::vector<std::pair<std::string, std::string>>& kv, PrefillOpts& o, std::string& err) {
  GpuShare none;
  for (auto& x : kv) {
    bool handled = false;
    std::string e;
    if (x.first == "gpu_frac" || x.first == "gpu_experts" || !prefill_option("prefill_" + x.first, x.second, o, none, nullptr, 0, nullptr, handled, e) || !handled)
      return (err = "manifest prefill." + x.first + ": " + (e.empty() ? "not a declared prefill value" : e)), false;
  }
  return true;
}

PromptPlan prompt_plan(const PrefillOpts& o, int n, const PromptBytes& b) {
  PromptPlan p;
  const double B = o.scratch_mb * 1e6;
  const bool bound = B > 0;
  n = std::max(1, n);
  const int sub_cap = bound && b.sub > 0 ? std::max(64, (int)(B / 4 / b.sub) / 64 * 64) : INT32_MAX;
  p.chunk = std::max(1, std::min(o.chunk, sub_cap));
  if (b.moe > 0) p.moe_block = std::min(n, o.moe_block > 0 ? o.moe_block : bound ? std::max(256, (int)(B / 4 / b.moe)) : n);
  const double rest = B - p.chunk * b.sub - p.moe_block * b.moe;
  const int span_cap = !bound || b.span <= 0 ? INT32_MAX : std::max(p.chunk, (int)std::min(2e9, rest / b.span));
  int span = std::min(std::max(p.chunk, o.span), span_cap);
  if (n > span) {
    int k = (n + span - 1) / span;
    // a short rest joins the span before (prefill_spans' rule) when the joined spans fit the bound
    const int tail = n - (k - 1) * span;
    if (k > 1 && tail < std::max(64, (int)(o.join * span)) && (n + k - 2) / (k - 1) <= span_cap) k--;
    span = (n + k - 1) / k;  // equal spans
  }
  p.span = span;
  if (o.lend && o.release) {
    const int sl = std::min(n, span);
    const double est = sl * b.span + std::min(p.chunk, sl) * b.sub + std::min(sl, p.moe_block) * b.moe + (sl > p.chunk ? b.snap : 0.0) - b.held;
    p.lend = (int64_t)std::max(0.0, est);
  }
  return p;
}

PromptLoan::PromptLoan(Core& core, int64_t b) : c(core), bytes(b) {
  if (bytes > 0) c.cache.lend(bytes);
}
PromptLoan::~PromptLoan() {
  if (bytes > 0) c.cache.lend(0);
}

bool prefill_spans(Core& core, int n, const std::vector<int>& capture, const std::atomic<bool>* cancel, const PrefillOpts& o,
                   const std::function<bool(int, int, bool, const std::vector<int>&)>& span, PrefillStats* st, int* spans_out) {
  struct NsOff {  // fast mode's rule off for the prompt (the batched pass is lossless; its state is exact mode's), except a
                  // short prompt under fast mode's leave-out rule (PrefillOpts::drop_upto)
    Approx& a;
    bool on;
    NsOff(Approx& x, bool keep) : a(x), on(x.on) { a.on = keep && x.on; }
    ~NsOff() { a.on = on; }
  } ns_off(core.ns, n <= o.drop_upto);
  struct InPrefill {  // the hooks' previous-token state follows the prompt (Core::chunk_prefill)
    Core& c;
    explicit InPrefill(Core& x) : c(x) { c.chunk_prefill = true; }
    ~InPrefill() { c.chunk_prefill = false; }
  } in_prefill(core);
  struct Apart {  // the prompt apart from the decode's cache (PrefillOpts::apart)
    Core& c;
    const bool on;
    const long t0;
    Apart(Core& x, bool want) : c(x), on(want && x.cache.hazard_evict && !x.cache.apart_on), t0(x.tok_clock) {
      if (on) c.cache.apart_begin();
    }
    ~Apart() {
      if (on) c.cache.apart_end(c.tok_clock - t0);
    }
  } apart(core, o.apart);
  std::vector<int> caps(capture);
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t b0 = core.cache.st.bytes_urgent + core.cache.st.bytes_pf;
  const double w0 = core.cache.st.wait_ms;
  const std::vector<uint32_t> r0 = st ? core.cache.n_reads : std::vector<uint32_t>();
  const int sp = std::max(1, o.span);
  int done = 0, spans = 0;
  bool ok = true;
  while (done < n) {
    if (cancel && cancel->load()) {
      ok = false;
      break;
    }
    int m = std::min(sp, n - done);
    if (sp > 1 && n - done - m > 0 && n - done - m < std::max(64, (int)(o.join * sp))) m = n - done;  // no short last span
    std::vector<int> cap_rel;
    for (int p : caps)
      if (p > done && p <= done + m) cap_rel.push_back(p - done);
    if (!span(done, m, done + m == n, cap_rel)) {
      ok = false;
      break;
    }
    done += m;
    spans++;
  }
  if (st) {
    st->positions = done;
    st->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    st->ms_wait = core.cache.st.wait_ms - w0;
    st->mb_read = (core.cache.st.bytes_urgent + core.cache.st.bytes_pf - b0) / MB;
    for (long& h : st->read_hist) h = 0;
    for (size_t i = 0; i < r0.size() && i < core.cache.n_reads.size(); i++) st->read_hist[std::min<uint32_t>(4, core.cache.n_reads[i] - r0[i])]++;
  }
  if (spans_out) *spans_out = spans;
  return ok;
}

}  // namespace dm
