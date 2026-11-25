// The batched prompt path's shared parts, the same for every family: its options (the prefill_* names of
// Model::set_option, and the values each manifest declares under "prefill"), the GPU share's setup, and the driver that
// cuts a prompt into spans, delivers checkpoints, honours a cancel and counts. A family supplies only its span pass (a
// layer-major or application-major forward over one span: what is architectural).
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "dmeng/lm.h"

namespace dm {

struct GpuGemm;
struct GpuHalf;
class Core;

struct PrefillOpts {
  // declared per model (manifest "prefill"; python/convert/<family>/moved-store.json): positions per sub-chunk (the
  // scratch bound; 1 = single steps), per layer-major span (a layer's experts read once per span), the span size from
  // which the next layer is named whole, sub-chunks of at least gpu_min positions using the GPU's share
  int chunk = 512, span = 8192, whole_from = 0, gpu_min = 256;
  // early names in a span: 0 none, 1 every expert of the next layer, 2 the next layer's released router on this layer's
  // MoE inputs, 3 automatic: 1 for spans of several sub-chunks from whole_from positions, else 2 if short_router (the
  // 35B), else none
  int stream = 3;
  bool short_router = false;
  // the last span: a rest under max(64, join x span) positions joins the span before (no short span reading every layer's
  // experts again for a few positions)
  double join = 0.25;
  // the experts of a span (Core::moe_chunk): in waves as they land (a wave given the GPU only from gpu_min_wave rows),
  // packed kernels from pack_min users (-1: the core's rule), down projections in dn_rows-row jobs
  bool waves = false;
  int gpu_min_wave = 128, pack_min = -1, dn_rows = 32;
  // the GPU's share of the projections' / experts' rows (exact; 0 = CPU only), its kernel (1 matrix units, 0 integer)
  double gpu_frac = 0, gpu_experts = 0;
  int gpu_kernel = 1;
  bool release = false;  // give the scratch and the GPU's buffers back after each prefill
  // the fast prompt path (fast mode only; exact mode never takes it): the prompt's 8- and 4-bit matrix products and its
  // experts on the GPU in half precision (gpu_half.h), the experts of a layer computed over MoE blocks of up to
  // moe_block positions (0: as the scratch bound allows). Not exact: decode after it is checked against exact prefill's.
  bool fast = false;
  int moe_block = 0;
  // the scratch bound of a prefill in MB (the span's streams first, then the sub-chunk and MoE-block buffers); 0: the
  // family's own default. A longer span reads each layer's experts once for more positions.
  double scratch_mb = 0;
  // fast mode: a prompt of at most drop_upto positions keeps the decode's leave-out rule (approx:drop) in its batched pass
  // (each position's light missing picks left out, as a decode step leaves them; its short-span names are only the picks
  // the rule would read), as the old runner ran its short prompts through its approximate decode; longer prompts, and
  // every prompt in exact mode, run lossless. 0: never (every family's default)
  int drop_upto = 0;
  // the early names of a span of n positions (multi: several sub-chunks): 0 none, 1 whole next layer, 2 router picks
  int names_for(int n, bool multi) const {
    if (stream != 3) return stream;
    return multi && n >= whole_from ? 1 : short_router ? 2 : 0;
  }
};

// The GPU's share of a prefill (Metal builds): created at the first non-zero share, the resident weights wrapped where
// they are, the expert slab(s) too once the experts get a share (FILE records stay on the CPU). Kept across prefills.
struct GpuShare {
  std::unique_ptr<GpuGemm> g;
  bool slab_wrapped = false;
  std::unique_ptr<GpuHalf> h;  // the fast prompt path's half-precision products (PrefillOpts::fast), the resident weights and
                               // the expert slab(s) wrapped
  GpuHalf* half() const { return h.get(); }
  GpuShare();
  ~GpuShare();
  GpuGemm* get() const { return g.get(); }
  explicit operator bool() const { return (bool)g; }
  GpuGemm* operator->() const { return g.get(); }
};

// One prefill_* option (or run_ahead's family-independent ones) into o; the GPU shares also set up `gpu` (res_base /
// res_size: the resident weights' region). false + err on a bad value; handled = false if the name is not a prefill option.
bool prefill_option(const std::string& name, const std::string& value, PrefillOpts& o, GpuShare& gpu, const void* res_base, size_t res_size,
                    Core* core, bool& handled, std::string& err);

// A span's MoE settings on the core (Core::chunk_waves, chunk_wave_gpu_min, chunk_pack_affine / _shared,
// chunk_dn_rows) for the life of the object, put back after: every family's span pass holds one around its layers
struct MoePass {
  Core& c;
  MoePass(Core& core, const PrefillOpts& o);
  ~MoePass();
};

// The values a manifest declares under "prefill" (LoadOpts::prefill: key -> value, e.g. chunk -> 512) into o, as the
// prefill_<key> options; false + err on an unknown key or a bad value
bool prefill_declared(const std::vector<std::pair<std::string, std::string>>& kv, PrefillOpts& o, std::string& err);

// The driver: the prompt in spans (o.span, o.join), each run by span(done, m, last, cap_rel) where cap_rel are the
// checkpoint positions inside the span relative to its start; a cancel checked before each span; fast mode's leave-out
// rule off while it runs (a span reads its positions' union once whatever the rule: the prompt's state is exact mode's).
// Fills st's common counters (positions, spans, wall ms, wait ms, MB read); the family adds its detail to st->json.
bool prefill_spans(Core& core, int n, const std::vector<int>& capture, const std::atomic<bool>* cancel, const PrefillOpts& o,
                   const std::function<bool(int done, int m, bool last, const std::vector<int>& cap_rel)>& span, PrefillStats* st, int* spans);

}  // namespace dm
