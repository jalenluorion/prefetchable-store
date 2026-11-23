// The model interface. A model implementation holds only its architecture: its sequence mixers (attention, recurrence,
// loop), norms, residual structure and state layout. Everything else (expert store, MoE step, early address, modes,
// sampler, command line) is the core's. At each MoE application the model calls core.moe_step() with a MoeView of the
// block (core.h).
//
// Extension points (marked EXTENSION below) are where the work in progress on decoding and serving plugs in:
//   - a chunk of positions in one pass (batched prefill; multi-position verify of drafted tokens)
//   - the state saved / restored at a position (session and prefix cache), split as include/dmeng/lm.h describes:
//     a fixed part (recurrent and convolution states, sliding rings) and a positional part (attention KV per position)
//   - a draft head, a tokenizer and chat formatter, the OpenAI-compatible server
// The serving layer (prefix cache, server worker) talks to one model through the narrow LM interface of lm.h; one
// adapter (Model + Core + State) implements it for every family: prefill -> forward over a chunk with pauses at the
// cache's checkpoint positions, step -> step, save/load_fixed and save/load_pos -> the state methods below,
// held_experts / preload_experts -> the core's expert store (Cache::tab, Cache::name + wait).
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "dmeng/core.h"
#include "dmeng/lm.h"

namespace dm {

struct LoadOpts {
  std::string dir;       // the model folder (store/, resident.safetensors, ...)
  std::string resident;  // the engine's 8-bit resident file (DMENGRES1); empty: <dir>/dmeng_resident.bin
  Mem res_mem = Mem::MLOCK;
  // from the manifest (moved-store.json): the configuration the model reads, in place of the resident file's cfg lines
  std::vector<std::pair<std::string, std::string>> cfg;
};

// Where a model folder's configuration and engine files come from. With a moved-store.json: the family and the
// configuration from the manifest, the resident file from the cache folder (<dir>/.dmeng/, FORMAT.md section 6),
// derived on first use (a legacy dmeng_resident.bin whose sha256 is the manifest's is linked there instead of
// re-derived). Without one: the resident file's cfg lines, as before. `resident` (given) overrides the file.
// derived_dir (non-empty): the cache folder to use instead of <dir>/.dmeng.
// emb_file (non-empty): read the embedding rows from this safetensors file (relative to dir, or absolute) instead of
// the manifest's (resident.safetensors): it must hold the manifest's embedding tensor, same name, dtype and shape (e.g.
// embeddings.safetensors, the table alone: phone-store make_embeddings.py). Empty: the manifest's file, or, when that
// file is absent, <dir>/embeddings.safetensors if present (the phone layout, without the 4.9 GB bf16 resident).
bool resolve_model(const std::string& dir, const std::string& resident, Mem res_mem, LoadOpts& lo, std::string& family,
                   std::string& err, const std::string& derived_dir = "", const std::string& emb_file = "",
                   const std::string& profile = "");  // profile: its experts override (Manifest::apply_profile_experts)

// Per-sequence state; its layout is private to the model.
class State {
 public:
  virtual ~State() = default;
  int pos = 0;  // positions consumed
};

// Where a model's own memory is (bytes), for the server's per-request log; the expert store's is the core's (Cache).
struct ModelMem {
  size_t resident = 0;         // the resident weights file as loaded
  size_t kv_reserved = 0;      // positional state reserved for the context (its pages become resident as positions fill)
  size_t recurrent = 0;        // the fixed part of the state (recurrent and convolution states, sliding rings)
  size_t prefill_scratch = 0;  // batched-prefill scratch (allocated capacity)
  size_t prefill_xqb = 0;      // batched-prefill quantised activation batches (allocated capacity)
};

// What forward() returns: nothing, the last position's logits, or (EXTENSION, n > 1) every position's.
// GREEDY: only the argmax of the last position is needed (exact greedy head); TOPK: only the exact top-k logits are
// needed (a top-k sampler; k is the model's head_topk option), the other rows may hold -inf. A family without a screened
// head treats both as LAST.
enum class Want { NONE, LAST, GREEDY, TOPK };

class Model {
 public:
  virtual ~Model() = default;
  virtual const ModelInfo& info() const = 0;
  // read the resident weights and fill info(); the core is built from info() afterwards
  virtual bool load(const LoadOpts& o, std::string& err) = 0;
  // bind to the core (scratch sized for its precision), after Core::init
  virtual bool bind(Core& core, std::string& err) = 0;

  // ---- state
  virtual std::unique_ptr<State> new_state(int ctx_max, std::string& err) = 0;
  virtual void reset(State& s) = 0;
  // timing of long contexts only: the state as if `pos` positions had been consumed (zero KV entries before it)
  virtual bool seek(State&, int pos) { return false; }
  // EXTENSION (session / prefix cache, include/dmeng/lm.h): fixed part and positional part of the state
  virtual size_t fixed_bytes(const State&) const { return 0; }
  virtual size_t pos_bytes() const { return 0; }
  virtual bool save_fixed(const State&, void*) const { return false; }
  virtual bool load_fixed(State&, const void*, int pos) { return false; }
  virtual bool save_pos(const State&, int p0, int p1, void*) const { return false; }
  virtual bool load_pos(State&, int p0, int p1, const void*) { return false; }

  // ---- compute: positions [s.pos, s.pos + n) of ids, advancing s.pos by n; logits() then holds the last position's.
  // EXTENSION (batched prefill, multi-position verify): this per-position loop is replaced by one pass over the chunk
  // (every weight row read once for all positions, each routed expert read once for the chunk: core.moe_chunk), with
  // the exactness contract of lm.h: the same state bits and logits as n single steps.
  void forward(State& s, const int32_t* ids, int n, Want want) {
    for (int i = 0; i < n; i++) step(s, ids[i], i + 1 == n ? want : Want::NONE);
  }
  virtual void step(State& s, int32_t id, Want want) = 0;
  // A run of positions for the serving layer (lm.h's contract): the same state bits and logits as n single steps.
  // capture: positions relative to s.pos, in (0, n], whose fixed state goes to sink as (absolute position, byte offset
  // in the fixed blob, data, bytes), every byte once. The default runs single steps; a model with a batched path
  // (Qwen3.6: layer-major, each expert read once per span) overrides it. False if *cancel was set (s.pos says where).
  virtual bool prefill(State& s, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture,
                       const LM::StateSink& sink, PrefillStats* st, const std::atomic<bool>* cancel);
  // EXTENSION (multi-position verify of drafted tokens): n <= 8 given positions in one pass, each one's logits kept
  // (logits_at(i)); commit(a) keeps the first a and rolls the rest back, leaving the state bits n - a fewer single steps
  // would have left. Default: not supported (false).
  virtual bool verify(State&, const int32_t* ids, int n) { return false; }
  virtual void commit(State&, int a) {}
  virtual const float* logits_at(int i) const { return nullptr; }
  virtual const std::vector<float>& logits() const = 0;  // [vocab_out]
  virtual int greedy() const { return -1; }             // argmax of the last step under Want::GREEDY, -1 if not kept

  // ---- options and statistics
  virtual bool set_option(const std::string& name, const std::string& value, std::string& err) {
    err = "unknown model option " + name;
    return false;
  }
  virtual const float* router_emb() const = 0;          // this token's `emb` input of the early-address routers
  virtual std::string stats_json(int ntok) const { return "{}"; }
  virtual void reset_stats() {}
  // memory held now (default: the fixed part of the state only)
  virtual ModelMem mem(const State& s) const {
    ModelMem m;
    m.recurrent = fixed_bytes(s);
    return m;
  }
  virtual const std::vector<float>* taps() const { return nullptr; }  // draft-head features of the last step, if kept

  // Draft head (Inkling's MTP): set_option("draft_head", "1") loads it, after which the model keeps the head's features
  // of every committed position (taken in its own steps and chunks, no extra pass). draft_head(tok, ...) drafts the
  // positions after the last committed one, tok being the newest token (not yet fed): at most max tokens, stopping before
  // the first whose top-1 probability is below thr; prob[i] = draft i's probability. One sequence (the dmeng loop).
  // logits1 (optional, [vocab_out]): draft 1's logits (the port check)
  virtual int draft_head(int32_t tok, int max, float thr, int32_t* out, float* prob, float* logits1 = nullptr) { return 0; }
  // features from outside (the port check on recorded features): rows [n, 4, hidden] for positions p0 ..
  virtual bool draft_head_feed(const float* rows, int n, int p0) { return false; }
  virtual size_t draft_head_bytes() const { return 0; }  // its weights and state

  // For draft heads trained outside (LightHeadSource, draft.h): the post-norm final hidden state (the output head's input)
  // of the last step (i < 0) or of position i of the last verify; null when not kept. The raw embedding row of a token
  // [hidden]. The argmax of the model's own output head over hn (the screened greedy head when the model has one), and
  // the top-1 softmax probability into *conf.
  virtual const float* final_hidden(int i) const { return nullptr; }
  virtual bool embed_row(int32_t tok, float* out) const { return false; }
  virtual int head_argmax(const float* hn, float* conf) { return -1; }

  // EXTENSION: tokenizer and chat formatter (per family), checked against the release.
};

// the families this build knows
std::unique_ptr<Model> make_model(const std::string& family, std::string& err);
// the family of a model folder / resident file (the resident file's cfg `model`, else the layout of its cfg)
std::string detect_family(const std::string& resident, std::string& err);

}  // namespace dm
