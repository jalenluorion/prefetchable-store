// Qwen3.8-Flash-Next (qwen4_exp), batch 1, on the CPU: load, state, the decode step and run-ahead (qwen4_exp.h for the
// architecture and the precision). The batched prefill is prefill.cpp; both call the per-position functions here.
#include "qwen4_exp.h"
#include "../common/gated_attn.h"

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../../core/gpu_gemm.h"

namespace dm {
namespace qwen4 {

// ------------------------------------------------------------------------------------------------ configuration
bool FCfg::from(const ResFile& R, std::string& err) {
  auto need = [&](const char* k) {
    if (!R.has_cfg(k) && err.empty()) err = std::string("qwen4_exp: no cfg ") + k;
  };
  for (const char* k : {"hidden", "hc", "hc_low", "layers", "layer_types", "experts", "topk", "moe_inter", "shared_inter", "lin_nk", "lin_nv",
                        "lin_dk", "lin_dv", "conv_k", "n_heads", "n_kv", "head_dim", "rot_dim", "ix_heads", "ix_dim", "ix_budget", "ix_ratio",
                        "ple_layer", "ple_dim", "ple_conv_k", "ng_size", "ng_hpn", "ng_base", "ng_div", "ng_seed", "eos", "ng_files", "ng_parts",
                        "ng_rows", "ng_rows_per_part", "ng_dim", "ng_bpr", "eps", "vocab", "emb_file", "emb_off", "rec_bytes", "store_layer_file"})
    need(k);
  if (!err.empty()) return false;
  H = R.geti("hidden");
  HC = R.geti("hc");
  LOW = R.geti("hc_low");
  NL = R.geti("layers");
  types = R.gets("layer_types");
  NE = R.geti("experts");
  K = R.geti("topk");
  I = R.geti("moe_inter");
  SI = R.geti("shared_inter");
  norm_topk = R.has_cfg("norm_topk") ? R.geti("norm_topk") : 1;
  NKH = R.geti("lin_nk");
  NVH = R.geti("lin_nv");
  DK = R.geti("lin_dk");
  DV = R.geti("lin_dv");
  CK = R.geti("conv_k");
  sigmoid_gate = !R.has_cfg("gate") || R.gets("gate") == "sigmoid";
  NQ = R.geti("n_heads");
  NKV = R.geti("n_kv");
  HD = R.geti("head_dim");
  ROT = R.geti("rot_dim");
  theta = R.has_cfg("rope_theta") ? R.getf("rope_theta") : 1e7;
  IXH = R.geti("ix_heads");
  IXKV = R.has_cfg("ix_kv") ? R.geti("ix_kv") : 1;
  IXD = R.geti("ix_dim");
  budget = R.geti("ix_budget");
  ratio = R.geti("ix_ratio");
  ple = R.geti("ple_layer");
  ple_dim = R.geti("ple_dim");
  ple_ck = R.geti("ple_conv_k");
  ng_size = R.geti("ng_size");
  hpn = R.geti("ng_hpn");
  ng_base = R.geti("ng_base");
  ng_div = R.geti("ng_div");
  ng_seed = R.geti("ng_seed");
  eos = R.geti("eos");
  ng_files = R.gets("ng_files");
  ng_parts = R.geti("ng_parts");
  ng_rows = R.geti("ng_rows");
  ng_rpp = R.geti("ng_rows_per_part");
  ng_dim = R.geti("ng_dim");
  ng_bpr = R.geti("ng_bpr");
  eps = (float)R.getf("eps");
  V = R.geti("vocab");
  emb_file = R.gets("emb_file");
  emb_off = R.geti("emb_off");
  rec = R.geti("rec_bytes");
  layer_file = R.gets("store_layer_file");
  const char* kinds[3] = {"gate_proj", "up_proj", "down_proj"};
  for (int i = 0; i < 3; i++) {
    off[i] = R.geti(std::string("off_") + kinds[i]);
    soff[i] = R.geti(std::string("soff_") + kinds[i]);
    boff[i] = R.geti(std::string("boff_") + kinds[i]);
  }
  if ((int)types.size() != NL) return (err = "qwen4_exp: layer_types does not match layers"), false;
  if (IXKV != 1 || NQ % NKV || NVH % NKH || ROT % 2 || ROT > HD || ROT > IXD || HD % 8 || HD > 512 || IXD % 4 || DK % 4 || NKH > 64)
    return (err = "qwen4_exp: unsupported head configuration"), false;
  if (ng_size != 3) return (err = "qwen4_exp: n-gram size 3 only (the release's)"), false;
  if (ple < 0 || ple >= NL || ng_dim * heads() != ple_dim || ng_bpr != 2 + ng_dim) return (err = "qwen4_exp: inconsistent PLE / n-gram configuration"), false;
  if (H % 64 || LOW % 64 || ple_dim % 64 || I % 64 || SI % 64) return (err = "qwen4_exp: sizes must be multiples of 64"), false;
  if (H > 8192 || IXD > 1024 || HC > 64 || heads() > 64) return (err = "qwen4_exp: sizes above the scratch bounds (hidden 8192, indexer head 1024, 64 streams / n-gram heads)"), false;
  return true;
}

// ------------------------------------------------------------------------------------------------ the n-gram hash
static uint64_t splitmix64(uint64_t v) {
  v = v + 0x9E3779B97F4A7C15ull;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}
static bool is_prime(int64_t v) {
  if (v < 2) return false;
  if (v % 2 == 0) return v == 2;
  for (int64_t d = 3; d * d <= v; d += 2)
    if (v % d == 0) return false;
  return true;
}
static int64_t nth_prime_after(int64_t start, int count) {
  int64_t p = start;
  for (int c = 0; c < count; c++) {
    p++;
    while (!is_prime(p)) p++;
  }
  return p;
}

void NGramHash::build(int vocab, int ngram_size_, int hpn_, int64_t base, int divisor, int ple_index, uint64_t seed, int eos_) {
  ngram_size = ngram_size_;
  hpn = hpn_;
  heads = (ngram_size - 1) * hpn;
  eos = eos_;
  const uint64_t mmax = ((1ull << 63) - 1) / (uint64_t)(vocab > 1 ? vocab : 1);
  const uint64_t half = mmax / 2 > 0 ? mmax / 2 : 1;
  const uint64_t bseed = seed + 10007ull * (uint64_t)ple_index;
  mult.resize(ngram_size);
  for (int i = 0; i < ngram_size; i++) mult[i] = 2 * (splitmix64(bseed + 0x9E3779B97F4A7C15ull * (uint64_t)(i + 1)) % half) + 1;
  sizes.resize(heads);
  offs.resize(heads);
  int64_t total = 0;
  for (int h = 0; h < heads; h++) {
    sizes[h] = nth_prime_after(base - 1, ple_index * heads + h + 1);
    offs[h] = total;
    total += sizes[h];
  }
  padded = (total + divisor - 1) / divisor * divisor;
}

// history h(t - 2), h(t - 1), h(t); a shift by a gives id(t - a) when none of id(t - 1) .. id(t - a) is EOS, else EOS.
// The products stay below 2^63 (multipliers < 2^63 / vocab), so unsigned 64-bit arithmetic is the reference's int64.
void NGramHash::rows(const int32_t* ctx, int32_t cur, int64_t* out) const {
  const uint64_t s0 = (uint64_t)cur;
  const uint64_t s1 = (uint64_t)ctx[1];  // id(t - 1), or EOS when it is EOS: the same value
  const uint64_t s2 = (ctx[1] != eos && ctx[0] != eos) ? (uint64_t)ctx[0] : (uint64_t)eos;
  const uint64_t m2 = (s0 * mult[0]) ^ (s1 * mult[1]);
  const uint64_t m3 = m2 ^ (s2 * mult[2]);
  for (int h = 0; h < heads; h++) out[h] = (int64_t)((h < hpn ? m2 : m3) % (uint64_t)sizes[h]) + offs[h];
}

// ------------------------------------------------------------------------------------------------ load
Flash::Flash() = default;
Flash::~Flash() = default;

static HCw load_hc(const ResFile& R, const std::string& n, bool combine) {
  HCw w;
  w.norm = R.f32(n + "_n");
  w.down = R.weight(n + "_d");
  w.up = R.weight(n + "_u");
  if (combine) w.inj = R.weight(n + "_i");
  return w;
}

bool Flash::load(const LoadOpts& op, std::string& err) {
  dir = op.dir;
  res_mem = op.res_mem;
  const std::string res = op.resident.empty() ? op.dir + "/dmeng_resident.bin" : op.resident;
  if (!R.load(res, op.res_mem, err)) return false;
  for (auto& kv : op.cfg) R.cfg[kv.first] = kv.second;  // the manifest's configuration
  if (!c.from(R, err)) return false;
  lw.resize(c.NL);
  for (int L = 0; L < c.NL; L++) {
    Lw& w = lw[L];
    const std::string s = std::to_string(L) + ".";
    w.lin = c.types[L] == 'L';
    w.ahc = load_hc(R, s + "ahc", true);
    w.mhc = load_hc(R, s + "mhc", true);
    if (w.lin) {
      w.qkv = R.weight(s + "qkv");
      w.z = R.weight(s + "z");
      w.wa = R.weight(s + "a");
      w.wb = R.weight(s + "b");
      w.o = R.weight(s + "o");
      w.g = {R.f32(s + "conv"), R.f32(s + "A_log"), R.f32(s + "dt_bias"), R.f32(s + "gnorm")};
    } else {
      w.q = R.weight(s + "q");
      w.k = R.weight(s + "k");
      w.v = R.weight(s + "v");
      w.o = R.weight(s + "o");
      w.ix = R.weight(s + "ix");
      w.qn = R.f32(s + "qn");
      w.kn = R.f32(s + "kn");
      w.ixqn = R.f32(s + "ixqn");
      w.ixkn = R.f32(s + "ixkn");
    }
    w.router = R.weight(s + "router");
    w.sg = R.weight(s + "sg");
    w.su = R.weight(s + "su");
    w.sd = R.weight(s + "sd");
    w.shgate = R.weight(s + "shgate");
    if (L == c.ple) {
      w.ple = true;
      w.pk = R.weight(s + "pk");
      w.pv = R.weight(s + "pv");
      w.pnk = R.f32(s + "pnk");
      w.pnq = R.f32(s + "pnq");
      w.pnc = R.f32(s + "pnc");
      w.pconv = R.f32(s + "pconv");
      // the hashing constants the release stores must be the ones the model builds (as the reference runner checks)
      ng.build(c.V, c.ng_size, c.hpn, c.ng_base, c.ng_div, 0, (uint64_t)c.ng_seed, c.eos);
      const ResFile::T &tm = R.info(s + "ng_mult"), &ts = R.info(s + "ng_sizes"), &to = R.info(s + "ng_offs");
      bool same = tm.nbytes == ng.mult.size() * 8 && ts.nbytes == ng.sizes.size() * 8 && to.nbytes == ng.offs.size() * 8;
      if (same) {
        const int64_t *m = (const int64_t*)R.ptr(s + "ng_mult"), *z = (const int64_t*)R.ptr(s + "ng_sizes"), *o = (const int64_t*)R.ptr(s + "ng_offs");
        for (size_t i = 0; i < ng.mult.size(); i++) same = same && (uint64_t)m[i] == ng.mult[i];
        for (size_t i = 0; i < ng.sizes.size(); i++) same = same && z[i] == ng.sizes[i] && o[i] == ng.offs[i];
      }
      if (!same) return (err = "qwen4_exp: the release's n-gram hashing constants differ from the ones built from config.json"), false;
      if (ng.padded != c.ng_rows) return (err = "qwen4_exp: the n-gram store's rows do not match the model's table"), false;
    }
  }
  fin = load_hc(R, "fhc", false);
  head = R.weight("head");
  if (!rows.open(c.emb_file[0] == '/' ? c.emb_file : op.dir + "/" + c.emb_file, c.emb_off, c.H, err)) return false;
  {  // the n-gram parts: read where they are, as published
    std::vector<std::string> paths;
    std::vector<int64_t> pr;
    for (int p = 0; p < c.ng_parts; p++) {
      char b[512];
      snprintf(b, sizeof b, c.ng_files.c_str(), p);
      paths.push_back(op.dir + "/" + b);
      pr.push_back(p + 1 < c.ng_parts ? c.ng_rpp : c.ng_rows - (int64_t)(c.ng_parts - 1) * c.ng_rpp);
    }
    const char* e = getenv("DMENG_NGRAM_READERS");
    if (!ngs.open(paths, pr, c.ng_rpp, c.ng_bpr, e ? std::max(1, atoi(e)) : 4, err)) return false;
  }
  rope_inv.resize(c.ROT / 2);
  for (int i = 0; i < c.ROT / 2; i++) rope_inv[i] = (float)std::pow(c.theta, -2.0 * i / c.ROT);
  gd.NKH = c.NKH;
  gd.NVH = c.NVH;
  gd.DK = c.DK;
  gd.DV = c.DV;
  gd.CK = c.CK;
  gd.eps = c.eps;
  gd.sigmoid_gate = c.sigmoid_gate;
  if (!gd.ok(err)) return false;
  // what the core needs
  mi.family = "qwen4_exp";
  mi.hidden = c.H;
  mi.vocab = mi.vocab_out = c.V;
  mi.layers = mi.steps = c.NL;
  mi.step_layer.resize(c.NL);
  for (int L = 0; L < c.NL; L++) mi.step_layer[L] = L;
  mi.step_pass.assign(c.NL, -1);
  mi.experts = c.NE;
  mi.top_k = c.K;
  mi.shared_inter = c.SI;
  ExpertFormat f;
  f.fmt = Fmt::AFF4;
  f.rec = c.rec;
  f.H = c.H;
  f.I = c.I;
  f.fused = false;
  f.gu_c = c.off[0], f.gu_s = c.soff[0], f.gu_b = c.boff[0];
  f.up_c = c.off[1], f.up_s = c.soff[1], f.up_b = c.boff[1];
  f.dn_c = c.off[2], f.dn_s = c.soff[2], f.dn_b = c.boff[2];
  f.touched = c.rec;
  mi.formats = {f};
  mi.layer_fmt.assign(c.NL, 0);
  mi.pool_share = {1.0};
  mi.layer_file = c.layer_file;
  mi.has_ref_path = true;
  mi.has_verify = true;     // the multi-position verify pass (prefill.cpp: the span's layer-major pass with rollback)
  mi.chunk_aff4 = true;     // ... with the packed 4-bit kernels and the GPU's share of the experts
  views.resize(c.NL);
  for (int L = 0; L < c.NL; L++) {
    MoeView& v = views[L];
    v.rule = MoeView::SOFTMAX_TOPK_RENORM;
    v.renorm = c.norm_topk != 0;
    v.router = &lw[L].router;
    v.act = MoeView::SWIGLU;
    v.shared = true;
    v.sh_gate = &lw[L].sg;
    v.sh_up = &lw[L].su;
    v.sh_down = &lw[L].sd;
    v.sh_scalar_gate = &lw[L].shgate;
  }
  return prefill_declared(op.prefill, po, err);  // the prompt path's values the manifest declares
}

bool Flash::bind(Core& cr, std::string& err) {
  core = &cr;
  prec = cr.o.prec;
  if (const char* e = getenv("DMENG_AB")) ab_on = atoi(e);  // measurement: arm = token parity, per-arm step and mixer ms
  ref = cr.o.ref;
  cr.views.resize(c.NL);
  for (int L = 0; L < c.NL; L++) cr.views[L] = &views[L];
  T = &cr.t;
  const int N = c.N(), C = c.conv_dim();
  const int kmax = std::max({N, c.H, c.ple_dim, c.LOW, c.NVH * c.DV, c.NQ * c.HD});
  xin.init(kmax, prec);
  xlo.init(kmax, prec);
  xatt.init(std::max(c.NVH * c.DV, c.NQ * c.HD), prec);
  xe.init(c.ple_dim, prec);
  hyp.assign(N, 0.f);
  nrm.assign(N, 0.f);
  mix.assign(c.H, 0.f);
  blk.assign(c.H, 0.f);
  lo.assign(c.LOW, 0.f);
  up.assign(N, 0.f);
  emb.assign(c.H, 0.f);
  x.assign(c.H, 0.f);
  lgts.assign(c.V, 0.f);
  buf.assign(std::max((size_t)2 * c.NQ * c.HD + 2 * c.NKV * c.HD + (c.IXH + 1) * c.IXD + 2 * c.NQ * c.HD,
                      (size_t)2 * C + c.NVH * c.DV + 2 * c.NVH) + 64, 0.f);
  att.assign(std::max(c.NVH * c.DV, c.NQ * c.HD), 0.f);
  ple_e.assign(c.ple_dim, 0.f);
  ple_out.assign((size_t)4 * N + c.H, 0.f);
  // the fixed blob: each Gated DeltaNet layer's conv state then recurrent state in layer order, the PLE conv state, the
  // n-gram history
  fixed_off.assign(c.NL, 0);
  size_t off = 0;
  for (int L = 0; L < c.NL; L++)
    if (lw[L].lin) {
      fixed_off[L] = off;
      off += ((size_t)(c.CK - 1) * C + (size_t)c.NVH * c.DV * c.DK) * 4;
    }
  fixed_ple = off;
  fixed_ng = off + (size_t)c.pstate() * N * 4;
  (void)err;
  return true;
}

// ------------------------------------------------------------------------------------------------ state
std::unique_ptr<State> Flash::new_state(int ctx_max, std::string& err) {
  std::unique_ptr<FState> s(new FState());
  s->ctx = ctx_max;
  s->ls.resize(c.NL);
  const int C = c.conv_dim();
  for (int L = 0; L < c.NL; L++) {
    LState& st = s->ls[L];
    if (lw[L].lin) {
      st.conv.assign((size_t)(c.CK - 1) * C, 0.f);
      st.S.assign((size_t)c.NVH * c.DV * c.DK, 0.f);
      continue;
    }
    const size_t es = ref ? 4 : 2;  // the reference path keeps K, V and the raw indexer keys in fp32
    st.kv_bytes = (size_t)c.NKV * ctx_max * c.HD * es;
    st.ix_bytes = (size_t)ctx_max * c.IXD * es;
    st.ixp_bytes = ((size_t)ctx_max / c.ratio + 1) * c.IXD * 4;
    // plain anonymous memory: only the pages a context touches become resident
    void *k = alloc_region(st.kv_bytes, Mem::PLAIN, &err), *v = alloc_region(st.kv_bytes, Mem::PLAIN, &err),
         *ix = alloc_region(st.ix_bytes, Mem::PLAIN, &err);
    st.ixp = (float*)alloc_region(st.ixp_bytes, Mem::PLAIN, &err);
    if (!k || !v || !ix || !st.ixp) return nullptr;
    if (ref) {
      st.Kf = (float*)k, st.Vf = (float*)v, st.ixkf = (float*)ix;
    } else {
      st.K = (_Float16*)k, st.V = (_Float16*)v, st.ixk = (_Float16*)ix;
    }
  }
  s->pconv.assign((size_t)c.pstate() * c.N(), 0.f);
  s->ng_ctx[0] = s->ng_ctx[1] = c.eos;
  return s;
}

void Flash::reset(State& s0) {
  FState& s = (FState&)s0;
  for (auto& st : s.ls) {
    std::fill(st.conv.begin(), st.conv.end(), 0.f);
    std::fill(st.S.begin(), st.S.end(), 0.f);
  }
  std::fill(s.pconv.begin(), s.pconv.end(), 0.f);
  s.ng_ctx[0] = s.ng_ctx[1] = c.eos;  // the n-gram history starts with EOS
  s.pos = 0;
}

// The state for the serving layer (include/dmeng/lm.h): the fixed part = every Gated DeltaNet layer's conv and fp32
// recurrent state, the PLE's conv state and the n-gram history (two token ids); the positional part = per position, for
// every QSA layer, its K and V (every KV head) and its raw indexer key. The pooled block keys are not saved: load_pos
// recomputes them from the raw keys with the decode path's own arithmetic.
size_t Flash::fixed_bytes(const State&) const { return fixed_ng + 8; }
size_t Flash::pos_bytes() const {
  int nq = 0;
  for (int L = 0; L < c.NL; L++) nq += !lw[L].lin;
  return (size_t)nq * (2 * c.NKV * c.HD + c.IXD) * (ref ? 4 : 2);
}
bool Flash::save_fixed(const State& s0, void* dst) const {
  const FState& s = (const FState&)s0;
  uint8_t* d = (uint8_t*)dst;
  for (int L = 0; L < c.NL; L++)
    if (lw[L].lin) {
      const LState& st = s.ls[L];
      std::memcpy(d + fixed_off[L], st.conv.data(), st.conv.size() * 4);
      std::memcpy(d + fixed_off[L] + st.conv.size() * 4, st.S.data(), st.S.size() * 4);
    }
  std::memcpy(d + fixed_ple, s.pconv.data(), s.pconv.size() * 4);
  std::memcpy(d + fixed_ng, s.ng_ctx, 8);
  return true;
}
bool Flash::load_fixed(State& s0, const void* src, int p) {
  FState& s = (FState&)s0;
  const uint8_t* d = (const uint8_t*)src;
  for (int L = 0; L < c.NL; L++)
    if (lw[L].lin) {
      LState& st = s.ls[L];
      std::memcpy(st.conv.data(), d + fixed_off[L], st.conv.size() * 4);
      std::memcpy(st.S.data(), d + fixed_off[L] + st.conv.size() * 4, st.S.size() * 4);
    }
  std::memcpy(s.pconv.data(), d + fixed_ple, s.pconv.size() * 4);
  std::memcpy(s.ng_ctx, d + fixed_ng, 8);
  s.pos = p;
  return true;
}
bool Flash::save_pos(const State& s0, int p0, int p1, void* dst) const {
  const FState& s = (const FState&)s0;
  if (p0 < 0 || p1 > s.ctx || p0 > p1) return false;
  uint8_t* d = (uint8_t*)dst;
  const size_t es = ref ? 4 : 2, row = (size_t)c.HD * es, ixr = (size_t)c.IXD * es;
  for (int p = p0; p < p1; p++)
    for (int L = 0; L < c.NL; L++) {
      if (lw[L].lin) continue;
      const LState& st = s.ls[L];
      const uint8_t* K = ref ? (const uint8_t*)st.Kf : (const uint8_t*)st.K;
      const uint8_t* V = ref ? (const uint8_t*)st.Vf : (const uint8_t*)st.V;
      for (int j = 0; j < c.NKV; j++) {
        std::memcpy(d, K + ((size_t)j * s.ctx + p) * row, row);
        std::memcpy(d + row, V + ((size_t)j * s.ctx + p) * row, row);
        d += 2 * row;
      }
      std::memcpy(d, (ref ? (const uint8_t*)st.ixkf : (const uint8_t*)st.ixk) + (size_t)p * ixr, ixr);
      d += ixr;
    }
  return true;
}
// the pooled key of block b of layer L from its raw keys (the cache's values), normed and rotated at the block's first
// position
void Flash::pool_key(int L, int b) const {
  const LState& st = S->ls[L];
  float mean[1024];
  for (int i = 0; i < c.IXD; i++) {
    float s = 0;
    for (int r = 0; r < c.ratio; r++) {
      const size_t at = (size_t)(b * c.ratio + r) * c.IXD + i;
      s += st.ixkf ? st.ixkf[at] : (float)st.ixk[at];
    }
    mean[i] = s / (float)c.ratio;
  }
  float* out = st.ixp + (size_t)b * c.IXD;
  rmsnorm(mean, lw[L].ixkn, c.IXD, c.eps, out);
  rope(out, b * c.ratio);
}
bool Flash::load_pos(State& s0, int p0, int p1, const void* src) {
  FState& s = (FState&)s0;
  if (p0 < 0 || p1 > s.ctx || p0 > p1) return false;
  const uint8_t* d = (const uint8_t*)src;
  const size_t es = ref ? 4 : 2, row = (size_t)c.HD * es, ixr = (size_t)c.IXD * es;
  for (int p = p0; p < p1; p++)
    for (int L = 0; L < c.NL; L++) {
      if (lw[L].lin) continue;
      LState& st = s.ls[L];
      uint8_t* K = ref ? (uint8_t*)st.Kf : (uint8_t*)st.K;
      uint8_t* V = ref ? (uint8_t*)st.Vf : (uint8_t*)st.V;
      for (int j = 0; j < c.NKV; j++) {
        std::memcpy(K + ((size_t)j * s.ctx + p) * row, d, row);
        std::memcpy(V + ((size_t)j * s.ctx + p) * row, d + row, row);
        d += 2 * row;
      }
      std::memcpy((ref ? (uint8_t*)st.ixkf : (uint8_t*)st.ixk) + (size_t)p * ixr, d, ixr);
      d += ixr;
    }
  S = &s;  // the blocks these positions complete (the raw keys before p0 are the caller's: the same tokens)
  for (int L = 0; L < c.NL; L++)
    if (!lw[L].lin)
      for (int p = p0; p < p1; p++)
        if ((p + 1) % c.ratio == 0) pool_key(L, p / c.ratio);
  return true;
}

// ------------------------------------------------------------------------------------------------ per-position pieces
void Flash::rope(float* v, int pos) const {
  const int half = c.ROT / 2;
  for (int i = 0; i < half; i++) {
    const float ang = (float)pos * rope_inv[i];  // position (fp32) x inverse frequency (fp32)
    const float cs = std::cos(ang), sn = std::sin(ang);
    const float a = v[i], b = v[i + half];
    v[i] = a * cs - b * sn;
    v[i + half] = b * cs + a * sn;
  }
}

void Flash::ng_row(const uint8_t* rec, float* out) const {
  uint16_t sb;
  std::memcpy(&sb, rec, 2);
  _Float16 sh;
  std::memcpy(&sh, &sb, 2);
  const float s = (float)sh;
  for (int i = 0; i < c.ng_dim; i++) out[i] = (float)(int8_t)rec[2 + i] * s;
}

void Flash::hc_norm(const HCw& w, const float* hyper, float* normed) const {
  for (int s = 0; s < c.HC; s++) rmsnorm(hyper + (size_t)s * c.H, w.norm + (size_t)s * c.H, c.H, c.eps, normed + (size_t)s * c.H);
}
void Flash::hc_lo(float* l) const {
  for (int i = 0; i < c.LOW; i++) l[i] = silu(l[i] / (float)c.HC);
}
// mixed[i] = (sum over the streams of sigmoid(up) normed) / HC, the products added in stream order (explicit fused
// multiply-adds); inj[s] = 2 sigmoid(r_s / HC)
void Flash::hc_mix(const float* normed, const float* u, const float* inj_raw, float* mixed, float* inj_out) const {
  const int H = c.H;
  for (int i = 0; i < H; i++) {
    float acc = 0;
    for (int s = 0; s < c.HC; s++) acc = __builtin_fmaf(sigm(u[(size_t)s * H + i]), normed[(size_t)s * H + i], acc);
    mixed[i] = acc / (float)c.HC;
  }
  if (inj_raw)
    for (int s = 0; s < c.HC; s++) inj_out[s] = 2.f * sigm(inj_raw[s] / (float)c.HC);
}

// One position of the PLE after its products: key_s = norm(key_s), query_s = norm(hyper_s), gate_s = signed sqrt of
// key_s . query_s / sqrt(H), gated_s = sigmoid(gate_s) value, normed = the per-stream norm of gated (the conv's input)
void Flash::ple_gate(const Lw& w, const float* key, const float* value, const float* hyper, float* gated, float* normed) const {
  const int H = c.H;
  float kn[8192], qn[8192], gt[64];
  for (int s = 0; s < c.HC; s++) {
    rmsnorm(key + (size_t)s * H, w.pnk + (size_t)s * H, H, c.eps, kn);
    rmsnorm(hyper + (size_t)s * H, w.pnq + (size_t)s * H, H, c.eps, qn);
    double d = 0;
    for (int i = 0; i < H; i++) d += (double)kn[i] * qn[i];
    const float g = (float)d / std::sqrt((float)H);
    const float sg = g > 0.f ? 1.f : (g < 0.f ? -1.f : 0.f);
    gt[s] = sg * std::sqrt(std::max(std::fabs(g), 1e-6f));
  }
  for (int s = 0; s < c.HC; s++) {
    const float sg = sigm(gt[s]);
    for (int i = 0; i < H; i++) gated[(size_t)s * H + i] = sg * value[i];
    rmsnorm(gated + (size_t)s * H, w.pnc + (size_t)s * H, H, c.eps, normed + (size_t)s * H);
  }
}
// channel ch of the PLE's dilated depthwise convolution over [state, this position] (tap k reads position k * ngram_size
// of the window, the taps as one fused multiply-add chain), its silu; then (commit) the channel's state moves by one
float Flash::ple_conv(const Lw& w, float* st, const float* normed, int ch, bool commit) const {
  const int N = c.N(), PS = c.pstate(), dil = c.ng_size, CKp = c.ple_ck;
  const float* wc = w.pconv + (size_t)ch * CKp;
  float v = 0;
  for (int k = 0; k < CKp; k++) {
    const int j = k * dil;
    v = __builtin_fmaf(j < PS ? st[(size_t)j * N + ch] : normed[ch], wc[k], v);
  }
  if (commit) {
    for (int j = 0; j + 1 < PS; j++) st[(size_t)j * N + ch] = st[(size_t)(j + 1) * N + ch];
    st[(size_t)(PS - 1) * N + ch] = normed[ch];
  }
  return silu(v);
}

// QSA, one position after its projections (see the header)
void Flash::qsa_kv(int L, int pos, float* qp, const float* kp, const float* vp, float* ixq, float* q, float* gate) {
  const Lw& w = lw[L];
  LState& st = S->ls[L];
  const int HD = c.HD, ctx = S->ctx, IXD = c.IXD;
  for (int h = 0; h < c.IXH; h++) {  // the indexer's queries: normed, rotated
    float* qh = ixq + (size_t)h * IXD;
    rmsnorm(qh, w.ixqn, IXD, c.eps, qh);
    rope(qh, pos);
  }
  const float* raw = ixq + (size_t)c.IXH * IXD;  // the indexer's key head, kept raw
  if (st.ixkf) std::memcpy(st.ixkf + (size_t)pos * IXD, raw, sizeof(float) * IXD);
  else
    for (int i = 0; i < IXD; i++) st.ixk[(size_t)pos * IXD + i] = (_Float16)raw[i];
  for (int h = 0; h < c.NQ; h++) {
    rmsnorm(qp + (size_t)h * 2 * HD, w.qn, HD, c.eps, q + (size_t)h * HD);
    rope(q + (size_t)h * HD, pos);
    std::memcpy(gate + (size_t)h * HD, qp + (size_t)h * 2 * HD + HD, sizeof(float) * HD);
  }
  for (int j = 0; j < c.NKV; j++) {
    float k[1024];
    rmsnorm(kp + (size_t)j * HD, w.kn, HD, c.eps, k);
    rope(k, pos);
    const size_t at = ((size_t)j * ctx + pos) * HD;
    if (st.Kf) {
      std::memcpy(st.Kf + at, k, sizeof(float) * HD);
      std::memcpy(st.Vf + at, vp + (size_t)j * HD, sizeof(float) * HD);
    } else {
      for (int i = 0; i < HD; i++) {
        st.K[at + i] = (_Float16)k[i];
        st.V[at + i] = (_Float16)vp[(size_t)j * HD + i];
      }
    }
  }
}

// The keys position pos attends to: every visible one within the budget, else the indexer's top blocks (by score, ties
// to the earlier block) and the incomplete tail block, ascending.
void Flash::qsa_select(int L, int pos, const float* ixq, std::vector<int>& sel, bool par) const {
  const int visible = pos + 1;
  sel.clear();
  if (visible <= c.budget) {
    for (int j = 0; j < visible; j++) sel.push_back(j);
    return;
  }
  const LState& st = S->ls[L];
  const int complete = visible / c.ratio, IXD = c.IXD;
  std::vector<float> sc(complete);  // (not thread_local: the pool's threads write it)
  std::vector<int> ord;
  const float rs = std::sqrt((float)IXD);
  auto score = [&](int b) {
    const float* pb = st.ixp + (size_t)b * IXD;
    float s = 0;
    for (int h = 0; h < c.IXH; h++) {
      const float* qh = ixq + (size_t)h * IXD;
      float32x4_t a = vdupq_n_f32(0);
      for (int i = 0; i < IXD; i += 4) a = vfmaq_f32(a, vld1q_f32(qh + i), vld1q_f32(pb + i));
      s += std::max(vaddvq_f32(a), 0.f);
    }
    sc[b] = s / rs;
  };
  static const int par_min = getenv("DMENG_QSA_PAR_MIN") ? atoi(getenv("DMENG_QSA_PAR_MIN")) : 1024;  // test switch
  if (par && complete >= par_min) {
    const int per = 256;
    parallel_for(*core->pool, (complete + per - 1) / per, [&](int j) {
      for (int b = j * per; b < std::min(complete, (j + 1) * per); b++) score(b);
    });
  } else {
    for (int b = 0; b < complete; b++) score(b);
  }
  const int kb = std::min(c.budget / c.ratio, complete);
  ord.resize(complete);
  for (int b = 0; b < complete; b++) ord[b] = b;
  std::nth_element(ord.begin(), ord.begin() + (kb - 1), ord.end(), [&](int a, int b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); });
  std::sort(ord.begin(), ord.begin() + kb);
  for (int i = 0; i < kb; i++)
    for (int r = 0; r < c.ratio; r++) sel.push_back(ord[i] * c.ratio + r);
  for (int j = complete * c.ratio; j < visible; j++) sel.push_back(j);
}

// One query head over the selected keys: scores (fp16 keys: the query rounded to fp16, products widened and summed in
// fp32 by FMLAL, as the 35B's attention; the reference path in fp32), softmax with the max subtracted and a double sum,
// the values weighted in key order, times sigmoid(gate)
void Flash::qsa_head(int L, int h, const float* q, const float* gate, const std::vector<int>& sel, float* o) const {
  const LState& st = S->ls[L];
  const int HD = c.HD, ctx = S->ctx, kvh = h / (c.NQ / c.NKV), n = (int)sel.size();
  static thread_local std::vector<float> sc;
  sc.resize(n);
  const float scale = 1.f / std::sqrt((float)HD);
  float mx = -INFINITY;
  if (st.Kf) {
    const float* K = st.Kf + (size_t)kvh * ctx * HD;
    for (int t = 0; t < n; t++) {
      const float* kr = K + (size_t)sel[t] * HD;
      float32x4_t a = vdupq_n_f32(0);
      for (int i = 0; i < HD; i += 4) a = vfmaq_f32(a, vld1q_f32(q + i), vld1q_f32(kr + i));
      sc[t] = vaddvq_f32(a) * scale;
      mx = std::max(mx, sc[t]);
    }
  } else {  // fp16 keys and values: the 35B's head over the selected positions (src/models/common/gated_attn.h)
    _Float16 q16[512];
    for (int i = 0; i < HD; i++) q16[i] = (_Float16)q[i];
    gated_attn::head_f16(q16, st.K + (size_t)kvh * ctx * HD, st.V + (size_t)kvh * ctx * HD, HD, n, [&](int t) { return sel[t]; }, gate, o, sc);
    return;
  }
  double zs = 0;
  for (int t = 0; t < n; t++) {
    sc[t] = std::exp(sc[t] - mx);
    zs += sc[t];
  }
  const float iz = (float)(1.0 / zs);
  float32x4_t acc[128];
  for (int i = 0; i < HD / 4; i++) acc[i] = vdupq_n_f32(0);
  for (int t = 0; t < n; t++) {
    const float32x4_t pv = vdupq_n_f32(sc[t] * iz);
    if (st.Vf) {
      const float* vr = st.Vf + ((size_t)kvh * ctx + sel[t]) * HD;
      for (int i = 0; i < HD; i += 4) acc[i / 4] = vfmaq_f32(acc[i / 4], vld1q_f32(vr + i), pv);
    } else {
      const _Float16* vr = st.V + ((size_t)kvh * ctx + sel[t]) * HD;
      for (int i = 0; i < HD; i += 8) {
        float16x8_t vv = vld1q_f16((const float16_t*)vr + i);
        acc[i / 4] = vfmaq_f32(acc[i / 4], vcvt_f32_f16(vget_low_f16(vv)), pv);
        acc[i / 4 + 1] = vfmaq_f32(acc[i / 4 + 1], vcvt_high_f32_f16(vv), pv);
      }
    }
  }
  for (int i = 0; i < HD; i += 4) vst1q_f32(o + i, acc[i / 4]);
  for (int i = 0; i < HD; i++) o[i] *= sigm(gate[i]);
}

// ------------------------------------------------------------------------------------------------ decode blocks
namespace {
struct HcCtx {
  const FCfg* c;
  const HCw* w;
  const float* hyper;
  float *nrm, *lo, *up, *mixed, *r;
  XQ *xn, *xl;
  int nd = 0, nu = 0;
  bool inj = false;
  alignas(128) std::atomic<int> next{0};
  alignas(128) std::atomic<int> streams_left{0};
  alignas(128) std::atomic<int> down_left{0};
  alignas(128) std::atomic<int> lo_ready{0};
};
void phase_hc(void* p, int, int) {
  HcCtx& X = *(HcCtx*)p;
  const FCfg& c = *X.c;
  const int H = c.H, HC = c.HC, LOW = c.LOW, nb = H / XB;
  const int a_down = HC, a_inj = a_down + X.nd, a_lo = a_inj + (X.inj ? 1 : 0), a_up = a_lo + 1, total = a_up + X.nu;
  auto wait = [](const std::atomic<int>& v, int want) {
    while (v.load(std::memory_order_acquire) != want) __builtin_arm_yield();
  };
  for (int job; (job = X.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
    if (job < a_down) {
      const int st = job;
      rmsnorm(X.hyper + (size_t)st * H, X.w->norm + (size_t)st * H, H, c.eps, X.nrm + (size_t)st * H);
      for (int b = 0; b < nb; b++) xq_block(*X.xn, st * nb + b, X.nrm + (size_t)st * H + XB * b);
      X.streams_left.fetch_sub(1, std::memory_order_release);
    } else if (job < a_lo) {
      wait(X.streams_left, 0);
      if (job < a_inj) {
        const int r0 = (job - a_down) * 8, r1 = std::min(LOW, r0 + 8);
        mv(X.w->down, r0, r1, X.xn, X.nrm, X.lo + r0);
      } else {
        mv(X.w->inj, 0, X.w->inj.rows, X.xn, X.nrm, X.r);
      }
      X.down_left.fetch_sub(1, std::memory_order_release);
    } else if (job == a_lo) {
      wait(X.down_left, 0);
      for (int i = 0; i < LOW; i++) X.lo[i] = silu(X.lo[i] / (float)HC);
      for (int b = 0; b < LOW / XB; b++) xq_block(*X.xl, b, X.lo + XB * b);
      X.lo_ready.store(1, std::memory_order_release);
    } else {
      wait(X.lo_ready, 1);
      const int i0 = (job - a_up) * 64, i1 = std::min(H, i0 + 64);
      for (int st = 0; st < HC; st++) mv(X.w->up, st * H + i0, st * H + i1, X.xl, X.lo, X.up + (size_t)st * H + i0);
      for (int i = i0; i < i1; i++) {
        float acc = 0;
        for (int st = 0; st < HC; st++) acc = __builtin_fmaf(sigm(X.up[(size_t)st * H + i]), X.nrm[(size_t)st * H + i], acc);
        X.mixed[i] = acc / (float)HC;
      }
    }
  }
}
}  // namespace

void Flash::hc_step(const HCw& w, const float* hyper, float* mixed, float* inj_out) {
  const int N = c.N();
  const auto t0 = clk::now();
  if (ab_on == 1 && ab_arm && !ref) {  // (measurement: the mixer in one fork-join)
    float r[64];
    HcCtx X;
    X.c = &c, X.w = &w, X.hyper = hyper;
    X.nrm = nrm.data(), X.lo = lo.data(), X.up = up.data(), X.mixed = mixed, X.r = r;
    xin.reset(N);
    xlo.reset(c.LOW);
    X.xn = &xin.q, X.xl = &xlo.q;
    X.nd = (c.LOW + 7) / 8, X.nu = (c.H + 63) / 64, X.inj = inj_out != nullptr;
    X.streams_left.store(c.HC), X.down_left.store(X.nd + (X.inj ? 1 : 0));
    core->pool->run_any(phase_hc, &X);
    if (inj_out)
      for (int st = 0; st < c.HC; st++) inj_out[st] = 2.f * sigm(r[st] / (float)c.HC);
    T->msg += ms_since(t0);
    return;
  }
  // the per-stream norms on the pool, one stream per task (each stream's arithmetic is the serial loop's)
  parallel_for(*core->pool, c.HC, [&](int s) {
    rmsnorm(hyper + (size_t)s * c.H, w.norm + (size_t)s * c.H, c.H, c.eps, nrm.data() + (size_t)s * c.H);
  });
  float r[64];
  Job jb[2] = {{&w.down, &xin.q, nrm.data(), lo.data()}, {&w.inj, &xin.q, nrm.data(), r}};
  {
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {nrm.data()};
    const int ks[1] = {N};
    run_rows_q(*core->pool, jb, inj_out ? 2 : 1, ab_on == 2 && ab_arm ? 8 : 16, ref, &T->bytes_res, qs, xs, ks, 1);
  }
  hc_lo(lo.data());
  Job ju = {&w.up, &xlo.q, lo.data(), up.data()};
  {
    XBuf* qs[1] = {&xlo};
    const float* xs[1] = {lo.data()};
    const int ks[1] = {c.LOW};
    run_rows_q(*core->pool, &ju, 1, ab_on == 2 && ab_arm ? 64 : 32, ref, &T->bytes_res, qs, xs, ks, 1);
  }
  // the mix on the pool in blocks of 256 outputs (every output's arithmetic is hc_mix's: the same bits)
  const int H = c.H, nb = (H + 255) / 256;
  const float* un = up.data();
  const float* nn = nrm.data();
  parallel_for(*core->pool, nb, [&](int b) {
    const int i1 = std::min(H, (b + 1) * 256);
    for (int i = b * 256; i < i1; i++) {
      float acc = 0;
      for (int s = 0; s < c.HC; s++) acc = __builtin_fmaf(sigm(un[(size_t)s * H + i]), nn[(size_t)s * H + i], acc);
      mixed[i] = acc / (float)c.HC;
    }
  });
  if (inj_out)
    for (int s = 0; s < c.HC; s++) inj_out[s] = 2.f * sigm(r[s] / (float)c.HC);
  T->msg += ms_since(t0);  // the hyper-connection mixers (norms, down / inject / up products): their own stage
}

void Flash::gdn_step(int L, const float* xh, float* y) {
  const Lw& W = lw[L];
  const int C = c.conv_dim(), ZD = c.NVH * c.DV;
  auto t0 = clk::now();
  float* qkv = buf.data();
  float* z = qkv + C;
  float* bb = z + ZD;
  float* aa = bb + c.NVH;
  float* cv = aa + c.NVH;
  stage_mark(ST_ATTN_PROJ);
  Job jb[4] = {{&W.qkv, &xin.q, xh, qkv}, {&W.z, &xin.q, xh, z}, {&W.wb, &xin.q, xh, bb}, {&W.wa, &xin.q, xh, aa}};
  {
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {xh};
    const int ks[1] = {c.H};
    run_rows_q(*core->pool, jb, 4, 32, ref, &T->bytes_res, qs, xs, ks, 1);
  }
  auto t1 = clk::now();
  T->attn_proj += ms_between(t0, t1);
  stage_mark(ST_ATTN_CORE);
  gdn::Ctx X;
  X.d = &gd;
  X.w = W.g;
  X.conv_st = S->ls[L].conv.data();
  X.S = S->ls[L].S.data();
  X.qkv = qkv, X.z = z, X.bb = bb, X.aa = aa, X.cv = cv, X.o = att.data();
  X.xq = &xatt.q;
  X.quant_heads = !ref && c.DV % XB == 0 && !(ab_on == 3 && ab_arm);  // (DMENG_AB=3: arm 1 the output projection after the heads)
  for (int j = 0; j < c.NKH; j++) X.qk_done[j].store(0, std::memory_order_relaxed);
  xatt.reset(ZD);
  if (X.quant_heads) gdn::fuse_out(X, &W.o, y, core->pool->size());  // the output projection in the heads' fork-join
  core->pool->run(gdn::phase_conv_heads, &X);
  auto t2 = clk::now();
  T->split_heads_out(t1, t2, X.o_chunks ? X.heads_done_ns.load() : 0);  // the fused output projection's share: attn_out
  if (!X.quant_heads) {
    if (!ref) xatt.prep(att.data(), ZD);
    Job jo = {&W.o, &xatt.q, att.data(), y};
    run_rows(*core->pool, &jo, 1, 32, ref, &T->bytes_res);
  } else {
    T->bytes_res += W.o.bytes();
  }
  T->attn_out += ms_since(t2);
}

void Flash::qsa_step(int L, const float* xh, float* y) {
  const Lw& W = lw[L];
  const int HD = c.HD, pos = S->pos;
  if (pos >= S->ctx) {
    fprintf(stderr, "context full (%d tokens)\n", S->ctx);
    abort();
  }
  auto t0 = clk::now();
  float* qp = buf.data();
  float* kp = qp + 2 * c.NQ * HD;
  float* vp = kp + c.NKV * HD;
  float* ixq = vp + c.NKV * HD;
  float* q = ixq + (c.IXH + 1) * c.IXD;
  float* gate = q + c.NQ * HD;
  stage_mark(ST_ATTN_PROJ);
  Job jb[4] = {{&W.q, &xin.q, xh, qp}, {&W.k, &xin.q, xh, kp}, {&W.v, &xin.q, xh, vp}, {&W.ix, &xin.q, xh, ixq}};
  {
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {xh};
    const int ks[1] = {c.H};
    run_rows_q(*core->pool, jb, 4, 32, ref, &T->bytes_res, qs, xs, ks, 1);
  }
  auto t1 = clk::now();
  T->attn_proj += ms_between(t0, t1);
  stage_mark(ST_ATTN_CORE);
  qsa_kv(L, pos, qp, kp, vp, ixq, q, gate);
  if ((pos + 1) % c.ratio == 0) pool_key(L, pos / c.ratio);
  std::vector<int>& sel = sel_;  // (a member: the pool's threads read it)
  qsa_select(L, pos, ixq, sel, true);
  parallel_for(*core->pool, c.NQ, [&](int h) { qsa_head(L, h, q + (size_t)h * HD, gate + (size_t)h * HD, sel, att.data() + (size_t)h * HD); });
  auto t2 = clk::now();
  T->attn_core += ms_between(t1, t2);
  stage_mark(ST_ATTN_OUT);
  xatt.reset(c.NQ * HD);
  if (!ref) xatt.prep(att.data(), c.NQ * HD);
  Job jo = {&W.o, &xatt.q, att.data(), y};
  run_rows(*core->pool, &jo, 1, 32, ref, &T->bytes_res);
  T->attn_out += ms_since(t2);
}

// the PLE of this step (its n-gram rows were issued at the step's start); commit = false: the run-ahead's evaluation
void Flash::ple_step(int L, const float* hyper, float* out, bool commit) {
  const Lw& W = lw[L];
  const int N = c.N();
  auto t0 = clk::now();
  ngs.wait(*ng_cur);
  for (int h = 0; h < c.heads(); h++) ng_row(ng_cur->data.data() + (size_t)h * c.ng_bpr, ple_e.data() + (size_t)h * c.ng_dim);
  float* key = out + N;
  float* value = key + N;
  Job jb[2] = {{&W.pk, &xe.q, ple_e.data(), key}, {&W.pv, &xe.q, ple_e.data(), value}};
  {
    XBuf* qs[1] = {&xe};
    const float* xs[1] = {ple_e.data()};
    const int ks[1] = {c.ple_dim};
    run_rows_q(*core->pool, jb, 2, 32, ref, &T->bytes_res, qs, xs, ks, 1);
  }
  float* gated = value + c.H;
  float* normed = gated + N;
  ple_gate(W, key, value, hyper, gated, normed);
  float* st = S->pconv.data();
  for (int ch = 0; ch < N; ch++) out[ch] = gated[ch] + ple_conv(W, st, normed, ch, commit);
  T->misc += ms_since(t0);
}

void Flash::head_step(const float* xh, Want want) {
  auto th = clk::now();
  stage_mark(ST_HEAD);
  xin.prep(xh, c.H);
  if (fast_head && (want == Want::GREEDY || (want == Want::TOPK && head_topk > 0))) {
    greedy_ = sh.run(*core->pool, xin.q, want == Want::GREEDY ? 1 : head_topk, lgts.data());
  } else {
    Job jh = {&head, &xin.q, xh, lgts.data()};
    run_rows(*core->pool, &jh, 1, 64, ref, &T->bytes_res);
  }
  T->head += ms_since(th);
}

// ------------------------------------------------------------------------------------------------ one token
void Flash::step(State& s0, int32_t tok, Want want) {
  S = (FState*)&s0;
  greedy_ = -1;
  const int H = c.H, N = c.N();
  auto t0 = clk::now();
  const double msg0 = T->msg;
  struct AbAcc {  // measurement (DMENG_AB): this step's time and mixer time to its arm
    Flash* f;
    clk::time_point t0;
    double msg0;
    ~AbAcc() {
      if (!f->ab_on) return;
      f->ab_ms[f->ab_arm] += ms_since(t0), f->ab_msg[f->ab_arm] += f->T->msg - msg0, f->ab_n[f->ab_arm]++;
      f->ab_arm ^= 1;
    }
  } ab_acc{this, t0, msg0};
  stage_mark(ST_EMBED);
  {  // the n-gram rows first: their reads run while the embedding and layer 0 compute
    int64_t r[64];
    ng.rows(S->ng_ctx, tok, r);
    ng_cur = ngs.issue(r, c.heads());
  }
  rows.row(tok, emb.data());
  for (int s = 0; s < c.HC; s++) std::memcpy(hyp.data() + (size_t)s * H, emb.data(), sizeof(float) * H);
  T->embed += ms_since(t0);
  core->token_start(S->pos, tok, emb.data());
  float inj1[64], inj2[64];
  for (int L = 0; L < c.NL; L++) {
    const Lw& W = lw[L];
    if (W.ple) {
      ple_step(L, hyp.data(), ple_out.data(), true);
      for (int i = 0; i < N; i++) hyp[i] = hyp[i] + ple_out[i];
    }
    stage_mark(ST_NORMQ);
    hc_step(W.ahc, hyp.data(), mix.data(), inj1);
    if (W.lin) gdn_step(L, mix.data(), blk.data());
    else qsa_step(L, mix.data(), blk.data());
    stage_mark(ST_NORMQ);
    for (int s = 0; s < c.HC; s++)
      for (int i = 0; i < H; i++) hyp[(size_t)s * H + i] = __builtin_fmaf(blk[i], inj1[s], hyp[(size_t)s * H + i]);
    hc_step(W.mhc, hyp.data(), mix.data(), inj2);
    cur_inj2 = inj2;
    core->moe_step(L, mix.data(), blk.data(), views[L]);
    for (int s = 0; s < c.HC; s++)
      for (int i = 0; i < H; i++) hyp[(size_t)s * H + i] = __builtin_fmaf(blk[i], inj2[s], hyp[(size_t)s * H + i]);
  }
  cur_inj2 = nullptr;
  if (want != Want::NONE) {
    hc_step(fin, hyp.data(), x.data(), nullptr);
    head_step(x.data(), want);
  }
  S->ng_ctx[0] = S->ng_ctx[1];
  S->ng_ctx[1] = tok;
  S->pos++;
  core->token_end();
  stage_mark(ST_OTHER);
}

bool Flash::embed_row(int32_t tok, float* out) const {
  rows.row(tok, out);
  return true;
}

// Run-ahead (Core::run_ahead): from layer L's approximate MoE output (its experts in memory only), layer L + 1 up to its
// router on the approximate state, its top picks. Nothing of the state changes: the PLE's conv and a Gated DeltaNet
// layer's conv and recurrence are evaluated without writing; a QSA layer writes its K, V and indexer key at this
// position, which the real pass writes again before anything reads them. The MoE input of layer L (the core reads it
// after this returns) and the injection weights are not touched: the run-ahead has its own. Gives up as soon as the read
// the step waits for lands (Core::ra_stopped). Approximate by design: it only decides reads.
int Flash::run_ahead_step(int L, const float* ypart, int32_t* picks, int maxk) {
  const int L1 = L + 1, H = c.H, N = c.N();
  if (L1 >= c.NL || !cur_inj2) return 0;
  const Times keep = *T;
  struct Restore {
    Times* T;
    Times keep;
    ~Restore() { *T = keep; }
  } restore{T, keep};
  const Lw& W = lw[L1];
  ra_hyp.resize(N);
  ra_mix.resize(H);
  ra_blk.resize(H);
  float ri[64];
  for (int s = 0; s < c.HC; s++)
    for (int i = 0; i < H; i++) ra_hyp[(size_t)s * H + i] = __builtin_fmaf(ypart[i], cur_inj2[s], hyp[(size_t)s * H + i]);
  if (W.ple) {
    ple_step(L1, ra_hyp.data(), ple_out.data(), false);
    for (int i = 0; i < N; i++) ra_hyp[i] += ple_out[i];
  }
  hc_step(W.ahc, ra_hyp.data(), ra_mix.data(), ri);
  if (core->ra_stopped()) return 0;
  if (W.lin) {  // the Gated DeltaNet without writing its states
    const int C = c.conv_dim(), ZD = c.NVH * c.DV;
    float* qkv = buf.data();
    float* z = qkv + C;
    float* bb = z + ZD;
    float* aa = bb + c.NVH;
    float* cv = aa + c.NVH;
    Job jb[4] = {{&W.qkv, &xin.q, ra_mix.data(), qkv}, {&W.z, &xin.q, ra_mix.data(), z}, {&W.wb, &xin.q, ra_mix.data(), bb}, {&W.wa, &xin.q, ra_mix.data(), aa}};
    XBuf* qs[1] = {&xin};
    const float* xs[1] = {ra_mix.data()};
    const int ks[1] = {H};
    run_rows_q(*core->pool, jb, 4, 32, ref, nullptr, qs, xs, ks, 1);
    if (core->ra_stopped()) return 0;
    gdn::conv_ro(gd, W.g, S->ls[L1].conv.data(), qkv, cv);
    const float* St = S->ls[L1].S.data();
    parallel_for(*core->pool, c.NVH, [&](int h) {
      gdn::head_ro(gd, W.g, St + (size_t)h * c.DV * c.DK, cv, z + h * c.DV, bb[h], aa[h], h, att.data() + h * c.DV);
    });
    if (core->ra_stopped()) return 0;
    xatt.reset(ZD);
    if (!ref) xatt.prep(att.data(), ZD);
    Job jo = {&W.o, &xatt.q, att.data(), ra_blk.data()};
    run_rows(*core->pool, &jo, 1, 32, ref);
  } else {
    qsa_step(L1, ra_mix.data(), ra_blk.data());
  }
  if (core->ra_stopped()) return 0;
  for (int s = 0; s < c.HC; s++)
    for (int i = 0; i < H; i++) ra_hyp[(size_t)s * H + i] = __builtin_fmaf(ra_blk[i], ri[s], ra_hyp[(size_t)s * H + i]);
  hc_step(W.mhc, ra_hyp.data(), ra_mix.data(), ri);
  if (core->ra_stopped()) return 0;
  if (ra_xq.cap < H) ra_xq.init(H, prec);
  ra_xq.prep(ra_mix.data(), H);
  ra_lg.resize(W.router.rows);
  Job jr = {&W.router, &ra_xq.q, ra_mix.data(), ra_lg.data()};
  run_rows(*core->pool, &jr, 1, 16, ref);
  if (core->ra_stopped()) return 0;
  std::vector<int> idx(c.NE);
  for (int e = 0; e < c.NE; e++) idx[e] = e;
  const int k = std::min(maxk, c.NE);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return ra_lg[a] > ra_lg[b] || (ra_lg[a] == ra_lg[b] && a < b); });
  for (int i = 0; i < k; i++) picks[i] = idx[i];
  core->ra_lgv.assign(ra_lg.begin(), ra_lg.begin() + c.NE);
  return k;
}

bool Flash::set_option(const std::string& name, const std::string& value, std::string& err) {
  if (name == "run_ahead") {  // > 0: run ahead one layer while a step waits on reads, naming this many picks
    const int k = std::stoi(value);
    if (k <= 0) return (core->run_ahead = nullptr), true;
    core->ra_k = k;
    core->run_ahead = [this](int step, const float* yp, int32_t* pk, int maxk) { return run_ahead_step(step, yp, pk, maxk); };
    return true;
  }
  if (name == "run_ahead_depth") {  // the 35B's measurement of deeper run-aheads only: refused, not ignored
    if (std::stoi(value) <= 1) return true;
    return (err = "run_ahead_depth > 1: not implemented for qwen4_exp (one layer ahead)"), false;
  }
  if (name == "fast_head") {  // 1: the screened head (greedy, and top-k with head_topk)
    if (std::stoi(value) <= 0 || fast_head) return true;
    if (prec != Prec::I16 || ref) return (err = "fast_head: needs I16 activations"), false;
    if (res_mem == Mem::FILE) return (err = "fast_head: the resident weights must be loaded (not mapped) to split the head"), false;
    if (head.kind != Weight::AFF8) return (err = "fast_head: needs an 8-bit head"), false;
    if (!sh.build(*core->pool, head, c.V, c.V, err)) return false;
    fast_head = true;
    return true;
  }
  if (name == "verify_old") return true;  // (the 35B's previous verify pass, for A/B: this family has one pass)
  if (name == "head_topk") return (head_topk = std::min(std::stoi(value), ScreenHead::KMAX)), true;
  if ((name == "prefill_batched_always" && std::stoi(value) == 0) || (name == "prefill_record" && value.empty())) return true;  // off
  if (name == "prefill_batched_always" || name == "prefill_record")  // refused, not ignored
    return (err = name + ": not implemented for qwen4_exp's prefill"), false;
  bool handled = false;  // the shared prefill_* options (dmeng/prefill.h)
  const bool ok = prefill_option(name, value, po, gpu, R.base, R.size, core, handled, err);
  if (handled) return ok;
  if (!core) return (err = "unknown option " + name + " (or bind the model first)"), false;
  return core->set_option(name, value, err);  // the engine's own options (Core::set_option)
}

std::string Flash::stats_json(int ntok) const {
  std::ostringstream o;
  const double n = std::max(1, ntok);
  o << "{\"ngram\":{\"rows_per_token\":" << ngs.n_rows.load() / n << ",\"reads_per_token\":" << ngs.n_reads.load() / n
    << ",\"mb_read_per_token\":" << ngs.n_bytes.load() / 1e6 / n << ",\"wait_ms_per_use\":" << ngs.wait_ms / std::max(1.0, ngs.uses)
    << ",\"lead_ms_per_use\":" << ngs.lead_ms / std::max(1.0, ngs.uses) << ",\"uses\":" << ngs.uses << "}";
  if (fast_head) {
    const double nc = std::max(1.0, sh.n_calls);
    o << ",\"fast_head\":{\"candidates_per_token\":" << sh.n_cand / nc << ",\"coarse_ms\":" << sh.ms_coarse / nc << ",\"exact_ms\":" << sh.ms_exact / nc << "}";
  }
  if (V) o << ",\"verify\":" << verify_json();
  if (ab_on)
    for (int a = 0; a < 2; a++)
      o << ",\"ab" << a << "\":{\"steps\":" << ab_n[a] << ",\"ms\":" << ab_ms[a] / std::max(1.0, ab_n[a]) << ",\"msg_ms\":" << ab_msg[a] / std::max(1.0, ab_n[a]) << "}";
  o << "}";
  return o.str();
}

void Flash::reset_stats() {
  sh.reset_stats();
  ngs.reset_stats();
  for (int a = 0; a < 2; a++) ab_ms[a] = ab_msg[a] = ab_n[a] = 0;
}

ModelMem Flash::mem(const State& s0) const {
  const FState& s = (const FState&)s0;
  ModelMem m;
  m.resident = R.size;
  for (auto& st : s.ls) m.kv_reserved += 2 * st.kv_bytes + st.ix_bytes + st.ixp_bytes;
  m.recurrent = fixed_bytes(s);
  m.prefill_scratch = span_bytes();
  return m;
}

}  // namespace qwen4

std::unique_ptr<Model> make_qwen4_exp() { return std::unique_ptr<Model>(new qwen4::Flash()); }

}  // namespace dm
