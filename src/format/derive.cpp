#include "derive.h"

#include <unistd.h>

#include <chrono>
#include <cstring>
#include <set>
#include <vector>

#include "files.h"

namespace ms {

namespace {

struct Ctx {
  const SafeTensors& st;
  DmWriter& w;
  DeriveStats& s;
  std::string& err;
};

bool fail(Ctx& c, const std::string& m) {
  if (c.err.empty()) c.err = m;
  return false;
}

const StTensor* need(Ctx& c, const std::string& key) {
  const StTensor* t = c.st.find(key);
  if (!t) fail(c, c.st.path + ": missing tensor " + key);
  return t;
}

Src src_of(const StTensor& t) { return t.dtype == "BF16" ? Src::BF16 : t.dtype == "F16" ? Src::F16 : Src::F32; }

bool read_f32(Ctx& c, const StTensor& t, std::vector<float>& out) {
  int64_t n = t.numel();
  out.resize(n);
  if (t.dtype == "F32") return c.st.read(t, out.data(), c.err);
  if (t.dtype != "BF16" && t.dtype != "F16") return fail(c, t.name + ": unsupported dtype " + t.dtype);
  std::vector<uint16_t> raw(n);
  if (!c.st.read(t, raw.data(), c.err)) return false;
  for (int64_t i = 0; i < n; i++) out[i] = t.dtype == "BF16" ? bf16_bits_to_f32(raw[i]) : f16_bits_to_f32(raw[i]);
  return true;
}

// f32 tensor (widened exactly from bf16 / f16), optionally + 1 (zero-centred norms), optionally reshaped
bool put_f32(Ctx& c, const std::string& name, const std::string& key, bool plus1 = false, std::vector<int64_t> shape = {}) {
  const StTensor* t = need(c, key);
  if (!t) return false;
  std::vector<float> v;
  if (!read_f32(c, *t, v)) return false;
  if (plus1)
    for (auto& x : v) x = x + 1.0f;
  if (shape.empty()) shape = t->shape;
  c.s.tensors++;
  return c.w.add(name, "f32", v.data(), v.size() * 4, shape) || fail(c, "write failed: " + name);
}

// bf16 tensor: the release's bytes (or round-to-nearest-even from f32, as mx astype does)
bool put_bf16(Ctx& c, const std::string& name, const std::string& key) {
  const StTensor* t = need(c, key);
  if (!t) return false;
  std::vector<uint16_t> v(t->numel());
  if (t->dtype == "BF16") {
    if (!c.st.read(*t, v.data(), c.err)) return false;
  } else {
    std::vector<float> f;
    if (!read_f32(c, *t, f)) return false;
    for (size_t i = 0; i < f.size(); i++) v[i] = f32_to_bf16_bits(f[i]);
  }
  c.s.tensors++;
  return c.w.add(name, "bf16", v.data(), v.size() * 2, t->shape) || fail(c, "write failed: " + name);
}

// an 8-bit (or 4-bit) affine matrix NAME.q / .s / .b from a 2-D tensor, read and quantised in row blocks
bool put_q(Ctx& c, const std::string& name, const std::string& key, Rule rule, int bits = 8, const char* qsuffix = ".q") {
  const StTensor* t = need(c, key);
  if (!t) return false;
  if (t->shape.size() != 2 || t->shape[1] % 64) return fail(c, key + ": not a [R, C % 64 == 0] matrix");
  const int64_t R = t->shape[0], C = t->shape[1], NG = C / 64, es = t->dtype == "F32" ? 4 : 2;
  const int64_t cb = bits == 8 ? C : C / 2;
  std::vector<uint16_t> S(R * NG), B(R * NG);
  int64_t block = std::max<int64_t>(1, (32ll << 20) / (C * es));
  std::vector<uint8_t> src(block * C * es), codes(block * cb);
  if (!c.w.begin(name + qsuffix, "u8", R * cb, {R, cb})) return fail(c, "write failed: " + name);
  for (int64_t r0 = 0; r0 < R; r0 += block) {
    int64_t n = std::min(block, R - r0);
    uint64_t off = c.st.data_offset(*t) + (uint64_t)(r0 * C * es);
    uint64_t want = n * C * es, got = 0;
    while (got < want) {
      ssize_t k = pread(c.st.fd, src.data() + got, want - got, off + got);
      if (k <= 0) return fail(c, "short read of " + key);
      got += k;
    }
    quantize_rows(src.data(), src_of(*t), (int)n, (int)C, bits, rule, codes.data(), S.data() + r0 * NG, B.data() + r0 * NG, 0,
                  (int)n, &c.s.q);
    if (!c.w.append(codes.data(), n * cb)) return fail(c, "write failed: " + name);
  }
  if (!c.w.finish() || !c.w.add(name + ".s", "f16", S.data(), S.size() * 2, {R, NG}) ||
      !c.w.add(name + ".b", "f16", B.data(), B.size() * 2, {R, NG}))
    return fail(c, "write failed: " + name);
  c.s.matrices++;
  c.s.tensors += 3;
  return true;
}

// the same for an in-memory f16 matrix [R, C] (router blocks)
bool put_q_mem(DmWriter& w, DeriveStats& s, const std::string& name, const uint16_t* f16, int64_t R, int64_t C, int bits,
               const char* qsuffix, std::string& err) {
  const int64_t NG = C / 64, cb = bits == 8 ? C : C / 2;
  std::vector<uint8_t> codes(R * cb);
  std::vector<uint16_t> S(R * NG), B(R * NG);
  quantize_rows(f16, Src::F16, (int)R, (int)C, bits, Rule::F16, codes.data(), S.data(), B.data(), 0, (int)R, &s.q);
  if (!w.add(name + qsuffix, "u8", codes.data(), codes.size(), {R, cb}) || !w.add(name + ".s", "f16", S.data(), S.size() * 2, {R, NG}) ||
      !w.add(name + ".b", "f16", B.data(), B.size() * 2, {R, NG})) {
    err = "write failed: " + name;
    return false;
  }
  s.matrices++;
  s.tensors += 3;
  return true;
}

// a calibrated 4-bit matrix NAME.q4 / .s / .b copied as stored from a calibrated export (see derive.h); key = the release's
// tensor, whose shape [R, C] the export's codes [R, C/2], scales and biases [R, C/64] and BASE.shape must match
bool put_q4_cal(Ctx& c, const SafeTensors& cal, std::set<std::string>& used, const std::string& name, const std::string& key) {
  const StTensor* t = need(c, key);
  if (!t) return false;
  if (t->shape.size() != 2 || t->shape[1] % 64) return fail(c, key + ": not a [R, C % 64 == 0] matrix");
  const int64_t R = t->shape[0], C = t->shape[1], NG = C / 64;
  if (key.size() < 8 || key.compare(key.size() - 7, 7, ".weight")) return fail(c, key + ": not a .weight key");
  const std::string base = key.substr(0, key.size() - 7);
  const StTensor *q = cal.find(base + ".codes"), *s = cal.find(base + ".scales"), *b = cal.find(base + ".biases"),
                 *sh = cal.find(base + ".shape");
  if (!q || !s || !b || !sh) return fail(c, cal.path + ": missing calibrated tensors for " + base);
  auto is = [](const StTensor* x, const char* dt, std::vector<int64_t> shp) { return x->dtype == dt && x->shape == shp; };
  if (!is(q, "U8", {R, C / 2}) || !is(s, "F16", {R, NG}) || !is(b, "F16", {R, NG}) || !is(sh, "I64", {2}))
    return fail(c, cal.path + ": " + base + ": dtype or shape does not match the release's [" + std::to_string(R) + ", " +
                       std::to_string(C) + "]");
  int64_t rc[2];
  if (!cal.read(*sh, rc, c.err)) return false;
  if (rc[0] != R || rc[1] != C) return fail(c, cal.path + ": " + base + ".shape differs from the release");
  std::vector<uint8_t> codes(R * C / 2);
  std::vector<uint16_t> S(R * NG), B(R * NG);
  if (!cal.read(*q, codes.data(), c.err) || !cal.read(*s, S.data(), c.err) || !cal.read(*b, B.data(), c.err)) return false;
  if (!c.w.add(name + ".q4", "u8", codes.data(), codes.size(), {R, C / 2}) || !c.w.add(name + ".s", "f16", S.data(), S.size() * 2, {R, NG}) ||
      !c.w.add(name + ".b", "f16", B.data(), B.size() * 2, {R, NG}))
    return fail(c, "write failed: " + name);
  if (!used.insert(base).second) return fail(c, base + ": used twice");
  c.s.matrices++;
  c.s.tensors += 3;
  c.s.calibrated++;
  return true;
}

bool resident_qwen3_5_moe(Ctx& c, const Manifest& m, int bits, const SafeTensors* cal = nullptr,
                          std::set<std::string>* used = nullptr) {
  const std::string P = "model.language_model.";
  const Rule F = Rule::F16;
  const char* qs = bits == 4 ? ".q4" : ".q";
  // every affine matrix of the 8-bit recipe at `bits` (the call sites below pass bits and the suffix through this);
  // with a calibrated export, its codes / scales / biases instead of today's quantiser
  auto put_q = [&](Ctx& cc, const std::string& name, const std::string& key, Rule rule) {
    return cal ? put_q4_cal(cc, *cal, *used, name, key) : ::ms::put_q(cc, name, key, rule, bits, qs);
  };
  for (int L = 0; L < m.physical_layers; L++) {
    std::string p = P + "layers." + std::to_string(L) + ".", s = std::to_string(L) + ".";
    if (!put_f32(c, s + "ln_in", p + "input_layernorm.weight", true) || !put_f32(c, s + "ln_post", p + "post_attention_layernorm.weight", true))
      return false;
    if (m.layer_types[L] == 'L') {
      std::string a = p + "linear_attn.";
      const StTensor* cw = need(c, a + "conv1d.weight");
      if (!cw) return false;
      if (!put_q(c, s + "qkv", a + "in_proj_qkv.weight", F) || !put_q(c, s + "z", a + "in_proj_z.weight", F) ||
          !put_bf16(c, s + "a", a + "in_proj_a.weight") || !put_bf16(c, s + "b", a + "in_proj_b.weight") ||
          !put_f32(c, s + "conv", a + "conv1d.weight", false, {cw->shape[0], cw->shape[2]}) || !put_f32(c, s + "A_log", a + "A_log") ||
          !put_f32(c, s + "dt_bias", a + "dt_bias") || !put_f32(c, s + "gnorm", a + "norm.weight") ||
          !put_q(c, s + "o", a + "out_proj.weight", F))
        return false;
    } else {
      std::string a = p + "self_attn.";
      if (!put_q(c, s + "q", a + "q_proj.weight", F) || !put_q(c, s + "k", a + "k_proj.weight", F) ||
          !put_q(c, s + "v", a + "v_proj.weight", F) || !put_q(c, s + "o", a + "o_proj.weight", F) ||
          !put_f32(c, s + "qn", a + "q_norm.weight", true) || !put_f32(c, s + "kn", a + "k_norm.weight", true))
        return false;
    }
    std::string mm = p + "mlp.";
    if (!put_bf16(c, s + "router", mm + "gate.weight") || !put_q(c, s + "sg", mm + "shared_expert.gate_proj.weight", F) ||
        !put_q(c, s + "su", mm + "shared_expert.up_proj.weight", F) || !put_q(c, s + "sd", mm + "shared_expert.down_proj.weight", F) ||
        !put_bf16(c, s + "shgate", mm + "shared_expert_gate.weight"))
      return false;
  }
  return put_f32(c, "norm", P + "norm.weight", true) && put_q(c, "head", "lm_head.weight", F);
}

bool resident_inkling2_loop(Ctx& c, const Manifest& m, int bits = 8, const SafeTensors* cal = nullptr,
                            std::set<std::string>* used = nullptr) {
  const std::string P = "model.llm.";
  const Rule B = Rule::BF16;
  // bits 4 (memory-limited phones, profile phone-small): the same matrices (head and the six attention projections) as
  // plain affine 4-bit g64 (NAME.q4 / .s / .b); everything the 8-bit file keeps exact stays as it is; with a calibrated
  // export (jobs/babel/ink_q4c_export.py), its codes / scales / biases instead of today's quantiser
  const char* qs = bits == 4 ? ".q4" : ".q";
  auto put_q = [&](Ctx& cc, const std::string& name, const std::string& key, Rule rule) {
    return cal ? put_q4_cal(cc, *cal, *used, name, key) : ::ms::put_q(cc, name, key, rule, bits, qs);
  };
  const Json& tc = m.config["text_config"];
  const int CS = (int)tc["loop_config"]["core_start"].i(), CE = (int)tc["loop_config"]["core_end"].i();
  const int H = (int)m.hidden, SC = (int)tc["sconv_kernel_size"].i();
  if (!put_f32(c, "embed_norm", P + "embed_norm.weight") || !put_f32(c, "final_q", P + "attn_res.output_query") ||
      !put_f32(c, "norm", P + "norm.weight"))
    return false;
  {  // message queries: the checkpoint's [query, mlp] and [key, value] rows per looped layer -> [q, k, v, mlp] (checkpoint.py)
    const StTensor *q = need(c, P + "middle_cycle_cross_pass_queries"), *kv = need(c, P + "middle_cycle_cross_pass_kv_queries");
    if (!q || !kv) return false;
    std::vector<float> a, b;
    if (!read_f32(c, *q, a) || !read_f32(c, *kv, b)) return false;
    int64_t D = q->shape[1], NLp = q->shape[0] / 2;
    std::vector<float> o(NLp * 4 * D);
    for (int64_t i = 0; i < NLp; i++) {
      memcpy(&o[(i * 4 + 0) * D], &a[(2 * i) * D], D * 4);
      memcpy(&o[(i * 4 + 1) * D], &b[(2 * i) * D], D * 4);
      memcpy(&o[(i * 4 + 2) * D], &b[(2 * i + 1) * D], D * 4);
      memcpy(&o[(i * 4 + 3) * D], &a[(2 * i + 1) * D], D * 4);
    }
    c.s.tensors++;
    if (!c.w.add("msgq", "f32", o.data(), o.size() * 4, {NLp, 4, D})) return fail(c, "write failed: msgq");
  }
  if (!put_q(c, "head", P + "unembed.weight", B)) return false;
  const char* proj[6][2] = {{"q", "wq_du"}, {"k", "wk_dv"}, {"v", "wv_dv"}, {"r", "wr_du"}, {"o", "wo_ud"}, {"g", "gate_attn"}};
  for (int L = 0; L < m.physical_layers; L++) {
    std::string p = P + "layers." + std::to_string(L) + ".", s = std::to_string(L) + ".";
    bool g = m.layer_types[L] == 'G';
    if (L > 0 && !put_f32(c, s + "aq", P + "attn_res.attn_queries." + std::to_string(L - 1))) return false;
    if (!put_f32(c, s + "mq", P + "attn_res.mlp_queries." + std::to_string(L))) return false;
    if (g) {
      for (const char* x : {"q", "k", "v"})
        if (!put_f32(c, s + "ln" + x, p + x + "_input_norm.weight")) return false;
    } else if (!put_f32(c, s + "ln_in", p + "attn_norm.weight")) {
      return false;
    }
    for (auto& pr : proj)
      if (!put_q(c, s + pr[0], p + "attn." + pr[1] + ".weight", B)) return false;
    if (!put_f32(c, s + "sink", p + "attn.attn_sink_proj.weight") || !put_f32(c, s + "qclip", p + "attn.q_l2_clip.scale") ||
        !put_f32(c, s + "rclip", p + "attn.rel_l2_clip.scale") || !put_f32(c, s + "rel", p + "attn.rel_logits_proj.proj"))
      return false;
    for (const char* x : {"k", "v"}) {
      std::string sc = p + "attn." + x + "_sconv.";
      if (g) {
        if (!put_f32(c, s + x + "stat", sc + "static_w") || !put_f32(c, s + x + "wp", sc + "weight_proj")) return false;
      } else {
        const StTensor* t = need(c, sc + "weight");  // [C, 1, K] in the release; MLX swaps to [C, K, 1]: same bytes
        if (!t || !put_f32(c, s + x + "conv", sc + "weight", false, {t->numel() / SC, SC})) return false;
      }
    }
    const StTensor *ac = need(c, p + "attn_sconv.weight"), *mc = need(c, p + "mlp_sconv.weight");
    if (!ac || !mc || !put_f32(c, s + "aconv", p + "attn_sconv.weight", false, {H, ac->numel() / H}) ||
        !put_f32(c, s + "mconv", p + "mlp_sconv.weight", false, {H, mc->numel() / H}) ||
        !put_f32(c, s + "mlnorm", p + "mlp_norm.weight") || !put_bf16(c, s + "router", p + "mlp.gate.weight") ||
        !put_f32(c, s + "rbias", p + "mlp.gate.bias"))
      return false;
    if (L >= CS && L < CE) {
      const char* mg[4][2] = {{"q", "attn"}, {"k", "k"}, {"v", "v"}, {"m", "mlp"}};
      for (auto& x : mg) {
        std::string q = p + "controllers." + x[1] + ".", n = s + "mg" + x[0];
        if (!put_bf16(c, n + ".down", q + "down_weight") || !put_bf16(c, n + ".forget", q + "forget_weight") ||
            !put_bf16(c, n + ".gain", q + "gain_weight"))
          return false;
      }
    }
  }
  return true;
}

}  // namespace

bool derive_resident(const Manifest& m, const std::string& out_path, DeriveStats& st, std::string& err, int bits,
                     const std::string& calibrated) {
  auto t0 = std::chrono::steady_clock::now();
  SafeTensors src;
  if (!src.open(m.dir + "/" + m.raw["resident"]["file"].str_or("resident.safetensors"), err)) return false;
  SafeTensors cal;
  std::set<std::string> used;
  if (!calibrated.empty()) {
    if (bits != 4 || (m.family != "qwen3_5_moe" && m.family != "inkling2_loop")) {
      err = "a calibrated resident is defined for qwen3_5_moe and inkling2_loop at 4 bits only";
      return false;
    }
    if (!cal.open(calibrated, err)) return false;
  }
  Cfg cfg;
  if (!m.engine_cfg(cfg, err)) return false;
  DmWriter w;
  if (!w.open(out_path, err)) return false;
  for (auto& kv : cfg) w.cfg(kv.first, kv.second);
  Ctx c{src, w, st, err};
  bool ok;
  if (m.family == "qwen3_5_moe" || m.family == "inkling2_loop") {
    if (bits == 4) w.cfg("resident_bits", "4");
    if (m.family == "qwen3_5_moe") {
      ok = resident_qwen3_5_moe(c, m, bits, calibrated.empty() ? nullptr : &cal, &used);
    } else {
      ok = resident_inkling2_loop(c, m, bits, calibrated.empty() ? nullptr : &cal, &used);
      w.cfg("fp16_inexact", std::to_string(st.q.f16_inexact));
    }
    if (ok && !calibrated.empty()) {  // every matrix of the export used, nothing else in it
      int extra = 0;
      for (auto& t : cal.tensors) {
        size_t dot = t.name.rfind('.');
        std::string base = t.name.substr(0, dot), sfx = dot == std::string::npos ? "" : t.name.substr(dot);
        if (!used.count(base) || (sfx != ".codes" && sfx != ".scales" && sfx != ".biases" && sfx != ".shape")) extra++;
      }
      if (extra || cal.tensors.size() != used.size() * 4) {
        err = calibrated + ": " + std::to_string(extra) + " tensors not used (" + std::to_string(used.size()) + " matrices used of " +
              std::to_string(cal.tensors.size()) + " tensors)";
        ok = false;
      }
    }
  } else {
    err = "no resident recipe for family " + m.family;
    return false;
  }
  if (!ok) return false;
  if (m.family == "inkling2_loop" && st.q.f16_inexact) {
    err = std::to_string(st.q.f16_inexact) + " scale/bias values not exact in fp16";
    return false;
  }
  if (!w.close(err)) return false;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return true;
}

bool derive_router(const std::string& path, const std::string& out_path, int bits, bool split, DeriveStats& st, std::string& err) {
  auto t0 = std::chrono::steady_clock::now();
  SafeTensors z;
  if (!z.open(path, err)) return false;
  std::string fam = z.meta("family");
  int lag = atoi(z.meta("lag").c_str()), targets = atoi(z.meta("targets").c_str()), hid = atoi(z.meta("hidden").c_str());
  if (fam.empty() || targets <= 0) {
    err = path + ": no router metadata";
    return false;
  }
  if (split && fam != "qwen3_5_moe") {
    err = "split router files exist for qwen3_5_moe only";
    return false;
  }
  auto T = [&](const std::string& k) { return z.find(k); };
  const StTensor* en0 = T("0.emb_norm.weight");
  const StTensor* lb0 = T("0.lin.bias");
  if (!en0 || !lb0) {
    err = path + ": not a router file";
    return false;
  }
  const int64_t H = en0->shape[0], NE = lb0->shape[0];
  DmWriter w;
  if (!w.open(out_path, err)) return false;
  const bool ink = fam == "inkling2_loop";
  const char* qs = ink && bits == 4 ? ".q4" : ".q";
  if (!ink) {
    w.cfg("lag", std::to_string(lag));
    w.cfg("feats", "full");
    w.cfg("layers", std::to_string(targets));
    w.cfg("hidden", std::to_string(H));
    w.cfg("hidden_mlp", std::to_string(hid));
    if (bits == 4) w.cfg("bits", "4");
    if (split) w.cfg("split_pre", "1");
  }
  std::string widths;
  auto rd16 = [&](const StTensor* t, std::vector<uint16_t>& v) {
    if (!t || t->dtype != "F16") return false;
    v.resize(t->numel());
    return z.read(*t, v.data(), err);
  };
  auto rd32 = [&](const StTensor* t, std::vector<float>& v) {
    if (!t || t->dtype != "F32") return false;
    v.resize(t->numel());
    return z.read(*t, v.data(), err);
  };
  for (int L = 0; L < targets; L++) {
    std::string p = std::to_string(L) + ".";
    std::vector<uint16_t> lw, m0, m2;
    std::vector<float> en, lb, m0b, m2b;
    const StTensor* lwt = T(p + "lin.weight");
    if (!rd16(lwt, lw) || !rd32(T(p + "emb_norm.weight"), en) || !rd32(T(p + "lin.bias"), lb) ||
        (hid && (!rd16(T(p + "mlp.0.weight"), m0) || !rd32(T(p + "mlp.0.bias"), m0b) || !rd16(T(p + "mlp.2.weight"), m2) ||
                 !rd32(T(p + "mlp.2.bias"), m2b)))) {
      if (err.empty()) err = path + ": target " + std::to_string(L) + " has missing or mistyped tensors";
      return false;
    }
    const int64_t W = lwt->shape[1], R = NE + (hid ? hid : 0);
    if (W != 2 * H && W != 3 * H) {
      err = path + ": unexpected input width at target " + std::to_string(L);
      return false;
    }
    widths += (L ? "," : "") + std::to_string(W);
    // stack = rows [lin; mlp.0], then column blocks
    auto block = [&](int64_t c0, int64_t c1, std::vector<uint16_t>& out) {
      out.resize(R * (c1 - c0));
      for (int64_t r = 0; r < R; r++) {
        const uint16_t* row = r < NE ? &lw[r * W] : &m0[(r - NE) * W];
        memcpy(&out[r * (c1 - c0)], row + c0, (c1 - c0) * 2);
      }
    };
    if (!w.add(p + "emb_norm", "f32", en.data(), en.size() * 4, {H})) return false;
    std::vector<uint16_t> blk;
    if (split) {
      block(0, H, blk);
      if (!put_q_mem(w, st, p + "pre_e", blk.data(), R, H, bits, qs, err)) return false;
      block(H, 2 * H, blk);
      if (!put_q_mem(w, st, p + "pre_p", blk.data(), R, H, bits, qs, err)) return false;
    } else {
      block(0, 2 * H, blk);
      if (!put_q_mem(w, st, p + "pre", blk.data(), R, 2 * H, bits, qs, err)) return false;
    }
    if (W == 3 * H) {
      block(2 * H, 3 * H, blk);
      if (!put_q_mem(w, st, p + "lagm", blk.data(), R, H, bits, qs, err)) return false;
    }
    std::vector<float> bp(lb);
    bp.insert(bp.end(), m0b.begin(), m0b.end());
    if (!w.add(p + "b_pre", "f32", bp.data(), bp.size() * 4, {(int64_t)bp.size()})) return false;
    if (hid) {
      if (!put_q_mem(w, st, p + "m2", m2.data(), NE, hid, bits, qs, err)) return false;
      if (!w.add(p + "m2_b", "f32", m2b.data(), m2b.size() * 4, {NE})) return false;
    }
    st.tensors += 2 + (hid ? 1 : 0);
  }
  if (!ink) {
    w.cfg("widths", widths);
  } else {
    w.cfg("kind", "ink_router");
    w.cfg("run", z.meta("run"));
    w.cfg("variant", z.meta("variant"));
    w.cfg("lag", std::to_string(lag));
    w.cfg("apps", std::to_string(targets));
    w.cfg("hidden", std::to_string(H));
    w.cfg("experts", std::to_string(NE));
    w.cfg("hidden_mlp", std::to_string(hid));
    w.cfg("bits", std::to_string(bits));
    w.cfg("emb", "embed_norm_output");
    w.cfg("rms_eps", "1.1920929e-07");
    w.cfg("source", z.meta("source"));
  }
  if (!w.close(err)) return false;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return true;
}

}  // namespace ms
