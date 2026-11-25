// The Gated DeltaNet of the Qwen3.5 line, shared by the families that have it (qwen3_5_moe: Qwen3.6-35B-A3B; qwen4_exp:
// Qwen3.8-Flash-Next): one position's causal convolution channel and one value head (l2-normalised q and k, the gated
// delta rule on the fp32 state, the gated RMSNorm), and the decode fork-joins built on them. Every path of every family
// (decode, verify, batched prefill) calls these functions, so a layer computes the same bits wherever it runs.
//
// The families differ in the output gate only: silu(z) (Qwen3.6) or sigmoid(z) (Flash-Next, output_gate_type).
#pragma once
#include <arm_neon.h>

#include <atomic>
#include <cmath>
#include <string>
#include <vector>

#include "dmeng/weights.h"

namespace dm {
namespace gdn {

struct Dims {
  int NKH = 0, NVH = 0, DK = 0, DV = 0, CK = 0;
  float eps = 1e-6f;
  bool sigmoid_gate = false;  // the gated norm's gate: sigmoid(z) (Flash-Next) or silu(z) (Qwen3.6)
  int C() const { return 2 * NKH * DK + NVH * DV; }
  // the head's buffers are on the stack (head_io): DK, DV at most 256, DK a multiple of 4 (the NEON loops)
  bool ok(std::string& err) const {
    if (DK > 0 && DK <= 256 && DV > 0 && DV <= 256 && DK % 4 == 0 && NKH > 0 && NVH % NKH == 0) return true;
    err = "Gated DeltaNet head sizes outside what the engine supports (key dim <= 256 and a multiple of 4, value dim <= 256)";
    return false;
  }
};

struct Weights {  // one layer's: conv [C, CK], A_log / dt_bias [NVH], the gated norm's weight [DV] (as released)
  const float *conv = nullptr, *A_log = nullptr, *dt_bias = nullptr, *gnorm = nullptr;
};

// one channel of the causal convolution for one position: the taps over the state (oldest first) as a fused
// multiply-add chain, the newest input's product rounded on its own and added; then the state shifts by one. _io: the
// state read from sin and the shifted state written to sout (another buffer, or sin itself)
inline float conv_channel_io(const float* sin, float* sout, const float* wc, float x, int C, int CK, int ch) {
  float v = 0;
  for (int k = 0; k < CK - 1; k++) v = __builtin_fmaf(sin[(size_t)k * C + ch], wc[k], v);
  v = add_unfused(v, x, wc[CK - 1]);
  for (int k = 0; k + 1 < CK - 1; k++) sout[(size_t)k * C + ch] = sin[(size_t)(k + 1) * C + ch];
  sout[(size_t)(CK - 2) * C + ch] = x;
  return silu(v);
}
inline float conv_channel(float* st, const float* wc, float x, int C, int CK, int ch) { return conv_channel_io(st, st, wc, x, C, CK, ch); }

// One value head h of one position. cv: the position's convolved q, k, v [C]; z: the head's gate inputs [DV]; b, a: the
// head's beta and alpha inputs (before their activations); Sin: the head's state [DV, DK]; Sout: where the updated state
// goes (Sin itself, or another buffer: the verify pass); o: the head's output [DV] (null: the state only, no output).
// q = q / |q| / sqrt(DK), k = k / |k| (eps 1e-6 under the root: mlx-vlm's rms_norm form); the state decays by
// g = exp(-exp(A_log) softplus(a + dt_bias)), then S += k (v - S k) beta, y = S q; the gated RMSNorm over the head.
inline void head_io(const Dims& d, const Weights& w, const float* Sin, float* Sout, const float* cv, const float* z, float b, float a, int h,
                    float* o) {
  const int DK = d.DK, DV = d.DV, rep = d.NVH / d.NKH;
  const int kh = h / rep;
  float q[256], k[256], ob[256];  // DK, DV <= 256 (checked at load)
  float* ov = o ? o : ob;
  const float* qs = cv + kh * DK;
  const float* ks = cv + d.NKH * DK + kh * DK;
  const float* v = cv + 2 * d.NKH * DK + h * DV;
  float qq = 0, kk = 0;
  for (int i = 0; i < DK; i++) {
    qq += qs[i] * qs[i];
    kk += ks[i] * ks[i];
  }
  const float rq = 1.f / std::sqrt(qq + 1e-6f) / std::sqrt((float)DK), rk = 1.f / std::sqrt(kk + 1e-6f);
  for (int i = 0; i < DK; i++) {
    q[i] = qs[i] * rq;
    k[i] = ks[i] * rk;
  }
  const float beta = sigm(b);
  const float xa = a + w.dt_bias[h];
  const float sp = xa > 20.f ? xa : std::log1p(std::exp(xa));
  const float g = std::exp(-std::exp(w.A_log[h]) * sp);
  for (int vi = 0; vi < DV; vi++) {
    const float* si = Sin + (size_t)vi * DK;
    float* s = Sout + (size_t)vi * DK;
    float32x4_t acc = vdupq_n_f32(0), gv = vdupq_n_f32(g);
    for (int i = 0; i < DK; i += 4) {
      float32x4_t sv = vmulq_f32(vld1q_f32(si + i), gv);
      vst1q_f32(s + i, sv);
      acc = vfmaq_f32(acc, sv, vld1q_f32(k + i));
    }
    const float delta = (v[vi] - vaddvq_f32(acc)) * beta;
    float32x4_t dv = vdupq_n_f32(delta), oa = vdupq_n_f32(0);
    for (int i = 0; i < DK; i += 4) {
      float32x4_t sv = vfmaq_f32(vld1q_f32(s + i), vld1q_f32(k + i), dv);
      vst1q_f32(s + i, sv);
      oa = vfmaq_f32(oa, sv, vld1q_f32(q + i));
    }
    ov[vi] = vaddvq_f32(oa);
  }
  if (!o) return;
  // gated RMSNorm over the head (weight as released), times the gate
  float ss = 0;
  for (int i = 0; i < DV; i++) ss += o[i] * o[i];
  const float r = 1.f / std::sqrt(ss / DV + d.eps);
  if (d.sigmoid_gate)
    for (int i = 0; i < DV; i++) o[i] = o[i] * r * w.gnorm[i] * sigm(z[i]);
  else
    for (int i = 0; i < DV; i++) o[i] = o[i] * r * w.gnorm[i] * silu(z[i]);
}
inline void head(const Dims& d, const Weights& w, float* St, const float* cv, const float* z, float b, float a, int h, float* o) {
  head_io(d, w, St, St, cv, z, b, a, h, o);
}

// The same head evaluated without writing anything (run-ahead: it only decides reads, so its arithmetic need not be the
// real pass's): the convolution from a read-only state, S_new = g S + k (x) delta, its output.
void conv_ro(const Dims& d, const Weights& w, const float* st, const float* qkv, float* cv);
void head_ro(const Dims& d, const Weights& w, const float* St, const float* cv, const float* z, float b, float a, int h, float* o);

// One position of a layer in the pool: the convolution (phase_conv) and the heads (phase_heads), or both in one fork-join
// with the output projection after the heads (phase_conv_heads, decode).
struct Ctx {
  const Dims* d = nullptr;
  Weights w;
  float* conv_st = nullptr;  // [CK - 1, C], oldest first
  float* S = nullptr;        // [NVH, DV, DK]
  float *qkv = nullptr, *z = nullptr, *bb = nullptr, *aa = nullptr, *cv = nullptr, *o = nullptr;
  XQ* xq = nullptr;          // quant_heads: each head quantises its own output blocks into it (the output projection's input)
  bool quant_heads = false;
  std::atomic<int> qk_done[64];  // phase_conv_heads: the conv of key head kh's q and k channels is done
  // phase_conv_heads: the output projection after the heads in the same fork-join (row chunks of 32 once every head is
  // done; the heads quantise their own outputs), its rows as run_rows computes them
  const Weight* wo = nullptr;
  float* rout = nullptr;
  int o_chunks = 0, o_nb = 0;  // claims of the output projection: o_nb of 32 rows, then 8-row claims (a short tail)
  std::atomic<int> heads_left{0};
  std::atomic<int64_t> heads_done_ns{0};  // steady-clock ns when the last head finished (with o_chunks: the stage split)
  alignas(128) std::atomic<int> next{0};
};
void phase_conv(void* p, int tid, int nt);
void phase_heads(void* p, int, int);
void phase_conv_heads(void* p, int, int);  // the decode path: both in one fork-join

}  // namespace gdn
}  // namespace dm
