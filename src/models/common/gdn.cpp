// The Gated DeltaNet's decode fork-joins (gdn.h), shared by qwen3_5_moe and qwen4_exp.
#include "gdn.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace dm {
namespace gdn {

void phase_conv(void* p, int tid, int nt) {
  Ctx& X = *(Ctx*)p;
  const int C = X.d->C(), CK = X.d->CK;
  float* st = X.conv_st;
  const float* w = X.w.conv;
  const int a = (int)((long)C * tid / nt), b = (int)((long)C * (tid + 1) / nt);
  for (int ch = a; ch < b; ch++) {
    X.cv[ch] = conv_channel(st, w + (size_t)ch * CK, X.qkv[ch], C, CK, ch);
  }
}

// one value head through head(), then its output blocks quantised (when the heads feed the output projection directly)
static void ctx_head(Ctx& X, int h) {
  const Dims& d = *X.d;
  float* o = X.o + h * d.DV;
  head(d, X.w, X.S + (size_t)h * d.DV * d.DK, X.cv, X.z + h * d.DV, X.bb[h], X.aa[h], h, o);
  if (X.quant_heads)
    for (int j = 0; j < d.DV / XB; j++) xq_block(*X.xq, (h * d.DV) / XB + j, o + j * XB);
}

void phase_heads(void* p, int, int) {
  Ctx& X = *(Ctx*)p;
  for (int h; (h = X.next.fetch_add(1, std::memory_order_relaxed)) < X.d->NVH;) ctx_head(X, h);
}

// The convolution and the heads in one fork-join: jobs [0, NKH) convolve key head kh's q and k channels (flagged when
// done), then one job per value head convolves its own v channels, waits for its key head's flag and runs the head.
// Every channel's conv_channel call and every head's arithmetic are phase_conv's and phase_heads'.
void phase_conv_heads(void* p, int, int) {
  Ctx& X = *(Ctx*)p;
  const Dims& d = *X.d;
  const int DK = d.DK, DV = d.DV, rep = d.NVH / d.NKH, C = d.C(), CK = d.CK;
  float* cst = X.conv_st;
  const float* cw = X.w.conv;
  for (int job; (job = X.next.fetch_add(1, std::memory_order_relaxed)) < d.NKH + d.NVH + X.o_chunks;) {
    if (job >= d.NKH + d.NVH) {
      while (X.heads_left.load(std::memory_order_acquire) > 0) __builtin_arm_yield();
      const int oc = job - d.NKH - d.NVH;
      const int r0 = oc < X.o_nb ? oc * 32 : X.o_nb * 32 + (oc - X.o_nb) * 8, r1 = std::min(X.wo->rows, r0 + (oc < X.o_nb ? 32 : 8));
      mv(*X.wo, r0, r1, X.xq, nullptr, X.rout + r0);
      continue;
    }
    if (job < d.NKH) {
      for (int part = 0; part < 2; part++) {  // q channels of key head kh, then its k channels
        const int c0 = part * d.NKH * DK + job * DK;
        for (int ch = c0; ch < c0 + DK; ch++) X.cv[ch] = conv_channel(cst, cw + (size_t)ch * CK, X.qkv[ch], C, CK, ch);
      }
      X.qk_done[job].store(1, std::memory_order_release);
      continue;
    }
    const int h = job - d.NKH;
    {
      const int c0 = 2 * d.NKH * DK + h * DV;
      for (int ch = c0; ch < c0 + DV; ch++) X.cv[ch] = conv_channel(cst, cw + (size_t)ch * CK, X.qkv[ch], C, CK, ch);
    }
    while (X.qk_done[h / rep].load(std::memory_order_acquire) == 0) __builtin_arm_yield();
    ctx_head(X, h);
    if (X.o_chunks) X.heads_left.fetch_sub(1, std::memory_order_release);
  }
}

void conv_ro(const Dims& d, const Weights& w, const float* st, const float* qkv, float* cv) {
  const int C = d.C(), CK = d.CK;
  for (int ch = 0; ch < C; ch++) {
    float v = qkv[ch] * w.conv[(size_t)ch * CK + CK - 1];
    for (int k = 0; k < CK - 1; k++) v += st[(size_t)k * C + ch] * w.conv[(size_t)ch * CK + k];
    cv[ch] = silu(v);
  }
}

void head_ro(const Dims& d, const Weights& w, const float* Sh, const float* cv, const float* z, float b, float a, int h, float* oh) {
  const int DK = d.DK, DV = d.DV, kh = h / (d.NVH / d.NKH);
  const float* qs = cv + kh * DK;
  const float* ks = cv + d.NKH * DK + kh * DK;
  const float* v = cv + 2 * d.NKH * DK + h * DV;
  float qq = 0, kk = 0;
  for (int i = 0; i < DK; i++) qq += qs[i] * qs[i], kk += ks[i] * ks[i];
  const float rq = 1.f / std::sqrt(qq + 1e-6f) / std::sqrt((float)DK), rk = 1.f / std::sqrt(kk + 1e-6f);
  std::vector<float> q(DK), k(DK);
  for (int i = 0; i < DK; i++) q[i] = qs[i] * rq, k[i] = ks[i] * rk;
  const float beta = sigm(b), xa = a + w.dt_bias[h];
  const float g = std::exp(-std::exp(w.A_log[h]) * (xa > 20.f ? xa : std::log1p(std::exp(xa))));
  for (int vi = 0; vi < DV; vi++) {
    const float* s = Sh + (size_t)vi * DK;
    float32x4_t acc = vdupq_n_f32(0), gv = vdupq_n_f32(g);
    for (int i = 0; i < DK; i += 4) acc = vfmaq_f32(acc, vmulq_f32(vld1q_f32(s + i), gv), vld1q_f32(k.data() + i));
    const float32x4_t dv = vdupq_n_f32((v[vi] - vaddvq_f32(acc)) * beta);
    float32x4_t oa = vdupq_n_f32(0);
    for (int i = 0; i < DK; i += 4)
      oa = vfmaq_f32(oa, vfmaq_f32(vmulq_f32(vld1q_f32(s + i), gv), vld1q_f32(k.data() + i), dv), vld1q_f32(q.data() + i));
    oh[vi] = vaddvq_f32(oa);
  }
  float ss = 0;
  for (int i = 0; i < DV; i++) ss += oh[i] * oh[i];
  const float r = 1.f / std::sqrt(ss / DV + d.eps);
  for (int i = 0; i < DV; i++) oh[i] = oh[i] * r * w.gnorm[i] * (d.sigmoid_gate ? sigm(z[i]) : silu(z[i]));
}

}  // namespace gdn
}  // namespace dm
