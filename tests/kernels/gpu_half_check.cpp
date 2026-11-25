// The fast prompt path's half-precision GPU products (src/core/gpu_half.h) against an fp64 reference of the same
// dequantised weights, on random data (no model), and their throughput. Not bit-exact by design: the check is the
// error relative to the outputs' size (fp16 inputs and weights, fp32 accumulation: about 1e-3), and the MoE block's
// combination order.
//   gpu_half_check [reps]
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "../../src/core/gpu_half.h"
#include "../../src/models/common/gdn.h"

using namespace dm;
using hc = std::chrono::steady_clock;

static uint8_t* region(size_t n) {
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  return p == MAP_FAILED ? nullptr : (uint8_t*)p;
}

struct Mat {
  Weight w;
  size_t bytes = 0;
};

// a random affine matrix (bits 8 or 4) at p; returns the bytes used (rounded to 256)
static size_t make(uint8_t* p, int bits, int R, int K, std::mt19937& rng, Weight& w) {
  const size_t cb = (size_t)R * (bits == 8 ? K : K / 2), sb = (size_t)R * (K / 64) * 2;
  std::uniform_int_distribution<int> code(0, 255);
  for (size_t i = 0; i < cb; i++) p[i] = (uint8_t)code(rng);
  _Float16* s = (_Float16*)(p + cb);
  _Float16* b = (_Float16*)(p + cb + sb);
  std::normal_distribution<float> nd(0.f, 1.f);
  const float q = bits == 8 ? 255.f : 15.f;
  for (size_t i = 0; i < (size_t)R * (K / 64); i++) {
    const float sc = 0.04f / q * (1.f + 0.3f * std::fabs(nd(rng)));
    s[i] = (_Float16)sc;
    b[i] = (_Float16)(-0.5f * q * sc + 0.002f * nd(rng));
  }
  w.kind = bits == 8 ? Weight::AFF8 : Weight::AFF4;
  w.rows = R;
  w.cols = K;
  w.c = p;
  w.s = s;
  w.b = b;
  return (cb + 2 * sb + 255) & ~size_t(255);
}

static double wv(const Weight& w, int r, int k) {
  const int NG = w.cols / 64;
  const double s = (double)((const _Float16*)w.s)[(size_t)r * NG + k / 64], b = (double)((const _Float16*)w.b)[(size_t)r * NG + k / 64];
  int c;
  if (w.kind == Weight::AFF8) c = w.c[(size_t)r * w.cols + k];
  else {
    const uint8_t by = w.c[(size_t)r * (w.cols / 2) + k / 2];
    c = (k & 1) ? by >> 4 : by & 15;
  }
  return c * s + b;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 5;
  GpuHalf g;
  if (!g.init()) {
    printf("{\"gpu\":false}\n");
    return 0;
  }
  Pool pool(8);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.f, 1.f);
  const size_t REG = (size_t)1600 << 20;
  uint8_t* reg = region(REG);
  if (!reg || !g.wrap(reg, REG)) return printf("wrap failed\n"), 1;
  size_t off = 0;
  int fail = 0;
  // ---- gemm sets: (bits, R, K, n)
  struct Case {
    int bits, R, K, n;
  } cases[] = {{8, 10240, 2560, 256}, {8, 2560, 6144, 256}, {8, 320, 10240, 256}, {8, 10240, 320, 256}, {4, 640, 2560, 37}, {8, 1024, 2560, 9}, {8, 2560, 640, 300}};
  for (const Case& c : cases) {
    Weight w;
    off += make(reg + off, c.bits, c.R, c.K, rng, w);
    std::vector<float> x((size_t)c.n * c.K);
    for (float& v : x) v = nd(rng);
    std::vector<const float*> xr(c.n);
    for (int i = 0; i < c.n; i++) xr[i] = x.data() + (size_t)i * c.K;
    g.begin();
    const int t = g.add(w, c.R, g.input(xr.data(), c.n, c.K), c.n);
    if (t < 0 || !g.run(pool)) return printf("gemm not run\n"), 1;
    const int ld = c.R;
    std::vector<float> yv((size_t)c.n * c.R);
    for (int q = 0; q < c.n; q++) g.row(t, q, yv.data() + (size_t)q * c.R);
    const float* y = yv.data();
    double emax = 0, rms = 0;
    int checked = 0;
    for (int q = 0; q < c.n; q += std::max(1, c.n / 7))
      for (int r = 0; r < c.R; r += std::max(1, c.R / 97)) {
        double ref = 0;
        for (int k = 0; k < c.K; k++) ref += wv(w, r, k) * (double)(_Float16)x[(size_t)q * c.K + k];
        emax = std::max(emax, std::fabs(ref - y[(size_t)q * ld + r]));
        rms += ref * ref;
        checked++;
      }
    rms = std::sqrt(rms / checked);
    // throughput: the same set again
    double gms = 1e30, ms = 1e30;  // the fastest of reps (a shared machine)
    for (int i = 0; i < reps; i++) {
      const double g0 = g.ms_gpu;
      auto t0 = hc::now();
      g.begin();
      g.add(w, c.R, g.input(xr.data(), c.n, c.K), c.n);
      g.run(pool);
      ms = std::min(ms, std::chrono::duration<double, std::milli>(hc::now() - t0).count());
      gms = std::min(gms, g.ms_gpu - g0);
    }
    const bool ok = emax / rms < 3e-3;
    fail += !ok;
    printf("{\"gemm\":[%d,%d,%d,%d],\"err_rel\":%.2e,\"ok\":%s,\"ms\":%.3f,\"tflops\":%.2f,\"gpu_ms\":%.3f,\"tflops_gpu\":%.2f}\n", c.bits, c.R, c.K, c.n,
           emax / rms, ok ? "true" : "false", ms, 2.0 * c.R * c.K * c.n / ms / 1e9, gms, 2.0 * c.R * c.K * c.n / gms / 1e9);
  }
  // ---- a MoE block: P positions, NE experts (4-bit), K picks each, the shared expert (8-bit)
  {
    const int P = 2048, H = 2560, I = 640, NE = 256, K = 10;
    std::vector<Weight> gw(NE), uw(NE), dw(NE);
    for (int e = 0; e < NE; e++) {
      off += make(reg + off, 4, I, H, rng, gw[e]);
      off += make(reg + off, 4, I, H, rng, uw[e]);
      off += make(reg + off, 4, H, I, rng, dw[e]);
    }
    Weight sg, su, sd;
    off += make(reg + off, 8, I, H, rng, sg);
    off += make(reg + off, 8, I, H, rng, su);
    off += make(reg + off, 8, H, I, rng, sd);
    if (off > REG) return printf("region too small\n"), 1;
    std::vector<float> x((size_t)P * H), wts((size_t)P * K), shw(P), y((size_t)P * H);
    for (float& v : x) v = nd(rng);
    std::vector<int> pk((size_t)P * K);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::vector<GpuHalf::Exp> ex(NE);
    for (int e = 0; e < NE; e++) ex[e].gate = &gw[e], ex[e].up = &uw[e], ex[e].down = &dw[e];
    for (int i = 0; i < P; i++) {
      std::vector<int> all(NE);
      for (int e = 0; e < NE; e++) all[e] = e;
      for (int j = 0; j < K; j++) {  // skewed picks
        int a = j + (int)(std::pow(u01(rng), 2.0f) * (NE - j));
        std::swap(all[j], all[a]);
        pk[(size_t)i * K + j] = all[j];
        wts[(size_t)i * K + j] = 0.1f * u01(rng);
        ex[all[j]].pos.push_back(i);
        ex[all[j]].row.push_back(i * K + j);
      }
      shw[i] = u01(rng);
    }
    std::vector<const float*> xr(P);
    std::vector<float*> yr(P);
    for (int i = 0; i < P; i++) xr[i] = x.data() + (size_t)i * H, yr[i] = y.data() + (size_t)i * H;
    if (!g.moe(pool, xr.data(), P, H, K, I, ex, wts.data(), &sg, &su, &sd, I, shw.data(), yr.data())) return printf("moe not run\n"), 1;
    // reference for a few positions
    double emax = 0, rms = 0;
    int checked = 0;
    auto mlp = [&](const Weight& G, const Weight& U, const Weight& D, const float* xi, std::vector<double>& out) {
      std::vector<double> act(I);
      for (int r = 0; r < I; r++) {
        double a = 0, b = 0;
        for (int k = 0; k < H; k++) a += wv(G, r, k) * (double)(_Float16)xi[k], b += wv(U, r, k) * (double)(_Float16)xi[k];
        act[r] = (double)(_Float16)(a / (1 + std::exp(-a)) * b);
      }
      out.assign(H, 0);
      for (int r = 0; r < H; r += 37) {
        double s = 0;
        for (int k = 0; k < I; k++) s += wv(D, r, k) * act[k];
        out[r] = s;
      }
    };
    for (int i = 0; i < P; i += 401) {
      std::vector<double> ref(H, 0), o;
      for (int j = 0; j < K; j++) {
        const int e = pk[(size_t)i * K + j];
        mlp(gw[e], uw[e], dw[e], xr[i], o);
        for (int r = 0; r < H; r += 37) ref[r] += wts[(size_t)i * K + j] * o[r];
      }
      mlp(sg, su, sd, xr[i], o);
      for (int r = 0; r < H; r += 37) {
        ref[r] += shw[i] * o[r];
        emax = std::max(emax, std::fabs(ref[r] - y[(size_t)i * H + r]));
        rms += ref[r] * ref[r];
        checked++;
      }
    }
    rms = std::sqrt(rms / checked);
    // the same block again: identical bits (fixed combination order)
    std::vector<float> y1 = y;
    double ms = 1e30, gms = 1e30;
    for (int r = 0; r < reps; r++) {
      const double g0 = g.ms_gpu;
      auto t0 = hc::now();
      g.moe(pool, xr.data(), P, H, K, I, ex, wts.data(), &sg, &su, &sd, I, shw.data(), yr.data());
      ms = std::min(ms, std::chrono::duration<double, std::milli>(hc::now() - t0).count());
      gms = std::min(gms, g.ms_gpu - g0);
    }
    const bool same = std::memcmp(y1.data(), y.data(), y.size() * 4) == 0, ok = emax / rms < 5e-3 && same;
    fail += !ok;
    const double fl = 2.0 * P * (K + 1) * 3.0 * I * H;
    printf("{\"moe\":[%d,%d,%d,%d,%d],\"err_rel\":%.2e,\"repeat_identical\":%s,\"ok\":%s,\"ms\":%.2f,\"tflops\":%.2f,\"gpu_ms\":%.1f,\"tflops_gpu\":%.2f}\n", P, NE, K,
           H, I, emax / rms, same ? "true" : "false", ok ? "true" : "false", ms, fl / ms / 1e9, gms, fl / gms / 1e9);
  }
  // ---- attention over listed keys: 64 positions, 2 key heads x 12 query heads, D 256, dense and selected lists
  {
    const int ctx = 4096, D = 256, NKV = 2, NQ = 24, n = 64;
    const size_t kvb = (size_t)NKV * ctx * D * 2;
    _Float16* Kc = (_Float16*)region(kvb);
    _Float16* Vc = (_Float16*)region(kvb);
    for (size_t i = 0; i < kvb / 2; i++) Kc[i] = (_Float16)nd(rng), Vc[i] = (_Float16)nd(rng);
    std::vector<std::vector<int>> sel(n);
    std::vector<const std::vector<int>*> sp(n);
    std::vector<float> q((size_t)n * NQ * D), o((size_t)n * NQ * D);
    std::vector<const float*> qp(n);
    std::vector<float*> op(n);
    std::uniform_int_distribution<int> pick(0, 3000);
    for (int i = 0; i < n; i++) {
      const int pos = 2000 + i * 13;
      if (i % 2 == 0)
        for (int j = 0; j <= pos; j++) sel[i].push_back(j);  // dense
      else {
        for (int j = 0; j < 777; j++) sel[i].push_back(pick(rng));
        std::sort(sel[i].begin(), sel[i].end());
      }
      sp[i] = &sel[i];
      for (int k = 0; k < NQ * D; k++) q[(size_t)i * NQ * D + k] = 0.5f * nd(rng);
      qp[i] = q.data() + (size_t)i * NQ * D, op[i] = o.data() + (size_t)i * NQ * D;
    }
    const bool ran = g.attn(pool, Kc, Vc, ctx, D, NQ, NKV, n, qp.data(), sp.data(), op.data());
    double emax = 0, rms = 0;
    int checked = 0;
    for (int i = 0; ran && i < n; i += 7)
      for (int h = 0; h < NQ; h += 5) {
        const int kv = h / (NQ / NKV);
        std::vector<double> sc(sel[i].size());
        double mx = -1e300, z = 0;
        for (size_t t = 0; t < sel[i].size(); t++) {
          double a = 0;
          for (int d = 0; d < D; d++) a += (double)(_Float16)q[((size_t)i * NQ + h) * D + d] * (double)Kc[((size_t)kv * ctx + sel[i][t]) * D + d];
          sc[t] = a / std::sqrt((double)D);
          mx = std::max(mx, sc[t]);
        }
        for (auto& v : sc) z += (v = std::exp(v - mx));
        for (int d = 0; d < D; d += 17) {
          double a = 0;
          for (size_t t = 0; t < sel[i].size(); t++) a += sc[t] / z * (double)Vc[((size_t)kv * ctx + sel[i][t]) * D + d];
          emax = std::max(emax, std::fabs(a - o[((size_t)i * NQ + h) * D + d]));
          rms += a * a;
          checked++;
        }
      }
    rms = std::sqrt(rms / std::max(1, checked));
    double ms = 1e30;
    for (int r = 0; r < reps; r++) {
      auto t0 = hc::now();
      g.attn(pool, Kc, Vc, ctx, D, NQ, NKV, n, qp.data(), sp.data(), op.data());
      ms = std::min(ms, std::chrono::duration<double, std::milli>(hc::now() - t0).count());
    }
    const bool ok = ran && emax / rms < 2e-2;
    fail += !ok;
    printf("{\"attn\":[%d,%d,%d,%d],\"ran\":%s,\"err_rel\":%.2e,\"ok\":%s,\"ms\":%.2f}\n", n, NQ, NKV, D, ran ? "true" : "false", emax / rms, ok ? "true" : "false", ms);
  }
  // ---- the Gated DeltaNet's heads over 96 positions against gdn::head_io (Flash-Next's sizes)
  {
    gdn::Dims d;
    d.NKH = 16, d.NVH = 48, d.DK = 128, d.DV = 128, d.CK = 4, d.eps = 1e-6f, d.sigmoid_gate = true;
    const int n = 96, C = d.C(), ZD = d.NVH * d.DV, ldp = C + ZD + 2 * d.NVH;
    std::vector<float> A(d.NVH), dt(d.NVH), gn(d.DV), cv((size_t)n * C), pj((size_t)n * ldp), S0((size_t)d.NVH * d.DV * d.DK);
    for (auto& v : A) v = -1.f + 0.3f * nd(rng);
    for (auto& v : dt) v = 0.5f * nd(rng);
    for (auto& v : gn) v = 1.f + 0.1f * nd(rng);
    for (auto& v : cv) v = nd(rng);
    for (auto& v : pj) v = nd(rng);
    for (auto& v : S0) v = 0.05f * nd(rng);
    gdn::Weights W;
    W.A_log = A.data(), W.dt_bias = dt.data(), W.gnorm = gn.data();
    std::vector<float> Sc = S0, Sg = S0, oc((size_t)n * ZD), og((size_t)n * ZD);
    for (int t = 0; t < n; t++)
      for (int h = 0; h < d.NVH; h++) {
        const float* pr = pj.data() + (size_t)t * ldp;
        gdn::head(d, W, Sc.data() + (size_t)h * d.DV * d.DK, cv.data() + (size_t)t * C, pr + C + h * d.DV, pr[C + ZD + h], pr[C + ZD + d.NVH + h], h,
                  oc.data() + (size_t)t * ZD + h * d.DV);
      }
    double ms = 1e30;
    bool ran = false;
    for (int r = 0; r < std::max(1, reps); r++) {
      Sg = S0;
      auto t0 = hc::now();
      ran = g.gdn(pool, n, d.NKH, d.NVH, d.DK, d.DV, Sg.data(), cv.data(), C, pj.data(), ldp, C, C + ZD, C + ZD + d.NVH, og.data(), ZD, A.data(), dt.data(),
                  gn.data(), true, d.eps);
      ms = std::min(ms, std::chrono::duration<double, std::milli>(hc::now() - t0).count());
    }
    double eo = 0, ro = 0, es = 0, rs = 0;
    for (size_t i = 0; i < oc.size(); i++) eo = std::max(eo, (double)std::fabs(oc[i] - og[i])), ro += (double)oc[i] * oc[i];
    for (size_t i = 0; i < Sc.size(); i++) es = std::max(es, (double)std::fabs(Sc[i] - Sg[i])), rs += (double)Sc[i] * Sc[i];
    ro = std::sqrt(ro / oc.size()), rs = std::sqrt(rs / Sc.size());
    const bool ok = ran && eo / ro < 1e-3 && es / rs < 1e-3;
    fail += !ok;
    printf("{\"gdn\":[%d,%d,%d],\"ran\":%s,\"out_err_rel\":%.2e,\"state_err_rel\":%.2e,\"ok\":%s,\"ms\":%.2f}\n", n, d.NVH, d.DK, ran ? "true" : "false",
           eo / ro, es / rs, ok ? "true" : "false", ms);
  }
  printf("{\"pass\":%s}\n", fail ? "false" : "true");
  return fail ? 1 : 0;
}
