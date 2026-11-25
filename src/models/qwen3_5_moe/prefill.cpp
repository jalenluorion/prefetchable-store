// Batched prefill for Qwen3.6-35B-A3B, layer-major: every layer runs over a whole span of positions (sub-chunks bound
// the scratch), so its weights are read once per sub-chunk and its experts once per span, each expert applied to all the
// positions that picked it. Every per-position computation is the decode path's (qwen3_5_moe.cpp): the same quantised
// activations, the same integer group dots (i8mm here, SDOT in decode: exact either way), the same float epilogues
// (kernels_batch.h), and the shared inline helpers of qwen3_5_moe.h for the rest. So a prefill of n positions leaves the
// same bits (logits, recurrent states, KV) as n single steps; tools/dmserve.cpp --check verifies it.
#include <arm_neon.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "../../core/gpu_gemm.h"
#include "../../core/gpu_split.h"
#include "dmeng/kernels_batch.h"
#include "qwen3_5_moe.h"
#include "../common/gated_attn.h"

namespace dm {
namespace qwen35 {
namespace {

inline double msd(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

// ------------------------------------------------------------------------------------------------ generic phases
template <class F>
struct ForCtx {
  const F* f;
  int n;
  alignas(128) std::atomic<int> next{0};
};
template <class F>
void for_phase(void* p, int tid, int nt) {
  auto& C = *(ForCtx<F>*)p;
  for (int i; (i = C.next.fetch_add(1, std::memory_order_relaxed)) < C.n;) (*C.f)(i);
}
template <class F>
void pfor(Pool& pool, int n, const F& f) {  // f(i) for i in [0, n), dynamic over the pool's threads
  if (n <= 0) return;
  ForCtx<F> C;
  C.f = &f;
  C.n = n;
  pool.run(for_phase<F>, &C);
}

// one product over the sub-chunk's packed inputs (rows of every position) with the GPU's share: the shared run_rows_n_split
// (src/core/gpu_split.h, every family's batched projections): the packed 8- or 4-bit kernels, the GPU the first frac of
// the rows (gemm bits), the same bits either way
struct Proj {
  const Weight* m;
  float* Y;
  size_t ldy;
};
void run_proj(Pool& pool, GpuGemm* gpu, double frac, const XQB& x, const float* xf, size_t ldx, int n, std::initializer_list<Proj> ps,
              GpuSplitAcct* acct) {
  static thread_local std::vector<std::vector<float*>> outs;
  static thread_local std::vector<const float*> fx;
  static thread_local std::vector<JobN> J;
  fx.resize(n);
  for (int i = 0; i < n; i++) fx[i] = xf + (size_t)i * ldx;
  outs.resize(ps.size());
  J.clear();
  size_t k = 0;
  for (const Proj& q : ps) {
    auto& o = outs[k++];
    o.resize(n);
    for (int i = 0; i < n; i++) o[i] = q.Y + (size_t)i * q.ldy;
    J.push_back({q.m, n, nullptr, fx.data(), o.data(), &x});
  }
  run_rows_n_split(pool, gpu, frac, J.data(), (int)J.size(), 64, false, nullptr, acct);
}

struct PrefillCounters {
  double ms_embed = 0, ms_attn_proj = 0, ms_attn_core = 0, ms_attn_out = 0, ms_router = 0, ms_experts = 0, ms_wait = 0,
         ms_cache = 0, ms_head = 0, ms_names = 0, ms_total = 0, ms_conv = 0, ms_heads = 0, ms_full = 0;
  bool gpu = false;  // the GPU split was on (then the JSON also reports "gpu": the split's account)
  GpuSplitAcct ga, ge;  // the split of the attention projections / of the experts (shared + routed: Core::moe_gpu_*)
  double gpu_frac = 0, gpu_frac_experts = 0;
  int gpu_kernel = -1;
  uint64_t positions = 0, chunks = 0, hits = 0, misses = 0, late = 0;  // the experts of the span's layers (Cache::st)
  uint64_t union_batches = 0;  // layer applications whose union ran in batches (Core::n_union_batches; phone sizes)
  void reset() { *this = PrefillCounters(); }
  std::string json() const;
};

}  // namespace

struct Prefill {
  Qwen35* M = nullptr;
  PrefillCounters k;
  // run positions ids[0..n_all) from the state; logits of the last one into M->lgts if want_logits. False: cancel was
  // set (checked after every layer); the state is then back at the span's start (fixed part restored, pos unchanged)
  bool forward_chunk(const int* ids, int n_all, bool want_logits);
  const std::atomic<bool>* cancel = nullptr;
  const PrefillProgress* progress = nullptr;  // (positions done incl. this span's layers so far, total, -)
  int prog_base = 0, prog_total = 0;
  std::vector<uint8_t> snap;                  // the fixed state at the span's start (for a cancel inside the span)
  std::vector<int> capture;  // positions relative to the span start, in (0, n], whose DeltaNet states go to sink
  LM::StateSink sink;
  // scratch (sized for the sub-chunk)
  int cap = 0;
  std::vector<float> X, Hn, Y, A, B, Cv, O, Z, BB, AA, Part, Qn, Cs, Sn;
  std::vector<int32_t> Pk;
  std::vector<int16_t> Rec;
  XQB qa, qo;
  void reserve(int n);
};

namespace {
std::string PrefillCounters::json() const {
  std::ostringstream s;
  s.setf(std::ios::fixed);
  s.precision(3);
  s << "{\"positions\":" << positions << ",\"chunks\":" << chunks << ",\"ms\":" << ms_total << ",\"tok_s\":" << positions / std::max(1e-9, ms_total / 1e3)
    << ",\"ms_by_stage\":{\"embed\":" << ms_embed << ",\"attn_proj\":" << ms_attn_proj << ",\"attn_core\":" << ms_attn_core << ",\"attn_out\":" << ms_attn_out
    << ",\"router\":" << ms_router << ",\"names\":" << ms_names << ",\"experts\":" << ms_experts << ",\"wait\":" << ms_wait << ",\"cache\":" << ms_cache
    << ",\"head\":" << ms_head;
  s << ",\"of_core_conv\":" << ms_conv << ",\"of_core_heads\":" << ms_heads << ",\"of_core_full\":" << ms_full << "}";
  if (gpu) {  // per part: GPU share of the products' multiply-adds, time waiting for the GPU after the CPU's rows, staging
    auto one = [&](const char* name, double f, const GpuSplitAcct& a) {
      s << "\"" << name << "\":{\"frac\":" << f << ",\"gmac_gpu\":" << a.macs_gpu / 1e9 << ",\"gmac_cpu\":" << a.macs_cpu / 1e9 << ",\"wait_ms\":" << a.ms_wait
        << ",\"stage_ms\":" << a.ms_stage << ",\"sets\":" << a.sets << ",\"failed\":" << a.failed << "}";
    };
    s << ",\"gpu\":{\"kernel\":\"" << (gpu_kernel == 1 ? "matrix" : "integer") << "\",";
    one("attn", gpu_frac, ga);
    s << ",";
    one("experts", gpu_frac_experts, ge);
    s << "}";
  }
  s
    << ",\"hits\":" << hits << ",\"late\":" << late << ",\"misses\":" << misses << ",\"union_batches\":" << union_batches << "}";
  return s.str();
}

}  // namespace

void Prefill::reserve(int n) {
  if (n <= cap) return;
  const QCfg& c = M->c;
  const int H = c.H, C = c.conv_dim(), ZD = c.NVH * c.DV, QD = c.NQ * c.HD;
  X.resize((size_t)n * H);
  Hn.resize((size_t)n * H);
  Y.resize((size_t)n * H);
  A.resize((size_t)n * std::max(C, 2 * QD));            // GDN qkv / full-attention q + gate
  B.resize((size_t)n * 2 * c.NKV * c.HD);                // full-attention k, v
  Cv.resize((size_t)n * C);
  O.resize((size_t)n * std::max(ZD, QD));
  Z.resize((size_t)n * ZD);
  BB.resize((size_t)n * c.NVH);
  AA.resize((size_t)n * c.NVH);
  Pk.resize((size_t)n * c.K);
  Qn.resize((size_t)n * QD);
  Cs.resize((size_t)n * 512);
  Sn.resize((size_t)n * 512);
  cap = n;
}

bool Prefill::forward_chunk(const int* ids, int n_all, bool want_logits) {
  Qwen35& Mo = *M;
  const QCfg& c = Mo.c;
  Pool& pool = *Mo.core->pool;
  QState& Sst = *Mo.S;
  const PrefillOpts& o = Mo.po;
  const int H = c.H, NL = c.NL, K = c.K, C = c.conv_dim(), ZD = c.NVH * c.DV, HD = c.HD, QD = c.NQ * HD;
  Core& Co = *Mo.core;
  const int base = Sst.pos, SUB = std::max(8, std::min(o.chunk, n_all));
  if (base + n_all > Sst.ctx) {
    fprintf(stderr, "context full (%d tokens)\n", Sst.ctx);
    abort();
  }
  reserve(SUB);
  if (X.size() < (size_t)n_all * H) X.resize((size_t)n_all * H);  // the residual stream of every position of the span
  if (cancel) {  // a cancel inside the span puts the state back here
    snap.resize(Mo.fixed_bytes(Sst));
    Mo.save_fixed(Sst, snap.data());
  }
  auto t_all = clk::now();
  // embeddings
  auto t0 = clk::now();
  stage_mark(ST_EMBED);
  pfor(pool, n_all, [&](int i) {
    Mo.rows.row(ids[i], X.data() + (size_t)i * H);
  });
  k.ms_embed += msd(t0);
  const int half = c.ROT / 2;
  static const int split_min = getenv("DMENG_SPLIT_MIN") ? atoi(getenv("DMENG_SPLIT_MIN")) : 2 * 256;
  constexpr int CHK = 256;
  GpuGemm* const gpu = (o.gpu_frac > 0 || o.gpu_experts > 0) ? Mo.gpu.get() : nullptr;  // the GPU's share, if on
  const int names = o.names_for(n_all, n_all > SUB || !capture.empty());  // the next layer's early names in this span

  // Layer-major: each layer runs over the whole span (sub-chunks of SUB positions bound the scratch), so a layer's
  // experts are read once per span, not once per chunk.
  // sub-chunk boundaries: every SUB positions, and at each capture position (a layer's state is copied there)
  std::vector<int> bnd;
  for (int p = SUB; p < n_all; p += SUB) bnd.push_back(p);
  for (int p : capture)
    if (p > 0 && p < n_all) bnd.push_back(p);
  bnd.push_back(n_all);
  std::sort(bnd.begin(), bnd.end());
  bnd.erase(std::unique(bnd.begin(), bnd.end()), bnd.end());
  auto captured = [&](int e) { return std::find(capture.begin(), capture.end(), e) != capture.end(); };
  Cache::ChunkClock chunk_clock(Co.cache, Co.tok_clock + n_all - 1);  // the span is one token to the policy
  // the MoE of every layer through Core::moe_chunk (the experts' step of every family's multi-position passes): the span's
  // positions from core.pos, the union of a layer's experts pinned across its sub-chunks while the span runs
  Co.pos = base;
  Co.span_mode = bnd.size() > 1;
  MoePass moe_pass(Co, o);  // the span's MoE settings (waves, packed kernels, down rows: this model's declared values)
  const Times t0c = Co.t;
  const auto st0 = Co.cache.st;
  const double gw0 = Co.moe_gpu_wait_ms, gs0 = Co.moe_gpu_stage_ms, gm0 = Co.moe_gpu_macs;
  const uint64_t gn0 = Co.moe_gpu_sets, gf0 = Co.moe_gpu_failed;
  const long ub0 = Co.n_union_batches;
  auto moe_counts = [&]() {  // the MoE's part of this span: the core's stage times and cache counters, the GPU's account
    k.ms_router += Co.t.router - t0c.router;
    k.ms_cache += Co.t.cache - t0c.cache;
    k.ms_names += Co.t.pf - t0c.pf;
    k.ms_experts += Co.t.experts - t0c.experts;
    k.ms_wait += Co.t.wait - t0c.wait;
    k.hits += Co.cache.st.n_hit - st0.n_hit;
    k.late += Co.cache.st.n_late - st0.n_late;
    k.misses += Co.cache.st.n_miss - st0.n_miss;
    k.ge.ms_wait += Co.moe_gpu_wait_ms - gw0;
    k.ge.ms_stage += Co.moe_gpu_stage_ms - gs0;
    k.ge.macs_gpu += Co.moe_gpu_macs - gm0;
    k.ge.sets += Co.moe_gpu_sets - gn0;
    k.ge.failed += Co.moe_gpu_failed - gf0;
    k.union_batches += Co.n_union_batches - ub0;
  };
  for (int L = 0; L < NL; L++) {
   if (L > 0) {
     if (progress && *progress) (*progress)(prog_base + (int)((long)n_all * L / NL), prog_total, 0.0);
     if (cancel && cancel->load()) {  // stop within one layer: names for this layer released, the fixed state restored
       Co.span_unname(L);
       Co.span_mode = false;
       Mo.load_fixed(Sst, snap.data(), base);
       moe_counts();
       k.ms_total += msd(t_all);
       return false;
     }
   }
   for (size_t bi = 0, s0 = 0; bi < bnd.size(); s0 = bnd[bi++]) {
    const int n = bnd[bi] - (int)s0, pos0 = base + (int)s0;
    float* Xb = X.data() + (size_t)s0 * H;
    // rope tables per position (as attn_full builds them)
    for (int i = 0; i < n; i++)
      for (int j = 0; j < half; j++) {
        double inv = std::pow(c.theta, -2.0 * j / c.ROT);
        float ang = (float)(pos0 + i) * (float)inv;
        Cs[(size_t)i * 512 + j] = std::cos(ang);
        Sn[(size_t)i * 512 + j] = std::sin(ang);
      }
    const Lw& W = Mo.lw[L];
    GpuGemm* const Ga = n >= o.gpu_min && o.gpu_frac > 0 ? gpu : nullptr;           // attention projections
    GpuGemm* const Ge = n >= o.gpu_min && o.gpu_experts > 0 ? gpu : nullptr;   // shared + routed experts
    // ---- attention input
    auto ta = clk::now();
    stage_mark(ST_ATTN_PROJ);
    qa.reserve(H, n);
    pfor(pool, n, [&](int i) {
      rmsnorm(Xb + (size_t)i * H, W.ln_in, H, c.eps, Hn.data() + (size_t)i * H);
      xqb_set(qa, i, Hn.data() + (size_t)i * H);
    });
    xqb_pad(qa);
    if (!W.full) {
      // ---- Gated DeltaNet
      run_proj(pool, Ga, o.gpu_frac, qa, Hn.data(), H, n, {{&W.qkv, A.data(), (size_t)C}, {&W.z, Z.data(), (size_t)ZD}}, &k.ga);
      pfor(pool, n, [&](int i) {
        const float* h = Hn.data() + (size_t)i * H;
        for (int r = 0; r < c.NVH; r++) BB[(size_t)i * c.NVH + r] = dot_bf16((const uint16_t*)W.wb.w + (size_t)r * H, h, H);
        for (int r = 0; r < c.NVH; r++) AA[(size_t)i * c.NVH + r] = dot_bf16((const uint16_t*)W.wa.w + (size_t)r * H, h, H);
      });
      auto tb = clk::now();
      stage_mark(ST_CONV);
      k.ms_attn_proj += std::chrono::duration<double, std::milli>(tb - ta).count();
      // conv: channel by channel, positions in order (phase_conv per position)
      const int CK = c.CK;
      float* st = Sst.ls[L].conv.data();
      const float* w = W.conv;
      const int CB = 256;
      pfor(pool, (C + CB - 1) / CB, [&](int blk) {
        for (int i = 0; i < n; i++)  // positions in order; each channel's own sequence is what phase_conv computes
          for (int ch = blk * CB; ch < std::min(C, blk * CB + CB); ch++)
            Cv[(size_t)i * C + ch] = conv_channel(st, w + (size_t)ch * CK, A[(size_t)i * C + ch], C, CK, ch);
      });
      k.ms_conv += msd(tb);
      auto th0 = clk::now();
      stage_mark(ST_ATTN_CORE);
      // heads: head by head, positions in order (the decode path's gdn::head per position)
      const gdn::Weights gw{W.conv, W.A_log, W.dt_bias, W.gnorm};
      pfor(pool, c.NVH, [&](int h) {
        float* S = Sst.ls[L].S.data() + (size_t)h * c.DV * c.DK;
        for (int i = 0; i < n; i++)
          gdn::head(Mo.gd, gw, S, Cv.data() + (size_t)i * C, Z.data() + (size_t)i * ZD + h * c.DV, BB[(size_t)i * c.NVH + h],
                    AA[(size_t)i * c.NVH + h], h, O.data() + (size_t)i * ZD + h * c.DV);
      });
      k.ms_heads += msd(th0);
      qo.reserve(ZD, n);
      pfor(pool, n, [&](int i) { xqb_set(qo, i, O.data() + (size_t)i * ZD); });
      xqb_pad(qo);
      auto tc = clk::now();
      stage_mark(ST_ATTN_OUT);
      k.ms_attn_core += std::chrono::duration<double, std::milli>(tc - tb).count();
      run_proj(pool, Ga, o.gpu_frac, qo, O.data(), ZD, n, {{&W.o, Y.data(), (size_t)H}}, &k.ga);
      k.ms_attn_out += msd(tc);
      if (sink && captured(bnd[bi])) {
        const LState& ls = Sst.ls[L];
        sink(base + bnd[bi], Mo.fixed_off[L], ls.conv.data(), ls.conv.size() * 4);
        sink(base + bnd[bi], Mo.fixed_off[L] + ls.conv.size() * 4, ls.S.data(), ls.S.size() * 4);
      }
    } else {
      // ---- gated full attention
      const int KD = c.NKV * HD;
      float* Kp = B.data();
      float* Vp = B.data() + (size_t)n * KD;
      run_proj(pool, Ga, o.gpu_frac, qa, Hn.data(), H, n, {{&W.q, A.data(), (size_t)2 * QD}, {&W.k, Kp, (size_t)KD}, {&W.v, Vp, (size_t)KD}}, &k.ga);
      auto tb = clk::now();
      stage_mark(ST_ATTN_CORE);
      k.ms_attn_proj += std::chrono::duration<double, std::milli>(tb - ta).count();
      LState& S = Sst.ls[L];
      // K (normed, rotated) and V into the cache at their positions (attn_full)
      pfor(pool, n, [&](int i) {
        const int pos = pos0 + i;
        const float* kp = Kp + (size_t)i * KD;
        const float* vp = Vp + (size_t)i * KD;
        for (int j = 0; j < c.NKV; j++) {
          float kk[1024], ss = 0;
          for (int t = 0; t < HD; t++) {
            kk[t] = kp[j * HD + t];
            ss += kk[t] * kk[t];
          }
          float r = 1.f / std::sqrt(ss / HD + c.eps);
          for (int t = 0; t < HD; t++) kk[t] = kk[t] * r * W.kn[t];
          {
            const float* cs = Cs.data() + (size_t)i * 512;
            const float* sn = Sn.data() + (size_t)i * 512;
            for (int t = 0; t < half; t++) {
              float a = kk[t], b = kk[t + half];
              kk[t] = a * cs[t] - b * sn[t];
              kk[t + half] = b * cs[t] + a * sn[t];
            }
          }
          _Float16* kd = S.K + ((size_t)j * Sst.ctx + pos) * HD;
          _Float16* vd = S.V + ((size_t)j * Sst.ctx + pos) * HD;
          for (int t = 0; t < HD; t++) {
            kd[t] = (_Float16)kk[t];
            vd[t] = (_Float16)vp[j * HD + t];
          }
        }
      });
      // queries: q_norm + rope, as both decode paths build them
      pfor(pool, n * c.NQ, [&](int jh) {
        const int i = jh / c.NQ, h = jh % c.NQ;
        const float* qs = A.data() + (size_t)i * 2 * QD + (size_t)h * 2 * HD;
        float ss = 0;
        for (int t = 0; t < HD; t++) ss += qs[t] * qs[t];
        float r = 1.f / std::sqrt(ss / HD + c.eps);
        float* q = Qn.data() + (size_t)i * QD + (size_t)h * HD;
        for (int t = 0; t < HD; t++) q[t] = qs[t] * r * W.qn[t];
        const float* cs = Cs.data() + (size_t)i * 512;
        const float* sn = Sn.data() + (size_t)i * 512;
        for (int t = 0; t < half; t++) {
          float a = q[t], b = q[t + half];
          q[t] = a * cs[t] - b * sn[t];
          q[t + half] = b * cs[t] + a * sn[t];
        }
      });
      const float scale = 1.f / std::sqrt((float)HD);
      const int rep = c.NQ / c.NKV;
      // short contexts (T <= split_min): one head per job, phase_full_heads
      int n_short = 0;
      while (n_short < n && pos0 + n_short + 1 <= split_min) n_short++;
      pfor(pool, n_short * c.NQ, [&](int jh) {  // the decode path's head (src/models/common/gated_attn.h)
        const int i = jh / c.NQ, h = jh % c.NQ, T = pos0 + i + 1;
        static thread_local std::vector<float> sc;
        _Float16 q16[1024];
        const float* q = Qn.data() + (size_t)i * QD + (size_t)h * HD;
        for (int t = 0; t < HD; t++) q16[t] = (_Float16)q[t];
        const int kvh = h / rep;
        gated_attn::head_f16(q16, S.K + (size_t)kvh * Sst.ctx * HD, S.V + (size_t)kvh * Sst.ctx * HD, HD, T, [](int t) { return t; },
                             A.data() + (size_t)i * 2 * QD + (size_t)h * 2 * HD + HD, O.data() + (size_t)i * QD + (size_t)h * HD, sc);
      });
      // long contexts: split-K per query (phase_split + the combination), queries in blocks to bound the partials
      const int QB = 32;
      for (int qb0 = n_short; qb0 < n; qb0 += QB) {
        const int qb1 = std::min(n, qb0 + QB);
        const int maxT = pos0 + qb1, maxch = (maxT + CHK - 1) / CHK, W2 = HD + 2;
        Part.resize((size_t)QB * maxch * c.NQ * W2);
        // one (query, chunk, kv head): phase_split's arithmetic for that query
        auto one = [&](int i, int ch, int kvh) {
          const int T = pos0 + i + 1;
          static thread_local std::vector<_Float16> q16v(8 * 256);
          static thread_local std::vector<float> scv(8 * 256);
          _Float16* q16 = q16v.data();
          float* sc = scv.data();
          const float* qn = Qn.data() + (size_t)i * QD;
          for (int g = 0; g < rep; g++)
            for (int u = 0; u < HD; u++) q16[(size_t)g * HD + u] = (_Float16)qn[(size_t)(kvh * rep + g) * HD + u];
          int t0 = ch * CHK, t1 = std::min(T, t0 + CHK), nn = t1 - t0;
          const _Float16* Kc = S.K + ((size_t)kvh * Sst.ctx + t0) * HD;
          const _Float16* Vc = S.V + ((size_t)kvh * Sst.ctx + t0) * HD;
          // scores: each (key, head) dot is phase_split's (two FMLAL accumulators over the dimensions in order, then
          // their sum); four heads run side by side so their chains overlap and the key row is loaded once for them
          for (int t = 0; t < nn; t++) {
            const float16_t* kr = (const float16_t*)(Kc + (size_t)t * HD);
            int g = 0;
            for (; g + 4 <= rep; g += 4) {
              const float16_t* q0 = (const float16_t*)q16 + (size_t)g * HD;
              float32x4_t a0 = vdupq_n_f32(0), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
              for (int u = 0; u < HD; u += 8) {
                float16x8_t kv = vld1q_f16(kr + u);
                float16x8_t x0 = vld1q_f16(q0 + u), x1 = vld1q_f16(q0 + HD + u), x2 = vld1q_f16(q0 + 2 * HD + u), x3 = vld1q_f16(q0 + 3 * HD + u);
                a0 = vfmlalq_low_f16(a0, kv, x0); b0 = vfmlalq_high_f16(b0, kv, x0);
                a1 = vfmlalq_low_f16(a1, kv, x1); b1 = vfmlalq_high_f16(b1, kv, x1);
                a2 = vfmlalq_low_f16(a2, kv, x2); b2 = vfmlalq_high_f16(b2, kv, x2);
                a3 = vfmlalq_low_f16(a3, kv, x3); b3 = vfmlalq_high_f16(b3, kv, x3);
              }
              sc[(size_t)(g + 0) * CHK + t] = vaddvq_f32(vaddq_f32(a0, b0)) * scale;
              sc[(size_t)(g + 1) * CHK + t] = vaddvq_f32(vaddq_f32(a1, b1)) * scale;
              sc[(size_t)(g + 2) * CHK + t] = vaddvq_f32(vaddq_f32(a2, b2)) * scale;
              sc[(size_t)(g + 3) * CHK + t] = vaddvq_f32(vaddq_f32(a3, b3)) * scale;
            }
            for (; g < rep; g++) {
              const float16_t* qq = (const float16_t*)q16 + (size_t)g * HD;
              float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
              for (int u = 0; u < HD; u += 8) {
                float16x8_t kv = vld1q_f16(kr + u), qv = vld1q_f16(qq + u);
                s0 = vfmlalq_low_f16(s0, kv, qv);
                s1 = vfmlalq_high_f16(s1, kv, qv);
              }
              sc[(size_t)g * CHK + t] = vaddvq_f32(vaddq_f32(s0, s1)) * scale;
            }
          }
          float* part = Part.data() + (size_t)(i - qb0) * maxch * c.NQ * W2;
          for (int g = 0; g < rep; g++) {
            float* sg = sc + (size_t)g * CHK;
            float mx = -INFINITY;
            for (int t = 0; t < nn; t++) mx = std::max(mx, sg[t]);
            double z = 0;
            for (int t = 0; t < nn; t++) z += (sg[t] = std::exp(sg[t] - mx));
            float* out = part + ((size_t)ch * c.NQ + kvh * rep + g) * W2;
            out[0] = mx;
            out[1] = (float)z;
          }
          // values: each output element is the same fused multiply-add sequence over t as phase_split's (starting from
          // 0), kept in registers per 8 dimensions x 8 heads instead of in memory; the weights transposed to [t][head]
          if (rep == 8) {
            static thread_local std::vector<float> wtv(256 * 8);
            float* wt = wtv.data();
            for (int t = 0; t < nn; t++)
              for (int g = 0; g < 8; g++) wt[t * 8 + g] = sc[(size_t)g * CHK + t];
            for (int u = 0; u < HD; u += 8) {
              float32x4_t a0[8], a1[8];
              for (int g = 0; g < 8; g++) a0[g] = a1[g] = vdupq_n_f32(0);
              for (int t = 0; t < nn; t++) {
                float16x8_t vv = vld1q_f16((const float16_t*)(Vc + (size_t)t * HD) + u);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(vv)), hi = vcvt_high_f32_f16(vv);
                float32x4_t w0 = vld1q_f32(wt + t * 8), w1 = vld1q_f32(wt + t * 8 + 4);
                a0[0] = vfmaq_laneq_f32(a0[0], lo, w0, 0); a1[0] = vfmaq_laneq_f32(a1[0], hi, w0, 0);
                a0[1] = vfmaq_laneq_f32(a0[1], lo, w0, 1); a1[1] = vfmaq_laneq_f32(a1[1], hi, w0, 1);
                a0[2] = vfmaq_laneq_f32(a0[2], lo, w0, 2); a1[2] = vfmaq_laneq_f32(a1[2], hi, w0, 2);
                a0[3] = vfmaq_laneq_f32(a0[3], lo, w0, 3); a1[3] = vfmaq_laneq_f32(a1[3], hi, w0, 3);
                a0[4] = vfmaq_laneq_f32(a0[4], lo, w1, 0); a1[4] = vfmaq_laneq_f32(a1[4], hi, w1, 0);
                a0[5] = vfmaq_laneq_f32(a0[5], lo, w1, 1); a1[5] = vfmaq_laneq_f32(a1[5], hi, w1, 1);
                a0[6] = vfmaq_laneq_f32(a0[6], lo, w1, 2); a1[6] = vfmaq_laneq_f32(a1[6], hi, w1, 2);
                a0[7] = vfmaq_laneq_f32(a0[7], lo, w1, 3); a1[7] = vfmaq_laneq_f32(a1[7], hi, w1, 3);
              }
              for (int g = 0; g < 8; g++) {
                float* out = part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2;
                vst1q_f32(out + u, a0[g]);
                vst1q_f32(out + u + 4, a1[g]);
              }
            }
          } else {
            for (int g = 0; g < rep; g++) std::fill(part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2, part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2 + HD, 0.f);
            for (int t = 0; t < nn; t++) {
              const float16_t* vr = (const float16_t*)(Vc + (size_t)t * HD);
              for (int u = 0; u < HD; u += 8) {
                float16x8_t vv = vld1q_f16(vr + u);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(vv)), hi = vcvt_high_f32_f16(vv);
                for (int g = 0; g < rep; g++) {
                  float* out = part + ((size_t)ch * c.NQ + kvh * rep + g) * W2 + 2;
                  float w2 = sc[(size_t)g * CHK + t];
                  vst1q_f32(out + u, vfmaq_n_f32(vld1q_f32(out + u), lo, w2));
                  vst1q_f32(out + u + 4, vfmaq_n_f32(vld1q_f32(out + u + 4), hi, w2));
                }
              }
            }
          }
        };
        // jobs: (chunk, kv head, group of queries); a chunk's keys and values stay in cache for the whole group
        const int nq = qb1 - qb0, cj = maxch * c.NKV;
        const int groups = std::max(1, std::min(nq, (2 * pool.size() + cj - 1) / cj));
        pfor(pool, cj * groups, [&](int jb) {
          const int grp = jb / cj, ch = (jb % cj) / c.NKV, kvh = jb % c.NKV;
          const int i0 = qb0 + nq * grp / groups, i1 = qb0 + nq * (grp + 1) / groups;
          for (int i = i0; i < i1; i++)
            if (ch * CHK < pos0 + i + 1) one(i, ch, kvh);
        });
        pfor(pool, (qb1 - qb0) * c.NQ, [&](int jh) {
          const int i = qb0 + jh / c.NQ, h = jh % c.NQ, T = pos0 + i + 1, nch = (T + CHK - 1) / CHK;
          const float* part = Part.data() + (size_t)(i - qb0) * maxch * c.NQ * W2;
          float mx = -INFINITY;
          for (int ch = 0; ch < nch; ch++) mx = std::max(mx, part[((size_t)ch * c.NQ + h) * W2]);
          std::vector<double> acc(HD, 0.0);
          double z = 0;
          for (int ch = 0; ch < nch; ch++) {
            const float* pp = part + ((size_t)ch * c.NQ + h) * W2;
            double f = std::exp((double)pp[0] - mx);
            z += f * pp[1];
            for (int u = 0; u < HD; u++) acc[u] += f * pp[2 + u];
          }
          float* oh = O.data() + (size_t)i * QD + (size_t)h * HD;
          const float* gate = A.data() + (size_t)i * 2 * QD + (size_t)h * 2 * HD + HD;
          for (int u = 0; u < HD; u++) oh[u] = (float)(acc[u] / z) * sigm(gate[u]);
        });
      }
      k.ms_full += msd(tb);
      qo.reserve(QD, n);
      pfor(pool, n, [&](int i) { xqb_set(qo, i, O.data() + (size_t)i * QD); });
      xqb_pad(qo);
      auto tc = clk::now();
      stage_mark(ST_ATTN_OUT);
      k.ms_attn_core += std::chrono::duration<double, std::milli>(tc - tb).count();
      run_proj(pool, Ga, o.gpu_frac, qo, O.data(), QD, n, {{&W.o, Y.data(), (size_t)H}}, &k.ga);
      k.ms_attn_out += msd(tc);
    }
    // ---- residual, MoE input
    stage_mark(ST_MOE_ROUTER);
    pfor(pool, n, [&](int i) {
      float* x = Xb + (size_t)i * H;
      const float* y = Y.data() + (size_t)i * H;
      for (int j = 0; j < H; j++) x[j] += y[j];
      rmsnorm(x, W.ln_post, H, c.eps, Hn.data() + (size_t)i * H);
    });
    // ---- the MoE: Core::moe_chunk (the released router per position, the union read once, every expert and the shared
    // expert computed for the positions that picked it, the fixed-order combination; the single step's bits), its
    // experts in waves as they land, and at the first sub-chunk the next layer's experts named behind this layer's own
    // reads (prefill_stream: 1 every expert of it, 2 its released router's picks on this layer's MoE inputs)
    std::vector<const float*> xm(n);
    std::vector<float*> ym(n);
    for (int i = 0; i < n; i++) xm[i] = Hn.data() + (size_t)i * H, ym[i] = Y.data() + (size_t)i * H;
    if (bi == 0 && names && L + 1 < NL) {
      Co.span_next = L + 1;
      Co.span_next_router = names == 2;
    }
    Co.moe_gpu = Ge;
    Co.moe_gpu_frac = o.gpu_experts;
    Co.moe_chunk(L, n, xm.data(), ym.data(), Mo.views[L], Mo.px.record ? Pk.data() : nullptr, nullptr, (int)s0);
    Co.moe_gpu = nullptr;
    auto tcmb = clk::now();
    stage_mark(ST_MOE_COMBINE);
    pfor(pool, n, [&](int i) {
      float* x = Xb + (size_t)i * H;
      const float* y = Y.data() + (size_t)i * H;
      for (int r = 0; r < H; r++) x[r] += y[r];
    });
    k.ms_experts += msd(tcmb);
    if (Mo.px.record) {
      if (Rec.size() < (size_t)n_all * NL * K) Rec.resize((size_t)n_all * NL * K);
      for (int i = 0; i < n; i++)
        for (int j = 0; j < K; j++) Rec[((size_t)((int)s0 + i) * NL + L) * K + j] = (int16_t)Pk[(size_t)i * K + j];
    }
   }
   Co.span_unname(L);  // the layer's union pins and its names
  }
  Co.span_mode = false;
  moe_counts();
  if (Mo.px.record) fwrite(Rec.data(), 2, (size_t)n_all * NL * K, Mo.px.record);
  // ---- head on the last position
  if (want_logits) {
    auto th = clk::now();
    stage_mark(ST_HEAD);
    rmsnorm(X.data() + (size_t)(n_all - 1) * H, Mo.fnorm, H, c.eps, Mo.h.data());
    Mo.xin.prep(Mo.h.data(), H);
    const int V = c.V, RB = 64;
    pfor(pool, (V + RB - 1) / RB, [&](int b) {
      int r0 = b * RB, r1 = std::min(V, r0 + RB);
      mv(Mo.head, r0, r1, &Mo.xin.q, nullptr, Mo.lgts.data() + r0);
    });
    k.ms_head += msd(th);
  }
  Sst.pos += n_all;
  Co.tok_clock += n_all;
  k.positions += n_all;
  k.chunks++;
  stage_mark(ST_OTHER);
  k.ms_total += msd(t_all);
  if (progress && *progress) (*progress)(prog_base + n_all, prog_total, 0.0);
  return true;
}


// ------------------------------------------------------------------------------------------------ the model's entry
Qwen35::Qwen35() = default;
Qwen35::~Qwen35() = default;

ModelMem Qwen35::mem(const State& s0) const {
  const QState& s = (const QState&)s0;
  ModelMem m;
  m.resident = R.size;
  for (const LState& st : s.ls) {
    m.kv_reserved += 2 * st.kv_bytes;
    m.recurrent += (st.conv.size() + st.S.size()) * sizeof(float);
  }
  if (P) {
    auto add = [&](const auto& v) { m.prefill_scratch += v.capacity() * sizeof(v[0]); };
    add(P->X); add(P->Hn); add(P->Y); add(P->A); add(P->B); add(P->Cv); add(P->O); add(P->Z); add(P->BB); add(P->AA);
    m.prefill_scratch += gpu ? gpu->buffer_bytes() : 0;  // the GPU split's staging and output buffers
    add(P->Part); add(P->Qn); add(P->Cs); add(P->Sn); add(P->Pk); add(P->Rec);
    m.prefill_xqb = P->qa.cap + P->qo.cap;
  }
  return m;
}

bool Qwen35::prefill(State& s0, const int32_t* ids, int n, bool want_logits, const std::vector<int>& caps, const LM::StateSink& sink,
                     PrefillStats* st, const std::atomic<bool>* cancel) {
  S = (QState*)&s0;
  greedy_ = -1;  // the logits (if wanted) come from the full head
  // one position per pass: the decode path (the batched kernels are I16, and take inputs in multiples of 256)
  const bool dims_ok = c.H % 256 == 0 && c.I % 256 == 0 && c.SI % 256 == 0 && (c.NVH * c.DV) % 256 == 0 && (c.NQ * c.HD) % 256 == 0;
  const bool tbt = po.chunk <= 1 || prec != Prec::I16 || !dims_ok;
  if (tbt && !px.batched_always) return Model::prefill(s0, ids, n, want_logits, caps, sink, st, cancel);
  if (!P) {
    P.reset(new Prefill());
    P->M = this;
  }
  P->k.reset();
  P->k.gpu = (po.gpu_frac > 0 || po.gpu_experts > 0) && gpu;
  P->k.gpu_frac = po.gpu_frac;
  if (gpu) gpu->set_kernel(po.gpu_kernel);
  P->k.gpu_kernel = gpu ? gpu->kernel() : -1;
  P->k.gpu_frac_experts = po.gpu_experts;
  // per-layer caps would refuse the next layer's early names (names never borrow past a cap): off for the pass
  Cache::CapsOff caps_off(core->cache);
  PrefillOpts o = po;
  if (tbt) o.span = 1;  // batched_always with the decode path's conditions: the batched code one position per pass
  const bool ok = prefill_spans(*core, n, caps, cancel, o, [&](int done, int m, bool last, const std::vector<int>& cap_rel) {
    P->capture = cap_rel;
    P->sink = sink;
    P->cancel = cancel;
    P->progress = st ? &st->progress : nullptr;
    P->prog_base = done;
    P->prog_total = n;
    const bool whole = P->forward_chunk((const int*)ids + done, m, last && want_logits);
    P->capture.clear();
    P->sink = nullptr;
    return whole;
  }, st, nullptr);
  if (st) {
    st->chunks = (int)P->k.chunks;
    st->json = P->k.json();
  }
  if (po.release) {  // the scratch back to the system (re-created by the next prefill)
    P.reset();
    if (gpu) gpu->trim();
  }
  return ok;
}

}  // namespace qwen35
}  // namespace dm
