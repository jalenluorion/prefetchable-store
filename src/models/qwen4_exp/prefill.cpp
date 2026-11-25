// Qwen3.8-Flash-Next's batched prefill: layer-major over a span of positions. Each layer runs over the whole span (in
// sub-chunks that bound the scratch), so its resident matrices are read once per sub-chunk (the packed i8mm gemm from 6
// positions, else the multi-vector kernels: the one-vector kernels' bits) and its experts once per span (Core::moe_chunk:
// each expert computed for every position that picked it, the same bits as single steps), with the GPU's share of the
// projections and of the experts when asked. Everything else is the decode path's own per-position arithmetic
// (qwen4_exp.cpp): the hyper-connections, the PLE and its convolution (channel blocks in parallel, positions in order),
// the Gated DeltaNet's convolution and heads (positions in order), QSA's keys, pooled blocks, selection and attention.
// The n-gram rows of the whole span are issued before its first layer. The state and the last logits equal those of n
// single steps, bit for bit (tests/prefill_check.sh, tests/qwen4_exp/tiny_check.py).
#include <algorithm>
#include <cstring>
#include <sstream>

#include "../../core/gpu_gemm.h"
#include "qwen4_exp.h"

namespace dm {
namespace qwen4 {

struct Flash::Span {
  int n_cap = 0, sub_cap = 0;
  std::vector<float> hyp;  // [n][N] the span's streams
  // per position of a sub-chunk
  std::vector<float> nrm, lo, up, r, mix, y, inj1, inj2, proj, cv, att, e, key, val, gated, normed, pout;
  std::vector<XBuf> xa, xb, xc;
  XQB qa, qc;
  std::vector<std::vector<int>> sel;
  std::vector<int32_t> hist;  // [n + 2] the n-gram history: the two ids before the span, then the span's ids
  std::shared_ptr<RowStore::Batch> rows;
  std::vector<uint8_t> snap;
  size_t bytes() const {
    size_t b = 4 * (hyp.capacity() + nrm.capacity() + lo.capacity() + up.capacity() + r.capacity() + mix.capacity() + y.capacity() + inj1.capacity() +
                    inj2.capacity() + proj.capacity() + cv.capacity() + att.capacity() + e.capacity() + key.capacity() + val.capacity() +
                    gated.capacity() + normed.capacity() + pout.capacity());
    for (auto* v : {&xa, &xb, &xc})
      for (auto& x : *v) b += xq_bytes(x.cap);
    return b + qa.cap + qc.cap;
  }
};

// A verify pass (Flash::verify): the span's own layer-major arithmetic over n <= TMAX positions, with what a rollback needs.
// The Gated DeltaNet: the committed state (A, the model state) is only read; the pass writes the state after its
// positions into B (the convolution channel by channel, the heads head by head, each over the positions in order, with the
// decode path's functions: gdn::conv_channel_io / head_io with the input and output buffers apart). commit(n) swaps A and
// B (no copy); commit(a < n) rebuilds B from A for positions 0..a-1 (the conv rows are the kept projections, the heads run
// again on the kept convolved inputs, state only) and swaps. The PLE's convolution state (small) is copied at the pass's
// start and moved again by the kept positions' inputs. QSA's keys, values and pooled blocks are append-only: a
// rolled-back position's entries are written again when it runs again. Every kept position's state bits and logits are a
// single step's.
struct Flash::Ver {
  int n = 0, pos0 = 0;
  bool live = false;
  Want want = Want::LAST;
  std::vector<std::vector<float>> altc, alts;  // [NL] Gated DeltaNet layers: B (conv state, recurrent state)
  std::vector<std::vector<float>> gp, cvp;     // [NL] Gated DeltaNet layers: [TMAX][PW] projections, [TMAX][C] convolved
  std::vector<float> pconv0, ple_in;           // the PLE's conv state at the pass's start; its inputs [TMAX][N]
  int32_t hist[TMAX + 2] = {0};                // the n-gram history: the two ids before the pass, then the pass's ids
  std::vector<float> embs, fx, lg;             // [TMAX][H] embeddings, [TMAX][H] final mixer outputs, [TMAX][V] logits
  std::vector<XBuf> xh;
  XQB qh;
  long steps = 0, positions = 0, rollbacks = 0;
  double ms_rollback = 0;
};

size_t Flash::span_bytes() const { return sp ? sp->bytes() : 0; }

// bytes of a sub-chunk's scratch per position (Span's per-position buffers and input buffers, as span_run sizes them)
double Flash::sub_bytes_per_pos() const {
  const double N = c.N(), H = c.H, PW = std::max((size_t)c.conv_dim() + c.NVH * c.DV + 2 * c.NVH,
                                                    (size_t)2 * c.NQ * c.HD + 2 * c.NKV * c.HD + (c.IXH + 1) * c.IXD + 2 * c.NQ * c.HD);
  const double floats = 6 * N + 3 * H + c.LOW + 3 * 64 + PW + c.conv_dim() + std::max(c.NVH * c.DV, c.NQ * c.HD) + c.ple_dim;
  const int ka = std::max(c.N(), c.H), kc = std::max({c.ple_dim, c.NVH * c.DV, c.NQ * c.HD});
  return 4 * floats + xq_bytes(ka) + xq_bytes(c.LOW) + xq_bytes(kc) + 2.0 * 4 * ka;
}

static size_t proj_width(const FCfg& c) {
  return std::max((size_t)c.conv_dim() + c.NVH * c.DV + 2 * c.NVH,
                  (size_t)2 * c.NQ * c.HD + 2 * c.NKV * c.HD + (c.IXH + 1) * c.IXD + 2 * c.NQ * c.HD);
}

bool Flash::span_run(const int32_t* ids, int n, bool want_last, const std::vector<int>* capture, const LM::StateSink* sink,
                     const std::atomic<bool>* cancel, const PrefillProgress* progress, int prog_base, int prog_total, Ver* ver) {
  const int H = c.H, N = c.N(), NL = c.NL, sub = ver ? n : std::max(1, std::min(po.chunk, n));
  Pool& pool = *core->pool;
  if (!sp) sp = std::make_shared<Span>();
  Span& P = *sp;
  // sub-chunk boundaries: every `sub` positions and at each capture position (a layer's state is delivered there)
  std::vector<int> bnd;
  for (int p = sub; p < n; p += sub) bnd.push_back(p);
  if (capture)
    for (int p : *capture)
      if (p > 0 && p < n) bnd.push_back(p);
  bnd.push_back(n);
  std::sort(bnd.begin(), bnd.end());
  bnd.erase(std::unique(bnd.begin(), bnd.end()), bnd.end());
  int wmax = 0;
  for (size_t b = 0, s0 = 0; b < bnd.size(); s0 = bnd[b++]) wmax = std::max(wmax, bnd[b] - (int)s0);
  const bool multi = bnd.size() > 1;
  auto captured = [&](int e) { return sink && *sink && capture && std::find(capture->begin(), capture->end(), e) != capture->end(); };
  // buffers
  if (P.n_cap < n) {
    P.hyp.resize((size_t)n * N);
    P.n_cap = n;
  }
  if (P.sub_cap < wmax) {
    const size_t w = wmax, PW = proj_width(c);
    P.nrm.resize(w * N);
    P.lo.resize(w * c.LOW);
    P.up.resize(w * N);
    P.r.resize(w * 64);
    P.mix.resize(w * H);
    P.y.resize(w * H);
    P.inj1.resize(w * 64);
    P.inj2.resize(w * 64);
    P.proj.resize(w * PW);
    P.cv.resize(w * c.conv_dim());
    P.att.resize(w * std::max(c.NVH * c.DV, c.NQ * c.HD));
    P.e.resize(w * c.ple_dim);
    P.key.resize(w * N);
    P.val.resize(w * H);
    P.gated.resize(w * N);
    P.normed.resize(w * N);
    P.pout.resize(w * N);
    // each per-position input buffer at the widest input it takes: xa the mixers' norms (N) and the mixers' outputs (H),
    // xb the mixers' low rank (LOW), xc the PLE rows and the attention outputs
    const int ka = std::max(N, H), kc = std::max({c.ple_dim, c.NVH * c.DV, c.NQ * c.HD});
    while ((int)P.xa.size() < wmax) {
      P.xa.emplace_back();
      P.xa.back().init(ka, prec);
      P.xb.emplace_back();
      P.xb.back().init(c.LOW, prec);
      P.xc.emplace_back();
      P.xc.back().init(kc, prec);
    }
    P.sel.resize(wmax);
    P.sub_cap = wmax;
  }
  // the n-gram rows of every position, read while the first layer computes
  P.hist.resize(n + 2);
  P.hist[0] = S->ng_ctx[0];
  P.hist[1] = S->ng_ctx[1];
  for (int i = 0; i < n; i++) P.hist[i + 2] = ids[i];
  {
    const int nh = c.heads();
    std::vector<int64_t> rws((size_t)n * nh);
    for (int i = 0; i < n; i++) ng.rows(P.hist.data() + i, ids[i], rws.data() + (size_t)i * nh);
    P.rows = ngs.issue(rws.data(), n * nh);
  }
  auto t0 = clk::now();
  stage_mark(ST_EMBED);
  {
    std::vector<float> e(H);
    for (int i = 0; i < n; i++) {
      rows.row(ids[i], e.data());
      for (int s = 0; s < c.HC; s++) std::memcpy(P.hyp.data() + (size_t)i * N + (size_t)s * H, e.data(), sizeof(float) * H);
      if (i + 1 == n) std::copy(e.begin(), e.end(), emb.begin());  // the routers' `emb` of the last position (as a step leaves it)
      if (ver) std::copy(e.begin(), e.end(), ver->embs.begin() + (size_t)i * H);
    }
  }
  if (ver) {  // what a rollback needs: the PLE's conv state and the n-gram history at the pass's start
    ver->pconv0 = S->pconv;
    for (int i = 0; i < n + 2; i++) ver->hist[i] = P.hist[i];
  }
  T->embed += ms_since(t0);
  if (cancel && multi) {  // a cancel inside the span puts the fixed state back here
    P.snap.resize(fixed_bytes(*S));
    save_fixed(*S, P.snap.data());
  }
  core->pos = S->pos;
  Cache::CapsOff caps_off(core->cache);  // a span names the next layer's union: caps off for the pass
  if (ver) {  // names inside a verify pass (the routers read the embeddings now, before the layers)
    const float* es[TMAX];
    for (int i = 0; i < n; i++) es[i] = ver->embs.data() + (size_t)i * H;
    core->chunk_start(n, ids, es);
  }
  core->span_mode = multi;
  Cache::ChunkClock span_clock(core->cache, core->tok_clock + n - 1);  // the whole span is one token to the policy
  const int names = po.names_for(n, multi);  // the next layer's early names in this span (prefill.h)
  MoePass moe_pass(*core, po);
  const bool packed_ok = !ref && prec == Prec::I16;
  GpuGemm* const G0 = gpu && po.gpu_frac > 0 ? gpu.get() : nullptr;
  if (gpu) gpu->set_kernel(po.gpu_kernel);
  const size_t PW = proj_width(c);
  // the n-gram history's slice of the fixed blob at position p (relative): ids p - 2, p - 1 of the history
  auto sink_ng = [&](int p) { (*sink)(S->pos + p, fixed_ng, P.hist.data() + p, 8); };
  for (int L = 0; L < NL; L++) {
    const Lw& W = lw[L];
    if (core->cache.budget_req.load(std::memory_order_relaxed) >= 0) core->cache.apply_budget();
    for (size_t bi = 0, s0 = 0; bi < bnd.size(); s0 = bnd[bi++]) {
      const int i0 = (int)s0, m = bnd[bi] - i0, pos0 = S->pos + i0;
      GpuGemm* const G = G0 && m >= po.gpu_min ? G0 : nullptr;
      const bool packed = packed_ok && m >= (ver ? 2 : 6);  // (the same bits either way; a verify pass packs from 2, as the 35B's)
      auto hp = [&](int i) { return P.hyp.data() + (size_t)(i0 + i) * N; };
      // inputs into per-position XBufs (and the packed batch): x[i] (K values) -> xs[i], qb
      auto quant = [&](std::vector<XBuf>& xs, XQB* qb, const float* base, size_t stride, int K) {
        if (packed) qb->reserve(K, m);
        parallel_for(pool, m, [&](int i) {
          xs[i].reset(K);
          if (!ref) xs[i].prep(base + (size_t)i * stride, K);
          if (packed) xqb_set_from(*qb, i, xs[i].q);
        });
        if (packed) xqb_pad(*qb);
      };
      std::vector<const XQ*> xq(m);
      std::vector<const float*> xf(m);
      std::vector<float*> o1(m), o2(m), o3(m), o4(m);
      auto inputs = [&](std::vector<XBuf>& xs, const float* base, size_t stride) {
        for (int i = 0; i < m; i++) xq[i] = &xs[i].q, xf[i] = base + (size_t)i * stride;
      };
      // ---- a GatedResidual over the sub-chunk: mixed [m][H] into mixo, inj [m][64] into injo (null: the final mixer)
      auto hc_chunk = [&](const HCw& w, float* mixo, float* injo) {
        const auto th = clk::now();
        parallel_for(pool, m, [&](int i) { hc_norm(w, hp(i), P.nrm.data() + (size_t)i * N); });
        quant(P.xa, &P.qa, P.nrm.data(), N, N);
        inputs(P.xa, P.nrm.data(), N);
        for (int i = 0; i < m; i++) o1[i] = P.lo.data() + (size_t)i * c.LOW, o2[i] = P.r.data() + (size_t)i * 64;
        JobN jb[2] = {{&w.down, m, xq.data(), xf.data(), o1.data(), packed ? &P.qa : nullptr}, {&w.inj, m, nullptr, xf.data(), o2.data(), nullptr}};
        run_rows_n_split(pool, G, po.gpu_frac, jb, injo ? 2 : 1, 16, ref, &T->bytes_res, &gacct);
        parallel_for(pool, m, [&](int i) { hc_lo(P.lo.data() + (size_t)i * c.LOW); });
        quant(P.xb, &P.qc, P.lo.data(), c.LOW, c.LOW);
        inputs(P.xb, P.lo.data(), c.LOW);
        for (int i = 0; i < m; i++) o1[i] = P.up.data() + (size_t)i * N;
        JobN ju = {&w.up, m, xq.data(), xf.data(), o1.data(), packed ? &P.qc : nullptr};
        run_rows_n_split(pool, G, po.gpu_frac, &ju, 1, 32, ref, &T->bytes_res, &gacct);
        parallel_for(pool, m, [&](int i) {
          hc_mix(P.nrm.data() + (size_t)i * N, P.up.data() + (size_t)i * N, injo ? P.r.data() + (size_t)i * 64 : nullptr, mixo + (size_t)i * H,
                 injo ? injo + (size_t)i * 64 : nullptr);
        });
        T->msg += ms_since(th);  // the hyper-connection mixers: their own stage, as in decode
      };
      auto inject = [&](const float* yv, const float* ij) {  // h_s += inj_s * y, per position
        const auto tj = clk::now();
        parallel_for(pool, m, [&](int i) {
          float* h = hp(i);
          const float* yy = yv + (size_t)i * H;
          const float* jj = ij + (size_t)i * 64;
          for (int s = 0; s < c.HC; s++)
            for (int k = 0; k < H; k++) h[(size_t)s * H + k] = __builtin_fmaf(yy[k], jj[s], h[(size_t)s * H + k]);
        });
        T->misc += ms_since(tj);
      };
      // ---- the PLE
      if (W.ple) {
        auto tp = clk::now();
        ngs.wait(*P.rows);
        const int nh = c.heads();
        parallel_for(pool, m, [&](int i) {
          for (int h = 0; h < nh; h++)
            ng_row(P.rows->data.data() + ((size_t)(i0 + i) * nh + h) * c.ng_bpr, P.e.data() + (size_t)i * c.ple_dim + (size_t)h * c.ng_dim);
        });
        quant(P.xc, &P.qa, P.e.data(), c.ple_dim, c.ple_dim);
        inputs(P.xc, P.e.data(), c.ple_dim);
        for (int i = 0; i < m; i++) o1[i] = P.key.data() + (size_t)i * N, o2[i] = P.val.data() + (size_t)i * H;
        JobN jb[2] = {{&W.pk, m, xq.data(), xf.data(), o1.data(), packed ? &P.qa : nullptr}, {&W.pv, m, xq.data(), xf.data(), o2.data(), packed ? &P.qa : nullptr}};
        run_rows_n_split(pool, G, po.gpu_frac, jb, 2, 32, ref, &T->bytes_res, &gacct);
        parallel_for(pool, m, [&](int i) {
          ple_gate(W, P.key.data() + (size_t)i * N, P.val.data() + (size_t)i * H, hp(i), P.gated.data() + (size_t)i * N, P.normed.data() + (size_t)i * N);
        });
        float* st = S->pconv.data();
        const int CB = 256;
        parallel_for(pool, (N + CB - 1) / CB, [&](int b) {  // channel blocks; each channel's positions in order
          for (int i = 0; i < m; i++)
            for (int ch = b * CB; ch < std::min(N, b * CB + CB); ch++)
              P.pout[(size_t)i * N + ch] = P.gated[(size_t)i * N + ch] + ple_conv(W, st, P.normed.data() + (size_t)i * N, ch, true);
        });
        parallel_for(pool, m, [&](int i) {
          float* h = hp(i);
          const float* po_ = P.pout.data() + (size_t)i * N;
          for (int k = 0; k < N; k++) h[k] = h[k] + po_[k];
        });
        if (ver) ver->ple_in.assign(P.normed.begin(), P.normed.begin() + (size_t)m * N);  // the conv's inputs (a rollback moves the state by them)
        T->misc += ms_since(tp);
        if (captured(bnd[bi])) (*sink)(S->pos + bnd[bi], fixed_ple, S->pconv.data(), S->pconv.size() * 4);
      }
      // ---- the mixer
      stage_mark(ST_NORMQ);
      hc_chunk(W.ahc, P.mix.data(), P.inj1.data());
      auto ta = clk::now();
      stage_mark(ST_ATTN_PROJ);
      quant(P.xa, &P.qa, P.mix.data(), H, H);
      inputs(P.xa, P.mix.data(), H);
      if (W.lin) {
        const int C = c.conv_dim(), ZD = c.NVH * c.DV;
        // a verify pass keeps its projections and convolved inputs per layer (a rollback replays the state from them)
        float* const PJ = ver ? ver->gp[L].data() : P.proj.data();
        float* const CV = ver ? ver->cvp[L].data() : P.cv.data();
        for (int i = 0; i < m; i++) {
          float* pr = PJ + (size_t)i * PW;
          o1[i] = pr, o2[i] = pr + C, o3[i] = pr + C + ZD, o4[i] = pr + C + ZD + c.NVH;
        }
        const XQB* pa = packed ? &P.qa : nullptr;
        JobN jb[4] = {{&W.qkv, m, xq.data(), xf.data(), o1.data(), pa}, {&W.z, m, xq.data(), xf.data(), o2.data(), pa},
                      {&W.wb, m, nullptr, xf.data(), o3.data(), nullptr}, {&W.wa, m, nullptr, xf.data(), o4.data(), nullptr}};
        run_rows_n_split(pool, G, po.gpu_frac, jb, 4, 32, ref, &T->bytes_res, &gacct);
        auto tb = clk::now();
        T->attn_proj += ms_between(ta, tb);
        stage_mark(ST_ATTN_CORE);
        // the state read from (cin, Sin) for the first position and written to (cout, Sout): the model state itself, or, in a
        // verify pass, the committed state read only and the second buffers written
        float* cst = S->ls[L].conv.data();
        const float* cin = cst;
        float* cout = ver ? ver->altc[L].data() : cst;
        const int CK = c.CK, CB = 256;
        parallel_for(pool, (C + CB - 1) / CB, [&](int b) {  // the convolution: channel blocks, positions in order
          for (int i = 0; i < m; i++) {
            const float* qkv = PJ + (size_t)i * PW;
            float* cv = CV + (size_t)i * C;
            for (int ch = b * CB; ch < std::min(C, b * CB + CB); ch++)
              cv[ch] = gdn::conv_channel_io(i == 0 ? cin : cout, cout, W.g.conv + (size_t)ch * CK, qkv[ch], C, CK, ch);
          }
        });
        const float* Sin = S->ls[L].S.data();
        float* Sst = ver ? ver->alts[L].data() : S->ls[L].S.data();
        parallel_for(pool, c.NVH, [&](int h) {  // the heads: head by head, positions in order (gdn::head_io, as decode)
          const size_t hb = (size_t)h * c.DV * c.DK;
          for (int i = 0; i < m; i++) {
            const float* pr = PJ + (size_t)i * PW;
            gdn::head_io(gd, W.g, (i == 0 ? Sin : Sst) + hb, Sst + hb, CV + (size_t)i * C, pr + C + h * c.DV, pr[C + ZD + h], pr[C + ZD + c.NVH + h], h,
                         P.att.data() + (size_t)i * ZD + h * c.DV);
          }
        });
        auto tc = clk::now();
        T->attn_core += ms_between(tb, tc);
        stage_mark(ST_ATTN_OUT);
        quant(P.xc, &P.qc, P.att.data(), ZD, ZD);
        inputs(P.xc, P.att.data(), ZD);
        for (int i = 0; i < m; i++) o1[i] = P.y.data() + (size_t)i * H;
        JobN jo = {&W.o, m, xq.data(), xf.data(), o1.data(), packed ? &P.qc : nullptr};
        run_rows_n_split(pool, G, po.gpu_frac, &jo, 1, 32, ref, &T->bytes_res, &gacct);
        T->attn_out += ms_since(tc);
        if (captured(bnd[bi])) {
          const LState& ls = S->ls[L];
          (*sink)(S->pos + bnd[bi], fixed_off[L], ls.conv.data(), ls.conv.size() * 4);
          (*sink)(S->pos + bnd[bi], fixed_off[L] + ls.conv.size() * 4, ls.S.data(), ls.S.size() * 4);
        }
      } else {
        const int HD = c.HD, QW = 2 * c.NQ * HD, KW = c.NKV * HD;
        for (int i = 0; i < m; i++) {
          float* pr = P.proj.data() + (size_t)i * PW;
          o1[i] = pr, o2[i] = pr + QW, o3[i] = pr + QW + KW, o4[i] = pr + QW + 2 * KW;
        }
        const XQB* pa = packed ? &P.qa : nullptr;
        JobN jb[4] = {{&W.q, m, xq.data(), xf.data(), o1.data(), pa}, {&W.k, m, xq.data(), xf.data(), o2.data(), pa},
                      {&W.v, m, xq.data(), xf.data(), o3.data(), pa}, {&W.ix, m, xq.data(), xf.data(), o4.data(), pa}};
        run_rows_n_split(pool, G, po.gpu_frac, jb, 4, 32, ref, &T->bytes_res, &gacct);
        auto tb = clk::now();
        T->attn_proj += ms_between(ta, tb);
        stage_mark(ST_ATTN_CORE);
        if (pos0 + m > S->ctx) {
          fprintf(stderr, "context full (%d tokens)\n", S->ctx);
          abort();
        }
        const size_t IXW = (size_t)(c.IXH + 1) * c.IXD;
        auto qv = [&](int i) { return P.proj.data() + (size_t)i * PW + QW + 2 * KW + IXW; };  // q [NQ HD] then gate [NQ HD]
        parallel_for(pool, m, [&](int i) {  // keys, values, indexer keys and queries of every position
          float* pr = P.proj.data() + (size_t)i * PW;
          qsa_kv(L, pos0 + i, pr, pr + QW, pr + QW + KW, pr + QW + 2 * KW, qv(i), qv(i) + (size_t)c.NQ * HD);
        });
        {  // the blocks these positions complete (their raw keys are all in the cache now)
          std::vector<int> blocks;
          for (int i = 0; i < m; i++)
            if ((pos0 + i + 1) % c.ratio == 0) blocks.push_back((pos0 + i) / c.ratio);
          parallel_for(pool, (int)blocks.size(), [&](int k) { pool_key(L, blocks[k]); });
        }
        parallel_for(pool, m, [&](int i) { qsa_select(L, pos0 + i, P.proj.data() + (size_t)i * PW + QW + 2 * KW, P.sel[i], false); });
        parallel_for(pool, m * c.NQ, [&](int j) {
          const int i = j / c.NQ, h = j % c.NQ;
          qsa_head(L, h, qv(i) + (size_t)h * HD, qv(i) + (size_t)c.NQ * HD + (size_t)h * HD, P.sel[i], P.att.data() + (size_t)i * c.NQ * HD + (size_t)h * HD);
        });
        auto tc = clk::now();
        T->attn_core += ms_between(tb, tc);
        stage_mark(ST_ATTN_OUT);
        quant(P.xc, &P.qc, P.att.data(), c.NQ * HD, c.NQ * HD);
        inputs(P.xc, P.att.data(), c.NQ * HD);
        for (int i = 0; i < m; i++) o1[i] = P.y.data() + (size_t)i * H;
        JobN jo = {&W.o, m, xq.data(), xf.data(), o1.data(), packed ? &P.qc : nullptr};
        run_rows_n_split(pool, G, po.gpu_frac, &jo, 1, 32, ref, &T->bytes_res, &gacct);
        T->attn_out += ms_since(tc);
      }
      inject(P.y.data(), P.inj1.data());
      // ---- the MoE
      hc_chunk(W.mhc, P.mix.data(), P.inj2.data());
      std::vector<const float*> xm(m);
      std::vector<float*> ym(m);
      for (int i = 0; i < m; i++) xm[i] = P.mix.data() + (size_t)i * H, ym[i] = P.y.data() + (size_t)i * H;
      core->moe_gpu = gpu && po.gpu_experts > 0 && m >= po.gpu_min ? gpu.get() : nullptr;  // the experts' share
      core->moe_gpu_frac = po.gpu_experts;
      if (bi == 0 && names && L + 1 < NL) {  // named inside moe_chunk, behind this layer's own reads
        core->span_next = L + 1;
        core->span_next_router = names == 2;
      }
      core->moe_chunk(L, m, xm.data(), ym.data(), views[L], nullptr, nullptr, i0);
      core->moe_gpu = nullptr;
      inject(P.y.data(), P.inj2.data());
      if (ver && L + 1 == NL) hc_chunk(fin, ver->fx.data(), nullptr);  // every position's final mixer (step()'s hc_step, batched)
      if (L == 0 && captured(bnd[bi])) sink_ng(bnd[bi]);
    }
    core->span_unname(L);  // its union pins and its names
    if (multi && progress && *progress) (*progress)(prog_base + (int)((long)n * (L + 1) / NL), prog_total, 0.0);
    if (cancel && multi && cancel->load() && L + 1 < NL) {  // stop: the fixed state back at the span's start, nothing else moved
      load_fixed(*S, P.snap.data(), S->pos);
      core->span_unname(L + 1);
      core->span_mode = false;
      return false;
    }
  }
  core->span_mode = false;
  if (ver) {  // every position's logits, as step() computes them under the pass's Want
    auto th = clk::now();
    stage_mark(ST_HEAD);
    const XQ* xq[TMAX];
    const float* xf[TMAX];
    float* lo[TMAX];
    for (int i = 0; i < n; i++) {
      xf[i] = ver->fx.data() + (size_t)i * H;
      ver->xh[i].reset(H);
      ver->xh[i].prep(xf[i], H);
      xq[i] = &ver->xh[i].q;
      lo[i] = ver->lg.data() + (size_t)i * c.V;
    }
    if (fast_head && (ver->want == Want::GREEDY || (ver->want == Want::TOPK && head_topk > 0))) {
      sh.run_n(pool, xq, n, ver->want == Want::GREEDY ? 1 : head_topk, lo, ver_best, nullptr);
    } else {
      const bool pk = packed_ok && n >= 2 && H % 256 == 0;
      if (pk) {
        ver->qh.reserve(H, n);
        for (int i = 0; i < n; i++) xqb_set_from(ver->qh, i, *xq[i]);
        xqb_pad(ver->qh);
      }
      JobN jh = {&head, n, xq, xf, lo, pk ? &ver->qh : nullptr};
      run_rows_n(pool, &jh, 1, 64, ref, &T->bytes_res);
      for (int i = 0; i < n; i++) ver_best[i] = -1;
    }
    T->head += ms_since(th);
  } else if (want_last) {  // the last position's logits, as step() computes them
    std::vector<float> xl(H);
    hc_step(fin, P.hyp.data() + (size_t)(n - 1) * N, xl.data(), nullptr);
    head_step(xl.data(), Want::LAST);
  }
  S->ng_ctx[0] = P.hist[n];
  S->ng_ctx[1] = P.hist[n + 1];
  S->pos += n;
  core->chunk_end(n);
  n_chunks += (long)bnd.size();
  stage_mark(ST_OTHER);
  return true;
}

bool Flash::prefill(State& s0, const int32_t* ids, int n, bool want_logits, const std::vector<int>& capture, const LM::StateSink& sink,
                    PrefillStats* st, const std::atomic<bool>* cancel) {
  S = (FState*)&s0;
  // with the screened head on, the next token is greedy(): the prompt's last logits are the full head's, so their argmax
  // (lowest index on a tie) is it
  auto argmax = [&]() {
    int id = 0;
    for (int r = 1; r < c.V; r++)
      if (lgts[r] > lgts[id]) id = r;
    return id;
  };
  if (po.chunk <= 1 || (core->ns.on && !core->ns.drop) || (core->force && S->pos < core->force_n)) {
    const bool r = Model::prefill(s0, ids, n, want_logits, capture, sink, st, cancel);  // single steps
    if (r && fast_head && want_logits) greedy_ = argmax();
    return r;
  }
  // fast mode (approx:drop): the prompt runs the batched pass losslessly (prefill_spans turns the rule off), as the 35B's
  // prefill does: a span reads the union of its positions' experts once whatever the rule, so leaving picks out saves no
  // reads there, and the prompt's state is exact mode's
  const double be0 = T->bytes_exp, br0 = T->bytes_res;
  const Times tt0 = *T;  // the stages of this prefill (ms in all, under the decode's names)
  const long c0 = n_chunks;
  const GpuSplitAcct ga0 = gacct;
  const double me_w0 = core->moe_gpu_wait_ms, me_m0 = core->moe_gpu_macs;
  const uint64_t me_n0 = core->moe_gpu_sets, me_f0 = core->moe_gpu_failed;
  int spans = 0;
  // The scratch is bounded by memory, not only by the front end's chunk and span: this model's per-position scratch is
  // large (four streams of the hidden state, 40 KB per position for the streams alone, and about 0.6 MB per position of a
  // sub-chunk), and on the 16 GB Mac the prompt path's peak decides whether it swaps. Sub-chunks of at most
  // scratch_mb / 2 (each layer's resident matrices are read once per sub-chunk: 4 GB / 256 positions = 16 MB per position,
  // under 1% of the position's compute) and spans of at most scratch_mb / 2 of streams (a span reads every expert of
  // every layer once, 68 GB: at 3,000 positions that is 20 s of drive under 75+ s of compute).
  const int sub_cap = std::max(64, (int)(px.scratch_mb * 1e6 / 2 / sub_bytes_per_pos()) / 64 * 64);
  const int chunk_eff = std::min(po.chunk, sub_cap);
  const int span_cap = std::max(chunk_eff, (int)(px.scratch_mb * 1e6 / 2 / (4.0 * c.N())) / chunk_eff * chunk_eff);
  const int span = std::min(std::max(chunk_eff, po.span), span_cap);
  const int chunk_keep = po.chunk;
  po.chunk = chunk_eff;  // (span_run's sub-chunks)
  PrefillOpts o = po;
  o.span = span;
  const bool ok = prefill_spans(*core, n, capture, cancel, o, [&](int done, int m, bool last, const std::vector<int>& cap_rel) {
    if (m == 1) {
      step(s0, ids[done], last && want_logits ? Want::LAST : Want::NONE);
      n_chunks++;
      if (sink && !cap_rel.empty()) {
        std::vector<uint8_t> blob(fixed_bytes(s0));
        save_fixed(s0, blob.data());
        sink(s0.pos, 0, blob.data(), blob.size());
      }
      return true;
    }
    return span_run(ids + done, m, last && want_logits, &cap_rel, &sink, cancel, st ? &st->progress : nullptr, done, n);
  }, st, &spans);
  if (st) {
    const int done = st->positions;
    st->chunks = (int)(n_chunks - c0);
    const double np_ = std::max(1, done);
    std::ostringstream o;
    o << "{\"chunk\":" << po.chunk << ",\"chunk_asked\":" << chunk_keep << ",\"span\":" << span << ",\"span_asked\":" << po.span << ",\"spans\":" << spans << ",\"union_batches\":" << core->n_union_batches
      << ",\"mb_read\":" << st->mb_read << ",\"read_mb_per_pos\":" << st->mb_read / np_ << ",\"expert_touched_mb_per_pos\":" << (T->bytes_exp - be0) / MB / np_
      << ",\"resident_touched_mb_per_pos\":" << (T->bytes_res - br0) / MB / np_ << ",\"wait_ms\":" << st->ms_wait
      << ",\"ngram_mb\":" << ngs.n_bytes.load() / 1e6;
    {
      Times d = *T;
      d.embed -= tt0.embed, d.attn_proj -= tt0.attn_proj, d.attn_core -= tt0.attn_core, d.attn_out -= tt0.attn_out, d.depth -= tt0.depth;
      d.msg -= tt0.msg, d.router -= tt0.router, d.experts -= tt0.experts, d.wait -= tt0.wait, d.cache -= tt0.cache, d.pf -= tt0.pf;
      d.head -= tt0.head, d.misc -= tt0.misc, d.ra -= tt0.ra;
      o << ",\"stages_ms\":{" << d.stages_json(1) << "}";
    }
    if (gpu && (po.gpu_frac > 0 || po.gpu_experts > 0))
      o << ",\"gpu\":{\"kernel\":\"" << (gpu->kernel() == 1 ? "matrix" : "integer") << "\",\"attn\":{\"frac\":" << po.gpu_frac
        << ",\"gmac_gpu\":" << (gacct.macs_gpu - ga0.macs_gpu) / 1e9 << ",\"gmac_cpu\":" << (gacct.macs_cpu - ga0.macs_cpu) / 1e9
        << ",\"wait_ms\":" << gacct.ms_wait - ga0.ms_wait << ",\"sets\":" << gacct.sets - ga0.sets << ",\"failed\":" << gacct.failed - ga0.failed
        << "},\"experts\":{\"frac\":" << po.gpu_experts << ",\"gmac_gpu\":" << (core->moe_gpu_macs - me_m0) / 1e9 << ",\"wait_ms\":" << core->moe_gpu_wait_ms - me_w0
        << ",\"sets\":" << core->moe_gpu_sets - me_n0 << ",\"failed\":" << core->moe_gpu_failed - me_f0 << "}}";
    o << "}";
    st->json = o.str();
  }
  po.chunk = chunk_keep;
  if (po.release) {  // the span's scratch (40 KB per position for the streams alone) and the GPU's buffers go back
    sp.reset();
    if (gpu) gpu->trim();
  }
  if (ok && fast_head && want_logits) greedy_ = argmax();
  return ok;
}

const float* Flash::logits_at(int i) const { return V && i >= 0 && i < V->n ? V->lg.data() + (size_t)i * c.V : nullptr; }

bool Flash::verify(State& s0, const int32_t* ids, int n, Want want) {
  S = (FState*)&s0;
  if (n < 1 || n > TMAX || (core->ns.on && !core->ns.drop) || (core->force && S->pos < core->force_n) || S->pos + n > S->ctx) return false;
  greedy_ = -1;
  const int H = c.H, C = c.conv_dim();
  const size_t PW = proj_width(c);
  if (!V) {
    V = std::make_shared<Ver>();
    Ver& W8 = *V;
    W8.altc.resize(c.NL);
    W8.alts.resize(c.NL);
    W8.gp.resize(c.NL);
    W8.cvp.resize(c.NL);
    for (int L = 0; L < c.NL; L++)
      if (lw[L].lin) {
        W8.gp[L].resize((size_t)TMAX * PW);
        W8.cvp[L].resize((size_t)TMAX * C);
      }
    W8.embs.resize((size_t)TMAX * H);
    W8.fx.resize((size_t)TMAX * H);
    W8.lg.resize((size_t)TMAX * c.V);
    W8.xh.resize(TMAX);
    for (auto& x : W8.xh) x.init(H, prec);
  }
  Ver& W8 = *V;
  for (int L = 0; L < c.NL; L++)  // B at the state's sizes
    if (lw[L].lin) {
      if (W8.altc[L].size() != S->ls[L].conv.size()) W8.altc[L].resize(S->ls[L].conv.size());
      if (W8.alts[L].size() != S->ls[L].S.size()) W8.alts[L].resize(S->ls[L].S.size());
    }
  W8.want = want;
  W8.pos0 = S->pos;
  span_run(ids, n, false, nullptr, nullptr, nullptr, nullptr, 0, n, &W8);
  W8.n = n;
  W8.live = true;
  W8.steps++;
  W8.positions += n;
  return true;
}

void Flash::commit(State& s0, int a) {
  S = (FState*)&s0;
  if (!V || !V->live) return;
  Ver& W8 = *V;
  W8.live = false;
  const int n = W8.n, H = c.H, N = c.N(), C = c.conv_dim(), ZD = c.NVH * c.DV, CK = c.CK;
  const size_t PW = proj_width(c);
  a = std::max(1, std::min(a, n));
  Pool& pool = *core->pool;
  auto t0 = clk::now();
  for (int L = 0; L < c.NL; L++) {
    if (!lw[L].lin) continue;
    LState& St = S->ls[L];
    if (a < n) {  // B again from A for positions 0..a-1
      // the conv state after a positions: the last CK - 1 rows of [A's rows, inputs 0 .. a-1]
      const int R = CK - 1;
      float* Bc = W8.altc[L].data();
      for (int r = 0; r < R; r++) {
        const int idx = a + r;  // index into [A rows 0 .. R-1, inputs 0 ..]
        if (idx < R) std::memcpy(Bc + (size_t)r * C, St.conv.data() + (size_t)idx * C, sizeof(float) * C);
        else std::memcpy(Bc + (size_t)r * C, W8.gp[L].data() + (size_t)(idx - R) * PW, sizeof(float) * C);
      }
      const Lw& W = lw[L];
      const float* Asn = St.S.data();
      float* Bs = W8.alts[L].data();
      const float* gp = W8.gp[L].data();
      const float* cvp = W8.cvp[L].data();
      parallel_for(pool, c.NVH, [&](int h) {  // the heads again on the kept positions' inputs (state only, as decode's)
        const size_t hb = (size_t)h * c.DV * c.DK;
        for (int i = 0; i < a; i++) {
          const float* pr = gp + (size_t)i * PW;
          gdn::head_io(gd, W.g, (i == 0 ? Asn : Bs) + hb, Bs + hb, cvp + (size_t)i * C, nullptr, pr[C + ZD + h], pr[C + ZD + c.NVH + h], h, nullptr);
        }
      });
    }
    St.conv.swap(W8.altc[L]);
    St.S.swap(W8.alts[L]);
  }
  if (a < n) {
    // the PLE's conv state: the pass's start, moved by the kept positions' inputs (ple_conv's commit)
    S->pconv = W8.pconv0;
    const int PS = c.pstate();
    float* st = S->pconv.data();
    for (int i = 0; i < a; i++) {
      const float* x = W8.ple_in.data() + (size_t)i * N;
      for (int j = 0; j + 1 < PS; j++) std::memcpy(st + (size_t)j * N, st + (size_t)(j + 1) * N, sizeof(float) * N);
      std::memcpy(st + (size_t)(PS - 1) * N, x, sizeof(float) * N);
    }
    S->ng_ctx[0] = W8.hist[a];
    S->ng_ctx[1] = W8.hist[a + 1];
    S->pos = W8.pos0 + a;
    core->tok_clock -= n - a;
    std::copy(W8.embs.begin() + (size_t)(a - 1) * H, W8.embs.begin() + (size_t)a * H, emb.begin());
    W8.rollbacks++;
    W8.ms_rollback += ms_since(t0);
  }
  std::memcpy(lgts.data(), W8.lg.data() + (size_t)(a - 1) * c.V, sizeof(float) * c.V);
  greedy_ = fast_head ? ver_best[a - 1] : -1;  // (the screened head's argmax of the last kept position; -1: the full head's logits)
  core->chunk_commit(a);
}

std::string Flash::verify_json() const {
  if (!V) return "{}";
  std::ostringstream o;
  o << "{\"steps\":" << V->steps << ",\"positions\":" << V->positions << ",\"rollbacks\":" << V->rollbacks << ",\"ms_rollback\":" << V->ms_rollback << "}";
  return o.str();
}


}  // namespace qwen4
}  // namespace dm
