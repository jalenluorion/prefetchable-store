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

size_t Flash::span_bytes() const { return sp ? sp->bytes() : 0; }

static size_t proj_width(const FCfg& c) {
  return std::max((size_t)c.conv_dim() + c.NVH * c.DV + 2 * c.NVH,
                  (size_t)2 * c.NQ * c.HD + 2 * c.NKV * c.HD + (c.IXH + 1) * c.IXD + 2 * c.NQ * c.HD);
}

bool Flash::span_run(const int32_t* ids, int n, bool want_last, const std::vector<int>* capture, const LM::StateSink* sink,
                     const std::atomic<bool>* cancel, const PrefillProgress* progress, int prog_base, int prog_total) {
  const int H = c.H, N = c.N(), NL = c.NL, sub = std::max(1, std::min(po.chunk, n));
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
    const int kmax = std::max({N, H, c.ple_dim, c.LOW, c.NVH * c.DV, c.NQ * c.HD});
    while ((int)P.xa.size() < wmax) {
      P.xa.emplace_back();
      P.xa.back().init(kmax, prec);
      P.xb.emplace_back();
      P.xb.back().init(kmax, prec);
      P.xc.emplace_back();
      P.xc.back().init(kmax, prec);
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
    }
  }
  T->embed += ms_since(t0);
  if (cancel && multi) {  // a cancel inside the span puts the fixed state back here
    P.snap.resize(fixed_bytes(*S));
    save_fixed(*S, P.snap.data());
  }
  core->pos = S->pos;
  Cache::CapsOff caps_off(core->cache);  // a span names the next layer's union: caps off for the pass
  core->span_mode = multi;
  Cache::ChunkClock span_clock(core->cache, core->tok_clock + n - 1);  // the whole span is one token to the policy
  const bool names = multi && po.names && n >= po.whole_from;
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
      const bool packed = packed_ok && m >= 6;
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
      };
      auto inject = [&](const float* yv, const float* ij) {  // h_s += inj_s * y, per position
        parallel_for(pool, m, [&](int i) {
          float* h = hp(i);
          const float* yy = yv + (size_t)i * H;
          const float* jj = ij + (size_t)i * 64;
          for (int s = 0; s < c.HC; s++)
            for (int k = 0; k < H; k++) h[(size_t)s * H + k] = __builtin_fmaf(yy[k], jj[s], h[(size_t)s * H + k]);
        });
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
        T->misc += ms_since(tp);
        if (captured(bnd[bi])) (*sink)(S->pos + bnd[bi], fixed_ple, S->pconv.data(), S->pconv.size() * 4);
      }
      // ---- the mixer
      auto ta = clk::now();
      stage_mark(ST_ATTN_PROJ);
      hc_chunk(W.ahc, P.mix.data(), P.inj1.data());
      quant(P.xa, &P.qa, P.mix.data(), H, H);
      inputs(P.xa, P.mix.data(), H);
      if (W.lin) {
        const int C = c.conv_dim(), ZD = c.NVH * c.DV;
        for (int i = 0; i < m; i++) {
          float* pr = P.proj.data() + (size_t)i * PW;
          o1[i] = pr, o2[i] = pr + C, o3[i] = pr + C + ZD, o4[i] = pr + C + ZD + c.NVH;
        }
        const XQB* pa = packed ? &P.qa : nullptr;
        JobN jb[4] = {{&W.qkv, m, xq.data(), xf.data(), o1.data(), pa}, {&W.z, m, xq.data(), xf.data(), o2.data(), pa},
                      {&W.wb, m, nullptr, xf.data(), o3.data(), nullptr}, {&W.wa, m, nullptr, xf.data(), o4.data(), nullptr}};
        run_rows_n_split(pool, G, po.gpu_frac, jb, 4, 32, ref, &T->bytes_res, &gacct);
        auto tb = clk::now();
        T->attn_proj += ms_between(ta, tb);
        stage_mark(ST_ATTN_CORE);
        float* cst = S->ls[L].conv.data();
        const int CK = c.CK, CB = 256;
        parallel_for(pool, (C + CB - 1) / CB, [&](int b) {  // the convolution: channel blocks, positions in order
          for (int i = 0; i < m; i++) {
            const float* qkv = P.proj.data() + (size_t)i * PW;
            float* cv = P.cv.data() + (size_t)i * C;
            for (int ch = b * CB; ch < std::min(C, b * CB + CB); ch++) cv[ch] = gdn::conv_channel(cst, W.g.conv + (size_t)ch * CK, qkv[ch], C, CK, ch);
          }
        });
        float* Sst = S->ls[L].S.data();
        parallel_for(pool, c.NVH, [&](int h) {  // the heads: head by head, positions in order (gdn::head, as decode)
          float* Sh = Sst + (size_t)h * c.DV * c.DK;
          for (int i = 0; i < m; i++) {
            const float* pr = P.proj.data() + (size_t)i * PW;
            gdn::head(gd, W.g, Sh, P.cv.data() + (size_t)i * C, pr + C + h * c.DV, pr[C + ZD + h], pr[C + ZD + c.NVH + h], h,
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
      core->moe_chunk(L, m, xm.data(), ym.data(), views[L], nullptr, nullptr, i0);
      core->moe_gpu = nullptr;
      inject(P.y.data(), P.inj2.data());
      if (bi == 0 && names && L + 1 < NL) core->span_name(L + 1);  // behind this layer's own reads
      if (L == 0 && captured(bnd[bi])) sink_ng(bnd[bi]);
    }
    if (multi) core->span_unname(L);  // its union pins and its names
    if (multi && progress && *progress) (*progress)(prog_base + (int)((long)n * (L + 1) / NL), prog_total, 0.0);
    if (cancel && multi && cancel->load() && L + 1 < NL) {  // stop: the fixed state back at the span's start, nothing else moved
      load_fixed(*S, P.snap.data(), S->pos);
      if (names) core->span_unname(L + 1);
      core->span_mode = false;
      return false;
    }
  }
  core->span_mode = false;
  if (want_last) {  // the last position's logits, as step() computes them
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
  if (po.chunk <= 1 || core->ns.on || (core->force && S->pos < core->force_n)) {
    const bool r = Model::prefill(s0, ids, n, want_logits, capture, sink, st, cancel);  // single steps
    if (r && fast_head && want_logits) greedy_ = argmax();
    return r;
  }
  std::vector<int> caps(capture);
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  auto t0 = clk::now();
  const uint64_t b0 = core->cache.st.bytes_urgent + core->cache.st.bytes_pf;
  const double w0 = core->cache.st.wait_ms, be0 = T->bytes_exp, br0 = T->bytes_res;
  const long c0 = n_chunks;
  const GpuSplitAcct ga0 = gacct;
  const double me_w0 = core->moe_gpu_wait_ms, me_m0 = core->moe_gpu_macs;
  const uint64_t me_n0 = core->moe_gpu_sets, me_f0 = core->moe_gpu_failed;
  int done = 0, spans = 0;
  const int span = std::max(po.chunk, po.span);
  bool ok = true;
  while (done < n) {
    if (cancel && cancel->load()) {
      ok = false;
      break;
    }
    int m = std::min(span, n - done);
    // no short last span (it would read every layer's experts again for a few positions): a rest under a quarter span
    // joins this one
    if (n - done - m > 0 && n - done - m < std::max(64, span / 4)) m = n - done;
    const bool last = done + m == n;
    std::vector<int> cap_rel;
    for (int p : caps)
      if (p > done && p <= done + m) cap_rel.push_back(p - done);
    if (m == 1) {
      step(s0, ids[done], last && want_logits ? Want::LAST : Want::NONE);
      n_chunks++;
      if (sink && !cap_rel.empty()) {
        std::vector<uint8_t> blob(fixed_bytes(s0));
        save_fixed(s0, blob.data());
        sink(s0.pos, 0, blob.data(), blob.size());
      }
    } else if (!span_run(ids + done, m, last && want_logits, &cap_rel, &sink, cancel, st ? &st->progress : nullptr, done, n)) {
      ok = false;
      break;
    }
    done += m;
    spans++;
  }
  if (st) {
    st->positions = done;
    st->chunks = (int)(n_chunks - c0);
    st->ms = ms_since(t0);
    st->ms_wait = core->cache.st.wait_ms - w0;
    st->mb_read = (core->cache.st.bytes_urgent + core->cache.st.bytes_pf - b0) / 1048576.0;
    const double np_ = std::max(1, done);
    std::ostringstream o;
    o << "{\"chunk\":" << po.chunk << ",\"span\":" << span << ",\"spans\":" << spans << ",\"union_batches\":" << core->n_union_batches
      << ",\"mb_read\":" << st->mb_read << ",\"read_mb_per_pos\":" << st->mb_read / np_ << ",\"expert_touched_mb_per_pos\":" << (T->bytes_exp - be0) / 1048576.0 / np_
      << ",\"resident_touched_mb_per_pos\":" << (T->bytes_res - br0) / 1048576.0 / np_ << ",\"wait_ms\":" << st->ms_wait
      << ",\"ngram_mb\":" << ngs.n_bytes.load() / 1e6;
    if (gpu && (po.gpu_frac > 0 || po.gpu_experts > 0))
      o << ",\"gpu\":{\"kernel\":\"" << (gpu->kernel() == 1 ? "matrix" : "integer") << "\",\"attn\":{\"frac\":" << po.gpu_frac
        << ",\"gmac_gpu\":" << (gacct.macs_gpu - ga0.macs_gpu) / 1e9 << ",\"gmac_cpu\":" << (gacct.macs_cpu - ga0.macs_cpu) / 1e9
        << ",\"wait_ms\":" << gacct.ms_wait - ga0.ms_wait << ",\"sets\":" << gacct.sets - ga0.sets << ",\"failed\":" << gacct.failed - ga0.failed
        << "},\"experts\":{\"frac\":" << po.gpu_experts << ",\"gmac_gpu\":" << (core->moe_gpu_macs - me_m0) / 1e9 << ",\"wait_ms\":" << core->moe_gpu_wait_ms - me_w0
        << ",\"sets\":" << core->moe_gpu_sets - me_n0 << ",\"failed\":" << core->moe_gpu_failed - me_f0 << "}}";
    o << "}";
    st->json = o.str();
  }
  if (gpu && po.release) gpu->trim();
  if (ok && fast_head && want_logits) greedy_ = argmax();
  return ok;
}

bool Flash::set_gpu_frac(double f, bool experts, std::string& err) {
  if (!(f >= 0 && f < 1)) return (err = experts ? "prefill_gpu_experts must be in [0, 1)" : "prefill_gpu_frac must be in [0, 1)"), false;
  if (f > 0 && !gpu) {
#if defined(DM_METAL)
    if (!R.base || !core) return (err = "prefill GPU split: load and bind the model first"), false;
    std::unique_ptr<GpuGemm> g(new GpuGemm());
    if (!g->init()) return (err = "prefill GPU split: GPU (Metal) init failed"), false;
    if (!g->wrap(R.base, R.size)) return (err = "prefill GPU split: the resident weights could not be wrapped"), false;
    gpu = std::move(g);
#else
    return (err = "prefill GPU split: this build has no Metal (DM_METAL off)"), false;
#endif
  }
  if (experts && f > 0 && !slab_wrapped) {  // the expert slab where it is (locked or plain; mapped records stay on the CPU)
    bool any = false;
    for (int p = 0; p < core->cache.npools; p++)
      if (core->cache.P[p].slab) any = gpu->wrap(core->cache.P[p].slab, core->cache.P[p].slab_bytes) || any;
    if (!any) return (err = "prefill GPU split: the expert slab could not be wrapped"), false;
    slab_wrapped = true;
  }
  (experts ? po.gpu_experts : po.gpu_frac) = f;
  return true;
}

}  // namespace qwen4
}  // namespace dm
