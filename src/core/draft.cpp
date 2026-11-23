// Draft and verify (include/dmeng/draft.h): the draft sources, the controller and the decode step, shared by dmeng and
// dmserve. Nothing here changes a computed bit: drafts only decide how many positions a pass carries.
#include "dmeng/draft.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>

#include "dmeng/core.h"
#include "dmeng/model.h"
#include "dmeng/sampler.h"
#include "../format/files.h"

namespace dm {

static inline uint64_t key2(int32_t a, int32_t b) { return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; }

// ------------------------------------------------------------------------------------------------ lookup
void LookupSource::Index::add(size_t from) {
  for (size_t j = std::max<size_t>(1, from); j < seq.size(); j++)
    if (seq[j - 1] >= 0 && seq[j] >= 0) pos[key2(seq[j - 1], seq[j])].push_back((int32_t)j);
}

int LookupSource::match(const std::vector<int32_t>& a, long ia, const std::vector<int32_t>& b, long ib, int cap) {
  int k = 0;
  while (k < cap && ia - k >= 0 && ib - k >= 0 && b[ib - k] >= 0 && a[ia - k] == b[ib - k]) k++;
  return k;
}

bool LookupSource::sync(const std::vector<int32_t>& h) {
  const size_t n0 = idx_.seq.size();
  bool ext = h.size() >= n0 && (n0 == 0 || h[n0 - 1] == idx_.seq[n0 - 1]);
  if (ext && n0 > 0) ext = std::equal(idx_.seq.begin() + (n0 > 64 ? n0 - 64 : 0), idx_.seq.end(), h.begin() + (n0 > 64 ? n0 - 64 : 0));
  if (!ext) {  // the sequence changed below what is indexed: index it again
    idx_.seq.clear();
    idx_.pos.clear();
  }
  const size_t from = idx_.seq.size();
  if (h.size() > from) {
    idx_.seq.insert(idx_.seq.end(), h.begin() + from, h.end());
    idx_.add(from);
  }
  return true;
}

void LookupSource::add_archive(const std::vector<int32_t>& s) {
  if (s.empty()) return;
  if (arch_.seq.size() + s.size() + 1 > archive_cap) {  // the oldest go: keep the newer half
    std::vector<int32_t> keep(arch_.seq.begin() + arch_.seq.size() / 2, arch_.seq.end());
    arch_.seq.swap(keep);
    arch_.pos.clear();
    arch_.add(0);
  }
  const size_t from = arch_.seq.size();
  arch_.seq.push_back(-1);  // separator: no match crosses it
  arch_.seq.insert(arch_.seq.end(), s.begin(), s.end());
  arch_.add(from);
}

bool LookupSource::propose(const std::vector<int32_t>& hist, int max, Draft& d) {
  d.clear();
  last_n = 0;
  sync(hist);
  const long end = (long)hist.size() - 1;
  if (end < 1 || max < 1) return false;
  const uint64_t k = key2(hist[end - 1], hist[end]);
  int best = 0;
  long bp = -1;
  auto it = idx_.pos.find(k);
  if (it != idx_.pos.end()) {
    const auto& v = it->second;
    int seen = 0;
    for (long q = (long)v.size() - 1; q >= 0 && seen < max_cand; q--) {
      const long p = v[q];
      if (p >= end) continue;
      seen++;
      const int m = match(hist, end, hist, p, max_n);
      if (m > best) best = m, bp = p;
      if (best >= max_n) break;
    }
  }
  bool from_arch = false;
  if (best < max_n && !arch_.seq.empty()) {
    auto ia = arch_.pos.find(k);
    if (ia != arch_.pos.end()) {
      const auto& v = ia->second;
      int seen = 0;
      for (long q = (long)v.size() - 1; q >= 0 && seen < max_cand; q--) {
        const long p = v[q];
        if (p + 1 >= (long)arch_.seq.size() || arch_.seq[p + 1] < 0) continue;
        seen++;
        const int m = match(hist, end, arch_.seq, p, max_n);
        if (m > best) best = m, bp = p, from_arch = true;
        if (best >= max_n) break;
      }
    }
  }
  if (best < min_n || bp < 0) return false;
  const std::vector<int32_t>& src = from_arch ? arch_.seq : hist;
  const long lim = from_arch ? (long)src.size() - 1 : end;
  for (long q = bp + 1; q <= lim && (int)d.toks.size() < max && src[q] >= 0; q++) d.toks.push_back(src[q]);
  if (d.toks.empty()) return false;
  static const float prior[4] = {0.4f, 0.6f, 0.85f, 0.95f};
  d.bucket = bucket_of(std::min(best, max_n));
  d.conf.assign(d.toks.size(), prior[d.bucket]);
  d.src = from_arch ? -1 : bp + 1;
  last_n = best;
  return true;
}

// ------------------------------------------------------------------------------------------------ skeleton
bool SkeletonSource::propose(const std::vector<int32_t>& hist, int max, Draft& d) {
  d.clear();
  if (frags_.empty() || hist.empty() || max < 1) return false;
  // every fragment position whose tokens end like the sequence (longest match); the draft is what all of them agree on
  // next (after "<tool_call>": the call's opening up to the tool name; after a unique name prefix: the rest of the name
  // and the first key; after "</parameter>": nothing, the next key or the closing differ)
  const long end = (long)hist.size() - 1;
  int best = 0;
  std::vector<std::pair<int, long>> at;
  for (int f = 0; f < (int)frags_.size(); f++) {
    const auto& F = frags_[f];
    for (long q = 0; q + 1 < (long)F.size(); q++) {
      if (F[q] != hist[end]) continue;
      int m = 1;
      while (m < max_back && end - m >= 0 && q - m >= 0 && F[q - m] == hist[end - m]) m++;
      if (m > best) best = m, at.clear();
      if (m == best) at.push_back({f, q});
    }
  }
  if (at.empty()) return false;
  for (int k = 1; k <= max; k++) {
    int tok = -1;
    bool agree = true;
    for (auto& fq : at) {
      const auto& F = frags_[fq.first];
      const long q = fq.second + k;
      const int t = q < (long)F.size() ? F[q] : -2;
      if (t < 0 || (tok >= 0 && t != tok)) {
        agree = false;
        break;
      }
      tok = t;
    }
    if (!agree) break;
    d.toks.push_back(tok);
  }
  static const float prior[3] = {0.6f, 0.85f, 0.95f};  // the format fixes what all matching fragments agree on
  d.bucket = bucket_of(best);
  d.conf.assign(d.toks.size(), prior[d.bucket]);
  d.src = -1;
  return !d.toks.empty();
}

// ------------------------------------------------------------------------------------------------ draft head
int HeadSource::bin(double p) {
  static const double lo[NB] = {0, 0.8, 0.9, 0.95, 0.98, 0.995};
  int b = 0;
  while (b + 1 < NB && p >= lo[b + 1]) b++;
  return b;
}

bool HeadSource::ask(const std::vector<int32_t>& hist, double best_gain) {
  now = (long)hist.size();
  if (backoff < 0) backoff = backoff0;
  if (now < backoff_until) return skipped_backoff++, false;
  if (best_gain > 0 && best_gain >= h_val) return skipped_lookup++, false;
  return true;
}

bool HeadSource::propose(const std::vector<int32_t>& hist, int max, Draft& d) {
  d.clear();
  if (!M || max < 1) return false;
  std::vector<int32_t> t(max);
  prob_.assign(max, 0.f);
  auto t0 = std::chrono::steady_clock::now();
  const int n = M->draft_head(hist.back(), max, thr, t.data(), prob_.data());
  last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  h_ms = runs ? 0.8 * h_ms + 0.2 * last_ms : last_ms;
  ms_total += last_ms;
  runs++;
  prob_.resize(std::max(0, n));
  if (n <= 0) return false;
  offers++;
  d.toks.assign(t.begin(), t.begin() + n);
  for (int k = 0; k < n; k++) d.conf.push_back((float)cal[bin(prob_[k])]);
  d.bucket = bin(prob_[0]);
  d.src = -1;  // (early names come from the lookup's source span only)
  return true;
}

void HeadSource::outcome(const DraftCtl& ctl, int l, double gain, bool chosen, int accepted, double step_ms) {
  const double hc = ctl.c1 > 0 ? h_ms / ctl.c1 : 0.1;  // the head's cost in one-position steps
  h_val = 0.8 * h_val + 0.2 * (l > 0 ? (gain * ctl.rr[l + 1]) / (ctl.rr[l + 1] + hc) : 1.0 / (1.0 + hc));
  if (chosen) {
    used++;
    acc += accepted;
    for (int k = 1; k <= l && k <= accepted + 1 && k <= (int)prob_.size(); k++) {
      double& cb = cal[bin(prob_[k - 1])];
      cb = 0.9 * cb + 0.1 * (k <= accepted ? 1.0 : 0.0);
    }
  }
  // a try wins when its draft ran, something was kept, and the step (the head in it) produced tokens at least as fast as
  // a plain step does; otherwise the head is backed off
  const bool faster = ctl.c1 <= 0 || (double)(1 + accepted) * ctl.c1 >= step_ms;
  if (chosen && accepted > 0 && faster) {
    backoff = backoff0;
  } else {
    lost++;
    backoff_until = now + backoff;
    backoff = std::min<long>(backoff_max, 2 * backoff);
  }
}

std::string HeadSource::stats_json() const {
  std::ostringstream o;
  o << "{\"thr\":" << thr << ",\"backoff\":[" << backoff0 << "," << backoff_max << "],\"runs\":" << runs << ",\"offers\":" << offers
    << ",\"steps\":" << used << ",\"accepted\":" << acc << ",\"lost\":" << lost << ",\"skipped_backoff\":" << skipped_backoff
    << ",\"skipped_lookup\":" << skipped_lookup << ",\"ms_total\":" << ms_total << ",\"ms_per_run\":" << ms_total / std::max<long>(1, runs)
    << ",\"memory_gb\":" << (M ? M->draft_head_bytes() / 1e9 : 0.0) << ",\"calibration\":[";
  for (int b = 0; b < NB; b++) o << (b ? "," : "") << cal[b];
  o << "]}";
  return o.str();
}

// ------------------------------------------------------------------------------------------------ light draft head
int LightHeadSource::bin(double p) {
  static const double lo[NB] = {0, 0.5, 0.7, 0.85, 0.95, 0.99};
  int b = 0;
  while (b + 1 < NB && p >= lo[b + 1]) b++;
  return b;
}

bool LightHeadSource::load(const std::string& path, int Dm, Pool* pool, bool quant, std::string& err) {
  pool_ = pool;
  q8 = quant;
  ms::SafeTensors f;
  if (!f.open(path, err)) return false;
  auto get = [&](const std::string& n, std::vector<long> shape, std::vector<uint16_t>& out) {
    const ms::StTensor* t = f.find(n);
    if (!t || t->dtype != "BF16") return (err = path + ": missing bf16 " + n), false;
    if (t->shape.size() != shape.size()) return (err = path + ": " + n + " has the wrong rank"), false;
    for (size_t i = 0; i < shape.size(); i++)
      if (shape[i] >= 0 && t->shape[i] != shape[i]) return (err = path + ": " + n + " has the wrong shape"), false;
    out.resize((size_t)t->numel());
    return f.read(*t, out.data(), err);
  };
  auto shape_of = [&](const std::string& n, int i) { const ms::StTensor* t = f.find(n); return t && (int)t->shape.size() > i ? (long)t->shape[i] : -1L; };
  D = Dm;
  H = (int)shape_of("mlp.1.up.weight", 0);
  if (H <= 0 || shape_of("mlp.1.up.weight", 1) != 2 * D) return (err = path + ": mlp.1.up.weight is not [H, 2 x hidden]"), false;
  ND = 0;
  while (f.find("mlp." + std::to_string(ND + 1) + ".up.weight")) ND++;
  lowrank = f.find("proj.weight") != nullptr;
  if ((D % 64) || (H % 64) || (2 * D) % 64) return (err = "light head: widths must be multiples of 64"), false;
  st_.assign(ND, Step());
  for (int j = 0; j < ND; j++) {
    const std::string p = "mlp." + std::to_string(j + 1) + ".";
    Step& S = st_[j];
    std::vector<uint16_t> v;
    auto vec = [&](const std::string& n, long len, std::vector<float>& o) {
      if (!get(p + n, {len}, v)) return false;
      o.resize(v.size());
      for (size_t i = 0; i < v.size(); i++) o[i] = bf16_to_f32(v[i]);
      return true;
    };
    if (!vec("norm_z.weight", D, S.nz) || !vec("norm_e.weight", D, S.ne) || !vec("up.bias", H, S.ub) || !vec("down.bias", D, S.db) ||
        !get(p + "up.weight", {H, 2L * D}, S.up16) || !get(p + "down.weight", {D, H}, S.dn16))
      return false;
    if (q8) {
      S.up.from_bf16(*pool, S.up16.data(), H, 2 * D);
      S.dn.from_bf16(*pool, S.dn16.data(), D, H);
      std::vector<uint16_t>().swap(S.up16);
      std::vector<uint16_t>().swap(S.dn16);
    }
  }
  if (lowrank) {
    R = (int)shape_of("proj.weight", 0);
    V = (int)shape_of("vocab.weight", 0);
    if (R <= 0 || R % 64 || V <= 0) return (err = path + ": proj / vocab shapes"), false;
    if (!get("proj.weight", {R, D}, proj16_) || !get("vocab.weight", {V, R}, vocab16_)) return false;
    if (q8) {
      proj_.from_bf16(*pool, proj16_.data(), R, D);
      vocab_.from_bf16(*pool, vocab16_.data(), V, R);
      std::vector<uint16_t>().swap(proj16_);
      std::vector<uint16_t>().swap(vocab16_);
    }
    xr_.init(R, Prec::I16);
    r_.resize(R);
    lg_.resize(V);
  }
  xu_.init(2 * D, Prec::I16);
  xa_.init(H, Prec::I16);
  xz_.init(D, Prec::I16);
  z_.resize(D), e_.resize(D), u_.resize(2 * D), a_.resize(H);
  return true;
}

size_t LightHeadSource::weight_bytes() const {
  size_t n = 0;
  for (auto& S : st_) n += S.up.bytes() + S.dn.bytes() + 2 * (S.up16.size() + S.dn16.size()) + 4 * (S.nz.size() + S.ne.size() + S.ub.size() + S.db.size());
  return n + proj_.bytes() + vocab_.bytes() + 2 * (proj16_.size() + vocab16_.size());
}

void LightHeadSource::mat(const OwnedAff8& q, const std::vector<uint16_t>& b16, int rows, int cols, XBuf& xb, const float* x, float* y) {
  if (q8) {
    xb.prep(x, cols);
    Job j{&q.w, &xb.q, x, y};
    run_rows(*pool_, &j, 1, 64);
  } else {
    Weight w;
    w.kind = Weight::BF16;
    w.rows = rows;
    w.cols = cols;
    w.w = b16.data();
    Job j{&w, nullptr, x, y};
    run_rows(*pool_, &j, 1, 64);
  }
}

// the vocabulary layer on the pool: each chunk of rows computed by one thread, which also takes the chunk's argmax and
// its softmax sum relative to its own maximum (an epilogue in the same fork-join); the chunks then combine: the argmax
// is the first index of the largest logit (ties to the lower index, as the trainer), conf = 1 / sum exp(l - max)
namespace {
struct VPost {
  const float* lg;
  int vo, chunk;
  std::vector<float> mx, se;
  std::vector<int> am;
};
void vpost(void* p, int r0, int r1) {
  VPost& P = *(VPost*)p;
  const int k = r0 / P.chunk;
  r1 = std::min(r1, P.vo);
  if (r0 >= r1) {
    P.mx[k] = -INFINITY, P.se[k] = 0, P.am[k] = -1;
    return;
  }
  float m = P.lg[r0];
  int a = r0;
  for (int r = r0 + 1; r < r1; r++)
    if (P.lg[r] > m) m = P.lg[r], a = r;
  float sum = 0;
  for (int r = r0; r < r1; r++) sum += std::exp(P.lg[r] - m);
  P.mx[k] = m, P.se[k] = sum, P.am[k] = a;
}
}  // namespace

void LightHeadSource::vocab_rows(int vo, int* id, float* conf) {
  const int chunk = 256;
  static thread_local VPost P;
  P.lg = lg_.data();
  P.vo = vo;
  P.chunk = chunk;
  const int nch = (V + chunk - 1) / chunk;
  P.mx.assign(nch, -INFINITY), P.se.assign(nch, 0.f), P.am.assign(nch, -1);
  if (q8) {
    xr_.prep(r_.data(), R);
    Job j{&vocab_.w, &xr_.q, r_.data(), lg_.data()};
    run_rows_post(*pool_, j, chunk, false, nullptr, vpost, &P);
  } else {
    Weight w;
    w.kind = Weight::BF16;
    w.rows = V;
    w.cols = R;
    w.w = vocab16_.data();
    Job j{&w, nullptr, r_.data(), lg_.data()};
    run_rows_post(*pool_, j, chunk, false, nullptr, vpost, &P);
  }
  float m = -INFINITY;
  int a = -1;
  for (int k = 0; k < nch; k++)
    if (P.am[k] >= 0 && P.mx[k] > m) m = P.mx[k], a = P.am[k];
  double z = 0;
  for (int k = 0; k < nch; k++)
    if (P.am[k] >= 0) z += (double)P.se[k] * std::exp((double)(P.mx[k] - m));
  *id = a;
  *conf = (float)(1.0 / z);
}

int LightHeadSource::run(const float* h, int32_t tok, int max, int32_t* out, float* conf) {
  auto t0 = std::chrono::steady_clock::now();
  const int32_t tok0 = tok;
  for (int i = 0; i < D; i++) {  // z_0 = h_t in bf16, as the trainer feeds it (round to nearest even)
    uint32_t u;
    std::memcpy(&u, h + i, 4);
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
    std::memcpy(&z_[i], &u, 4);
  }
  int nd = 0;
  for (int j = 0; j < std::min(max, ND); j++) {
    auto ta = std::chrono::steady_clock::now();
    if (!embed || !embed(tok, e_.data())) break;
    auto tb = std::chrono::steady_clock::now();
    ms_embed += std::chrono::duration<double, std::milli>(tb - ta).count();
    Step& S = st_[j];
    rmsnorm(z_.data(), S.nz.data(), D, 1e-6f, u_.data());
    rmsnorm(e_.data(), S.ne.data(), D, 1e-6f, u_.data() + D);
    mat(S.up, S.up16, H, 2 * D, xu_, u_.data(), a_.data());
    for (int i = 0; i < H; i++) {
      const float v = a_[i] + S.ub[i];
      a_[i] = v / (1.f + std::exp(-v));
    }
    std::vector<float> dz(D);
    mat(S.dn, S.dn16, D, H, xa_, a_.data(), dz.data());
    for (int i = 0; i < D; i++) z_[i] += dz[i] + S.db[i];
    auto tc = std::chrono::steady_clock::now();
    ms_mlp += std::chrono::duration<double, std::milli>(tc - tb).count();
    int id = -1;
    float c = 0;
    if (lowrank) {
      mat(proj_, proj16_, R, D, xz_, z_.data(), r_.data());
      const int vo = vocab_out > 0 ? std::min(vocab_out, V) : V;
      vocab_rows(vo, &id, &c);
    } else if (head) {
      id = head(z_.data(), &c);
    }
    ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc).count();
    n_drafts_run++;
    if (id < 0) break;
    out[nd] = id;
    conf[nd] = c;
    nd++;
    tok = id;
    if (gate && !more(nd, conf)) break;  // the next draft is computed only when the drafts so far are worth extending
  }
  ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  runs++;
  if (dump) {
    std::vector<int32_t> o(ND, -1);
    std::vector<float> c(ND, 0.f);
    for (int k = 0; k < nd; k++) o[k] = out[k], c[k] = conf[k];
    const int32_t tk = tok0, n = nd, ps = (int32_t)dump_pos;
    fwrite(&ps, 4, 1, dump);
    fwrite(h, 4, D, dump);
    fwrite(&tk, 4, 1, dump);
    fwrite(&n, 4, 1, dump);
    fwrite(o.data(), 4, ND, dump);
    fwrite(c.data(), 4, ND, dump);
  }
  return nd;
}

// The width, gated by the controller's own counters. q(j, c) = the measured share of draft j kept at confidence bin(c),
// given drafts 1..j-1 kept (the confidence itself until 8 of that bin are seen). Drafts 1..j kept with p = prod q.
// Draft j+1 is computed when p x qbar(j+1) x t1 > (est(j+2) - est(j+1)) + h, where t1 = est(1) is a plain step (the
// token it would save), est(n + 1) - est(n) the controller's cost of one more position in a pass, h the head's own ms per
// draft, qbar(j+1) the measured share of draft j+1 kept over all bins (prior 0.5); the first draft is offered at all only
// when q(1, c1) x t1 > est(2) - est(1) + h. So the thresholds are break-even acceptances from the controller's costs.
double LightHeadSource::q_of(int j, float c) const {
  const int b = bin(c);
  return acc_n_[j][b] >= 8 ? acc_k_[j][b] / acc_n_[j][b] : (double)c;
}

bool LightHeadSource::more(int nd, const float* conf) {
  if (!est) return true;
  const double t1 = est(1), hd = n_drafts_run > 0 ? (ms_embed + ms_mlp + ms_out) / n_drafts_run : 0;
  if (t1 <= 0) return false;
  double p = 1;
  for (int k = 0; k < nd; k++) p *= q_of(k, conf[k]);
  if (nd == 1 && p * t1 <= est(2) - est(1) + hd) {  // the first draft does not pay for a pass: nothing is offered
    first_declined_++;
    return false;
  }
  if (nd >= ND) return false;
  double kn = 0, kk = 0;
  for (int b = 0; b < NB; b++) kn += acc_n_[nd][b], kk += acc_k_[nd][b];
  const double qbar = kn >= 8 ? kk / kn : 0.5;
  const double e2 = est(nd + 2), e1 = est(nd + 1);
  const bool go = e2 > 0 && e1 > 0 && p * qbar * t1 > (e2 - e1) + hd;
  if (!go) stopped_early_++;
  return go;
}

bool LightHeadSource::propose(const std::vector<int32_t>& hist, int max, Draft& d) {
  d.toks.clear();
  d.conf.clear();
  const float* h = hidden ? hidden() : nullptr;
  if (!h || hist.empty()) return false;
  int32_t o[16];
  float c[16];
  dump_pos = (long)hist.size() - 1;
  int n = run(h, hist.back(), std::min(max, 16), o, c);
  if (gate && n == 1 && first_declined_ > declined_seen_) n = 0;  // the first draft did not pay: no offer
  declined_seen_ = first_declined_;
  last_conf_.clear();
  for (int k = 0; k < n; k++) {
    if (c[k] < thr) break;
    d.toks.push_back(o[k]);
    d.conf.push_back(c[k]);  // the head's own probability: the controller's prior for a cell it has not seen
    last_conf_.push_back(c[k]);
  }
  if (d.toks.empty()) return false;
  d.bucket = bin(last_conf_[0]);
  d.src = -1;
  offers++;
  return true;
}

bool LightHeadSource::ask(const std::vector<int32_t>& hist, double best_gain) {
  now_ = (long)hist.size();
  if (backoff_ < 0) backoff_ = backoff0;
  if (now_ < until_) return skipped_backoff++, false;
  if (best_gain > 0 && backoff_max > 0) return skipped_earlier++, false;  // an earlier source's draft is chosen this step
  return true;
}

void LightHeadSource::outcome(const DraftCtl& ctl, int l, double gain, bool chosen, int accepted, double step_ms) {
  const bool faster = ctl.c1 <= 0 || (double)(1 + accepted) * ctl.c1 >= step_ms;  // step_ms has the head's own time in it
  if (chosen && accepted > 0 && faster) {
    backoff_ = backoff0;
  } else {
    lost++;
    until_ = now_ + backoff_;
    backoff_ = std::min<long>(backoff_max, 2 * backoff_);
  }
}

void LightHeadSource::feedback(int offered, int accepted) {
  drafts += offered;
  kept += accepted;
  for (int k = 0; k < offered && k <= accepted && k < (int)last_conf_.size() && k < ND; k++) {
    double& cb = cal[bin(last_conf_[k])];
    cb = 0.9 * cb + 0.1 * (k < accepted ? 1.0 : 0.0);
    const int b = bin(last_conf_[k]);
    acc_n_[k][b] = 0.98 * acc_n_[k][b] + 1, acc_k_[k][b] = 0.98 * acc_k_[k][b] + (k < accepted ? 1.0 : 0.0);
  }
}

std::string LightHeadSource::stats_json() const {
  std::ostringstream o;
  o << "{\"nd\":" << ND << ",\"hidden\":" << H << ",\"rank\":" << R << ",\"lowrank\":" << (lowrank ? "true" : "false") << ",\"q8\":"
    << (q8 ? "true" : "false") << ",\"weight_mb\":" << weight_bytes() / 1e6 << ",\"runs\":" << runs << ",\"offers\":" << offers
    << ",\"drafts\":" << drafts << ",\"kept\":" << kept << ",\"ms_per_run\":" << ms / std::max<long>(1, runs)
    << ",\"drafts_per_run\":" << (double)n_drafts_run / std::max<long>(1, runs) << ",\"first_declined\":" << first_declined_
    << ",\"stopped_early\":" << stopped_early_ << ",\"break_even_first\":"
    << (est && est(1) > 0 ? (est(2) - est(1) + (n_drafts_run ? (ms_embed + ms_mlp + ms_out) / n_drafts_run : 0)) / est(1) : -1.0) << ",\"lost\":" << lost << ",\"skipped_backoff\":"
    << skipped_backoff << ",\"skipped_earlier\":" << skipped_earlier << ",\"ms_per_draft\":{\"embedding_row\":"
    << ms_embed / std::max<long>(1, n_drafts_run) << ",\"mlp\":" << ms_mlp / std::max<long>(1, n_drafts_run) << ",\"output_layer\":"
    << ms_out / std::max<long>(1, n_drafts_run) << "}}";
  return o.str();
}

// ------------------------------------------------------------------------------------------------ controller
const double* DraftCtl::cost_prior(const std::string& family) {
  // pass of n positions / one position, measured on the base M4 (dmeng step costs; the controller replaces them with
  // its own measurements as it runs)
  static const double ink[DMAX + 1] = {0, 1.0, 1.64, 2.07, 2.50, 2.95, 3.41, 3.86, 4.32};
  static const double q36[DMAX + 1] = {0, 1.0, 1.64, 2.07, 2.50, 2.95, 3.41, 3.86, 4.32};
  return family == "qwen3_5_moe" ? q36 : ink;
}

void DraftCtl::init(const std::vector<int>& nb, const double* prior) {
  pa.assign(nb.size(), {});
  for (size_t s = 0; s < nb.size(); s++) {
    std::array<double, DMAX> a;
    a.fill(-1.0);
    pa[s].assign(nb[s], a);
  }
  for (int k = 0; k <= DMAX; k++) rr[k] = prior[k], seen[k] = 0;
  c1 = 0;
  offered = declined = probes = since_probe = 0;
}

double DraftCtl::expected(int s, const Draft& d, int l) const {
  double e = 1, pr = 1;
  for (int k = 1; k <= l; k++) {
    const double q = pa[s][d.bucket][k] >= 0 ? pa[s][d.bucket][k] : d.conf[k - 1];
    pr *= q;
    e += pr;
  }
  return e;
}

int DraftCtl::choose(int s, const Draft& d, int max_len, double* gain) {
  const int L = std::min<int>(max_len, (int)d.toks.size());
  if (fixed) {
    if (gain) *gain = (double)L;
    return L;
  }
  double best = 1.0;
  int bl = 0;
  for (int l = 1; l <= L; l++) {
    const double v = expected(s, d, l) / rr[l + 1];
    if (v > best) best = v, bl = l;
  }
  if (gain) *gain = best;
  return bl;
}

void DraftCtl::accepted(int s, const Draft& d, int l, int acc) {
  for (int k = 1; k <= l && k <= acc + 1; k++) {
    double& c = pa[s][d.bucket][k];
    if (c < 0) c = d.conf[k - 1];
    c = decay * c + (1 - decay) * (k <= acc ? 1.0 : 0.0);
  }
}

void DraftCtl::cost(int n, double ms) {
  if (n < 1 || n > DMAX) return;
  if (++seen[n] <= 2) return;  // a width's first passes allocate buffers: not its cost
  if (n == 1) c1 = c1 > 0 ? 0.8 * c1 + 0.2 * ms : ms;
  else if (c1 > 0) rr[n] = 0.8 * rr[n] + 0.2 * (ms / c1);
  // widths not measured yet: the measured passes' cost per extra position (their mean slope), not the prior's (a pass's
  // cost depends on how much of its union is named, which the prior table, measured without names, does not know)
  double sl = 0, w = 0;
  for (int k = 2; k <= DMAX; k++)
    if (seen[k] > 2) sl += (rr[k] - 1.0) / (k - 1) * seen[k], w += seen[k];
  if (w > 0)
    for (int k = 2; k <= DMAX; k++)
      if (seen[k] <= 2) rr[k] = 1.0 + sl / w * (k - 1);
}

// ------------------------------------------------------------------------------------------------ the decode step
void Drafter::init(const std::string& family, Core* c) {
  srcs_ = {&lookup, &skeleton};
  for (auto* s : extra) srcs_.push_back(s);
  std::vector<int> nb;
  for (auto* s : srcs_) nb.push_back(s->buckets());
  ctl.init(nb, DraftCtl::cost_prior(family));
  ctl.probe_every = o.probe_every;
  ctl.fixed = o.mode == "fixed";
  o.max = std::max(1, std::min(o.max, DraftCtl::DMAX - 1));
  core = c;
  if (core && o.early > 0) core->pick_log = &pick_log;
  const double* pr = DraftCtl::cost_prior(family);
  for (int k = 0; k <= DraftCtl::DMAX; k++) rho0_[k] = pr[k];
}

void Drafter::reset_stats() {
  steps = tokens = verifies = drafted = acc_total = early_names = early_cancelled = prenamed = explored = probes_held = 0;
  for (int k = 0; k < 2; k++) rd_steps[k] = rd_tok[k] = rd_miss[k] = rd_late[k] = rd_urg[k] = rd_pf[k] = rd_pos[k] = 0;
  ms_steps = 0;
  std::fill(std::begin(by_n), std::end(by_n), 0);
  std::fill(std::begin(by_src_steps), std::end(by_src_steps), 0);
  std::fill(std::begin(by_src_acc), std::end(by_src_acc), 0);
  ctl.offered = ctl.declined = ctl.probes = 0;
}

void Drafter::end() {
  pend_n_ = 0;  // a pass whose next step is in another generation is not measured
  prev_pass_ = false;
  if (!core) return;
  for (auto& kv : ea_held_)
    for (auto& x : kv.second) core->cache.unpin_pf(x[0], x[1], x[2]);
  ea_held_.clear();
}

// the step runs positions p .. p + L; the source positions src - 1 .. ran the same tokens before, so their released picks
// are named for this step's positions and the `early` after them (the next step's, if this step's drafts are accepted).
// Names no longer wanted are unpinned. Only reads move: the released router decides.
void Drafter::early(const Draft& d, int L, long p) {
  Core& C = *core;
  std::vector<std::pair<long, long>> want;
  if (d.src >= 1)
    for (long tp = p; tp <= p + L + o.early; tp++) {
      const long sp = d.src - 1 + (tp - p);
      if (sp < p) want.push_back({tp, sp});
    }
  for (auto it = ea_held_.begin(); it != ea_held_.end();) {
    if (std::find(want.begin(), want.end(), it->first) == want.end()) {  // (e.g. past a rejected draft): not read if queued
      for (auto& x : it->second) early_cancelled += C.cache.drop_name(x[0], x[1], x[2]);
      it = ea_held_.erase(it);
    } else {
      ++it;
    }
  }
  const size_t row = (size_t)C.steps * C.K;
  for (auto& w : want) {
    if (ea_held_.count(w) || (size_t)(w.second + 1) * row > pick_log.size()) continue;
    const int16_t* pk = pick_log.data() + (size_t)w.second * row;
    if (pk[0] < 0) continue;  // a position that never ran in this process
    auto& lst = ea_held_[w];
    for (int s = 0; s < C.steps; s++) {
      const int Ly = C.mi.step_layer[s];
      for (int j = 0; j < C.K; j++) {
        const int e = pk[(size_t)s * C.K + j];
        if (e < 0) continue;
        const int sl = C.cache.name(Ly, e, (C.tok_clock + (w.first - p)) * C.steps + s);
        if (sl >= 0) lst.push_back({Ly, sl, e});
        early_names++;
      }
    }
  }
}

bool Drafter::Fit::solve(double a0, double b0, double pw, double& a, double& b) const {
  // least squares of y = a + b x with pw pseudo-observations of the prior line at x = 0 and x = 5
  const double W = sw + 2 * pw, X = sx + pw * 5, Y = sy + pw * (2 * a0 + 5 * b0), XX = sxx + pw * 25, XY = sxy + pw * 5 * (a0 + 5 * b0);
  const double den = W * XX - X * X;
  if (den <= 1e-9) return false;
  b = (W * XY - X * Y) / den;
  a = (Y - b * X) / W;
  return true;
}

void Drafter::measure(int n, int tok, double cmp, double wait, double urg, double tot) {
  gen_steps_++;
  WidthStat& W = ws[n];
  W.n++;
  W.tokens += tok;
  W.s_cmp += cmp, W.s_wait += wait, W.s_urg += urg, W.s_tot += tot;
  if (gen_steps_ > 2) {
    if (n == 1) {
      c1 = plain_n == 0 ? cmp : 0.9 * c1 + 0.1 * cmp;
      plain_n++;
      if (pend_n_ > 1) {  // the step after a pass: what the pass left it to wait for
        const double x = ws[1].wait >= 0 ? wait - ws[1].wait : 0;
        nx_s = 0.95 * nx_s + x, nx_w = 0.95 * nx_w + 1;
      } else {  // a plain step after a plain step: the baseline (decayed means)
        W.d_n = 0.98 * W.d_n + 1, W.d_wait = 0.98 * W.d_wait + wait, W.d_urg = 0.98 * W.d_urg + urg, W.d_tot = 0.98 * W.d_tot + tot;
        W.wait = W.d_wait / W.d_n, W.urg = W.d_urg / W.d_n, W.rho = 1;
      }
    } else if (c1 > 0 && W.n > 1) {  // (a width's first pass allocates its buffers: not its cost)
      // pooled over every width: compute ratio and urgent MB as lines in the width, wait per urgent MB as a ratio of sums
      const double x = n - 2;
      f_rho.add(x, cmp / c1 - 1, 0.97);
      f_urg.add(x, urg, 0.97);
      f_wait.add(x, wait, 0.97);
      k_wait = 0.97 * k_wait + wait, k_urg = 0.97 * k_urg + urg;
      W.rho = cmp / c1, W.wait = wait, W.urg = urg;  // (the last pass, for the statistics)
      W.d_n = 0.95 * W.d_n + 1, W.d_cmp = 0.95 * W.d_cmp + cmp, W.d_ref = 0.95 * W.d_ref + c1, W.d_wait = 0.95 * W.d_wait + wait;
    }
  }
  pend_n_ = n;
}

double Drafter::est(int n) const {
  if (c1 <= 0 || plain_n < 2) return -1;
  const double w1 = std::max(0.0, ws[1].wait), u1 = std::max(0.0, ws[1].urg);
  if (n <= 1) return c1 + w1;
  // compute: rho(n) = 1 + a + b (n - 2), fitted over every pass of every width (prior: the family's table, weight 3)
  double a = rho0_[2] - 1, b = rho0_[3] - rho0_[2];
  f_rho.solve(rho0_[2] - 1, rho0_[3] - rho0_[2], 3, a, b);
  const double rho = 1 + std::max(0.05, a) + std::max(0.05, b) * (n - 2);
  // wait: urgent MB U(n) = ua + ub (n - 2) fitted over every pass (prior: each drafted position reads two plain steps'
  // urgent MB, weight 2), times the passes' wait per urgent MB (prior: plain steps', weight 5 MB)
  double ua = 3 * u1, ub = 2 * u1;
  f_urg.solve(3 * u1, 2 * u1, 2, ua, ub);
  const double U = std::max(0.0, ua + std::max(0.0, ub) * (n - 2));
  // the wait itself as a line in the width, fitted over every pass (prior: those urgent MB at the plain steps' wait per
  // urgent MB, weight 2): small urgent reads are latency-bound, so a wait per MB measured on two-position probes does not
  // carry to wide passes, whose reads run side by side
  const double kp = u1 > 0.5 ? w1 / u1 : 0.3;
  double wa = kp * std::max(0.0, ua), wb = kp * std::max(0.0, ub);
  f_wait.solve(wa, wb, 2, wa, wb);
  const double wt = std::max(0.0, wa + std::max(0.0, wb) * (n - 2));
  // what a pass leaves the next step (pooled; may be negative: a pass can read what the next step needs)
  const double nx = nx_w > 0 ? nx_s / nx_w : 0;
  (void)U;
  // a width with three or more recent passes: its own decayed ratio of sums and mean wait (the lines extrapolate from
  // other widths; a wide pass of a copied span is not a narrow probe's line continued)
  const WidthStat& M = ws[std::min(n, DraftCtl::DMAX)];
  if (M.d_n >= 3 && M.d_ref > 0) return c1 * (M.d_cmp / M.d_ref) + M.d_wait / M.d_n + std::max(-w1, nx);
  return c1 * rho + wt + std::max(-w1, nx);
}

bool Drafter::probe_ok(int l) const {
  // what a probe of l drafts may cost beyond plain steps if only its first token is kept: est(l + 1) - est(1) ms, and
  // MB: the measured excess per drafted position of earlier probes (prior: two plain steps' urgent MB per position)
  const double t1 = est(1), tl = est(l + 1);
  if (t1 <= 0 || tl <= 0) return false;
  const double xb = (pb_xb >= 0 ? pb_xb : 2 * std::max(0.0, ws[1].urg)) * l;
  return pb_ms >= tl - t1 && pb_mb >= xb;
}

int Drafter::step(DraftTarget& T, const std::vector<int32_t>& hist, int vocab, int budget, const std::vector<int>& stop, std::vector<int>& out) {
  auto t0 = std::chrono::steady_clock::now();
  const int last = hist.back();
  auto is_stop = [&](int g) { return std::find(stop.begin(), stop.end(), g) != stop.end(); };
  Draft d, dl;
  int bs = -1, L = 0;
  double t_timed = 0;                        // proposal ms of the timed sources (left out of the pass costs)
  std::vector<std::array<double, 3>> asked;  // (source, chosen length, gain) of each source that proposed
  double bg = 0;  // the best offer's value (expected tokens per estimated ms over a plain step's)
  if (on() && budget > 1) {
    const int maxL = std::min(o.max, budget - 1);
    int bl = 0;
    Draft cand;
    for (int s = 0; s < (int)srcs_.size(); s++) {
      if (srcs_[s] == &skeleton && skeleton.empty()) continue;
      if (srcs_[s] == &lookup && !o.lookup) continue;
      if (!srcs_[s]->ask(hist, bs >= 0 && bl > 0 ? bg : 0)) continue;
      auto tp = std::chrono::steady_clock::now();
      const bool got = srcs_[s]->propose(hist, maxL, cand) && !cand.toks.empty();
      if (srcs_[s]->timed) t_timed += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp).count();
      if (!got) {
        if (srcs_[s]->timed) asked.push_back({(double)s, 0, 1.0});  // ran, no draft: a lost try
        continue;
      }
      if (s == 0) dl = cand;
      double g = 0;
      int l = 0;
      if (o.gate && !ctl.fixed) {  // the cost model: expected tokens per estimated ms, against a plain step
        const double t1 = est(1);
        const int Lm = std::min<int>(maxL, (int)cand.toks.size());
        double best = 0;
        for (int k = 1; k <= Lm && t1 > 0; k++) {
          const double tk = est(k + 1);
          const double v = tk > 0 ? ctl.expected(s, cand, k) * t1 / tk : 0;
          if (v > best) best = v, l = k;
        }
        if (best < 1 + o.margin) l = 0;
        // exploration: every 6th pass that follows a fully kept one runs two drafts more than the estimate chose, so the
        // fitted lines see more than one width (a run of equal passes leaves the slope to the prior)
        if (l > 0 && l < Lm && last_full_ && ++explore_ % 6 == 0 && probe_ok(2)) l = std::min(Lm, l + 2), explored++, gamble_ = true;
        g = best;
      } else {
        l = ctl.choose(s, cand, maxL, &g);
      }
      asked.push_back({(double)s, (double)l, g});
      if (bs < 0 || g > bg || (g == bg && l > bl)) bs = s, bg = g, bl = l, d = cand;
    }
    if (bs >= 0) {
      ctl.offered++;
      L = bl;
      if (L == 0) {
        ctl.declined++;
        // a probe: one draft (the narrowest pass; a rejected probe pays two positions, not the controller's full length)
        // the probe's length: the most drafts the learned acceptance expects all kept with probability >= 0.5 (at least one;
        // a draft index never run is not counted: wide probes from the prior alone failed on the 35B's coding answer):
        // a long copied span is probed wide, where its value is (a two-position probe of it measures a pass that never
        // pays), an uncertain proposal with one draft
        if (est(1) > 0 && ++ctl.since_probe >= ctl.probe_every) {
          int lp = 1;
          double pr = 1;
          for (int k = 1; k <= std::min<int>(maxL, (int)d.toks.size()); k++) {
            const double q = ctl.pa[bs][d.bucket][k];  // learned only: an unseen index stops the probe there
            if (q < 0) break;
            pr *= q;
            if (pr >= 0.5) lp = k;
            else break;
          }
          if (probe_ok(lp)) {
            L = lp, ctl.since_probe = 0, ctl.probes++, gamble_ = true;
          } else {
            ctl.since_probe = ctl.probe_every;  // due: runs once the budget allows
            probes_held++;
          }
        }
      }
    }
  }
  const long p = T.pos();
  // early names: for a pass, its positions and `early` past them (the next pass's); for a plain step, only when a probe is
  // due next: the next positions of the copied span, so the probe finds its drafted position's experts read, as a pass in
  // a run of passes does (a probe without that lead waits for all of them and measures a pass sustained drafting never
  // runs). Otherwise none: names for a declined draft are reads a plain step pays for (naming every step whose offer was
  // near paying read 85 MB per token more on Inkling's file rewrite at 7.8 GB for one pass).
  const bool pre = L == 0 && bs >= 0 && o.gate && !ctl.fixed && ctl.since_probe + 1 >= ctl.probe_every && probe_ok(1);
  if (pre) prenamed++;
  if (core && o.early > 0) early(L > 0 ? d : pre ? (dl.src >= 1 ? dl : d) : Draft(), L, p);
  int n = 1, acc = 0;
  uint64_t r0[4] = {0, 0, 0, 0};
  double w0 = 0;
  if (reads) {
    const auto& S = reads->cache.st;
    r0[0] = S.n_miss, r0[1] = S.n_late, r0[2] = S.bytes_urgent, r0[3] = S.bytes_pf;
    w0 = S.wait_ms;
  }
  if (L == 0) {
    T.step(last);
    out.push_back(pick ? pick(T.logits(), 0) : argmax(T.logits(), vocab));
  } else {
    int toks[DraftCtl::DMAX];
    toks[0] = last;
    for (int i = 0; i < L; i++) toks[i + 1] = d.toks[i];
    n = 1 + L;
    if (!T.verify(toks, n)) {  // no multi-position verify: plain decoding from here on
      o.mode = "none";
      n = 1;
      T.step(last);
      out.push_back(pick ? pick(T.logits(), 0) : argmax(T.logits(), vocab));
    } else {
      int g = pick ? pick(T.logits_at(0), 0) : argmax(T.logits_at(0), vocab);
      out.push_back(g);
      int produced = 1;
      while (acc < L && d.toks[acc] == g && produced < budget && !is_stop(g)) {
        acc++;
        g = pick ? pick(T.logits_at(acc), acc) : argmax(T.logits_at(acc), vocab);
        out.push_back(g);
        produced++;
      }
      T.commit(1 + acc);
      last_full_ = acc == L;
      ctl.accepted(bs, d, L, acc);
      srcs_[bs]->feedback(L, acc);
      verifies++;
      drafted += L;
      acc_total += acc;
      if (bs < 4) by_src_steps[bs]++, by_src_acc[bs] += acc;
    }
  }
  if (core && o.early > 0)  // names of committed positions are spent
    for (auto it = ea_held_.begin(); it != ea_held_.end();) {
      if (it->first.first < T.pos()) {
        for (auto& x : it->second) core->cache.unpin_pf(x[0], x[1], x[2]);
        it = ea_held_.erase(it);
      } else {
        ++it;
      }
    }
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  for (auto& x : asked) {
    const int s = (int)x[0];
    const bool chosen = s == bs && n > 1;
    srcs_[s]->outcome(ctl, (int)x[1], x[2], chosen, chosen ? acc : 0, ms);
  }
  if (on()) {
    double wt = 0, ur = 0, tt = 0;
    if (reads) {
      const auto& S = reads->cache.st;
      wt = S.wait_ms - w0;
      ur = (double)(S.bytes_urgent - r0[2]) / 1e6;
      tt = ur + (double)(S.bytes_pf - r0[3]) / 1e6;
    }
    ctl.cost(n, ms - t_timed);
    // the probe budget: credit 1% of every step's ms and MB; a probe, an exploration or a step that named ahead for one is
    // debited what it cost beyond plain steps for the tokens it produced (kept drafts pay it back); a chosen pass's gain
    // over plain steps refills the credit
    const double sms = std::max(0.0, ms - t_timed), t1 = est(1), m1 = ws[1].d_n > 0 ? ws[1].d_tot / ws[1].d_n : -1;
    pb_ms += 0.01 * sms, pb_mb += 0.01 * tt;
    if (t1 > 0 && m1 >= 0) {
      const double xm = sms - (1 + acc) * t1, xb = tt - (1 + acc) * m1;
      if (gamble_ || pre) {
        pb_ms -= xm, pb_mb -= xb;
        if (n > 1 && L > 0) pb_xb = pb_xb < 0 ? xb / L : 0.8 * pb_xb + 0.2 * xb / L;
      } else if (n > 1) {
        pb_ms += std::max(0.0, -xm), pb_mb += std::max(0.0, -xb);
      }
    }
    gamble_ = false;
    measure(n, 1 + acc, std::max(0.0, ms - t_timed - wt), wt, ur, tt);
  }
  if (reads) {
    const auto& S = reads->cache.st;
    const int k = n > 1;
    rd_steps[k] += 1;
    rd_pos[k] += n;
    rd_tok[k] += 1 + acc;
    rd_miss[k] += (double)(S.n_miss - r0[0]);
    rd_late[k] += (double)(S.n_late - r0[1]);
    rd_urg[k] += (double)(S.bytes_urgent - r0[2]);
    rd_pf[k] += (double)(S.bytes_pf - r0[3]);
  }
  ms_steps += ms;
  steps++;
  tokens += 1 + acc;
  by_n[n]++;
  last_n = n;
  return 1 + acc;
}

std::string Drafter::stats_json() const {
  std::ostringstream o2;
  o2.setf(std::ios::fixed);
  o2.precision(4);
  o2 << "{\"mode\":\"" << o.mode << "\",\"max\":" << o.max << ",\"steps\":" << steps << ",\"tokens\":" << tokens
     << ",\"tokens_per_step\":" << (double)tokens / std::max<long>(1, steps) << ",\"verifies\":" << verifies << ",\"drafted\":" << drafted
     << ",\"accepted\":" << acc_total << ",\"prenamed_steps\":" << prenamed << ",\"explored\":" << explored << ",\"probes_held\":" << probes_held << ",\"probe_budget_ms\":" << pb_ms << ",\"probe_budget_mb\":" << pb_mb << ",\"offered\":" << ctl.offered << ",\"declined\":" << ctl.declined << ",\"probes\":" << ctl.probes
     << ",\"steps_by_positions\":[";
  for (int k = 1; k <= DraftCtl::DMAX; k++) o2 << (k > 1 ? "," : "") << by_n[k];
  o2 << "],\"verifies_by_source\":{\"lookup\":" << by_src_steps[0] << ",\"skeleton\":" << by_src_steps[1] << ",\"extra\":" << by_src_steps[2]
     << "},\"accepted_by_source\":{\"lookup\":" << by_src_acc[0] << ",\"skeleton\":" << by_src_acc[1] << ",\"extra\":" << by_src_acc[2]
     << "},\"cost_ratio\":[";
  for (int k = 1; k <= DraftCtl::DMAX; k++) o2 << (k > 1 ? "," : "") << ctl.rr[k];
  o2 << "],\"one_position_ms\":" << ctl.c1 << ",\"ms_per_step\":" << ms_steps / std::max<long>(1, steps);
  o2 << ",\"model\":{\"on\":" << (o.gate ? "true" : "false") << ",\"margin\":" << o.margin << ",\"plain_compute_ms\":" << c1
     << ",\"plain_wait_ms\":" << ws[1].wait << ",\"plain_mb_urgent\":" << ws[1].urg << ",\"widths\":[";
  bool first = true;
  for (int k = 1; k <= DraftCtl::DMAX; k++) {
    const WidthStat& W = ws[k];
    if (!W.n) continue;
    const double tk = (double)std::max<long>(1, W.tokens);
    o2 << (first ? "" : ",") << "{\"positions\":" << k << ",\"passes\":" << W.n << ",\"tokens_per_pass\":" << (double)W.tokens / W.n
       << ",\"compute_ms_per_token\":" << W.s_cmp / tk << ",\"wait_ms_per_token\":" << W.s_wait / tk << ",\"mb_urgent_per_token\":" << W.s_urg / tk
       << ",\"mb_total_per_token\":" << W.s_tot / tk << ",\"rho\":" << W.rho << ",\"wait_ms\":" << W.wait << ",\"mb_urgent\":" << W.urg
       << ",\"est_ms\":" << est(k) << "}";
    first = false;
  }
  {
    double ra = 0, rb = 0, ua = 0, ub = 0;
    f_rho.solve(rho0_[2] - 1, rho0_[3] - rho0_[2], 3, ra, rb);
    f_urg.solve(3 * std::max(0.0, ws[1].urg), 2 * std::max(0.0, ws[1].urg), 2, ua, ub);
    o2 << "],\"fit\":{\"rho_at2\":" << 1 + ra << ",\"rho_per_position\":" << rb << ",\"mb_urgent_at2\":" << ua << ",\"mb_urgent_per_position\":" << ub
       << ",\"wait_ms_per_urgent_mb\":" << k_wait / std::max(1e-9, k_urg) << ",\"wait_ms_at2\":" << [&] {
         double a = 0, b = 0;
         f_wait.solve(0, 0, 0, a, b);
         return a;
       }() << ",\"next_excess_wait_ms\":" << (nx_w > 0 ? nx_s / nx_w : 0) << "}}";
  }
  if (o.early > 0)
    o2 << ",\"early\":" << o.early << ",\"early_names_per_step\":" << (double)early_names / std::max<long>(1, steps)
       << ",\"early_reads_cancelled\":" << early_cancelled;
  if (reads) {
    o2 << ",\"reads\":{";
    for (int k = 0; k < 2; k++) {
      const double st = std::max(1.0, rd_steps[k]), tk = std::max(1.0, rd_tok[k]);
      o2 << (k ? ",\"verify\":" : "\"plain\":") << "{\"steps\":" << rd_steps[k] << ",\"positions\":" << rd_pos[k] << ",\"tokens\":" << rd_tok[k]
         << ",\"misses_per_step\":" << rd_miss[k] / st << ",\"late_per_step\":" << rd_late[k] / st << ",\"misses_per_token\":" << rd_miss[k] / tk
         << ",\"mb_urgent_per_token\":" << rd_urg[k] / 1e6 / tk << ",\"mb_named_per_token\":" << rd_pf[k] / 1e6 / tk
         << ",\"mb_total_per_token\":" << (rd_urg[k] + rd_pf[k]) / 1e6 / tk << ",\"mb_urgent_per_step\":" << rd_urg[k] / 1e6 / st << "}";
    }
    const double tk = std::max(1.0, rd_tok[0] + rd_tok[1]);
    o2 << ",\"all\":{\"tokens\":" << rd_tok[0] + rd_tok[1] << ",\"misses_per_token\":" << (rd_miss[0] + rd_miss[1]) / tk
       << ",\"mb_urgent_per_token\":" << (rd_urg[0] + rd_urg[1]) / 1e6 / tk << ",\"mb_total_per_token\":" << (rd_urg[0] + rd_urg[1] + rd_pf[0] + rd_pf[1]) / 1e6 / tk
       << "}}";
  }
  o2 << "}";
  return o2.str();
}

}  // namespace dm
