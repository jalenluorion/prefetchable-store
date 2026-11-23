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
}

void Drafter::reset_stats() {
  steps = tokens = verifies = drafted = acc_total = early_names = early_cancelled = 0;
  for (int k = 0; k < 2; k++) rd_steps[k] = rd_tok[k] = rd_miss[k] = rd_late[k] = rd_urg[k] = rd_pf[k] = rd_pos[k] = 0;
  ms_steps = 0;
  std::fill(std::begin(by_n), std::end(by_n), 0);
  std::fill(std::begin(by_src_steps), std::end(by_src_steps), 0);
  std::fill(std::begin(by_src_acc), std::end(by_src_acc), 0);
  ctl.offered = ctl.declined = ctl.probes = 0;
}

void Drafter::end() {
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

int Drafter::step(DraftTarget& T, const std::vector<int32_t>& hist, int vocab, int budget, const std::vector<int>& stop, std::vector<int>& out) {
  auto t0 = std::chrono::steady_clock::now();
  const int last = hist.back();
  auto is_stop = [&](int g) { return std::find(stop.begin(), stop.end(), g) != stop.end(); };
  Draft d, dl;
  int bs = -1, L = 0;
  double t_timed = 0;                        // proposal ms of the timed sources (left out of the pass costs)
  std::vector<std::array<double, 3>> asked;  // (source, chosen length, gain) of each source that proposed
  if (on() && budget > 1) {
    const int maxL = std::min(o.max, budget - 1);
    double bg = 0;
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
      const int l = ctl.choose(s, cand, maxL, &g);
      asked.push_back({(double)s, (double)l, g});
      if (bs < 0 || g > bg || (g == bg && l > bl)) bs = s, bg = g, bl = l, d = cand;
    }
    if (bs >= 0) {
      ctl.offered++;
      L = bl;
      if (L == 0) {
        ctl.declined++;
        if (++ctl.since_probe >= ctl.probe_every) L = std::min<int>(2, (int)d.toks.size()), ctl.since_probe = 0, ctl.probes++;
      }
    }
  }
  const long p = T.pos();
  if (core && o.early > 0) early(L > 0 ? d : dl, L, p);
  int n = 1, acc = 0;
  uint64_t r0[4] = {0, 0, 0, 0};
  if (reads) {
    const auto& S = reads->cache.st;
    r0[0] = S.n_miss, r0[1] = S.n_late, r0[2] = S.bytes_urgent, r0[3] = S.bytes_pf;
  }
  if (L == 0) {
    T.step(last);
    out.push_back(argmax(T.logits(), vocab));
  } else {
    int toks[DraftCtl::DMAX];
    toks[0] = last;
    for (int i = 0; i < L; i++) toks[i + 1] = d.toks[i];
    n = 1 + L;
    if (!T.verify(toks, n)) {  // no multi-position verify: plain decoding from here on
      o.mode = "none";
      n = 1;
      T.step(last);
      out.push_back(argmax(T.logits(), vocab));
    } else {
      int g = argmax(T.logits_at(0), vocab);
      out.push_back(g);
      int produced = 1;
      while (acc < L && d.toks[acc] == g && produced < budget && !is_stop(g)) {
        acc++;
        g = argmax(T.logits_at(acc), vocab);
        out.push_back(g);
        produced++;
      }
      T.commit(1 + acc);
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
  if (on()) ctl.cost(n, ms - t_timed);
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
     << ",\"accepted\":" << acc_total << ",\"offered\":" << ctl.offered << ",\"declined\":" << ctl.declined << ",\"probes\":" << ctl.probes
     << ",\"steps_by_positions\":[";
  for (int k = 1; k <= DraftCtl::DMAX; k++) o2 << (k > 1 ? "," : "") << by_n[k];
  o2 << "],\"verifies_by_source\":{\"lookup\":" << by_src_steps[0] << ",\"skeleton\":" << by_src_steps[1] << ",\"extra\":" << by_src_steps[2]
     << "},\"accepted_by_source\":{\"lookup\":" << by_src_acc[0] << ",\"skeleton\":" << by_src_acc[1] << ",\"extra\":" << by_src_acc[2]
     << "},\"cost_ratio\":[";
  for (int k = 1; k <= DraftCtl::DMAX; k++) o2 << (k > 1 ? "," : "") << ctl.rr[k];
  o2 << "],\"one_position_ms\":" << ctl.c1 << ",\"ms_per_step\":" << ms_steps / std::max<long>(1, steps);
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
