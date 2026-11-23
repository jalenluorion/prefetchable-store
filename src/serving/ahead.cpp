// Idle-time prefill (src/serving/ahead.h).
#include "ahead.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>

#include "dmeng/sampler.h"

namespace dm {

using sclk = std::chrono::steady_clock;
static double ms_from_(sclk::time_point t) { return std::chrono::duration<double, std::milli>(sclk::now() - t).count(); }

void AheadRunner::submit(long id, std::vector<int> ids, std::vector<int> marks) {
  std::lock_guard<std::mutex> lk(mu_);
  id_ = id;
  ids_ = std::move(ids);
  marks_ = std::move(marks);
  stop_.store(false);
  has_.store(true);
}

void AheadRunner::on_request() {
  std::lock_guard<std::mutex> lk(mu_);
  has_.store(false);
  if (running_.load()) {
    t_stop_ = sclk::now();
    stop_.store(true);
    lm_.cancel();  // the progress hook repeats it if this landed before the prefill call reset its flag
  }
}

std::string AheadRunner::run() {
  std::vector<int> ids, marks;
  long id;
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!has_.load()) return "";
    ids = ids_;
    marks = marks_;
    id = id_;
    has_.store(false);  // one attempt per submit; the front end submits again when it is still wanted
    stop_.store(false);
    running_.store(true);
  }
  std::string e;
  if (span_ > 0) lm_.model().set_option("prefill_span", std::to_string(span_), e);
  PrefillProgress prev = lm_.on_progress;
  lm_.on_progress = [this, prev](int done, int total, double mb) {
    if (prev) prev(done, total, mb);
    if (stop_.load()) lm_.cancel();
  };
  auto t0 = sclk::now();
  SyncResult r = pc_.ahead(ids, marks, park_old_);
  const double ms = ms_from_(t0);
  lm_.on_progress = prev;
  if (span_ > 0) lm_.model().set_option("prefill_span", std::to_string(normal_span_), e);
  double ms_stop = -1;
  {
    std::lock_guard<std::mutex> lk(mu_);
    running_.store(false);
    if (stop_.load()) ms_stop = ms_from_(t_stop_);  // from the request's arrival to the end of the cancelled run
  }
  n_runs_++;
  ms_total_ += r.ms_prefill;
  std::ostringstream s;
  s << "A " << id << " {\"predicted\":" << ids.size() << ",\"lcp_live\":" << r.lcp_live << ",\"live_len\":" << r.live_len << ",\"reused\":" << r.reused
    << ",\"computed\":" << r.computed << ",\"source\":\"" << r.source << "\",\"cancelled\":" << (r.cancelled ? "true" : "false")
    << ",\"ms\":" << ms << ",\"ms_prefill\":" << r.ms_prefill << ",\"ms_restore\":" << r.ms_restore << ",\"ms_cancel_to_stop\":" << ms_stop
    << ",\"mb_read\":" << r.pf.mb_read << ",\"span\":" << span_ << ",\"pos\":" << lm_.pos() << ",\"runs\":" << n_runs_ << ",\"ms_prefill_total\":" << ms_total_
    << ",\"prefill\":" << (r.pf.json.empty() ? "{}" : r.pf.json) << "}";
  return s.str();
}

// ---------------------------------------------------------------------------------------------------- the check
namespace {
struct Snap {
  std::vector<float> lg;
  std::vector<uint8_t> fx, pd;
  std::vector<int> gen;
};
void set_prefill(Model& m, int chunk, int span, int stream) {
  std::string e;
  m.set_option("prefill_chunk", std::to_string(chunk), e);
  m.set_option("prefill_span", std::to_string(span), e);
  m.set_option("prefill_stream", std::to_string(stream), e);
  m.set_option("prefill_batched_always", "0", e);
}
// the state now (after the prompt), then the greedy continuation (which moves the state on)
void snap_and_gen(LM& lm, PrefixCache* pc, std::vector<float>& lg, int gen, Snap& s) {
  s.lg = lg;
  s.fx.resize(lm.info().fixed_bytes);
  lm.save_fixed(s.fx.data());
  s.pd.resize((size_t)lm.pos() * lm.info().pos_bytes);
  lm.save_pos(0, lm.pos(), s.pd.data());
  std::mt19937_64 rng(0);
  for (int i = 0; i < gen; i++) {
    int t = dm::sample(lg.data(), (int)lg.size(), 0, 0, 1, rng);
    s.gen.push_back(t);
    if (pc) pc->step(t, lg.data());
    else lm.step(t, lg.data());
  }
}
std::string cmp(const Snap& a, const Snap& b, bool& ok) {
  size_t lb = 0, fb = 0, pb = 0;
  for (size_t i = 0; i < a.lg.size() && i < b.lg.size(); i++) lb += memcmp(&a.lg[i], &b.lg[i], 4) != 0;
  for (size_t i = 0; i < a.fx.size() && i < b.fx.size(); i++) fb += a.fx[i] != b.fx[i];
  for (size_t i = 0; i < a.pd.size() && i < b.pd.size(); i++) pb += a.pd[i] != b.pd[i];
  const bool same = lb == 0 && fb == 0 && pb == 0 && a.pd.size() == b.pd.size() && a.gen == b.gen;
  ok = ok && same;
  std::ostringstream s;
  s << "{\"logit_bits_differ\":" << lb << ",\"fixed_state_bytes_differ\":" << fb << ",\"kv_bytes_differ\":" << pb << ",\"kv_len\":[" << a.pd.size() << ","
    << b.pd.size() << "],\"greedy_identical\":" << (a.gen == b.gen ? "true" : "false") << "}";
  return s.str();
}
}  // namespace

int check_ckpt(ModelLM& lm, const SessionOptions& so0, const std::vector<int>& L0, int chunk, int span, int stream, int gen) {
  const int V = lm.info().vocab, n = (int)L0.size(), d = n * 6 / 10;
  std::vector<int> L(L0);
  for (int& t : L) t %= V;  // (ids of a larger vocabulary folded into this model's)
  SessionOptions so = so0;
  so.disk_dir = "";  // memory checkpoints only
  std::vector<int> D(L.begin(), L.begin() + d);  // diverges from L at 60% of its length
  for (int i = d; i < n; i++) D.push_back((L[i] + 1) % V);
  std::vector<float> lg(V);
  auto cold = [&](const std::vector<int>& ids, Snap& s) {  // token by token from an empty state
    lm.reset();
    set_prefill(lm.model(), 1, 1, stream);
    lm.prefill(ids.data(), (int)ids.size(), lg.data(), {}, nullptr, nullptr);
    snap_and_gen(lm, nullptr, lg, gen, s);
  };
  Snap rl, rd;
  cold(L, rl);
  cold(D, rd);
  lm.reset();
  set_prefill(lm.model(), chunk, span, stream);
  PrefixCache pc(lm, so);
  bool ok = true;
  std::ostringstream out;
  out << "{\"mode\":\"check_ckpt\",\"model\":\"" << lm.info().name << "\",\"n\":" << n << ",\"diverge_at\":" << d << ",\"ckpt_every\":" << so.ckpt_every
      << ",\"ckpt_min\":" << so.ckpt_min;
  const char* names[3] = {"long", "identical", "diverging"};
  const std::vector<int>* reqs[3] = {&L, &L, &D};
  const Snap* refs[3] = {&rl, &rl, &rd};
  for (int k = 0; k < 3; k++) {
    SyncResult r = pc.sync(*reqs[k], {}, lg.data());
    Snap got;
    snap_and_gen(lm, &pc, lg, gen, got);  // the reply (moves the live sequence on, as a served request does)
    const bool restored = k == 0 || (r.reused > 0 && r.reused <= (k == 2 ? d : n));
    ok = ok && restored;
    out << ",\"" << names[k] << "\":{\"cmp\":" << cmp(*refs[k], got, ok) << ",\"reused\":" << r.reused << ",\"computed\":" << r.computed
        << ",\"source\":\"" << r.source << "\",\"restored_from_a_checkpoint\":" << (restored ? "true" : "false") << "}";
  }
  out << ",\"session\":" << pc.stats_json() << ",\"passed\":" << (ok ? "true" : "false") << "}";
  printf("%s\n", out.str().c_str());
  fflush(stdout);
  return ok ? 0 : 1;
}

int check_ahead(ModelLM& lm, const SessionOptions& so0, const std::vector<int>& A, const std::vector<int>& B, int ahead_len, int chunk,
                int span, int ahead_span, int stream, int gen) {
  const int V = lm.info().vocab;
  SessionOptions so = so0;
  so.disk_dir = "";  // the check never reads or writes the disk prefix
  std::vector<float> lg(V);
  // reference: B token by token from an empty state
  lm.reset();
  set_prefill(lm.model(), 1, 1, stream);
  auto t = sclk::now();
  lm.prefill(B.data(), (int)B.size(), lg.data(), {}, nullptr, nullptr);
  const double ms_ref = ms_from_(t);
  Snap ref;
  snap_and_gen(lm, nullptr, lg, gen, ref);
  bool ok = true;
  std::ostringstream out;
  out << "{\"mode\":\"check_ahead\",\"model\":\"" << lm.info().name << "\",\"a\":" << A.size() << ",\"b\":" << B.size() << ",\"ahead_len\":" << ahead_len
      << ",\"ms_ref\":" << ms_ref;
  for (int with_ahead = 0; with_ahead < 2; with_ahead++) {
    lm.reset();
    set_prefill(lm.model(), chunk, span, stream);
    PrefixCache pc(lm, so);
    std::vector<int> marks;  // A's end and B's prediction end as message boundaries
    pc.sync(A, {(int)A.size()}, lg.data());
    std::mt19937_64 rng(0);  // a few generated tokens after A, like a reply
    for (int i = 0; i < 8; i++) {
      int t2 = dm::sample(lg.data(), V, 0, 0, 1, rng);
      pc.step(t2, lg.data());
    }
    std::string a1 = "{}", a2 = "{}";
    if (with_ahead) {
      std::vector<int> S(B.begin(), B.begin() + ahead_len);
      // first attempt: cancelled once the run has passed into its second span (from the progress hook, as a request does)
      AheadRunner run(lm, pc, ahead_span, span, true);
      const int base = std::min<int>(ahead_len, [&] {
        int k = 0;
        while (k < (int)A.size() && k < ahead_len && A[k] == B[k]) k++;
        return k;
      }());
      PrefillProgress prev = lm.on_progress;
      bool fired = false;
      lm.on_progress = [&](int done, int total, double) {
        if (!fired && done > std::min(ahead_span, (ahead_len - base) / 2)) {
          fired = true;
          run.on_request();
        }
      };
      run.submit(1, S, {});
      // on_request() only stops a running job: the hook above fires inside run(), after it took the job
      a1 = run.run();
      lm.on_progress = prev;
      run.submit(2, S, {});
      a2 = run.run();
      a1 = a1.substr(a1.find('{'));
      a2 = a2.substr(a2.find('{'));
    }
    SyncResult r = pc.sync(B, {}, lg.data());
    Snap got;
    snap_and_gen(lm, &pc, lg, gen, got);
    out << ",\"" << (with_ahead ? "with_ahead" : "request_path") << "\":{\"cmp\":" << cmp(ref, got, ok) << ",\"b_reused\":" << r.reused
        << ",\"b_computed\":" << r.computed << ",\"b_source\":\"" << r.source << "\",\"ahead_cancelled_run\":" << a1 << ",\"ahead_resumed_run\":" << a2
        << ",\"session\":" << pc.stats_json() << "}";
  }
  out << ",\"passed\":" << (ok ? "true" : "false") << "}";
  printf("%s\n", out.str().c_str());
  fflush(stdout);
  return ok ? 0 : 1;
}

}  // namespace dm
