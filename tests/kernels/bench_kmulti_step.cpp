// The engine's expert step (expert_step: the same fork-join, chunking, claim rules and pool as Core::run_experts) on
// real records, with the losses against plain streaming taken out one at a time. Each "application" is one step of 8
// routed experts (the 35B: + its shared expert, synthetic AFF8 of the real shape) from a set of distinct records larger
// than the caches; its arm is drawn at random, so every arm sees the same load. Per arm: GB/s of weight bytes through
// the call(s), spin thread-ms per application.
//   bench_kmulti_step ink|q36|flash MODEL_DIR [applications] [reader GB/s]
// (flash: Qwen3.8-Flash-Next, 10 routed experts per step from store/layer_00.bin + a synthetic AFF8 shared expert of 640)
// Arms: engine (one call, as the engine's first wave) | nodep (down rows do not wait for gate-up) | noact (no activation
// or its quantisation) | ponly (efficiency-core threads claim nothing) | ptail (they stop at the down tail) | smalltail
// (single quarters from 8 per thread left) | twojoins (the step split into two calls) | split1 (every expert its own call) | warm (the previous application's
// records again) | reader (a concurrent reader: F_NOCACHE preads of another layer file at the given rate) | stream (the
// same matrices' rows claimed 64 at a time, no activation, no dependency: the kernels' streaming rate in this process).
// The name matches engine/bench/guard.py's model-process pattern, so the queue waits for it. No model process.
#include <fcntl.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "dmeng/expert_step.h"
#include "dmeng/store.h"

using namespace dm;
using sclk = std::chrono::steady_clock;

static bool read_all(int fd, uint8_t* dst, size_t n, off_t off) {
  size_t done = 0;
  while (done < n) {
    ssize_t r = pread(fd, dst + done, n - done, off + (off_t)done);
    if (r <= 0) return false;
    done += (size_t)r;
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: bench_kmulti_step ink|q36|flash MODEL_DIR [applications] [reader GB/s]\n");
    return 2;
  }
  const bool ink = std::string(argv[1]) == "ink", flash = std::string(argv[1]) == "flash";
  const std::string md = argv[2];
  const int NAPP = argc > 3 ? atoi(argv[3]) : 6000;
  const double RD_GBS = argc > 4 ? atof(argv[4]) : 2.0;
  const int BUSY = argc > 5 ? atoi(argv[5]) : 0;  // co-runners: threads spinning at default QoS for the whole run
  set_qos_interactive();
  const int K = flash ? 10 : 8;
  ExpertFormat F;
  int NREC, H, I, SH = 0;
  std::string err;
  if (ink) {
    F.fmt = Fmt::NVFP4, F.rec = 4489216, F.H = H = 2304, F.I = I = 1152, F.fused = true;
    F.gu_c = 0, F.gu_s = 2654208, F.dn_c = 2985984, F.dn_s = 4313088, F.trailer = 4478976;
    NREC = 48;  // 215 MB
  } else if (flash) {  // the published record layout (store/layer_NN.bin, 2768896 B per expert)
    F.fmt = Fmt::AFF4, F.rec = 2768896, F.H = H = 2560, F.I = I = 640, F.fused = false;
    F.gu_c = 0, F.gu_s = 819200, F.gu_b = 870400;
    F.up_c = 921600, F.up_s = 1740800, F.up_b = 1792000;
    F.dn_c = 1843200, F.dn_s = 2662400, F.dn_b = 2713600;
    SH = 640;
    NREC = 80;  // 221 MB
  } else {
    std::map<std::string, std::string> cfg;
    if (!ResFile::read_cfg(md + "/dmeng_resident.bin", cfg, err)) return fprintf(stderr, "%s\n", err.c_str()), 1;
    auto gi = [&](const std::string& k) { return atol(cfg.at(k).c_str()); };
    F.fmt = Fmt::AFF4, F.rec = gi("rec_bytes"), F.H = H = (int)gi("hidden"), F.I = I = (int)gi("moe_inter"), F.fused = false;
    F.gu_c = gi("off_gate_proj"), F.gu_s = gi("soff_gate_proj"), F.gu_b = gi("boff_gate_proj");
    F.up_c = gi("off_up_proj"), F.up_s = gi("soff_up_proj"), F.up_b = gi("boff_up_proj");
    F.dn_c = gi("off_down_proj"), F.dn_s = gi("soff_down_proj"), F.dn_b = gi("boff_down_proj");
    SH = (int)gi("shared_inter");
    NREC = 120;  // 212 MB
  }
  std::vector<uint8_t> recs((size_t)NREC * F.rec);
  {
    int fd = open((md + "/store/layer_00.bin").c_str(), O_RDONLY);
    if (fd < 0 || !read_all(fd, recs.data(), recs.size(), 0)) return fprintf(stderr, "cannot read layer_00.bin\n"), 1;
    close(fd);
  }
  std::vector<ExpertJob> all(NREC);
  for (int r = 0; r < NREC; r++) {
    Weight gu, up, dn;
    F.mats(recs.data() + (size_t)r * F.rec, gu, up, dn);
    all[r] = {gu, up, dn, I, 0};
  }
  // the 35B's shared expert: AFF8 of its shape (resident in the engine), synthetic codes
  std::vector<uint8_t> shc;
  std::vector<_Float16> shs;
  ExpertJob shj;
  if (SH > 0) {
    shc.resize((size_t)3 * SH * H);
    std::mt19937 g(3);
    for (auto& b : shc) b = (uint8_t)g();
    shs.assign((size_t)3 * SH * H / 64 * 2, (_Float16)0.01f);
    auto mk = [&](int rows, int cols, size_t off, size_t soff) {
      Weight w;
      w.kind = Weight::AFF8, w.rows = rows, w.cols = cols, w.c = shc.data() + off;
      w.s = shs.data() + soff, w.b = shs.data() + soff + (size_t)SH * H / 64;
      return w;
    };
    shj.gu = mk(SH, H, 0, 0);
    shj.up = mk(SH, H, (size_t)SH * H, (size_t)2 * SH * H / 64);
    shj.dn = mk(H, SH, (size_t)2 * SH * H, (size_t)4 * SH * H / 64);
    shj.inter = SH;
    shj.slot_out = K;
  }
  const int IA = std::max(I, SH);
  Pool pool(8);
  XBuf xin;
  xin.init(H, Prec::I16);
  std::vector<float> xf(H);
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  for (auto& v : xf) v = nd(rng) * 0.5f;
  xin.prep(xf.data(), H);
  std::vector<XBuf> act(K + 1);
  for (auto& a : act) a.init(IA, Prec::I16);
  std::vector<float> eout((size_t)(K + 1) * H), actf((size_t)(K + 1) * IA);
  std::atomic<int64_t> spin{0};
  ExpertStep S;
  S.pool = &pool, S.IA = IA, S.H = H, S.softcap_act = ink, S.gcap = 30.f, S.ucap = 30.f;
  S.xq = &xin.q, S.xf = xf.data(), S.actf = actf.data(), S.eout = eout.data(), S.act = act.data(), S.spin_ns = &spin;
  std::atomic<int64_t> rr_n{0};
  S.reruns = &rr_n;

  std::atomic<bool> busy_stop{false};
  std::vector<std::thread> busy;
  for (int b = 0; b < BUSY; b++)
    busy.emplace_back([&]() {
      pthread_set_qos_class_self_np(QOS_CLASS_DEFAULT, 0);
      volatile uint64_t x = 0;
      while (!busy_stop.load(std::memory_order_relaxed)) x = x + 1;
    });
  // the concurrent reader
  std::atomic<bool> rd_on{false}, rd_stop{false};
  std::thread reader([&]() {
    set_qos_interactive();
    int fd = open((md + "/store/layer_05.bin").c_str(), O_RDONLY);
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
    const size_t piece = 1 << 20;
    std::vector<uint8_t> buf(piece);
    const off_t fsz = lseek(fd, 0, SEEK_END);
    off_t off = 0;
    auto t0 = sclk::now();
    double sent = 0;
    while (!rd_stop.load()) {
      if (!rd_on.load(std::memory_order_relaxed)) {
        t0 = sclk::now(), sent = 0;
        std::this_thread::yield();
        continue;
      }
      if (pread(fd, buf.data(), piece, off) <= 0) off = 0;
      off = (off + (off_t)piece) % (fsz - (off_t)piece);
      sent += piece;
      const double due = sent / (RD_GBS * 1e9);
      while (std::chrono::duration<double>(sclk::now() - t0).count() < due && rd_on.load(std::memory_order_relaxed)) std::this_thread::yield();
    }
    close(fd);
  });

  // the stream reference: every matrix's rows claimed 64 at a time
  struct Stream {
    std::vector<std::pair<const Weight*, int>> ch;  // (matrix, first row)
    std::atomic<int> next{0};
    const XQ *xg, *xd[17];
  } SR;
  auto stream_phase = [](void* p, int, int) {
    Stream& R = *(Stream*)p;
    float y[64];
    for (int c; (c = R.next.fetch_add(1, std::memory_order_relaxed)) < (int)R.ch.size();) {
      const Weight* w = R.ch[c].first;
      const int r0 = R.ch[c].second;
      mv(*w, r0, std::min(w->rows, r0 + 64), w->cols == R.xg->K ? R.xg : R.xd[0], nullptr, y, false);
    }
  };

  const char* NAMES[] = {"engine", "nodep", "noact", "ponly", "ptail", "smalltail", "twojoins", "warm", "reader", "stream",
                         "parked", "wave7+1", "e_no_gateup", "e_no_down", "rerun", "split1"};
  const unsigned ARMS[] = {0, STEP_NODEP, STEP_NOACT, STEP_PONLY, STEP_PTAIL, STEP_SMALLTAIL, 0, 0, 0, 0, 0, 0, STEP_ENOGU, STEP_ENODN, STEP_RERUN, 0};
  const int NA = 16;
  {  // the arms that keep the arithmetic give the engine arm's outputs bit for bit (all work done, each row once)
    ExpertJob jobs[17];
    int n = 0;
    if (SH > 0) jobs[n++] = shj;
    for (int k = 0; k < K; k++) jobs[n] = all[k], jobs[n].slot_out = k, n++;
    std::vector<float> ref;
    long bad = 0;
    for (int a : {0, 3, 4, 5, 6, 12, 13, 14, 0}) {
      std::fill(eout.begin(), eout.end(), 0.f);
      S.arms = ARMS[a];
      if (a == 6) expert_step(S, jobs, n / 2), expert_step(S, jobs + n / 2, n - n / 2);
      else expert_step(S, jobs, n);
      if (ref.empty()) ref = eout;
      else bad += std::memcmp(ref.data(), eout.data(), eout.size() * 4) != 0;
    }
    for (int r = 0; r < 300; r++) {  // the re-run arm again and again (DMENG_RERUN_LIM_NS=0 re-runs whatever is not done)
      S.arms = STEP_RERUN;
      expert_step(S, jobs, n);
      bad += std::memcmp(ref.data(), eout.data(), eout.size() * 4) != 0;
    }
    printf("{\"arms_bits_differ\":%ld,\"reruns_in_check\":%lld}\n", bad, (long long)rr_n.load());
    rr_n.store(0);
  }
  double ns[NA] = {0}, by[NA] = {0}, sp[NA] = {0}, cnt[NA] = {0};
  int base = 0, last = 0;
  uint32_t lcg = 2463534242u;
  for (int app = -200; app < NAPP; app++) {  // (the first 200 warm the pool and the caches, unrecorded)
    lcg = lcg * 1664525u + 1013904223u;
    const int arm = app < 0 ? 0 : (int)((lcg >> 8) % NA);
    const int b0 = arm == 7 ? last : base;
    if (arm != 7) base = (base + K) % NREC, last = b0;
    ExpertJob jobs[17];
    int n = 0;
    if (SH > 0) jobs[n++] = shj;
    for (int k = 0; k < K; k++) {
      jobs[n] = all[(b0 + k) % NREC];
      jobs[n].slot_out = k;
      n++;
    }
    double bytes = 0;
    for (int j = 0; j < n; j++) bytes += jobs[j].gu.bytes() + (jobs[j].up.kind == Weight::NONE ? 0 : jobs[j].up.bytes()) + jobs[j].dn.bytes();
    if (arm == 8) rd_on.store(true);
    if (arm == 9) {
      SR.ch.clear();
      for (int j = 0; j < n; j++)
        for (const Weight* w : {&jobs[j].gu, &jobs[j].up, &jobs[j].dn})
          if (w->kind != Weight::NONE)
            for (int r = 0; r < w->rows; r += 64) SR.ch.push_back({w, r});
      SR.next.store(0);
      SR.xg = &xin.q;
      SR.xd[0] = &act[0].q;
    }
    if (arm == 10) {  // as after a wait on a read: the workers parked, then woken by the step
      pool.park();
      const auto tw = sclk::now();
      while (std::chrono::duration<double, std::micro>(sclk::now() - tw).count() < 300) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    const int64_t s0 = spin.load();
    const auto t0 = sclk::now();
    if (arm == 9) {
      pool.run_any(stream_phase, &SR);
    } else if (arm == 6) {
      S.arms = 0;
      expert_step(S, jobs, n / 2);
      expert_step(S, jobs + n / 2, n - n / 2);
    } else if (arm == 15) {
      S.arms = STEP_RERUN;
      for (int j = 0; j < n; j++) expert_step(S, jobs + j, 1);
    } else if (arm == 11) {
      S.arms = 0;
      expert_step(S, jobs, n - 1);
      expert_step(S, jobs + n - 1, 1);
    } else {
      S.arms = ARMS[arm];
      expert_step(S, jobs, n);
    }
    const double t = std::chrono::duration<double, std::nano>(sclk::now() - t0).count();
    if (arm == 8) rd_on.store(false);
    if (app < 0) continue;
    ns[arm] += t, by[arm] += bytes, cnt[arm] += 1, sp[arm] += (spin.load() - s0) / 1e6;
  }
  rd_stop.store(true);
  reader.join();
  busy_stop.store(true);
  for (auto& t : busy) t.join();
  printf("{\"reruns_in_rerun_arm\":%lld}\n", (long long)rr_n.load());
  const double g0 = by[0] / ns[0];
  for (int a = 0; a < NA; a++)
    printf("{\"model\":\"%s\",\"busy\":%d,\"arm\":\"%s\",\"apps\":%.0f,\"gb_s\":%.1f,\"vs_engine\":%.3f,\"us_per_app\":%.1f,\"spin_thread_ms_per_app\":%.3f}\n",
           ink ? "inkling" : flash ? "flash" : "35b", BUSY, NAMES[a], cnt[a], by[a] / ns[a], (by[a] / ns[a]) / g0, ns[a] / cnt[a] / 1e3, sp[a] / cnt[a]);
  return 0;
}
