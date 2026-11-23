// One fork-join of the MoE expert step (the engine's Core::run_experts, and tests/kernels/bench_kmulti_step.cpp): every
// job's gate-up rows, its activation (SwiGLU, or Inkling's softcapped form) quantised for the down projection, then its
// down rows into the job's output row; claimed from shared counters on the pool (see moe.cpp phase_gudn).
#pragma once
#include <atomic>
#include <cstdint>

#include "dmeng/pool.h"
#include "dmeng/weights.h"

namespace dm {

struct ExpertJob {
  Weight gu, up, dn;  // up.kind NONE: gu is the fused gate-up matrix (row 2j = gate j, 2j + 1 = up j)
  int inter = 0, slot_out = 0;
};

// measurement arms (the bench only, except STEP_RERUN, which the engine sets). They change timing, and NODEP / NOACT also
// the outputs.
enum : unsigned {
  STEP_NODEP = 1,      // down rows do not wait for their expert's gate-up
  STEP_NOACT = 2,      // no activation and no quantisation of it (the down rows read the previous contents)
  STEP_PONLY = 4,      // threads on efficiency cores claim nothing
  STEP_PTAIL = 8,      // threads on efficiency cores stop claiming down rows once few remain
  STEP_SMALLTAIL = 16, // single down quarters from 8 per thread left (instead of 4)
  STEP_ENOGU = 32,     // threads on efficiency cores claim no gate-up chunks
  STEP_ENODN = 64,     // threads on efficiency cores claim no down rows
  STEP_RERUN = 128     // a gate-up chunk claimed long ago and not done is re-run by a thread that would spin (the engine's
                       // default: claimed more than 4x a chunk's running time ago; the re-run computes into its own scratch
                       // and one compare-and-swap publishes either copy)
};

struct ExpertStep {
  Pool* pool = nullptr;
  int IA = 0, H = 0;  // activation row stride (the largest intermediate size), hidden size
  bool ref = false, softcap_act = false;
  float gcap = 0.f, ucap = 0.f;
  const XQ* xq = nullptr;  // the step's input, quantised; xf: the same in fp32
  const float* xf = nullptr;
  float* actf = nullptr;  // [slots][IA] activations (fp32)
  float* eout = nullptr;  // [slots][H] expert outputs
  XBuf* act = nullptr;    // [slots] activations quantised
  std::atomic<int64_t>* spin_ns = nullptr;  // thread time down claims spent waiting on gate-up (all threads)
  std::atomic<int64_t>* reruns = nullptr;   // STEP_RERUN: gate-up chunks re-run
  unsigned arms = 0;
};

void expert_step(const ExpertStep& s, const ExpertJob* jobs, int n);

// the CPU the calling thread runs on and whether it is an efficiency core (Apple M4: CPUs 0-5 efficiency, 6-9
// performance; hw.perflevel1.logicalcpu efficiency cores numbered first)
int cpu_now();
bool on_ecore();

}  // namespace dm
