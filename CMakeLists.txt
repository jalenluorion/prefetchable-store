cmake_minimum_required(VERSION 3.24)
project(dmeng LANGUAGES CXX)
# One build: the engine library (core + model families) and the dmeng binary.
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# The kernels use dotprod, i8mm (usdot) and fp16fml; the default targets Apple M4.
set(DM_ARCH "-mcpu=apple-m4" CACHE STRING "CPU flags for the kernels")
# Floating-point contraction: the engine's token ids are defined with -ffp-contract=fast; changing it changes logits
# (tests/fingerprint.py shows it). Kept as the reference until every exactness-relevant step is explicit.
set(DM_CONTRACT "-ffp-contract=fast" CACHE STRING "floating-point contraction flag")

# The weight-repo format: manifest and JSON reader, safetensors / DMENGRES1 files, the MLX-exact quantiser and the
# derivations of the engine's files. One IEEE float32 operation per source operation, as in MLX's kernel.
add_library(msfmt STATIC src/format/json.cpp src/format/files.cpp src/format/manifest.cpp src/format/quant.cpp src/format/derive.cpp)
target_compile_options(msfmt PRIVATE -O2 -ffp-contract=off -fno-fast-math -Wall -Wno-deprecated-declarations)
# The format tools alone (any platform: a Linux x86 machine can make the engine's derived files where the bf16 sources
# are, so they need not sit on the Mac; the quantiser is IEEE float32 operations only, the same bits everywhere):
#   cmake -S . -B build-fmt -DDM_FORMAT_ONLY=ON && cmake --build build-fmt --target ms_derive manifest_dump
if(NOT APPLE)
  set(DM_FORMAT_ONLY ON)
endif()
option(DM_FORMAT_ONLY "build only the format library and its tools (ms_derive, manifest_dump)" OFF)
if(DM_FORMAT_ONLY)
  foreach(t manifest_dump ms_derive)
    add_executable(${t} tools/${t}.cpp)
    target_include_directories(${t} PRIVATE src/format)
    target_link_libraries(${t} PRIVATE msfmt)
  endforeach()
  return()
endif()

add_library(dmengcore STATIC
  src/core/kernels.cpp src/core/pool.cpp src/core/weights.cpp src/core/store.cpp src/core/moe.cpp src/core/moe_chunk.cpp src/core/greedy_head.cpp src/core/screen_head.cpp
  src/core/address.cpp src/core/planner.cpp src/core/prefill.cpp src/core/presets.cpp src/core/sampler.cpp src/core/draft.cpp src/core/model_lm.cpp src/core/kernels_batch.cpp src/core/kernels_sme.cpp
  src/core/gpu_check.cpp src/core/gpu_split.cpp src/core/rowstore.cpp
  src/models/registry.cpp src/models/common/gdn.cpp src/models/qwen3_5_moe/qwen3_5_moe.cpp src/models/qwen3_5_moe/prefill.cpp src/models/qwen3_5_moe/verify.cpp
  src/models/inkling2_loop/inkling2_loop.cpp src/models/inkling2_loop/ink_draft.cpp
  src/models/qwen4_exp/qwen4_exp.cpp src/models/qwen4_exp/prefill.cpp)
target_include_directories(dmengcore PUBLIC include)
target_link_libraries(dmengcore PUBLIC msfmt)
target_include_directories(dmengcore PRIVATE src/format)
target_compile_options(dmengcore PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall -Wno-unused-function)
# The GPU share of the batched prefill's products (src/core/gpu_gemm.mm, Metal; 8- and 4-bit; exact, see gpu_gemm.h). Built
# on Apple platforms by default; the code path stays off unless a run asks for it (prefill_gpu_frac / prefill_gpu_experts
# > 0, dmserve --gpu-frac / --gpu-experts).
if(APPLE)
  option(DM_METAL "build the Metal split of the prefill products" ON)
else()
  set(DM_METAL OFF)
endif()
if(DM_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  target_sources(dmengcore PRIVATE src/core/gpu_gemm.mm src/core/gpu_half.mm)
  set_target_properties(dmengcore PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
  set_source_files_properties(src/core/gpu_gemm.mm src/core/gpu_half.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_definitions(dmengcore PRIVATE DM_METAL=1)
  target_link_libraries(dmengcore PUBLIC "-framework Metal" "-framework Foundation")
  # the GPU kernels against the CPU kernels, bit for bit, on random data (no model): tests/kernels/gpu_gemm_check.cpp
  add_executable(gpu_gemm_check tests/kernels/gpu_gemm_check.cpp)
  target_link_libraries(gpu_gemm_check PRIVATE dmengcore)
  target_compile_options(gpu_gemm_check PRIVATE ${DM_ARCH})
  # the fast prompt path's half-precision products against an fp32 reference on random data, and their throughput
  add_executable(gpu_half_check tests/kernels/gpu_half_check.cpp)
  target_link_libraries(gpu_half_check PRIVATE dmengcore)
  target_compile_options(gpu_half_check PRIVATE ${DM_ARCH})
  target_compile_definitions(gpu_half_check PRIVATE DM_METAL=1)
endif()
# A counting build: per-thread instructions and cycles of the pool's work, the main thread's serial work and its
# spin-waits (macOS thread_selfcounts), printed to stderr at exit when DM_COUNTS is set (see bench/README.md).
# OFF compiles to exactly the default engine.
option(DM_COUNTERS "count instructions and cycles per phase (bench builds only)" OFF)
if(DM_COUNTERS)
  target_compile_definitions(dmengcore PRIVATE DM_COUNTERS=1)
endif()
# A join-trace build: per phase function, each thread's entry, exit and clock and the join's waits ("DMJOIN" lines at
# exit, src/core/pool.cpp). OFF compiles to exactly the default engine.
option(DM_POOLTRACE "trace every fork-join's threads (diagnostic builds only)" OFF)
if(DM_POOLTRACE)
  target_compile_definitions(dmengcore PRIVATE DM_POOLTRACE=1)
endif()

add_executable(dmeng tools/dmeng.cpp)
target_link_libraries(dmeng PRIVATE dmengserve dmengcore)
target_compile_options(dmeng PRIVATE ${DM_ARCH} -Wall)

# The serving worker (one model process behind python/serve/dmserve.py): prefix cache + any model of this build.
add_library(dmengserve STATIC src/serving/session.cpp src/serving/ahead.cpp src/serving/hotset.cpp src/serving/memwatch.cpp)
target_include_directories(dmengserve PUBLIC include)
target_compile_options(dmengserve PRIVATE ${DM_ARCH} -Wall)
add_executable(dmserve tools/dmserve.cpp)
target_link_libraries(dmserve PRIVATE dmengserve dmengcore)
target_compile_options(dmserve PRIVATE ${DM_ARCH} -Wall)

# Format tools: read / check a model folder's manifest; derive the cache files explicitly
foreach(t manifest_dump ms_derive)
  add_executable(${t} tools/${t}.cpp)
  target_include_directories(${t} PRIVATE src/format)
  target_link_libraries(${t} PRIVATE msfmt)
endforeach()

# The C API (bindings/c/dmeng_c.h): the boundary the iPhone app (apps/ios) links against
add_library(dmeng_c STATIC bindings/c/dmeng_c.cpp)
target_include_directories(dmeng_c PUBLIC bindings/c include PRIVATE src/format)
target_link_libraries(dmeng_c PUBLIC dmengserve dmengcore)
target_compile_options(dmeng_c PRIVATE ${DM_ARCH} -Wall)
# Spare memory's mechanics on a small store of known bytes, no model (tests/spare_tier/spare_tier_check.cpp)
add_executable(spare_tier_check tests/spare_tier/spare_tier_check.cpp)
target_link_libraries(spare_tier_check PRIVATE dmengcore)
add_executable(spare_tier_bench tests/spare_tier/spare_tier_bench.cpp)  # spare memory's cost to the compute, synthetic store
target_link_libraries(spare_tier_bench PRIVATE dmengcore)
# The C API's gate: ids through the C API only (tests/c_api/dm_gate.c)
enable_language(C)
add_executable(dm_gate tests/c_api/dm_gate.c)
target_link_libraries(dm_gate PRIVATE dmeng_c)
target_compile_options(dm_gate PRIVATE ${DM_ARCH} -Wall)

# SME2 split microbenchmark (tests/kernels/sme_split.cpp): NEON i8mm vs one SME2 thread per cluster vs the split
add_executable(sme_split tests/kernels/sme_split.cpp)
target_link_libraries(sme_split PRIVATE dmengcore)
target_compile_options(sme_split PRIVATE ${DM_ARCH} -Wall)

# Inkling's expert kernels by users per expert (tests/kernels/expert_users.cpp): packed NVFP4 gemm vs multi-vector, MXFP8
add_executable(expert_users tests/kernels/expert_users.cpp)
target_link_libraries(expert_users PRIVATE dmengcore)
target_compile_options(expert_users PRIVATE ${DM_ARCH} -Wall)

# the one-vector NVFP4 I16 kernel against the v1 kernel it replaced, bit for bit (tests/kernels/nv_i16_check.cpp)
add_executable(nv_i16_check tests/kernels/nv_i16_check.cpp)
target_link_libraries(nv_i16_check PRIVATE dmengcore)
target_compile_options(nv_i16_check PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall)

# one decode position of the 35B's 4-bit expert kernel: NEON against streaming SVE on the SME units, alone and together
add_executable(spinner tests/kernels/spinner.cpp)  # a co-runner for load tests

# the engine's expert step on real records, its losses against streaming taken out one at a time (tests/kernels/bench_kmulti_step.cpp)
add_executable(bench_kmulti_step tests/kernels/bench_kmulti_step.cpp)
target_link_libraries(bench_kmulti_step PRIVATE dmengcore)
target_compile_options(bench_kmulti_step PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall)

# Inkling's decode expert kernels and candidate rewrites on real records (tests/kernels/bench_kmulti_nv.cpp)
add_executable(bench_kmulti_nv tests/kernels/bench_kmulti_nv.cpp)
target_link_libraries(bench_kmulti_nv PRIVATE dmengcore)
target_compile_options(bench_kmulti_nv PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall)

# the in-memory 8-bit matrices' decode fork-joins, Flash-Next's shapes beside the 35B's (tests/kernels/bench_mats.cpp)
add_executable(bench_mats tests/kernels/bench_mats.cpp)
target_link_libraries(bench_mats PRIVATE dmengcore)
target_compile_options(bench_mats PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall)

# the decode's norms one row at a time against side by side (rmsnorm_n), the bits checked first (tests/kernels/bench_norms.cpp)
add_executable(bench_norms tests/kernels/bench_norms.cpp)
target_link_libraries(bench_norms PRIVATE dmengcore)
target_compile_options(bench_norms PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall)

add_executable(bench_kmulti_sme4 tests/kernels/bench_kmulti_sme4.cpp)
target_link_libraries(bench_kmulti_sme4 PRIVATE dmengcore)
target_compile_options(bench_kmulti_sme4 PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall)
