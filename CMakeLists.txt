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
# Floating-point contraction: the token ids of the research engines were produced with -ffp-contract=fast; changing it
# changes logits (see the regression gate). Kept as the reference until every exactness-relevant step is explicit.
set(DM_CONTRACT "-ffp-contract=fast" CACHE STRING "floating-point contraction flag")

# The weight-repo format: manifest and JSON reader, safetensors / DMENGRES1 files, the MLX-exact quantiser and the
# derivations of the engine's files. One IEEE float32 operation per source operation, as in MLX's kernel.
add_library(msfmt STATIC src/format/json.cpp src/format/files.cpp src/format/manifest.cpp src/format/quant.cpp src/format/derive.cpp)
target_compile_options(msfmt PRIVATE -O2 -ffp-contract=off -fno-fast-math -Wall -Wno-deprecated-declarations)

add_library(dmengcore STATIC
  src/core/kernels.cpp src/core/pool.cpp src/core/weights.cpp src/core/store.cpp src/core/moe.cpp src/core/moe_chunk.cpp src/core/greedy_head.cpp
  src/core/address.cpp src/core/planner.cpp src/core/presets.cpp src/core/sampler.cpp src/core/draft.cpp src/core/model_lm.cpp src/core/kernels_batch.cpp src/core/kernels_sme.cpp
  src/models/registry.cpp src/models/qwen3_5_moe/qwen3_5_moe.cpp src/models/qwen3_5_moe/prefill.cpp src/models/qwen3_5_moe/verify.cpp
  src/models/inkling2_loop/inkling2_loop.cpp src/models/inkling2_loop/ink_draft.cpp)
target_include_directories(dmengcore PUBLIC include)
target_link_libraries(dmengcore PUBLIC msfmt)
target_include_directories(dmengcore PRIVATE src/format)
target_compile_options(dmengcore PRIVATE ${DM_ARCH} ${DM_CONTRACT} -Wall -Wno-unused-function)
# The GPU half of the batched prefill's 8-bit products (src/core/gpu_gemm.mm, Metal; exact, see gpu_gemm.h). Built on Apple
# platforms by default; the code path stays off unless a run asks for it (prefill_gpu_frac > 0, dmserve --gpu-frac).
if(APPLE)
  option(DM_METAL "build the Metal split of the 8-bit prefill products" ON)
else()
  set(DM_METAL OFF)
endif()
if(DM_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  target_sources(dmengcore PRIVATE src/core/gpu_gemm.mm)
  set_target_properties(dmengcore PROPERTIES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED ON)
  set_source_files_properties(src/core/gpu_gemm.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_definitions(dmengcore PRIVATE DM_METAL=1)
  target_link_libraries(dmengcore PUBLIC "-framework Metal" "-framework Foundation")
endif()
# A counting build: per-thread instructions and cycles of the pool's work, the main thread's serial work and its
# spin-waits (macOS thread_selfcounts), printed to stderr at exit when DM_COUNTS is set (see bench/README.md).
# OFF compiles to exactly the default engine.
option(DM_COUNTERS "count instructions and cycles per phase (bench builds only)" OFF)
if(DM_COUNTERS)
  target_compile_definitions(dmengcore PRIVATE DM_COUNTERS=1)
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

# The C API (bindings/c/dmeng_c.h): the boundary the iPhone app (apps/ios, P1) links against
add_library(dmeng_c STATIC bindings/c/dmeng_c.cpp)
target_include_directories(dmeng_c PUBLIC bindings/c include PRIVATE src/format)
target_link_libraries(dmeng_c PUBLIC dmengserve dmengcore)
target_compile_options(dmeng_c PRIVATE ${DM_ARCH} -Wall)
# The P1 gate through the C API only (tests/c_api/dm_gate.c)
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
