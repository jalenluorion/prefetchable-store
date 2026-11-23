// P0 of the dmeng iOS port (2026-11-22): measurement-only C API, called from the app's launch-flag test modes
// (-kernelBench, -fileCacheTest; ios/PhoneStoreApp/PhoneStoreApp/Probe.swift). No model is loaded.
//   dmb_kernel_bench    the engine's NEON kernels (copied from moved-store port-ios: kernels.cpp, kernels_batch.cpp,
//                       pool.cpp) on this model's shapes, DRAM-fed, per thread count and QoS; the raw read bandwidth;
//                       exact Metal kernels (8-bit and 4-bit matvec, 8-bit batched group dots), checked bit for bit
//                       against the CPU kernels; CPU and GPU together; sustained runs with thermal state per second
//   dmb_filecache_test  the OS file cache as the expert cache: mmap of the store files (PROT_READ, MAP_SHARED),
//                       mincore residency, footprint, available memory, warnings, read paths' own cost, interference
//                       with a compute load
// Both return a malloc'd JSON string (free with dmb_free) and report progress lines through `log`.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*dmb_log_fn)(const char* line);

// spec: comma-separated key=value, e.g. "cell=3,sustain=60,threads=1:2:4:6,mb=384,metal=1" (see bench_cpu.cpp)
char* dmb_kernel_bench(const char* spec, dmb_log_fn log);
// variant: grow | dirty | paths | bg; paths: the store files to map; spec: key=value as above (see filecache.mm)
char* dmb_filecache_test(const char* variant, const char* const* paths, int npaths, const char* spec, dmb_log_fn log);
void dmb_free(char* p);

// from the app: memory warnings (UIApplication) and foreground state (scene / application notifications)
void dmb_note_memory_warning(void);
void dmb_set_foreground(int fg);

#ifdef __cplusplus
}
#endif
