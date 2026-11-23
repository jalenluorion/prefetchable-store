// P0 kernel bench, Metal part. Exact kernels: every output is the CPU kernels' bits (checked here, on the device):
//   mv8 / mv4   batch-1 matvec of 8-bit / 4-bit affine g64 matrices against the I16 activations (v = hi * 256 + lo).
//               4 threads per row, thread j owns the 64-groups g = j (mod 4), i.e. the CPU kernel's float lane j:
//               a = fma(float(dot_g), scale_g * d_g, a); a = fma(bias_g, gsum_g, a) in group order; the row's result is
//               (a0 + a1) + (a2 + a3) (vaddvq_f32's pairwise order). The integer group dots are exact int32.
//   gd          the batched 8-bit group dot of deep-memory/engine/src/gpu_gemm.mm (16 x 16 tiles), unchanged
//   bw          a read-bandwidth kernel
// MTLMathModeSafe: IEEE fp32, fma fused, no reassociation.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

#include "bench.h"

using namespace dm;

namespace dmb {

static const char* MSRC = R"(
#include <metal_stdlib>
using namespace metal;
struct MV { ulong co, so, bo; uint R, K; };
kernel void mv8(device const uchar* W [[buffer(0)]], device const short* v [[buffer(1)]], device const float* d [[buffer(2)]],
                device const float* gs [[buffer(3)]], device float* y [[buffer(4)]], constant MV& p [[buffer(5)]],
                uint gid [[thread_position_in_grid]]) {
  const uint r = gid >> 2, lane = gid & 3, NG = p.K / 64;
  const bool ok = r < p.R;
  float a = 0.0f;
  if (ok) {
    device const uchar* c = W + p.co + (ulong)r * p.K;
    device const half* sc = (device const half*)(W + p.so) + (ulong)r * NG;
    device const half* bi = (device const half*)(W + p.bo) + (ulong)r * NG;
    for (uint g = lane; g < NG; g += 4) {
      device const uchar4* c4 = (device const uchar4*)(c + g * 64);
      device const short4* v4 = (device const short4*)(v + g * 64);
      int dot = 0;
      for (uint j = 0; j < 16; j++) {
        int4 cc = int4(c4[j]);
        int4 vv = int4(v4[j]);
        dot += cc.x * vv.x + cc.y * vv.y + cc.z * vv.z + cc.w * vv.w;
      }
      float m = float(sc[g]) * d[g];
      a = fma(float(dot), m, a);
      a = fma(float(bi[g]), gs[g], a);
    }
  }
  float a1 = simd_shuffle_xor(a, 1u);
  float a2 = simd_shuffle_xor(a, 2u);
  float a3 = simd_shuffle_xor(a, 3u);
  if (ok && lane == 0) y[r] = (a + a1) + (a2 + a3);
}
kernel void mv4(device const uchar* W [[buffer(0)]], device const short* v [[buffer(1)]], device const float* d [[buffer(2)]],
                device const float* gs [[buffer(3)]], device float* y [[buffer(4)]], constant MV& p [[buffer(5)]],
                uint gid [[thread_position_in_grid]]) {
  const uint r = gid >> 2, lane = gid & 3, NG = p.K / 64;
  const bool ok = r < p.R;
  float a = 0.0f;
  if (ok) {
    device const uchar* c = W + p.co + (ulong)r * (p.K / 2);
    device const half* sc = (device const half*)(W + p.so) + (ulong)r * NG;
    device const half* bi = (device const half*)(W + p.bo) + (ulong)r * NG;
    for (uint g = lane; g < NG; g += 4) {
      device const uchar4* c4 = (device const uchar4*)(c + g * 32);
      device const short4* v4 = (device const short4*)(v + g * 64);
      int dot = 0;
      for (uint j = 0; j < 8; j++) {
        uint4 b = uint4(c4[j]);
        int4 lo = int4(b & 15u), hi = int4(b >> 4);
        int4 va = int4(v4[2 * j]), vb = int4(v4[2 * j + 1]);
        dot += lo.x * va.x + hi.x * va.y + lo.y * va.z + hi.y * va.w + lo.z * vb.x + hi.z * vb.y + lo.w * vb.z + hi.w * vb.w;
      }
      float m = float(sc[g]) * d[g];
      a = fma(float(dot), m, a);
      a = fma(float(bi[g]), gs[g], a);
    }
  }
  float a1 = simd_shuffle_xor(a, 1u);
  float a2 = simd_shuffle_xor(a, 2u);
  float a3 = simd_shuffle_xor(a, 3u);
  if (ok && lane == 0) y[r] = (a + a1) + (a2 + a3);
}
struct P { ulong co, so, bo; uint R, K, N; };
kernel void gd(device const uchar* res [[buffer(0)]], device const short* v [[buffer(1)]], device const float* d [[buffer(2)]],
               device const float* gs [[buffer(3)]], device float* y [[buffer(4)]], constant P& p [[buffer(5)]],
               uint2 tg [[threadgroup_position_in_grid]], uint2 lt [[thread_position_in_threadgroup]]) {
  const uint R = p.R, K = p.K, N = p.N, NG = K / 64;
  device const uchar* codes = res + p.co;
  device const half* sc = (device const half*)(res + p.so);
  device const half* bi = (device const half*)(res + p.bo);
  const uint r = tg.x * 16 + lt.x, q = tg.y * 16 + lt.y;
  threadgroup uchar wt[16][64];
  threadgroup short xt[16][64];
  float4 acc = float4(0.0f);
  const uint tid = lt.y * 16 + lt.x;
  for (uint g = 0; g < NG; g++) {
    for (uint k = tid; k < 1024; k += 256) {
      uint rr = k / 64, kk = k % 64;
      wt[rr][kk] = (tg.x * 16 + rr < R) ? codes[(ulong)(tg.x * 16 + rr) * K + g * 64 + kk] : 0;
      xt[rr][kk] = (tg.y * 16 + rr < N) ? v[(ulong)(tg.y * 16 + rr) * K + g * 64 + kk] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int dot = 0;
    for (uint kk = 0; kk < 64; kk++) dot += int(wt[lt.x][kk]) * int(xt[lt.y][kk]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r < R && q < N) {
      const uint lane = g & 3;
      float m = float(sc[(ulong)r * NG + g]) * d[(ulong)q * NG + g];
      float a = acc[lane];
      a = fma(float(dot), m, a);
      a = fma(float(bi[(ulong)r * NG + g]), gs[(ulong)q * NG + g], a);
      acc[lane] = a;
    }
  }
  if (r < R && q < N) y[(ulong)q * R + r] = (acc.x + acc.y) + (acc.z + acc.w);
}
kernel void bw(device const uint4* p [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint gid [[thread_position_in_grid]], uint gsz [[threads_per_grid]]) {
  uint4 s = uint4(0);
  for (uint i = gid; i < n; i += gsz) s += p[i];
  out[gid] = s.x + s.y + s.z + s.w;
}
kernel void nop(device uint* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) { if (gid == 0) out[0] += 1; }
)";

struct MVArgs { uint64_t co, so, bo; uint32_t R, K; };
struct PArgs { uint64_t co, so, bo; uint32_t R, K, N; };

struct Gpu {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> mv8, mv4, gd, bw, nop;
  bool ok = false;
  std::string err;
  Gpu() {
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) { err = "no Metal device"; return; }
    NSError* e = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
    co.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:MSRC] options:co error:&e];
    if (!lib) { err = e ? e.localizedDescription.UTF8String : "compile failed"; return; }
    auto mk = [&](const char* n) { return [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@(n)] error:nil]; };
    mv8 = mk("mv8"); mv4 = mk("mv4"); gd = mk("gd"); bw = mk("bw"); nop = mk("nop");
    q = [dev newCommandQueue];
    ok = mv8 && mv4 && gd && bw && nop && q;
    if (!ok) err = "pipeline failed";
  }
};

// a MatBuf wrapped without copy, plus the activations of one position
struct GBuf {
  id<MTLBuffer> w, v, d, g, y;
  void wrap(Gpu& G, const MatBuf& b, const XQ& x) {
    size_t len = (((size_t)b.nmat * b.stride) + 16383) & ~size_t(16383);
    w = [G.dev newBufferWithBytesNoCopy:(void*)b.buf length:len options:MTLResourceStorageModeShared deallocator:nil];
    const int K = b.c.cols, NG = K / 64;
    std::vector<int16_t> vv(K);
    for (int k = 0; k < K; k++) vv[k] = (int16_t)(x.hn[k] * 256 + x.ln[k]);
    v = [G.dev newBufferWithBytes:vv.data() length:K * 2 options:MTLResourceStorageModeShared];
    d = [G.dev newBufferWithBytes:x.d length:NG * 4 options:MTLResourceStorageModeShared];
    g = [G.dev newBufferWithBytes:x.gsum length:NG * 4 options:MTLResourceStorageModeShared];
    y = [G.dev newBufferWithLength:(size_t)std::max(b.c.rows, 1) * 4 * 8 options:MTLResourceStorageModeShared];
  }
};

static void enc_mv(id<MTLComputeCommandEncoder> en, Gpu& G, const MatBuf& b, GBuf& gb, int m) {
  MVArgs a{(uint64_t)((size_t)m * b.stride), (uint64_t)((size_t)m * b.stride + b.code_bytes),
           (uint64_t)((size_t)m * b.stride + b.code_bytes + b.scale_bytes), (uint32_t)b.c.rows, (uint32_t)b.c.cols};
  [en setComputePipelineState:b.c.fmt == Fmt::AFF4 ? G.mv4 : G.mv8];
  [en setBuffer:gb.w offset:0 atIndex:0];
  [en setBuffer:gb.v offset:0 atIndex:1];
  [en setBuffer:gb.d offset:0 atIndex:2];
  [en setBuffer:gb.g offset:0 atIndex:3];
  [en setBuffer:gb.y offset:(size_t)(m & 7) * b.c.rows * 4 atIndex:4];
  [en setBytes:&a length:sizeof a atIndex:5];
  [en dispatchThreads:MTLSizeMake((NSUInteger)b.c.rows * 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// one pass over every matrix: one command buffer, one dispatch per matrix; returns wall seconds, gpu seconds
static std::pair<double, double> gpu_pass(Gpu& G, const MatBuf& b, GBuf& gb, int nmat = -1) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [G.q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    int n = nmat < 0 ? b.nmat : nmat;
    for (int m = 0; m < n; m++) enc_mv(en, G, b, gb, m);
    [en endEncoding];
    double t0 = now();
    [cb commit];
    [cb waitUntilCompleted];
    double w = now() - t0;
    return {w, cb.GPUEndTime - cb.GPUStartTime};
  }
}

static size_t check_mv(Gpu& G, const Case& c) {
  MatBuf b;
  b.make(c, 1, 11);
  std::vector<uint8_t> mem(xq_bytes(c.cols) + 64);
  XQ x;
  xq_bind(x, (void*)(((uintptr_t)mem.data() + 63) & ~uintptr_t(63)), c.cols, Prec::I16);
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> xf(c.cols);
  for (auto& v : xf) v = nd(rng) * (rng() % 50 == 0 ? 20.f : 1.f);
  xq_prep(x, xf.data());
  std::vector<float> yc(c.rows);
  if (c.fmt == Fmt::AFF4) mv_aff4(b.codes(0), b.scales(0), b.biases(0), c.cols, 0, c.rows, x, yc.data());
  else mv_aff8(b.codes(0), b.scales(0), b.biases(0), c.cols, 0, c.rows, x, yc.data());
  GBuf gb;
  gb.wrap(G, b, x);
  gpu_pass(G, b, gb, 1);
  const float* yg = (const float*)gb.y.contents;
  size_t bad = 0;
  for (int r = 0; r < c.rows; r++) bad += std::memcmp(&yc[r], &yg[r], 4) != 0;
  gb = GBuf();
  b.release();
  return bad;
}

void metal_bench(J& j, const Spec& sp, dmb_log_fn log) {
  @autoreleasepool {
    Gpu G;
    j.key("metal").obj();
    if (!G.ok) {
      j.kv("error", G.err).end_obj();
      logf(log, "KBENCH metal error: %s", G.err.c_str());
      return;
    }
    const double cell = sp.num("cell", 3), sustain = sp.num("sustain", 60);
    const size_t mb = (size_t)sp.num("mb", 384);
    const int cput = (int)sp.num("cputhreads", 2);
    j.kv("device", std::string(G.dev.name.UTF8String));
    if (@available(iOS 16.0, macOS 13.0, *)) j.kv("max_working_set", (int64_t)G.dev.recommendedMaxWorkingSetSize);

    // exactness against the CPU kernels on this device
    j.key("bit_identity").obj();
    for (const char* id : {"a4gu", "a4dn", "a8rt", "a8out", "a4out"}) {
      size_t bad = check_mv(G, *find_case(id));
      j.kv(id, (int64_t)bad);
      logf(log, "KBENCH metal exact %s: %zu of %d rows differ from the CPU kernel", id, bad, find_case(id)->rows);
    }
    {  // batched gd vs gemm_aff8_i16, 1024 x 2048, N = 64
      Case c{"gdcheck", "", Fmt::AFF8, 1024, 2048};
      MatBuf b;
      b.make(c, 1, 13);
      const int N = 64, K = c.cols, NG = K / 64;
      XQB xb;
      xb.reserve(K, N);
      std::mt19937 rng(9);
      std::normal_distribution<float> nd(0.f, 1.f);
      std::vector<float> X((size_t)N * K);
      for (auto& v : X) v = nd(rng) * (rng() % 50 == 0 ? 20.f : 1.f);
      for (int i = 0; i < N; i++) xqb_set(xb, i, X.data() + (size_t)i * K);
      xqb_pad(xb);
      std::vector<float> Yc((size_t)xb.NP * c.rows);
      for (int r = 0; r < c.rows; r += 64) gemm_aff8_i16(b.codes(0), b.scales(0), b.biases(0), K, r, r + 64, xb, 0, N, Yc.data() + r, c.rows);
      std::vector<int16_t> vv((size_t)N * K);
      for (int i = 0; i < N; i++)
        for (int k = 0; k < K; k++) {
          size_t o = ((size_t)(i >> 1) * (K / 8) + (k >> 3)) * 16 + (i & 1) * 8 + (k & 7);
          vv[(size_t)i * K + k] = (int16_t)(xb.hn[o] * 256 + xb.ln[o]);
        }
      size_t len = (b.stride + 16383) & ~size_t(16383);
      id<MTLBuffer> w = [G.dev newBufferWithBytesNoCopy:b.buf length:len options:MTLResourceStorageModeShared deallocator:nil];
      id<MTLBuffer> bv = [G.dev newBufferWithBytes:vv.data() length:vv.size() * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> bd = [G.dev newBufferWithBytes:xb.d length:(size_t)N * NG * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> bg = [G.dev newBufferWithBytes:xb.gsum length:(size_t)N * NG * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> by = [G.dev newBufferWithLength:(size_t)N * c.rows * 4 options:MTLResourceStorageModeShared];
      PArgs a{0, b.code_bytes, b.code_bytes + b.scale_bytes, (uint32_t)c.rows, (uint32_t)K, (uint32_t)N};
      id<MTLCommandBuffer> cb = [G.q commandBuffer];
      id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
      [en setComputePipelineState:G.gd];
      [en setBuffer:w offset:0 atIndex:0];
      [en setBuffer:bv offset:0 atIndex:1];
      [en setBuffer:bd offset:0 atIndex:2];
      [en setBuffer:bg offset:0 atIndex:3];
      [en setBuffer:by offset:0 atIndex:4];
      [en setBytes:&a length:sizeof a atIndex:5];
      [en dispatchThreadgroups:MTLSizeMake((c.rows + 15) / 16, (N + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [en endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      const float* yg = (const float*)by.contents;
      size_t bad = 0;
      for (size_t i = 0; i < (size_t)N * c.rows; i++) bad += std::memcmp(&yg[i], &Yc[i], 4) != 0;
      j.kv("gd_a8_1024x2048_N64", (int64_t)bad);
      logf(log, "KBENCH metal exact gd N=64: %zu of %d outputs differ", bad, N * c.rows);
      b.release();
    }
    j.end_obj();

    // round trip of an empty command buffer, and of one router matvec (a decode step's sync point)
    {
      id<MTLBuffer> o = [G.dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
      std::vector<double> ts;
      for (int i = 0; i < 200; i++) {
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [G.q commandBuffer];
          id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
          [en setComputePipelineState:G.nop];
          [en setBuffer:o offset:0 atIndex:0];
          [en dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
          [en endEncoding];
          double t0 = now();
          [cb commit];
          [cb waitUntilCompleted];
          ts.push_back(now() - t0);
        }
      }
      std::sort(ts.begin(), ts.end());
      j.kv("roundtrip_us_p50", ts[100] * 1e6).kv("roundtrip_us_p90", ts[180] * 1e6);
      logf(log, "KBENCH metal command buffer round trip p50 %.0f us p90 %.0f us", ts[100] * 1e6, ts[180] * 1e6);
    }

    // GPU read bandwidth
    {
      size_t n = mb << 20;
      id<MTLBuffer> p = [G.dev newBufferWithLength:n options:MTLResourceStorageModeShared];
      std::memset(p.contents, 1, n);
      id<MTLBuffer> o = [G.dev newBufferWithLength:(1 << 16) * 4 options:MTLResourceStorageModeShared];
      uint32_t n4 = (uint32_t)(n / 16);
      std::vector<double> ts;
      double t0 = now();
      while (now() - t0 < cell || ts.size() < 3) {
        @autoreleasepool {
          id<MTLCommandBuffer> cb = [G.q commandBuffer];
          id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
          [en setComputePipelineState:G.bw];
          [en setBuffer:p offset:0 atIndex:0];
          [en setBuffer:o offset:0 atIndex:1];
          [en setBytes:&n4 length:4 atIndex:2];
          [en dispatchThreads:MTLSizeMake(1 << 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          [en endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
        }
      }
      std::sort(ts.begin(), ts.end());
      double g = n / ts[ts.size() / 2] / 1e9;
      j.kv("read_gbps", g);
      logf(log, "KBENCH metal read bandwidth %.1f GB/s", g);
    }

    // matvec rates, DRAM-fed (one command buffer per pass, one dispatch per matrix)
    j.key("matvec").arr();
    for (const char* id : {"a4gu", "a4dn", "a8qkv", "a8out", "a8rt", "a8head", "a4qkv", "a4out", "a4head"}) {
      const Case& c = *find_case(id);
      MatBuf b;
      b.make(c, mb << 20, 1);
      std::vector<uint8_t> mem(xq_bytes(c.cols) + 64);
      XQ x;
      xq_bind(x, (void*)(((uintptr_t)mem.data() + 63) & ~uintptr_t(63)), c.cols, Prec::I16);
      std::vector<float> xf(c.cols, 0.5f);
      xq_prep(x, xf.data());
      GBuf gb;
      gb.wrap(G, b, x);
      gpu_pass(G, b, gb);
      double w = 0, gt = 0, t0 = now();
      int n = 0;
      while (now() - t0 < cell || n < 2) {
        auto r = gpu_pass(G, b, gb);
        w += r.first;
        gt += r.second;
        n++;
      }
      double gbps = b.bytes_per_pass() * n / w / 1e9, gm = b.macs_per_pass() * n / w / 1e9;
      j.obj().kv("case", c.name).kv("nmat", b.nmat).kv("gbps_wall", gbps).kv("gmacs_wall", gm)
          .kv("gbps_gpu", b.bytes_per_pass() * n / gt / 1e9).kv("us_per_matrix_gpu", gt / n / b.nmat * 1e6)
          .kv("thermal", thermal_name(thermal())).end_obj();
      logf(log, "KBENCH metal mv %s %.1f GB/s %.1f GMAC/s (gpu-time %.1f GB/s, %.1f us/mat)", id, gbps, gm,
           b.bytes_per_pass() * n / gt / 1e9, gt / n / b.nmat * 1e6);
      gb = GBuf();
      b.release();
    }
    j.end_arr();

    // batched gd (prefill) 8192 x 2048 at N = 64, 256
    j.key("batched").arr();
    {
      Case c = *find_case("a8qkv");
      MatBuf b;
      b.make(c, 1, 17);
      size_t len = (b.stride + 16383) & ~size_t(16383);
      id<MTLBuffer> w = [G.dev newBufferWithBytesNoCopy:b.buf length:len options:MTLResourceStorageModeShared deallocator:nil];
      for (int N : {64, 256}) {
        const int K = c.cols, NG = K / 64;
        id<MTLBuffer> bv = [G.dev newBufferWithLength:(size_t)N * K * 2 options:MTLResourceStorageModeShared];
        std::memset(bv.contents, 1, (size_t)N * K * 2);
        id<MTLBuffer> bd = [G.dev newBufferWithLength:(size_t)N * NG * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bg = [G.dev newBufferWithLength:(size_t)N * NG * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> by = [G.dev newBufferWithLength:(size_t)N * c.rows * 4 options:MTLResourceStorageModeShared];
        PArgs a{0, b.code_bytes, b.code_bytes + b.scale_bytes, (uint32_t)c.rows, (uint32_t)K, (uint32_t)N};
        std::vector<double> ts;
        double t0 = now();
        while (now() - t0 < cell || ts.size() < 3) {
          @autoreleasepool {
            id<MTLCommandBuffer> cb = [G.q commandBuffer];
            id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
            [en setComputePipelineState:G.gd];
            [en setBuffer:w offset:0 atIndex:0];
            [en setBuffer:bv offset:0 atIndex:1];
            [en setBuffer:bd offset:0 atIndex:2];
            [en setBuffer:bg offset:0 atIndex:3];
            [en setBuffer:by offset:0 atIndex:4];
            [en setBytes:&a length:sizeof a atIndex:5];
            [en dispatchThreadgroups:MTLSizeMake((c.rows + 15) / 16, (N + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [en endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            ts.push_back(cb.GPUEndTime - cb.GPUStartTime);
          }
        }
        std::sort(ts.begin(), ts.end());
        double gm = (double)c.rows * K * N / ts[ts.size() / 2] / 1e9;
        j.obj().kv("case", "a8qkv").kv("N", N).kv("gmacs", gm).kv("ms", ts[ts.size() / 2] * 1e3).end_obj();
        logf(log, "KBENCH metal gd a8qkv N=%d %.1f GMAC/s", N, gm);
      }
      b.release();
    }
    j.end_arr();

    // CPU and GPU together on the experts' kernel (disjoint buffers): does the GPU add to the CPU's rate?
    auto together = [&](const char* key, double secs, bool series) {
      const Case& c = *find_case("a4gu");
      MatBuf b;
      b.make(c, mb << 20, 21);
      std::vector<uint8_t> mem(xq_bytes(c.cols) + 64);
      XQ x;
      xq_bind(x, (void*)(((uintptr_t)mem.data() + 63) & ~uintptr_t(63)), c.cols, Prec::I16);
      std::vector<float> xf(c.cols, 0.5f);
      xq_prep(x, xf.data());
      GBuf gb;
      gb.wrap(G, b, x);
      j.key(key).obj().kv("case", "a4gu").kv("cpu_threads", cput).kv("seconds", secs);
      // GPU alone
      double t0 = now(), w = 0, tsec = t0;
      int64_t bsec = 0;
      int n = 0;
      std::vector<std::array<double, 3>> ser;
      while (now() - t0 < secs || n < 2) {
        w += gpu_pass(G, b, gb).first;
        n++;
        bsec += (int64_t)b.bytes_per_pass();
        double t = now();
        if (series && t - tsec >= 1.0) { ser.push_back({t - t0, bsec / (t - tsec) / 1e9, (double)thermal()}); tsec = t; bsec = 0; }
      }
      double g_alone = b.bytes_per_pass() * n / w / 1e9;
      j.kv("gpu_alone_gbps", g_alone);
      if (series) {
        j.key("gpu_alone_series").arr();
        for (auto& s : ser) j.arr().v(s[0]).v(s[1]).v(thermal_name((int)s[2])).end_arr();
        j.end_arr();
      }
      // CPU alone
      CpuLoad L;
      L.start(c, cput, "ui", mb);
      auto m0 = L.mark();
      std::this_thread::sleep_for(std::chrono::milliseconds((int)(secs * 1000)));
      auto m1 = L.mark();
      double c_alone = CpuLoad::gbps(m0, m1);
      // together
      ser.clear();
      t0 = now(); w = 0; n = 0; tsec = t0; bsec = 0;
      auto m2 = L.mark(), ms = m2;
      while (now() - t0 < secs || n < 2) {
        w += gpu_pass(G, b, gb).first;
        n++;
        double t = now();
        if (series && t - tsec >= 1.0) {
          auto mm = L.mark();
          ser.push_back({t - t0, CpuLoad::gbps(ms, mm), (double)thermal()});
          ms = mm;
          tsec = t;
        }
      }
      double tt = now() - t0;
      auto m3 = L.mark();
      double g_tog = b.bytes_per_pass() * n / tt / 1e9, c_tog = CpuLoad::gbps(m2, m3);
      L.stop();
      j.kv("cpu_alone_gbps", c_alone).kv("gpu_together_gbps", g_tog).kv("cpu_together_gbps", c_tog)
          .kv("sum_together_gbps", g_tog + c_tog).kv("thermal_end", thermal_name(thermal()));
      if (series) {
        j.key("cpu_together_series").arr();
        for (auto& s : ser) j.arr().v(s[0]).v(s[1]).v(thermal_name((int)s[2])).end_arr();
        j.end_arr();
      }
      j.end_obj();
      logf(log, "KBENCH metal %s: gpu alone %.1f, cpu(%d) alone %.1f, together gpu %.1f + cpu %.1f = %.1f GB/s [%s]", key, g_alone,
           cput, c_alone, g_tog, c_tog, g_tog + c_tog, thermal_name(thermal()));
      gb = GBuf();
      b.release();
    };
    together("together", cell * 2, false);
    if (sustain > 0) together("sustained_together", sustain, true);
    j.end_obj();
  }
}

}  // namespace dmb
