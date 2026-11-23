// Exact 8-bit batched matrix products on the GPU (see gpu_gemm.h). Objective-C++ with ARC (-fobjc-arc).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "gpu_gemm.h"

#include <cstring>
#include <vector>

namespace dm {

// The arithmetic, per (row r, position q) of a matrix of K inputs in groups of 64:
//   dot_g  = sum over the group of int(code) * int(act), act = hi * 256 + lo (the XQB planes): an exact int32, as the
//            CPU's USMMLA / USDOT sums (|dot| < 64 * 255 * 32767 < 2^31), in any order
//   lane   = g mod 4;  a = fma(float(dot_g), float(scale_g) * d_g, a);  a = fma(float(bias_g), gsum_g, a)
//   y      = (lane0 + lane1) + (lane2 + lane3)
// which is gemm_tile_loop<8>'s epilogue (vfmaq_f32 per lane, vaddvq_f32 at the end) operation for operation, compiled with
// MTLMathModeSafe so no operation is reassociated, contracted or approximated: the same bits.
static const char* GSRC = R"(
#include <metal_stdlib>
using namespace metal;
struct P { ulong co, so, bo; uint R, K, N; };
// one thread per (row, position): per 64-input group the exact int32 dot of the 8-bit codes with the 15-bit activations,
// then mv_aff8's epilogue (lane = group mod 4, fma(dot, scale * d), fma(bias, gsum), pairwise sum), IEEE fp32
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
)";

struct GImpl {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> ps;
  id<MTLBuffer> res;
  const uint8_t* base = nullptr;
  size_t bytes = 0;
  struct Slot {
    id<MTLBuffer> v, d, g;
    size_t cap = 0;
    int K = 0, n = 0;
  } slot[4];
  struct Out {
    id<MTLBuffer> y;
    size_t cap = 0;
    int R = 0, n = 0;
  };
  std::vector<Out> outs;
  int nout = 0;
  id<MTLCommandBuffer> cb = nil;
  id<MTLComputeCommandEncoder> en = nil;
  std::vector<id<MTLCommandBuffer>> inflight;
};

bool GpuGemm::init(const void* res_base, size_t res_bytes) {
  @autoreleasepool {
    GImpl* g = new GImpl();
    g->dev = MTLCreateSystemDefaultDevice();
    if (!g->dev || ((uintptr_t)res_base & 16383)) {
      delete g;
      return false;
    }
    NSError* err = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
    co.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib = [g->dev newLibraryWithSource:[NSString stringWithUTF8String:GSRC] options:co error:&err];
    if (!lib) {
      delete g;
      return false;
    }
    g->ps = [g->dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gd"] error:&err];
    g->q = [g->dev newCommandQueue];
    const size_t pg = 16384;
    g->base = (const uint8_t*)res_base;
    g->bytes = (res_bytes + pg - 1) / pg * pg;
    g->res = [g->dev newBufferWithBytesNoCopy:(void*)res_base length:g->bytes options:MTLResourceStorageModeShared deallocator:nil];
    if (!g->ps || !g->q || !g->res) {
      delete g;
      return false;
    }
    impl = g;
    return true;
  }
}

GpuGemm::~GpuGemm() { delete (GImpl*)impl; }

bool GpuGemm::holds(const void* p, size_t n) const {
  if (!impl) return false;
  const GImpl& g = *(const GImpl*)impl;
  const uint8_t* b = (const uint8_t*)p;
  return b >= g.base && b + n <= g.base + g.bytes;
}

void GpuGemm::stage(const XQB& x, int s) {
  GImpl& g = *(GImpl*)impl;
  GImpl::Slot& S = g.slot[s];
  const int K = x.K, n = x.N, NG = K / 64;
  const size_t need = (size_t)n * K * 2;
  if (need > S.cap) {
    S.v = [g.dev newBufferWithLength:need options:MTLResourceStorageModeShared];
    S.d = [g.dev newBufferWithLength:(size_t)n * NG * 4 options:MTLResourceStorageModeShared];
    S.g = [g.dev newBufferWithLength:(size_t)n * NG * 4 options:MTLResourceStorageModeShared];
    S.cap = need;
  }
  // the 15-bit activations hi * 256 + lo in natural order, out of the pair-interleaved planes (kernels_batch.cpp pk_off)
  int16_t* v = (int16_t*)S.v.contents;
  for (int i = 0; i < n; i++)
    for (int k = 0; k < K; k++) {
      const size_t o = ((size_t)(i >> 1) * (K / 8) + (k >> 3)) * 16 + (i & 1) * 8 + (k & 7);
      v[(size_t)i * K + k] = (int16_t)(x.hn[o] * 256 + x.ln[o]);
    }
  std::memcpy(S.d.contents, x.d, (size_t)n * NG * 4);
  std::memcpy(S.g.contents, x.gsum, (size_t)n * NG * 4);
  S.K = K;
  S.n = n;
}

int GpuGemm::enqueue(const uint8_t* codes, const _Float16* scales, const _Float16* biases, int K, int r1, int n, int s) {
  @autoreleasepool {
    GImpl& g = *(GImpl*)impl;
    if (!g.cb) {
      g.cb = [g.q commandBuffer];
      g.en = [g.cb computeCommandEncoder];
      [g.en setComputePipelineState:g.ps];
      g.nout = 0;
    }
    if ((int)g.outs.size() <= g.nout) g.outs.resize(g.nout + 1);
    GImpl::Out& O = g.outs[g.nout];
    const size_t need = (size_t)n * r1 * 4;
    if (need > O.cap) {
      O.y = [g.dev newBufferWithLength:need options:MTLResourceStorageModeShared];
      O.cap = need;
    }
    O.R = r1;
    O.n = n;
    struct {
      uint64_t co, so, bo;
      uint32_t R, K, N;
    } p = {(uint64_t)(codes - g.base), (uint64_t)((const uint8_t*)scales - g.base), (uint64_t)((const uint8_t*)biases - g.base), (uint32_t)r1,
           (uint32_t)K, (uint32_t)n};
    GImpl::Slot& S = g.slot[s];
    [g.en setBuffer:g.res offset:0 atIndex:0];
    [g.en setBuffer:S.v offset:0 atIndex:1];
    [g.en setBuffer:S.d offset:0 atIndex:2];
    [g.en setBuffer:S.g offset:0 atIndex:3];
    [g.en setBuffer:O.y offset:0 atIndex:4];
    [g.en setBytes:&p length:sizeof(p) atIndex:5];
    [g.en dispatchThreadgroups:MTLSizeMake((r1 + 15) / 16, (n + 15) / 16, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    return g.nout++;
  }
}

void GpuGemm::commit() {
  @autoreleasepool {
    GImpl& g = *(GImpl*)impl;
    if (!g.cb) return;
    [g.en endEncoding];
    [g.cb commit];
    g.inflight.push_back(g.cb);
    g.cb = nil;
    g.en = nil;
  }
}

void GpuGemm::wait() {
  @autoreleasepool {
    GImpl& g = *(GImpl*)impl;
    for (auto& c : g.inflight) [c waitUntilCompleted];
    g.inflight.clear();
  }
}

void GpuGemm::finish(int t, float* Y, size_t ldy) {
  GImpl& g = *(GImpl*)impl;
  wait();
  const GImpl::Out& O = g.outs[t];
  const float* y = (const float*)O.y.contents;
  for (int i = 0; i < O.n; i++) std::memcpy(Y + (size_t)i * ldy, y + (size_t)i * O.R, (size_t)O.R * 4);
}

}  // namespace dm
