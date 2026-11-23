// Memory regions, weight views and their matvec dispatch, the DMENGRES1 reader, embedding rows, row phases, and the
// exact reference kernels.
#include "dmeng/weights.h"
#include <vector>

#include <arm_neon.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace dm {

// ---------------------------------------------------------------------------------------------- memory
uint8_t* alloc_region(size_t bytes, Mem mode, std::string* err) {
  bytes = (bytes + 16383) & ~size_t(16383);
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (p == MAP_FAILED) {
    if (err) *err = "mmap failed";
    return nullptr;
  }
  if (mode == Mem::MLOCK && mlock(p, bytes) != 0) {
    if (err) *err = std::string("mlock failed: ") + strerror(errno);
    munmap(p, bytes);
    return nullptr;
  }
  return (uint8_t*)p;
}

void free_region(uint8_t* p, size_t bytes) {
  if (!p) return;
  munmap(p, (bytes + 16383) & ~size_t(16383));
}

Mem mem_of(const std::string& s) {
  return s == "mlock" ? Mem::MLOCK : s == "purgeable" ? Mem::PURGEABLE : s == "file" ? Mem::FILE : Mem::PLAIN;
}

// ---------------------------------------------------------------------------------------------- weights
size_t Weight::bytes() const {
  switch (kind) {
    case AFF8: return (size_t)rows * cols + (size_t)rows * (cols / 64) * 4;
    case AFF4: return (size_t)rows * cols / 2 + (size_t)rows * (cols / 64) * 4;
    case NVFP4: return (size_t)rows * cols / 2 + (size_t)rows * (cols / 16);
    case MXFP8: return (size_t)rows * cols + (size_t)rows * (cols / 32);
    case BF16: return (size_t)rows * cols * 2;
    case F32: return (size_t)rows * cols * 4;
    default: return 0;
  }
}

void mv(const Weight& w, int r0, int r1, const XQ* xq, const float* xf, float* y, bool ref) {
  switch (w.kind) {
    case Weight::AFF4: mv_aff4(w.c, (const _Float16*)w.s, (const _Float16*)w.b, w.cols, r0, r1, *xq, y); break;
    case Weight::AFF8:
      if (ref && xf) kref_aff8(w, r0, r1, xf, y);
      else mv_aff8(w.c, (const _Float16*)w.s, (const _Float16*)w.b, w.cols, r0, r1, *xq, y);
      break;
    case Weight::NVFP4:
      if (ref && xf) kref_nvfp4(w.c, (const uint8_t*)w.s, w.gmul, w.cols, r0, r1, xf, y);
      else mv_nvfp4(w.c, (const uint8_t*)w.s, w.gmul, w.cols, r0, r1, *xq, y);
      break;
    case Weight::MXFP8:
      if (ref && xf) kref_mxfp8(w.c, (const uint8_t*)w.s, w.cols, r0, r1, xf, y);
      else mv_mxfp8(w.c, (const uint8_t*)w.s, w.cols, r0, r1, *xq, y);
      break;
    case Weight::BF16:
      for (int r = r0; r < r1; r++) y[r - r0] = dot_bf16((const uint16_t*)w.w + (size_t)r * w.cols, xf, w.cols);
      break;
    case Weight::F32: mv_f32((const float*)w.w, w.cols, r0, r1, xf, y); break;
    default: fprintf(stderr, "mv: empty weight\n"); abort();
  }
}

XBuf::~XBuf() { std::free(mem); }
void XBuf::init(int K, Prec p) {
  K = (K + 63) / 64 * 64;
  std::free(mem);
  mem = std::aligned_alloc(64, (xq_bytes(K) + 63) & ~size_t(63));
  xq_bind(q, mem, K, p);
  cap = K;
}
void XBuf::prep(const float* x, int K) {
  q.K = K;
  xq_prep(q, x);
}

// ---------------------------------------------------------------------------------------------- ResFile
static bool parse_header(int fd, const std::string& path, std::map<std::string, std::string>& cfg,
                         std::map<std::string, ResFile::T>* t, size_t& hdr, size_t& data, std::string& err) {
  char first[64] = {0};
  if (pread(fd, first, 63, 0) <= 0 || strncmp(first, "DMENGRES1 ", 10) != 0) {
    err = path + ": not a DMENGRES1 file";
    return false;
  }
  hdr = strtoull(first + 10, nullptr, 10);
  std::string head(hdr, '\0');
  if (pread(fd, &head[0], hdr, 0) != (ssize_t)hdr) {
    err = "short header in " + path;
    return false;
  }
  std::istringstream in(head);
  std::string line;
  std::getline(in, line);
  data = 0;
  while (std::getline(in, line)) {
    if (line == "end") break;
    std::istringstream ls(line);
    std::string kind;
    ls >> kind;
    if (kind == "cfg") {
      std::string k, v;
      ls >> k;
      std::getline(ls, v);
      if (!v.empty() && v[0] == ' ') v.erase(0, 1);
      cfg[k] = v;
    } else if (kind == "t" && t) {
      std::string n;
      ResFile::T x;
      int nd;
      ls >> n >> x.dtype >> x.off >> x.nbytes >> nd;
      x.shape.resize(nd);
      for (auto& d : x.shape) ls >> d;
      data = std::max(data, x.off + x.nbytes);
      (*t)[n] = x;
    }
  }
  return true;
}

bool ResFile::read_cfg(const std::string& path, std::map<std::string, std::string>& cfg, std::string& err) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + path;
    return false;
  }
  size_t hdr = 0, data = 0;
  bool ok = parse_header(fd, path, cfg, nullptr, hdr, data, err);
  close(fd);
  return ok;
}

bool ResFile::load(const std::string& path, Mem mode, std::string& err) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + path;
    return false;
  }
  size_t data = 0;
  if (!parse_header(fd, path, cfg, &t, hdr, data, err)) {
    close(fd);
    return false;
  }
  size = (data + 16383) & ~size_t(16383);
  mem = mode;
  if (mode == Mem::FILE) {  // mapped, not loaded: the pages stay clean and file-backed (the OS may drop and re-read them)
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < hdr + data) {
      err = "short file " + path;
      close(fd);
      return false;
    }
    map_size = (size_t)st.st_size;
    void* p = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
      err = "mmap " + path + ": " + strerror(errno);
      map_size = 0;
      return false;
    }
    map = (uint8_t*)p;
    base = map + hdr;
    return true;
  }
  base = alloc_region(size, mode, &err);
  if (!base) {
    close(fd);
    return false;
  }
  const size_t CH = 64 << 20;  // big sequential reads straight into the (possibly locked) region
  for (size_t o = 0; o < data; o += CH) {
    size_t n = std::min(CH, data - o);
    if (pread(fd, base + o, n, hdr + o) != (ssize_t)n) {
      err = "short read of " + path;
      close(fd);
      return false;
    }
  }
  close(fd);
  return true;
}

ResFile::~ResFile() {
  if (map) munmap(map, map_size);
  else free_region(base, size);
}

const ResFile::T& ResFile::info(const std::string& n) const {
  auto it = t.find(n);
  if (it == t.end()) {
    fprintf(stderr, "tensor missing: %s\n", n.c_str());
    abort();
  }
  return it->second;
}

const float* ResFile::f32(const std::string& n) const {
  if (info(n).dtype != "f32") {
    fprintf(stderr, "%s is %s, expected f32\n", n.c_str(), info(n).dtype.c_str());
    abort();
  }
  return (const float*)ptr(n);
}

Weight ResFile::weight(const std::string& n) const {
  Weight w;
  const bool q4 = has(n + ".q4");
  if (q4 || has(n + ".q")) {
    const T& q = info(n + (q4 ? ".q4" : ".q"));
    const bool four = q4 || (has_cfg("bits") && geti("bits") == 4);
    w.kind = four ? Weight::AFF4 : Weight::AFF8;
    w.rows = (int)q.shape[0];
    w.cols = (int)q.shape[1] * (four ? 2 : 1);
    w.c = (const uint8_t*)ptr(n + (q4 ? ".q4" : ".q"));
    w.s = ptr(n + ".s");
    w.b = ptr(n + ".b");
    return w;
  }
  const T& x = info(n);
  w.rows = (int)x.shape[0];
  w.cols = 1;
  for (size_t i = 1; i < x.shape.size(); i++) w.cols *= (int)x.shape[i];
  if (x.dtype == "bf16") w.kind = Weight::BF16;
  else if (x.dtype == "f32") w.kind = Weight::F32;
  else {
    fprintf(stderr, "%s: unsupported dtype %s\n", n.c_str(), x.dtype.c_str());
    abort();
  }
  w.w = ptr(n);
  return w;
}

Weight ResFile::bf16_row(const std::string& n) const {
  const T& x = info(n);
  Weight w;
  w.kind = Weight::BF16;
  w.rows = 1;
  w.cols = 1;
  for (long d : x.shape) w.cols *= (int)d;
  w.w = ptr(n);
  return w;
}

long ResFile::geti(const std::string& k) const {
  auto it = cfg.find(k);
  if (it == cfg.end()) {
    fprintf(stderr, "cfg missing: %s\n", k.c_str());
    abort();
  }
  return strtol(it->second.c_str(), nullptr, 10);
}
double ResFile::getf(const std::string& k) const { return strtod(cfg.at(k).c_str(), nullptr); }
std::string ResFile::gets(const std::string& k) const { return cfg.at(k); }

// ---------------------------------------------------------------------------------------------- embedding rows
bool EmbedRows::open(const std::string& path, long offset, int hidden, std::string& err) {
  fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + path;
    return false;
  }
  off = offset;
  H = hidden;
  return true;
}

void EmbedRows::row(int tok, float* out) const {
  std::vector<uint16_t> r(H);
  if (pread(fd, r.data(), 2 * (size_t)H, off + (long)tok * 2 * H) != 2 * H) {
    fprintf(stderr, "embedding read failed for token %d\n", tok);
    abort();
  }
  for (int i = 0; i < H; i++) out[i] = bf16_to_f32(r[i]);
}

EmbedRows::~EmbedRows() {
  if (fd >= 0) close(fd);
}

// ---------------------------------------------------------------------------------------------- row phases
namespace {
struct RowsCtx {
  const Job* j;
  int n, chunk;
  bool ref;
  int* start;
  alignas(128) std::atomic<int> next{0};
};
void phase_rows(void* p, int, int) {
  RowsCtx& C = *(RowsCtx*)p;
  const int total = C.start[C.n];
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
    int i = 0;
    while (ch >= C.start[i + 1]) i++;
    const Job& J = C.j[i];
    int r0 = (ch - C.start[i]) * C.chunk, r1 = std::min(J.w->rows, r0 + C.chunk);
    mv(*J.w, r0, r1, J.xq, J.xf, J.out + r0, C.ref);
  }
}
}  // namespace

void run_rows(Pool& pool, const Job* jobs, int n, int chunk, bool ref, double* bytes) {
  RowsCtx C;
  C.j = jobs;
  C.n = n;
  C.chunk = chunk;
  C.ref = ref;
  int small[65];
  std::vector<int> big;
  if (n > 64) big.resize(n + 1);
  C.start = n > 64 ? big.data() : small;
  C.start[0] = 0;
  for (int i = 0; i < n; i++) {
    C.start[i + 1] = C.start[i] + (jobs[i].w->rows + chunk - 1) / chunk;
    if (bytes) *bytes += (double)jobs[i].w->bytes();
  }
  pool.run(phase_rows, &C);
}

namespace {
struct RowsQCtx {
  const Job* j;
  int n, chunk;
  bool ref;
  int* start;
  XBuf* const* qs;
  const float* const* xs;
  int qstart[9];  // block prefix over the inputs
  std::atomic<int> left{0};
  alignas(128) std::atomic<int> next{0};
};
void phase_rows_q(void* p, int, int) {
  RowsQCtx& C = *(RowsQCtx*)p;
  const int nqb = C.qstart[8], total = nqb + C.start[C.n];
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
    if (ch < nqb) {
      int i = 0;
      while (ch >= C.qstart[i + 1]) i++;
      const int b = ch - C.qstart[i];
      xq_block(C.qs[i]->q, b, C.xs[i] + XB * b);
      C.left.fetch_sub(1, std::memory_order_release);
      continue;
    }
    while (C.left.load(std::memory_order_acquire) > 0) __builtin_arm_yield();
    const int c2 = ch - nqb;
    int i = 0;
    while (c2 >= C.start[i + 1]) i++;
    const Job& J = C.j[i];
    const int r0 = (c2 - C.start[i]) * C.chunk, r1 = std::min(J.w->rows, r0 + C.chunk);
    mv(*J.w, r0, r1, J.xq, J.xf, J.out + r0, C.ref);
  }
}

struct PostCtx {
  const Job* j;
  int chunk, nch;
  bool ref;
  void (*post)(void*, int, int);
  void* ctx;
  alignas(128) std::atomic<int> next{0};
};
void phase_rows_post(void* p, int, int) {
  PostCtx& C = *(PostCtx*)p;
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < C.nch;) {
    const int r0 = ch * C.chunk, r1 = std::min(C.j->w->rows, r0 + C.chunk);
    mv(*C.j->w, r0, r1, C.j->xq, C.j->xf, C.j->out + r0, C.ref);
    C.post(C.ctx, r0, r1);
  }
}
}  // namespace

void run_rows_q(Pool& pool, const Job* jobs, int n, int chunk, bool ref, double* bytes, XBuf* const* qs, const float* const* xs,
                const int* Ks, int nq) {
  if (ref || nq > 8) {  // the reference path reads fp32 inputs; or more inputs than the context holds: prep first
    for (int i = 0; i < nq; i++) qs[i]->prep(xs[i], Ks[i]);
    run_rows(pool, jobs, n, chunk, ref, bytes);
    return;
  }
  RowsQCtx C;
  C.j = jobs;
  C.n = n;
  C.chunk = chunk;
  C.ref = ref;
  C.qs = qs;
  C.xs = xs;
  int small[65];
  std::vector<int> big;
  if (n > 64) big.resize(n + 1);
  C.start = n > 64 ? big.data() : small;
  C.start[0] = 0;
  for (int i = 0; i < n; i++) {
    C.start[i + 1] = C.start[i] + (jobs[i].w->rows + chunk - 1) / chunk;
    if (bytes) *bytes += (double)jobs[i].w->bytes();
  }
  C.qstart[0] = 0;
  for (int i = 0; i < 8; i++) {
    const int nb = i < nq ? Ks[i] / XB : 0;
    if (i < nq) qs[i]->q.K = Ks[i];
    C.qstart[i + 1] = C.qstart[i] + nb;
  }
  C.left.store(C.qstart[8], std::memory_order_relaxed);
  pool.run(phase_rows_q, &C);
}

void run_rows_post(Pool& pool, const Job& job, int chunk, bool ref, double* bytes, void (*post)(void*, int, int), void* ctx) {
  PostCtx C;
  C.j = &job;
  C.chunk = chunk;
  C.nch = (job.w->rows + chunk - 1) / chunk;
  C.ref = ref;
  C.post = post;
  C.ctx = ctx;
  if (bytes) *bytes += (double)job.w->bytes();
  pool.run(phase_rows_post, &C);
}

void mv_multi(const Weight& w, int r0, int r1, const XQ* const* xq, const float* const* xf, int n, float* const* y, bool ref) {
  const bool i16 = !ref && n > 1 && xq && xq[0]->prec == Prec::I16;
  for (int k0 = 0; k0 < n; k0 += 8) {
    const int k = std::min(8, n - k0);
    float* ys[8];
    for (int i = 0; i < k; i++) ys[i] = y[k0 + i];
    if (i16 && w.kind == Weight::AFF8) {
      mv_aff8_multi(w.c, (const _Float16*)w.s, (const _Float16*)w.b, w.cols, r0, r1, xq + k0, k, ys);
    } else if (i16 && w.kind == Weight::NVFP4) {
      mv_nvfp4_multi(w.c, (const uint8_t*)w.s, w.gmul, w.cols, r0, r1, xq + k0, k, ys);
    } else if (i16 && w.kind == Weight::MXFP8) {
      mv_mxfp8_multi(w.c, (const uint8_t*)w.s, w.cols, r0, r1, xq + k0, k, ys);
    } else {
      for (int i = 0; i < k; i++) mv(w, r0, r1, xq ? xq[k0 + i] : nullptr, xf ? xf[k0 + i] : nullptr, ys[i], ref);
    }
  }
}

namespace {
struct RowsNCtx {
  const JobN* j;
  int n, chunk;
  bool ref;
  int* start;
  alignas(128) std::atomic<int> next{0};
};
void phase_rows_n(void* p, int, int) {
  RowsNCtx& C = *(RowsNCtx*)p;
  const int total = C.start[C.n];
  static thread_local std::vector<float> tmp;
  static thread_local std::vector<float*> ys;
  for (int ch; (ch = C.next.fetch_add(1, std::memory_order_relaxed)) < total;) {
    int i = 0;
    while (ch >= C.start[i + 1]) i++;
    const JobN& J = C.j[i];
    const Weight& w = *J.w;
    const int r0 = (ch - C.start[i]) * C.chunk, r1 = std::min(w.rows, r0 + C.chunk);
    if (!C.ref && J.xb && w.kind == Weight::AFF8 && !(r0 & 3) && !(r1 & 3) && w.cols % 256 == 0) {
      tmp.resize((size_t)J.n * C.chunk);
      gemm_aff8_i16(w.c, (const _Float16*)w.s, (const _Float16*)w.b, w.cols, r0, r1, *J.xb, 0, J.n, tmp.data(), C.chunk);
      for (int k = 0; k < J.n; k++) std::memcpy(J.out[k] + r0, tmp.data() + (size_t)k * C.chunk, sizeof(float) * (r1 - r0));
      continue;
    }
    ys.resize(J.n);
    for (int k = 0; k < J.n; k++) ys[k] = J.out[k] + r0;
    mv_multi(w, r0, r1, J.xq, J.xf, J.n, ys.data(), C.ref);
  }
}
}  // namespace

void run_rows_n(Pool& pool, const JobN* jobs, int nj, int chunk, bool ref, double* bytes) {
  RowsNCtx C;
  C.j = jobs;
  C.n = nj;
  C.chunk = chunk;
  C.ref = ref;
  std::vector<int> start(nj + 1);
  C.start = start.data();
  C.start[0] = 0;
  for (int i = 0; i < nj; i++) {
    C.start[i + 1] = C.start[i] + (jobs[i].w->rows + chunk - 1) / chunk;
    if (bytes) *bytes += (double)jobs[i].w->bytes();
  }
  pool.run(phase_rows_n, &C);
}

// ---------------------------------------------------------------------------------------------- numerics
// The arithmetic is written out, so it does not depend on where the compiler inlines or vectorises this function: 16 at
// a time with fused multiply-adds into four accumulators; then the tail, as the research engines' out-of-line build
// computed it: products rounded and added in order for whole groups of 4, fused multiply-adds for the last 0-3.
float dot_bf16(const uint16_t* w, const float* x, int n) {
#pragma clang fp contract(off)
  float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
  int i = 0;
  for (; i + 16 <= n; i += 16) {
    uint16x8_t v0 = vld1q_u16(w + i), v1 = vld1q_u16(w + i + 8);
    a0 = vfmaq_f32(a0, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v0), 16)), vld1q_f32(x + i));
    a1 = vfmaq_f32(a1, vreinterpretq_f32_u32(vshll_high_n_u16(v0, 16)), vld1q_f32(x + i + 4));
    a2 = vfmaq_f32(a2, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v1), 16)), vld1q_f32(x + i + 8));
    a3 = vfmaq_f32(a3, vreinterpretq_f32_u32(vshll_high_n_u16(v1, 16)), vld1q_f32(x + i + 12));
  }
  float t = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
  for (const int n4 = i + ((n - i) & ~3); i < n4; i++) {
    const float p = bf16_to_f32(w[i]) * x[i];
    t = t + p;
  }
  for (; i < n; i++) t = __builtin_fmaf(bf16_to_f32(w[i]), x[i], t);
  return t;
}

// ---------------------------------------------------------------------------------------------- reference kernels
void kref_aff8(const Weight& w, int r0, int r1, const float* x, float* y) {
  const int K = w.cols, NG = K / 64;
  const _Float16* s = (const _Float16*)w.s;
  const _Float16* b = (const _Float16*)w.b;
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = w.c + (size_t)r * K;
    double t = 0;
    for (int g = 0; g < NG; g++) {
      float d = 0, sx = 0;
      for (int i = 0; i < 64; i++) {
        d += (float)c[64 * g + i] * x[64 * g + i];
        sx += x[64 * g + i];
      }
      t += (float)s[(size_t)r * NG + g] * d + (float)b[(size_t)r * NG + g] * sx;
    }
    y[r - r0] = (float)t;
  }
}

static float e4m3(uint8_t b) {
  int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? std::ldexp((float)m / 8.f, -6) : std::ldexp(1.f + (float)m / 8.f, e - 7);
  return s ? -v : v;
}
static const float E2M1[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -.5f, -1, -1.5f, -2, -3, -4, -6};

void kref_nvfp4(const uint8_t* codes, const uint8_t* sc, float gmul, int K, int r0, int r1, const float* x, float* y) {
  const int NG = K / 16;
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * (K / 2);
    const uint8_t* s = sc + (size_t)r * NG;
    float t = 0;
    for (int g = 0; g < NG; g++) {
      float d = 0;
      for (int i = 0; i < 16; i++) {
        int j = 16 * g + i;
        uint8_t b = c[j / 2];
        d += E2M1[(j & 1) ? (b >> 4) : (b & 15)] * x[j];
      }
      t += d * e4m3(s[g]);
    }
    y[r - r0] = t * gmul;
  }
}

void kref_mxfp8(const uint8_t* codes, const uint8_t* sc, int K, int r0, int r1, const float* x, float* y) {
  const int NG = K / 32;
  for (int r = r0; r < r1; r++) {
    const uint8_t* c = codes + (size_t)r * K;
    const uint8_t* s = sc + (size_t)r * NG;
    float t = 0;
    for (int g = 0; g < NG; g++) {
      float d = 0;
      for (int i = 0; i < 32; i++) d += e4m3(c[32 * g + i]) * x[32 * g + i];
      t += d * std::ldexp(1.f, (int)s[g] - 127);
    }
    y[r - r0] = t;
  }
}

}  // namespace dm
