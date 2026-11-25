#include "files.h"

#if defined(__APPLE__) && !defined(MS_PORTABLE_SHA256)
#include <CommonCrypto/CommonDigest.h>
#endif
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <vector>

namespace ms {

// reads that leave no pages in the OS file cache where the system has the switch (macOS F_NOCACHE); elsewhere plain
static inline void no_cache(int fd) {
#ifdef F_NOCACHE
  fcntl(fd, F_NOCACHE, 1);
#else
  (void)fd;
#endif
}

#if !defined(__APPLE__) || defined(MS_PORTABLE_SHA256)
// SHA-256 (FIPS 180-4) for platforms without CommonCrypto (the format tools on Linux): the same digests
namespace {
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint8_t blk[64];
  size_t nb = 0;
  uint64_t bits = 0;
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t* p) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
        0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
        0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
        0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3), s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
  void update(const uint8_t* p, size_t n) {
    bits += (uint64_t)n * 8;
    while (n) {
      const size_t k = std::min(n, 64 - nb);
      memcpy(blk + nb, p, k);
      nb += k, p += k, n -= k;
      if (nb == 64) block(blk), nb = 0;
    }
  }
  void final(unsigned char* out) {
    const uint64_t L = bits;
    const uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (nb != 56) update(&zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[i] = (uint8_t)(L >> (56 - 8 * i));
    update(len, 8);
    for (int i = 0; i < 8; i++)
      for (int j = 0; j < 4; j++) out[4 * i + j] = (uint8_t)(h[i] >> (24 - 8 * j));
  }
};
}  // namespace
#endif

int64_t StTensor::numel() const {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

static bool pread_all(int fd, void* dst, uint64_t n, uint64_t off) {
  uint8_t* p = (uint8_t*)dst;
  while (n) {
    size_t c = n > (64u << 20) ? (64u << 20) : (size_t)n;
    ssize_t r = pread(fd, p, c, (off_t)off);
    if (r <= 0) return false;
    p += r;
    n -= r;
    off += r;
  }
  return true;
}

bool SafeTensors::open(const std::string& p, std::string& err) {
  path = p;
  fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + p;
    return false;
  }
  no_cache(fd);
  uint64_t n = 0;
  if (pread(fd, &n, 8, 0) != 8 || n > (1ull << 30)) {
    err = p + ": not a safetensors file";
    return false;
  }
  header_len = n;
  std::string h(n, '\0');
  if (!pread_all(fd, &h[0], n, 8)) {
    err = p + ": short header";
    return false;
  }
  Json j;
  if (!json_parse(h, j, err)) {
    err = p + ": " + err;
    return false;
  }
  for (auto& kv : j.obj) {
    if (kv.first == "__metadata__") {
      for (auto& m : kv.second.obj) metadata.emplace_back(m.first, m.second.kind == Json::STR ? m.second.text : py_str(m.second));
      continue;
    }
    StTensor t;
    t.name = kv.first;
    t.dtype = kv.second["dtype"].s();
    for (auto& d : kv.second["shape"].arr) t.shape.push_back(d.i());
    t.begin = kv.second["data_offsets"][0].i();
    t.end = kv.second["data_offsets"][1].i();
    index[t.name] = tensors.size();
    tensors.push_back(std::move(t));
  }
  return true;
}
SafeTensors::~SafeTensors() {
  if (fd >= 0) ::close(fd);
}
const StTensor* SafeTensors::find(const std::string& n) const {
  auto it = index.find(n);
  return it == index.end() ? nullptr : &tensors[it->second];
}
bool SafeTensors::read(const StTensor& t, void* dst, std::string& err) const {
  if (!pread_all(fd, dst, t.end - t.begin, data_offset(t))) {
    err = path + ": short read of " + t.name;
    return false;
  }
  return true;
}
std::string SafeTensors::meta(const std::string& k, const std::string& d) const {
  for (auto& kv : metadata)
    if (kv.first == k) return kv.second;
  return d;
}

// ---------------------------------------------------------------------------------------------- npz
namespace {
uint16_t le16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t le32(const uint8_t* p) { return (uint32_t)le16(p) | ((uint32_t)le16(p + 2) << 16); }
uint64_t le64(const uint8_t* p) { return (uint64_t)le32(p) | ((uint64_t)le32(p + 4) << 32); }
// the value of key k in a .npy header dict ("{'descr': '<f4', 'fortran_order': False, 'shape': (512, 7680), }")
std::string npy_field(const std::string& h, const std::string& k) {
  size_t p = h.find("'" + k + "'");
  if (p == std::string::npos) return "";
  p = h.find(':', p);
  if (p == std::string::npos) return "";
  p++;
  while (p < h.size() && h[p] == ' ') p++;
  if (p < h.size() && (h[p] == '\'' || h[p] == '"')) {
    const char q = h[p];
    size_t e = h.find(q, p + 1);
    return e == std::string::npos ? "" : h.substr(p + 1, e - p - 1);
  }
  if (p < h.size() && h[p] == '(') {
    size_t e = h.find(')', p);
    return e == std::string::npos ? "" : h.substr(p, e - p + 1);
  }
  size_t e = h.find_first_of(",}", p);
  return h.substr(p, e - p);
}
}  // namespace

bool Npz::open(const std::string& p, std::string& err) {
  path = p;
  fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + p;
    return false;
  }
  struct stat sb;
  if (fstat(fd, &sb) != 0 || sb.st_size < 22) {
    err = p + ": not a zip archive";
    return false;
  }
  const uint64_t size = (uint64_t)sb.st_size;
  // the end of central directory record: within the last 64 KiB + 22 bytes (its comment)
  const uint64_t tail = std::min<uint64_t>(size, 65557);
  std::vector<uint8_t> t(tail);
  if (!pread_all(fd, t.data(), tail, size - tail)) {
    err = p + ": short read";
    return false;
  }
  int64_t eo = -1;
  for (int64_t i = (int64_t)tail - 22; i >= 0; i--)
    if (le32(&t[i]) == 0x06054b50) {
      eo = i;
      break;
    }
  if (eo < 0) {
    err = p + ": no zip end record";
    return false;
  }
  uint64_t n_ent = le16(&t[eo + 10]), cd_size = le32(&t[eo + 12]), cd_off = le32(&t[eo + 16]);
  if (n_ent == 0xFFFF || cd_size == 0xFFFFFFFFu || cd_off == 0xFFFFFFFFu) {  // zip64: the locator just before it
    if (eo < 20 || le32(&t[eo - 20]) != 0x07064b50) {
      err = p + ": zip64 archive without a locator";
      return false;
    }
    const uint64_t z64 = le64(&t[eo - 20 + 8]);
    uint8_t r[56];
    if (!pread_all(fd, r, 56, z64) || le32(r) != 0x06064b50) {
      err = p + ": bad zip64 end record";
      return false;
    }
    n_ent = le64(r + 32);
    cd_size = le64(r + 40);
    cd_off = le64(r + 48);
  }
  if (cd_off + cd_size > size) {
    err = p + ": central directory outside the file";
    return false;
  }
  std::vector<uint8_t> cd(cd_size);
  if (!pread_all(fd, cd.data(), cd_size, cd_off)) {
    err = p + ": short read of the central directory";
    return false;
  }
  size_t q = 0;
  for (uint64_t e = 0; e < n_ent; e++) {
    if (q + 46 > cd.size() || le32(&cd[q]) != 0x02014b50) {
      err = p + ": bad central directory entry";
      return false;
    }
    const uint16_t method = le16(&cd[q + 10]), nl = le16(&cd[q + 28]), xl = le16(&cd[q + 30]), cl = le16(&cd[q + 32]);
    uint64_t csize = le32(&cd[q + 20]), usize = le32(&cd[q + 24]), loff = le32(&cd[q + 42]);
    std::string name((const char*)&cd[q + 46], nl);
    for (size_t x = q + 46 + nl; x + 4 <= q + 46 + nl + xl;) {  // the zip64 extra field: the 0xFFFFFFFF values in order
      const uint16_t id = le16(&cd[x]), len = le16(&cd[x + 2]);
      if (id == 0x0001) {
        size_t y = x + 4;
        if (usize == 0xFFFFFFFFu) usize = le64(&cd[y]), y += 8;
        if (csize == 0xFFFFFFFFu) csize = le64(&cd[y]), y += 8;
        if (loff == 0xFFFFFFFFu) loff = le64(&cd[y]);
      }
      x += 4 + len;
    }
    q += 46 + nl + xl + cl;
    if (method != 0 || csize != usize) {
      err = p + ": member " + name + " is compressed (np.savez_compressed); only stored members are read";
      return false;
    }
    uint8_t lh[30];
    if (!pread_all(fd, lh, 30, loff) || le32(lh) != 0x04034b50) {
      err = p + ": bad local header of " + name;
      return false;
    }
    const uint64_t data = loff + 30 + le16(lh + 26) + le16(lh + 28);
    // the .npy header
    uint8_t mg[12];
    if (usize < 12 || !pread_all(fd, mg, 12, data) || memcmp(mg, "\x93NUMPY", 6) != 0) {
      err = p + ": member " + name + " is not a .npy array";
      return false;
    }
    const int major = mg[6];
    const uint64_t hl = major == 1 ? le16(mg + 8) : le32(mg + 8), h0 = data + (major == 1 ? 10 : 12);
    std::string h(hl, '\0');
    if (!pread_all(fd, &h[0], hl, h0)) {
      err = p + ": short .npy header of " + name;
      return false;
    }
    StTensor tt;
    tt.name = name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0 ? name.substr(0, name.size() - 4) : name;
    const std::string descr = npy_field(h, "descr"), fo = npy_field(h, "fortran_order"), shp = npy_field(h, "shape");
    static const std::pair<const char*, const char*> DT[] = {{"<f4", "F32"}, {"<f2", "F16"}, {"<i8", "I64"}, {"<i4", "I32"},
                                                             {"|u1", "U8"},  {"|i1", "I8"},  {"<u4", "U32"}, {"<f8", "F64"},
                                                             {"|b1", "BOOL"}};
    for (auto& d : DT)
      if (descr == d.first) tt.dtype = d.second;
    if (tt.dtype.empty() || fo.find("True") != std::string::npos) {
      err = p + ": member " + name + " has dtype " + descr + (fo.find("True") != std::string::npos ? " in Fortran order" : "") + " (not read)";
      return false;
    }
    for (size_t i = 1; i < shp.size();) {  // "(512, 7680)", "(7680,)", "()"
      while (i < shp.size() && (shp[i] == ' ' || shp[i] == ',')) i++;
      if (i >= shp.size() || shp[i] == ')') break;
      tt.shape.push_back(strtoll(shp.c_str() + i, nullptr, 10));
      while (i < shp.size() && shp[i] != ',' && shp[i] != ')') i++;
    }
    tt.begin = h0 + hl;
    tt.end = data + usize;
    const int es = tt.dtype == "F32" || tt.dtype == "I32" || tt.dtype == "U32" ? 4 : tt.dtype == "F16" ? 2 : tt.dtype == "I64" || tt.dtype == "F64" ? 8 : 1;
    if ((int64_t)(tt.end - tt.begin) != tt.numel() * es) {
      err = p + ": member " + name + ": data size does not match its shape";
      return false;
    }
    index[tt.name] = tensors.size();
    tensors.push_back(std::move(tt));
  }
  return true;
}
Npz::~Npz() {
  if (fd >= 0) ::close(fd);
}
const StTensor* Npz::find(const std::string& n) const {
  auto it = index.find(n);
  return it == index.end() ? nullptr : &tensors[it->second];
}
bool Npz::read(const StTensor& t, void* dst, std::string& err) const {
  if (!pread_all(fd, dst, t.end - t.begin, t.begin)) {
    err = path + ": short read of " + t.name;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------------- DMENGRES1
bool DmFile::open(const std::string& p, std::string& err) {
  path = p;
  int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + p;
    return false;
  }
  char first[64] = {0};
  if (pread(fd, first, 63, 0) <= 0 || strncmp(first, "DMENGRES1 ", 10) != 0) {
    err = p + ": not a DMENGRES1 file";
    ::close(fd);
    return false;
  }
  hdr = strtoull(first + 10, nullptr, 10);
  std::string head(hdr, '\0');
  bool ok = pread_all(fd, &head[0], hdr, 0);
  ::close(fd);
  if (!ok) {
    err = p + ": short header";
    return false;
  }
  std::istringstream in(head.c_str());
  std::string line;
  std::getline(in, line);
  while (std::getline(in, line)) {
    if (line == "end") break;
    if (!line.compare(0, 4, "cfg ")) {
      size_t sp = line.find(' ', 4);
      cfg.emplace_back(line.substr(4, sp - 4), sp == std::string::npos ? "" : line.substr(sp + 1));
    } else if (!line.compare(0, 2, "t ")) {
      std::istringstream ls(line.substr(2));
      DmTensor t;
      int nd;
      ls >> t.name >> t.dtype >> t.off >> t.nbytes >> nd;
      t.shape.resize(nd);
      for (auto& d : t.shape) ls >> d;
      index[t.name] = tensors.size();
      tensors.push_back(t);
    }
  }
  return true;
}
const std::string* DmFile::get(const std::string& k) const {
  for (auto& kv : cfg)
    if (kv.first == k) return &kv.second;
  return nullptr;
}
const DmTensor* DmFile::find(const std::string& n) const {
  auto it = index.find(n);
  return it == index.end() ? nullptr : &tensors[it->second];
}
bool DmFile::read(const DmTensor& t, std::vector<uint8_t>& out, std::string& err) const {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + path;
    return false;
  }
  no_cache(fd);
  out.resize(t.nbytes);
  bool ok = pread_all(fd, out.data(), t.nbytes, hdr + t.off);
  ::close(fd);
  if (!ok) err = path + ": short read of " + t.name;
  return ok;
}

bool DmWriter::open(const std::string& path, std::string& err) {
  path_ = path;
  tmp_ = path + ".tmp";
  f_ = fopen(tmp_.c_str(), "wb");
  if (!f_) {
    err = "cannot write " + tmp_;
    return false;
  }
  std::vector<char> z(RESERVE, 0);
  bad_ = fwrite(z.data(), 1, z.size(), f_) != z.size();
  return !bad_;
}
bool DmWriter::begin(const std::string& name, const char* dtype, uint64_t nbytes, const std::vector<int64_t>& shape) {
  if (!f_ || bad_) return false;
  tab_.push_back({name, dtype, off_, nbytes, shape});
  want_ = nbytes;
  got_ = 0;
  return true;
}
bool DmWriter::append(const void* data, uint64_t n) {
  if (!f_ || bad_) return false;
  bad_ |= fwrite(data, 1, n, f_) != n;
  got_ += n;
  return !bad_;
}
bool DmWriter::finish() {
  if (!f_ || bad_) return false;
  if (got_ != want_) {
    bad_ = true;
    return false;
  }
  uint64_t end = (off_ + want_ + ALIGN - 1) / ALIGN * ALIGN;
  static const std::vector<char> z(ALIGN, 0);
  uint64_t pad = end - off_ - want_;
  if (pad) bad_ |= fwrite(z.data(), 1, pad, f_) != pad;
  off_ = end;
  return !bad_;
}
bool DmWriter::add(const std::string& name, const char* dtype, const void* data, uint64_t nbytes, const std::vector<int64_t>& shape) {
  return begin(name, dtype, nbytes, shape) && append(data, nbytes) && finish();
}
bool DmWriter::close(std::string& err) {
  if (!f_) return false;
  std::string head = "DMENGRES1 " + std::to_string(RESERVE) + "\n";
  for (auto& kv : cfg_) head += "cfg " + kv.first + " " + kv.second + "\n";
  for (auto& t : tab_) {
    head += "t " + t.name + " " + t.dtype + " " + std::to_string(t.off) + " " + std::to_string(t.nbytes) + " " + std::to_string(t.shape.size()) + " ";
    for (size_t i = 0; i < t.shape.size(); i++) head += (i ? " " : "") + std::to_string(t.shape[i]);
    head += "\n";
  }
  head += "end\n";
  if (head.size() > RESERVE) {
    err = "header larger than its reserved region";
    bad_ = true;
  }
  if (!bad_) {
    fseek(f_, 0, SEEK_SET);
    bad_ |= fwrite(head.data(), 1, head.size(), f_) != head.size();
  }
  bad_ |= fclose(f_) != 0;
  f_ = nullptr;
  if (bad_) {
    if (err.empty()) err = "write failed: " + tmp_;
    unlink(tmp_.c_str());
    return false;
  }
  if (rename(tmp_.c_str(), path_.c_str()) != 0) {
    err = "rename failed: " + path_;
    return false;
  }
  return true;
}
DmWriter::~DmWriter() {
  if (f_) {
    fclose(f_);
    unlink(tmp_.c_str());
  }
}

// ---------------------------------------------------------------------------------------------- hashing
std::string sha256_hex(const unsigned char* d) {
  static const char* hx = "0123456789abcdef";
  std::string s(64, '0');
  for (int i = 0; i < 32; i++) {
    s[2 * i] = hx[d[i] >> 4];
    s[2 * i + 1] = hx[d[i] & 15];
  }
  return s;
}

std::string sha256_file(const std::string& path, std::string& err, bool nocache) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    err = "cannot open " + path;
    return "";
  }
  if (nocache) no_cache(fd);
#if defined(__APPLE__) && !defined(MS_PORTABLE_SHA256)
  CC_SHA256_CTX c;
  CC_SHA256_Init(&c);
#else
  Sha256 c;
#endif
  std::vector<uint8_t> buf(8 << 20);
  for (;;) {
    ssize_t r = ::read(fd, buf.data(), buf.size());
    if (r < 0) {
      err = "read failed: " + path;
      ::close(fd);
      return "";
    }
    if (r == 0) break;
#if defined(__APPLE__) && !defined(MS_PORTABLE_SHA256)
    CC_SHA256_Update(&c, buf.data(), (CC_LONG)r);
#else
    c.update(buf.data(), (size_t)r);
#endif
  }
  ::close(fd);
  unsigned char d[32];
#if defined(__APPLE__) && !defined(MS_PORTABLE_SHA256)
  CC_SHA256_Final(d, &c);
#else
  c.final(d);
#endif
  return sha256_hex(d);
}

bool file_size(const std::string& path, uint64_t& out) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  out = st.st_size;
  return true;
}

bool files_identical(const std::string& a, const std::string& b, std::string& why) {
  uint64_t na, nb;
  if (!file_size(a, na) || !file_size(b, nb)) {
    why = "missing file";
    return false;
  }
  if (na != nb) {
    why = "sizes differ: " + std::to_string(na) + " vs " + std::to_string(nb);
    return false;
  }
  int fa = ::open(a.c_str(), O_RDONLY), fb = ::open(b.c_str(), O_RDONLY);
  no_cache(fa);
  no_cache(fb);
  std::vector<uint8_t> x(16 << 20), y(16 << 20);
  uint64_t off = 0;
  bool same = true;
  while (off < na) {
    size_t n = (size_t)std::min<uint64_t>(x.size(), na - off);
    if (!pread_all(fa, x.data(), n, off) || !pread_all(fb, y.data(), n, off)) {
      why = "read failed";
      same = false;
      break;
    }
    if (memcmp(x.data(), y.data(), n)) {
      size_t k = 0;
      while (x[k] == y[k]) k++;
      why = "first difference at byte " + std::to_string(off + k);
      same = false;
      break;
    }
    off += n;
  }
  ::close(fa);
  ::close(fb);
  return same;
}

}  // namespace ms
