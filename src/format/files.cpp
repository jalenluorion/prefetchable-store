#include "files.h"

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <sstream>

namespace ms {

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
  fcntl(fd, F_NOCACHE, 1);
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
  fcntl(fd, F_NOCACHE, 1);
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
  if (nocache) fcntl(fd, F_NOCACHE, 1);
  CC_SHA256_CTX c;
  CC_SHA256_Init(&c);
  std::vector<uint8_t> buf(8 << 20);
  for (;;) {
    ssize_t r = ::read(fd, buf.data(), buf.size());
    if (r < 0) {
      err = "read failed: " + path;
      ::close(fd);
      return "";
    }
    if (r == 0) break;
    CC_SHA256_Update(&c, buf.data(), (CC_LONG)r);
  }
  ::close(fd);
  unsigned char d[32];
  CC_SHA256_Final(d, &c);
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
  fcntl(fa, F_NOCACHE, 1);
  fcntl(fb, F_NOCACHE, 1);
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
