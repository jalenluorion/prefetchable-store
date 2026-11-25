// File readers and writers the format needs: safetensors (header + tensor reads), the engine's DMENGRES1 files
// (reader and the streaming writer, the layout of python/derive/qwen3_5_moe.py's Writer), sha256 (CommonCrypto on Apple platforms, a portable one elsewhere).
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "json.h"

namespace ms {

// ---------------------------------------------------------------------------------------------- safetensors
struct StTensor {
  std::string name, dtype;   // F16, BF16, F32, U8, U32, ...
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // byte range inside the data section
  int64_t numel() const;
};

struct SafeTensors {
  std::string path;
  uint64_t header_len = 0;          // the u64 at the front
  std::vector<StTensor> tensors;    // header order
  std::map<std::string, size_t> index;
  std::vector<std::pair<std::string, std::string>> metadata;  // __metadata__ (string map)
  int fd = -1;
  bool open(const std::string& p, std::string& err);
  ~SafeTensors();
  const StTensor* find(const std::string& n) const;
  uint64_t data_offset(const StTensor& t) const { return 8 + header_len + t.begin; }  // absolute file offset
  bool read(const StTensor& t, void* dst, std::string& err) const;  // the whole tensor (F_NOCACHE reads)
  std::string meta(const std::string& k, const std::string& d = "") const;
};

// ---------------------------------------------------------------------------------------------- npz
// numpy's np.savez archive read where it lies: the zip's central directory (zip64 included, as numpy writes it) gives
// each member's local header, each member must be stored (not deflated), and its .npy header gives dtype and shape. A
// tensor is then a byte range of the archive, read like a safetensors tensor. Names drop the ".npy"; dtypes are named as
// safetensors names them (F32, F16, I64, ...); begin / end are absolute file offsets. C order only.
struct Npz {
  std::string path;
  std::vector<StTensor> tensors;
  std::map<std::string, size_t> index;
  int fd = -1;
  bool open(const std::string& p, std::string& err);
  ~Npz();
  const StTensor* find(const std::string& n) const;
  bool read(const StTensor& t, void* dst, std::string& err) const;
};

// ---------------------------------------------------------------------------------------------- DMENGRES1
struct DmTensor {
  std::string name, dtype;
  uint64_t off = 0, nbytes = 0;
  std::vector<int64_t> shape;
};
struct DmFile {
  std::string path;
  uint64_t hdr = 0;
  std::vector<std::pair<std::string, std::string>> cfg;  // header order
  std::vector<DmTensor> tensors;
  std::map<std::string, size_t> index;
  bool open(const std::string& p, std::string& err);   // header only
  const std::string* get(const std::string& k) const;
  const DmTensor* find(const std::string& n) const;
  bool read(const DmTensor& t, std::vector<uint8_t>& out, std::string& err) const;
};

// The writer of python/derive/qwen3_5_moe.py (and the two Inkling scripts): a 1 MiB region reserved for the text header,
// then every tensor at a 16 KiB-aligned offset, zero-padded; the header is written at close. Writes PATH.tmp, renames.
class DmWriter {
 public:
  static constexpr uint64_t RESERVE = 1 << 20, ALIGN = 16384;
  bool open(const std::string& path, std::string& err);
  void cfg(const std::string& k, const std::string& v) { cfg_.emplace_back(k, v); }
  bool add(const std::string& name, const char* dtype, const void* data, uint64_t nbytes, const std::vector<int64_t>& shape);
  // streaming form of add: begin, any number of appends totalling nbytes, finish
  bool begin(const std::string& name, const char* dtype, uint64_t nbytes, const std::vector<int64_t>& shape);
  bool append(const void* data, uint64_t n);
  bool finish();
  bool close(std::string& err);
  ~DmWriter();
  uint64_t payload_bytes() const { return off_; }

 private:
  std::string path_, tmp_;
  FILE* f_ = nullptr;
  uint64_t off_ = 0;
  bool bad_ = false;
  uint64_t want_ = 0, got_ = 0;
  std::vector<std::pair<std::string, std::string>> cfg_;
  std::vector<DmTensor> tab_;
};

// ---------------------------------------------------------------------------------------------- hashing
std::string sha256_file(const std::string& path, std::string& err, bool nocache = true);
std::string sha256_hex(const unsigned char* digest32);
bool file_size(const std::string& path, uint64_t& out);
bool files_identical(const std::string& a, const std::string& b, std::string& why);

}  // namespace ms
