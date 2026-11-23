// Saved hot set (src/serving/hotset.h).
#include "hotset.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace dm {

bool HotSet::load() {
  FILE* f = fopen(path_.c_str(), "rb");
  if (!f) return false;
  char mg[4];
  uint32_t ver = 0;
  int32_t nl = 0, ne = 0, n = 0;
  bool ok = fread(mg, 1, 4, f) == 4 && !memcmp(mg, "DMHS", 4) && fread(&ver, 4, 1, f) == 1 && ver == 1 && fread(&nl, 4, 1, f) == 1 &&
            fread(&ne, 4, 1, f) == 1 && fread(&n, 4, 1, f) == 1 && nl == lm_.info().n_layers && ne == lm_.info().n_experts && n >= 0 &&
            n <= nl * ne;
  if (ok) {
    ids_.resize(n);
    ok = (int)fread(ids_.data(), 4, n, f) == n;
  }
  fclose(f);
  if (!ok) ids_.clear();
  next_ = 0;
  return ok && !ids_.empty();
}

bool HotSet::save() const {
  std::vector<int32_t> ids = lm_.held_experts();
  if (ids.empty() || path_.empty()) return false;
  const std::string tmp = path_ + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  const uint32_t ver = 1;
  const int32_t nl = lm_.info().n_layers, ne = lm_.info().n_experts, n = (int32_t)ids.size();
  bool ok = fwrite("DMHS", 1, 4, f) == 4 && fwrite(&ver, 4, 1, f) == 1 && fwrite(&nl, 4, 1, f) == 1 && fwrite(&ne, 4, 1, f) == 1 &&
            fwrite(&n, 4, 1, f) == 1 && (int)fwrite(ids.data(), 4, n, f) == n;
  ok = (fclose(f) == 0) && ok;
  if (!ok) {
    remove(tmp.c_str());
    return false;
  }
  return rename(tmp.c_str(), path_.c_str()) == 0;
}

std::string HotSet::step(bool request_waiting) {
  if (reported_) return "";
  if (!request_waiting && !stopped_.load() && next_ < ids_.size()) {
    const size_t e = std::min(ids_.size(), next_ + (size_t)batch_);
    std::vector<int32_t> b(ids_.begin() + next_, ids_.begin() + e);
    auto t = std::chrono::steady_clock::now();
    lm_.preload_experts(b);  // waits until this batch has landed
    ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    next_ = e;
    if (next_ < ids_.size() && !stopped_.load()) return "";
  }
  reported_ = true;
  std::ostringstream s;
  s << "H 0 {\"hotset\":" << ids_.size() << ",\"refilled\":" << next_ << ",\"ms\":" << ms_ << ",\"stopped\":" << (next_ < ids_.size() ? "true" : "false") << "}";
  return s.str();
}

}  // namespace dm
