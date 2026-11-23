// A small JSON reader (no third-party code): objects keep their key order, numbers keep their source text, so a value
// can be printed back the way Python's json + str() print it (py_str). Enough for config.json, store/meta.json,
// moved-store.json and safetensors headers.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ms {

struct Json {
  enum Kind { NUL, BOOL, NUM, STR, ARR, OBJ } kind = NUL;
  bool b = false;
  double num = 0;
  bool is_int = false;       // a number written without '.', 'e' or 'E'
  std::string text;          // STR: the string; NUM: the source text
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  bool is_null() const { return kind == NUL; }
  const Json* get(const std::string& k) const;          // OBJ member or nullptr
  const Json& operator[](const std::string& k) const;   // OBJ member or a static null
  const Json& operator[](size_t i) const { return arr.at(i); }
  size_t size() const { return kind == ARR ? arr.size() : obj.size(); }
  long long i(long long dflt = 0) const;
  double f(double dflt = 0) const;
  const std::string& s() const { return text; }
  std::string str_or(const std::string& d) const { return kind == STR ? text : d; }
};

bool json_parse(const std::string& src, Json& out, std::string& err);
bool json_load(const std::string& path, Json& out, std::string& err);
std::string slurp(const std::string& path, bool* ok = nullptr);

// Python's repr() of a float (shortest round-trip digits; fixed notation for 1e-4 <= |x| < 1e16, else d.ddde-XX)
std::string py_float(double x);
// How Python's str() prints a value json.load returned: int -> digits, float -> py_float, bool -> True/False, str -> as is
std::string py_str(const Json& v);

}  // namespace ms
