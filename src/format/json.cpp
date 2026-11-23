#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace ms {

static const Json kNull;

const Json* Json::get(const std::string& k) const {
  if (kind != OBJ) return nullptr;
  for (auto& kv : obj)
    if (kv.first == k) return &kv.second;
  return nullptr;
}
const Json& Json::operator[](const std::string& k) const {
  const Json* p = get(k);
  return p ? *p : kNull;
}
long long Json::i(long long d) const {
  if (kind == NUM) return is_int ? strtoll(text.c_str(), nullptr, 10) : (long long)num;
  if (kind == BOOL) return b;
  return d;
}
double Json::f(double d) const { return kind == NUM ? num : (kind == BOOL ? (double)b : d); }

namespace {
struct P {
  const char* s;
  const char* e;
  std::string err;
  void ws() {
    while (s < e && (*s == ' ' || *s == '\n' || *s == '\r' || *s == '\t')) s++;
  }
  bool fail(const char* m) {
    if (err.empty()) err = m;
    return false;
  }
  static void utf8(std::string& o, uint32_t c) {
    if (c < 0x80) {
      o += (char)c;
    } else if (c < 0x800) {
      o += (char)(0xC0 | (c >> 6));
      o += (char)(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
      o += (char)(0xE0 | (c >> 12));
      o += (char)(0x80 | ((c >> 6) & 0x3F));
      o += (char)(0x80 | (c & 0x3F));
    } else {
      o += (char)(0xF0 | (c >> 18));
      o += (char)(0x80 | ((c >> 12) & 0x3F));
      o += (char)(0x80 | ((c >> 6) & 0x3F));
      o += (char)(0x80 | (c & 0x3F));
    }
  }
  bool hex4(uint32_t& v) {
    if (e - s < 4) return fail("short \\u escape");
    v = 0;
    for (int k = 0; k < 4; k++) {
      char c = *s++;
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else return fail("bad \\u escape");
    }
    return true;
  }
  bool str(std::string& o) {
    if (*s != '"') return fail("expected string");
    s++;
    while (s < e && *s != '"') {
      if (*s == '\\') {
        s++;
        if (s >= e) return fail("bad escape");
        char c = *s++;
        switch (c) {
          case '"': o += '"'; break;
          case '\\': o += '\\'; break;
          case '/': o += '/'; break;
          case 'b': o += '\b'; break;
          case 'f': o += '\f'; break;
          case 'n': o += '\n'; break;
          case 'r': o += '\r'; break;
          case 't': o += '\t'; break;
          case 'u': {
            uint32_t v;
            if (!hex4(v)) return false;
            if (v >= 0xD800 && v < 0xDC00 && e - s >= 6 && s[0] == '\\' && s[1] == 'u') {
              s += 2;
              uint32_t lo;
              if (!hex4(lo)) return false;
              v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
            }
            utf8(o, v);
            break;
          }
          default: return fail("bad escape");
        }
      } else {
        o += *s++;
      }
    }
    if (s >= e) return fail("unterminated string");
    s++;
    return true;
  }
  bool val(Json& v) {
    ws();
    if (s >= e) return fail("unexpected end");
    char c = *s;
    if (c == '{') {
      v.kind = Json::OBJ;
      s++;
      ws();
      if (s < e && *s == '}') {
        s++;
        return true;
      }
      for (;;) {
        ws();
        std::string k;
        if (!str(k)) return false;
        ws();
        if (s >= e || *s != ':') return fail("expected ':'");
        s++;
        v.obj.emplace_back(k, Json());
        if (!val(v.obj.back().second)) return false;
        ws();
        if (s < e && *s == ',') {
          s++;
          continue;
        }
        if (s < e && *s == '}') {
          s++;
          return true;
        }
        return fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      v.kind = Json::ARR;
      s++;
      ws();
      if (s < e && *s == ']') {
        s++;
        return true;
      }
      for (;;) {
        v.arr.emplace_back();
        if (!val(v.arr.back())) return false;
        ws();
        if (s < e && *s == ',') {
          s++;
          continue;
        }
        if (s < e && *s == ']') {
          s++;
          return true;
        }
        return fail("expected ',' or ']'");
      }
    }
    if (c == '"') {
      v.kind = Json::STR;
      return str(v.text);
    }
    if (e - s >= 4 && !strncmp(s, "true", 4)) {
      v.kind = Json::BOOL;
      v.b = true;
      s += 4;
      return true;
    }
    if (e - s >= 5 && !strncmp(s, "false", 5)) {
      v.kind = Json::BOOL;
      s += 5;
      return true;
    }
    if (e - s >= 4 && !strncmp(s, "null", 4)) {
      s += 4;
      return true;
    }
    // number (also Python's NaN / Infinity, which json.dump writes)
    const char* b0 = s;
    if (e - s >= 3 && !strncmp(s, "NaN", 3)) {
      s += 3;
    } else if (e - s >= 8 && !strncmp(s, "Infinity", 8)) {
      s += 8;
    } else if (e - s >= 9 && !strncmp(s, "-Infinity", 9)) {
      s += 9;
    } else {
      while (s < e && (isdigit((unsigned char)*s) || *s == '-' || *s == '+' || *s == '.' || *s == 'e' || *s == 'E')) s++;
    }
    if (s == b0) return fail("unexpected character");
    v.kind = Json::NUM;
    v.text.assign(b0, s);
    v.is_int = v.text.find_first_of(".eEIN") == std::string::npos;
    if (v.text == "NaN") v.num = NAN;
    else if (v.text == "Infinity") v.num = INFINITY;
    else if (v.text == "-Infinity") v.num = -INFINITY;
    else v.num = strtod(v.text.c_str(), nullptr);
    return true;
  }
};
}  // namespace

bool json_parse(const std::string& src, Json& out, std::string& err) {
  P p{src.data(), src.data() + src.size(), {}};
  out = Json();
  if (!p.val(out)) {
    err = p.err + " at byte " + std::to_string(p.s - src.data());
    return false;
  }
  return true;
}

std::string slurp(const std::string& path, bool* ok) {
  std::ifstream f(path, std::ios::binary);
  if (ok) *ok = (bool)f;
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool json_load(const std::string& path, Json& out, std::string& err) {
  bool ok;
  std::string s = slurp(path, &ok);
  if (!ok) {
    err = "cannot read " + path;
    return false;
  }
  if (!json_parse(s, out, err)) {
    err = path + ": " + err;
    return false;
  }
  return true;
}

std::string py_float(double x) {
  if (std::isnan(x)) return "nan";
  if (std::isinf(x)) return x > 0 ? "inf" : "-inf";
  if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64];
  int p = 1;
  for (; p <= 17; p++) {
    snprintf(buf, sizeof buf, "%.*e", p - 1, x);
    if (strtod(buf, nullptr) == x) break;
  }
  // buf = [-]d[.ddd]e(+|-)XX
  std::string s(buf);
  bool neg = s[0] == '-';
  if (neg) s.erase(0, 1);
  size_t epos = s.find('e');
  int ex = atoi(s.c_str() + epos + 1);
  std::string d;
  for (size_t k = 0; k < epos; k++)
    if (s[k] != '.') d += s[k];
  while (d.size() > 1 && d.back() == '0') d.pop_back();
  std::string o;
  if (ex >= -4 && ex < 16) {
    if (ex >= 0) {
      std::string ip = d.substr(0, std::min<size_t>(d.size(), ex + 1));
      while ((int)ip.size() < ex + 1) ip += '0';
      std::string fp = (int)d.size() > ex + 1 ? d.substr(ex + 1) : "0";
      o = ip + "." + fp;
    } else {
      o = "0." + std::string(-ex - 1, '0') + d;
    }
  } else {
    o = d.substr(0, 1);
    if (d.size() > 1) o += "." + d.substr(1);
    char eb[16];
    snprintf(eb, sizeof eb, "e%c%02d", ex < 0 ? '-' : '+', ex < 0 ? -ex : ex);
    o += eb;
  }
  return neg ? "-" + o : o;
}

std::string py_str(const Json& v) {
  switch (v.kind) {
    case Json::NUM: return v.is_int ? v.text : py_float(v.num);
    case Json::BOOL: return v.b ? "True" : "False";
    case Json::STR: return v.text;
    case Json::NUL: return "None";
    default: return "<container>";
  }
}

}  // namespace ms
