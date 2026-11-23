// Derived engine files (FORMAT.md section 6), made in C++ from the shipped parts:
//   resident_q8   the engine's resident file (DMENGRES1) from resident.safetensors: 8-bit matrices by the family's rule
//                 (quant.h), every other tensor as today's Python converters write it (python/derive/qwen3_5_moe.py
//                 `resident()` for qwen3_5_moe, python/derive/inkling2_loop.py --bits 8 for inkling2_loop)
//   router .dmr   the engine's router file from routers/<id>.safetensors (python/derive/qwen3_5_moe.py `router()` / inkling2_loop_routers.py)
// Both are meant to be byte-identical to the files the Python converters write today.
#pragma once
#include <string>

#include "manifest.h"
#include "quant.h"

namespace ms {

struct DeriveStats {
  QStats q;
  int matrices = 0, tensors = 0;
  double seconds = 0;
};

// bits 8: resident_q8 (the default); bits 4 (qwen3_5_moe, P2): resident_q4, the same matrices by the same rule at 4 bits
// (NAME.q4 / .s / .b), everything the 8-bit file keeps exact kept exactly as there
bool derive_resident(const Manifest& m, const std::string& out_path, DeriveStats& st, std::string& err, int bits = 8);
// family: the router file's family (its metadata); bits 8 or 4; split: q36 only (pre_e / pre_p)
bool derive_router(const std::string& router_safetensors, const std::string& out_path, int bits, bool split,
                   DeriveStats& st, std::string& err);

}  // namespace ms
