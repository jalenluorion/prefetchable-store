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
  int calibrated = 0;  // matrices taken from a calibrated export (calibrated != "")
  double seconds = 0;
};

// bits 8: resident_q8 (the default); bits 4 (qwen3_5_moe, P2): resident_q4, the same matrices by the same rule at 4 bits
// (NAME.q4 / .s / .b), everything the 8-bit file keeps exact kept exactly as there.
// calibrated (bits 4, qwen3_5_moe or inkling2_loop): a calibrated 4-bit export (jobs/babel/q36_q4c_export.py or ink_q4c_export.py: per matrix BASE.codes u8
// [R, C/2] two per byte with element 2j in the low nibble, BASE.scales / BASE.biases f16 [R, C/64], BASE.shape i64 [R, C];
// BASE = the release key without ".weight"): every 4-bit matrix's codes / scales / biases are copied from it as stored
// (no re-quantisation); every other tensor is derived exactly as for resident_q4. Every matrix of the export must be used.
bool derive_resident(const Manifest& m, const std::string& out_path, DeriveStats& st, std::string& err, int bits = 8,
                     const std::string& calibrated = "");
// family: the router file's family (its metadata); bits 8 or 4; split: q36 only (pre_e / pre_p)
bool derive_router(const std::string& router_safetensors, const std::string& out_path, int bits, bool split,
                   DeriveStats& st, std::string& err);

}  // namespace ms
