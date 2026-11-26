// swift-tools-version: 5.9
// The engine as a Swift package for the iPhone app (apps/ios): the core, the model families, the
// serving layer and the C API (bindings/c/dmeng_c.h) built from this repo's sources, plus a thin Swift wrapper
// (apps/ios/DmengKit). CMake (CMakeLists.txt) stays the build for the Mac tools; this file only adds the package view.
//
// Flags: one set for every Apple platform this package builds for, the A16's (-mcpu=apple-a16: dotprod, i8mm, fp16fml,
// bf16; no SME), so the Mac Catalyst / macOS build of the app runs the phone's code paths (the M4 is a superset). The
// SME kernels (src/core/kernels_sme.cpp, measurement only, never called) are left out. The Metal prefill split
// (src/core/gpu_gemm.mm, exact; off unless dm_open_opts.prefill_gpu_frac / prefill_gpu_experts > 0) is built in, with
// DM_METAL defined and Metal linked, on iOS and macOS. -ffp-contract=fast is the
// reference contraction of the core (CMakeLists.txt DM_CONTRACT); the format code (quantiser, derivations) is built
// with -ffp-contract=off as in CMake, as its own target.
import PackageDescription

let core: [CXXSetting] = [
  .headerSearchPath("include"),
  .unsafeFlags(["-O3", "-ffp-contract=fast", "-mcpu=apple-a16", "-Wno-unused-function"]),
]

let package = Package(
  name: "dmeng",
  platforms: [.macOS(.v14), .iOS(.v17)],
  products: [.library(name: "DmengKit", targets: ["DmengKit"])],
  targets: [
    .target(
      name: "DmengFormat",
      path: "src/format",
      publicHeadersPath: "public",
      cxxSettings: [.headerSearchPath("."), .unsafeFlags(["-O2", "-ffp-contract=off", "-fno-fast-math", "-Wno-deprecated-declarations"])]
    ),
    .target(
      name: "CDmeng",
      dependencies: ["DmengFormat"],
      path: ".",
      exclude: ["src/core/kernels_sme.cpp"],
      sources: ["src/core", "src/models", "src/serving", "bindings/c/dmeng_c.cpp"],
      publicHeadersPath: "bindings/c/public",
      cxxSettings: core + [.headerSearchPath("src/format"), .define("DM_METAL", to: "1")],
      linkerSettings: [.linkedLibrary("c++"), .linkedFramework("Metal"), .linkedFramework("Foundation")]
    ),
    .target(name: "DmengKit", dependencies: ["CDmeng"], path: "apps/ios/DmengKit"),
  ],
  cxxLanguageStandard: .cxx17
)
