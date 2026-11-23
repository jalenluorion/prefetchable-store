# apps/ios: the iPhone app on the unified engine

Status 2026-11-22: **P1** (the engine runs inside the app through its C API; gated on the Mac; nothing run on the
phone yet). Results: `results/ios/p0/P0.md` (measurements), `results/ios/p1/P1.md` (this stage).

## Where things live until the merge

| piece | where | why |
|---|---|---|
| C API | `bindings/c/dmeng_c.h`, `dmeng_c.cpp` (this repo) | engine-side: one boundary for Swift, Kotlin, Python |
| Swift package | `Package.swift` (this repo's root) + `apps/ios/DmengKit/Dmeng.swift` | builds `src/core`, `src/models`, `src/serving`, `src/format` and the C API from this repo's sources (no prebuilt binary), A16 flags, no SME file; Swift wrapper `DmengModel` |
| app sources | phone-store repo, `ios/PhoneStoreApp` (`DmengChatEngine.swift`, `-engine dmeng`) | the app (UI, tokenizer and chat template, tool loop, the MLX engine beside it) still lives there |
| text codec | phone-store `native/Sources/PhoneStore/DmengSupport.swift` (`TextCodec`) | the tokenizer is swift-transformers, already a dependency of the app's `PhoneStore` package |

How they connect: phone-store's `native/dmeng` is a symbolic link to this repo's checkout (a checkout of the `port-ios` branch
today); `native/Package.swift` depends on it with `.package(path: "dmeng")` and links the `DmengKit` product into
`PhoneStore`, which re-exports it (`@_exported import DmengKit`), so the app target needs no project change beyond
the new source file. Point the link at another checkout to build against it.

The choice that keeps the move a directory move: the engine-side pieces (C API, `Package.swift`, `DmengKit`) are
already here; the app-side pieces (`DmengChatEngine.swift`, `TextCodec`) are in the app's own folders and reference
the engine only through the `dmeng` package. Moving `phone-store/ios/PhoneStoreApp` and `phone-store/native` into
`apps/ios/` then needs only `native/Package.swift`'s `.package(path: "dmeng")` changed to the repo root (`../../..`).

## Build

See phone-store `ios/PhoneStoreApp/README.md` ("The dmeng engine"): the iOS device build and the Mac Catalyst build
are the app's usual `xcodebuild` lines; `run_dmeng_gate_mac.sh` builds for Catalyst and runs the identity gate
through the Swift path. The Mac tools stay on CMake (`cmake -S . -B build && cmake --build build -j3`, targets
`dmeng`, `dmserve`, `dm_gate`).
