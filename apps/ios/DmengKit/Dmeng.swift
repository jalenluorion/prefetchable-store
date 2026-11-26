// DmengKit: the Swift view of the engine's C API (bindings/c/dmeng_c.h). Token ids in, logits out; no tokenizer here.
//
// Threads (the C API's contract): one caller at a time for open / sync / step / persist / stats / bench, which this
// wrapper serialises on its own queue (`run`); cancel() and setCacheBytes(_:) may be called from any thread.
@_exported import CDmeng
import Foundation

public struct DmengError: Error, CustomStringConvertible {
  public let code: Int32
  public let message: String
  public var description: String { "dmeng error \(code): \(message)" }
}

public struct DmengOptions: Sendable {
  public enum Memory: Int32, Sendable { case plain = 0, mlock = 1, purgeable = 2, file = 3 }
  public enum QoS: Int32, Sendable { case userInteractive = 0, userInitiated = 1, standard = 2, utility = 3 }
  public var threads = 0                   // 0: the platform default (iOS 4, macOS 8)
  public var qos = QoS.userInteractive
  public var perfCoresOnly = false
  public var readers = 0                   // 0: 4
  public var cacheBytes: Int64 = 0         // 0: 3.3e9
  public var memory: Memory? = nil         // nil: the platform default (iOS file, macOS mlock)
  public var residentMemory: Memory? = nil // nil: the platform default (iOS plain, macOS mlock)
  public var routersId: String? = nil      // nil: the profile's; "none" (cache only), "default", "linear_k2", ...
  public var routersFile: String? = nil
  public var routerM = 0                   // 0: 8
  public var derivedDir: String? = nil
  public var diskDir: String? = nil
  public var diskGB = 0.0
  public var checkpointGB = 0.0            // 0: the platform default (0.25 GB macOS; iOS: the pinned boundary and the newest)
  public var parkGB = 0.0                  // 0: the platform default (0.9 macOS, 0.3 iOS)
  public var ctx = 16384
  public var fileCheck = 1
  public var fileEvictHint = 0             // 0 none, 1 MADV_DONTNEED, 2 msync(MS_INVALIDATE) on eviction
  public var fileAdapt: Bool? = nil        // nil: the platform default (iOS on, macOS off)
  public var fileMinBytes: Int64 = 0       // 0: 1e9
  public var fileCoreBytes: Int64 = 0      // 0: no locked core
  public var embeddings: String? = nil     // nil: the manifest's file, else <model>/embeddings.safetensors
  // phone stage (P2): eviction policy, per-layer caps, the read planner, the readers' QoS
  public var evict: String? = nil          // nil: the core default (auto = hazard); "hazard", "freq"
  public var layerCaps = 0                 // 0 the core default (auto), 1 uniform per-layer caps, 2 one pool
  public var plan = 0                      // 0: fixed top routerM; 1: planner, marginal-cost rule; 2: planner, byte bucket
  public var planU = 0.0                   // 0: 0.9
  public var readerQoS: QoS? = nil         // nil: user-initiated
  public var readerIOLow: Bool? = nil      // the readers' disk I/O one tier below the app's; nil: the platform's default (iOS on, macOS off)
  public var routerTau = 0.0               // > 0: name the top routerCand with calibrated probability >= tau
  public var routerCand = 0                // 0: 16
  public var split = 0                     // pieces per record read; 0: 1
  public var mode: String? = nil           // nil: the profile's; "exact" or "fast" (the app's two choices)
  public var profile: String? = nil        // nil: from the device (phone / phone-small); "none": no profile defaults
  public var xtok: String? = nil           // nil: the profile's cross-token routers; "none": off
  public var residentBits = 0              // 0 / 8: the 8-bit resident; 4: the 4-bit variant (memory-limited devices)
  public var prefillThreads = 0            // 0: the C API default (iOS + 4-bit: 6)
  public var diskMin = 0                   // 0: the C API default (iOS 16)
  public var preloadExperts = false
  public var modelOptions: String? = nil  // "name=value;..." (nil: the C API default; iOS prefill_release=1)
  public var prefillGpuFrac = 0.0          // batched prefill: share of the attention projections' rows on the GPU (exact); 0: the profile's, < 0 off
  public var prefillGpuExperts = 0.0       // ... of the experts' rows (routed experts in the slab, shared expert); 0: the profile's, < 0 off
  public var prefillGpuMin = 0             // sub-chunks of at least this many positions use the GPU; 0: 256
  public var prefillGpuIntKernel = false   // the integer GPU kernel instead of the matrix-unit one (the same bits)
  public var prefillGpuMinWave = 0         // a wave of landed experts uses the GPU from this many assignments; 0: 128
  public var runAhead = 0                  // picks the run-ahead names while a step waits; 0: the profile's, < 0 off
  public var runAheadDepth = 0             // 0: the profile's (1)
  public var runAheadBudget = 0.0          // speculative ms per token at most this x compute (0: the profile's, < 0 no limit)
  public var runAheadMax = 0               // run-aheads per token at most (0: the profile's, < 0 no limit)
  public var runAheadSatSkip = false       // true: none while every reader is busy and reads are queued (false: the profile's)
  public var runAheadAB = false            // measurement: off / on alternated token by token (dm_stats_json ra_ab)
  public init() {}
}

public struct DmengSync: Sendable {
  public var reused: Int, computed: Int
  public var source: String
  public var msTotal: Double, msRestore: Double, msPrefill: Double, msWaitReads: Double, mbRead: Double
  public var cancelled: Bool
}

public struct DmengSampler: Sendable, Equatable {
  public var temp = 0.0, topK = 0, topP = 1.0, minP = 0.0
  public var presencePenalty = 0.0, repetitionPenalty = 1.0, frequencyPenalty = 0.0
  public var seed: UInt64 = 0
  public init() {}
  public static let greedy = DmengSampler()
  public var isGreedy: Bool { temp <= 0 && presencePenalty == 0 && repetitionPenalty == 1 && frequencyPenalty == 0 }
}

/// One open model (dm_engine). Not Sendable by the compiler's rules; the documented threads above make it safe.
public final class DmengModel: @unchecked Sendable {
  private var e: OpaquePointer?
  public let info: dm_info_t
  public let vocab: Int
  /// The serial queue every call except cancel / setCacheBytes runs on (user-interactive QoS).
  public let queue = DispatchQueue(label: "dmeng.engine", qos: .userInteractive)

  public init(modelDir: String, options o: DmengOptions = DmengOptions()) throws {
    var c = dm_open_opts()
    dm_default_opts(&c)
    c.threads = Int32(o.threads)
    c.qos = o.qos.rawValue
    c.core_class = o.perfCoresOnly ? Int32(DM_CORES_PERF) : Int32(DM_CORES_ANY)
    c.readers = Int32(o.readers)
    c.cache_bytes = o.cacheBytes
    if let m = o.memory { c.memory_mode = m.rawValue }
    if let m = o.residentMemory { c.res_memory_mode = m.rawValue }
    c.router_m = Int32(o.routerM)
    c.disk_gb = o.diskGB
    c.ckpt_gb = o.checkpointGB
    c.park_gb = o.parkGB
    c.ctx = Int32(o.ctx)
    c.file_check = Int32(o.fileCheck)
    c.file_evict_hint = Int32(o.fileEvictHint)
    if let a = o.fileAdapt { c.file_adapt = a ? 1 : 0 }
    c.file_min_bytes = o.fileMinBytes
    c.file_core_bytes = o.fileCoreBytes
    c.layer_caps = Int32(o.layerCaps)
    c.plan = Int32(o.plan)
    c.plan_u = o.planU
    if let q = o.readerQoS { c.reader_qos = q.rawValue + 1 }
    c.router_tau = o.routerTau
    c.router_cand = Int32(o.routerCand)
    c.split = Int32(o.split)
    c.resident_bits = Int32(o.residentBits)
    c.prefill_threads = Int32(o.prefillThreads)
    c.disk_min = Int32(o.diskMin)
    c.preload_experts = o.preloadExperts ? 1 : 0
    c.prefill_gpu_frac = o.prefillGpuFrac
    c.prefill_gpu_experts = o.prefillGpuExperts
    c.prefill_gpu_min = Int32(o.prefillGpuMin)
    c.prefill_gpu_int_kernel = o.prefillGpuIntKernel ? 1 : 0
    c.prefill_gpu_min_wave = Int32(o.prefillGpuMinWave)
    c.run_ahead = Int32(o.runAhead)
    c.run_ahead_depth = Int32(o.runAheadDepth)
    c.run_ahead_budget = o.runAheadBudget
    c.run_ahead_max = Int32(o.runAheadMax)
    c.run_ahead_sat_skip = o.runAheadSatSkip ? 1 : 0
    c.run_ahead_ab = o.runAheadAB ? 1 : 0
    if let v = o.readerIOLow { c.reader_io_low = v ? 1 : -1 }
    // the C strings live for the dm_open call only (dm_open copies what it keeps)
    let strs = [o.routersId, o.routersFile, o.derivedDir, o.diskDir, o.embeddings, o.evict, o.mode, o.profile, o.xtok, o.modelOptions].map { $0.flatMap { strdup($0) } }
    defer { strs.forEach { free($0) } }
    c.routers_id = UnsafePointer(strs[0])
    c.routers = UnsafePointer(strs[1])
    c.derived_dir = UnsafePointer(strs[2])
    c.disk_dir = UnsafePointer(strs[3])
    c.embeddings = UnsafePointer(strs[4])
    c.evict = UnsafePointer(strs[5])
    c.mode = UnsafePointer(strs[6])
    c.profile = UnsafePointer(strs[7])
    c.xtok = UnsafePointer(strs[8])
    c.model_options = UnsafePointer(strs[9])
    guard let h = dm_open(modelDir, &c) else {
      throw DmengError(code: Int32(DM_ELOAD), message: String(cString: dm_last_error(nil)))
    }
    e = h
    var i = dm_info_t()
    dm_info(h, &i)
    info = i
    vocab = Int(i.vocab)
  }

  deinit { if let e { dm_close(e) } }

  public var position: Int {
    var i = dm_info_t()
    dm_info(e, &i)
    return Int(i.pos)
  }

  public var routersFile: String { withUnsafeBytes(of: info.routers) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) } }
  public var name: String { withUnsafeBytes(of: info.name) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) } }

  /// Bring the live state to `ids` (the least recomputation) and fill `logits` (vocab floats). Call on `queue`.
  /// `progress(done, total, mbRead)` runs on this thread during the prefill.
  public func sync(_ ids: [Int32], marks: [Int32] = [], logits: inout [Float],
                   progress: ((Int, Int, Double) -> Void)? = nil) throws -> DmengSync {
    precondition(logits.count >= vocab)
    final class Box { let f: ((Int, Int, Double) -> Void)?; init(_ f: ((Int, Int, Double) -> Void)?) { self.f = f } }
    let box = Box(progress)
    let user = Unmanaged.passUnretained(box).toOpaque()
    var r = dm_sync_result()
    let rc = ids.withUnsafeBufferPointer { ip in
      marks.withUnsafeBufferPointer { mp in
        logits.withUnsafeMutableBufferPointer { lp in
          dm_sync(e, ip.baseAddress, Int32(ip.count), mp.baseAddress, Int32(mp.count), lp.baseAddress,
                  { u, done, total, mb in
                    Unmanaged<Box>.fromOpaque(u!).takeUnretainedValue().f?(Int(done), Int(total), mb)
                  }, user, &r)
        }
      }
    }
    withExtendedLifetime(box) {}
    if rc != DM_OK && rc != DM_ECANCELLED { throw DmengError(code: rc, message: "dm_sync") }
    let src = withUnsafeBytes(of: r.source) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
    return DmengSync(reused: Int(r.reused), computed: Int(r.computed), source: src, msTotal: r.ms_total, msRestore: r.ms_restore,
                     msPrefill: r.ms_prefill, msWaitReads: r.ms_wait_reads, mbRead: r.mb_read, cancelled: rc == DM_ECANCELLED)
  }

  /// State only (no logits): bring the live state to `ids`, restoring a saved state at its end whole (idle time: the
  /// system boundary, a chat up to the next user header). The next call must be `sync` (with logits), not `step`.
  public func prefill(_ ids: [Int32], marks: [Int32] = []) throws -> DmengSync {
    var r = dm_sync_result()
    let rc = ids.withUnsafeBufferPointer { ip in
      marks.withUnsafeBufferPointer { mp in
        dm_sync(e, ip.baseAddress, Int32(ip.count), mp.baseAddress, Int32(mp.count), nil, nil, nil, &r)
      }
    }
    if rc != DM_OK && rc != DM_ECANCELLED { throw DmengError(code: rc, message: "dm_sync (state only)") }
    let src = withUnsafeBytes(of: r.source) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
    return DmengSync(reused: Int(r.reused), computed: Int(r.computed), source: src, msTotal: r.ms_total, msRestore: r.ms_restore,
                     msPrefill: r.ms_prefill, msWaitReads: r.ms_wait_reads, mbRead: r.mb_read, cancelled: rc == DM_ECANCELLED)
  }

  /// One generated token; false if a cancel was pending (nothing computed). Call on `queue`.
  public func step(_ id: Int32, logits: inout [Float]) throws -> Bool {
    let rc = logits.withUnsafeMutableBufferPointer { dm_step(e, id, $0.baseAddress) }
    if rc == DM_ECANCELLED { return false }
    if rc != DM_OK { throw DmengError(code: rc, message: "dm_step") }
    return true
  }

  /// The C sampler (identical on the Mac and the phone). `history`: this reply's tokens so far.
  public static func sample(_ logits: [Float], vocab: Int, _ s: DmengSampler, history: [Int32]) -> Int32 {
    var c = dm_sampler()
    dm_sampler_default(&c)
    c.temp = s.temp
    c.top_k = Int32(s.topK)
    c.top_p = s.topP
    c.min_p = s.minP
    c.presence_penalty = s.presencePenalty
    c.repetition_penalty = s.repetitionPenalty
    c.frequency_penalty = s.frequencyPenalty
    c.seed = s.seed
    return logits.withUnsafeBufferPointer { lp in
      history.withUnsafeBufferPointer { hp in dm_sample(lp.baseAddress, Int32(vocab), &c, hp.baseAddress, Int32(hp.count)) }
    }
  }

  /// The prefill GPU split's self-test on this device (no model; ~8 x seconds): both GPU kernels against the CPU kernels
  /// bit for bit, then GMAC/s of each alone, the CPU's 4-bit kernel on `threads`, and both together. JSON ("exact": true).
  public static func gpuCheck(threads: Int = 4, seconds: Double = 2) -> String {
    String(cString: dm_gpu_check(Int32(threads), seconds))
  }

  public static func argmax(_ logits: [Float], vocab: Int) -> Int32 {
    logits.withUnsafeBufferPointer { dm_argmax($0.baseAddress, Int32(vocab)) }
  }

  /// Any thread.
  public func cancel() { dm_cancel(e) }
  /// Any thread: applied at once when idle, else at the next token / prefill layer.
  public func setCacheBytes(_ b: Int64) { dm_set_cache_bytes(e, b) }
  /// Spare memory (the system's file cache under the expert cache): stop adding to it for the rest of the session.
  /// Any thread; false when it was not on.
  @discardableResult public func pauseSpareMemory() -> Bool { dm_spare_memory_pause(e) == DM_OK }
  /// The model's expert files the system's file cache holds now, bytes (nil when they are not mapped: spare memory
  /// off). Any thread.
  public var storeResidentBytes: Int64? { let b = dm_store_resident_bytes(e); return b < 0 ? nil : b }

  public func persist(_ pos: Int) { dm_persist(e, Int32(pos)) }
  /// Drop a pending cancel that found nothing running (call on `queue`, after a cancel of background work).
  public func clearCancel() { dm_cancel_clear(e) }
  /// The last sync's prefill account (JSON). Call on `queue`.
  public var lastPrefillJSON: String { String(cString: dm_last_prefill_json(e)) }

  public var counters: dm_counters_t {
    var c = dm_counters_t()
    dm_counters(e, &c)
    return c
  }

  /// JSON of what ran since the last call (call on `queue`).
  public func statsJSON() -> String { String(cString: dm_stats_json(e)) }

  public func bench(_ kind: String, spec: String = "") -> String? {
    var out: UnsafeMutablePointer<CChar>? = nil
    guard dm_bench(e, kind, spec, &out) == DM_OK, let o = out else { return nil }
    defer { free(o) }
    return String(cString: o)
  }

  /// Run `f` on the engine's queue and await it.
  public func run<T: Sendable>(_ f: @escaping @Sendable () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { (k: CheckedContinuation<T, Error>) in
      queue.async {
        do { k.resume(returning: try f()) } catch { k.resume(throwing: error) }
      }
    }
  }
}
