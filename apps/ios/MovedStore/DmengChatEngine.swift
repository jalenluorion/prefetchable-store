import Foundation
import os
import DmengKit
#if canImport(UIKit)
import UIKit
#endif

/// The engine (DmengKit, the engine's C API) behind the app's ChatEngine protocol.
///
/// The engine takes token ids and returns logits; the app keeps the text side: the model folder's tokenizer and chat
/// template (TextCodec: swift-transformers for Qwen3.6, InklingChat for Inkling 2 Mini), the history rendering
/// (templateMessages), sampling (GenSettings mapped onto the engine's C sampler, so the Mac and the phone draw
/// identically), the think-block tracker and budget, and the tool loop (ChatModel). The engine's prefix cache brings every
/// prompt in with the least recomputation (live state, checkpoint, disk copy of the system prompt or of a chat), lossless.
///
/// Memory warnings shrink the cache budget only when the available memory is below 1 GB, backgrounding sets it to the
/// floor, the next foreground restores it (dm_set_cache_bytes).
enum DmengLaunch {
    /// The selected model (Settings > Model; ModelKind) and its folder.
    static var kind: ModelKind { ModelKind.selected }
    static var modelDir: URL { kind.folder }
    /// What is missing in a model folder for this device's profile (empty: complete): manifest, the 40 store layers, the
    /// profile's in-memory weights, the embedding table, tokenizer and template. Inkling: inklingCompleteness.
    static func completeness(_ d: URL, kind: ModelKind = .qwen36) -> [String] {
        if kind == .inkling { return inklingCompleteness(d) }
        let fm = FileManager.default
        func sz(_ p: String) -> Int64 { ((try? fm.attributesOfItem(atPath: d.appendingPathComponent(p).path))?[.size] as? NSNumber)?.int64Value ?? -1 }
        var miss: [String] = []
        for f in ["moved-store.json", "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"]
            where sz(f) <= 0 { miss.append(f) }
        if sz("embeddings.safetensors") != 1_017_119_224 { miss.append("embeddings.safetensors") }
        for L in 0..<40 where sz(String(format: "store/layer_%02d.bin", L)) != 452_984_832 { miss.append(String(format: "store/layer_%02d.bin", L)) }
        // the in-memory weights of the device's profile, in derived/ (the Hub's layout) or .dmeng/
        let want4 = profile == "phone-small"
        let (pre, bytes): (String, Int64) = want4 ? ("resident_q4", 1_135_001_600) : ("resident_q8", 2_091_827_200)
        let found = ["derived", ".dmeng"].contains { sub in
            ((try? fm.contentsOfDirectory(atPath: d.appendingPathComponent(sub).path)) ?? []).contains { $0.hasPrefix(pre + ".") && sz(sub + "/" + $0) == bytes }
        }
        if !found { miss.append("derived/\(pre) (\(bytes) B)") }
        return miss
    }
    /// Inkling: every part of the profile's list (the downloaded manifest's) at its size (a derived file also counts in
    /// .dmeng/), documentation files and moved-store.json's size aside; without a manifest, the files the engine needs
    /// by name.
    static func inklingCompleteness(_ d: URL) -> [String] {
        func sz(_ p: String) -> Int64 { ModelDownloader.resolvedSize(d.appendingPathComponent(p).path) ?? -1 }
        var want: [(String, Int64)] = []
        for f in [".download/manifest.json", "moved-store.json"] {
            if let data = try? Data(contentsOf: d.appendingPathComponent(f)),
               let m = try? MovedStoreManifest.manifest(from: data, profile: profile) {
                want = m.files.map { ($0.local, $0.size) }
                break
            }
        }
        if want.isEmpty {
            want = ["moved-store.json", "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                    "embeddings.safetensors"].map { ($0, 0) } + (0..<24).map { (String(format: "store/layer_%02d.bin", $0), 0) }
        }
        let docs: Set<String> = ["LICENSE", "NOTICE.md", "README.md", "sha256.txt"]
        var miss: [String] = []
        for (p, n) in want where !docs.contains(p) {
            var got = sz(p)
            if got < 0, p.hasPrefix("derived/") { got = sz(".dmeng/" + (p as NSString).lastPathComponent) }
            if got <= 0 || (n > 0 && p != "moved-store.json" && got != n) { miss.append(p) }
        }
        return miss
    }
    /// The expert cache: 0 on iOS, the profile's slab rule decides (phone-small: profiles.phone-small.cache_slab, the
    /// footprint target less the model's measured non-cache footprint; phone: what iOS makes available); on the Mac 3.3 GB.
    static var cacheGB: Double {
        #if os(iOS) && !targetEnvironment(macCatalyst)
        return 0
        #else
        return 3.3
        #endif
    }

    // MARK: profiles and modes (two modes, three profiles picked from the device)
    /// The memory iOS makes available to the app at launch (0 on the Mac).
    static let availableAtOpen: UInt64 = {
        #if os(iOS) && !targetEnvironment(macCatalyst)
        return UInt64(os_proc_available_memory())
        #else
        return 0
        #endif
    }()
    /// The memory class at open: `mac`, `phone` (iOS, at least 5.0 GB available), `phone-small` (less: the 6 GB class).
    static let resident4BelowBytes: UInt64 = 5_000_000_000
    static var profile: String {
        #if os(iOS) && !targetEnvironment(macCatalyst)
        return availableAtOpen < resident4BelowBytes ? "phone-small" : "phone"
        #else
        return "mac"
        #endif
    }
    static var isPhone: Bool {
        #if os(iOS) && !targetEnvironment(macCatalyst)
        return true
        #else
        return false
        #endif
    }
    /// The user's choice (Settings: Exact / Fast), else the profile's default: Fast on phone and phone-small, Exact elsewhere.
    static var userMode: String {
        get {
            #if SELFCHECK
            if let o = SelfCheck.modeOverride { return o }
            #endif
            return UserDefaults.standard.string(forKey: kind.modeKey) ?? kind.defaultMode(profile: profile)
        }
        set { UserDefaults.standard.set(newValue, forKey: kind.modeKey) }
    }
    /// The engine mode for dm_open: nil = exact; Fast is the model's fast preset (ModelKind.fastEngineMode).
    static var engineMode: String? {
        switch userMode {
        case "fast": return kind.fastEngineMode
        default: return nil
        }
    }
    /// The in-memory weights' precision: 4-bit on phone-small when the folder holds the 4-bit copy, else 8-bit.
    static func residentBits(dir: String) -> Int {
        let has4 = ["/derived", "/.dmeng"].contains { ((try? FileManager.default.contentsOfDirectory(atPath: dir + $0)) ?? []).contains { $0.hasPrefix("resident_q4.") } }
        return profile == "phone-small" && has4 ? 4 : 8
    }
    /// Spare memory for the model's files (the system's file cache under the expert cache): off by default; the
    /// switch (Diagnostics) is offered on phones with at least 11 GB of memory, and a choice made with it is passed to
    /// the engine as an explicit override at the next open.
    static let spareMemoryKey = "engine.spareMemory"
    static var spareMemoryOffered: Bool { isPhone && ProcessInfo.processInfo.physicalMemory >= 11_000_000_000 }
    /// The user's choice, nil when never set (the engine's rule).
    static var spareMemoryChoice: Bool? {
        get { UserDefaults.standard.object(forKey: spareMemoryKey) as? Bool }
        set { UserDefaults.standard.set(newValue, forKey: spareMemoryKey) }
    }
    /// The 35B's early router (the manifest's linear_k2); Inkling takes its profile's routers.
    static let defaultRouters = "linear_k2"
    /// The selected model's saved states (one folder per model: a 35B state is never offered to Inkling).
    static var diskDir: URL { prefixDir(kind) }
    static func prefixDir(_ k: ModelKind) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent(Bundle.main.bundleIdentifier ?? "app", isDirectory: true)
            .appendingPathComponent(k.prefixDirName, isDirectory: true)
    }
}

/// Which moved-store.json the engine opens with: the release line's, bundled with the app at build time (the Xcode build
/// phase "Bundle the line's manifests" copies python/convert/<family>/moved-store.json into manifests/<family>.json), or
/// the model folder's own (what the Hub served). The bundled one is used when all of these hold, else the folder's:
///   - the two differ in content (equal: nothing to do);
///   - the bundled one is not older: its `generated.date` (ISO yyyy-mm-dd) is at least the folder's (a Hub manifest
///     with a later date wins, so the bundle becomes a no-op once the Hub catches up), and its `format_version` major is
///     the one this engine reads;
///   - every part its profile lists for this device (profiles.<profile>.parts) is on disk at the listed size (a derived
///     file may sit in .dmeng/), moved-store.json and documentation files aside.
/// The engine reads <dir>/moved-store.json, so the bundled one is opened through a folder of links: every entry of the
/// model folder linked, and the bundled manifest as its moved-store.json (Application Support/<bundle>/manifest-line/).
/// No setting is copied into Swift: the engine reads the whole manifest as it would from the Hub.
enum ManifestChoice {
    struct Pick { var dir: String; var bundled: Bool; var label: String }
    static func family(_ k: ModelKind) -> String { k == .qwen36 ? "qwen3_5_moe" : "inkling2_loop" }
    static func bundledURL(_ k: ModelKind) -> URL? {
        Bundle.main.url(forResource: family(k), withExtension: "json", subdirectory: "manifests")
    }
    private static func stamp(_ j: [String: Any]) -> (date: String, major: Int) {
        let d = (j["generated"] as? [String: Any])?["date"] as? String ?? ""
        let v = (j["format_version"] as? String) ?? (j["format_version"] as? NSNumber)?.stringValue ?? ""
        return (d, Int(v.split(separator: ".").first ?? "") ?? -1)
    }
    static func choose(kind: ModelKind, dir: String, profile: String) -> Pick {
        let fm = FileManager.default
        let own = URL(fileURLWithPath: dir).appendingPathComponent("moved-store.json")
        func folder(_ why: String) -> Pick {
            log("MANIFEST the model folder's (downloaded): " + why)
            return Pick(dir: dir, bundled: false, label: "downloaded (" + why + ")")
        }
        guard let bu = bundledURL(kind), let bd = try? Data(contentsOf: bu),
              let bj = (try? JSONSerialization.jsonObject(with: bd)) as? [String: Any] else { return folder("no bundled manifest") }
        guard let od = try? Data(contentsOf: own), let oj = (try? JSONSerialization.jsonObject(with: od)) as? [String: Any] else {
            return folder("the folder has no readable moved-store.json")
        }
        if od == bd { return folder("identical to the bundled one") }
        let b = stamp(bj), o = stamp(oj)
        if b.major != 1 { return folder("the bundled one's format_version major \(b.major) is not 1") }
        if b.date.isEmpty || b.date < o.date { return folder("the bundled one (generated \(b.date)) is older than the folder's (\(o.date))") }
        // every part the bundled profile lists, at its size
        var missing: [String] = []
        if let m = try? MovedStoreManifest.manifest(from: bd, profile: profile) {
            let docs: Set<String> = ["LICENSE", "NOTICE.md", "README.md", "sha256.txt", "moved-store.json"]
            for f in m.files where !docs.contains(f.local) {
                var got = ModelDownloader.resolvedSize(dir + "/" + f.local) ?? -1
                if got < 0, f.local.hasPrefix("derived/") { got = ModelDownloader.resolvedSize(dir + "/.dmeng/" + (f.local as NSString).lastPathComponent) ?? -1 }
                if got != f.size { missing.append(f.local) }
            }
        } else {
            missing.append("profiles.\(profile).parts")
        }
        if !missing.isEmpty {
            return folder("the bundled one's profile \(profile) lists \(missing.count) part(s) not on disk at their size: "
                          + missing.prefix(3).joined(separator: ", "))
        }
        // the folder of links
        let base = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? "app", isDirectory: true)
            .appendingPathComponent("manifest-line", isDirectory: true).appendingPathComponent(kind.rawValue, isDirectory: true)
        do {
            try? fm.removeItem(at: base)
            try fm.createDirectory(at: base, withIntermediateDirectories: true)
            for e in try fm.contentsOfDirectory(atPath: dir) where e != "moved-store.json" {
                try fm.createSymbolicLink(atPath: base.appendingPathComponent(e).path, withDestinationPath: dir + "/" + e)
            }
            try bd.write(to: base.appendingPathComponent("moved-store.json"))
        } catch {
            return folder("the folder of links could not be made (\(error.localizedDescription))")
        }
        let why = "generated \(b.date) against the folder's \(o.date.isEmpty ? "none" : o.date), contents differ, every part of profile \(profile) present"
        log("MANIFEST the bundled one (the release line's): " + why)
        return Pick(dir: base.path, bundled: true, label: "bundled with the app (the release line's: " + why + ")")
    }
}

final class DmengChatEngine: ChatEngine, @unchecked Sendable {
    private(set) var info = EngineInfo(name: "dmeng", detail: "not loaded")
    private let lock = NSLock()
    private var _thinking = false
    private var model: DmengModel?
    private var codec: TextCodec?
    private var _last: ReplyDetail?
    private var cancelled = false
    private var openBytes: Int64 = 0
    private var observers: [NSObjectProtocol] = []
    /// The model this engine opened (nil before the first load and after unload).
    private(set) var loadedKind: ModelKind?
    /// The engine's settings after precedence, read once at open (dm_stats_json "resolved"): the Diagnostics rows.
    private var resolvedAtOpen: [String: Any] = [:]
    var prefixCacheEnabled = true  // the engine's prefix cache is always on (in memory; disk for the system prompt)

    var thinking: Bool {
        get { lock.withLock { _thinking } }
        set { lock.withLock { _thinking = newValue } }
    }
    var lastDetail: ReplyDetail? { lock.withLock { _last } }
    private var _replyRows: [(String, String)] = []
    /// The last finished reply from the engine's counters (dm_counters deltas over the reply's decode), plain words.
    var replyRows: [(String, String)] { lock.withLock { _replyRows } }
    /// The profile as the engine read it at open (dm_stats_json policy.preset): engine_mode, engine_mode_from.
    private var presetAtOpen: [String: Any] = [:]
    /// The manifest this open used (ManifestChoice): the Engine row "manifest".
    private var manifestPick: ManifestChoice.Pick?
    /// Spare memory as the engine resolved it at open (nil before a model is open).
    var spareMemoryResolved: Bool? {
        lock.withLock { model == nil ? nil : (resolvedAtOpen["spare_memory"] as? Bool) == true }
    }

    /// Rows for one reply: the counters at the decode's start and end, its tokens and seconds.
    /// `storeResident`: the model's files the system's file cache held at the end (nil: not tracked, spare memory off);
    /// `bigMemory`: a phone where the spare-memory switch is offered (its rows show whether it is on or off).
    static func replyRows(_ c0: dm_counters_t, _ c1: dm_counters_t, tokens: Int, speed: GenSpeed, storeResident: Int64? = nil,
                          bigMemory: Bool = false) -> [(String, String)] {
        let seconds = speed.seconds
        let n = Double(max(1, tokens))
        let ms = 1000 * seconds / n
        let wait = (c1.wait_ms - c0.wait_ms) / n
        let read = Double(c1.bytes_read &- c0.bytes_read)
        let spare = Double(c1.bytes_spare &- c0.bytes_spare)
        let busy = Double(c1.read_busy_ns &- c0.read_busy_ns) / 1e9
        let picks = Double(c1.picks &- c0.picks)
        let vm = MemProbe.vmInfo()
        var r: [(String, String)] = [
            ("Speed", speed.detail),  // this round's; ChatModel.lastReplyRows shows the message's (GenSpeed)
            ("Read from storage", String(format: "%.1f MB per token", read / 1e6 / n)),
        ]
        if c1.spare_memory != 0 || bigMemory {
            r.append(("From spare memory", String(format: "%.1f MB per token", spare / 1e6 / n)))
            r.append(("Model files held by the system", storeResident.map { String(format: "%.2f GB at the end of the reply", Double($0) / 1e9) }
                      ?? "not tracked (spare memory off)"))
        }
        r.append(("Read but never used", String(format: "%.1f MB per token (named ahead, dropped before use)",
                                                Double(c1.bytes_named_unused &- c0.bytes_named_unused) / 1e6 / n)))
        r += [
            ("Waiting on storage", String(format: "%.0f ms per token (%.0f%% of each token)", wait, ms > 0 ? 100 * wait / ms : 0)),
            ("Experts found in memory", picks > 0 ? String(format: "%.1f%% of picks", 100 * Double(c1.hits &- c0.hits) / picks) : "-"),
            ("Storage read speed", busy > 0 ? String(format: "%.2f GB/s while reading", read / 1e9 / busy) : "-"),
            ("Expert cache in use", String(format: "%.2f GB of %.2f GB", Double(c1.held_bytes) / 1e9, Double(c1.budget_bytes) / 1e9)),
            ("Peak memory use", String(format: "%.2f GB", Double(vm?.ledger_phys_footprint_peak ?? 0) / 1e9)),
            ("Device temperature at end", thermalName(ProcessInfo.processInfo.thermalState)),
        ]
        return r
    }

    var settings: [(String, String)] {
        let m = lock.withLock { model }
        var s: [(String, String)] = [("engine", "dmeng (unified C++ engine)"),
                                     ("model", (lock.withLock { loadedKind } ?? DmengLaunch.kind).displayName
                                        + (lock.withLock { codec }.map { $0.family.isEmpty ? "" : " (\($0.family))" } ?? "")),
                                     ("profile", String(format: "%@ (%.2f GB available at open)", DmengLaunch.profile, Double(DmengLaunch.availableAtOpen) / 1e9)),
                                     ("mode", DmengLaunch.userMode + " (passed to the engine as \"\(DmengLaunch.engineMode ?? "exact")\")"),
                                     ("in-memory weights", "\(DmengLaunch.residentBits(dir: DmengLaunch.modelDir.path))-bit"),
                                     ("model dir", DmengLaunch.modelDir.path),
                                     ("manifest", lock.withLock { manifestPick }?.label ?? "not opened yet")]
        if let m {
            // every engine row from what the engine RESOLVED at open (dm_stats_json "resolved"), never from the app's own
            // guess of the profile, so this screen cannot disagree with the engine
            let i = m.info
            let r = lock.withLock { resolvedAtOpen }
            func d(_ k: String) -> [String: Any] { r[k] as? [String: Any] ?? [:] }
            func num(_ x: Any?) -> Double { (x as? NSNumber)?.doubleValue ?? 0 }
            func file(_ x: Any?) -> String { ((x as? String) ?? "").isEmpty ? "none" : (((x as? String) ?? "") as NSString).lastPathComponent }
            let mem = ["plain", "mlock", "purgeable", "file"][Int(max(0, min(3, i.memory_mode)))]
            let io = DmengLaunch.isPhone ? "below the app's" : "the app's"
            let nm = d("names"), ra = d("run_ahead"), gpu = d("prefill_gpu"), bg = d("background")
            let names = (nm["rule"] as? String) == "confidence"
                ? String(format: "confidence rule: p >= %.2f over the top %d", num(nm["tau"]), Int(num(nm["candidates"])))
                : "top \(Int(num(nm["m"])))"
            let picks = Int(num(ra["picks"]))
            let pre = lock.withLock { presetAtOpen }
            s += [("engine mode", (r["mode"] as? String) ?? "?"),
                  ("profile's engine_mode", ((pre["engine_mode"] as? String) ?? "?") + ", from "
                      + ((pre["engine_mode_from"] as? String) ?? "the manifest's modes (no per-profile override reported)")),
                  ("threads", "\(i.threads) compute, \(Int(num(r["readers"]))) readers (disk I/O: \(io))"),
                  ("run-ahead", picks > 0 ? String(format: "%d names, depth %d, budget %.2f x compute", picks, Int(num(ra["depth"])), num(ra["budget"])) : "off"),
                  ("expert cache", String(format: "%@, %.2f GB (resolved %.2f GB), eviction %@%@", mem, Double(i.cache_bytes) / 1e9, num(r["cache_gb"]),
                                          (r["eviction"] as? String) ?? "?", ((r["layer_caps"] as? String) ?? "").isEmpty ? "" : ", caps \((r["layer_caps"] as? String)!)")),
                  ("routers", "\(file(r["routers"])), \(names)"),
                  ("cross-token routers", file(r["cross_token"]) == "none" ? "none" : String(format: "%@ (tau %.2f), priority reads %@", file(r["cross_token"]),
                                                                                             num(r["cross_token_tau"]), num(r["priority_reads"]) > 0 ? "on" : "off")),
                  ("prompt GPU share", num(gpu["projections"]) > 0 || num(gpu["experts"]) > 0
                      ? String(format: "projections %.2f, experts %.2f", num(gpu["projections"]), num(gpu["experts"])) : "off"),
                  ("background reads", Int(num(bg["ranks"])) > 0 ? "ranks to \(Int(num(bg["ranks"])))" : "off"),
                  ("spare memory", (r["spare_memory"] as? Bool) == true
                      ? (m.counters.spare_memory == 2 ? "paused after a memory warning" : "on (model files kept in the system's spare memory)")
                      : "off"),
                  ("context", "\(i.max_ctx) positions")]
        }
        s += [("sampling", GenSettings.current(defaults: lock.withLock { codec }?.generationDefaults ?? .modelDefault).label),
              ("history", HistoryMode.current.label), ("prefix cache", prefixCacheStatus)]
        return s
    }

    var prefixCacheStatus: String {
        let d = DmengLaunch.diskDir
        let files = (try? FileManager.default.contentsOfDirectory(atPath: d.path)) ?? []
        return "engine prefix cache (in memory + system prompt on disk: \(files.count) file(s) in \(d.lastPathComponent)/)"
    }

    func clearPrefixCache() async {
        try? FileManager.default.removeItem(at: DmengLaunch.diskDir)
    }

    func waitIdle() async {
        guard let m = lock.withLock({ model }) else { return }
        _ = try? await m.run { 0 }
    }

    /// The engine keeps one live state and brings any prompt to it with the least recomputation: nothing to forget.
    func forget() {}

    /// Close the model (dm_close when the last reference goes) and drop its text side, so another one can be opened in
    /// the memory it held: the engine's queue is drained first (a reply, a warm or a type-ahead running on it ends).
    func unload() async {
        cancel()
        await waitIdle()
        #if canImport(UIKit)
        for o in observers { NotificationCenter.default.removeObserver(o) }
        #endif
        let m = lock.withLock { () -> DmengModel? in
            let m = model
            model = nil; codec = nil; _last = nil; _replyRows = []; headCache = [:]; taHeadKey = ""; taHead = []
            observers = []; budgetNow = 0; openBytes = 0; loadedKind = nil
            info.detail = "not loaded"
            return m
        }
        // queued type-aheads see `cancelled` and return; then nothing on the queue holds the model, and dm_close runs
        // when `m` goes out of scope here
        let had = m != nil
        if let m { m.queue.sync {} }
        lock.withLock { cancelled = false }
        log("DMENG closed \(had ? "the model" : "nothing (no model was open)")")
    }

    func load(progress: @escaping (String) -> Void) async throws {
        let dir = DmengLaunch.modelDir.path
        let kind = DmengLaunch.kind
        AppClock.start()
        progress("tokenizer + chat template from \(DmengLaunch.modelDir.lastPathComponent)/ (\(kind.displayName))")
        let c = try await TextCodec.load(dir: dir)
        log("DMENG model \(kind.rawValue) (\(kind.displayName)): family \(c.family.isEmpty ? "unknown" : c.family), front end \(c.isInkling ? "InklingChat (stop ids \(c.stopIds.sorted()))" : "the chat template (Jinja)")")
        let exact = DmengLaunch.engineMode == nil
        let q = kind == .qwen36
        var o = DmengOptions()
        o.cacheBytes = Int64(DmengLaunch.cacheGB * 1e9)
        // the 35B names its experts with the manifest's linear_k2 router; Inkling takes its profile's routers (naming one
        // here would turn the profile's cross-token router and its priority reads off)
        o.routersId = q ? DmengLaunch.defaultRouters : nil
        o.routerM = 0
        o.threads = 0
        o.readers = 0  // the profile's: phone and phone-small declare 8 (4 readers 0.745 GB/s, 8 readers 0.84 on the phone's drive)
        // the derived files: the Hub's layout (derived/), else .dmeng/
        // the manifest the engine opens with (ManifestChoice): the release line's bundled one, or the folder's own
        let pick = ManifestChoice.choose(kind: kind, dir: dir, profile: DmengLaunch.profile)
        lock.withLock { manifestPick = pick }
        if FileManager.default.fileExists(atPath: dir + "/derived") { o.derivedDir = dir + "/derived" }
        else if FileManager.default.fileExists(atPath: dir + "/.dmeng") { o.derivedDir = dir + "/.dmeng" }
        // saved states belong to the weights and the mode they were computed with (a Fast decode's positions differ from
        // Exact's; the 4-bit in-memory copy differs from the 8-bit one): one folder per (mode, in-memory weights file by
        // size / mtime / inode), the two most recent kept
        let fm = FileManager.default
        let bitsNow = DmengLaunch.residentBits(dir: dir)
        let resName = bitsNow == 4 ? "resident_q4." : "resident_q8."
        var resKey = "none"
        for sub in ["derived", ".dmeng"] {
            let sd = dir + "/" + sub
            if let f = ((try? fm.contentsOfDirectory(atPath: sd)) ?? []).first(where: { $0.hasPrefix(resName) && $0.hasSuffix(".bin") }),
               let a = try? fm.attributesOfItem(atPath: sd + "/" + f) {
                resKey = "\(sub)/\(f)|\((a[.size] as? NSNumber)?.int64Value ?? 0)|\(Int(((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)))|\((a[.systemFileNumber] as? NSNumber)?.int64Value ?? 0)"
                break
            }
        }
        let key = (DmengLaunch.engineMode ?? "exact") + "|" + resKey + (pick.bundled ? "|line-manifest" : "")  // Fast's rule is the manifest's
        var hsh: UInt64 = 1469598103934665603
        for b in key.utf8 { hsh = (hsh ^ UInt64(b)) &* 1099511628211 }
        let sub = DmengLaunch.diskDir.appendingPathComponent(String(format: "s%016llx", hsh), isDirectory: true)
        try? fm.createDirectory(at: sub, withIntermediateDirectories: true)
        try? key.write(to: sub.appendingPathComponent("key.txt"), atomically: true, encoding: .utf8)
        let others = ((try? fm.contentsOfDirectory(at: DmengLaunch.diskDir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
            .filter { $0.lastPathComponent != sub.lastPathComponent }
            .sorted { ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast)
                    > ((try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
        for (i, u) in others.enumerated() where i >= 1 || !u.lastPathComponent.hasPrefix("s") { try? fm.removeItem(at: u) }
        log("DMENG saved states in \(sub.lastPathComponent)/ (\(key))")
        o.diskDir = sub.path
        #if !targetEnvironment(macCatalyst)
        o.diskGB = 0.6  // the pinned system-prompt states + the last few chats (~65 MB each)
        // checkpoints in memory: the C API's rule for every model (the pinned system boundary and the newest, whatever
        // their size; the rest are restored from their disk copies); no parked sequences
        o.checkpointGB = 0
        o.parkGB = 0.001
        #endif
        o.ctx = 0  // the profile's positions (Inkling's phone-small: 4096), else the engine's 16384
        // Inkling: the profile named explicitly (its phone-small set's NVFP4 layers 20-23 come from the profile's experts
        // override); the 35B: the C API's own choice from the memory available at open
        if kind == .inkling { o.profile = DmengLaunch.profile }
        o.fileMinBytes = 0
        o.fileCoreBytes = 0
        o.fileEvictHint = 0
        o.layerCaps = 0  // the core's auto
        o.plan = 0
        o.planU = 0
        o.routerTau = 0
        o.routerCand = 0
        #if os(iOS) && !targetEnvironment(macCatalyst)
        // the 35B: Fast names linear_k2's plain top 3 per step, Exact the confidence rule (tau 0.5 over 16); Inkling: the
        // profile's names rule (the confidence rule, tau 0.5 over 16, in both modes)
        if q {
            if exact {
                o.routerTau = 0.5
                o.routerCand = 16
            } else {
                o.routerM = 3
                o.routerTau = -1
            }
        }
        #endif
        o.split = 1
        // passed as "exact" explicitly: the engine's profile would otherwise choose its own default mode (phone-small: fast)
        o.mode = DmengLaunch.engineMode ?? "exact"
        o.residentBits = bitsNow
        o.prefillThreads = 0
        // the batched prefill split with the GPU (exact: the GPU's rows are the CPU kernels' bits): on iOS the profile's
        // (phone and phone-small declare the 35B 0.65 / 0.65, Inkling 0 / 0, with their iPhone measurements); off on the Mac
        #if os(iOS) && !targetEnvironment(macCatalyst)
        let gpu = (projections: 0.0, experts: 0.0)
        #else
        let gpu = (projections: -1.0, experts: -1.0)
        #endif
        o.prefillGpuFrac = gpu.projections
        o.prefillGpuExperts = gpu.experts
        o.prefillGpuMin = 0
        o.prefillGpuIntKernel = false
        #if os(iOS) && !targetEnvironment(macCatalyst)
        // a disk restore also reads the expert set saved with it (the system prompt's, at launch in idle time): a new
        // chat's first message then reads only its own tokens' misses
        o.preloadExperts = true
        #else
        o.preloadExperts = false
        #endif
        o.readerQoS = nil
        o.perfCoresOnly = false
        // the readers' disk I/O one tier below the app's own on the phone, so the UI's page faults do not wait behind
        // queued expert reads
        o.readerIOLow = DmengLaunch.isPhone
        #if SELFCHECK
        if let x = SelfCheck.optsOverride { o.modelOptions = x }  // a lever's model options / run flags (this process only)
        #endif
        if DmengLaunch.spareMemoryOffered, let on = DmengLaunch.spareMemoryChoice {  // the switch, as an explicit override
            o.modelOptions = [o.modelOptions, "run:spare-mem=\(on ? 1 : 0)"].compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: ";")
        }
        // run-ahead (while a step waits on a read, the next layer's router runs on the experts in memory and its reads are
        // issued; the real pass recomputes exactly): the 35B in Exact 8 picks, depth 1, speculative work <= 0.2 x compute,
        // none while every reader is busy; off in Fast; Inkling: the profile's (0)
        o.runAhead = q ? (exact ? 8 : -1) : 0
        o.runAheadDepth = 1
        o.runAheadBudget = q ? (exact ? 0.2 : -1) : 0
        o.runAheadMax = -1
        o.runAheadSatSkip = q && exact
        o.runAheadAB = false
        progress("opening \(dir) (in-memory weights, routers \(o.routersFile.map { ($0 as NSString).lastPathComponent } ?? o.routersId ?? "the profile's"), cache \(DmengLaunch.cacheGB) GB)")
        let t0 = Date()
        let openDir = pick.dir
        let m = try await Task.detached(priority: .userInitiated) { try DmengModel(modelDir: openDir, options: o) }.value
        let i = m.info
        lock.withLock {
            model = m
            codec = c
            loadedKind = kind
            openBytes = i.cache_bytes
            info.detail = String(format: "%d threads, %@ cache %.2f GB, routers %@; opened in %.1f s",
                                 i.threads, ["plain", "mlock", "purgeable", "file"][Int(max(0, min(3, i.memory_mode)))],
                                 Double(i.cache_bytes) / 1e9, m.routersFile.isEmpty ? "none" : (m.routersFile as NSString).lastPathComponent,
                                 Date().timeIntervalSince(t0))
        }
        // the settings the engine applied after precedence (explicit option > profile > manifest > built-in)
        let st = m.statsJSON()
        if let r = st.range(of: "\"resolved\":") { log("DMENG RESOLVED " + String(st[r.upperBound...].dropLast())) }
        let sj = (try? JSONSerialization.jsonObject(with: Data(st.utf8))) as? [String: Any]
        let res = sj?["resolved"] as? [String: Any] ?? [:]
        let pre = (sj?["policy"] as? [String: Any])?["preset"] as? [String: Any] ?? [:]  // the profile as the engine read it
        lock.withLock { resolvedAtOpen = res; presetAtOpen = pre }
        log("DMENG MODE \(DmengLaunch.engineMode ?? "exact") -> \((res["mode"] as? String) ?? "?") (profile default engine_mode \((pre["engine_mode"] as? String) ?? "?"), from \((pre["engine_mode_from"] as? String) ?? "the manifest's modes"))")
        progress("ready: " + info.detail)
        observeMemory()
    }

    /// Memory warnings and the app's background / foreground state, mapped onto the engine's run-time cache budget.
    private func observeMemory() {
        #if canImport(UIKit)
        let nc = NotificationCenter.default
        let floorBytes: Int64 = 900_000_000  // the engine clamps to its own floor (a prefill layer's union + names)
        observers.append(nc.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil) { [weak self] _ in
            guard let self else { return }
            // spare memory on: stop adding to the system's file cache for the rest of the session (reads go to storage
            // as without it); the pages already there are the system's to drop
            if let m = self.lock.withLock({ self.model }), m.counters.spare_memory == 1, m.pauseSpareMemory() {
                log("DMENG memory warning: spare memory paused for the rest of the session")
            }
            // iOS sends warnings during a prefill's 3.3 GB peak with 0.9-1.8 GB still available, and halving the
            // slab then halved the chat's decode speed. Shrink only when the available memory is below the floor, and
            // only by the shortfall (plus one record of slack per layer).
            let avail = Int64(os_proc_available_memory())
            let floorAvail: Int64 = 1_000_000_000  // at 0.6 GB the chat path compressed 0.6 GB of state (2.4 tok/s)
            guard avail < floorAvail else {
                log(String(format: "DMENG memory warning: available %.2f GB >= %.2f GB floor: cache budget kept", Double(avail) / 1e9, Double(floorAvail) / 1e9))
                return
            }
            let short = floorAvail - avail + 70_000_000
            let b = self.setBudget { max(floorBytes / 2, $0 - short) }
            log(String(format: "DMENG memory warning: available %.2f GB: cache budget -> %.2f GB (shortfall %.2f GB)", Double(avail) / 1e9, Double(b) / 1e9, Double(short) / 1e9))
        })
        observers.append(nc.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: nil) { [weak self] _ in
            guard let self else { return }
            let b = self.setBudget { _ in floorBytes }
            log(String(format: "DMENG background: cache budget -> %.2f GB", Double(b) / 1e9))
        })
        observers.append(nc.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: nil) { [weak self] _ in
            guard let self else { return }
            let b = self.setBudget { _ in self.lock.withLock { self.openBytes } }
            log(String(format: "DMENG foreground: cache budget -> %.2f GB", Double(b) / 1e9))
        })
        #endif
    }

    private var budgetNow: Int64 = 0  // the budget last asked for (0: the one given at open)
    /// Ask the engine for a new cache budget computed from the current one (any thread); returns the bytes asked for.
    @discardableResult
    private func setBudget(_ f: (Int64) -> Int64) -> Int64 {
        guard let m = lock.withLock({ model }) else { return 0 }
        let cur = lock.withLock { budgetNow > 0 ? budgetNow : openBytes }
        let b = f(cur)
        lock.withLock { budgetNow = b }
        m.setCacheBytes(b)
        return b
    }

    func cancel() {
        lock.withLock {
            cancelled = true
            model?.cancel()
        }
    }

    /// Bring the system prompt's state into the engine (idle time: at launch and when a new chat opens).
    func warmSystem(_ system: String) async {
        await warm([ChatMessage(role: .system, content: system)], why: "system prompt")
    }

    /// The ids of the next turn's head: `msgs` (rendered for the engine) and the next user message's header, i.e. where
    /// two renders of msgs + one more user message with different contents stop agreeing (the Qwen template refuses a
    /// system message alone, so the head is found this way).
    static func nextHead(_ c: TextCodec, _ msgs: [[String: String]], thinking: Bool) -> [Int]? {
        guard let a = try? c.promptIds(messages: msgs + [["role": "user", "content": "a"]], thinking: thinking),
              let b = try? c.promptIds(messages: msgs + [["role": "user", "content": "b"]], thinking: thinking) else { return nil }
        var n = 0
        while n < min(a.count, b.count) && a[n] == b[n] { n += 1 }
        return n > 0 ? Array(a.prefix(n)) : nil
    }
    private var headCache: [String: [Int]] = [:]  // system content -> the system boundary's ids (system turn + user header)
    func systemBoundary(_ c: TextCodec, system: [String: String], thinking: Bool) -> [Int]? {
        let key = (system["content"] ?? "") + (thinking ? "\u{1}" : "")
        if let v = lock.withLock({ headCache[key] }) { return v }
        guard let h = Self.nextHead(c, [system], thinking: thinking) else { return nil }
        lock.withLock { headCache[key] = h }
        return h
    }

    /// Idle-time work (launch, a new chat, a chat opened, the end of a reply): the live state brought to `messages`
    /// plus the next user message's header, state only (dm_sync without logits), and kept on disk there. The system
    /// boundary is the first mark (its disk file pinned); a chat's file is the one a relaunch restores. A message sent
    /// next then computes only its own text and the generation prompt.
    func warm(_ messages: [ChatMessage], why: String) async {
        guard let m = lock.withLock({ model }), let c = lock.withLock({ codec }) else { return }
        let think = thinking
        let msgs = templateMessages(messages, mode: HistoryMode.current)
        guard let head = Self.nextHead(c, msgs, thinking: think), head.count > 1 else { return }
        var marks: [Int32] = []
        if msgs.first?["role"] == "system", let sb = systemBoundary(c, system: msgs[0], thinking: think)?.count, sb <= head.count {
            marks.append(Int32(sb))
        }
        if !marks.contains(Int32(head.count)) { marks.append(Int32(head.count)) }
        let mk = marks
        let t0 = Date()
        _ = try? await m.run {
            let sr = try m.prefill(head.map { Int32($0) }, marks: mk)
            let tp = Date()
            if let sb = mk.first, Int(sb) != head.count { m.persist(Int(sb)) }
            m.persist(head.count)
            log(String(format: "DMENG WARM %@: %d tokens (next user header included): %@, reused %d, computed %d, restore %.0f ms, prefill %.0f ms, persist %.0f ms, %.0f ms in all; %@",
                       why, head.count, sr.source, sr.reused, sr.computed, sr.msRestore, sr.msPrefill, 1000 * Date().timeIntervalSince(tp),
                       1000 * Date().timeIntervalSince(t0), m.lastPrefillJSON))
            return 0
        }
    }

    // MARK: type-ahead: the draft's stable prefix synced while the user types; exact (the prefix cache
    // restores only states it computed for the same ids; the reply's own sync starts from the longest common prefix)
    private var taGen = 0              // the latest type-ahead request (a queued older one is skipped)
    private var taInflight: [Int]? = nil  // the ids of the type-ahead sync running now
    private var taInflightFull = false
    private var taHeadKey = "", taHead: [Int] = []
    private var taInflightText: String? = nil  // the draft text the running type-ahead read (stable part, or the whole draft)
    private(set) var taRuns = 0, taPositions = 0

    /// Bring the live state, in the background, to the longest prefix two renders of the chat with the draft's stable
    /// text agree on (`a`, `b`: the same messages, the last user turn's content followed by " a" / " b"), less one
    /// token (a partial text's last token can merge with what follows). State only, never generates; a mark there.
    /// A running type-ahead whose ids are not a prefix of the new ones is cancelled.
    ///
    /// `full` (after a pause in typing): instead the whole prompt as it would be sent now (the draft as the user message
    /// and the generation prompt), with its logits, so a send of exactly this draft computes nothing. Speculative: any
    /// keystroke cancels it (draftEdited).
    func typeAhead(_ a: [ChatMessage], _ b: [ChatMessage], full: [ChatMessage]? = nil, text: String = "") {
        let tReq = ProcessInfo.processInfo.systemUptime
        guard let m = lock.withLock({ model }), let c = lock.withLock({ codec }) else { return }
        let think = thinking
        // the head (history + the user header) is rendered once per draft; the draft's stable text is appended by the
        // tokenizer alone (no template render per keystroke; a boundary token that differs only costs the reuse: the
        // send's sync starts from the longest common prefix)
        let hist = Array(a.dropLast())
        let hkey = "\(hist.count)|\(hist.last?.content.hashValue ?? 0)|\(hist.first?.content.hashValue ?? 0)|\(think)"
        var head: [Int]? = lock.withLock { taHeadKey == hkey ? taHead : nil }
        if head == nil {
            head = Self.nextHead(c, templateMessages(hist, mode: HistoryMode.current), thinking: think)
            if let h = head { lock.withLock { taHeadKey = hkey; taHead = h } }
        }
        guard let head, !text.isEmpty else { return }
        let body = c.encode(text)
        let n = head.count + body.count - 1
        guard n > head.count else { return }
        var ids = Array((head + body).prefix(n))
        var marks: [Int32] = [Int32(n)]
        if let full, let F = try? c.promptIds(messages: templateMessages(full, mode: HistoryMode.current), thinking: think),
           F.count > n, Array(F.prefix(n)) == ids {
            ids = F
            marks.append(Int32(F.count))
        } else if full != nil { return }
        let isFull = full != nil
        let (gen, stop) = lock.withLock { () -> (Int, Bool) in
            taGen += 1
            let stop = taInflight.map { !ids.starts(with: $0) } ?? false
            return (taGen, stop)
        }
        let tRendered = ProcessInfo.processInfo.systemUptime
        if stop { m.cancel(); m.queue.async { m.clearCancel() } }
        let mk = marks, nn = n
        m.queue.async { [weak self] in
            guard let self, self.lock.withLock({ self.taGen == gen && !self.cancelled }) else { return }
            self.lock.withLock { self.taInflight = ids; self.taInflightFull = isFull; self.taInflightText = text }
            let t0 = Date()
            let tStart = ProcessInfo.processInfo.systemUptime
            var lg = [Float](repeating: 0, count: isFull ? m.vocab : 0)
            let sr = isFull ? try? m.sync(ids.map { Int32($0) }, marks: mk, logits: &lg) : try? m.prefill(ids.map { Int32($0) }, marks: mk)
            self.lock.withLock { self.taInflight = nil; self.taInflightFull = false; self.taInflightText = nil; self.taRuns += 1; self.taPositions += sr?.computed ?? 0 }
            log(String(format: "DMENG TYPEAHEAD %@ %d positions (stable prefix %d): %@, reused %d, computed %d%@, %.0f ms (prefill %.0f, wait %.0f, %.0f MB); t %.2f: asked, +%.2f rendered, +%.2f started",
                       isFull ? "full prompt" : "prefix", ids.count, nn, sr?.source ?? "error", sr?.reused ?? 0, sr?.computed ?? 0,
                       sr?.cancelled == true ? " (cancelled)" : "", 1000 * Date().timeIntervalSince(t0), sr?.msPrefill ?? 0, sr?.msWaitReads ?? 0, sr?.mbRead ?? 0,
                       tReq, tRendered - tReq, tStart - tReq))
        }
    }

    /// A keystroke: a running speculative full-prompt type-ahead is cancelled (the draft it read is not the one sent).
    func draftEdited() {
        guard let m = lock.withLock({ model }) else { return }
        let stop = lock.withLock { () -> Bool in
            guard taInflightFull else { return false }
            taGen += 1
            return true
        }
        if stop { m.cancel(); m.queue.async { m.clearCancel() } }
    }

    /// Before a reply: queued type-aheads are dropped; a running one is cancelled unless its ids are a prefix of the
    /// reply's prompt (then the reply's sync continues from it).
    private func typeAheadSend(_ finalText: String?, _ m: DmengModel) {
        let stop = lock.withLock { () -> Bool in
            taGen += 1
            guard taInflight != nil else { return false }
            guard let f = finalText, let t = taInflightText else { return true }
            return taInflightFull ? f != t : !f.hasPrefix(t)  // the running sync's text is the sent one, or a prefix of it
        }
        if stop { m.cancel(); m.queue.async { m.clearCancel() } }
        log(String(format: "DMENG SEND t %.2f: type-ahead %@", ProcessInfo.processInfo.systemUptime, stop ? "cancelled" : "kept (or none running)"))
    }

    /// Prompt tokens of `messages` by this engine's tokenizer and template (the tool loop's counts).
    func tokenCount(_ messages: [[String: String]], thinking: Bool) -> Int? {
        guard let c = lock.withLock({ codec }) else { return nil }
        return try? c.promptIds(messages: messages, thinking: thinking).count
    }

    static func sampler(_ p: SamplingParams) -> DmengSampler {
        var s = DmengSampler()
        s.temp = Double(p.temperature)
        s.topK = p.topK
        s.topP = Double(p.topP)
        s.minP = Double(p.minP)
        s.presencePenalty = Double(p.presencePenalty)
        s.repetitionPenalty = Double(p.repetitionPenalty)
        s.frequencyPenalty = Double(p.frequencyPenalty)
        s.seed = p.seed
        return s
    }

    func generate(messages: [ChatMessage], maxNew: Int) -> AsyncThrowingStream<GenEvent, Error> {
        let (m, c, think) = lock.withLock { () -> (DmengModel?, TextCodec?, Bool) in
            cancelled = false
            return (model, codec, _thinking)
        }
        let msgs = templateMessages(messages, mode: HistoryMode.current)
        if let m { typeAheadSend(messages.last(where: { $0.role == .user })?.content, m) }
        return AsyncThrowingStream { cont in
            guard let m, let c else { cont.finish(throwing: EngineError.notLoaded); return }
            let gen = GenSettings.current(defaults: c.generationDefaults)
            m.queue.async { [weak self] in
                guard let self else { cont.finish(); return }
                // The tool-call terminator cancels the previous decode. Reset its native flag on the engine
                // queue before this round's sync, while preserving a Stop that arrived after generate().
                self.lock.withLock {
                    if self.cancelled { m.cancel() } else { m.clearCancel() }
                }
                do {
                    try self.reply(m, c, msgs: msgs, thinking: think, maxNew: maxNew, gen: gen, cont: cont)
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
            cont.onTermination = { reason in
                if case .cancelled = reason { self.cancel() }
            }
        }
    }

    private var isCancelled: Bool { lock.withLock { cancelled } }

    #if SELFCHECK
    struct Greedy: Sendable { var ids: [Int]; var source: String; var seconds: Double; var prefillSeconds: Double = 0; var decodeStats: String = ""; var prefillJSON: String = "" }
    /// SelfCheck: greedy decoding of raw prompt ids on the open model (sync, then argmax / step, as a reply does). The
    /// engine's counters are restarted after the sync, so `decodeStats` (dm_stats_json) covers the decode steps only.
    func selfCheckGreedy(_ prompt: [Int], n: Int) async throws -> Greedy {
        guard let m = lock.withLock({ model }) else { throw EngineError.notLoaded }
        return try await m.run {
            var logits = [Float](repeating: 0, count: m.vocab)
            let tp = Date()
            let sr = try m.sync(prompt.map { Int32($0) }, logits: &logits)
            let prefillSeconds = Date().timeIntervalSince(tp)
            let pj = m.lastPrefillJSON
            _ = m.statsJSON()
            var out: [Int] = []
            let t0 = Date()
            while out.count < n {
                let tk = DmengModel.argmax(logits, vocab: m.vocab)
                out.append(Int(tk))
                if out.count == n { break }
                guard try m.step(tk, logits: &logits) else { break }
            }
            let secs = Date().timeIntervalSince(t0)
            return Greedy(ids: out, source: sr.source, seconds: secs, prefillSeconds: prefillSeconds, decodeStats: m.statsJSON(), prefillJSON: pj)
        }
    }
    var selfCheckStats: String { lock.withLock { model }?.statsJSON() ?? "" }
    struct Forced: Sendable { var agree: Int; var n: Int; var firstDiff: Int; var prefillSeconds: Double; var prefillJSON: String; var top: [Int] }
    /// SelfCheck: the prompt synced, then `cont` teacher-forced: at each position the argmax against the next forced id.
    func selfCheckForced(_ prompt: [Int], cont: [Int]) async throws -> Forced {
        guard let m = lock.withLock({ model }) else { throw EngineError.notLoaded }
        return try await m.run {
            var logits = [Float](repeating: 0, count: m.vocab)
            let tp = Date()
            _ = try m.sync(prompt.map { Int32($0) }, logits: &logits)
            let ps = Date().timeIntervalSince(tp)
            let pj = m.lastPrefillJSON
            var agree = 0, first = -1
            var top: [Int] = []
            for (i, t) in cont.enumerated() {
                let a = Int(DmengModel.argmax(logits, vocab: m.vocab))
                top.append(a)
                if a == t { agree += 1 } else if first < 0 { first = i }
                if i + 1 == cont.count { break }
                guard try m.step(Int32(t), logits: &logits) else { break }
            }
            return Forced(agree: agree, n: cont.count, firstDiff: first, prefillSeconds: ps, prefillJSON: pj, top: top)
        }
    }
    #endif

    /// One reply on the engine's queue: render, sync (prefill with progress), decode.
    private func reply(_ m: DmengModel, _ c: TextCodec, msgs: [[String: String]], thinking: Bool, maxNew: Int, gen: GenSettings,
                       cont: AsyncThrowingStream<GenEvent, Error>.Continuation) throws {
        let tStart = Date()
        let ids = try c.promptIds(messages: msgs, thinking: thinking)
        // checkpoint marks: the system boundary (system turn + the user header: the disk prefix every new chat starts from)
        var marks: [Int32] = []
        var sysMark: Int? = nil
        if msgs.first?["role"] == "system", let h = systemBoundary(c, system: msgs[0], thinking: thinking), h.count < ids.count,
           Array(ids.prefix(h.count)) == h {
            marks.append(Int32(h.count))
            sysMark = h.count
        }
        // the end of the conversation's messages (before the generation prompt): stable when the next turn re-renders it
        if let h = try? c.prefixIds(messages: msgs, thinking: thinking), h.count < ids.count, Array(ids.prefix(h.count)) == h,
           !marks.contains(Int32(h.count)) {
            marks.append(Int32(h.count))
        }
        var logits = [Float](repeating: 0, count: m.vocab)
        let tPre = Date()
        let sr = try m.sync(ids.map { Int32($0) }, marks: marks, logits: &logits) { done, total, mb in
            cont.yield(.prefill(PrefillTick(kind: "prefill", promptTokens: ids.count, reusedTokens: ids.count - total, total: total,
                                            done: done, mbRead: mb, relandMB: 0, elapsed: Date().timeIntervalSince(tStart),
                                            prefillElapsed: Date().timeIntervalSince(tPre))))
        }
        log("DMENG PREFILL " + m.lastPrefillJSON)  // the prefill's split (wall, read wait, union, names used / unused)
        let kind = sr.source == "cold" ? "cold" : (sr.source == "live" ? "live prefix" : "restored prefix")
        cont.yield(.prefill(PrefillTick(kind: kind, promptTokens: ids.count, reusedTokens: sr.reused, total: sr.computed,
                                        done: sr.computed, mbRead: sr.mbRead, relandMB: 0, elapsed: Date().timeIntervalSince(tStart),
                                        prefillElapsed: sr.msPrefill / 1000)))
        let sampler = Self.sampler(gen.sampling)
        var tracker = ThinkingTracker(promptIds: ids, tokens: c.thinkTokens, budget: thinking ? gen.thinkingBudget : nil)
        var new: [Int] = [], hist: [Int32] = []
        var reason: StopReason = sr.cancelled ? .cancelled : .maxNew
        var prev = 0, done = 0
        var held = ""
        var first = true, openShown = false
        let ink = c.isInkling
        let lead = c.inklingLead(thinking: thinking)
        var stopId: Int? = nil
        var shownScalars: [Unicode.Scalar] = []
        let rec = 1_769_472.0
        let c0 = m.counters
        var cl = c0
        let t0 = Date()
        let clock = GenClock()
        if !sr.cancelled {
            for _ in 0..<maxNew {
                if isCancelled { reason = .cancelled; break }
                let forced = tracker.forced(lastText: { c.decode(new) })
                let tk = forced ?? Int(sampler.isGreedy ? DmengModel.argmax(logits, vocab: m.vocab)
                                                       : DmengModel.sample(logits, vocab: m.vocab, sampler, history: hist))
                if c.stopIds.contains(tk) { reason = .eos; stopId = tk; break }
                new.append(tk)
                hist.append(Int32(tk))
                let phase = tracker.observe(tk)
                // the token is shown as soon as it is drawn; its step (the logits for the next one) runs after
                // incremental detokenisation: only the tokens since the last emitted piece
                var piece = ""
                if ink {
                    // Inkling: the turn so far, decoded with its special tokens and parsed (InklingChat.parseTurn), in the
                    // app's conventions (TextCodec.inklingAppText); the part past what is shown is the piece
                    let raw = lead + c.decode(new)
                    if !raw.hasSuffix("\u{FFFD}") {
                        let ds = Array(TextCodec.inklingAppText(raw).unicodeScalars)
                        if ds.count > shownScalars.count && Array(ds[..<shownScalars.count]) == shownScalars {
                            piece = String(String.UnicodeScalarView(ds[shownScalars.count...]))
                            shownScalars = ds
                        }
                    }
                } else {
                    let text = c.decode(Array(new[prev...]))
                    let ts = Array(text.unicodeScalars), hs = held.unicodeScalars.count
                    if ts.last != "\u{FFFD}" && ts.count > hs {
                        piece = String(String.UnicodeScalarView(ts[hs...]))
                        prev = done
                        done = new.count
                        held = piece
                    }
                    if first && (phase == .thinking || phase == .close) { piece = "<think>\n" + piece; openShown = true }
                    if phase == .open { if openShown { piece = "" } else { openShown = true } }
                }
                first = false
                let cn = m.counters
                let now = Date()
                let picks = Double(cn.picks &- cl.picks)
                cont.yield(.text(piece, TokenStats(index: new.count - 1, ms: 1000 * clock.last,
                                                   tokPerSec: clock.last > 0 ? 1 / clock.last : 0,
                                                   mbRead: Double(cn.bytes_read &- cl.bytes_read) / 1e6,
                                                   inMemory: picks > 0 ? Double(cn.hits &- cl.hits) / picks : 0,
                                                   expertsInMemory: Int(Double(cn.held_bytes) / rec),
                                                   mbTotal: Double(cn.bytes_read &- c0.bytes_read) / 1e6, phase: phase.rawValue,
                                                   speed: clock.speed)))
                cl = cn
                if new.count == 1 {
                    log(String(format: "DMENG REPLY first token %.0f ms after the reply started (prompt %d tokens: %@, reused %d, computed %d, prefill %.0f ms, restore %.0f ms; sampler %@)",
                               1000 * now.timeIntervalSince(tStart), ids.count, sr.source, sr.reused, sr.computed, sr.msPrefill,
                               sr.msRestore, sampler.isGreedy ? "greedy" : String(format: "temperature %g top_k %d top_p %g seed %llu", sampler.temp, sampler.topK, sampler.topP, sampler.seed)))
                }
                let a = clock.begin()
                let stepped = try m.step(Int32(tk), logits: &logits)
                clock.end(a, counted: stepped)
                guard stepped else { reason = .cancelled; break }
            }
        }
        let dt = Date().timeIntervalSince(t0)
        // the system boundary's disk copy (the prefill's dense file of the first mark is completed only by a persist at
        // exactly its position). The chat's own disk copy is written by the idle-time warm after the reply (ChatModel:
        // the chat up to the next user header), the state a relaunch restores.
        if let sysEnd = sysMark {
            let tp = Date()
            m.persist(Int(sysEnd))
            log(String(format: "DMENG persisted the system boundary (%d positions) in %.0f ms", Int(sysEnd), 1000 * Date().timeIntervalSince(tp)))
        }
        let n = max(1, new.count)
        let cEnd = m.counters
        let picks = Double(cEnd.picks &- c0.picks)
        var split: [String: Double] = ["wait_ms_per_token": (cEnd.wait_ms - c0.wait_ms) / Double(n),
                                       "mb_issued_per_token": Double(cEnd.bytes_issued &- c0.bytes_issued) / 1e6 / Double(n),
                                       "file_refaults": Double(cEnd.file_refaults &- c0.file_refaults),
                                       "prefill_ms": sr.msPrefill, "restore_ms": sr.msRestore,
                                       "gen_greedy": sampler.isGreedy ? 1 : 0, "gen_temperature": sampler.temp,
                                       "gen_top_k": Double(sampler.topK), "gen_top_p": sampler.topP, "gen_min_p": sampler.minP,
                                       "gen_seed": Double(sampler.seed), "gen_thinking_budget": Double(gen.thinkingBudget ?? -1),
                                       "think_tokens": Double(tracker.thinkingTokens), "think_closed": tracker.closed ? 1 : 0,
                                       "think_forced_close": tracker.forcedClose ? 1 : 0]
        let sp = clock.speed
        split["compute_ms_per_token"] = 1000 * sp.seconds / Double(max(1, sp.steps)) - split["wait_ms_per_token"]!
        // the engine counters' own account of the same steps (dm_counters: decode steps, time inside dm_step), for the
        // RATE line and the self-check (ChatModel adds them over a message's rounds as it adds the shown speed)
        split["engine_steps"] = Double(cEnd.decode_steps - c0.decode_steps)
        split["engine_step_s"] = Double(cEnd.step_ns &- c0.step_ns) / 1e9
        split["gen_steps"] = Double(sp.steps)
        split["gen_s"] = sp.seconds
        split["wall_decode_s"] = dt
        let mbTok = Double(cEnd.bytes_read &- c0.bytes_read) / 1e6 / Double(n)
        let d = ReplyDetail(promptIds: ids, ids: new, prefill: kind, prefillTokens: sr.computed, prefillSeconds: sr.msPrefill / 1000,
                            seconds: sp.seconds, tokPerSec: sp.rate, mibPerToken: mbTok * 1e6 / 1048576,
                            releasedTop8InMemory: picks > 0 ? Double(cEnd.hits &- c0.hits) / picks : 0,
                            heldExpertsMean: Double(cEnd.held_bytes) / rec, syncReads: Int(cEnd.misses &- c0.misses), split: split,
                            prefillStats: ["prefill_s": sr.msPrefill / 1000, "prefill_mb_read": sr.mbRead,
                                           "prefill_reused": Double(sr.reused), "prefill_computed": Double(sr.computed)])
        let rows = Self.replyRows(c0, cEnd, tokens: new.count, speed: sp, storeResident: m.storeResidentBytes,
                                  bigMemory: DmengLaunch.spareMemoryOffered)
        lock.withLock { _last = d; _replyRows = rows }
        log("DMENG REPLY ids " + new.map(String.init).joined(separator: ","))
        if ink { log("DMENG REPLY raw " + (lead + c.decode(new)).debugDescription + " (stop: \(reason.rawValue)\(stopId.map { ", token \($0)" } ?? ""))") }
        log(String(format: "DMENG REPLY done: %d tokens (%@), %.2f tok/s, %.1f MB read/token, hit %.3f, refaults %.0f, budget %.2f GB", new.count,
                   reason.rawValue, sp.rate, mbTok, d.releasedTop8InMemory, split["file_refaults"] ?? 0,
                   Double(cEnd.budget_bytes) / 1e9))
        cont.yield(.finished(GenSummary(reason: reason, tokens: new.count, seconds: sp.seconds,
                                        meanTokPerSec: sp.rate, meanMBPerToken: mbTok, speed: sp)))
    }
}
