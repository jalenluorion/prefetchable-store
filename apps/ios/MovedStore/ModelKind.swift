import Foundation

/// The models the app runs: Qwen3.6-35B-A3B and Inkling 2 Mini. One is loaded at a time;
/// each has its own folder, Hub repo, prefix-cache folder, mode setting and front end (TextCodec picks the front end from
/// the folder's moved-store.json `family`).
///
/// - Selection: the first-run screen, then Settings > Model (only models whose files are present), stored in
///   `engine.model`. At launch a selected model whose files are missing gives way to the other one when that one is
///   complete (resolveAtLaunch).
/// - Chats remember the model they were made with (Conversation.model); a chat made with the other model opens read-only
///   with a one-line notice (ChatView), so a model switch never re-renders one model's history with the other's tokens.
/// - Thinking: Inkling has no thinking switch but a reasoning effort; the chat's thinking toggle maps to effort "none"
///   when off and the template's default 0.9 when on (TextCodec.inklingEffortOn / inklingEffortOff), and the thinking
///   limit setting ends a long reasoning segment as it does Qwen's think block.
enum ModelKind: String, CaseIterable, Identifiable, Codable {
    case qwen36, inkling
    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .qwen36: return "Qwen3.6 35B"
        case .inkling: return "Inkling 2 Mini"
        }
    }
    /// The Model files footer: who made it, the license.
    var credit: String {
        switch self {
        case .qwen36: return "Qwen3.6-35B-A3B by the Qwen team, Apache 2.0."
        case .inkling: return "Inkling 2 Mini by Thinking Machines, Apache 2.0."
        }
    }
    /// Documents/<folder>: the Hub layout (moved-store.json, store/, derived/, the small files).
    var folderName: String { self == .qwen36 ? "dmeng-model" : "dmeng-model-inkling" }
    var folder: URL { docsURL.appendingPathComponent(folderName, isDirectory: true) }
    var hubRepo: String { self == .qwen36 ? "jalenluorion/Qwen3.6-35B-A3B-prefetchable-store" : "jalenluorion/Inkling-2-Mini-prefetchable-store" }
    /// The engine's saved states (Application Support/<bundle>/<name>): one folder per model, so a state computed with
    /// one model's weights is never offered to the other.
    var prefixDirName: String { self == .qwen36 ? "dmeng-prefix" : "dmeng-prefix-inkling" }
    /// Positions the engine reserves. Inkling on phone-small: 4096 (16K would reserve 0.54 GB of attention cache).
    func ctx(profile: String) -> Int { self == .inkling && profile == "phone-small" ? 4096 : 16384 }
    /// The mode setting's key: one per model (each model has its default per profile).
    var modeKey: String { self == .qwen36 ? "engine.mode" : "engine.mode.inkling" }
    /// Fast on both phone profiles (phone, phone-small), for both models; Exact elsewhere (the manifests' profiles agree).
    func defaultMode(profile: String) -> String { profile == "phone" || profile == "phone-small" ? "fast" : "exact" }
    /// The engine's mode string for the app's Fast. 35B: approx:drop at the threshold measured for the calibrated 4-bit
    /// in-memory weights (0.08); Inkling: the engine's own fast preset.
    var fastEngineMode: String { self == .qwen36 ? "approx:drop:w=0.08" : "fast" }
    /// phone-small's expert cache for a 2.96 GB app footprint. 35B: 2.96 - 1.54 GB measured beside the cache. Inkling:
    /// 1.7 GB (its 4-bit in-memory weights and 4-bit late expert layers at the same footprint).
    var phoneSmallSlabGB: Double { self == .qwen36 ? DmengLaunch.footprintTargetGB - DmengLaunch.nonSlabGB : 1.7 }
    /// The download's size before the manifest is in (the Hub manifest's profile bytes).
    func approxBytes(profile: String) -> Int64 {
        switch self {
        case .qwen36: return profile == "phone-small" ? 20_352_447_020 : 21_309_272_620
        case .inkling: return profile == "phone-small" ? 15_526_167_637 : 17_992_385_621
        }
    }
    /// Decode speed measured on the iPhone 15 Plus (phone-small), tokens/s; nil elsewhere (not measured).
    func tokensPerSec(profile: String, fast: Bool) -> Double? {
        guard profile == "phone-small" else { return nil }
        switch self {
        case .qwen36: return fast ? 3.9 : 3.1
        case .inkling: return fast ? 2.4 : 1.5
        }
    }
    /// The engine's GPU share of the batched prefill on iOS. 35B: 0.65 / 0.65 (measured). Inkling: off (its in-memory
    /// projections are 4-bit on phone-small and the GPU split takes only 8-bit ones).
    var iosPrefillGpu: (projections: Double, experts: Double) { self == .qwen36 ? (0.65, 0.65) : (-1, -1) }

    // MARK: selection

    static let selectedKey = "engine.model"
    /// A switch in this process (switchModel).
    private static var switched: ModelKind?
    static var selected: ModelKind {
        get { switched ?? UserDefaults.standard.string(forKey: selectedKey).flatMap(ModelKind.init(rawValue:)) ?? .qwen36 }
        set {
            switched = newValue
            UserDefaults.standard.set(newValue.rawValue, forKey: selectedKey)
        }
    }
    var other: ModelKind { self == .qwen36 ? .inkling : .qwen36 }
    /// What is missing in this model's folder for the device's profile (empty: complete).
    var problems: [String] { DmengLaunch.completeness(folder, kind: self) }
    var isComplete: Bool { problems.isEmpty }
    /// Bytes of this model's files on the phone (its folder, the download's partial files included).
    var presentBytes: Int64 {
        guard let e = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else { return 0 }
        var n: Int64 = 0
        for case let u as URL in e {
            if let v = try? u.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), v.isRegularFile == true { n += Int64(v.fileSize ?? 0) }
        }
        return n
    }
    /// Models whose files are present (the Model choice in Settings).
    static var present: [ModelKind] { allCases.filter { $0.isComplete } }

    /// At launch: the selected model's files are missing but the other model's are complete: select the other one.
    static func resolveAtLaunch() {
        let s = selected
        if !s.isComplete && s.other.isComplete {
            log("MODEL \(s.displayName) is not complete here; \(s.other.displayName) is: selected")
            selected = s.other
        }
    }
}
