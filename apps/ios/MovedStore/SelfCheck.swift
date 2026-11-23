#if SELFCHECK
import Foundation
import UIKit

/// A build check, compiled only with SELFCHECK in SWIFT_ACTIVE_COMPILATION_CONDITIONS (README "Checking a build"). Launched
/// with `-selfcheck`, the app records its first screen, profile and mode, hashes the selected model's files against the
/// Hub manifest (read-only: nothing is moved or deleted), sends one chat message in the default mode through the chat's
/// own reply path (the chat is deleted afterwards), reopens the engine in Exact and decodes a prompt-ids file greedily,
/// then writes Documents/selfcheck.json, logs `SELFCHECK DONE` and exits.
@MainActor
enum SelfCheck {
    static let requested = ProcessInfo.processInfo.arguments.contains("-selfcheck")
    /// Read by DmengLaunch.userMode: the Exact pass's mode for this process only (never saved).
    nonisolated(unsafe) static var modeOverride: String?
    /// Documents/selfcheck-ids.txt (comma-separated ids) if present, else the Pittsburgh prompt.
    static let builtInIds = [760, 3712, 314, 279, 3177, 314, 26635, 11690, 440]
    static let exactTokens = 400
    static let chatPrompt = "In one sentence, why is the sky blue?"
    private static var started = false
    private static var out: [String: Any] = [:]

    static func setupShown() {
        guard requested, !started else { return }
        started = true
        out["first_screen"] = "setup"
        out["missing"] = ModelKind.selected.problems
        finish()
    }

    static func start(_ model: ChatModel) {
        guard !started else { return }
        started = true
        Task { await run(model) }
    }

    private static func run(_ model: ChatModel) async {
        let kind = ModelKind.selected
        out["first_screen"] = SetupGate.shared.showSetup ? "setup" : "chat"
        out["model"] = kind.rawValue
        out["profile"] = DmengLaunch.profile
        out["available_at_open_gb"] = Double(DmengLaunch.availableAtOpen) / 1e9
        out["mode_default"] = DmengLaunch.userMode
        out["mode_saved"] = UserDefaults.standard.string(forKey: kind.modeKey) ?? NSNull()
        let convsBefore = ConversationStore.loadAll().count
        out["conversations_before"] = convsBefore
        log("SELFCHECK screen \(out["first_screen"]!) model \(kind.rawValue) profile \(DmengLaunch.profile) mode \(DmengLaunch.userMode)")

        // 1. the selected model's files against the Hub manifest's profile parts
        out["verify"] = await verify(kind)

        // 2. one chat message in the default mode (the app's launch, then the chat's reply path); the chat is deleted
        model.startOnce()
        for _ in 0..<900 where ![EngineState.ready].contains(model.engineState) {
            if case .failed = model.engineState { break }
            if case .refused = model.engineState { break }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        out["engine_default"] = model.engineState.label
        if model.engineState == .ready {
            out["resolved_default"] = resolved((model.engine as? DmengChatEngine)?.selfCheckStats ?? "")
            let r = await model.reply(to: chatPrompt, maxNew: 160)
            let id = model.conversations.first { $0.messages.contains { $0.role == .user && $0.content == chatPrompt } }?.id
            out["chat"] = ["prompt": chatPrompt, "text": r.text, "tokens": r.tokens, "reason": r.reason,
                           "first_token_ms": r.firstTokenMs ?? -1, "ui_ttft_ms": r.uiTtftMs ?? -1,
                           "tok_per_s": r.uiDecodeTokPerSec ?? r.meanTokPerSec, "mean_tok_per_s": r.meanTokPerSec,
                           "prefill_kind": r.prefillKind ?? "", "error": r.error ?? ""] as [String: Any]
            if let id { model.delete(id) }
        }
        out["conversations_after"] = ConversationStore.loadAll().count

        // 3. Exact: the engine reopened in Exact, greedy decoding of the prompt ids
        if let d = model.engine as? DmengChatEngine {
            modeOverride = "exact"
            await d.unload()
            model.engineState = .idle
            await model.load()
            out["engine_exact"] = model.engineState.label
            if model.engineState == .ready {
                out["resolved_exact"] = resolved(d.selfCheckStats)
                let ids = promptIds()
                do {
                    let g = try await d.selfCheckGreedy(ids, n: exactTokens)
                    try? g.ids.map(String.init).joined(separator: ",").write(to: docsURL.appendingPathComponent("selfcheck_exact.ids"),
                                                                            atomically: true, encoding: .utf8)
                    out["exact"] = ["prompt_ids": ids.count, "tokens": g.ids.count, "source": g.source,
                                    "decode_s": g.seconds, "tok_per_s": Double(g.ids.count) / max(1e-9, g.seconds),
                                    "ids": g.ids] as [String: Any]
                } catch {
                    out["exact"] = ["error": String(describing: error)]
                }
            }
            modeOverride = nil
        }
        finish()
    }

    private static func promptIds() -> [Int] {
        let f = docsURL.appendingPathComponent("selfcheck-ids.txt")
        if let s = try? String(contentsOf: f, encoding: .utf8) {
            let v = s.split { $0 == "," || $0.isWhitespace }.compactMap { Int($0) }
            if !v.isEmpty { return v }
        }
        return builtInIds
    }

    /// The engine's settings after precedence (`resolved` in dm_stats_json), as JSON if it parses.
    private static func resolved(_ stats: String) -> Any {
        guard let d = stats.data(using: .utf8), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return NSNull() }
        return j["resolved"] ?? NSNull()
    }

    /// Size and SHA-256 of every required part of the device's profile (a derived file also counts in .dmeng/, as the
    /// completeness check takes it). The manifest: the Hub's, else the downloader's cached copy.
    private static func verify(_ kind: ModelKind) async -> [String: Any] {
        var data: Data?
        if let (d, r) = try? await URLSession.shared.data(from: URL(string: "https://huggingface.co/\(kind.hubRepo)/resolve/main/moved-store.json")!),
           (r as? HTTPURLResponse)?.statusCode == 200 { data = d }
        let source = data != nil ? "hub" : "cached"
        if data == nil { data = try? Data(contentsOf: kind.folder.appendingPathComponent(".download/manifest.json")) }
        guard let data, let m = (try? MovedStoreManifest.manifest(from: data, profile: DmengLaunch.profile)) ?? nil else {
            return ["error": "no manifest"]
        }
        var parts: [[String: Any]] = []
        var ok = 0
        let t0 = Date()
        for f in m.files where f.required {
            var p = kind.folder.appendingPathComponent(f.local).path
            if ModelDownloader.resolvedSize(p) == nil, f.local.hasPrefix("derived/") {
                p = kind.folder.appendingPathComponent(".dmeng/" + (f.local as NSString).lastPathComponent).path
            }
            let size = ModelDownloader.resolvedSize(p)
            var sha: String?
            if size == f.size { sha = await ModelDownloader.sha256(URL(fileURLWithPath: (p as NSString).resolvingSymlinksInPath)) { _ in } }
            let good = sha == f.sha256
            if good { ok += 1 }
            parts.append(["part": f.local, "ok": good, "size": size ?? -1, "want_size": f.size,
                          "at": String(p.dropFirst(kind.folder.path.count + 1))])
            log("SELFCHECK verify \(f.local): \(good ? "ok" : "BAD (size \(size ?? -1), want \(f.size))")")
        }
        log("SELFCHECK verify \(ok) of \(parts.count) parts ok in \(String(format: "%.0f", Date().timeIntervalSince(t0))) s (manifest: \(source))")
        return ["manifest": source, "profile": DmengLaunch.profile, "parts_ok": ok, "parts": parts.count, "seconds": Date().timeIntervalSince(t0),
                "per_part": parts]
    }

    private static func finish() {
        let url = docsURL.appendingPathComponent("selfcheck.json")
        if let d = try? JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]) { try? d.write(to: url) }
        let v = out["verify"] as? [String: Any], e = out["exact"] as? [String: Any], c = out["chat"] as? [String: Any]
        log("SELFCHECK DONE screen=\(out["first_screen"] ?? "?") profile=\(out["profile"] ?? "?") mode=\(out["mode_default"] ?? "?") "
            + "verify=\(v?["parts_ok"] ?? "?")/\(v?["parts"] ?? "?") exact_tokens=\(e?["tokens"] ?? "?") "
            + "chat_ttft_ms=\(c?["first_token_ms"] ?? "?") chat_tok_s=\(c?["tok_per_s"] ?? "?") -> \(url.lastPathComponent)")
        exit(0)
    }
}
#endif
