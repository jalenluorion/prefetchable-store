#if SELFCHECK
import Foundation
import UIKit

/// A build check, compiled only with SELFCHECK in SWIFT_ACTIVE_COMPILATION_CONDITIONS (README "Checking a build"). Launched
/// with `-selfcheck`, the app records its first screen, profile and mode, hashes the selected model's files against the
/// Hub manifest with `-selfcheck-verify` (read-only: nothing is moved or deleted), sends one chat message in the default mode through the chat's
/// own reply path (the chat is deleted afterwards), reopens the engine in Exact and in Fast and decodes the Pittsburgh
/// and coding prompts greedily, then writes Documents/selfcheck.json, logs `SELFCHECK DONE` and exits.
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

    /// The A/B rows (one launch): the chat in the default mode (a new chat's first message, then a follow-up), then the
    /// engine reopened in Exact and in Fast, each decoding the Pittsburgh prompt and the coding prompt (357 ids) greedily
    /// for `exactTokens`, with the engine's counters for the decode steps only. `-selfcheck-verify` also hashes the files.
    static let codeIds = [248045,846,198,40,2688,3133,383,264,2526,12654,2393,421,43303,18805,1433,7189,321,17657,287,1070,310,12977,13,9170,2706,1010,7189,513,5384,321,22217,279,1788,3150,369,5158,10598,13,5514,369,279,523,25,271,71093,12305,198,464,29174,11,854,11,2857,271,1005,417,27150,25,198,262,687,1259,2251,3676,701,11,1752,11,1866,11789,28,16,15,15,11,1866,30997,625,28,17,13,15,1590,198,285,638,3745,283,1752,198,285,638,6469,11789,283,1866,11789,198,285,638,6469,30997,625,283,1866,30997,625,198,285,638,43007,283,2958,198,285,638,7155,24230,283,2168,198,285,638,20362,283,29174,30371,363,198,285,29174,30977,8373,7424,1385,66032,11,37993,3497,553,2388,363,271,262,687,884,1153,11,1347,1590,198,285,440,638,20362,25,198,309,413,524,638,43007,25,198,388,638,7155,24230,283,854,6177,363,198,309,638,43007,1989,20009,8,198,285,413,2343,1153,43007,8,2518,638,6469,11789,25,198,309,638,22320,363,271,262,687,696,66032,1153,1590,198,285,1345,2912,25,198,309,854,10795,7,15,13,16,8,198,309,413,638,7155,24230,321,854,6177,363,471,638,7155,24230,835,638,6469,30997,625,25,198,388,638,22320,363,271,262,687,17657,1153,1590,198,285,3470,283,638,43007,198,285,440,1724,1153,3745,11,328,64,871,430,281,25,198,309,364,427,303,3470,25,198,388,281,3708,9028,20868,2515,8,478,2823,77,871,198,285,440,638,20362,25,198,309,638,43007,283,2958,198,309,638,7155,24230,283,2168,198,71093,271,9592,1396,6741,303,411,1970,11,10033,1204,1754,799,31130,466,40922,7189,11,321,2873,264,34773,2243,13,13262,279,569,3622,279,1788,11,321,884,264,3160,363,1654,421,17657,287,8554,369,2047,13,248046,198,248045,74455,198,248068,198]
    static let followPrompt = "And why are sunsets red?"
    static func thermal() -> String { thermalName(ProcessInfo.processInfo.thermalState) }
    static func mem(_ label: String) -> Any {
        let m = MemProbe.read(label)
        return ["phys_footprint_gb": Double(m.physFootprint) / 1e9, "peak_gb": Double(m.physFootprintPeak) / 1e9,
                "available_gb": Double(m.osProcAvailable) / 1e9, "thermal": m.thermal] as [String: Any]
    }
    static func json(_ s: String) -> Any {
        guard let d = s.data(using: .utf8), let j = try? JSONSerialization.jsonObject(with: d) else { return NSNull() }
        return j
    }
    /// Size and modification time of every file in the model folder (a re-derive or rewrite at open shows here).
    static func listing(_ kind: ModelKind) -> [String: String] {
        var r: [String: String] = [:]
        let base = kind.folder
        guard let e = FileManager.default.enumerator(at: base, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]) else { return r }
        for case let u as URL in e {
            guard let v = try? u.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]), v.isRegularFile == true else { continue }
            r[String(u.path.dropFirst(base.path.count + 1))] = "\(v.fileSize ?? -1)|\(Int(v.contentModificationDate?.timeIntervalSince1970 ?? 0))"
        }
        return r
    }
    static func chatRow(_ r: ReplyRecord, thermal0: String) -> [String: Any] {
        ["text": r.text, "tokens": r.tokens, "reason": r.reason, "first_token_ms": r.firstTokenMs ?? -1, "ui_ttft_ms": r.uiTtftMs ?? -1,
         "tok_per_s": r.uiDecodeTokPerSec ?? r.meanTokPerSec, "mean_tok_per_s": r.meanTokPerSec, "prefill_kind": r.prefillKind ?? "",
         "prompt_tokens": r.promptTokens ?? -1, "memory_warnings": r.memoryWarnings, "error": r.error ?? "",
         "thermal_start": thermal0, "thermal_end": thermal()]
    }

    private static func run(_ model: ChatModel) async {
        let kind = ModelKind.selected
        let tStart = Date()
        out["thermal_start"] = thermal()
        out["first_screen"] = SetupGate.shared.showSetup ? "setup" : "chat"
        out["model"] = kind.rawValue
        out["profile"] = DmengLaunch.profile
        out["available_at_open_gb"] = Double(DmengLaunch.availableAtOpen) / 1e9
        out["mode_default"] = DmengLaunch.userMode
        out["mode_saved"] = UserDefaults.standard.string(forKey: kind.modeKey) ?? NSNull()
        let convsBefore = ConversationStore.loadAll().count
        out["conversations_before"] = convsBefore
        out["files_before"] = listing(kind)
        log("SELFCHECK screen \(out["first_screen"]!) model \(kind.rawValue) profile \(DmengLaunch.profile) mode \(DmengLaunch.userMode)")

        // 1. the selected model's files against the Hub manifest's profile parts (only with -selfcheck-verify)
        if ProcessInfo.processInfo.arguments.contains("-selfcheck-verify") { out["verify"] = await verify(kind) }

        // 2. the chat in the default mode (the app's launch, then the chat's reply path): a new chat's first message and a
        //    follow-up, each sent after 10 s idle; the chat is deleted
        model.startOnce()
        for _ in 0..<900 where ![EngineState.ready].contains(model.engineState) {
            if case .failed = model.engineState { break }
            if case .refused = model.engineState { break }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        out["ready_s"] = Date().timeIntervalSince(tStart)
        out["engine_default"] = model.engineState.label
        if model.engineState == .ready {
            out["resolved_default"] = resolved((model.engine as? DmengChatEngine)?.selfCheckStats ?? "")
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            var th = thermal()
            let r1 = await model.reply(to: chatPrompt, maxNew: 160)
            let id = model.conversations.first { $0.messages.contains { $0.role == .user && $0.content == chatPrompt } }?.id
            out["chat"] = chatRow(r1, thermal0: th)
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            th = thermal()
            let r2 = await model.reply(to: followPrompt, maxNew: 160)
            out["chat_followup"] = chatRow(r2, thermal0: th)
            if let id { model.delete(id) }
            out["mem_after_chat"] = mem("after chat")
        }
        out["conversations_after"] = ConversationStore.loadAll().count

        // 3. Exact, then Fast: the engine reopened in the mode, greedy decoding of the Pittsburgh prompt (the Exact gate) and
        //    of the coding prompt
        if let d = model.engine as? DmengChatEngine {
            for mode in ["exact", "fast"] {
                modeOverride = mode
                await d.unload()
                model.engineState = .idle
                let tl = Date()
                await model.load()
                out["engine_\(mode)"] = model.engineState.label
                out["load_s_\(mode)"] = Date().timeIntervalSince(tl)
                guard model.engineState == .ready else { continue }
                out["resolved_\(mode)"] = resolved(d.selfCheckStats)
                for (row, ids) in [("pitt", builtInIds), ("code", codeIds)] {
                    let th = thermal()
                    do {
                        let g = try await d.selfCheckGreedy(ids, n: exactTokens)
                        try? g.ids.map(String.init).joined(separator: ",").write(to: docsURL.appendingPathComponent("selfcheck_\(mode)_\(row).ids"),
                                                                                atomically: true, encoding: .utf8)
                        out["\(mode)_\(row)"] = ["prompt_ids": ids.count, "tokens": g.ids.count, "source": g.source,
                                                 "prefill_s": g.prefillSeconds, "decode_s": g.seconds,
                                                 "tok_per_s": Double(g.ids.count) / max(1e-9, g.seconds),
                                                 "ids": g.ids, "stats": json(g.decodeStats), "mem": mem("\(mode) \(row)"),
                                                 "thermal_start": th, "thermal_end": thermal()] as [String: Any]
                        log("SELFCHECK \(mode) \(row): \(g.ids.count) tokens, \(String(format: "%.2f", Double(g.ids.count) / max(1e-9, g.seconds))) tok/s, thermal \(th) -> \(thermal())")
                    } catch {
                        out["\(mode)_\(row)"] = ["error": String(describing: error)]
                    }
                }
            }
            if let e = out["exact_pitt"] as? [String: Any] { out["exact"] = e }
            modeOverride = nil
        }
        out["memory_warnings"] = model.memoryWarnings
        out["mem_end"] = mem("selfcheck end")
        out["thermal_end"] = thermal()
        out["files_after"] = listing(kind)
        out["total_s"] = Date().timeIntervalSince(tStart)
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
        if let v = try? URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [.volumeAvailableCapacityKey, .volumeAvailableCapacityForImportantUsageKey]) {
            out["volume_free_gb"] = Double(v.volumeAvailableCapacity ?? -1) / 1e9
            out["volume_free_important_gb"] = Double(v.volumeAvailableCapacityForImportantUsage ?? -1) / 1e9
        }
        // only JSON values (a non-finite number or another type would make the whole write fail)
        func clean(_ v: Any) -> Any {
            switch v {
            case let d as [String: Any]: return d.mapValues(clean)
            case let a as [Any]: return a.map(clean)
            case let x as Double: return x.isFinite ? x : String(describing: x)
            case let x as Float: return x.isFinite ? Double(x) : String(describing: x)
            case is String, is Int, is Int64, is Bool, is NSNull, is NSNumber: return v
            default: return String(describing: v)
            }
        }
        let outClean = clean(out)
        var data: Data?
        do { data = try JSONSerialization.data(withJSONObject: outClean, options: [.sortedKeys]) } catch { log("SELFCHECK JSON error: \(error)") }
        if let d = data {
            do { try d.write(to: url) } catch { log("SELFCHECK could not write \(url.lastPathComponent): \(error)") }
            // also on the console (a full drive cannot take the file): base64 in chunks, "SELFCHECK JSON i/n <chunk>"
            let b = d.base64EncodedString(), n = (b.count + 799) / 800
            for i in 0..<n {
                let lo = b.index(b.startIndex, offsetBy: i * 800), hi = b.index(lo, offsetBy: min(800, b.count - i * 800))
                log("SELFCHECK JSON \(i + 1)/\(n) " + String(b[lo..<hi]))
            }
        }
        let v = out["verify"] as? [String: Any], e = out["exact"] as? [String: Any], c = out["chat"] as? [String: Any]
        log("SELFCHECK DONE screen=\(out["first_screen"] ?? "?") profile=\(out["profile"] ?? "?") mode=\(out["mode_default"] ?? "?") "
            + "verify=\(v?["parts_ok"] ?? "?")/\(v?["parts"] ?? "?") exact_tokens=\(e?["tokens"] ?? "?") "
            + "chat_ttft_ms=\(c?["first_token_ms"] ?? "?") chat_tok_s=\(c?["tok_per_s"] ?? "?") -> \(url.lastPathComponent)")
        exit(0)
    }
}
#endif
