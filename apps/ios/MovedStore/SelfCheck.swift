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
    /// A lever's model options for the engine's next open (-selfcheck-lever-opts), this process only.
    nonisolated(unsafe) static var optsOverride: String?
    static func arg(_ name: String) -> String? {
        let a = ProcessInfo.processInfo.arguments
        guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
        return a[i + 1]
    }
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
    /// -selfcheck-chat-long: the new chat's first message is this long request (about 460 tokens) instead of chatPrompt.
    static let longPrompt = """
    I am writing a small command-line tool in Python that reads a folder of CSV files, each one a day of readings from a set     of temperature sensors, and produces a weekly summary. Every file has a header row with the columns timestamp,     sensor_id, temperature_c and status, where status is either ok, warn or fault. Some files have rows out of order, some     have duplicated rows, and a few have temperatures recorded as empty strings when a sensor dropped out. The tool should     accept the folder path and an optional start date on the command line, read only the seven files that fall in the week     beginning on that date (or the most recent full week when no date is given), and skip any file it cannot parse while     printing a warning that names the file and the line number of the first bad row. For each sensor it should report the     number of readings, the minimum, maximum and mean temperature over the week ignoring empty values, the number of warn     and fault rows, and the longest stretch of consecutive hours with no reading at all. Duplicated rows should be counted     once. The summary should be printed as an aligned table sorted by sensor id, and also written to a JSON file next to the     input folder when a flag asks for it. Please explain how you would structure the program into functions, which parts of     the standard library you would use for the dates, the CSV parsing and the command-line arguments, how you would handle     time zones if the timestamps carry an offset, and how you would test the gap computation and the duplicate handling     without real sensor files. Then write the core function that takes the parsed rows for one sensor and returns its     summary, with comments, and list the edge cases you would cover in its tests, such as a sensor with a single reading,     a week where a sensor never reports, readings exactly on the hour boundaries, and a fault row with an empty temperature.
    """
    static var firstPrompt: String { ProcessInfo.processInfo.arguments.contains("-selfcheck-chat-long") ? longPrompt : chatPrompt }
    /// Inkling's prompts: Documents/p2/ink_pitt_ids.txt and ink_code_ids.txt (the Nov 23 phone runs' files), else its
    /// Pittsburgh prompt built in (the coding row is then left out).
    static let inkPittBuiltIn = [371, 151698, 149879, 149876, 152652, 149879, 220, 136558, 165300, 149899]
    static func idsFile(_ rel: String) -> [Int]? {
        guard let s = try? String(contentsOf: docsURL.appendingPathComponent(rel), encoding: .utf8) else { return nil }
        let v = s.split { $0 == "," || $0.isWhitespace }.compactMap { Int($0) }
        return v.isEmpty ? nil : v
    }
    static var pittIds: [Int] { ModelKind.selected == .inkling ? (idsFile("p2/ink_pitt_ids.txt") ?? inkPittBuiltIn) : builtInIds }
    static var codeRowIds: [Int]? { ModelKind.selected == .inkling ? idsFile("p2/ink_code_ids.txt") : codeIds }
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
         "thermal_start": thermal0, "thermal_end": thermal(),
         "rate_shown": r.split?["rate_shown"] ?? -1, "rate_engine": r.split?["rate_engine"] ?? -1, "rate_ok": rateOK(r)]
    }
    /// The shown tokens/s within 2% of the engine counters' (dm_counters: steps / time inside dm_step).
    static func rateOK(_ r: ReplyRecord) -> Bool {
        guard let s = r.split?["rate_shown"], let e = r.split?["rate_engine"], e > 0, r.tokens > 0 else { return r.tokens == 0 }
        return abs(s - e) / e <= 0.02
    }

    /// The drive's free space (GB): what the volume reports, and what iOS would make available for important use.
    static func freeGB() -> [String: Double] {
        guard let v = try? URL(fileURLWithPath: NSHomeDirectory()).resourceValues(forKeys: [.volumeAvailableCapacityKey, .volumeAvailableCapacityForImportantUsageKey]) else { return [:] }
        return ["free_gb": Double(v.volumeAvailableCapacity ?? -1) / 1e9, "free_important_gb": Double(v.volumeAvailableCapacityForImportantUsage ?? -1) / 1e9]
    }
    /// `-selfcheck-cleanup`: the app's own leftovers only (old profiling logs and test outputs in Documents, the stale
    /// temporary download, half-written saved states); never the model folders, the conversations or a finished state.
    static func cleanup() -> [[String: Any]] {
        let fm = FileManager.default
        var removed: [[String: Any]] = []
        func rm(_ u: URL) {
            let n = ((try? fm.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?.int64Value ?? -1
            do { try fm.removeItem(at: u); removed.append(["path": u.path.replacingOccurrences(of: NSHomeDirectory() + "/", with: ""), "bytes": n])
                 log("SELFCHECK cleanup removed \(u.lastPathComponent) (\(n) bytes)") } catch { log("SELFCHECK cleanup could not remove \(u.path): \(error)") }
        }
        let docs = (try? fm.contentsOfDirectory(at: docsURL, includingPropertiesForKeys: nil)) ?? []
        for u in docs {
            let n = u.lastPathComponent
            if (n.hasPrefix("gpuprof-") && n.hasSuffix(".txt")) || n.hasPrefix("p3out_") || (n.hasPrefix("tf-") && n.hasSuffix(".json"))
                || n == "ui.json" || n == "probe.json" || (n.hasPrefix("selfcheck") && (n.hasSuffix(".json") || n.hasSuffix(".ids"))) { rm(u) }
        }
        // conversations a self-check made and could not delete (a run that was stopped): only the self-check's own two
        // prompts in them, nothing else
        let mine: Set<String> = [chatPrompt, followPrompt, longPrompt]
        for c in ConversationStore.loadAll() {
            let users = c.messages.filter { $0.role == .user }.map(\.content)
            if !users.isEmpty && users.allSatisfy({ mine.contains($0) }) {
                ConversationStore.delete(c.id)
                removed.append(["conversation": c.id.uuidString, "messages": c.messages.count])
                log("SELFCHECK cleanup removed the self-check's conversation \(c.id.uuidString) (\(c.messages.count) messages)")
            }
        }
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
        for u in (try? fm.contentsOfDirectory(at: tmp, includingPropertiesForKeys: nil)) ?? []
            where u.lastPathComponent.hasPrefix("CFNetworkDownload_") && u.lastPathComponent.hasSuffix(".tmp") { rm(u) }
        for d in ModelKind.allCases.flatMap({ (try? fm.contentsOfDirectory(at: DmengLaunch.prefixDir($0), includingPropertiesForKeys: nil)) ?? [] }) {
            for u in (try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: nil)) ?? [] {
                let size = ((try? fm.attributesOfItem(atPath: u.path))?[.size] as? NSNumber)?.int64Value ?? -1
                let since = Double(arg("-selfcheck-states-since") ?? "") ?? .infinity
                let mt = ((try? fm.attributesOfItem(atPath: u.path))?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
                let mine = u.lastPathComponent.hasSuffix(".bin") && !u.lastPathComponent.hasSuffix(".pin.bin") && mt >= since
                if u.lastPathComponent.hasSuffix(".tmp") || (u.lastPathComponent.hasSuffix(".bin") && size == 0) || mine { rm(u) }
            }
        }
        return removed
    }

    /// The saved chat states (not the pinned system-prompt states) in the app's state folders, by path.
    static func chatStates() -> Set<String> {
        let fm = FileManager.default
        var r = Set<String>()
        for d in ModelKind.allCases.flatMap({ (try? fm.contentsOfDirectory(at: DmengLaunch.prefixDir($0), includingPropertiesForKeys: nil)) ?? [] }) {
            for u in (try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: nil)) ?? []
                where u.lastPathComponent.hasSuffix(".bin") && !u.lastPathComponent.hasSuffix(".pin.bin") { r.insert(u.path) }
        }
        return r
    }
    /// The chat states this run's chat left (its conversation is deleted, so they can never be used): removed, logged.
    static func removeStates(_ paths: Set<String>) -> [[String: Any]] {
        var removed: [[String: Any]] = []
        for p in paths.sorted() {
            let n = ((try? FileManager.default.attributesOfItem(atPath: p))?[.size] as? NSNumber)?.int64Value ?? -1
            if (try? FileManager.default.removeItem(atPath: p)) != nil {
                removed.append(["path": (p as NSString).lastPathComponent, "bytes": n])
                log("SELFCHECK removed the self-check chat's state \((p as NSString).lastPathComponent) (\(n) bytes)")
            }
        }
        return removed
    }

    /// `-selfcheck-remove-35b`: the swap's removal (approved procedure): the 35B's model folder (Documents/dmeng-model) and
    /// its saved states (Application Support/<bundle>/dmeng-prefix) are deleted, every file logged with its size; nothing
    /// else (conversations, Inkling's folder and states, settings) is touched. Then the result is written and the app exits.
    static func remove35B() -> [String: Any] {
        let fm = FileManager.default
        var r: [String: Any] = ["free_before": freeGB()]
        var files: [[String: Any]] = []
        var total: Int64 = 0
        for root in [ModelKind.qwen36.folder, DmengLaunch.prefixDir(.qwen36)] {
            if let e = fm.enumerator(at: root, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) {
                for case let u as URL in e {
                    guard let v = try? u.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), v.isRegularFile == true else { continue }
                    let n = Int64(v.fileSize ?? 0)
                    total += n
                    files.append(["path": u.path.replacingOccurrences(of: NSHomeDirectory() + "/", with: ""), "bytes": n])
                    log("SELFCHECK remove-35b \(u.path.replacingOccurrences(of: NSHomeDirectory() + "/", with: "")) (\(n) bytes)")
                }
            }
            do { try fm.removeItem(at: root); log("SELFCHECK remove-35b removed \(root.lastPathComponent)/") }
            catch {
                // a folder copied onto the phone by cable is not the app's to unlink from: each file is emptied in place
                // instead (its space freed; the empty name stays, and a later copy overwrites it)
                log("SELFCHECK remove-35b could not remove \(root.lastPathComponent)/ (\(error.localizedDescription)): emptying its files")
                var emptied = 0
                if let e = fm.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) {
                    for case let u as URL in e {
                        guard (try? u.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true else { continue }
                        if (try? fm.removeItem(at: u)) != nil { emptied += 1; continue }
                        if let h = try? FileHandle(forWritingTo: u) {
                            do { try h.truncate(atOffset: 0); emptied += 1 } catch { log("SELFCHECK remove-35b could not empty \(u.lastPathComponent): \(error)") }
                            try? h.close()
                        } else { log("SELFCHECK remove-35b could not open \(u.lastPathComponent)") }
                    }
                }
                r["emptied_\(root.lastPathComponent)"] = emptied
                log("SELFCHECK remove-35b emptied \(emptied) files in \(root.lastPathComponent)/")
            }
        }
        r["files"] = files
        r["bytes"] = total
        r["free_after"] = freeGB()
        r["conversations"] = ConversationStore.loadAll().count
        log("SELFCHECK remove-35b: \(files.count) files, \(total) bytes; free \(freeGB())")
        return r
    }

    /// The footprint by kind (GB): task_vm_info's internal / external / compressed and the graphics (Metal, IOSurface)
    /// and media ledgers, and the malloc zones' bytes in use and allocated.
    static func regions() -> [String: Double] {
        var r: [String: Double] = [:]
        if let v = MemProbe.vmInfo() {
            r["footprint"] = Double(v.phys_footprint) / 1e9
            r["internal"] = Double(v.internal) / 1e9
            r["external"] = Double(v.external) / 1e9
            r["compressed"] = Double(v.compressed) / 1e9
            r["graphics"] = Double(v.ledger_tag_graphics_footprint) / 1e9
            r["media"] = Double(v.ledger_tag_media_footprint) / 1e9
            r["purgeable_nonvolatile"] = Double(v.ledger_purgeable_nonvolatile) / 1e9
        }
        var ms = malloc_statistics_t()
        malloc_zone_statistics(nil, &ms)
        r["malloc_in_use"] = Double(ms.size_in_use) / 1e9
        r["malloc_allocated"] = Double(ms.size_allocated) / 1e9
        return r
    }

    private static func run(_ model: ChatModel) async {
        optsOverride = arg("-selfcheck-opts")  // model options for every open of this launch (a measurement's setting)
        if let m = arg("-selfcheck-mode") { modeOverride = m; out["mode_override"] = m }  // the launch's mode (exact / fast)
        if let o = optsOverride { out["opts"] = o }
        if ProcessInfo.processInfo.arguments.contains("-selfcheck-remove-35b") {
            out["remove_35b"] = remove35B()
            finish()
            return
        }
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
        out["free_start"] = freeGB()
        if ProcessInfo.processInfo.arguments.contains("-selfcheck-cleanup") {
            out["cleanup"] = cleanup()
            out["free_after_cleanup"] = freeGB()
        }
        log("SELFCHECK screen \(out["first_screen"]!) model \(kind.rawValue) profile \(DmengLaunch.profile) mode \(DmengLaunch.userMode)")

        // -selfcheck-lever NAME -selfcheck-lever-opts "OPTS" [-selfcheck-lever-mode fast|exact] [-selfcheck-lever-tokens N]:
        // one launch, the engine opened twice in the mode (default, then with OPTS as its model options / run: flags); per
        // arm a cold short prompt (the coding prompt's first 48 ids: a short batch) and the Pittsburgh prompt decoded
        // greedily for N tokens (200), with the decode's counters
        if let lever = arg("-selfcheck-lever"), let d = model.engine as? DmengChatEngine {
            let mode = arg("-selfcheck-lever-mode") ?? "fast"
            let n = Int(arg("-selfcheck-lever-tokens") ?? "") ?? 200
            out["lever"] = lever
            out["lever_opts"] = arg("-selfcheck-lever-opts") ?? ""
            out["lever_mode"] = mode
            var arms: [String: Any] = [:]
            // default, lever, default again: the second default shows the order's own effect (the drive's and the OS's
            // caches are warmer for a later arm)
            for (arm, opts) in [("default", nil), ("lever", arg("-selfcheck-lever-opts")), ("default2", nil)] as [(String, String?)] {
                modeOverride = mode
                optsOverride = opts ?? arg("-selfcheck-opts")
                await d.unload()
                model.engineState = .idle
                let tl = Date()
                await model.load()
                var r: [String: Any] = ["engine": model.engineState.label, "load_s": Date().timeIntervalSince(tl)]
                if model.engineState == .ready {
                    r["resolved"] = resolved(d.selfCheckStats)
                    let th = thermal()
                    if let g = try? await d.selfCheckGreedy(Array((codeRowIds ?? codeIds).prefix(48)), n: 1) {
                        r["short_prefill_s"] = g.prefillSeconds
                        r["short_prefill"] = json(g.prefillJSON)
                    }
                    if let g = try? await d.selfCheckGreedy(pittIds, n: n) {
                        r["pitt"] = ["tokens": g.ids.count, "decode_s": g.seconds, "tok_per_s": Double(g.ids.count) / max(1e-9, g.seconds),
                                     "ids": g.ids, "stats": json(g.decodeStats)] as [String: Any]
                        log("SELFCHECK lever \(lever) \(arm): \(String(format: "%.2f", Double(g.ids.count) / max(1e-9, g.seconds))) tok/s")
                    }
                    r["thermal"] = "\(th) -> \(thermal())"
                    r["mem"] = mem("lever \(arm)")
                }
                arms[arm] = r
            }
            modeOverride = nil
            optsOverride = nil
            out["arms"] = arms
            finish()
            return
        }

        // -selfcheck-agree N: the coding prompt in Exact, N greedy ids E (with the prompt's time and reads); then in Fast the
        // same prompt and E teacher-forced: top-choice agreement with Exact over the N positions after the prompt
        if let ns = arg("-selfcheck-agree"), let n = Int(ns), let d = model.engine as? DmengChatEngine, let prompt = codeRowIds {
            var res: [String: Any] = ["prompt_ids": prompt.count, "n": n]
            for mode in ["exact", "fast"] {
                modeOverride = mode
                await d.unload()
                model.engineState = .idle
                await model.load()
                guard model.engineState == .ready else { res["\(mode)_engine"] = model.engineState.label; continue }
                if mode == "exact" {
                    if let g = try? await d.selfCheckGreedy(prompt, n: n) {
                        res["exact_ids"] = g.ids
                        res["exact_prefill_s"] = g.prefillSeconds
                        res["exact_prefill"] = json(g.prefillJSON)
                    }
                } else if let e = res["exact_ids"] as? [Int], let f = try? await d.selfCheckForced(prompt, cont: e) {
                    res["fast_agree"] = Double(f.agree) / Double(max(1, f.n))
                    res["fast_first_diff"] = f.firstDiff
                    res["fast_prefill_s"] = f.prefillSeconds
                    res["fast_prefill"] = json(f.prefillJSON)
                    res["fast_top"] = f.top
                    res["fast_stats"] = json(d.selfCheckStats)
                    log("SELFCHECK agree fast vs exact \(f.agree)/\(f.n), first diff \(f.firstDiff), fast prompt \(String(format: "%.1f", f.prefillSeconds)) s")
                }
                res["\(mode)_mem"] = mem("agree \(mode)")
            }
            modeOverride = nil
            out["agree"] = res
            finish()
            return
        }

        // -selfcheck-memrow: only the memory of one long prompt (the coding prompt, 357 ids) in the default mode: the
        // footprint before it, its highest value while the prompt is read (sampled every 50 ms) and after ~10 s of decode
        if ProcessInfo.processInfo.arguments.contains("-selfcheck-memrow") {
            optsOverride = arg("-selfcheck-lever-opts")  // a diagnostic's model options for this launch (e.g. the GPU share off)
            out["memrow_opts"] = optsOverride ?? ""
            model.startOnce()
            for _ in 0..<900 where model.engineState != .ready {
                if case .failed = model.engineState { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            if let d = model.engine as? DmengChatEngine, model.engineState == .ready {
                try? await Task.sleep(nanoseconds: 10_000_000_000)  // the launch's warm-up in idle time
                let before = MemProbe.read("memrow before")
                let regBefore = regions()
                let statsBefore = json(d.selfCheckStats)
                final class Peak: @unchecked Sendable { var v = 0; var on = true; var reg: [String: Double] = [:] }
                let pk = Peak()
                let sampler = Thread {
                    while pk.on {
                        let f = MemProbe.read("").physFootprint
                        if f > pk.v { pk.v = f; pk.reg = regions() }
                        usleep(50_000)
                    }
                }
                sampler.start()
                let t0 = Date()
                let g = try? await d.selfCheckGreedy(codeRowIds ?? codeIds, n: 26)
                let secs = Date().timeIntervalSince(t0)
                pk.on = false
                let after = MemProbe.read("memrow after")
                var mr: [String: Any] = [:]
                mr["before_gb"] = Double(before.physFootprint) / 1e9
                mr["prompt_peak_gb"] = Double(pk.v) / 1e9
                mr["after_gb"] = Double(after.physFootprint) / 1e9
                mr["lifetime_peak_gb"] = Double(after.physFootprintPeak) / 1e9
                mr["prefill_s"] = g?.prefillSeconds ?? -1
                mr["decode_s"] = g?.seconds ?? -1
                mr["tokens"] = g?.ids.count ?? -1
                mr["seconds"] = secs
                mr["thermal"] = after.thermal
                mr["regions_before"] = regBefore
                mr["regions_peak"] = pk.reg
                mr["regions_after"] = regions()
                mr["stats_before"] = (statsBefore as? [String: Any])?["memory"] ?? NSNull()
                mr["stats_after"] = (json(d.selfCheckStats) as? [String: Any])?["memory"] ?? NSNull()
                out["memrow"] = mr
                log("SELFCHECK memrow before \(before.physFootprint) peak \(pk.v) after \(after.physFootprint) (decode \(g?.seconds ?? -1) s)")
            }
            finish()
            return
        }

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
            let statesBefore = chatStates()
            var th = thermal()
            final class Peak: @unchecked Sendable { var v = 0; var on = true }
            let pk = Peak()
            Thread { while pk.on { pk.v = max(pk.v, MemProbe.read("").physFootprint); usleep(50_000) } }.start()
            out["chat_regions_before"] = regions()
            let r1 = await model.reply(to: firstPrompt, maxNew: 160)
            let id = model.conversations.first { $0.messages.contains { $0.role == .user && $0.content == firstPrompt } }?.id
            out["chat"] = chatRow(r1, thermal0: th)
            try? await Task.sleep(nanoseconds: 5_000_000_000)
            out["chat_peak_gb"] = Double(pk.v) / 1e9
            out["chat_regions_after"] = regions()
            out["chat_engine_after"] = { let j = json((model.engine as? DmengChatEngine)?.selfCheckStats ?? "") as? [String: Any]
                                         return ["memory": j?["memory"] ?? NSNull(), "prefix_cache": j?["prefix_cache"] ?? NSNull()] as [String: Any] }()
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            th = thermal()
            let r2 = await model.reply(to: followPrompt, maxNew: 160)
            out["chat_followup"] = chatRow(r2, thermal0: th)
            pk.on = false
            out["chat_peak_all_gb"] = Double(pk.v) / 1e9
            out["chat_regions_end"] = regions()
            if let id { model.delete(id) }
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            await (model.engine as? DmengChatEngine)?.waitIdle()  // the chat's warm-up after the reply has written its state
            out["chat_states_removed"] = removeStates(chatStates().subtracting(statesBefore))
            out["mem_after_chat"] = mem("after chat")
        }
        out["conversations_after"] = ConversationStore.loadAll().count
        if ProcessInfo.processInfo.arguments.contains("-selfcheck-chat-only") {  // the chat rows only
            out["mem_end"] = mem("selfcheck end")
            out["memory_warnings"] = model.memoryWarnings
            finish()
            return
        }

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
                // -selfcheck-code-mode M: the coding row in mode M only (both when not given)
                let codeMode = arg("-selfcheck-code-mode")
                var rows: [(String, [Int])] = [("pitt", pittIds)]
                if let c = codeRowIds, codeMode == nil || codeMode == mode { rows.append(("code", c)) }
                for (row, ids) in rows {
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
        out["free_end"] = freeGB()
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
        // the shown speed must be the engine's: a chat reply whose shown tokens/s differs from the counters' by over 2% fails
        let rateRows = ["chat", "chat_followup"].compactMap { out[$0] as? [String: Any] }
        if rateRows.contains(where: { ($0["rate_ok"] as? Bool) == false }) {
            log("SELFCHECK FAIL rate: shown tokens/s differs from the engine counters' by over 2%: "
                + rateRows.map { "\($0["rate_shown"] ?? "?") vs \($0["rate_engine"] ?? "?")" }.joined(separator: ", "))
            exit(3)
        }
        exit(0)
    }
}
#endif
