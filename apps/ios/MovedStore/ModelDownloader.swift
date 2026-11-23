import Foundation
import CryptoKit
import UIKit
import Network

// The model download. The app fetches the model's `moved-store.json` from its Hub repo (DownloadConfig), then every file
// the device's profile lists into Documents/<model folder>/ at its repo path.
//
// - One URLSession download task per file, all queued at once on a BACKGROUND session (nsurlsessiond keeps them going
//   while the app is suspended or killed; iOS relaunches the app for the completion events), at most
//   DownloadConfig.concurrentFiles transferring at once (httpMaximumConnectionsPerHost).
// - Resumable: pause = cancel(byProducingResumeData:), the resume data goes to <model folder>/.download/<key>.resume;
//   a network error's resume data likewise; a relaunch adopts the session's live tasks (getAllTasks) and resumes the
//   rest from their resume data (HTTP Range).
// - Each finished file is moved to .download/<key>.part, its SHA-256 computed in 4 MiB streaming reads (CryptoKit) and
//   compared with the manifest; only then is it renamed into place. Mismatch, bad HTTP status or network error: retry
//   with backoff (2, 4, 8 ... 60 s), DownloadConfig.maxAttempts times, then the file is marked failed.
// - Before starting: remaining bytes + 1 GB must fit in the volume's free space (volumeAvailableCapacityForImportantUsage),
//   else a refusal with the numbers.
// - Verified files are recorded in .download/verified.json (size, mtime, sha256), so a relaunch does not rehash 20 GB.
// - Cellular: off by default (setting `download.cellular`); two background sessions, one with allowsCellularAccess off.

struct ManifestFile: Codable, Identifiable, Hashable {
    var path: String          // in the repo (URL = DownloadConfig.baseURL + path)
    var local: String         // relative to the model folder
    var size: Int64
    var sha256: String
    var required: Bool
    var kind: String?
    var id: String { local }
    var key: String { local.replacingOccurrences(of: "/", with: "__") }
    var name: String { (local as NSString).lastPathComponent }
}

struct Manifest: Codable {
    var format: String
    var version: String
    var model: String
    var license: String
    var repo: String?
    var requiredBytes: Int64
    var files: [ManifestFile]
}

enum FileState: Equatable {
    case queued, downloading, verifying, done, paused
    case waiting(String)      // retry scheduled (reason)
    case failed(String)
    var label: String {
        switch self {
        case .queued: return "queued"
        case .downloading: return "downloading"
        case .verifying: return "checking"
        case .done: return "done"
        case .paused: return "paused"
        case .waiting(let r): return "retrying (\(r))"
        case .failed(let r): return "failed: \(r)"
        }
    }
}

struct FileProgress: Identifiable {
    var file: ManifestFile
    var state: FileState = .queued
    var bytes: Int64 = 0              // received (downloading) or hashed (verifying)
    var attempts = 0
    var resumedFrom: Int64 = 0        // the offset of the last resume (resume data or adopted task)
    var started: Date?
    var finished: Date?
    var downloadedBytes: Int64 = 0    // bytes that came over the network for this file in this launch
    var shaOK: Bool?
    var source = ""                   // "network", "already here", "adopted", "resumed"
    var id: String { file.local }
    var fraction: Double { file.size > 0 ? min(1, Double(bytes) / Double(file.size)) : 1 }
}

enum DownloadPhase: Equatable {
    case idle                  // nothing started
    case fetchingManifest
    case checking              // hashing files already in the folder
    case refused(String)       // not enough space (or no manifest)
    case offline               // the file list could not be fetched and none is cached: waits for a connection
    case downloading
    case paused
    case complete
    case failed(String)
}

/// One verified file: what was hashed (size + mtime), so an unchanged file is not hashed again.
private struct VerifiedRecord: Codable { var size: Int64; var mtime: Double; var sha256: String }

@MainActor
final class ModelDownloader: ObservableObject {
    static let shared = ModelDownloader()

    @Published private(set) var manifest: Manifest?
    @Published private(set) var phase: DownloadPhase = .idle
    @Published private(set) var files: [FileProgress] = []
    @Published private(set) var bytesPerSecond: Double = 0
    @Published private(set) var freeBytes: Int64?
    @Published private(set) var note = ""
    @Published var cellular = DownloadConfig.cellularAllowed {
        didSet {
            UserDefaults.standard.set(cellular, forKey: DownloadConfig.cellularKey)
            if phase == .downloading { Task { await self.switchSession() } }
        }
    }
    /// Verify-files action (manage screen): files hashed so far / to hash, and the files that did not match.
    @Published private(set) var verifyResult: String?
    @Published private(set) var verifying = false

    /// Set by the AppDelegate when iOS relaunches the app for background session events.
    var backgroundCompletion: (() -> Void)?
    /// Called on the main actor when every selected file is done.
    var onComplete: (() -> Void)?

    private let delegate = SessionDelegate()
    private var sessions: [Bool: URLSession] = [:]     // keyed by allowsCellularAccess
    private var index: [String: Int] = [:]              // local -> files[i]
    private var tasks: [String: URLSessionDownloadTask] = [:]
    private var pausing = false
    private var cancelling = false
    private var speedSamples: [(Date, Int64)] = []
    private var sessionBytes: Int64 = 0                 // network bytes this launch (all files)
    private var startedAt: Date?
    private var timer: Timer?
    private var lastLog = Date.distantPast
    private var verified: [String: VerifiedRecord] = [:]

    /// The network as NWPathMonitor sees it (the set-up screen's "Waiting for Wi-Fi"); a change to online retries a
    /// download that is waiting for one.
    @Published private(set) var network: NetworkState = .unknown
    enum NetworkState { case unknown, wifi, cellularOnly, none }
    private let monitor = NWPathMonitor()
    private var autoRetry: Timer?

    private init() {
        delegate.owner = self
        monitor.pathUpdateHandler = { [weak self] path in
            let st: NetworkState = path.status != .satisfied ? .none : (path.isExpensive ? .cellularOnly : .wifi)
            Task { @MainActor in self?.networkChanged(st) }
        }
        monitor.start(queue: DispatchQueue(label: "download.path"))
        verified = Self.loadVerified()
        // create the session(s) at once: a relaunch for background events must reconnect before the events arrive
        _ = session(cellular: cellular)
        _ = session(cellular: !cellular)
    }

    // MARK: derived figures

    var selected: [FileProgress] { files }
    var totalBytes: Int64 { files.isEmpty ? DownloadConfig.approxRequiredBytes : files.reduce(0) { $0 + $1.file.size } }
    var doneBytes: Int64 {
        files.reduce(0) { a, f in a + (f.state == .done ? f.file.size : (f.state == .verifying ? 0 : f.bytes)) }
    }
    var remainingBytes: Int64 { max(0, totalBytes - doneBytes) }
    var fraction: Double { totalBytes > 0 ? Double(doneBytes) / Double(totalBytes) : 0 }
    var etaSeconds: Double? { bytesPerSecond > 1e5 ? Double(remainingBytes) / bytesPerSecond : nil }
    var active: Bool { phase == .downloading || phase == .fetchingManifest || phase == .checking }
    var doneCount: Int { files.filter { $0.state == .done }.count }

    static var wantKey = "download.want"
    /// The user started a download and has not paused or cancelled it: a relaunch continues it.
    var wantDownload: Bool {
        get { UserDefaults.standard.bool(forKey: Self.wantKey) }
        set { UserDefaults.standard.set(newValue, forKey: Self.wantKey) }
    }

    // MARK: sessions

    private func session(cellular: Bool) -> URLSession {
        if let s = sessions[cellular] { return s }
        let cfg = URLSessionConfiguration.background(withIdentifier: cellular ? DownloadConfig.backgroundCellularSessionID : DownloadConfig.backgroundSessionID)
        cfg.isDiscretionary = false
        cfg.sessionSendsLaunchEvents = true
        cfg.allowsCellularAccess = cellular
        cfg.allowsExpensiveNetworkAccess = cellular
        cfg.httpMaximumConnectionsPerHost = DownloadConfig.concurrentFiles
        cfg.timeoutIntervalForRequest = 120
        cfg.timeoutIntervalForResource = 7 * 24 * 3600
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        q.name = "download"
        let s = URLSession(configuration: cfg, delegate: delegate, delegateQueue: q)
        sessions[cellular] = s
        return s
    }

    private func allTasks() async -> [URLSessionTask] {
        var out: [URLSessionTask] = []
        for s in sessions.values { out += await s.allTasks }
        return out
    }

    // MARK: launch

    /// At launch (setup screen shown): continue a download that was running.
    func resumeIfWanted() {
        log("DOWNLOAD config " + DownloadConfig.summary)
        if DownloadConfig.memoryTooLow {   // the set-up screen says so; nothing to download for
            log("DOWNLOAD not offered: this device's memory is below the model's minimum")
            Task { await refreshFree() }
        } else if wantDownload {
            Task { await start() }
        } else {
            Task { await refreshFree(); await loadManifestQuietly() }
        }
    }

    private func loadManifestQuietly() async {
        if manifest != nil { return }
        if let m = try? await fetchManifest() { setManifest(m) }
    }

    func refreshFree() async {
        freeBytes = Self.volumeFree()
    }

    static func volumeFree() -> Int64? {
        ensureTargetDir()
        let v = try? DownloadConfig.targetDir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey])
        if let i = v?.volumeAvailableCapacityForImportantUsage, i > 0 { return i }
        return v?.volumeAvailableCapacity.map { Int64($0) }
    }

    // MARK: manifest

    private func fetchManifest() async throws -> Manifest {
        let url = DownloadConfig.manifestURL
        let cached = DownloadConfig.stateDir.appendingPathComponent("manifest.json")
        do {
            var req = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 30)
            req.allowsCellularAccess = true   // a few KB; the files themselves follow the cellular setting
            let (data, resp) = try await URLSession.shared.data(for: req)
            if let h = resp as? HTTPURLResponse, !(200..<300).contains(h.statusCode) {
                throw NSError(domain: "download", code: h.statusCode,
                              userInfo: [NSLocalizedDescriptionKey: "manifest: HTTP \(h.statusCode) from \(url.absoluteString)"])
            }
            guard let m = try MovedStoreManifest.manifest(from: data, profile: DownloadConfig.profile) else {
                throw NSError(domain: "download", code: 1, userInfo: [NSLocalizedDescriptionKey: "manifest: no profile \(DownloadConfig.profile) in \(url.absoluteString)"])
            }
            try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
            try? data.write(to: cached, options: .atomic)
            log("DOWNLOAD manifest \(url.absoluteString): \(m.files.count) files, required \(m.requiredBytes) B, version \(m.version)")
            return m
        } catch {
            if let d = try? Data(contentsOf: cached),
               let m = (try? MovedStoreManifest.manifest(from: d, profile: DownloadConfig.profile)) ?? nil {
                log("DOWNLOAD manifest fetch failed (\(error.localizedDescription)); using the cached copy (version \(m.version))")
                return m
            }
            throw error
        }
    }

    /// The model folder exists and is excluded from backup.
    nonisolated static func ensureTargetDir() {
        var u = DownloadConfig.targetDir
        try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        var rv = URLResourceValues()
        rv.isExcludedFromBackup = true
        try? u.setResourceValues(rv)
    }

    /// Size of a regular file, following links (nil: absent or not a file).
    nonisolated static func resolvedSize(_ path: String) -> Int64? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: (path as NSString).resolvingSymlinksInPath),
              (a[.type] as? FileAttributeType) == .typeRegular else { return nil }
        return (a[.size] as? NSNumber)?.int64Value
    }

    /// Settings › Model files picked another model's files (one model at a time): the downloader's state is that model's
    /// from here (its folder, its verified-files record, its manifest). Refused while a download or check runs.
    @discardableResult
    func retarget(_ k: ModelKind) -> Bool {
        if k == DownloadConfig.kind { return true }
        guard !active, !verifying else { return false }
        DownloadConfig.target = k == ModelKind.selected ? nil : k
        UserDefaults.standard.set(k.rawValue, forKey: DownloadConfig.targetKey)
        files = []; index = [:]; manifest = nil; phase = .idle; verifyResult = nil; note = ""
        verified = Self.loadVerified()
        log("DOWNLOAD target: \(k.rawValue) (\(DownloadConfig.targetDir.path), repo \(DownloadConfig.repo), profile \(DownloadConfig.profile))")
        Task { await refreshFree(); await loadManifestQuietly() }
        return true
    }

    private func setManifest(_ m: Manifest) {
        manifest = m
        let sel = m.files.filter(\.required)
        // keep the progress of files already tracked (a restart after pause)
        let old = Dictionary(uniqueKeysWithValues: files.map { ($0.file.local, $0) })
        files = sel.map { f in
            var p = old[f.local] ?? FileProgress(file: f)
            p.file = f
            // a file with a matching verified record (size + mtime) is shown as done without hashing it again
            if old[f.local] == nil, isVerified(f, at: DownloadConfig.targetDir.appendingPathComponent(f.local)) {
                p.state = .done; p.bytes = f.size; p.shaOK = true; p.source = "already here"
            }
            return p
        }
        index = Dictionary(uniqueKeysWithValues: files.enumerated().map { ($1.file.local, $0) })
    }

    // MARK: start / pause / cancel

    func start() async {
        guard !active else { return }
        Self.ensureTargetDir()
        pausing = false
        cancelling = false
        note = ""
        // the user asked for the model: a relaunch, a reconnect or freed space continues by itself until it is done,
        // paused or cancelled
        wantDownload = true
        phase = .fetchingManifest
        let m: Manifest
        do { m = try await fetchManifest() } catch {
            log("DOWNLOAD failed manifest: \(error.localizedDescription)")
            phase = .offline   // no dialog: retried when the network comes back, every 15 s, and when the app is opened
            scheduleAutoRetry(15)
            return
        }
        setManifest(m)
        log("DOWNLOAD plan \(files.count) file(s), \(totalBytes) B")

        // 1. what is already here: verified record, or a matching size to hash; leftovers in .download to finish
        phase = .checking
        let live = await allTasks()
        let liveLocals = Set(live.compactMap { $0.taskDescription })
        for i in files.indices where files[i].state != .done {
            let f = files[i].file
            if liveLocals.contains(f.local) { continue }   // being downloaded by the session (adopted below)
            let dst = DownloadConfig.targetDir.appendingPathComponent(f.local)
            let part = partURL(f)
            if let sz = Self.size(dst.path), sz == f.size {
                if isVerified(f, at: dst) {
                    files[i].state = .done; files[i].bytes = f.size; files[i].source = "already here"; files[i].shaOK = true
                    continue
                }
                if await verify(i, at: dst, moveTo: nil) { files[i].source = "already here"; continue }
            } else if let sz = Self.size(part.path), sz == f.size {
                if await verify(i, at: part, moveTo: dst) { files[i].source = "network (earlier launch)"; continue }
            }
            files[i].state = .queued
        }
        if cancelling { return }

        // 2. free space: what is left + 1 GB
        let need = files.filter { $0.state != .done }.reduce(Int64(0)) { $0 + $1.file.size }
        let free = Self.volumeFree()
        freeBytes = free
        if need > 0, let free, need + DownloadConfig.headroomBytes > free {
            let msg = String(format: "Not enough space: the download needs %.2f GB (%.2f GB of files + 1 GB to spare) and this device has %.2f GB free. Free up %.2f GB (Settings > General > iPhone Storage) and try again.",
                             Double(need + DownloadConfig.headroomBytes) / 1e9, Double(need) / 1e9, Double(free) / 1e9,
                             Double(need + DownloadConfig.headroomBytes - free) / 1e9)
            phase = .refused(msg)
            log("DOWNLOAD refused need=\(need) headroom=\(DownloadConfig.headroomBytes) free=\(free)")
            scheduleAutoRetry(20)   // starts by itself once the space is there
            return
        }
        log("DOWNLOAD space need=\(need) + headroom \(DownloadConfig.headroomBytes) <= free \(free.map(String.init) ?? "unknown")")

        // 3. tasks: adopt the session's live ones, resume from resume data, or start fresh
        wantDownload = true
        phase = .downloading
        startedAt = startedAt ?? Date()
        startTimer()
        for t in live {
            guard let local = t.taskDescription, let i = index[local], let dt = t as? URLSessionDownloadTask else {
                t.cancel(); continue
            }
            if files[i].state == .done { t.cancel(); continue }
            if t.state == .suspended { t.resume() }
            tasks[local] = dt
            files[i].state = .downloading
            files[i].bytes = t.countOfBytesReceived
            files[i].resumedFrom = t.countOfBytesReceived
            files[i].source = "adopted"
            files[i].started = files[i].started ?? Date()
            log("DOWNLOAD adopt \(local) task \(t.taskIdentifier) at \(t.countOfBytesReceived) B (state \(t.state.rawValue))")
        }
        for i in files.indices where files[i].state == .queued || files[i].state == .paused { launch(i) }
        checkAllDone()
    }

    // MARK: recovering by itself

    /// The app came to the front (SetupScreen): pick up a download that waits for space or a connection, or failed.
    func appBecameActive() {
        Task { await refreshFree() }
        retryIfWanted("app opened")
    }

    private func networkChanged(_ st: NetworkState) {
        let was = network
        network = st
        if was != st { log("DOWNLOAD network \(st)") }
        if st != .none && was != st { retryIfWanted("network back") }
    }

    private func scheduleAutoRetry(_ seconds: Double) {
        autoRetry?.invalidate()
        autoRetry = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) { [weak self] _ in
            Task { @MainActor in self?.retryIfWanted("timer") }
        }
    }

    /// Start again when the user wanted the download and it stopped for a reason that may have gone away.
    private func retryIfWanted(_ why: String) {
        guard wantDownload, !SetupGate.shared.needsRestart else { return }
        switch phase {
        case .offline:
            log("DOWNLOAD retry (\(why)): waiting for the file list")
            Task { await start() }
        case .refused:
            let need = remainingBytes + DownloadConfig.headroomBytes
            if let f = Self.volumeFree(), f >= need {
                log("DOWNLOAD retry (\(why)): space is there now")
                Task { await start() }
            } else { scheduleAutoRetry(20) }
        case .failed:
            log("DOWNLOAD retry (\(why)): failed files")
            Task { await retryFailed() }
        default:
            break
        }
    }

    private func launch(_ i: Int) {
        let f = files[i].file
        guard tasks[f.local] == nil else { return }
        let s = session(cellular: cellular)
        let task: URLSessionDownloadTask
        let rd = resumeURL(f)
        if let data = try? Data(contentsOf: rd) {
            try? FileManager.default.removeItem(at: rd)
            task = s.downloadTask(withResumeData: data)
            let off = Self.resumeOffset(data)
            files[i].resumedFrom = off
            files[i].bytes = off
            files[i].source = "resumed"
            log("DOWNLOAD resume \(f.local) from resume data at \(off) B")
        } else {
            var req = URLRequest(url: DownloadConfig.baseURL.appendingPathComponent(f.path))
            req.timeoutInterval = 120
            task = s.downloadTask(with: req)
            files[i].bytes = 0
            files[i].source = "network"
            log("DOWNLOAD start \(f.local) \(f.size) B <- \(req.url!.absoluteString)")
        }
        task.taskDescription = f.local
        task.countOfBytesClientExpectsToReceive = f.size
        tasks[f.local] = task
        files[i].state = .queued   // becomes .downloading at the first bytes
        files[i].started = nil
        task.resume()
    }

    func pause() async {
        guard phase == .downloading else { return }
        pausing = true
        wantDownload = false
        for t in await allTasks() {
            guard let dt = t as? URLSessionDownloadTask, let local = dt.taskDescription else { t.cancel(); continue }
            let data = await withCheckedContinuation { (c: CheckedContinuation<Data?, Never>) in dt.cancel { c.resume(returning: $0) } }
            if let data, let i = index[local] {
                try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
                try? data.write(to: resumeURL(files[i].file), options: .atomic)
                log("DOWNLOAD paused \(local) at \(Self.resumeOffset(data)) B (resume data saved)")
            }
        }
        tasks.removeAll()
        for i in files.indices where files[i].state != .done { files[i].state = .paused }
        phase = .paused
        bytesPerSecond = 0
        stopTimer()
    }

    /// Stop and throw away partial files and resume data; finished, verified files stay.
    func cancel() async {
        cancelling = true
        wantDownload = false
        for t in await allTasks() { t.cancel() }
        tasks.removeAll()
        for f in files {
            try? FileManager.default.removeItem(at: partURL(f.file))
            try? FileManager.default.removeItem(at: resumeURL(f.file))
        }
        for i in files.indices where files[i].state != .done { files[i].state = .queued; files[i].bytes = 0 }
        phase = .idle
        bytesPerSecond = 0
        stopTimer()
        log("DOWNLOAD cancelled")
    }

    private func switchSession() async {
        log("DOWNLOAD cellular -> \(cellular): moving the transfers to the other session")
        await pause()
        await start()
    }

    // MARK: delegate events (main actor)

    fileprivate func didWrite(_ local: String, written: Int64, total: Int64) {
        guard let i = index[local] else { return }
        if files[i].state == .queued || files[i].state == .paused { files[i].state = .downloading }
        if files[i].started == nil { files[i].started = Date() }
        files[i].bytes = total
        files[i].downloadedBytes += written
        sessionBytes += written
    }

    fileprivate func didResume(_ local: String, offset: Int64) {
        guard let i = index[local] else { return }
        files[i].resumedFrom = offset
        files[i].bytes = offset
        log("DOWNLOAD resumed \(local) at offset \(offset) B (HTTP Range)")
    }

    fileprivate func didFinish(_ local: String, taskID: Int, outcome: SessionDelegate.Outcome, error: Error?, resumeData: Data?) {
        // an event of an older task for a file that has a newer one running (a force-quit's cancellations are delivered
        // at the next launch, possibly after start() launched the file again): ignore it, so the file is not fetched twice
        if let cur = tasks[local], cur.taskIdentifier != taskID {
            log("DOWNLOAD \(local): ignored the end of an older task \(taskID) (task \(cur.taskIdentifier) is running)")
            return
        }
        tasks[local] = nil
        // resume data first, even for an event that arrives before start() has the file list (a relaunch)
        if let resumeData, !cancelling {
            try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
            try? resumeData.write(to: DownloadConfig.stateDir.appendingPathComponent(local.replacingOccurrences(of: "/", with: "__") + ".resume"),
                                  options: .atomic)
            log("DOWNLOAD \(local): task ended with resume data at \(Self.resumeOffset(resumeData)) B (saved)")
        }
        guard let i = index[local] else {
            log("DOWNLOAD event for \(local) before the file list (kept: .part or resume data)")
            return
        }
        if cancelling { return }
        if let error {
            let ns = error as NSError
            if pausing && ns.code == NSURLErrorCancelled { return }
            retry(i, reason: (resumeData != nil ? "network error, resumable: " : "network error: ") + ns.localizedDescription)
            return
        }
        switch outcome {
        case .moved:
            Task {
                let ok = await verify(i, at: partURL(files[i].file), moveTo: DownloadConfig.targetDir.appendingPathComponent(files[i].file.local))
                if !ok { retry(i, reason: "sha256 mismatch") }
                checkAllDone()
            }
        case .badStatus(let code):
            retry(i, reason: "HTTP \(code)")
        case .moveFailed(let why):
            retry(i, reason: "could not store the file: \(why)")
        case .none:
            retry(i, reason: "no file")
        }
    }

    private func retry(_ i: Int, reason: String) {
        files[i].attempts += 1
        let a = files[i].attempts
        if a >= DownloadConfig.maxAttempts {
            files[i].state = .failed(reason)
            log("DOWNLOAD failed \(files[i].file.local) after \(a) attempts: \(reason)")
            checkAllDone()
            return
        }
        let d = DownloadConfig.backoff(attempt: a)
        files[i].state = .waiting(reason)
        log("DOWNLOAD retry \(files[i].file.local) in \(Int(d)) s (attempt \(a + 1) of \(DownloadConfig.maxAttempts)): \(reason)")
        let local = files[i].file.local
        DispatchQueue.main.asyncAfter(deadline: .now() + d) { [weak self] in
            guard let self, self.phase == .downloading, let j = self.index[local], case .waiting = self.files[j].state else { return }
            self.launch(j)
        }
    }

    /// Retry the failed files (the screen's button).
    func retryFailed() async {
        for i in files.indices { if case .failed = files[i].state { files[i].state = .queued; files[i].attempts = 0 } }
        if phase == .downloading { for i in files.indices where files[i].state == .queued { launch(i) } } else { await start() }
    }

    fileprivate func sessionEventsFinished() {
        log("DOWNLOAD background session events delivered")
        backgroundCompletion?()
        backgroundCompletion = nil
    }

    private func checkAllDone() {
        guard phase == .downloading, !files.isEmpty else { return }
        func isFailed(_ p: FileProgress) -> Bool { if case .failed = p.state { return true }; return false }
        if files.contains(where: { $0.state != .done && !isFailed($0) }) { return }
        stopTimer()
        bytesPerSecond = 0
        let failed = files.filter(isFailed)
        if !failed.isEmpty {
            phase = .failed("\(failed.count) file(s) could not be downloaded; tap Retry.")
            log("DOWNLOAD FAILED \(failed.count) file(s): " + failed.map { $0.file.local }.joined(separator: ", "))
            scheduleAutoRetry(60)
            return
        }
        // the manifest next to the files it describes: the engine reads it as moved-store.json
        if !files.contains(where: { $0.file.local == "moved-store.json" }),
           let d = try? Data(contentsOf: DownloadConfig.stateDir.appendingPathComponent("manifest.json")) {
            try? d.write(to: DownloadConfig.targetDir.appendingPathComponent("moved-store.json"), options: .atomic)
        }
        phase = .complete
        wantDownload = false
        let dt = startedAt.map { Date().timeIntervalSince($0) } ?? 0
        log(String(format: "DOWNLOAD DONE %d file(s), %lld B; %lld B over the network this launch in %.1f s (%.1f MB/s)",
                   files.count, totalBytes, sessionBytes, dt, dt > 0 ? Double(sessionBytes) / 1e6 / dt : 0))
        let missing = DownloadConfig.installedProblems
        log("DOWNLOAD store check: \(missing.isEmpty ? "complete" : "incomplete, \(missing.count) problem(s): \(missing.prefix(5).joined(separator: ", "))")")
        onComplete?()
    }

    // MARK: verification

    private func isVerified(_ f: ManifestFile, at url: URL) -> Bool {
        guard let r = verified[f.local], r.sha256 == f.sha256, r.size == f.size,
              let m = Self.mtime(url.path) else { return false }
        return abs(m - r.mtime) < 1
    }

    /// Hash `url` (streaming) against the manifest; on a match move it to `moveTo` (if given) and record it.
    private func verify(_ i: Int, at url: URL, moveTo: URL?) async -> Bool {
        let f = files[i].file
        files[i].state = .verifying
        files[i].bytes = 0
        let t0 = Date()
        let sha = await Self.sha256(url) { [weak self] n in
            Task { @MainActor in
                if let self, let j = self.index[f.local], self.files[j].state == .verifying { self.files[j].bytes = n }
            }
        }
        let dt = Date().timeIntervalSince(t0)
        guard let sha, sha == f.sha256 else {
            log("DOWNLOAD sha256 MISMATCH \(f.local): got \(sha ?? "unreadable"), want \(f.sha256)")
            try? FileManager.default.removeItem(at: url)
            files[i].shaOK = false
            files[i].state = .queued
            files[i].bytes = 0
            return false
        }
        var final = url
        if let dst = moveTo {
            let fm = FileManager.default
            try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
            try? fm.removeItem(at: dst)
            do { try fm.moveItem(at: url, to: dst) } catch {
                log("DOWNLOAD could not move \(f.local) into place: \(error.localizedDescription)")
                files[i].state = .queued
                return false
            }
            final = dst
        }
        if let m = Self.mtime(final.path) {
            verified[f.local] = VerifiedRecord(size: f.size, mtime: m, sha256: f.sha256)
            saveVerified()
        }
        try? FileManager.default.removeItem(at: resumeURL(f))
        files[i].state = .done
        files[i].bytes = f.size
        files[i].shaOK = true
        files[i].finished = Date()
        let net = files[i].started.map { files[i].finished!.timeIntervalSince($0) }
        log(String(format: "DOWNLOAD verified %@ %lld B sha256 ok (hash %.2f s, %.0f MB/s)%@", f.local, f.size, dt,
                   dt > 0 ? Double(f.size) / 1e6 / dt : 0,
                   net.map { String(format: "; transfer %.1f s, %.1f MB/s, resumed from %lld B", $0,
                                    $0 > 0 ? Double(files[i].downloadedBytes) / 1e6 / $0 : 0, files[i].resumedFrom) } ?? ""))
        return true
    }

    nonisolated static func sha256(_ url: URL, progress: @escaping @Sendable (Int64) -> Void) async -> String? {
        await Task.detached(priority: .utility) { () -> String? in
            guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? h.close() }
            var hasher = SHA256()
            var n: Int64 = 0
            var lastReport: Int64 = 0
            while true {
                let chunk: Data? = autoreleasepool { try? h.read(upToCount: 4 << 20) }
                guard let c = chunk, !c.isEmpty else { break }
                hasher.update(data: c)
                n += Int64(c.count)
                if n - lastReport >= 64 << 20 { progress(n); lastReport = n }
            }
            progress(n)
            return hasher.finalize().map { String(format: "%02x", $0) }.joined()
        }.value
    }

    /// Settings > Model files > Verify files: hash every required file in the folder against the manifest.
    func verifyAll() async {
        guard !verifying, !active else { return }
        verifying = true
        verifyResult = "Getting the file list..."
        defer { verifying = false }
        guard let m = try? await fetchManifest() else { verifyResult = "Could not get the file list (offline?)."; return }
        if manifest == nil || files.isEmpty { setManifest(m) }
        var bad: [String] = [], missing: [String] = [], ok = 0
        let req = m.files.filter(\.required)
        for (n, f) in req.enumerated() {
            verifyResult = "Checking \(n + 1) of \(req.count): \(f.local)"
            let url = DownloadConfig.targetDir.appendingPathComponent(f.local)
            guard let sz = Self.resolvedSize(url.path) else { missing.append(f.local); continue }
            if sz != f.size { bad.append(f.local); continue }
            let sha = await Self.sha256(url) { _ in }
            if sha == f.sha256 {
                ok += 1
                if let mt = Self.mtime(url.path) { verified[f.local] = VerifiedRecord(size: f.size, mtime: mt, sha256: f.sha256) }
            } else {
                bad.append(f.local)
                verified[f.local] = nil
            }
        }
        saveVerified()
        var s = "\(ok) of \(req.count) files match the manifest."
        if !bad.isEmpty { s += " Damaged: " + bad.joined(separator: ", ") + "." }
        if !missing.isEmpty { s += " Missing: " + missing.joined(separator: ", ") + "." }
        if !bad.isEmpty || !missing.isEmpty { s += " Tap Download to repair them." }
        verifyResult = s
        log("DOWNLOAD verify-all ok=\(ok) bad=\(bad) missing=\(missing)")
        // damaged files are removed so the next download replaces them (and the store check stops passing them)
        for b in bad { try? FileManager.default.removeItem(at: DownloadConfig.targetDir.appendingPathComponent(b)) }
    }

    /// Settings > Model files > Delete model: every file of the model folder and the engine's saved states for this model.
    /// `k`: the model whose folder is deleted (default: the one this screen shows); the other model's files and the chats
    /// are never touched.
    func deleteModel(_ k: ModelKind? = nil) async {
        if let k, k != DownloadConfig.kind {
            guard retarget(k) else { note = "A download is running; try again when it ends."; return }
        }
        await cancel()
        let fm = FileManager.default
        for name in (try? fm.contentsOfDirectory(atPath: DownloadConfig.targetDir.path)) ?? [] {
            try? fm.removeItem(at: DownloadConfig.targetDir.appendingPathComponent(name))
        }
        try? fm.removeItem(at: DmengLaunch.prefixDir(DownloadConfig.kind))  // the engine's saved states of this model
        verified = [:]
        Self.ensureTargetDir()
        files = []
        index = [:]
        manifest = nil
        phase = .idle
        verifyResult = nil
        cancelling = false
        freeBytes = Self.volumeFree()
        log("DOWNLOAD model deleted: \(DownloadConfig.targetDir.path)")
        await loadManifestQuietly()
    }

    // MARK: speed

    private func startTimer() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func stopTimer() { timer?.invalidate(); timer = nil }

    private func tick() {
        let now = Date()
        speedSamples.append((now, sessionBytes))
        speedSamples.removeAll { now.timeIntervalSince($0.0) > 8 }
        if let first = speedSamples.first, now.timeIntervalSince(first.0) > 0.9 {
            bytesPerSecond = Double(sessionBytes - first.1) / now.timeIntervalSince(first.0)
        }
        if now.timeIntervalSince(lastLog) >= 5 {
            lastLog = now
            let act = files.filter { $0.state == .downloading }.map { String(format: "%@ %.0f%%", $0.file.name, 100 * $0.fraction) }
            log(String(format: "DOWNLOAD progress %.3f of %.3f GB (%.1f%%), %.1f MB/s, eta %@; active: %@", Double(doneBytes) / 1e9,
                       Double(totalBytes) / 1e9, 100 * fraction, bytesPerSecond / 1e6, etaSeconds.map { formatDuration($0) } ?? "-",
                       act.joined(separator: ", ")))
        }
    }

    // MARK: files

    func partURL(_ f: ManifestFile) -> URL { DownloadConfig.stateDir.appendingPathComponent(f.key + ".part") }
    func resumeURL(_ f: ManifestFile) -> URL { DownloadConfig.stateDir.appendingPathComponent(f.key + ".resume") }

    nonisolated static func size(_ path: String) -> Int64? {
        guard let a = try? FileManager.default.attributesOfItem(atPath: path), (a[.type] as? FileAttributeType) == .typeRegular
        else { return nil }
        return (a[.size] as? NSNumber)?.int64Value
    }

    nonisolated static func mtime(_ path: String) -> Double? {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.modificationDate] as? Date)?.timeIntervalSince1970
    }

    /// The byte offset a resume data blob continues from (its plist's NSURLSessionResumeBytesReceived), for the log.
    nonisolated static func resumeOffset(_ data: Data) -> Int64 {
        let key = "NSURLSessionResumeBytesReceived"
        // older systems: a plain plist dictionary; current ones: a keyed archive of one
        if let p = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
           let n = p[key] as? NSNumber { return n.int64Value }
        if let u = try? NSKeyedUnarchiver(forReadingFrom: data) {
            u.requiresSecureCoding = false
            defer { u.finishDecoding() }
            // CFNetwork archives under the literal key "NSKeyedArchiveRootObjectKey" (Foundation's constant is "root")
            for root in ["NSKeyedArchiveRootObjectKey", NSKeyedArchiveRootObjectKey] {
                if let d = u.decodeObject(forKey: root) as? [String: Any], let n = d[key] as? NSNumber { return n.int64Value }
            }
        }
        return -1
    }

    private static var verifiedURL: URL { DownloadConfig.stateDir.appendingPathComponent("verified.json") }
    private static func loadVerified() -> [String: VerifiedRecord] {
        guard let d = try? Data(contentsOf: verifiedURL) else { return [:] }
        return (try? JSONDecoder().decode([String: VerifiedRecord].self, from: d)) ?? [:]
    }
    private func saveVerified() {
        try? FileManager.default.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
        if let d = try? JSONEncoder().encode(verified) { try? d.write(to: Self.verifiedURL, options: .atomic) }
    }
}

func formatDuration(_ s: Double) -> String {
    if s < 90 { return "\(Int(s.rounded())) s" }
    if s < 5400 { return "\(Int((s / 60).rounded())) min" }
    return String(format: "%.1f h", s / 3600)
}

/// The URLSession delegate. Runs on the session's serial queue; the finished file must be moved before
/// didFinishDownloadingTo returns, so that happens here; everything else goes to the main actor.
final class SessionDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    enum Outcome { case none, moved, badStatus(Int), moveFailed(String) }
    weak var owner: ModelDownloader?
    private var outcomes: [Int: Outcome] = [:]   // task id -> outcome (delegate queue only)
    private var lastWrite: [Int: Date] = [:]

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let local = downloadTask.taskDescription else { return }
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else { outcomes[downloadTask.taskIdentifier] = .badStatus(status); return }
        let part = DownloadConfig.stateDir.appendingPathComponent(local.replacingOccurrences(of: "/", with: "__") + ".part")
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: DownloadConfig.stateDir, withIntermediateDirectories: true)
            try? fm.removeItem(at: part)
            try fm.moveItem(at: location, to: part)
            outcomes[downloadTask.taskIdentifier] = .moved
        } catch {
            outcomes[downloadTask.taskIdentifier] = .moveFailed(error.localizedDescription)
        }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let local = downloadTask.taskDescription else { return }
        Task { @MainActor [weak owner] in owner?.didWrite(local, written: bytesWritten, total: totalBytesWritten) }
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didResumeAtOffset fileOffset: Int64,
                    expectedTotalBytes: Int64) {
        guard let local = downloadTask.taskDescription else { return }
        Task { @MainActor [weak owner] in owner?.didResume(local, offset: fileOffset) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let local = task.taskDescription else { return }
        let outcome = outcomes.removeValue(forKey: task.taskIdentifier) ?? Outcome.none
        let resume = (error as NSError?)?.userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        let id = task.taskIdentifier
        Task { @MainActor [weak owner] in owner?.didFinish(local, taskID: id, outcome: outcome, error: error, resumeData: resume) }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor [weak owner] in owner?.sessionEventsFinished() }
    }
}

/// A model repo's moved-store.json read as the downloader's manifest: `profiles.<name>.parts` ({path, bytes, sha256}
/// objects) become the required files, kept at their repo paths under the model folder (store/, derived/,
/// embeddings.safetensors, the small files); `profiles.<name>.bytes` is the total shown on the set-up screen. nil when the
/// data is not a moved-store manifest.
enum MovedStoreManifest {
    static func manifest(from data: Data, profile: String) throws -> Manifest? {
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any], j["format"] as? String == "moved-store",
              let profiles = j["profiles"] as? [String: Any] else { return nil }
        guard let p = (profiles[profile] ?? profiles["phone-small"]) as? [String: Any], let parts = p["parts"] as? [[String: Any]] else {
            throw NSError(domain: "download", code: 1, userInfo: [NSLocalizedDescriptionKey: "moved-store.json: no profile \(profile)"])
        }
        var files: [ManifestFile] = []
        for x in parts {
            guard let path = x["path"] as? String, let sha = x["sha256"] as? String,
                  let n = (x["bytes"] as? NSNumber)?.int64Value else { continue }
            files.append(ManifestFile(path: path, local: path, size: n, sha256: sha, required: true,
                                      kind: path.hasPrefix("store/") ? "store" : (path.hasPrefix("derived/") ? "derived" : "small")))
        }
        let total = (p["bytes"] as? NSNumber)?.int64Value ?? files.reduce(0) { $0 + $1.size }
        let gen = (j["generated"] as? [String: Any])?["date"] as? String ?? "?"
        let lic = ((j["source"] as? [String: Any])?["license"] as? String) ?? "apache-2.0"
        log("DOWNLOAD moved-store.json: profile \(profile) (\((p["precision"] as? String) ?? "?"), default mode \((p["mode"] as? String) ?? "?")): \(files.count) parts, \(total) B")
        return Manifest(format: "moved-store/\(profile)", version: gen, model: (j["name"] as? String) ?? "moved-store",
                        license: lic, repo: DownloadConfig.repo, requiredBytes: total, files: files)
    }
}
