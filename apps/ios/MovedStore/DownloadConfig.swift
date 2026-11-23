import Foundation
import os

/// Where the app downloads a model from, and the downloader's knobs.
///
/// Each model's Hub repo holds `moved-store.json` (the manifest) and the files it lists. The app fetches the manifest,
/// picks the device's profile (`phone` with at least 5.0 GB available to the app at launch, `phone-small` below), and
/// downloads exactly `profiles.<name>.parts` (objects {path, bytes, sha256}) into Documents/<model folder>/ at their repo
/// paths, each checked against its SHA-256. A file's URL is the manifest URL's folder + the file's `path`.
enum DownloadConfig {
    /// The model whose files the downloader works on (one at a time): the one Settings > Model files shows
    /// (ModelDownloader.retarget), else the selected model. A download that was running when the app closed continues
    /// for its model at the next launch.
    static let targetKey = "download.model"
    static var target: ModelKind? = {
        guard UserDefaults.standard.bool(forKey: "download.want") else { return nil }
        return UserDefaults.standard.string(forKey: targetKey).flatMap(ModelKind.init(rawValue:))
    }()
    static var kind: ModelKind { target ?? ModelKind.selected }
    static var repo: String { kind.hubRepo }
    static let revision = "main"
    static var manifestURL: URL { URL(string: "https://huggingface.co/\(repo)/resolve/\(revision)/moved-store.json")! }
    /// The download's folder: Documents/<the model's folder>.
    static var targetDir: URL { kind.folder }
    /// The profile whose parts are downloaded: the device's.
    static var profile: String { DmengLaunch.profile }
    /// What is missing in the installed folder (empty: complete).
    static var installedProblems: [String] { DmengLaunch.completeness(targetDir, kind: kind) }
    /// The folder the manifest's `path`s are relative to.
    static var baseURL: URL { manifestURL.deletingLastPathComponent() }
    static var repoPageURL: URL { URL(string: "https://huggingface.co/\(repo)")! }

    /// Shown before the manifest has been fetched: the profile's bytes.
    static var approxRequiredBytes: Int64 { kind.approxBytes(profile: profile) }
    /// Free space the volume must keep after the download: remaining bytes + this, or the start is refused.
    static let headroomBytes: Int64 = 1_000_000_000
    /// Files transferred at once (URLSessionConfiguration.httpMaximumConnectionsPerHost; the hub's CDN is one host).
    static let concurrentFiles = 3
    /// Retries per file (network errors, bad status, sha256 mismatch) before it is marked failed; backoff 2, 4, 8 ... 60 s.
    static let maxAttempts = 8
    static func backoff(attempt: Int) -> Double { min(60, pow(2, Double(max(1, attempt)))) }

    /// This iPhone gives the app too little memory to run a model (the 4-bit profile runs from about 4 GB available), so
    /// the set-up screen says so instead of offering the download. iOS only (os_proc_available_memory is 0 on the Mac and
    /// in the Simulator).
    static var memoryTooLow: Bool {
        let avail = Double(os_proc_available_memory())
        return avail > 0 && avail < 3_000_000_000
    }

    /// The background URLSession identifiers, derived from the bundle id.
    static var backgroundSessionID: String { (Bundle.main.bundleIdentifier ?? "app") + ".model-download" }
    static var backgroundCellularSessionID: String { backgroundSessionID + ".cellular" }

    static let cellularKey = "download.cellular"
    static var cellularAllowed: Bool { UserDefaults.standard.bool(forKey: cellularKey) }

    /// Download bookkeeping inside the model folder (excluded from backup with it): partial files, resume data, the
    /// cached manifest, the verified-files record.
    static var stateDir: URL { targetDir.appendingPathComponent(".download", isDirectory: true) }

    static var summary: String {
        "manifest=\(manifestURL.absoluteString) cellular=\(cellularAllowed) dir=\(targetDir.lastPathComponent) profile=\(profile) model=\(kind.rawValue)"
    }
}
