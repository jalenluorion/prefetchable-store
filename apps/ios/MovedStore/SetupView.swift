import SwiftUI
import UIKit

/// Whether the app shows the first-run setup (download) screen instead of the chat. Decided at launch, and again when a
/// download completes or the model is deleted.
@MainActor
final class SetupGate: ObservableObject {
    static let shared = SetupGate()
    @Published var showSetup: Bool
    /// The chat (and with it an engine) was created in this process: after Delete + a new download the app asks for a
    /// restart instead of loading a second engine next to the first.
    static var engineCreated = false
    @Published var needsRestart = false

    private init() {
        ModelKind.resolveAtLaunch()  // a selected model whose files are gone gives way to the other one when it is complete
        showSetup = Self.needsSetup()
    }

    /// The selected model's files are not all here (a download of the other model, from Model files, never blocks the
    /// chat).
    static func needsSetup() -> Bool { !ModelKind.selected.isComplete }

    /// The download finished while the app was not in front (the usual case: the phone locked for the 20-30 minutes):
    /// the chat, and with it the engine load, waits until the app is opened.
    private var pendingOpen = false

    func downloadFinished() {
        if DownloadConfig.kind != ModelKind.selected {  // the other model, from Model files
            log("SETUP \(DownloadConfig.kind.displayName) downloaded: it can be chosen in Settings › Model")
            return
        }
        guard DownloadConfig.installedProblems.isEmpty else { return }
        // The chat (and its engine) was created in this process before the model was deleted from Settings: the app asks
        // for a relaunch instead of creating a second chat beside it. A first run never gets here with an engine.
        if Self.engineCreated { needsRestart = true; return }
        #if !targetEnvironment(macCatalyst)
        if UIApplication.shared.applicationState != .active {
            pendingOpen = true
            log("SETUP store complete in the background: the chat opens when the app is opened")
            return
        }
        #endif
        log("SETUP store complete: opening the chat")
        showSetup = false
    }

    /// The app came to the front: open the chat a background completion left waiting.
    func appBecameActive() {
        guard pendingOpen else { return }
        pendingOpen = false
        downloadFinished()
    }

    func modelDeleted() { showSetup = true }
}

/// Strings shared by several screens.
enum Copy {
    static let lowMemory = "This iPhone doesn't have enough memory to run the model."
    static var modelName: String { ModelKind.selected.displayName }
}

/// The setup screen: first launch (full screen) and Settings > Model files (`manage`: also Check files and Delete).
struct SetupView: View {
    enum Mode { case firstRun, manage }
    var mode: Mode
    /// Called before the model is deleted (the chat stops a reply in progress).
    var beforeDelete: (() -> Void)? = nil
    /// Called after the selected model was deleted (the chat switches to the other model when it is complete); nil: the
    /// set-up screen.
    var afterDelete: (() -> Void)? = nil
    @EnvironmentObject var d: ModelDownloader
    @ObservedObject var gate = SetupGate.shared
    @State private var confirmDelete = false
    @State private var confirmCancel = false
    @State private var showFiles = false

    var body: some View {
        Group {
            if mode == .firstRun {
                NavigationStack { form.navigationTitle("").navigationBarTitleDisplayMode(.inline) }
            } else {
                form.navigationTitle("Model files").navigationBarTitleDisplayMode(.inline)
            }
        }
        .task { await d.refreshFree() }
    }

    var form: some View {
        Form {
            if mode == .firstRun { intro }
            if mode == .firstRun && !lowMemory && !started { firstRunPicker }
            if mode == .firstRun && lowMemory {
                Section {
                    ExpectationsView(lowMemory: true, grouped: true)
                }
                .listRowBackground(Color.clear)
                .listRowInsets(EdgeInsets())
            } else {
                if mode == .manage || started || spaceShort { progressSection }
                if mode == .firstRun { primarySection }
            }
            if mode == .manage { modelPicker }
            if mode == .manage { manageSection; filesSection }
            aboutSection
        }
    }

    var lowMemory: Bool { DownloadConfig.memoryTooLow }

    /// Not enough free space: refused, or (before the first tap) the free space already short of the download.
    var spaceShort: Bool {
        if case .refused = d.phase { return true }
        guard d.phase == .idle, let f = d.freeBytes else { return false }
        return f < d.remainingBytes + DownloadConfig.headroomBytes
    }

    /// Anything happened yet (the first screen shows only the intro and the button until then).
    var started: Bool {
        if case .idle = d.phase { return d.doneCount > 0 }
        return true
    }

    // MARK: sections

    var intro: some View {
        Section {
            VStack(alignment: .leading, spacing: 14) {
                Image(systemName: "iphone.gen3.radiowaves.left.and.right")
                    .font(.system(.largeTitle)).imageScale(.large).foregroundStyle(Color.accentColor)
                    .accessibilityHidden(true)
                Text("An AI assistant that lives on your iPhone").font(.title2.bold())
                if !lowMemory { VStack(alignment: .leading, spacing: 8) {
                    Text("Download the model once, \(gbs(d.totalBytes)) over Wi-Fi.")
                    Text("Keep your iPhone plugged in. You can leave the app while it downloads.")
                    Text("After that, it works without a connection.")
                }
                .font(.body).foregroundStyle(.secondary) }
            }
            .padding(.vertical, 8)
            .listRowBackground(Color.clear)
        }
    }

    @ViewBuilder var progressSection: some View {
        Section {
            switch spaceShort ? .refused("") : d.phase {
            case .refused:
                Label { VStack(alignment: .leading, spacing: 4) {
                    Text("Not enough storage").font(.headline)
                    Text(spaceLine)
                    Text("Free up space in Settings › General › iPhone Storage, then come back.").foregroundStyle(.secondary)
                } } icon: { Image(systemName: "externaldrive.badge.xmark").foregroundStyle(.orange) }
            case .failed:
                Label { VStack(alignment: .leading, spacing: 4) {
                    Text("Download interrupted").font(.headline)
                    Text("Check your connection and try again.").foregroundStyle(.secondary)
                } } icon: { Image(systemName: "wifi.exclamationmark").foregroundStyle(.orange) }
            case .offline:
                Label { VStack(alignment: .leading, spacing: 4) {
                    Text("Waiting for a connection").font(.headline)
                    Text("The download starts by itself when you're online.").foregroundStyle(.secondary)
                } } icon: { Image(systemName: "wifi.slash").foregroundStyle(.secondary) }
            case .complete:
                Label(gate.needsRestart ? "Downloaded. Close the app and open it again to finish."
                                        : (mode == .manage ? "Model downloaded" : "Downloaded. Getting ready…"),
                      systemImage: "checkmark.circle.fill").foregroundStyle(.green)
            default:
                EmptyView()
            }
            if d.doneBytes > 0 || d.active || d.phase == .paused || mode == .manage {
                VStack(alignment: .leading, spacing: 8) {
                    ProgressView(value: d.fraction)
                    Text(progressLine).font(.callout).monospacedDigit()
                    if !statusWord.isEmpty { Text(statusWord).font(.footnote).foregroundStyle(.secondary) }
                }
                .padding(.vertical, 4)
            }
            if mode == .manage, let f = d.freeBytes { LabeledContent("Available", value: gbs(f)) }
        }
    }

    /// "4.2 of 18 GB · 23 MB/s · about 12 min left"
    var progressLine: String {
        var parts = [ofText(d.doneBytes, d.totalBytes)]
        if d.phase == .downloading && d.bytesPerSecond >= 1e5 && !reconnecting {
            parts.append(String(format: "%.0f MB/s", d.bytesPerSecond / 1e6))
            if let e = d.etaSeconds { parts.append(etaText(e)) }
        }
        return parts.joined(separator: " · ")
    }

    /// A file waits for its retry (the connection dropped).
    var reconnecting: Bool { d.files.contains { if case .waiting = $0.state { return true }; return false } }

    var statusWord: String {
        switch d.phase {
        case .idle: return d.doneCount > 0 ? "Paused" : ""
        case .fetchingManifest: return "Starting…"
        case .checking: return "Checking files…"
        case .downloading:
            if d.network == .none { return "Waiting for a connection" }
            if d.network == .cellularOnly && !d.cellular { return "Waiting for Wi-Fi" }
            if reconnecting { return "Reconnecting…" }
            return d.bytesPerSecond < 1e5 && d.doneBytes == 0 ? "Starting…" : "Downloading"
        case .paused: return "Paused"
        case .refused, .failed, .offline: return ""
        case .complete: return d.files.isEmpty ? "" : "All files checked"
        }
    }

    var spaceLine: String {
        let need = d.remainingBytes + DownloadConfig.headroomBytes
        return "The model needs \(gbs(need)). This iPhone has \(gbs(d.freeBytes ?? 0)) free."
    }

    var primarySection: some View {
        Section {
            switch d.phase {
            case .downloading, .fetchingManifest, .checking:
                Button { Task { await d.pause() } } label: { Text("Pause").frame(maxWidth: .infinity) }
                    .buttonStyle(.bordered).controlSize(.large)
                    .disabled(d.phase != .downloading)
            case .paused:
                Button { Task { await d.start() } } label: { Text("Resume").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
            case .failed:
                Button { Task { await d.retryFailed() } } label: { Text("Try again").frame(maxWidth: .infinity) }
                    .buttonStyle(.borderedProminent).controlSize(.large)
            case .complete:
                EmptyView()
            case .idle, .refused, .offline:
                Button { Task { await d.start() } } label: {
                    Text(spaceShort ? "Check again" : (d.doneCount > 0 ? "Resume" : (d.phase == .idle ? "Download" : "Try again"))).frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(gate.needsRestart)
            }
        } footer: {
            Toggle("Use cellular data", isOn: $d.cellular)
                .font(.footnote).foregroundStyle(.secondary)
                .padding(.top, 10)
        }
        .listRowBackground(Color.clear)
        .listRowInsets(EdgeInsets())
    }

    /// First run: which model to download (the other one can be added later in Settings › Model files).
    @State private var firstChoice = ModelKind.selected
    var firstRunPicker: some View {
        Section {
            Picker("Model", selection: Binding(get: { firstChoice }, set: { k in
                guard k != firstChoice, d.retarget(k) else { return }
                ModelKind.selected = k
                DownloadConfig.target = nil
                firstChoice = k
            })) {
                ForEach(ModelKind.allCases) { k in Text(k.displayName).tag(k) }
            }
            .pickerStyle(.segmented)
        } footer: {
            Text("You can add the other model later in Settings › Model files.")
        }
    }

    /// Model files: which model's files this screen shows (one at a time; ModelKind). Not while a download runs.
    @State private var shown = DownloadConfig.kind
    var modelPicker: some View {
        Section {
            Picker("Model", selection: Binding(get: { shown }, set: { k in if d.retarget(k) { shown = k } })) {
                ForEach(ModelKind.allCases) { k in Text(k.displayName).tag(k) }
            }
            .pickerStyle(.segmented)
            .disabled(d.active || d.verifying)
            LabeledContent(shown.isComplete ? "Downloaded" : "Not downloaded", value: gbs(d.totalBytes))
        } footer: {
            if shown != ModelKind.selected && shown.isComplete {
                Text("To use it, choose it under Settings › Model.")
            }
        }
    }

    var manageSection: some View {
        Section {
            switch d.phase {
            case .downloading, .fetchingManifest, .checking:
                Button("Pause") { Task { await d.pause() } }.disabled(d.phase != .downloading)
                Button("Cancel download", role: .destructive) { confirmCancel = true }
            case .paused, .failed, .idle, .refused, .offline:
                if d.doneCount < d.files.count || d.files.isEmpty {
                    Button(d.doneCount > 0 ? "Resume download" : "Download") { Task { await d.start() } }
                        .disabled(gate.needsRestart)
                }
            case .complete:
                EmptyView()
            }
            Button("Check files") { Task { await d.verifyAll() } }
                .disabled(d.verifying || d.active)
            if let r = d.verifyResult { Text(r).font(.footnote).foregroundStyle(.secondary) }
            Toggle("Use cellular data", isOn: $d.cellular)
            Button("Delete model", role: .destructive) { confirmDelete = true }
                .disabled(d.verifying)
        } footer: {
            Text("Check files makes sure nothing is damaged. It takes about a minute.")
        }
        .confirmationDialog("Cancel the download?", isPresented: $confirmCancel, titleVisibility: .visible) {
            Button("Cancel download", role: .destructive) { Task { await d.cancel() } }
            Button("Keep downloading", role: .cancel) {}
        } message: {
            Text("Finished files are kept.")
        }
        .confirmationDialog("Delete the model?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete model", role: .destructive) {
                let k = shown
                if k == ModelKind.selected { beforeDelete?() }
                Task {
                    await d.deleteModel(k)
                    if k == ModelKind.selected { (afterDelete ?? { gate.modelDeleted() })() }
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(ModelKind.selected.other.isComplete || shown != ModelKind.selected
                 ? "This frees about \(gbs(d.totalBytes)). Your chats are kept."
                 : "This frees about \(gbs(d.totalBytes)). Your chats are kept. To use the app again, you'll need to download the model again.")
        }
    }

    var filesSection: some View {
        Section {
            DisclosureGroup(isExpanded: $showFiles) {
                ForEach(d.files) { f in FileRow(f: f) }
            } label: {
                Text(d.files.isEmpty ? "Files" : "Files · \(d.doneCount) of \(d.files.count)")
            }
        }
    }

    var aboutSection: some View {
        Section {
            if mode == .manage {
                Link(destination: DownloadConfig.repoPageURL) { LabeledContent("Source", value: "Hugging Face") }
                LabeledContent("License", value: d.manifest?.license ?? "Apache-2.0")
            }
        } footer: {
            Text(DownloadConfig.kind.credit)
        }
    }
}

/// "4.2 of 18 GB", "25 of 90 MB"
func ofText(_ done: Int64, _ total: Int64) -> String {
    if total >= 1_000_000_000 {
        let t = Double(total) / 1e9
        return String(format: "%.1f of %@ GB", Double(done) / 1e9, t >= 10 ? String(format: "%.0f", t) : String(format: "%.1f", t))
    }
    return String(format: "%.0f of %.0f MB", Double(done) / 1e6, Double(total) / 1e6)
}

/// "about 12 min left"
func etaText(_ s: Double) -> String {
    if s < 60 { return "less than a minute left" }
    let m = Int((s / 60).rounded())
    if m < 90 { return "about \(m) min left" }
    return "about \(m / 60) h \(m % 60) min left"
}

struct FileRow: View {
    let f: FileProgress
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(f.file.local).font(.caption.monospaced()).lineLimit(1)
                Spacer()
                Text(gbs(f.file.size)).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            HStack(spacing: 6) {
                Image(systemName: icon).foregroundStyle(color).font(.caption)
                Text(word).font(.caption2).foregroundStyle(color).lineLimit(1)
                if f.state == .downloading || f.state == .verifying {
                    ProgressView(value: f.fraction).frame(maxWidth: 120)
                    Text(String(format: "%.0f%%", 100 * f.fraction)).font(.caption2).monospacedDigit()
                }
            }
        }
    }
    /// The state in one word (FileState.label stays the download.json form).
    var word: String {
        switch f.state {
        case .queued: return "Waiting"
        case .downloading: return "Downloading"
        case .verifying: return "Checking"
        case .done: return "Done"
        case .paused: return "Paused"
        case .waiting: return "Retrying soon"
        case .failed: return "Couldn't download"
        }
    }
    var icon: String {
        switch f.state {
        case .done: return "checkmark.circle.fill"
        case .downloading: return "arrow.down.circle"
        case .verifying: return "checkmark.shield"
        case .failed: return "exclamationmark.triangle.fill"
        case .waiting: return "clock.arrow.circlepath"
        case .paused: return "pause.circle"
        case .queued: return "circle.dashed"
        }
    }
    var color: Color {
        switch f.state {
        case .done: return .green
        case .failed: return .red
        case .waiting: return .orange
        default: return .secondary
        }
    }
}

func gbs(_ b: Int64) -> String {
    b >= 1_000_000_000 ? String(format: "%.1f GB", Double(b) / 1e9) : String(format: "%.0f MB", Double(b) / 1e6)
}
