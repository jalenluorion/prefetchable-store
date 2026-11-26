import SwiftUI
import UIKit

/// Settings: Model, Generation, Prefix cache, and everything the sheet showed before under Diagnostics.
struct SettingsView: View {
    @EnvironmentObject var m: ChatModel
    @Environment(\.dismiss) private var dismiss
    @AppStorage(AppSettings.defaultThinkingKey) private var defaultThinking = false
    @AppStorage(AppSettings.maxTokensKey) private var maxTokens = 1024
    @AppStorage(AppSettings.typeAheadKey) private var typeAhead = true
    @AppStorage(AppSettings.greedyKey) private var greedy = false
    @AppStorage(AppSettings.temperatureKey) private var temperature = AppSettings.modelTemperature
    @AppStorage(AppSettings.topPKey) private var topP = AppSettings.modelTopP
    @AppStorage(AppSettings.topKKey) private var topK = AppSettings.modelTopK
    @AppStorage(AppSettings.minPKey) private var minP = AppSettings.modelMinP
    @AppStorage(AppSettings.repetitionPenaltyKey) private var repetitionPenalty = AppSettings.modelRepetitionPenalty
    @AppStorage(AppSettings.presencePenaltyKey) private var presencePenalty = AppSettings.modelPresencePenalty
    @AppStorage(AppSettings.thinkingBudgetKey) private var thinkingBudget = 0
    @AppStorage(ToolSettings.enabledKey) private var toolsOn = true
    @AppStorage(ToolSettings.backendKey) private var backend = ToolBackend.duckduckgo.rawValue
    @AppStorage(ToolSettings.pageReadsKey) private var pageReads = true
    @State private var usage = PrefixCacheUsage.read()
    @State private var confirmClear = false
    @State private var share: ShareItem?
    @State private var confirmDeleteModel = false

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                generationSection
                toolsSection
                samplingSection
                prefixSection
                Section {
                    Button("Export as Markdown") { share = ChatExport.file(m.conversations, json: false).map { ShareItem(items: [$0]) } }
                    Button("Export as JSON") { share = ChatExport.file(m.conversations, json: true).map { ShareItem(items: [$0]) } }
                } header: {
                    Text("Your chats")
                } footer: {
                    Text(m.conversations.isEmpty ? "No chats yet." : (m.conversations.count == 1 ? "Your chat in one file." : "All \(m.conversations.count) chats in one file."))
                }
                .disabled(m.conversations.isEmpty)
                Section {
                    NavigationLink { DiagnosticsView().environmentObject(m) } label: {
                        Label("Diagnostics", systemImage: "stethoscope")
                    }
                } footer: {
                    Text("Speed, memory, engine settings and model details.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
            .onAppear { usage = PrefixCacheUsage.read() }
            .onDisappear { Task { await m.warmUp() } }  // a changed system prompt is warmed into the cache
            .sheet(item: $share) { ShareSheet(items: $0.items).presentationDetents([.medium, .large]) }
        }
    }

    // MARK: Model

    var stateColor: Color {
        switch m.engineState {
        case .ready: return .green
        case .loading, .idle: return .secondary
        case .refused, .failed: return .red
        }
    }

    var stateWord: String {
        switch m.engineState {
        case .ready: return "Ready"
        case .idle, .loading: return "Loading…"
        case .refused, .failed: return "Not available"
        }
    }

    /// The selected model's files on this phone, and the free space.
    @State private var presentBytes = ModelKind.selected.presentBytes
    @State private var freeBytes = ModelDownloader.volumeFree() ?? 0

    /// The mode of the selected model (each model has its own setting: ModelKind.modeKey).
    @State private var modeShown = DmengLaunch.userMode
    var engineMode: Binding<String> {
        Binding(get: { modeShown }, set: { DmengLaunch.userMode = $0; modeShown = $0 })
    }
    /// Settings › Model: the models whose files are present (ModelKind); a choice closes the open model and opens the
    /// other one (a new chat starts with it).
    @State private var modelShown = ModelKind.selected
    @State private var switching = false
    var modelChoice: Binding<ModelKind> {
        Binding(get: { modelShown }, set: { k in
            guard k != modelShown else { return }
            modelShown = k
            switching = true
            Task {
                await m.switchModel(to: k)
                modelShown = ModelKind.selected
                modeShown = DmengLaunch.userMode
                switching = false
            }
        })
    }

    @ViewBuilder var modelSection: some View {
        Section {
            LabeledContent("Status") { Text(stateWord).foregroundStyle(stateColor) }
            if ModelKind.present.count > 1 {
                Picker("Model", selection: modelChoice) {
                    ForEach(ModelKind.present) { k in Text(k.displayName).tag(k) }
                }
                .disabled(switching || m.isGenerating)
            } else {
                LabeledContent("Model", value: Copy.modelName)
            }
            if presentBytes > 0 {
                LabeledContent("Storage", value: "\(gbs(presentBytes)) used · \(gbs(max(0, freeBytes))) free")
            }
            NavigationLink {
                SetupView(mode: .manage, beforeDelete: { m.stop() }, afterDelete: { afterDeletingSelected() })
                    .environmentObject(ModelDownloader.shared)
            } label: {
                Text("Model files")
            }
            Picker("Mode", selection: engineMode) {
                Text("Fast (default)").tag("fast")
                Text("Exact").tag("exact")
            }
            Button("Delete model", role: .destructive) { confirmDeleteModel = true }
                .disabled(presentBytes == 0)
                .confirmationDialog("Delete the model?", isPresented: $confirmDeleteModel, titleVisibility: .visible) {
                    Button("Delete model", role: .destructive) {
                        m.stop()
                        Task {
                            await ModelDownloader.shared.deleteModel(ModelKind.selected)
                            afterDeletingSelected()
                        }
                    }
                    Button("Cancel", role: .cancel) {}
                } message: {
                    Text(ModelKind.selected.other.isComplete
                         ? "This frees \(gbs(presentBytes)). Your chats are kept. The app switches to \(ModelKind.selected.other.displayName)."
                         : "This frees \(gbs(presentBytes)). Your chats are kept. To use the app again, you'll need to download the model again.")
                }
        } header: {
            Text("Model")
        } footer: {
            Text((switching ? "Switching to \(modelShown.displayName)…\n" : "")
                 + (engineMode.wrappedValue == "fast"
                    ? "Fast: replies come sooner. Answers can differ a little from the full model's.\nApplies the next time you open the app."
                    : "Exact: answers identical to the full model's. Replies are slower.\nApplies the next time you open the app."))
        }
    }

    /// The selected model's files were deleted: the other model when its files are here, else the set-up screen.
    func afterDeletingSelected() {
        let other = ModelKind.selected.other
        if other.isComplete {
            Task { await m.switchModel(to: other); modelShown = ModelKind.selected; modeShown = DmengLaunch.userMode }
        } else {
            SetupGate.shared.modelDeleted()
        }
    }

    // MARK: Generation

    var systemPromptPreview: String {
        let s = UserDefaults.standard.string(forKey: AppSettings.systemPromptKey) ?? AppSettings.defaultSystemPrompt
        return s.isEmpty ? "None" : s
    }

    @ViewBuilder var generationSection: some View {
        Section {
            Toggle("Thinking in new chats", isOn: $defaultThinking)
            Stepper(value: $thinkingBudget, in: 0...4096, step: 128) {
                LabeledContent("Thinking limit", value: thinkingBudget == 0 ? "None" : "\(thinkingBudget) tokens")
            }
            Stepper(value: $maxTokens, in: 128...4096, step: 128) {
                LabeledContent("Longest reply", value: "\(maxTokens) tokens")
            }
            NavigationLink {
                CustomInstructionsEditor()
            } label: {
                LabeledContent("Custom instructions", value: CustomInstructions.isSet ? "On" : "Off")
            }
        } header: {
            Text("Chat")
        } footer: {
            Text(ModelKind.selected == .inkling
                 ? "The brain button turns thinking on or off for each chat. With it off, Inkling 2 Mini answers without reasoning first."
                 : "The brain button turns thinking on or off for each chat.")
        }
    }

    // MARK: Tools

    @ViewBuilder var toolsSection: some View {
        Section {
            Toggle("Web search", isOn: $toolsOn)
            if toolsOn {
                Picker("Search with", selection: $backend) {
                    ForEach(ToolBackend.allCases) { Text($0.short).tag($0.rawValue) }
                }
                if let b = ToolBackend(rawValue: backend), b.needsKey {
                    NavigationLink { SearchKeyEditor(backend: b) } label: {
                        LabeledContent("\(b.short) API key", value: SearchKeychain.has(b) ? "Saved" : "Not set")
                    }
                }
                Toggle("Read result pages", isOn: $pageReads)
            }
        } header: {
            Text("Web search")
        } footer: {
            Text("When a question needs current facts, the model can search the web. The globe button turns search on or off for each chat.")
        }
    }

    @ViewBuilder var samplingSection: some View {
        Section {
            Toggle("Always pick the likeliest word", isOn: $greedy)
            if !greedy {
                Stepper(value: $temperature, in: 0...2, step: 0.05) {
                    LabeledContent("Temperature", value: String(format: "%.2f", temperature))
                }
                Stepper(value: $topP, in: 0.05...1, step: 0.05) {
                    LabeledContent("Top-p", value: String(format: "%.2f", topP))
                }
                Stepper(value: $topK, in: 0...200, step: 1) {
                    LabeledContent("Top-k", value: topK == 0 ? "Off" : "\(topK)")
                }
                Stepper(value: $minP, in: 0...0.5, step: 0.01) {
                    LabeledContent("Min-p", value: minP == 0 ? "Off" : String(format: "%.2f", minP))
                }
                Stepper(value: $repetitionPenalty, in: 1...2, step: 0.05) {
                    LabeledContent("Repetition penalty", value: repetitionPenalty == 1 ? "Off" : String(format: "%.2f", repetitionPenalty))
                }
                Stepper(value: $presencePenalty, in: 0...2, step: 0.1) {
                    LabeledContent("Presence penalty", value: presencePenalty == 0 ? "Off" : String(format: "%.1f", presencePenalty))
                }
            }
            Button("Reset to defaults") {
                for k in AppSettings.samplingKeys { UserDefaults.standard.removeObject(forKey: k) }
                greedy = false
                temperature = AppSettings.modelTemperature
                topP = AppSettings.modelTopP
                topK = AppSettings.modelTopK
                minP = AppSettings.modelMinP
                repetitionPenalty = AppSettings.modelRepetitionPenalty
                presencePenalty = AppSettings.modelPresencePenalty
                for k in AppSettings.samplingKeys { UserDefaults.standard.removeObject(forKey: k) }
            }
        } header: {
            Text("Advanced")
        }
    }

    // MARK: Prefix cache

    @ViewBuilder var prefixSection: some View {
        Section {
            Toggle("Save chat progress", isOn: $m.prefixCache)
                .disabled(m.isGenerating)
            Toggle("Read my message while I type", isOn: $typeAhead)
            LabeledContent("Storage used", value: String(format: "%.0f MB of %.1f GB", Double(usage.bytes) / 1e6, usage.capBytes / 1e9))
            Button("Clear saved progress", role: .destructive) { confirmClear = true }
                .disabled(m.isGenerating || usage.entries == 0)
                .confirmationDialog("Clear saved progress?", isPresented: $confirmClear, titleVisibility: .visible) {
                    Button("Clear", role: .destructive) {
                        m.clearPrefixCache()
                        Task { try? await Task.sleep(nanoseconds: 500_000_000); usage = PrefixCacheUsage.read() }
                    }
                } message: {
                    Text("Your chats are kept, but the next message in each one starts slower.")
                }
        } header: {
            Text("Speed")
        } footer: {
            Text("Saving progress lets chats continue quickly instead of starting over.")
        }
    }
}

/// Settings > Custom instructions: two plain fields rendered into the system prompt for new chats
/// (CustomInstructions.render); both empty = the default prompt exactly, so its cached state still applies.
struct CustomInstructionsEditor: View {
    @AppStorage(CustomInstructions.aboutKey) private var about = ""
    @AppStorage(CustomInstructions.howKey) private var how = ""

    var body: some View {
        Form {
            Section {
                TextField("For example: I'm a researcher in Pittsburgh.", text: $about, axis: .vertical).lineLimit(3...8)
            } header: {
                Text("About you")
            }
            Section {
                TextField("For example: Keep answers short and use bullet points.", text: $how, axis: .vertical).lineLimit(3...8)
            } header: {
                Text("How to answer")
            } footer: {
                Text("Used for new chats. The first message after a change starts a little slower.")
            }
            if !about.isEmpty || !how.isEmpty {
                Section { Button("Clear", role: .destructive) { about = ""; how = "" } }
            }
        }
        .navigationTitle("Custom instructions")
        .navigationBarTitleDisplayMode(.inline)
        .onChange(of: about) { _, _ in CustomInstructions.apply() }
        .onChange(of: how) { _, _ in CustomInstructions.apply() }
    }
}

enum CustomInstructions {
    static let aboutKey = "chat.aboutYou", howKey = "chat.howToAnswer"
    static var about: String { (UserDefaults.standard.string(forKey: aboutKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
    static var how: String { (UserDefaults.standard.string(forKey: howKey) ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
    static var isSet: Bool { !about.isEmpty || !how.isEmpty }

    static func render(about: String, how: String) -> String {
        var s = AppSettings.defaultSystemPrompt
        if !about.isEmpty { s += "\n\nAbout the user: " + about }
        if !how.isEmpty { s += "\n\nHow to answer: " + how }
        return s
    }

    /// Write the rendered prompt as the system prompt for new chats.
    static func apply() {
        UserDefaults.standard.set(render(about: about, how: how), forKey: AppSettings.systemPromptKey)
    }
}

/// The raw system prompt for new chats (Diagnostics; existing chats keep the one they started with).
struct SystemPromptEditor: View {
    @AppStorage(AppSettings.systemPromptKey) private var prompt = AppSettings.defaultSystemPrompt

    var body: some View {
        Form {
            Section {
                TextEditor(text: $prompt)
                    .frame(minHeight: 160)
            } footer: {
                Text("Used for new chats. Each chat keeps the prompt it started with.")
            }
            Section {
                Button("Restore default") { prompt = AppSettings.defaultSystemPrompt }
                Button("Clear", role: .destructive) { prompt = "" }
            }
        }
        .navigationTitle("System prompt")
        .navigationBarTitleDisplayMode(.inline)
    }
}

/// Speed, memory, the engine's settings and the device.
struct DiagnosticsView: View {
    @EnvironmentObject var m: ChatModel
    @State private var copied = false
    @State private var spareChoice = DmengLaunch.spareMemoryChoice
    var spareMemory: Binding<Bool> {
        Binding(get: { spareChoice ?? (m.engine as? DmengChatEngine)?.spareMemoryResolved ?? true },
                set: { DmengLaunch.spareMemoryChoice = $0; spareChoice = $0 })
    }

    /// Every Engine and Last reply row as plain text, one "label: value" per line (Copy diagnostics).
    var diagnosticsText: String {
        var lines = ["MovedStore diagnostics, " + machineName() + ", " + ProcessInfo.processInfo.operatingSystemVersionString
                     + String(format: ", %.1f GB memory", Double(ProcessInfo.processInfo.physicalMemory) / 1e9)]
        lines.append("")
        lines.append("Last reply in detail")
        let rows = m.lastReplyRows
        lines += rows.isEmpty ? ["(no reply yet)"] : rows.map { "\($0.0): \($0.1)" }
        lines.append("")
        lines.append("Engine")
        lines.append("State: \(m.engineState.label)")
        lines.append(m.engine.info.detail)
        lines += m.engine.settings.map { "\($0.0): \($0.1)" }
        return lines.joined(separator: "\n")
    }

    var body: some View {
        Form {
            Section("Last reply") {
                Text(m.prefixStatus.isEmpty ? m.engine.prefixCacheStatus : m.prefixStatus).font(.caption).foregroundStyle(.secondary)
                row("Decode tok/s", String(format: "%.2f (last token %.2f)", m.replyTokPerSec, m.lastTokPerSec))
                row("Time to first token", (m.isGenerating ? m.replyTtftMs : m.lastTtftMs).map { String(format: "%.2f s", $0 / 1000) } ?? "-")
                row("MB / token", String(format: "%.1f", m.mbPerToken))
                row("Experts already in memory", String(format: "%.1f %%", m.inMemory * 100))
                if let s = m.lastSummary {
                    row("Last reply", String(format: "%d tokens, %@, %.2f tok/s, %.1f MB/tok", s.tokens, s.reason.rawValue,
                                             m.replyTokPerSec, s.meanMBPerToken))
                }
                if let k = m.lastPrefillKind, let p = m.lastPrefill {
                    row("Prefill", String(format: "%@: %.0f reused, %.0f computed, %.2f s", k, p["prefill_reused"] ?? 0,
                                          p["prefill_computed"] ?? 0, p["prefill_s"] ?? 0))
                }
            }
            Section("Memory") {
                if let s = m.latestMem {
                    row("Footprint", gb(s.physFootprint))
                    row("Footprint peak", gb(s.physFootprintPeak))
                    row("Available to the app", gb(s.osProcAvailable))
                    if let l = s.limitEstimate { row("Limit (footprint + available)", gb(l)) }
                }
                row("Memory warnings", "\(m.memoryWarnings)")
                Button("Sample memory now") { m.memSample("settings") }
            }
            if !m.lastReplyRows.isEmpty {
                Section {
                    ForEach(Array(m.lastReplyRows.enumerated()), id: \.offset) { _, kv in
                        row(kv.0, kv.1).font(.caption)
                    }
                } header: {
                    Text("Last reply in detail")
                } footer: {
                    Text("From the engine's own counters for the most recent reply.")
                }
            }
            if DmengLaunch.spareMemoryOffered {
                Section {
                    Toggle("Use spare memory for the model's files", isOn: spareMemory)
                } footer: {
                    Text("Keeps recently used parts of the model in memory the system has spare, so fewer are read from storage. "
                         + "On by default on this phone. Applies the next time you open the app.")
                }
            }
            Section {
                Button(copied ? "Copied" : "Copy diagnostics") {
                    UIPasteboard.general.string = diagnosticsText
                    copied = true
                }
            } footer: {
                Text("Copies the Last reply and Engine rows as text, to paste into a message.")
            }
            Section("Engine") {
                row("State", m.engineState.label)
                Text(m.engine.info.detail).font(.caption).foregroundStyle(.secondary)
                ForEach(Array(m.engine.settings.enumerated()), id: \.offset) { _, kv in
                    row(kv.0, kv.1).font(.caption)
                }
                ForEach(Array(m.loadLog.enumerated()), id: \.offset) { _, l in
                    Text(l).font(.caption.monospaced())
                }
            }
            Section("Device") {
                row("Machine", machineName())
                row("OS", ProcessInfo.processInfo.operatingSystemVersionString)
                row("Memory", gb(Int(ProcessInfo.processInfo.physicalMemory)))
                row("Thermal", m.thermal)
                row("Low power mode", ProcessInfo.processInfo.isLowPowerModeEnabled ? "on" : "off")
            }
            Section("System prompt") {
                NavigationLink { SystemPromptEditor() } label: {
                    Text(UserDefaults.standard.string(forKey: AppSettings.systemPromptKey) ?? AppSettings.defaultSystemPrompt)
                        .font(.caption).foregroundStyle(.secondary).lineLimit(3)
                }
            }
        }
        .navigationTitle("Diagnostics")
        .navigationBarTitleDisplayMode(.inline)
    }

    func row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k)
            Spacer()
            Text(v).foregroundStyle(.secondary).multilineTextAlignment(.trailing)
        }
    }
}

/// Paste or clear a search API key; it goes to the Keychain only (SearchKeychain), never to settings or logs.
struct SearchKeyEditor: View {
    let backend: ToolBackend
    @State private var key = ""
    @State private var status = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Form {
            Section {
                SecureField("API key", text: $key)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button("Save") {
                    let e = SearchKeychain.set(backend, key)
                    status = e ?? (key.isEmpty ? "Key removed." : "Key saved.")
                    key = ""
                }
                if SearchKeychain.has(backend) {
                    Button("Remove key", role: .destructive) {
                        SearchKeychain.set(backend, "")
                        status = "Key removed."
                    }
                }
            } footer: {
                Text(status.isEmpty ? (backend == .brave ? "Get a free key at api-dashboard.search.brave.com." : "Get a key at app.tavily.com.") : status)
            }
        }
        .navigationTitle("\(backend.short) key")
    }
}
