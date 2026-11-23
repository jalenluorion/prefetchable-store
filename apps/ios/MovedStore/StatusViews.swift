import SwiftUI

extension ChatModel.Level {
    var color: Color {
        switch self {
        case .good: return .green
        case .fair: return .yellow
        case .serious: return .orange
        case .critical: return .red
        case .unknown: return .gray
        }
    }
}

/// One line under the composer: the last reply's decode rate and time to first token (the reply in progress while
/// generating), and two dots: memory headroom and thermal state. Tapping it opens StatusSheet.
struct StatusLineView: View {
    @EnvironmentObject var m: ChatModel
    var elsewhere = false

    var body: some View {
        HStack(spacing: 10) {
            if !elsewhere && !m.isGenerating && (m.startup == .checking || m.startup == .loading || m.startup == .warming) {
                ProgressView().controlSize(.mini)  // the launch's background load ("Loading model…")
            }
            Text(elsewhere ? "Replying in another chat" : m.statusText)
                .monospacedDigit()
                .lineLimit(1)
            Spacer(minLength: 4)
            dot(m.memoryLevel, "memorychip")
            dot(m.thermalLevel, "thermometer.medium")
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 4)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityHint("Shows speed, memory and temperature")
    }

    func dot(_ l: ChatModel.Level, _ symbol: String) -> some View {
        HStack(spacing: 3) {
            Circle().fill(l.color).frame(width: 7, height: 7)
            Image(systemName: symbol).imageScale(.small)
        }
    }
}

/// What the status line's numbers and dots mean, in plain words (the figures are in Settings > Diagnostics).
struct StatusSheet: View {
    @EnvironmentObject var m: ChatModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    if let s = m.lastSummary, s.tokens > 0 {
                        LabeledContent("Speed", value: String(format: "%.1f tokens/s", s.meanTokPerSec))
                        if let t = m.lastTtftMs { LabeledContent("Started in", value: String(format: "%.1f s", t / 1000)) }
                    } else {
                        Text("No replies yet.").foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Last reply")
                } footer: {
                    Text("Replies appear a few words per second. Longer messages take longer to start.")
                }

                Section {
                    LabeledContent {
                        Text(memoryWord).foregroundStyle(m.memoryLevel.color)
                    } label: {
                        Label("Memory", systemImage: "memorychip")
                    }
                } footer: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Green: plenty of memory.")
                        Text("Yellow: memory is getting low.")
                        Text("Red: iOS may close the app soon.")
                        if m.memoryLevel == .unknown { Text("Gray: this device has no memory limit.") }
                    }
                }

                Section {
                    LabeledContent {
                        Text(thermalWord).foregroundStyle(m.thermalLevel.color)
                    } label: {
                        Label("Temperature", systemImage: "thermometer.medium")
                    }
                } footer: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Green: normal.")
                        Text("Yellow: warm.")
                        Text("Orange: hot, so replies slow down.")
                        Text("Red: too hot. Let your iPhone cool down.")
                    }
                }
            }
            .navigationTitle("Status")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .onAppear { m.refreshLiveMemory() }
        }
        .presentationDetents([.medium, .large])
    }

    var memoryWord: String {
        switch m.memoryLevel {
        case .good: return "Good"
        case .fair: return "Low"
        case .serious, .critical: return "Very low"
        case .unknown: return "No limit"
        }
    }

    var thermalWord: String {
        switch m.thermalLevel {
        case .good: return "Normal"
        case .fair: return "Warm"
        case .serious: return "Hot"
        case .critical: return "Too hot"
        case .unknown: return "Unknown"
        }
    }
}
