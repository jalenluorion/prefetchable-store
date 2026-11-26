import SwiftUI
import UIKit
import SafariServices

// Chat-app features around the chat: share and export, what to expect, the empty chat, citations.

/// The system share sheet for text or files.
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

/// Something to share, for `.sheet(item:)`.
struct ShareItem: Identifiable {
    let id = UUID()
    let items: [Any]
}

// MARK: export

extension Conversation {
    /// The conversation as Markdown: title, then each turn; thinking left out, a search shown as one line.
    var markdown: String {
        var out = "# \(displayTitle)\n\n"
        out += "_\(created.formatted(date: .abbreviated, time: .shortened))_\n\n"
        for msg in visible {
            switch msg.role {
            case .user:
                out += "**You:** \(msg.content)\n\n"
            case .assistant:
                let answer = splitThinking(msg.content).1
                if let c = ToolCall.split(answer) {
                    if !c.before.isEmpty { out += "**Assistant:** \(c.before)\n\n" }
                    let q = ToolCall.parse(c.call)?.query ?? ""
                    out += "_Searched the web: \(q)_\n\n"
                } else if !answer.isEmpty {
                    out += "**Assistant:** \(answer)\n\n"
                }
            case .tool:
                if let t = meta[msg.id]?.tool, !t.results.isEmpty {
                    out += t.results.map { "- [\($0.title)](\($0.url))" }.joined(separator: "\n") + "\n\n"
                }
            case .system:
                break
            }
        }
        return out
    }
}

enum ChatExport {
    /// Every saved chat in one file in the temporary folder: Markdown (all chats, newest first) or JSON (the app's own
    /// conversation format, one object per chat).
    static func file(_ chats: [Conversation], json: Bool) -> URL? {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        let stamp = f.string(from: Date())
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("Chats \(stamp).\(json ? "json" : "md")")
        let data: Data?
        if json {
            data = try? ConversationStore.encoder().encode(chats.map(ConversationFile.init))
        } else {
            data = chats.map(\.markdown).joined(separator: "\n---\n\n").data(using: .utf8)
        }
        guard let data, (try? data.write(to: url, options: .atomic)) != nil else { return nil }
        return url
    }
}

// MARK: what to expect

/// What to expect on this phone: the decode speed measured on the iPhone 15 Plus (6 GB) for the selected model and mode
/// on phone-small; nothing on other devices (not measured).
enum Expectations {
    static let wordsPerToken = 0.75

    struct Estimate: Codable, Equatable {
        var tokensPerSec: Double
        var wordsPerSec: Double
    }

    /// "About 3 words per second" (one significant digit).
    static func speedText(_ e: Estimate) -> String {
        let w = e.wordsPerSec
        let shown = w >= 1 ? String(Int(w.rounded())) : String(format: "%.1f", w)
        return "About \(shown) word\(shown == "1" ? "" : "s") per second"
    }

    static var stored: Estimate? {
        guard let tps = DmengLaunch.kind.tokensPerSec(profile: DmengLaunch.profile, fast: DmengLaunch.shownMode == "fast") else { return nil }
        return Estimate(tokensPerSec: tps, wordsPerSec: tps * wordsPerToken)
    }
}

/// The low-memory card on the Set up screen.
struct ExpectationsView: View {
    var lowMemory = false
    var grouped = false   // on a grouped list (the Set up screen)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("What to expect").font(.headline)
            if lowMemory {
                Text(Copy.lowMemory).font(.subheadline)
            }
            VStack(alignment: .leading, spacing: 6) {
                Label("This is an experimental research app.", systemImage: "flask")
                if !lowMemory { Label("Replies take a few seconds to start.", systemImage: "clock") }
            }
            .font(.subheadline).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(grouped ? .secondarySystemGroupedBackground : .secondarySystemBackground),
                    in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

}

/// The empty chat: what the app is, what this phone does, and three prompts that show it (one offline, one long paste,
/// one current question that searches the web).
struct EmptyChatView: View {
    var onPick: (String, Bool) -> Void = { _, _ in }   // (text, send now)

    struct Suggestion: Identifiable {
        let id = UUID()
        let icon: String, title: String, prompt: String, send: Bool
    }
    static let suggestions = [
        Suggestion(icon: "airplane", title: "Works in airplane mode", prompt: "Explain how a refrigerator works, in simple terms.", send: true),
        Suggestion(icon: "doc.text", title: "Summarize something long", prompt: "Summarize this in three bullet points:\n\n", send: false),
        Suggestion(icon: "globe", title: "What's in the news today?", prompt: "What are today's top news headlines?", send: true),
    ]

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "iphone.gen3").font(.largeTitle).foregroundStyle(.tertiary).accessibilityHidden(true)
            Text("On this iPhone · works offline").font(.headline)
            if let e = Expectations.stored {
                Text(Expectations.speedText(e) + " on this iPhone").font(.subheadline).foregroundStyle(.secondary)
            }
            VStack(spacing: 8) {
                ForEach(Self.suggestions) { s in
                    Button { onPick(s.prompt, s.send) } label: {
                        HStack(spacing: 10) {
                            Image(systemName: s.icon).frame(width: 20).foregroundStyle(Color.accentColor)
                            Text(s.title).foregroundStyle(.primary).multilineTextAlignment(.leading)
                            Spacer(minLength: 0)
                        }
                        .font(.subheadline)
                        .padding(.horizontal, 14).padding(.vertical, 11)
                        .background(Color(.secondarySystemBackground), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.top, 6)
        }
        .multilineTextAlignment(.center)
        .frame(maxWidth: 420)
        .frame(maxWidth: .infinity)
        .padding(.top, 40)
        .padding(.horizontal, 8)
    }
}

// MARK: citations

/// A web page inside the app.
struct SafariView: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> SFSafariViewController { SFSafariViewController(url: url) }
    func updateUIViewController(_ vc: SFSafariViewController, context: Context) {}
}

struct WebPage: Identifiable { let url: URL; var id: String { url.absoluteString } }

/// The sources a reply drew on, as tappable chips under it (title and site); a tap opens the page in the app.
struct CitationChips: View {
    let sources: [SearchResult]
    @State private var open: WebPage?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Sources").font(.caption).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(Array(sources.prefix(6).enumerated()), id: \.offset) { i, s in
                        Button { open = URL(string: s.url).map(WebPage.init) } label: {
                            HStack(spacing: 6) {
                                Text("\(i + 1)").font(.caption2.weight(.semibold)).foregroundStyle(.white)
                                    .frame(width: 18, height: 18).background(Color.accentColor, in: Circle())
                                VStack(alignment: .leading, spacing: 0) {
                                    Text(s.title).font(.caption.weight(.medium)).lineLimit(1)
                                    Text(s.host).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                }
                                .frame(maxWidth: 180, alignment: .leading)
                            }
                            .padding(.horizontal, 10).padding(.vertical, 7)
                            .background(Color(.secondarySystemBackground), in: Capsule())
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel("Source \(i + 1): \(s.title)")
                    }
                }
            }
        }
        .sheet(item: $open) { SafariView(url: $0.url).ignoresSafeArea() }
    }
}
