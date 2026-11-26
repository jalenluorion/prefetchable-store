import SwiftUI
import UIKit

enum Route: Hashable { case chat }

/// The app: the conversation list, with the open chat pushed on top (at launch, a new chat, so typing can start while
/// the model loads).
struct RootView: View {
    @EnvironmentObject var m: ChatModel
    @State private var path: [Route] = [.chat]

    var body: some View {
        NavigationStack(path: $path) {
            ConversationListView(path: $path)
                .navigationDestination(for: Route.self) { _ in ChatView() }
        }
        #if UITEST
        .onReceive(NotificationCenter.default.publisher(for: UITest.navNote)) { n in
            path = (n.object as? String) == "back" ? [] : [.chat]
        }
        #endif
    }
}

/// One conversation: messages, the composer pinned above the keyboard, and the status line under it.
struct ChatView: View {
    @EnvironmentObject var m: ChatModel
    @State private var draft = ""
    @State private var follow = true
    @State private var atBottom = true
    @State private var showSettings = false
    @State private var showStatus = false
    @State private var renaming = false
    @State private var renameText = ""
    @State private var confirmDelete = false
    @State private var editingID: UUID?
    @State private var share: ShareItem?
    @FocusState private var focused: Bool
    private let bottomID = "bottom"

    var generatingHere: Bool { m.isGenerating && m.generatingConvID == m.conv.id }
    var generatingElsewhere: Bool { m.isGenerating && m.generatingConvID != m.conv.id }
    var lastAssistantID: UUID? { m.conv.messages.last(where: { $0.role == .assistant })?.id }

    private func scrollToLatest(_ proxy: ScrollViewProxy) {
        var transaction = Transaction(animation: nil)
        transaction.disablesAnimations = true
        withTransaction(transaction) { proxy.scrollTo(bottomID, anchor: .bottom) }
    }

    /// The rows: user and assistant messages; a tool result is not a row of its own but part of its call's card.
    struct Row: Identifiable {
        var message: ChatMessage
        var toolResult: MessageMeta?
        /// The web pages the answer drew on (the searches since the user's message), shown as citations under it.
        var sources: [SearchResult] = []
        var id: UUID { message.id }
    }
    var rows: [Row] {
        let v = m.conv.visible
        var out: [Row] = []
        var turnSources: [SearchResult] = []
        for (i, msg) in v.enumerated() {
            if msg.role == .user { turnSources = [] }
            if msg.role == .tool {
                for r in m.conv.meta[msg.id]?.tool?.results ?? [] where !turnSources.contains(where: { $0.url == r.url }) { turnSources.append(r) }
                continue
            }
            let next = i + 1 < v.count && v[i + 1].role == .tool ? (m.conv.meta[v[i + 1].id] ?? MessageMeta()) : nil
            let final = msg.role == .assistant && next == nil
            out.append(Row(message: msg, toolResult: next, sources: final ? turnSources : []))
        }
        return out
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 20) {
                    StartupCard()
                    if m.conv.visible.isEmpty {
                        EmptyChatView { text, now in
                            if now && m.acceptsMessages { m.send(text); follow = true } else { draft = text; focused = true }
                        }
                    }
                    ForEach(rows) { r in
                        MessageRow(message: r.message, meta: m.conv.meta[r.message.id], toolResult: r.toolResult,
                                   isLastAssistant: r.message.id == lastAssistantID, sources: r.sources,
                                   onEdit: { msg in editingID = msg.id; draft = msg.content; focused = true })
                            .id(r.message.id)
                            #if UITEST
                            .modifier(UITestRowProbe(id: r.message.id))
                            #endif
                    }
                    Color.clear.frame(height: 1).id(bottomID)
                        .onAppear { if FollowTracking.legacy { atBottom = true; follow = true } }
                        .onDisappear { if FollowTracking.legacy { atBottom = false } }
                }
                .padding(.horizontal, 16)
                .padding(.top, 12)
                .padding(.bottom, 8)
                #if UITEST
                .coordinateSpace(name: UITest.contentSpace)
                #endif
            }
            .scrollDismissesKeyboard(.interactively)
            .modifier(FollowTracking(follow: $follow, atBottom: $atBottom))
            #if UITEST
            .onChange(of: follow) { _, v in UITest.rec.follow = v; UITest.rec.mark("follow \(v)") }
            .onChange(of: atBottom) { _, v in UITest.rec.atBottom = v }
            #endif
            .onChange(of: m.conv.messages.last) { _, _ in
                // Tool rounds replace the last row as streaming updates it. Use the same scroll
                // transaction for both so an animated row change cannot compete with token updates.
                if follow { scrollToLatest(proxy) }
            }
            .onChange(of: m.conv.id) { _, _ in
                follow = true
                scrollToLatest(proxy)
            }
            .onAppear { scrollToLatest(proxy) }
            .overlay(alignment: .bottomTrailing) {
                if !atBottom && !m.conv.visible.isEmpty {
                    Button {
                        follow = true
                        withAnimation { proxy.scrollTo(bottomID, anchor: .bottom) }
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.body.weight(.semibold))
                            .frame(width: 36, height: 36)
                            .background(.regularMaterial, in: Circle())
                            .shadow(color: .black.opacity(0.12), radius: 4, y: 1)
                    }
                    .padding(16)
                    .accessibilityLabel("Scroll to the latest message")
                    .transition(.opacity)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) { composer }
        }
        .navigationTitle(m.conv.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Menu {
                    Button { renameText = m.conv.title; renaming = true } label: { Label("Rename", systemImage: "pencil") }
                        .disabled(!m.conv.hasUserMessage)
                    Button { UIPasteboard.general.string = m.conv.markdown } label: { Label("Copy chat", systemImage: "doc.on.doc") }
                        .disabled(!m.conv.hasUserMessage)
                    Button { share = ShareItem(items: [m.conv.markdown]) } label: { Label("Share chat", systemImage: "square.and.arrow.up") }
                        .disabled(!m.conv.hasUserMessage)
                    Button { showSettings = true } label: { Label("Settings", systemImage: "gearshape") }
                    Button(role: .destructive) { confirmDelete = true } label: { Label("Delete chat", systemImage: "trash") }
                        .disabled(!m.conv.hasUserMessage)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                Button { m.newConversation(); draft = ""; focused = true } label: { Image(systemName: "square.and.pencil") }
                    .accessibilityLabel("New chat")
            }
        }
        .sheet(isPresented: $showSettings, onDismiss: { m.settingsClosed() }) { SettingsView().environmentObject(m) }
        .sheet(isPresented: $showStatus) { StatusSheet().environmentObject(m) }
        .sheet(item: $share) { ShareSheet(items: $0.items).presentationDetents([.medium, .large]) }
        .sensoryFeedback(.impact(weight: .light), trigger: m.conv.messages.filter { $0.role == .user }.count)   // send
        .sensoryFeedback(.impact(weight: .light), trigger: m.replyTtftMs != nil) { _, new in new }               // first token
        .onChange(of: m.conv.id) { _, _ in editingID = nil }
        .alert("Rename chat", isPresented: $renaming) {
            TextField("Title", text: $renameText)
            Button("Save") { m.rename(m.conv.id, to: renameText) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Leave blank to use the first message.")
        }
        .confirmationDialog("Delete this chat?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) { m.delete(m.conv.id) }
        }
        .task {
            while !Task.isCancelled {
                m.refreshLiveMemory()
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && m.acceptsMessages  // a send while loading waits
    }

    var composer: some View {
        VStack(spacing: 6) {
            if let other = m.readOnlyModel {  // a chat made with the other model: read-only (ModelKind)
                Label("Made with \(other.displayName). To continue it, choose that model in Settings.", systemImage: "lock")
                    .font(.footnote).foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 4)
            }
            if editingID != nil {
                HStack {
                    Label("Editing message", systemImage: "pencil").font(.footnote).foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel") { editingID = nil; draft = "" }.font(.footnote)
                }
                .padding(.horizontal, 4)
            }
            HStack(alignment: .bottom, spacing: 8) {
                Button { m.thinking.toggle() } label: {
                    Image(systemName: "brain")
                        .font(.body)
                        .foregroundStyle(m.thinking ? Color.white : Color.secondary)
                        .frame(width: 36, height: 36)
                        .background(m.thinking ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(Color(.secondarySystemBackground)),
                                    in: Circle())
                }
                .disabled(m.isGenerating)
                .accessibilityLabel(m.thinking ? "Thinking on" : "Thinking off")
                .accessibilityHint("The model thinks before it answers. Replies take longer.")

                Button { m.setTools(!m.conv.toolsOn) } label: {
                    Image(systemName: "globe")
                        .font(.body)
                        .foregroundStyle(m.conv.toolsOn ? Color.white : Color.secondary)
                        .frame(width: 36, height: 36)
                        .background(m.conv.toolsOn ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(Color(.secondarySystemBackground)),
                                    in: Circle())
                }
                .disabled(m.isGenerating)
                .accessibilityLabel(m.conv.toolsOn ? "Web search on" : "Web search off")
                .accessibilityHint("The model can search the web for current facts.")

                TextField("Message", text: $draft, axis: .vertical)
                    .lineLimit(1...6)
                    .focused($focused)
                    .onChange(of: draft) { _, v in if editingID == nil { m.draftChanged(v) } }  // type-ahead
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))

                if generatingHere {
                    Button { m.stop() } label: {
                        Image(systemName: "stop.circle.fill").font(.largeTitle).symbolRenderingMode(.hierarchical)
                    }
                    .accessibilityLabel("Stop")
                } else {
                    Button {
                        if let id = editingID { m.editAndResend(id, text: draft) } else { m.send(draft) }
                        editingID = nil
                        draft = ""
                        follow = true
                    } label: {
                        Image(systemName: "arrow.up.circle.fill").font(.largeTitle)
                    }
                    .disabled(!canSend)
                    .accessibilityLabel("Send")
                }
            }
            Button { showStatus = true } label: { StatusLineView(elsewhere: generatingElsewhere) }
                .buttonStyle(.plain)
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 4)
        .background(.bar)
    }
}

/// Auto-scroll: follow the stream while the view is at the bottom; stop when the user scrolls up; resume at the bottom.
/// iOS 18+ reads the scroll geometry and phase; on iOS 17 the bottom marker's appearance stands in (always follows).
struct FollowTracking: ViewModifier {
    @Binding var follow: Bool
    @Binding var atBottom: Bool
    @State private var userScrolling = false

    static var legacy: Bool {
        #if UITEST
        if UITest.forceLegacy { return true }
        #endif
        if #available(iOS 18.0, *) { return false }
        return true
    }

    func body(content: Content) -> some View {
        if #available(iOS 18.0, *), !Self.legacy {
            content
                .onScrollPhaseChange { _, p in
                    userScrolling = p == .tracking || p == .interacting || p == .decelerating
                }
                .onScrollGeometryChange(for: Bool.self) { g in
                    g.contentOffset.y >= g.contentSize.height - g.containerSize.height + g.contentInsets.bottom - 48
                } action: { _, bottom in
                    atBottom = bottom
                    if bottom { follow = true } else if userScrolling { follow = false }
                }
        } else {
            content
        }
    }
}

/// A failure state at the top of the chat. Nothing while the model loads (the status line under the composer says so)
/// nor when it is ready.
struct StartupCard: View {
    @EnvironmentObject var m: ChatModel

    var body: some View {
        if m.startup == .unavailable {
            card(icon: "exclamationmark.triangle", tint: .orange, title: "The model couldn't load",
                 detail: "Close the app and open it again. Details are in Settings › Diagnostics.")
        }
    }

    func card(spinner: Bool = false, icon: String = "", tint: Color = .secondary, title: String, detail: String?) -> some View {
        HStack(alignment: .top, spacing: 10) {
            if spinner { ProgressView().controlSize(.small).padding(.top, 2) } else { Image(systemName: icon).foregroundStyle(tint) }
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.subheadline.weight(.semibold))
                if let d = detail { Text(d).font(.subheadline).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 0)
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

/// A user message is a bubble; an assistant message is full-width text with its thinking block, caption and actions.
struct MessageRow: View {
    @EnvironmentObject var m: ChatModel
    let message: ChatMessage
    let meta: MessageMeta?
    /// The tool turn after this message, when it is a tool call (its meta carries the search's sources).
    var toolResult: MessageMeta? = nil
    let isLastAssistant: Bool
    var sources: [SearchResult] = []
    var onEdit: ((ChatMessage) -> Void)? = nil

    var streaming: Bool { m.streamingID == message.id }

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 48)
                Text(message.content)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 9)
                    .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .textSelection(.enabled)
                    .contextMenu {
                        Button { UIPasteboard.general.string = message.content } label: { Label("Copy", systemImage: "doc.on.doc") }
                        if let onEdit {
                            Button { onEdit(message) } label: { Label("Edit", systemImage: "pencil") }
                                .disabled(m.isGenerating)
                        }
                    }
            }
        } else {
            assistant
        }
    }

    @ViewBuilder var assistant: some View {
        let thinkingOn = streaming ? m.thinking : (meta?.thinking ?? false)
        let (thought, fullAnswer) = splitThinking(message.content, thinkingOn: thinkingOn)
        let call = ToolCall.split(fullAnswer)
        let answer = call?.before ?? fullAnswer
        VStack(alignment: .leading, spacing: 10) {
            // One inline indicator from send to the first token (restore, prefill): with thinking on it is the thinking
            // block's own header (the label appears beside it at the first token, no layout jump); with thinking off the
            // first streamed text replaces it. The prefill figures stay in the caption, the status line and its popover.
            if thinkingOn, let t = thought, !(t.isEmpty && !streaming) {
                ThinkingBlock(text: t, live: streaming && fullAnswer.isEmpty, start: streaming ? m.thinkStart : nil,
                              seconds: streaming ? m.liveThinkingSeconds : meta?.thinkingSeconds)
            } else if let t = thought, !t.isEmpty {
                ThinkingBlock(text: t, live: false, start: nil, seconds: meta?.thinkingSeconds)
            } else if streaming && message.content.isEmpty {
                LoadingDots()
            }
            if !answer.isEmpty {
                MarkdownView(text: answer)
                    .textSelection(.enabled)
                    .contextMenu {
                        Button { UIPasteboard.general.string = answer } label: { Label("Copy", systemImage: "doc.on.doc") }
                    }
            }
            if let c = call {
                ToolCard(call: ToolCall.parse(c.call), complete: c.complete, result: toolResult?.tool,
                         running: m.toolRunningID == message.id, streaming: streaming)
            }
            if let e = meta?.error {
                Label(e, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.red)
            }
            if !streaming, call == nil, !sources.isEmpty { CitationChips(sources: sources) }
            if !streaming, call == nil, let meta {
                HStack(alignment: .firstTextBaseline, spacing: 14) {
                    ReplyCaption(meta: meta)
                    Spacer(minLength: 0)
                    Button { UIPasteboard.general.string = answer.isEmpty ? message.content : answer } label: {
                        Image(systemName: "doc.on.doc")
                    }
                    .accessibilityLabel("Copy")
                    ShareLink(item: answer.isEmpty ? message.content : answer) { Image(systemName: "square.and.arrow.up") }
                        .accessibilityLabel("Share")
                    if isLastAssistant {
                        Button { m.regenerate() } label: { Image(systemName: "arrow.clockwise") }
                            .disabled(m.isGenerating || m.engineState != .ready)
                            .accessibilityLabel("Regenerate")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .buttonStyle(.borderless)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The quiet line under a reply (speed, time to start); a tap shows the reply's details.
struct ReplyCaption: View {
    let meta: MessageMeta
    @State private var showDetail = false

    var body: some View {
        Button { showDetail = true } label: {
            Text(meta.shortCaption)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .monospacedDigit()
        }
        .buttonStyle(.plain)
        .disabled(meta.detailLines.isEmpty)
        .accessibilityHint("Shows details about this reply")
        .popover(isPresented: $showDetail) {
            VStack(alignment: .leading, spacing: 6) {
                ForEach(Array(meta.detailLines.enumerated()), id: \.offset) { _, kv in
                    HStack(spacing: 16) {
                        Text(kv.0).foregroundStyle(.secondary)
                        Spacer(minLength: 0)
                        Text(kv.1).monospacedDigit()
                    }
                }
            }
            .font(.footnote)
            .padding(14)
            .frame(minWidth: 240)
            .presentationCompactAdaptation(.popover)
        }
    }
}

/// The reasoning trace: open while the model thinks, collapsed once the answer starts (a tap toggles it), with its time.
struct ThinkingBlock: View {
    let text: String
    let live: Bool
    let start: Date?
    let seconds: Double?
    @State private var expanded: Bool?

    var isExpanded: Bool { expanded ?? live }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.snappy) { expanded = !isExpanded }
            } label: {
                HStack(spacing: 6) {
                    // live: the reply's loading indicator (it was there from the send); the label joins it at the first
                    // token (start set)
                    if live { LoadingDots() } else { Image(systemName: "brain").imageScale(.small) }
                    if live, let start {
                        TimelineView(.periodic(from: .now, by: 1)) { ctx in
                            Text("Thinking… " + duration(ctx.date.timeIntervalSince(start)))
                        }
                    } else if !live {
                        Text(seconds.map { "Thought for " + duration($0) } ?? "Thoughts")
                    }
                    if !(live && start == nil) {
                        Image(systemName: "chevron.right")
                            .imageScale(.small)
                            .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    }
                }
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .monospacedDigit()
            }
            .buttonStyle(.plain)
            if isExpanded && !text.isEmpty {
                Text(text)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .padding(.leading, 12)
                    .overlay(alignment: .leading) { RoundedRectangle(cornerRadius: 1).fill(.quaternary).frame(width: 2) }
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

func duration(_ s: Double) -> String {
    let t = Int(s.rounded())
    return t < 60 ? "\(t) s" : "\(t / 60) min \(t % 60) s"
}

/// The reply's loading indicator: three pulsing dots at the start of
/// the reply from the send to the first token (restore, prefill, a search's result). No numbers; a tap shows the live
/// prefill detail (tokens, MB read) in a small popover. The engine's progress events still arrive (ChatModel `prefill`)
/// and feed the status line.
struct LoadingDots: View {
    @EnvironmentObject var m: ChatModel
    @State private var showDetail = false

    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .frame(width: 7, height: 7)
                    .phaseAnimator([0.25, 1.0]) { c, ph in c.opacity(ph) } animation: { _ in
                        .easeInOut(duration: 0.55).delay(Double(i) * 0.18)
                    }
            }
        }
        .foregroundStyle(.secondary)
        .frame(height: 20)
        .contentShape(Rectangle())
        .onTapGesture { showDetail = true }
        .popover(isPresented: $showDetail) {
            PrefillDetail().environmentObject(m).presentationCompactAdaptation(.popover)
        }
        .accessibilityElement()
        .accessibilityLabel("Preparing the reply")
    }
}

/// The popover behind the loading indicator: what the engine is doing right now.
struct PrefillDetail: View {
    @EnvironmentObject var m: ChatModel

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { ctx in
            VStack(alignment: .leading, spacing: 4) {
                ForEach(lines(ctx.date), id: \.self) { Text($0) }
            }
            .font(.caption.monospacedDigit())
            .padding(12)
            .frame(minWidth: 220, alignment: .leading)
        }
    }

    func lines(_ now: Date) -> [String] {
        if let q = m.toolRunningQuery { return ["Searching the web", q] }
        guard let p = m.prefill else { return [m.isGenerating ? "Starting" : "Idle"] }
        if p.waitingForWarmup {
            return [p.waitingFor == "the one-time setup" ? "Waiting for setup to finish" : "Waiting for the model", "Your message is next"]
        }
        var out: [String] = []
        let el = now.timeIntervalSince(p.start)
        if let t = p.tick {
            switch t.kind {
            case "restored prefix": out.append("Reused \(t.reusedTokens) tokens from cache")
            case "live prefix": out.append("Continued from \(t.reusedTokens) tokens")
            default: out.append("Starting fresh")
            }
            out.append("Read \(t.done) of \(t.total) tokens")
            out.append(t.mbRead >= 1000 ? String(format: "%.1f GB from storage", t.mbRead / 1000) : String(format: "%.0f MB from storage", t.mbRead))
        } else {
            out.append(p.restoring ? "Restoring the chat" : "Reading your message")
            out.append("About \(p.expectedTokens) tokens")
        }
        out.append(duration(el))
        return out
    }
}

/// A tool call and its result, in place of the call's raw text: "Searching: <query>" (open_url: "Reading: <page>") with a
/// spinner while the call streams and the tool runs, then the sources as links (marked when the page itself was read),
/// Tavily's answer if any, and a caption (results, pages read, the result's tokens as the model reads them against the
/// budget, the time, the backend).
struct ToolCard: View {
    let call: ToolCall?
    let complete: Bool
    let result: ToolMeta?
    let running: Bool
    let streaming: Bool

    var query: String { call?.query ?? result?.query ?? "" }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                if let r = result {
                    Image(systemName: r.error == nil ? "globe" : "exclamationmark.triangle")
                        .foregroundStyle(r.error == nil ? Color.secondary : Color.orange)
                } else if running || streaming {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "globe").foregroundStyle(.tertiary)
                }
                Text(displayTitle).font(.subheadline).lineLimit(2)
            }
            if let r = result {
                if let a = r.answer {
                    Text(a).font(.caption).foregroundStyle(.secondary).lineLimit(3)
                }
                ForEach(Array(r.results.enumerated()), id: \.offset) { _, s in
                    if let u = URL(string: s.url) {
                        Link(destination: u) {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(s.title).font(.footnote.weight(.medium)).lineLimit(1)
                                Text(s.host + (s.read == true ? " · read" : "")).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                            }
                        }
                    }
                }
                Text(displayCaption(r)).font(.caption2).foregroundStyle(.tertiary).monospacedDigit()
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// What the card shows; `title` and `caption` below are the forms the console log uses.
    var displayTitle: String {
        let q = query.isEmpty ? "…" : (isPageRead ? SearchResult(title: "", url: query, snippet: "").shortURL : "“" + query + "”")
        if let r = result {
            if r.error == "limit" { return isPageRead ? "Page limit reached" : "Search limit reached" }
            if isPageRead { return r.error == nil ? "Read " + q : "Couldn't open " + q }
            return r.error == nil ? "Searched " + q : "Couldn't search " + q
        }
        if running || streaming { return (isPageRead ? "Reading " : "Searching ") + q }
        return (isPageRead ? "Didn't open " : "Didn't search ") + q
    }

    func displayCaption(_ r: ToolMeta) -> String {
        if let e = r.error { return e.contains("timeout") || e.contains("timed out") ? "Timed out" : (e == "limit" ? "" : "Try again later") }
        var parts: [String] = []
        if r.name != "open_url" { parts.append("\(r.results.count) source" + (r.results.count == 1 ? "" : "s")) }
        parts.append(r.backend)
        return parts.joined(separator: " · ")
    }

    /// open_url: "Reading: host/path" (call or result).
    var isPageRead: Bool { call?.isPageRead ?? (result?.name == "open_url") }

    var title: String {
        let q = query.isEmpty ? "…" : (isPageRead ? SearchResult(title: "", url: query, snippet: "").shortURL : query)
        if let r = result {
            if r.error == "limit" { return isPageRead ? "Page limit reached" : "Search limit reached" }
            if isPageRead { return r.error == nil ? "Read: " + q : "Could not read: " + q }
            return r.error == nil ? "Searched: " + q : "Search failed: " + q
        }
        if running || (streaming && !complete) || streaming { return (isPageRead ? "Reading: " : "Searching: ") + q }
        return (isPageRead ? "Page not read: " : "Search not run: ") + q
    }

    /// "3 results · 2 pages read · 109 / 112 tokens · 1.4 s · DuckDuckGo": the tokens as the model reads them (the engine's
    /// tokenizer) against the budget.
    func caption(_ r: ToolMeta) -> String {
        var parts: [String] = []
        if let e = r.error { parts.append(e) } else if r.name != "open_url" {
            parts.append("\(r.results.count) result" + (r.results.count == 1 ? "" : "s"))
            if let p = r.pages, p > 0 { parts.append("\(p) page\(p == 1 ? "" : "s") read") }
        }
        if let t = r.resultTokens { parts.append(r.budget.map { "\(t) / \($0) tokens" } ?? "\(t) tokens") }
        if r.ms > 0 { parts.append(String(format: "%.1f s", r.ms / 1000)) }
        parts.append(r.backend)
        return parts.joined(separator: " · ")
    }
}
