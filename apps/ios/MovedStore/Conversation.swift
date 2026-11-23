import Foundation

// Conversations: in memory a list of ChatMessage (the engine's input, a leading system message included) plus per-reply
// metadata; on disk one JSON file per conversation in Documents/conversations/<id>.json (README "Conversations").

/// What happened for one assistant reply: shown as the caption under the message, saved with the conversation.
struct MessageMeta: Codable, Equatable, Sendable {
    var tokens = 0
    var reason = ""                   // eos | max_new | cancelled | error
    var decodeTokPerSec: Double?      // tokens / the engine's decode clock (prefill excluded)
    var ttftMs: Double?               // send -> first piece (wall)
    var prefillKind: String?          // "cold", "live prefix", "restored prefix"
    var promptTokens: Int?            // the whole prompt the template gave
    var prefilledTokens: Int?         // prompt tokens actually run through the model for this reply
    var reusedTokens: Int?            // prompt tokens taken from the prefix cache or the live conversation
    var restoreMs: Double?            // prefix-cache restore (state load + expert re-land)
    var thinkingSeconds: Double?      // first token -> end of the thinking block
    var mbPerToken: Double?           // MB (1e6 B) read from flash per generated token
    var thinking: Bool?               // generated with thinking on
    var error: String?
    /// A tool-call turn (assistant): the parsed call. A tool turn: what the tool did (the card's sources and caption).
    var toolCall: ToolCall?
    var tool: ToolMeta?

    /// One quiet line: what the prompt cost, then the reply's speed.
    var caption: String {
        var parts: [String] = []
        if let p = prefilledTokens {
            switch prefillKind {
            case "restored prefix": parts.append("restored \(reusedTokens ?? 0) tokens · prefilled \(p)")
            case "live prefix": parts.append("continued \(reusedTokens ?? 0) tokens · prefilled \(p)")
            default: parts.append("prefilled \(p) tokens (cold)")
            }
        }
        if let t = ttftMs { parts.append(String(format: "first token %.1f s", t / 1000)) }
        if let r = decodeTokPerSec, tokens > 0 { parts.append(String(format: "%.1f tok/s", r)) }
        if reason == "cancelled" { parts.append("stopped") }
        if reason == "max_new" { parts.append("length limit") }
        return parts.joined(separator: " · ")
    }

    /// What the chat shows under a reply: the speed and how long the reply took to start. `caption`
    /// above stays the technical line (console `CAPTION`); the tap-through shows `detailLines`.
    var shortCaption: String {
        var parts: [String] = []
        if let r = decodeTokPerSec, tokens > 0 { parts.append(String(format: "%.1f tokens/s", r)) }
        if let t = ttftMs { parts.append(String(format: "started in %.1f s", t / 1000)) }
        if reason == "cancelled" { parts.append("stopped") }
        if reason == "max_new" { parts.append("reached the length limit") }
        guard let first = parts.first else { return "" }
        parts[0] = first.prefix(1).uppercased() + first.dropFirst()
        return parts.joined(separator: " · ")
    }

    /// The reply's details, one per line (the caption's tap-through).
    var detailLines: [(String, String)] {
        var out: [(String, String)] = []
        if let r = decodeTokPerSec, tokens > 0 { out.append(("Speed", String(format: "%.1f tokens/s", r))) }
        if let t = ttftMs { out.append(("Started in", String(format: "%.1f s", t / 1000))) }
        if tokens > 0 { out.append(("Length", "\(tokens) tokens")) }
        if let s = thinkingSeconds { out.append(("Thinking", String(format: "%.0f s", s))) }
        if let p = prefilledTokens {
            switch prefillKind {
            case "restored prefix": out.append(("Reused from cache", "\(reusedTokens ?? 0) tokens"))
            case "live prefix": out.append(("Continued", "\(reusedTokens ?? 0) tokens"))
            default: break
            }
            out.append(("Read", "\(p) tokens"))
        }
        if let m = mbPerToken { out.append(("Storage read", String(format: "%.0f MB per token", m))) }
        return out
    }
}

struct Conversation: Identifiable, Equatable {
    var id = UUID()
    var title = ""
    var titleIsCustom = false
    var created = Date()
    var updated = Date()
    var thinking = false
    /// Tools for this chat (the globe button); nil: the setting (Settings > Tools).
    var tools: Bool?
    /// The engine's input: a leading system message (if any), then user / assistant turns.
    var messages: [ChatMessage] = []
    var meta: [UUID: MessageMeta] = [:]
    /// Prompt + generated tokens after the last reply (the engine's count; 0 until the first reply).
    var contextTokens = 0
    /// The model the chat is made with (ModelKind); a file without it is the 35B's.
    var model: ModelKind = ModelKind.selected

    init(system: String = "", thinking: Bool = false) {
        self.thinking = thinking
        let s = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !s.isEmpty { messages = [ChatMessage(role: .system, content: s)] }
    }

    var systemPrompt: String { messages.first?.role == .system ? messages[0].content : "" }
    var visible: [ChatMessage] { messages.filter { $0.role != .system } }
    /// The chat's tool mode: its own override, else the setting.
    var toolsOn: Bool { tools ?? ToolSettings.enabled }
    var hasUserMessage: Bool { messages.contains { $0.role == .user } }
    var displayTitle: String { title.isEmpty ? "New chat" : title }

    /// The first line of the first user message, shortened.
    static func title(from text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline).first.map(String.init) ?? text
        let plain = line.replacingOccurrences(of: "**", with: "").replacingOccurrences(of: "__", with: "")
            .replacingOccurrences(of: "`", with: "")
        let t = plain.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "#>*-")))
        return t.count > 48 ? String(t.prefix(47)) + "…" : t
    }

    /// The last assistant message's answer text (thinking removed), for the list's preview line.
    var preview: String {
        guard let a = messages.last(where: { $0.role == .assistant }) else { return "" }
        var ans = splitThinking(a.content).1
        if let c = ToolCall.split(ans) { ans = c.before.isEmpty ? "Searched the web" : c.before }
        ans = ans.replacingOccurrences(of: "\n", with: " ")
        return ans.count > 90 ? String(ans.prefix(89)) + "…" : ans
    }
}

/// The on-disk format, one file per conversation (snake_case keys, ISO 8601 dates, pretty-printed).
struct ConversationFile: Codable {
    struct Message: Codable {
        var id: UUID
        var role: ChatMessage.Role
        var content: String
        var meta: MessageMeta?
    }
    var format = ConversationStore.format
    var id: UUID
    var title: String
    var titleIsCustom: Bool
    var created: Date
    var updated: Date
    var systemPrompt: String
    var thinking: Bool
    var tools: Bool?
    var contextTokens: Int
    var model: String?        // ModelKind's raw value (absent: the 35B)
    var messages: [Message]   // user and assistant turns, oldest first (the system prompt is the field above)

    init(_ c: Conversation) {
        id = c.id; title = c.title; titleIsCustom = c.titleIsCustom; created = c.created; updated = c.updated
        systemPrompt = c.systemPrompt; thinking = c.thinking; tools = c.tools; contextTokens = c.contextTokens
        model = c.model.rawValue
        messages = c.visible.map { Message(id: $0.id, role: $0.role, content: $0.content, meta: c.meta[$0.id]) }
    }

    var conversation: Conversation {
        var c = Conversation(system: systemPrompt, thinking: thinking)
        c.id = id; c.title = title; c.titleIsCustom = titleIsCustom; c.created = created; c.updated = updated
        c.contextTokens = contextTokens
        c.tools = tools
        c.model = model.flatMap(ModelKind.init(rawValue:)) ?? .qwen36
        for m in messages where m.role != .system {
            c.messages.append(ChatMessage(id: m.id, role: m.role, content: m.content))
            if let x = m.meta { c.meta[m.id] = x }
        }
        return c
    }
}

/// Documents/conversations/<uuid>.json, written atomically after every reply, rename and thinking change.
enum ConversationStore {
    static let format = "movedstore.conversation.v1"
    static var dir: URL { docsURL.appendingPathComponent("conversations", isDirectory: true) }

    static func url(_ id: UUID) -> URL { dir.appendingPathComponent(id.uuidString + ".json") }

    static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        return e
    }

    static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        d.dateDecodingStrategy = .iso8601
        return d
    }

    /// Every readable conversation, newest first. A file that does not parse is skipped (and logged), never deleted.
    static func loadAll() -> [Conversation] {
        let fm = FileManager.default
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return [] }
        var out: [Conversation] = []
        for n in names where n.hasSuffix(".json") {
            let u = dir.appendingPathComponent(n)
            do {
                let f = try decoder().decode(ConversationFile.self, from: Data(contentsOf: u))
                out.append(f.conversation)
            } catch {
                log("CONVERSATION skipped unreadable \(n): \(error.localizedDescription)")
            }
        }
        return out.sorted { $0.updated > $1.updated }
    }

    /// Writes and deletes in order on one background queue (a write can wait seconds while the engine keeps the drive
    /// busy); `flush()` waits for them.
    static let io = DispatchQueue(label: "conversation.store", qos: .utility)
    static func flush() { io.sync {} }

    static func save(_ c: Conversation) {
        io.async { saveNow(c) }
    }

    static func saveNow(_ c: Conversation) {
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try encoder().encode(ConversationFile(c)).write(to: url(c.id), options: .atomic)
        } catch {
            log("CONVERSATION save failed \(c.id): \(error.localizedDescription)")
        }
    }

    static func delete(_ id: UUID) {
        io.async { try? FileManager.default.removeItem(at: url(id)) }
    }
}

/// The engine's saved states on disk (Settings > Speed): files and bytes in the selected model's folder, and the
/// engine's disk budget for them.
struct PrefixCacheUsage {
    var entries: Int
    var bytes: Int
    var capBytes: Double

    static func read() -> PrefixCacheUsage {
        var n = 0, b = 0
        if let e = FileManager.default.enumerator(at: DmengLaunch.diskDir, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) {
            for case let u as URL in e where u.lastPathComponent != "key.txt" {
                if let v = try? u.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), v.isRegularFile == true {
                    n += 1
                    b += v.fileSize ?? 0
                }
            }
        }
        return PrefixCacheUsage(entries: n, bytes: b, capBytes: 0.6e9)
    }
}

/// Splits Qwen's `<think>...</think>` block from the answer. While the block is open, everything is thought. A reply that
/// has `</think>` without the opening tag (the template put `<think>` in the prompt) is split at the closing tag; with
/// `thinkingOn` and no tag at all yet, all of it is thought (a reply still in its thinking block).
func splitThinking(_ s: String, thinkingOn: Bool = false) -> (String?, String) {
    if let open = s.range(of: "<think>") {
        let after = s[open.upperBound...]
        if let close = after.range(of: "</think>") {
            let thought = after[..<close.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
            let answer = after[close.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            return (thought, answer)
        }
        return (after.trimmingCharacters(in: .whitespacesAndNewlines), "")
    }
    if let close = s.range(of: "</think>") {
        return (s[..<close.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines),
                s[close.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines))
    }
    if thinkingOn { return (s.trimmingCharacters(in: .whitespacesAndNewlines), "") }
    return (nil, s)
}

/// A rough prompt-token count for the prefill progress bar (Qwen's tokenizer averages ~3.8 characters per token on
/// English prose; each message adds ~5 template tokens). Only used for the estimate; the engine's count is reported after.
func estimateTokens(_ text: String) -> Int { max(1, Int((Double(text.utf8.count) / 3.8).rounded(.up))) + 5 }
