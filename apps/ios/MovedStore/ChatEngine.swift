import Foundation

// The contract between the chat UI and the engine (DmengChatEngine): one event per generated token carrying the
// incremental detokenized piece and that token's stats, prefill progress before the first token, cancellation between
// tokens.

struct ChatMessage: Identifiable, Codable, Equatable, Sendable {
    /// `tool`: a tool's result (Tools.swift), rendered by the template as `<|im_start|>user\n<tool_response>\n…\n</tool_response>`.
    enum Role: String, Codable, Sendable { case system, user, assistant, tool }
    let id: UUID
    var role: Role
    var content: String

    init(id: UUID = UUID(), role: Role, content: String) {
        self.id = id
        self.role = role
        self.content = content
    }

    /// The chat-template input.
    var templateDict: [String: String] { ["role": role.rawValue, "content": content] }
}

/// Stats for one generated token.
struct TokenStats: Codable, Sendable, Equatable {
    var index: Int              // 0-based generated-token index
    var ms: Double              // wall time of this token
    var tokPerSec: Double       // 1000 / ms
    var mbRead: Double          // bytes read from flash for this token, in MB (1e6 B)
    var inMemory: Double        // fraction of the token's experts already in memory (0...1)
    var expertsInMemory: Int    // expert records in the engine's cache right now
    /// The engine's decode clock: seconds from the reply's first sample to the end of this token (prefill and the prefix
    /// restore excluded). (index + 1) / decodeSeconds is the reply's decode rate so far, the engine's tokens / seconds.
    var decodeSeconds: Double = 0
    /// MB (1e6 B) read from flash since the prefill ended (this reply's tokens so far).
    var mbTotal: Double = 0
    /// The token's place in the think block, by token id (ThinkPhase: "open", "thinking", "close", "answer").
    var phase: String? = nil
}

enum StopReason: String, Codable, Sendable { case eos, maxNew = "max_new", cancelled }

struct GenSummary: Codable, Sendable, Equatable {
    var reason: StopReason
    var tokens: Int
    var seconds: Double
    var meanTokPerSec: Double
    var meanMBPerToken: Double
}

/// The prompt prefill in progress: one before the first prefilled token and one after each.
struct PrefillTick: Sendable, Equatable {
    var kind: String          // "cold" | "live prefix" | "restored prefix"
    var promptTokens: Int     // the whole prompt
    var reusedTokens: Int     // restored or continued (not run)
    var total: Int            // tokens this prefill runs
    var done: Int             // of which run so far
    var mbRead: Double        // MB (1e6 B) read from flash by the prefill so far (the restore's re-land excluded)
    var relandMB: Double      // MB the prefix restore read (0 without one)
    var elapsed: Double       // seconds since the engine started the reply (template + restore + prefill so far)
    var prefillElapsed: Double // seconds of prefill so far (restore and state saves excluded)
}

enum GenEvent: Sendable {
    /// The prompt prefill's progress, before the first token.
    case prefill(PrefillTick)
    /// One token: the incremental detokenized text and its stats.
    case text(String, TokenStats)
    /// Always the last event of a stream, also after cancel().
    case finished(GenSummary)
}

struct EngineInfo: Sendable {
    var name: String
    var detail: String
}

/// What the engine knows about the last reply beyond GenSummary.
struct ReplyDetail: Codable, Sendable {
    var promptIds: [Int]
    var ids: [Int]
    var prefill: String               // "cold", "live prefix" or "restored prefix"
    var prefillTokens: Int
    var prefillSeconds: Double
    var seconds: Double
    var tokPerSec: Double
    var mibPerToken: Double           // MiB (1048576 B)
    var releasedTop8InMemory: Double
    var heldExpertsMean: Double
    var syncReads: Int
    /// The engine's per-token split for the reply: wait_ms_per_token, compute_ms_per_token, mb_issued_per_token, ...
    var split: [String: Double]?
    /// The prompt prefill before the first token: prefill_s, prefill_mb_read, prefill_reused, prefill_computed
    var prefillStats: [String: Double]?
}

protocol ChatEngine: AnyObject {
    var info: EngineInfo { get }
    /// Every setting in force, in display order (Settings > Diagnostics).
    var settings: [(String, String)] { get }
    /// The last finished reply's detail.
    var lastDetail: ReplyDetail? { get }
    /// Maps to the chat template's `enable_thinking`. Default off.
    var thinking: Bool { get set }

    /// Open the selected model (ModelKind) from its folder.
    func load(progress: @escaping (String) -> Void) async throws
    /// Stream one assistant reply. The stream always ends with `.finished`.
    func generate(messages: [ChatMessage], maxNew: Int) -> AsyncThrowingStream<GenEvent, Error>
    /// Stop the current reply after the token in flight. Safe to call from any thread, and when idle.
    func cancel()
    /// A new conversation (the engine keeps one live state and brings any prompt to it: nothing to forget).
    func forget()
    /// The engine keeps model states (in memory, and on disk for the system prompt and each chat's last turn) and restores
    /// the longest one a prompt starts with. Settings toggle; default on.
    var prefixCacheEnabled: Bool { get set }
    /// One line for Diagnostics.
    var prefixCacheStatus: String { get }
    /// Delete every saved state.
    func clearPrefixCache() async
    /// Returns when the engine has finished its queued work.
    func waitIdle() async
}

enum EngineError: LocalizedError {
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .notLoaded: return "engine not loaded"
        }
    }
}

// MARK: token counts for the tool loop

extension ChatEngine {
    /// Prompt tokens of `messages` (string messages, generation prompt added) by the engine's own tokenizer and template;
    /// nil before load.
    func promptTokenCount(_ messages: [[String: String]], thinking: Bool) -> Int? {
        (self as? DmengChatEngine)?.tokenCount(messages, thinking: thinking)
    }

    /// Tokens of a tool result as the model reads it (the content between `<tool_response>\n` and `\n</tool_response>`):
    /// the render with the result minus the render with an empty one. nil before load.
    func toolResultTokens(_ result: String) -> Int? {
        let q = ["role": "user", "content": "q"]
        guard let a = promptTokenCount([q, ["role": "tool", "content": result]], thinking: false),
              let b = promptTokenCount([q, ["role": "tool", "content": ""]], thinking: false) else { return nil }
        return a - b
    }
}

/// The chat-template input. An earlier assistant reply that was cut inside its think block (max tokens or stop) gets the
/// template's close ("\n</think>\n\n") so the template drops its reasoning like any earlier turn's (the release's
/// template strips reasoning only when `</think>` is present; left unclosed, the whole truncated trace would be fed back
/// as the turn's answer).
/// History mode `live` (HistoryMode, ChatModel.swift): a thinking-off reply (no think tags in its text) is given back
/// as the model generated it, after the generation prompt's empty block "<think>\n\n</think>\n\n", with
/// `reasoning_content` set so the template neither splits nor strips it: the rendered history is then the engine's own
/// fed tokens and the next prompt continues the live state. `strict`: the template's bytes.
func templateMessages(_ messages: [ChatMessage], mode: HistoryMode = .strict) -> [[String: String]] {
    messages.filter { !($0.role == .assistant && $0.content.isEmpty) }.map { m in
        var d = m.templateDict
        guard m.role == .assistant else { return d }
        if let o = m.content.range(of: "<think>"), m.content[o.upperBound...].range(of: "</think>") == nil {
            d["content"] = m.content + (m.content.hasSuffix("\n") ? "</think>\n\n" : "\n</think>\n\n")
        } else if mode == .live, !m.content.contains("<think>"), !m.content.contains("</think>") {
            d["content"] = "<think>\n\n</think>\n\n" + m.content
            d["reasoning_content"] = ""
        }
        return d
    }
}

/// Sampling and the thinking budget for the app's replies.
///
/// UserDefaults keys (Settings writes them; absent = the model's default from generation_config.json):
///   gen.greedy (Bool)                 true: argmax; default false
///   gen.temperature, gen.topP, gen.topK, gen.minP, gen.repetitionPenalty, gen.presencePenalty
///   gen.seed (Int)                    absent: a fresh random seed per reply
///   gen.thinkingBudget (Int)          0 / absent: none; N: after N tokens inside the think block its close is forced
struct GenSettings {
    var sampling: SamplingParams
    var thinkingBudget: Int?
    var greedyBecause: String?

    static func current(defaults: SamplingParams, store: UserDefaults = .standard) -> GenSettings {
        func num(_ k: String) -> NSNumber? { store.object(forKey: k) as? NSNumber }
        let budget = num("gen.thinkingBudget")?.intValue
        let tb = (budget ?? 0) > 0 ? budget : nil
        if store.bool(forKey: "gen.greedy") { return GenSettings(sampling: .greedy, thinkingBudget: tb, greedyBecause: "gen.greedy") }
        var p = defaults
        if let v = num("gen.temperature") { p.temperature = v.floatValue }
        if let v = num("gen.topP") { p.topP = v.floatValue }
        if let v = num("gen.topK") { p.topK = v.intValue }
        if let v = num("gen.minP") { p.minP = v.floatValue }
        if let v = num("gen.repetitionPenalty") { p.repetitionPenalty = v.floatValue }
        if let v = num("gen.presencePenalty") { p.presencePenalty = v.floatValue }
        if let v = num("gen.seed").map({ UInt64(truncatingIfNeeded: $0.int64Value) }) {
            p.seed = v
        } else {
            p.seed = UInt64.random(in: 0..<(1 << 32))
        }
        return GenSettings(sampling: p, thinkingBudget: tb, greedyBecause: nil)
    }

    var label: String {
        if let why = greedyBecause { return "greedy (\(why))" }
        let seedFixed = UserDefaults.standard.object(forKey: "gen.seed") != nil
        let s = sampling.summary
        return seedFixed ? s : s.replacingOccurrences(of: ", seed \(sampling.seed)", with: ", seed random per reply")
    }
}
