import Foundation
import SwiftUI
import UIKit

enum EngineState: Equatable {
    case idle
    case loading
    case ready
    case refused(String)
    case failed(String)

    var label: String {
        switch self {
        case .idle: return "idle"
        case .loading: return "loading"
        case .ready: return "ready"
        case .refused(let s): return "refused: " + s
        case .failed(let s): return "failed: " + s
        }
    }
}

/// One finished reply (a round of the tool loop), for the logs.
struct ReplyRecord: Codable {
    var prompt: String
    var thinking: Bool
    var tokens: Int
    var reason: String
    var seconds: Double
    var meanTokPerSec: Double
    var meanMBPerToken: Double
    var firstTokenMs: Double?
    var cancelRequestedAtToken: Int?
    var text: String
    var error: String?
    var maxNew: Int = 0
    /// The generated token ids (store-backed engines), for offline identity checks.
    var ids: [Int]?
    var promptTokens: Int?
    /// The engine's per-token split for this reply (ReplyDetail.split).
    var split: [String: Double]?
    /// The prompt prefill before the first token, and the first token itself (ReplyDetail.prefillStats).
    var prefill: [String: Double]?
    /// Memory warnings received while this reply was in progress (prefill included).
    var memoryWarnings: Int = 0
    /// "cold", "live prefix" or "restored prefix" (ReplyDetail.prefill).
    var prefillKind: String?
    /// What the UI showed at the end of the reply: the speed (GenSpeed.rate over the message's rounds so far), the time to
    /// the first token (send -> first piece, wall), MB per token (1e6 B) since the prefill.
    var uiDecodeTokPerSec: Double?
    var uiTtftMs: Double?
    var uiMBPerToken: Double?
    /// The engine counters' rate over the same rounds (dm_counters: decode steps / time inside dm_step) and the shown
    /// speed's difference from it, in percent.
    var engineTokPerSec: Double?
    var uiRateDiffPct: Double?
    /// The round's wall time, generate call to the stream's end (tool rounds).
    var wallMs: Double?
}

/// The app's own settings (UserDefaults). The sampling keys `gen.*` are written here and read by the engine wrapper.
enum AppSettings {
    static let systemPromptKey = "chat.systemPrompt"
    static let defaultThinkingKey = "chat.defaultThinking"
    static let maxTokensKey = "chat.maxTokens"
    static let prefillSecPerTokenKey = "chat.prefillSecPerToken"
    static let restoreSecKey = "chat.restoreSec"
    static let liveContinuationKey = "chat.liveContinuationWorks"

    /// The default system prompt for new conversations: short (it is prefilled once at launch and restored after), and it
    /// asks for brevity because replies stream at ~3 tokens per second on the 15 Plus.
    static let defaultSystemPrompt = "You are a helpful assistant running entirely on this iPhone. Replies stream at about three words per second, so be concise: lead with the answer, use short paragraphs or lists, and skip filler."

    static var d: UserDefaults { .standard }
    /// The setting, else the default above.
    static var systemPrompt: String { d.string(forKey: systemPromptKey) ?? defaultSystemPrompt }
    static var defaultThinking: Bool { d.object(forKey: defaultThinkingKey) as? Bool ?? false }
    /// "Read my message while I type" (type-ahead: the draft's finished words are read in the background, so a sent
    /// message only computes its last word). On by default.
    static let typeAheadKey = "chat.typeAhead"
    static var typeAhead: Bool { d.object(forKey: typeAheadKey) as? Bool ?? true }
    static var maxTokens: Int { (d.object(forKey: maxTokensKey) as? Int).map { max(16, $0) } ?? 1024 }
    /// Seconds per prompt token for the prefill estimate (learned from the replies' prefill stats).
    static var prefillSecPerToken: Double { d.object(forKey: prefillSecPerTokenKey) as? Double ?? 0.27 }
    static var restoreSec: Double { d.object(forKey: restoreSecKey) as? Double ?? 2.0 }
    /// Whether the last follow-up in a live conversation continued the engine's state ("live prefix").
    static var liveContinuationWorks: Bool { d.object(forKey: liveContinuationKey) as? Bool ?? false }

    // Sampling (written by Settings, read by GenSettings; absent = the model's defaults below)
    static let temperatureKey = "gen.temperature", topPKey = "gen.topP", topKKey = "gen.topK", minPKey = "gen.minP"
    static let repetitionPenaltyKey = "gen.repetitionPenalty", greedyKey = "gen.greedy"
    static let presencePenaltyKey = "gen.presencePenalty", thinkingBudgetKey = "gen.thinkingBudget"
    static let modelTemperature = 1.0, modelTopP = 0.95, modelTopK = 20, modelMinP = 0.0, modelRepetitionPenalty = 1.0
    static let modelPresencePenalty = 0.0
    static let samplingKeys = [temperatureKey, topPKey, topKKey, minPKey, repetitionPenaltyKey, greedyKey, presencePenaltyKey]
}

/// How earlier assistant turns go into the chat template (templateMessages).
/// - live (the default): a thinking-off reply keeps the empty `<think>\n\n</think>\n\n` block it was generated after,
///   so the history is the model's own tokens and a follow-up continues the engine's live state (only the new turn is
///   prefilled); a thinking reply's reasoning is dropped as the template does.
/// - strict: exactly the template's bytes (the reasoning and the empty block stripped from earlier turns).
enum HistoryMode: String, CaseIterable {
    case live, strict
    static let key = "chat.historyMode"
    static var current: HistoryMode { HistoryMode(rawValue: UserDefaults.standard.string(forKey: key) ?? "") ?? .live }
    var label: String {
        switch self {
        case .live: return "live: thinking-off turns keep their empty think block"
        case .strict: return "strict template: the template's bytes"
        }
    }
}

/// The reply's pending UI update (ChatModel.runRound): text and stats not yet shown, flushed at most every `interval`.
@MainActor final class UIBatch {
    static let interval: TimeInterval = 0.1
    var text = ""
    var stats: TokenStats?
    var n = 0
    var mbSum = 0.0
    var last: TimeInterval = 0       // the last flush (system uptime)
    var lastTick: TimeInterval = 0   // the last published prefill tick
    var scheduled = false
}

/// The app's start-up as the chat screen shows it.
enum StartupPhase: Equatable {
    case checking        // probing memory and the model files
    case loading         // engine load (resident weights, routers, slabs)
    case warming         // a system-prompt state prefilled into the prefix cache (cold: once per prompt; the status line says so)
    case ready
    case unavailable     // refused or failed (engineState says why)
}

/// The prefill of the reply in progress: an estimate from the text until the engine's first progress event, then the
/// engine's own figures (`tick`: tokens run of the tokens to run, MB read, seconds).
struct PrefillProgress: Equatable {
    var start: Date
    var expectedTokens: Int
    var expectedSeconds: Double
    var restoring: Bool       // a cached prefix is expected to be restored first
    var waitingForWarmup: Bool
    /// What a message sent before the engine was ready waits for ("the model to load", "the one-time setup", "the system
    /// prompt"); nil when it does not wait.
    var waitingFor: String? = nil
    /// The engine's latest prefill event (nil until the first arrives).
    var tick: PrefillTick? = nil
    /// When `tick` arrived (the card extrapolates the elapsed time from it).
    var tickAt: Date? = nil
}

@MainActor
final class ChatModel: ObservableObject {
    /// The open conversation (a draft until its first message, then also in `conversations` and on disk).
    @Published var conv = Conversation()
    /// Saved conversations, newest first (Documents/conversations/).
    @Published var conversations: [Conversation] = []
    /// The open conversation's messages: what the engine is given (a leading system message included).
    var messages: [ChatMessage] {
        get { conv.messages }
        set { conv.messages = newValue }
    }
    @Published var isGenerating = false
    @Published var engineState: EngineState = .idle
    @Published var startup: StartupPhase = .checking
    @Published var loadLog: [String] = []
    @Published var thermal = thermalName(ProcessInfo.processInfo.thermalState)
    /// Thinking for the next reply (the chat template's enable_thinking); per conversation in the UI.
    @Published var thinking = false {
        didSet {
            engine.thinking = thinking
            if conv.thinking != thinking { updateConv(conv.id) { $0.thinking = thinking } }
        }
    }
    // the reply in progress, for the views
    /// The conversation and assistant message being generated (nil when idle).
    @Published var generatingConvID: UUID?
    @Published var streamingID: UUID?
    /// Cold-prefill estimate while waiting for the first token (nil once it arrives).
    @Published var prefill: PrefillProgress?
    /// When the reply's thinking block started and how long it took (live; saved in the message's meta at the end).
    @Published var thinkStart: Date?
    @Published var liveThinkingSeconds: Double?
    @Published var liveTokens = 0
    /// The latest memory reading for the status dots (taken every few seconds without logging) and the last warning's time.
    @Published var liveMem: MemSample?
    @Published var lastMemoryWarning: Date?
    /// The system contents (the tools block included when on) whose state is in the prefix cache from this launch's
    /// warm-ups: with tools on and off the system turn differs, so each mode has its own cached state.
    private(set) var warmedSystems: Set<String> = []
    private var warmTask: Task<Void, Never>?
    /// The launch (the engine load), awaited by a message sent before it ends.
    private var launchTask: Task<Void, Never>?
    @Published private(set) var launchPending = false
    /// The system content whose state the engine's slab holds right now: set by a warm-up (its records landed), cleared
    /// as soon as any reply runs (its prefill and decode replace them). A new chat whose system content is cached but not
    /// in the slab is warmed again in the background (newConversation): a restore of the cached state, no new tokens.
    private var slabSystem: String?
    /// The conversation whose state the engine holds live (its last reply), for the prefill estimate.
    private var liveConvID: UUID?
    /// The prefix cache toggle (settings; default on, remembered).
    @Published var prefixCache = true {
        didSet {
            engine.prefixCacheEnabled = prefixCache
            UserDefaults.standard.set(prefixCache, forKey: "prefixCache")
            prefixStatus = engine.prefixCacheStatus
        }
    }
    @Published var prefixStatus = ""

    // live stats. replyTokPerSec: the one speed figure (GenSpeed.rate: tokens generated / the engine's generating time,
    // over every round of the message so far; the prompt, tool rounds and background time excluded). The live line, the
    // caption, the status line, Diagnostics and the Copy text all read it (or replySpeed, its parts).
    @Published var lastTokPerSec: Double = 0
    @Published private(set) var replyTokPerSec: Double = 0
    /// The figure's parts: the message's rounds so far (live) / at the end of its last round.
    private(set) var replySpeed = GenSpeed()
    /// The finished rounds of the message being generated: the shown speed's and the engine counters' (RATE line).
    private var messageSpeed = GenSpeed(), messageEngine = GenSpeed(), messageRounds = 0
    private func showSpeed(_ sp: GenSpeed) {
        replySpeed = sp
        replyTokPerSec = sp.rate
    }
    /// The engine's "Last reply in detail" rows with its Speed row as the message's (the one figure).
    var lastReplyRows: [(String, String)] {
        engine.lastReplyRows.map { $0.0 == "Speed" ? ("Speed", replySpeed.detail) : $0 }
    }
    @Published var mbPerToken: Double = 0
    /// Time to the first token of the reply in progress (send -> first piece, wall ms); nil until it arrives.
    @Published var replyTtftMs: Double?
    /// The last finished reply: time to first token, prefill kind, prefix restore (re-land MB / ms), MB per token.
    @Published var lastTtftMs: Double?
    @Published var lastPrefillKind: String?
    @Published var inMemory: Double = 0
    @Published var expertsInMemory: Int = 0
    @Published var lastSummary: GenSummary?
    // diagnostics
    @Published var latestMem: MemSample?
    @Published var lastSplit: [String: Double]?
    @Published var lastPrefill: [String: Double]?
    @Published var memoryWarnings = 0
    private var replyNo = 0          // replies started (1-based number of the one in progress)
    private var replyTokens = 0      // tokens of the reply in progress
    private var inReply = false
    // the tool loop: the call being run (its assistant message) and its query, for the card and status line
    @Published var toolRunningID: UUID?
    @Published var toolRunningQuery: String?
    private var userStopped = false
    private var toolTask: Task<(text: String, meta: ToolMeta), Never>?
    private(set) var lastReply: ReplyRecord?

    let engine: ChatEngine
    private var genTask: Task<Void, Never>?
    private var started = false
    private var events: [String] = []

    init() {
        #if UITEST
        engine = UITest.active ? FakeChatEngine() : DmengChatEngine()
        #else
        engine = DmengChatEngine()
        #endif
        engine.thinking = false
        let pref = (UserDefaults.standard.object(forKey: "prefixCache") as? Bool) ?? true
        prefixCache = pref
        engine.prefixCacheEnabled = pref
        // saved conversations, and a new draft with the settings' system prompt and thinking default
        conversations = ConversationStore.loadAll()
        conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
        thinking = conv.thinking  // (no didSet in init)
        engine.thinking = thinking
        _ = appStart
        log("LAUNCH model \(ModelKind.selected.displayName), profile \(DmengLaunch.profile), mode \(DmengLaunch.userMode)")
        memSample("launch")
        let nc = NotificationCenter.default
        nc.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main) { [weak self] _ in
            let t = thermalName(ProcessInfo.processInfo.thermalState)
            Task { @MainActor in self?.thermal = t; self?.event("thermal -> \(t)") }
        }
        nc.addObserver(forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: .main) { [weak self] _ in
            // advisory: a reply is never stopped by one (the engine shrinks its cache when memory is really short)
            Task { @MainActor in self?.memoryWarning() }
        }
    }

    func event(_ s: String) {
        events.append(s)
        log("EVENT " + s)
    }

    /// Take a memory sample: console line and Diagnostics.
    @discardableResult
    func memSample(_ label: String) -> MemSample {
        let m = MemProbe.read(label)
        latestMem = m
        log(m.line)
        return m
    }

    func memoryWarning() {
        memoryWarnings += 1
        lastMemoryWarning = Date()
        let wh = inReply ? "reply \(replyNo), token \(replyTokens)" : (engineState == .loading ? "during load" : "idle")
        let m = memSample("memory warning #\(memoryWarnings) (\(wh))")
        event(String(format: "memory warning (%@; footprint %.3f GB, available %.3f GB); not stopped", wh,
                     Double(m.physFootprint) / 1e9, Double(m.osProcAvailable) / 1e9))
    }

    // MARK: launch

    func startOnce() {
        if started { return }
        started = true
        Task { await launchApp() }
    }

    /// The app's launch: the chat is shown at once and the engine loads in the background; then the open draft's system
    /// state is brought in during idle time. A message sent meanwhile waits for the load (reply awaits launchTask).
    func launchApp() async {
        startup = .checking
        launchPending = true
        let t0 = Date()
        let t = Task { @MainActor in
            await self.load()
            if self.engineState == .ready { Task { await self.warmUp() } }
            log(String(format: "LAUNCH ready in %.2f s: engine %@", Date().timeIntervalSince(t0), self.engineState.label))
        }
        launchTask = t
        await t.value
        launchPending = false
    }

    /// Settings closed: a changed system prompt or tools definition gets its state (in the background).
    func settingsClosed() {
        guard engineState == .ready, warmTask == nil, !isGenerating, !launchPending else { return }
        Task { await self.warmUp() }
    }

    /// Whether a message can be sent now: the engine is ready, or still loading at launch (the message then waits).
    var acceptsMessages: Bool {
        if isGenerating || readOnlyModel != nil { return false }
        switch engineState {
        case .ready: return true
        case .idle, .loading: return launchPending
        default: return false
        }
    }

    func load() async {
        if engineState == .loading { return }
        loadLog = []
        defer { startup = engineState == .ready ? .ready : (engineState == .loading ? .loading : .unavailable) }
        engineState = .loading
        startup = .loading
        do {
            try await engine.load { [weak self] s in
                log("LOAD " + s)
                Task { @MainActor in self?.loadLog.append(s) }
            }
            engineState = .ready
            log("LOAD ready: " + engine.info.detail)
            engine.prefixCacheEnabled = prefixCache
            prefixStatus = engine.prefixCacheStatus
            for (k, v) in engine.settings { log("SETTING \(k) = \(v)") }
            memSample("after load")
        } catch {
            engineState = .failed(error.localizedDescription)
            log("LOAD failed: \(error.localizedDescription) [\(String(describing: error))]")
        }
    }

    // MARK: chat

    /// The send button. `maxNew`: the reply's token limit (default: the settings' longest reply).
    func send(_ text: String, maxNew: Int? = nil) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, acceptsMessages else { return }
        let mx = maxNew ?? AppSettings.maxTokens
        draftTask?.cancel()
        draftFullTask?.cancel()
        draftStable = ""
        genTask = Task { _ = await self.reply(to: t, maxNew: mx) }
    }

    func stop() {
        userStopped = true
        toolTask?.cancel()
        engine.cancel()
    }

    /// Tools on or off for the open chat (the globe button): the chat's override; the system state of the new mode is
    /// warmed in the background so the next message restores it.
    func setTools(_ on: Bool) {
        guard !isGenerating else { return }
        updateConv(conv.id) { $0.tools = on == ToolSettings.enabled ? nil : on }
        persist(conv.id)
        Task { await warmUp() }
    }

    func clearPrefixCache() {
        Task {
            await engine.clearPrefixCache()
            prefixStatus = engine.prefixCacheStatus
            warmedSystems.removeAll()
            slabSystem = nil
            event("prefix cache cleared")
            // keep the promise that a first message only prefills the user turn: the system state again (background)
            await warmUp()
        }
    }

    // MARK: conversations

    /// A new draft with the settings' system prompt and thinking default. The engine keeps its state: the next reply
    /// decides (a prompt that does not continue it restores the longest saved prefix, i.e. the system prompt's). The new
    /// chat's system state is brought in again in the background while the user types; a message sent before it finishes
    /// waits for it (reply awaits `warmTask`), as at launch.
    func newConversation() {
        if !conv.hasUserMessage && conv.systemPrompt == AppSettings.systemPrompt && conv.model == DmengLaunch.kind { prepareNewChat(); return }  // already a fresh draft
        conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
        thinking = conv.thinking
        prepareNewChat()
    }

    /// The open draft's system state into the slab in the background (see newConversation); no-op when it is there already,
    /// when a reply is running (reply() calls this again when it ends) or a warm-up is in progress.
    func prepareNewChat() {
        guard !conv.hasUserMessage else { return }
        Task { await warmUp() }
    }

    func open(_ id: UUID) {
        guard conv.id != id, let c = conversations.first(where: { $0.id == id }) else { return }
        conv = c
        thinking = c.thinking
        if readOnlyModel == nil { warmChat(id, why: "chat opened") }
    }

    /// The open chat was made with the other model (ModelKind): it opens read-only (the notice above the composer says
    /// which model to choose in Settings); nil when it can be continued.
    var readOnlyModel: ModelKind? {
        guard conv.hasUserMessage, conv.model != DmengLaunch.kind else { return nil }
        return conv.model
    }

    // MARK: two models (ModelKind)

    /// Settings > Model: close the open model (the engine's queue drained, dm_close, its memory released), open `k` from
    /// its own folder with its own profile, options and saved states, and start a new chat with it. True when it loaded.
    @discardableResult
    func switchModel(to k: ModelKind) async -> Bool {
        guard let d = engine as? DmengChatEngine else { return false }
        if d.loadedKind == k && engineState == .ready { return true }
        stop()
        await genTask?.value
        if let w = warmTask { await w.value }
        draftTask?.cancel()
        draftFullTask?.cancel()
        let from = d.loadedKind
        let m0 = memSample("model switch: before closing \(from?.rawValue ?? "none")")
        engineState = .loading
        startup = .loading
        await d.unload()
        let m1 = memSample("model switch: \(from?.rawValue ?? "none") closed")
        ModelKind.selected = k
        liveConvID = nil
        slabSystem = nil
        warmedSystems.removeAll()
        engineState = .idle
        let t0 = Date()
        await load()
        let m2 = memSample("model switch: \(k.rawValue) open")
        log(String(format: "MODEL SWITCH %@ -> %@: footprint %.2f GB before, %.2f GB after closing (%.2f GB released), %.2f GB with %@ open (%.1f s to open); engine %@",
                   from?.rawValue ?? "none", k.rawValue, Double(m0.physFootprint) / 1e9, Double(m1.physFootprint) / 1e9,
                   Double(m0.physFootprint - m1.physFootprint) / 1e9, Double(m2.physFootprint) / 1e9, k.rawValue,
                   Date().timeIntervalSince(t0), engineState.label))
        if !conv.hasUserMessage || conv.model != k {
            conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
            thinking = conv.thinking
        }
        guard engineState == .ready else { return false }
        Task { await self.warmUp() }
        return true
    }

    /// Type-ahead: the composer's text changed. When the draft's stable part (everything before its last word)
    /// changes, it is synced in the background (DmengChatEngine.typeAhead): at once at a word boundary or a paste, else
    /// after 300 ms without a keystroke. Off with the setting, in Low Power Mode, when the phone is hot (serious or
    /// critical), while a reply runs, and while a sent message is being edited (the caller passes nothing then).
    private var draftTask: Task<Void, Never>?, draftFullTask: Task<Void, Never>?
    private var draftStable = ""
    func draftChanged(_ text: String) {
        guard let d = engine as? DmengChatEngine, AppSettings.typeAhead, engineState == .ready, prefixCache, !isGenerating,
              readOnlyModel == nil else { return }
        d.draftEdited()
        draftFullTask?.cancel()
        if ProcessInfo.processInfo.isLowPowerModeEnabled { return }
        if ProcessInfo.processInfo.thermalState.rawValue >= ProcessInfo.ThermalState.serious.rawValue { return }
        let t = String(text.drop(while: { $0.isWhitespace }))
        let id = conv.id
        let dateNote = ToolSettings.dateNote()
        func msgs(_ c: Conversation, _ content: String) -> [ChatMessage] {
            engineMessages(c.messages + [ChatMessage(role: .user, content: content, modelNote: dateNote)], tools: c.toolsOn)
        }
        // after a pause (300 ms without a keystroke): the whole prompt as it would be sent now, with its logits
        let whole = t.trimmingCharacters(in: .whitespacesAndNewlines)
        if !whole.isEmpty {
            draftFullTask = Task {
                try? await Task.sleep(nanoseconds: 300_000_000)
                guard !Task.isCancelled, !self.isGenerating, let c = self.conversation(id) else { return }
                let a = msgs(c, whole + " a"), b = msgs(c, whole + " b"), f = msgs(c, whole)
                let sent = f.last(where: { $0.role == .user })?.content ?? whole
                await Task.detached(priority: .userInitiated) { d.typeAhead(a, b, full: f, text: sent) }.value
            }
        }
        guard let r = t.rangeOfCharacter(from: .whitespacesAndNewlines, options: .backwards) else { return }
        let stable = String(t[..<r.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stable.isEmpty, stable != draftStable else { return }
        let boundary = text.last?.isWhitespace == true || stable.count > draftStable.count + 20  // a word just ended, or a paste
        draftStable = stable
        draftTask?.cancel()
        draftTask = Task {
            if !boundary { try? await Task.sleep(nanoseconds: 300_000_000) }
            guard !Task.isCancelled, !self.isGenerating, let c = self.conversation(id) else { return }
            let a = msgs(c, stable + " a"), b = msgs(c, stable + " b")
            await Task.detached(priority: .userInitiated) { d.typeAhead(a, b, text: stable) }.value
        }
    }

    /// Idle-time work on a chat (opened, or a reply just ended): the engine's live state brought to the chat
    /// as the next message will render it, up to the next user header, and kept on disk there (DmengChatEngine.warm); the
    /// next message then computes only its own text. Queued behind any warm-up in progress; a message sent meanwhile
    /// waits for it (reply awaits warmTask).
    func warmChat(_ id: UUID, why: String) {
        guard let d = engine as? DmengChatEngine, engineState == .ready, prefixCache, !isGenerating,
              let c = conversation(id), c.hasUserMessage, c.model == d.loadedKind else { return }
        // rendered with an (empty) next user message, then without it: earlier turns exactly as the next request renders them
        let msgs = Array(engineMessages(c.messages + [ChatMessage(role: .user, content: "")], tools: c.toolsOn).dropLast())
        let prev = warmTask
        let t = Task { await prev?.value; await d.warm(msgs, why: why) }
        warmTask = t
        Task { await t.value; if self.warmTask == t { self.warmTask = nil } }
    }

    func delete(_ id: UUID) {
        if generatingConvID == id { stop() }
        conversations.removeAll { $0.id == id }
        ConversationStore.delete(id)
        if liveConvID == id { liveConvID = nil }
        if conv.id == id {
            conv = Conversation(system: AppSettings.systemPrompt, thinking: AppSettings.defaultThinking)
            thinking = conv.thinking
        }
    }

    /// An empty title goes back to the one taken from the first message.
    func rename(_ id: UUID, to title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        updateConv(id) { c in
            if t.isEmpty {
                c.titleIsCustom = false
                c.title = Conversation.title(from: c.messages.first { $0.role == .user }?.content ?? "")
            } else {
                c.title = t
                c.titleIsCustom = true
            }
        }
        persist(id)
    }

    /// Drop the last reply (and the message it answered) and send that message again.
    func regenerate() {
        guard !isGenerating, engineState == .ready, readOnlyModel == nil,
              let a = conv.messages.lastIndex(where: { $0.role == .assistant }),
              let u = conv.messages[..<a].lastIndex(where: { $0.role == .user }) else { return }
        let text = conv.messages[u].content
        let gone = conv.messages[u...].map(\.id)
        updateConv(conv.id) { c in
            c.messages.removeSubrange(u...)
            for id in gone { c.meta[id] = nil }
        }
        send(text)
    }

    /// Edit a sent message: the chat is cut back to before it (like regenerate), then the new text is sent. The engine
    /// restores the longest cached state of the shortened prompt (the previous reply's end, or the system prompt), so
    /// only the new message is read.
    func editAndResend(_ id: UUID, text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !isGenerating, acceptsMessages,
              let u = conv.messages.firstIndex(where: { $0.id == id && $0.role == .user }) else { return }
        let gone = conv.messages[u...].map(\.id)
        log("EDIT message \(u) of \(conv.messages.count): the chat is cut back and the new text sent")
        updateConv(conv.id) { c in
            c.messages.removeSubrange(u...)
            for g in gone { c.meta[g] = nil }
            if !c.titleIsCustom && !c.hasUserMessage { c.title = "" }   // the first message: the title follows the edit
        }
        send(t)
    }

    /// Apply `f` to conversation `id` wherever it is: the open one and the saved list (kept in step).
    func updateConv(_ id: UUID, _ f: (inout Conversation) -> Void) {
        if conv.id == id { f(&conv) }
        if let i = conversations.firstIndex(where: { $0.id == id }) {
            if conv.id == id { conversations[i] = conv } else { f(&conversations[i]) }
        }
    }

    func conversation(_ id: UUID) -> Conversation? { conv.id == id ? conv : conversations.first { $0.id == id } }

    /// Write conversation `id` to Documents/conversations/.
    func persist(_ id: UUID) {
        guard let c = conversation(id), c.hasUserMessage else { return }
        ConversationStore.save(c)
    }

    // MARK: warm-up: the system prompt's state in idle time

    /// The open chat's system content (the tools block included when tools are on) brought into the engine's state and
    /// kept on disk, so a first message computes only its own turn (DmengChatEngine.warmSystem). Called at launch, for a
    /// new chat, when Settings closes and when the globe button changes the tools.
    func warmUp() async {
        let system = effectiveSystem(conv).trimmingCharacters(in: .whitespacesAndNewlines)
        guard engineState == .ready, prefixCache, !system.isEmpty, !isGenerating, warmTask == nil,
              let d = engine as? DmengChatEngine else { return }
        let t = Task { await d.warmSystem(system) }
        warmTask = t
        await t.value
        warmTask = nil
    }

    // MARK: prefill estimate (until the engine's first progress event)

    /// What the next reply's prefill should cost: the tokens the engine will have to run (an estimate from the text) at
    /// the measured seconds per prompt token, plus a prefix-cache restore when one is expected.
    func expectedPrefill(history: [ChatMessage], convID: UUID, waiting: Bool, waitingFor: String? = nil) -> PrefillProgress {
        let sys = history.first?.role == .system ? history[0].content : ""
        let rest = history.filter { $0.role != .system }
        var tokens = 0
        var restoring = false
        if liveConvID == convID && AppSettings.liveContinuationWorks, let u = rest.last {
            tokens = estimateTokens(u.content) + 6          // the new turn only: the engine continues its live state
        } else {
            tokens = rest.map { estimateTokens($0.content) }.reduce(0, +) + 4
            if !sys.isEmpty {
                if prefixCache && warmedSystems.contains(sys) { restoring = slabSystem != sys } else { tokens += estimateTokens(sys) }
            }
        }
        let secs = Double(tokens) * AppSettings.prefillSecPerToken + (restoring ? AppSettings.restoreSec : 0) + 0.5
        return PrefillProgress(start: Date(), expectedTokens: tokens, expectedSeconds: secs, restoring: restoring,
                               waitingForWarmup: waiting, waitingFor: waiting ? (waitingFor ?? "the system prompt") : nil)
    }

    /// Keep the estimate's two figures measured: seconds per prefilled prompt token and the restore time.
    private func learnTiming(_ d: ReplyDetail) {
        guard let p = d.prefillStats else { return }
        if let n = p["prefill_tokens"], let s = p["prefill_s"], n >= 8, s > 0 {
            let v = 0.5 * AppSettings.prefillSecPerToken + 0.5 * (s / n)
            UserDefaults.standard.set(v, forKey: AppSettings.prefillSecPerTokenKey)
        }
        if let r = p["prefix_restore_ms"], (p["prefix_hit"] ?? 0) > 0 {
            UserDefaults.standard.set(0.5 * AppSettings.restoreSec + 0.5 * r / 1000, forKey: AppSettings.restoreSecKey)
        }
    }

    /// The status dots' memory reading (no console line, no mem.jsonl entry).
    func refreshLiveMemory() { liveMem = MemProbe.read("status") }

    // MARK: status line (under the composer)

    enum Level: String { case good, fair, serious, critical, unknown }

    /// Memory headroom before iOS may close the app: os_proc_available_memory (iOS only; the Mac has no per-app limit).
    /// good >= 0.7 GB, fair >= 0.3 GB or a warning in the last 2 minutes, critical below. Warnings are advisory.
    var memoryLevel: Level {
        guard let m = liveMem ?? latestMem, m.osProcAvailable > 0 else { return .unknown }
        let recentWarning = lastMemoryWarning.map { Date().timeIntervalSince($0) < 120 } ?? false
        if m.osProcAvailable < 300_000_000 { return .critical }
        if m.osProcAvailable < 700_000_000 || recentWarning { return .fair }
        return .good
    }

    var thermalLevel: Level {
        switch thermal {
        case "nominal": return .good
        case "fair": return .fair
        case "serious": return .serious
        case "critical": return .critical
        default: return .unknown
        }
    }

    /// One line: the last reply's decode rate and time to first token; the reply in progress while generating.
    var statusLine: String {
        if isGenerating {
            if let q = toolRunningQuery { return "Searching: " + q }
            if replyTtftMs == nil {
                if prefill?.waitingForWarmup == true { return "Waiting for " + (prefill?.waitingFor ?? "the system prompt") }
                if let t = prefill?.tick, t.total > 0 { return "Reading the prompt · \(t.done) of \(t.total) tokens" }
                return "Reading the prompt"
            }
            return String(format: "%d tokens · %.1f tok/s", liveTokens, replyTokPerSec)
        }
        if let s = lastSummary, s.tokens > 0 {
            var parts = [String(format: "%.1f tok/s", replyTokPerSec)]
            if let t = lastTtftMs { parts.append(String(format: "first token %.1f s", t / 1000)) }
            return parts.joined(separator: " · ")
        }
        switch startup {
        case .checking, .loading: return "Loading model…"
        case .warming: return "Preparing the system prompt (once)…"
        case .ready: return "Ready"
        case .unavailable: return "Model unavailable"
        }
    }

    /// What the status line shows: the same state as `statusLine` in plain words.
    var statusText: String {
        if isGenerating {
            if let q = toolRunningQuery { return q.hasPrefix("http") ? "Reading a web page…" : "Searching the web…" }
            if replyTtftMs == nil {
                if prefill?.waitingForWarmup == true {
                    return prefill?.waitingFor == "the one-time setup" ? "Sending when setup finishes" : "Sending when the model is ready"
                }
                return "Reading your message…"
            }
            return replyTokPerSec > 0 ? String(format: "Replying · %.1f tokens/s", replyTokPerSec) : "Replying…"
        }
        if let s = lastSummary, s.tokens > 0 {
            var parts = [String(format: "%.1f tokens/s", replyTokPerSec)]
            if let t = lastTtftMs { parts.append(String(format: "started in %.1f s", t / 1000)) }
            return parts.joined(separator: " · ")
        }
        switch startup {
        case .checking, .loading: return "Loading model…"
        case .warming: return "Updating instructions…"
        case .ready: return "Ready"
        case .unavailable: return "Model unavailable"
        }
    }

    /// Sends `prompt` in the open conversation and streams the reply into it. If `cancelAfter` is set, calls stop() after
    /// that many tokens. The reply keeps writing to its own conversation if another one is opened meanwhile.
    func reply(to prompt: String, cancelAfter: Int? = nil, maxNew: Int? = nil) async -> ReplyRecord {
        let convID = conv.id
        let assistant = ChatMessage(role: .assistant, content: "")
        var aid = assistant.id
        updateConv(convID) { c in
            c.messages.append(ChatMessage(role: .user, content: prompt, modelNote: ToolSettings.dateNote()))
            if !c.titleIsCustom && c.title.isEmpty { c.title = Conversation.title(from: prompt) }
        }
        let history = conv.messages
        updateConv(convID) { c in
            c.messages.append(assistant)
            c.updated = Date()
        }
        if let i = conversations.firstIndex(where: { $0.id == convID }) { conversations.remove(at: i) }
        conversations.insert(conv, at: 0)
        persist(convID)
        isGenerating = true
        generatingConvID = convID
        streamingID = aid
        liveTokens = 0
        thinkStart = nil
        liveThinkingSeconds = nil
        userStopped = false
        let continuing = liveConvID == convID
        defer {
            isGenerating = false
            inReply = false
            generatingConvID = nil
            streamingID = nil
            prefill = nil
            thinkStart = nil
            liveThinkingSeconds = nil
            toolRunningID = nil
            toolRunningQuery = nil
            // a new chat opened while this reply ran: its system state into the slab now (newConversation); else the chat
            // itself, up to the next user header (idle time while the reply is read)
            if conv.id != convID { prepareNewChat() } else if !userStopped { warmChat(convID, why: "after the reply") }
        }
        // tools: the tools block in the system turn when this chat has tools on
        let tools = conversation(convID)?.toolsOn ?? false
        var engineHistory = engineMessages(history, tools: tools)
        if launchPending, let lt = launchTask {  // sent while the model loads at launch
            prefill = expectedPrefill(history: engineMessages(history, tools: tools), convID: convID, waiting: true,
                                      waitingFor: "the model to load")
            log("QUEUED reply=\(replyNo + 1): sent while the model loads at launch; starts when it is ready")
            await lt.value
        }
        if engineState != .ready || userStopped {
            let why = userStopped ? "stopped before the model was ready" : "the model is not available (\(engineState.label))"
            let shownText = userStopped ? "Stopped." : "The model isn't available right now."
            updateConv(convID) { c in
                if let i = c.messages.firstIndex(where: { $0.id == aid }) { c.messages[i].content = shownText }
            }
            persist(convID)
            log("REPLY not run: " + why)
            let r = ReplyRecord(prompt: prompt, thinking: thinking, tokens: 0, reason: userStopped ? "cancelled" : "error", seconds: 0,
                                meanTokPerSec: 0, meanMBPerToken: 0, firstTokenMs: nil, cancelRequestedAtToken: nil, text: "", error: why)
            lastReply = r
            return r
        }
        if let w = warmTask {  // the system prompt's warm-up is still running: the reply starts after it
            prefill = expectedPrefill(history: engineHistory, convID: convID, waiting: true)
            log("QUEUED reply=\(replyNo + 1): sent during the system prompt's warm-up; starts when it ends")
            await w.value
        }
        slabSystem = nil  // this reply's prefill and decode replace the warmed records
        let mx = maxNew ?? AppSettings.maxTokens
        let think = engine.thinking
        let tSend = Date()
        messageSpeed = GenSpeed(); messageEngine = GenSpeed(); messageRounds = 0  // the speed over this message's rounds
        // round 1; then, while the answer ends in a complete tool call: run it, append the call and the result, generate
        // again from the engine's live state (only the result is prefilled)
        var rounds: [ReplyRecord] = []
        var searches: [ToolMeta] = []
        var calls = 0
        // open_url's `result` names a result of the last search ; the last one
        // may be in an earlier message ("give me more detail" follows the searched answer), so it comes from the history
        var (lastResults, lastQuery) = lastSearch(convID)
        var freeRetryUsed = false
        let currentUser = engineHistory.lastIndex(where: { $0.role == .user })
        while true {
            let rec = await runRound(prompt: prompt, convID: convID, aid: aid, history: engineHistory, think: think, mx: mx,
                                     cancelAfter: cancelAfter, continuing: continuing || !rounds.isEmpty, tools: tools,
                                     toolsAllowed: tools && calls < ToolSettings.maxCalls + 1)
            rounds.append(rec)
            if tools, let user = currentUser, searches.isEmpty, !userStopped, rec.error == nil, rec.reason == "eos",
               engineHistory[user].content.contains(SearchHint.hint),
               SearchHint.isSearchPreamble(splitThinking(rec.text, thinkingOn: think).1) {
                let retry = rounds.count == 1
                updateConv(convID) { c in
                    if let i = c.messages.firstIndex(where: { $0.id == aid }) {
                        c.messages[i].content = retry ? "" : "I couldn’t start the web search."
                    }
                }
                if retry {
                    engineHistory[user].content += "\n\n" + SearchHint.retryInstruction
                    log("TOOL reply=\(replyNo) ended after announcing a search; retrying once")
                    continue
                }
                persist(convID)
                break
            }
            guard tools, !userStopped, rec.error == nil, calls <= ToolSettings.maxCalls,
                  let (call, kept) = completedCall(convID: convID, aid: aid, think: think) else { break }
            calls += 1
            // the call turn: its text ends at </tool_call> (anything after it is dropped, as the template would)
            updateConv(convID) { c in
                if let i = c.messages.firstIndex(where: { $0.id == aid }) { c.messages[i].content = kept }
                c.meta[aid]?.toolCall = call
                c.meta[aid]?.reason = "tool_call"
            }
            toolRunningID = aid
            toolRunningQuery = call.query
            log("TOOL reply=\(replyNo) call \(call.name) \(call.arguments) (\(rec.tokens) tokens generated, reason \(rec.reason))")
            let limited = calls > ToolSettings.maxCalls
            let eng = engine
            // open_url ranks the page's sentences by the user's message and this message's searches
            let msgs = conversation(convID)?.messages ?? []
            let lastUser = msgs.lastIndex(where: { $0.role == .user }) ?? 0
            let queries = searches.filter { $0.name == "web_search" }.map(\.query)
            let context = ([msgs.indices.contains(lastUser) ? msgs[lastUser].content : ""]
                + (queries.isEmpty ? [lastQuery].compactMap { $0 } : queries)).joined(separator: " ")
            let last = lastResults
            let task = Task { @MainActor () -> (text: String, meta: ToolMeta) in
                if limited {
                    let t = "Error: tool limit reached. Answer with what you have."
                    return (t, ToolMeta(name: call.name, query: call.query, backend: "-", resultTokens: eng.toolResultTokens(t), error: "limit"))
                }
                #if UITEST
                if UITest.active { return await UITest.fakeSearch(call) }
                #endif
                return await WebSearch.run(call, context: context, last: last, count: { eng.toolResultTokens($0) })
            }
            toolTask = task
            let (text, meta) = await task.value
            toolTask = nil
            searches.append(meta)
            if meta.name == "web_search", meta.error == nil, !meta.results.isEmpty { (lastResults, lastQuery) = (meta.results, meta.query) }
            // a failed open_url of a URL the model wrote (not a search result) does not use up a call, once per message:
            // its error lists the results by number, so the next call can open one (ToolSettings.maxCalls)
            if call.isPageRead, !limited, meta.error != nil, meta.resultIndex == nil, !lastResults.isEmpty, !freeRetryUsed {
                freeRetryUsed = true
                calls -= 1
                log("TOOL reply=\(replyNo) open_url of an unlisted URL failed (\(meta.error ?? "")): not counted (\(calls) of \(ToolSettings.maxCalls) calls used)")
            }
            log(String(format: "TOOL reply=%d result: %d results, %d pages read (%.0f ms), %@ / %@ tokens, %.0f ms, backend %@%@\nTOOLRESULT %@", replyNo, meta.results.count,
                       meta.pages ?? 0, meta.pagesMs ?? 0, meta.resultTokens.map(String.init) ?? "n/a", meta.budget.map(String.init) ?? "n/a",
                       meta.ms, meta.backend, meta.error.map { ", error " + $0 } ?? "",
                       text.replacingOccurrences(of: "\n", with: " | ")))
            let note = ToolSettings.toolNote(meta, canCallAgain: calls < ToolSettings.maxCalls)
            let toolMsg = ChatMessage(role: .tool, content: text, modelNote: note)
            let next = ChatMessage(role: .assistant, content: "")
            var tm = MessageMeta()
            tm.tool = meta
            tm.reason = meta.error == nil ? "ok" : "error"
            updateConv(convID) { c in
                c.messages.append(toolMsg)
                c.meta[toolMsg.id] = tm
                c.messages.append(next)
                c.updated = Date()
            }
            persist(convID)
            toolRunningID = nil
            toolRunningQuery = nil
            if userStopped { break }
            aid = next.id
            streamingID = aid
            engineHistory = engineMessages(conversation(convID)?.messages ?? [], tools: tools)
        }
        // the answer after tool rounds must have text: an answer turn that is empty
        // or only another (unrun) tool call gets one more generation, without tools, after "Answer now with what you
        // have."; if that is empty too, the message says what the searches found
        if tools, !searches.isEmpty, !userStopped, rounds.last?.error == nil, answerIsEmpty(convID, aid, think: think) {
            log("TOOL reply=\(replyNo) the answer after \(searches.count) tool rounds is empty (\(rounds.last?.tokens ?? 0) tokens, reason \(rounds.last?.reason ?? "-")): re-prompting once without tools")
            let unrun = completedCall(convID: convID, aid: aid, think: think)
            let nudge = "Error: no more tools. Answer now with what you have."
            let toolMsg = ChatMessage(role: .tool, content: nudge)
            let next = ChatMessage(role: .assistant, content: "")
            var tm = MessageMeta()
            tm.tool = ToolMeta(name: unrun?.0.name ?? "web_search", query: unrun?.0.query ?? "", backend: "-",
                               resultTokens: engine.toolResultTokens(nudge), error: "limit")
            tm.reason = "error"
            updateConv(convID) { c in
                if let i = c.messages.firstIndex(where: { $0.id == aid }) {
                    if let (call, kept) = unrun {  // the unrun call stays as the call turn it was generated as
                        c.messages[i].content = kept
                        c.meta[aid]?.toolCall = call
                        c.meta[aid]?.reason = "tool_call"
                    } else {  // no answer text (empty, or a think block only): the nudge follows the tool result directly
                        c.messages.remove(at: i); c.meta[aid] = nil
                    }
                }
                c.messages.append(toolMsg)
                c.meta[toolMsg.id] = tm
                c.messages.append(next)
                c.updated = Date()
            }
            persist(convID)
            aid = next.id
            streamingID = aid
            engineHistory = engineMessages(conversation(convID)?.messages ?? [], tools: tools)
            let rec = await runRound(prompt: prompt, convID: convID, aid: aid, history: engineHistory, think: think, mx: mx,
                                     cancelAfter: cancelAfter, continuing: true, tools: tools, toolsAllowed: false)
            rounds.append(rec)
            if !userStopped, answerIsEmpty(convID, aid, think: think) {
                let found = searches.filter { $0.error == nil }.flatMap(\.results)
                let text = found.isEmpty ? "I couldn't find an answer: the searches gave nothing usable."
                    : "I couldn't write an answer. What the search found:\n\n" + found.prefix(4).map {
                        "- **\($0.title)** (\($0.host)): \(Passages.cut($0.snippet, words: 40))"
                    }.joined(separator: "\n")
                updateConv(convID) { c in
                    if let i = c.messages.firstIndex(where: { $0.id == aid }) { c.messages[i].content = text }
                }
                persist(convID)
                log("TOOL reply=\(replyNo) the re-prompted answer is empty too: showing the search results (\(found.count) sources)")
            } else {
                log("TOOL reply=\(replyNo) re-prompted answer: \(rec.tokens) tokens, reason \(rec.reason)")
            }
        }
        // an empty last assistant message (stopped during a search) is removed
        updateConv(convID) { c in
            if let i = c.messages.lastIndex(where: { $0.id == aid }), c.messages[i].content.isEmpty, c.messages[i].role == .assistant,
               i > 0, c.messages[i - 1].role == .tool { c.messages.remove(at: i); c.meta[aid] = nil }
        }
        if !searches.isEmpty { logToolTrip(rounds: rounds, searches: searches, total: Date().timeIntervalSince(tSend)); persist(convID) }
        return rounds.last!
    }

    /// Render hidden app notes and preserve the assistant prefix across tool calls.
    func engineMessages(_ msgs: [ChatMessage], tools: Bool) -> [ChatMessage] {
        var out = msgs.filter { !($0.role == .assistant && $0.content.isEmpty) }
        if out.first?.role == .system {
            out[0].content = ToolSettings.systemContent(out[0].content, tools: tools)
        } else {
            let system = ToolSettings.systemContent("", tools: tools)
            if !system.isEmpty { out.insert(ChatMessage(role: .system, content: system), at: 0) }
        }
        for i in out.indices {
            if tools && out[i].role == .user {
                out[i].content = SearchHint.engineContent(out[i].content, hasDeviceDate: out[i].modelNote != nil)
            }
            if let note = out[i].modelNote { out[i].content += "\n\n" + note }
        }
        if let u = out.lastIndex(where: { $0.role == .user }) {
            for i in out.indices where i > u && out[i].role == .assistant && !out[i].content.contains("</think>") {
                out[i].content = "<think>\n\n</think>\n\n" + out[i].content
            }
        }
        return out
    }

    /// The effective system content of a chat (what the warm-up caches).
    func effectiveSystem(_ c: Conversation) -> String {
        ToolSettings.systemContent(c.systemPrompt, tools: c.toolsOn)
    }

    /// The last successful web_search of the conversation: its results as the model was shown them ([1] first) and its query.
    private func lastSearch(_ convID: UUID) -> ([SearchResult], String?) {
        guard let c = conversation(convID) else { return ([], nil) }
        for m in c.messages.reversed() where m.role == .tool {
            if let t = c.meta[m.id]?.tool, t.name == "web_search", t.error == nil, !t.results.isEmpty { return (t.results, t.query) }
        }
        return ([], nil)
    }

    /// Message `aid` has no answer text: empty, only a think block, or only a tool call (one the round did not run).
    private func answerIsEmpty(_ convID: UUID, _ aid: UUID, think: Bool) -> Bool {
        guard let content = conversation(convID)?.messages.first(where: { $0.id == aid })?.content else { return true }
        var answer = splitThinking(content, thinkingOn: think).1
        if let sp = ToolCall.split(answer) { answer = sp.before }
        return answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// A complete tool call at the end of message `aid`'s answer: the call and the message text cut after `</tool_call>`.
    private func completedCall(convID: UUID, aid: UUID, think: Bool) -> (ToolCall, String)? {
        guard let content = conversation(convID)?.messages.first(where: { $0.id == aid })?.content else { return nil }
        let answer = splitThinking(content, thinkingOn: think).1
        guard let sp = ToolCall.split(answer), sp.complete, let call = ToolCall.parse(sp.call),
              let close = content.range(of: ToolPrompt.callClose, options: .backwards) else { return nil }
        return (call, String(content[..<close.upperBound]))
    }

    /// The round trip's cost split (console `TOOLTRIP`): the tool-call round (prefill, first token, the call's
    /// tokens), the search, the result prefill of the continuation (live prefix: only the result turn is new) and its first
    /// token and decode.
    private func logToolTrip(rounds: [ReplyRecord], searches: [ToolMeta], total: Double) {
        func f(_ v: Double?, _ k: Double = 1000) -> String { v.map { String(format: "%.2f", $0 / k) } ?? "n/a" }
        for (i, r) in rounds.enumerated() {
            let p = r.prefill ?? [:]
            let decode = r.seconds
            let computed = Int(p["prefill_computed"] ?? p["prefill_tokens"] ?? 0)
            let reused = Int(p["prefill_reused"] ?? Double(max(0, (r.promptTokens ?? 0) - computed)))
            log("TOOLTRIP reply=\(replyNo) round=\(i + 1)/\(rounds.count) kind=\(r.prefillKind ?? "n/a") prompt_tokens=\(r.promptTokens ?? 0) reused=\(reused) prefilled=\(computed) prefill_s=\(f(p["prefill_s"], 1)) prefill_mb=\(f(p["prefill_mb_read"], 1)) phone_read_s=\(f(p["prefill_mb_read"], 750)) restore_s=\(f(p["prefix_restore_ms"])) first_token_s=\(f(r.firstTokenMs)) generated=\(r.tokens) decode_s=\(String(format: "%.2f", decode)) wall_s=\(f(r.wallMs)) reason=\(r.reason)")
        }
        for (i, m) in searches.enumerated() {
            log("TOOLTRIP reply=\(replyNo) search=\(i + 1) tool=\(m.name) query=\"\(m.query)\" backend=\(m.backend) results=\(m.results.count) pages=\(m.pages ?? 0) pages_s=\(f(m.pagesMs)) result_tokens=\(m.resultTokens.map(String.init) ?? "n/a") budget=\(m.budget.map(String.init) ?? "n/a") search_s=\(f(m.ms))\(m.error.map { " error=" + $0 } ?? "")")
        }
        log(String(format: "TOOLTRIP reply=%d total_s=%.2f (send to the last token)", replyNo, total))
    }

    /// One generation round of a reply: stream into assistant message `aid`, then the caption, persistence and the logs.
    /// With `toolsAllowed`, a complete `<tool_call>…</tool_call>` in the answer stops the round (the model normally ends its
    /// turn right after it; the stop is the safeguard).
    private func runRound(prompt: String, convID: UUID, aid: UUID, history: [ChatMessage], think: Bool, mx: Int,
                          cancelAfter: Int?, continuing: Bool, tools: Bool, toolsAllowed: Bool) async -> ReplyRecord {
        replyNo += 1
        replyTokens = 0
        inReply = true
        let warnings0 = memoryWarnings
        defer { inReply = false }
        liveTokens = 0
        thinkStart = nil
        liveThinkingSeconds = nil
        var toolStopSent = false
        prefill = expectedPrefill(history: history, convID: convID, waiting: false)

        showSpeed(messageSpeed)  // the message's rounds before this one (none: 0)
        var roundSpeed = GenSpeed()
        mbPerToken = 0
        replyTtftMs = nil
        var rec = ReplyRecord(prompt: prompt, thinking: think, tokens: 0, reason: StopReason.cancelled.rawValue,
                              seconds: 0, meanTokPerSec: 0, meanMBPerToken: 0, firstTokenMs: nil,
                              cancelRequestedAtToken: nil, text: "", error: nil, maxNew: mx)
        let t0 = Date()
        var n = 0, mbSum = 0.0
        var finished = false
        var lastTick: PrefillTick? = nil
        // the UI follows the reply at most every `UIBatch.interval` (the first token at once): the
        // streamed text goes into the conversation and the stats bar in one update per batch, not one per token or per
        // prefill layer (each update re-rendered the chat; on a phone whose UI pages iOS keeps dropping, every render that
        // touches cold code waits for the drive)
        let batch = UIBatch()
        var full = conversation(convID)?.messages.last(where: { $0.id == aid })?.content ?? ""
        func flush() {
            batch.scheduled = false
            batch.last = ProcessInfo.processInfo.systemUptime
            let piece = batch.text
            batch.text = ""
            if !piece.isEmpty {
                updateConv(convID) { c in
                    if let i = c.messages.lastIndex(where: { $0.id == aid }) { c.messages[i].content += piece }
                }
            }
            if let st = batch.stats {
                batch.stats = nil
                liveTokens = batch.n
                lastTokPerSec = st.tokPerSec
                showSpeed(messageSpeed + st.speed)
                mbPerToken = st.mbTotal > 0 ? st.mbTotal / Double(st.index + 1) : batch.mbSum / Double(max(1, batch.n))
                inMemory = st.inMemory
                expertsInMemory = st.expertsInMemory
            }
        }
        do {
            for try await ev in engine.generate(messages: history, maxNew: mx) {
                switch ev {
                case .prefill(let p):
                    // the engine's own prefill progress replaces the estimate (the card and the status line read it); the
                    // published value at most every 0.25 s (the card extrapolates between ticks) and at the end
                    let tnow = ProcessInfo.processInfo.systemUptime
                    if var pp = prefill, lastTick == nil || p.done == p.total || tnow - batch.lastTick >= 0.25 {
                        pp.tick = p
                        pp.tickAt = Date()
                        prefill = pp
                        batch.lastTick = tnow
                    }
                    lastTick = p
                    if p.done == 0 || p.done == p.total || p.done % 10 == 0 {
                        log(String(format: "PROGRESS reply=%d %@ done=%d/%d reused=%d read=%.1f MB reland=%.1f MB prefill=%.2f s elapsed=%.2f s",
                                   replyNo, p.kind, p.done, p.total, p.reusedTokens, p.mbRead, p.relandMB, p.prefillElapsed, p.elapsed))
                    }
                case .text(let piece, let st):
                    var firstNow = false
                    if rec.firstTokenMs == nil {
                        rec.firstTokenMs = Date().timeIntervalSince(t0) * 1000
                        replyTtftMs = rec.firstTokenMs
                        log(String(format: "REPLY reply=%d first token at %.1f s", replyNo, (rec.firstTokenMs ?? 0) / 1000))
                        prefill = nil
                        if think { thinkStart = Date() }
                        firstNow = true
                    }
                    full += piece
                    // a complete tool call in the answer ends the round (by string: </tool_call> is one token, 248059)
                    if toolsAllowed, !toolStopSent, piece.contains("tool_call>"), st.phase == nil || st.phase == "answer",
                       ToolCall.split(splitThinking(full, thinkingOn: think).1)?.complete == true {
                        toolStopSent = true
                        log("TOOL reply=\(replyNo) </tool_call> at token \(n + 1): stopping the round")
                        engine.cancel()
                    }
                    if st.phase == "close" || st.phase == "answer" || (st.phase == nil && piece.contains("</think>")),
                       let s = thinkStart, liveThinkingSeconds == nil {
                        liveThinkingSeconds = Date().timeIntervalSince(s)
                    }
                    n += 1
                    replyTokens = n
                    if n == 1 || n % 10 == 0 { memSample(n == 1 ? "reply \(replyNo) first token" : "reply \(replyNo) token \(n)") }
                    mbSum += st.mbRead
                    batch.text += piece
                    batch.stats = st
                    roundSpeed = st.speed
                    batch.n = n
                    batch.mbSum = mbSum
                    let since = ProcessInfo.processInfo.systemUptime - batch.last
                    if firstNow || since >= UIBatch.interval || toolStopSent {
                        flush()
                    } else if !batch.scheduled {
                        batch.scheduled = true
                        let wait = UIBatch.interval - since
                        Task { @MainActor in
                            try? await Task.sleep(nanoseconds: UInt64(wait * 1e9))
                            if batch.scheduled { flush() }
                        }
                    }
                    if let c = cancelAfter, n == c, rec.cancelRequestedAtToken == nil {
                        rec.cancelRequestedAtToken = n
                        stop()
                    }
                case .finished(let s):
                    finished = true
                    lastSummary = s
                    rec.reason = s.reason.rawValue
                    rec.tokens = s.tokens
                    rec.seconds = s.seconds
                    rec.meanTokPerSec = s.meanTokPerSec
                    rec.meanMBPerToken = s.meanMBPerToken
                    roundSpeed = s.speed  // the round's every step, the one that produced the stop included
                }
            }
        } catch {
            rec.error = error.localizedDescription
            event("generate error: \(error.localizedDescription)")
        }
        flush()  // the batch's rest (the text and the stats bar before the caption)
        if !finished {
            rec.tokens = n
            rec.seconds = roundSpeed.seconds
            rec.meanTokPerSec = roundSpeed.rate
            rec.meanMBPerToken = n > 0 ? mbSum / Double(n) : 0
        }
        // the message's speed: this round added to the ones before it (the figure every screen shows from here)
        messageSpeed = messageSpeed + roundSpeed
        messageRounds += 1
        showSpeed(messageSpeed)
        rec.text = conversation(convID)?.messages.last(where: { $0.id == aid })?.content ?? ""
        rec.wallMs = Date().timeIntervalSince(t0) * 1000
        rec.memoryWarnings = memoryWarnings - warnings0
        defer { lastReply = rec }
        let detail = finished ? engine.lastDetail : nil
        if let d = detail {
            rec.ids = d.ids
            rec.promptTokens = d.promptIds.count
            rec.split = d.split
            rec.prefill = d.prefillStats
            rec.prefillKind = d.prefill
            lastSplit = d.split
            lastPrefill = d.prefillStats
            lastPrefillKind = d.prefill
            if let es = d.split?["engine_steps"], let et = d.split?["engine_step_s"] {
                messageEngine = messageEngine + GenSpeed(steps: Int(es), seconds: et)
            }
        }
        // the shown figure beside the engine counters' over the same rounds (SelfCheck.rateOK: within 2%)
        if messageEngine.steps > 0 {
            rec.engineTokPerSec = messageEngine.rate
            rec.split?["rate_shown"] = replyTokPerSec
            rec.split?["rate_engine"] = messageEngine.rate
        }
        // what the UI shows at the end of the reply (the stats bar, the settings sheet), logged next to the engine's
        lastTtftMs = rec.firstTokenMs
        if n > 0 {
            rec.uiDecodeTokPerSec = replyTokPerSec
            rec.uiMBPerToken = mbPerToken
            rec.uiTtftMs = rec.firstTokenMs
            if let e = rec.engineTokPerSec, e > 0 { rec.uiRateDiffPct = 100 * (replyTokPerSec - e) / e }  // (RATE line)
        }
        // the message's caption and the conversation's token count, saved with it
        var meta = MessageMeta(tokens: rec.tokens, reason: rec.error != nil ? "error" : rec.reason)
        meta.decodeTokPerSec = n > 0 ? replyTokPerSec : nil
        meta.ttftMs = rec.firstTokenMs
        meta.mbPerToken = n > 0 ? mbPerToken : nil
        meta.error = rec.error
        meta.thinking = think
        if think, let s = thinkStart { meta.thinkingSeconds = liveThinkingSeconds ?? Date().timeIntervalSince(s) }
        if let d = detail {
            meta.prefillKind = d.prefill
            meta.promptTokens = d.promptIds.count
            meta.prefilledTokens = d.prefillTokens
            meta.reusedTokens = d.promptIds.count - d.prefillTokens
            meta.restoreMs = d.prefillStats?["prefix_restore_ms"]
        }
        updateConv(convID) { c in
            c.meta[aid] = meta
            c.updated = Date()
            if let d = detail { c.contextTokens = d.promptIds.count + d.ids.count }
            else { c.contextTokens = c.messages.map { estimateTokens($0.content) }.reduce(0, +) }  // an estimate
        }
        persist(convID)
        if let d = detail {
            if continuing { UserDefaults.standard.set(d.prefill == "live prefix", forKey: AppSettings.liveContinuationKey) }
            liveConvID = convID
            learnTiming(d)
        }
        log("CAPTION reply=\(replyNo) " + meta.caption + (continuing ? " (continued conversation)" : " (first reply of this launch in it)"))
        prefixStatus = engine.prefixCacheStatus
        log(String(format: "REPLY tokens=%d reason=%@ %.2f tok/s %.1f MB/token", rec.tokens, rec.reason,
                   rec.meanTokPerSec, rec.meanMBPerToken))
        // one line per reply: the shown figure (the message's rounds so far) beside the engine counters' over the same rounds
        log(String(format: "RATE reply=%d shown %.3f tokens/s (%d steps, %.2f s generating, %d round%@; this round %.2f s wall) engine counters %@ caption %@",
                   replyNo, replySpeed.rate, replySpeed.steps, replySpeed.seconds, messageRounds, messageRounds == 1 ? "" : "s",
                   (rec.wallMs ?? 0) / 1000,
                   messageEngine.steps > 0 ? String(format: "%.3f tokens/s (%d steps, %.2f s in dm_step) diff %+.2f%%", messageEngine.rate,
                                                    messageEngine.steps, messageEngine.seconds, rec.uiRateDiffPct ?? 0) : "n/a",
                   meta.decodeTokPerSec.map { String(format: "%.3f", $0) } ?? "none"))
        memSample("end of reply \(replyNo) (\(rec.tokens) tokens, \(rec.reason))")
        return rec
    }

    // MARK: scene phase

    func phaseChanged(_ p: ScenePhase) {
        #if !targetEnvironment(macCatalyst)
        if p == .background && isGenerating {
            // iOS suspends GPU work in the background: stop after the current token
            event("scene -> background during generation; stopping")
            stop()
        }
        #endif
    }
}
