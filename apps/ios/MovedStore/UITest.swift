#if UITEST
import QuartzCore
import SwiftUI
import UIKit

// The chat view's scroll harness, compiled only with UITEST in SWIFT_ACTIVE_COMPILATION_CONDITIONS (README "Checking the
// chat's scrolling"; normal builds contain none of this). Launched with `-uitest <scenario>`, the app skips the setup
// screen, replaces the engine with scripted replies (FakeChatEngine: the real ChatModel reply path, tool loop and UI
// batching, no model) and a fake web search, seeds saved chats, runs the scenario's script, and records the chat's
// scroll view every display frame to Documents/uitest-<scenario>.json (rewritten every 2 s and at the end, where it logs
// `UITEST DONE`). apps/ios/tests/scroll_check.py reads the file and applies the jump checks.
//
// Scenarios: follow (two replies at the bottom: thinking, its collapse, Markdown, tool rounds, the retry of a reply that
// only announced a search, sources, captions), keyboard (the keyboard up and down and a draft growing to 4 lines while a
// reply streams at the bottom), hold (a long chat: a reply starts, the user drags up during it and holds still; the reply
// then collapses its thinking, runs a tool round and finishes, and the keyboard comes up and goes down), history (a long
// chat opened, then the user scrolls up through it), open (chats opened from the list and switched in place).
// Touches come from outside (the Simulator's HID input): the harness logs `UITEST TOUCH NOW` when it waits for them.

/// Flags read by the fake engine's task (off the main actor).
enum UITestFlags {
    /// The fake engine keeps the thinking phase going (slowly) while this is true (scenario hold).
    nonisolated(unsafe) static var holdThinking = false
    /// The fake engine's step in flight waits while the app is in the background (scenario rate).
    nonisolated(unsafe) static var stallInBackground = false
    nonisolated(unsafe) static var tokenInterval: Double = {
        let a = ProcessInfo.processInfo.arguments
        if let i = a.firstIndex(of: "-uitestTokenMs"), i + 1 < a.count, let v = Double(a[i + 1]) { return v / 1000 }
        return 0.1
    }()
}

@MainActor
enum UITest {
    static let args = ProcessInfo.processInfo.arguments
    static let scenario: String? = {
        guard let i = args.firstIndex(of: "-uitest"), i + 1 < args.count else { return nil }
        return args[i + 1]
    }()
    nonisolated static var active: Bool {
        let a = ProcessInfo.processInfo.arguments
        return a.contains("-uitest")
    }
    /// The iOS 17 follow path (the bottom marker's appearance) on a newer runtime.
    nonisolated static var forceLegacy: Bool { ProcessInfo.processInfo.arguments.contains("-uitestLegacy") }
    static let navNote = Notification.Name("uitest.nav")
    static let rec = Recorder()
    private static var started = false
    static var longID = UUID(), otherID = UUID(), inklingID = UUID()

    // MARK: script

    static func start(_ m: ChatModel) {
        guard let s = scenario, !started else { return }
        started = true
        seed(m)
        rec.start(scenario: s, model: m)
        Task { await run(s, m) }
    }

    static let msg1 = "Explain how a refrigerator works, with an example."
    static let msg2 = "What are today's top news headlines?"

    private static func run(_ s: String, _ m: ChatModel) async {
        await waitReady(m)
        try? await Task.sleep(nanoseconds: 800_000_000)
        if s == "keyboard" || s == "hold" {
            // the Simulator's first keyboard stalls the app for seconds: bring it up once before the measured part
            focus(); await sleep(4); unfocus(); await sleep(2)
            rec.mark("keyboard warmed up")
        }
        switch s {
        case "follow":
            m.thinking = true
            send(m, msg1)
            await waitIdle(m)
            await sleep(1.5)
            send(m, msg2)
            await waitIdle(m)
            await sleep(2)
        case "keyboard":
            m.open(otherID)
            await sleep(1.5)
            m.thinking = true
            send(m, msg1)
            await sleep(3); focus()
            await sleep(2.5)
            for w in draftWords { type(w); await sleep(0.15) }
            await sleep(2); unfocus()
            await sleep(3); focus()
            await sleep(2); clearDraft()
            await sleep(2); unfocus()
            await waitIdle(m)
            await sleep(2)
        case "hold":
            m.open(longID)
            await sleep(1.5)
            m.thinking = true
            UITestFlags.holdThinking = true
            send(m, msg1)
            await sleep(1.5)
            log("UITEST TOUCH NOW: drag up and let go")
            await rec.waitForHold()
            rec.beginHold()
            await sleep(1.5)
            UITestFlags.holdThinking = false
            rec.mark("thinking released")
            await sleep(8)
            focus()
            await sleep(2)
            for w in draftWords { type(w); await sleep(0.15) }
            await sleep(2); unfocus()
            await waitIdle(m)
            await sleep(2)
        case "history":
            rec.mark("switch long")
            m.open(longID)
            await sleep(2.5)
            log("UITEST TOUCH NOW: scroll up through the chat")
            // -uitestQuiet N: wait N s after the last touch before ending (taps on the arrow, which is not a scroll)
            let quiet = args.firstIndex(of: "-uitestQuiet").flatMap { i in i + 1 < args.count ? Double(args[i + 1]) : nil } ?? 6
            await rec.waitForTouches(quiet: 6)
            await sleep(max(0, quiet - 6))
        case "rate":
            // the speed figure (README "Checking the speed figure"): a normal reply, a reply with tool rounds (the figure
            // over all its rounds), one with the app in the background ~10 s in the middle (driven from outside on
            // `UITEST BACKGROUND NOW`; the step in flight waits while the app is away, as a suspended step does, and the
            // app stops the reply after it), a stopped one. The fake engine's step is one token interval and the
            // thinking hold pauses 0.2 s between steps (not generating): every case must show the normal reply's figure
            // (the Simulator's actual step rate) within 2%, and the caption and the status line the same figure
            var ref = 0.0
            func check(_ what: String) {
                let shown = m.replyTokPerSec
                if ref == 0 { ref = shown }
                let cap = m.conv.messages.last(where: { $0.role == .assistant }).flatMap { m.conv.meta[$0.id]?.decodeTokPerSec } ?? -1
                let status = m.statusText
                let ok = abs(shown - ref) / ref <= 0.02 && abs(cap - shown) < 1e-9 && status.hasPrefix(String(format: "%.1f tokens/s", shown))
                    && abs(shown * UITestFlags.tokenInterval - 1) < 0.1
                log(String(format: "UITEST RATE %@ shown %.3f tokens/s (%d steps, %.2f s generating), caption %.3f, status \"%@\"; tokens / wall %.3f; the normal reply's %.3f, scripted %.3f: %@",
                           what, shown, m.replySpeed.steps, m.replySpeed.seconds, cap, status,
                           Double(m.lastReply?.tokens ?? 0) / max(1e-9, (m.lastReply?.wallMs ?? 0) / 1000), ref, 1 / UITestFlags.tokenInterval,
                           ok ? "OK" : "FAIL"))
            }
            m.thinking = true
            send(m, msg1)
            await waitIdle(m)
            check("normal")
            await sleep(1.5)
            send(m, msg2)
            await waitIdle(m)
            check("tool rounds")
            await sleep(1.5)
            UITestFlags.holdThinking = true
            UITestFlags.stallInBackground = true
            send(m, msg1)
            await sleep(3)
            let bg0 = AppClock.background
            log("UITEST BACKGROUND NOW")
            var k = 0
            while AppClock.background - bg0 < 8 || UIApplication.shared.applicationState != .active {
                await sleep(0.5)
                k += 1
                if k % 4 == 0 { log(String(format: "UITEST waiting: background %.1f s, state %d", AppClock.background - bg0, UIApplication.shared.applicationState.rawValue)) }
            }
            await sleep(1)
            UITestFlags.holdThinking = false
            await waitIdle(m)
            UITestFlags.stallInBackground = false
            log(String(format: "UITEST background %.1f s", AppClock.background - bg0))
            check("background")
            await sleep(1.5)
            UITestFlags.holdThinking = true
            send(m, msg1)
            await sleep(3)
            m.stop()
            await waitIdle(m)
            UITestFlags.holdThinking = false
            check("stopped (\(m.lastSummary?.reason.rawValue ?? "?"))")
            await sleep(1)
        case "open":
            rec.mark("switch long"); m.open(longID); await sleep(3)
            rec.mark("back"); nav("back"); await sleep(2)
            rec.mark("push other"); m.open(otherID); nav("chat"); await sleep(3)
            rec.mark("switch long"); m.open(longID); await sleep(3)
            rec.mark("back"); nav("back"); await sleep(2)
            rec.mark("push long"); nav("chat"); await sleep(3)
            rec.mark("switch other"); m.open(otherID); await sleep(3)
        case "shots":
            // screenshots: a chat, the list (driven from outside), a chat made with the other model
            m.open(otherID)
            log("UITEST SHOTS chat open")
            await sleep(4)
            nav("back")
            log("UITEST SHOTS list")
            await sleep(4)
            m.open(inklingID); nav("chat")
            log("UITEST SHOTS other model")
            await sleep(4)
            m.newConversation()
            log("UITEST SHOTS new chat")
            await sleep(4)
        default:
            log("UITEST unknown scenario \(s)")
        }
        rec.mark("script done")
        rec.write()
        log("UITEST DONE \(s) -> uitest-\(s).json")
    }

    static let draftWords = ("Also, could you say how a heat pump differs from it, which one uses less power in a mild "
        + "winter, and whether the same idea is behind the air conditioner in a car? A short answer is fine.")
        .split(separator: " ").map { String($0) + " " }

    static func sleep(_ s: Double) async { try? await Task.sleep(nanoseconds: UInt64(s * 1e9)) }

    static func waitReady(_ m: ChatModel) async {
        while m.engineState != .ready { await sleep(0.05) }
    }

    static func waitIdle(_ m: ChatModel) async {
        await sleep(0.3)
        while m.isGenerating { await sleep(0.05) }
    }

    static func send(_ m: ChatModel, _ text: String) {
        rec.mark("send: " + text)
        m.send(text)
    }

    static func nav(_ to: String) { NotificationCenter.default.post(name: navNote, object: to) }

    static var window: UIWindow? {
        UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.flatMap(\.windows).first { $0.isKeyWindow }
    }

    static func composer() -> UITextView? {
        func find(_ v: UIView) -> UITextView? {
            if let t = v as? UITextView, t.isEditable { return t }
            for s in v.subviews { if let t = find(s) { return t } }
            return nil
        }
        return window.flatMap(find)
    }

    static func focus() { rec.mark("keyboard up"); _ = composer()?.becomeFirstResponder() }
    static func unfocus() { rec.mark("keyboard down"); window?.endEditing(true) }
    static func type(_ s: String) { composer()?.insertText(s) }
    static func clearDraft() {
        rec.mark("draft cleared")
        guard let t = composer() else { return }
        t.selectAll(nil)
        t.deleteBackward()
    }

    // MARK: fake search

    static func fakeSearch(_ call: ToolCall) async -> (text: String, meta: ToolMeta) {
        await sleep(1.5)
        let results = (1...3).map { i in
            SearchResult(title: "\(call.query.capitalized): source \(i)", url: "https://example.org/\(i)",
                         snippet: "A passage about \(call.query) from source \(i), with a few figures and a date.", read: i < 3)
        }
        let text = results.enumerated().map { "[\($0.offset + 1)] \($0.element.title)\n\($0.element.snippet)" }.joined(separator: "\n\n")
        return (text, ToolMeta(name: call.name, query: call.query, backend: "DuckDuckGo", results: results, resultTokens: 112,
                               ms: 1500, pages: 2, budget: 128))
    }

    // MARK: seeded chats

    static func seed(_ m: ChatModel) {
        var long = Conversation(system: AppSettings.systemPrompt, thinking: false)
        long.id = longID
        long.title = "Long chat with Markdown"
        for i in 0..<18 {
            let u = ChatMessage(role: .user, content: "Question \(i + 1): " + seedQuestions[i % seedQuestions.count])
            var content = longReply(i)
            var meta = MessageMeta(tokens: 260 + 37 * i, reason: "eos", decodeTokPerSec: 3.4, ttftMs: 4100 + 50 * Double(i))
            if i % 4 == 1 {
                content = "The user asks about part \(i + 1). I should give a structured answer with an example and a table.\n</think>\n\n" + content
                meta.thinking = true
                meta.thinkingSeconds = 9 + Double(i)
            }
            let a = ChatMessage(role: .assistant, content: content)
            long.messages += [u, a]
            long.meta[a.id] = meta
        }
        long.updated = Date().addingTimeInterval(-600)
        var other = Conversation(system: AppSettings.systemPrompt, thinking: false)
        other.id = otherID
        other.title = "Weekend plans"
        for i in 0..<6 {
            let u = ChatMessage(role: .user, content: "Another question, number \(i + 1)?")
            let a = ChatMessage(role: .assistant, content: longReply(i + 3).components(separatedBy: "\n\n").prefix(3).joined(separator: "\n\n"))
            other.messages += [u, a]
            other.meta[a.id] = MessageMeta(tokens: 120, reason: "eos", decodeTokPerSec: 3.6, ttftMs: 3900)
        }
        other.updated = Date().addingTimeInterval(-60)
        var ink = Conversation(system: AppSettings.systemPrompt, thinking: false)
        ink.id = inklingID
        ink.title = "Packing list for a trip"
        ink.model = .inkling
        let iu = ChatMessage(role: .user, content: "What should I pack for three days of hiking?")
        let ia = ChatMessage(role: .assistant, content: "- Layers: a base layer, a fleece and a rain shell\n- Two pairs of socks per day\n- A map, a headlamp and a first-aid kit")
        ink.messages += [iu, ia]
        ink.meta[ia.id] = MessageMeta(tokens: 60, reason: "eos", decodeTokPerSec: 2.4, ttftMs: 5200)
        ink.updated = Date().addingTimeInterval(-7200)
        m.conversations = [other, long, ink] + m.conversations.filter { ![longID, otherID, inklingID].contains($0.id) }
    }

    static let seedQuestions = [
        "How do vaccines train the immune system?", "Compare three ways to sort a list in Swift.",
        "What makes sourdough rise?", "Plan a week of simple dinners.", "Why is the sky blue at noon and red at sunset?",
        "Summarize the causes of the French Revolution.",
    ]

    static func longReply(_ i: Int) -> String {
        let para = "This is a paragraph of an earlier reply, long enough to wrap over several lines on a phone. It explains "
            + "one idea with a little detail, so the row's height depends on the text and the width."
        var parts = ["## Part \(i + 1)", para]
        for k in 0..<(i % 3 + 1) { parts.append(para + " Point \(k + 1) adds one more sentence to vary the length.") }
        parts.append((1...(3 + i % 4)).map { "- Item \($0): a short line" + String(repeating: " that runs on", count: $0 % 3) }.joined(separator: "\n"))
        if i % 2 == 0 {
            parts.append("```swift\nlet values = [3, 1, 2]\nlet sorted = values.sorted()\nprint(sorted)  // [1, 2, 3]\n```")
        }
        if i % 3 == 0 {
            parts.append("| Method | Cost | Notes |\n|---|---|---|\n| sorted() | n log n | stable |\n| sort(by:) | n log n | in place |\n| heap | n log n | not stable |")
        }
        parts.append(para)
        return parts.joined(separator: "\n\n")
    }

    // MARK: scripted replies

    enum Segment { case text(String), holdThinking }

    /// The reply for this prompt, by where the conversation is in the script.
    nonisolated static func plan(_ messages: [ChatMessage], thinking: Bool) -> [Segment] {
        let lastUser = messages.last(where: { $0.role == .user })?.content ?? ""
        let afterTool = messages.last?.role == .tool
        func think(_ s: String) -> [Segment] { thinking ? [.text(s + "\n</think>\n\n")] : [] }
        let shortThought = "The search gave what I need. I will answer with the main points and keep it short."
        if afterTool {
            if lastUser.contains("headlines") { return think(shortThought) + [.text(answerC)] }
            return think(shortThought) + [.text(answerB)]
        }
        if lastUser.contains(SearchHint.retryInstruction) { return think("I should search now.") + [.text(call("top news headlines today"))] }
        if lastUser.contains(SearchHint.hint) { return think("The user wants the news. I need to search.") + [.text("Let me check the latest news.")] }
        var out: [Segment] = []
        if thinking {
            out.append(.text(longThought))
            out.append(.holdThinking)
            out.append(.text("\n</think>\n\n"))
        }
        out.append(.text(answerA))
        out.append(.text(call("how refrigerators move heat")))
        return out
    }

    nonisolated static func call(_ q: String) -> String {
        "\n\n<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"\(q)\"}}\n</tool_call>"
    }

    nonisolated static let longThought = "The user wants to know how a refrigerator works. I should start with the core idea: "
        + "it does not make cold, it moves heat from inside the box to the room. The refrigerant evaporates in the coils "
        + "inside, which absorbs heat, then the compressor squeezes the vapor, which makes it hot, and the coils at the back "
        + "release that heat into the kitchen. Then the liquid passes an expansion valve, cools sharply, and the cycle "
        + "repeats. An example would help: a can of soda put in at room temperature. I could add a table of the four "
        + "stages and a short code sketch of a thermostat loop. I should also look up a figure for typical efficiency."

    nonisolated static let answerA = "A refrigerator doesn't make cold: it **moves heat** from the inside of the box to "
        + "the room, using a fluid that boils at a low temperature.\n\nThe cycle has four steps:\n\n1. **Evaporate.** "
        + "The refrigerant boils in the coils inside and takes heat from the food.\n2. **Compress.** The compressor "
        + "squeezes the vapor, which makes it hot.\n3. **Condense.** The coils at the back give that heat to the kitchen.\n"
        + "4. **Expand.** The liquid passes a narrow valve, drops in pressure and gets cold again.\n\nLet me check one figure."

    nonisolated static let answerB = "For example, a can of soda put in at 22 °C reaches about 4 °C in a few hours: the "
        + "fridge removes the heat a little at a time while the compressor runs.\n\nA thermostat keeps it in range:\n\n"
        + "```swift\nwhile true {\n    let t = sensor.read()\n    if t > target + 1 { compressor.on() }\n    if t < target - 1 { compressor.off() }\n    sleep(30)\n}\n```\n\n"
        + "| Stage | Where | Temperature |\n|---|---|---|\n| Evaporate | inside coils | −20 °C |\n| Compress | compressor | 60 °C |\n"
        + "| Condense | back coils | 40 °C |\n| Expand | valve | −25 °C |\n\nA modern fridge moves about two to three "
        + "times as much heat as the electricity it uses [1]. Keeping the coils clean and the door closed helps it most."

    nonisolated static let answerC = "Here are today's top stories:\n\n- **Weather:** a storm system moves east with heavy "
        + "rain [1].\n- **Science:** a new probe reaches orbit around Jupiter's moon [2].\n- **Sports:** the final goes "
        + "to extra time [3].\n\nI can open any of these for more detail."

    /// Word-sized pieces (a word with its trailing space), the tool call's tags as their own pieces.
    nonisolated static func pieces(_ s: String) -> [String] {
        var out: [String] = []
        var cur = ""
        var sawText = false
        for ch in s {
            if ch.isWhitespace { cur.append(ch); continue }
            if sawText && !cur.isEmpty && cur.last!.isWhitespace { out.append(cur); cur = "" }
            cur.append(ch)
            sawText = true
        }
        if !cur.isEmpty { out.append(cur) }
        // keep "<tool_call>" and "</tool_call>" whole (the round stops at the close tag's piece)
        return out
    }
}

/// Scripted replies through the real reply path.
final class FakeChatEngine: ChatEngine {
    var info = EngineInfo(name: "Scripted replies", detail: "UITEST harness, no model")
    var settings: [(String, String)] { [("engine", "scripted replies (UITEST)")] }
    var lastDetail: ReplyDetail? { nil }
    var thinking = false
    var prefixCacheEnabled = false
    var prefixCacheStatus: String { "off (UITEST)" }
    private let lock = NSLock()
    private var stopped = false
    private var isStopped: Bool { lock.lock(); defer { lock.unlock() }; return stopped }

    func load(progress: @escaping (String) -> Void) async throws {
        progress("scripted engine")
        try? await Task.sleep(nanoseconds: 300_000_000)
    }

    func generate(messages: [ChatMessage], maxNew: Int) -> AsyncThrowingStream<GenEvent, Error> {
        lock.lock(); stopped = false; lock.unlock()
        let plan = UITest.plan(messages, thinking: thinking)
        return AsyncThrowingStream { cont in
            Task.detached { [self] in
                let dt = UITestFlags.tokenInterval
                for i in 0...8 where !isStopped {
                    cont.yield(.prefill(PrefillTick(kind: "restored prefix", promptTokens: 400, reusedTokens: 360, total: 40,
                                                    done: i * 5, mbRead: Double(i) * 40, relandMB: 0, elapsed: Double(i) * 0.1,
                                                    prefillElapsed: Double(i) * 0.1)))
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                AppClock.start()
                let clock = GenClock()
                var n = 0
                func emit(_ p: String) async {
                    n += 1
                    cont.yield(.text(p, TokenStats(index: n - 1, ms: clock.last * 1000, tokPerSec: clock.last > 0 ? 1 / clock.last : 0,
                                                   mbRead: 45, inMemory: 0.6, expertsInMemory: 400,
                                                   mbTotal: 45 * Double(n), speed: clock.speed)))
                    let a = clock.begin()  // the scripted "step": one token interval
                    try? await Task.sleep(nanoseconds: UInt64(dt * 1e9))
                    // (scenario rate) a step the app's suspension caught midway: it ends after the app is back
                    while UITestFlags.stallInBackground && AppClock.inBackground { try? await Task.sleep(nanoseconds: 200_000_000) }
                    clock.end(a)  // a step that ran is counted, as the engine's (a stop is seen before the next one)
                }
                outer: for seg in plan {
                    switch seg {
                    case .text(let s):
                        for p in UITest.pieces(s) {
                            if isStopped { break outer }
                            await emit(p)
                        }
                    case .holdThinking:
                        var k = 0
                        while UITestFlags.holdThinking && !isStopped {
                            k += 1
                            await emit(k % 8 == 0 ? "more. " : "thinking ")
                            try? await Task.sleep(nanoseconds: 200_000_000)
                        }
                    }
                }
                cont.yield(.finished(GenSummary(reason: isStopped ? .cancelled : .eos, tokens: n, seconds: clock.speed.seconds,
                                                meanTokPerSec: clock.speed.rate, meanMBPerToken: 45, speed: clock.speed)))
                cont.finish()
            }
        }
    }

    func cancel() { lock.lock(); stopped = true; lock.unlock() }
    func forget() {}
    func clearPrefixCache() async {}
    func waitIdle() async {}
}

/// Row frames in the list's content space, reported by each row (MessageRow's probe in ChatView).
struct UITestRowProbe: ViewModifier {
    let id: UUID
    func body(content: Content) -> some View {
        content
            .onGeometryChange(for: CGRect.self) { $0.frame(in: .named(UITest.contentSpace)) } action: { f in
                UITest.rec.rows[id] = f
            }
            .onDisappear { UITest.rec.rows[id] = nil }
    }
}

extension UITest {
    static let contentSpace = "uitest.content"
}

/// Every display frame: the chat's scroll view (offset, content and container height, insets, the touch state, the pan's
/// translation), the follow state, the keyboard, the tracked row and the top visible row on screen.
@MainActor
final class Recorder: NSObject {
    var rows: [UUID: CGRect] = [:]
    var follow = true, atBottom = true
    /// The distance to the bottom as the chat view computes it (iOS 18 FollowTracking), to compare with `dist`.
    var geoDist: CGFloat = .nan
    private var scenario = ""
    private weak var model: ChatModel?
    private weak var scroll: UIScrollView?
    private var link: CADisplayLink?
    private let t0 = CACurrentMediaTime()
    private var samples: [[Any]] = []
    private var events: [[Any]] = []
    private var keyboard: CGFloat = 0
    private var frameNo = 0
    private var lastTouch: Double = -1
    private var touches = 0
    private var wasTouching = false
    private var tracked: UUID?
    private var lastWrite: Double = 0
    private var lastState: [Int] = []

    var now: Double { CACurrentMediaTime() - t0 }

    func start(scenario: String, model: ChatModel) {
        self.scenario = scenario
        self.model = model
        let l = CADisplayLink(target: self, selector: #selector(tick))
        l.add(to: .main, forMode: .common)
        link = l
        let nc = NotificationCenter.default
        nc.addObserver(forName: UIResponder.keyboardWillChangeFrameNotification, object: nil, queue: .main) { [weak self] n in
            let f = (n.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect) ?? .zero
            let h = max(0, UIScreen.main.bounds.height - f.minY)
            Task { @MainActor in self?.keyboard = h; self?.mark("keyboard frame \(Int(h))") }
        }
    }

    func mark(_ s: String) {
        events.append([round3(now), s])
        log("UITEST t=\(String(format: "%.2f", now)) \(s)")
    }

    func beginHold() {
        guard let s = scroll else { return }
        let top = visibleTop(s)
        let ids = Set((model?.conv.visible ?? []).map(\.id))
        let rows = self.rows.filter { ids.contains($0.key) }
        tracked = rows.filter { $0.value.minY >= top - 1 }.min { $0.value.minY < $1.value.minY }?.key
            ?? rows.filter { $0.value.maxY > top }.min { $0.value.minY < $1.value.minY }?.key
        mark("hold start, tracked row \(tracked.map { short($0) } ?? "none")")
    }

    func waitForHold() async {
        while true {
            if touches > 0 && !wasTouching && now - lastTouch > 0.5 { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    func waitForTouches(quiet: Double) async {
        while true {
            if touches > 0 && !wasTouching && now - lastTouch > quiet { return }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }

    private func short(_ id: UUID) -> String { String(id.uuidString.prefix(6)) }
    private func round3(_ v: Double) -> Double { (v * 1000).rounded() / 1000 }
    private func r1(_ v: CGFloat) -> Double { (Double(v) * 10).rounded() / 10 }

    /// The content-space y at the top of the visible area (below the navigation bar).
    private func visibleTop(_ s: UIScrollView) -> CGFloat { s.contentOffset.y + s.adjustedContentInset.top }

    private func findScroll() -> UIScrollView? {
        var best: UIScrollView?
        func walk(_ v: UIView) {
            if let s = v as? UIScrollView, !(s is UITextView), !(s is UICollectionView), !(s is UITableView), s.window != nil,
               s.bounds.height > (best?.bounds.height ?? 200) { best = s }
            for c in v.subviews { walk(c) }
        }
        if let w = UITest.window { walk(w) }
        return best
    }

    @objc private func tick(_ l: CADisplayLink) {
        frameNo += 1
        if scroll == nil || scroll?.window == nil || frameNo % 20 == 0 { scroll = findScroll() }
        let t = now
        guard let s = scroll, let m = model else {
            samples.append([round3(t)])
            return
        }
        let touching = s.isTracking || s.isDragging || s.isDecelerating
        if touching { lastTouch = t; if !wasTouching { touches += 1; mark("touch \(touches) begins") } }
        if !touching && wasTouching { mark("touch \(touches) ends") }
        wasTouching = touching
        let o = s.contentOffset.y, ch = s.contentSize.height, bh = s.bounds.height
        let it = s.adjustedContentInset.top, ib = s.adjustedContentInset.bottom
        // the row frames are in the scroll content's space
        let top = visibleTop(s), bottom = o + bh - ib
        let ids = Set(m.conv.visible.map(\.id))
        let visible = rows.filter { ids.contains($0.key) && $0.value.maxY > top + 1 && $0.value.minY < bottom }
        let topRow = visible.min { $0.value.minY < $1.value.minY }
        let trackedY: Any = tracked.flatMap { rows[$0] }.map { r1($0.minY - top) } ?? NSNull()
        let last = m.conv.messages.last
        // what changed in the chat this frame, as events
        let state = [m.conv.visible.count, (last?.content.contains("</think>") ?? false) ? 1 : 0, m.isGenerating ? 1 : 0,
                     m.toolRunningID != nil ? 1 : 0, last?.content.isEmpty ?? true ? 1 : 0]
        if state != lastState, !lastState.isEmpty {
            if state[0] != lastState[0] { mark("rows \(lastState[0]) -> \(state[0])") }
            if state[1] == 1 && lastState[1] == 0 { mark("answer starts (thinking collapses)") }
            if state[2] != lastState[2] { mark(state[2] == 1 ? "reply starts" : "reply ends (caption, sources)") }
            if state[3] != lastState[3] { mark(state[3] == 1 ? "tool runs" : "tool result") }
            if state[4] == 1 && lastState[4] == 0 && state[0] == lastState[0] { mark("last row emptied (retry)") }
        }
        lastState = state
        samples.append([
            round3(t), r1(o), r1(ch), r1(bh), r1(it), r1(ib), r1(ch + ib - bh - o),
            s.isTracking ? 1 : 0, s.isDecelerating ? 1 : 0, r1(s.panGestureRecognizer.translation(in: s).y),
            follow ? 1 : 0, atBottom ? 1 : 0, r1(keyboard), trackedY,
            topRow.map { short($0.key) } ?? NSNull(), topRow.map { r1($0.value.minY - top) } ?? NSNull(),
            m.conv.visible.count, last?.content.count ?? 0, m.isGenerating ? 1 : 0, m.toolRunningID != nil ? 1 : 0,
            String(m.conv.id.uuidString.prefix(6)), geoDist.isNaN ? NSNull() : r1(geoDist),
        ])
        if t - lastWrite > 2 { write() }
    }

    func write() {
        lastWrite = now
        let out: [String: Any] = [
            "scenario": scenario,
            "legacy": FollowTracking.legacy,
            "os": UIDevice.current.systemVersion,
            "token_interval_s": UITestFlags.tokenInterval,
            "fields": ["t", "offset", "content_h", "container_h", "inset_top", "inset_bottom", "dist", "tracking",
                       "decelerating", "pan_y", "follow", "at_bottom", "keyboard", "tracked_y", "top_row", "top_row_y",
                       "rows", "last_len", "generating", "tool_running", "conv", "geo_dist"],
            "events": events,
            "samples": samples,
        ]
        if let d = try? JSONSerialization.data(withJSONObject: out) {
            try? d.write(to: docsURL.appendingPathComponent("uitest-\(scenario).json"))
        }
    }
}
#endif
