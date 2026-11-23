// The text side the engine leaves to the app: the model folder's tokenizer and chat template (swift-transformers), stop
// ids and the think-block tokens.
//
// Two front ends, chosen by the folder's moved-store.json `family`:
// - Qwen3.6-35B-A3B (qwen3_5_moe): the release's Jinja template through swift-transformers (ChatPrompt);
// - Inkling 2 Mini (inkling2_loop): InklingChat, the hand-written renderer of its template (InklingChatTemplate.swift;
//   swift-jinja is never used for it). The app's messages keep the Qwen conventions (a `<think>` block and `<tool_call>`
//   JSON in an assistant's content, the tools block in the system content: ToolPrompt), and are mapped onto Inkling's
//   messages here (inklingMessages); a generated Inkling turn is shown in the same conventions (inklingAppText).
import Foundation
import Tokenizers

public final class TextCodec: @unchecked Sendable {
    public let tokenizer: Tokenizer
    public let stopIds: Set<Int>
    public let thinkTokens: ThinkTokens
    public let dir: String
    /// The folder's model family (moved-store.json `family`), "" when unknown.
    public let family: String
    public var isInkling: Bool { family == "inkling2_loop" }
    /// Inkling: the app's thinking switch. On: the template's default effort (0.9, "high"); the thinking limit setting
    /// still ends a long reasoning segment. Off: effort "none" (0) and the model's turn opened as an answer
    /// (`<|message_model|><|content_text|>` after the prompt): at effort 0 alone it still reasons first. This is the counterpart of Qwen's thinking-off generation
    /// prompt (an empty think block), and the history renders it back exactly (an answer-only turn).
    public var inklingEffortOn: InklingChat.Effort? = nil
    public static let inklingEffortOff = InklingChat.Effort.name("none")
    public static let inklingAnswerLead = "<|message_model|><|content_text|>"
    public static let inklingAnswerLeadIds = [200001, 200004]
    /// The text the prompt already holds of the model's turn (thinking off: the answer's opening), for parsing a reply.
    public func inklingLead(thinking: Bool) -> String { isInkling && !thinking ? Self.inklingAnswerLead : "" }

    init(tokenizer: Tokenizer, stopIds: Set<Int>, dir: String, family: String) {
        self.tokenizer = tokenizer
        self.dir = dir
        self.family = family
        if family == "inkling2_loop" {
            // <|content_thinking|> 200008 opens the reasoning, <|end_message|> 200010 closes it; a forced close is
            // <|end_message|><|message_model|><|content_text|> (the answer's segment)
            self.thinkTokens = ThinkTokens(open: 200008, close: 200010, forcedClose: [200010, 200001, 200004])
            self.stopIds = stopIds.union(InklingChat.stopIds)
        } else {
            self.thinkTokens = ThinkTokens(tokenizer)
            self.stopIds = stopIds
        }
    }

    /// tokenizer.json, tokenizer_config.json, chat_template.jinja and generation_config.json of `dir`.
    public static func load(dir: String) async throws -> TextCodec {
        let tk = try await AutoTokenizer.from(modelFolder: URL(fileURLWithPath: dir))
        var stop = Set<Int>()
        if let d = try? Data(contentsOf: URL(fileURLWithPath: dir + "/generation_config.json")),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            if let a = j["eos_token_id"] as? [NSNumber] { a.forEach { stop.insert($0.intValue) } }
            if let a = j["eos_token_id"] as? NSNumber { stop.insert(a.intValue) }
        }
        if let e = tk.eosTokenId { stop.insert(e) }
        return TextCodec(tokenizer: tk, stopIds: stop, dir: dir, family: family(dir: dir))
    }

    /// moved-store.json's `family` (a small read: the key is near the top of the file).
    public static func family(dir: String) -> String {
        guard let d = try? Data(contentsOf: URL(fileURLWithPath: dir + "/moved-store.json")),
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return "" }
        return j["family"] as? String ?? ""
    }

    /// The prompt ids (add_generation_prompt, enable_thinking = `thinking`). Inkling: its
    /// render (no generation prompt: the model opens its own turn), effort from `thinking`.
    public func promptIds(messages: [[String: String]], thinking: Bool) throws -> [Int] {
        if isInkling { return encode(try inklingRender(messages, thinking: thinking)) + (thinking ? [] : Self.inklingAnswerLeadIds) }
        return try ChatPrompt.ids(tokenizer, messages: messages, thinking: thinking)
    }

    /// The ids of `messages` without the generation prompt (a prefix of promptIds when the template is prefix-stable).
    /// Inkling has no generation prompt: its render (thinking off: promptIds less the answer's opening).
    public func prefixIds(messages: [[String: String]], thinking: Bool) throws -> [Int] {
        if isInkling { return encode(try inklingRender(messages, thinking: thinking)) }
        return try ChatPrompt.ids(tokenizer, messages: messages.map { $0 as [String: any Sendable] }, thinking: thinking, tools: nil,
                                  addGenerationPrompt: false)
    }

    public func decode(_ ids: [Int]) -> String { tokenizer.decode(tokens: ids) }
    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
    public var generationDefaults: SamplingParams { SamplingParams.generationConfig(dir: dir) }

    // MARK: Inkling

    public func inklingRender(_ messages: [[String: String]], thinking: Bool) throws -> String {
        let (msgs, tools) = Self.inklingMessages(messages)
        return try InklingChat.render(msgs, tools: tools, effort: thinking ? inklingEffortOn : Self.inklingEffortOff)
    }

    /// The app's messages (Qwen conventions) as Inkling's: the tools block of a system content (ToolPrompt.systemContent)
    /// becomes the declared tools and the rest the system text; an assistant's `<think>\nR\n</think>\n\n` becomes its
    /// reasoning (kept for every turn, as it was generated: the next turn then continues the engine's state) and each
    /// `<tool_call>\n{"name": N, "arguments": A}\n</tool_call>` a tool call; a tool result is named by the last call
    /// before it.
    public static func inklingMessages(_ messages: [[String: String]]) -> ([InklingChat.Message], [InklingChat.JSON]?) {
        var out: [InklingChat.Message] = []
        var tools: [InklingChat.JSON]? = nil
        var lastCall: String? = nil
        for d in messages {
            let role = d["role"] ?? "user"
            var content = d["content"] ?? ""
            switch role {
            case "system":
                if content.hasPrefix(ToolPrompt.head), let t = content.range(of: ToolPrompt.tail) {
                    let block = content[content.index(content.startIndex, offsetBy: ToolPrompt.head.count)..<t.lowerBound]
                    tools = block.split(separator: "\n").compactMap { InklingChat.JSON.parse(String($0)) }
                    content = String(content[t.upperBound...])
                    if content.hasPrefix("\n\n") { content.removeFirst(2) }
                }
                out.append(.init(role: "system", content: content))
            case "assistant":
                var reasoning: String? = nil
                if content.hasPrefix("<think>"), let c = content.range(of: "</think>") {
                    var r = String(content[content.index(content.startIndex, offsetBy: 7)..<c.lowerBound])
                    if r.hasPrefix("\n") { r.removeFirst() }
                    if r.hasSuffix("\n") { r.removeLast() }
                    reasoning = r
                    content = String(content[c.upperBound...])
                    if content.hasPrefix("\n\n") { content.removeFirst(2) }
                }
                var calls: [InklingChat.ToolCall] = []
                while let o = content.range(of: ToolPrompt.callOpen) {
                    let rest = content[o.upperBound...]
                    let close = rest.range(of: ToolPrompt.callClose)
                    let body = String(close.map { rest[..<$0.lowerBound] } ?? rest).trimmingCharacters(in: .whitespacesAndNewlines)
                    if let j = InklingChat.JSON.parse(body), let name = j["name"]?.string {
                        var args = j["arguments"] ?? j["args"] ?? .object([])
                        if case .string(let s) = args { args = InklingChat.JSON.parse(s) ?? .object([]) }
                        calls.append(.init(name: name, arguments: args))
                        lastCall = name
                    }
                    let after = close.map { String(rest[$0.upperBound...]) } ?? ""
                    content = String(content[..<o.lowerBound]) + after
                }
                out.append(.init(role: "assistant", content: content, reasoning: reasoning, toolCalls: calls))
            case "tool":
                out.append(.init(role: "tool", content: content, name: lastCall))
            default:
                out.append(.init(role: role, content: content))
            }
        }
        return (out, tools)
    }

    /// A generated Inkling turn (decoded with its special tokens) in the app's conventions: the reasoning as a `<think>`
    /// block (closed once its segment ended), the answer, each call as `<tool_call>` JSON (name, arguments).
    public static func inklingAppText(_ raw: String) -> String {
        let t = InklingChat.parseTurn(raw)
        var s = ""
        if let r = t.reasoning {
            s += "<think>\n" + r
            var closed = !t.content.isEmpty || !t.calls.isEmpty
            if !closed, let o = raw.range(of: "<|content_thinking|>", options: .backwards) {
                closed = raw[o.upperBound...].contains("<|end_message|>")
            }
            if closed { s += "\n</think>\n\n" }
        }
        s += t.content
        for c in t.calls {
            s += ToolPrompt.callOpen + "\n{\"name\": " + InklingChat.JSON.string(c.name).dumps() + ", \"arguments\": " + c.arguments + "}\n"
                + ToolPrompt.callClose
        }
        return s
    }
}
