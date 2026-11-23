// The chat prompt and the thinking segment for Qwen3.6-35B-A3B.
//
// Prompt: the release's own chat template (chat_template.jinja in the model folder), rendered by swift-transformers'
// Jinja with add_generation_prompt = true and enable_thinking passed as given (nil: not passed, the template's default =
// thinking on). The ids equal transformers' apply_chat_template.
//
// Tools: `ToolPrompt.systemContent` renders the template's tools system turn into a plain system message's content, which
// is how the app declares its web_search tool.
//
// Thinking segment, by token id (`<think>` and `</think>` are single added tokens that also decode as text): Qwen3.6's
// generation prompt already opens the block, so with thinking on the reply starts INSIDE it. ThinkingTracker starts
// inside when the prompt's last assistant header is followed by an unclosed `<think>`, marks each generated token open /
// thinking / close / answer, and, with a budget, forces the template's close (`\n</think>\n\n`, or `</think>\n\n` after
// a newline) once the block has used `budget` tokens; the forced tokens are fed like sampled ones and are part of the reply.
import Foundation
import Tokenizers

public enum ChatPrompt {
    /// The prompt ids for `messages` (role/content dictionaries), add_generation_prompt = true, enable_thinking =
    /// `thinking` (nil: not passed).
    public static func ids(_ tk: Tokenizer, messages: [[String: String]], thinking: Bool?) throws -> [Int] {
        let msgs: [Message] = messages.map { $0 as [String: any Sendable] }
        return try tk.applyChatTemplate(messages: msgs, tools: nil, additionalContext: thinking.map { ["enable_thinking": $0] })
    }

    /// The template's own render with `tools` and `addGenerationPrompt` explicit (the app passes tools: nil and renders the
    /// tools block into the system content itself, ToolPrompt.systemContent).
    public static func ids(_ tk: Tokenizer, messages: [[String: any Sendable]], thinking: Bool?, tools: [[String: any Sendable]]?,
                           addGenerationPrompt: Bool = true) throws -> [Int] {
        try tk.applyChatTemplate(messages: messages, chatTemplate: nil, addGenerationPrompt: addGenerationPrompt, truncation: false,
                                 maxLength: nil, tools: tools, additionalContext: thinking.map { ["enable_thinking": $0] })
    }
}

/// Tools in the system turn. The app renders string messages without `tools`; with tools the release's template writes
/// `<|im_start|>system\n` + the tools block + ("\n\n" + system prompt, trimmed, if any) + `<|im_end|>\n`, and without tools `<|im_start|>system\n` + content (trimmed) + `<|im_end|>\n`.
/// So a system message whose content is the tools block + "\n\n" + the prompt renders the same bytes. The block's text
/// is the template's, verbatim; each tool is one line of JSON as transformers' `tojson` writes it (json.dumps: ", " and
/// ": " separators, keys in the spec's order; the app's specs are written with sorted keys, as swift-jinja would).
public enum ToolPrompt {
    public static let head = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
    public static let tail = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

    /// The system message content that renders as the template's tools system turn. `toolsJSON`: one JSON line per tool.
    public static func systemContent(system: String, toolsJSON: [String]) -> String {
        var s = head
        for j in toolsJSON { s += "\n" + j }
        s += tail
        let t = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { s += "\n\n" + t }
        return s
    }

    /// The app's search tool: terse, one parameter, keys sorted. `n` (number of results, default 3) is accepted
    /// by the app but not declared: declaring it costs 19 more system-turn tokens (272 -> 291 with a one-line system prompt).
    public static let webSearchJSON = #"{"function": {"description": "Search the web.", "name": "web_search", "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"], "type": "object"}}, "type": "function"}"#

    /// The tool call tags (single added tokens 248058 / 248059 in Qwen3.6's vocabulary, special = false: they decode as text).
    public static let callOpen = "<tool_call>", callClose = "</tool_call>"
}

/// The think-block token ids of a tokenizer (nil when it has none) and the template's close continuation.
public struct ThinkTokens: Sendable {
    public let open: Int?  // <think>
    public let close: Int?  // </think>
    public let imStart: Int?  // <|im_start|>
    /// "\n</think>\n\n" and "</think>\n\n" (after a newline): the template's `reasoning + '\n</think>\n\n' + content`.
    public let closeAfterText: [Int]
    public let closeAfterNewline: [Int]

    public init(_ tk: Tokenizer) {
        open = tk.convertTokenToId("<think>")
        close = tk.convertTokenToId("</think>")
        imStart = tk.convertTokenToId("<|im_start|>")
        closeAfterText = close == nil ? [] : tk.encode(text: "\n</think>\n\n", addSpecialTokens: false)
        closeAfterNewline = close == nil ? [] : tk.encode(text: "</think>\n\n", addSpecialTokens: false)
    }

    /// A model whose reasoning segment has other ids (Inkling 2 Mini: opened by `<|content_thinking|>`, closed by
    /// `<|end_message|>`, a forced close continues with the answer's header).
    public init(open: Int, close: Int, forcedClose: [Int]) {
        self.open = open
        self.close = close
        imStart = nil
        closeAfterText = forcedClose
        closeAfterNewline = forcedClose
    }
}

public enum ThinkPhase: String, Sendable {
    case open  // a <think> token (the model opening a block itself, or a redundant one inside the block)
    case thinking  // inside the block
    case close  // the </think> token
    case answer  // after the block (or no block)
}

/// The thinking segment of one reply, by token id.
public struct ThinkingTracker: Sendable {
    public let tokens: ThinkTokens
    /// Tokens inside the block (the open/close tags excluded) after which the close is forced; nil: no budget.
    public let budget: Int?
    /// The prompt ends inside a think block (Qwen3.6's generation prompt with thinking on: `<|im_start|>assistant\n<think>\n`).
    public let startedInside: Bool
    public private(set) var inside: Bool
    public private(set) var thinkingTokens = 0
    public private(set) var closed = false
    public private(set) var forcedClose = false
    private var pendingForce: [Int] = []

    public init(promptIds: [Int], tokens: ThinkTokens, budget: Int?) {
        self.tokens = tokens
        self.budget = budget
        var ins = false
        if let o = tokens.open {
            let from = tokens.imStart.flatMap { promptIds.lastIndex(of: $0) } ?? 0
            for id in promptIds[from...] {
                if id == o { ins = true } else if id == tokens.close { ins = false }
            }
        }
        startedInside = ins
        inside = ins
    }

    /// The phase of a token fed after the prompt (sampled or forced), updating the state.
    public mutating func observe(_ id: Int) -> ThinkPhase {
        if id == tokens.open { inside = true; return .open }
        if inside && id == tokens.close { inside = false; closed = true; return .close }
        if inside { thinkingTokens += 1; return .thinking }
        return .answer
    }

    /// The next token to feed instead of sampling, when the budget forces the close (nil: sample). `lastText` is the
    /// decoded reply so far (decides whether the close needs its leading newline).
    public mutating func forced(lastText: () -> String) -> Int? {
        if pendingForce.isEmpty, let b = budget, inside, !forcedClose, thinkingTokens >= b, tokens.close != nil {
            pendingForce = lastText().hasSuffix("\n") ? tokens.closeAfterNewline : tokens.closeAfterText
            forcedClose = true
        }
        return pendingForce.isEmpty ? nil : pendingForce.removeFirst()
    }

    public var summary: [String: Any] {
        ["started_inside": startedInside, "thinking_tokens": thinkingTokens, "closed": closed, "forced_close": forcedClose,
         "budget": budget.map { $0 as Any } ?? NSNull()]
    }
}
