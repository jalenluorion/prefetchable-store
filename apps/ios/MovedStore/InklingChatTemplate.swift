import Foundation

/// Inkling 2 Mini's text front end without a Jinja engine: a hand-written renderer equivalent to the release's
/// chat_template.jinja (thinkingmachines/Inkling-2-Mini) and the response parser of its tokenizer_config.json
/// response_template. The spec is the engine's Python front end (python/serve/generic/chat_format.py: jinja2 with
/// transformers' tojson = json.dumps(ensure_ascii=False, ...)).
///
/// The format, in short:
/// - every message is `<|message_ROLE|>[name]<|content_KIND|>TEXT<|end_message|>`; an assistant turn ends with
///   `<|content_model_end_sampling|>` (the stop token, id 200006);
/// - a system message `Thinking effort level: E` goes before the first non-system message (E from reasoning_effort:
///   none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh / max 0.99, or a number in [0, 0.99]; default 0.9);
/// - tools are declared first: `<|message_system|>tool_declare<|content_xml|>` + the specs as sorted compact JSON;
/// - a tool call: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>`;
///   a tool result: `<|message_tool|>NAME<|content_text|>RESULT<|end_message|>`;
/// - there is NO generation prompt: the model starts its own turn with `<|message_model|>`.
public enum InklingChat {
    // MARK: - JSON with Python's json.dumps output (ensure_ascii=False) and insertion order kept

    public indirect enum JSON: Equatable, Sendable {
        case null, bool(Bool), int(Int64), double(Double), string(String), array([JSON]), object([(String, JSON)])

        public static func == (a: JSON, b: JSON) -> Bool {
            switch (a, b) {
            case (.null, .null): return true
            case let (.bool(x), .bool(y)): return x == y
            case let (.int(x), .int(y)): return x == y
            case let (.double(x), .double(y)): return x == y
            case let (.string(x), .string(y)): return x == y
            case let (.array(x), .array(y)): return x == y
            case let (.object(x), .object(y)): return x.count == y.count && zip(x, y).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 }
            default: return false
            }
        }

        public var isTruthy: Bool {  // Jinja truthiness
            switch self {
            case .null: return false
            case .bool(let b): return b
            case .int(let i): return i != 0
            case .double(let d): return d != 0
            case .string(let s): return !s.isEmpty
            case .array(let a): return !a.isEmpty
            case .object(let o): return !o.isEmpty
            }
        }

        public subscript(key: String) -> JSON? {
            if case .object(let o) = self { return o.first(where: { $0.0 == key })?.1 }
            return nil
        }

        public var string: String? { if case .string(let s) = self { return s }; return nil }

        /// json.dumps(self, ensure_ascii=False, separators=(item, key), sort_keys=sorted)
        public func dumps(item: String = ", ", key: String = ": ", sorted: Bool = false) -> String {
            var out = ""
            write(&out, item, key, sorted)
            return out
        }

        private func write(_ o: inout String, _ item: String, _ key: String, _ sorted: Bool) {
            switch self {
            case .null: o += "null"
            case .bool(let b): o += b ? "true" : "false"
            case .int(let i): o += String(i)
            case .double(let d): o += JSON.pyFloat(d)
            case .string(let s): JSON.quote(s, into: &o)
            case .array(let a):
                o += "["
                for (i, v) in a.enumerated() {
                    if i > 0 { o += item }
                    v.write(&o, item, key, sorted)
                }
                o += "]"
            case .object(let kv):
                let items = sorted ? kv.sorted { JSON.pyLess($0.0, $1.0) } : kv
                o += "{"
                for (i, (k, v)) in items.enumerated() {
                    if i > 0 { o += item }
                    JSON.quote(k, into: &o)
                    o += key
                    v.write(&o, item, key, sorted)
                }
                o += "}"
            }
        }

        /// Python's str < str: code point order
        static func pyLess(_ a: String, _ b: String) -> Bool {
            a.unicodeScalars.lexicographicallyPrecedes(b.unicodeScalars) { $0.value < $1.value }
        }

        /// Python's float repr (shortest round-trip; Swift's description agrees on the forms used here: 0.5, 1.0, 1e+16, 1e-05)
        static func pyFloat(_ d: Double) -> String {
            if d.isNaN { return "NaN" }
            if d.isInfinite { return d > 0 ? "Infinity" : "-Infinity" }
            return d.description
        }

        static func quote(_ s: String, into o: inout String) {
            o += "\""
            for u in s.unicodeScalars {
                switch u {
                case "\"": o += "\\\""
                case "\\": o += "\\\\"
                case "\n": o += "\\n"
                case "\r": o += "\\r"
                case "\t": o += "\\t"
                case "\u{08}": o += "\\b"
                case "\u{0C}": o += "\\f"
                default:
                    if u.value < 0x20 { o += String(format: "\\u%04x", u.value) } else { o.unicodeScalars.append(u) }
                }
            }
            o += "\""
        }

        /// An order-keeping parser (Python's json.loads keeps object order; ints stay ints).
        public static func parse(_ text: String) -> JSON? {
            var p = Parser(Array(text.unicodeScalars))
            guard let v = p.value() else { return nil }
            p.ws()
            return p.i == p.s.count ? v : nil
        }

        struct Parser {
            let s: [Unicode.Scalar]
            var i = 0
            init(_ s: [Unicode.Scalar]) { self.s = s }
            mutating func ws() { while i < s.count, " \t\n\r".unicodeScalars.contains(s[i]) { i += 1 } }
            mutating func lit(_ w: String) -> Bool {
                let a = Array(w.unicodeScalars)
                guard i + a.count <= s.count, Array(s[i..<i + a.count]) == a else { return false }
                i += a.count
                return true
            }
            mutating func value() -> JSON? {
                ws()
                guard i < s.count else { return nil }
                switch s[i] {
                case "{":
                    i += 1
                    var kv: [(String, JSON)] = []
                    ws()
                    if i < s.count, s[i] == "}" { i += 1; return .object(kv) }
                    while true {
                        ws()
                        guard i < s.count, s[i] == "\"", let k = str() else { return nil }
                        ws()
                        guard i < s.count, s[i] == ":" else { return nil }
                        i += 1
                        guard let v = value() else { return nil }
                        if let j = kv.firstIndex(where: { $0.0 == k }) { kv[j].1 = v } else { kv.append((k, v)) }  // last wins, first place
                        ws()
                        guard i < s.count else { return nil }
                        if s[i] == "," { i += 1; continue }
                        if s[i] == "}" { i += 1; return .object(kv) }
                        return nil
                    }
                case "[":
                    i += 1
                    var a: [JSON] = []
                    ws()
                    if i < s.count, s[i] == "]" { i += 1; return .array(a) }
                    while true {
                        guard let v = value() else { return nil }
                        a.append(v)
                        ws()
                        guard i < s.count else { return nil }
                        if s[i] == "," { i += 1; continue }
                        if s[i] == "]" { i += 1; return .array(a) }
                        return nil
                    }
                case "\"": return str().map { .string($0) }
                case "t": return lit("true") ? .bool(true) : nil
                case "f": return lit("false") ? .bool(false) : nil
                case "n": return lit("null") ? .null : nil
                default: return number()
                }
            }
            mutating func str() -> String? {
                i += 1
                var out = String.UnicodeScalarView()
                while i < s.count {
                    let c = s[i]
                    i += 1
                    if c == "\"" { return String(out) }
                    if c != "\\" { out.append(c); continue }
                    guard i < s.count else { return nil }
                    let e = s[i]
                    i += 1
                    switch e {
                    case "\"": out.append("\"")
                    case "\\": out.append("\\")
                    case "/": out.append("/")
                    case "b": out.append("\u{08}")
                    case "f": out.append("\u{0C}")
                    case "n": out.append("\n")
                    case "r": out.append("\r")
                    case "t": out.append("\t")
                    case "u":
                        guard var v = hex4() else { return nil }
                        if (0xD800..<0xDC00).contains(v), i + 1 < s.count, s[i] == "\\", s[i + 1] == "u" {
                            i += 2
                            guard let lo = hex4() else { return nil }
                            v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00)
                        }
                        guard let u = Unicode.Scalar(v) else { return nil }
                        out.append(u)
                    default: return nil
                    }
                }
                return nil
            }
            mutating func hex4() -> UInt32? {
                guard i + 4 <= s.count, let v = UInt32(String(String.UnicodeScalarView(s[i..<i + 4])), radix: 16) else { return nil }
                i += 4
                return v
            }
            mutating func number() -> JSON? {
                let st = i
                var isFloat = false
                while i < s.count, "+-0123456789.eE".unicodeScalars.contains(s[i]) {
                    if ".eE".unicodeScalars.contains(s[i]) { isFloat = true }
                    i += 1
                }
                let t = String(String.UnicodeScalarView(s[st..<i]))
                if !isFloat, let v = Int64(t) { return .int(v) }
                return Double(t).map { .double($0) }
            }
        }
    }

    // MARK: - Messages

    public struct ToolCall: Sendable {
        public var id: String?
        public var name: String
        public var arguments: JSON  // a parsed object (the template refuses a JSON string)
        public init(id: String? = nil, name: String, arguments: JSON) { self.id = id; self.name = name; self.arguments = arguments }
    }

    public struct Message: Sendable {
        public var role: String  // system | user | assistant | tool
        public var content: String
        public var name: String?
        public var reasoning: String?  // reasoning_content of an assistant turn in the history
        public var toolCalls: [ToolCall]
        public var toolCallId: String?
        public init(role: String, content: String, name: String? = nil, reasoning: String? = nil, toolCalls: [ToolCall] = [], toolCallId: String? = nil) {
            self.role = role; self.content = content; self.name = name; self.reasoning = reasoning; self.toolCalls = toolCalls; self.toolCallId = toolCallId
        }
    }

    /// reasoning_effort: a name, a number, or nil (the template's default 0.9)
    public enum Effort: Sendable { case name(String), value(Double) }

    public enum RenderError: Error, Equatable { case unknownRole(String), badEffort(String), toolName(String) }

    public static let roleToken = ["user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>"]
    static let effortMap: [String: Double] = ["none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99]

    /// Token ids that end generation: generation_config.json's eos (<|content_model_end_sampling|>) and the pad token.
    public static let stopIds: Set<Int> = [200006, 199999]
    public static let endSampling = "<|content_model_end_sampling|>"

    static func effortText(_ e: Effort?) throws -> String {
        var num: Double
        switch e {
        case nil: num = 0.9
        case .name(let s):
            let k = s.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let v = effortMap[k] else { throw RenderError.badEffort(s) }
            num = v
        case .value(let v): num = v
        }
        guard num >= 0, num <= 0.99 else { throw RenderError.badEffort("\(num)") }
        return "<|message_system|><|content_text|>Thinking effort level: " + (num == 0 ? "0" : JSON.pyFloat(num)) + "<|end_message|>"
    }

    /// The prompt text (no generation prompt: the model opens its own turn). tools: OpenAI tool objects
    /// ({"type": "function", "function": {name, description, parameters}}) as JSON.
    public static func render(_ messages: [Message], tools: [JSON]? = nil, effort: Effort? = nil) throws -> String {
        var out = ""
        if let tools, !tools.isEmpty {
            var specs: [JSON] = []
            for t in tools {
                let fn = t["function"] ?? t
                let desc = fn["description"].flatMap { $0.isTruthy ? $0 : nil } ?? .string("")
                let params = fn["parameters"].flatMap { $0.isTruthy ? $0 : nil } ?? .object([])
                let type = t["type"].flatMap { $0.isTruthy ? $0 : nil } ?? .string("function")
                specs.append(.object([("description", desc), ("name", fn["name"] ?? .null), ("parameters", params), ("type", type)]))
            }
            out += "<|message_system|>tool_declare<|content_xml|>" + JSON.array(specs).dumps(item: ",", key: ":", sorted: true) + "<|end_message|>"
        }
        var effortDone = false
        for m in messages {
            guard let rt = roleToken[m.role] else { throw RenderError.unknownRole(m.role) }
            if !effortDone && m.role != "system" {
                out += try effortText(effort)
                effortDone = true
            }
            var header = rt
            if m.role == "tool" {
                if let n = m.name, !n.isEmpty {
                    header += n
                } else if let id = m.toolCallId, !id.isEmpty {
                    for prev in messages where prev.role == "assistant" {  // every match, in order (as the template's loops)
                        for tc in prev.toolCalls where tc.id == id { header += tc.name }
                    }
                }
            } else if m.role == "assistant", let n = m.name, !n.isEmpty {
                header += n
            }
            let hasCalls = !m.toolCalls.isEmpty, hasReasoning = !(m.reasoning ?? "").isEmpty
            if m.role == "assistant", let r = m.reasoning, !r.isEmpty {
                out += header + "<|content_thinking|>" + r + "<|end_message|>"
            }
            if !m.content.isEmpty || !(m.role == "assistant" && (hasCalls || hasReasoning)) {
                out += header + "<|content_text|>" + m.content + "<|end_message|>"
            }
            if m.role == "assistant" {
                for tc in m.toolCalls {
                    if let n = m.name, !n.isEmpty, n != tc.name { throw RenderError.toolName(tc.name) }
                    var args: [(String, JSON)] = []
                    switch tc.arguments {
                    case .object(let kv): args = kv
                    case .null: args = []
                    default: throw RenderError.toolName(tc.name)  // the template: arguments must be an object
                    }
                    out += "<|message_model|>" + tc.name + "<|content_invoke_tool_json|>"
                    out += "{\"name\":" + JSON.string(tc.name).dumps(item: ",", key: ":") + ",\"args\":{"
                    let sortedArgs = args.sorted { JSON.pyLess($0.0, $1.0) }  // dictsort(case_sensitive=true)
                    for (i, (k, v)) in sortedArgs.enumerated() {
                        if i > 0 { out += "," }
                        out += JSON.string(k).dumps() + ":" + v.dumps(item: ",", key: ":")  // nested objects keep their order
                    }
                    out += "}}<|end_message|>"
                }
                out += endSampling
            }
        }
        if !effortDone { out += try effortText(effort) }
        return out
    }

    // MARK: - The generated turn (decoded with special tokens kept)

    public struct Turn: Equatable, Sendable {
        public var reasoning: String?
        public var content: String
        public var calls: [(name: String, arguments: String)]  // arguments: json.dumps(args, ensure_ascii=False), OpenAI style
        public static func == (a: Turn, b: Turn) -> Bool {
            a.reasoning == b.reasoning && a.content == b.content && a.calls.count == b.calls.count
                && zip(a.calls, b.calls).allSatisfy { $0.name == $1.name && $0.arguments == $1.arguments }
        }
    }

    enum Field { case reasoning, content, toolCall }
    static let opens: [(Field, NSRegularExpression)] = [
        (.reasoning, try! NSRegularExpression(pattern: "(?:<\\|message_model\\|>[^<]*)?<\\|content_thinking\\|>", options: [.dotMatchesLineSeparators])),
        (.content, try! NSRegularExpression(pattern: "(?:<\\|message_model\\|>[^<]*)?<\\|content_(?:text|tool_error|invoke_tool_text)\\|>", options: [.dotMatchesLineSeparators])),
        (.toolCall, try! NSRegularExpression(pattern: "(?:<\\|message_model\\|>[^<]*)?<\\|content_invoke_tool_json\\|>", options: [.dotMatchesLineSeparators])),
    ]
    static let close = "<|end_message|>"

    /// = chat_format.parse_turn: segments in order (each opened by its field's marker, closed by <|end_message|>); text
    /// fields joined unstripped; a tool call's JSON {"name", "args"} becomes (name, json.dumps(args)); an unfinished call
    /// (no valid JSON) is not a call; text outside any field is dropped.
    public static func parseTurn(_ text: String) -> Turn {
        let ns = text as NSString
        var pos = 0
        var reasoning: String?, content: String?
        var calls: [(String, String)] = []
        while pos < ns.length {
            var best: (Field, NSRange)?
            for (f, re) in opens {
                if let m = re.firstMatch(in: text, range: NSRange(location: pos, length: ns.length - pos)),
                   best == nil || m.range.location < best!.1.location {
                    best = (f, m.range)
                }
            }
            guard let (f, r) = best else { break }
            let start = r.location + r.length
            let cr = ns.range(of: close, range: NSRange(location: start, length: ns.length - start))
            let segEnd = cr.location == NSNotFound ? ns.length : cr.location
            let seg = ns.substring(with: NSRange(location: start, length: segEnd - start))
            switch f {
            case .reasoning: reasoning = (reasoning ?? "") + seg
            case .content: content = (content ?? "") + seg
            case .toolCall:
                if let j = JSON.parse(seg) {  // valid JSON: a call (a non-object gives an empty name and {}, as Python)
                    let name = j["name"]?.string ?? ""
                    let a = j["args"]
                    let args: JSON = (a == nil || a == .null) ? .object([]) : a!
                    calls.append((name, args.string ?? args.dumps()))
                }
            }
            if cr.location == NSNotFound { break }
            pos = max(cr.location + cr.length, start)
        }
        return Turn(reasoning: reasoning, content: content ?? "", calls: calls)
    }
}
