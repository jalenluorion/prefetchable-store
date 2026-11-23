import Foundation
import Security
// Web search through the model's native tool calling.
//
// The tool is declared in the system turn exactly as the release's chat template writes it for `tools=[...]`
// (ToolPrompt.systemContent). The model answers with a `<tool_call>`; ChatModel stops the round there, runs the search,
// appends the call turn and a `tool` turn and generates again from the engine's live state (only the result is
// prefilled). A search reads the top result pages and gives the model selected sentences (facts), not titles: per source
// the title, the host and 2-4 sentences ranked by overlap with the query, within `ToolSettings.resultTokenCap` tokens of
// the engine's tokenizer. `open_url(result | url)` (declared when page reads are on) returns one page's selected
// sentences within the same budget; `result` is the [n] of the last search's results (the model is shown hosts, not URLs,
// so a URL it writes is usually invented).

/// The search backends. DuckDuckGo's HTML endpoint needs no key; Brave Search API and Tavily take a key the user pastes
/// in Settings > Web search (kept in the Keychain).
enum ToolBackend: String, CaseIterable, Identifiable, Codable {
    case duckduckgo, brave, tavily
    var id: String { rawValue }
    var label: String {
        switch self {
        case .duckduckgo: return "DuckDuckGo (no key)"
        case .brave: return "Brave Search API"
        case .tavily: return "Tavily"
        }
    }
    var short: String {
        switch self {
        case .duckduckgo: return "DuckDuckGo"
        case .brave: return "Brave"
        case .tavily: return "Tavily"
        }
    }
    var needsKey: Bool { self != .duckduckgo }
}

/// Settings > Web search (UserDefaults; the keys are in the Keychain).
enum ToolSettings {
    static let enabledKey = "tools.enabled"
    static let backendKey = "tools.backend"
    static let pageReadsKey = "tools.pageReads"
    static var d: UserDefaults { .standard }

    /// Tools in new chats (default on).
    static var enabled: Bool { d.object(forKey: enabledKey) as? Bool ?? true }
    static var backend: ToolBackend { ToolBackend(rawValue: d.string(forKey: backendKey) ?? "") ?? .duckduckgo }
    /// Page reads (default on): a search reads its top result pages, and `open_url` is declared. Off: snippets only, one
    /// tool (the open_url declaration costs system-turn tokens in every chat).
    static var pageReads: Bool { d.object(forKey: pageReadsKey) as? Bool ?? true }
    /// Tool calls run per message; a further call gets an error result and the model answers with what it has. A failed
    /// open_url of a URL that was not among the last search's results (an invented one) does not count, once per message
    /// (the error lists the results, so the retry can succeed). Each round costs the phone tens of seconds.
    static let maxCalls = 3
    /// The tool result's token budget (the engine's tokenizer): the result turn's first prefill chunk (112 + the turn's
    /// 16 template tokens = 128 positions).
    static let defaultResultTokens = 112
    static var resultTokenCap: Int { defaultResultTokens }
    static let defaultResults = 3
    /// Result pages a search reads (the top ones; the rest keep their snippets).
    static let pagesPerSearch = 2
    /// Bytes of a page read at most (the rest is not downloaded).
    static let pageByteCap = 1 << 20
    static let timeout: TimeInterval = 10

    /// The second tool (declared when page reads are on), keys sorted as swift-jinja's `tojson` writes them (the same form
    /// as ToolPrompt.webSearchJSON). `result`: the [n] of a result of the last search (resolved to its stored URL, so the
    /// model never has to know one); `url` stays as the alternative. Neither is required.
    static let openURLJSON = #"{"function": {"description": "Open a result from the last search by its number, e.g. result 1. Use only if the results do not contain the answer.", "name": "open_url", "parameters": {"properties": {"result": {"description": "the [n] number", "type": "integer"}, "url": {"type": "string"}}, "type": "object"}}, "type": "function"}"#
    static var toolsJSON: [String] { pageReads ? [ToolPrompt.webSearchJSON, openURLJSON] : [ToolPrompt.webSearchJSON] }

    /// The system prompt's addition when tools are on (after the tools block and the user's system prompt). It names the
    /// current month: without a date the model takes events after its training as still to come and never searches. The
    /// month, not the day, so the saved system state changes once a month instead of daily.
    static var systemAddition: String {
        "It is \(currentMonth). Use web_search only when the answer needs current or specific facts: search once, with a short query. Answer in the user's language and name the source (title or site) you used."
    }

    static var currentMonth: String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "MMMM yyyy"
        return f.string(from: Date())
    }

    /// The system message content the engine is given: the template's tools block, the system prompt and the addition.
    static func systemContent(_ system: String, tools: Bool) -> String {
        let s = system.trimmingCharacters(in: .whitespacesAndNewlines)
        guard tools else { return s }
        return ToolPrompt.systemContent(system: s.isEmpty ? systemAddition : s + "\n\n" + systemAddition,
                                        toolsJSON: toolsJSON)
    }
}

/// API keys for the keyed backends: generic passwords in the Keychain (service <bundle id>.search, account = the
/// backend). Never logged, never written to UserDefaults or the conversation files.
enum SearchKeychain {
    static let service = (Bundle.main.bundleIdentifier ?? "app") + ".search"

    static func get(_ b: ToolBackend) -> String? {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: b.rawValue, kSecReturnData as String: true,
                                kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }

    /// Stores `key` (empty: deletes it). Returns nil on success, else a short reason.
    @discardableResult
    static func set(_ b: ToolBackend, _ key: String) -> String? {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                   kSecAttrAccount as String: b.rawValue]
        SecItemDelete(base as CFDictionary)
        let k = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !k.isEmpty else { return nil }
        var add = base
        add[kSecValueData as String] = Data(k.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let st = SecItemAdd(add as CFDictionary, nil)
        return st == errSecSuccess ? nil : "Keychain error \(st)"
    }

    static func has(_ b: ToolBackend) -> Bool { !(get(b) ?? "").isEmpty }
}

// MARK: the call

/// One parsed tool call. The template's format is XML-like (`<function=NAME>` + `<parameter=K>\nV\n</parameter>`); a JSON
/// body (`{"name": …, "arguments": {…}}`, Qwen2.5 / Hermes style) is accepted too.
struct ToolCall: Codable, Equatable, Sendable {
    var name: String
    var arguments: [String: String]

    /// The search query, or the page of an `open_url` call ("result N" when it names a search result).
    var query: String { arguments["query"] ?? arguments["q"] ?? arguments["url"] ?? arguments["result"].map { "result " + $0 } ?? "" }
    /// open_url's `result` (the [n] of a search result): "1", "[1]", "result 1" all give 1. A `url` that is only a number
    /// counts too (a model that put the number in the other parameter).
    var resultNumber: Int? {
        for v in [arguments["result"], arguments["url"]] {
            guard let v else { continue }
            let digits = v.lowercased().replacingOccurrences(of: "result", with: "")
                .trimmingCharacters(in: CharacterSet(charactersIn: " []#\n"))
            if let n = Int(digits) { return n }
        }
        return nil
    }
    var isPageRead: Bool { name == "open_url" }

    /// The text before the call (shown as the answer) and the call's own text, when `answer` contains `<tool_call>`.
    static func split(_ answer: String) -> (before: String, call: String, complete: Bool)? {
        guard let o = answer.range(of: ToolPrompt.callOpen) else { return nil }
        let before = String(answer[..<o.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let rest = answer[o.upperBound...]
        if let c = rest.range(of: ToolPrompt.callClose) {
            return (before, String(rest[..<c.lowerBound]), true)
        }
        return (before, String(rest), false)
    }

    /// Parses a call body (between the tags). Partial bodies (still streaming) give what is there so far.
    static func parse(_ body: String) -> ToolCall? {
        let b = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if b.hasPrefix("{"), let d = b.data(using: .utf8), let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
           let name = j["name"] as? String {
            var args: [String: String] = [:]
            var a: Any? = j["arguments"] ?? j["parameters"]
            if let s = a as? String, let d2 = s.data(using: .utf8) { a = try? JSONSerialization.jsonObject(with: d2) }
            for (k, v) in (a as? [String: Any]) ?? [:] { args[k] = (v as? String) ?? "\(v)" }
            return ToolCall(name: name, arguments: args)
        }
        guard let f = b.range(of: "<function=") else { return nil }
        let afterF = b[f.upperBound...]
        guard let gt = afterF.firstIndex(of: ">") else { return nil }
        let name = String(afterF[..<gt]).trimmingCharacters(in: .whitespaces)
        var args: [String: String] = [:]
        var rest = afterF[afterF.index(after: gt)...]
        while let p = rest.range(of: "<parameter=") {
            let a = rest[p.upperBound...]
            guard let g = a.firstIndex(of: ">") else { break }
            let key = String(a[..<g]).trimmingCharacters(in: .whitespaces)
            let vStart = a.index(after: g)
            let end = a[vStart...].range(of: "</parameter>")
            let raw = end.map { a[vStart..<$0.lowerBound] } ?? a[vStart...]
            // the template writes "\n" + value + "\n" around each value
            var v = String(raw)
            if v.hasPrefix("\n") { v.removeFirst() }
            if v.hasSuffix("\n") { v.removeLast() }
            args[key] = v
            guard let e = end else { break }
            rest = a[e.upperBound...]
        }
        return name.isEmpty ? nil : ToolCall(name: name, arguments: args)
    }
}

// MARK: results

struct SearchResult: Codable, Equatable, Sendable {
    var title: String
    var url: String
    /// The backend's snippet; in a tool meta, the sentences the model was given for this source.
    var snippet: String
    /// The page itself was read (its selected sentences, not the snippet, went to the model). nil: not read.
    var read: Bool? = nil

    /// host + path without the scheme, "www.", query and fragment (what the model is given, to save tokens).
    var shortURL: String {
        guard let u = URLComponents(string: url), var host = u.host else { return url }
        if host.hasPrefix("www.") { host.removeFirst(4) }
        // at most the first two path segments, 32 characters
        let segs = u.path.split(separator: "/").prefix(2)
        var path = segs.isEmpty ? "" : "/" + segs.joined(separator: "/")
        if path.count > 32 { path = String(path.prefix(32)) }
        return host + path
    }
    var host: String {
        guard var h = URLComponents(string: url)?.host else { return url }
        if h.hasPrefix("www.") { h.removeFirst(4) }
        return h
    }
}

/// What a tool round did, saved in the tool message's meta (the card reads it).
struct ToolMeta: Codable, Equatable, Sendable {
    var name: String
    var query: String
    var backend: String
    var results: [SearchResult] = []
    /// Tokens of the result text as the model reads it (the engine's tokenizer; nil before the model loads).
    var resultTokens: Int?
    /// Wall time of the search (network + parsing + trimming).
    var ms: Double = 0
    var error: String?
    /// Pages read for this result (the top results' pages; open_url: 1), and the token budget it was fitted to.
    var pages: Int? = nil
    var budget: Int? = nil
    /// Tavily's `answer` (include_answer), when given to the model.
    var answer: String? = nil
    /// Wall time of the page reads alone (inside `ms`).
    var pagesMs: Double? = nil
    /// open_url: the [n] of the last search's results the page was (1-based; from `result`, or a `url` equal to one of
    /// them), nil when the URL was not among them (the model wrote it).
    var resultIndex: Int? = nil
}

/// The web search: fetch, parse, trim to the token budget, format for the model.
enum WebSearch {
    static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"

    static var session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = ToolSettings.timeout
        c.timeoutIntervalForResource = ToolSettings.timeout
        c.httpAdditionalHeaders = ["User-Agent": userAgent, "Accept-Language": "en-US,en;q=0.8"]
        return URLSession(configuration: c)
    }()

    enum Failure: LocalizedError {
        case noKey(ToolBackend), http(Int), parse, empty, badURL, notHTML, noText
        var errorDescription: String? {
            switch self {
            case .noKey(let b): return "no \(b.short) API key"
            case .http(let c): return c == 202 || c == 429 ? "rate-limited (HTTP \(c)); try again later or use a key" : "HTTP \(c)"
            case .parse: return "unreadable response"
            case .empty: return "no results"
            case .badURL: return "not a public http(s) URL"
            case .notHTML: return "not a web page"
            case .noText: return "no readable text"
            }
        }
    }

    /// Runs the call. Returns the text given to the model (the passages, or a short error) and the meta for the card.
    /// `context`: the user's message (open_url ranks the page's sentences by it and by the last search's query).
    /// `last`: the last search's results as the model was shown them ([1] = last[0]); open_url's `result` names one.
    /// `count`: measures a candidate result's tokens (the engine's tokenizer); nil: an estimate of ~3.8 characters / token.
    static func run(_ call: ToolCall, context: String = "", last: [SearchResult] = [], count: (String) -> Int?) async -> (text: String, meta: ToolMeta) {
        let backend = ToolSettings.backend
        var meta = ToolMeta(name: call.name, query: call.query, backend: backend.short)
        let t0 = Date()
        let budget = ToolSettings.resultTokenCap
        let known = call.name == "web_search" || (call.name == "open_url" && ToolSettings.pageReads)
        guard known else {
            meta.error = "unknown tool \(call.name)"
            let text = "Error: unknown tool \(call.name). Answer without tools."
            meta.resultTokens = count(text)
            return (text, meta)
        }
        if call.isPageRead { return await openURL(call, context: context, last: last, budget: budget, count: count) }
        let q = call.query.trimmingCharacters(in: .whitespacesAndNewlines)
        let n = min(max(Int(call.arguments["n"] ?? "") ?? ToolSettings.defaultResults, 1), 5)
        do {
            guard !q.isEmpty else { throw Failure.empty }
            var rs: [SearchResult]
            var answer: String?
            switch backend {
            case .duckduckgo: rs = try await duckDuckGo(q)
            case .brave: rs = try await brave(q, n: n)
            case .tavily: (rs, answer) = try await tavily(q, n: n)
            }
            rs = Array(rs.filter { !$0.title.isEmpty && !$0.url.isEmpty }.prefix(n))
            guard !rs.isEmpty else { throw Failure.empty }
            // the top result pages (Tavily's `content` is already the page's relevant extract)
            var pages: [Int: [String]] = [:]
            if ToolSettings.pageReads && backend != .tavily {
                let tp = Date()
                // the top `pagesPerSearch` pages; one more is fetched alongside and used only in place of a page that
                // fails or has no text (a script-rendered page: weather.com's forecast is in its JavaScript)
                let picks = Array(rs.indices.filter { Readable.worthReading(rs[$0].url) }.prefix(ToolSettings.pagesPerSearch + 1))
                let got = await withTaskGroup(of: (Int, [String]?).self) { g in
                    for i in picks { g.addTask { (i, try? await readPage(rs[i].url).blocks) } }
                    var out: [Int: [String]] = [:]
                    for await (i, b) in g { if let b, b.reduce(0, { $0 + $1.count }) >= 200 { out[i] = b } }
                    return out
                }
                for i in picks where got[i] != nil && pages.count < ToolSettings.pagesPerSearch { pages[i] = got[i] }
                meta.pagesMs = Date().timeIntervalSince(tp) * 1000
            }
            let terms = Passages.terms(q)
            var sources: [Passages.Source] = rs.enumerated().map { i, r in
                let t = shortTitle(r.title)
                var ranked = pages[i].map { Passages.rank(blocks: $0, terms: terms, lead: false, title: t) } ?? []
                let fromPage = !ranked.isEmpty
                if !fromPage { ranked = Passages.rank(blocks: [r.snippet], terms: terms, lead: true, title: t) }
                return Passages.Source(title: t, host: r.host, url: r.url, sentences: ranked, read: fromPage, maxKeep: fromPage ? 4 : 2)
            }
            pages = pages.filter { i, _ in sources[i].read }
            if let a = answer?.trimmingCharacters(in: .whitespacesAndNewlines), !a.isEmpty {
                meta.answer = Passages.cut(a, words: 60)
            }
            let (text, kept, toks) = Passages.compose(&sources, answer: meta.answer, budget: budget, count: count)
            meta.results = kept
            meta.resultTokens = toks
            meta.pages = pages.count
            meta.budget = budget
            meta.ms = Date().timeIntervalSince(t0) * 1000
            return (text, meta)
        } catch {
            meta.error = reason(error)
            meta.ms = Date().timeIntervalSince(t0) * 1000
            let text = "Error: search failed (\(meta.error!))."
            meta.resultTokens = count(text)
            return (text, meta)
        }
    }

    static func reason(_ error: Error) -> String {
        if let u = error as? URLError {
            return u.code == .timedOut ? "timeout" : (u.code == .notConnectedToInternet ? "offline" : "network error \(u.code.rawValue)")
        }
        return error.localizedDescription
    }

    /// `open_url(result | url)`: the page's sentences most relevant to the user's message and the last search (the context),
    /// in page order, within the budget. A page with nothing matching gives its first sentences. `result` N opens the last
    /// search's [N] (its stored URL); a `url` equal to a result's URL (or only its host, as the model is shown it) is that
    /// result. Any other URL is still tried, but a failure (an invented URL: HTTP 404) returns an error that lists the
    /// results by number, so the next call can open one.
    static func openURL(_ call: ToolCall, context: String, last: [SearchResult] = [], budget: Int, count: (String) -> Int?) async -> (text: String, meta: ToolMeta) {
        var meta = ToolMeta(name: call.name, query: call.query, backend: "page")
        let t0 = Date()
        var raw = (call.arguments["url"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        func listed() -> String {
            last.enumerated().map { "[\($0.offset + 1)] \($0.element.title) (\($0.element.host))" }.joined(separator: " ")
        }
        func fail(_ text: String, _ why: String) -> (text: String, meta: ToolMeta) {
            meta.error = why
            meta.ms = Date().timeIntervalSince(t0) * 1000
            meta.resultTokens = count(text)
            return (text, meta)
        }
        if let n = call.resultNumber {
            guard last.indices.contains(n - 1) else {
                return fail(last.isEmpty ? "Error: no search results to open; use web_search first."
                                         : "Error: there is no result \(n). Results: \(listed())", "no result \(n)")
            }
            meta.resultIndex = n
            raw = last[n - 1].url
        } else if let i = resultIndex(of: raw, in: last) {
            meta.resultIndex = i + 1
            raw = last[i].url
        }
        guard !raw.isEmpty else {
            return fail(last.isEmpty ? "Error: no page given." : "Error: no page given. Open a result by number: \(listed())", "no page given")
        }
        meta.query = raw
        do {
            guard let u = Readable.safeURL(raw) else { throw Failure.badURL }
            let page = try await readPage(u.absoluteString)
            meta.pagesMs = Date().timeIntervalSince(t0) * 1000
            guard !page.blocks.isEmpty else { throw Failure.noText }
            let title = shortTitle(page.title.isEmpty ? (u.host ?? raw) : page.title)
            var src = [Passages.Source(title: title, host: SearchResult(title: "", url: u.absoluteString, snippet: "").host,
                                       url: u.absoluteString,
                                       sentences: Passages.rank(blocks: page.blocks, terms: Passages.terms(context), lead: false, title: title),
                                       read: true, maxKeep: 40)]
            let (text, kept, toks) = Passages.compose(&src, answer: nil, budget: budget, count: count)
            meta.results = kept
            meta.resultTokens = toks
            meta.pages = 1
            meta.budget = budget
            meta.ms = Date().timeIntervalSince(t0) * 1000
            return (text, meta)
        } catch {
            let why = reason(error)
            // a URL the model wrote (not a result): say which pages exist instead of a bare 404
            if meta.resultIndex == nil && !last.isEmpty {
                return fail("Error: could not read that page (\(why)). Open a search result by number instead: \(listed())", why)
            }
            return fail("Error: could not read the page (\(why)).", why)
        }
    }

    /// The index in `last` of the result `raw` names: the same URL (scheme, "www.", a trailing "/" and the fragment
    /// ignored), or only the host of exactly one result.
    static func resultIndex(of raw: String, in last: [SearchResult]) -> Int? {
        guard !raw.isEmpty, !last.isEmpty else { return nil }
        func key(_ s: String) -> String {
            var t = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if let r = t.range(of: "://") { t = String(t[r.upperBound...]) }
            if t.hasPrefix("www.") { t.removeFirst(4) }
            if let h = t.firstIndex(of: "#") { t = String(t[..<h]) }
            while t.hasSuffix("/") { t.removeLast() }
            return t
        }
        let k = key(raw)
        if let i = last.firstIndex(where: { key($0.url) == k }) { return i }
        if !k.contains("/") {
            let byHost = last.indices.filter { last[$0].host.lowercased() == k }
            if byHost.count == 1 { return byHost[0] }
        }
        return nil
    }

    /// Downloads a page (10 s, at most `pageByteCap` bytes; HTML or plain text only) and extracts its readable blocks.
    static func readPage(_ url: String) async throws -> (title: String, blocks: [String]) {
        guard let u = Readable.safeURL(url) else { throw Failure.badURL }
        var req = URLRequest(url: u)
        req.timeoutInterval = ToolSettings.timeout
        req.setValue("text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5", forHTTPHeaderField: "Accept")
        let (bytes, resp) = try await session.bytes(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw Failure.http(code) }
        let type = ((resp as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") ?? "text/html").lowercased()
        guard type.contains("html") || type.contains("text/plain") else { throw Failure.notHTML }
        var data = Data()
        data.reserveCapacity(256 << 10)
        var buf = [UInt8]()
        buf.reserveCapacity(64 << 10)
        for try await b in bytes {
            buf.append(b)
            if buf.count == 64 << 10 { data.append(contentsOf: buf); buf.removeAll(keepingCapacity: true) }
            if data.count + buf.count >= ToolSettings.pageByteCap { break }
        }
        data.append(contentsOf: buf)
        let html = String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
        let plainText = type.contains("text/plain")
        return await Task.detached(priority: .userInitiated) {
            plainText ? ("", Readable.plainBlocks(html)) : Readable.extract(html)
        }.value
    }

    /// The first `n` words (with "…" when cut).
    static func words(_ s: String, _ n: Int) -> String {
        let w = s.split(separator: " ")
        return w.count <= n ? s : w.prefix(n).joined(separator: " ") + "…"
    }

    /// A title without its site suffix (" | AccuWeather", " - The Weather Channel": the host names the site), at most 70
    /// characters.
    static func shortTitle(_ t: String) -> String {
        var s = t
        for sep in [" | ", " - ", " – ", " — "] {
            if let r = s.range(of: sep, options: .backwards), s.distance(from: s.startIndex, to: r.lowerBound) >= 16 {
                s = String(s[..<r.lowerBound])
            }
        }
        if s.hasSuffix(" ...") { s = String(s.dropLast(4)) }
        return s.count > 70 ? String(s.prefix(70)) : s
    }

    static func firstSentence(_ s: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        // the first ". ", "! " or "? " after 20 characters ends the sentence
        let chars = Array(t)
        var i = 20
        while i < chars.count - 1 {
            if ".!?".contains(chars[i]) && chars[i + 1] == " " { return String(chars[...i]) }
            i += 1
        }
        return t
    }

    // MARK: backends

    static func get(_ req: URLRequest) async throws -> Data {
        let (d, r) = try await session.data(for: req)
        let code = (r as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw Failure.http(code) }
        return d
    }

    /// DuckDuckGo's HTML endpoint (html.duckduckgo.com/html/, a POST form like the no-JavaScript page): the organic
    /// results' title links and snippets; ads (`result--ad`) skipped.
    static func duckDuckGo(_ q: String) async throws -> [SearchResult] {
        var req = URLRequest(url: URL(string: "https://html.duckduckgo.com/html/")!)
        req.httpMethod = "POST"
        req.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        var form = URLComponents()
        form.queryItems = [URLQueryItem(name: "q", value: q), URLQueryItem(name: "kl", value: "us-en")]
        req.httpBody = (form.percentEncodedQuery ?? "").replacingOccurrences(of: "+", with: "%2B").data(using: .utf8)
        let d = try await get(req)
        guard let html = String(data: d, encoding: .utf8) else { throw Failure.parse }
        return parseDuckDuckGo(html)
    }

    static func parseDuckDuckGo(_ html: String) -> [SearchResult] {
        var out: [SearchResult] = []
        // one block per result: from `class="result results_links` to the next one
        let blocks = html.components(separatedBy: "class=\"result results_links").dropFirst()
        let aRe = try! NSRegularExpression(pattern: #"class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#, options: [.dotMatchesLineSeparators])
        let sRe = try! NSRegularExpression(pattern: #"class="result__snippet"[^>]*>(.*?)</a>"#, options: [.dotMatchesLineSeparators])
        for b in blocks {
            let head = b.prefix(200)
            if head.contains("result--ad") { continue }
            let ns = b as NSString
            guard let m = aRe.firstMatch(in: b, range: NSRange(location: 0, length: ns.length)) else { continue }
            let href = decodeDDGLink(entities(ns.substring(with: m.range(at: 1))))
            let title = plain(ns.substring(with: m.range(at: 2)))
            var snip = ""
            if let s = sRe.firstMatch(in: b, range: NSRange(location: 0, length: ns.length)) { snip = plain(ns.substring(with: s.range(at: 1))) }
            if href.contains("duckduckgo.com/y.js") { continue }  // an ad link
            out.append(SearchResult(title: title, url: href, snippet: snip))
        }
        return out
    }

    /// `//duckduckgo.com/l/?uddg=<url>&…` redirect links → the target.
    static func decodeDDGLink(_ h: String) -> String {
        var s = h.hasPrefix("//") ? "https:" + h : h
        if let c = URLComponents(string: s), c.host?.hasSuffix("duckduckgo.com") == true, c.path == "/l/",
           let u = c.queryItems?.first(where: { $0.name == "uddg" })?.value {
            s = u
        }
        return s
    }

    /// Brave Search API (api.search.brave.com/res/v1/web/search; header X-Subscription-Token).
    static func brave(_ q: String, n: Int) async throws -> [SearchResult] {
        guard let key = SearchKeychain.get(.brave), !key.isEmpty else { throw Failure.noKey(.brave) }
        var c = URLComponents(string: "https://api.search.brave.com/res/v1/web/search")!
        c.queryItems = [URLQueryItem(name: "q", value: q), URLQueryItem(name: "count", value: "\(n)"),
                        URLQueryItem(name: "extra_snippets", value: "true")]
        var req = URLRequest(url: c.url!)
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        req.setValue(key, forHTTPHeaderField: "X-Subscription-Token")
        let d = try await get(req)
        guard let j = try JSONSerialization.jsonObject(with: d) as? [String: Any] else { throw Failure.parse }
        let rs = ((j["web"] as? [String: Any])?["results"] as? [[String: Any]]) ?? []
        // `extra_snippets` (up to 5 more excerpts per result, on plans that include them) are facts from the page
        return rs.map { r in
            let extra = (r["extra_snippets"] as? [String] ?? []).map(plain)
            return SearchResult(title: plain(r["title"] as? String ?? ""), url: r["url"] as? String ?? "",
                                snippet: ([plain(r["description"] as? String ?? "")] + extra).filter { !$0.isEmpty }.joined(separator: "\n"))
        }
    }

    /// Tavily (api.tavily.com/search; Authorization: Bearer key): `include_answer` gives a short answer written from the
    /// results, and each result's `content` is the page's passage most relevant to the query (facts, not a teaser), so no
    /// page is read for Tavily.
    static func tavily(_ q: String, n: Int) async throws -> ([SearchResult], String?) {
        guard let key = SearchKeychain.get(.tavily), !key.isEmpty else { throw Failure.noKey(.tavily) }
        var req = URLRequest(url: URL(string: "https://api.tavily.com/search")!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["query": q, "max_results": n, "search_depth": "basic",
                                                                   "include_answer": true])
        let d = try await get(req)
        guard let j = try JSONSerialization.jsonObject(with: d) as? [String: Any] else { throw Failure.parse }
        let rs = (j["results"] as? [[String: Any]]) ?? []
        let out = rs.map { SearchResult(title: plain($0["title"] as? String ?? ""), url: $0["url"] as? String ?? "",
                                        snippet: plain($0["content"] as? String ?? "")) }
        return (out, (j["answer"] as? String).map(plain))
    }

    // MARK: HTML text

    static func plain(_ s: String) -> String {
        let noTags = s.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
        return entities(noTags).replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func entities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var t = s
        for (k, v) in [("&quot;", "\""), ("&#x27;", "'"), ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">"),
                       ("&nbsp;", " "), ("&#x2F;", "/"), ("&hellip;", "…"), ("&mdash;", "—"), ("&ndash;", "–"),
                       ("&deg;", "°"), ("&rsquo;", "’"), ("&lsquo;", "‘"), ("&rdquo;", "”"), ("&ldquo;", "“"),
                       ("&apos;", "'"), ("&middot;", "·"), ("&times;", "×"), ("&minus;", "−"), ("&plusmn;", "±"),
                       ("&frac12;", "½"), ("&copy;", "©"), ("&reg;", "®"), ("&trade;", "™"), ("&euro;", "€"),
                       ("&pound;", "£"), ("&eacute;", "é"), ("&amp;", "&")] {
            t = t.replacingOccurrences(of: k, with: v)
        }
        // numeric entities
        let re = try! NSRegularExpression(pattern: "&#(x?)([0-9a-fA-F]+);")
        let ns = t as NSString
        var out = ""
        var last = 0
        for m in re.matches(in: t, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let hex = ns.substring(with: m.range(at: 1)) == "x"
            let num = ns.substring(with: m.range(at: 2))
            if let v = UInt32(num, radix: hex ? 16 : 10), let u = Unicode.Scalar(v) { out.append(Character(u)) }
            last = m.range.location + m.range.length
        }
        out += ns.substring(from: last)
        return out
    }
}

// MARK: page text

/// The readable text of a web page: a simple readability heuristic, no DOM. Scripts, styles, navigation, headers,
/// footers, asides, forms and comments are cut; the page's `<main>` (or its one large `<article>`) is the scope when it
/// has text, else `<body>`; block-level tags end a block; blocks that are mostly link text (menus, link lists) and short
/// boilerplate (cookies, sign-in, subscribe) are dropped, unless the page is nothing but links (a front page), whose
/// link blocks are then the text.
enum Readable {
    /// Result pages a search reads: http(s), not a video / social / PDF page whose HTML carries no article text.
    static func worthReading(_ url: String) -> Bool {
        guard let u = safeURL(url), let h = u.host?.lowercased() else { return false }
        let skip = ["youtube.com", "youtu.be", "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com",
                    "pinterest.com", "linkedin.com", "reddit.com"]
        if skip.contains(where: { h == $0 || h.hasSuffix("." + $0) }) { return false }
        let p = u.path.lowercased()
        return ![".pdf", ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".zip", ".doc", ".docx", ".xls", ".ppt"].contains(where: p.hasSuffix)
    }

    /// An http(s) URL to a public host (no localhost, .local, private / loopback / link-local IPv4, IPv6 literals); a bare
    /// host gets https://.
    static func safeURL(_ s: String) -> URL? {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.contains("://") { t = "https://" + t }
        guard let u = URL(string: t), let sch = u.scheme?.lowercased(), sch == "http" || sch == "https",
              let h = u.host?.lowercased(), !h.isEmpty, !h.contains(":") else { return nil }
        if h == "localhost" || [".local", ".localhost", ".internal", ".lan", ".home"].contains(where: h.hasSuffix) { return nil }
        let p = h.split(separator: ".").compactMap { Int($0) }
        if p.count == 4 {
            let a = p[0], b = p[1]
            if a == 0 || a == 10 || a == 127 || (a == 169 && b == 254) || (a == 172 && (16...31).contains(b))
                || (a == 192 && b == 168) || (a == 100 && (64...127).contains(b)) || a >= 224 { return nil }
        }
        return u
    }

    static func re(_ p: String) -> NSRegularExpression {
        try! NSRegularExpression(pattern: p, options: [.caseInsensitive, .dotMatchesLineSeparators])
    }
    static func replace(_ r: NSRegularExpression, _ s: String, _ with: String) -> String {
        r.stringByReplacingMatches(in: s, range: NSRange(location: 0, length: (s as NSString).length), withTemplate: with)
    }
    static func groups(_ r: NSRegularExpression, _ s: String, _ g: Int) -> [String] {
        let ns = s as NSString
        return r.matches(in: s, range: NSRange(location: 0, length: ns.length)).compactMap {
            $0.range(at: g).location == NSNotFound ? nil : ns.substring(with: $0.range(at: g))
        }
    }

    static let cutTags = ["script", "style", "noscript", "svg", "template", "iframe", "head", "nav", "header", "footer",
                          "aside", "form", "button", "select", "dialog", "menu"].map { re("<\($0)\\b[^>]*>.*?</\($0)\\s*>") }
    static let comment = re("<!--.*?-->")
    static let titleRe = re("<title[^>]*>(.*?)</title>")
    static let mainRe = re("<main\\b[^>]*>(.*?)</main\\s*>")
    static let articleRe = re("<article\\b[^>]*>(.*?)</article\\s*>")
    static let bodyRe = re("<body[^>]*>(.*)</body>")
    /// A tag's attributes, quoted values included (a value may contain ">": Wikipedia's data-mw JSON does).
    static let attrs = #"(?:[^>"']|"[^"]*"|'[^']*')*"#
    static let aOpen = re("<a\\b" + attrs + ">"), aClose = re("</a\\s*>")
    static let blockTag = re("</?(p|div|li|ul|ol|h[1-6]|tr|table|tbody|thead|tfoot|section|article|main|br|hr|dd|dt|dl|blockquote|pre|figure|figcaption|caption|label|option|summary|details|time)\\b" + attrs + ">")
    static let cellTag = re("</?(td|th)\\b" + attrs + ">")
    static let anyTag = re("<" + attrs + ">")
    static let spaces = try! NSRegularExpression(pattern: "[ \\t\\r\\u00a0\\u200b]+")
    /// Wikipedia-style footnote markers ("[6]", "[citation needed]") and the spaces tag removal leaves before punctuation.
    static let footnote = try! NSRegularExpression(pattern: "\\[(\\d{1,3}|[a-z]|citation needed|note \\d+)\\]")
    static let spaceBeforePunct = try! NSRegularExpression(pattern: "\\s+([,.;:!?)\\]])")
    static let spaceAfterOpen = try! NSRegularExpression(pattern: "([(\\[])\\s+")
    static let boiler = ["cookie", "subscribe", "sign in", "sign up", "log in", "newsletter", "all rights reserved",
                         "privacy policy", "terms of use", "terms of service", "javascript", "advertisement",
                         "skip to content", "your browser", "accept all", "download the app", "follow us",
                         "click for", "click here", "get premium", "create your account", "you are here"]

    static func extract(_ html0: String) -> (title: String, blocks: [String]) {
        let title = groups(titleRe, html0, 1).first.map(WebSearch.plain) ?? ""
        var html = replace(comment, html0, " ")
        for r in cutTags { html = replace(r, html, " ") }
        var scope = groups(bodyRe, html, 1).first ?? html
        let mains = groups(mainRe, html, 1)
        let arts = groups(articleRe, html, 1)
        if let m = mains.max(by: { $0.count < $1.count }), WebSearch.plain(m).count >= 400 {
            scope = m
        } else if arts.count <= 2, let a = arts.max(by: { $0.count < $1.count }), WebSearch.plain(a).count >= 400 {
            scope = a
        }
        // anchors marked (\u{1} … \u{2}) so a block's link text can be measured once the tags are gone
        var t = replace(aOpen, scope, "\u{1}")
        t = replace(aClose, t, "\u{2}")
        t = replace(blockTag, t, "\n")
        t = replace(cellTag, t, " ")
        t = replace(anyTag, t, "")
        t = WebSearch.entities(t)
        var all: [(text: String, links: Bool)] = []
        var seen = Set<String>()
        for line in t.components(separatedBy: "\n") {
            // link text measured per anchor: a menu is several short links; a headline (one long link) is text
            var inLink = false, linkChars = 0, text = "", anchors = 0, maxAnchorWords = 0, cur = ""
            for ch in line {
                if ch == "\u{1}" || ch == "\u{2}" {
                    if ch == "\u{1}" { inLink = true; anchors += 1; cur = "" } else if inLink {
                        inLink = false
                        maxAnchorWords = max(maxAnchorWords, cur.split(separator: " ").count)
                    }
                    if text.last.map({ !$0.isWhitespace }) ?? false { text.append(" ") }
                    continue
                }
                text.append(ch)
                if inLink { cur.append(ch); if !ch.isWhitespace { linkChars += 1 } }
            }
            text = replace(spaces, text, " ")
            text = replace(footnote, text, "")
            text = replace(spaceBeforePunct, text, "$1")
            text = replace(spaceAfterOpen, text, "$1").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let words = text.split(separator: " ").count
            let letters = text.filter { !$0.isWhitespace }.count
            if words < 2 && !text.contains(where: \.isNumber) { continue }
            let low = text.lowercased()
            if words < 30 && boiler.contains(where: low.contains) { continue }
            guard seen.insert(text).inserted else { continue }
            let headline = anchors <= 2 && maxAnchorWords >= 5
            all.append((text, letters > 0 && Double(linkChars) / Double(letters) > 0.6 && words < 40 && !headline))
            if all.count >= 800 { break }
        }
        var blocks = all.filter { !$0.links }.map(\.text)
        if blocks.reduce(0, { $0 + $1.count }) < 300 { blocks = all.map(\.text) }  // a page of links: they are its text
        return (title, mergeShort(blocks))
    }

    /// Runs of short blocks (table cells, infobox rows, "36°F (2°C)", "Wind Chill 39°F") joined into one block with
    /// "; " (up to ~24 words), so they are ranked as one fact line instead of dropped as fragments.
    static func mergeShort(_ blocks: [String]) -> [String] {
        var out: [String] = []
        var run: [String] = []
        var runWords = 0
        func flush() {
            if !run.isEmpty { out.append(run.joined(separator: "; ")) }
            run = []; runWords = 0
        }
        for b in blocks {
            let w = b.split(separator: " ").count
            if w < 6 && !".!?".contains(b.last ?? " ") {
                if runWords + w > 24 { flush() }
                run.append(b); runWords += w
            } else {
                flush()
                out.append(b)
            }
        }
        flush()
        return out
    }

    static func plainBlocks(_ s: String) -> [String] {
        s.components(separatedBy: "\n").map { replace(spaces, $0, " ").trimmingCharacters(in: .whitespaces) }
            .filter { $0.split(separator: " ").count >= 3 }.prefix(800).map { $0 }
    }
}

// MARK: passage selection

/// Picks the sentences that carry the facts: every block is split into sentences, each scored by its overlap with the
/// query's terms (plus a few expansions: "weather" also matches temperatures, "population" census counts, ...), a small
/// bonus for numbers and for the start of the page; the best ones per source are kept, in page order, within the budget.
enum Passages {
    struct Sentence: Equatable {
        var text: String
        var score: Double
        var pos: Int
    }
    struct Source {
        var title: String
        var host: String
        var url: String
        /// Best first.
        var sentences: [Sentence]
        var read: Bool
        var maxKeep: Int
        var kept: [Sentence] = []
    }

    static let stop: Set<String> = ["the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "are", "was",
        "were", "be", "what", "who", "whom", "which", "when", "where", "why", "how", "does", "do", "did", "it", "its", "this",
        "that", "these", "those", "with", "by", "from", "as", "about", "me", "my", "i", "you", "your", "tell", "please",
        "current", "currently", "now", "right", "today", "latest", "recent", "new", "much", "many", "there", "has", "have",
        "can", "could", "would", "should", "will", "s", "vs", "than", "into", "out", "up"]
    static let expansions: [String: [String]] = [
        "weather": ["temperature", "°f", "°c", "°", "forecast", "humidity", "wind", "feels", "cloudy", "sunny", "rain", "clear", "degrees"],
        "temperature": ["°f", "°c", "°", "degrees"],
        "population": ["residents", "inhabitants", "census", "people", "estimated", "estimate"],
        "ceo": ["chief", "executive", "officer"],
        "won": ["beat", "defeated", "win", "victory", "final", "score", "champion"],
        "win": ["beat", "defeated", "won", "victory", "final", "score", "champion"],
        "winner": ["beat", "defeated", "won", "victory", "final", "champion"],
        "result": ["beat", "defeated", "won", "score", "final"],
        "score": ["beat", "defeated", "won", "final"],
        "story": ["points", "comments"],
    ]

    static func words(_ s: String) -> [String] {
        s.lowercased().split(whereSeparator: { !($0.isLetter || $0.isNumber || $0 == "°") }).map(String.init)
    }
    static func norm(_ w: String) -> String {
        var t = w
        if t.count > 4 && t.hasSuffix("ies") { t = String(t.dropLast(3)) + "y" }
        else if t.count > 3 && t.hasSuffix("s") && !t.hasSuffix("ss") { t.removeLast() }
        return t
    }

    /// The query's terms and weights: its words (stop words out) 1.0, their expansions 0.5.
    static func terms(_ q: String) -> [String: Double] {
        var out: [String: Double] = [:]
        for w in words(q) where !stop.contains(w) && (w.count > 1 || w.first?.isNumber == true) {
            out[norm(w)] = 1.0
            for e in expansions[w] ?? expansions[norm(w)] ?? [] {
                let k = e.contains("°") ? e : norm(e)
                if out[k] == nil { out[k] = 0.5 }
            }
        }
        return out
    }

    static func cut(_ s: String, words n: Int) -> String {
        let w = s.split(separator: " ")
        return w.count <= n ? s : w.prefix(n).joined(separator: " ") + "…"
    }

    /// A sentence ends at . ! ? before a capital or digit, but not after an initial ("G. Sterman", "U.S.").
    static let sentenceEnd = try! NSRegularExpression(pattern: "(?<=[.!?])(?<!\\b[A-Z][.])\\s+(?=[\"“(\\[]?[A-Z0-9])")

    static func sentences(_ block: String) -> [String] {
        let ns = block as NSString
        var out: [String] = []
        var last = 0
        for m in sentenceEnd.matches(in: block, range: NSRange(location: 0, length: ns.length)) {
            out.append(ns.substring(with: NSRange(location: last, length: m.range.location - last)))
            last = m.range.location + m.range.length
        }
        out.append(ns.substring(from: last))
        return out.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }

    /// Every sentence of `blocks`, scored and ranked (best first). `snippet`: the backend's snippet, already chosen for
    /// the query (a bonus, so it ranks by position when nothing matches).
    static func rank(blocks: [String], terms: [String: Double], lead snippet: Bool, title: String = "") -> [Sentence] {
        var out: [Sentence] = []
        var seen = Set<String>()
        let titleWords = Set(words(title).map(norm))
        var pos = 0
        for b in blocks {
            for s0 in sentences(b) {
                let s = cut(s0, words: 30)
                let ws = words(s)
                guard ws.count >= 3, seen.insert(s).inserted else { continue }
                let set = Set(ws.map(norm))
                var sc = 0.0
                let low = s.lowercased()
                for (t, w) in terms where t.contains("°") ? low.contains(t) : set.contains(t) { sc += w }
                if s.contains(where: \.isNumber) { sc += 0.3 }
                if ws.count < 5 { sc -= 0.3 }
                sc += 0.4 * max(0, 1 - Double(pos) / 40) + (pos < 3 && !snippet ? 0.6 : 0)  // a page's lead defines its subject
                // a heading (few words, no sentence end, no number) or a repeat of the title carries no fact
                let numeric = s.contains(where: \.isNumber)
                if ws.count < 8 && !numeric && !".!?".contains(s.last ?? " ") { sc -= 0.8 }
                if !titleWords.isEmpty && ws.count <= titleWords.count + 2 && set.isSubset(of: titleWords) { continue }
                // a merged run of short items that are mostly not numbers is a menu ("Radar & Maps; News & Features; …")
                let items = s.components(separatedBy: "; ")
                if items.count >= 3 && items.filter({ $0.contains(where: \.isNumber) }).count * 2 < items.count { sc -= 1.2 }
                if snippet { sc += 1 }
                out.append(Sentence(text: s, score: sc, pos: pos))
                pos += 1
                if pos >= 1500 { break }
            }
        }
        return out.sorted { $0.score != $1.score ? $0.score > $1.score : $0.pos < $1.pos }
    }

    /// The model's text: `Answer: …` (Tavily), then per source `[i] Title (host)` and its kept sentences in page order.
    static func render(_ sources: [Source], answer: String?) -> String {
        var lines: [String] = []
        if let a = answer { lines.append("Answer: " + a) }
        var i = 0
        for s in sources where !s.kept.isEmpty {
            i += 1
            lines.append("[\(i)] \(s.title) (\(s.host))")
            lines.append(s.kept.sorted { $0.pos < $1.pos }.map(\.text).joined(separator: " "))
        }
        return lines.joined(separator: "\n")
    }

    /// Fits the sources to `budget` tokens: each source's first one or two sentences (two for a page read), then the best
    /// remaining sentences of any source while they fit (an estimate calibrated on this text by one count), then exact
    /// counts, dropping the lowest-scored sentence while over.
    static func compose(_ src: inout [Source], answer: String?, budget: Int, count: (String) -> Int?) -> (String, [SearchResult], Int?) {
        for i in src.indices { src[i].kept = [] }
        let sample = String(src.flatMap { $0.sentences.prefix(12).map(\.text) }.joined(separator: " ").prefix(3000))
        var cpt = 3.8  // characters per token
        if sample.count > 200, let n = count(sample), n > 0 { cpt = Double(sample.utf8.count) / Double(n) }
        func est(_ s: String) -> Int { Int((Double(s.utf8.count) / cpt).rounded(.up)) }
        let headerCost = { (s: Source) in est("[0] \(s.title) (\(s.host))\n") + 1 }
        var used = answer.map { est("Answer: " + $0 + "\n") } ?? 0
        let target = Int(Double(budget) * 0.97)
        var next = Array(repeating: 0, count: src.count)
        func tryAdd(_ i: Int) -> Bool {
            while next[i] < src[i].sentences.count {
                let s = src[i].sentences[next[i]]
                next[i] += 1
                let c = est(s.text + " ") + (src[i].kept.isEmpty ? headerCost(src[i]) : 0)
                if used + c <= target { src[i].kept.append(s); used += c; return true }
            }
            return false
        }
        for i in src.indices { _ = tryAdd(i) }
        for i in src.indices where src[i].read && src[i].maxKeep >= 2 { _ = tryAdd(i) }
        while true {
            let open = src.indices.filter { next[$0] < src[$0].sentences.count && src[$0].kept.count < src[$0].maxKeep && !src[$0].kept.isEmpty }
            guard let i = open.max(by: { src[$0].sentences[next[$0]].score < src[$1].sentences[next[$1]].score }) else { break }
            if !tryAdd(i) { next[i] = src[i].sentences.count }
        }
        var text = render(src, answer: answer)
        var toks = count(text)
        var guardN = 0
        while let t = toks, t > budget, guardN < 60 {
            guardN += 1
            // the lowest-scored kept sentence of a source with more than one; else the last source with any
            let multi = src.indices.filter { src[$0].kept.count > 1 }
            if let i = multi.min(by: { src[$0].kept.map(\.score).min()! < src[$1].kept.map(\.score).min()! }) {
                let j = src[i].kept.indices.min(by: { src[i].kept[$0].score < src[i].kept[$1].score })!
                src[i].kept.remove(at: j)
            } else if let i = src.indices.last(where: { !src[$0].kept.isEmpty }), src.filter({ !$0.kept.isEmpty }).count > 1 {
                src[i].kept = []
            } else if let i = src.indices.first(where: { !src[$0].kept.isEmpty }) {
                src[i].kept[0].text = cut(src[i].kept[0].text, words: max(5, src[i].kept[0].text.split(separator: " ").count * 2 / 3))
            } else { break }
            text = render(src, answer: answer)
            toks = count(text)
        }
        let kept = src.filter { !$0.kept.isEmpty }.map {
            SearchResult(title: $0.title, url: $0.url, snippet: $0.kept.sorted { $0.pos < $1.pos }.map(\.text).joined(separator: " "), read: $0.read)
        }
        return (text, kept, toks ?? est(text))
    }
}

// MARK: search pre-check

/// A deterministic check on the user's message: a message about live or changing facts
/// (weather, news, prices, scores, office holders, "right now" / today / latest ...) gets a one-line hint appended in the
/// engine's copy of the user turn, "(Use web_search.)", so the model calls the tool instead of answering from memory. The
/// displayed message and the conversation file are unchanged; the hint is a pure function of the message text, so every
/// later render of the same history gives the same tokens (the live prefix and the prefix cache still match).
enum SearchHint {
    static let hint = "(Use web_search.)"
    static var enabled: Bool {
        defaultOn
    }
    static let defaultOn = true

    static let triggers: [NSRegularExpression] = [
        // time: the answer is about now
        #"\b(right now|currently|current|today'?s?|tonight|tomorrow|yesterday|this (week|weekend|month|year|morning|afternoon|evening)|latest|most recent|recently|newest|live|breaking|as of now|so far this|this season|at the moment|these days|nowadays)\b"#,
        // topics whose answer changes
        #"\b(weather|forecast|temperature (in|outside|today)|rain(ing)?|snow(ing)?|storms?|hurricanes?|tornado(es)?|news|headlines?|prices?|cost of|stocks?|share price|market cap|bitcoin|ethereum|crypto|exchange rate|inflation|interest rates?|mortgage rates?|scores?|standings|won the|winner of|who won|playoffs?|election|polls?|schedule|release date|box office|top stor(y|ies)|trending)\b"#,
        // office holders and positions
        #"\b(ceo|cfo|cto|president|prime minister|chancellor|governor|mayor|head coach|secretary of|leader of|chair(man|woman|person)? of|owner of|king|queen|pope)\b"#,
    ].map { try! NSRegularExpression(pattern: $0, options: [.caseInsensitive]) }

    static func applies(_ text: String) -> Bool {
        let r = NSRange(location: 0, length: (text as NSString).length)
        return triggers.contains { $0.firstMatch(in: text, range: r) != nil }
    }

    /// The user turn as the engine is given it.
    static func engineContent(_ text: String) -> String {
        guard enabled, applies(text), !text.contains(hint) else { return text }
        return text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n" + hint
    }
}
