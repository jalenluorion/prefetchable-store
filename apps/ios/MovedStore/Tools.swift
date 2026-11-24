import Foundation
import libxml2
import PDFKit
import WebKit
import Security

// Web search through the model's native tool calling.
//
// The tool is declared in the system turn exactly as the release's chat template writes it for `tools=[...]`
// (ToolPrompt.systemContent). The
// model answers with `<tool_call>\n<function=web_search>\n<parameter=query>\n…\n</parameter>\n</function>\n</tool_call>`;
// ChatModel stops the round there, runs the search, appends the call turn and a `tool` turn and generates again from the
// engine's live state (only the result is prefilled). The no-key backend checks page content before showing a
// result. open_url can then rank the cached page for the question, within the read budget, without fetching it again.

/// DuckDuckGo's HTML endpoint needs no key. Brave Search API and Tavily use keys stored in the Keychain.
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

    static var enabled: Bool { d.object(forKey: enabledKey) as? Bool ?? true }
    static var backend: ToolBackend { ToolBackend(rawValue: d.string(forKey: backendKey) ?? "") ?? .duckduckgo }
    static var pageReads: Bool { d.object(forKey: pageReadsKey) as? Bool ?? true }

    /// Tool calls run per message; a further call gets an error result and the model answers with what it has. A failed
    /// open_url of a URL that was not among the last search's results (an invented one) does not count, once per message
    /// because the error lists the numbered results to retry.
    static let maxCalls = 3
    // Keep search previews short. Explicit reads have room for a passage and its context. The larger-memory
    // budgets start from the 3.5× prefill estimate for the 12 GB class.
    static let defaultResultTokens = 128

    static func evidenceBudget(reading: Bool, memory: UInt64) -> Int {
        let largerPhone = memory >= UInt64(11) << 30
        return reading ? (largerPhone ? 896 : 256) : (largerPhone ? 448 : defaultResultTokens)
    }
    static let defaultResults = 3
    /// Result pages a search reads (the top ones; the rest keep their snippets).
    static let pagesPerSearch = 2
    static let candidateLimit = 6
    static let retryCandidateLimit = 4
    /// Bytes of a page read at most (the rest is not downloaded).
    static let pageByteCap = 1 << 20
    static let timeout: TimeInterval = 10

    /// The second tool (declared when page reads are on), keys sorted as swift-jinja's `tojson` writes them (the same form
    /// as ToolPrompt.webSearchJSON). `result`: the [n] of a result of the last search (resolved to its stored URL, so the
    /// model never has to know one); `url` stays as the alternative. Neither is required.
    static let openURLJSON = #"{"function": {"description": "Read a search result using its numbered result parameter. Use url only for an exact URL supplied by the user.", "name": "open_url", "parameters": {"properties": {"result": {"description": "the [n] number", "type": "integer"}, "url": {"type": "string"}}, "type": "object"}}, "type": "function"}"#
    static var toolsJSON: [String] { pageReads ? [ToolPrompt.webSearchJSON, openURLJSON] : [ToolPrompt.webSearchJSON] }

    static let searchReadNote = "Call open_url with a result number before answering."
    static let followupReadNote = "Read or search again if needed. Otherwise, answer with citations."

    static func toolNote(_ meta: ToolMeta, canCallAgain: Bool) -> String? {
        guard pageReads, canCallAgain, meta.error == nil else { return nil }
        if meta.name == "web_search", !meta.results.isEmpty { return searchReadNote }
        if meta.name == "open_url" { return followupReadNote }
        return nil
    }

    static let systemAddition = "Use web_search to find sources for current or specific facts. Start with a short query. Cite supporting sources by title or site. Include a URL only if it was provided to you. Say what you could not verify. Answer in the user's language."

    static func dateNote(now: Date = Date(), timeZone: TimeZone = .autoupdatingCurrent) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.timeZone = timeZone
        formatter.dateFormat = "yyyy-MM-dd"
        return "Device date: " + formatter.string(from: now)
    }

    /// The day changes in the user turn, leaving the cached system prefix and prior turns stable.
    static func systemContent(_ system: String, tools: Bool) -> String {
        let s = system.trimmingCharacters(in: .whitespacesAndNewlines)
        var parts = s.isEmpty ? [] : [s]
        parts.append("Use the latest Device date in the conversation as today's local date.")
        if tools { parts.append(systemAddition) }
        let content = parts.joined(separator: "\n\n")
        return tools ? ToolPrompt.systemContent(system: content, toolsJSON: toolsJSON) : content
    }
}

/// API keys for the keyed backends: generic passwords in the Keychain (service <bundle id>.search, account =
/// the backend). Never logged, never written to UserDefaults or the conversation files.
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
    /// An in-memory page snapshot, which can expire between turns.
    var pageID: UUID? = nil

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
    /// Tokens of the result text as the model reads it (the engine's tokenizer; nil for the stub).
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
    static let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 PhoneStoreApp/1.0"

    static var session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = ToolSettings.timeout
        c.timeoutIntervalForResource = ToolSettings.timeout
        c.httpAdditionalHeaders = ["User-Agent": userAgent, "Accept-Language": "en-US,en;q=0.8"]
        return URLSession(configuration: c)
    }()

    private final class PageRedirects: NSObject, URLSessionTaskDelegate {
        func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
            let allowed = request.url.flatMap { Readable.safeURL($0.absoluteString) } != nil
            completionHandler(allowed ? request : nil)
        }
    }

    static var pageSession: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForResource = 5
        c.httpAdditionalHeaders = ["User-Agent": userAgent, "Accept-Language": "en-US,en;q=0.8"]
        return URLSession(configuration: c, delegate: PageRedirects(), delegateQueue: nil)
    }()

    private final class CachedPage {
        let page: Readable.Page
        let saved = Date()
        init(_ page: Readable.Page) { self.page = page }
    }

    private static let pages: NSCache<NSUUID, CachedPage> = {
        let cache = NSCache<NSUUID, CachedPage>()
        cache.countLimit = 24
        cache.totalCostLimit = 2 << 20
        return cache
    }()

    private struct Verified {
        let source: Passages.Source
        let page: Readable.Page
        let score: Double
    }

    private static func verify(_ results: [SearchResult], terms: [String: Double], requireStateNames: Bool,
                               tried: inout Set<String>, limit: Int, render: Bool, desired: Int) async -> [Verified] {
        guard desired > 0, !Task.isCancelled else { return [] }
        var candidates: [SearchResult] = []
        for result in results {
            guard Readable.worthReading(result.url), tried.insert(canonical(result.url)).inserted else { continue }
            candidates.append(result)
            if candidates.count == limit { break }
        }
        let fetched = await withTaskGroup(of: (Int, Readable.Page?).self) { group in
            for (i, result) in candidates.enumerated() {
                group.addTask { (i, try? await readPage(result.url)) }
            }
            var result: [Int: Readable.Page] = [:]
            for await (i, page) in group {
                if let page { result[i] = page }
                if Task.isCancelled { group.cancelAll() }
            }
            return result
        }
        guard !Task.isCancelled else { return [] }

        var verified: [Verified] = []
        var seen = Set<String>()
        func keep(_ page: Readable.Page, index: Int) {
            guard let score = Passages.evidence(page, terms: terms, requireStateNames: requireStateNames) else { return }
            let result = candidates[index]
            let url = page.url?.absoluteString ?? result.url
            let title = shortTitle(page.title.isEmpty ? (result.title.isEmpty ? url : result.title) : page.title)
            let ranked = Passages.rank(blocks: page.blocks, terms: terms, lead: false, title: title, passages: true, requireStateNames: requireStateNames)
                .map { Passages.Sentence(text: Passages.preview($0.text, terms: terms), score: $0.score, pos: $0.pos) }
            guard let lead = ranked.first, seen.insert(canonical(url)).inserted,
                  seen.insert(lead.text.lowercased()).inserted else { return }
            let id = UUID()
            let host = SearchResult(title: "", url: url, snippet: "").host
            let source = Passages.Source(title: title, host: host, url: url, sentences: ranked, read: true, maxKeep: 3, pageID: id)
            verified.append(Verified(source: source, page: page, score: score + 0.25 / Double(index + 1)))
        }
        for i in candidates.indices {
            if let page = fetched[i] { keep(page, index: i) }
        }
        if render, verified.count < desired, !Task.isCancelled,
           let i = candidates.indices.first(where: {
               guard let page = fetched[$0] else { return false }
               return page.renderable && Passages.evidence(page, terms: terms, requireStateNames: requireStateNames) == nil
           }), let page = fetched[i], let rendered = await Readable.render(page, terms: terms, requireStateNames: requireStateNames) {
            keep(rendered, index: i)
        }
        return verified
    }

    private static func canonical(_ raw: String) -> String {
        guard var parts = URLComponents(string: raw) else { return raw }
        parts.fragment = nil
        parts.host = parts.host?.lowercased()
        let items = parts.queryItems?.filter { !$0.name.hasPrefix("utm_") && !["fbclid", "gclid"].contains($0.name) }
        parts.queryItems = items?.isEmpty == false ? items : nil
        return parts.string ?? raw
    }

    private static func retryQuery(_ query: String, context: String, requireStateNames: Bool) -> String? {
        let existing = Set(Passages.words(query).map(Passages.norm))
        var added = Set<String>()
        let terms = Passages.words(context).filter { word in
            let normalized = Passages.norm(word)
            return word.count > 2 && !Passages.stop.contains(word) && !existing.contains(normalized) && added.insert(normalized).inserted
        }.prefix(4)
        if requireStateNames && !query.lowercased().contains("by state") { return query + " by state" }
        if !terms.isEmpty { return query + " " + terms.joined(separator: " ") }
        return existing.contains("official") ? nil : query + " official"
    }

    enum Failure: LocalizedError {
        case noKey(ToolBackend), http(Int), parse, empty, badURL, notHTML, noText, noEvidence
        var errorDescription: String? {
            switch self {
            case .noKey(let b): return "no \(b.short) API key"
            case .http(let c): return c == 202 || c == 429 ? "rate-limited (HTTP \(c)); try again later or use a key" : "HTTP \(c)"
            case .parse: return "unreadable response"
            case .empty: return "no results"
            case .badURL: return "not a public http(s) URL"
            case .notHTML: return "not a web page"
            case .noText: return "no readable text"
            case .noEvidence: return "no readable source contained relevant evidence"
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
        let budget = ToolSettings.evidenceBudget(reading: call.isPageRead, memory: ProcessInfo.processInfo.physicalMemory)
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
            if backend == .duckduckgo && ToolSettings.pageReads {
                let start = Date()
                let terms = Passages.questionTerms(q + " " + context)
                let requireStateNames = Passages.asksForUSStates(context)
                let limit = max(n, ToolSettings.pagesPerSearch)
                var attempted = Set<String>()
                var verified = await verify(rs, terms: terms, requireStateNames: requireStateNames, tried: &attempted, limit: ToolSettings.candidateLimit,
                                            render: true, desired: limit)
                if verified.count < min(2, n), !Task.isCancelled, let retry = retryQuery(q, context: context, requireStateNames: requireStateNames),
                   let more = try? await duckDuckGo(retry) {
                    verified += await verify(more, terms: terms, requireStateNames: requireStateNames, tried: &attempted,
                                             limit: ToolSettings.retryCandidateLimit, render: false, desired: limit - verified.count)
                }
                try Task.checkCancellation()
                var urls = Set<String>(), hosts = Set<String>()
                let best = verified.sorted { $0.score > $1.score }.filter {
                    urls.insert(canonical($0.source.url)).inserted
                }
                // Prefer independent sources. Keep a second page from a site only if no other site qualified.
                let independent = best.filter { hosts.insert($0.source.host).inserted }
                let duplicates = best.filter { item in independent.contains { $0.source.host == item.source.host && $0.source.url != item.source.url } }
                let selected = Array((independent + duplicates).prefix(n))
                var sources = selected.map { $0.source }
                let (text, kept, toks) = Passages.compose(&sources, answer: nil, budget: budget, count: count)
                guard !kept.isEmpty else { throw Failure.noEvidence }
                for result in kept {
                    guard let id = result.pageID, let match = selected.first(where: { $0.source.pageID == id }) else { continue }
                    let bytes = match.page.blocks.reduce(0) { $0 + $1.utf8.count }
                    pages.setObject(CachedPage(match.page), forKey: id as NSUUID, cost: bytes)
                }
                meta.results = kept
                meta.resultTokens = toks
                meta.pages = kept.count
                meta.pagesMs = Date().timeIntervalSince(start) * 1000
                meta.budget = budget
                meta.ms = Date().timeIntervalSince(t0) * 1000
                return (text, meta)
            }
            rs = Array(rs.filter { !$0.title.isEmpty && !$0.url.isEmpty }.prefix(n))
            guard !rs.isEmpty else { throw Failure.empty }
            // Tavily already supplies a passage from each page.
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
            let terms = Passages.questionTerms(context)
            let requireStateNames = Passages.asksForUSStates(context)
            let page: Readable.Page
            if let i = meta.resultIndex, let id = last[i - 1].pageID, let cached = pages.object(forKey: id as NSUUID),
               Date().timeIntervalSince(cached.saved) < 300 {
                page = cached.page
                meta.backend = "page cache"
            } else {
                let direct = try await readPage(u.absoluteString)
                if direct.renderable && Passages.evidence(direct, terms: terms, requireStateNames: requireStateNames) == nil,
                   let rendered = await Readable.render(direct, terms: terms, requireStateNames: requireStateNames) {
                    page = rendered
                    meta.backend = "rendered page"
                } else {
                    page = direct
                }
            }
            meta.pagesMs = Date().timeIntervalSince(t0) * 1000
            guard !page.blocks.isEmpty else { throw Failure.noText }
            let title = shortTitle(page.title.isEmpty ? (u.host ?? raw) : page.title)
            guard terms.isEmpty || Passages.evidence(page, terms: terms, requireStateNames: requireStateNames) != nil else { throw Failure.noEvidence }
            let ranked = Passages.rank(blocks: page.blocks, terms: terms, lead: false, title: title, passages: true, requireStateNames: requireStateNames)
            guard !ranked.isEmpty else { throw Failure.noEvidence }
            let url = page.url?.absoluteString ?? u.absoluteString
            var src = [Passages.Source(title: title, host: SearchResult(title: "", url: url, snippet: "").host,
                                       url: url, sentences: ranked, read: true, maxKeep: 12)]
            let (text, kept, toks) = Passages.compose(&src, answer: nil, budget: budget, count: count)
            guard !kept.isEmpty else { throw Failure.noEvidence }
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
    static func readPage(_ url: String) async throws -> Readable.Page {
        guard let u = Readable.safeURL(url) else { throw Failure.badURL }
        var req = URLRequest(url: u)
        req.timeoutInterval = 5
        req.setValue("text/html,application/xhtml+xml,text/plain,application/pdf;q=0.8", forHTTPHeaderField: "Accept")
        let (bytes, resp) = try await pageSession.bytes(for: req)
        guard let finalURL = resp.url, Readable.safeURL(finalURL.absoluteString) != nil else { throw Failure.badURL }
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw Failure.http(code) }
        let type = ((resp as? HTTPURLResponse)?.value(forHTTPHeaderField: "Content-Type") ?? "text/html").lowercased()
        let pdf = type.contains("application/pdf") || (type.contains("application/octet-stream") && finalURL.pathExtension.lowercased() == "pdf")
        let plainText = type.contains("text/plain")
        guard pdf || plainText || type.contains("html") else { throw Failure.notHTML }
        let byteCap = pdf ? 4 << 20 : ToolSettings.pageByteCap
        var data = Data()
        data.reserveCapacity(256 << 10)
        var buf = [UInt8]()
        buf.reserveCapacity(64 << 10)
        for try await b in bytes {
            buf.append(b)
            if buf.count == 64 << 10 { data.append(contentsOf: buf); buf.removeAll(keepingCapacity: true) }
            if data.count + buf.count >= byteCap { break }
        }
        data.append(contentsOf: buf)
        try Task.checkCancellation()
        if pdf {
            guard let document = PDFDocument(data: data), !document.isLocked else { throw Failure.noText }
            let blocks = (0..<min(document.pageCount, 20)).compactMap { document.page(at: $0)?.string }
                .flatMap { Readable.plainBlocks(String($0.prefix(12_000))) }
            return Readable.Page(title: document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String ?? "",
                                 blocks: Array(blocks.prefix(800)), url: finalURL)
        }
        let html = String(data: data, encoding: .utf8) ?? String(decoding: data, as: UTF8.self)
        if plainText { return Readable.Page(title: "", blocks: Readable.plainBlocks(html), url: finalURL) }
        return Readable.extractPage(html, url: finalURL)
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

/// Extracts readable blocks from the page DOM while keeping headings and table labels with their text.
enum Readable {
    /// Candidate URLs with a supported content type and a public-looking host.
    static func worthReading(_ url: String) -> Bool {
        guard let u = safeURL(url), let h = u.host?.lowercased() else { return false }
        let skip = ["youtube.com", "youtu.be", "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com",
                    "pinterest.com", "linkedin.com", "reddit.com"]
        if skip.contains(where: { h == $0 || h.hasSuffix("." + $0) }) { return false }
        let p = u.path.lowercased()
        return ![".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".zip", ".doc", ".docx", ".xls", ".ppt"].contains(where: p.hasSuffix)
    }

    /// An http(s) URL to a public host (no localhost, .local, private / loopback / link-local IPv4, IPv6 literals); a bare
    /// host gets https://.
    static func safeURL(_ s: String) -> URL? {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.contains("://") { t = "https://" + t }
        guard let u = URL(string: t), let sch = u.scheme?.lowercased(), sch == "http" || sch == "https",
              let h = u.host?.lowercased(), h.contains("."), !h.contains(":"), u.user == nil, u.password == nil else { return nil }
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

    static let spaces = try! NSRegularExpression(pattern: "[ \\t\\r\\u00a0\\u200b]+")
    /// Wikipedia-style footnote markers ("[6]", "[citation needed]") and the spaces tag removal leaves before punctuation.
    static let footnote = try! NSRegularExpression(pattern: "\\[(\\d{1,3}|[a-z]|citation needed|note \\d+)\\]")
    static let spaceBeforePunct = try! NSRegularExpression(pattern: "\\s+([,.;:!?)\\]])")
    static let spaceAfterOpen = try! NSRegularExpression(pattern: "([(\\[])\\s+")
    static let boiler = ["cookie", "subscribe", "sign in", "sign up", "log in", "newsletter", "all rights reserved",
                         "privacy policy", "terms of use", "terms of service", "javascript", "advertisement",
                         "skip to content", "your browser", "accept all", "download the app", "follow us",
                         "click for", "click here", "get premium", "create your account", "you are here"]

    struct Page: Sendable {
        let title: String
        let blocks: [String]
        var url: URL? = nil
        var renderable = false
    }

    static func extract(_ html: String) -> (title: String, blocks: [String]) {
        let page = extractPage(html)
        return (page.title, page.blocks)
    }

    static func extractPage(_ html: String, url: URL? = nil) -> Page {
        let options = HTML_PARSE_RECOVER.rawValue | HTML_PARSE_NONET.rawValue
            | HTML_PARSE_NOERROR.rawValue | HTML_PARSE_NOWARNING.rawValue
        let doc = html.withCString { htmlReadMemory($0, Int32(html.utf8.count), nil, "UTF-8", Int32(options)) }
        guard let doc else { return Page(title: "", blocks: [], url: url, renderable: true) }
        defer { xmlFreeDoc(doc) }
        var title = "", heading = "", blocks: [String] = [], seen = Set<String>()
        var excludedSection = false
        var main: xmlNodePtr?
        let excluded: Set<String> = ["head", "style", "noscript", "svg", "template", "iframe", "nav", "footer", "aside", "form",
                                      "button", "select", "dialog", "menu"]
        let content: Set<String> = ["p", "li", "dt", "dd", "blockquote", "pre", "figcaption"]
        let headings: Set<String> = ["h1", "h2", "h3", "h4", "h5", "h6"]
        func tag(_ n: xmlNodePtr) -> String {
            n.pointee.name.map { String(cString: $0).lowercased() } ?? ""
        }
        func text(_ n: xmlNodePtr) -> String {
            guard let value = xmlNodeGetContent(n) else { return "" }
            defer { xmlFree(value) }
            return String(cString: value).split(whereSeparator: \.isWhitespace).joined(separator: " ")
        }
        func attribute(_ n: xmlNodePtr, _ name: String) -> String? {
            guard let value = name.withCString({ xmlGetProp(n, UnsafeRawPointer($0).assumingMemoryBound(to: xmlChar.self)) }) else { return nil }
            defer { xmlFree(value) }
            return String(cString: value)
        }
        func children(_ n: xmlNodePtr, _ visit: (xmlNodePtr) -> Void) {
            var child = n.pointee.children
            while let node = child {
                if node.pointee.type == XML_ELEMENT_NODE { visit(node) }
                child = node.pointee.next
            }
        }
        func add(_ value: String, section: String = "") {
            let clean = replace(footnote, replace(responsiveBreak, value, " "), "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !clean.isEmpty, blocks.count < 800 else { return }
            let words = clean.split(whereSeparator: \.isWhitespace)
            guard words.count >= 3,
                  !clean.lowercased().hasPrefix("explore the "),
                  !(words.count < 16 && boiler.contains(where: clean.lowercased().hasPrefix)),
                  seen.insert(clean).inserted else { return }
            let prefix = section.isEmpty ? "" : String(section.split(separator: " ").prefix(8).joined(separator: " ")) + " — "
            // Keep long paragraphs searchable beyond their opening lines.
            let limit = min(words.count, 900)
            var start = 0
            while start < limit && blocks.count < 800 {
                let end = min(start + 60, limit)
                blocks.append(prefix + words[start..<end].joined(separator: " "))
                if end == limit { break }
                start += 44
            }
        }
        func scan(_ node: xmlNodePtr, depth: Int) {
            guard depth < 80 else { return }
            let name = tag(node)
            if name == "title", title.isEmpty { title = text(node) }
            if name == "main", main == nil { main = node }
            children(node) { scan($0, depth: depth + 1) }
        }
        func table(_ node: xmlNodePtr, depth: Int) {
            var labels: [String] = [], caption = heading
            func row(_ child: xmlNodePtr, level: Int) {
                guard level < 8, blocks.count < 800 else { return }
                switch tag(child) {
                case "caption": caption = text(child)
                case "tr":
                    var cells: [String] = []
                    var headers = 0
                    children(child) {
                        let name = tag($0)
                        if name == "th" { headers += 1 }
                        if name == "th" || name == "td" { cells.append(text($0)) }
                    }
                    if !cells.isEmpty && headers == cells.count { labels = cells }
                    else if !cells.isEmpty {
                        let values = cells.enumerated().map { i, cell in
                            labels.indices.contains(i) && !labels[i].isEmpty ? "\(labels[i]): \(cell)" : cell
                        }
                        add(values.joined(separator: " | "), section: caption)
                    }
                case "thead", "tbody", "tfoot", "table":
                    children(child) { row($0, level: level + 1) }
                default: break
                }
            }
            row(node, level: depth)
        }
        func walk(_ node: xmlNodePtr, depth: Int) {
            guard depth < 80, blocks.count < 800 else { return }
            let name = tag(node)
            if excluded.contains(name) || attribute(node, "hidden") != nil || attribute(node, "aria-hidden") == "true" || attribute(node, "role") == "navigation" { return }
            if name == "script" { return }
            if headings.contains(name) {
                heading = String(text(node).prefix(100))
                excludedSection = ["references", "external links", "further reading", "see also", "notes",
                                   "media", "press contacts", "related articles", "site navigation", "contents", "table of contents"].contains(heading.lowercased())
                return
            }
            if name == "table", !excludedSection {
                table(node, depth: 0)
            } else if content.contains(name) {
                if !excludedSection {
                    // textContent does not insert spaces at inline element boundaries such as <br>.
                    add(text(node), section: heading)
                }
            } else {
                var hasElements = false
                children(node) { _ in hasElements = true }
                if !hasElements && ["div", "article", "main", "section", "span"].contains(name) {
                    let value = text(node)
                    if !excludedSection && (name != "span" || value.split(separator: " ").count >= 8) {
                        add(value, section: heading)
                    }
                } else {
                    children(node) { walk($0, depth: depth + 1) }
                }
            }
        }
        guard let root = xmlDocGetRootElement(doc) else { return Page(title: "", blocks: [], url: url, renderable: true) }
        scan(root, depth: 0)
        walk(main ?? root, depth: 0)
        if main != nil && blocks.isEmpty {
            heading = ""
            excludedSection = false
            walk(root, depth: 0)
        }

        // Only accept an actual article body from structured metadata, not a title or description.
        for payload in groups(jsonLD, html, 1).prefix(12) {
            guard let value = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) else { continue }
            var queue: [Any] = [value]
            for _ in 0..<64 {
                guard let next = queue.popLast() else { break }
                if let list = next as? [Any] { queue.append(contentsOf: list.prefix(32)) }
                if let fields = next as? [String: Any] {
                    if let body = fields["articleBody"] as? String {
                        for paragraph in body.components(separatedBy: .newlines).prefix(100) { add(WebSearch.plain(paragraph)) }
                    }
                    if let graph = fields["@graph"] { queue.append(graph) }
                }
            }
        }
        return Page(title: title, blocks: blocks, url: url, renderable: true)
    }

    private static let responsiveBreak = re(#":br\([a-z]+\):"#)
    private static let jsonLD = re(#"<script\b[^>]*type\s*=\s*['"]application/ld\+json['"][^>]*>(.*?)</script\s*>"#)

    @MainActor
    private final class BrowserNavigation: NSObject, WKNavigationDelegate {
        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            let allowed = action.targetFrame?.isMainFrame == true
                && action.request.url.flatMap { Readable.safeURL($0.absoluteString) } != nil
            decisionHandler(allowed ? .allow : .cancel)
        }
    }

    @MainActor
    static func render(_ page: Page, terms: [String: Double], requireStateNames: Bool) async -> Page? {
        guard let url = page.url, safeURL(url.absoluteString) != nil, !Task.isCancelled else { return nil }
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        let browser = WKWebView(frame: .zero, configuration: config)
        let navigation = BrowserNavigation()
        browser.navigationDelegate = navigation
        defer { browser.stopLoading(); browser.navigationDelegate = nil }
        browser.load(URLRequest(url: url))
        for _ in 0..<6 {
            do { try await Task.sleep(nanoseconds: 500_000_000) } catch { return nil }
            guard let current = browser.url, safeURL(current.absoluteString) != nil else { continue }
            let js = "document.documentElement?.outerHTML.slice(0, 1048576) ?? ''"
            guard let html = try? await browser.evaluateJavaScript(js) as? String, !html.isEmpty else { continue }
            let result = await Task.detached { extractPage(html, url: current) }.value
            if Passages.evidence(result, terms: terms, requireStateNames: requireStateNames) != nil { return result }
        }
        return nil
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
        var pageID: UUID? = nil
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
        if w == "democratic" { return "democrat" }
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

    static func asksForUSStates(_ question: String) -> Bool {
        let lower = question.lowercased()
        return (lower.contains("which states") || lower.contains("list the states") || lower.contains("what states"))
            && ["senate", "united states", "u.s.", "democrat", "republican"].contains(where: lower.contains)
    }

    private static let usStates = try! NSRegularExpression(pattern: #"\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming)\b"#)

    private static func hasUSState(_ text: String) -> Bool {
        usStates.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    static func questionTerms(_ text: String) -> [String: Double] {
        let range = NSRange(text.startIndex..., in: text)
        let clean = urls.stringByReplacingMatches(in: text, range: range, withTemplate: "")
        return terms(clean).filter { !readWords.contains($0.key) }
    }

    private static let urls = try! NSRegularExpression(pattern: #"https?://[^\s]+"#, options: .caseInsensitive)
    private static let readWords: Set<String> = ["read", "open", "page", "website", "link", "detail", "give", "more", "summarize"]

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
    static func rank(blocks: [String], terms: [String: Double], lead snippet: Bool, title: String = "", passages: Bool = false, requireStateNames: Bool = false) -> [Sentence] {
        var out: [Sentence] = []
        var seen = Set<String>()
        let titleWords = Set(words(title).map(norm))
        var pos = 0
        for b in blocks {
            for s0 in (passages ? [b] : sentences(b)) {
                let s = cut(s0, words: passages ? 70 : 30)
                let body = passages ? passageBody(s) : s
                if requireStateNames && !hasUSState(body) { continue }
                let ws = words(body)
                guard ws.count >= 3, seen.insert(s).inserted else { continue }
                let set = Set(ws.map(norm))
                var sc = 0.0
                let low = body.lowercased()
                for (t, w) in terms where t.contains("°") ? low.contains(t) : set.contains(t) { sc += w }
                if passages && !terms.isEmpty && sc == 0 { continue }
                if body.contains(where: \.isNumber) { sc += 0.3 }
                if ws.count < 5 { sc -= 0.3 }
                if passages && ws.count >= 12 { sc += 0.45 }
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

    static func preview(_ passage: String, terms: [String: Double]) -> String {
        let body = passageBody(passage)
        let chunks = sentences(body)
        guard chunks.count > 1 else { return cut(passage, words: 46) }
        func score(_ sentence: String) -> Double {
            Set(words(sentence).map(norm)).reduce(0) { $0 + (terms[$1] ?? 0) }
        }
        let best = chunks.indices.max { score(chunks[$0]) < score(chunks[$1]) } ?? 0
        let prefix = String(passage.prefix(passage.count - body.count))
        if best > 0 {
            let withPrevious = prefix + chunks[(best - 1)...best].joined(separator: " ")
            if withPrevious.split(separator: " ").count <= 46 { return withPrevious }
        }
        let end = min(chunks.count, best + 2)
        return cut(prefix + chunks[best..<end].joined(separator: " "), words: 46)
    }

    private static func passageBody(_ block: String) -> String {
        guard let boundary = block.range(of: " — "), block.distance(from: block.startIndex, to: boundary.lowerBound) <= 100 else {
            return block
        }
        return String(block[boundary.upperBound...])
    }

    /// Require query words in a real body passage. A matching title alone cannot qualify a result.
    static func evidence(_ page: Readable.Page, terms: [String: Double], requireStateNames: Bool = false) -> Double? {
        let title = Set(words(page.title).map(norm))
        let exact = terms.filter { $0.value == 1 }
        var best = 0.0
        for block in page.blocks {
            let body = passageBody(block)
            if requireStateNames && !hasUSState(body) { continue }
            let wordsInBody = words(body)
            guard wordsInBody.count >= 5 else { continue }
            let set = Set(wordsInBody.map(norm))
            guard set.subtracting(title).count >= 3 else { continue }
            if terms.isEmpty { return 1 }
            let hits = terms.reduce(0.0) { sum, term in
                sum + ((term.key.contains("°") ? body.lowercased().contains(term.key) : set.contains(term.key)) ? term.value : 0)
            }
            let titleHits = exact.keys.filter { title.contains($0) }.count
            guard hits >= 1, hits + Double(titleHits) >= Double(min(2, exact.count)) else { continue }
            best = max(best, hits + 0.25 * Double(titleHits))
        }
        return best > 0 ? best : nil
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
        let text = lines.joined(separator: "\n")
        let range = NSRange(text.startIndex..., in: text)
        return controlTags.stringByReplacingMatches(in: text, range: range, withTemplate: "[markup]")
    }

    private static let controlTags = try! NSRegularExpression(
        pattern: #"<\|[^<>\s]{1,40}\|>|</?(?:tool_call|tool_response|think)(?:\s[^<>]*)?>|</?(?:parameter|function)(?:=[^<>]*)?>"#,
        options: .caseInsensitive
    )

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
                let header = src[i].kept.isEmpty ? headerCost(src[i]) : 0
                let c = est(s.text + " ") + header
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
        if let toks, toks > budget { return ("", [], 0) }
        let kept = src.filter { !$0.kept.isEmpty }.map {
            SearchResult(title: $0.title, url: $0.url, snippet: $0.kept.sorted { $0.pos < $1.pos }.map(\.text).joined(separator: " "), read: $0.read, pageID: $0.pageID)
        }
        return (text, kept, toks ?? est(text))
    }
}

// MARK: search pre-check

/// A deterministic check on the user's message (docs/TOOLS.md "Search decision"): a message about live or changing facts
/// (weather, news, prices, scores, office holders, "right now" / today / latest ...) gets a one-line hint appended in the
/// engine's copy of the user turn, "(Use web_search.)", so the model calls the tool instead of answering from memory. The
/// displayed message and the conversation file are unchanged; the hint is a pure function of the message text, so every
/// later render of the same history gives the same tokens (the live prefix and the prefix cache still match).
/// `-searchHint 0|1` overrides the default.
enum SearchHint {
    static let hint = "(Use web_search.)"
    static let retryInstruction = "Call web_search now. Skip the preamble."
    private static let searchPreamble = try! NSRegularExpression(
        pattern: #"^(?:i[’']ll|i will|let me)\s+(?:quickly\s+)?(?:check|search|look up|look for|verify)\b[^.!?\n]*[.!?]?$"#,
        options: [.caseInsensitive]
    )

    /// A reply that ends after announcing a search has no answer or tool call.
    static func isSearchPreamble(_ answer: String) -> Bool {
        let text = answer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.count <= 120 else { return false }
        return searchPreamble.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }
    static let localDateQuestion = try! NSRegularExpression(
        pattern: #"^\s*(what (is|day is)|what'?s) (the date( today)?|today'?s date|today)\s*[?.!]?\s*$"#,
        options: [.caseInsensitive]
    )

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
    static func engineContent(_ text: String, hasDeviceDate: Bool = false) -> String {
        guard applies(text), !text.contains(hint) else { return text }
        let range = NSRange(text.startIndex..., in: text)
        if hasDeviceDate, localDateQuestion.firstMatch(in: text, range: range) != nil { return text }
        return text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n" + hint
    }
}
