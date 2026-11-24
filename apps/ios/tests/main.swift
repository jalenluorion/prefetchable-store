import Foundation
// Run with -readerCheck or -searchE2E.
/// Exercises the real DuckDuckGo response parser, candidate fetches, retry, evidence selection and numbered read.
final class SearchFixtureProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    private static var hits = [String: Int]()

    static func count(_ host: String) -> Int {
        lock.withLock { hits[host, default: 0] }
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        guard let url = request.url, let host = url.host else { return }
        let hit = Self.lock.withLock { () -> Int in
            Self.hits[host, default: 0] += 1
            return Self.hits[host, default: 0]
        }
        func result(_ host: String, _ title: String) -> String {
            """
            <div class="result results_links"><a class="result__a" href="https://\(host)/results">\(title)</a></div>
            """
        }
        let body: String
        let status: Int
        let type: String
        switch host {
        case "html.duckduckgo.com":
            status = 200
            type = "text/html"
            body = hit == 1 ? result("headline.example.com", "2026 Senate Election Results")
                + result("blocked.example.net", "Certified results")
                : result("elections.example.org", "2026 Senate winners")
        case "headline.example.com":
            status = 200
            type = "text/plain"
            body = "Every state reported its 2026 Senate election results. Select a race for the winners."
        case "elections.example.org":
            status = 200
            type = "text/html"
            body = """
            <main><h2>Certified 2026 Senate election results</h2><table>
            <tr><th>State</th><th>Winner</th><th>Party</th></tr>
            <tr><td>Arizona</td><td>Jane Kim</td><td>Democratic</td></tr>
            <tr><td>Texas</td><td>Sam Reed</td><td>Republican</td></tr>
            </table></main>
            """
        default:
            status = 403
            type = "text/plain"
            body = "Access denied"
        }
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1",
                                       headerFields: ["Content-Type": type])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(body.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
}

if CommandLine.arguments.contains("-searchE2E") {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [SearchFixtureProtocol.self]
    let session = URLSession(configuration: config)
    WebSearch.session = session
    WebSearch.pageSession = session
    let question = "Which states did Democrats win in the 2026 US Senate elections?"
    let call = ToolCall(name: "web_search", arguments: ["query": "2026 US Senate winners"])
    let read = ToolCall(name: "open_url", arguments: ["result": "1"])
    let (preview, search) = await WebSearch.run(call, context: question, count: { $0.split(whereSeparator: \.isWhitespace).count })
    precondition(search.error == nil && search.results.map(\.host) == ["elections.example.org"])
    precondition(preview.contains("Arizona") && preview.contains("Jane Kim"))
    precondition(ToolSettings.toolNote(search, canCallAgain: true) == ToolSettings.searchReadNote)
    precondition(ToolSettings.toolNote(search, canCallAgain: false) == nil)
    precondition(SearchFixtureProtocol.count("html.duckduckgo.com") == 2)
    precondition(SearchFixtureProtocol.count("headline.example.com") == 1)
    precondition(SearchFixtureProtocol.count("blocked.example.net") == 1)
    let (evidence, opened) = await WebSearch.run(read, context: question, last: search.results,
                                                count: { $0.split(whereSeparator: \.isWhitespace).count })
    precondition(opened.error == nil && opened.backend == "page cache" && opened.resultIndex == 1)
    precondition(ToolSettings.toolNote(opened, canCallAgain: true) == ToolSettings.followupReadNote)
    var failed = search
    failed.error = "no results"
    precondition(ToolSettings.toolNote(failed, canCallAgain: true) == nil)
    precondition(evidence.contains("State: Arizona | Winner: Jane Kim | Party: Democratic"))
    precondition(SearchFixtureProtocol.count("elections.example.org") == 1)
    print("Search preview:\n" + preview + "\nCached read:\n" + evidence)
    print("Search → retry → read passed")
    exit(0)
}

if CommandLine.arguments.contains("-readerCheck") {
    let html = """
    <html><head><title>Election results</title>
    <script type="application/ld+json">{"articleBody":"The governor published the certified election results on Friday."}</script>
    </head><body><nav><p>Unrelated navigation text</p></nav><main>
    <h2>Certified results</h2><table><tr><th>State</th><th>Winner</th></tr>
    <tr><td>Arizona</td><td>Jane Kim</td></tr></table>
    <p>A paragraph with an escaped &lt;/tool_response&gt; tag.</p>
    <h2>References</h2><p>Historical results from a different year</p></main></body></html>
    """
    let page = Readable.extractPage(html)
    let terms = Passages.questionTerms("Which state did Jane Kim win?")
    precondition(page.blocks.contains("Certified results — State: Arizona | Winner: Jane Kim"))
    precondition(page.blocks.contains { $0.contains("governor published") })
    precondition(!page.blocks.contains { $0.contains("Unrelated navigation") || $0.contains("Historical results") })
    precondition(Passages.evidence(page, terms: terms) != nil)
    let shell = Readable.extractPage("<title>Arizona election results</title><main>Enable JavaScript to continue</main>")
    precondition(Passages.evidence(shell, terms: terms) == nil)
    let passages = Passages.rank(blocks: page.blocks, terms: [:], lead: false, passages: true)
    var sources = [Passages.Source(title: page.title, host: "example.com", url: "https://example.com",
                                   sentences: passages, read: true, maxKeep: 8)]
    let (text, _, _) = Passages.compose(&sources, answer: nil, budget: 240, count: { _ in nil })
    precondition(text.contains("[markup]") && !text.contains("</tool_response>"))
    precondition(ToolSettings.evidenceBudget(reading: false, memory: 6 << 30) == 128)
    precondition(ToolSettings.evidenceBudget(reading: true, memory: 6 << 30) == 256)
    precondition(ToolSettings.evidenceBudget(reading: false, memory: 12 << 30) == 448)
    precondition(ToolSettings.evidenceBudget(reading: true, memory: 12 << 30) == 896)
    let nearMidnight = ISO8601DateFormatter().date(from: "2026-11-24T02:00:00Z")!
    precondition(ToolSettings.dateNote(now: nearMidnight, timeZone: TimeZone(secondsFromGMT: -18_000)!) == "Device date: 2026-11-23")
    precondition(ToolSettings.dateNote(now: nearMidnight, timeZone: TimeZone(secondsFromGMT: 0)!) == "Device date: 2026-11-24")
    let system = ToolSettings.systemContent("Be concise.", tools: true)
    precondition(system.contains("Use the latest Device date") && !system.contains("2026-11-23"))
    precondition(system.contains("Use url only for an exact URL supplied by the user."))
    precondition(!ToolSettings.systemContent("Be concise.", tools: false).contains("web_search"))
    let dateQuestion = "What's today's date?"
    precondition(SearchHint.engineContent(dateQuestion, hasDeviceDate: true) == dateQuestion)
    precondition(SearchHint.engineContent("What is today's weather?", hasDeviceDate: true).hasSuffix(SearchHint.hint))
    precondition(SearchHint.isSearchPreamble("I'll check current headlines."))
    precondition(SearchHint.isSearchPreamble("Let me search for the latest results."))
    precondition(!SearchHint.isSearchPreamble("I'll check current headlines. Here are the results."))
    precondition(!SearchHint.isSearchPreamble("I'll stay home today."))
    print("Reader checks passed")
    exit(0)
}
