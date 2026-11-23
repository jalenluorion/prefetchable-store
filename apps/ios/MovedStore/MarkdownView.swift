import SwiftUI
import UIKit

/// Block-level Markdown for streamed replies: headings, paragraphs, bullet and numbered lists (nested by indent), block
/// quotes, fenced code blocks (monospaced, scrolls sideways, copy button), horizontal rules and pipe tables. Inline syntax
/// (bold, italic, `code`, links) goes through Foundation's inline Markdown parser. An unclosed fence while a reply is still
/// streaming renders as code so far. Re-parsed per update: replies arrive at ~3 tokens per second, so this is cheap.
struct MarkdownView: View {
    let text: String

    var body: some View {
        let blocks = MarkdownParser.cached(text)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, b in
                MarkdownBlockView(block: b).equatable()  // an unchanged block (all but the streamed tail) is not rebuilt
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct MDListItem: Equatable {
    var indent: Int
    var marker: String   // "•" or "3."
    var text: String
}

enum MDBlock: Equatable {
    case heading(Int, String)
    case paragraph(String)
    case code(String, String)    // language, text
    case list([MDListItem])
    case quote(String)
    case rule
    case table([[String]])
}

enum MarkdownParser {
    /// The parsed blocks of recent texts (every message of the open chat re-rendered on each update of the reply; only
    /// the streamed one changes). Main thread only (SwiftUI bodies).
    nonisolated(unsafe) private static var cache: [String: [MDBlock]] = [:]
    nonisolated(unsafe) private static var order: [String] = []
    static func cached(_ s: String) -> [MDBlock] {
        if let b = cache[s] { return b }
        let b = parse(s)
        cache[s] = b
        order.append(s)
        if order.count > 64 { cache[order.removeFirst()] = nil }
        return b
    }

    static func parse(_ s: String) -> [MDBlock] {
        var out: [MDBlock] = []
        var para: [String] = []
        var list: [MDListItem] = []
        var quote: [String] = []
        var table: [[String]] = []
        var code: [String]? = nil
        var lang = ""

        func flushPara() { if !para.isEmpty { out.append(.paragraph(para.joined(separator: "\n"))); para = [] } }
        func flushList() { if !list.isEmpty { out.append(.list(list)); list = [] } }
        func flushQuote() { if !quote.isEmpty { out.append(.quote(quote.joined(separator: "\n"))); quote = [] } }
        func flushTable() { if !table.isEmpty { out.append(.table(table)); table = [] } }
        func flushAll() { flushPara(); flushList(); flushQuote(); flushTable() }

        for line in s.components(separatedBy: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            if code != nil {
                if t.hasPrefix("```") {
                    out.append(.code(lang, code!.joined(separator: "\n")))
                    code = nil
                } else {
                    code!.append(line)
                }
                continue
            }
            if t.hasPrefix("```") {
                flushAll()
                code = []
                lang = String(t.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                continue
            }
            if t.isEmpty { flushPara(); flushQuote(); flushTable(); continue }
            if let h = heading(t) { flushAll(); out.append(.heading(h.0, h.1)); continue }
            if isRule(t) { flushAll(); out.append(.rule); continue }
            if t.hasPrefix("|") {
                flushPara(); flushList(); flushQuote()
                if !isTableSeparator(t) { table.append(cells(t)) }
                continue
            }
            if t.hasPrefix(">") {
                flushPara(); flushList(); flushTable()
                quote.append(String(t.dropFirst()).trimmingCharacters(in: .whitespaces))
                continue
            }
            if let item = listItem(line) {
                flushPara(); flushQuote(); flushTable()
                list.append(item)
                continue
            }
            if !list.isEmpty, line.first?.isWhitespace == true {  // a wrapped list item
                list[list.count - 1].text += " " + t
                continue
            }
            flushList(); flushQuote(); flushTable()
            para.append(t)
        }
        if let c = code { out.append(.code(lang, c.joined(separator: "\n"))) }
        flushAll()
        return out
    }

    static func heading(_ t: String) -> (Int, String)? {
        let n = t.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(n), t.count > n, t[t.index(t.startIndex, offsetBy: n)] == " " else { return nil }
        return (n, String(t.dropFirst(n + 1)).trimmingCharacters(in: CharacterSet(charactersIn: " #")))
    }

    static func isRule(_ t: String) -> Bool {
        let c = t.replacingOccurrences(of: " ", with: "")
        guard c.count >= 3, let f = c.first, "-*_".contains(f) else { return false }
        return c.allSatisfy { $0 == f }
    }

    static func isTableSeparator(_ t: String) -> Bool { t.allSatisfy { "|-: ".contains($0) } && t.contains("-") }

    static func cells(_ t: String) -> [String] {
        var parts = t.split(separator: "|", omittingEmptySubsequences: false).map { $0.trimmingCharacters(in: .whitespaces) }
        if parts.first == "" { parts.removeFirst() }
        if parts.last == "" { parts.removeLast() }
        return parts
    }

    static func listItem(_ line: String) -> MDListItem? {
        var spaces = 0
        for ch in line {
            if ch == " " { spaces += 1 } else if ch == "\t" { spaces += 4 } else { break }
        }
        let t = line.trimmingCharacters(in: .whitespaces)
        guard let f = t.first else { return nil }
        if "-*+".contains(f), t.dropFirst().first == " " {
            return MDListItem(indent: spaces / 2, marker: "•", text: String(t.dropFirst(2)))
        }
        let digits = t.prefix(while: { $0.isNumber })
        if (1...3).contains(digits.count) {
            let rest = t.dropFirst(digits.count)
            if let d = rest.first, d == "." || d == ")", rest.dropFirst().first == " " {
                return MDListItem(indent: spaces / 2, marker: digits + ".", text: String(rest.dropFirst(2)))
            }
        }
        return nil
    }

    /// Bold, italic, inline code and links; whitespace and line breaks kept. Plain text if it does not parse.
    static func inline(_ s: String) -> AttributedString {
        let o = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace,
                                                        failurePolicy: .returnPartiallyParsedIfPossible)
        return (try? AttributedString(markdown: s, options: o)) ?? AttributedString(s)
    }
}

struct MarkdownBlockView: View, Equatable {
    let block: MDBlock

    var body: some View {
        switch block {
        case .heading(let level, let text):
            Text(MarkdownParser.inline(text))
                .font(level == 1 ? .title2.bold() : level == 2 ? .title3.bold() : level == 3 ? .headline : .subheadline.bold())
                .padding(.top, 4)
        case .paragraph(let text):
            Text(MarkdownParser.inline(text))
        case .code(let lang, let text):
            CodeBlockView(language: lang, code: text)
        case .list(let items):
            VStack(alignment: .leading, spacing: 5) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, it in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(it.marker).monospacedDigit().foregroundStyle(.secondary)
                            .frame(minWidth: 16, alignment: .trailing)
                        Text(MarkdownParser.inline(it.text))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.leading, CGFloat(min(it.indent, 4)) * 16)
                }
            }
        case .quote(let text):
            HStack(alignment: .top, spacing: 8) {
                RoundedRectangle(cornerRadius: 1.5).fill(.quaternary).frame(width: 3)
                Text(MarkdownParser.inline(text)).foregroundStyle(.secondary)
            }
            .fixedSize(horizontal: false, vertical: true)
        case .rule:
            Divider()
        case .table(let rows):
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { i, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                Text(MarkdownParser.inline(cell)).font(.subheadline).fontWeight(i == 0 ? .semibold : .regular)
                            }
                        }
                        if i == 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                    }
                }
                .padding(10)
            }
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}

struct CodeBlockView: View {
    let language: String
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button {
                    UIPasteboard.general.string = code
                    copied = true
                    Task { try? await Task.sleep(nanoseconds: 1_500_000_000); copied = false }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc").font(.caption)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 12).padding(.top, 8).padding(.bottom, 4)
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(.horizontal, 12).padding(.bottom, 10)
            }
        }
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
    }
}
