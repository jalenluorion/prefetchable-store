import SwiftUI

/// Saved conversations, newest first: title (from the first message, or renamed), date, preview, token count.
/// Swipe left to delete, right to rename; long-press for both.
struct ConversationListView: View {
    @EnvironmentObject var m: ChatModel
    @Binding var path: [Route]
    @State private var search = ""
    @State private var showSettings = false
    @State private var renameID: UUID?
    @State private var renameText = ""
    @State private var deleteID: UUID?

    var filtered: [Conversation] {
        let q = search.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return m.conversations }
        return m.conversations.filter { c in
            c.displayTitle.localizedCaseInsensitiveContains(q) || c.messages.contains { $0.content.localizedCaseInsensitiveContains(q) }
        }
    }

    var body: some View {
        List {
            ForEach(filtered) { c in
                Button {
                    m.open(c.id)
                    path = [.chat]
                } label: {
                    ConversationRow(c: c, replying: m.generatingConvID == c.id)
                }
                .swipeActions(edge: .trailing) {
                    Button(role: .destructive) { deleteID = c.id } label: { Label("Delete", systemImage: "trash") }
                }
                .swipeActions(edge: .leading) {
                    Button { renameText = c.title; renameID = c.id } label: { Label("Rename", systemImage: "pencil") }
                        .tint(.indigo)
                }
                .contextMenu {
                    Button { renameText = c.title; renameID = c.id } label: { Label("Rename", systemImage: "pencil") }
                    Button(role: .destructive) { deleteID = c.id } label: { Label("Delete", systemImage: "trash") }
                }
            }
        }
        .listStyle(.plain)
        .overlay {
            if m.conversations.isEmpty {
                ContentUnavailableView("No chats yet", systemImage: "bubble.left.and.bubble.right",
                                       description: Text("Tap the compose button to start one."))
            } else if filtered.isEmpty {
                ContentUnavailableView.search(text: search)
            }
        }
        .searchable(text: $search, prompt: "Search chats")
        .navigationTitle("Chats")
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button { showSettings = true } label: { Image(systemName: "gearshape") }
                    .accessibilityLabel("Settings")
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    m.newConversation()
                    path = [.chat]
                } label: { Image(systemName: "square.and.pencil") }
                    .accessibilityLabel("New chat")
            }
        }
        .safeAreaInset(edge: .bottom) {
            StatusLineView().padding(.horizontal, 16).padding(.vertical, 6).background(.bar)
        }
        .sheet(isPresented: $showSettings) { SettingsView().environmentObject(m) }
        .alert("Rename chat", isPresented: Binding(get: { renameID != nil }, set: { if !$0 { renameID = nil } })) {
            TextField("Title", text: $renameText)
            Button("Save") { if let id = renameID { m.rename(id, to: renameText) }; renameID = nil }
            Button("Cancel", role: .cancel) { renameID = nil }
        } message: {
            Text("Leave blank to use the first message.")
        }
        .confirmationDialog("Delete this chat?", isPresented: Binding(get: { deleteID != nil }, set: { if !$0 { deleteID = nil } }),
                            titleVisibility: .visible) {
            Button("Delete", role: .destructive) { if let id = deleteID { m.delete(id) }; deleteID = nil }
        }
    }
}

struct ConversationRow: View {
    let c: Conversation
    let replying: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .firstTextBaseline) {
                Text(c.displayTitle).font(.headline).foregroundStyle(.primary).lineLimit(1)
                Spacer()
                Text(dateText).font(.caption).foregroundStyle(.secondary)
            }
            if !c.preview.isEmpty {
                Text(c.preview).font(.subheadline).foregroundStyle(.secondary).lineLimit(2)
            }
            HStack(spacing: 6) {
                if replying {
                    ProgressView().controlSize(.mini)
                    Text("Replying…")
                } else {
                    let n = c.visible.filter { $0.role != .tool }.count
                    Text(n == 1 ? "1 message" : "\(n) messages")
                }
                if c.thinking { Image(systemName: "brain").imageScale(.small) }
            }
            .font(.caption)
            .foregroundStyle(.tertiary)
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
    }

    var dateText: String {
        Calendar.current.isDateInToday(c.updated)
            ? c.updated.formatted(date: .omitted, time: .shortened)
            : c.updated.formatted(.dateTime.month(.abbreviated).day())
    }
}
