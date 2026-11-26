import SwiftUI
import UIKit

@main
struct MovedStoreApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var gate = SetupGate.shared
    @StateObject private var launch = LaunchGate()
    @Environment(\.scenePhase) private var scenePhase

    init() {
        setvbuf(stdout, nil, _IOLBF, 0)   // line-buffered console output
    }

    var body: some Scene {
        WindowGroup {
            // First launch without a complete model: the setup (download) screen. The chat, and with it the engine, is
            // only created once the model's files are complete.
            if gate.showSetup {
                SetupScreen()
            } else if launch.deferred {
                Color.clear  // iOS launched the app in the background (download events): no chat, no engine (LaunchGate)
            } else {
                ChatRoot()
            }
        }
        .onChange(of: scenePhase) { _, p in
            if p == .active { launch.opened() }
        }
    }
}

/// A launch in the background is iOS relaunching the app for the model download's background URLSession
/// (AppDelegate.handleEventsForBackgroundURLSession): it only services the download events. The chat, and with it the
/// engine load, waits until the app is opened (scene active). The Mac is never deferred.
@MainActor
final class LaunchGate: ObservableObject {
    @Published private(set) var deferred: Bool

    init() {
        #if targetEnvironment(macCatalyst)
        deferred = false
        #else
        deferred = UIApplication.shared.applicationState == .background
        #endif
        if deferred { log("LAUNCH in the background without arguments (the download session's events): the chat and the engine load wait until the app is opened") }
    }

    func opened() {
        guard deferred else { return }
        deferred = false
        log("LAUNCH opened: starting the chat")
    }
}

/// The setup screen as the app's root; opens the chat by itself when the download completes.
struct SetupScreen: View {
    @StateObject private var d = ModelDownloader.shared
    @Environment(\.scenePhase) private var phase
    var body: some View {
        SetupView(mode: .firstRun)
            .environmentObject(d)
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true   // the screen stays on while it downloads in front
                #if SELFCHECK
                SelfCheck.setupShown()
                #endif
                d.onComplete = { SetupGate.shared.downloadFinished() }
                d.resumeIfWanted()
            }
            .onChange(of: phase) { _, p in
                guard p == .active else { return }
                SetupGate.shared.appBecameActive()   // a download that finished in the background opens the chat now
                d.appBecameActive()                  // no space / no connection / failed: try again by itself
            }
    }
}

/// The chat app.
struct ChatRoot: View {
    @StateObject private var model = ChatModel()
    @Environment(\.scenePhase) private var phase

    var body: some View {
        RootView()
            .environmentObject(model)
            .onAppear {
                SetupGate.engineCreated = true
                UIApplication.shared.isIdleTimerDisabled = true
                #if SELFCHECK
                if SelfCheck.requested { SelfCheck.start(model) } else { model.startOnce() }
                #else
                model.startOnce()
                #endif
                #if UITEST
                UITest.start(model)
                #endif
            }
            .onChange(of: phase) { _, new in
                model.phaseChanged(new)
            }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    /// iOS relaunched the app (in the background) because the model download session has events: the downloader's
    /// session is recreated by ModelDownloader.shared, and this handler is called when its events are delivered.
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        log("DOWNLOAD background session events for \(identifier)")
        Task { @MainActor in ModelDownloader.shared.backgroundCompletion = completionHandler }
    }
}
