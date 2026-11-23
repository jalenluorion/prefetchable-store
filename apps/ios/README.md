# Moved Store for iPhone

A chat app that runs Qwen3.6-35B-A3B or Inkling 2 Mini on the iPhone itself, with the engine in this repository. The
experts stay on the phone's storage and are read as they are needed; the app holds about 2.9 GB in memory. Once a model
is downloaded, chats work without a connection.

What the app does:

- **Two models.** Qwen3.6-35B-A3B and Inkling 2 Mini, one open at a time, switched in Settings.
- **In-app download.** Each model comes from its Hugging Face repo in the profile that fits the phone, every file
  checked against its SHA-256, resumable, in the background.
- **Exact or Fast.** Exact gives the model's own choices; Fast leaves out a few low-weight experts and replies sooner.
- **Reads while you type.** The app processes your message in the background as you write it, so a reply starts
  sooner after you send.
- **Chats on the phone.** Saved conversations, thinking on or off per chat, Markdown replies, export as Markdown or
  JSON. Optional web search (DuckDuckGo by default; on by default, off in Settings or per chat with the globe button).

## What you need

- **A Mac with Xcode 16.3 or later.** The package dependencies need Swift tools 6.1. The app was built with Xcode 26.3.
- **An iPhone with an A16 chip or newer, on iOS 17 or later** (iPhone 14 Pro and later, iPhone 15 and later). The
  engine is compiled for the A16's instruction set.
- **An Apple ID.** A free one works; Xcode signs the app for 7 days, after which you build and run it again.
- **Free storage on the phone** for the model you choose, plus 1 GB while it downloads and up to 0.6 GB for saved chat
  states:

| model | profile `phone-small` | profile `phone` |
|---|---|---|
| Qwen3.6-35B-A3B | 20.35 GB (20,352,447,020 bytes, 49 files) | 21.31 GB (21,309,272,620 bytes, 49 files) |
| Inkling 2 Mini | 15.53 GB (15,526,167,637 bytes, 35 files) | 17.99 GB (17,992,385,621 bytes, 35 files) |

The app picks the profile from the memory iOS makes available to it when it opens:

- **`phone`**: at least 5.0 GB available to the app. 8-bit in-memory weights, Exact by default.
- **`phone-small`**: the 6 GB class (iPhone 14 Pro, iPhone 15, iPhone 15 Plus). Calibrated 4-bit in-memory weights,
  Fast by default.

Below 3 GB available the app says the phone does not have enough memory instead of offering the download.

The app asks iOS for more memory with the `com.apple.developer.kernel.increased-memory-limit` entitlement
(`MovedStore/MovedStore.entitlements`, device builds only). Xcode's automatic signing with a free Apple ID accepted it
in our use.

## Build and run

1. Clone the repository and open the project:

   ```bash
   git clone https://github.com/jalenluorion/moved-store && cd moved-store
   open apps/ios/MovedStore.xcodeproj
   ```

   Xcode fetches the one package from outside the repository, swift-transformers (the tokenizer and the Qwen chat
   template), at the versions pinned in the project. The engine is the Swift package at the repository's root
   (`Package.swift`), built from source.

2. Select the **MovedStore** target, open **Signing & Capabilities**, choose your own **Team** (your Apple ID; add it
   under Xcode › Settings › Accounts if it is not listed) and change the **Bundle Identifier** from
   `com.example.movedstore` to one of your own, such as `com.yourname.movedstore`.
3. Connect the iPhone with a cable, unlock it and trust the Mac. Turn on **Developer Mode** (Settings › Privacy &
   Security › Developer Mode; the phone restarts).
4. Choose the iPhone as the run destination and press **Run**. The shared scheme builds the Release configuration,
   which is what the numbers below were measured with. The first build compiles the engine and takes a few minutes.
5. On the first launch of an app from a free Apple ID, iOS asks you to trust the developer: Settings › General ›
   VPN & Device Management, then your Apple ID, then Trust.

The model does not run in the Simulator. The Simulator shows the screens only.

To check that the project builds without a phone or a signing team:

```bash
cd apps/ios
xcodebuild -project MovedStore.xcodeproj -target MovedStore -configuration Release -sdk iphoneos \
  CODE_SIGNING_ALLOWED=NO SYMROOT="$PWD/build" build
```

`SYMROOT` keeps the packages' build products in one folder, where the app's build finds them.

## First launch

The first screen asks which model to download: Qwen3.6-35B-A3B or Inkling 2 Mini. It shows the size for this phone's
profile (the table above) and starts when you tap **Download**.

- The app reads the model's `moved-store.json` from its Hugging Face repo
  ([Qwen3.6-35B-A3B](https://huggingface.co/jalenluorion/Qwen3.6-35B-A3B-moved-store),
  [Inkling 2 Mini](https://huggingface.co/jalenluorion/Inkling-2-Mini-moved-store)) and downloads exactly the files of
  this phone's profile.
- Every file is checked against the SHA-256 in that manifest before it is put in place. A damaged or interrupted file
  is fetched again, resuming where it stopped.
- The download runs in the background: you can leave the app or lock the phone. Keep it plugged in. It uses Wi-Fi only
  unless you turn on **Use cellular data**. **Pause** and **Resume** are on the same screen.
- If the phone does not have the space (the files plus 1 GB), the screen says how much to free.
- When the last file checks out, the chat opens. The model loads in about 3 seconds; you can type meanwhile, and a
  message sent before the model is ready waits for it.

## Using the app

**Exact and Fast** are under Settings › Model › Mode, one setting per model. The default is Fast on `phone-small` and
Exact on `phone`. A change applies the next time you open the app (or when you switch models).

**The other model.** Settings › Model files shows one model's files at a time: choose the other model in its picker
and tap **Download**. Once both are on the phone, a **Model** picker appears in Settings; choosing a model closes the open
one and opens the other with a new chat. A chat made with one model opens read-only while the other is selected.

Both models together take 35.9 GB on a 6 GB phone. With less free storage, keep one model at a time: delete one before
you download the other.

**Removing a model.** Settings › Model › **Delete model** removes the selected model's files and its saved chat states;
Settings › Model files › **Delete model** does the same for the model shown there. Your chats are kept. Deleting the
app removes everything.

**Settings › Speed** turns off saved chat progress (the states that let a chat continue without reading it again) or
reading your message while you type. Settings › Diagnostics shows the engine's settings, the last reply's speed and
the app's memory.

## Speed on an iPhone 15 Plus (6 GB)

Measured on the device, profile `phone-small`. Phones with more memory have not been measured.

**Qwen3.6-35B-A3B** (calibrated 4-bit in-memory weights, about 2.9 GB of app memory)

| | Exact | Fast |
|---|---|---|
| decode, tokens/s | 2.9 to 3.4 | 3.7 to 4.1 |
| top choice equal to the exact 8-bit engine, prose / agent text | 0.913 / 0.884 | 0.891 / 0.881 |

| first token | seconds |
|---|---|
| first reply to a short message | 4.8 |
| a follow-up | 4.0 to 4.2 |
| a chat reopened after relaunching the app | 4.2 to 4.3 |
| a typed message (the app reads it while you type) | 3.3 to 3.9 |
| a 470-token prompt pasted and sent at once | 19.5 |
| launch to ready | 2.7 to 3.2 |

Exact output on the phone is identical to the Mac engine's at the same weights.

**Inkling 2 Mini** (calibrated 4-bit in-memory weights, late expert layers stored at 4-bit, about 2.85 GB of app
memory): 1.46 tokens/s in Exact, identical to the Mac at the same profile, and 2.44 tokens/s in Fast. Its first-token
times have not been measured.

Both models are bound by the phone's storage read rate, 0.8 to 0.9 GB/s measured.

## Checking a build

Add `'SWIFT_ACTIVE_COMPILATION_CONDITIONS=$(inherited) SELFCHECK'` to the `xcodebuild` command (normal builds contain none of this), install, then run `xcrun devicectl device process launch --console --device <id> <bundle id> -- -selfcheck`.
It hashes the selected model's files against the Hub manifest, sends one chat message (then deletes that chat), decodes 400 tokens greedily in Exact, writes `Documents/selfcheck.json`, logs `SELFCHECK DONE` and exits.

## Troubleshooting

- **"Failed to register bundle identifier" or "No profiles for ... were found".** The bundle identifier is taken or
  still the placeholder: set one of your own (step 2) and choose your team.
- **"Untrusted Developer" when the app opens.** Trust your Apple ID on the phone (step 5).
- **The iPhone is not offered as a destination, or Xcode asks for Developer Mode.** Turn on Developer Mode (step 3)
  and unlock the phone.
- **The app stops opening after a week.** A free Apple ID's signature lasts 7 days: build and run again. Your models
  and chats stay.
- **"Not enough storage".** Free the space the screen names (Settings › General › iPhone Storage); the download starts
  by itself once the space is there.
- **The download waits.** It uses Wi-Fi unless **Use cellular data** is on, and continues when the connection returns.

## Files

| path | what it is |
|---|---|
| `MovedStore.xcodeproj` | the Xcode project; it uses the repository's `Package.swift` (product `DmengKit`) by relative path |
| `MovedStore/DmengChatEngine.swift` | the engine behind the chat: profile, mode and options per model, prefill and decode, type-ahead, saved states |
| `MovedStore/TextCodec.swift`, `ChatTemplate.swift`, `InklingChatTemplate.swift` | tokenizer, Qwen's chat template, Inkling's front end |
| `MovedStore/ModelDownloader.swift`, `DownloadConfig.swift`, `SetupView.swift` | the download from the Hub by profile, checks, the first-run and Model files screens |
| `MovedStore/ModelKind.swift` | the two models: folders, repos, defaults per profile |
| `MovedStore/ChatModel.swift`, `ChatView.swift`, `SettingsView.swift`, ... | the chat app |
| `DmengKit/` | the Swift wrapper over the engine's C API (`bindings/c/dmeng_c.h`) |

## License

MIT, as the rest of the repository (`LICENSE`). The models are Apache-2.0: Qwen3.6-35B-A3B by the Qwen team, Inkling 2
Mini by Thinking Machines.
