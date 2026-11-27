# PhoneStore for iPhone

PhoneStore is a chat app that runs Qwen3.6-35B-A3B or Inkling 2 Mini on the iPhone itself, with the engine in this
repository. The experts stay on the phone's storage and are read as they are needed; the app holds about 2.9 GB in
memory. Once a model is downloaded, chats work without a connection.

What the app does:

- **Two models.** Qwen3.6-35B-A3B and Inkling 2 Mini, one open at a time, switched in Settings.
- **In-app download.** Each model comes from its Hugging Face repo in the profile that fits the phone, every file
  checked against its SHA-256, resumable, in the background.
- **`fast` or `exact`.** `fast`, the default, leaves out a few low-weight experts and replies sooner; `exact` gives
  answers identical to the full model's, more slowly.
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

- **`phone`**: at least 5.0 GB available to the app. 8-bit in-memory weights, Inkling 2 Mini's experts as released,
  `fast` by default.
- **`phone-small`**: the 6 GB class (iPhone 14 Pro, iPhone 15, iPhone 15 Plus). Calibrated 4-bit in-memory weights,
  `fast` by default.

Below 3 GB available the app says the phone does not have enough memory instead of offering the download.

The app asks iOS for more memory with the `com.apple.developer.kernel.increased-memory-limit` entitlement
(`MovedStore/MovedStore.entitlements`, device builds only). Xcode's automatic signing with a free Apple ID accepted it
in testing.

## Build and run

1. Clone the repository and open the project:

   ```bash
   git clone https://github.com/jalenluorion/prefetchable-store && cd prefetchable-store
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
   VPN & Device Management, then your Apple ID, then Trust. Then open PhoneStore from the home screen.

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
  ([Qwen3.6-35B-A3B](https://huggingface.co/jalenluorion/Qwen3.6-35B-A3B-prefetchable-store),
  [Inkling 2 Mini](https://huggingface.co/jalenluorion/Inkling-2-Mini-prefetchable-store)) and downloads exactly the
  files of this phone's profile.
- Every file is checked against the SHA-256 in that manifest before it is put in place. A damaged or interrupted file
  is fetched again, resuming where it stopped.
- The download runs in the background: you can leave the app or lock the phone. Keep it plugged in. It uses Wi-Fi only
  unless you turn on **Use cellular data**. **Pause** and **Resume** are on the same screen.
- If the phone does not have the space (the files plus 1 GB), the screen says how much to free.
- When the last file checks out, the chat opens. The model loads in about 2 seconds; you can type meanwhile, and a
  message sent before the model is ready waits for it.

## Using the app

**Mode.** `fast` and `exact` are under Settings › Model › Mode, one setting per model. `fast` is the default for both
models on every phone; choose `exact` for answers identical to the full model's, at a slower speed. A change applies
the next time you open the app (or when you switch models).

**Spare memory.** On a phone with 11 GB or more of memory (the iPhone 17 Pro class), the model's files that the system
keeps in memory the app does not own are used before reading storage. It is on by default there in both modes
([docs/MANIFEST-PRESETS.md](../../docs/MANIFEST-PRESETS.md), `spare_memory`). Settings › Diagnostics › Use spare memory for the model's files is shown where it is on; a choice made
there holds in either mode from the next open. It pauses for the rest of the session after a memory warning.

**Web search.** With page reads enabled, DuckDuckGo finds candidate URLs and the app fetches their pages to look for
relevant passages. It can retry with a more specific query when fewer than two sources qualify. The model is prompted
to open a numbered result before answering, and recent numbered results reuse the page already fetched by the app. The
evidence budgets are 128 tokens per search and 256 per read on iPhone 15, and 448/896 on the 12 GB class such as
iPhone 17 Pro, which uses a 3.5× prefill-speed estimate. The phone's local date is passed with each user turn so the
warmed system prompt stays the same across days.

**The other model.** Settings › Model files shows one model's files at a time: choose the other model in its picker
and tap **Download**. Once both are on the phone, a **Model** picker appears in Settings; choosing a model closes the
open one and opens the other with a new chat. A chat made with one model opens read-only while the other is selected.

Both models together take 35.9 GB on a 6 GB phone. With less free storage, keep one model at a time: delete one before
you download the other.

**Removing a model.** Settings › Model › **Delete model** removes the selected model's files and its saved chat states;
Settings › Model files › **Delete model** does the same for the model shown there. Your chats are kept. Deleting the
app removes everything.

**Settings › Speed** turns off saved chat progress (the states that let a chat continue without reading it again) or
reading your message while you type. Settings › Diagnostics shows the engine's settings, the last reply's speed and
the app's memory; **Copy diagnostics** in the same screen copies the last reply's numbers and the engine's settings as
text.

## Speed on an iPhone 15 Plus (6 GB)

Measured on the device, profile `phone-small`. Estimates for an iPhone 17 Pro (12 GB) are in the
[main README](../../README.md#iphone-17-pro-estimated).

**Qwen3.6-35B-A3B** (calibrated 4-bit in-memory weights, about 2.9 GB of app memory)

| | `exact` | `fast` |
|---|---|---|
| decode (tokens/s) | 3.0-3.7 | 5.0-6.4 |
| top choice equal to the `exact` 8-bit engine, prose / agent text | 0.913 / 0.884 | 0.860 / 0.868 |

| first token (s) | `exact` | `fast` |
|---|---|---|
| first reply to a short message | 4.6-5.4 | 3.2-3.3 |
| a follow-up | 4.0-4.3 | 2.6 |
| a chat reopened after relaunching the app | 4.4-4.6 | 2.7 |
| a typed message (the app reads it while you type) | 3.2-3.3 | 2.3 |
| a 470-token prompt pasted and sent at once | 15.4 | 16.9 |

Launch to ready takes 1.9 to 2.4 s. With the phone's small cache, `fast` leaves out more than it does on the Mac: that
is what its speed costs here (the main README's Quality section has both).

`exact` output on the phone is identical to the Mac engine's at the same weights.

**Inkling 2 Mini** (calibrated 4-bit in-memory weights, late expert layers stored at 4-bit, about 2.85 GB of app
memory): 1.4 to 1.5 tokens per second in `exact`, identical to the Mac at the same profile, and 4.5 to 5.3 in `fast`.

| first token (s) | `exact` | `fast` |
|---|---|---|
| first reply to a short message | 10.5-11.1 | 2.1-2.6 |
| a follow-up | 6.1-6.2 | 1.4-1.6 |
| a typed message (the app reads it while you type) | 5.2 | 1.1 |

Launch to ready takes 1.4 to 1.8 s.

Both models are bound by the phone's storage read rate, 0.8 to 0.9 GB/s measured.

## Drafting in the C API

The app has no setting for drafting, and it is off in every profile. Opened with `model_options` holding
`run:draft=lookup` (or `run:draft=head` / `run:draft=both` for Inkling 2 Mini's draft head, `run:draft=none` to turn a
profile's default off; optional `run:draft-max=N`, 1 to 7), the engine drafts when the reply is decoded with
`dm_step_n` instead of `dm_step` + `dm_sample`. Pass it the token just chosen; the sampler (NULL where the loop takes
`dm_argmax`, temperature 0: `dm_sample` would apply the penalties there); the reply's tokens so far (the same `hist`
given to `dm_sample`, that token included); the stop ids; and `max_out` (at most 8; the number of tokens before one the
thinking budget will force, 1 when the next token is forced). Handle the returned tokens in order as if each had been
sampled; the next call feeds the last one. The tokens are the ones the current loop produces, only several per step
where drafts are kept.

Text the reply is likely to copy (a file being rewritten, a tool's call format) goes to `dm_set_draft_fragments` before
the reply; the prompt and the chat's earlier replies are searched without it. With drafting on, the slab is smaller by
drafting's memory (`dm_stats_json` `policy.slab_rule.draft_reserve_gb`; Inkling 2 Mini's head adds about 0.7 GB), and
`dm_stats_json` reports `draft` (tokens per step, kept drafts, the first pass's memory).

## Checking a build

Run `bash apps/ios/tests/search_check.sh` from the repository root to check page extraction, device budgets and dates,
and a fixture-backed search through the numbered cache read.

For the self-check on the device, add `'SWIFT_ACTIVE_COMPILATION_CONDITIONS=$(inherited) SELFCHECK'` to the
`xcodebuild` command (normal builds contain none of this), install, then run
`xcrun devicectl device process launch --console --device <id> <bundle id> -- -selfcheck`. It hashes the selected
model's files against the Hub manifest, sends one chat message (then deletes that chat), decodes 400 tokens greedily
in `exact`, writes `Documents/selfcheck.json`, logs `SELFCHECK DONE` and exits. With `-selfcheck-keep-chat` the chat is
kept instead; the next launch with `-selfcheck-resume` opens it, sends a follow-up, then deletes it and the saved
states it made.

## Checking the chat's scrolling

A Simulator build with `'SWIFT_ACTIVE_COMPILATION_CONDITIONS=$(inherited) UITEST'` (and `-sdk iphonesimulator ARCHS=arm64`;
normal builds contain none of this) runs scripted replies through the chat instead of a model
(`MovedStore/UITest.swift`). Launch it with `-uitest <scenario>`: `follow` (two replies at the bottom: thinking, tool
rounds, sources), `keyboard` (the keyboard and a growing draft during a reply), `hold` (a long chat; drag up during the
reply and let go), `history` (scroll up through a long chat) or `open` (chats opened and switched). The app records the
chat's scroll view every frame to `Documents/uitest-<scenario>.json`; `python3 apps/ios/tests/scroll_check.py <file>`
checks it: at the bottom within 1 pt while following, a held position within 1 pt while the reply below changes,
content moving with the finger. `hold` and `history` wait for touches from the Simulator (`UITEST TOUCH NOW` in the
console; `-uitestQuiet N` keeps `history` open N seconds after the last scroll, for a tap on the arrow).

## Checking the speed figure

The tokens/s the app shows (the live line, the caption under a reply, the status line, Diagnostics and Copy
diagnostics) is one figure, `GenSpeed.rate` in `ChatEngine.swift`: tokens generated divided by the time the engine
spent stepping them, on a monotonic clock, with any time the app spent in the background left out. A message made of
several rounds (web search) adds its rounds, the search time in none; the prompt keeps its own "started in" figure.
Every reply logs `RATE` with the shown figure beside the engine counters' (decode steps over the time inside
`dm_step`); the self-check fails when they differ by more than 2%. The UITEST scenario `rate` (`-uitest rate`) runs a
normal reply, one with tool rounds, one with the app in the background about 10 s in the middle (it logs
`UITEST BACKGROUND NOW`: bring another app to the front for 10 s, then this one), and a stopped one, and logs
`UITEST RATE ... OK` when the figure is the scripted rate.

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
| `MovedStore/TextCodec.swift`, `ChatTemplate.swift`, `InklingChatTemplate.swift` | tokenizer, Qwen's chat template, Inkling 2 Mini's front end |
| `MovedStore/ModelDownloader.swift`, `DownloadConfig.swift`, `SetupView.swift` | the download from the Hub by profile, checks, the first-run and Model files screens |
| `MovedStore/ModelKind.swift` | the two models: folders, repos, defaults per profile |
| `MovedStore/ChatModel.swift`, `ChatView.swift`, `SettingsView.swift`, ... | the chat app |
| `DmengKit/` | the Swift wrapper over the engine's C API (`bindings/c/dmeng_c.h`) |

## License

MIT, as the rest of the repository (`LICENSE`). The models are Apache-2.0: Qwen3.6-35B-A3B by the Qwen team, Inkling 2
Mini by Thinking Machines.
