#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
build_dir=$(mktemp -d)
trap 'rm -r "$build_dir"' EXIT

# Compile the actual tool definitions and reader without the app or model weights.
awk '/^public enum ToolPrompt \{/{ printing=1 } printing{ print } printing && /^\}/{ exit }' \
    ../MovedStore/ChatTemplate.swift > "$build_dir/ToolPrompt.swift"
xcrun swiftc -O ../MovedStore/Tools.swift "$build_dir/ToolPrompt.swift" main.swift -o "$build_dir/search_check"
"$build_dir/search_check" -readerCheck
"$build_dir/search_check" -searchE2E
