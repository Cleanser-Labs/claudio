#!/bin/bash
# Claude Code hook to speak assistant responses via TTS
#
# Setup:
# 1. Start TTS server: just tts-server
# 2. Add hook to Claude Code settings (~/.config/claude-code/settings.json):
#
# {
#   "hooks": {
#     "assistant_response": {
#       "command": "/path/to/voice-control/hooks/speak-response.sh"
#     }
#   }
# }

# Read the response from stdin (Claude Code pipes it)
response=$(cat)

# Extract just the text content (skip tool calls, code blocks, etc.)
# This is a simple heuristic - adjust as needed
text=$(echo "$response" | grep -v '^\s*```' | grep -v '^\s*<' | head -c 500)

# Send to TTS server
if [ -n "$text" ]; then
  curl -s -X POST http://127.0.0.1:8765/speak \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$text\", \"speed\": 1.2}" \
    | afplay -
fi
