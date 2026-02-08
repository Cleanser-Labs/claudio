# Claudio

Give Claude a voice with real-time TTS.




https://github.com/user-attachments/assets/f535dcd9-23c2-46df-b77e-3df8ddf65284



```bash
uv tool install git+https://github.com/cleanser-labs/claudio
claudio init    # picks the best TTS for your system
claudio         # launches Claude Code with voice
```

## Install

### From source

```bash
git clone https://github.com/cleanser-labs/claudio.git
cd claudio
uv sync
```

## Usage

```bash
claudio                              # Launch Claude with voice
claudio --speak text                 # Speak all text, skip code blocks
claudio --tts kokoro --voice nova    # Pick backend and voice
claudio --speed 1.2                  # Faster speech
claudio --persona narrator           # Use a persona
claudio voices                       # List available voices
claudio try --voice nova             # Audition a voice
```

## Using with raw `claude`

You can run the proxy standalone and point any Claude Code session at it:

```bash
# Terminal 1: start the proxy
claudio proxy --speak text

# Terminal 2: launch claude with the proxy
ANTHROPIC_BASE_URL=http://127.0.0.1:9000 claude
```

The proxy intercepts Claude API traffic, extracts text, and speaks it through your configured TTS engine. All other API behavior is unchanged.

### Speak modes

| Mode | Flag | Behavior |
|------|------|----------|
| `tags` | `--speak tags` | Only speaks text inside `<say>...</say>` tags (default). Claude needs a system prompt telling it to use `<say>` tags — `claudio` injects this automatically, but with raw `claude` you'll need to add it yourself. |
| `text` | `--speak text` | Speaks all text automatically, skips code blocks and JSON. No special prompting needed. |
| `all` | `--speak all` | Speaks everything including code blocks. |
| `off` | `--speak off` | TTS disabled, proxy just forwards requests. |

For most setups with raw `claude`, `--speak text` is the easiest since it requires no prompt changes.

## Guides

- [Configuration](docs/configuration.md) — YAML schema, examples, CLI overrides
- [Multi-Destination Routing](docs/multi-destination-routing.md) — Path patterns, API key injection
- [Session Logging](docs/session-logging.md) — JSONL/SQLite storage, querying logs
- [TTS Backends](docs/tts-backends.md) — Backend comparison, voice cloning, performance
