# Claudio

Give Claude a voice with real-time TTS.

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
claudio --speak auto                 # Speak all text (not just <say> tags)
claudio --tts kokoro --voice nova    # Pick backend and voice
claudio --speed 1.2                  # Faster speech
claudio --persona narrator           # Use a persona
claudio voices                       # List available voices
claudio try --voice nova             # Audition a voice
```

## Guides

- [Configuration](docs/configuration.md) — YAML schema, examples, CLI overrides
- [Multi-Destination Routing](docs/multi-destination-routing.md) — Path patterns, API key injection
- [Session Logging](docs/session-logging.md) — JSONL/SQLite storage, querying logs
- [TTS Backends](docs/tts-backends.md) — Backend comparison, voice cloning, performance
