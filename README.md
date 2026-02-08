# Claudio

Voice I/O for Claude Code.

```bash
uv tool install git+https://github.com/cleanser-labs/claudio
claudio init    # picks the best TTS for your system
claudio         # launches Claude Code with voice
```

## How It Works

`claudio` starts a local proxy, sets `ANTHROPIC_BASE_URL`, and launches Claude Code. Claude uses `<say>` tags to control what gets spoken — tags are stripped before reaching your terminal. You see clean text, hear the voice.

```
you ──► claudio proxy ──► anthropic / openai / local LLM
              │
              ├── strips <say> tags
              ├── speaks text via TTS
              ├── logs requests, responses, audio
              └── returns clean response to client
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
claudio                          # Launch Claude with voice
claudio --speak auto             # Speak all text (not just <say> tags)
claudio --tts kokoro --voice nova # Pick backend and voice
claudio --speed 1.2              # Faster speech
claudio --persona narrator       # Use a persona
```

### Standalone proxy

```bash
claudio proxy                        # Start proxy server
claudio proxy --config claudio.yaml  # With YAML config
claudio proxy --port 8080            # Custom port
```

### Voice discovery

```bash
claudio voices                       # List available voices
claudio try                          # Test default voice
claudio try --voice nova             # Test specific voice
```

## Features

### TTS Backends

Six backends from fast-and-simple to high-quality neural:

```bash
claudio proxy --tts say          # macOS say (instant, no deps)
claudio proxy --tts kokoro       # 82M param neural (fast, good quality)
claudio proxy --tts pocket       # 100M param CPU-friendly, voice cloning
claudio proxy --tts soprano      # Neural TTS with vocoder
claudio proxy --tts qwen3        # 1.7B param (best quality, needs GPU)
```

Use `auto` (the default) to pick the best available backend.

Pocket-TTS supports voice cloning from a WAV file:

```python
from claudio import tts

engine = tts.Pocket()
engine.clone_voice("my_voice.wav", name="me")
engine.speak("Hello in my voice", voice="me")
```

See [TTS Backends Guide](docs/tts-backends.md) for details.

### Multi-Destination Routing

Route requests to Anthropic, OpenAI, local LLMs, or any API from a single proxy. Configure in `claudio.yaml`:

```yaml
destinations:
  - name: anthropic
    url: https://api.anthropic.com
    paths: [/v1/messages, /v1/messages/*]
    api_key_env: ANTHROPIC_API_KEY

  - name: openai
    url: https://api.openai.com
    paths: [/openai/*, /v1/chat/*]
    api_key_env: OPENAI_API_KEY
    tts_enabled: false           # don't speak GPT responses

  - name: ollama
    url: http://localhost:11434
    paths: [/local/*]

default_destination: anthropic
```

Paths use glob patterns (`*` matches one level, `**` matches any depth). API keys are injected automatically from environment variables.

See [Routing Guide](docs/multi-destination-routing.md) for details.

### Session Logging

Log every request, response, and generated audio file for debugging or replay:

```yaml
logging:
  enabled: true
  storage: jsonl    # or 'sqlite'
  dir: .claudio/sessions
  save_audio: true
```

Logs are organized by date and session:

```
.claudio/sessions/
  2025-01-28/
    abc123/
      events.jsonl
      requests/req001.json
      responses/req001.json
      audio/req001_000.wav
```

Use SQLite for indexed storage:

```yaml
logging:
  storage: sqlite
```

```bash
sqlite3 .claudio/sessions/2025-01-28/sessions.db \
  "SELECT request_id, elapsed, model FROM responses WHERE elapsed > 5.0"
```

See [Session Logging Guide](docs/session-logging.md) for details.

### Personas

Define voice characters in markdown files:

```markdown
---
name: Narrator
voices: [nova, daniel]
priority: 60
speed: 0.9
---

You narrate code reviews like a documentary.
Speak in a calm, measured tone.
```

```bash
claudio --persona narrator
claudio --persona alert          # Terse, high-priority alerts
claudio --persona coder          # Technical, fast-paced
```

### Triggers

Global hotkeys and wake words for hands-free control:

```python
from claudio.triggers import on_hotkey, on_wake_word, run_triggers

@on_hotkey('cmd+shift+c')
def start_coding():
    return {'action': 'start_session', 'persona': 'coder'}

@on_wake_word('hey claude')
def wake_up():
    return {'action': 'listen_start'}

run_triggers()
```

Train custom wake words using claudio's built-in TTS:

```python
from claudio.wakeword_trainer import WakeWordTrainer

trainer = WakeWordTrainer('hey_claude')
trainer.generate_synthetic(count=5000)  # TTS-generated samples
trainer.augment()                       # Noise, pitch, reverb
trainer.train()
trainer.export('models/hey_claude.onnx')
```

## Configuration

Claudio looks for config files in this order:

1. `--config /path/to/file.yaml` (explicit)
2. `./claudio.yaml`
3. `./.claudio/config.yaml`
4. `~/.claudio/config.yaml`

Full schema:

```yaml
destinations:
  - name: anthropic
    url: https://api.anthropic.com
    paths: [/v1/messages]
    api_key_env: ANTHROPIC_API_KEY
    tts_enabled: true

default_destination: anthropic

logging:
  enabled: true
  storage: jsonl
  dir: .claudio/sessions
  log_requests: true
  log_responses: true
  save_audio: true

tts:
  backend: auto
  voice: null
  speed: 1.0
  device: null

proxy:
  host: 127.0.0.1
  port: 9000
  speak_mode: markers
  notify: null
```

See [Configuration Guide](docs/configuration.md) for all options.

## CLI Reference

```bash
# Main launcher
claudio                              # Launch Claude with voice
claudio --speak auto                 # Speak all text
claudio --speak tags                 # Only <say> tags (default)
claudio --speak off                  # No TTS
claudio --tts kokoro --voice nova    # Backend and voice
claudio --speed 1.2                  # Faster speech
claudio --persona narrator           # Use persona
claudio --notify tools               # Announce tool use

# Proxy (standalone)
claudio proxy                        # Start proxy server
claudio proxy --config claudio.yaml  # With YAML config
claudio proxy --port 8080            # Custom port
claudio proxy --tts pocket           # Pocket-TTS backend

# Voice discovery
claudio voices                       # List available voices
claudio try                          # Test default voice
claudio try --voice nova             # Test specific voice
claudio try "Custom text"            # Test with custom text
```

## Status Endpoint

The proxy exposes a status endpoint at `GET /status`:

```bash
curl http://localhost:9000/status | jq
```

```json
{
  "status": "ok",
  "proxy": { "request_count": 42, "uptime_human": "1h 23m" },
  "tts": { "backend": "Kokoro", "voice": "nova" },
  "routing": {
    "destinations": [
      { "name": "anthropic", "url": "https://api.anthropic.com" },
      { "name": "openai", "url": "https://api.openai.com" }
    ]
  },
  "logging": { "enabled": true, "storage": "jsonl" }
}
```

## Architecture

```
src/claudio/
  cli.py               # Typer CLI, launches Claude with voice
  proxy.py             # API proxy, tag extraction, TTS pipeline
  config.py            # YAML configuration (claudio.yaml)
  session_logger.py    # Full session logging (JSONL + SQLite)
  personas.py          # Voice personas and voice store
  triggers.py          # Hotkeys and wake word triggers
  wakeword_trainer.py  # Synthetic wake word training
  tts/
    __init__.py        # Factory and lazy imports
    base.py            # TTS base class and Config
    say.py             # macOS say
    avfoundation.py    # macOS AVSpeechSynthesizer
    soprano.py         # Soprano neural TTS
    kokoro.py          # Kokoro 82M
    pocket.py          # Pocket-TTS 100M (voice cloning)
    qwen3.py           # Qwen3-TTS 1.7B
```

## Guides

- [Configuration](docs/configuration.md) — YAML schema, examples, CLI overrides
- [Multi-Destination Routing](docs/multi-destination-routing.md) — Path patterns, API key injection
- [Session Logging](docs/session-logging.md) — JSONL/SQLite storage, querying logs
- [TTS Backends](docs/tts-backends.md) — Backend comparison, voice cloning, performance
