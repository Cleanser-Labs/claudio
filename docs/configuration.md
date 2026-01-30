# Claudio Configuration

Claudio uses YAML configuration files for proxy settings, routing, TTS, and logging.

## Config File Discovery

The proxy searches for configuration in this order:

1. Explicit path: `claudio proxy --config /path/to/config.yaml`
2. Current directory: `./claudio.yaml`
3. Local .claudio: `./.claudio/config.yaml`
4. Home .claudio: `~/.claudio/config.yaml`

## Full Configuration Schema

```yaml
# Multi-destination routing
destinations:
  - name: anthropic
    url: https://api.anthropic.com
    paths:
      - /v1/messages
      - /v1/messages/*
    headers: {}
    api_key_env: ANTHROPIC_API_KEY
    tts_enabled: true

  - name: openai
    url: https://api.openai.com
    paths:
      - /openai/*
      - /v1/chat/*
    api_key_env: OPENAI_API_KEY
    tts_enabled: false

# Default when no path matches
default_destination: anthropic

# Session logging
logging:
  enabled: true
  storage: jsonl        # 'jsonl' or 'sqlite'
  dir: .claudio/sessions
  log_requests: true
  log_responses: true
  save_audio: true
  max_request_size: 0   # 0 = unlimited
  max_response_size: 0

# Text-to-speech
tts:
  backend: auto         # auto, say, soprano, kokoro, pocket, qwen3
  voice: null           # Backend-specific voice name
  speed: 1.0            # 0.5 to 2.0
  device: null          # Audio device ID

# Proxy server
proxy:
  host: 127.0.0.1
  port: 9000
  speak_mode: markers   # auto, markers, off
  notify: null          # tools, approval, or comma-separated names
```

## Section Reference

### destinations

Route requests to different API endpoints based on path patterns.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Unique identifier |
| `url` | string | required | Base URL for the API |
| `paths` | list[string] | required | Path patterns (glob syntax) |
| `headers` | dict | `{}` | Headers to add to requests |
| `api_key_env` | string | null | Environment variable for API key |
| `tts_enabled` | bool | `true` | Enable TTS for this destination |

See: [Multi-Destination Routing](./multi-destination-routing.md)

### logging

Configure session logging for requests, responses, and audio.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Enable session logging |
| `storage` | string | `jsonl` | Storage backend: `jsonl` or `sqlite` |
| `dir` | string | `.claudio/sessions` | Log directory |
| `log_requests` | bool | `true` | Log full request bodies |
| `log_responses` | bool | `true` | Log full response content |
| `save_audio` | bool | `true` | Save TTS audio files |
| `max_request_size` | int | `0` | Max request size (0=unlimited) |
| `max_response_size` | int | `0` | Max response size (0=unlimited) |

See: [Session Logging](./session-logging.md)

### tts

Configure text-to-speech synthesis.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `auto` | TTS backend to use |
| `voice` | string | null | Voice name (backend-specific) |
| `speed` | float | `1.0` | Speech rate (0.5-2.0) |
| `device` | int | null | Audio output device ID |

**Available backends:**
- `auto` - Auto-detect best available
- `say` - macOS say command
- `soprano` - Neural TTS
- `kokoro` - 82M param fast TTS
- `pocket` - 100M param CPU-friendly with voice cloning
- `qwen3` - 1.7B param high-quality
- `avfoundation` - macOS AVSpeechSynthesizer

See: [TTS Backends](./tts-backends.md)

### proxy

General proxy server settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | string | `127.0.0.1` | Listen address |
| `port` | int | `9000` | Listen port |
| `speak_mode` | string | `markers` | TTS mode |
| `notify` | string | null | Tool notification config |

**Speak modes:**
- `auto` - Speak all response text
- `markers` - Only speak `<say>...</say>` content
- `off` - No TTS

**Notify options:**
- `tools` - Announce all tool uses
- `approval` - Announce Edit, Write, Bash, NotebookEdit
- `Tool1,Tool2` - Comma-separated tool names

## Example Configurations

### Minimal (Anthropic only)

```yaml
# Just enable logging
logging:
  enabled: true
```

### Development Setup

```yaml
destinations:
  - name: local
    url: http://localhost:11434
    paths: [/**]

logging:
  enabled: true
  storage: sqlite

tts:
  backend: say  # Fast, no GPU
  speed: 1.2

proxy:
  speak_mode: auto  # Hear everything
```

### Production Multi-Provider

```yaml
destinations:
  - name: anthropic
    url: https://api.anthropic.com
    paths: [/v1/messages, /anthropic/**]
    api_key_env: ANTHROPIC_API_KEY

  - name: openai
    url: https://api.openai.com
    paths: [/v1/chat/*, /openai/**]
    api_key_env: OPENAI_API_KEY
    tts_enabled: false

  - name: embeddings
    url: https://api.openai.com
    paths: [/v1/embeddings]
    tts_enabled: false

default_destination: anthropic

logging:
  enabled: true
  storage: sqlite
  save_audio: false  # Save space

tts:
  backend: kokoro
  voice: nova
  speed: 1.1

proxy:
  port: 9000
  speak_mode: markers
  notify: approval
```

### CI/Testing (No TTS)

```yaml
destinations:
  - name: mock
    url: http://localhost:8080
    paths: [/**]

logging:
  enabled: true
  storage: jsonl
  save_audio: false

tts:
  backend: auto  # Will be disabled if no audio

proxy:
  speak_mode: off
```

## CLI Override

CLI arguments override config file values:

```bash
# Config file specifies port 9000, override to 8080
claudio proxy --config claudio.yaml --port 8080

# Override TTS
claudio proxy --tts kokoro --voice nova --speed 1.2

# Override speak mode
claudio proxy --speak auto
```

## Environment Variables

Some settings can come from environment:

```bash
# API keys (referenced in destinations)
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...

# Session logging
export CLAUDIO_SESSION_LOG=1
export CLAUDIO_SESSION_DIR=.claudio/sessions
export CLAUDIO_SESSION_STORAGE=sqlite

# Legacy log file (simple JSONL)
export CLAUDIO_LOG_FILE=.claudio/proxy.log
```

## Validation

The proxy validates configuration on startup. Invalid config shows warnings:

```
[yellow]Config load warning:[/yellow] Unknown field 'foo' in destinations[0]
```

Missing required fields cause startup failure.
