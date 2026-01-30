# Multi-Destination Proxy Routing

Claudio proxy can route requests to multiple API endpoints based on path patterns. This enables a single proxy to handle Anthropic, OpenAI, local LLMs, and custom endpoints.

## Quick Start

Configure destinations in `claudio.yaml`:

```yaml
destinations:
  - name: anthropic
    url: https://api.anthropic.com
    paths: [/v1/messages, /v1/messages/*]
    api_key_env: ANTHROPIC_API_KEY

  - name: openai
    url: https://api.openai.com
    paths: [/openai/*, /v1/chat/*, /v1/completions/*]
    api_key_env: OPENAI_API_KEY
    tts_enabled: false

default_destination: anthropic
```

Start the proxy:

```bash
claudio proxy --config claudio.yaml
# or just: claudio proxy (auto-discovers claudio.yaml)
```

## Path Patterns

Patterns use glob-like syntax:

| Pattern | Matches |
|---------|---------|
| `/v1/messages` | Exact path only |
| `/v1/messages/*` | One level: `/v1/messages/stream` |
| `/api/**` | Any depth: `/api/v1/users/123` |
| `/v1/?hat/*` | Single char: `/v1/chat/completions` |

### Examples

```yaml
destinations:
  # Anthropic - exact Claude API paths
  - name: anthropic
    url: https://api.anthropic.com
    paths:
      - /v1/messages
      - /v1/messages/*

  # OpenAI - all chat/completion paths
  - name: openai
    url: https://api.openai.com
    paths:
      - /v1/chat/completions
      - /v1/completions
      - /v1/embeddings

  # OpenAI via prefix (for namespacing)
  - name: openai-prefixed
    url: https://api.openai.com
    paths:
      - /openai/**

  # Local Ollama
  - name: ollama
    url: http://localhost:11434
    paths:
      - /ollama/**
      - /local/**

  # Custom inference server
  - name: vllm
    url: http://gpu-server:8000
    paths:
      - /vllm/**
```

## Destination Configuration

### Full Schema

```yaml
destinations:
  - name: anthropic              # Required: unique identifier
    url: https://api.anthropic.com  # Required: base URL
    paths:                       # Required: path patterns to match
      - /v1/messages
      - /v1/messages/*
    headers:                     # Optional: headers to add
      X-Custom-Header: value
    api_key_env: ANTHROPIC_API_KEY  # Optional: env var for API key
    tts_enabled: true            # Optional: enable TTS (default: true)
```

### API Key Injection

The proxy automatically injects API keys from environment variables:

```yaml
# For Anthropic-style APIs (x-api-key header)
- name: anthropic
  api_key_env: ANTHROPIC_API_KEY
  # Adds: x-api-key: $ANTHROPIC_API_KEY

# For OpenAI-style APIs (Authorization: Bearer)
- name: openai
  api_key_env: OPENAI_API_KEY
  # Adds: Authorization: Bearer $OPENAI_API_KEY
```

The header format is determined by the destination name:
- Names containing "openai" → `Authorization: Bearer`
- All others → `x-api-key`

### Per-Destination TTS

Disable TTS for destinations that don't benefit from it:

```yaml
destinations:
  - name: anthropic
    tts_enabled: true   # Speak Claude responses

  - name: openai
    tts_enabled: false  # Don't speak GPT responses

  - name: embeddings
    url: https://api.openai.com
    paths: [/v1/embeddings]
    tts_enabled: false  # Embeddings are data, not text
```

### Custom Headers

Add headers for authentication, routing, or metadata:

```yaml
destinations:
  - name: internal-api
    url: https://api.internal.com
    paths: [/internal/**]
    headers:
      X-Internal-Auth: secret-token
      X-Request-Source: claudio-proxy
```

## Default Destination

When no path matches, requests go to the default:

```yaml
default_destination: anthropic  # Use anthropic if no match
```

If not specified, the first destination is used as default.

## Routing Examples

### Single Proxy for Multiple Providers

```yaml
destinations:
  - name: claude
    url: https://api.anthropic.com
    paths: [/claude/*, /v1/messages]
    api_key_env: ANTHROPIC_API_KEY

  - name: gpt
    url: https://api.openai.com
    paths: [/gpt/*, /v1/chat/*]
    api_key_env: OPENAI_API_KEY

  - name: local
    url: http://localhost:11434
    paths: [/local/*]

default_destination: claude
```

Usage:
```bash
# Claude (default)
curl http://localhost:9000/v1/messages -d '...'

# GPT via prefix
curl http://localhost:9000/gpt/v1/chat/completions -d '...'

# Local via prefix
curl http://localhost:9000/local/api/generate -d '...'
```

### Development vs Production

```yaml
# dev-claudio.yaml
destinations:
  - name: mock
    url: http://localhost:8080
    paths: [/**]  # Catch all

# prod-claudio.yaml
destinations:
  - name: anthropic
    url: https://api.anthropic.com
    paths: [/v1/messages]
    api_key_env: ANTHROPIC_API_KEY
```

### Load Balancing (Manual)

```yaml
destinations:
  - name: gpu-1
    url: http://gpu1:8000
    paths: [/gpu1/**]

  - name: gpu-2
    url: http://gpu2:8000
    paths: [/gpu2/**]

  - name: primary
    url: http://gpu1:8000
    paths: [/v1/**]  # Default to gpu1
```

## Status Endpoint

Check routing configuration:

```bash
curl http://localhost:9000/status | jq .routing
```

```json
{
  "routing": {
    "destinations": [
      {
        "name": "anthropic",
        "url": "https://api.anthropic.com",
        "paths": ["/v1/messages", "/v1/messages/*"],
        "tts_enabled": true
      },
      {
        "name": "openai",
        "url": "https://api.openai.com",
        "paths": ["/openai/*"],
        "tts_enabled": false
      }
    ],
    "default": "anthropic"
  }
}
```

## Startup Output

The proxy shows routing on startup:

```
Claudio Proxy
  Listening: http://127.0.0.1:9000
  Routing:
    anthropic → https://api.anthropic.com TTS
      /v1/messages, /v1/messages/*
    openai → https://api.openai.com no-TTS
      /openai/*, /v1/chat/*, /v1/completions/*
  TTS:       Kokoro (nova)
  Speak:     markers
```
