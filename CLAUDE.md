# Claudio

Voice I/O for Claude Code — local TTS proxy with streaming speech, multi-destination routing, personas, and session logging.

## Why Claudio?

Claude Code is powerful — but silent. You're staring at a terminal while an AI writes, debugs, and refactors your code, and you can't look away or you'll miss what it's doing. Claudio gives Claude a voice. Hear it narrate its work while you keep your eyes on your editor. No more context-switching to read terminal output. It's just a proxy — sits between Claude Code and the API, extracts speech, forwards everything else untouched. Your workflow doesn't change, it just gets an audio channel.

## Voice Output

Use `<say>` tags for speech. Tags are stripped before display:

```xml
<say>Found the bug - missing null check on line 42.</say>
<say speed="fast">Quick update.</say>
```

## Quick Reference

```bash
claudio                              # Launch Claude with voice
claudio --speak tags                 # Only speak <say> tags (default)
claudio --speak text                 # Speak all text, skip code
claudio --speak all                  # Speak everything incl. code
claudio --speak off                  # No TTS
claudio --tts kokoro --voice nova    # Backend and voice
claudio --speed 1.2                  # Faster speech
claudio --persona narrator           # Use persona
claudio --notify tools               # Announce tool usage

# Discover voices
claudio voices                       # List voices ranked by quality
claudio try                          # Try system default voice
claudio try --voice Samantha         # Try specific voice

# Proxy (standalone)
claudio proxy                        # Start proxy server
claudio proxy --config claudio.yaml  # With YAML config
claudio proxy --port 8080            # Custom port
```

## Options

**Speech:**
- `--speak`: tags (default), text, all, off
- `--notify`: tools, approval, or comma-separated tool names

**TTS:**
- `--tts`: auto, say, avfoundation, kokoro, pocket, soprano, qwen3, off
- `--voice`: Voice name
- `--speed`: Speed multiplier (0.5-2.0)

**Connection:**
- `--proxy-url`: Use existing proxy at URL
- `--port`: Proxy port (default 9000)

**Other:**
- `--persona`: Voice persona (narrator, alert, coder)
- `--config`: Path to claudio.yaml
- `--device`: Audio output device ID
- `--asr`: off, whisper, deepgram (WIP)

## How It Works

1. `claudio` starts proxy in background (or connects to existing)
2. Sets `ANTHROPIC_BASE_URL` to route through proxy
3. Injects voice skill via `--append-system-prompt`
4. Proxy strips `<say>` tags, extracts text for TTS
5. Session logs written to `.claudio/sessions/`

## Architecture

```
src/claudio/
  cli.py               # Typer CLI, launches Claude with voice
  proxy.py             # API proxy, tag extraction, TTS pipeline
  config.py            # YAML configuration (claudio.yaml)
  session_logger.py    # Session logging (JSONL + SQLite)
  personas.py          # Voice personas and voice store
  triggers.py          # Hotkeys and wake word triggers
  wakeword_trainer.py  # Synthetic wake word training
  tts/                 # TTS backends (say, avfoundation, kokoro, pocket, soprano, qwen3)
```
