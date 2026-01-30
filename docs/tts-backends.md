# TTS Backends

Claudio supports multiple text-to-speech backends for different use cases, from fast CPU-only synthesis to high-quality neural voices.

## Available Backends

| Backend | Size | Quality | Speed | GPU | Voice Cloning |
|---------|------|---------|-------|-----|---------------|
| `say` | - | Basic | Instant | No | No |
| `avfoundation` | - | Good | Fast | No | No |
| `soprano` | ~500MB | Very Good | Fast | Optional | No |
| `kokoro` | 82M | Good | Very Fast | Optional | No |
| `pocket` | 100M | Good | Fast | No | Yes |
| `qwen3` | 1.7B | Excellent | Medium | Yes | No |

## Quick Start

```python
from claudio import tts

# Auto-detect best backend
engine = tts.create()
engine.speak("Hello world")

# Specify backend
engine = tts.create(tts.Config(backend='kokoro'))
engine.speak("Hello world")
```

## Backend Details

### say (macOS)

Uses the macOS `say` command. No dependencies, instant startup.

```yaml
tts:
  backend: say
  voice: Samantha  # Any macOS voice
  speed: 1.0
```

**Voices:** Run `say -v '?'` to list available voices.

**Pros:** Zero startup time, no dependencies
**Cons:** macOS only, robotic quality

### avfoundation (macOS)

Uses AVSpeechSynthesizer for better quality than `say`.

```yaml
tts:
  backend: avfoundation
  voice: com.apple.voice.compact.en-US.Samantha
```

**Pros:** Better quality than say, native macOS
**Cons:** macOS only, still synthetic-sounding

### soprano

Neural TTS with good quality and fast inference.

```yaml
tts:
  backend: soprano
  voice: default
  speed: 1.0
```

**Installation:**
```bash
uv sync --extra soprano
# or: uv pip install soprano-tts
```

**Pros:** Good quality, reasonable speed
**Cons:** ~500MB model download

### kokoro

82M parameter model, very fast inference, multi-language.

```yaml
tts:
  backend: kokoro
  voice: nova      # or: echo, fable, onyx, shimmer
  speed: 1.1
```

**Installation:**
```bash
uv sync --extra kokoro
# or: uv pip install kokoro-tts
```

**Languages:** English, Japanese, Chinese, French, Spanish, Italian, Portuguese, Hindi

**Voices:**
- `nova` - Warm, conversational
- `echo` - Clear, neutral
- `fable` - Expressive, storytelling
- `onyx` - Deep, authoritative
- `shimmer` - Bright, energetic

**Pros:** Very fast, small model, good quality
**Cons:** Limited voice customization

### pocket (NEW)

100M parameter model from Kyutai Labs. CPU-friendly with voice cloning support.

```yaml
tts:
  backend: pocket
  voice: default   # or path to cloned voice
  speed: 1.0
```

**Installation:**
```bash
uv sync --extra pocket
# or: uv pip install pocket-tts
```

**Voice Cloning:**
```python
from claudio import tts

engine = tts.Pocket()

# Clone voice from audio file
voice = engine.clone_voice("speaker.wav", name="my-voice")

# Use cloned voice
engine.speak("Hello in cloned voice", voice="my-voice")
```

Cloned voices are stored in `~/.pocket-tts/voices/`.

**Pros:** CPU-friendly, voice cloning, small model
**Cons:** Newer, less tested

### qwen3

1.7B parameter model with excellent quality.

```yaml
tts:
  backend: qwen3
  voice: default
  speed: 1.0
```

**Installation:**
```bash
uv sync --extra qwen3
# or: uv pip install "mlx-audio>=0.3.1"
```

**Languages:** Chinese, English, Japanese, Korean, and more

**Pros:** Best quality, multilingual
**Cons:** Large model, requires GPU for good speed

## Auto Backend Selection

When `backend: auto`, Claudio selects based on availability:

1. Check for GPU → prefer `kokoro` or `qwen3`
2. Check for `soprano` installation
3. Check for `pocket` installation
4. macOS → use `say`
5. Fallback → error

## Configuration

### Via YAML

```yaml
tts:
  backend: kokoro
  voice: nova
  speed: 1.1
  device: null  # Audio device ID
```

### Via CLI

```bash
claudio proxy --tts kokoro --voice nova --speed 1.1
```

### Via Code

```python
from claudio import tts

config = tts.Config(
    backend='kokoro',
    voice='nova',
    rate=1.1,
    output_device=None,
)
engine = tts.create(config)
```

## API Reference

### TTS Base Class

All backends implement:

```python
class TTS:
    def speak(self, text: str, voice: str = None, rate: float = 1.0) -> None:
        """Speak text through audio output."""

    def generate(self, text: str, voice: str = None, rate: float = 1.0) -> bytes:
        """Generate audio bytes (WAV format)."""

    def voices(self) -> list[dict]:
        """List available voices."""

    def play(self, audio_bytes: bytes) -> None:
        """Play audio bytes."""
```

### Pocket-specific

```python
class Pocket(TTS):
    def clone_voice(self, audio_path: str, name: str = None) -> dict:
        """Clone voice from audio file.

        Args:
            audio_path: Path to WAV file with speaker audio
            name: Name for the cloned voice (auto-generated if None)

        Returns:
            Voice info dict with 'name', 'path', 'source'
        """
```

## Performance Tips

### Reduce Latency

1. **Lazy loading:** Models load on first use
2. **Keep engine alive:** Reuse the TTS instance
3. **Use streaming:** The proxy generates while playing

### Reduce Memory

1. Use smaller backends (`say`, `pocket`, `kokoro`)
2. Unload when not needed: `del engine`

### GPU vs CPU

| Backend | GPU Speedup |
|---------|-------------|
| kokoro | 2-3x |
| soprano | 2-3x |
| qwen3 | 5-10x (required for real-time) |
| pocket | None (CPU-optimized) |
| say | N/A |

## Troubleshooting

### No audio output

```python
# List audio devices
import sounddevice as sd
print(sd.query_devices())

# Specify device
config = tts.Config(output_device=2)
```

### Model download fails

```bash
# Manual download for kokoro
python -c "import kokoro; kokoro.download_model()"

# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--*kokoro*
```

### Slow startup

Models are loaded lazily on first use. Pre-warm in your application:

```python
engine = tts.create(tts.Config(backend='kokoro'))
engine.speak("")  # Trigger model load
```
