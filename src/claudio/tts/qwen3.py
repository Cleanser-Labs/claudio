"""Qwen3-TTS neural TTS backend (MLX)."""

from __future__ import annotations

from typing import Iterator

from .base import TTS, Config


class Qwen3(TTS):
  """Qwen3-TTS via mlx-audio.

  Supports multiple model variants:
  - base: Fast, uses predefined speaker names (Chelsie, Ethan, etc.)
  - voicedesign: Slower, creates voices from text descriptions
  """

  # Model variants (smaller = faster)
  MODELS = {
    'base-6bit': 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit',  # Fastest
    'base': 'mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16',
    'custom-5bit': 'mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-5bit',
    'voicedesign': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',  # Slowest
  }

  # Default speakers for base model
  SPEAKERS = ['Chelsie', 'Ethan', 'Vivian', 'Bella', 'Lucas']

  @classmethod
  def available(cls) -> bool:
    try:
      import mlx_audio  # noqa: F401
      return True
    except ImportError:
      return False

  @classmethod
  def list_voices(cls) -> list[dict]:
    return [
      {'name': s, 'id': s, 'lang': 'en', 'quality': 'neural'}
      for s in cls.SPEAKERS
    ]

  def download_model(self) -> None:
    from mlx_audio.tts.utils import load_model
    model_id = self.MODELS.get(self._model_variant, self.MODELS['base-6bit'])
    load_model(model_id)

  def __init__(self, config: Config | None = None, model: str = 'base-6bit'):
    super().__init__(config)
    self._model = None
    self._model_variant = model

  def __repr__(self) -> str:
    voice = self.config.voice or 'Chelsie'
    return f'qwen3:{voice} (mlx, {self._model_variant})'

  def _get_model(self):
    if self._model is None:
      try:
        from mlx_audio.tts.utils import load_model
      except ImportError:
        raise ImportError('mlx-audio>=0.3.1 not installed. Run: uv sync --extra qwen3')

      model_id = self.MODELS.get(self._model_variant, self.MODELS['base'])
      self._model = load_model(model_id)
    return self._model


  def _generate_audio(self, text: str, voice: str | None = None):
    """Generate audio using the appropriate model API."""
    import numpy as np

    model = self._get_model()

    # Estimate max_tokens based on text length (roughly 10 tokens per word)
    # Add some headroom but avoid excessive generation
    word_count = len(text.split())
    max_tokens = min(max(256, word_count * 50), 2048)

    if self._model_variant == 'voicedesign':
      # VoiceDesign: voice is a text description
      instruct = voice or self.config.voice or 'A calm, clear, professional voice.'
      results = list(model.generate_voice_design(
        text=text,
        language='English',
        instruct=instruct,
        max_tokens=max_tokens,
      ))
    else:
      # Base model: voice is a speaker name
      speaker = voice or self.config.voice or 'Chelsie'
      results = list(model.generate(
        text=text,
        voice=speaker,
        max_tokens=max_tokens,
      ))

    if not results:
      return None

    # Collect audio chunks
    chunks = [np.asarray(r.audio).flatten() for r in results]
    return np.concatenate(chunks)

  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    import sounddevice as sd
    import numpy as np

    audio = self._generate_audio(text, voice)
    if audio is None:
      return

    # Qwen3-TTS uses 24000 Hz sample rate
    sample_rate = 24000

    # Apply rate adjustment by resampling
    effective_rate = rate * self.config.rate
    if effective_rate != 1.0:
      new_length = int(len(audio) / effective_rate)
      indices = np.linspace(0, len(audio) - 1, new_length)
      audio = np.interp(indices, np.arange(len(audio)), audio)

    sd.play(audio, samplerate=sample_rate, blocking=True)

  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    import numpy as np

    text = text.strip()
    if not text or not any(c.isalnum() for c in text):
      return b''

    audio = self._generate_audio(text, voice)
    if audio is None:
      return b''

    # Apply rate adjustment
    effective_rate = rate * self.config.rate
    if effective_rate != 1.0:
      new_length = int(len(audio) / effective_rate)
      indices = np.linspace(0, len(audio) - 1, new_length)
      audio = np.interp(indices, np.arange(len(audio)), audio)

    # Convert to WAV (24kHz)
    return self._to_wav(audio, 24000)

  def stream(self, text: str, voice: str | None = None, rate: float = 1.0) -> Iterator[bytes]:
    model = self._get_model()

    word_count = len(text.split())
    max_tokens = min(max(256, word_count * 50), 2048)

    if self._model_variant == 'voicedesign':
      instruct = voice or self.config.voice or 'A calm, clear, professional voice.'
      gen = model.generate_voice_design(text=text, language='English', instruct=instruct, max_tokens=max_tokens)
    else:
      speaker = voice or self.config.voice or 'Chelsie'
      gen = model.generate(text=text, voice=speaker, max_tokens=max_tokens)

    for result in gen:
      yield result.audio.tobytes()

  def voices(self) -> list[dict]:
    """Return available voices (speakers for base, descriptions for voicedesign)."""
    if self._model_variant == 'voicedesign':
      return [
        {'name': 'neutral', 'id': 'A calm, clear, professional voice with natural pacing.', 'lang': 'en', 'quality': 'neural'},
        {'name': 'friendly', 'id': 'A warm, friendly voice with a slight smile in the tone.', 'lang': 'en', 'quality': 'neural'},
        {'name': 'narrator', 'id': 'A deep, authoritative male narrator voice, like a documentary.', 'lang': 'en', 'quality': 'neural'},
        {'name': 'assistant', 'id': 'A helpful, clear female assistant voice with natural inflection.', 'lang': 'en', 'quality': 'neural'},
      ]
    else:
      return self.list_voices()

  def _to_wav(self, audio, sample_rate: int) -> bytes:
    """Convert numpy audio to WAV bytes."""
    import struct
    import numpy as np

    audio = np.clip(audio, -1, 1)
    audio_16bit = (audio * 32767).astype(np.int16)
    audio_bytes = audio_16bit.tobytes()

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio_bytes)

    header = struct.pack(
      '<4sI4s4sIHHIIHH4sI',
      b'RIFF',
      36 + data_size,
      b'WAVE',
      b'fmt ',
      16,
      1,
      num_channels,
      sample_rate,
      byte_rate,
      block_align,
      bits_per_sample,
      b'data',
      data_size,
    )
    return header + audio_bytes
