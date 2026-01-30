"""Kokoro neural TTS backend (MLX)."""

from __future__ import annotations

from typing import Iterator

from .base import TTS, Config


class Kokoro(TTS):
  """Kokoro TTS via mlx-audio."""

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
      {'name': 'af_heart', 'id': 'af_heart', 'lang': 'en', 'quality': 'neural'},
      {'name': 'af_bella', 'id': 'af_bella', 'lang': 'en', 'quality': 'neural'},
      {'name': 'af_nicole', 'id': 'af_nicole', 'lang': 'en', 'quality': 'neural'},
      {'name': 'af_sarah', 'id': 'af_sarah', 'lang': 'en', 'quality': 'neural'},
      {'name': 'af_sky', 'id': 'af_sky', 'lang': 'en', 'quality': 'neural'},
      {'name': 'am_adam', 'id': 'am_adam', 'lang': 'en', 'quality': 'neural'},
      {'name': 'am_michael', 'id': 'am_michael', 'lang': 'en', 'quality': 'neural'},
      {'name': 'bf_emma', 'id': 'bf_emma', 'lang': 'en-gb', 'quality': 'neural'},
      {'name': 'bf_isabella', 'id': 'bf_isabella', 'lang': 'en-gb', 'quality': 'neural'},
      {'name': 'bm_george', 'id': 'bm_george', 'lang': 'en-gb', 'quality': 'neural'},
      {'name': 'bm_lewis', 'id': 'bm_lewis', 'lang': 'en-gb', 'quality': 'neural'},
    ]

  def download_model(self) -> None:
    from mlx_audio.tts.utils import load_model
    load_model('mlx-community/Kokoro-82M-bf16')

  def __init__(self, config: Config | None = None):
    super().__init__(config)
    self._model = None
    self._pipeline = None

  def __repr__(self) -> str:
    voice = self.config.voice or 'af_heart'
    return f'kokoro:{voice} (mlx)'

  def _get_pipeline(self):
    if self._pipeline is None:
      try:
        from mlx_audio.tts.models.kokoro import KokoroPipeline
        from mlx_audio.tts.utils import load_model
      except ImportError:
        raise ImportError('mlx-audio not installed. Run: uv sync --extra kokoro')

      model_id = 'mlx-community/Kokoro-82M-bf16'
      self._model = load_model(model_id)
      self._pipeline = KokoroPipeline(lang_code='a', model=self._model, repo_id=model_id)
    return self._pipeline

  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    import sounddevice as sd
    import numpy as np

    pipeline = self._get_pipeline()
    v = voice or self.config.voice or 'af_heart'

    # Collect audio chunks
    chunks = []
    for _, _, audio in pipeline(text, voice=v, speed=rate * self.config.rate):
      chunk = audio[0] if isinstance(audio, tuple) else audio
      chunks.append(np.asarray(chunk).flatten())

    # Play (mono, float32)
    audio = np.concatenate(chunks)
    sd.play(audio, samplerate=24000, blocking=True)

  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    import numpy as np

    # Skip empty or non-pronounceable text
    text = text.strip()
    if not text or not any(c.isalnum() for c in text):
      return b''

    pipeline = self._get_pipeline()
    v = voice or self.config.voice or 'af_heart'

    # Collect audio
    chunks = []
    for _, _, audio in pipeline(text, voice=v, speed=rate * self.config.rate):
      chunks.append(audio[0] if isinstance(audio, tuple) else audio)

    if not chunks:
      return b''

    audio = np.concatenate(chunks)

    # Convert to WAV bytes
    return self._to_wav(audio, 24000)

  def stream(self, text: str, voice: str | None = None, rate: float = 1.0) -> Iterator[bytes]:
    pipeline = self._get_pipeline()
    v = voice or self.config.voice or 'af_heart'

    for _, _, audio in pipeline(text, voice=v, speed=rate * self.config.rate):
      chunk = audio[0] if isinstance(audio, tuple) else audio
      yield chunk.tobytes()

  def voices(self) -> list[dict]:
    return self.list_voices()

  def _to_wav(self, audio, sample_rate: int) -> bytes:
    """Convert numpy audio to WAV bytes."""
    import struct
    import numpy as np

    # Normalize and convert to 16-bit PCM
    audio = np.clip(audio, -1, 1)
    audio_16bit = (audio * 32767).astype(np.int16)
    audio_bytes = audio_16bit.tobytes()

    # WAV header
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
      16,  # fmt chunk size
      1,   # PCM format
      num_channels,
      sample_rate,
      byte_rate,
      block_align,
      bits_per_sample,
      b'data',
      data_size,
    )
    return header + audio_bytes
