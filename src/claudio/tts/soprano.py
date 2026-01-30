"""Soprano neural TTS backend."""

from __future__ import annotations

from typing import Iterator

from .base import TTS, Config


class Soprano(TTS):
  """Soprano neural TTS."""

  @classmethod
  def available(cls) -> bool:
    try:
      import soprano  # noqa: F401
      return True
    except ImportError:
      return False

  @classmethod
  def list_voices(cls) -> list[dict]:
    return [{'name': 'default', 'id': 'default', 'lang': 'en'}]

  def __init__(self, config: Config | None = None):
    super().__init__(config)
    self._model = None

  def _get_model(self):
    if self._model is None:
      try:
        from soprano import SopranoTTS
      except ImportError:
        raise ImportError('soprano-tts not installed. Run: uv sync --extra soprano')
      self._model = SopranoTTS(
        backend='auto',
        device='auto',
        cache_size_mb=self.config.cache_size_mb,
      )
    return self._model

  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    import sounddevice as sd
    from soprano.utils.streaming import play_stream

    if self.config.output_device is not None:
      sd.default.device = (sd.default.device[0], self.config.output_device)

    stream = self._get_model().infer_stream(
      text,
      chunk_size=1,
      temperature=self.config.temperature,
      top_p=self.config.top_p,
    )
    play_stream(stream)

  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    import numpy as np

    chunks = list(self._get_model().infer_stream(
      text,
      chunk_size=1,
      temperature=self.config.temperature,
      top_p=self.config.top_p,
    ))
    return np.concatenate(chunks).tobytes()

  def stream(self, text: str, voice: str | None = None, rate: float = 1.0) -> Iterator[bytes]:
    for chunk in self._get_model().infer_stream(
      text,
      chunk_size=1,
      temperature=self.config.temperature,
      top_p=self.config.top_p,
    ):
      yield chunk.tobytes()

  def voices(self) -> list[dict]:
    return self.list_voices()
