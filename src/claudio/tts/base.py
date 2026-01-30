"""TTS base class and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

import msgspec

if TYPE_CHECKING:
  from .outputs import Sink


class Config(msgspec.Struct, kw_only=True):
  """TTS configuration."""
  backend: str = 'kokoro'
  voice: str | None = 'af_heart'
  rate: float = 1.1  # Slightly faster default
  output_device: int | str | None = None
  # Soprano-specific
  cache_size_mb: int = 100
  temperature: float = 0.3
  top_p: float = 0.95


class TTS(ABC):
  """Base TTS interface."""

  def __init__(self, config: Config | None = None):
    self.config = config or Config()

  @classmethod
  def available(cls) -> bool:
    """Check if this backend's dependencies are installed."""
    return True

  @classmethod
  def list_voices(cls) -> list[dict]:
    """List voices without instantiating. Subclasses override."""
    return []

  def download_model(self) -> None:
    """Download model weights. No-op for system backends."""
    pass

  @abstractmethod
  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    """Speak text (blocking)."""

  @abstractmethod
  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    """Generate audio bytes."""

  def play(self, audio: bytes) -> None:
    """Play WAV audio bytes via sounddevice."""
    if not audio:
      return
    import struct
    import numpy as np
    import sounddevice as sd

    if audio[:4] != b'RIFF':
      raise ValueError('Not a WAV file')

    # Parse sample rate from fmt chunk
    fmt_pos = audio.find(b'fmt ')
    sample_rate = struct.unpack('<I', audio[fmt_pos + 12:fmt_pos + 16])[0]

    # Find data chunk
    pos = 12
    while pos < len(audio) - 8:
      chunk_id = audio[pos:pos + 4]
      chunk_size = struct.unpack('<I', audio[pos + 4:pos + 8])[0]
      if chunk_id == b'data':
        audio_data = audio[pos + 8:pos + 8 + chunk_size]
        break
      pos += 8 + chunk_size
    else:
      raise ValueError('No data chunk in WAV')

    samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    samples *= (1.0 / 32767)

    if self.config.output_device is not None:
      sd.default.device = (sd.default.device[0], self.config.output_device)

    # Resample to device rate if needed
    device_info = sd.query_devices(sd.default.device[1])
    device_rate = int(device_info['default_samplerate'])
    if sample_rate != device_rate:
      duration = len(samples) / sample_rate
      new_length = int(duration * device_rate)
      indices = np.linspace(0, len(samples) - 1, new_length)
      samples = np.interp(indices, np.arange(len(samples)), samples)
      sample_rate = device_rate

    sd.play(samples, samplerate=sample_rate, blocking=True)

  def stream(self, text: str, voice: str | None = None, rate: float = 1.0) -> Iterator[bytes]:
    """Stream audio chunks."""
    yield self.generate(text, voice, rate)

  def to(self, sink: 'Sink', text: str, voice: str | None = None, rate: float = 1.0) -> None:
    """Generate audio and write to sink."""
    audio = self.generate(text, voice, rate)
    sink.write(audio)

  def __repr__(self) -> str:
    name = type(self).__name__.lower()
    voice = self.config.voice or 'default'
    return f'{name}:{voice}'

  @abstractmethod
  def voices(self) -> list[dict]:
    """List available voices."""
