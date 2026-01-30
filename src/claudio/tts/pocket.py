"""Kyutai Pocket-TTS backend - lightweight TTS with voice cloning.

Supports two backends:
- pocket (torch, via pocket_tts.TTSModel) - CPU
- pocket-mlx (via mlx_audio, model mlx-community/pocket-tts-8bit) - Apple Silicon
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Iterator

from .base import TTS, Config


# Built-in speaker presets
SPEAKERS = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']


def _has_pocket_tts() -> bool:
  try:
    import pocket_tts  # noqa: F401
    return True
  except ImportError:
    return False


def _has_mlx_audio() -> bool:
  try:
    import mlx_audio  # noqa: F401
    return True
  except ImportError:
    return False


class Pocket(TTS):
  """Pocket-TTS - 100M param TTS from Kyutai Labs.

  Supports torch (CPU) and mlx (Apple Silicon) backends.
  Install: uv sync --extra pocket
  """

  @classmethod
  def available(cls) -> bool:
    return _has_pocket_tts() or _has_mlx_audio()

  @classmethod
  def list_voices(cls) -> list[dict]:
    voices = [
      {'name': s, 'id': s, 'lang': 'en', 'quality': 'neural'}
      for s in SPEAKERS
    ]
    # Check for cloned voices
    clone_dir = Path.home() / '.pocket-tts' / 'voices'
    if clone_dir.exists():
      for wav_file in clone_dir.glob('*.wav'):
        voices.append({
          'name': wav_file.stem,
          'id': str(wav_file),
          'lang': 'en',
          'quality': 'cloned',
        })
    return voices

  def download_model(self) -> None:
    if self._backend_name == 'mlx':
      from mlx_audio.tts.utils import load_model
      load_model('mlx-community/pocket-tts-8bit')

  def __init__(self, config: Config | None = None, backend: str = 'auto'):
    super().__init__(config)
    self._model = None
    self._sample_rate = 24000

    # Resolve backend
    if backend == 'auto':
      self._backend_name = 'mlx' if _has_mlx_audio() else 'torch'
    else:
      self._backend_name = backend

  def __repr__(self) -> str:
    voice = self.config.voice or 'default'
    return f'pocket:{voice} ({self._backend_name})'

  def _get_model(self):
    if self._model is None:
      if self._backend_name == 'mlx':
        try:
          from mlx_audio.tts.utils import load_model
        except ImportError:
          raise ImportError('mlx-audio not installed. Run: uv sync --extra pocket')
        self._model = load_model('mlx-community/pocket-tts-8bit')
      else:
        try:
          from pocket_tts import PocketTTS
        except ImportError:
          raise ImportError('pocket-tts not installed. Run: uv sync --extra pocket')
        self._model = PocketTTS()
    return self._model

  def _audio_to_wav(self, audio, sample_rate: int) -> bytes:
    import numpy as np
    if audio.dtype != np.int16:
      if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
      else:
        audio = audio.astype(np.int16)

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(audio) * bits_per_sample // 8

    wav = io.BytesIO()
    wav.write(b'RIFF')
    wav.write(struct.pack('<I', 36 + data_size))
    wav.write(b'WAVE')
    wav.write(b'fmt ')
    wav.write(struct.pack('<I', 16))
    wav.write(struct.pack('<H', 1))
    wav.write(struct.pack('<H', num_channels))
    wav.write(struct.pack('<I', sample_rate))
    wav.write(struct.pack('<I', byte_rate))
    wav.write(struct.pack('<H', block_align))
    wav.write(struct.pack('<H', bits_per_sample))
    wav.write(b'data')
    wav.write(struct.pack('<I', data_size))
    wav.write(audio.tobytes())
    return wav.getvalue()

  def _synthesize(self, text: str, voice: str | None = None, rate: float = 1.0):
    import numpy as np
    model = self._get_model()
    voice = voice or self.config.voice

    if self._backend_name == 'mlx':
      # mlx_audio generate API
      results = list(model.generate(text=text, voice=voice or SPEAKERS[0]))
      if not results:
        return np.array([], dtype=np.float32)
      chunks = [np.asarray(r.audio).flatten() for r in results]
      audio = np.concatenate(chunks)
    else:
      # torch pocket_tts API
      if voice and Path(voice).exists():
        audio = model.synthesize(text, reference_audio=voice)
      elif voice:
        try:
          audio = model.synthesize(text, speaker=voice)
        except (ValueError, TypeError):
          audio = model.synthesize(text)
      else:
        audio = model.synthesize(text)

    # Apply rate adjustment (combine call-site rate with config rate)
    effective_rate = rate * self.config.rate
    if effective_rate != 1.0:
      original_len = len(audio)
      new_len = int(original_len / effective_rate)
      indices = np.linspace(0, original_len - 1, new_len)
      audio = np.interp(indices, np.arange(original_len), audio)

    return audio

  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    import sounddevice as sd

    audio = self._synthesize(text, voice, rate)

    if self.config.output_device is not None:
      sd.default.device = (sd.default.device[0], self.config.output_device)

    sd.play(audio, self._sample_rate)
    sd.wait()

  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    audio = self._synthesize(text, voice, rate)
    return self._audio_to_wav(audio, self._sample_rate)

  def stream(self, text: str, voice: str | None = None, rate: float = 1.0) -> Iterator[bytes]:
    audio = self._synthesize(text, voice, rate)
    chunk_samples = int(self._sample_rate * 0.1)
    for i in range(0, len(audio), chunk_samples):
      chunk = audio[i:i + chunk_samples]
      yield self._audio_to_wav(chunk, self._sample_rate)

  def voices(self) -> list[dict]:
    return self.list_voices()

  def clone_voice(self, audio_path: str | Path, name: str | None = None) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.exists():
      raise FileNotFoundError(f'Audio file not found: {audio_path}')

    name = name or audio_path.stem
    clone_dir = Path.home() / '.pocket-tts' / 'voices'
    clone_dir.mkdir(parents=True, exist_ok=True)

    dest = clone_dir / f'{name}.wav'
    if dest != audio_path:
      import shutil
      shutil.copy2(audio_path, dest)

    return {
      'name': name,
      'id': str(dest),
      'lang': 'en',
      'quality': 'cloned',
    }
