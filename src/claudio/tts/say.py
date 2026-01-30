"""macOS `say` command backend."""

from __future__ import annotations

import os
import subprocess
import tempfile

from .base import TTS, Config


class Say(TTS):
  """macOS `say` command."""

  _voice_cache = None

  @classmethod
  def available(cls) -> bool:
    import sys
    return sys.platform == 'darwin'

  @classmethod
  def list_voices(cls) -> list[dict]:
    if cls._voice_cache is not None:
      return cls._voice_cache
    import re
    import subprocess
    try:
      result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True)
    except FileNotFoundError:
      cls._voice_cache = []
      return cls._voice_cache
    voices = []
    for line in result.stdout.strip().split('\n'):
      match = re.match(r'^(.+?)\s+([a-z]{2}_[A-Z]{2})\s+#\s*(.*)$', line.strip())
      if match:
        name, lang, sample = match.groups()
        voices.append({
          'name': name.strip(),
          'id': name.strip(),
          'lang': lang,
          'sample': sample,
        })
    cls._voice_cache = voices
    return cls._voice_cache

  def __init__(self, config: Config | None = None):
    super().__init__(config)
    self._base_rate = 175  # words per minute

  def _rate(self, rate: float) -> int:
    return int(self._base_rate * rate * self.config.rate)

  def _voice_args(self, voice: str | None) -> list[str]:
    v = voice or self.config.voice
    return ['-v', v] if v else []

  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    cmd = ['say', *self._voice_args(voice), '-r', str(self._rate(rate)), text]
    subprocess.run(cmd, check=True, capture_output=True)

  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    fd, path = tempfile.mkstemp(suffix='.aiff', prefix='claudio_')
    os.close(fd)
    try:
      cmd = ['say', '-o', path, *self._voice_args(voice), '-r', str(self._rate(rate)), text]
      subprocess.run(cmd, check=True, capture_output=True)
      with open(path, 'rb') as f:
        return f.read()
    finally:
      os.unlink(path)

  def voices(self) -> list[dict]:
    return self.list_voices()
