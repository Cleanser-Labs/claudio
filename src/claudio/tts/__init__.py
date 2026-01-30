"""Text-to-speech with pluggable backends."""

from __future__ import annotations

import sys
from pathlib import Path

from .base import TTS, Config
from .say import Say
from . import inputs
from . import outputs

__all__ = [
  'TTS',
  'Config',
  'Say',
  'AVFoundation',
  'Soprano',
  'Kokoro',
  'Qwen3',
  'Pocket',
  'inputs',
  'outputs',
  'create',
  'get',
  'speak',
  'available_backends',
  'all_backends',
]


# --- Lightweight availability checks (no heavy imports) ---

def _check_available(name: str) -> bool:
  """Check if backend deps are installed without importing heavy modules."""
  if name == 'say':
    return sys.platform == 'darwin'
  if name == 'avfoundation':
    try:
      import AVFoundation  # noqa: F401
      return True
    except ImportError:
      return False
  if name in ('kokoro', 'qwen3'):
    try:
      import mlx_audio  # noqa: F401
      return True
    except ImportError:
      return False
  if name == 'soprano':
    try:
      import soprano  # noqa: F401
      return True
    except ImportError:
      return False
  if name == 'pocket':
    try:
      import pocket_tts  # noqa: F401
      return True
    except ImportError:
      pass
    try:
      import mlx_audio  # noqa: F401
      return True
    except ImportError:
      return False
  return False


# --- Static voice lists (no module imports needed) ---

_KOKORO_VOICES = [
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

_QWEN3_SPEAKERS = ['Chelsie', 'Ethan', 'Vivian', 'Bella', 'Lucas']

_POCKET_SPEAKERS = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']


def _list_voices_static(name: str) -> list[dict]:
  """Return voice list without importing backend modules."""
  if name == 'kokoro':
    return list(_KOKORO_VOICES)
  if name == 'qwen3':
    return [{'name': s, 'id': s, 'lang': 'en', 'quality': 'neural'} for s in _QWEN3_SPEAKERS]
  if name == 'pocket':
    voices = [{'name': s, 'id': s, 'lang': 'en', 'quality': 'neural'} for s in _POCKET_SPEAKERS]
    clone_dir = Path.home() / '.pocket-tts' / 'voices'
    if clone_dir.exists():
      for wav_file in clone_dir.glob('*.wav'):
        voices.append({'name': wav_file.stem, 'id': str(wav_file), 'lang': 'en', 'quality': 'cloned'})
    return voices
  if name == 'soprano':
    return [{'name': 'default', 'id': 'default', 'lang': 'en'}]
  if name == 'say':
    return Say.list_voices()
  if name == 'avfoundation':
    try:
      from .avfoundation import AVFoundation as _AVF
      return _AVF.list_voices()
    except ImportError:
      return []
  return []


# --- Backend registry ---

# name -> (module, class_name)
BACKENDS = {
  'kokoro': ('kokoro', 'Kokoro'),
  'pocket': ('pocket', 'Pocket'),
  'qwen3': ('qwen3', 'Qwen3'),
  'soprano': ('soprano', 'Soprano'),
  'say': ('say', 'Say'),
  'avfoundation': ('avfoundation', 'AVFoundation'),
}


class _BackendInfo:
  """Lightweight proxy that exposes available() and list_voices() without importing the backend."""

  def __init__(self, name: str):
    self.name = name
    self._cls = None

  def available(self) -> bool:
    return _check_available(self.name)

  def list_voices(self) -> list[dict]:
    return _list_voices_static(self.name)

  def get_class(self) -> type[TTS]:
    if self._cls is None:
      import importlib
      module_name, class_name = BACKENDS[self.name]
      mod = importlib.import_module(f'.{module_name}', package=__package__)
      self._cls = getattr(mod, class_name)
    return self._cls


# Cached backend info objects
_backend_info: dict[str, _BackendInfo] = {}


def _get_info(name: str) -> _BackendInfo:
  if name not in _backend_info:
    _backend_info[name] = _BackendInfo(name)
  return _backend_info[name]


def all_backends() -> dict[str, _BackendInfo]:
  """Return {name: info} for all backends (fast, no heavy imports)."""
  return {name: _get_info(name) for name in BACKENDS}


def available_backends() -> dict[str, _BackendInfo]:
  """Return {name: info} for installed backends (fast, no heavy imports)."""
  return {name: info for name, info in all_backends().items() if info.available()}


def _get_backend_class(name: str) -> type[TTS]:
  """Import and return a backend class by name."""
  return _get_info(name).get_class()


# --- Lazy constructors ---

def AVFoundation(config: Config | None = None):
  from .avfoundation import AVFoundation as _AVFoundation
  return _AVFoundation(config)


def Soprano(config: Config | None = None):
  from .soprano import Soprano as _Soprano
  return _Soprano(config)


def Kokoro(config: Config | None = None):
  from .kokoro import Kokoro as _Kokoro
  return _Kokoro(config)


def Qwen3(config: Config | None = None):
  from .qwen3 import Qwen3 as _Qwen3
  return _Qwen3(config)


def Pocket(config: Config | None = None):
  from .pocket import Pocket as _Pocket
  return _Pocket(config)


def create(config: Config | None = None) -> TTS:
  """Create TTS instance based on config."""
  config = config or Config()
  backend = config.backend

  if backend == 'auto':
    backend = 'pocket' if _check_available('pocket') else 'kokoro'

  if backend == 'say':
    return Say(config)
  elif backend == 'avfoundation':
    return AVFoundation(config)
  elif backend == 'soprano':
    return Soprano(config)
  elif backend == 'kokoro':
    return Kokoro(config)
  elif backend == 'qwen3':
    return Qwen3(config)
  elif backend == 'pocket':
    return Pocket(config)
  else:
    raise ValueError(f'Unknown backend: {backend}')


# Module-level singleton
_tts: TTS | None = None


def get(config: Config | None = None) -> TTS:
  """Get or create default TTS instance."""
  global _tts
  if _tts is None:
    _tts = create(config)
  return _tts


def speak(text: str, voice: str | None = None, rate: float = 1.0) -> None:
  """Speak text using default TTS."""
  get().speak(text, voice, rate)
