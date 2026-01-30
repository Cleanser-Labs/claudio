"""Claudio - Voice I/O for Claude Code.

Cleanser Labs Audio - speak and listen for Claude.
"""

from claudio import tts
from claudio.tts import TTS, Config, Say, speak

__all__ = [
  'tts',
  'TTS',
  'Config',
  'Say',
  'speak',
]

# Optional ASR (requires claudio[asr])
try:
  from claudio.asr import ASREngine, ASRConfig, ASRDaemon
  __all__.extend(['ASREngine', 'ASRConfig', 'ASRDaemon'])
except ImportError:
  pass
