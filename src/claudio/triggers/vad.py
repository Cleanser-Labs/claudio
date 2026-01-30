"""Voice activity detection trigger - activates when speech detected."""

from __future__ import annotations

import threading
import time

import msgspec
import numpy as np

from .base import Trigger, TriggerConfig


class VADConfig(TriggerConfig, kw_only=True):
  """Voice activity detection trigger configuration."""
  # Detection thresholds
  activation_threshold: float = 0.02  # RMS level to start
  deactivation_threshold: float = 0.01  # RMS level to stop

  # Timing
  min_speech_duration: float = 0.3  # Minimum speech to activate (seconds)
  silence_duration: float = 1.5  # Silence duration to deactivate (seconds)

  # Audio settings
  device: int | None = None
  sample_rate: int = 16000
  chunk_duration: float = 0.1  # Audio chunk size in seconds


class VADTrigger(Trigger):
  """Voice activity detection trigger.

  Automatically activates when speech is detected and deactivates
  after a period of silence. No explicit trigger needed.

  Uses simple energy-based detection. For more accurate detection,
  consider using Silero VAD or WebRTC VAD.
  """

  def __init__(self, config: VADConfig | None = None):
    super().__init__(config)
    self.config: VADConfig = config or VADConfig()
    self._thread: threading.Thread | None = None
    self._stream = None
    self._active = False

  def start(self) -> None:
    """Start voice activity detection."""
    if self._running:
      return

    self._running = True
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    """Stop voice activity detection."""
    self._running = False
    if self._stream:
      self._stream.stop()
      self._stream = None
    if self._thread:
      self._thread.join(timeout=2.0)
      self._thread = None
    self._active = False

  def _run(self):
    """Main VAD loop."""
    import sounddevice as sd

    chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)

    # State tracking
    speech_start: float | None = None
    silence_start: float | None = None

    def callback(indata, frames, time_info, status):
      nonlocal speech_start, silence_start

      if not self._running:
        return

      # Calculate RMS energy
      rms = np.sqrt(np.mean(indata ** 2))
      now = time.time()

      if not self._active:
        # Not active - looking for speech
        if rms > self.config.activation_threshold:
          if speech_start is None:
            speech_start = now
          elif now - speech_start > self.config.min_speech_duration:
            # Enough speech detected - activate
            self._active = True
            speech_start = None
            silence_start = None
            self.activate()
        else:
          speech_start = None
      else:
        # Active - looking for silence
        if rms < self.config.deactivation_threshold:
          if silence_start is None:
            silence_start = now
          elif now - silence_start > self.config.silence_duration:
            # Enough silence - deactivate
            self._active = False
            silence_start = None
            self.deactivate()
        else:
          silence_start = None

    self._stream = sd.InputStream(
      samplerate=self.config.sample_rate,
      channels=1,
      dtype='float32',
      blocksize=chunk_samples,
      device=self.config.device,
      callback=callback,
    )
    self._stream.start()

    # Keep thread alive while running
    while self._running:
      sd.sleep(100)
