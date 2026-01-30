"""Wake word trigger - voice-activated trigger like 'Hey Siri'."""

from __future__ import annotations

import threading
from typing import Literal

import msgspec
import numpy as np

from .base import Trigger, TriggerConfig


class WakeWordConfig(TriggerConfig, kw_only=True):
  """Wake word trigger configuration."""
  # Backend: porcupine (commercial, lightweight) or openwakeword (open source)
  backend: Literal['porcupine', 'openwakeword'] = 'openwakeword'

  # Porcupine settings
  porcupine_access_key: str | None = None
  # Built-in keywords: alexa, americano, blueberry, bumblebee, computer,
  # grapefruit, grasshopper, hey google, hey siri, jarvis, ok google,
  # picovoice, porcupine, terminator
  porcupine_keyword: str = 'jarvis'

  # OpenWakeWord settings
  openwakeword_model: str = 'hey_jarvis'
  openwakeword_threshold: float = 0.5

  # Audio settings
  device: int | None = None
  sample_rate: int = 16000


class WakeWordTrigger(Trigger):
  """Wake word detection trigger.

  Continuously listens for a wake word using minimal resources.
  When detected, fires activate callback to start dictation.
  """

  def __init__(self, config: WakeWordConfig | None = None):
    super().__init__(config)
    self.config: WakeWordConfig = config or WakeWordConfig()
    self._thread: threading.Thread | None = None
    self._stream = None

  def start(self) -> None:
    """Start listening for wake word."""
    if self._running:
      return

    self._running = True
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self) -> None:
    """Stop listening for wake word."""
    self._running = False
    if self._stream:
      self._stream.stop()
      self._stream = None
    if self._thread:
      self._thread.join(timeout=2.0)
      self._thread = None

  def _run(self):
    """Main wake word detection loop."""
    if self.config.backend == 'porcupine':
      self._run_porcupine()
    else:
      self._run_openwakeword()

  def _run_porcupine(self):
    """Run Porcupine wake word detection."""
    import pvporcupine
    import sounddevice as sd

    if not self.config.porcupine_access_key:
      raise ValueError('Porcupine requires access_key. Get one at https://picovoice.ai/')

    porcupine = pvporcupine.create(
      access_key=self.config.porcupine_access_key,
      keywords=[self.config.porcupine_keyword],
    )

    try:
      def callback(indata, frames, time_info, status):
        if not self._running:
          return
        # Convert to int16 for Porcupine
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        if porcupine.process(pcm) >= 0:
          self.activate()

      self._stream = sd.InputStream(
        samplerate=porcupine.sample_rate,
        channels=1,
        dtype='float32',
        blocksize=porcupine.frame_length,
        device=self.config.device,
        callback=callback,
      )
      self._stream.start()

      # Keep thread alive while running
      while self._running:
        sd.sleep(100)

    finally:
      porcupine.delete()

  def _run_openwakeword(self):
    """Run OpenWakeWord detection."""
    import sounddevice as sd
    from openwakeword.model import Model

    # Load model
    model = Model(wakeword_models=[self.config.openwakeword_model])
    chunk_size = 1280  # ~80ms at 16kHz

    try:
      def callback(indata, frames, time_info, status):
        if not self._running:
          return
        # OpenWakeWord expects int16
        audio = (indata[:, 0] * 32767).astype(np.int16)
        prediction = model.predict(audio)

        # Check if wake word detected
        score = prediction.get(self.config.openwakeword_model, 0)
        if score > self.config.openwakeword_threshold:
          self.activate()
          # Reset to avoid repeated triggers
          model.reset()

      self._stream = sd.InputStream(
        samplerate=self.config.sample_rate,
        channels=1,
        dtype='float32',
        blocksize=chunk_size,
        device=self.config.device,
        callback=callback,
      )
      self._stream.start()

      # Keep thread alive while running
      while self._running:
        sd.sleep(100)

    finally:
      pass  # Model cleanup if needed
