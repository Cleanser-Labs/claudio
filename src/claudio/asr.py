"""Automatic speech recognition using Parakeet-MLX."""

from __future__ import annotations

import threading
import time
from typing import Callable

import msgspec


class ASRConfig(msgspec.Struct, kw_only=True):
  """ASR configuration."""
  model: str = 'mlx-community/parakeet-tdt-0.6b-v3'
  input_device: int | None = None  # None = system default
  window_seconds: float = 10.0
  step_seconds: float = 0.5
  sample_rate: int = 16000
  # Trigger config
  trigger_phrase: str | None = 'send'  # Say this to send transcript
  silence_threshold: float = 0.01  # RMS threshold for silence detection
  silence_duration: float = 1.5  # Seconds of silence before auto-send


class ASREngine:
  """ASR engine using Parakeet-MLX."""

  def __init__(self, config: ASRConfig | None = None):
    self.config = config or ASRConfig()
    self._model = None

  def _load_model(self):
    if self._model is None:
      try:
        from parakeet_mlx import from_pretrained
      except ImportError:
        raise ImportError('parakeet-mlx not installed. Run: uv sync --extra asr')
      print(f'Loading ASR model {self.config.model}...', flush=True)
      self._model = from_pretrained(self.config.model)
      print('ASR model loaded.', flush=True)
    return self._model

  def transcribe(self, audio) -> str:
    """Transcribe audio array to text."""
    import numpy as np
    from parakeet_mlx.audio import get_logmel

    model = self._load_model()

    # Convert to float32 if needed
    if audio.dtype != np.float32:
      audio = audio.astype(np.float32)

    # Get log-mel spectrogram
    mel = get_logmel(audio, self.config.sample_rate)

    # Generate transcription
    result = model.generate(mel)
    return result.strip()


class ASRDaemon:
  """Background daemon for continuous speech recognition."""

  def __init__(
    self,
    config: ASRConfig | None = None,
    on_transcript: Callable[[str], None] | None = None,
    on_partial: Callable[[str], None] | None = None,
  ):
    self.config = config or ASRConfig()
    self.engine = ASREngine(self.config)
    self.on_transcript = on_transcript  # Called when transcript is ready to send
    self.on_partial = on_partial  # Called with partial transcripts

    self._running = False
    self._thread: threading.Thread | None = None
    self._audio_buffer: list = []
    self._buffer_lock = threading.Lock()
    self._last_transcript = ''

  def start(self):
    """Start the ASR daemon."""
    if self._running:
      return

    self._running = True
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

  def stop(self):
    """Stop the ASR daemon."""
    self._running = False
    if self._thread:
      self._thread.join(timeout=2.0)
      self._thread = None

  def send(self):
    """Manually trigger sending the current transcript."""
    if self._last_transcript and self.on_transcript:
      self.on_transcript(self._last_transcript)
      self._last_transcript = ''
      with self._buffer_lock:
        self._audio_buffer.clear()

  def _run(self):
    """Main daemon loop."""
    import numpy as np
    import sounddevice as sd

    # Pre-load model
    self.engine._load_model()

    # Audio callback
    def audio_callback(indata, frames, time_info, status):
      with self._buffer_lock:
        self._audio_buffer.append(indata.copy())

    # Calculate buffer sizes
    window_samples = int(self.config.window_seconds * self.config.sample_rate)
    step_samples = int(self.config.step_seconds * self.config.sample_rate)

    # Start audio stream
    with sd.InputStream(
      samplerate=self.config.sample_rate,
      channels=1,
      dtype='float32',
      device=self.config.input_device,
      callback=audio_callback,
      blocksize=step_samples,
    ):
      last_transcribe = time.time()
      silence_start = None

      while self._running:
        time.sleep(0.1)

        # Check if we have enough audio
        with self._buffer_lock:
          if not self._audio_buffer:
            continue

          # Concatenate buffer
          audio = np.concatenate(self._audio_buffer, axis=0).flatten()

          # Keep only the window
          if len(audio) > window_samples:
            audio = audio[-window_samples:]
            # Trim buffer
            keep_chunks = int(np.ceil(window_samples / step_samples))
            self._audio_buffer = self._audio_buffer[-keep_chunks:]

        # Check for silence
        rms = np.sqrt(np.mean(audio[-step_samples:] ** 2))
        is_silent = rms < self.config.silence_threshold

        if is_silent:
          if silence_start is None:
            silence_start = time.time()
          elif time.time() - silence_start > self.config.silence_duration:
            # Auto-send after silence
            if self._last_transcript and self.on_transcript:
              self.on_transcript(self._last_transcript)
              self._last_transcript = ''
              with self._buffer_lock:
                self._audio_buffer.clear()
            silence_start = None
        else:
          silence_start = None

        # Transcribe periodically
        now = time.time()
        if now - last_transcribe >= self.config.step_seconds:
          last_transcribe = now

          try:
            transcript = self.engine.transcribe(audio)

            # Check for trigger phrase
            if self.config.trigger_phrase:
              lower = transcript.lower()
              trigger = self.config.trigger_phrase.lower()
              if trigger in lower:
                # Remove trigger phrase and send
                idx = lower.rfind(trigger)
                clean = transcript[:idx].strip()
                if clean and self.on_transcript:
                  self.on_transcript(clean)
                self._last_transcript = ''
                with self._buffer_lock:
                  self._audio_buffer.clear()
                continue

            self._last_transcript = transcript
            if self.on_partial:
              self.on_partial(transcript)

          except Exception as e:
            print(f'ASR error: {e}', flush=True)


# Global engine for reuse
_engine: ASREngine | None = None


def get_engine(config: ASRConfig | None = None) -> ASREngine:
  global _engine
  if _engine is None:
    _engine = ASREngine(config)
  return _engine


def transcribe(audio) -> str:
  """Convenience function to transcribe audio."""
  return get_engine().transcribe(audio)
