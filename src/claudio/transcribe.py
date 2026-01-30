"""Speech-to-text transcription using MLX-based models.

Uses parakeet-mlx for real-time streaming on Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Protocol


@dataclass
class Chunk:
  """A transcription chunk from the ASR model."""

  text: str
  is_final: bool = False
  latency_ms: float | None = None  # Time to transcribe
  window_seconds: float | None = None  # Audio window size


class TranscriberProtocol(Protocol):
  """Protocol for transcriber implementations."""

  def stream(self) -> Generator[Chunk, None, None]:
    """Stream transcription chunks from microphone input."""
    ...

  def stop(self) -> None:
    """Stop the transcription stream."""
    ...


class ParakeetTranscriber:
  """Real-time transcriber using parakeet-mlx with sliding window."""

  def __init__(
    self,
    model: str = 'mlx-community/parakeet-tdt-0.6b-v3',
    device: int | None = None,
    window_seconds: float = 10.0,
    step_seconds: float = 0.5,
  ):
    self.model_name = model
    self.device = device
    self.window_seconds = window_seconds
    self.step_seconds = step_seconds
    self._running = False
    self._model = None

  def _load_model(self):
    if self._model is None:
      try:
        from parakeet_mlx import from_pretrained
      except ImportError:
        raise ImportError('parakeet-mlx not installed. Run: uv sync --extra asr')
      print(f'Loading model {self.model_name}...')
      self._model = from_pretrained(self.model_name)
      print('Model loaded.')
    return self._model

  def stream(self) -> Generator[Chunk, None, None]:
    """Stream transcription using sliding window approach."""
    import queue

    import mlx.core as mx
    import numpy as np
    import sounddevice as sd

    from parakeet_mlx.audio import get_logmel

    model = self._load_model()

    sample_rate = model.preprocessor_config.sample_rate
    window_samples = int(self.window_seconds * sample_rate)
    step_samples = int(self.step_seconds * sample_rate)

    audio_queue: queue.Queue = queue.Queue()
    self._running = True

    def audio_callback(indata, frames, time_info, status):
      if status:
        print(f'Audio status: {status}')
      audio_queue.put(indata.copy())

    # Rolling buffer - collect audio chunks
    chunks: list = []
    last_text = ''

    with sd.InputStream(
      samplerate=sample_rate,
      channels=1,
      dtype='float32',
      blocksize=step_samples,
      device=self.device,
      callback=audio_callback,
    ):
      while self._running:
        try:
          audio_data = audio_queue.get(timeout=0.1)
          chunks.append(audio_data.flatten())

          # Keep only last N seconds worth of chunks
          max_chunks = int(self.window_seconds / self.step_seconds)
          if len(chunks) > max_chunks:
            chunks = chunks[-max_chunks:]

          # Need at least 2 chunks to transcribe
          if len(chunks) < 2:
            continue

          # Concatenate and transcribe
          import time
          t0 = time.perf_counter()

          buffer = np.concatenate(chunks)
          window_secs = len(buffer) / sample_rate
          audio_mx = mx.array(buffer)
          mel = get_logmel(audio_mx, model.preprocessor_config)
          results = model.generate(mel)

          latency_ms = (time.perf_counter() - t0) * 1000

          if results and results[0].text:
            text = results[0].text.strip()
            if text and text != last_text:
              yield Chunk(
                text=text,
                is_final=False,
                latency_ms=latency_ms,
                window_seconds=window_secs,
              )
              last_text = text

        except queue.Empty:
          continue

  def stop(self) -> None:
    """Stop the transcription stream."""
    self._running = False


class WhisperTranscriber:
  """Transcriber using mlx-audio whisper models."""

  def __init__(
    self,
    model: str = 'mlx-community/whisper-large-v3-turbo-asr-fp16',
    device: int | None = None,
    chunk_duration: float = 1.0,
  ):
    self.model_name = model
    self.device = device
    self.chunk_duration = chunk_duration
    self._running = False
    self._model = None

  def _load_model(self):
    if self._model is None:
      try:
        from mlx_audio.stt.utils import load_model
      except ImportError:
        raise ImportError('mlx-audio not installed. Run: uv sync --extra asr')
      self._model = load_model(self.model_name)
    return self._model

  def stream(self) -> Generator[Chunk, None, None]:
    """Stream transcription from microphone using chunked processing."""
    import sounddevice as sd
    import numpy as np

    model = self._load_model()

    try:
      from mlx_audio.stt.utils import transcribe
    except ImportError:
      raise ImportError('mlx-audio not installed. Run: uv sync --extra asr')

    sample_rate = 16000
    chunk_samples = int(self.chunk_duration * sample_rate)
    self._running = True

    def callback(indata, frames, time_info, status):
      if status:
        print(f'Audio status: {status}')
      audio_queue.put(indata.copy())

    import queue
    audio_queue = queue.Queue()

    with sd.InputStream(
      samplerate=sample_rate,
      channels=1,
      dtype='float32',
      blocksize=chunk_samples,
      device=self.device,
      callback=callback,
    ):
      buffer = np.array([], dtype='float32')

      while self._running:
        try:
          chunk = audio_queue.get(timeout=0.1)
          buffer = np.concatenate([buffer, chunk.flatten()])

          if len(buffer) >= chunk_samples:
            # Transcribe the buffer
            result = transcribe(buffer[:chunk_samples], model=model)
            text = result.get('text', '').strip()

            if text:
              yield Chunk(text=text, is_final=True)

            buffer = buffer[chunk_samples:]

        except queue.Empty:
          continue

  def stop(self) -> None:
    """Stop the transcription stream."""
    self._running = False


class AppleTranscriber:
  """Transcriber using Apple's built-in Speech framework.

  Uses on-device neural speech recognition via SFSpeechRecognizer.
  Works offline, low latency, uses Neural Engine on Apple Silicon.

  Requires macOS 10.15+ and microphone permission.
  """

  def __init__(
    self,
    device: int | None = None,
    locale: str = 'en-US',
  ):
    self.device = device  # Note: device selection not supported with AVAudioEngine
    self.locale = locale
    self._running = False
    self._engine = None
    self._request = None
    self._task = None

  def stream(self) -> Generator[Chunk, None, None]:
    """Stream transcription from microphone using Apple Speech."""
    import queue
    import time as time_module

    try:
      from AVFoundation import AVAudioEngine
      from Foundation import NSLocale
      from Speech import (
        SFSpeechRecognizer,
        SFSpeechAudioBufferRecognitionRequest,
      )
    except ImportError as e:
      raise ImportError(
        f'PyObjC Speech framework not available: {e}. '
        'Run: uv pip install pyobjc-framework-Speech'
      )

    # Create recognizer for locale
    ns_locale = NSLocale.localeWithLocaleIdentifier_(self.locale)
    recognizer = SFSpeechRecognizer.alloc().initWithLocale_(ns_locale)

    if not recognizer or not recognizer.isAvailable():
      raise RuntimeError(
        'Speech recognition not available. '
        'Check System Preferences > Privacy > Speech Recognition'
      )

    # Create audio engine and recognition request
    self._engine = AVAudioEngine.alloc().init()
    self._request = SFSpeechAudioBufferRecognitionRequest.alloc().init()
    self._request.setShouldReportPartialResults_(True)

    # Use on-device recognition if available
    if hasattr(self._request, 'setRequiresOnDeviceRecognition_'):
      self._request.setRequiresOnDeviceRecognition_(True)

    # Queue for results
    result_queue: queue.Queue = queue.Queue()
    self._running = True

    # Result handler
    def handle_result(result, error):
      if not self._running:
        return
      if error:
        err_desc = error.localizedDescription() if hasattr(error, 'localizedDescription') else str(error)
        result_queue.put(('__ERROR__', err_desc))
        return
      if result:
        text = result.bestTranscription().formattedString()
        is_final = result.isFinal()
        result_queue.put((text, is_final))

    # Start recognition task
    self._task = recognizer.recognitionTaskWithRequest_resultHandler_(
      self._request, handle_result
    )

    # Get input node and its format
    input_node = self._engine.inputNode()
    record_format = input_node.outputFormatForBus_(0)

    # Install tap on input node to get audio
    def tap_block(buffer, when):
      if self._running and self._request:
        self._request.appendAudioPCMBuffer_(buffer)

    input_node.installTapOnBus_bufferSize_format_block_(
      0,  # bus
      1024,  # buffer size
      record_format,
      tap_block,
    )

    # Start audio engine
    self._engine.prepare()
    success, error = self._engine.startAndReturnError_(None)
    if not success:
      raise RuntimeError(f'Failed to start audio engine: {error}')

    last_text = ''
    start_time = time_module.perf_counter()

    try:
      while self._running:
        try:
          item = result_queue.get(timeout=0.1)
          if item[0] == '__ERROR__':
            # Log error but continue
            continue

          text, is_final = item
          if text and text != last_text:
            latency = (time_module.perf_counter() - start_time) * 1000
            yield Chunk(text=text, is_final=is_final, latency_ms=latency)
            last_text = text
            start_time = time_module.perf_counter()

          if is_final:
            last_text = ''

        except queue.Empty:
          continue

    finally:
      self._cleanup()

  def _cleanup(self):
    """Clean up audio engine and recognition."""
    if self._engine:
      self._engine.stop()
      self._engine.inputNode().removeTapOnBus_(0)
      self._engine = None

    if self._request:
      self._request.endAudio()
      self._request = None

    if self._task:
      self._task.cancel()
      self._task = None

  def stop(self) -> None:
    """Stop the transcription stream."""
    self._running = False
    self._cleanup()


def get_transcriber(
  backend: str = 'parakeet',
  device: int | None = None,
  window_seconds: float = 10.0,
  step_seconds: float = 0.5,
  **kwargs,
) -> TranscriberProtocol:
  """Get a transcriber instance.

  Args:
      backend: "parakeet", "whisper", or "apple"
      device: Audio input device index
      window_seconds: Audio window size
      step_seconds: Step between transcriptions
      **kwargs: Additional args passed to transcriber

  Returns:
      Transcriber instance

  Backends:
      parakeet: MLX-based, fast, good accuracy (requires parakeet-mlx)
      whisper: MLX Whisper, very accurate (requires mlx-audio)
      apple: macOS built-in Speech framework, no deps, uses Neural Engine
  """
  if backend == 'parakeet':
    return ParakeetTranscriber(
      device=device,
      window_seconds=window_seconds,
      step_seconds=step_seconds,
      **kwargs,
    )
  elif backend == 'whisper':
    return WhisperTranscriber(device=device, **kwargs)
  elif backend == 'apple':
    return AppleTranscriber(device=device, **kwargs)
  else:
    raise ValueError(f'Unknown backend: {backend}. Options: parakeet, whisper, apple')
