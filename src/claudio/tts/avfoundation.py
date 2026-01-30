"""macOS AVFoundation backend."""

from __future__ import annotations

import io
import threading

from .base import TTS, Config


class AVFoundation(TTS):
  """macOS AVFoundation (native API)."""

  @classmethod
  def available(cls) -> bool:
    try:
      import AVFoundation  # noqa: F401
      return True
    except ImportError:
      return False

  @classmethod
  def list_voices(cls) -> list[dict]:
    try:
      import AVFoundation
    except ImportError:
      return []
    voices = []
    for v in AVFoundation.AVSpeechSynthesisVoice.speechVoices():
      quality = 'premium' if v.quality() >= 2 else 'standard'
      voices.append({
        'name': v.name(),
        'id': v.identifier(),
        'lang': v.language(),
        'quality': quality,
      })
    return voices

  def __init__(self, config: Config | None = None):
    super().__init__(config)
    self._synth = None
    self._delegate = None
    self._voice_map = None

  def _get_synth(self):
    if self._synth is None:
      import AVFoundation
      import objc

      class SynthDelegate(AVFoundation.NSObject):
        def init(self):
          self = objc.super(SynthDelegate, self).init()
          self.done_event = None
          return self

        def speechSynthesizer_didFinishSpeechUtterance_(self, synth, utterance):
          if self.done_event:
            self.done_event.set()

      self._synth = AVFoundation.AVSpeechSynthesizer.new()
      self._delegate = SynthDelegate.new()
      self._synth.setDelegate_(self._delegate)
    return self._synth

  def _get_voice(self, voice: str | None):
    import AVFoundation
    v = voice or self.config.voice
    if not v:
      return AVFoundation.AVSpeechSynthesisVoice.voiceWithLanguage_('en-US')

    found = AVFoundation.AVSpeechSynthesisVoice.voiceWithIdentifier_(v)
    if found:
      return found

    # Build name->voice map on first call for O(1) lookup
    if self._voice_map is None:
      self._voice_map = {av.name(): av for av in AVFoundation.AVSpeechSynthesisVoice.speechVoices()}

    if v in self._voice_map:
      return self._voice_map[v]

    return AVFoundation.AVSpeechSynthesisVoice.voiceWithLanguage_('en-US')

  def _rate(self, rate: float) -> float:
    return 0.5 * rate * self.config.rate

  def speak(self, text: str, voice: str | None = None, rate: float = 1.0) -> None:
    import AVFoundation

    synth = self._get_synth()
    utterance = AVFoundation.AVSpeechUtterance.speechUtteranceWithString_(text)
    utterance.setVoice_(self._get_voice(voice))
    utterance.setRate_(self._rate(rate))

    done = threading.Event()
    self._delegate.done_event = done
    synth.speakUtterance_(utterance)
    done.wait(timeout=60)

  def generate(self, text: str, voice: str | None = None, rate: float = 1.0) -> bytes:
    import ctypes
    import AVFoundation

    synth = self._get_synth()
    utterance = AVFoundation.AVSpeechUtterance.speechUtteranceWithString_(text)
    utterance.setVoice_(self._get_voice(voice))
    utterance.setRate_(self._rate(rate))

    buffer = io.BytesIO()
    done = threading.Event()

    def callback(audio_buffer):
      if audio_buffer is None:
        done.set()
      else:
        data = audio_buffer.floatChannelData()[0]
        frame_count = audio_buffer.frameLength()
        buffer.write(bytes((ctypes.c_float * frame_count).from_address(data)))

    synth.writeUtterance_toBufferCallback_(utterance, callback)
    done.wait(timeout=60)
    return buffer.getvalue()

  def voices(self) -> list[dict]:
    return self.list_voices()
