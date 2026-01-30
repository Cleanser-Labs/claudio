"""Integration tests for claudio TTS pipeline."""

import sys
import time
import threading
import pytest
from unittest.mock import Mock, patch

from claudio import tts
from claudio.proxy import TTSQueue, MarkerBuffer, SpeechRequest, JsonLogger


pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')


class MockTTS(tts.TTS):
  """Mock TTS for testing without audio playback."""

  def __init__(self):
    super().__init__(tts.Config())
    self.generated = []
    self.played = []
    self.generate_delay = 0.01  # Simulate generation time
    self.play_delay = 0.01     # Simulate playback time

  def speak(self, text, voice=None, rate=1.0):
    self.generated.append(text)
    self.played.append(text)

  def generate(self, text, voice=None, rate=1.0):
    time.sleep(self.generate_delay)
    self.generated.append(text)
    # Return fake WAV bytes
    return b'RIFF' + b'\x00' * 100

  def play(self, audio):
    time.sleep(self.play_delay)
    self.played.append(audio)

  def voices(self):
    return [{'name': 'mock', 'id': 'mock'}]


class TestTTSQueueUnit:
  """Unit tests for TTSQueue with mocked TTS."""

  def test_queue_processes_text(self):
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)

    queue.add('Hello world')
    time.sleep(0.1)  # Give threads time to process

    assert 'Hello world' in mock_tts.generated
    queue.stop()

  def test_queue_processes_speech_request(self):
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)

    req = SpeechRequest(text='Test message', speed=1.5)
    queue.add(req)
    time.sleep(0.1)

    assert 'Test message' in mock_tts.generated
    queue.stop()

  def test_queue_tracks_stats(self):
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)

    queue.add('Hello')
    queue.add('World')
    time.sleep(0.2)

    stats = queue.get_stats()
    assert stats['chars'] == 10  # 'Hello' + 'World'
    queue.stop()

  def test_queue_reset_stats(self):
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)

    queue.add('First')
    time.sleep(0.1)
    queue.reset_stats()

    stats = queue.get_stats()
    assert stats['chars'] == 0
    queue.stop()


@pytest.mark.slow
class TestTTSQueueWithKokoro:
  """Integration tests with real Kokoro TTS. Run with: pytest -m slow"""

  @pytest.fixture
  def kokoro_queue(self):
    engine = tts.create(tts.Config(backend='kokoro'))
    queue = TTSQueue(engine)
    yield queue
    queue.stop()

  def test_generates_audio(self, kokoro_queue):
    """Test that Kokoro actually generates audio."""
    kokoro_queue.add('Hello')
    time.sleep(2.0)  # Give time for generation

    stats = kokoro_queue.get_stats()
    assert stats['chars'] == 5

  def test_multiple_requests(self, kokoro_queue):
    """Test queuing multiple requests."""
    kokoro_queue.add('One')
    kokoro_queue.add('Two')
    kokoro_queue.add('Three')
    time.sleep(5.0)  # More time for multiple requests

    stats = kokoro_queue.get_stats()
    assert stats['chars'] == 11  # One + Two + Three


class TestMarkerBufferToTTSQueue:
  """Test the full pipeline from MarkerBuffer to TTSQueue."""

  def test_streaming_to_tts(self):
    """Simulate streaming response through MarkerBuffer to TTS."""
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)
    buffer = MarkerBuffer(on_speech=queue.add)

    # Simulate streaming chunks
    chunks = [
      '<say>',
      'Hello! ',
      'How are you today?',
      '</say>',
    ]

    for chunk in chunks:
      buffer.add(chunk)

    buffer.flush()
    time.sleep(0.2)

    # Check that text was processed
    assert len(mock_tts.generated) > 0
    all_generated = ' '.join(mock_tts.generated)
    assert 'Hello' in all_generated
    queue.stop()

  def test_multiple_tags_streaming(self):
    """Test multiple <say> tags in a streaming response."""
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)
    buffer = MarkerBuffer(on_speech=queue.add)

    # Claude-style response with multiple say blocks around code
    chunks = [
      "<say>Here's a simple function:</say>\n\n",
      '```python\ndef hello():\n    print("hi")\n```\n\n',
      '<say>This prints hello to the console.</say>',
    ]

    for chunk in chunks:
      buffer.add(chunk)

    buffer.flush()
    time.sleep(0.3)

    assert len(mock_tts.generated) == 2
    assert "Here's a simple function" in mock_tts.generated[0]
    assert 'prints hello' in mock_tts.generated[1]
    queue.stop()


class TestEndToEnd:
  """End-to-end tests simulating full proxy flow."""

  def test_full_response_flow(self):
    """Test a complete response flow like the proxy handles."""
    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts)
    buffer = MarkerBuffer(on_speech=queue.add)

    # Simulate a typical Claude Code response
    response_chunks = [
      '{"type":"content_block_start"}\n',
      '<say',
      '>',
      'Found the issue. ',
      "It's on line 42.",
      '</say>',
      '\n\nHere\'s the fix:\n\n```python\n',
      'x = None\nif x:\n    print(x)\n',
      '```\n\n',
      '<say>I\'ve updated the code.</say>',
    ]

    for chunk in response_chunks:
      # Only feed text content to buffer (skip JSON metadata)
      if not chunk.startswith('{'):
        buffer.add(chunk)

    buffer.flush()
    time.sleep(0.3)

    # Should have captured spoken content
    assert len(mock_tts.generated) >= 2
    all_text = ' '.join(mock_tts.generated)
    assert 'Found the issue' in all_text
    assert 'updated the code' in all_text
    # Should NOT contain code
    assert 'print(x)' not in all_text
    queue.stop()


@pytest.mark.slow
class TestTTSBackendIntegration:
  """Test actual TTS backends generate valid audio. Run with: pytest -m slow"""

  def test_kokoro_generates_playable_wav(self):
    """Kokoro should generate valid WAV that can be played."""
    engine = tts.create(tts.Config(backend='kokoro'))
    audio = engine.generate('Test audio generation')

    # Verify WAV structure
    assert audio[:4] == b'RIFF'
    assert audio[8:12] == b'WAVE'

    # Find data chunk
    assert b'data' in audio

    # Should have reasonable size (not empty)
    assert len(audio) > 10000

  def test_say_generates_playable_aiff(self):
    """macOS say should generate valid AIFF."""
    engine = tts.create(tts.Config(backend='say'))
    audio = engine.generate('Test audio')

    # Verify AIFF structure
    assert audio[:4] == b'FORM'
    assert b'AIFF' in audio[:12]

  def test_kokoro_different_voices(self):
    """Test Kokoro with different voice presets."""
    engine = tts.create(tts.Config(backend='kokoro'))

    voices_to_test = ['af_heart', 'af_bella', 'am_adam']
    for voice in voices_to_test:
      audio = engine.generate('Hello', voice=voice)
      assert len(audio) > 1000, f'Voice {voice} produced no audio'


class TestTTSQueueEdgeCases:
  """Test TTSQueue handles edge cases correctly."""

  def test_queue_skips_non_pronounceable_text(self):
    """Queue should skip non-pronounceable text without hanging."""
    mock_tts = MockTTS()
    # Make mock return empty bytes for non-pronounceable text
    original_generate = mock_tts.generate
    def generate_with_skip(text, voice=None, rate=1.0):
      if not any(c.isalnum() for c in text):
        return b''
      return original_generate(text, voice, rate)
    mock_tts.generate = generate_with_skip

    queue = TTSQueue(mock_tts)

    # Add mix of pronounceable and non-pronounceable
    queue.add('#')
    queue.add('Hello')
    queue.add('...')
    queue.add('World')
    time.sleep(0.3)

    # Only pronounceable text should be played
    assert len(mock_tts.played) == 2
    queue.stop()

  def test_queue_handles_empty_generate_result(self):
    """Queue should not hang when generate returns empty bytes."""
    mock_tts = MockTTS()
    mock_tts.generate = lambda *args, **kwargs: b''  # Always return empty

    queue = TTSQueue(mock_tts)
    queue.add('Test')
    time.sleep(0.2)

    # Should not hang, nothing played
    assert len(mock_tts.played) == 0
    queue.stop()

  def test_queue_continues_after_empty(self):
    """Queue should continue processing after empty result."""
    mock_tts = MockTTS()
    call_count = [0]
    original_generate = mock_tts.generate

    def generate_alternating(text, voice=None, rate=1.0):
      call_count[0] += 1
      if call_count[0] == 2:  # Second call returns empty
        return b''
      return original_generate(text, voice, rate)

    mock_tts.generate = generate_alternating

    queue = TTSQueue(mock_tts)
    queue.add('First')
    queue.add('Skip')  # Will return empty
    queue.add('Third')
    time.sleep(0.3)

    # First and Third should be played, Skip should be skipped
    assert len(mock_tts.played) == 2
    queue.stop()


class TestLogging:
  """Test that TTS events are properly logged."""

  def test_queue_logs_events(self, tmp_path):
    """Test that TTSQueue logs to JsonLogger."""
    log_file = tmp_path / 'test.jsonl'
    logger = JsonLogger(log_file)

    mock_tts = MockTTS()
    queue = TTSQueue(mock_tts, logger=logger)

    queue.add('Test logging')
    time.sleep(0.2)

    queue.stop()
    logger.close()

    # Read log file
    log_content = log_file.read_text()
    assert 'tts_queued' in log_content
    assert 'tts_generate_start' in log_content
    assert 'tts_generate_done' in log_content
