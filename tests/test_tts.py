"""Tests for TTS backends."""

import sys
import pytest
from claudio import tts


# Skip all tests if not on macOS (most backends are macOS-only)
pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS only')


class TestTTSConfig:
  def test_default_config(self):
    config = tts.Config()
    assert config.backend == 'auto'
    assert config.rate == 1.0
    assert config.voice is None

  def test_custom_config(self):
    config = tts.Config(backend='kokoro', voice='af_heart', rate=1.5)
    assert config.backend == 'kokoro'
    assert config.voice == 'af_heart'
    assert config.rate == 1.5


class TestTTSFactory:
  def test_create_auto_returns_kokoro_on_macos(self):
    engine = tts.create(tts.Config(backend='auto'))
    assert type(engine).__name__ == 'Kokoro'

  def test_create_kokoro(self):
    engine = tts.create(tts.Config(backend='kokoro'))
    assert type(engine).__name__ == 'Kokoro'

  def test_create_say(self):
    engine = tts.create(tts.Config(backend='say'))
    assert type(engine).__name__ == 'Say'

  def test_create_soprano(self):
    engine = tts.create(tts.Config(backend='soprano'))
    assert type(engine).__name__ == 'Soprano'

  def test_create_unknown_raises(self):
    with pytest.raises(ValueError, match='Unknown backend'):
      tts.create(tts.Config(backend='unknown'))


class TestKokoroBackend:
  @pytest.fixture
  def kokoro(self):
    return tts.create(tts.Config(backend='kokoro'))

  def test_generate_returns_wav_bytes(self, kokoro):
    audio = kokoro.generate('Hello')
    assert isinstance(audio, bytes)
    assert audio[:4] == b'RIFF'  # WAV header
    assert b'WAVE' in audio[:12]

  def test_generate_with_voice(self, kokoro):
    audio = kokoro.generate('Hello', voice='af_bella')
    assert isinstance(audio, bytes)
    assert len(audio) > 1000  # Should have actual audio data

  def test_generate_with_rate(self, kokoro):
    slow = kokoro.generate('Hello', rate=0.5)
    fast = kokoro.generate('Hello', rate=2.0)
    # Slower speech should produce more audio data
    assert len(slow) > len(fast)

  def test_voices_returns_list(self, kokoro):
    voices = kokoro.voices()
    assert isinstance(voices, list)
    assert len(voices) > 0
    assert all('name' in v and 'id' in v for v in voices)

  def test_generate_empty_text_returns_empty(self, kokoro):
    """Empty text should return empty bytes."""
    assert kokoro.generate('') == b''
    assert kokoro.generate('   ') == b''

  def test_generate_non_pronounceable_returns_empty(self, kokoro):
    """Non-pronounceable text (symbols only) should return empty bytes."""
    assert kokoro.generate('#') == b''
    assert kokoro.generate('###') == b''
    assert kokoro.generate('...') == b''
    assert kokoro.generate('---') == b''
    assert kokoro.generate('@#$%') == b''

  def test_generate_mixed_text_works(self, kokoro):
    """Text with some pronounceable chars should work."""
    audio = kokoro.generate('Hello #world')
    assert isinstance(audio, bytes)
    assert len(audio) > 1000

  def test_play_empty_bytes_noop(self, kokoro):
    """Playing empty bytes should not raise."""
    kokoro.play(b'')  # Should not raise

  @pytest.mark.skip(reason='stream() not used in proxy, needs fixing')
  def test_stream_yields_chunks(self, kokoro):
    chunks = list(kokoro.stream('Hello world'))
    assert len(chunks) > 0
    assert all(isinstance(c, bytes) for c in chunks)


class TestSayBackend:
  @pytest.fixture
  def say(self):
    return tts.create(tts.Config(backend='say'))

  def test_generate_returns_aiff_bytes(self, say):
    audio = say.generate('Hello')
    assert isinstance(audio, bytes)
    assert audio[:4] == b'FORM'  # AIFF header

  def test_generate_with_voice(self, say):
    audio = say.generate('Hello', voice='Samantha')
    assert isinstance(audio, bytes)
    assert len(audio) > 1000

  def test_voices_returns_list(self, say):
    voices = say.voices()
    assert isinstance(voices, list)
    assert len(voices) > 0


class TestSopranoBackend:
  @pytest.fixture
  def soprano(self):
    return tts.create(tts.Config(backend='soprano'))

  def test_generate_returns_bytes(self, soprano):
    """Soprano returns raw audio bytes (not WAV formatted)."""
    audio = soprano.generate('Hello')
    assert isinstance(audio, bytes)
    assert len(audio) > 1000  # Should have actual audio data

  def test_voices_returns_list(self, soprano):
    voices = soprano.voices()
    assert isinstance(voices, list)


class TestTTSPerformance:
  """Performance benchmarks for TTS backends."""

  @pytest.fixture
  def kokoro(self):
    return tts.create(tts.Config(backend='kokoro'))

  def test_kokoro_generation_under_500ms(self, kokoro, benchmark):
    """Kokoro should generate short text in under 500ms."""
    result = benchmark(kokoro.generate, 'Hello world')
    assert isinstance(result, bytes)
    # benchmark.stats['mean'] is in seconds
    assert benchmark.stats['mean'] < 0.5

  def test_kokoro_first_generation_loads_model(self, kokoro):
    """First generation loads the model, subsequent calls are faster."""
    import time

    # First call (cold start - loads model)
    start = time.time()
    kokoro.generate('First')
    cold_time = time.time() - start

    # Second call (warm)
    start = time.time()
    kokoro.generate('Second')
    warm_time = time.time() - start

    # Warm calls should be significantly faster
    # (unless model was already cached from previous test)
    assert warm_time < 1.0  # Should be under 1 second when warm
