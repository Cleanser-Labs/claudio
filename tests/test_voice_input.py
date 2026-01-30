"""Tests for voice input module."""

import pytest
from unittest.mock import Mock, patch, call


class TestTextInput:
  """Test text input functions."""

  def test_paste_text_uses_clipboard(self):
    """type_text uses clipboard+paste by default."""
    from claudio.voice_input import type_text

    with patch('subprocess.Popen') as mock_popen, patch('subprocess.run') as mock_run:
      mock_proc = Mock()
      mock_proc.communicate.return_value = (b'', b'')
      mock_proc.returncode = 0
      mock_popen.return_value = mock_proc
      mock_run.return_value = Mock(returncode=0)

      type_text('Hello world')

      # Should have used pbcopy
      mock_popen.assert_called_once()
      assert mock_popen.call_args[0][0] == ['pbcopy']

      # Should have pasted with Cmd+V
      mock_run.assert_called_once()
      script = mock_run.call_args[0][0][2]
      assert 'keystroke "v" using command down' in script

  def test_keystroke_text_escapes_special_chars(self):
    """AppleScript keystroke escaping."""
    from claudio.voice_input import _keystroke_text

    with patch('subprocess.run') as mock_run:
      mock_run.return_value = Mock(returncode=0)
      _keystroke_text('Hello "world"')

      # Should have escaped the quotes
      mock_run.assert_called_once()
      script = mock_run.call_args[0][0][2]
      assert '\\"world\\"' in script

  def test_press_enter(self):
    """Press enter key."""
    from claudio.voice_input import press_enter

    with patch('subprocess.run') as mock_run:
      mock_run.return_value = Mock(returncode=0)
      result = press_enter()

      assert result is True
      mock_run.assert_called_once()
      script = mock_run.call_args[0][0][2]
      assert 'keystroke return' in script


class TestVoiceInputConfig:
  """Test configuration."""

  def test_default_config(self):
    from claudio.voice_input import VoiceInputConfig

    config = VoiceInputConfig()
    assert config.backend == 'parakeet'
    assert config.trigger_phrase == 'send'
    assert config.silence_seconds == 1.5
    assert config.auto_submit is False

  def test_custom_config(self):
    from claudio.voice_input import VoiceInputConfig

    config = VoiceInputConfig(
      backend='whisper',
      trigger_phrase='done',
      silence_seconds=2.0,
      auto_submit=True,
    )
    assert config.backend == 'whisper'
    assert config.trigger_phrase == 'done'
    assert config.silence_seconds == 2.0
    assert config.auto_submit is True


class TestVoiceInputUI:
  """Test the terminal UI."""

  def test_show_listening(self, capsys):
    from claudio.voice_input import VoiceInputUI

    ui = VoiceInputUI()
    ui.show_listening()

    captured = capsys.readouterr()
    assert '🎤' in captured.out
    assert 'Listening' in captured.out

  def test_show_partial(self, capsys):
    from claudio.voice_input import VoiceInputUI

    ui = VoiceInputUI()
    ui.show_partial('Hello world')

    captured = capsys.readouterr()
    assert 'Hello world' in captured.out

  def test_show_partial_truncates_long_text(self, capsys):
    from claudio.voice_input import VoiceInputUI

    ui = VoiceInputUI()
    long_text = 'a' * 100
    ui.show_partial(long_text)

    captured = capsys.readouterr()
    assert '...' in captured.out

  def test_show_submitted(self, capsys):
    from claudio.voice_input import VoiceInputUI

    ui = VoiceInputUI()
    ui.show_submitted('Test message')

    captured = capsys.readouterr()
    assert '✓' in captured.out
    assert 'Sent' in captured.out


class TestVoiceInputController:
  """Test the VoiceInput controller."""

  def test_init(self):
    from claudio.voice_input import VoiceInput, VoiceInputConfig

    config = VoiceInputConfig()
    on_start = Mock()
    on_partial = Mock()
    on_submit = Mock()
    on_stop = Mock()

    voice = VoiceInput(
      config=config,
      on_start=on_start,
      on_partial=on_partial,
      on_submit=on_submit,
      on_stop=on_stop,
    )

    assert voice.config == config
    assert voice.is_running is False

  def test_start_calls_on_start(self):
    from claudio.voice_input import VoiceInput, VoiceInputConfig

    on_start = Mock()
    voice = VoiceInput(
      config=VoiceInputConfig(),
      on_start=on_start,
    )

    # Mock the _run method to avoid actual ASR
    voice._run = Mock()

    voice.start()
    assert voice.is_running is True
    on_start.assert_called_once()

    voice.stop()

  def test_stop_calls_on_stop(self):
    from claudio.voice_input import VoiceInput, VoiceInputConfig

    on_stop = Mock()
    voice = VoiceInput(
      config=VoiceInputConfig(),
      on_stop=on_stop,
    )

    voice._running = True
    voice.stop()

    on_stop.assert_called_once()

  def test_submit_calls_on_submit(self):
    from claudio.voice_input import VoiceInput, VoiceInputConfig

    on_submit = Mock()
    voice = VoiceInput(
      config=VoiceInputConfig(auto_submit=False),
      on_submit=on_submit,
    )

    voice._current_text = 'Hello world'
    voice.submit()

    on_submit.assert_called_once_with('Hello world')
    assert voice._current_text == ''

  def test_submit_with_auto_submit(self):
    from claudio.voice_input import VoiceInput, VoiceInputConfig

    on_submit = Mock()
    voice = VoiceInput(
      config=VoiceInputConfig(auto_submit=True),
      on_submit=on_submit,
    )

    voice._current_text = 'Hello'

    with patch('claudio.voice_input.press_enter') as mock_enter:
      mock_enter.return_value = True
      voice.submit()

      on_submit.assert_called_once_with('Hello')
      mock_enter.assert_called_once()
