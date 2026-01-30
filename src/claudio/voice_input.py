"""Voice input - stream speech to focused input field.

Uses ASR to transcribe speech and injects text into the active application.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Callable

import msgspec


class VoiceInputConfig(msgspec.Struct, kw_only=True):
  """Voice input configuration."""
  # ASR settings
  backend: str = 'parakeet'  # parakeet or whisper
  model: str | None = None  # None = use default for backend
  device: int | None = None  # Audio input device
  window_seconds: float = 10.0
  step_seconds: float = 0.1  # 100ms for fast updates

  # Trigger settings
  trigger_phrase: str | None = 'send'  # Say this to submit
  silence_seconds: float = 1.0  # Silence duration to auto-submit
  silence_threshold: float = 0.01  # RMS threshold

  # Output settings
  auto_submit: bool = False  # Press Enter after pasting
  continuous: bool = True  # Keep running after submit (Ctrl+C to stop)


def type_text(text: str) -> bool:
  """Copy text to clipboard. User must press Cmd+V to paste."""
  return _copy_to_clipboard(text)


def _copy_to_clipboard(text: str) -> bool:
  """Copy text to clipboard using pbcopy."""
  import subprocess

  try:
    proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
    proc.communicate(text.encode('utf-8'))
    return proc.returncode == 0
  except Exception:
    return False


def _paste_from_clipboard() -> bool:
  """Paste from clipboard using Cmd+V. Requires Accessibility permission."""
  import subprocess

  script = '''
    tell application "System Events"
      keystroke "v" using command down
    end tell
  '''
  try:
    result = subprocess.run(
      ['osascript', '-e', script],
      capture_output=True,
      timeout=2.0,
    )
    return result.returncode == 0
  except Exception:
    return False


def check_accessibility_permission() -> bool:
  """Check if we have Accessibility permission for keystrokes."""
  import subprocess

  # Try a harmless keystroke test
  script = '''
    tell application "System Events"
      key code 999
    end tell
  '''
  try:
    result = subprocess.run(
      ['osascript', '-e', script],
      capture_output=True,
      timeout=2.0,
    )
    # If we get error 1002, we don't have permission
    # key code 999 doesn't exist but permission error comes first
    stderr = result.stderr.decode() if result.stderr else ''
    return '1002' not in stderr and 'not allowed' not in stderr.lower()
  except Exception:
    return False


def _keystroke_text(text: str) -> bool:
  """Type text using AppleScript keystroke. Needs Accessibility permission."""
  # Escape special characters for AppleScript
  escaped = text.replace('\\', '\\\\').replace('"', '\\"')

  script = f'''
    tell application "System Events"
      keystroke "{escaped}"
    end tell
  '''

  try:
    subprocess.run(
      ['osascript', '-e', script],
      check=True,
      capture_output=True,
      timeout=5.0,
    )
    return True
  except subprocess.CalledProcessError as e:
    print(f'AppleScript error: {e.stderr.decode()}')
    return False
  except subprocess.TimeoutExpired:
    print('AppleScript timeout')
    return False


def press_enter() -> bool:
  """Press Enter key in the frontmost application."""
  script = '''
    tell application "System Events"
      keystroke return
    end tell
  '''
  try:
    subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
    return True
  except subprocess.CalledProcessError:
    return False


def clear_line() -> bool:
  """Clear the current line (Cmd+A, Delete)."""
  script = '''
    tell application "System Events"
      keystroke "a" using command down
      key code 51
    end tell
  '''
  try:
    subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
    return True
  except subprocess.CalledProcessError:
    return False


class VoiceInput:
  """Voice input controller - streams ASR to focused input field."""

  def __init__(
    self,
    config: VoiceInputConfig | None = None,
    on_start: Callable[[], None] | None = None,
    on_partial: Callable[[str], None] | None = None,
    on_submit: Callable[[str], None] | None = None,
    on_stop: Callable[[], None] | None = None,
  ):
    self.config = config or VoiceInputConfig()
    self.on_start = on_start
    self.on_partial = on_partial
    self.on_submit = on_submit
    self.on_stop = on_stop

    self._running = False
    self._thread: threading.Thread | None = None
    self._current_text = ''
    self._last_submitted = ''  # Track to avoid duplicate submits
    self._submit_cooldown = 0.0  # Time until next submit allowed

  @property
  def is_running(self) -> bool:
    return self._running

  def start(self):
    """Start voice input."""
    if self._running:
      return

    self._running = True
    self._current_text = ''
    self._typed_text = ''
    self._thread = threading.Thread(target=self._run, daemon=True)
    self._thread.start()

    if self.on_start:
      self.on_start()

  def stop(self):
    """Stop voice input."""
    self._running = False

    # Don't join if called from within the thread itself
    if self._thread and self._thread is not threading.current_thread():
      self._thread.join(timeout=2.0)
      self._thread = None

    if self.on_stop:
      self.on_stop()

  def submit(self):
    """Submit current transcript - copy to clipboard and paste."""
    text = self._current_text.strip()

    # Skip if empty, duplicate, or in cooldown
    if not text:
      return
    if text == self._last_submitted:
      return
    if time.time() < self._submit_cooldown:
      return

    self._current_text = ''
    self._last_submitted = text
    self._submit_cooldown = time.time() + 1.0  # 1s cooldown

    # Copy to clipboard and try to paste
    pasted = self._paste_final_text(text)

    if self.on_submit:
      self.on_submit((text, pasted))  # Pass tuple with paste status

    if pasted and self.config.auto_submit:
      time.sleep(0.1)  # Small delay before Enter
      press_enter()

  def _run(self):
    """Main voice input loop."""
    import numpy as np
    import sounddevice as sd

    from .transcribe import get_transcriber

    # Get transcriber
    transcriber = get_transcriber(
      backend=self.config.backend,
      device=self.config.device,
      window_seconds=self.config.window_seconds,
      step_seconds=self.config.step_seconds,
    )

    # Track silence for auto-submit
    silence_start: float | None = None
    step_samples = int(self.config.step_seconds * 16000)

    # Audio monitoring for silence detection
    audio_rms = [0.0]

    def audio_monitor(indata, frames, time_info, status):
      rms = np.sqrt(np.mean(indata ** 2))
      audio_rms[0] = rms

    # Start audio monitor in parallel
    monitor_stream = sd.InputStream(
      samplerate=16000,
      channels=1,
      dtype='float32',
      blocksize=step_samples,
      device=self.config.device,
      callback=audio_monitor,
    )

    try:
      monitor_stream.start()

      for chunk in transcriber.stream():
        if not self._running:
          break

        text = chunk.text.strip()
        if not text:
          continue

        # Check for trigger phrase
        if self.config.trigger_phrase:
          trigger = self.config.trigger_phrase.lower()
          lower = text.lower()
          if trigger in lower:
            # Remove trigger and submit
            idx = lower.rfind(trigger)
            clean = text[:idx].strip()
            if clean:
              self._current_text = clean
            self.submit()
            if not self.config.continuous:
              self.stop()
              return
            # Reset for next utterance
            silence_start = None
            continue

        # Update current text (shown in preview, pasted on submit)
        self._current_text = text

        if self.on_partial:
          self.on_partial(text)

        # Check silence for auto-submit
        rms = audio_rms[0]
        if rms < self.config.silence_threshold:
          if silence_start is None:
            silence_start = time.time()
          elif time.time() - silence_start > self.config.silence_seconds:
            if self._current_text:
              self.submit()
              if not self.config.continuous:
                self.stop()
                return
              # Reset for next utterance
              silence_start = None
        else:
          silence_start = None

    finally:
      monitor_stream.stop()
      transcriber.stop()

  def _paste_final_text(self, text: str) -> bool:
    """Copy text to clipboard and paste it."""
    if not text:
      return False

    # Copy to clipboard
    if not _copy_to_clipboard(text):
      return False

    # Small delay for clipboard to be ready
    time.sleep(0.05)

    # Try to paste
    return _paste_from_clipboard()


class VoiceInputUI:
  """Simple terminal UI for voice input status."""

  def __init__(self, continuous: bool = True):
    self._last_line_len = 0
    self.continuous = continuous

  def show_listening(self):
    """Show listening indicator."""
    print('\r🎤 Listening...', end='', flush=True)
    self._last_line_len = 15

  def show_partial(self, text: str):
    """Show partial transcript."""
    display = f'\r🎤 {text[:60]}{"..." if len(text) > 60 else ""}'
    # Clear previous line
    clear = ' ' * max(0, self._last_line_len - len(display))
    print(display + clear, end='', flush=True)
    self._last_line_len = len(display)

  def show_submitted(self, result):
    """Show submitted text."""
    # Handle tuple (text, pasted) or just text
    if isinstance(result, tuple):
      text, pasted = result
    else:
      text, pasted = result, False

    # Clear the line and show confirmation
    clear = ' ' * self._last_line_len
    print(f'\r{clear}', end='')

    if pasted:
      print(f'\r✓ Pasted: {text[:50]}{"..." if len(text) > 50 else ""}')
    else:
      print(f'\r📋 Copied: {text[:40]}{"..." if len(text) > 40 else ""} [Cmd+V]')

    if self.continuous:
      print('\r🎤 Listening...', end='', flush=True)
      self._last_line_len = 15
    else:
      self._last_line_len = 0

  def show_stopped(self):
    """Show stopped indicator."""
    clear = ' ' * self._last_line_len
    print(f'\r{clear}', end='')
    print('\r⏹ Stopped')
    self._last_line_len = 0


def run_voice_input(
  config: VoiceInputConfig | None = None,
  show_ui: bool = True,
):
  """Run voice input with optional terminal UI.

  Press Ctrl+C to stop.
  """
  config = config or VoiceInputConfig()
  ui = VoiceInputUI(continuous=config.continuous) if show_ui else None

  def on_start():
    if ui:
      ui.show_listening()

  def on_partial(text: str):
    if ui:
      ui.show_partial(text)

  def on_submit(text: str):
    if ui:
      ui.show_submitted(text)

  def on_stop():
    if ui:
      ui.show_stopped()

  voice = VoiceInput(
    config=config,
    on_start=on_start,
    on_partial=on_partial,
    on_submit=on_submit,
    on_stop=on_stop,
  )

  try:
    voice.start()
    # Wait for completion or Ctrl+C
    while voice.is_running:
      time.sleep(0.1)
  except KeyboardInterrupt:
    voice.stop()


def run_triggered_voice_input(
  trigger_type: str = 'hotkey',
  voice_config: VoiceInputConfig | None = None,
  show_ui: bool = True,
  **trigger_kwargs,
):
  """Run voice input activated by a trigger.

  Args:
      trigger_type: Trigger type (hotkey, doubletap, wakeword, vad)
      voice_config: Voice input configuration
      show_ui: Show terminal UI
      **trigger_kwargs: Trigger-specific config (e.g., hotkey='cmd+shift+d')

  Examples:
      # Hotkey activation (Cmd+Shift+D)
      run_triggered_voice_input('hotkey', hotkey='cmd+shift+d')

      # Double-tap Control key
      run_triggered_voice_input('doubletap', key='ctrl')

      # Wake word activation
      run_triggered_voice_input('wakeword', backend='openwakeword')

      # Voice activity detection (just start talking)
      run_triggered_voice_input('vad')
  """
  from .triggers import get_trigger

  voice_config = voice_config or VoiceInputConfig()
  ui = VoiceInputUI(continuous=True) if show_ui else None

  # Create voice input
  voice: VoiceInput | None = None

  def start_dictation():
    nonlocal voice
    if voice and voice.is_running:
      return

    def on_start():
      if ui:
        ui.show_listening()

    def on_partial(text: str):
      if ui:
        ui.show_partial(text)

    def on_submit(text: str):
      if ui:
        ui.show_submitted(text)

    def on_stop():
      if ui:
        ui.show_stopped()

    voice = VoiceInput(
      config=voice_config,
      on_start=on_start,
      on_partial=on_partial,
      on_submit=on_submit,
      on_stop=on_stop,
    )
    voice.start()

  def stop_dictation():
    nonlocal voice
    if voice and voice.is_running:
      voice.submit()  # Submit any pending text
      voice.stop()
      voice = None

  # Set up trigger
  trigger = get_trigger(trigger_type, **trigger_kwargs)
  trigger.on_activate(start_dictation)
  trigger.on_deactivate(stop_dictation)

  if show_ui:
    print(f'🎯 Waiting for trigger ({trigger_type})...')
    if trigger_type == 'hotkey':
      hotkey = trigger_kwargs.get('hotkey', 'cmd+shift+d')
      print(f'   Press {hotkey} to start/stop dictation')
    elif trigger_type == 'doubletap':
      key = trigger_kwargs.get('key', 'ctrl')
      print(f'   Double-tap {key} to start/stop dictation')
    elif trigger_type == 'wakeword':
      print('   Say the wake word to start dictation')
    elif trigger_type == 'vad':
      print('   Just start speaking to activate')
    print('   Press Ctrl+C to quit')
    print()

  try:
    trigger.start()
    # Keep running until Ctrl+C
    while True:
      time.sleep(0.1)
  except KeyboardInterrupt:
    trigger.stop()
    if voice:
      voice.stop()
    if show_ui:
      print('\n👋 Goodbye!')
