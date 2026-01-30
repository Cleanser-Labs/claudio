"""Trigger system for global keyboard shortcuts and wake words.

Example usage:

```python
from claudio.triggers import on_hotkey, on_wake_word, run_triggers

@on_hotkey('cmd+shift+c')
def start_coding():
    '''Start a coding session.'''
    return {'action': 'start_session', 'persona': 'coder'}

@on_hotkey('cmd+shift+m')
def toggle_mute():
    '''Toggle mute on default session.'''
    return {'action': 'toggle_mute', 'session': 'default'}

@on_wake_word('hey claude')
def wake_up():
    '''Start listening when wake word is detected.'''
    return {'action': 'listen_start', 'session': 'default'}

@on_wake_word(['stop', 'cancel', 'nevermind'])
def cancel():
    '''Cancel current operation.'''
    return {'action': 'listen_stop'}

if __name__ == '__main__':
    run_triggers()
```
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

import httpx
from rich.console import Console

console = Console()


# --- Trigger Registry ---

@dataclass
class TriggerDef:
  """Definition of a registered trigger."""
  name: str
  trigger_type: str  # 'hotkey', 'wake_word', 'event'
  pattern: str | list[str]  # Hotkey combo, wake word(s), or event name
  handler: Callable[[], dict | None]
  description: str = ''
  enabled: bool = True


class TriggerRegistry:
  """Global registry for triggers."""

  def __init__(self):
    self._triggers: list[TriggerDef] = []
    self._lock = threading.Lock()

  def register(self, trigger: TriggerDef):
    with self._lock:
      self._triggers.append(trigger)

  def get_all(self) -> list[TriggerDef]:
    with self._lock:
      return list(self._triggers)

  def get_by_type(self, trigger_type: str) -> list[TriggerDef]:
    with self._lock:
      return [t for t in self._triggers if t.trigger_type == trigger_type]

  def clear(self):
    with self._lock:
      self._triggers.clear()


# Global registry
_registry = TriggerRegistry()


def get_registry() -> TriggerRegistry:
  return _registry


# --- Decorators ---

def on_hotkey(keys: str, *, enabled: bool = True):
  """Decorator to register a hotkey trigger.

  Args:
    keys: Hotkey combination like 'cmd+shift+c', 'ctrl+alt+h', 'f12'
    enabled: Whether trigger is enabled by default

  Supported modifiers: cmd, ctrl, alt, shift, super
  Keys can be single letters, numbers, or special keys (f1-f12, space, etc.)

  Example:
    @on_hotkey('cmd+shift+c')
    def start_coding():
        return {'action': 'start_session', 'persona': 'coder'}
  """
  def decorator(func: Callable[[], dict | None]) -> Callable[[], dict | None]:
    trigger = TriggerDef(
      name=func.__name__,
      trigger_type='hotkey',
      pattern=keys.lower(),
      handler=func,
      description=func.__doc__ or '',
      enabled=enabled,
    )
    _registry.register(trigger)
    return func
  return decorator


def on_wake_word(words: str | list[str], *, enabled: bool = True):
  """Decorator to register a wake word trigger.

  Args:
    words: Wake word or list of wake words to listen for
    enabled: Whether trigger is enabled by default

  Example:
    @on_wake_word('hey claude')
    def wake_up():
        return {'action': 'listen_start'}

    @on_wake_word(['stop', 'cancel'])
    def cancel():
        return {'action': 'listen_stop'}
  """
  def decorator(func: Callable[[], dict | None]) -> Callable[[], dict | None]:
    pattern = words if isinstance(words, list) else [words]
    trigger = TriggerDef(
      name=func.__name__,
      trigger_type='wake_word',
      pattern=[w.lower() for w in pattern],
      handler=func,
      description=func.__doc__ or '',
      enabled=enabled,
    )
    _registry.register(trigger)
    return func
  return decorator


def on_event(event_name: str, *, enabled: bool = True):
  """Decorator to register an event trigger.

  Args:
    event_name: Event name to listen for
    enabled: Whether trigger is enabled by default

  Events:
    - 'startup': When trigger manager starts
    - 'shutdown': When trigger manager stops
    - 'session_start': When a session starts
    - 'session_end': When a session ends
    - 'speech_start': When TTS starts
    - 'speech_end': When TTS ends

  Example:
    @on_event('startup')
    def on_startup():
        print('Triggers active!')
        return {'action': 'speak', 'text': 'Ready'}
  """
  def decorator(func: Callable[[], dict | None]) -> Callable[[], dict | None]:
    trigger = TriggerDef(
      name=func.__name__,
      trigger_type='event',
      pattern=event_name,
      handler=func,
      description=func.__doc__ or '',
      enabled=enabled,
    )
    _registry.register(trigger)
    return func
  return decorator


# --- Action Executor ---

class ActionExecutor:
  """Execute actions returned by triggers."""

  def __init__(self, server_url: str = 'http://127.0.0.1:8765'):
    self.server_url = server_url
    self._client = httpx.Client(timeout=5.0)

  def execute(self, action: dict | None) -> bool:
    """Execute an action dict returned by a trigger handler.

    Returns True if action was executed successfully.
    """
    if not action:
      return True

    action_type = action.get('action')
    if not action_type:
      return False

    try:
      if action_type == 'start_session':
        session_id = action.get('session', f'trigger-{int(time.time())}')
        persona = action.get('persona')
        self._client.get(f'{self.server_url}/sessions/{session_id}')
        if persona:
          self._client.post(
            f'{self.server_url}/sessions/{session_id}/persona',
            json={'persona_id': persona},
          )
        console.print(f'[green]Started session:[/green] {session_id}')
        return True

      elif action_type == 'listen_start':
        session_id = action.get('session', 'default')
        self._client.post(f'{self.server_url}/listen/{session_id}/start')
        console.print(f'[green]Listening:[/green] {session_id}')
        return True

      elif action_type == 'listen_stop':
        session_id = action.get('session', 'default')
        self._client.post(f'{self.server_url}/listen/{session_id}/stop')
        console.print(f'[yellow]Stopped listening:[/yellow] {session_id}')
        return True

      elif action_type == 'listen_send':
        session_id = action.get('session', 'default')
        self._client.post(f'{self.server_url}/listen/{session_id}/send')
        console.print(f'[cyan]Sent transcript:[/cyan] {session_id}')
        return True

      elif action_type == 'mute':
        session_id = action.get('session', 'default')
        self._client.post(f'{self.server_url}/sessions/{session_id}/mute')
        console.print(f'[yellow]Muted:[/yellow] {session_id}')
        return True

      elif action_type == 'unmute':
        session_id = action.get('session', 'default')
        self._client.post(f'{self.server_url}/sessions/{session_id}/unmute')
        console.print(f'[green]Unmuted:[/green] {session_id}')
        return True

      elif action_type == 'toggle_mute':
        session_id = action.get('session', 'default')
        resp = self._client.get(f'{self.server_url}/sessions/{session_id}')
        if resp.status_code == 200:
          session = resp.json().get('session', {})
          endpoint = 'unmute' if session.get('muted') else 'mute'
          self._client.post(f'{self.server_url}/sessions/{session_id}/{endpoint}')
          console.print(f'[cyan]Toggled mute:[/cyan] {session_id}')
        return True

      elif action_type == 'speak':
        text = action.get('text', '')
        session_id = action.get('session')
        if session_id:
          self._client.post(
            f'{self.server_url}/talk/{session_id}',
            json={'text': text},
          )
        else:
          self._client.post(
            f'{self.server_url}/speak',
            json={'text': text},
          )
        return True

      elif action_type == 'set_priority':
        session_id = action.get('session', 'default')
        priority = action.get('priority', 50)
        self._client.patch(
          f'{self.server_url}/sessions/{session_id}',
          json={'priority': priority},
        )
        console.print(f'[cyan]Set priority:[/cyan] {session_id} -> {priority}')
        return True

      elif action_type == 'set_persona':
        session_id = action.get('session', 'default')
        persona = action.get('persona')
        if persona:
          self._client.post(
            f'{self.server_url}/sessions/{session_id}/persona',
            json={'persona_id': persona},
          )
          console.print(f'[cyan]Set persona:[/cyan] {session_id} -> {persona}')
        return True

      elif action_type == 'delete_session':
        session_id = action.get('session', 'default')
        self._client.delete(f'{self.server_url}/sessions/{session_id}')
        console.print(f'[red]Deleted session:[/red] {session_id}')
        return True

      elif action_type == 'reload_personas':
        self._client.post(f'{self.server_url}/personas/reload')
        console.print('[green]Reloaded personas[/green]')
        return True

      elif action_type == 'custom':
        # Custom action - just call the callback if provided
        callback = action.get('callback')
        if callable(callback):
          callback()
        return True

      else:
        console.print(f'[yellow]Unknown action:[/yellow] {action_type}')
        return False

    except Exception as e:
      console.print(f'[red]Action failed:[/red] {e}')
      return False


# --- Hotkey Listener ---

class HotkeyListener:
  """Listen for global keyboard shortcuts using pynput."""

  def __init__(self, executor: ActionExecutor, registry: TriggerRegistry):
    self.executor = executor
    self.registry = registry
    self._running = False
    self._listener = None
    self._current_keys: set[str] = set()

  def _parse_hotkey(self, hotkey: str) -> set[str]:
    """Parse hotkey string into set of key names."""
    parts = hotkey.lower().replace(' ', '').split('+')
    keys = set()
    for part in parts:
      # Normalize modifier names
      if part in ('cmd', 'command', 'meta', 'super', 'win'):
        keys.add('cmd')
      elif part in ('ctrl', 'control'):
        keys.add('ctrl')
      elif part in ('alt', 'option'):
        keys.add('alt')
      elif part in ('shift',):
        keys.add('shift')
      else:
        keys.add(part)
    return keys

  def _key_to_name(self, key) -> str | None:
    """Convert pynput key to name string."""
    from pynput import keyboard

    if hasattr(key, 'char') and key.char:
      return key.char.lower()
    elif hasattr(key, 'name'):
      name = key.name.lower()
      # Normalize names
      if name in ('cmd', 'cmd_l', 'cmd_r'):
        return 'cmd'
      elif name in ('ctrl', 'ctrl_l', 'ctrl_r'):
        return 'ctrl'
      elif name in ('alt', 'alt_l', 'alt_r', 'alt_gr'):
        return 'alt'
      elif name in ('shift', 'shift_l', 'shift_r'):
        return 'shift'
      return name
    return None

  def _on_press(self, key):
    """Handle key press."""
    name = self._key_to_name(key)
    if name:
      self._current_keys.add(name)
      self._check_hotkeys()

  def _on_release(self, key):
    """Handle key release."""
    name = self._key_to_name(key)
    if name:
      self._current_keys.discard(name)

  def _check_hotkeys(self):
    """Check if current keys match any registered hotkey."""
    for trigger in self.registry.get_by_type('hotkey'):
      if not trigger.enabled:
        continue
      required = self._parse_hotkey(trigger.pattern)
      if required == self._current_keys:
        console.print(f'[dim]Hotkey:[/dim] {trigger.pattern} -> {trigger.name}')
        try:
          result = trigger.handler()
          self.executor.execute(result)
        except Exception as e:
          console.print(f'[red]Handler error:[/red] {e}')

  def start(self):
    """Start listening for hotkeys."""
    try:
      from pynput import keyboard
    except ImportError:
      console.print('[yellow]pynput not installed - hotkeys disabled[/yellow]')
      console.print('[dim]Install with: uv sync --extra triggers[/dim]')
      return

    self._running = True
    self._listener = keyboard.Listener(
      on_press=self._on_press,
      on_release=self._on_release,
    )
    self._listener.start()
    console.print('[green]Hotkey listener started[/green]')

  def stop(self):
    """Stop listening for hotkeys."""
    self._running = False
    if self._listener:
      self._listener.stop()
      self._listener = None


# --- Wake Word Listener ---

# Pre-trained openWakeWord models available
OPENWAKEWORD_MODELS = {
  'hey_jarvis': 'hey_jarvis_v0.1',
  'alexa': 'alexa_v0.1',
  'hey_mycroft': 'hey_mycroft_v0.1',
  'hey_rhasspy': 'hey_rhasspy_v0.1',
  'ok_nabu': 'ok_nabu_v0.1',
  'timer': 'timer_v0.1',
  'weather': 'weather_v0.1',
}

# Mapping of common wake phrases to models
WAKE_PHRASE_TO_MODEL = {
  'hey jarvis': 'hey_jarvis',
  'hey claude': 'hey_jarvis',  # Use hey_jarvis as proxy for hey_claude
  'hey cloud': 'hey_jarvis',
  'a claude': 'hey_jarvis',
  'alexa': 'alexa',
  'hey mycroft': 'hey_mycroft',
}


class WakeWordListener:
  """Listen for wake words using openWakeWord.

  Uses Google's speech embeddings with lightweight classifiers for
  efficient, low-latency wake word detection. Processes 80ms audio
  chunks and can run multiple models simultaneously.

  Falls back to ASR-based detection for custom wake phrases without
  trained models.
  """

  def __init__(
    self,
    executor: ActionExecutor,
    registry: TriggerRegistry,
    threshold: float = 0.5,
    cooldown: float = 1.0,
  ):
    self.executor = executor
    self.registry = registry
    self.threshold = threshold  # Confidence threshold for activation
    self.cooldown = cooldown  # Seconds between activations
    self._running = False
    self._thread: threading.Thread | None = None
    self._last_activation: dict[str, float] = {}  # Track cooldowns per trigger
    self._oww_model = None
    self._use_fallback = False

  def _get_required_models(self) -> list[str]:
    """Determine which openWakeWord models are needed."""
    models_needed = set()

    for trigger in self.registry.get_by_type('wake_word'):
      if not trigger.enabled:
        continue
      patterns = trigger.pattern if isinstance(trigger.pattern, list) else [trigger.pattern]
      for pattern in patterns:
        pattern_lower = pattern.lower()
        # Check if pattern maps to a known model
        if pattern_lower in WAKE_PHRASE_TO_MODEL:
          models_needed.add(WAKE_PHRASE_TO_MODEL[pattern_lower])
        # Check if pattern is a model name
        elif pattern_lower in OPENWAKEWORD_MODELS:
          models_needed.add(pattern_lower)

    return list(models_needed)

  def _init_openwakeword(self) -> bool:
    """Initialize openWakeWord models."""
    try:
      import openwakeword
      from openwakeword.model import Model
    except ImportError:
      console.print('[yellow]openwakeword not installed[/yellow]')
      console.print('[dim]Install with: uv sync --extra triggers[/dim]')
      return False

    models = self._get_required_models()
    if not models:
      console.print('[dim]No openWakeWord models needed[/dim]')
      return False

    try:
      # Download models if needed
      openwakeword.utils.download_models(models)

      # Load models
      self._oww_model = Model(
        wakeword_models=models,
        inference_framework='onnx',
      )
      console.print(f'[green]Loaded openWakeWord models:[/green] {", ".join(models)}')
      return True
    except Exception as e:
      console.print(f'[yellow]openWakeWord init failed:[/yellow] {e}')
      return False

  def _normalize_text(self, text: str) -> str:
    """Normalize text for matching."""
    import re
    text = ' '.join(text.lower().split())
    text = re.sub(r'[^\w\s]', '', text)
    return text

  def _check_wake_words_oww(self, predictions: dict[str, float]):
    """Check openWakeWord predictions against registered triggers."""
    now = time.time()

    for trigger in self.registry.get_by_type('wake_word'):
      if not trigger.enabled:
        continue

      patterns = trigger.pattern if isinstance(trigger.pattern, list) else [trigger.pattern]
      for pattern in patterns:
        pattern_lower = pattern.lower()

        # Get model name for this pattern
        model_name = None
        if pattern_lower in WAKE_PHRASE_TO_MODEL:
          model_name = WAKE_PHRASE_TO_MODEL[pattern_lower]
        elif pattern_lower in OPENWAKEWORD_MODELS:
          model_name = pattern_lower

        if not model_name:
          continue

        # Check if model detected wake word
        confidence = predictions.get(model_name, 0.0)
        if confidence >= self.threshold:
          # Check cooldown
          if trigger.name in self._last_activation:
            if now - self._last_activation[trigger.name] < self.cooldown:
              continue

          self._last_activation[trigger.name] = now
          console.print(
            f'[dim]Wake word:[/dim] "{pattern}" ({confidence:.2f}) -> {trigger.name}'
          )
          try:
            result = trigger.handler()
            self.executor.execute(result)
          except Exception as e:
            console.print(f'[red]Handler error:[/red] {e}')
          return  # Only trigger once per detection

  def _check_wake_words_fallback(self, text: str):
    """Fallback: check text against registered wake words (ASR-based)."""
    normalized = self._normalize_text(text)
    now = time.time()

    for trigger in self.registry.get_by_type('wake_word'):
      if not trigger.enabled:
        continue

      patterns = trigger.pattern if isinstance(trigger.pattern, list) else [trigger.pattern]
      for pattern in patterns:
        pattern_normalized = self._normalize_text(pattern)
        if pattern_normalized in normalized:
          # Check cooldown
          if trigger.name in self._last_activation:
            if now - self._last_activation[trigger.name] < self.cooldown:
              continue

          self._last_activation[trigger.name] = now
          console.print(f'[dim]Wake word (ASR):[/dim] "{pattern}" -> {trigger.name}')
          try:
            result = trigger.handler()
            self.executor.execute(result)
          except Exception as e:
            console.print(f'[red]Handler error:[/red] {e}')
          return

  def _listen_loop_oww(self):
    """Main loop using openWakeWord."""
    import numpy as np

    try:
      import pyaudio
    except ImportError:
      console.print('[yellow]pyaudio not installed - wake words disabled[/yellow]')
      return

    # Audio settings for openWakeWord (16kHz, mono, 16-bit)
    CHUNK_SIZE = 1280  # 80ms at 16kHz
    SAMPLE_RATE = 16000
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    audio = pyaudio.PyAudio()
    stream = None

    try:
      stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
      )
      console.print('[green]Wake word listener started (openWakeWord)[/green]')

      while self._running:
        try:
          # Read audio chunk
          audio_bytes = stream.read(CHUNK_SIZE, exception_on_overflow=False)
          audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

          # Run prediction
          predictions = self._oww_model.predict(audio_data)
          self._check_wake_words_oww(predictions)

        except Exception as e:
          if self._running:
            console.print(f'[red]Wake word error:[/red] {e}')
          time.sleep(0.1)

    finally:
      if stream:
        stream.stop_stream()
        stream.close()
      audio.terminate()

  def _listen_loop_fallback(self):
    """Fallback loop using ASR."""
    try:
      from claudio.asr import ASREngine, ASRConfig
    except ImportError:
      console.print('[yellow]ASR not available - wake words disabled[/yellow]')
      return

    config = ASRConfig(
      silence_duration=1.0,
      min_audio_length=0.3,
    )

    try:
      engine = ASREngine(config)
    except Exception as e:
      console.print(f'[yellow]ASR init failed:[/yellow] {e}')
      return

    console.print('[green]Wake word listener started (ASR fallback)[/green]')

    while self._running:
      try:
        audio = engine.record(max_duration=3.0)
        if audio is not None and len(audio) > 0:
          text = engine.transcribe(audio)
          if text:
            self._check_wake_words_fallback(text)
      except Exception as e:
        if self._running:
          console.print(f'[red]ASR error:[/red] {e}')
        time.sleep(0.5)

  def start(self):
    """Start listening for wake words."""
    self._running = True

    # Try to initialize openWakeWord
    if self._init_openwakeword():
      self._use_fallback = False
      self._thread = threading.Thread(target=self._listen_loop_oww, daemon=True)
    else:
      # Fall back to ASR-based detection
      console.print('[dim]Falling back to ASR-based wake word detection[/dim]')
      self._use_fallback = True
      self._thread = threading.Thread(target=self._listen_loop_fallback, daemon=True)

    self._thread.start()

  def stop(self):
    """Stop listening for wake words."""
    self._running = False
    if self._thread:
      self._thread.join(timeout=2.0)
      self._thread = None
    self._oww_model = None


# --- Trigger Manager ---

class TriggerManager:
  """Manages all trigger types and executes actions."""

  def __init__(
    self,
    server_url: str = 'http://127.0.0.1:8765',
    enable_hotkeys: bool = True,
    enable_wake_words: bool = True,
  ):
    self.server_url = server_url
    self.registry = get_registry()
    self.executor = ActionExecutor(server_url)

    self._hotkey_listener: HotkeyListener | None = None
    self._wake_word_listener: WakeWordListener | None = None

    self.enable_hotkeys = enable_hotkeys
    self.enable_wake_words = enable_wake_words
    self._running = False

  def _fire_event(self, event_name: str):
    """Fire an event trigger."""
    for trigger in self.registry.get_by_type('event'):
      if trigger.enabled and trigger.pattern == event_name:
        try:
          result = trigger.handler()
          self.executor.execute(result)
        except Exception as e:
          console.print(f'[red]Event handler error:[/red] {e}')

  def start(self):
    """Start all listeners."""
    self._running = True

    # Start hotkey listener
    if self.enable_hotkeys:
      hotkey_triggers = self.registry.get_by_type('hotkey')
      if hotkey_triggers:
        self._hotkey_listener = HotkeyListener(self.executor, self.registry)
        self._hotkey_listener.start()
        for t in hotkey_triggers:
          console.print(f'  [cyan]{t.pattern}[/cyan] -> {t.name}')

    # Start wake word listener
    if self.enable_wake_words:
      wake_word_triggers = self.registry.get_by_type('wake_word')
      if wake_word_triggers:
        self._wake_word_listener = WakeWordListener(self.executor, self.registry)
        self._wake_word_listener.start()
        for t in wake_word_triggers:
          patterns = t.pattern if isinstance(t.pattern, list) else [t.pattern]
          console.print(f'  [cyan]{", ".join(patterns)}[/cyan] -> {t.name}')

    # Fire startup event
    self._fire_event('startup')

  def stop(self):
    """Stop all listeners."""
    self._running = False

    # Fire shutdown event
    self._fire_event('shutdown')

    if self._hotkey_listener:
      self._hotkey_listener.stop()
      self._hotkey_listener = None

    if self._wake_word_listener:
      self._wake_word_listener.stop()
      self._wake_word_listener = None

  def run(self):
    """Run the trigger manager until interrupted."""
    console.print()
    console.print('[bold green]Claudio Triggers[/bold green]')
    console.print(f'  Server: {self.server_url}')
    console.print()

    self.start()

    try:
      while self._running:
        time.sleep(0.1)
    except KeyboardInterrupt:
      console.print()
      console.print('[yellow]Stopping...[/yellow]')
    finally:
      self.stop()

    console.print('[dim]Triggers stopped[/dim]')


# --- Loader ---

def load_triggers_file(path: Path) -> int:
  """Load triggers from a Python file.

  Returns number of triggers loaded.
  """
  if not path.exists():
    raise FileNotFoundError(f'Triggers file not found: {path}')

  # Get count before loading
  before = len(_registry.get_all())

  # Import the module
  spec = importlib.util.spec_from_file_location('triggers_module', path)
  if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

  # Return count of new triggers
  return len(_registry.get_all()) - before


def load_triggers_dir(directory: Path) -> int:
  """Load all trigger files from a directory.

  Returns total number of triggers loaded.
  """
  total = 0
  if directory.exists():
    for path in sorted(directory.glob('*.py')):
      if not path.name.startswith('_'):
        try:
          count = load_triggers_file(path)
          total += count
          console.print(f'[dim]Loaded {count} triggers from {path.name}[/dim]')
        except Exception as e:
          console.print(f'[red]Failed to load {path.name}:[/red] {e}')
  return total


# --- Convenience function ---

def run_triggers(
  server_url: str = 'http://127.0.0.1:8765',
  enable_hotkeys: bool = True,
  enable_wake_words: bool = True,
  triggers_dir: Path | str | None = None,
):
  """Run the trigger manager with registered triggers.

  Args:
    server_url: URL of the claudio server
    enable_hotkeys: Enable keyboard shortcuts
    enable_wake_words: Enable wake word detection
    triggers_dir: Optional directory to load additional triggers from
  """
  # Load triggers from directory if specified
  if triggers_dir:
    triggers_path = Path(triggers_dir)
    if triggers_path.is_file():
      load_triggers_file(triggers_path)
    elif triggers_path.is_dir():
      load_triggers_dir(triggers_path)

  # Create and run manager
  manager = TriggerManager(
    server_url=server_url,
    enable_hotkeys=enable_hotkeys,
    enable_wake_words=enable_wake_words,
  )
  manager.run()
