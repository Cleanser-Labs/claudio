"""Double-tap trigger - activate by tapping a key twice quickly."""

from __future__ import annotations

import time

from .base import Trigger, TriggerConfig


class DoubleTapConfig(TriggerConfig, kw_only=True):
  """Double-tap trigger configuration."""
  # Key to double-tap: ctrl, alt/option, shift, cmd, fn
  key: str = 'ctrl'
  # Maximum time between taps (seconds)
  max_interval: float = 0.4
  # Toggle mode: double-tap to start, double-tap again to stop
  toggle: bool = True


class DoubleTapTrigger(Trigger):
  """Double-tap trigger.

  Activates when a modifier key is tapped twice quickly.
  Common pattern used by apps like Alfred and Raycast.
  """

  def __init__(self, config: DoubleTapConfig | None = None):
    super().__init__(config)
    self.config: DoubleTapConfig = config or DoubleTapConfig()
    self._listener = None
    self._last_tap: float = 0
    self._tap_count: int = 0
    self._active = False
    self._target_key = self._parse_key(self.config.key)

  def _parse_key(self, key: str):
    """Parse key name to pynput Key."""
    from pynput.keyboard import Key

    key_map = {
      'ctrl': Key.ctrl,
      'control': Key.ctrl,
      'alt': Key.alt,
      'option': Key.alt,
      'shift': Key.shift,
      'cmd': Key.cmd,
      'command': Key.cmd,
      'fn': Key.f1,  # fn key not directly accessible, use f1 as placeholder
    }

    key_lower = key.lower()
    if key_lower not in key_map:
      raise ValueError(f'Unknown modifier key: {key}. Use: ctrl, alt, shift, cmd')
    return key_map[key_lower]

  def _on_press(self, _key):
    """Handle key press."""
    pass  # We only care about release for tap detection

  def _on_release(self, key):
    """Handle key release - detect taps."""
    if key != self._target_key:
      return

    now = time.time()

    # Check if this is part of a double-tap sequence
    if now - self._last_tap < self.config.max_interval:
      self._tap_count += 1
    else:
      self._tap_count = 1

    self._last_tap = now

    # Double-tap detected
    if self._tap_count >= 2:
      self._tap_count = 0

      if self.config.toggle:
        if self._active:
          self._active = False
          self.deactivate()
        else:
          self._active = True
          self.activate()
      else:
        # Non-toggle mode: always activate
        self.activate()

  def start(self) -> None:
    """Start listening for double-tap."""
    if self._running:
      return

    from pynput import keyboard

    self._running = True
    self._listener = keyboard.Listener(
      on_press=self._on_press,
      on_release=self._on_release,
    )
    self._listener.start()

  def stop(self) -> None:
    """Stop listening for double-tap."""
    if not self._running:
      return

    self._running = False
    if self._listener:
      self._listener.stop()
      self._listener = None
    self._tap_count = 0
    self._active = False
