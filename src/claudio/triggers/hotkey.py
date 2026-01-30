"""Hotkey trigger - global keyboard shortcut activation."""

from __future__ import annotations

import threading
from typing import Set

import msgspec

from .base import Trigger, TriggerConfig


class HotkeyConfig(TriggerConfig, kw_only=True):
  """Hotkey trigger configuration."""
  # Key combo as string: 'cmd+shift+d', 'ctrl+alt+space', etc.
  hotkey: str = 'cmd+shift+d'
  # Toggle mode: press once to start, again to stop
  toggle: bool = True
  # Push-to-talk mode: hold to talk, release to stop
  push_to_talk: bool = False


class HotkeyTrigger(Trigger):
  """Global hotkey trigger using pynput.

  Listens for keyboard shortcuts even when app isn't focused.
  Requires Accessibility permission on macOS.
  """

  def __init__(self, config: HotkeyConfig | None = None):
    super().__init__(config)
    self.config: HotkeyConfig = config or HotkeyConfig()
    self._listener: threading.Thread | None = None
    self._active = False
    self._current_keys: Set = set()
    self._hotkey_keys = self._parse_hotkey(self.config.hotkey)

  def _parse_hotkey(self, hotkey: str) -> set:
    """Parse hotkey string like 'cmd+shift+d' into key set."""
    from pynput.keyboard import Key, KeyCode

    key_map = {
      'cmd': Key.cmd,
      'command': Key.cmd,
      'ctrl': Key.ctrl,
      'control': Key.ctrl,
      'alt': Key.alt,
      'option': Key.alt,
      'shift': Key.shift,
      'space': Key.space,
      'enter': Key.enter,
      'return': Key.enter,
      'tab': Key.tab,
      'esc': Key.esc,
      'escape': Key.esc,
    }

    keys = set()
    for part in hotkey.lower().split('+'):
      part = part.strip()
      if part in key_map:
        keys.add(key_map[part])
      elif len(part) == 1:
        keys.add(KeyCode.from_char(part))
      else:
        raise ValueError(f'Unknown key: {part}')
    return keys

  def _on_press(self, key):
    """Handle key press."""
    self._current_keys.add(key)

    if self._hotkey_keys.issubset(self._current_keys):
      if self.config.push_to_talk:
        # Push-to-talk: activate on press
        if not self._active:
          self._active = True
          self.activate()
      else:
        # Toggle mode: flip state on press
        if self._active:
          self._active = False
          self.deactivate()
        else:
          self._active = True
          self.activate()

  def _on_release(self, key):
    """Handle key release."""
    self._current_keys.discard(key)

    if self.config.push_to_talk and self._active:
      # Push-to-talk: deactivate on release
      if not self._hotkey_keys.issubset(self._current_keys):
        self._active = False
        self.deactivate()

  def start(self) -> None:
    """Start listening for hotkey."""
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
    """Stop listening for hotkey."""
    if not self._running:
      return

    self._running = False
    if self._listener:
      self._listener.stop()
      self._listener = None
    self._current_keys.clear()
    self._active = False
