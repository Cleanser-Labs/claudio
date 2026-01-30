"""Trigger base class - defines interface for activation triggers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import msgspec


class TriggerConfig(msgspec.Struct, kw_only=True):
  """Base trigger configuration."""
  enabled: bool = True


class Trigger(ABC):
  """Base trigger interface.

  Triggers detect when to start/stop voice input.
  Examples: hotkey, wake word, voice activity, double-tap.
  """

  def __init__(self, config: TriggerConfig | None = None):
    self.config = config or TriggerConfig()
    self._on_activate: Callable[[], None] | None = None
    self._on_deactivate: Callable[[], None] | None = None
    self._running = False

  def on_activate(self, callback: Callable[[], None]) -> 'Trigger':
    """Set callback for when trigger activates (start listening)."""
    self._on_activate = callback
    return self

  def on_deactivate(self, callback: Callable[[], None]) -> 'Trigger':
    """Set callback for when trigger deactivates (stop listening)."""
    self._on_deactivate = callback
    return self

  def activate(self):
    """Fire the activate callback."""
    if self._on_activate:
      self._on_activate()

  def deactivate(self):
    """Fire the deactivate callback."""
    if self._on_deactivate:
      self._on_deactivate()

  @abstractmethod
  def start(self) -> None:
    """Start listening for trigger."""

  @abstractmethod
  def stop(self) -> None:
    """Stop listening for trigger."""

  @property
  def is_running(self) -> bool:
    return self._running

  def __enter__(self) -> 'Trigger':
    self.start()
    return self

  def __exit__(self, *_args) -> None:
    self.stop()
