"""Composite trigger - combine multiple triggers."""

from __future__ import annotations

from .base import Trigger


class CompositeTrigger(Trigger):
  """Combine multiple triggers.

  Any of the child triggers can activate/deactivate.
  Useful for having multiple ways to start dictation.

  Example:
      trigger = CompositeTrigger([
          HotkeyTrigger(HotkeyConfig(hotkey='cmd+shift+d')),
          DoubleTapTrigger(DoubleTapConfig(key='ctrl')),
      ])
  """

  def __init__(self, triggers: list[Trigger]):
    super().__init__()
    self.triggers = triggers

    # Wire up all child triggers to our callbacks
    for t in self.triggers:
      t.on_activate(self._child_activated)
      t.on_deactivate(self._child_deactivated)

  def _child_activated(self):
    """Called when any child trigger activates."""
    self.activate()

  def _child_deactivated(self):
    """Called when any child trigger deactivates."""
    self.deactivate()

  def start(self) -> None:
    """Start all child triggers."""
    if self._running:
      return

    self._running = True
    for t in self.triggers:
      t.start()

  def stop(self) -> None:
    """Stop all child triggers."""
    if not self._running:
      return

    self._running = False
    for t in self.triggers:
      t.stop()
