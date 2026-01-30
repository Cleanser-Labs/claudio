"""Triggers - activation methods for voice input.

Triggers detect when to start/stop voice input:
- HotkeyTrigger: Global keyboard shortcuts (Cmd+Shift+D)
- DoubleTapTrigger: Double-tap modifier key (Ctrl, Option)
- WakeWordTrigger: Voice activation ('Hey Jarvis')
- VADTrigger: Voice activity detection (just start talking)
- CompositeTrigger: Combine multiple triggers

Example:
    from claudio.triggers import HotkeyTrigger, HotkeyConfig

    trigger = HotkeyTrigger(HotkeyConfig(hotkey='cmd+shift+d'))
    trigger.on_activate(start_dictation)
    trigger.on_deactivate(stop_dictation)
    trigger.start()
"""

from .base import Trigger, TriggerConfig
from .composite import CompositeTrigger
from .doubletap import DoubleTapConfig, DoubleTapTrigger
from .hotkey import HotkeyConfig, HotkeyTrigger
from .vad import VADConfig, VADTrigger
from .wakeword import WakeWordConfig, WakeWordTrigger

__all__ = [
  # Base
  'Trigger',
  'TriggerConfig',
  # Implementations
  'HotkeyTrigger',
  'HotkeyConfig',
  'DoubleTapTrigger',
  'DoubleTapConfig',
  'WakeWordTrigger',
  'WakeWordConfig',
  'VADTrigger',
  'VADConfig',
  'CompositeTrigger',
]


def get_trigger(name: str, **kwargs) -> Trigger:
  """Get a trigger by name with config kwargs.

  Args:
      name: Trigger name (hotkey, doubletap, wakeword, vad)
      **kwargs: Config parameters

  Example:
      trigger = get_trigger('hotkey', hotkey='cmd+shift+d')
      trigger = get_trigger('doubletap', key='ctrl')
      trigger = get_trigger('wakeword', backend='openwakeword')
  """
  triggers = {
    'hotkey': (HotkeyTrigger, HotkeyConfig),
    'doubletap': (DoubleTapTrigger, DoubleTapConfig),
    'wakeword': (WakeWordTrigger, WakeWordConfig),
    'vad': (VADTrigger, VADConfig),
  }

  if name not in triggers:
    raise ValueError(f'Unknown trigger: {name}. Options: {list(triggers.keys())}')

  trigger_cls, config_cls = triggers[name]
  config = config_cls(**kwargs)
  return trigger_cls(config)
