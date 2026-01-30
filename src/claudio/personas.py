"""Voice and Persona system for Claudio.

Voices define TTS configuration (how speech sounds).
Personas define agent identity (behavior, voice preferences, priority).

Voice file format (voices/*.md):
```markdown
---
name: Rachel
model: eleven_labs
voice_id: rachel

# traits for selection/filtering
gender: female
accent: american
tone: warm
energy: calm
quality: high
latency: medium

# runtime defaults
speed: 1.0
pitch: 0
---

Warm, professional voice. Good for customer-facing.
```

Persona file format (personas/*.md):
```markdown
---
name: Alert
voices: [alert-custom, rachel, nova]
priority: 80
interruptible: false
speed: 1.3
---

You are an alert system. Keep messages terse.
```
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import msgspec


# =============================================================================
# YAML Parsing (simple, no PyYAML dependency)
# =============================================================================

FRONTMATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n(.*)$', re.DOTALL)
LIST_ITEM_RE = re.compile(r'^\s*-\s*(.+)$')
INLINE_LIST_RE = re.compile(r'^\[([^\]]*)\]$')


def parse_yaml_simple(yaml_str: str) -> dict[str, Any]:
  """Parse simple YAML (key: value pairs, lists).

  Supports:
  - key: value
  - key: [item1, item2]
  - key:
      - item1
      - item2
  """
  result: dict[str, Any] = {}
  current_key: str | None = None
  current_list: list[str] | None = None

  for line in yaml_str.split('\n'):
    stripped = line.strip()

    # Skip empty lines and comments
    if not stripped or stripped.startswith('#'):
      if current_list is not None:
        # Empty line ends list
        result[current_key] = current_list
        current_key = None
        current_list = None
      continue

    # Check for list item
    list_match = LIST_ITEM_RE.match(line)
    if list_match and current_key is not None:
      if current_list is None:
        current_list = []
      current_list.append(_parse_value(list_match.group(1).strip()))
      continue

    # If we were building a list, save it
    if current_list is not None:
      result[current_key] = current_list
      current_key = None
      current_list = None

    # Parse key: value
    if ':' not in stripped:
      continue

    key, _, value = stripped.partition(':')
    key = key.strip()
    value = value.strip()

    if not value:
      # Might be start of a list
      current_key = key
      current_list = None  # Will be created when we see first item
      continue

    # Check for inline list [a, b, c]
    inline_match = INLINE_LIST_RE.match(value)
    if inline_match:
      items = [_parse_value(v.strip()) for v in inline_match.group(1).split(',') if v.strip()]
      result[key] = items
    else:
      result[key] = _parse_value(value)

  # Save any pending list
  if current_list is not None and current_key is not None:
    result[current_key] = current_list

  return result


def _parse_value(value: str) -> Any:
  """Parse a YAML value to appropriate Python type."""
  # Remove quotes
  if (value.startswith('"') and value.endswith('"')) or \
     (value.startswith("'") and value.endswith("'")):
    return value[1:-1]

  # Booleans
  if value.lower() in ('true', 'yes'):
    return True
  if value.lower() in ('false', 'no'):
    return False

  # Null
  if value.lower() in ('null', 'none', '~', ''):
    return None

  # Numbers
  try:
    if '.' in value:
      return float(value)
    return int(value)
  except ValueError:
    pass

  return value


def parse_frontmatter(content: str, config_cls: type) -> tuple[Any, str]:
  """Parse YAML frontmatter from markdown content.

  Returns (config, body).
  """
  match = FRONTMATTER_RE.match(content)
  if not match:
    return config_cls(), content

  yaml_str, body = match.groups()
  data = parse_yaml_simple(yaml_str)

  # Convert to config class, ignoring unknown fields
  config_data = {}
  for field in config_cls.__struct_fields__:
    if field in data:
      config_data[field] = data[field]

  return config_cls(**config_data), body


# =============================================================================
# Voice Configuration
# =============================================================================

class VoiceConfig(msgspec.Struct, kw_only=True):
  """Voice configuration from frontmatter."""
  name: str = 'default'

  # Provider settings
  model: str | None = None  # TTS model/provider: eleven_labs, kokoro, say, etc.
  voice_id: str | None = None  # Provider-specific voice identifier
  sample: str | None = None  # Path to sample audio for cloning

  # Traits for selection/filtering
  gender: str | None = None  # male, female, neutral
  accent: str | None = None  # american, british, australian, etc.
  age: str | None = None  # child, young, adult, senior
  tone: str | None = None  # warm, cold, urgent, calm, playful
  energy: str | None = None  # low, medium, high
  quality: str = 'medium'  # high, medium, low - fidelity
  latency: str = 'medium'  # low, medium, high - time to first byte
  cost: str = 'medium'  # low, medium, high

  # Runtime defaults (overridable by persona/session)
  speed: float = 1.0
  pitch: float = 0.0
  stability: float | None = None  # Provider-specific


class Voice(msgspec.Struct, kw_only=True):
  """A voice with config and description."""
  id: str  # filename without extension
  config: VoiceConfig
  description: str  # Body text (voice prompt or documentation)
  path: Path | None = None

  @classmethod
  def from_markdown(cls, path: Path) -> 'Voice':
    """Load a voice from a markdown file."""
    content = path.read_text()
    config, description = parse_frontmatter(content, VoiceConfig)
    return cls(
      id=path.stem,
      config=config,
      description=description.strip(),
      path=path,
    )

  def matches_traits(self, traits: dict[str, str]) -> bool:
    """Check if this voice matches the given trait requirements."""
    for key, value in traits.items():
      voice_value = getattr(self.config, key, None)
      if voice_value is not None and voice_value != value:
        return False
    return True


class VoiceStore:
  """Manages loading and selecting voices."""

  def __init__(self, directory: Path | str | None = None):
    self.directory = Path(directory) if directory else None
    self._voices: dict[str, Voice] = {}
    self._lock = threading.Lock()

    if self.directory and self.directory.exists():
      self.reload()

  def reload(self) -> None:
    """Reload voices from directory."""
    if not self.directory or not self.directory.exists():
      return

    with self._lock:
      self._voices.clear()
      for path in self.directory.glob('*.md'):
        try:
          voice = Voice.from_markdown(path)
          self._voices[voice.id] = voice
        except Exception as e:
          print(f'Warning: Failed to load voice {path}: {e}')

  def get(self, voice_id: str) -> Voice | None:
    """Get a voice by ID."""
    with self._lock:
      return self._voices.get(voice_id)

  def list(self) -> list[Voice]:
    """List all voices."""
    with self._lock:
      return list(self._voices.values())

  def select(
    self,
    preferences: list[str] | None = None,
    fallback_traits: dict[str, str] | None = None,
    active_models: set[str] | None = None,
  ) -> Voice | None:
    """Select a voice based on preferences and constraints.

    Selection logic:
    1. Walk preferences list in order
    2. If active_models set, prefer voices using already-active models
    3. Fall back to first preference or trait-matched voice
    """
    with self._lock:
      if not self._voices:
        return None

      # Try explicit preferences first
      if preferences:
        # First pass: prefer voices with already-active models
        if active_models:
          for voice_id in preferences:
            voice = self._voices.get(voice_id)
            if voice and voice.config.model in active_models:
              return voice

        # Second pass: any matching preference
        for voice_id in preferences:
          voice = self._voices.get(voice_id)
          if voice:
            return voice

      # Try trait-based fallback
      if fallback_traits:
        for voice in self._voices.values():
          if voice.matches_traits(fallback_traits):
            return voice

      # Return first available voice
      return next(iter(self._voices.values()), None)

  def __contains__(self, voice_id: str) -> bool:
    with self._lock:
      return voice_id in self._voices

  def __len__(self) -> int:
    with self._lock:
      return len(self._voices)


# =============================================================================
# Persona Configuration
# =============================================================================

class PersonaConfig(msgspec.Struct, kw_only=True):
  """Persona configuration from frontmatter."""
  name: str = 'default'

  # Voice selection (preference order)
  voices: list[str] | None = None  # List of voice IDs in preference order
  fallback: dict[str, str] | None = None  # Trait requirements for fallback

  # Scheduling
  priority: int = 50  # 0-100, higher speaks first
  interruptible: bool = True  # Can higher priority cut in mid-utterance?

  # Voice overrides (runtime params)
  speed: float | None = None
  pitch: float | None = None


class Persona(msgspec.Struct, kw_only=True):
  """A persona with config and system prompt."""
  id: str  # filename without extension
  config: PersonaConfig
  prompt: str  # System prompt (body text)
  path: Path | None = None

  @classmethod
  def from_markdown(cls, path: Path) -> 'Persona':
    """Load a persona from a markdown file."""
    content = path.read_text()
    config, prompt = parse_frontmatter(content, PersonaConfig)
    return cls(
      id=path.stem,
      config=config,
      prompt=prompt.strip(),
      path=path,
    )

  @classmethod
  def default(cls) -> 'Persona':
    """Create default persona."""
    return cls(
      id='default',
      config=PersonaConfig(),
      prompt='',
    )


class PersonaStore:
  """Manages loading and caching personas."""

  def __init__(self, directory: Path | str | None = None):
    self.directory = Path(directory) if directory else None
    self._personas: dict[str, Persona] = {}
    self._lock = threading.Lock()

    if self.directory and self.directory.exists():
      self.reload()

  def reload(self) -> None:
    """Reload personas from directory."""
    if not self.directory or not self.directory.exists():
      return

    with self._lock:
      self._personas.clear()
      for path in self.directory.glob('*.md'):
        try:
          persona = Persona.from_markdown(path)
          self._personas[persona.id] = persona
        except Exception as e:
          print(f'Warning: Failed to load persona {path}: {e}')

  def get(self, persona_id: str) -> Persona | None:
    """Get a persona by ID."""
    with self._lock:
      return self._personas.get(persona_id)

  def list(self) -> list[Persona]:
    """List all personas."""
    with self._lock:
      return list(self._personas.values())

  def add(self, persona: Persona) -> None:
    """Add a persona programmatically."""
    with self._lock:
      self._personas[persona.id] = persona

  def __contains__(self, persona_id: str) -> bool:
    with self._lock:
      return persona_id in self._personas

  def __len__(self) -> int:
    with self._lock:
      return len(self._personas)


# =============================================================================
# Speech Scheduler
# =============================================================================

class Utterance(msgspec.Struct):
  """A single speech utterance in the queue."""
  session_id: str
  text: str
  priority: int  # Effective priority (higher = higher)
  interruptible: bool
  enqueued_at: float
  voice_id: str | None = None


class Scheduler:
  """Coordinates speech across sessions by priority.

  - One session holds speech lock at a time
  - When lock released, highest priority waiting session acquires
  - If higher priority session queues while lower is speaking:
    - If speaker is interruptible → interrupt, yield lock
    - Else → wait for current utterance to finish
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._condition = threading.Condition(self._lock)
    self._speaking: Utterance | None = None
    self._interrupted = False
    self._active_models: set[str] = set()

  @property
  def active_models(self) -> set[str]:
    """Get currently active TTS models."""
    with self._lock:
      return self._active_models.copy()

  def set_model_active(self, model: str) -> None:
    """Mark a model as currently active."""
    with self._lock:
      self._active_models.add(model)

  def set_model_inactive(self, model: str) -> None:
    """Mark a model as no longer active."""
    with self._lock:
      self._active_models.discard(model)

  def acquire(
    self,
    session_id: str,
    priority: int,
    interruptible: bool = True,
    timeout: float | None = None,
  ) -> bool:
    """Acquire the speech lock for a session.

    Returns True if acquired, False if timed out.
    Higher priority (higher number) sessions speak first.
    """
    with self._condition:
      start = time.time()

      while True:
        # If no one is speaking, we can speak
        if self._speaking is None:
          self._speaking = Utterance(
            session_id=session_id,
            text='',
            priority=priority,
            interruptible=interruptible,
            enqueued_at=time.time(),
          )
          self._interrupted = False
          return True

        # If we're already speaking, allow re-entry
        if self._speaking.session_id == session_id:
          return True

        # Check if we should interrupt current speaker
        if priority > self._speaking.priority and self._speaking.interruptible:
          # Signal interruption and take over
          self._interrupted = True
          self._speaking = Utterance(
            session_id=session_id,
            text='',
            priority=priority,
            interruptible=interruptible,
            enqueued_at=time.time(),
          )
          self._condition.notify_all()
          return True

        # Wait for current speaker to finish
        remaining = None
        if timeout is not None:
          elapsed = time.time() - start
          remaining = timeout - elapsed
          if remaining <= 0:
            return False

        self._condition.wait(timeout=remaining or 1.0)

  def release(self, session_id: str) -> None:
    """Release the speech lock."""
    with self._condition:
      if self._speaking and self._speaking.session_id == session_id:
        self._speaking = None
        self._interrupted = False
        self._condition.notify_all()

  def was_interrupted(self) -> bool:
    """Check if the current session was interrupted."""
    with self._lock:
      return self._interrupted

  def is_speaking(self, session_id: str) -> bool:
    """Check if a session is currently speaking."""
    with self._lock:
      return self._speaking is not None and self._speaking.session_id == session_id

  def current_speaker(self) -> str | None:
    """Get the current speaker's session ID."""
    with self._lock:
      return self._speaking.session_id if self._speaking else None

  def current_priority(self) -> int | None:
    """Get the current speaker's priority."""
    with self._lock:
      return self._speaking.priority if self._speaking else None


# =============================================================================
# Session Runtime
# =============================================================================

class Session(msgspec.Struct, kw_only=True):
  """Runtime session for a persona."""
  id: str
  persona_id: str | None = None
  priority_override: int | None = None  # Overrides persona priority if set
  voice_id: str | None = None  # Resolved voice selection
  state: str = 'idle'  # idle, queued, speaking

  def effective_priority(self, persona: Persona | None = None) -> int:
    """Get effective priority (override or persona default)."""
    if self.priority_override is not None:
      return self.priority_override
    if persona:
      return persona.config.priority
    return 50  # Default middle priority


# =============================================================================
# Global State & Factories
# =============================================================================

DEFAULT_VOICES_DIR = Path.home() / '.claudio' / 'voices'
DEFAULT_PERSONAS_DIR = Path.home() / '.claudio' / 'personas'

_voice_store: VoiceStore | None = None
_persona_store: PersonaStore | None = None
_scheduler: Scheduler | None = None


def get_voice_store(directory: Path | str | None = None) -> VoiceStore:
  """Get or create the global voice store."""
  global _voice_store
  if _voice_store is None:
    _voice_store = VoiceStore(directory or DEFAULT_VOICES_DIR)
  return _voice_store


def get_persona_store(directory: Path | str | None = None) -> PersonaStore:
  """Get or create the global persona store."""
  global _persona_store
  if _persona_store is None:
    _persona_store = PersonaStore(directory or DEFAULT_PERSONAS_DIR)
  return _persona_store


def get_scheduler() -> Scheduler:
  """Get or create the global scheduler."""
  global _scheduler
  if _scheduler is None:
    _scheduler = Scheduler()
  return _scheduler


# Backwards compatibility aliases
get_store = get_persona_store
get_coordinator = get_scheduler
SpeechCoordinator = Scheduler
