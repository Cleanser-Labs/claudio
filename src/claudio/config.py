"""YAML configuration for claudio proxy and services."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import msgspec


class Destination(msgspec.Struct, kw_only=True):
  """A proxy destination with path-based routing."""
  name: str
  url: str
  # Path patterns to match (supports glob-like patterns)
  # e.g., "/openai/*", "/v1/chat/*", "/anthropic/**"
  paths: list[str] = []
  # Headers to add to requests
  headers: dict[str, str] = {}
  # API key environment variable name
  api_key_env: str | None = None
  # Whether to enable TTS for this destination
  tts_enabled: bool = True


class SessionLogging(msgspec.Struct, kw_only=True):
  """Session logging configuration."""
  enabled: bool = True
  # Storage backend: 'jsonl' or 'sqlite'
  storage: str = 'jsonl'
  # Directory for session logs
  dir: str = '.claudio/sessions'
  # Log full request bodies
  log_requests: bool = True
  # Log full response content
  log_responses: bool = True
  # Save generated audio files
  save_audio: bool = True
  # Max request body size to log (bytes, 0 = unlimited)
  max_request_size: int = 0
  # Max response size to log (bytes, 0 = unlimited)
  max_response_size: int = 0


class TTSConfig(msgspec.Struct, kw_only=True):
  """TTS configuration."""
  backend: str = 'kokoro'
  voice: str | None = 'af_heart'
  speed: float = 1.0
  device: int | None = None


class ProxySettings(msgspec.Struct, kw_only=True):
  """General proxy settings."""
  host: str = '127.0.0.1'
  port: int = 9000
  # Speak mode: 'tags', 'text', 'all', 'off'
  speak_mode: str = 'tags'
  # Notify on tool use: 'tools', 'approval', or comma-separated names
  notify: str | None = None


class ClaudioConfig(msgspec.Struct, kw_only=True):
  """Root configuration for claudio.yaml."""
  # Proxy destinations with path-based routing
  destinations: list[Destination] = []
  # Default destination when no path matches
  default_destination: str | None = None
  # Session logging
  logging: SessionLogging = msgspec.field(default_factory=SessionLogging)
  # TTS settings
  tts: TTSConfig = msgspec.field(default_factory=TTSConfig)
  # Proxy settings
  proxy: ProxySettings = msgspec.field(default_factory=ProxySettings)


def _pattern_to_regex(pattern: str) -> re.Pattern:
  """Convert glob-like path pattern to regex.

  Supports:
    * - matches any characters except /
    ** - matches any characters including /
    ? - matches single character
  """
  # Escape regex special chars except our glob chars
  regex = ''
  i = 0
  while i < len(pattern):
    c = pattern[i]
    if c == '*':
      if i + 1 < len(pattern) and pattern[i + 1] == '*':
        regex += '.*'
        i += 2
      else:
        regex += '[^/]*'
        i += 1
    elif c == '?':
      regex += '.'
      i += 1
    elif c in '.^$+{}[]|()\\':
      regex += '\\' + c
      i += 1
    else:
      regex += c
      i += 1

  return re.compile(f'^{regex}$')


class Router:
  """Path-based router for multiple destinations."""

  def __init__(self, config: ClaudioConfig):
    self.config = config
    self._destinations: dict[str, Destination] = {}
    self._routes: list[tuple[re.Pattern, Destination]] = []
    self._default: Destination | None = None

    # Index destinations by name
    for dest in config.destinations:
      self._destinations[dest.name] = dest
      # Compile path patterns
      for path in dest.paths:
        pattern = _pattern_to_regex(path)
        self._routes.append((pattern, dest))

    # Set default destination
    if config.default_destination and config.default_destination in self._destinations:
      self._default = self._destinations[config.default_destination]
    elif config.destinations:
      self._default = config.destinations[0]

  def match(self, path: str) -> Destination | None:
    """Find destination for a given path."""
    for pattern, dest in self._routes:
      if pattern.match(path):
        return dest
    return self._default

  def get(self, name: str) -> Destination | None:
    """Get destination by name."""
    return self._destinations.get(name)

  def all(self) -> Iterator[Destination]:
    """Iterate all destinations."""
    yield from self.config.destinations


def load_config(path: Path | str | None = None) -> ClaudioConfig:
  """Load configuration from YAML file.

  Search order:
    1. Explicit path if provided
    2. ./claudio.yaml
    3. ./.claudio/config.yaml
    4. ~/.claudio/config.yaml
  """
  try:
    import yaml
  except ImportError:
    raise ImportError('pyyaml not installed. Run: uv sync --extra config')

  search_paths = []

  if path:
    search_paths.append(Path(path))
  else:
    search_paths.extend([
      Path.cwd() / 'claudio.yaml',
      Path.cwd() / '.claudio' / 'config.yaml',
      Path.home() / '.claudio' / 'config.yaml',
    ])

  for p in search_paths:
    if p.exists():
      with open(p) as f:
        data = yaml.safe_load(f) or {}
      return msgspec.convert(data, ClaudioConfig)

  # No config found — write default to ./claudio.yaml
  default_path = Path.cwd() / 'claudio.yaml'
  save_config(DEFAULT_CONFIG, default_path)
  print(f'Created default config: {default_path}')
  return msgspec.convert(msgspec.to_builtins(DEFAULT_CONFIG), ClaudioConfig)


def save_config(config: ClaudioConfig, path: Path | str) -> None:
  """Save configuration to YAML file."""
  try:
    import yaml
  except ImportError:
    raise ImportError('pyyaml not installed. Run: uv sync --extra config')

  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)

  # Convert to dict
  data = msgspec.to_builtins(config)

  with open(path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Default configuration example
DEFAULT_CONFIG = ClaudioConfig(
  destinations=[
    Destination(
      name='anthropic',
      url='https://api.anthropic.com',
      paths=['/v1/messages', '/v1/messages/*'],
      api_key_env='ANTHROPIC_API_KEY',
    ),
    Destination(
      name='openai',
      url='https://api.openai.com',
      paths=['/openai/*', '/v1/chat/*', '/v1/completions/*'],
      api_key_env='OPENAI_API_KEY',
      tts_enabled=False,
    ),
  ],
  default_destination='anthropic',
  logging=SessionLogging(
    enabled=False,
    dir='.claudio/sessions',
    log_requests=True,
    log_responses=True,
    save_audio=True,
  ),
  tts=TTSConfig(
    backend='pocket',
    voice='alba',
    speed=1.0,
  ),
  proxy=ProxySettings(
    host='127.0.0.1',
    port=9000,
    speak_mode='tags',
  ),
)
