"""Claude API proxy with auto-TTS for streaming responses.

Supports:
- Multi-destination routing (OpenAI, Anthropic, custom)
- Full session logging (requests, responses, audio)
- YAML configuration via claudio.yaml
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import httpx
import msgspec
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from claudio import tts
from claudio.config import (
  ClaudioConfig,
  Destination,
  Router,
  load_config,
  DEFAULT_CONFIG,
)
from claudio.session_logger import SessionLogger, get_session_logger

# Suppress Soprano vocoder warnings
warnings.filterwarnings('ignore', category=UserWarning, module='soprano')

console = Console()


class LogRecord(msgspec.Struct):
  """A log record."""
  ts: str
  event: str
  data: dict = {}


class JsonLogger:
  """Log events to a JSONL file."""

  def __init__(self, path: Path | None = None):
    self.path = path
    self._file = None
    self._encoder = msgspec.json.Encoder()
    if path:
      self._file = open(path, 'ab')  # binary mode for msgspec

  def log(self, event: str, **data):
    """Log an event with data."""
    if not self._file:
      return
    record = LogRecord(
      ts=datetime.now().isoformat(),
      event=event,
      data=data,
    )
    self._file.write(self._encoder.encode(record) + b'\n')
    self._file.flush()

  def close(self):
    if self._file:
      self._file.close()


# Global logger
_logger: JsonLogger | None = None


def get_logger() -> JsonLogger:
  global _logger
  if _logger is None:
    log_file = os.environ.get('CLAUDIO_LOG_FILE')
    _logger = JsonLogger(Path(log_file) if log_file else None)
  return _logger


class ProxyConfig(msgspec.Struct, kw_only=True):
  """Proxy configuration."""
  anthropic_url: str = 'https://api.anthropic.com'
  listen_host: str = '127.0.0.1'
  listen_port: int = 9000
  # Speak mode: 'auto' (all text), 'tags' (<say>...</say> only), 'off'
  speak_mode: str = 'tags'
  speak_code: bool = False  # Don't speak code blocks
  min_sentence_length: int = 10  # Min chars before speaking
  # Notify: 'tools', 'approval', or comma-separated tool names
  notify: str | None = None


class ProxyStats:
  """Track proxy statistics."""

  def __init__(self):
    self.started_at: float = time.time()
    self.request_count: int = 0
    self.total_chars_spoken: int = 0
    self.total_tts_time: float = 0.0
    self.models_seen: set[str] = set()
    self.last_model: str | None = None
    self.last_request_at: float = 0.0
    self._lock = threading.Lock()

  def record_request(self, model: str | None = None):
    with self._lock:
      self.request_count += 1
      self.last_request_at = time.time()
      if model:
        self.models_seen.add(model)
        self.last_model = model

  def record_tts(self, chars: int, tts_time: float):
    with self._lock:
      self.total_chars_spoken += chars
      self.total_tts_time += tts_time

  def to_dict(self) -> dict:
    uptime = time.time() - self.started_at
    with self._lock:
      return {
        'started_at': self.started_at,
        'started_at_iso': datetime.fromtimestamp(self.started_at).isoformat(),
        'uptime_seconds': uptime,
        'uptime_human': _format_uptime(uptime),
        'request_count': self.request_count,
        'total_chars_spoken': self.total_chars_spoken,
        'total_tts_time': self.total_tts_time,
        'models_seen': list(self.models_seen),
        'last_model': self.last_model,
        'last_request_at': self.last_request_at,
      }


def _format_uptime(seconds: float) -> str:
  """Format uptime in human-readable form."""
  if seconds < 60:
    return f'{int(seconds)}s'
  elif seconds < 3600:
    mins = int(seconds / 60)
    secs = int(seconds % 60)
    return f'{mins}m {secs}s'
  else:
    hours = int(seconds / 3600)
    mins = int((seconds % 3600) / 60)
    return f'{hours}h {mins}m'


def is_json_like(text: str) -> bool:
  """Check if text looks like JSON rather than natural language."""
  stripped = text.strip()
  # Check for JSON object/array patterns
  if stripped.startswith(('{', '[')) or stripped.endswith(('}', ']')):
    return True
  # Check for JSON key-value patterns
  if re.match(r'^\s*"[^"]+"\s*:', stripped):
    return True
  # Check for common structured patterns
  if stripped.lower() in ('none', 'null', 'true', 'false'):
    return True
  return False


class SpeechRequest(msgspec.Struct):
  """A request to speak text with optional parameters."""
  text: str
  tone: str = 'neutral'  # neutral, calm, excited, whisper
  speed: float = 1.0     # 0.5 to 2.0
  voice: str = 'default' # default, narrator, reader


# Speed word mappings
SPEED_MAP = {'slow': 0.75, 'fast': 1.5, 'slower': 0.5, 'faster': 2.0}


def parse_speed(val: str) -> float:
  """Parse speed value (word or number)."""
  val = val.lower()
  if val in SPEED_MAP:
    return SPEED_MAP[val]
  try:
    return float(val)
  except ValueError:
    return 1.0




# Patterns for <say> tags
SAY_OPEN = re.compile(r'<say(\s+[^>]*)?\s*>', re.IGNORECASE)
SAY_CLOSE = re.compile(r'</say\s*>', re.IGNORECASE)
SAY_BLOCK = re.compile(r'<say(\s+[^>]*)?\s*>(.*?)</say\s*>', re.IGNORECASE | re.DOTALL)


def parse_xml_attrs(tag: str) -> dict:
  """Parse attributes from XML tag like '<say speed=\"fast\">'."""
  attrs = {}
  for match in re.finditer(r'(\w+)=["\']([^"\']*)["\']', tag):
    attrs[match.group(1).lower()] = match.group(2)
  return attrs


def extract_speech_requests(text: str) -> list[SpeechRequest]:
  """Extract speech requests from <say>...</say> tags."""
  requests = []
  for match in SAY_BLOCK.finditer(text):
    attrs_str, content = match.groups()
    attrs = parse_xml_attrs(f'<say{attrs_str or ""}>') if attrs_str else {}
    content = ' '.join(content.split())
    if content:
      requests.append(SpeechRequest(
        text=content,
        tone=attrs.get('tone', 'neutral'),
        speed=parse_speed(attrs.get('speed', '1.0')),
        voice=attrs.get('voice', 'default'),
      ))
  return requests


def strip_say_tags(text: str, strip_invisible: bool = True) -> str:
  """Remove <say> tags but keep their content.

  If strip_invisible=True, also removes content of <say visible="false"> tags entirely.
  """
  if strip_invisible:
    # First, remove invisible blocks entirely (content and tags)
    text = re.sub(
      r'<say\s+[^>]*visible\s*=\s*["\']false["\'][^>]*>.*?</say\s*>',
      '',
      text,
      flags=re.IGNORECASE | re.DOTALL,
    )
  # Remove remaining opening tags
  text = SAY_OPEN.sub('', text)
  # Remove closing tags
  text = SAY_CLOSE.sub('', text)
  return text


class SentenceBuffer:
  """Buffer that emits complete sentences for TTS."""

  # Sentence-ending patterns
  SENTENCE_END = re.compile(r'[.!?]\s*$')
  CODE_BLOCK = re.compile(r'```')

  def __init__(self, on_sentence: callable):
    self.on_sentence = on_sentence
    self._buffer = ''
    self._in_code_block = False

  def add(self, text: str):
    """Add text to buffer, emit sentences as they complete."""
    self._buffer += text

    # Track code blocks
    code_matches = list(self.CODE_BLOCK.finditer(self._buffer))
    if len(code_matches) % 2 == 1:
      self._in_code_block = True
      return
    else:
      self._in_code_block = False

    # Don't emit while in code block
    if self._in_code_block:
      return

    # Find sentence boundaries
    while True:
      match = self.SENTENCE_END.search(self._buffer)
      if match:
        # Emit the sentence
        sentence = self._buffer[:match.end()].strip()
        self._buffer = self._buffer[match.end():]

        # Skip code blocks and very short sentences
        if sentence and not sentence.startswith('```'):
          self.on_sentence(sentence)
      else:
        break

  def flush(self):
    """Flush remaining buffer."""
    if self._buffer.strip() and not self._in_code_block:
      self.on_sentence(self._buffer.strip())
      self._buffer = ''


class MarkerBuffer:
  """Buffer that streams sentences from within <say>...</say> tags."""

  SENTENCE_END = re.compile(r'[.!?]\s+')

  def __init__(self, on_speech: callable):
    self.on_speech = on_speech
    self._buffer = ''
    self._in_tag = False
    self._attrs = {}

  def add(self, text: str):
    """Add text, stream sentences from tags."""
    self._buffer += text

    while True:
      if not self._in_tag:
        # Look for opening tag
        match = SAY_OPEN.search(self._buffer)
        if match:
          self._attrs = parse_xml_attrs(match.group(0))
          self._in_tag = True
          self._buffer = self._buffer[match.end():]
          continue

        # Keep tail in case of partial tag
        if len(self._buffer) > 30:
          self._buffer = self._buffer[-30:]
        break
      else:
        # Inside tag - look for closing or sentences
        close_match = SAY_CLOSE.search(self._buffer)

        if close_match:
          remaining = self._buffer[:close_match.start()].strip()
          if remaining:
            self._emit(remaining)
          self._buffer = self._buffer[close_match.end():]
          self._in_tag = False
          self._attrs = {}
        else:
          # Emit sentences as they complete
          match = self.SENTENCE_END.search(self._buffer)
          if match:
            sentence = self._buffer[:match.end()].strip()
            if sentence:
              self._emit(sentence)
            self._buffer = self._buffer[match.end():]
          else:
            break

  def _emit(self, text: str):
    """Emit text with current tag attributes."""
    if not text:
      return
    text = ' '.join(text.split())
    req = SpeechRequest(
      text=text,
      tone=self._attrs.get('tone', 'neutral'),
      speed=parse_speed(self._attrs.get('speed', '1.0')),
      voice=self._attrs.get('voice', 'default'),
    )
    self.on_speech(req)

  def flush(self):
    """Flush any remaining text in tag."""
    if self._in_tag:
      text = SAY_CLOSE.sub('', self._buffer).strip()
      text = ' '.join(text.split())
      if text:
        self._emit(text)
    self._buffer = ''
    self._in_tag = False


class TTSQueue:
  """Two-stage TTS pipeline: generate ahead while playing."""

  def __init__(
    self,
    engine: tts.TTS,
    logger: JsonLogger | None = None,
    coordinator=None,
    session_id: str = 'default',
    priority: int = 50,
    interruptible: bool = True,
    session_logger: SessionLogger | None = None,
  ):
    self.engine = engine
    self.logger = logger
    self.coordinator = coordinator
    self.session_id = session_id
    self.priority = priority
    self.interruptible = interruptible
    self.session_logger = session_logger

    # Stage 1: text -> audio
    self._text_queue: list[SpeechRequest | str] = []
    self._text_lock = threading.Lock()
    # Stage 2: audio -> playback
    self._audio_queue: list[tuple[bytes, str, str | None]] = []  # (audio_bytes, text, request_id)
    self._audio_lock = threading.Lock()

    self._running = True
    self._current_request_id: str | None = None

    # Start both worker threads
    self._gen_thread = threading.Thread(target=self._generator, daemon=True)
    self._play_thread = threading.Thread(target=self._player, daemon=True)
    self._gen_thread.start()
    self._play_thread.start()

    # Stats
    self.total_chars = 0
    self.total_time = 0.0
    self.utterances = 0
    self.captured_text: list[str] = []
    self.captured_chars = 0

  def set_request_id(self, request_id: str | None):
    """Set current request ID for audio logging."""
    self._current_request_id = request_id

  def add(self, item: SpeechRequest | str):
    """Add text or speech request to queue."""
    with self._text_lock:
      self._text_queue.append(item)
      text = item.text if isinstance(item, SpeechRequest) else item
      self.captured_text.append(text)
      self.captured_chars += len(text)
      if self.logger:
        self.logger.log('tts_queued', text=text[:50], queue_size=len(self._text_queue))

  def add_requests(self, requests: list[SpeechRequest]):
    """Add multiple speech requests."""
    for req in requests:
      self.add(req)

  def stop(self):
    """Stop the queue."""
    self._running = False

  def wait(self, timeout: float = 30.0) -> bool:
    """Wait for queues to drain. Returns True if drained, False if timeout."""
    import time
    start = time.time()
    while time.time() - start < timeout:
      with self._text_lock:
        text_empty = len(self._text_queue) == 0
      with self._audio_lock:
        audio_empty = len(self._audio_queue) == 0
      if text_empty and audio_empty:
        return True
      time.sleep(0.1)
    return False

  def is_idle(self) -> bool:
    """Check if both queues are empty."""
    with self._text_lock:
      text_empty = len(self._text_queue) == 0
    with self._audio_lock:
      audio_empty = len(self._audio_queue) == 0
    return text_empty and audio_empty

  def reset_stats(self):
    """Reset stats for a new request."""
    self.total_chars = 0
    self.total_time = 0.0
    self.utterances = 0
    self.captured_text = []
    self.captured_chars = 0

  def get_stats(self) -> dict:
    """Get current stats."""
    return {
      'chars': self.captured_chars,
      'tts_time': self.total_time,
      'utterances': self.utterances,
      'text': ''.join(self.captured_text),
    }

  def _generator(self):
    """Generate audio from text, queue for playback."""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='soprano')

    while self._running:
      item = None
      with self._text_lock:
        if self._text_queue:
          item = self._text_queue.pop(0)

      if item:
        try:
          if isinstance(item, SpeechRequest):
            text, speed, voice = item.text, item.speed, item.voice
          else:
            text, speed, voice = item, 1.0, 'default'

          # Capture request_id for this audio
          request_id = self._current_request_id

          # Generate audio bytes
          v = None if voice == 'default' else voice
          if self.logger:
            self.logger.log('tts_generate_start', text=text[:50])
          audio_bytes = self.engine.generate(text, voice=v, rate=speed)
          if self.logger:
            self.logger.log('tts_generate_done', bytes=len(audio_bytes))
          # Skip empty audio (non-pronounceable text)
          if audio_bytes:
            with self._audio_lock:
              self._audio_queue.append((audio_bytes, text, request_id))
        except Exception as e:
          if self.logger:
            self.logger.log('tts_generate_error', error=str(e), error_type=type(e).__name__)
          console.print(f'[red]TTS gen error:[/red] {e}')
      else:
        time.sleep(0.02)

  def _player(self):
    """Play audio from queue."""
    while self._running:
      audio_item = None
      with self._audio_lock:
        if self._audio_queue:
          audio_item = self._audio_queue.pop(0)

      if audio_item:
        audio_bytes, text, request_id = audio_item
        try:
          # Save audio to session log if configured
          if self.session_logger and request_id:
            self.session_logger.save_audio(request_id, audio_bytes, text=text)

          # Acquire speech priority if coordinator is set
          if self.coordinator:
            acquired = self.coordinator.acquire(
              self.session_id,
              self.priority,
              interruptible=self.interruptible,
              timeout=30.0,
            )
            if not acquired:
              if self.logger:
                self.logger.log('tts_priority_timeout', session=self.session_id)
              continue

          start = time.time()
          if self.logger:
            self.logger.log('tts_play_start', text=text[:50], session=self.session_id)
          try:
            self.engine.play(audio_bytes)
          finally:
            # Always release priority after playing
            if self.coordinator:
              self.coordinator.release(self.session_id)

          elapsed = time.time() - start
          if self.logger:
            self.logger.log('tts_play_done', elapsed=elapsed)
          self.total_chars += len(text)
          self.total_time += elapsed
          self.utterances += 1
        except Exception as e:
          if self.logger:
            self.logger.log('tts_play_error', error=str(e), error_type=type(e).__name__)
          console.print(f'[red]TTS play error:[/red] {e}')
          # Release priority on error
          if self.coordinator:
            self.coordinator.release(self.session_id)
      else:
        time.sleep(0.02)


async def proxy_stream(
  request_body: dict,
  config: ProxyConfig,
  tts_queue: TTSQueue | None,
  headers: dict,
  url: str,
  tracker: ResponseTracker,
  logger: JsonLogger,
  session_logger: SessionLogger | None = None,
  request_id: str | None = None,
) -> AsyncIterator[bytes]:
  """Proxy a streaming request to Claude API, with optional TTS."""

  # Set up buffer based on speak mode
  buffer = None
  if config.speak_mode == 'auto' and tts_queue:
    buffer = SentenceBuffer(on_sentence=tts_queue.add)
  elif config.speak_mode == 'tags' and tts_queue:
    buffer = MarkerBuffer(on_speech=tts_queue.add)

  async with httpx.AsyncClient() as client:
    async with client.stream(
      'POST',
      url,
      headers=headers,
      json=request_body,
      timeout=300.0,
    ) as response:
      # Only process for successful responses
      process = response.status_code == 200

      # Accumulator for stripping tags across chunks
      tag_buffer = ''

      async for chunk in response.aiter_bytes():
        if not process or config.speak_mode == 'off':
          yield chunk
          continue

        # Decode and process chunk
        chunk_text = chunk.decode('utf-8', errors='ignore')
        modified_lines = []

        for line in chunk_text.split('\n'):
          if not line.startswith('data: '):
            modified_lines.append(line)
            continue

          payload = line[6:].strip()
          if not payload or payload == '[DONE]':
            modified_lines.append(line)
            continue

          try:
            data = json.loads(payload)
            event_type = data.get('type', '')

            # Strip say tags from text deltas
            if event_type == 'content_block_delta':
              delta = data.get('delta', {})
              if delta.get('type') == 'text_delta':
                text = delta.get('text', '')
                if text:
                  # Accumulate and strip tags
                  tag_buffer += text
                  stripped = strip_say_tags(tag_buffer)
                  # Keep potential partial tags in buffer
                  if '<' in tag_buffer and '>' not in tag_buffer[tag_buffer.rfind('<'):]:
                    # Partial opening tag at end
                    last_lt = tag_buffer.rfind('<')
                    stripped = strip_say_tags(tag_buffer[:last_lt])
                    tag_buffer = tag_buffer[last_lt:]
                  else:
                    tag_buffer = ''
                  delta['text'] = stripped
                  data['delta'] = delta

            modified_lines.append(f'data: {json.dumps(data)}')
          except json.JSONDecodeError:
            modified_lines.append(line)

        # Yield modified chunk
        modified_chunk = '\n'.join(modified_lines)
        yield modified_chunk.encode('utf-8')

        # Now parse for TTS (using original data)
        try:
          for line in chunk_text.split('\n'):
            if not line.startswith('data: '):
              continue
            payload = line[6:].strip()
            if not payload or payload == '[DONE]':
              continue
            data = json.loads(payload)
            event_type = data.get('type', '')

            if event_type == 'message_start':
              msg = data.get('message', {})
              tracker.model = msg.get('model', '')

            elif event_type == 'content_block_start':
              idx = data.get('index', 0)
              block = data.get('content_block', {})
              tracker.start_block(idx, block)
              # Log block start
              if block.get('type') == 'text':
                logger.log('block_start', index=idx, type='text')

            elif event_type == 'content_block_delta':
              idx = data.get('index', 0)
              delta = data.get('delta', {})
              tracker.add_delta(idx, delta)

              # Log text deltas
              if delta.get('type') == 'text_delta':
                text = delta.get('text', '')
                if text:
                  logger.log('text', content=text)

              # TTS for text deltas in text blocks
              if buffer and idx in tracker.blocks:
                block = tracker.blocks[idx]
                if block.type == 'text' and delta.get('type') == 'text_delta':
                  text = delta.get('text', '')
                  if text:
                    # For auto mode, skip JSON-like content
                    if isinstance(buffer, SentenceBuffer) and is_json_like(block.text):
                      pass
                    else:
                      try:
                        buffer.add(text)
                      except Exception as e:
                        logger.log('error', type='buffer', message=str(e))

            elif event_type == 'content_block_stop':
              logger.log('block_stop', index=data.get('index', 0))

            elif event_type == 'message_delta':
              delta = data.get('delta', {})
              tracker.stop_reason = delta.get('stop_reason', '')

        except Exception as e:
          logger.log('error', type='parse', message=str(e))

      # Flush remaining buffer
      if buffer:
        buffer.flush()


class ContentBlock:
  """A content block from the response."""
  def __init__(self, index: int, block_type: str):
    self.index = index
    self.type = block_type
    self.text = ''
    self.tool_name = ''
    self.tool_id = ''
    self.tool_input = ''


class ResponseTracker:
  """Track response content for logging."""
  def __init__(self):
    self.blocks: dict[int, ContentBlock] = {}
    self.model = ''
    self.stop_reason = ''

  def start_block(self, index: int, block: dict):
    block_type = block.get('type', 'unknown')
    cb = ContentBlock(index, block_type)
    if block_type == 'tool_use':
      cb.tool_name = block.get('name', '')
      cb.tool_id = block.get('id', '')
    self.blocks[index] = cb

  def add_delta(self, index: int, delta: dict):
    if index not in self.blocks:
      return
    cb = self.blocks[index]
    delta_type = delta.get('type', '')
    if delta_type == 'text_delta':
      cb.text += delta.get('text', '')
    elif delta_type == 'input_json_delta':
      cb.tool_input += delta.get('partial_json', '')

  def get_text_content(self) -> str:
    """Get only text content for TTS."""
    return ''.join(b.text for b in self.blocks.values() if b.type == 'text')


def should_notify(tool_name: str, notify: str | None) -> bool:
  """Check if we should notify for this tool."""
  if not notify:
    return False
  if notify == 'tools':
    return True
  if notify == 'approval':
    # Notify for tools that might require approval
    return tool_name in ('Edit', 'Write', 'Bash', 'NotebookEdit')
  # Check if tool name is in comma-separated list
  return tool_name in [t.strip() for t in notify.split(',')]


def log_response(
  tracker: ResponseTracker,
  elapsed: float,
  tts_stats: dict,
  logger: JsonLogger,
  config: 'ProxyConfig | None' = None,
  tts_queue: 'TTSQueue | None' = None,
):
  """Log response stats after streaming."""
  is_child = os.environ.get('CLAUDIO_CHILD') == '1'

  # Log tool use blocks
  for idx in sorted(tracker.blocks.keys()):
    block = tracker.blocks[idx]
    if block.type == 'tool_use' and block.tool_name:
      logger.log('tool_use', name=block.tool_name, input=block.tool_input[:500])

      # Notify via TTS if configured
      if config and tts_queue and should_notify(block.tool_name, config.notify):
        tts_queue.add(f"Using {block.tool_name}")

      if not is_child:
        input_display = block.tool_input[:200] + '...' if len(block.tool_input) > 200 else block.tool_input
        panel = Panel(
          Text(input_display or '{}', style='yellow'),
          title=f'[bold yellow]tool_use[/bold yellow] [dim]{block.tool_name}[/dim]',
          border_style='yellow',
          padding=(0, 1),
        )
        console.print(panel)

  # Log stats
  model_short = tracker.model.split('-')[1] if '-' in tracker.model else tracker.model
  chars = tts_stats.get('chars', 0)
  tts_time = tts_stats.get('tts_time', 0)

  logger.log('response_end',
    model=tracker.model,
    elapsed=elapsed,
    chars=chars,
    tts_time=tts_time,
    stop_reason=tracker.stop_reason,
  )

  if not is_child:
    stats = f"  [blue]{model_short}[/blue] [green]{elapsed:.1f}s[/green]"
    if chars > 0:
      stats += f" [dim]|[/dim] [yellow]{chars} chars[/yellow]"
      if tts_time > 0:
        stats += f" [dim]in[/dim] [yellow]{tts_time:.1f}s[/yellow]"
    if tracker.stop_reason:
      stats += f" [dim]({tracker.stop_reason})[/dim]"
    console.print(stats)
    console.print()


async def run_proxy(
  config: ProxyConfig | None = None,
  output_device: int | None = None,
  tts_backend: str = 'auto',
  tts_voice: str | None = None,
  tts_speed: float = 1.0,
  persona_id: str | None = None,
  personas_dir: str | None = None,
  session_id: str = 'default',
  config_file: str | None = None,
):
  """Run the proxy server.

  Args:
    config: Proxy configuration
    output_device: Audio output device ID
    tts_backend: TTS backend to use
    tts_voice: Voice name (overridden by persona if set)
    tts_speed: Speech speed (overridden by persona if set)
    persona_id: Persona to use for this session
    personas_dir: Directory containing persona markdown files
    session_id: Session ID for speech coordination
    config_file: Path to claudio.yaml config file
  """
  from pathlib import Path
  from starlette.applications import Starlette
  from starlette.responses import StreamingResponse, JSONResponse, Response, HTMLResponse
  from starlette.routing import Route
  from starlette.requests import Request

  from claudio.personas import PersonaStore, VoiceStore, get_scheduler

  # Load YAML configuration if available
  yaml_config: ClaudioConfig | None = None
  router: Router | None = None
  try:
    yaml_config = load_config(config_file)
    if yaml_config.destinations:
      router = Router(yaml_config)
      # Override settings from YAML config if not explicitly provided
      if config is None:
        config = ProxyConfig(
          listen_host=yaml_config.proxy.host,
          listen_port=yaml_config.proxy.port,
          speak_mode=yaml_config.proxy.speak_mode,
          notify=yaml_config.proxy.notify,
        )
      if tts_backend == 'auto' and yaml_config.tts.backend != 'auto':
        tts_backend = yaml_config.tts.backend
      if tts_voice is None and yaml_config.tts.voice:
        tts_voice = yaml_config.tts.voice
      if tts_speed == 1.0 and yaml_config.tts.speed != 1.0:
        tts_speed = yaml_config.tts.speed
      if output_device is None and yaml_config.tts.device is not None:
        output_device = yaml_config.tts.device
  except Exception as e:
    console.print(f'[yellow]Config load warning:[/yellow] {e}')

  config = config or ProxyConfig()

  # Initialize loggers
  logger = get_logger()

  # Initialize session logger for full request/response/audio logging
  session_logger: SessionLogger | None = None
  if yaml_config and yaml_config.logging.enabled:
    session_logger = get_session_logger(yaml_config.logging, session_id)
    console.print(f'[green]Session logging:[/green] {session_logger.session_dir}')

  # Load voice store
  voice_store = None
  for path in [
    Path.cwd() / 'voices',
    Path.cwd() / '.claudio' / 'voices',
    Path.home() / '.claudio' / 'voices',
  ]:
    if path.exists():
      voice_store = VoiceStore(path)
      break

  # Load persona if specified
  persona = None
  priority = 50  # Default middle priority
  interruptible = True
  if persona_id:
    # Try to load persona
    search_paths = [
      Path(personas_dir) if personas_dir else None,
      Path.cwd() / 'personas',
      Path.cwd() / '.claudio' / 'personas',
      Path.home() / '.claudio' / 'personas',
    ]
    for path in search_paths:
      if path and path.exists():
        store = PersonaStore(path)
        persona = store.get(persona_id)
        if persona:
          break

    if persona:
      priority = persona.config.priority
      interruptible = persona.config.interruptible

      # Resolve voice from persona preferences
      if voice_store and persona.config.voices:
        voice = voice_store.select(
          preferences=persona.config.voices,
          fallback_traits=persona.config.fallback,
        )
        if voice:
          tts_voice = voice.config.voice_id or voice.config.name
          tts_speed = voice.config.speed
          if voice.config.model:
            tts_backend = voice.config.model

      # Persona speed override
      if persona.config.speed:
        tts_speed = persona.config.speed

      console.print(f'[green]Using persona:[/green] {persona.config.name} (priority {priority})')
    else:
      console.print(f'[yellow]Persona not found:[/yellow] {persona_id}')

  # Get speech scheduler for priority management
  scheduler = get_scheduler()

  # Initialize stats tracking
  proxy_stats = ProxyStats()

  # Initialize TTS
  tts_config_kw = dict(backend=tts_backend, output_device=output_device, rate=tts_speed)
  if tts_voice is not None:
    tts_config_kw['voice'] = tts_voice
  tts_engine = tts.create(tts.Config(**tts_config_kw))
  tts_queue = TTSQueue(
    tts_engine,
    logger=logger,
    coordinator=scheduler,
    session_id=session_id,
    priority=priority,
    interruptible=interruptible,
    session_logger=session_logger,
  )

  async def proxy_messages(request: Request):
    """Proxy /v1/messages endpoint."""
    request_start = time.time()

    body_bytes = await request.body()
    if not body_bytes:
      return JSONResponse({'error': 'Empty request body'}, status_code=400)
    body = json.loads(body_bytes)

    # Generate request ID for session logging
    request_id = session_logger.new_request_id() if session_logger else None

    # Record request
    proxy_stats.record_request(body.get('model'))

    # Find destination using router or use default
    path = str(request.url.path)
    destination = router.match(path) if router else None
    dest_url = destination.url if destination else config.anthropic_url
    dest_name = destination.name if destination else 'anthropic'

    # Build URL with query params
    url = f'{dest_url}{path}'
    if request.url.query:
      url = f'{url}?{request.url.query}'

    # Forward all headers except hop-by-hop headers
    skip_headers = {'host', 'connection', 'keep-alive', 'transfer-encoding', 'te', 'trailer', 'upgrade'}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}

    # Add destination-specific headers
    if destination and destination.headers:
      headers.update(destination.headers)

    # Add API key from environment if configured
    if destination and destination.api_key_env:
      api_key = os.environ.get(destination.api_key_env)
      if api_key:
        # Determine header name based on destination
        if 'openai' in dest_name.lower():
          headers['Authorization'] = f'Bearer {api_key}'
        else:
          headers['x-api-key'] = api_key

    # Log request to session logger
    if session_logger and request_id:
      session_logger.log_request(
        request_id=request_id,
        method='POST',
        path=path,
        headers=dict(headers),
        body=body,
        destination=dest_name,
      )

    # Set request ID for audio logging
    if tts_queue:
      tts_queue.set_request_id(request_id)

    # Reset TTS stats
    tts_queue.reset_stats()

    # Check if streaming
    is_streaming = body.get('stream', False)

    # Check if TTS is enabled for this destination
    tts_enabled = destination.tts_enabled if destination else True
    active_tts_queue = tts_queue if tts_enabled else None

    if is_streaming:
      tracker = ResponseTracker()

      async def stream_with_logging():
        chunk_count = 0
        async for chunk in proxy_stream(body, config, active_tts_queue, dict(headers), url, tracker, logger, session_logger, request_id):
          chunk_count += 1
          yield chunk
        # Log after stream completes
        elapsed = time.time() - request_start
        tts_stats = tts_queue.get_stats() if active_tts_queue else {}
        log_response(tracker, elapsed, tts_stats, logger, config, active_tts_queue)
        # Log to session logger
        if session_logger and request_id:
          session_logger.log_streaming_complete(
            request_id=request_id,
            total_chunks=chunk_count,
            full_content=tracker.get_text_content(),
            model=tracker.model,
            elapsed=elapsed,
          )
        # Record TTS stats
        proxy_stats.record_tts(tts_stats.get('chars', 0), tts_stats.get('tts_time', 0))
        # Record model from response
        if tracker.model:
          proxy_stats.record_request(tracker.model)

      return StreamingResponse(stream_with_logging(), media_type='text/event-stream')
    else:
      # Non-streaming: build tracker manually from response
      tracker = ResponseTracker()

      async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=body, timeout=300.0)
        data = response.json()

        if response.status_code == 200:
          tracker.model = data.get('model', '')
          tracker.stop_reason = data.get('stop_reason', '')

          content = data.get('content', [])
          if isinstance(content, list):
            for idx, block in enumerate(content):
              if isinstance(block, dict):
                block_type = block.get('type', 'unknown')
                cb = ContentBlock(idx, block_type)
                if block_type == 'text':
                  cb.text = block.get('text', '')
                  # TTS based on speak mode (only if enabled for destination)
                  if active_tts_queue:
                    if cb.text and config.speak_mode == 'auto' and not is_json_like(cb.text):
                      active_tts_queue.add(cb.text)
                    elif cb.text and config.speak_mode == 'tags':
                      requests = extract_speech_requests(cb.text)
                      if requests:
                        active_tts_queue.add_requests(requests)
                elif block_type == 'tool_use':
                  cb.tool_name = block.get('name', '')
                  cb.tool_id = block.get('id', '')
                  cb.tool_input = json.dumps(block.get('input', {}))
                tracker.blocks[idx] = cb

        # Log response
        elapsed = time.time() - request_start
        tts_stats = tts_queue.get_stats() if active_tts_queue else {}
        log_response(tracker, elapsed, tts_stats, logger, config, active_tts_queue)

        # Log to session logger
        if session_logger and request_id:
          session_logger.log_response(
            request_id=request_id,
            status_code=response.status_code,
            headers=dict(response.headers),
            body=data,
            elapsed=elapsed,
            model=tracker.model,
          )

        # Record TTS stats
        proxy_stats.record_tts(tts_stats.get('chars', 0), tts_stats.get('tts_time', 0))
        # Record model from response
        if tracker.model:
          proxy_stats.record_request(tracker.model)

        return JSONResponse(data, status_code=response.status_code)

  async def dashboard(request: Request):
    """GET / - Status page."""
    stats = proxy_stats.to_dict()
    tts_name = type(tts_engine).__name__ if tts_engine else 'none'
    tts_repr = repr(tts_engine) if tts_engine else 'none'
    queue_idle = tts_queue.is_idle() if tts_queue else True

    # Destinations
    dest_rows = ''
    if router:
      for dest in router.all():
        tts_badge = '<span style="color:#4a4">tts</span>' if dest.tts_enabled else '<span style="color:#888">no-tts</span>'
        default_badge = ' <span style="color:#888">(default)</span>' if yaml_config and dest.name == yaml_config.default_destination else ''
        paths = ', '.join(dest.paths)
        dest_rows += f'<tr><td>{dest.name}{default_badge}</td><td>{dest.url}</td><td>{paths}</td><td>{tts_badge}</td></tr>\n'

    # Logging
    log_status = 'enabled' if session_logger else 'disabled'
    log_detail = ''
    if session_logger:
      log_detail = f'{yaml_config.logging.storage} &rarr; {session_logger.session_dir}'

    html = f'''<!doctype html>
<html><head>
<meta charset="utf-8">
<title>claudio proxy</title>
<style>
  body {{ font-family: ui-monospace, monospace; background: #1a1a1a; color: #e0e0e0; margin: 2em; line-height: 1.6; }}
  h1 {{ color: #fff; font-size: 1.2em; }}
  h2 {{ color: #aaa; font-size: 1em; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; }}
  td, th {{ padding: 0.2em 1em 0.2em 0; text-align: left; }}
  .dim {{ color: #888; }}
  .ok {{ color: #4a4; }}
  .val {{ color: #6bf; }}
  a {{ color: #6bf; }}
</style>
</head><body>
<h1>claudio proxy</h1>

<h2>tts</h2>
<table>
  <tr><td class="dim">engine</td><td class="val">{tts_repr}</td></tr>
  <tr><td class="dim">voice</td><td class="val">{tts_voice or 'default'}</td></tr>
  <tr><td class="dim">speed</td><td class="val">{tts_speed}x</td></tr>
  <tr><td class="dim">queue</td><td class="{"ok" if queue_idle else "val"}">{"idle" if queue_idle else "busy"}</td></tr>
  <tr><td class="dim">speak mode</td><td class="val">{config.speak_mode}</td></tr>
</table>

<h2>proxy</h2>
<table>
  <tr><td class="dim">requests</td><td class="val">{stats.get("total_requests", 0)}</td></tr>
  <tr><td class="dim">upstream</td><td class="val">{config.anthropic_url}</td></tr>
  <tr><td class="dim">session</td><td class="val">{session_id}</td></tr>
  <tr><td class="dim">persona</td><td class="val">{persona.config.name if persona else "none"}</td></tr>
</table>

{"<h2>destinations</h2><table><tr><th>name</th><th>url</th><th>paths</th><th>tts</th></tr>" + dest_rows + "</table>" if dest_rows else ""}

<h2>logging</h2>
<table>
  <tr><td class="dim">status</td><td class="val">{log_status}</td></tr>
  {"<tr><td class='dim'>storage</td><td class='val'>" + log_detail + "</td></tr>" if log_detail else ""}
</table>

<h2>endpoints</h2>
<table>
  <tr><td><a href="/health">/health</a></td><td class="dim">health check</td></tr>
  <tr><td><a href="/status">/status</a></td><td class="dim">status JSON</td></tr>
  <tr><td class="dim">/v1/messages</td><td class="dim">Claude API proxy</td></tr>
  <tr><td><a href="/wait">/wait</a></td><td class="dim">wait for TTS queue</td></tr>
</table>

</body></html>'''
    return HTMLResponse(html)

  async def health(request: Request):
    return JSONResponse({'status': 'ok', 'mode': 'proxy'})

  async def proxy_status(request: Request):
    """GET /status - Proxy status with stats."""
    stats = proxy_stats.to_dict()

    # Build destinations list
    destinations_info = []
    if router:
      for dest in router.all():
        destinations_info.append({
          'name': dest.name,
          'url': dest.url,
          'paths': dest.paths,
          'tts_enabled': dest.tts_enabled,
        })

    return JSONResponse({
      'status': 'ok',
      'mode': 'proxy',
      'proxy': stats,
      'tts': {
        'backend': type(tts_engine).__name__,
        'voice': tts_voice,
        'speed': tts_speed,
        'queue_idle': tts_queue.is_idle() if tts_queue else True,
      },
      'config': {
        'speak_mode': config.speak_mode,
        'notify': config.notify,
        'upstream': config.anthropic_url,
      },
      'routing': {
        'destinations': destinations_info,
        'default': yaml_config.default_destination if yaml_config else None,
      },
      'logging': {
        'enabled': session_logger is not None,
        'session_id': session_logger.session_id if session_logger else None,
        'session_dir': str(session_logger.session_dir) if session_logger else None,
        'storage': yaml_config.logging.storage if yaml_config else None,
      },
      'session': {
        'id': session_id,
        'priority': priority,
        'interruptible': interruptible,
        'persona': persona.config.name if persona else None,
      },
    })

  async def wait_tts(request: Request):
    """Wait for TTS queue to drain before shutdown."""
    if tts_queue:
      # Wait up to 30s for TTS to finish
      drained = tts_queue.wait(timeout=30.0)
      return JSONResponse({'status': 'drained' if drained else 'timeout'})
    return JSONResponse({'status': 'no_tts'})

  async def passthrough(request: Request):
    """Pass through any other request using multi-destination routing."""
    body = await request.body()

    # Find destination using router
    path = str(request.url.path)
    destination = router.match(path) if router else None
    dest_url = destination.url if destination else config.anthropic_url
    dest_name = destination.name if destination else 'anthropic'

    # Build URL with query params
    url = f'{dest_url}{path}'
    if request.url.query:
      url = f'{url}?{request.url.query}'

    # Forward all headers except hop-by-hop
    skip_headers = {'host', 'connection', 'keep-alive', 'transfer-encoding', 'te', 'trailer', 'upgrade'}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}

    # Add destination-specific headers
    if destination and destination.headers:
      headers.update(destination.headers)

    # Add API key from environment if configured
    if destination and destination.api_key_env:
      api_key = os.environ.get(destination.api_key_env)
      if api_key:
        if 'openai' in dest_name.lower():
          headers['Authorization'] = f'Bearer {api_key}'
        else:
          headers['x-api-key'] = api_key

    # Generate request ID and log if session logger enabled
    request_id = session_logger.new_request_id() if session_logger else None
    if session_logger and request_id:
      session_logger.log_request(
        request_id=request_id,
        method=request.method,
        path=path,
        headers=dict(headers),
        body=body if body else None,
        destination=dest_name,
      )

    request_start = time.time()
    async with httpx.AsyncClient() as client:
      response = await client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
        timeout=300.0,
      )

      # Log response if session logger enabled
      elapsed = time.time() - request_start
      if session_logger and request_id:
        session_logger.log_response(
          request_id=request_id,
          status_code=response.status_code,
          headers=dict(response.headers),
          body=response.content,
          elapsed=elapsed,
        )

      return Response(
        content=response.content,
        status_code=response.status_code,
      )

  routes = [
    Route('/', dashboard, methods=['GET']),
    Route('/v1/messages', proxy_messages, methods=['POST']),
    Route('/health', health, methods=['GET']),
    Route('/status', proxy_status, methods=['GET']),
    Route('/wait', wait_tts, methods=['GET']),
    Route('/{path:path}', passthrough, methods=['GET', 'POST', 'PUT', 'DELETE']),
  ]

  app = Starlette(routes=routes)

  import os
  import uvicorn

  # Check if we're running as a child process
  is_child = os.environ.get('CLAUDIO_CHILD') == '1'

  if not is_child:
    console.print()
    console.print('[bold green]Claudio Proxy[/bold green]')
    console.print(f'  Listening: [cyan]http://{config.listen_host}:{config.listen_port}[/cyan]')

    # Show routing/destinations
    if router and yaml_config and yaml_config.destinations:
      console.print('  Routing:')
      for dest in yaml_config.destinations:
        paths_str = ', '.join(dest.paths[:3])
        if len(dest.paths) > 3:
          paths_str += f' +{len(dest.paths) - 3} more'
        tts_indicator = '[green]TTS[/green]' if dest.tts_enabled else '[dim]no-TTS[/dim]'
        console.print(f'    [cyan]{dest.name}[/cyan] → [dim]{dest.url}[/dim] {tts_indicator}')
        console.print(f'      [dim]{paths_str}[/dim]')
    else:
      console.print(f'  Upstream:  [dim]{config.anthropic_url}[/dim]')

    console.print(f'  TTS:       [cyan]{type(tts_engine).__name__}[/cyan]' + (f' [dim]({tts_voice})[/dim]' if tts_voice else ''))
    mode_color = {'auto': 'yellow', 'tags': 'green', 'off': 'red'}.get(config.speak_mode, 'white')
    console.print(f'  Speak:     [{mode_color}]{config.speak_mode}[/{mode_color}]')
    if config.speak_mode == 'tags':
      console.print('             [dim]Use <say>text</say> in responses[/dim]')

    # Show session logging status
    if session_logger:
      storage_type = yaml_config.logging.storage if yaml_config else 'jsonl'
      console.print(f'  Logging:   [green]enabled[/green] [dim]({storage_type})[/dim]')
      console.print(f'             [dim]{session_logger.session_dir}[/dim]')

    console.print()
    console.print('[dim]Set ANTHROPIC_BASE_URL=http://{0}:{1}[/dim]'.format(config.listen_host, config.listen_port))
    console.print()

  # Speak startup greeting
  import random
  greetings = [
    # Normal greetings
    "Ready when you are.",
    "Voice enabled. ... What are we building?",
    # Retro/funny greetings
    "Bum, bum, bum, bummm... Windows 95 successfully loaded. ... Just kidding, I'm Claude.",
    "Pshhhkkkkkkrrrr, kaking, kaking, tsh, ch, ch, ch... ... dial-up complete. ... Welcome to the internet. It's 2025, we have AI now.",
    "Insert disk 2... of 47... to continue. ... Actually no, I'm ready. SSDs are nice.",
    "Reticulating splines... ... Adjusting flux capacitor... ... Okay I'm just stalling. Let's code.",
    "Beep boop. ... I am a robot. ... I am here to help. ... Beep. ... Okay I'll stop.",
    "Good morning starshine! ... The earth says hello!",
    "I'd make a startup sound, ... but my lawyers said no. ... So, ... hi!",
  ]
  tts_queue.add(random.choice(greetings))

  log_level = 'warning' if is_child else 'info'
  await uvicorn.Server(
    uvicorn.Config(app, host=config.listen_host, port=config.listen_port, log_level=log_level)
  ).serve()


def main():
  """CLI entry point for proxy."""
  import argparse

  parser = argparse.ArgumentParser(description='Claudio API Proxy')
  parser.add_argument('--config', '-c', default=None,
                      help='Path to claudio.yaml config file')
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', '-p', type=int, default=9000)
  parser.add_argument('--speak', '-s', choices=['auto', 'tags', 'off'], default='tags',
                      help='Speak mode: auto (all text), tags (<say>...</say> only), off')
  parser.add_argument('--tts', '-t', default='auto',
                      help='TTS backend: auto, say, soprano, kokoro, pocket')
  parser.add_argument('--voice', '-v', default=None,
                      help='Voice name (for say backend)')
  parser.add_argument('--speed', type=float, default=1.0,
                      help='Speech speed multiplier (0.5-2.0)')
  parser.add_argument('--notify', '-n', default=None,
                      help='Notify on: tools, approval, or comma-separated tool names')
  parser.add_argument('--device', '-d', type=int, default=None,
                      help='Audio output device ID')
  parser.add_argument('--persona', default=None,
                      help='Persona ID to use for voice settings')
  parser.add_argument('--personas-dir', default=None,
                      help='Directory containing persona markdown files')
  parser.add_argument('--session', default='default',
                      help='Session ID for speech coordination')
  args = parser.parse_args()

  config = ProxyConfig(
    listen_host=args.host,
    listen_port=args.port,
    speak_mode=args.speak,
    notify=args.notify,
  )

  asyncio.run(run_proxy(
    config,
    output_device=args.device,
    tts_backend=args.tts,
    tts_voice=args.voice,
    tts_speed=args.speed,
    persona_id=args.persona,
    personas_dir=args.personas_dir,
    session_id=args.session,
    config_file=args.config,
  ))


if __name__ == '__main__':
  main()
