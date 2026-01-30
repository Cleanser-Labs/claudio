"""Full session logging with pluggable storage backends.

Supports:
- JSONL (JSON Lines) - simple file-based logging
- SQLite - structured database storage
"""

from __future__ import annotations

import abc
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import msgspec

from .config import SessionLogging


class SessionRecord(msgspec.Struct, kw_only=True):
  """A log record for session events."""
  ts: str
  event: str
  session_id: str
  request_id: str | None = None
  data: dict = {}


class RequestRecord(msgspec.Struct, kw_only=True):
  """Full request record."""
  request_id: str
  session_id: str
  ts: str
  method: str
  path: str
  destination: str | None = None
  headers: dict = {}
  body: dict | str | None = None


class ResponseRecord(msgspec.Struct, kw_only=True):
  """Full response record."""
  request_id: str
  session_id: str
  ts: str
  status_code: int
  elapsed: float | None = None
  model: str | None = None
  streaming: bool = False
  total_chunks: int | None = None
  content: dict | str | None = None


class AudioRecord(msgspec.Struct, kw_only=True):
  """Audio generation record."""
  request_id: str
  session_id: str
  ts: str
  index: int
  text: str | None = None
  file_path: str | None = None
  size: int = 0


# Storage backend interface
class StorageBackend(abc.ABC):
  """Abstract storage backend for session logging."""

  @abc.abstractmethod
  def log_event(self, record: SessionRecord) -> None:
    """Log a session event."""
    pass

  @abc.abstractmethod
  def log_request(self, record: RequestRecord) -> None:
    """Log a full request."""
    pass

  @abc.abstractmethod
  def log_response(self, record: ResponseRecord) -> None:
    """Log a full response."""
    pass

  @abc.abstractmethod
  def log_audio(self, record: AudioRecord) -> None:
    """Log audio generation."""
    pass

  @abc.abstractmethod
  def close(self) -> None:
    """Close the storage backend."""
    pass

  def get_session_events(self, session_id: str) -> Iterator[SessionRecord]:
    """Get all events for a session (optional implementation)."""
    return iter([])

  def get_request(self, request_id: str) -> RequestRecord | None:
    """Get a request by ID (optional implementation)."""
    return None

  def get_response(self, request_id: str) -> ResponseRecord | None:
    """Get a response by ID (optional implementation)."""
    return None


class JsonlStorage(StorageBackend):
  """JSONL (JSON Lines) file storage backend."""

  def __init__(self, session_dir: Path):
    self.session_dir = session_dir
    self._encoder = msgspec.json.Encoder()
    self._lock = threading.Lock()

    # Create directories
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / 'requests').mkdir(exist_ok=True)
    (session_dir / 'responses').mkdir(exist_ok=True)

    # Open event log
    self._events_file = open(session_dir / 'events.jsonl', 'ab')

  def log_event(self, record: SessionRecord) -> None:
    with self._lock:
      self._events_file.write(self._encoder.encode(record) + b'\n')
      self._events_file.flush()

  def log_request(self, record: RequestRecord) -> None:
    request_file = self.session_dir / 'requests' / f'{record.request_id}.json'
    with open(request_file, 'wb') as f:
      f.write(self._encoder.encode(record))

  def log_response(self, record: ResponseRecord) -> None:
    response_file = self.session_dir / 'responses' / f'{record.request_id}.json'
    with open(response_file, 'wb') as f:
      f.write(self._encoder.encode(record))

  def log_audio(self, record: AudioRecord) -> None:
    # Audio metadata is logged as events
    self.log_event(SessionRecord(
      ts=record.ts,
      event='audio_saved',
      session_id=record.session_id,
      request_id=record.request_id,
      data={
        'index': record.index,
        'text': record.text[:100] if record.text else None,
        'file': record.file_path,
        'size': record.size,
      },
    ))

  def close(self) -> None:
    with self._lock:
      if self._events_file:
        self._events_file.close()

  def get_session_events(self, session_id: str) -> Iterator[SessionRecord]:
    events_file = self.session_dir / 'events.jsonl'
    if not events_file.exists():
      return
    decoder = msgspec.json.Decoder(SessionRecord)
    with open(events_file, 'rb') as f:
      for line in f:
        if line.strip():
          record = decoder.decode(line)
          if record.session_id == session_id:
            yield record

  def get_request(self, request_id: str) -> RequestRecord | None:
    request_file = self.session_dir / 'requests' / f'{request_id}.json'
    if not request_file.exists():
      return None
    decoder = msgspec.json.Decoder(RequestRecord)
    with open(request_file, 'rb') as f:
      return decoder.decode(f.read())

  def get_response(self, request_id: str) -> ResponseRecord | None:
    response_file = self.session_dir / 'responses' / f'{request_id}.json'
    if not response_file.exists():
      return None
    decoder = msgspec.json.Decoder(ResponseRecord)
    with open(response_file, 'rb') as f:
      return decoder.decode(f.read())


class SqliteStorage(StorageBackend):
  """SQLite database storage backend."""

  def __init__(self, db_path: Path):
    self.db_path = db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)

    self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
    self._lock = threading.Lock()
    self._encoder = msgspec.json.Encoder()

    self._init_schema()

  def _init_schema(self):
    """Initialize database schema."""
    with self._lock:
      self._conn.executescript('''
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          event TEXT NOT NULL,
          session_id TEXT NOT NULL,
          request_id TEXT,
          data TEXT
        );

        CREATE TABLE IF NOT EXISTS requests (
          request_id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          ts TEXT NOT NULL,
          method TEXT NOT NULL,
          path TEXT NOT NULL,
          destination TEXT,
          headers TEXT,
          body TEXT
        );

        CREATE TABLE IF NOT EXISTS responses (
          request_id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          ts TEXT NOT NULL,
          status_code INTEGER NOT NULL,
          elapsed REAL,
          model TEXT,
          streaming INTEGER DEFAULT 0,
          total_chunks INTEGER,
          content TEXT
        );

        CREATE TABLE IF NOT EXISTS audio (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          request_id TEXT NOT NULL,
          session_id TEXT NOT NULL,
          ts TEXT NOT NULL,
          idx INTEGER NOT NULL,
          text TEXT,
          file_path TEXT,
          size INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
        CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id);
        CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id);
        CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
        CREATE INDEX IF NOT EXISTS idx_audio_request ON audio(request_id);
      ''')
      self._conn.commit()

  def log_event(self, record: SessionRecord) -> None:
    with self._lock:
      self._conn.execute(
        'INSERT INTO events (ts, event, session_id, request_id, data) VALUES (?, ?, ?, ?, ?)',
        (record.ts, record.event, record.session_id, record.request_id,
         self._encoder.encode(record.data).decode() if record.data else None),
      )
      self._conn.commit()

  def log_request(self, record: RequestRecord) -> None:
    with self._lock:
      body_json = None
      if record.body:
        if isinstance(record.body, dict):
          body_json = self._encoder.encode(record.body).decode()
        else:
          body_json = record.body

      self._conn.execute(
        '''INSERT OR REPLACE INTO requests
           (request_id, session_id, ts, method, path, destination, headers, body)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (record.request_id, record.session_id, record.ts, record.method,
         record.path, record.destination,
         self._encoder.encode(record.headers).decode() if record.headers else None,
         body_json),
      )
      self._conn.commit()

  def log_response(self, record: ResponseRecord) -> None:
    with self._lock:
      content_json = None
      if record.content:
        if isinstance(record.content, dict):
          content_json = self._encoder.encode(record.content).decode()
        else:
          content_json = record.content

      self._conn.execute(
        '''INSERT OR REPLACE INTO responses
           (request_id, session_id, ts, status_code, elapsed, model, streaming, total_chunks, content)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (record.request_id, record.session_id, record.ts, record.status_code,
         record.elapsed, record.model, 1 if record.streaming else 0,
         record.total_chunks, content_json),
      )
      self._conn.commit()

  def log_audio(self, record: AudioRecord) -> None:
    with self._lock:
      self._conn.execute(
        '''INSERT INTO audio (request_id, session_id, ts, idx, text, file_path, size)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (record.request_id, record.session_id, record.ts, record.index,
         record.text, record.file_path, record.size),
      )
      self._conn.commit()

  def close(self) -> None:
    with self._lock:
      self._conn.close()

  def get_session_events(self, session_id: str) -> Iterator[SessionRecord]:
    with self._lock:
      cursor = self._conn.execute(
        'SELECT ts, event, session_id, request_id, data FROM events WHERE session_id = ? ORDER BY id',
        (session_id,),
      )
      for row in cursor:
        data = json.loads(row[4]) if row[4] else {}
        yield SessionRecord(
          ts=row[0],
          event=row[1],
          session_id=row[2],
          request_id=row[3],
          data=data,
        )

  def get_request(self, request_id: str) -> RequestRecord | None:
    with self._lock:
      cursor = self._conn.execute(
        'SELECT request_id, session_id, ts, method, path, destination, headers, body FROM requests WHERE request_id = ?',
        (request_id,),
      )
      row = cursor.fetchone()
      if not row:
        return None
      return RequestRecord(
        request_id=row[0],
        session_id=row[1],
        ts=row[2],
        method=row[3],
        path=row[4],
        destination=row[5],
        headers=json.loads(row[6]) if row[6] else {},
        body=json.loads(row[7]) if row[7] and row[7].startswith(('{', '[')) else row[7],
      )

  def get_response(self, request_id: str) -> ResponseRecord | None:
    with self._lock:
      cursor = self._conn.execute(
        'SELECT request_id, session_id, ts, status_code, elapsed, model, streaming, total_chunks, content FROM responses WHERE request_id = ?',
        (request_id,),
      )
      row = cursor.fetchone()
      if not row:
        return None
      return ResponseRecord(
        request_id=row[0],
        session_id=row[1],
        ts=row[2],
        status_code=row[3],
        elapsed=row[4],
        model=row[5],
        streaming=bool(row[6]),
        total_chunks=row[7],
        content=json.loads(row[8]) if row[8] and row[8].startswith(('{', '[')) else row[8],
      )


class SessionLogger:
  """Log full session data with pluggable storage backends.

  Storage options:
  - jsonl: JSON Lines files in directory structure
  - sqlite: Single SQLite database file

  Directory structure (for jsonl):
    sessions/
      {date}/
        {session_id}/
          events.jsonl        - Event log
          requests/
            {request_id}.json - Full request body
          responses/
            {request_id}.json - Full response content
          audio/
            {request_id}_{index}.wav - Generated audio files
  """

  def __init__(
    self,
    config: SessionLogging | None = None,
    session_id: str | None = None,
  ):
    self.config = config or SessionLogging()
    self.session_id = session_id or str(uuid.uuid4())[:8]
    self._lock = threading.Lock()
    self._audio_counter: dict[str, int] = {}
    self._storage: StorageBackend | None = None
    self._session_dir: Path | None = None
    self._audio_dir: Path | None = None

    if self.config.enabled:
      self._init_storage()

  def _init_storage(self):
    """Initialize storage backend."""
    base_dir = Path(self.config.dir)
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Determine storage type from config or file extension
    storage_type = getattr(self.config, 'storage', 'jsonl')

    if storage_type == 'sqlite':
      # SQLite: single database file per day
      db_dir = base_dir / date_str
      db_dir.mkdir(parents=True, exist_ok=True)
      db_path = db_dir / 'sessions.db'
      self._storage = SqliteStorage(db_path)
      self._session_dir = db_dir

      # Audio still goes to filesystem
      self._audio_dir = db_dir / 'audio' / self.session_id
      if self.config.save_audio:
        self._audio_dir.mkdir(parents=True, exist_ok=True)
    else:
      # JSONL: directory structure
      self._session_dir = base_dir / date_str / self.session_id
      self._storage = JsonlStorage(self._session_dir)

      # Audio directory
      if self.config.save_audio:
        self._audio_dir = self._session_dir / 'audio'
        self._audio_dir.mkdir(parents=True, exist_ok=True)

  @property
  def session_dir(self) -> Path | None:
    """Get the session directory path."""
    return self._session_dir

  @property
  def storage(self) -> StorageBackend | None:
    """Get the storage backend."""
    return self._storage

  def log(self, event: str, request_id: str | None = None, **data):
    """Log an event with optional data."""
    if not self.config.enabled or not self._storage:
      return

    record = SessionRecord(
      ts=datetime.now().isoformat(),
      event=event,
      session_id=self.session_id,
      request_id=request_id,
      data=data,
    )
    self._storage.log_event(record)

  def log_request(
    self,
    request_id: str,
    method: str,
    path: str,
    headers: dict,
    body: bytes | dict | None,
    destination: str | None = None,
  ):
    """Log a full request with body."""
    if not self.config.enabled or not self.config.log_requests or not self._storage:
      return

    # Log event
    self.log(
      'request',
      request_id=request_id,
      method=method,
      path=path,
      destination=destination,
      content_length=len(body) if body else 0,
    )

    # Parse body
    parsed_body: dict | str | None = None
    if body:
      if isinstance(body, dict):
        parsed_body = body
      else:
        try:
          parsed_body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
          parsed_body = body.decode('utf-8', errors='replace') if isinstance(body, bytes) else body

      # Check size limit
      body_size = len(body) if isinstance(body, bytes) else len(json.dumps(body))
      if self.config.max_request_size > 0 and body_size > self.config.max_request_size:
        self.log('request_truncated', request_id=request_id, original_size=body_size)
        return

    # Filter sensitive headers
    safe_headers = {k: v for k, v in headers.items()
                    if k.lower() not in ('authorization', 'x-api-key')}

    record = RequestRecord(
      request_id=request_id,
      session_id=self.session_id,
      ts=datetime.now().isoformat(),
      method=method,
      path=path,
      destination=destination,
      headers=safe_headers,
      body=parsed_body,
    )
    self._storage.log_request(record)

  def log_response(
    self,
    request_id: str,
    status_code: int,
    headers: dict | None,
    body: bytes | dict | str | None,
    elapsed: float | None = None,
    model: str | None = None,
  ):
    """Log a full response with body."""
    if not self.config.enabled or not self.config.log_responses or not self._storage:
      return

    # Parse body
    parsed_body: dict | str | None = None
    if body:
      if isinstance(body, dict):
        parsed_body = body
      elif isinstance(body, str):
        try:
          parsed_body = json.loads(body) if body.strip().startswith(('{', '[')) else body
        except json.JSONDecodeError:
          parsed_body = body
      else:
        try:
          parsed_body = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
          parsed_body = body.decode('utf-8', errors='replace')

      # Check size limit
      body_size = len(body) if isinstance(body, bytes) else len(str(body))
      if self.config.max_response_size > 0 and body_size > self.config.max_response_size:
        self.log('response_truncated', request_id=request_id, original_size=body_size)
        return

    # Log event
    self.log(
      'response',
      request_id=request_id,
      status_code=status_code,
      elapsed=elapsed,
      model=model,
    )

    record = ResponseRecord(
      request_id=request_id,
      session_id=self.session_id,
      ts=datetime.now().isoformat(),
      status_code=status_code,
      elapsed=elapsed,
      model=model,
      content=parsed_body,
    )
    self._storage.log_response(record)

  def log_streaming_complete(
    self,
    request_id: str,
    total_chunks: int,
    full_content: str | None = None,
    model: str | None = None,
    elapsed: float | None = None,
  ):
    """Log streaming completion with accumulated content."""
    if not self.config.enabled or not self._storage:
      return

    self.log(
      'stream_complete',
      request_id=request_id,
      total_chunks=total_chunks,
      model=model,
      elapsed=elapsed,
      content_length=len(full_content) if full_content else 0,
    )

    if self.config.log_responses:
      record = ResponseRecord(
        request_id=request_id,
        session_id=self.session_id,
        ts=datetime.now().isoformat(),
        status_code=200,
        elapsed=elapsed,
        model=model,
        streaming=True,
        total_chunks=total_chunks,
        content=full_content,
      )
      self._storage.log_response(record)

  def save_audio(
    self,
    request_id: str,
    audio_bytes: bytes,
    text: str | None = None,
    index: int | None = None,
  ) -> Path | None:
    """Save generated audio to file.

    Returns path to saved audio file.
    """
    if not self.config.enabled or not self.config.save_audio or not self._audio_dir:
      return None

    # Auto-increment index if not provided
    if index is None:
      with self._lock:
        index = self._audio_counter.get(request_id, 0)
        self._audio_counter[request_id] = index + 1

    audio_file = self._audio_dir / f'{request_id}_{index:03d}.wav'

    try:
      with open(audio_file, 'wb') as f:
        f.write(audio_bytes)

      # Log audio record
      if self._storage:
        record = AudioRecord(
          request_id=request_id,
          session_id=self.session_id,
          ts=datetime.now().isoformat(),
          index=index,
          text=text[:500] if text else None,
          file_path=str(audio_file),
          size=len(audio_bytes),
        )
        self._storage.log_audio(record)

      return audio_file
    except Exception as e:
      self.log('audio_save_error', request_id=request_id, error=str(e))
      return None

  def new_request_id(self) -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())[:8]

  def close(self):
    """Close the session logger."""
    if self._storage:
      self._storage.close()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()


# Global session logger
_session_logger: SessionLogger | None = None


def get_session_logger(
  config: SessionLogging | None = None,
  session_id: str | None = None,
) -> SessionLogger:
  """Get or create the global session logger."""
  global _session_logger

  if _session_logger is None:
    # Check for config from environment
    if config is None:
      log_dir = os.environ.get('CLAUDIO_SESSION_DIR', '.claudio/sessions')
      storage = os.environ.get('CLAUDIO_SESSION_STORAGE', 'jsonl')
      config = SessionLogging(
        enabled=os.environ.get('CLAUDIO_SESSION_LOG', '1') == '1',
        dir=log_dir,
      )
      # Add storage type dynamically
      object.__setattr__(config, 'storage', storage)

    _session_logger = SessionLogger(config, session_id)

  return _session_logger


def reset_session_logger():
  """Reset the global session logger (for testing)."""
  global _session_logger
  if _session_logger:
    _session_logger.close()
  _session_logger = None
