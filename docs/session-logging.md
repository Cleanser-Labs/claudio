# Session Logging

Claudio supports full session logging to capture all proxy requests, responses, and generated audio for debugging, analytics, and replay.

## Quick Start

Enable logging in `claudio.yaml`:

```yaml
logging:
  enabled: true
  storage: jsonl  # or 'sqlite'
  dir: .claudio/sessions
  save_audio: true
```

Or via environment variables:

```bash
export CLAUDIO_SESSION_LOG=1
export CLAUDIO_SESSION_DIR=.claudio/sessions
export CLAUDIO_SESSION_STORAGE=jsonl
```

## Storage Backends

### JSONL (JSON Lines)

Simple file-based storage with one directory per session.

**Structure:**
```
.claudio/sessions/
  2025-01-28/
    abc123/                    # session ID
      events.jsonl             # chronological event log
      requests/
        req001.json            # full request body
        req002.json
      responses/
        req001.json            # full response content
        req002.json
      audio/
        req001_000.wav         # generated audio files
        req001_001.wav
```

**Pros:**
- Human-readable files
- Easy to grep/search
- Git-friendly
- No dependencies

**Cons:**
- Many small files
- Slower for large sessions
- No indexing

### SQLite

Structured database storage with indexing.

**Structure:**
```
.claudio/sessions/
  2025-01-28/
    sessions.db                # all sessions for the day
    audio/
      abc123/                  # session ID
        req001_000.wav
```

**Schema:**
```sql
-- Events table (chronological log)
CREATE TABLE events (
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  event TEXT NOT NULL,
  session_id TEXT NOT NULL,
  request_id TEXT,
  data TEXT  -- JSON
);

-- Full request bodies
CREATE TABLE requests (
  request_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  method TEXT NOT NULL,
  path TEXT NOT NULL,
  destination TEXT,
  headers TEXT,  -- JSON
  body TEXT      -- JSON
);

-- Full response content
CREATE TABLE responses (
  request_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  elapsed REAL,
  model TEXT,
  streaming INTEGER DEFAULT 0,
  total_chunks INTEGER,
  content TEXT  -- JSON or text
);

-- Audio metadata
CREATE TABLE audio (
  id INTEGER PRIMARY KEY,
  request_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  idx INTEGER NOT NULL,
  text TEXT,
  file_path TEXT,
  size INTEGER DEFAULT 0
);
```

**Pros:**
- Fast queries
- Indexed by session/request
- Single file per day
- Efficient for large sessions

**Cons:**
- Binary format
- Requires sqlite3

## Configuration Options

```yaml
logging:
  # Enable/disable logging
  enabled: true

  # Storage backend: 'jsonl' or 'sqlite'
  storage: jsonl

  # Base directory for logs
  dir: .claudio/sessions

  # Log full request bodies
  log_requests: true

  # Log full response content
  log_responses: true

  # Save generated audio files
  save_audio: true

  # Max request body size (0 = unlimited)
  max_request_size: 0

  # Max response size (0 = unlimited)
  max_response_size: 0
```

## What Gets Logged

### Events
- `request` - incoming request received
- `response` - response sent to client
- `stream_complete` - streaming response finished
- `audio_saved` - TTS audio file saved
- `tts_queued` - text added to TTS queue
- `tts_generate_start/done` - TTS generation
- `tts_play_start/done` - audio playback

### Request Data
- Method (POST, GET, etc.)
- Path (`/v1/messages`)
- Headers (excluding `Authorization`, `x-api-key`)
- Full body (model, messages, etc.)
- Destination name (anthropic, openai, etc.)

### Response Data
- Status code
- Model name
- Elapsed time
- Full content (text, tool_use blocks)
- Streaming metadata (chunk count)

### Audio Files
- WAV format (from TTS engine)
- Named by request ID and index
- Text snippet stored in metadata

## Querying Logs

### JSONL

```bash
# Find all requests for a session
cat .claudio/sessions/2025-01-28/abc123/events.jsonl | jq 'select(.event == "request")'

# Get all response times
cat events.jsonl | jq 'select(.event == "response") | .data.elapsed'

# List audio files
ls .claudio/sessions/2025-01-28/abc123/audio/
```

### SQLite

```bash
# Open database
sqlite3 .claudio/sessions/2025-01-28/sessions.db

# List sessions
SELECT DISTINCT session_id FROM events;

# Get requests for a session
SELECT * FROM requests WHERE session_id = 'abc123';

# Find slow responses
SELECT request_id, elapsed, model FROM responses WHERE elapsed > 5.0;

# Audio by request
SELECT * FROM audio WHERE request_id = 'req001';
```

## Programmatic Access

```python
from claudio.session_logger import SessionLogger, JsonlStorage, SqliteStorage

# Read JSONL session
storage = JsonlStorage(Path('.claudio/sessions/2025-01-28/abc123'))
for event in storage.get_session_events('abc123'):
    print(event.event, event.data)

# Read SQLite
storage = SqliteStorage(Path('.claudio/sessions/2025-01-28/sessions.db'))
request = storage.get_request('req001')
response = storage.get_response('req001')
```

## Privacy Notes

- API keys are **never** logged (filtered from headers)
- Request/response bodies may contain sensitive data
- Consider `max_request_size`/`max_response_size` limits
- Audio files may contain spoken content
- Use `.gitignore` for session directories
