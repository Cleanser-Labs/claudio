"""Claudio unified server - TTS, ASR, MCP, and HTTP API."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, HTMLResponse
from starlette.routing import Route
from starlette.requests import Request

from claudio import tts
from claudio.personas import (
  Persona,
  PersonaStore,
  VoiceStore,
  Scheduler,
  get_persona_store,
  get_voice_store,
  get_scheduler,
)

if TYPE_CHECKING:
  from claudio.asr import ASREngine


class Session(msgspec.Struct):
  """A voice session for a conversation."""
  id: str
  transcript_buffer: str = ''
  persona_id: str | None = None
  priority: int = 50  # 0-100, higher speaks first
  voice_id: str | None = None  # Resolved voice
  muted: bool = False
  created_at: float = 0.0
  last_speech_at: float = 0.0
  speech_count: int = 0

  def __post_init__(self):
    self._lock = threading.Lock()
    self._persona: Persona | None = None
    if self.created_at == 0.0:
      object.__setattr__(self, 'created_at', time.time())

  @property
  def lock(self):
    if not hasattr(self, '_lock'):
      self._lock = threading.Lock()
    return self._lock

  @property
  def persona(self) -> Persona | None:
    """Get the loaded persona object."""
    if not hasattr(self, '_persona'):
      self._persona = None
    return self._persona

  @persona.setter
  def persona(self, value: Persona | None):
    self._persona = value
    if value:
      self.persona_id = value.id
      self.priority = value.config.priority

  @property
  def interruptible(self) -> bool:
    """Check if this session can be interrupted."""
    if self._persona:
      return self._persona.config.interruptible
    return True

  def to_dict(self) -> dict:
    """Convert session to dict for API responses."""
    return {
      'id': self.id,
      'persona_id': self.persona_id,
      'persona_name': self._persona.config.name if self._persona else None,
      'voice_id': self.voice_id,
      'priority': self.priority,
      'muted': self.muted,
      'interruptible': self.interruptible,
      'created_at': self.created_at,
      'last_speech_at': self.last_speech_at,
      'speech_count': self.speech_count,
    }


class ServerState:
  """Shared server state."""

  def __init__(self):
    self.tts: tts.TTS | None = None
    self.asr: 'ASREngine | None' = None
    self.sessions: dict[str, Session] = {}
    self._sessions_lock = threading.Lock()
    self.persona_store: PersonaStore | None = None
    self.voice_store: VoiceStore | None = None
    self.scheduler: Scheduler | None = None
    self.started_at: float = 0.0
    self.request_count: int = 0
    self.total_speeches: int = 0

  def get_session(self, session_id: str) -> Session:
    """Get or create a session."""
    with self._sessions_lock:
      if session_id not in self.sessions:
        self.sessions[session_id] = Session(id=session_id)
      return self.sessions[session_id]

  def set_persona(self, session_id: str, persona_id: str) -> Persona | None:
    """Set persona for a session and resolve voice."""
    session = self.get_session(session_id)
    if self.persona_store:
      persona = self.persona_store.get(persona_id)
      if persona:
        session.persona = persona
        # Resolve voice from persona preferences
        if self.voice_store and persona.config.voices:
          active_models = self.scheduler.active_models if self.scheduler else None
          voice = self.voice_store.select(
            preferences=persona.config.voices,
            fallback_traits=persona.config.fallback,
            active_models=active_models,
          )
          if voice:
            session.voice_id = voice.id
        return persona
    return None


state = ServerState()


# --- HTTP API ---

async def speak(request: Request) -> Response:
  """POST /speak - Generate and play speech.

  Body: {"text": "Hello", "wait": true}
  """
  body = await request.json()
  text = body.get('text', '')
  wait = body.get('wait', True)

  if not text:
    return JSONResponse({'error': 'No text provided'}, status_code=400)

  if wait:
    state.tts.speak(text)
  else:
    threading.Thread(target=state.tts.speak, args=(text,), daemon=True).start()

  return JSONResponse({'status': 'ok', 'text': text})


async def talk(request: Request) -> Response:
  """POST /talk/{session_id} - Session-based voice interaction.

  Body: {"text": "Hello", "mode": "play|stream|file", "wait_for_priority": true}
    - play: Play audio server-side (default)
    - stream: Stream WAV bytes back to client
    - file: Return path to generated file
    - wait_for_priority: If true, wait for speech priority (default: true)

  Uses session's persona for voice settings if set.
  Returns audio or status depending on mode.
  """
  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  body = await request.json()
  text = body.get('text', '')
  mode = body.get('mode', 'play')
  wait_for_priority = body.get('wait_for_priority', True)

  if not text:
    return JSONResponse({'error': 'No text provided'}, status_code=400)

  # Check if session is muted
  if session.muted:
    return JSONResponse({
      'status': 'muted',
      'text': text,
      'session': session_id,
      'message': 'Session is muted',
    })

  # Resolve voice from session's resolved voice or persona preferences
  voice = None
  rate = 1.0

  if session.voice_id and state.voice_store:
    voice_obj = state.voice_store.get(session.voice_id)
    if voice_obj:
      voice = voice_obj.config.voice_id or voice_obj.config.name
      rate = voice_obj.config.speed

  # Persona speed override
  if session.persona and session.persona.config.speed:
    rate = session.persona.config.speed

  # Request params override
  voice = body.get('voice', voice)
  rate = body.get('speed', rate)

  if mode == 'stream':
    # Stream audio bytes back to client
    audio_bytes = state.tts.generate(text, voice=voice, rate=rate)
    return Response(
      content=audio_bytes,
      media_type='audio/x-aiff',
      headers={'Content-Disposition': 'inline; filename="speech.aiff"'},
    )

  elif mode == 'file':
    # Generate and return bytes as base64
    import base64
    audio_bytes = state.tts.generate(text, voice=voice, rate=rate)
    return JSONResponse({'status': 'ok', 'audio': base64.b64encode(audio_bytes).decode()})

  else:  # mode == 'play'
    # Coordinate speech priority (higher priority = speaks first)
    if wait_for_priority and state.scheduler:
      acquired = state.scheduler.acquire(
        session_id,
        session.priority,
        interruptible=session.interruptible,
        timeout=30.0,
      )
      if not acquired:
        return JSONResponse({'error': 'Timeout waiting for speech priority'}, status_code=408)
      try:
        state.tts.speak(text, voice=voice, rate=rate)
        # Update session stats
        session.last_speech_at = time.time()
        session.speech_count += 1
      finally:
        state.scheduler.release(session_id)
    else:
      state.tts.speak(text, voice=voice, rate=rate)
      session.last_speech_at = time.time()
      session.speech_count += 1

    return JSONResponse({
      'status': 'ok',
      'text': text,
      'session': session_id,
      'persona': session.persona_id,
      'voice': session.voice_id,
    })


async def talk_transcript(request: Request) -> JSONResponse:
  """GET /talk/{session_id}/transcript - Get session transcript."""
  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  with session.lock:
    transcript = session.transcript_buffer
    session.transcript_buffer = ''

  return JSONResponse({'transcript': transcript, 'session': session_id})


async def get_transcript(request: Request) -> JSONResponse:
  """GET /transcript - Get and clear pending voice transcript (legacy)."""
  # Use default session for backwards compatibility
  session = state.get_session('default')
  with session.lock:
    transcript = session.transcript_buffer
    session.transcript_buffer = ''
  return JSONResponse({'transcript': transcript})


async def listen_start(request: Request) -> JSONResponse:
  """POST /listen/{session_id}/start - Start ASR daemon for a session."""
  from claudio.asr import ASRDaemon, ASRConfig

  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  # Check if already listening
  if hasattr(session, '_asr_daemon') and session._asr_daemon:
    return JSONResponse({'status': 'already_running', 'session': session_id})

  def on_transcript(text: str):
    """Called when transcript is ready to send."""
    with session.lock:
      session.transcript_buffer = text

  def on_partial(text: str):
    """Called with partial transcripts."""
    pass  # Could emit via websocket in future

  body = await request.json() if request.headers.get('content-type') == 'application/json' else {}
  config = ASRConfig(
    trigger_phrase=body.get('trigger_phrase', 'send'),
    silence_duration=body.get('silence_duration', 1.5),
  )

  daemon = ASRDaemon(config=config, on_transcript=on_transcript, on_partial=on_partial)
  daemon.start()
  session._asr_daemon = daemon

  return JSONResponse({'status': 'started', 'session': session_id})


async def listen_stop(request: Request) -> JSONResponse:
  """POST /listen/{session_id}/stop - Stop ASR daemon for a session."""
  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  if hasattr(session, '_asr_daemon') and session._asr_daemon:
    session._asr_daemon.stop()
    session._asr_daemon = None
    return JSONResponse({'status': 'stopped', 'session': session_id})

  return JSONResponse({'status': 'not_running', 'session': session_id})


async def listen_send(request: Request) -> JSONResponse:
  """POST /listen/{session_id}/send - Manually trigger sending transcript."""
  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  if hasattr(session, '_asr_daemon') and session._asr_daemon:
    session._asr_daemon.send()
    return JSONResponse({'status': 'sent', 'session': session_id})

  return JSONResponse({'status': 'not_running', 'session': session_id})


async def devices(request: Request) -> JSONResponse:
  """GET /devices - List audio devices."""
  import sounddevice as sd
  devices_list = []
  for i, dev in enumerate(sd.query_devices()):
    devices_list.append({
      'id': i,
      'name': dev['name'],
      'inputs': dev['max_input_channels'],
      'outputs': dev['max_output_channels'],
      'default_input': i == sd.default.device[0],
      'default_output': i == sd.default.device[1],
    })
  return JSONResponse({'devices': devices_list})


async def health(request: Request) -> JSONResponse:
  """GET /health - Health check."""
  return JSONResponse({
    'status': 'ok',
    'tts_loaded': state.tts is not None,
    'asr_loaded': state.asr is not None,
    'sessions': list(state.sessions.keys()),
    'personas_loaded': state.persona_store is not None and len(state.persona_store) > 0,
    'voices_loaded': state.voice_store is not None and len(state.voice_store) > 0,
  })


async def status(request: Request) -> JSONResponse:
  """GET /status - Detailed server status."""
  from datetime import datetime

  uptime = time.time() - state.started_at if state.started_at else 0

  # Get session stats
  sessions = []
  total_speeches = 0
  with state._sessions_lock:
    for session in state.sessions.values():
      total_speeches += session.speech_count
      sessions.append(session.to_dict())

  # Get active models from scheduler
  active_models = []
  if state.scheduler:
    active_models = list(state.scheduler.active_models) if hasattr(state.scheduler, 'active_models') else []

  return JSONResponse({
    'status': 'ok',
    'server': {
      'started_at': state.started_at,
      'started_at_iso': datetime.fromtimestamp(state.started_at).isoformat() if state.started_at else None,
      'uptime_seconds': uptime,
      'uptime_human': _format_uptime(uptime),
    },
    'stats': {
      'total_sessions': len(sessions),
      'total_speeches': total_speeches,
      'request_count': state.request_count,
    },
    'tts': {
      'loaded': state.tts is not None,
      'backend': type(state.tts).__name__ if state.tts else None,
    },
    'personas': {
      'loaded': state.persona_store is not None,
      'count': len(state.persona_store) if state.persona_store else 0,
    },
    'voices': {
      'loaded': state.voice_store is not None,
      'count': len(state.voice_store) if state.voice_store else 0,
    },
    'scheduler': {
      'active_models': active_models,
      'speaking_session': state.scheduler._speaking_session if state.scheduler and hasattr(state.scheduler, '_speaking_session') else None,
    },
    'sessions': sessions,
  })


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


# --- Persona API ---

async def list_personas(request: Request) -> JSONResponse:
  """GET /personas - List available personas."""
  if not state.persona_store:
    return JSONResponse({'personas': []})

  personas = []
  for p in state.persona_store.list():
    personas.append({
      'id': p.id,
      'name': p.config.name,
      'voices': p.config.voices,
      'priority': p.config.priority,
      'interruptible': p.config.interruptible,
      'speed': p.config.speed,
    })

  return JSONResponse({'personas': personas})


async def get_persona(request: Request) -> JSONResponse:
  """GET /personas/{persona_id} - Get a persona's details."""
  persona_id = request.path_params.get('persona_id')

  if not state.persona_store:
    return JSONResponse({'error': 'No persona store'}, status_code=500)

  persona = state.persona_store.get(persona_id)
  if not persona:
    return JSONResponse({'error': f'Persona not found: {persona_id}'}, status_code=404)

  return JSONResponse({
    'id': persona.id,
    'name': persona.config.name,
    'voices': persona.config.voices,
    'fallback': persona.config.fallback,
    'priority': persona.config.priority,
    'interruptible': persona.config.interruptible,
    'speed': persona.config.speed,
    'prompt': persona.prompt,
  })


async def get_session_persona(request: Request) -> JSONResponse:
  """GET /sessions/{session_id}/persona - Get session's current persona."""
  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  if not session.persona:
    return JSONResponse({
      'session': session_id,
      'persona': None,
      'voice': session.voice_id,
      'priority': session.priority,
    })

  return JSONResponse({
    'session': session_id,
    'persona': {
      'id': session.persona.id,
      'name': session.persona.config.name,
      'voices': session.persona.config.voices,
      'priority': session.persona.config.priority,
      'interruptible': session.persona.config.interruptible,
    },
    'voice': session.voice_id,
    'priority': session.priority,
  })


async def set_session_persona(request: Request) -> JSONResponse:
  """POST /sessions/{session_id}/persona - Set session's persona.

  Body: {"persona_id": "assistant"} or {"priority": 80}
  """
  session_id = request.path_params.get('session_id', 'default')
  session = state.get_session(session_id)

  body = await request.json()
  persona_id = body.get('persona_id')
  priority = body.get('priority')

  if persona_id:
    persona = state.set_persona(session_id, persona_id)
    if not persona:
      return JSONResponse({'error': f'Persona not found: {persona_id}'}, status_code=404)
  elif priority is not None:
    session.priority = int(priority)

  return JSONResponse({
    'session': session_id,
    'persona': session.persona_id,
    'voice': session.voice_id,
    'priority': session.priority,
  })


async def reload_personas(request: Request) -> JSONResponse:
  """POST /personas/reload - Reload personas from disk."""
  if state.persona_store:
    state.persona_store.reload()
  if state.voice_store:
    state.voice_store.reload()
  return JSONResponse({
    'status': 'ok',
    'personas': len(state.persona_store) if state.persona_store else 0,
    'voices': len(state.voice_store) if state.voice_store else 0,
  })


# --- Session API ---

async def list_sessions(request: Request) -> JSONResponse:
  """GET /sessions - List all sessions."""
  sessions = []
  with state._sessions_lock:
    for session in state.sessions.values():
      sessions.append(session.to_dict())

  # Sort by priority (highest first) then by created_at
  sessions.sort(key=lambda s: (-s['priority'], s['created_at']))

  return JSONResponse({'sessions': sessions})


async def get_session(request: Request) -> JSONResponse:
  """GET /sessions/{session_id} - Get session details."""
  session_id = request.path_params.get('session_id')

  with state._sessions_lock:
    if session_id not in state.sessions:
      return JSONResponse({'error': f'Session not found: {session_id}'}, status_code=404)
    session = state.sessions[session_id]

  return JSONResponse({'session': session.to_dict()})


async def update_session(request: Request) -> JSONResponse:
  """PATCH /sessions/{session_id} - Update session settings.

  Body: {"priority": 80, "voice_id": "daniel", "muted": true}
  """
  session_id = request.path_params.get('session_id')

  with state._sessions_lock:
    if session_id not in state.sessions:
      return JSONResponse({'error': f'Session not found: {session_id}'}, status_code=404)
    session = state.sessions[session_id]

  body = await request.json()

  if 'priority' in body:
    session.priority = int(body['priority'])
  if 'voice_id' in body:
    session.voice_id = body['voice_id']
  if 'muted' in body:
    session.muted = bool(body['muted'])

  return JSONResponse({'session': session.to_dict()})


async def delete_session(request: Request) -> JSONResponse:
  """DELETE /sessions/{session_id} - Delete a session."""
  session_id = request.path_params.get('session_id')

  with state._sessions_lock:
    if session_id not in state.sessions:
      return JSONResponse({'error': f'Session not found: {session_id}'}, status_code=404)

    session = state.sessions[session_id]
    # Stop ASR if running
    if hasattr(session, '_asr_daemon') and session._asr_daemon:
      session._asr_daemon.stop()

    del state.sessions[session_id]

  return JSONResponse({'status': 'deleted', 'session_id': session_id})


async def mute_session(request: Request) -> JSONResponse:
  """POST /sessions/{session_id}/mute - Mute a session."""
  session_id = request.path_params.get('session_id')

  with state._sessions_lock:
    if session_id not in state.sessions:
      return JSONResponse({'error': f'Session not found: {session_id}'}, status_code=404)
    session = state.sessions[session_id]

  session.muted = True
  return JSONResponse({'session': session.to_dict()})


async def unmute_session(request: Request) -> JSONResponse:
  """POST /sessions/{session_id}/unmute - Unmute a session."""
  session_id = request.path_params.get('session_id')

  with state._sessions_lock:
    if session_id not in state.sessions:
      return JSONResponse({'error': f'Session not found: {session_id}'}, status_code=404)
    session = state.sessions[session_id]

  session.muted = False
  return JSONResponse({'session': session.to_dict()})


# --- Voice API ---

async def list_voices(request: Request) -> JSONResponse:
  """GET /voices - List available voices."""
  if not state.voice_store:
    return JSONResponse({'voices': []})

  voices = []
  for v in state.voice_store.list():
    voices.append({
      'id': v.id,
      'name': v.config.name,
      'model': v.config.model,
      'gender': v.config.gender,
      'tone': v.config.tone,
      'energy': v.config.energy,
      'quality': v.config.quality,
      'latency': v.config.latency,
    })

  return JSONResponse({'voices': voices})


async def get_voice(request: Request) -> JSONResponse:
  """GET /voices/{voice_id} - Get voice details."""
  voice_id = request.path_params.get('voice_id')

  if not state.voice_store:
    return JSONResponse({'error': 'No voice store'}, status_code=500)

  voice = state.voice_store.get(voice_id)
  if not voice:
    return JSONResponse({'error': f'Voice not found: {voice_id}'}, status_code=404)

  return JSONResponse({
    'id': voice.id,
    'name': voice.config.name,
    'model': voice.config.model,
    'voice_id': voice.config.voice_id,
    'gender': voice.config.gender,
    'accent': voice.config.accent,
    'tone': voice.config.tone,
    'energy': voice.config.energy,
    'quality': voice.config.quality,
    'latency': voice.config.latency,
    'speed': voice.config.speed,
    'description': voice.description,
  })


# --- Web Dashboard ---

DASHBOARD_HTML = '''<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Claudio Dashboard</title>
  <style>
    :root { --bg: #1a1a2e; --surface: #16213e; --primary: #0f3460; --accent: #e94560; --text: #eee; --muted: #888; }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; padding: 20px; }
    h1 { margin-bottom: 20px; font-size: 1.5rem; }
    h2 { margin: 20px 0 10px; font-size: 1.2rem; color: var(--muted); }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
    .card { background: var(--surface); border-radius: 8px; padding: 15px; }
    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
    .card-title { font-weight: 600; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; background: var(--primary); }
    .badge.high { background: var(--accent); }
    .badge.muted { background: var(--muted); opacity: 0.6; }
    .stats { display: flex; gap: 15px; font-size: 0.85rem; color: var(--muted); margin: 8px 0; }
    .actions { display: flex; gap: 8px; margin-top: 10px; }
    button { background: var(--primary); color: var(--text); border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 0.85rem; }
    button:hover { opacity: 0.85; }
    button.danger { background: var(--accent); }
    button.mute { background: var(--muted); }
    input, select { background: var(--bg); color: var(--text); border: 1px solid var(--primary); padding: 6px 10px; border-radius: 4px; font-size: 0.85rem; }
    .empty { color: var(--muted); font-style: italic; padding: 20px; text-align: center; }
    .row { display: flex; gap: 10px; align-items: center; margin: 5px 0; }
    .label { color: var(--muted); font-size: 0.85rem; min-width: 60px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 8px; border-bottom: 1px solid var(--primary); }
    th { color: var(--muted); font-size: 0.85rem; }
    .refresh { position: fixed; bottom: 20px; right: 20px; }
  </style>
</head>
<body>
  <h1>Claudio Voice Server</h1>

  <h2>Active Sessions</h2>
  <div class="grid" id="sessions"></div>

  <h2>Personas</h2>
  <div class="grid" id="personas"></div>

  <h2>Voices</h2>
  <table id="voices">
    <thead>
      <tr><th>Name</th><th>Model</th><th>Gender</th><th>Tone</th><th>Quality</th></tr>
    </thead>
    <tbody></tbody>
  </table>

  <button class="refresh" onclick="refresh()">Refresh</button>

  <script>
    const api = async (path, opts = {}) => {
      const res = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
        body: opts.body ? JSON.stringify(opts.body) : undefined,
      });
      return res.json();
    };

    const formatTime = (ts) => ts ? new Date(ts * 1000).toLocaleTimeString() : '-';

    async function loadSessions() {
      const { sessions } = await api('/sessions');
      const el = document.getElementById('sessions');
      if (!sessions.length) {
        el.innerHTML = '<div class="empty">No active sessions</div>';
        return;
      }
      el.innerHTML = sessions.map(s => `
        <div class="card" data-id="${s.id}">
          <div class="card-header">
            <span class="card-title">${s.id}</span>
            <span class="badge ${s.priority >= 70 ? 'high' : ''} ${s.muted ? 'muted' : ''}">
              ${s.muted ? 'MUTED' : 'P' + s.priority}
            </span>
          </div>
          <div class="row"><span class="label">Persona:</span> ${s.persona_name || '-'}</div>
          <div class="row"><span class="label">Voice:</span> ${s.voice_id || '-'}</div>
          <div class="stats">
            <span>Created: ${formatTime(s.created_at)}</span>
            <span>Speeches: ${s.speech_count}</span>
          </div>
          <div class="row">
            <span class="label">Priority:</span>
            <input type="number" value="${s.priority}" min="0" max="100" style="width:60px"
              onchange="updateSession('${s.id}', {priority: +this.value})">
          </div>
          <div class="actions">
            <button class="mute" onclick="toggleMute('${s.id}', ${!s.muted})">
              ${s.muted ? 'Unmute' : 'Mute'}
            </button>
            <button class="danger" onclick="deleteSession('${s.id}')">Delete</button>
          </div>
        </div>
      `).join('');
    }

    async function loadPersonas() {
      const { personas } = await api('/personas');
      const el = document.getElementById('personas');
      if (!personas.length) {
        el.innerHTML = '<div class="empty">No personas loaded</div>';
        return;
      }
      el.innerHTML = personas.map(p => `
        <div class="card">
          <div class="card-header">
            <span class="card-title">${p.name}</span>
            <span class="badge ${p.priority >= 70 ? 'high' : ''}">P${p.priority}</span>
          </div>
          <div class="row"><span class="label">ID:</span> ${p.id}</div>
          <div class="row"><span class="label">Voices:</span> ${(p.voices || []).join(', ') || '-'}</div>
          <div class="row"><span class="label">Speed:</span> ${p.speed || '1.0'}</div>
          <div class="row"><span class="label">Interrupt:</span> ${p.interruptible ? 'Yes' : 'No'}</div>
        </div>
      `).join('');
    }

    async function loadVoices() {
      const { voices } = await api('/voices');
      const tbody = document.querySelector('#voices tbody');
      if (!voices.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty">No voices loaded</td></tr>';
        return;
      }
      tbody.innerHTML = voices.map(v => `
        <tr>
          <td>${v.name}</td>
          <td>${v.model || '-'}</td>
          <td>${v.gender || '-'}</td>
          <td>${v.tone || '-'}</td>
          <td>${v.quality || '-'}</td>
        </tr>
      `).join('');
    }

    async function updateSession(id, data) {
      await api(`/sessions/${id}`, { method: 'PATCH', body: data });
      loadSessions();
    }

    async function toggleMute(id, mute) {
      await api(`/sessions/${id}/${mute ? 'mute' : 'unmute'}`, { method: 'POST' });
      loadSessions();
    }

    async function deleteSession(id) {
      if (confirm(`Delete session "${id}"?`)) {
        await api(`/sessions/${id}`, { method: 'DELETE' });
        loadSessions();
      }
    }

    function refresh() {
      loadSessions();
      loadPersonas();
      loadVoices();
    }

    // Initial load
    refresh();
    // Auto-refresh every 5 seconds
    setInterval(loadSessions, 5000);
  </script>
</body>
</html>
'''


async def dashboard(request: Request) -> HTMLResponse:
  """GET / - Web dashboard landing page."""
  return HTMLResponse(DASHBOARD_HTML)


routes = [
  # Dashboard
  Route('/', dashboard, methods=['GET']),
  # Core voice API
  Route('/speak', speak, methods=['POST']),
  Route('/talk/{session_id}', talk, methods=['POST']),
  Route('/talk/{session_id}/transcript', talk_transcript, methods=['GET']),
  Route('/listen/{session_id}/start', listen_start, methods=['POST']),
  Route('/listen/{session_id}/stop', listen_stop, methods=['POST']),
  Route('/listen/{session_id}/send', listen_send, methods=['POST']),
  Route('/transcript', get_transcript, methods=['GET']),
  Route('/devices', devices, methods=['GET']),
  Route('/health', health, methods=['GET']),
  Route('/status', status, methods=['GET']),
  # Session routes
  Route('/sessions', list_sessions, methods=['GET']),
  Route('/sessions/{session_id}', get_session, methods=['GET']),
  Route('/sessions/{session_id}', update_session, methods=['PATCH']),
  Route('/sessions/{session_id}', delete_session, methods=['DELETE']),
  Route('/sessions/{session_id}/mute', mute_session, methods=['POST']),
  Route('/sessions/{session_id}/unmute', unmute_session, methods=['POST']),
  Route('/sessions/{session_id}/persona', get_session_persona, methods=['GET']),
  Route('/sessions/{session_id}/persona', set_session_persona, methods=['POST']),
  # Persona routes
  Route('/personas', list_personas, methods=['GET']),
  Route('/personas/reload', reload_personas, methods=['POST']),
  Route('/personas/{persona_id}', get_persona, methods=['GET']),
  # Voice routes
  Route('/voices', list_voices, methods=['GET']),
  Route('/voices/{voice_id}', get_voice, methods=['GET']),
]


def create_app(
  output_device: int | None = None,
  personas_dir: Path | str | None = None,
  voices_dir: Path | str | None = None,
) -> Starlette:
  """Create the Starlette app with optional device config."""

  @asynccontextmanager
  async def lifespan(app: Starlette):
    """Initialize TTS, voices, personas, and scheduler on startup."""
    state.started_at = time.time()
    state.tts = tts.create(tts.Config(output_device=output_device))
    state.scheduler = get_scheduler()

    # Load voices from directory
    if voices_dir:
      state.voice_store = VoiceStore(voices_dir)
    else:
      for path in [
        Path.cwd() / 'voices',
        Path.cwd() / '.claudio' / 'voices',
        Path.home() / '.claudio' / 'voices',
      ]:
        if path.exists():
          state.voice_store = VoiceStore(path)
          break
      else:
        state.voice_store = VoiceStore()

    # Load personas from directory
    if personas_dir:
      state.persona_store = PersonaStore(personas_dir)
    else:
      for path in [
        Path.cwd() / 'personas',
        Path.cwd() / '.claudio' / 'personas',
        Path.home() / '.claudio' / 'personas',
      ]:
        if path.exists():
          state.persona_store = PersonaStore(path)
          break
      else:
        state.persona_store = PersonaStore()

    yield

    # Cleanup
    for session in state.sessions.values():
      if hasattr(session, '_asr_daemon') and session._asr_daemon:
        session._asr_daemon.stop()
    state.tts = None
    state.asr = None
    state.persona_store = None
    state.voice_store = None
    state.scheduler = None

  return Starlette(routes=routes, lifespan=lifespan)


# Default app for backwards compatibility
app = create_app()


# --- MCP Protocol ---

def handle_mcp_request(request: dict) -> dict:
  """Handle a single MCP JSON-RPC request."""
  method = request.get('method', '')
  params = request.get('params', {})
  req_id = request.get('id')

  if method == 'initialize':
    return {
      'jsonrpc': '2.0',
      'id': req_id,
      'result': {
        'protocolVersion': '2024-11-05',
        'capabilities': {'tools': {}},
        'serverInfo': {'name': 'claudio', 'version': '0.1.0'},
      },
    }

  elif method == 'tools/list':
    return {
      'jsonrpc': '2.0',
      'id': req_id,
      'result': {
        'tools': [
          {
            'name': 'speak',
            'description': 'Speak text aloud to the user. Use for verbal communication.',
            'inputSchema': {
              'type': 'object',
              'properties': {
                'text': {
                  'type': 'string',
                  'description': 'The text to speak aloud',
                },
                'wait': {
                  'type': 'boolean',
                  'description': 'Wait for speech to complete (default: true)',
                  'default': True,
                },
              },
              'required': ['text'],
            },
          },
          {
            'name': 'get_transcript',
            'description': 'Get pending voice transcript from user. Returns empty if none.',
            'inputSchema': {
              'type': 'object',
              'properties': {},
            },
          },
        ],
      },
    }

  elif method == 'tools/call':
    tool_name = params.get('name', '')
    tool_args = params.get('arguments', {})

    if tool_name == 'speak':
      text = tool_args.get('text', '')
      wait = tool_args.get('wait', True)

      if not text:
        return {
          'jsonrpc': '2.0',
          'id': req_id,
          'result': {'content': [{'type': 'text', 'text': 'Error: no text provided'}]},
        }

      # Ensure TTS is loaded
      if state.tts is None:
        state.tts = tts.create()

      if wait:
        state.tts.speak(text)
      else:
        threading.Thread(target=state.tts.speak, args=(text,), daemon=True).start()

      return {
        'jsonrpc': '2.0',
        'id': req_id,
        'result': {'content': [{'type': 'text', 'text': f'Spoke: "{text}"'}]},
      }

    elif tool_name == 'get_transcript':
      session = state.get_session('mcp')
      with session.lock:
        transcript = session.transcript_buffer
        session.transcript_buffer = ''

      return {
        'jsonrpc': '2.0',
        'id': req_id,
        'result': {'content': [{'type': 'text', 'text': transcript or '(no pending transcript)'}]},
      }

    else:
      return {
        'jsonrpc': '2.0',
        'id': req_id,
        'error': {'code': -32601, 'message': f'Unknown tool: {tool_name}'},
      }

  elif method == 'notifications/initialized':
    # No response needed for notifications
    return None

  else:
    return {
      'jsonrpc': '2.0',
      'id': req_id,
      'error': {'code': -32601, 'message': f'Unknown method: {method}'},
    }


def run_mcp_stdio(output_device: int | None = None):
  """Run MCP server over stdio."""
  # Initialize TTS lazily on first speak
  state.tts = tts.create(tts.Config(output_device=output_device))

  for line in sys.stdin:
    line = line.strip()
    if not line:
      continue

    try:
      request = json.loads(line)
      response = handle_mcp_request(request)

      if response is not None:
        print(json.dumps(response), flush=True)

    except json.JSONDecodeError as e:
      error_response = {
        'jsonrpc': '2.0',
        'id': None,
        'error': {'code': -32700, 'message': f'Parse error: {e}'},
      }
      print(json.dumps(error_response), flush=True)
    except Exception as e:
      error_response = {
        'jsonrpc': '2.0',
        'id': None,
        'error': {'code': -32603, 'message': f'Internal error: {e}'},
      }
      print(json.dumps(error_response), flush=True)


def run_http(
  host: str = '127.0.0.1',
  port: int = 8765,
  output_device: int | None = None,
  personas_dir: Path | str | None = None,
  voices_dir: Path | str | None = None,
):
  """Run HTTP server."""
  import uvicorn
  app = create_app(
    output_device=output_device,
    personas_dir=personas_dir,
    voices_dir=voices_dir,
  )
  print(f'Starting Claudio HTTP server on {host}:{port}')
  if state.voice_store and len(state.voice_store) > 0:
    print(f'Loaded {len(state.voice_store)} voices')
  if state.persona_store and len(state.persona_store) > 0:
    print(f'Loaded {len(state.persona_store)} personas')
  uvicorn.run(app, host=host, port=port)


def run_daemon():
  """Start server as background daemon."""
  import subprocess

  # Check if already running
  try:
    import httpx
    response = httpx.get('http://127.0.0.1:8765/health', timeout=1.0)
    if response.status_code == 200:
      print('Claudio server already running', file=sys.stderr)
      return
  except:
    pass

  # Start as background process
  subprocess.Popen(
    ['claudio-server', '--http'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )
  print('Started Claudio server', file=sys.stderr)


def main():
  """Entry point."""
  parser = argparse.ArgumentParser(description='Claudio Voice Server')
  parser.add_argument('--mcp', action='store_true', help='Run as MCP server (stdio)')
  parser.add_argument('--http', action='store_true', help='Run as HTTP server')
  parser.add_argument('--daemon', action='store_true', help='Start as background daemon')
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', '-p', type=int, default=8765)
  parser.add_argument('--device', '-d', type=int, default=None, help='Output device ID')
  parser.add_argument('--personas', type=Path, default=None, help='Personas directory')
  parser.add_argument('--voices', type=Path, default=None, help='Voices directory')
  args = parser.parse_args()

  if args.daemon:
    run_daemon()
  elif args.mcp:
    run_mcp_stdio(output_device=args.device)
  elif args.http:
    run_http(
      host=args.host,
      port=args.port,
      output_device=args.device,
      personas_dir=args.personas,
      voices_dir=args.voices,
    )
  else:
    # Default to MCP mode
    run_mcp_stdio(output_device=args.device)


if __name__ == '__main__':
  main()
