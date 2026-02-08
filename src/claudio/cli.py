"""Claudio CLI - voice commands for Claude Code."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
  name='claudio',
  help='Voice I/O for Claude Code',
  rich_markup_mode='rich',
)
console = Console()

SERVER_URL = 'http://127.0.0.1:8765'
DEFAULT_PROXY_PORT = 9000

# Voice skill instructions injected into Claude's system prompt
VOICE_SKILL_TAGS = '''
# Voice Output

You have text-to-speech enabled. Use `<say>` tags to speak to the user. Tags can wrap multiple lines of text.

## Guidelines
- Wrap ALL your spoken text in `<say>` tags — if it's not in tags, the user won't hear it
- Speak most of your response: explanations, reasoning, answers, summaries, and narration should all be spoken
- NEVER read code blocks, terminal output, file paths, URLs, or markdown formatting aloud — instead, narrate what they contain (e.g. "Here's the updated function" or "The test output shows three failures")
- Break long responses into multiple `<say>` blocks, placing them before and after code/formatting blocks
- Keep the user informed as you work — narrate tool calls, searches, and findings

## Syntax
```xml
<say>Your normal spoken text goes here.</say>
<say>You can wrap multiple lines
of text in a single tag.</say>
<say speed="fast">Quick updates like this.</say>
```

## Example Response
```
<say>I found the issue — there's a missing null check on line 42. Here's a fix using a caching decorator:</say>

```python
@cache(ttl=300)
def get_users():
    return db.query("SELECT * FROM users")
```

<say>This caches the query results for 5 minutes so we avoid hitting the database on every request. Want me to add error handling too?</say>
```
'''.strip()

VOICE_SKILL_TEXT = '''
# Voice Output

Text-to-speech is enabled. All your text responses are spoken automatically.
Code blocks are skipped. Keep responses conversational and concise.
'''.strip()


def _get_voice_skill(speak_mode: str) -> str | None:
  """Get voice skill instructions for Claude based on speak mode."""
  if speak_mode == 'tags':
    return VOICE_SKILL_TAGS
  elif speak_mode == 'text':
    return VOICE_SKILL_TEXT
  # 'all' and 'off' modes don't need special instructions
  return None


def _stop_process_on_port(port: int, label: str) -> None:
  """Find and kill the process listening on a port."""
  import signal
  try:
    result = subprocess.run(
      ['lsof', '-ti', f'tcp:{port}'],
      capture_output=True, text=True,
    )
  except FileNotFoundError:
    console.print('[red]lsof not found[/red]')
    raise typer.Exit(1)

  pids = result.stdout.strip().split('\n')
  pids = [p.strip() for p in pids if p.strip()]
  if not pids:
    console.print(f'[dim]{label} is not running on port {port}[/dim]')
    return

  for pid in pids:
    try:
      os.kill(int(pid), signal.SIGTERM)
    except (ProcessLookupError, ValueError):
      pass
  console.print(f'[green]{label} stopped[/green] [dim](port {port})[/dim]')


def is_server_running(url: str = SERVER_URL) -> bool:
  """Check if Claudio server is running."""
  try:
    response = httpx.get(f'{url}/health', timeout=1.0)
    return response.status_code == 200
  except (httpx.ConnectError, httpx.ReadTimeout):
    return False


def ensure_server(url: str = SERVER_URL) -> None:
  """Start server if not running."""
  import subprocess
  import time

  if is_server_running(url):
    return

  with console.status('[bold blue]Starting Claudio server...'):
    subprocess.Popen(
      ['claudio-server', '--http'],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )

    for _ in range(30):
      if is_server_running(url):
        console.print('[green]Server ready[/green]')
        return
      time.sleep(0.5)

  console.print('[red]Server failed to start[/red]')
  raise typer.Exit(1)


# --- Direct mode (no server) ---

@app.command()
def say(
  text: Annotated[str, typer.Argument(help='Text to speak')],
  device: Annotated[Optional[int], typer.Option('-d', '--device', help='Output device ID')] = None,
):
  """Speak text directly (no server)."""
  from claudio import tts

  engine = tts.create(tts.Config(output_device=device))
  engine.speak(text)


@app.command()
def config():
  """Show resolved config with effective values."""
  from claudio import tts as tts_mod
  from claudio.config import load_config

  search_paths = [
    Path.cwd() / 'claudio.yaml',
    Path.cwd() / '.claudio' / 'config.yaml',
    Path.home() / '.claudio' / 'config.yaml',
  ]

  found = None
  for p in search_paths:
    if p.exists():
      found = p
      break

  if not found:
    console.print('[dim]No config file found[/dim]')
    console.print('[dim]Searched:[/dim]')
    for p in search_paths:
      console.print(f'  {p}')
    console.print()
    console.print('[dim]Run `claudio init` to create one[/dim]')
    return

  cfg = load_config(found)
  console.print(f'[bold]Config:[/bold] {found}')
  console.print()

  # TTS — resolve effective values
  tts_backend = cfg.tts.backend
  if tts_backend == 'auto':
    tts_backend = tts_mod.Config().backend
  tts_voice = cfg.tts.voice
  if not tts_voice:
    # Show what the backend will actually use
    backends = tts_mod.available_backends()
    if tts_backend in backends:
      voice_list = backends[tts_backend].list_voices()
      tts_voice = voice_list[0]['id'] if voice_list else 'default'
    else:
      tts_voice = 'default'

  console.print('[bold]tts:[/bold]')
  console.print(f'  backend: [cyan]{tts_backend}[/cyan]', end='')
  if cfg.tts.backend == 'auto':
    console.print(f' [dim](auto)[/dim]')
  else:
    console.print()
  console.print(f'  voice:   [cyan]{tts_voice}[/cyan]', end='')
  if not cfg.tts.voice:
    console.print(f' [dim](default)[/dim]')
  else:
    console.print()
  console.print(f'  speed:   {cfg.tts.speed}')
  if cfg.tts.device is not None:
    console.print(f'  device:  {cfg.tts.device}')
  console.print()

  # Proxy
  console.print('[bold]proxy:[/bold]')
  console.print(f'  host:       {cfg.proxy.host}')
  console.print(f'  port:       {cfg.proxy.port}')
  console.print(f'  speak_mode: {cfg.proxy.speak_mode}')
  if cfg.proxy.notify:
    console.print(f'  notify:     {cfg.proxy.notify}')
  console.print()

  # Destinations
  if cfg.destinations:
    console.print('[bold]destinations:[/bold]')
    for d in cfg.destinations:
      tts_tag = ' [dim](no tts)[/dim]' if not d.tts_enabled else ''
      default_tag = ' [dim](default)[/dim]' if d.name == cfg.default_destination else ''
      console.print(f'  [cyan]{d.name}[/cyan]{default_tag}{tts_tag}')
      console.print(f'    url:   {d.url}')
      console.print(f'    paths: {", ".join(d.paths)}')
      if d.api_key_env:
        key_set = bool(os.environ.get(d.api_key_env))
        status = '[green]set[/green]' if key_set else '[red]not set[/red]'
        console.print(f'    key:   ${d.api_key_env} ({status})')
    console.print()

  # Logging
  console.print('[bold]logging:[/bold]')
  if cfg.logging.enabled:
    console.print(f'  enabled:  [green]yes[/green]')
    console.print(f'  storage:  {cfg.logging.storage}')
    console.print(f'  dir:      {cfg.logging.dir}')
  else:
    console.print(f'  enabled:  [dim]no[/dim]')


@app.command()
def init():
  """Interactive setup — pick backend, voice, download models, save config."""
  from claudio import tts
  from claudio.config import ClaudioConfig, TTSConfig, save_config

  console.print()
  console.print('[bold]Claudio Setup[/bold]')
  console.print()

  # 1. Detect installed backends
  backends = tts.available_backends()
  if not backends:
    console.print('[red]No TTS backends available.[/red]')
    console.print('[dim]Install one: uv sync --extra kokoro[/dim]')
    raise typer.Exit(1)

  console.print('[bold]Available backends:[/bold]')
  backend_names = list(backends.keys())
  for i, name in enumerate(backend_names, 1):
    cls = backends[name]
    voice_count = len(cls.list_voices())
    console.print(f'  {i}. [cyan]{name}[/cyan] ({voice_count} voices)')

  # 2. Pick backend
  console.print()
  choice = typer.prompt('Pick a backend', default='1')
  try:
    idx = int(choice) - 1
    backend_name = backend_names[idx]
  except (ValueError, IndexError):
    # Try as name
    if choice in backends:
      backend_name = choice
    else:
      console.print(f'[red]Invalid choice: {choice}[/red]')
      raise typer.Exit(1)

  backend_cls = backends[backend_name]
  console.print(f'[green]Selected:[/green] {backend_name}')

  # 3. Show voices, pick one
  voice_list = backend_cls.list_voices()
  voice_id = None
  if voice_list:
    console.print()
    console.print('[bold]Voices:[/bold]')
    for i, v in enumerate(voice_list[:20], 1):
      lang = v.get('lang', '')
      console.print(f'  {i}. [cyan]{v["name"]}[/cyan] ({lang})')
    if len(voice_list) > 20:
      console.print(f'  [dim]...and {len(voice_list) - 20} more[/dim]')

    console.print()
    vchoice = typer.prompt('Pick a voice (number or name)', default='1')
    try:
      vidx = int(vchoice) - 1
      voice_id = voice_list[vidx]['id']
    except (ValueError, IndexError):
      voice_id = vchoice

    console.print(f'[green]Selected voice:[/green] {voice_id}')

  # 4. Download model if neural backend
  is_neural = backend_name in ('kokoro', 'qwen3', 'pocket')
  if is_neural:
    console.print()
    if typer.confirm('Download model weights?', default=True):
      with console.status('[bold blue]Downloading model...'):
        engine = tts.create(tts.Config(backend=backend_name, voice=voice_id))
        engine.download_model()
      console.print('[green]Model downloaded[/green]')

  # 5. Test voice
  console.print()
  if typer.confirm('Test voice?', default=True):
    with console.status('[bold blue]Generating speech...'):
      engine = tts.create(tts.Config(backend=backend_name, voice=voice_id))
      engine.speak('Hello, this is Claudio. Nice to meet you!')
    console.print('[green]Done[/green]')

  # 6. Save config
  console.print()
  config_path = Path.cwd() / 'claudio.yaml'
  if typer.confirm(f'Save config to {config_path}?', default=True):
    cfg = ClaudioConfig(
      tts=TTSConfig(backend=backend_name, voice=voice_id),
    )
    save_config(cfg, config_path)
    console.print(f'[green]Saved:[/green] {config_path}')


# --- Server commands ---

@app.command()
def speak(
  text: Annotated[Optional[str], typer.Argument(help='Text to speak (or stdin)')] = None,
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  session: Annotated[str, typer.Option('--session')] = 'default',
  mode: Annotated[str, typer.Option('-m', '--mode', help='play|stream|file')] = 'play',
):
  """Speak text via server."""
  if text is None:
    text = sys.stdin.read().strip()
  else:
    text = text.strip()
  if not text:
    return

  ensure_server(server)

  if mode == 'stream':
    response = httpx.post(
      f'{server}/talk/{session}',
      json={'text': text, 'mode': 'stream'},
      timeout=120.0,
    )
    response.raise_for_status()
    sys.stdout.buffer.write(response.content)
  else:
    response = httpx.post(
      f'{server}/talk/{session}',
      json={'text': text, 'mode': mode},
      timeout=120.0,
    )
    response.raise_for_status()
    if mode == 'file':
      console.print(response.json().get('path', ''))


# --- Listen commands (requires server) ---

listen_app = typer.Typer(help='Voice input control (requires server)')
app.add_typer(listen_app, name='listen')


@listen_app.command('start')
def listen_start(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  session: Annotated[str, typer.Option('--session')] = 'default',
  trigger: Annotated[str, typer.Option('-t', '--trigger')] = 'send',
):
  """Start listening for voice input."""
  ensure_server(server)
  response = httpx.post(
    f'{server}/listen/{session}/start',
    json={'trigger_phrase': trigger},
    timeout=30.0,
  )
  response.raise_for_status()
  console.print(f'[green]Listening[/green] (trigger: "{trigger}")')


@listen_app.command('stop')
def listen_stop(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  session: Annotated[str, typer.Option('--session')] = 'default',
):
  """Stop listening."""
  ensure_server(server)
  response = httpx.post(f'{server}/listen/{session}/stop', timeout=5.0)
  response.raise_for_status()
  console.print('[yellow]Stopped[/yellow]')


@listen_app.command('get')
def listen_get(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  session: Annotated[str, typer.Option('--session')] = 'default',
):
  """Get pending transcript."""
  ensure_server(server)
  response = httpx.get(f'{server}/talk/{session}/transcript', timeout=5.0)
  response.raise_for_status()
  transcript = response.json().get('transcript', '')
  if transcript:
    console.print(transcript)


@listen_app.command('send')
def listen_send(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  session: Annotated[str, typer.Option('--session')] = 'default',
):
  """Manually send current transcript."""
  ensure_server(server)
  response = httpx.post(f'{server}/listen/{session}/send', timeout=5.0)
  response.raise_for_status()
  console.print('[green]Sent[/green]')


# --- Persona commands ---

persona_app = typer.Typer(help='Manage personas for voice sessions')
app.add_typer(persona_app, name='persona')


@persona_app.command('list')
def persona_list(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  local: Annotated[bool, typer.Option('--local', '-l', help='List from local directory')] = False,
  directory: Annotated[Optional[Path], typer.Option('--dir', '-d')] = None,
):
  """List available personas."""
  if local:
    from claudio.personas import PersonaStore
    store = PersonaStore(directory or Path.cwd() / 'personas')
    personas = store.list()
  else:
    ensure_server(server)
    response = httpx.get(f'{server}/personas', timeout=5.0)
    response.raise_for_status()
    personas = response.json().get('personas', [])

  if not personas:
    console.print('[dim]No personas found[/dim]')
    return

  table = Table(title='Personas')
  table.add_column('ID', style='cyan')
  table.add_column('Name', style='green')
  table.add_column('Voices')
  table.add_column('Priority')
  table.add_column('Int.', justify='center')  # Interruptible

  for p in personas:
    if hasattr(p, 'id'):
      # Local Persona object
      voices = ', '.join(p.config.voices) if p.config.voices else '-'
      interr = 'Y' if p.config.interruptible else 'N'
      table.add_row(p.id, p.config.name, voices, str(p.config.priority), interr)
    else:
      # Server dict
      voices = ', '.join(p.get('voices') or []) or '-'
      interr = 'Y' if p.get('interruptible', True) else 'N'
      table.add_row(p['id'], p['name'], voices, str(p.get('priority', 50)), interr)

  console.print(table)


@persona_app.command('show')
def persona_show(
  persona_id: Annotated[str, typer.Argument(help='Persona ID to show')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  local: Annotated[bool, typer.Option('--local', '-l', help='Load from local directory')] = False,
  directory: Annotated[Optional[Path], typer.Option('--dir', '-d')] = None,
):
  """Show persona details and prompt."""
  if local:
    from claudio.personas import PersonaStore
    store = PersonaStore(directory or Path.cwd() / 'personas')
    persona = store.get(persona_id)
    if not persona:
      console.print(f'[red]Persona not found: {persona_id}[/red]')
      raise typer.Exit(1)
    console.print(f'[bold cyan]{persona.config.name}[/bold cyan] ({persona.id})')
    voices = ', '.join(persona.config.voices) if persona.config.voices else 'default'
    console.print(f'Voices: {voices}')
    console.print(f'Priority: {persona.config.priority}')
    console.print(f'Interruptible: {"yes" if persona.config.interruptible else "no"}')
    if persona.config.speed:
      console.print(f'Speed override: {persona.config.speed}')
    console.print()
    console.print('[dim]Prompt:[/dim]')
    console.print(persona.prompt)
  else:
    ensure_server(server)
    response = httpx.get(f'{server}/personas/{persona_id}', timeout=5.0)
    if response.status_code == 404:
      console.print(f'[red]Persona not found: {persona_id}[/red]')
      raise typer.Exit(1)
    response.raise_for_status()
    p = response.json()
    console.print(f'[bold cyan]{p["name"]}[/bold cyan] ({p["id"]})')
    voices = ', '.join(p.get('voices') or []) or 'default'
    console.print(f'Voices: {voices}')
    console.print(f'Priority: {p.get("priority", 50)}')
    console.print(f'Interruptible: {"yes" if p.get("interruptible", True) else "no"}')
    if p.get('speed'):
      console.print(f'Speed override: {p["speed"]}')
    console.print()
    console.print('[dim]Prompt:[/dim]')
    console.print(p.get('prompt', ''))


@persona_app.command('set')
def persona_set(
  persona_id: Annotated[str, typer.Argument(help='Persona ID to set')],
  session: Annotated[str, typer.Option('--session')] = 'default',
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Set persona for a session."""
  ensure_server(server)
  response = httpx.post(
    f'{server}/sessions/{session}/persona',
    json={'persona_id': persona_id},
    timeout=5.0,
  )
  if response.status_code == 404:
    console.print(f'[red]Persona not found: {persona_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  result = response.json()
  console.print(f'[green]Set persona[/green] {result["persona"]} for session {result["session"]}')
  console.print(f'Priority: {result["priority"]}')


@persona_app.command('get')
def persona_get(
  session: Annotated[str, typer.Option('--session')] = 'default',
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Get current persona for a session."""
  ensure_server(server)
  response = httpx.get(f'{server}/sessions/{session}/persona', timeout=5.0)
  response.raise_for_status()
  result = response.json()

  if result.get('persona'):
    p = result['persona']
    console.print(f'Session [cyan]{result["session"]}[/cyan]')
    console.print(f'Persona: [green]{p["name"]}[/green] ({p["id"]})')
    console.print(f'Voice: {p.get("voice") or "default"}')
    console.print(f'Speed: {p.get("speed", 1.0)}')
    console.print(f'Priority: {result["priority"]}')
  else:
    console.print(f'Session [cyan]{result["session"]}[/cyan]')
    console.print('[dim]No persona set[/dim]')
    console.print(f'Priority: {result["priority"]}')


@persona_app.command('priority')
def persona_priority(
  priority: Annotated[int, typer.Argument(help='Priority level (lower = higher priority)')],
  session: Annotated[str, typer.Option('--session')] = 'default',
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Set priority for a session (without changing persona)."""
  ensure_server(server)
  response = httpx.post(
    f'{server}/sessions/{session}/persona',
    json={'priority': priority},
    timeout=5.0,
  )
  response.raise_for_status()
  result = response.json()
  console.print(f'[green]Set priority[/green] {result["priority"]} for session {result["session"]}')


@persona_app.command('reload')
def persona_reload(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Reload personas from disk."""
  ensure_server(server)
  response = httpx.post(f'{server}/personas/reload', timeout=5.0)
  response.raise_for_status()
  result = response.json()
  console.print(f'[green]Reloaded[/green] {result["personas"]} personas, {result["voices"]} voices')


# --- Session commands ---

session_app = typer.Typer(help='Manage voice sessions')
app.add_typer(session_app, name='session')


@session_app.command('list')
def session_list(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """List active sessions."""
  ensure_server(server)
  response = httpx.get(f'{server}/sessions', timeout=5.0)
  response.raise_for_status()
  sessions = response.json().get('sessions', [])

  if not sessions:
    console.print('[dim]No active sessions[/dim]')
    return

  table = Table(title='Sessions')
  table.add_column('ID', style='cyan')
  table.add_column('Persona', style='green')
  table.add_column('Voice')
  table.add_column('Priority', justify='right')
  table.add_column('Muted', justify='center')
  table.add_column('Speeches', justify='right')

  for s in sessions:
    muted = '[red]Yes[/red]' if s.get('muted') else '[green]No[/green]'
    table.add_row(
      s['id'],
      s.get('persona_name') or '-',
      s.get('voice_id') or '-',
      str(s.get('priority', 50)),
      muted,
      str(s.get('speech_count', 0)),
    )

  console.print(table)


@session_app.command('show')
def session_show(
  session_id: Annotated[str, typer.Argument(help='Session ID')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Show session details."""
  ensure_server(server)
  response = httpx.get(f'{server}/sessions/{session_id}', timeout=5.0)
  if response.status_code == 404:
    console.print(f'[red]Session not found: {session_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  s = response.json().get('session', {})

  console.print(f'[bold cyan]{s["id"]}[/bold cyan]')
  console.print(f'  Persona:  {s.get("persona_name") or "-"} ({s.get("persona_id") or "none"})')
  console.print(f'  Voice:    {s.get("voice_id") or "-"}')
  console.print(f'  Priority: {s.get("priority", 50)}')
  console.print(f'  Muted:    {"yes" if s.get("muted") else "no"}')
  console.print(f'  Interruptible: {"yes" if s.get("interruptible", True) else "no"}')
  console.print(f'  Speeches: {s.get("speech_count", 0)}')
  if s.get('created_at'):
    from datetime import datetime
    created = datetime.fromtimestamp(s['created_at']).strftime('%H:%M:%S')
    console.print(f'  Created:  {created}')
  if s.get('last_speech_at'):
    from datetime import datetime
    last = datetime.fromtimestamp(s['last_speech_at']).strftime('%H:%M:%S')
    console.print(f'  Last speech: {last}')


@session_app.command('mute')
def session_mute(
  session_id: Annotated[str, typer.Argument(help='Session ID')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Mute a session."""
  ensure_server(server)
  response = httpx.post(f'{server}/sessions/{session_id}/mute', timeout=5.0)
  if response.status_code == 404:
    console.print(f'[red]Session not found: {session_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  console.print(f'[yellow]Muted[/yellow] session {session_id}')


@session_app.command('unmute')
def session_unmute(
  session_id: Annotated[str, typer.Argument(help='Session ID')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Unmute a session."""
  ensure_server(server)
  response = httpx.post(f'{server}/sessions/{session_id}/unmute', timeout=5.0)
  if response.status_code == 404:
    console.print(f'[red]Session not found: {session_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  console.print(f'[green]Unmuted[/green] session {session_id}')


@session_app.command('priority')
def session_priority(
  session_id: Annotated[str, typer.Argument(help='Session ID')],
  priority: Annotated[int, typer.Argument(help='Priority (0-100, higher speaks first)')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Set session priority."""
  ensure_server(server)
  response = httpx.patch(
    f'{server}/sessions/{session_id}',
    json={'priority': priority},
    timeout=5.0,
  )
  if response.status_code == 404:
    console.print(f'[red]Session not found: {session_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  console.print(f'[green]Set priority[/green] {priority} for session {session_id}')


@session_app.command('voice')
def session_voice(
  session_id: Annotated[str, typer.Argument(help='Session ID')],
  voice_id: Annotated[str, typer.Argument(help='Voice ID')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Set session voice."""
  ensure_server(server)
  response = httpx.patch(
    f'{server}/sessions/{session_id}',
    json={'voice_id': voice_id},
    timeout=5.0,
  )
  if response.status_code == 404:
    console.print(f'[red]Session not found: {session_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  console.print(f'[green]Set voice[/green] {voice_id} for session {session_id}')


@session_app.command('delete')
def session_delete(
  session_id: Annotated[str, typer.Argument(help='Session ID')],
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  force: Annotated[bool, typer.Option('-f', '--force', help='Skip confirmation')] = False,
):
  """Delete a session."""
  ensure_server(server)
  if not force:
    confirm = typer.confirm(f'Delete session {session_id}?')
    if not confirm:
      raise typer.Abort()
  response = httpx.delete(f'{server}/sessions/{session_id}', timeout=5.0)
  if response.status_code == 404:
    console.print(f'[red]Session not found: {session_id}[/red]')
    raise typer.Exit(1)
  response.raise_for_status()
  console.print(f'[red]Deleted[/red] session {session_id}')


# --- TUI ---

@app.command()
def tui(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  refresh: Annotated[float, typer.Option('-r', '--refresh', help='Refresh interval in seconds')] = 2.0,
):
  """Interactive TUI for session management.

  Keyboard controls:
    j/k or arrows  - Navigate sessions
    m              - Mute/unmute selected session
    d              - Delete selected session
    +/-            - Adjust priority
    r              - Force refresh
    q              - Quit
  """
  from rich.live import Live
  from rich.panel import Panel
  from rich.text import Text
  from rich.layout import Layout
  import sys
  import select
  import tty
  import termios

  ensure_server(server)

  selected_idx = 0
  sessions = []
  personas = []
  voices = []
  message = ''

  def fetch_data():
    nonlocal sessions, personas, voices
    try:
      sessions = httpx.get(f'{server}/sessions', timeout=2.0).json().get('sessions', [])
      personas = httpx.get(f'{server}/personas', timeout=2.0).json().get('personas', [])
      voices = httpx.get(f'{server}/voices', timeout=2.0).json().get('voices', [])
    except Exception as e:
      pass

  def make_display() -> Layout:
    nonlocal selected_idx, message

    layout = Layout()
    layout.split_column(
      Layout(name='header', size=3),
      Layout(name='main'),
      Layout(name='footer', size=3),
    )

    # Header
    header = Text()
    header.append('Claudio TUI', style='bold cyan')
    header.append(f'  |  {len(sessions)} sessions  |  {len(personas)} personas  |  {len(voices)} voices')
    layout['header'].update(Panel(header, border_style='dim'))

    # Main content
    layout['main'].split_row(
      Layout(name='sessions', ratio=2),
      Layout(name='personas', ratio=1),
    )

    # Sessions panel
    session_text = Text()
    if not sessions:
      session_text.append('No active sessions\n', style='dim italic')
    else:
      for i, s in enumerate(sessions):
        prefix = '> ' if i == selected_idx else '  '
        style = 'bold' if i == selected_idx else ''

        session_text.append(prefix, style='cyan' if i == selected_idx else '')
        session_text.append(f'{s["id"]}', style=style)

        # Status badges
        if s.get('muted'):
          session_text.append(' [MUTED]', style='red')
        session_text.append(f' P{s.get("priority", 50)}', style='dim')

        session_text.append('\n')

        if i == selected_idx:
          # Show details for selected session
          session_text.append(f'    Persona: {s.get("persona_name") or "-"}\n', style='dim')
          session_text.append(f'    Voice:   {s.get("voice_id") or "-"}\n', style='dim')
          session_text.append(f'    Speeches: {s.get("speech_count", 0)}\n', style='dim')

    layout['sessions'].update(Panel(session_text, title='Sessions', border_style='green'))

    # Personas panel
    persona_text = Text()
    if not personas:
      persona_text.append('No personas loaded\n', style='dim italic')
    else:
      for p in personas:
        persona_text.append(f'{p["name"]}', style='cyan')
        persona_text.append(f' P{p.get("priority", 50)}\n', style='dim')

    layout['personas'].update(Panel(persona_text, title='Personas', border_style='blue'))

    # Footer
    footer = Text()
    if message:
      footer.append(message)
      footer.append(' | ')
    footer.append('j/k:nav  m:mute  d:delete  +/-:priority  r:refresh  q:quit', style='dim')
    layout['footer'].update(Panel(footer, border_style='dim'))

    return layout

  def get_key():
    """Get a single keypress (non-blocking)."""
    if select.select([sys.stdin], [], [], 0.1)[0]:
      return sys.stdin.read(1)
    return None

  # Fetch initial data
  fetch_data()

  # Set terminal to raw mode
  old_settings = termios.tcgetattr(sys.stdin)
  try:
    tty.setraw(sys.stdin.fileno())

    with Live(make_display(), refresh_per_second=4, console=console, screen=True) as live:
      last_fetch = time.time()

      while True:
        # Auto-refresh
        if time.time() - last_fetch >= refresh:
          fetch_data()
          last_fetch = time.time()
          live.update(make_display())

        key = get_key()
        if key is None:
          continue

        message = ''

        if key in ('q', '\x03'):  # q or Ctrl+C
          break

        elif key in ('j', '\x1b[B'):  # j or down arrow
          if sessions:
            selected_idx = (selected_idx + 1) % len(sessions)

        elif key in ('k', '\x1b[A'):  # k or up arrow
          if sessions:
            selected_idx = (selected_idx - 1) % len(sessions)

        elif key == 'm':  # mute/unmute
          if sessions and 0 <= selected_idx < len(sessions):
            s = sessions[selected_idx]
            endpoint = 'unmute' if s.get('muted') else 'mute'
            try:
              httpx.post(f'{server}/sessions/{s["id"]}/{endpoint}', timeout=2.0)
              message = f'{"Unmuted" if s.get("muted") else "Muted"} {s["id"]}'
              fetch_data()
            except Exception as e:
              message = f'Error: {e}'

        elif key == 'd':  # delete
          if sessions and 0 <= selected_idx < len(sessions):
            s = sessions[selected_idx]
            try:
              httpx.delete(f'{server}/sessions/{s["id"]}', timeout=2.0)
              message = f'Deleted {s["id"]}'
              fetch_data()
              if selected_idx >= len(sessions):
                selected_idx = max(0, len(sessions) - 1)
            except Exception as e:
              message = f'Error: {e}'

        elif key == '+' or key == '=':  # increase priority
          if sessions and 0 <= selected_idx < len(sessions):
            s = sessions[selected_idx]
            new_priority = min(100, s.get('priority', 50) + 10)
            try:
              httpx.patch(f'{server}/sessions/{s["id"]}', json={'priority': new_priority}, timeout=2.0)
              message = f'Priority -> {new_priority}'
              fetch_data()
            except Exception as e:
              message = f'Error: {e}'

        elif key == '-':  # decrease priority
          if sessions and 0 <= selected_idx < len(sessions):
            s = sessions[selected_idx]
            new_priority = max(0, s.get('priority', 50) - 10)
            try:
              httpx.patch(f'{server}/sessions/{s["id"]}', json={'priority': new_priority}, timeout=2.0)
              message = f'Priority -> {new_priority}'
              fetch_data()
            except Exception as e:
              message = f'Error: {e}'

        elif key == 'r':  # force refresh
          fetch_data()
          last_fetch = time.time()
          message = 'Refreshed'

        live.update(make_display())

  finally:
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


@app.command()
def dashboard(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
):
  """Open the web dashboard in browser."""
  ensure_server(server)
  import webbrowser
  webbrowser.open(server)
  console.print(f'[green]Opened[/green] {server}')


@app.command()
def status(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  proxy: Annotated[str, typer.Option('-p', '--proxy')] = 'http://127.0.0.1:9000',
  json_output: Annotated[bool, typer.Option('--json', help='Output as JSON')] = False,
):
  """Show server and proxy status.

  Shows startup time, active sessions, models in use, and statistics.
  """
  from rich.panel import Panel

  server_status = None
  proxy_status = None

  # Try to get server status
  try:
    response = httpx.get(f'{server}/status', timeout=2.0)
    if response.status_code == 200:
      server_status = response.json()
  except (httpx.ConnectError, httpx.ReadTimeout):
    pass

  # Try to get proxy status
  try:
    response = httpx.get(f'{proxy}/status', timeout=2.0)
    if response.status_code == 200:
      proxy_status = response.json()
  except (httpx.ConnectError, httpx.ReadTimeout):
    pass

  if json_output:
    import json
    console.print(json.dumps({
      'server': server_status,
      'proxy': proxy_status,
    }, indent=2))
    return

  # Display server status
  if server_status:
    srv = server_status.get('server', {})
    stats = server_status.get('stats', {})
    sessions = server_status.get('sessions', [])

    console.print()
    console.print('[bold cyan]Server Status[/bold cyan]')
    console.print(f'  URL:     {server}')
    console.print(f'  Started: {srv.get("started_at_iso", "unknown")}')
    console.print(f'  Uptime:  [green]{srv.get("uptime_human", "?")}[/green]')
    console.print()
    console.print(f'  Sessions:  [cyan]{stats.get("total_sessions", 0)}[/cyan]')
    console.print(f'  Speeches:  [cyan]{stats.get("total_speeches", 0)}[/cyan]')
    console.print(f'  Personas:  {server_status.get("personas", {}).get("count", 0)}')
    console.print(f'  Voices:    {server_status.get("voices", {}).get("count", 0)}')

    if sessions:
      console.print()
      console.print('  [dim]Active Sessions:[/dim]')
      for s in sessions[:5]:
        muted = ' [red][MUTED][/red]' if s.get('muted') else ''
        console.print(f'    - {s["id"]} (P{s.get("priority", 50)}){muted}')
      if len(sessions) > 5:
        console.print(f'    [dim]...and {len(sessions) - 5} more[/dim]')
  else:
    console.print()
    console.print('[yellow]Server:[/yellow] Not running')
    console.print(f'  [dim]Expected at {server}[/dim]')

  # Display proxy status
  if proxy_status:
    prx = proxy_status.get('proxy', {})
    tts_info = proxy_status.get('tts', {})
    session = proxy_status.get('session', {})

    console.print()
    console.print('[bold cyan]Proxy Status[/bold cyan]')
    console.print(f'  URL:     {proxy}')
    console.print(f'  Started: {prx.get("started_at_iso", "unknown")}')
    console.print(f'  Uptime:  [green]{prx.get("uptime_human", "?")}[/green]')
    console.print()
    console.print(f'  Requests:     [cyan]{prx.get("request_count", 0)}[/cyan]')
    console.print(f'  Chars spoken: [cyan]{prx.get("total_chars_spoken", 0)}[/cyan]')
    console.print(f'  TTS time:     [cyan]{prx.get("total_tts_time", 0):.1f}s[/cyan]')

    models = prx.get('models_seen', [])
    if models:
      console.print()
      console.print('  [dim]Models seen:[/dim]')
      for model in models:
        is_last = model == prx.get('last_model')
        marker = ' [green](active)[/green]' if is_last else ''
        console.print(f'    - {model}{marker}')

    console.print()
    console.print(f'  TTS:      [cyan]{tts_info.get("backend", "?")}[/cyan]' +
                  (f' [dim]({tts_info.get("voice")})[/dim]' if tts_info.get('voice') else ''))
    console.print(f'  Session:  {session.get("id", "default")} (P{session.get("priority", 50)})')
    if session.get('persona'):
      console.print(f'  Persona:  {session.get("persona")}')
  else:
    console.print()
    console.print('[yellow]Proxy:[/yellow] Not running')
    console.print(f'  [dim]Expected at {proxy}[/dim]')

  console.print()


# --- Trigger commands ---

@app.command()
def triggers(
  server: Annotated[str, typer.Option('-s', '--server')] = SERVER_URL,
  triggers_dir: Annotated[Optional[Path], typer.Option('-d', '--dir', help='Directory with trigger files')] = None,
  triggers_file: Annotated[Optional[Path], typer.Option('-f', '--file', help='Single trigger file to load')] = None,
  no_hotkeys: Annotated[bool, typer.Option('--no-hotkeys', help='Disable keyboard shortcuts')] = False,
  no_wake_words: Annotated[bool, typer.Option('--no-wake-words', help='Disable wake word detection')] = False,
):
  """Run trigger listeners for keyboard shortcuts and wake words.

  Loads trigger definitions from Python files and starts listening.

  Example trigger file:

      from claudio.triggers import on_hotkey, on_wake_word

      @on_hotkey('cmd+shift+c')
      def start_coding():
          return {'action': 'start_session', 'persona': 'coder'}

      @on_wake_word('hey claude')
      def wake_up():
          return {'action': 'listen_start'}
  """
  from claudio.triggers import run_triggers, load_triggers_file, load_triggers_dir

  # Determine triggers source
  source = triggers_file or triggers_dir
  if not source:
    # Look in default locations
    for path in [
      Path.cwd() / 'triggers',
      Path.cwd() / '.claudio' / 'triggers',
      Path.home() / '.claudio' / 'triggers',
    ]:
      if path.exists():
        source = path
        break

    # Also check for single file
    if not source:
      for path in [
        Path.cwd() / 'triggers.py',
        Path.cwd() / '.claudio' / 'triggers.py',
        Path.home() / '.claudio' / 'triggers.py',
      ]:
        if path.exists():
          source = path
          break

  if not source:
    console.print('[yellow]No triggers found[/yellow]')
    console.print('[dim]Create triggers.py or a triggers/ directory[/dim]')
    raise typer.Exit(1)

  console.print(f'[dim]Loading triggers from {source}[/dim]')

  run_triggers(
    server_url=server,
    enable_hotkeys=not no_hotkeys,
    enable_wake_words=not no_wake_words,
    triggers_dir=source,
  )


@app.command('triggers-list')
def triggers_list(
  triggers_dir: Annotated[Optional[Path], typer.Option('-d', '--dir', help='Directory with trigger files')] = None,
  triggers_file: Annotated[Optional[Path], typer.Option('-f', '--file', help='Single trigger file to load')] = None,
):
  """List registered triggers without running them."""
  from claudio.triggers import get_registry, load_triggers_file, load_triggers_dir

  # Determine triggers source
  source = triggers_file or triggers_dir
  if not source:
    for path in [
      Path.cwd() / 'triggers',
      Path.cwd() / '.claudio' / 'triggers',
      Path.home() / '.claudio' / 'triggers',
      Path.cwd() / 'triggers.py',
      Path.cwd() / '.claudio' / 'triggers.py',
      Path.home() / '.claudio' / 'triggers.py',
    ]:
      if path.exists():
        source = path
        break

  if not source:
    console.print('[yellow]No triggers found[/yellow]')
    return

  # Load triggers
  if source.is_file():
    load_triggers_file(source)
  else:
    load_triggers_dir(source)

  # Display
  registry = get_registry()
  triggers = registry.get_all()

  if not triggers:
    console.print('[dim]No triggers registered[/dim]')
    return

  table = Table(title=f'Triggers from {source}')
  table.add_column('Type', style='cyan')
  table.add_column('Pattern')
  table.add_column('Name', style='green')
  table.add_column('Description')

  for t in triggers:
    pattern = t.pattern
    if isinstance(pattern, list):
      pattern = ', '.join(pattern)
    desc = t.description.split('\n')[0] if t.description else ''
    table.add_row(t.trigger_type, pattern, t.name, desc)

  console.print(table)


# --- Wake word training ---

wakeword_app = typer.Typer(help='Train custom wake word models')
app.add_typer(wakeword_app, name='wakeword')


@wakeword_app.command('train')
def wakeword_train(
  wake_word: Annotated[str, typer.Argument(help='Wake word to train (e.g., "hey claude")')],
  output_dir: Annotated[Optional[Path], typer.Option('-o', '--output', help='Output directory')] = None,
  synthetic: Annotated[int, typer.Option('--synthetic', '-s', help='Number of synthetic samples')] = 5000,
  user_samples: Annotated[int, typer.Option('--user-samples', '-u', help='Number of user samples to record')] = 20,
  augment_multiplier: Annotated[int, typer.Option('--augment', '-a', help='Augmentation multiplier')] = 5,
  skip_record: Annotated[bool, typer.Option('--skip-record', help='Skip user recording')] = False,
):
  """Train a custom wake word model.

  Full pipeline: generate synthetic TTS samples, optionally record
  user voice samples, augment data, train classifier, export to ONNX.

  Examples:
    claudio wakeword train "hey claude"
    claudio wakeword train "hey claude" --synthetic 2000 --skip-record
    claudio wakeword train "activate" -o ./models/activate
  """
  from claudio.wakeword_trainer import WakeWordTrainer

  trainer = WakeWordTrainer(wake_word, output_dir)

  if skip_record:
    user_samples = 0

  model_path = trainer.run_full_pipeline(
    synthetic_count=synthetic,
    user_samples=user_samples,
    augment_multiplier=augment_multiplier,
  )

  if model_path:
    console.print()
    console.print('[bold green]Training complete![/bold green]')
    console.print(f'Model: {model_path}')
    console.print()
    console.print('[dim]To use this model:[/dim]')
    console.print(f'  1. Copy to your models directory')
    console.print(f'  2. Add to OPENWAKEWORD_MODELS in triggers.py')
    console.print(f'  3. Update WAKE_PHRASE_TO_MODEL mapping')


@wakeword_app.command('generate')
def wakeword_generate(
  wake_word: Annotated[str, typer.Argument(help='Wake word')],
  count: Annotated[int, typer.Option('-c', '--count', help='Number of samples')] = 5000,
  output_dir: Annotated[Optional[Path], typer.Option('-o', '--output', help='Output directory')] = None,
):
  """Generate synthetic TTS samples for wake word training.

  Uses multiple TTS voices to create diverse training data.
  Supports Piper TTS, claudio's TTS, and edge-tts as backends.

  Examples:
    claudio wakeword generate "hey claude"
    claudio wakeword generate "hey claude" --count 10000
  """
  from claudio.wakeword_trainer import WakeWordTrainer

  trainer = WakeWordTrainer(wake_word, output_dir)
  generated = trainer.generate_synthetic(count)

  console.print()
  console.print(f'[green]Generated {generated} synthetic samples[/green]')
  console.print(f'Output: {trainer.config.output_dir / "synthetic"}')


@wakeword_app.command('record')
def wakeword_record(
  wake_word: Annotated[str, typer.Argument(help='Wake word to record')],
  count: Annotated[int, typer.Option('-c', '--count', help='Number of samples')] = 20,
  output_dir: Annotated[Optional[Path], typer.Option('-o', '--output', help='Output directory')] = None,
  auto: Annotated[bool, typer.Option('--auto', help='Auto-continue without prompts')] = False,
):
  """Record user voice samples for wake word training.

  Interactive recording session - you'll be prompted to say the
  wake word multiple times. More samples = better personalization.

  Examples:
    claudio wakeword record "hey claude"
    claudio wakeword record "hey claude" --count 50
    claudio wakeword record "hey claude" --auto  # No prompts
  """
  from claudio.wakeword_trainer import WakeWordTrainer

  trainer = WakeWordTrainer(wake_word, output_dir)
  recorded = trainer.record_samples(count, auto_continue=auto)

  console.print()
  console.print(f'[green]Recorded {len(recorded)} samples[/green]')
  console.print(f'Output: {trainer.config.output_dir / "user_samples"}')


@wakeword_app.command('augment')
def wakeword_augment(
  wake_word: Annotated[str, typer.Argument(help='Wake word (for directory location)')],
  multiplier: Annotated[int, typer.Option('-m', '--multiplier', help='Augmentation multiplier')] = 5,
  output_dir: Annotated[Optional[Path], typer.Option('-o', '--output', help='Output directory')] = None,
):
  """Augment existing samples with noise, pitch, tempo variations.

  Creates multiple variations of each sample to improve
  model robustness. Requires librosa for pitch/tempo shifts.

  Examples:
    claudio wakeword augment "hey claude"
    claudio wakeword augment "hey claude" --multiplier 10
  """
  from claudio.wakeword_trainer import WakeWordTrainer

  trainer = WakeWordTrainer(wake_word, output_dir)
  augmented = trainer.augment(multiplier)

  console.print()
  console.print(f'[green]Created {augmented} augmented samples[/green]')
  console.print(f'Output: {trainer.config.output_dir / "augmented"}')


@wakeword_app.command('export')
def wakeword_export(
  wake_word: Annotated[str, typer.Argument(help='Wake word (for directory location)')],
  output_dir: Annotated[Optional[Path], typer.Option('-o', '--output', help='Training directory')] = None,
  model_path: Annotated[Optional[Path], typer.Option('-m', '--model', help='Output model path')] = None,
):
  """Export trained model to ONNX format.

  After training, exports the model for use with openWakeWord.

  Examples:
    claudio wakeword export "hey claude"
    claudio wakeword export "hey claude" -m ./models/hey_claude.onnx
  """
  from claudio.wakeword_trainer import WakeWordTrainer

  trainer = WakeWordTrainer(wake_word, output_dir)
  path = trainer.export(model_path)

  if path:
    console.print()
    console.print(f'[green]Exported model:[/green] {path}')
  else:
    console.print('[red]Export failed - no trained model found[/red]')
    raise typer.Exit(1)


# --- Voice ranking and discovery ---

@app.command()
def voices(
  backend: Annotated[Optional[str], typer.Option('-b', '--backend', help='Show only this backend')] = None,
  all_backends_flag: Annotated[bool, typer.Option('-a', '--all', help='Show all backends (including unavailable)')] = False,
):
  """List available voices grouped by backend."""
  from claudio import tts

  if all_backends_flag:
    backends = tts.all_backends()
  else:
    backends = tts.available_backends()

  if backend:
    if backend not in backends:
      console.print(f'[red]Backend not found: {backend}[/red]')
      available = ', '.join(backends.keys())
      console.print(f'[dim]Available: {available}[/dim]')
      raise typer.Exit(1)
    backends = {backend: backends[backend]}

  if not backends:
    console.print('[yellow]No TTS backends available[/yellow]')
    console.print('[dim]Install one: uv sync --extra kokoro[/dim]')
    return

  table = Table(title='TTS Voices')
  table.add_column('Backend', style='cyan')
  table.add_column('Voice', style='green')
  table.add_column('Language')
  table.add_column('Quality')

  for name, cls in backends.items():
    is_available = cls.available()
    voice_list = cls.list_voices()

    if not voice_list:
      status = '[green]ready[/green]' if is_available else '[red]not installed[/red]'
      table.add_row(name, '[dim]-[/dim]', '', status)
      continue

    for i, v in enumerate(voice_list):
      backend_col = name if i == 0 else ''
      quality = v.get('quality', '')
      if not is_available:
        quality = '[red]not installed[/red]'
      table.add_row(backend_col, v['name'], v.get('lang', ''), quality)

  console.print()
  console.print(table)
  console.print()
  console.print('[dim]Try: claudio try --voice kokoro:af_heart "Hello world"[/dim]')
  console.print('[dim]     claudio try --voice pocket:fantine[/dim]')
  console.print('[dim]Setup: claudio init[/dim]')


def _resolve_voice(voice_str: str) -> tuple[str | None, str]:
  """Parse 'backend:voice' or plain 'voice'. Returns (backend, voice)."""
  if ':' in voice_str:
    backend, voice = voice_str.split(':', 1)
    return backend, voice
  return None, voice_str


def _find_backend_for_voice(voice: str) -> str | None:
  """Find which backend owns a voice (fast, no heavy imports)."""
  from claudio import tts
  for bname, info in tts.available_backends().items():
    for vinfo in info.list_voices():
      if vinfo['id'] == voice or vinfo['name'] == voice:
        return bname
  return None


def _suppress_generate(engine, text):
  """Generate audio with tqdm and stderr suppressed. Returns audio bytes."""
  import contextlib, io
  with contextlib.redirect_stderr(io.StringIO()):
    os.environ['TQDM_DISABLE'] = '1'
    audio = engine.generate(text)
    os.environ.pop('TQDM_DISABLE', None)
  return audio


@app.command('try')
def try_voice(
  text: Annotated[Optional[str], typer.Argument(help='Text to speak')] = None,
  voice: Annotated[str, typer.Option('-v', '--voice', help='Voice: name, backend:name, or comma-sep')] = 'default',
  backend_name: Annotated[Optional[str], typer.Option('-b', '--backend', help='TTS backend')] = None,
  speed: Annotated[float, typer.Option('-s', '--speed', help='Speed multiplier')] = 1.0,
):
  """Try voice(s) with sample text, then enter interactive mode.

  Voices can be specified as backend:voice to skip auto-detection.
  After the first playback, enter an interactive loop to keep testing.

  Interactive commands:
    <text>                    Speak the text
    (empty enter)             Repeat last text
    /voices [b1,b2]           List voices (current or specified backends)
    /voice <name|num>         Switch voice within current backend
    /voice backend:name       Switch backend and voice
    /quit, /q                 Exit

  Examples:
    claudio try                              # Try system default
    claudio try --voice af_heart             # Auto-detect backend
    claudio try --voice pocket:fantine       # Explicit backend
    claudio try --voice "af_heart,pocket:fantine" # Mix styles
    claudio try --backend say --voice Daniel # Backend flag
    claudio try "Hello world"               # Custom text
  """
  from claudio import tts

  # Parse voice specs
  if voice == 'default':
    # Use Config defaults (kokoro:af_heart)
    defaults = tts.Config()
    specs = [(defaults.backend, defaults.voice)]
  else:
    specs = [_resolve_voice(v.strip()) for v in voice.split(',')]

  if not specs:
    console.print('[yellow]No voices found[/yellow]')
    return

  multiple = len(specs) > 1
  if multiple:
    console.print(f'[dim]Trying {len(specs)} voices...[/dim]')
    console.print()

  # Track current state for interactive mode
  current_backend = None
  current_voice = None
  current_engine = None
  last_text = None
  engine_cache = {}  # (backend, voice) -> engine

  for i, (explicit_backend, v) in enumerate(specs):
    say_text = text or f"Hi, this is Claudio speaking to you using the {v or 'default'} voice. What do you think?"

    # Resolve backend: inline prefix > --backend flag > voice lookup > auto
    resolved = explicit_backend or backend_name
    if not resolved and v:
      resolved = _find_backend_for_voice(v)
    resolved = resolved or 'auto'

    engine = tts.create(tts.Config(backend=resolved, voice=v, rate=speed))

    # Warmup: load model
    console.print(f'\n  [dim]loading {repr(engine)}...[/dim]')
    t0 = time.perf_counter()
    _suppress_generate(engine, 'test')
    load_time = time.perf_counter() - t0
    console.print(f'  [cyan]{repr(engine)}[/cyan] [dim]loaded in {load_time:.4g}s[/dim]')

    # Timed generation
    t0 = time.perf_counter()
    audio = _suppress_generate(engine, say_text)
    gen_time = time.perf_counter() - t0
    console.print(f'  [dim]generated in {gen_time:.4g}s[/dim]')
    if audio:
      engine.play(audio)

    current_backend = resolved
    current_voice = v
    current_engine = engine
    engine_cache[(resolved, v)] = engine
    last_text = say_text

    if multiple and i < len(specs) - 1:
      time.sleep(0.5)

  # Interactive loop
  current_engines = [current_engine]
  console.print()
  console.print('[dim]Interactive mode. Type text to speak, /q to quit, /voices to list voices.[/dim]')

  try:
    while True:
      try:
        line = input('> ').strip()
      except EOFError:
        break

      if not line:
        # Repeat last text
        if not last_text:
          continue
        line = last_text
      elif line in ('/quit', '/q'):
        break
      elif line.startswith('/voices') or line.startswith('/v ') or line == '/v':
        # /voices [backend,...] — list voices for backends (default: current)
        arg = ''
        if line.startswith('/voices '):
          arg = line[8:].strip()
        elif line.startswith('/v '):
          arg = line[3:].strip()
        backends = tts.available_backends()
        if arg:
          show_backends = [b.strip() for b in arg.split(',')]
        else:
          actual = current_backend
          if actual == 'auto':
            actual = tts.Config().backend
          show_backends = [actual]
        for bname in show_backends:
          if bname not in backends:
            console.print(f'  [red]Unknown backend: {bname}[/red]')
            continue
          voice_list = backends[bname].list_voices()
          parts = [f'{i+1}. {v["name"]}' for i, v in enumerate(voice_list)]
          console.print(f'  [cyan]{bname}:[/cyan] ' + '  '.join(parts))
        continue
      elif line.startswith('/voice '):
        arg = line[7:].strip()
        if not arg:
          console.print('  [dim]Usage: /voice <name>, /voice a,b,c, /voice backend:name[/dim]')
          continue
        backends = tts.available_backends()
        actual = current_backend
        if actual == 'auto':
          actual = tts.Config().backend
        voice_list = backends[actual].list_voices() if actual in backends else []
        # Parse comma-separated specs into (backend, voice) pairs
        # Supports: name, number, glob (al*), backend:name, backend:glob (pocket:*)
        from fnmatch import fnmatch
        engines = []
        for part in arg.split(','):
          part = part.strip()
          if not part:
            continue
          if ':' in part:
            b, pattern = part.split(':', 1)
            if b not in backends:
              console.print(f'  [red]Unknown backend: {b}[/red]')
              continue
            if any(c in pattern for c in '*?['):
              # Glob: expand against backend's voice list
              bvoices = backends[b].list_voices()
              matched = [v['id'] for v in bvoices if fnmatch(v['name'], pattern) or fnmatch(v['id'], pattern)]
              if not matched:
                console.print(f'  [dim]no voices matching {pattern} in {b}[/dim]')
              engines.extend((b, v) for v in matched)
            else:
              engines.append((b, pattern))
          else:
            if any(c in part for c in '*?['):
              # Glob within current backend
              matched = [v['id'] for v in voice_list if fnmatch(v['name'], part) or fnmatch(v['id'], part)]
              if not matched:
                console.print(f'  [dim]no voices matching {part}[/dim]')
              engines.extend((current_backend, v) for v in matched)
            else:
              # Resolve number or name within current backend
              resolved = None
              try:
                idx = int(part) - 1
                if 0 <= idx < len(voice_list):
                  resolved = voice_list[idx]['id']
              except ValueError:
                resolved = part
              if resolved:
                engines.append((current_backend, resolved))
        if not engines:
          continue
        # Build engine list, warmup each (use cache when available)
        current_engines = []
        for b, v in engines:
          cache_key = (b, v)
          if cache_key in engine_cache:
            eng = engine_cache[cache_key]
            console.print(f'  [dim]cached {repr(eng)}[/dim]')
          else:
            eng = tts.create(tts.Config(backend=b, voice=v, rate=speed))
            console.print(f'  [dim]loading {repr(eng)}...[/dim]')
            t0 = time.perf_counter()
            _suppress_generate(eng, 'test')
            load_time = time.perf_counter() - t0
            console.print(f'  [dim]loaded in {load_time:.4g}s[/dim]')
            engine_cache[cache_key] = eng
          current_engines.append(eng)
        # Update current to last one for single-voice fallback
        current_backend = engines[-1][0]
        current_voice = engines[-1][1]
        current_engine = current_engines[-1]
        names = ', '.join(repr(e) for e in current_engines)
        console.print(f'  switched to [cyan]{names}[/cyan]')
        continue
      elif line.startswith('/'):
        console.print('  [dim]Commands: /voices [b1,b2] /voice <a,b,backend:c> /quit[/dim]')
        continue

      # Generate and play through all active engines
      last_text = line
      for eng in current_engines:
        console.print(f'  [dim]{repr(eng)}[/dim]')
        t0 = time.perf_counter()
        audio = _suppress_generate(eng, line)
        gen_time = time.perf_counter() - t0
        console.print(f'  [dim]generated in {gen_time:.4g}s[/dim]')
        if audio:
          eng.play(audio)

  except KeyboardInterrupt:
    pass

  console.print()


@app.command()
def devices(
  server: Annotated[Optional[str], typer.Option('-s', '--server')] = None,
):
  """List audio devices."""
  if server:
    ensure_server(server)
    response = httpx.get(f'{server}/devices', timeout=5.0)
    response.raise_for_status()
    devs = response.json().get('devices', [])
  else:
    import sounddevice as sd
    devs = []
    for i, dev in enumerate(sd.query_devices()):
      devs.append({
        'id': i,
        'name': dev['name'],
        'inputs': dev['max_input_channels'],
        'outputs': dev['max_output_channels'],
        'default_input': i == sd.default.device[0],
        'default_output': i == sd.default.device[1],
      })

  table = Table(title='Audio Devices')
  table.add_column('ID', style='cyan', justify='right')
  table.add_column('Name')
  table.add_column('In', justify='right')
  table.add_column('Out', justify='right')
  table.add_column('Default')

  for dev in devs:
    default = []
    if dev.get('default_input'):
      default.append('[green]IN[/green]')
    if dev.get('default_output'):
      default.append('[blue]OUT[/blue]')
    table.add_row(
      str(dev['id']),
      dev['name'],
      str(dev['inputs']),
      str(dev['outputs']),
      ' '.join(default),
    )

  console.print(table)


# --- Server modes ---

@app.command()
def asr(
  backend: Annotated[str, typer.Option('-b', '--backend', help='ASR backend: parakeet, whisper, apple')] = 'parakeet',
  device: Annotated[Optional[int], typer.Option('-d', '--device', help='Input device ID')] = None,
  window: Annotated[float, typer.Option('-w', '--window', help='Audio window size (seconds)')] = 10.0,
  step: Annotated[float, typer.Option('-s', '--step', help='Step between transcriptions (seconds)')] = 0.1,
  locale: Annotated[str, typer.Option('-l', '--locale', help='Locale for Apple backend')] = 'en-US',
):
  """Live ASR testing - see transcription as you speak.

  Shows real-time speech recognition with latency and accuracy stats.
  Useful for testing different backends and tuning parameters.

  Backends:
    parakeet  - MLX Parakeet (fast, good accuracy, requires parakeet-mlx)
    whisper   - MLX Whisper (very accurate, requires mlx-audio)
    apple     - macOS Speech (no deps, uses Neural Engine, works offline)

  Examples:
    claudio asr                          # Default (parakeet)
    claudio asr -b apple                 # Apple Neural Engine
    claudio asr -b whisper               # Whisper (most accurate)
    claudio asr -b parakeet -w 5 -s 0.2  # Faster updates, shorter window
  """
  from rich.live import Live
  from rich.panel import Panel
  from rich.text import Text

  try:
    from claudio.transcribe import get_transcriber
  except ImportError as e:
    console.print(f'[red]Transcription requires ASR:[/red] uv sync --extra asr')
    console.print(f'[dim]{e}[/dim]')
    raise typer.Exit(1)

  console.print()
  console.print('[bold green]Live ASR Test[/bold green]')
  console.print(f'  Backend:  [cyan]{backend}[/cyan]')
  if backend == 'parakeet':
    console.print(f'  Window:   [cyan]{window}s[/cyan]')
    console.print(f'  Step:     [cyan]{step}s[/cyan]')
  elif backend == 'apple':
    console.print(f'  Locale:   [cyan]{locale}[/cyan]')
    console.print(f'  Engine:   [dim]Neural Engine (on-device)[/dim]')
  console.print()
  console.print('[dim]Speak into your microphone. Press Ctrl+C to stop.[/dim]')
  console.print()

  # Stats tracking
  stats = {
    'chunks': 0,
    'total_latency_ms': 0.0,
    'min_latency_ms': float('inf'),
    'max_latency_ms': 0.0,
    'last_text': '',
    'start_time': time.time(),
  }

  def make_display() -> Panel:
    """Create the live display panel."""
    text = Text()

    # Current transcript
    transcript = stats['last_text'] or '[dim]Listening...[/dim]'
    text.append('📝 ', style='bold')
    text.append(transcript + '\n\n', style='white')

    # Stats
    elapsed = time.time() - stats['start_time']
    text.append('Stats\n', style='bold dim')
    text.append(f'  Duration:    {elapsed:.1f}s\n', style='dim')
    text.append(f'  Chunks:      {stats["chunks"]}\n', style='dim')

    if stats['chunks'] > 0:
      avg_latency = stats['total_latency_ms'] / stats['chunks']
      text.append(f'  Avg latency: {avg_latency:.0f}ms\n', style='dim')
      text.append(f'  Min latency: {stats["min_latency_ms"]:.0f}ms\n', style='dim')
      text.append(f'  Max latency: {stats["max_latency_ms"]:.0f}ms\n', style='dim')

    return Panel(text, title=f'[cyan]{backend}[/cyan] ASR', border_style='green')

  # Get transcriber with appropriate kwargs
  kwargs = {}
  if backend == 'parakeet':
    kwargs = {'window_seconds': window, 'step_seconds': step}
  elif backend == 'apple':
    kwargs = {'locale': locale}

  try:
    transcriber = get_transcriber(backend=backend, device=device, **kwargs)
  except Exception as e:
    console.print(f'[red]Failed to initialize {backend}:[/red] {e}')
    raise typer.Exit(1)

  try:
    with Live(make_display(), refresh_per_second=10, console=console) as live:
      for chunk in transcriber.stream():
        stats['chunks'] += 1
        stats['last_text'] = chunk.text

        if chunk.latency_ms is not None:
          stats['total_latency_ms'] += chunk.latency_ms
          stats['min_latency_ms'] = min(stats['min_latency_ms'], chunk.latency_ms)
          stats['max_latency_ms'] = max(stats['max_latency_ms'], chunk.latency_ms)

        live.update(make_display())

  except KeyboardInterrupt:
    pass
  finally:
    transcriber.stop()

  # Final summary
  console.print()
  console.print('[bold]Session Summary[/bold]')
  elapsed = time.time() - stats['start_time']
  console.print(f'  Duration:    {elapsed:.1f}s')
  console.print(f'  Chunks:      {stats["chunks"]}')
  if stats['chunks'] > 0:
    avg_latency = stats['total_latency_ms'] / stats['chunks']
    console.print(f'  Avg latency: {avg_latency:.0f}ms')
    console.print(f'  Min latency: {stats["min_latency_ms"]:.0f}ms')
    console.print(f'  Max latency: {stats["max_latency_ms"]:.0f}ms')
  console.print()


@app.command()
def dictate(
  backend: Annotated[str, typer.Option('-b', '--backend', help='ASR backend: parakeet, whisper')] = 'parakeet',
  device: Annotated[Optional[int], typer.Option('-d', '--device', help='Input device ID')] = None,
  trigger: Annotated[Optional[str], typer.Option('-t', '--trigger', help='Trigger phrase to submit')] = 'send',
  silence: Annotated[float, typer.Option('--silence', help='Silence duration to auto-submit')] = 1.5,
  auto_submit: Annotated[bool, typer.Option('--auto-submit', help='Press Enter after pasting')] = False,
  once: Annotated[bool, typer.Option('--once', help='Stop after first submit (default: continuous)')] = False,
):
  """Voice dictation - speak and text appears in focused app.

  Start speaking after the 🎤 indicator appears. Your words are typed
  into whatever app/field is focused.

  To submit: say "send" (or custom trigger), or wait for silence.
  Keeps running for multiple dictations - Ctrl+C to stop.

  Examples:
    claudio dictate                    # Continuous mode (default)
    claudio dictate --once             # Stop after first submit
    claudio dictate --trigger done     # Say "done" to submit
    claudio dictate --auto-submit      # Press Enter after text
    claudio dictate --silence 2.0      # Wait 2s of silence
  """
  try:
    from claudio.voice_input import VoiceInputConfig, run_voice_input, check_accessibility_permission
  except ImportError as e:
    console.print(f'[red]Voice input requires ASR:[/red] uv sync --extra asr')
    console.print(f'[dim]{e}[/dim]')
    raise typer.Exit(1)

  # Check accessibility permission for auto-paste
  has_accessibility = check_accessibility_permission()
  if not has_accessibility:
    console.print()
    console.print('[yellow]⚠ Accessibility permission needed for auto-paste[/yellow]')
    console.print('[dim]  System Settings → Privacy & Security → Accessibility[/dim]')
    console.print('[dim]  Add your terminal app (Terminal, iTerm2, etc.)[/dim]')
    console.print('[dim]  Without it, text is copied to clipboard only.[/dim]')

  config = VoiceInputConfig(
    backend=backend,
    device=device,
    trigger_phrase=trigger,
    silence_seconds=silence,
    auto_submit=auto_submit,
    continuous=not once,
  )

  console.print()
  console.print('[bold green]Voice Input[/bold green]')
  console.print(f'  Backend:  [cyan]{backend}[/cyan]')
  console.print(f'  Trigger:  [cyan]"{trigger}"[/cyan] [dim](say this to submit)[/dim]')
  console.print(f'  Silence:  [cyan]{silence}s[/cyan] [dim](auto-submit after silence)[/dim]')
  if auto_submit:
    console.print(f'  Submit:   [cyan]auto[/cyan] [dim](press Enter after paste)[/dim]')
  if not once:
    console.print(f'  Mode:     [cyan]continuous[/cyan] [dim](keeps listening after submit)[/dim]')
  console.print()
  console.print('[dim]Focus your target app, then speak. Ctrl+C to stop.[/dim]')
  console.print()

  run_voice_input(config, show_ui=True)


# --- Server commands ---

server_app = typer.Typer(help='Claudio HTTP/MCP server')
app.add_typer(server_app, name='server')


@server_app.callback(invoke_without_command=True)
def server_start(
  ctx: typer.Context,
  mcp: Annotated[bool, typer.Option('--mcp', help='Run as MCP server (stdio)')] = False,
  host: Annotated[str, typer.Option('-h', '--host')] = '127.0.0.1',
  port: Annotated[int, typer.Option('-p', '--port')] = 8765,
  device: Annotated[Optional[int], typer.Option('-d', '--device', help='Output device ID')] = None,
):
  """Start the Claudio server (requires: claudio[server])."""
  if ctx.invoked_subcommand is not None:
    return

  try:
    from claudio.server import run_http, run_mcp_stdio
  except ImportError:
    console.print('[red]Server requires:[/red] uv sync --extra server')
    raise typer.Exit(1)

  if mcp:
    run_mcp_stdio(output_device=device)
  else:
    run_http(host=host, port=port, output_device=device)


@server_app.command('stop')
def server_stop(
  port: Annotated[int, typer.Option('-p', '--port')] = 8765,
):
  """Stop a running Claudio server."""
  _stop_process_on_port(port, 'Server')


# --- Proxy commands ---

proxy_app = typer.Typer(help='Claude API proxy with TTS')
app.add_typer(proxy_app, name='proxy')


@proxy_app.callback(invoke_without_command=True)
def proxy_start(
  ctx: typer.Context,
  host: Annotated[str, typer.Option('-h', '--host')] = '127.0.0.1',
  port: Annotated[int, typer.Option('-p', '--port')] = 9000,
  device: Annotated[Optional[int], typer.Option('-d', '--device', help='Output device ID')] = None,
  speak: Annotated[str, typer.Option('-s', '--speak', help='Speak mode: tags, text, all, off')] = 'tags',
  tts: Annotated[str, typer.Option('-t', '--tts', help='TTS backend: auto, say, soprano')] = 'auto',
  voice: Annotated[Optional[str], typer.Option('-v', '--voice', help='Voice name (for say backend)')] = None,
  speed: Annotated[float, typer.Option('--speed', help='Speech speed multiplier')] = 1.0,
  persona: Annotated[Optional[str], typer.Option('--persona', help='Persona ID to use')] = None,
  personas_dir: Annotated[Optional[Path], typer.Option('--personas-dir', help='Personas directory')] = None,
  session: Annotated[str, typer.Option('--session', help='Session ID for speech coordination')] = 'default',
):
  """Start Claude API proxy with TTS. Use <say>text</say> tags for speech."""
  if ctx.invoked_subcommand is not None:
    return

  try:
    from claudio.proxy import run_proxy, ProxyConfig
  except ImportError:
    console.print('[red]Proxy requires:[/red] uv sync --extra server')
    raise typer.Exit(1)

  proxy_url = f'http://{host}:{port}'
  console.print()
  console.print(f'[bold green]Claudio Proxy[/bold green] starting on [cyan]{proxy_url}[/cyan]')
  console.print()
  console.print('[dim]To use with Claude Code:[/dim]')
  console.print(f'  export ANTHROPIC_BASE_URL={proxy_url}')
  console.print()

  import asyncio
  config = ProxyConfig(
    listen_host=host,
    listen_port=port,
    speak_mode=speak,
  )
  asyncio.run(run_proxy(
    config,
    output_device=device,
    tts_backend=tts,
    tts_voice=voice,
    tts_speed=speed,
    persona_id=persona,
    personas_dir=str(personas_dir) if personas_dir else None,
    session_id=session,
  ))


@proxy_app.command('stop')
def proxy_stop(
  port: Annotated[int, typer.Option('-p', '--port')] = 9000,
):
  """Stop a running proxy."""
  _stop_process_on_port(port, 'Proxy')


# --- Main command: launch Claude with voice ---

@app.callback(invoke_without_command=True)
def callback(
  ctx: typer.Context,
  # Mode selection
  mode: Annotated[str, typer.Option('-m', '--mode', help='Mode: proxy (default), mcp, or off')] = 'proxy',
  # TTS options (None = read from claudio.yaml)
  tts: Annotated[Optional[str], typer.Option('-t', '--tts', help='TTS backend: pocket, kokoro, say, off')] = None,
  voice: Annotated[Optional[str], typer.Option('-v', '--voice', help='Voice name')] = None,
  speed: Annotated[Optional[float], typer.Option('--speed', help='Speech speed multiplier')] = None,
  # Speech mode
  speak: Annotated[Optional[str], typer.Option('-s', '--speak', help='Speak mode: tags, text, all, off')] = None,
  # Notifications
  notify: Annotated[Optional[str], typer.Option('-n', '--notify', help='Notify on: tools, approval, or tool names (comma-sep)')] = None,
  # ASR options
  asr: Annotated[str, typer.Option('-a', '--asr', help='ASR backend: off, whisper, deepgram')] = 'off',
  # Audio device
  device: Annotated[Optional[int], typer.Option('-d', '--device', help='Output device ID')] = None,
  # Network
  port: Annotated[int, typer.Option('-p', '--port', help='Proxy port')] = DEFAULT_PROXY_PORT,
  proxy_url: Annotated[Optional[str], typer.Option('--proxy-url', help='Use existing proxy at URL')] = None,
  # Persona options
  persona: Annotated[Optional[str], typer.Option('--persona', help='Persona ID to use')] = None,
  personas_dir: Annotated[Optional[Path], typer.Option('--personas-dir', help='Personas directory')] = None,
  session: Annotated[str, typer.Option('--session', help='Session ID for speech coordination')] = 'default',
):
  """Launch Claude Code with voice I/O.

  Examples:
    claudio                      # Start with proxy + TTS (tags mode)
    claudio --speak text         # Speak text, skip code
    claudio --speak all          # Speak everything incl. code
    claudio --notify tools       # Announce tool usage
    claudio --tts say --voice Daniel
    claudio --persona claude     # Use the 'claude' persona
  """
  if ctx.invoked_subcommand is not None:
    return

  # Load config, use as defaults for any unset CLI options
  from claudio.config import load_config
  try:
    cfg = load_config()
  except Exception:
    from claudio.config import ClaudioConfig
    cfg = ClaudioConfig()

  # Speak mode map: config uses proxy speak_mode names, CLI uses same names
  speak_mode_map = {'tags': 'tags', 'text': 'text', 'all': 'all', 'off': 'off'}

  # Auto-detect backend from voice name if --tts not explicitly set
  resolved_tts = tts or cfg.tts.backend
  resolved_voice = voice or cfg.tts.voice
  if not tts and voice:
    detected = _find_backend_for_voice(voice)
    if detected:
      resolved_tts = detected

  _launch_claude(
    mode=mode,
    tts=resolved_tts,
    voice=resolved_voice,
    speed=speed if speed is not None else cfg.tts.speed,
    speak=speak or speak_mode_map.get(cfg.proxy.speak_mode, 'tags'),
    notify=notify or cfg.proxy.notify,
    asr=asr,
    device=device if device is not None else cfg.tts.device,
    port=port,
    proxy_url=proxy_url,
    persona=persona,
    personas_dir=str(personas_dir) if personas_dir else None,
    session=session,
  )


def _check_proxy_running(url: str) -> bool:
  """Check if a proxy is already running at the given URL."""
  try:
    resp = httpx.get(f'{url}/health', timeout=1.0)
    return resp.status_code == 200
  except (httpx.ConnectError, httpx.ReadTimeout):
    return False


def _launch_claude(
  mode: str,
  tts: str,
  voice: str | None,
  speed: float,
  speak: str,
  notify: str | None,
  asr: str,
  device: int | None,
  port: int,
  proxy_url: str | None,
  persona: str | None = None,
  personas_dir: str | None = None,
  session: str = 'default',
):
  """Launch Claude with voice configuration."""
  proxy_process = None
  used_proxy_url = None

  try:
    if mode == 'proxy':
      # Check if using existing proxy or starting new one
      if proxy_url:
        # User specified a proxy URL - check if it's running
        if _check_proxy_running(proxy_url):
          used_proxy_url = proxy_url
          console.print(f'[green]Using existing proxy:[/green] {proxy_url}')
        else:
          console.print(f'[yellow]Proxy not running at {proxy_url}, starting new one...[/yellow]')

      if not used_proxy_url:
        # Start proxy in background
        proxy_process = _start_proxy(
          port=port,
          tts=tts,
          voice=voice,
          speed=speed,
          speak=speak,
          notify=notify,
          device=device,
          persona=persona,
          personas_dir=personas_dir,
          session=session,
        )
        used_proxy_url = f'http://127.0.0.1:{port}'

      # Set environment for Claude
      env = os.environ.copy()
      env['ANTHROPIC_BASE_URL'] = used_proxy_url

      # Resolve actual backend
      resolved_backend = tts
      if tts == 'auto':
        from claudio.tts import Config as _TtsConfig
        resolved_backend = _TtsConfig().backend

      # Get voice skill
      skill = _get_voice_skill(speak)

      console.print()
      console.print('[bold green]Claudio[/bold green] [dim]Voice I/O for Claude Code[/dim]')
      console.print(f'  Proxy:   [cyan]{used_proxy_url}[/cyan]' + (' [dim](external)[/dim]' if proxy_url and not proxy_process else ''))
      if proxy_process:
        tts_display = f'[cyan]{resolved_backend}[/cyan]'
        if tts == 'auto':
          tts_display += ' [dim](auto)[/dim]'
        if voice:
          tts_display += f' [dim]voice={voice}[/dim]'
        if speed != 1.0:
          tts_display += f' [dim]speed={speed}[/dim]'
        console.print(f'  TTS:     {tts_display}')
        speak_display = f'[cyan]{speak}[/cyan]'
        if speak == 'tags':
          speak_display += ' [dim](<say> tags)[/dim]'
        elif speak == 'text':
          speak_display += ' [dim](all text, skip code)[/dim]'
        elif speak == 'all':
          speak_display += ' [dim](everything incl. code)[/dim]'
        console.print(f'  Speak:   {speak_display}')
        if notify:
          console.print(f'  Notify:  [cyan]{notify}[/cyan]')
      if asr != 'off':
        console.print(f'  ASR:     [cyan]{asr}[/cyan]')
      if skill:
        console.print(f'  Skill:   [dim]voice prompt injected[/dim]')
      console.print()

      # Launch Claude
      cmd = ['claude']
      if skill:
        cmd.extend(['--append-system-prompt', skill])
      result = subprocess.run(cmd, env=env)
      sys.exit(result.returncode)

    elif mode == 'mcp':
      # Configure Claude to use MCP server
      console.print('[yellow]MCP mode not yet implemented[/yellow]')
      console.print('[dim]Use: claudio server --mcp[/dim]')
      raise typer.Exit(1)

    elif mode == 'off':
      # Just launch Claude normally
      result = subprocess.run(['claude'])
      sys.exit(result.returncode)

    else:
      console.print(f'[red]Unknown mode:[/red] {mode}')
      raise typer.Exit(1)

  except KeyboardInterrupt:
    pass
  finally:
    if proxy_process:
      # Wait for TTS to finish before terminating
      try:
        httpx.get(f'{used_proxy_url}/wait', timeout=30.0)
      except Exception:
        pass  # Proxy may already be dead
      proxy_process.terminate()
      proxy_process.wait(timeout=5)


def _start_proxy(
  port: int,
  tts: str,
  voice: str | None,
  speed: float,
  speak: str,
  notify: str | None,
  device: int | None,
  persona: str | None = None,
  personas_dir: str | None = None,
  session: str = 'default',
) -> subprocess.Popen:
  """Start proxy server in background, wait for it to be ready."""
  # Map speak modes to proxy speak modes (1:1 now)
  proxy_speak = {'tags': 'tags', 'text': 'text', 'all': 'all', 'off': 'off'}.get(speak, 'tags')

  cmd = [
    sys.executable, '-m', 'claudio.proxy',
    '--port', str(port),
    '--speak', proxy_speak,
  ]

  if tts != 'off':
    cmd.extend(['--tts', tts])

  if voice:
    cmd.extend(['--voice', voice])

  if speed != 1.0:
    cmd.extend(['--speed', str(speed)])

  if notify:
    cmd.extend(['--notify', notify])

  if device is not None:
    cmd.extend(['--device', str(device)])

  if persona:
    cmd.extend(['--persona', persona])

  if personas_dir:
    cmd.extend(['--personas-dir', personas_dir])

  if session != 'default':
    cmd.extend(['--session', session])

  # Create log directory
  from datetime import datetime
  log_dir = Path('.claudio/logs')
  log_dir.mkdir(parents=True, exist_ok=True)
  log_file = log_dir / f'proxy-{datetime.now().strftime("%Y%m%d-%H%M%S")}.jsonl'

  # Tell proxy to log to file
  env = os.environ.copy()
  env['CLAUDIO_CHILD'] = '1'
  env['CLAUDIO_LOG_FILE'] = str(log_file)

  # Always redirect stdout/stderr when running as child
  process = subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env=env,
    start_new_session=True,
  )

  # Wait for proxy to be ready
  proxy_url = f'http://127.0.0.1:{port}'
  for _ in range(50):  # 5 seconds max
    try:
      resp = httpx.get(f'{proxy_url}/health', timeout=0.5)
      if resp.status_code == 200:
        return process
    except (httpx.ConnectError, httpx.ReadTimeout):
      pass
    time.sleep(0.1)

  console.print('[red]Proxy failed to start[/red]')
  process.terminate()
  raise typer.Exit(1)


def main():
  """Entry point for claudio command."""
  app()


if __name__ == '__main__':
  main()
