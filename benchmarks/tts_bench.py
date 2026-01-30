"""TTS benchmarks - compare backends, voices, and settings."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import duckdb
import pyarrow as pa
import typer

app = typer.Typer(help='TTS benchmarks')

# Sample texts of varying lengths
TEXTS = {
  'tiny': 'Hello.',
  'short': 'The quick brown fox jumps over the lazy dog.',
  'medium': '''
    Claude is an AI assistant created by Anthropic to be helpful, harmless, and honest.
    It can help with analysis, coding, math, writing, and many other tasks.
  '''.strip(),
  'long': '''
    Text-to-speech technology has evolved significantly over the past decade.
    Modern neural TTS systems can produce remarkably natural-sounding speech,
    with appropriate prosody, intonation, and emotional expression.
    The challenge remains in achieving low latency for real-time applications
    while maintaining high quality output. Different backends offer various
    trade-offs between speed, quality, and resource usage.
  '''.strip(),
}

# Default voices to test
VOICES = ['Samantha', 'Daniel', 'Karen', None]  # None = system default

DB_PATH = Path(__file__).parent / 'results.duckdb'
AUDIO_DIR = Path(__file__).parent / 'audio'


def init_db(db: duckdb.DuckDBPyConnection):
  """Initialize database schema."""
  db.execute('''
    CREATE TABLE IF NOT EXISTS runs (
      id VARCHAR PRIMARY KEY,
      timestamp TIMESTAMP,
      backend VARCHAR,
      voice VARCHAR,
      text_id VARCHAR,
      text_len INTEGER,
      chunk_size INTEGER,
      generate_ms DOUBLE,
      audio_bytes INTEGER,
      audio_hash VARCHAR,
      error VARCHAR
    )
  ''')


def audio_path(run_id: str) -> Path:
  """Get audio file path for a run."""
  return AUDIO_DIR / f'{run_id}.aiff'


def run_benchmark(
  backend: str,
  voice: str | None,
  text_id: str,
  text: str,
  chunk_size: int | None = None,
) -> dict:
  """Run a single benchmark."""
  from claudio.tts import Config, create

  run_id = f'{backend}_{voice or "default"}_{text_id}_{chunk_size or "full"}_{int(time.time() * 1000)}'

  config = Config(backend=backend, voice=voice)

  try:
    tts = create(config)

    # Generate
    start = time.perf_counter()
    if chunk_size and len(text) > chunk_size:
      # Chunked generation
      chunks = []
      for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
        chunks.append(tts.generate(chunk_text))
      audio = b''.join(chunks)
    else:
      audio = tts.generate(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Save audio
    AUDIO_DIR.mkdir(exist_ok=True)
    audio_path(run_id).write_bytes(audio)

    return {
      'id': run_id,
      'timestamp': datetime.now(),
      'backend': backend,
      'voice': voice or 'default',
      'text_id': text_id,
      'text_len': len(text),
      'chunk_size': chunk_size,
      'generate_ms': elapsed_ms,
      'audio_bytes': len(audio),
      'audio_hash': hashlib.md5(audio).hexdigest(),
      'error': None,
    }
  except Exception as e:
    return {
      'id': run_id,
      'timestamp': datetime.now(),
      'backend': backend,
      'voice': voice or 'default',
      'text_id': text_id,
      'text_len': len(text),
      'chunk_size': chunk_size,
      'generate_ms': None,
      'audio_bytes': None,
      'audio_hash': None,
      'error': str(e),
    }


def save_result(db: duckdb.DuckDBPyConnection, result: dict):
  """Save benchmark result to database."""
  db.execute('''
    INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  ''', [
    result['id'],
    result['timestamp'],
    result['backend'],
    result['voice'],
    result['text_id'],
    result['text_len'],
    result['chunk_size'],
    result['generate_ms'],
    result['audio_bytes'],
    result['audio_hash'],
    result['error'],
  ])


@app.command()
def run(
  backends: list[str] = typer.Option(['say', 'avfoundation'], '--backend', '-b', help='Backends to test'),
  voices: list[str] = typer.Option(['Samantha'], '--voice', '-v', help='Voices to test'),
  texts: list[str] = typer.Option(['short', 'medium'], '--text', '-t', help='Text sizes to test'),
  chunk_sizes: list[int] = typer.Option([], '--chunk', '-c', help='Chunk sizes to test'),
  iterations: int = typer.Option(3, '--iterations', '-n', help='Iterations per config'),
):
  """Run TTS benchmarks."""
  from rich.console import Console
  from rich.table import Table

  console = Console()
  db = duckdb.connect(str(DB_PATH))
  init_db(db)

  # Add None for full text (no chunking)
  chunk_options = [None] + chunk_sizes if chunk_sizes else [None]

  # Add default voice if not specified
  voice_options = voices if voices else [None]

  total = len(backends) * len(voice_options) * len(texts) * len(chunk_options) * iterations
  console.print(f'Running {total} benchmarks...\n')

  results = []
  with console.status('[bold blue]Running benchmarks...') as status:
    for backend in backends:
      for voice in voice_options:
        for text_id in texts:
          text = TEXTS.get(text_id, text_id)  # Allow custom text
          for chunk_size in chunk_options:
            for i in range(iterations):
              status.update(f'[bold blue]{backend}/{voice or "default"}/{text_id} (iter {i+1}/{iterations})')

              result = run_benchmark(backend, voice, text_id, text, chunk_size)
              save_result(db, result)
              results.append(result)

              if result['error']:
                console.print(f'[red]Error: {result["error"]}[/red]')
              else:
                console.print(f'[dim]{backend}/{voice or "default"}/{text_id}: {result["generate_ms"]:.1f}ms[/dim]')

  db.close()
  console.print(f'\n[green]Done! Results saved to {DB_PATH}[/green]')


@app.command()
def report(
  last: int = typer.Option(None, '--last', '-l', help='Show last N runs'),
  backend: str = typer.Option(None, '--backend', '-b', help='Filter by backend'),
):
  """Show benchmark results."""
  from rich.console import Console
  from rich.table import Table

  console = Console()
  db = duckdb.connect(str(DB_PATH), read_only=True)

  # Summary stats
  query = '''
    SELECT
      backend,
      voice,
      text_id,
      chunk_size,
      COUNT(*) as runs,
      AVG(generate_ms) as avg_ms,
      MIN(generate_ms) as min_ms,
      MAX(generate_ms) as max_ms,
      AVG(audio_bytes) as avg_bytes,
      AVG(generate_ms / text_len * 1000) as ms_per_1k_chars
    FROM runs
    WHERE error IS NULL
  '''
  if backend:
    query += f" AND backend = '{backend}'"
  query += '''
    GROUP BY backend, voice, text_id, chunk_size
    ORDER BY backend, voice, text_id, chunk_size
  '''

  result = db.execute(query).fetchall()

  table = Table(title='TTS Benchmark Summary')
  table.add_column('Backend', style='cyan')
  table.add_column('Voice', style='green')
  table.add_column('Text', style='yellow')
  table.add_column('Chunk')
  table.add_column('Runs', justify='right')
  table.add_column('Avg (ms)', justify='right', style='bold')
  table.add_column('Min (ms)', justify='right')
  table.add_column('Max (ms)', justify='right')
  table.add_column('ms/1k chars', justify='right')

  for row in result:
    table.add_row(
      row[0],
      row[1],
      row[2],
      str(row[3]) if row[3] else 'full',
      str(row[4]),
      f'{row[5]:.1f}',
      f'{row[6]:.1f}',
      f'{row[7]:.1f}',
      f'{row[9]:.1f}' if row[9] else '-',
    )

  console.print(table)

  # Recent runs
  if last:
    console.print(f'\n[bold]Last {last} runs:[/bold]')
    recent = db.execute(f'''
      SELECT timestamp, backend, voice, text_id, generate_ms, error
      FROM runs
      ORDER BY timestamp DESC
      LIMIT {last}
    ''').fetchall()
    for row in recent:
      if row[5]:
        console.print(f'  [red]{row[0]} {row[1]}/{row[2]}/{row[3]}: ERROR - {row[5]}[/red]')
      else:
        console.print(f'  [dim]{row[0]}[/dim] {row[1]}/{row[2]}/{row[3]}: {row[4]:.1f}ms')

  db.close()


@app.command()
def compare(
  backend1: str = typer.Argument(..., help='First backend'),
  backend2: str = typer.Argument(..., help='Second backend'),
):
  """Compare two backends."""
  from rich.console import Console
  from rich.table import Table

  console = Console()
  db = duckdb.connect(str(DB_PATH), read_only=True)

  query = '''
    WITH b1 AS (
      SELECT voice, text_id, AVG(generate_ms) as avg_ms
      FROM runs WHERE backend = ? AND error IS NULL
      GROUP BY voice, text_id
    ),
    b2 AS (
      SELECT voice, text_id, AVG(generate_ms) as avg_ms
      FROM runs WHERE backend = ? AND error IS NULL
      GROUP BY voice, text_id
    )
    SELECT
      COALESCE(b1.voice, b2.voice) as voice,
      COALESCE(b1.text_id, b2.text_id) as text_id,
      b1.avg_ms as ms1,
      b2.avg_ms as ms2,
      CASE
        WHEN b1.avg_ms < b2.avg_ms THEN ?
        WHEN b2.avg_ms < b1.avg_ms THEN ?
        ELSE 'tie'
      END as winner,
      ABS(b1.avg_ms - b2.avg_ms) as diff_ms
    FROM b1
    FULL OUTER JOIN b2 ON b1.voice = b2.voice AND b1.text_id = b2.text_id
    ORDER BY voice, text_id
  '''

  result = db.execute(query, [backend1, backend2, backend1, backend2]).fetchall()

  table = Table(title=f'Comparison: {backend1} vs {backend2}')
  table.add_column('Voice', style='green')
  table.add_column('Text', style='yellow')
  table.add_column(backend1, justify='right')
  table.add_column(backend2, justify='right')
  table.add_column('Winner', style='bold')
  table.add_column('Diff (ms)', justify='right')

  for row in result:
    winner_style = 'green' if row[4] != 'tie' else 'dim'
    table.add_row(
      row[0],
      row[1],
      f'{row[2]:.1f}' if row[2] else '-',
      f'{row[3]:.1f}' if row[3] else '-',
      f'[{winner_style}]{row[4]}[/{winner_style}]',
      f'{row[5]:.1f}' if row[5] else '-',
    )

  console.print(table)
  db.close()


@app.command()
def export(
  output: Path = typer.Argument(..., help='Output path (.parquet or .csv)'),
):
  """Export results to parquet or CSV."""
  db = duckdb.connect(str(DB_PATH), read_only=True)

  if output.suffix == '.parquet':
    db.execute(f"COPY runs TO '{output}' (FORMAT PARQUET)")
  elif output.suffix == '.csv':
    db.execute(f"COPY runs TO '{output}' (FORMAT CSV, HEADER)")
  else:
    raise ValueError(f'Unknown format: {output.suffix}')

  print(f'Exported to {output}')
  db.close()


@app.command()
def play(run_id: str):
  """Play audio from a benchmark run."""
  import subprocess
  path = audio_path(run_id)
  if not path.exists():
    print(f'Audio not found: {path}')
    raise typer.Exit(1)
  subprocess.run(['afplay', str(path)])


@app.command()
def clean():
  """Delete all benchmark data."""
  import shutil
  if DB_PATH.exists():
    DB_PATH.unlink()
    print(f'Deleted {DB_PATH}')
  if AUDIO_DIR.exists():
    shutil.rmtree(AUDIO_DIR)
    print(f'Deleted {AUDIO_DIR}')


if __name__ == '__main__':
  app()
