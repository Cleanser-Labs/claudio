"""Benchmark TTS pipeline - trace throughput and thread overlap.

Measures:
- Generation time (text → audio bytes)
- Playback time (audio bytes → sound)
- Queue latency (time between generation complete and playback start)
- Thread overlap (concurrent execution ratio)
- GIL contention (implied by lack of overlap)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterator

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help='TTS pipeline benchmarks')


@dataclass
class TraceEvent:
  """A traced pipeline event."""
  name: str
  stage: str  # 'gen' or 'play'
  event: str  # 'start' or 'end'
  ts: float   # timestamp (perf_counter)
  text: str = ''
  audio_bytes: int = 0


def play_audio(engine, audio_bytes: bytes):
  """Play audio using engine.play() or sounddevice fallback."""
  if hasattr(engine, 'play'):
    engine.play(audio_bytes)
  else:
    # Fallback: parse AIFF/WAV and play with sounddevice
    import struct
    import numpy as np
    import sounddevice as sd

    # Detect format and extract audio data
    if audio_bytes[:4] == b'FORM':
      # AIFF format (from macOS say)
      # Find SSND chunk
      pos = 12
      sample_rate = 22050  # Default for say
      while pos < len(audio_bytes) - 8:
        chunk_id = audio_bytes[pos:pos+4]
        chunk_size = struct.unpack('>I', audio_bytes[pos+4:pos+8])[0]
        if chunk_id == b'COMM':
          # Parse COMM chunk for sample rate
          if chunk_size >= 18:
            sr_bytes = audio_bytes[pos+16:pos+26]
            # 80-bit extended precision float - approximate
            sample_rate = 22050  # Common default
        elif chunk_id == b'SSND':
          # Audio data starts after 8-byte offset/block
          audio_data = audio_bytes[pos+16:pos+8+chunk_size]
          break
        pos += 8 + chunk_size
      else:
        return

      # Convert from 16-bit big-endian
      audio_array = np.frombuffer(audio_data, dtype='>i2').astype(np.float32) / 32767

    elif audio_bytes[:4] == b'RIFF':
      # WAV format
      pos = 12
      sample_rate = 24000
      while pos < len(audio_bytes) - 8:
        chunk_id = audio_bytes[pos:pos+4]
        chunk_size = struct.unpack('<I', audio_bytes[pos+4:pos+8])[0]
        if chunk_id == b'fmt ':
          sample_rate = struct.unpack('<I', audio_bytes[pos+12:pos+16])[0]
        elif chunk_id == b'data':
          audio_data = audio_bytes[pos+8:pos+8+chunk_size]
          break
        pos += 8 + chunk_size
      else:
        return

      audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32767

    else:
      return

    sd.play(audio_array, samplerate=sample_rate, blocking=True)


class TracedTTSQueue:
  """TTS queue with detailed tracing."""

  def __init__(self, engine, on_trace: callable):
    self.engine = engine
    self.on_trace = on_trace

    self._text_queue: list[str] = []
    self._text_lock = threading.Lock()
    self._audio_queue: list[tuple[bytes, str]] = []
    self._audio_lock = threading.Lock()

    self._running = True
    self._gen_thread = threading.Thread(target=self._generator, daemon=True)
    self._play_thread = threading.Thread(target=self._player, daemon=True)
    self._gen_thread.start()
    self._play_thread.start()

  def add(self, text: str):
    with self._text_lock:
      self._text_queue.append(text)

  def stop(self):
    self._running = False

  def wait(self, timeout: float = 60.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
      with self._text_lock:
        text_empty = len(self._text_queue) == 0
      with self._audio_lock:
        audio_empty = len(self._audio_queue) == 0
      if text_empty and audio_empty:
        return True
      time.sleep(0.01)
    return False

  def _generator(self):
    while self._running:
      text = None
      with self._text_lock:
        if self._text_queue:
          text = self._text_queue.pop(0)

      if text:
        self.on_trace(TraceEvent('gen', 'gen', 'start', time.perf_counter(), text=text))
        try:
          audio = self.engine.generate(text)
          self.on_trace(TraceEvent('gen', 'gen', 'end', time.perf_counter(), text=text, audio_bytes=len(audio)))
          if audio:
            with self._audio_lock:
              self._audio_queue.append((audio, text))
        except Exception as e:
          self.on_trace(TraceEvent('gen_error', 'gen', 'end', time.perf_counter(), text=str(e)))
      else:
        time.sleep(0.01)

  def _player(self):
    while self._running:
      audio_item = None
      with self._audio_lock:
        if self._audio_queue:
          audio_item = self._audio_queue.pop(0)

      if audio_item:
        audio, text = audio_item
        self.on_trace(TraceEvent('play', 'play', 'start', time.perf_counter(), text=text, audio_bytes=len(audio)))
        try:
          play_audio(self.engine, audio)
          self.on_trace(TraceEvent('play', 'play', 'end', time.perf_counter(), text=text))
        except Exception as e:
          self.on_trace(TraceEvent('play_error', 'play', 'end', time.perf_counter(), text=str(e)))
      else:
        time.sleep(0.01)


def analyze_traces(events: list[TraceEvent]) -> dict:
  """Analyze trace events to compute metrics."""
  # Pair up start/end events
  gen_times = []
  play_times = []
  queue_latencies = []

  gen_active = []  # [(start, end), ...]
  play_active = []

  first_gen_start = None
  first_gen_end = None
  first_play_start = None

  i = 0
  gen_ends = {}  # text -> end_ts
  while i < len(events):
    ev = events[i]

    if ev.name == 'gen' and ev.event == 'start':
      if first_gen_start is None:
        first_gen_start = ev.ts

      # Find matching end
      for j in range(i + 1, len(events)):
        if events[j].name == 'gen' and events[j].event == 'end' and events[j].text == ev.text:
          duration = events[j].ts - ev.ts
          gen_times.append((ev.text, duration, events[j].audio_bytes))
          gen_active.append((ev.ts, events[j].ts))
          gen_ends[ev.text] = events[j].ts
          if first_gen_end is None:
            first_gen_end = events[j].ts
          break

    elif ev.name == 'play' and ev.event == 'start':
      if first_play_start is None:
        first_play_start = ev.ts

      # Queue latency = play_start - gen_end
      if ev.text in gen_ends:
        queue_latencies.append(ev.ts - gen_ends[ev.text])

      # Find matching end
      for j in range(i + 1, len(events)):
        if events[j].name == 'play' and events[j].event == 'end' and events[j].text == ev.text:
          duration = events[j].ts - ev.ts
          play_times.append((ev.text, duration, ev.audio_bytes))
          play_active.append((ev.ts, events[j].ts))
          break

    i += 1

  # Calculate overlap - time when both gen and play are active
  overlap_time = 0.0
  total_time = 0.0
  if gen_active and play_active:
    all_starts = sorted([s for s, e in gen_active + play_active])
    all_ends = sorted([e for s, e in gen_active + play_active])
    total_time = all_ends[-1] - all_starts[0] if all_starts and all_ends else 0

    # Calculate overlap using timeline sweep
    for gen_s, gen_e in gen_active:
      for play_s, play_e in play_active:
        overlap_s = max(gen_s, play_s)
        overlap_e = min(gen_e, play_e)
        if overlap_e > overlap_s:
          overlap_time += overlap_e - overlap_s

  # Time to first audio (generation latency for first sentence)
  time_to_first_audio = first_gen_end - first_gen_start if first_gen_start and first_gen_end else 0
  # Time to first playback
  time_to_first_play = first_play_start - first_gen_start if first_gen_start and first_play_start else 0

  return {
    'gen_times': gen_times,
    'play_times': play_times,
    'queue_latencies': queue_latencies,
    'overlap_time': overlap_time,
    'total_time': total_time,
    'overlap_ratio': overlap_time / total_time if total_time > 0 else 0,
    'time_to_first_audio': time_to_first_audio,
    'time_to_first_play': time_to_first_play,
  }


# Test texts
TEXTS = [
  'Hello world.',
  'The quick brown fox jumps over the lazy dog.',
  'Text to speech technology has evolved significantly.',
  'Modern neural TTS systems can produce natural speech.',
  'Low latency is the key challenge for real-time applications.',
]


@app.command()
def run(
  backend: str = typer.Option('kokoro', '--backend', '-b', help='TTS backend'),
  voice: str = typer.Option(None, '--voice', '-v', help='Voice'),
  iterations: int = typer.Option(1, '--iterations', '-n', help='Iterations'),
  dry_run: bool = typer.Option(False, '--dry', help='No audio playback'),
):
  """Run pipeline benchmark with tracing."""
  from claudio.tts import Config, create

  console = Console()
  console.print(f'\n[bold]TTS Pipeline Benchmark[/bold]')
  console.print(f'Backend: {backend}')
  console.print(f'Voice: {voice or "default"}')
  console.print(f'Sentences: {len(TEXTS)}')
  console.print()

  # Initialize TTS
  config = Config(backend=backend, voice=voice)
  engine = create(config)

  # Warmup
  console.print('[dim]Warming up...[/dim]')
  try:
    audio = engine.generate('Warmup.')
    if not dry_run:
      play_audio(engine, audio)
  except Exception as e:
    console.print(f'[red]Warmup failed: {e}[/red]')
    raise typer.Exit(1)

  all_metrics = []

  for iteration in range(iterations):
    if iterations > 1:
      console.print(f'\n[bold]Iteration {iteration + 1}/{iterations}[/bold]')

    # Collect traces
    events: list[TraceEvent] = []

    def on_trace(ev: TraceEvent):
      events.append(ev)

    # Create traced queue
    queue = TracedTTSQueue(engine, on_trace)

    # Queue all texts
    start = time.perf_counter()
    for text in TEXTS:
      queue.add(text)

    # Wait for completion
    queue.wait(timeout=120.0)
    total_elapsed = time.perf_counter() - start
    queue.stop()

    # Analyze
    metrics = analyze_traces(events)
    metrics['total_elapsed'] = total_elapsed
    metrics['texts'] = len(TEXTS)
    metrics['total_chars'] = sum(len(t) for t in TEXTS)
    all_metrics.append(metrics)

    # Print trace timeline
    console.print(f'\n[bold]Timeline:[/bold]')
    base_ts = events[0].ts if events else 0
    for ev in events:
      rel_ts = (ev.ts - base_ts) * 1000
      arrow = '→' if ev.event == 'start' else '←'
      text_preview = ev.text[:30] + '...' if len(ev.text) > 30 else ev.text
      style = 'cyan' if ev.stage == 'gen' else 'yellow'
      extra = f' ({ev.audio_bytes} bytes)' if ev.audio_bytes else ''
      console.print(f'  [{style}]{rel_ts:7.1f}ms {arrow} {ev.stage:4} {ev.event:5}[/{style}] {text_preview}{extra}')

    # Print metrics
    console.print(f'\n[bold]Metrics:[/bold]')

    # First audio latency (key UX metric)
    ttfa = metrics['time_to_first_audio'] * 1000
    ttfp = metrics['time_to_first_play'] * 1000
    console.print(f'  [bold]Time to first audio: {ttfa:.0f}ms[/bold]')
    console.print(f'  Time to first play:  {ttfp:.0f}ms')

    # Generation stats
    if metrics['gen_times']:
      avg_gen = sum(t for _, t, _ in metrics['gen_times']) / len(metrics['gen_times']) * 1000
      total_gen = sum(t for _, t, _ in metrics['gen_times']) * 1000
      total_bytes = sum(b for _, _, b in metrics['gen_times'])
      console.print(f'  Generation:   {total_gen:.0f}ms total, {avg_gen:.0f}ms avg/sentence')
      console.print(f'  Audio output: {total_bytes:,} bytes')

    # Playback stats
    if metrics['play_times']:
      avg_play = sum(t for _, t, _ in metrics['play_times']) / len(metrics['play_times']) * 1000
      total_play = sum(t for _, t, _ in metrics['play_times']) * 1000
      console.print(f'  Playback:     {total_play:.0f}ms total, {avg_play:.0f}ms avg/sentence')

    # Queue latency
    if metrics['queue_latencies']:
      avg_queue = sum(metrics['queue_latencies']) / len(metrics['queue_latencies']) * 1000
      max_queue = max(metrics['queue_latencies']) * 1000
      console.print(f'  Queue wait:   {avg_queue:.1f}ms avg, {max_queue:.1f}ms max')

    # Overlap analysis
    overlap_pct = metrics['overlap_ratio'] * 100
    console.print(f'\n  [bold]Thread overlap: {overlap_pct:.1f}%[/bold]')

    # Calculate generation speed vs playback speed
    if metrics['gen_times'] and metrics['play_times']:
      total_gen = sum(t for _, t, _ in metrics['gen_times'])
      total_play = sum(t for _, t, _ in metrics['play_times'])
      gen_speed_ratio = total_play / total_gen if total_gen > 0 else 0

      if gen_speed_ratio > 5:
        console.print(f'  [green]✓ Generation {gen_speed_ratio:.1f}x faster than playback[/green]')
        console.print(f'  [dim]  (Low overlap is expected - gen finishes before play catches up)[/dim]')
      elif gen_speed_ratio > 1:
        console.print(f'  [green]✓ Generation {gen_speed_ratio:.1f}x faster than playback[/green]')
      else:
        console.print(f'  [yellow]⚠ Generation slower than playback - may cause gaps[/yellow]')

    # First sentence queue latency - the one that matters for UX
    if metrics['queue_latencies']:
      first_queue = metrics['queue_latencies'][0] * 1000 if metrics['queue_latencies'] else 0
      if first_queue < 50:
        console.print(f'  [green]✓ First audio queued immediately ({first_queue:.0f}ms)[/green]')
      elif first_queue < 200:
        console.print(f'  [green]✓ First audio queue latency: {first_queue:.0f}ms[/green]')
      else:
        console.print(f'  [yellow]⚠ First audio delayed ({first_queue:.0f}ms) - check GIL contention[/yellow]')

    # Throughput
    chars_per_sec = metrics['total_chars'] / total_elapsed
    sentences_per_sec = len(TEXTS) / total_elapsed
    console.print(f'\n  [bold]Throughput:[/bold]')
    console.print(f'    {chars_per_sec:.0f} chars/sec')
    console.print(f'    {sentences_per_sec:.2f} sentences/sec')
    console.print(f'    {total_elapsed:.2f}s wall time')

  # Summary across iterations
  if iterations > 1:
    console.print(f'\n[bold]Summary ({iterations} iterations):[/bold]')
    avg_overlap = sum(m['overlap_ratio'] for m in all_metrics) / iterations * 100
    avg_throughput = sum(m['total_chars'] / m['total_elapsed'] for m in all_metrics) / iterations
    console.print(f'  Avg overlap:    {avg_overlap:.1f}%')
    console.print(f'  Avg throughput: {avg_throughput:.0f} chars/sec')


@app.command()
def compare(
  backends: list[str] = typer.Option(['kokoro', 'say'], '--backend', '-b', help='Backends'),
  iterations: int = typer.Option(3, '--iterations', '-n', help='Iterations'),
):
  """Compare backends pipeline performance."""
  from claudio.tts import Config, create

  console = Console()
  console.print(f'\n[bold]Pipeline Comparison[/bold]')

  results = {}

  for backend in backends:
    console.print(f'\n[cyan]Testing {backend}...[/cyan]')

    try:
      config = Config(backend=backend)
      engine = create(config)

      # Warmup
      audio = engine.generate('Warmup.')
      play_audio(engine, audio)
    except Exception as e:
      console.print(f'[red]Failed: {e}[/red]')
      continue

    metrics_list = []
    for i in range(iterations):
      events = []
      queue = TracedTTSQueue(engine, lambda ev: events.append(ev))
      start = time.perf_counter()
      for text in TEXTS:
        queue.add(text)
      queue.wait(timeout=120.0)
      elapsed = time.perf_counter() - start
      queue.stop()

      metrics = analyze_traces(events)
      metrics['total_elapsed'] = elapsed
      metrics['total_chars'] = sum(len(t) for t in TEXTS)
      metrics_list.append(metrics)

    results[backend] = metrics_list

  # Summary table
  table = Table(title='Pipeline Comparison')
  table.add_column('Backend', style='cyan')
  table.add_column('Overlap %', justify='right')
  table.add_column('Gen (ms/sent)', justify='right')
  table.add_column('Play (ms/sent)', justify='right')
  table.add_column('Queue (ms)', justify='right')
  table.add_column('Chars/sec', justify='right', style='bold')

  for backend, metrics_list in results.items():
    avg_overlap = sum(m['overlap_ratio'] for m in metrics_list) / len(metrics_list) * 100

    avg_gen = sum(
      sum(t for _, t, _ in m['gen_times']) / len(m['gen_times']) * 1000
      for m in metrics_list if m['gen_times']
    ) / len(metrics_list)

    avg_play = sum(
      sum(t for _, t, _ in m['play_times']) / len(m['play_times']) * 1000
      for m in metrics_list if m['play_times']
    ) / len(metrics_list)

    avg_queue = sum(
      sum(m['queue_latencies']) / len(m['queue_latencies']) * 1000
      for m in metrics_list if m['queue_latencies']
    ) / len(metrics_list)

    avg_throughput = sum(m['total_chars'] / m['total_elapsed'] for m in metrics_list) / len(metrics_list)

    overlap_style = 'green' if avg_overlap > 50 else 'yellow' if avg_overlap > 20 else 'red'

    table.add_row(
      backend,
      f'[{overlap_style}]{avg_overlap:.1f}[/{overlap_style}]',
      f'{avg_gen:.0f}',
      f'{avg_play:.0f}',
      f'{avg_queue:.1f}',
      f'{avg_throughput:.0f}',
    )

  console.print()
  console.print(table)

  console.print('\n[dim]Overlap % indicates how much gen and play threads run in parallel.[/dim]')
  console.print('[dim]Higher overlap = less GIL blocking = better parallelism.[/dim]')


if __name__ == '__main__':
  app()
