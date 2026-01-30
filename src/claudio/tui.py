"""Terminal UI for live transcription testing."""

from __future__ import annotations

import signal
import sys
from threading import Thread

from voice_control.transcribe import get_transcriber


def run_simple(
  backend: str = 'parakeet',
  device: int | None = None,
  window: float = 10.0,
  step: float = 0.5,
):
  """Simple terminal display of live transcription.

  Args:
      backend: "parakeet" or "whisper"
      device: Audio input device index (None for default)
      window: Audio window size in seconds
      step: Step size in seconds
  """
  import time

  transcriber = get_transcriber(
    backend,
    device=device,
    window_seconds=window,
    step_seconds=step,
  )

  # Print settings
  print(f'Backend: {backend}')
  print(f'Device: {device or "default"}')
  if hasattr(transcriber, 'model_name'):
    print(f'Model: {transcriber.model_name}')
  if hasattr(transcriber, 'window_seconds'):
    print(f'Window: {transcriber.window_seconds}s')
  if hasattr(transcriber, 'step_seconds'):
    print(f'Step: {transcriber.step_seconds}s')
  print()
  print('Speak into your microphone. Press Ctrl+C to stop.')
  print('-' * 60)

  def handle_signal(sig, frame):
    print('\n\nStopping...')
    transcriber.stop()
    sys.exit(0)

  signal.signal(signal.SIGINT, handle_signal)

  start_time = time.time()
  last_text = ''

  for chunk in transcriber.stream():
    text = chunk.text
    if text == last_text:
      continue
    last_text = text
    elapsed = time.time() - start_time

    # Build status with timing info
    latency = f'{chunk.latency_ms:3.0f}ms' if chunk.latency_ms else '---'
    window = f'{chunk.window_seconds:.1f}s' if chunk.window_seconds else '---'
    print(f'[{elapsed:5.1f}s | {latency} | {window}] {text}')


def run_textual(backend: str = 'parakeet', device: int | None = None):
  """Rich TUI using textual library."""
  try:
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Static, Log
    from textual.containers import Container
  except ImportError:
    print('textual not installed. Run: uv sync --extra tui')
    print('Falling back to simple mode...')
    return run_simple(backend, device)

  class TranscriptionApp(App):
    """Live transcription TUI."""

    CSS = '''
    #status {
      dock: top;
      height: 3;
      padding: 1;
      background: $surface;
    }

    #transcript {
      height: 1fr;
      padding: 1;
      border: solid green;
    }

    #current {
      dock: bottom;
      height: 5;
      padding: 1;
      background: $surface-darken-1;
      border: solid $primary;
    }
    '''

    BINDINGS = [
      ('q', 'quit', 'Quit'),
      ('c', 'clear', 'Clear'),
    ]

    def __init__(self, backend: str, device: int | None):
      super().__init__()
      self.backend = backend
      self.device = device
      self.transcriber = None
      self.thread = None

    def compose(self) -> ComposeResult:
      yield Header()
      yield Static(f'Backend: {self.backend} | Device: {self.device or "default"}', id='status')
      yield Log(id='transcript', highlight=True)
      yield Static('Listening...', id='current')
      yield Footer()

    def on_mount(self) -> None:
      self.transcriber = get_transcriber(self.backend, device=self.device)
      self.thread = Thread(target=self._transcribe_loop, daemon=True)
      self.thread.start()

    def _transcribe_loop(self) -> None:
      transcript_log = self.query_one('#transcript', Log)
      current_display = self.query_one('#current', Static)

      for chunk in self.transcriber.stream():
        if chunk.is_final:
          self.call_from_thread(transcript_log.write_line, chunk.text)
          self.call_from_thread(current_display.update, 'Listening...')
        else:
          self.call_from_thread(current_display.update, f'> {chunk.text}')

    def action_clear(self) -> None:
      self.query_one('#transcript', Log).clear()

    def action_quit(self) -> None:
      if self.transcriber:
        self.transcriber.stop()
      self.exit()

  app = TranscriptionApp(backend, device)
  app.run()


def main():
  """CLI entry point."""
  import argparse

  parser = argparse.ArgumentParser(description='Live transcription TUI')
  parser.add_argument(
    '--backend',
    '-b',
    choices=['parakeet', 'whisper'],
    default='parakeet',
    help='ASR backend to use',
  )
  parser.add_argument(
    '--device',
    '-d',
    type=int,
    default=None,
    help='Audio input device index',
  )
  parser.add_argument(
    '--window',
    '-w',
    type=float,
    default=10.0,
    help='Audio window size in seconds (default: 10)',
  )
  parser.add_argument(
    '--step',
    type=float,
    default=0.5,
    help='Step size in seconds (default: 0.5)',
  )
  parser.add_argument(
    '--simple',
    '-s',
    action='store_true',
    help='Use simple terminal output instead of TUI',
  )
  parser.add_argument(
    '--list-devices',
    '-l',
    action='store_true',
    help='List available audio devices and exit',
  )

  args = parser.parse_args()

  if args.list_devices:
    import sounddevice as sd
    print(sd.query_devices())
    return

  if args.simple:
    run_simple(args.backend, args.device, args.window, args.step)
  else:
    run_textual(args.backend, args.device)


if __name__ == '__main__':
  main()
