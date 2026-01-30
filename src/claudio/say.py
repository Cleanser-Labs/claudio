"""CLI to speak text via TTS server. Auto-starts server if needed."""

import argparse
import subprocess
import sys
import time

import httpx


SERVER_URL = 'http://127.0.0.1:8765'


def is_server_running(url: str = SERVER_URL) -> bool:
  """Check if TTS server is running."""
  try:
    response = httpx.get(f'{url}/health', timeout=1.0)
    return response.status_code == 200
  except (httpx.ConnectError, httpx.ReadTimeout):
    return False


def start_server(url: str = SERVER_URL) -> None:
  """Start TTS server in background and wait for it to be ready."""
  print('Starting TTS server...', file=sys.stderr)

  # Start server as background process
  subprocess.Popen(
    ['uv', 'run', 'voice-tts-server'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )

  # Wait for server to be ready (model loading takes time)
  for i in range(60):  # Wait up to 60 seconds
    if is_server_running(url):
      print('TTS server ready.', file=sys.stderr)
      return
    time.sleep(1)
    if i % 5 == 0 and i > 0:
      print(f'  Still loading model... ({i}s)', file=sys.stderr)

  print('TTS server failed to start.', file=sys.stderr)
  sys.exit(1)


def main():
  parser = argparse.ArgumentParser(description='Speak text via TTS server')
  parser.add_argument('text', nargs='?', help='Text to speak (or read from stdin)')
  parser.add_argument('--server', '-s', default=SERVER_URL)
  parser.add_argument('--temperature', '-t', type=float, default=0.3)
  args = parser.parse_args()

  # Get text from arg or stdin
  if args.text:
    text = args.text
  else:
    text = sys.stdin.read()

  text = text.strip()
  if not text:
    return

  # Start server if not running
  if not is_server_running(args.server):
    start_server(args.server)

  # Send to TTS server (plays on server side)
  payload = {'text': text, 'temperature': args.temperature}
  response = httpx.post(
    f'{args.server}/speak',
    json=payload,
    timeout=120.0,
  )
  response.raise_for_status()


if __name__ == '__main__':
  main()
