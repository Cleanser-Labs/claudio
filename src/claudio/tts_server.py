"""TTS server using Starlette - keeps model loaded for fast responses."""

from __future__ import annotations

import asyncio
import queue
import threading
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.responses import Response, JSONResponse
from starlette.routing import Route
from starlette.requests import Request

from voice_control.tts import TTSEngine, TTSConfig


# Global engine and audio queue
engine: TTSEngine | None = None
audio_queue: queue.Queue | None = None
audio_thread: threading.Thread | None = None


def audio_worker(q: queue.Queue, eng: TTSEngine):
  """Dedicated thread for audio playback."""
  while True:
    item = q.get()
    if item is None:  # Shutdown signal
      break
    text, event = item
    try:
      eng.play(text)
    except Exception as e:
      print(f'TTS error: {e}')
    finally:
      event.set()  # Signal completion


@asynccontextmanager
async def lifespan(app: Starlette):
  """Load model on startup."""
  global engine, audio_queue, audio_thread

  engine = TTSEngine(TTSConfig())
  engine._load_model()  # Pre-load

  # Start audio worker thread
  audio_queue = queue.Queue()
  audio_thread = threading.Thread(
    target=audio_worker,
    args=(audio_queue, engine),
    daemon=True,
  )
  audio_thread.start()

  yield

  # Shutdown
  if audio_queue:
    audio_queue.put(None)
  engine = None


async def speak(request: Request) -> Response:
  """Generate and play speech from text.

  POST /speak
  Body: {"text": "Hello world", "voice": "optional", "speed": 1.0}

  Plays audio on server side, returns when done.
  """
  body = await request.json()
  text = body.get('text', '')

  if not text:
    return JSONResponse({'error': 'No text provided'}, status_code=400)

  # Update config if provided
  if 'temperature' in body:
    engine.config.temperature = body['temperature']
  if 'top_p' in body:
    engine.config.top_p = body['top_p']

  # Queue for playback and wait for completion
  done_event = threading.Event()
  audio_queue.put((text, done_event))

  # Wait in thread pool to not block event loop
  loop = asyncio.get_event_loop()
  await loop.run_in_executor(None, done_event.wait)

  return JSONResponse({'status': 'ok', 'text': text})


async def health(_request: Request) -> JSONResponse:
  """Health check."""
  return JSONResponse({
    'status': 'ok',
    'engine': 'soprano',
    'loaded': engine._model is not None if engine else False,
  })


routes = [
  Route('/speak', speak, methods=['POST']),
  Route('/health', health, methods=['GET']),
]

app = Starlette(routes=routes, lifespan=lifespan)


def main():
  """Run the TTS server."""
  import argparse
  import uvicorn

  parser = argparse.ArgumentParser(description='TTS Server (Soprano)')
  parser.add_argument('--host', default='127.0.0.1')
  parser.add_argument('--port', '-p', type=int, default=8765)
  args = parser.parse_args()

  print(f'Starting TTS server on {args.host}:{args.port}')
  uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
  main()
