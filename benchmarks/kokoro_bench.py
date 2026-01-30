"""Benchmark Kokoro TTS generation speed."""

import time
import soundfile as sf

# Sample texts
TEXTS = {
  'tiny': 'Hello.',
  'short': 'The quick brown fox jumps over the lazy dog.',
  'medium': 'Text to speech technology has evolved significantly over the past decade. Modern neural TTS systems can produce remarkably natural sounding speech.',
  'long': 'Text to speech technology has evolved significantly over the past decade. Modern neural TTS systems can produce remarkably natural sounding speech with appropriate prosody intonation and emotional expression. The challenge remains in achieving low latency for real time applications while maintaining high quality output. Different backends offer various trade offs between speed quality and resource usage.',
}


def bench_kokoro():
  from mlx_audio.tts.models.kokoro import KokoroPipeline
  from mlx_audio.tts.utils import load_model

  model_id = 'mlx-community/Kokoro-82M-bf16'

  print('Loading Kokoro model...')
  start = time.perf_counter()
  model = load_model(model_id)
  pipeline = KokoroPipeline(lang_code='a', model=model, repo_id=model_id)
  print(f'Model loaded: {time.perf_counter() - start:.2f}s\n')

  # Warmup
  print('Warmup...')
  for _, _, audio in pipeline('warmup', voice='af_heart', speed=1):
    pass

  print('\nKokoro benchmarks:')
  print('-' * 50)

  for name, text in TEXTS.items():
    times = []
    for _ in range(3):
      start = time.perf_counter()
      for _, _, audio in pipeline(text, voice='af_heart', speed=1):
        pass
      times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f'{name:8} ({len(text):3} chars): {avg:6.0f}ms  [{", ".join(f"{t:.0f}" for t in times)}]')


def bench_say():
  import subprocess
  import tempfile
  import os

  print('\nmacOS say benchmarks:')
  print('-' * 50)

  for name, text in TEXTS.items():
    times = []
    for _ in range(3):
      fd, path = tempfile.mkstemp(suffix='.aiff')
      os.close(fd)
      start = time.perf_counter()
      subprocess.run(['say', '-o', path, text], capture_output=True)
      times.append((time.perf_counter() - start) * 1000)
      os.unlink(path)

    avg = sum(times) / len(times)
    print(f'{name:8} ({len(text):3} chars): {avg:6.0f}ms  [{", ".join(f"{t:.0f}" for t in times)}]')


if __name__ == '__main__':
  bench_kokoro()
  bench_say()
