#!/usr/bin/env python3
"""Test wake word detection.

Usage:
    # Install dependencies first
    cd experiments/claudio
    uv pip install -e ".[triggers,asr]"

    # Run test with default wake word (hey_jarvis)
    python scripts/test_wakeword.py

    # Or specify a different model
    python scripts/test_wakeword.py --model alexa
"""

import argparse
import time


def test_wakeword(model_name: str = 'hey_jarvis', threshold: float = 0.5):
  """Test wake word detection with OpenWakeWord."""
  import numpy as np
  import sounddevice as sd
  from openwakeword.model import Model

  print(f'Loading wake word model: {model_name}')
  print('This may take a moment on first run (downloading model)...')
  print()

  model = Model(wakeword_models=[model_name])

  print(f'Ready! Say "{model_name.replace("_", " ")}" to test.')
  print('Press Ctrl+C to stop.')
  print()

  sample_rate = 16000
  chunk_size = 1280  # ~80ms

  detected_count = 0

  def callback(indata, frames, time_info, status):
    nonlocal detected_count

    audio = (indata[:, 0] * 32767).astype(np.int16)
    prediction = model.predict(audio)

    score = prediction.get(model_name, 0)
    if score > threshold:
      detected_count += 1
      print(f'🎯 Wake word detected! (score: {score:.2f}, count: {detected_count})')
      model.reset()

  try:
    with sd.InputStream(
      samplerate=sample_rate,
      channels=1,
      dtype='float32',
      blocksize=chunk_size,
      callback=callback,
    ):
      while True:
        time.sleep(0.1)
  except KeyboardInterrupt:
    print(f'\nStopped. Detected {detected_count} times.')


def list_models():
  """List available pre-trained models."""
  print('Available pre-trained OpenWakeWord models:')
  print()
  models = [
    ('alexa', 'Amazon Alexa wake word'),
    ('hey_jarvis', 'Hey Jarvis - good for testing'),
    ('hey_mycroft', 'Hey Mycroft - open source assistant'),
    ('hey_rhasspy', 'Hey Rhasspy - another open source option'),
    ('timer', 'Detects timer/alarm related phrases'),
    ('weather', 'Detects weather related phrases'),
  ]
  for name, desc in models:
    print(f'  {name:15} - {desc}')
  print()
  print('To train a custom wake word like "Hey Claudio", see:')
  print('  https://github.com/dscripka/openWakeWord#training-custom-models')


def train_personal_wakeword(name: str = 'hey_claudio', num_samples: int = 5):
  """Train a personal wake word with your voice.

  Records a few samples of you saying the wake word,
  then creates a model that recognizes your voice.
  """
  import os
  import numpy as np
  import sounddevice as sd
  from pathlib import Path

  sample_rate = 16000
  duration = 2.0  # seconds per sample

  # Create directory for samples
  samples_dir = Path(f'.wakeword_samples/{name}')
  samples_dir.mkdir(parents=True, exist_ok=True)

  print(f'Training personal wake word: "{name.replace("_", " ")}"')
  print(f'We\'ll record {num_samples} samples of you saying the phrase.')
  print()

  samples = []

  for i in range(num_samples):
    input(f'Press Enter, then say "{name.replace("_", " ")}" ({i+1}/{num_samples})...')
    print('  🎤 Recording...', end='', flush=True)

    audio = sd.rec(
      int(duration * sample_rate),
      samplerate=sample_rate,
      channels=1,
      dtype='float32'
    )
    sd.wait()

    # Save sample
    sample_path = samples_dir / f'sample_{i}.npy'
    np.save(sample_path, audio)
    samples.append(audio.flatten())

    print(' done!')

  print()
  print('Training model...')

  # Use OpenWakeWord's personal model training
  try:
    from openwakeword import utils

    # Convert samples to format expected by OpenWakeWord
    model_path = samples_dir / f'{name}.onnx'

    # OpenWakeWord personal models use embedding comparison
    # For a simple approach, we'll just save the samples and use
    # a threshold-based comparison at runtime

    print(f'Samples saved to: {samples_dir}')
    print()
    print('Note: Full personal wake word training requires additional setup.')
    print('For now, you can use the pre-trained models like "hey_jarvis".')
    print()
    print('To properly train a custom model, see:')
    print('  https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb')

  except Exception as e:
    print(f'Training helper not available: {e}')
    print('Samples saved - you can use them for manual training.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Test wake word detection')
  parser.add_argument('--model', default='hey_jarvis', help='Wake word model name')
  parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
  parser.add_argument('--list', action='store_true', help='List available models')
  parser.add_argument('--train', metavar='NAME', help='Train personal wake word')
  parser.add_argument('--samples', type=int, default=5, help='Number of training samples')

  args = parser.parse_args()

  if args.list:
    list_models()
  elif args.train:
    train_personal_wakeword(args.train, args.samples)
  else:
    test_wakeword(args.model, args.threshold)
