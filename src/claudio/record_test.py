"""Record and transcribe test audio (non-streaming)."""

import argparse


def main():
  import sounddevice as sd
  import mlx.core as mx
  from parakeet_mlx import from_pretrained
  from parakeet_mlx.audio import get_logmel

  parser = argparse.ArgumentParser()
  parser.add_argument('--seconds', '-s', type=int, default=5)
  parser.add_argument('--device', '-d', type=int, default=None)
  args = parser.parse_args()

  print(f'Recording {args.seconds}s from device {args.device or "default"}...')
  audio = sd.rec(
    int(args.seconds * 16000),
    samplerate=16000,
    channels=1,
    dtype='float32',
    device=args.device,
  )
  sd.wait()
  print('Done recording. Transcribing...')

  model = from_pretrained('mlx-community/parakeet-tdt-0.6b-v3')

  # Convert to log-mel spectrogram and transcribe
  audio_mx = mx.array(audio.flatten())
  mel = get_logmel(audio_mx, model.preprocessor_config)
  results = model.generate(mel)

  print(f'\nResult: {results[0].text}')


if __name__ == '__main__':
  main()
