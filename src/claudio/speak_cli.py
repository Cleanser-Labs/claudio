"""CLI to speak text directly (no server needed)."""

import argparse
import sys


def main():
  parser = argparse.ArgumentParser(description='Speak text via TTS')
  parser.add_argument('text', nargs='?', help='Text to speak (or read from stdin)')
  parser.add_argument('--instruct', '-i', default='A calm, clear male voice with moderate pace')
  parser.add_argument('--speed', '-s', type=float, default=1.0)
  args = parser.parse_args()

  # Get text from arg or stdin
  if args.text:
    text = args.text
  else:
    text = sys.stdin.read()

  text = text.strip()
  if not text:
    return

  from voice_control.tts import TTSEngine, TTSConfig

  config = TTSConfig(instruct=args.instruct, speed=args.speed)
  engine = TTSEngine(config)
  engine.play(text)


if __name__ == '__main__':
  main()
