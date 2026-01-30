"""Input sources for TTS text."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Iterator, TextIO


class Source(ABC):
  """Base input source."""

  @abstractmethod
  def read(self) -> str:
    """Read all text."""

  def chunks(self, sep: str = '\n') -> Iterator[str]:
    """Yield text chunks."""
    yield self.read()

  def __iter__(self) -> Iterator[str]:
    return self.chunks()


class Text(Source):
  """Text string source."""

  def __init__(self, text: str):
    self.text = text

  def read(self) -> str:
    return self.text

  def chunks(self, sep: str = '\n') -> Iterator[str]:
    for chunk in self.text.split(sep):
      if chunk.strip():
        yield chunk


class File(Source):
  """Read text from file."""

  def __init__(self, path: str | Path):
    self.path = Path(path)

  def read(self) -> str:
    return self.path.read_text()

  def chunks(self, sep: str = '\n') -> Iterator[str]:
    with open(self.path) as f:
      for line in f:
        line = line.strip()
        if line:
          yield line


class Handle(Source):
  """Read from file handle."""

  def __init__(self, handle: TextIO):
    self.handle = handle

  def read(self) -> str:
    return self.handle.read()

  def chunks(self, sep: str = '\n') -> Iterator[str]:
    for line in self.handle:
      line = line.strip()
      if line:
        yield line


class Stdin(Source):
  """Read from stdin."""

  def read(self) -> str:
    return sys.stdin.read()

  def chunks(self, sep: str = '\n') -> Iterator[str]:
    for line in sys.stdin:
      line = line.strip()
      if line:
        yield line


class Lines(Source):
  """Iterator of text lines."""

  def __init__(self, lines: Iterator[str] | list[str]):
    self._lines = list(lines) if not isinstance(lines, list) else lines

  def read(self) -> str:
    return '\n'.join(self._lines)

  def chunks(self, sep: str = '\n') -> Iterator[str]:
    for line in self._lines:
      if line.strip():
        yield line


# Convenience functions
def text(t: str) -> Text:
  return Text(t)


def file(path: str | Path) -> File:
  return File(path)


def handle(h: TextIO) -> Handle:
  return Handle(h)


def stdin() -> Stdin:
  return Stdin()


def lines(l: Iterator[str] | list[str]) -> Lines:
  return Lines(l)
