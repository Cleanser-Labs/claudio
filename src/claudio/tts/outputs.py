"""Output sinks for TTS audio."""

from __future__ import annotations

import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Iterator


class Sink(ABC):
  """Base output sink."""

  @abstractmethod
  def write(self, audio: bytes) -> None:
    """Write audio bytes."""

  def close(self) -> None:
    """Close the sink."""
    pass

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()


class Speaker(Sink):
  """Play audio through speakers."""

  def write(self, audio: bytes) -> None:
    # afplay needs a file, write temp
    fd, path = tempfile.mkstemp(suffix='.aiff', prefix='claudio_')
    try:
      os.write(fd, audio)
      os.close(fd)
      subprocess.run(['afplay', path], check=True, capture_output=True)
    finally:
      os.unlink(path)


class File(Sink):
  """Write audio to file."""

  def __init__(self, path: str | Path):
    self.path = Path(path)
    self._file: BinaryIO | None = None

  def write(self, audio: bytes) -> None:
    if self._file is None:
      self._file = open(self.path, 'wb')
    self._file.write(audio)

  def close(self) -> None:
    if self._file:
      self._file.close()
      self._file = None


class Handle(Sink):
  """Write audio to file handle."""

  def __init__(self, handle: BinaryIO):
    self.handle = handle

  def write(self, audio: bytes) -> None:
    self.handle.write(audio)


class Buffer(Sink):
  """Accumulate audio in memory."""

  def __init__(self):
    self.chunks: list[bytes] = []

  def write(self, audio: bytes) -> None:
    self.chunks.append(audio)

  def getvalue(self) -> bytes:
    return b''.join(self.chunks)

  def __bytes__(self) -> bytes:
    return self.getvalue()


class Pipe(Sink):
  """Write audio to named pipe (FIFO)."""

  def __init__(self, path: str | Path):
    self.path = Path(path)
    self._fd: int | None = None

  def write(self, audio: bytes) -> None:
    if self._fd is None:
      if not self.path.exists():
        os.mkfifo(self.path)
      self._fd = os.open(self.path, os.O_WRONLY)
    os.write(self._fd, audio)

  def close(self) -> None:
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None


class Chain(Sink):
  """Write to multiple sinks."""

  def __init__(self, *sinks: Sink):
    self.sinks = sinks

  def write(self, audio: bytes) -> None:
    for sink in self.sinks:
      sink.write(audio)

  def close(self) -> None:
    for sink in self.sinks:
      sink.close()


# Convenience functions
def speaker() -> Speaker:
  return Speaker()


def file(path: str | Path) -> File:
  return File(path)


def handle(h: BinaryIO) -> Handle:
  return Handle(h)


def buffer() -> Buffer:
  return Buffer()


def pipe(path: str | Path) -> Pipe:
  return Pipe(path)
