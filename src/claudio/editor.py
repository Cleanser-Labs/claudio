"""Built-in editor functions for voice control."""

from voice_control.func import func


@func
def insert(position: str, content: str) -> None:
  """Insert content at the specified position.

  Args:
      position: Where to insert ("start", "end", "line:N", "after:text")
      content: The text or markdown to insert
  """
  pass  # Implementation comes in Phase 4


@func
def delete(start: int, end: int | None = None) -> None:
  """Delete a range of text.

  Args:
      start: Start offset (0-indexed)
      end: End offset (exclusive), defaults to end of document
  """
  pass


@func
def replace(start: int, end: int, content: str) -> None:
  """Replace a range of text with new content.

  Args:
      start: Start offset (0-indexed)
      end: End offset (exclusive)
      content: The replacement text
  """
  pass


@func
def select(start: int, end: int) -> None:
  """Select a range of text.

  Args:
      start: Start offset (0-indexed)
      end: End offset (exclusive)
  """
  pass


@func
def move(direction: str, amount: int = 1) -> None:
  """Move cursor in a direction.

  Args:
      direction: "up", "down", "left", "right", "start", "end"
      amount: Number of units to move (default 1)
  """
  pass


@func
def undo() -> None:
  """Undo the last operation."""
  pass


@func
def redo() -> None:
  """Redo the last undone operation."""
  pass
