"""LLM integration for parsing speech to function calls."""

from __future__ import annotations

import os
from typing import Any

import httpx
import msgspec

from voice_control.func import FunctionCall, Registry


class EditorContext(msgspec.Struct, frozen=True):
  """Context for the LLM to understand the current state."""

  document: str = ''  # Current document content
  cursor: int = 0  # Cursor position (offset)
  selection: tuple[int, int] | None = None  # Selection range
  history: list[FunctionCall] = []  # Recent operations


SYSTEM_PROMPT = '''You are a voice-controlled editor assistant. Convert spoken commands into function calls.

The user speaks commands and you respond with the appropriate function call. Be concise and direct.

Rules:
1. Parse the user's intent, ignoring filler words (uh, um, like)
2. Handle corrections: "no wait, I meant..." should use the corrected intent
3. Use the document context to resolve references like "this line", "the word"
4. If the command is ambiguous, make a reasonable choice
5. For dictation (user wants to type text), use insert()

Current document state:
{context}
'''


class LLMParser:
  """Parse speech transcripts into function calls using an LLM."""

  def __init__(
    self,
    registry: Registry,
    model: str = 'claude-sonnet-4-20250514',
    api_key: str | None = None,
  ):
    self.registry = registry
    self.model = model
    self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not self.api_key:
      raise ValueError('ANTHROPIC_API_KEY not set')

    self._client = httpx.Client(
      base_url='https://api.anthropic.com/v1',
      headers={
        'x-api-key': self.api_key,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      timeout=30.0,
    )

  def parse(
    self,
    transcript: str,
    context: EditorContext | None = None,
  ) -> FunctionCall | None:
    """Parse a speech transcript into a function call.

    Returns None if no valid function call could be extracted.
    """
    context = context or EditorContext()
    tools = self.registry.tools_schema()

    # Build context string
    ctx_parts = []
    if context.document:
      ctx_parts.append(f'Document:\n"""\n{context.document}\n"""')
    ctx_parts.append(f'Cursor position: {context.cursor}')
    if context.selection:
      ctx_parts.append(f'Selection: {context.selection[0]}-{context.selection[1]}')
    if context.history:
      recent = [f'{c.name}({c.args})' for c in context.history[-3:]]
      ctx_parts.append(f'Recent operations: {", ".join(recent)}')

    context_str = '\n'.join(ctx_parts) if ctx_parts else 'Empty document'

    response = self._client.post(
      '/messages',
      json={
        'model': self.model,
        'max_tokens': 1024,
        'system': SYSTEM_PROMPT.format(context=context_str),
        'tools': tools,
        'tool_choice': {'type': 'auto'},
        'messages': [{'role': 'user', 'content': transcript}],
      },
    )
    response.raise_for_status()
    data = response.json()

    # Extract tool use from response
    for block in data.get('content', []):
      if block.get('type') == 'tool_use':
        return FunctionCall(
          name=block['name'],
          args=block.get('input', {}),
        )

    return None

  def parse_multi(
    self,
    transcript: str,
    context: EditorContext | None = None,
  ) -> list[FunctionCall]:
    """Parse a transcript that may contain multiple commands."""
    context = context or EditorContext()
    tools = self.registry.tools_schema()

    ctx_parts = []
    if context.document:
      ctx_parts.append(f'Document:\n"""\n{context.document}\n"""')
    ctx_parts.append(f'Cursor position: {context.cursor}')
    if context.selection:
      ctx_parts.append(f'Selection: {context.selection[0]}-{context.selection[1]}')
    if context.history:
      recent = [f'{c.name}({c.args})' for c in context.history[-3:]]
      ctx_parts.append(f'Recent operations: {", ".join(recent)}')

    context_str = '\n'.join(ctx_parts) if ctx_parts else 'Empty document'

    # For multi-command, we use a different prompt
    system = SYSTEM_PROMPT.format(context=context_str) + '''

The user may give multiple commands in one utterance. Call each function in order.
Example: "bold this and then move to the end" = format(bold) then move(end)'''

    response = self._client.post(
      '/messages',
      json={
        'model': self.model,
        'max_tokens': 2048,
        'system': system,
        'tools': tools,
        'tool_choice': {'type': 'any'},  # Force tool use
        'messages': [{'role': 'user', 'content': transcript}],
      },
    )
    response.raise_for_status()
    data = response.json()

    calls = []
    for block in data.get('content', []):
      if block.get('type') == 'tool_use':
        calls.append(FunctionCall(
          name=block['name'],
          args=block.get('input', {}),
        ))

    return calls


# Convenience function
_parser: LLMParser | None = None


def parse(
  transcript: str,
  context: EditorContext | None = None,
  registry: Registry | None = None,
) -> FunctionCall | None:
  """Parse a speech transcript into a function call."""
  global _parser
  if _parser is None:
    # Import editor to register functions
    import voice_control.editor  # noqa: F401
    _parser = LLMParser(registry or Registry())
  return _parser.parse(transcript, context)
