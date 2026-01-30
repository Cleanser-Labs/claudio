"""Function decorator and registry for voice control.

Define functions with @func decorator, auto-generates JSON schema from type hints.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, get_type_hints

import msgspec


class FunctionDef(msgspec.Struct, frozen=True):
  """Definition of a registered function."""

  name: str
  description: str
  parameters: dict[str, Any]  # JSON Schema
  fn: Callable = msgspec.field(default=None)

  def __call__(self, **kwargs):
    return self.fn(**kwargs)


class FunctionCall(msgspec.Struct, frozen=True):
  """A parsed function call from speech."""

  name: str
  args: dict[str, Any]


# Global registry for @func decorated functions
_registry: dict[str, FunctionDef] = {}


def func(fn: Callable) -> Callable:
  """Decorator to register a function for voice control.

  Extracts JSON schema from type hints and description from docstring.

  Example:
      @func
      def insert(position: str, content: str) -> None:
          '''Insert content at position.

          Args:
              position: Where to insert ("start", "end", "line:5")
              content: The text to insert
          '''
          ...
  """
  name = fn.__name__
  description, param_docs = _parse_docstring(fn.__doc__ or '')
  parameters = _extract_schema(fn, param_docs)

  fn_def = FunctionDef(
    name=name,
    description=description,
    parameters=parameters,
    fn=fn,
  )
  _registry[name] = fn_def
  return fn


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
  """Parse docstring into description and parameter docs.

  Returns:
      (description, {param_name: param_description})
  """
  if not docstring:
    return '', {}

  lines = docstring.strip().split('\n')
  description_lines = []
  param_docs = {}
  in_args = False
  current_param = None

  for line in lines:
    stripped = line.strip()

    if stripped.lower() in ('args:', 'arguments:', 'parameters:'):
      in_args = True
      continue

    if stripped.lower() in ('returns:', 'return:', 'raises:', 'example:', 'examples:'):
      in_args = False
      continue

    if in_args:
      # Check for "param_name: description" pattern
      match = re.match(r'^(\w+):\s*(.*)$', stripped)
      if match:
        current_param = match.group(1)
        param_docs[current_param] = match.group(2)
      elif current_param and stripped:
        # Continuation of previous param description
        param_docs[current_param] += ' ' + stripped
    else:
      if stripped:
        description_lines.append(stripped)

  description = ' '.join(description_lines)
  return description, param_docs


def _extract_schema(fn: Callable, param_docs: dict[str, str]) -> dict[str, Any]:
  """Extract JSON Schema from function signature."""
  hints = get_type_hints(fn)
  sig = inspect.signature(fn)

  properties = {}
  required = []

  for param_name, param in sig.parameters.items():
    if param_name in ('self', 'cls'):
      continue

    param_type = hints.get(param_name, Any)
    param_schema = _type_to_schema(param_type)

    if param_name in param_docs:
      param_schema['description'] = param_docs[param_name]

    properties[param_name] = param_schema

    if param.default is inspect.Parameter.empty:
      required.append(param_name)

  schema = {
    'type': 'object',
    'properties': properties,
  }
  if required:
    schema['required'] = required

  return schema


def _type_to_schema(t: type) -> dict[str, Any]:
  """Convert Python type to JSON Schema."""
  # Handle None/NoneType
  if t is type(None):
    return {'type': 'null'}

  # Handle basic types
  type_map = {
    str: {'type': 'string'},
    int: {'type': 'integer'},
    float: {'type': 'number'},
    bool: {'type': 'boolean'},
    Any: {},
  }

  if t in type_map:
    return type_map[t].copy()

  # Handle typing constructs
  origin = getattr(t, '__origin__', None)

  if origin is list:
    args = getattr(t, '__args__', (Any,))
    return {'type': 'array', 'items': _type_to_schema(args[0])}

  if origin is dict:
    args = getattr(t, '__args__', (str, Any))
    return {
      'type': 'object',
      'additionalProperties': _type_to_schema(args[1]) if len(args) > 1 else {},
    }

  # Handle Union (including Optional)
  if origin is type(int | str):  # UnionType in 3.10+
    args = t.__args__
    # Check if it's Optional (Union with None)
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and len(args) == 2:
      # It's Optional[X]
      return _type_to_schema(non_none[0])
    return {'anyOf': [_type_to_schema(a) for a in args]}

  # Handle Literal
  if origin is type(None):
    return {'type': 'null'}

  # Try msgspec for complex types (Struct, etc)
  try:
    return msgspec.json.schema(t)
  except Exception:
    return {}


class Registry:
  """Registry of functions for voice control."""

  def __init__(self, functions: dict[str, FunctionDef] | None = None):
    self._functions = dict(functions) if functions else dict(_registry)

  def __iter__(self):
    return iter(self._functions.values())

  def __len__(self):
    return len(self._functions)

  def __getitem__(self, name: str) -> FunctionDef:
    return self._functions[name]

  def get(self, name: str) -> FunctionDef | None:
    return self._functions.get(name)

  def add(self, fn_def: FunctionDef) -> None:
    """Add a function definition to the registry."""
    self._functions[fn_def.name] = fn_def

  def schema(self) -> dict[str, Any]:
    """Generate combined JSON Schema for all functions.

    Returns schema for structured generation that produces a FunctionCall.
    """
    fn_schemas = []
    for fn_def in self._functions.values():
      fn_schemas.append({
        'type': 'object',
        'properties': {
          'name': {'type': 'string', 'const': fn_def.name},
          'args': fn_def.parameters,
        },
        'required': ['name', 'args'],
        'additionalProperties': False,
      })

    return {
      '$schema': 'https://json-schema.org/draft/2020-12/schema',
      'oneOf': fn_schemas,
    }

  def tools_schema(self) -> list[dict[str, Any]]:
    """Generate OpenAI-style tools schema for LLM."""
    tools = []
    for fn_def in self._functions.values():
      tools.append({
        'type': 'function',
        'function': {
          'name': fn_def.name,
          'description': fn_def.description,
          'parameters': fn_def.parameters,
        },
      })
    return tools

  def call(self, name: str, args: dict[str, Any]) -> Any:
    """Execute a function by name with args."""
    fn_def = self._functions.get(name)
    if not fn_def:
      raise ValueError(f'Unknown function: {name}')
    return fn_def.fn(**args)

  def validate(self, call: FunctionCall) -> bool:
    """Validate a function call against the registry."""
    fn_def = self._functions.get(call.name)
    if not fn_def:
      return False
    # Basic validation - check required params present
    required = fn_def.parameters.get('required', [])
    return all(r in call.args for r in required)

  def parse(self, data: dict | str) -> FunctionCall:
    """Parse a dict or JSON string into a FunctionCall."""
    if isinstance(data, str):
      data = msgspec.json.decode(data)
    return FunctionCall(name=data['name'], args=data.get('args', {}))
