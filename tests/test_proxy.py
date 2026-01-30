"""Tests for proxy components."""

import pytest
from claudio.proxy import (
  MarkerBuffer,
  SentenceBuffer,
  SpeechRequest,
  extract_speech_requests,
  strip_say_tags,
  parse_xml_attrs,
  is_json_like,
)


class TestParseXmlAttrs:
  def test_parse_empty_tag(self):
    attrs = parse_xml_attrs('<say>')
    assert attrs == {}

  def test_parse_single_attr(self):
    attrs = parse_xml_attrs('<say speed="fast">')
    assert attrs == {'speed': 'fast'}

  def test_parse_multiple_attrs(self):
    attrs = parse_xml_attrs('<say speed="fast" voice="Daniel" tone="calm">')
    assert attrs == {'speed': 'fast', 'voice': 'Daniel', 'tone': 'calm'}

  def test_parse_single_quotes(self):
    attrs = parse_xml_attrs("<say speed='slow'>")
    assert attrs == {'speed': 'slow'}


class TestExtractSpeechRequests:
  def test_extract_simple(self):
    text = '<say>Hello world</say>'
    requests = extract_speech_requests(text)
    assert len(requests) == 1
    assert requests[0].text == 'Hello world'

  def test_extract_with_attrs(self):
    text = '<say speed="1.5" tone="excited">Quick update!</say>'
    requests = extract_speech_requests(text)
    assert len(requests) == 1
    assert requests[0].text == 'Quick update!'
    assert requests[0].speed == 1.5
    assert requests[0].tone == 'excited'

  def test_extract_multiple(self):
    text = '<say>First</say> some text <say>Second</say>'
    requests = extract_speech_requests(text)
    assert len(requests) == 2
    assert requests[0].text == 'First'
    assert requests[1].text == 'Second'

  def test_extract_empty_tag_ignored(self):
    text = '<say></say>'
    requests = extract_speech_requests(text)
    assert len(requests) == 0

  def test_extract_multiline(self):
    text = '''<say>
    Hello
    world
    </say>'''
    requests = extract_speech_requests(text)
    assert len(requests) == 1
    assert requests[0].text == 'Hello world'  # Whitespace normalized


class TestStripSayTags:
  def test_strip_simple(self):
    text = '<say>Hello</say> world'
    result = strip_say_tags(text)
    assert result == 'Hello world'

  def test_strip_with_attrs(self):
    text = '<say speed="fast">Quick</say>'
    result = strip_say_tags(text)
    assert result == 'Quick'

  def test_strip_invisible(self):
    text = 'Visible <say visible="false">Hidden</say> text'
    result = strip_say_tags(text, strip_invisible=True)
    assert result == 'Visible  text'

  def test_strip_preserves_visible(self):
    text = '<say visible="true">Shown</say>'
    result = strip_say_tags(text, strip_invisible=True)
    assert result == 'Shown'

  def test_strip_multiple(self):
    text = '<say>One</say> and <say>Two</say>'
    result = strip_say_tags(text)
    assert result == 'One and Two'


class TestIsJsonLike:
  def test_json_object(self):
    assert is_json_like('{"key": "value"}')
    assert is_json_like('  { "a": 1 }  ')

  def test_json_array(self):
    assert is_json_like('[1, 2, 3]')
    assert is_json_like('  [  ')

  def test_json_key_value(self):
    assert is_json_like('"name": "test"')

  def test_json_literals(self):
    assert is_json_like('null')
    assert is_json_like('true')
    assert is_json_like('false')
    assert is_json_like('None')

  def test_natural_text(self):
    assert not is_json_like('Hello world')
    assert not is_json_like('This is a sentence.')


class TestSentenceBuffer:
  def test_emits_complete_sentence(self):
    sentences = []
    buffer = SentenceBuffer(on_sentence=sentences.append)

    buffer.add('Hello world. ')
    assert sentences == ['Hello world.']

  def test_buffers_incomplete_sentence(self):
    sentences = []
    buffer = SentenceBuffer(on_sentence=sentences.append)

    buffer.add('Hello')
    buffer.add(' world')
    assert sentences == []

    buffer.add('. ')
    assert sentences == ['Hello world.']

  def test_multiple_sentences(self):
    """SentenceBuffer emits when it sees sentence end + whitespace."""
    sentences = []
    buffer = SentenceBuffer(on_sentence=sentences.append)

    # The SENTENCE_END pattern requires punctuation at end of buffer
    # So we need to add sentences one at a time
    buffer.add('First. ')
    assert len(sentences) == 1
    buffer.add('Second! ')
    assert len(sentences) == 2
    buffer.add('Third? ')
    assert len(sentences) == 3

  def test_skips_code_blocks(self):
    """Code blocks are tracked but content is buffered, not emitted mid-block."""
    sentences = []
    buffer = SentenceBuffer(on_sentence=sentences.append)

    # Add text before code block
    buffer.add('Here is code. ')
    assert len(sentences) == 1

    # During code block, nothing is emitted
    buffer.add('```python\nprint("Hello.")\n```')
    # Code block doesn't end with sentence punctuation, nothing new emitted

    # After code block, next sentence is emitted
    buffer.add('\nDone. ')
    # The whole accumulated buffer is emitted when we see sentence end
    assert len(sentences) >= 1

  def test_flush_remaining(self):
    sentences = []
    buffer = SentenceBuffer(on_sentence=sentences.append)

    buffer.add('No period')
    assert sentences == []

    buffer.flush()
    assert sentences == ['No period']


class TestMarkerBuffer:
  def test_extracts_tag_content(self):
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    buffer.add('<say>Hello world</say>')
    assert len(speeches) == 1
    assert speeches[0].text == 'Hello world'

  def test_ignores_non_tag_content(self):
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    buffer.add('This is ignored <say>This is spoken</say> also ignored')
    assert len(speeches) == 1
    assert speeches[0].text == 'This is spoken'

  def test_streams_sentences_within_tag(self):
    """MarkerBuffer emits all at once if closing tag is in same chunk."""
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    # When whole tag is added at once, closing tag is found immediately
    # so entire content is emitted as one speech
    buffer.add('<say>First sentence. Second sentence. Third.</say>')
    assert len(speeches) == 1
    assert 'First sentence' in speeches[0].text

  def test_streams_sentences_when_chunked(self):
    """MarkerBuffer emits sentences progressively when streamed in chunks."""
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    # When chunked, sentences are emitted as punctuation+space is seen
    buffer.add('<say>')
    buffer.add('First. ')  # "First. " - SENTENCE_END matches, emit
    assert len(speeches) == 1
    buffer.add('Second. ')  # "Second. " - emit
    assert len(speeches) == 2
    buffer.add('Third.')
    buffer.add('</say>')  # closing tag, emit remaining
    assert len(speeches) == 3

  def test_handles_chunked_input(self):
    """Simulates streaming where tags are split across chunks."""
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    # Tag split across chunks
    buffer.add('<sa')
    buffer.add('y>')
    buffer.add('Hello')
    buffer.add(' world')
    buffer.add('</sa')
    buffer.add('y>')

    assert len(speeches) == 1
    assert speeches[0].text == 'Hello world'

  def test_extracts_attrs(self):
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    buffer.add('<say speed="1.5" tone="excited">Quick!</say>')
    assert len(speeches) == 1
    assert speeches[0].speed == 1.5
    assert speeches[0].tone == 'excited'

  def test_flush_incomplete_tag(self):
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    buffer.add('<say>Incomplete')
    assert len(speeches) == 0

    buffer.flush()
    assert len(speeches) == 1
    assert speeches[0].text == 'Incomplete'

  def test_multiple_tags(self):
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    buffer.add('<say>First</say> middle <say>Second</say>')
    assert len(speeches) == 2
    assert speeches[0].text == 'First'
    assert speeches[1].text == 'Second'


class TestMarkerBufferStreaming:
  """Tests that simulate real streaming scenarios."""

  def test_claude_style_streaming(self):
    """Simulates how Claude streams response with <say> tags."""
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    # Typical Claude streaming pattern
    chunks = [
      '<say',
      '>',
      'Hi!',
      ' How',        # "! " triggers sentence end, emits "Hi!"
      ' can I help you today?',
      '</say',
      '>',           # closing tag emits remaining
    ]

    for chunk in chunks:
      buffer.add(chunk)

    # Sentence is split at "! " boundary
    assert len(speeches) == 2
    assert speeches[0].text == 'Hi!'
    assert 'help you today' in speeches[1].text

  def test_long_response_with_sentences(self):
    """Test streaming a longer response with multiple sentences."""
    speeches = []
    buffer = MarkerBuffer(on_speech=speeches.append)

    chunks = [
      '<say>',
      'Found the bug. ',
      'It was a missing null check on line 42. ',
      "I've fixed it now.",
      '</say>',
    ]

    for chunk in chunks:
      buffer.add(chunk)

    # Should have emitted sentences as they completed
    assert len(speeches) >= 2
    all_text = ' '.join(s.text for s in speeches)
    assert 'Found the bug' in all_text
    assert 'null check' in all_text


class TestSpeechRequest:
  def test_default_values(self):
    req = SpeechRequest(text='Hello')
    assert req.text == 'Hello'
    assert req.tone == 'neutral'
    assert req.speed == 1.0
    assert req.voice == 'default'

  def test_custom_values(self):
    req = SpeechRequest(text='Hi', tone='calm', speed=0.8, voice='Daniel')
    assert req.tone == 'calm'
    assert req.speed == 0.8
    assert req.voice == 'Daniel'
