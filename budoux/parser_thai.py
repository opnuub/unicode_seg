# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thai BudouX parser using grapheme-cluster-based features."""

import json
import os
import queue
import typing
from html.parser import HTMLParser

from icu import BreakIterator, Locale

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
SEP = '\ue000'
HTMLAttr = typing.List[typing.Tuple[str, typing.Union[str, None]]]
PARENT_CSS_STYLE = 'word-break: keep-all; overflow-wrap: anywhere;'
with open(
    os.path.join(os.path.dirname(__file__), 'skip_nodes.json'),
    encoding='utf-8') as f:
  SKIP_NODES: typing.Set[str] = set(json.load(f))


class _ElementState(object):
  """Represents the state for an element."""

  def __init__(self, tag: str, to_skip: bool) -> None:
    self.tag = tag
    self.to_skip = to_skip


class _TextContentExtractor(HTMLParser):
  """An HTML parser to extract text content."""
  output = ''

  def handle_data(self, data: str) -> None:
    self.output += data


class _HTMLChunkResolver(HTMLParser):
  """An HTML parser to resolve the given HTML string and semantic chunks."""
  output = ''

  def __init__(self, chunks: typing.List[str], separator: str):
    HTMLParser.__init__(self)
    self.chunks_joined = SEP.join(chunks)
    self.separator = separator
    self.to_skip = False
    self.scan_index = 0
    self.element_stack: queue.LifoQueue[_ElementState] = queue.LifoQueue()

  def handle_starttag(self, tag: str, attrs: HTMLAttr) -> None:
    attr_pairs = []
    for attr in attrs:
      if attr[1] is None:
        attr_pairs.append(' ' + attr[0])
      else:
        attr_pairs.append(' %s="%s"' % (attr[0], attr[1]))
    encoded_attrs = ''.join(attr_pairs)
    self.element_stack.put(_ElementState(tag, self.to_skip))
    if tag.upper() in SKIP_NODES:
      if not self.to_skip and self.chunks_joined[self.scan_index] == SEP:
        self.scan_index += 1
        self.output += self.separator
      self.to_skip = True
    self.output += '<%s%s>' % (tag, encoded_attrs)

  def handle_endtag(self, tag: str) -> None:
    self.output += '</%s>' % (tag)
    while not self.element_stack.empty():
      state = self.element_stack.get_nowait()
      if state.tag == tag:
        self.to_skip = state.to_skip
        break

  def handle_data(self, data: str) -> None:
    for char in data:
      if not char == self.chunks_joined[self.scan_index]:
        if not self.to_skip:
          self.output += self.separator
        self.scan_index += 1
      self.output += char
      self.scan_index += 1


def _get_text(html: str) -> str:
  """Gets the text content from the input HTML string."""
  text_content_extractor = _TextContentExtractor()
  text_content_extractor.feed(html)
  return text_content_extractor.output


def _resolve(phrases: typing.List[str],
             html: str,
             separator: str = '\u200b') -> str:
  """Wraps phrases in the HTML string with non-breaking markup."""
  resolver = _HTMLChunkResolver(phrases, separator)
  resolver.feed(html)
  result = '<span style="%s">%s</span>' % (PARENT_CSS_STYLE, resolver.output)
  return result


def _thai_char_type_from_codepoint(char: str) -> str:
  """Returns the Thai character type for a single code point."""
  codepoint = ord(char)
  if 0x0E40 <= codepoint <= 0x0E44:
    return 'LV'
  if char == '\u0E31' or 0x0E34 <= codepoint <= 0x0E37:
    return 'AV'
  if 0x0E38 <= codepoint <= 0x0E39:
    return 'BV'
  if char == '\u0E30' or char == '\u0E32':
    return 'RV'
  return 'C'


def _get_left_thai_char_type(cluster: str) -> str:
  """Returns the Thai character type at the right edge of the left cluster."""
  return _thai_char_type_from_codepoint(cluster[-1])


def _get_right_thai_char_type(cluster: str) -> str:
  """Returns the Thai character type at the left edge of the right cluster."""
  return _thai_char_type_from_codepoint(cluster[0])


def _get_left_edge_codepoint(cluster: str) -> str:
  """Returns the rightmost code point in the left grapheme cluster."""
  return cluster[-1]


def _get_right_edge_codepoint(cluster: str) -> str:
  """Returns the leftmost code point in the right grapheme cluster."""
  return cluster[0]


def _to_grapheme_clusters(text: str) -> typing.List[str]:
  """Splits text into ICU grapheme clusters."""
  iterator = BreakIterator.createCharacterInstance(Locale.getRoot())
  iterator.setText(text)
  breakpoints = [0]
  for breakpoint in iterator:
    breakpoints.append(breakpoint)
  return [
      text[breakpoints[i]:breakpoints[i + 1]]
      for i in range(len(breakpoints) - 1)
  ]


class ThaiParser:
  """BudouX parser variant for Thai grapheme-cluster features."""

  def __init__(self, model: typing.Dict[str, typing.Dict[str, int]]):
    self.model = model

  def parse(self, sentence: str) -> typing.List[str]:
    """Parses the input sentence and returns a list of semantic chunks."""
    if sentence == '':
      return []

    clusters = _to_grapheme_clusters(sentence)
    chunks = [clusters[0]]
    base_score = -sum(sum(g.values()) for g in self.model.values()) * 0.5
    for i in range(1, len(clusters)):
      score = base_score
      if i > 2:
        score += self.model.get('UW1', {}).get(clusters[i - 3], 0)
      if i > 1:
        score += self.model.get('UW2', {}).get(clusters[i - 2], 0)
      score += self.model.get('UW3', {}).get(clusters[i - 1], 0)
      score += self.model.get('UW4', {}).get(clusters[i], 0)
      if i + 1 < len(clusters):
        score += self.model.get('UW5', {}).get(clusters[i + 1], 0)
      if i + 2 < len(clusters):
        score += self.model.get('UW6', {}).get(clusters[i + 2], 0)

      if i > 1:
        score += self.model.get('BW1', {}).get(clusters[i - 2] + clusters[i - 1], 0)
      score += self.model.get('BW2', {}).get(clusters[i - 1] + clusters[i], 0)
      if i + 1 < len(clusters):
        score += self.model.get('BW3', {}).get(clusters[i] + clusters[i + 1], 0)

      if i > 2:
        score += self.model.get('TW1', {}).get(
            clusters[i - 3] + clusters[i - 2] + clusters[i - 1], 0)
      if i > 1:
        score += self.model.get('TW2', {}).get(
            clusters[i - 2] + clusters[i - 1] + clusters[i], 0)
      if i + 1 < len(clusters):
        score += self.model.get('TW3', {}).get(
            clusters[i - 1] + clusters[i] + clusters[i + 1], 0)
      if i + 2 < len(clusters):
        score += self.model.get('TW4', {}).get(
            clusters[i] + clusters[i + 1] + clusters[i + 2], 0)

      left_type = _get_left_thai_char_type(clusters[i - 1])
      right_type = _get_right_thai_char_type(clusters[i])
      if left_type == 'LV' and right_type == 'C':
        score += self.model.get(_get_left_edge_codepoint(clusters[i - 1]),
                                {}).get(_get_right_edge_codepoint(clusters[i]),
                                        0)
      if left_type == 'C' and right_type == 'RV':
        score += self.model.get(_get_left_edge_codepoint(clusters[i - 1]),
                                {}).get(_get_right_edge_codepoint(clusters[i]),
                                        0)
      if left_type == 'C' and right_type == 'AV':
        score += self.model.get(_get_left_edge_codepoint(clusters[i - 1]),
                                {}).get(_get_right_edge_codepoint(clusters[i]),
                                        0)

      if score > 0:
        chunks.append(clusters[i])
      else:
        chunks[-1] += clusters[i]
    return chunks

  def translate_html_string(self, html: str) -> str:
    """Translates the given HTML string with markups for semantic line breaks."""
    text_content = _get_text(html)
    chunks = self.parse(text_content)
    return _resolve(chunks, html)


def load_default_thai_parser() -> ThaiParser:
  """Loads a Thai grapheme-cluster parser equipped with the default Thai model."""
  with open(os.path.join(MODEL_DIR, 'th.json'), encoding='utf-8') as f:
    model = json.load(f)
  return ThaiParser(model)
