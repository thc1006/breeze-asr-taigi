"""Regroup per-word ASR timings into short transcript lines.

Engine-agnostic and dependency-free so it can be unit-tested without a model.
A "word" is anything with ``start`` / ``end`` (seconds) and ``word`` (text)
attributes, e.g. ``faster_whisper.transcribe.Word``.

Two facts about faster-whisper's word timings shape the heuristics below:

* Whisper's DTW alignment is contiguous: each word's ``end`` equals the next
  word's ``start``, so pauses are absorbed into word durations. A real gap
  between words only appears where VAD cut out silence (with this project's
  VAD settings a gap of ``g`` seconds means roughly ``g + 0.4`` s of actual
  silence) or where faster-whisper clamps an over-long word at ``。！？``.
* For Chinese output each word is usually a single character with no leading
  space, and ``"".join(w.word for w in seg.words)`` reproduces ``seg.text``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from taigi_asr.segments import TimestampedSegment

# Boundaries right after these are preferred when an over-long line has to be
# cut and the word timings carry no pause information.
_PUNCTUATION_BREAKS = frozenset("，。！？、；：,.!?;:")
_WHITESPACE_RE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _WHITESPACE_RE.sub("", text)


def words_match_text(words: Sequence[Any], text: str) -> bool:
    """True iff every word has timings and the words spell exactly ``text``.

    faster-whisper hands words to segments by token count, so when a segment
    starts with punctuation its words shift into the previous segment and the
    last segment can end up with ``words == []``. Regrouping such a segment
    from its words would duplicate or drop text, so callers should fall back
    to the segment-level line whenever this returns False. Whitespace is
    ignored: it only affects how words are delimited, not what was said.
    """
    if not words:
        return False
    if any(w.start is None or w.end is None for w in words):
        return False
    return _squash("".join(w.word for w in words)) == _squash(text)


def group_words(
    words: Sequence[Any],
    *,
    max_gap: float,
    max_span: float,
) -> list[TimestampedSegment]:
    """Collapse per-word timings into lines no longer than ``max_span`` seconds.

    A new line starts whenever the silence between two words is at least
    ``max_gap`` seconds. A line that reaches ``max_span`` is cut at the best
    internal boundary (see :func:`_best_cut`). Words are assumed to be in time
    order and to carry timings; check :func:`words_match_text` first.
    """
    lines: list[TimestampedSegment] = []
    buf: list[Any] = []

    def emit(chunk: Sequence[Any]) -> None:
        text = "".join(w.word for w in chunk).strip()
        if not text:
            return
        start = float(chunk[0].start)
        end = max(float(chunk[-1].end), start)
        lines.append(TimestampedSegment(start, end, text))

    for w in words:
        if buf and float(w.start) - float(buf[-1].end) >= max_gap:
            emit(buf)
            buf = []
        buf.append(w)
        while len(buf) > 1 and float(buf[-1].end) - float(buf[0].start) >= max_span:
            cut = _best_cut(buf)
            emit(buf[:cut])
            buf = buf[cut:]
    if buf:
        emit(buf)
    return lines


def _best_cut(buf: Sequence[Any]) -> int:
    """Index ``i`` (1 <= i < len(buf)) to split ``buf`` into ``buf[:i]`` / ``buf[i:]``.

    Preference order:

    1. the widest positive gap between consecutive words (a real pause);
    2. right after a punctuation mark;
    3. the latest boundary, i.e. fill the line up to ``max_span``.

    Without (1) and (2), DTW timings carry no pause information, so (3) is no
    worse than any other boundary and keeps lines as long as allowed.
    """
    best_i = len(buf) - 1
    best_key: tuple[float, bool] | None = None
    for i in range(1, len(buf)):
        gap = max(float(buf[i].start) - float(buf[i - 1].end), 0.0)
        after_punct = buf[i - 1].word.strip()[-1:] in _PUNCTUATION_BREAKS
        key = (round(gap, 3), after_punct)
        # ``>=`` so ties go to the later boundary.
        if best_key is None or key >= best_key:
            best_i, best_key = i, key
    return best_i
