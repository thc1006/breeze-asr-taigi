"""Unit tests for word-timing -> line regrouping (no model, no GPU)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from taigi_asr.engines.faster_whisper import FasterWhisperEngine
from taigi_asr.segments import TimestampedSegment
from taigi_asr.word_grouping import group_words, words_match_text


@dataclass
class W:
    """Shape-compatible stand-in for ``faster_whisper.transcribe.Word``."""

    start: float | None
    end: float | None
    word: str


def contiguous(text: str, start: float, step: float = 0.25) -> list[W]:
    """One word per character with zero gaps, as Whisper DTW produces them."""
    return [W(start + i * step, start + (i + 1) * step, ch) for i, ch in enumerate(text)]


def group(words, max_gap: float = 0.4, max_span: float = 8.0):
    return group_words(words, max_gap=max_gap, max_span=max_span)


# --------------------------------------------------------------------------- #
# group_words
# --------------------------------------------------------------------------- #


def test_empty_words_yield_no_lines() -> None:
    assert group([]) == []


def test_short_contiguous_run_is_one_line() -> None:
    assert group(contiguous("今天天氣真好", 1.0)) == [TimestampedSegment(1.0, 2.5, "今天天氣真好")]


def test_splits_at_pause_at_or_above_threshold() -> None:
    words = contiguous("早安", 0.0) + contiguous("大家好", 0.9)  # gap = 0.4 s
    assert [s.text for s in group(words)] == ["早安", "大家好"]
    assert group(words)[1].start_time == pytest.approx(0.9)


def test_does_not_split_below_threshold() -> None:
    words = contiguous("早安", 0.0) + contiguous("大家好", 0.85)  # gap = 0.35 s
    assert [s.text for s in group(words)] == ["早安大家好"]


def test_span_cap_without_pause_info_fills_line_to_cap() -> None:
    # 40 chars x 0.25 s = 10 s, all gaps zero: nothing better than the cap.
    lines = group(contiguous("一" * 40, 0.0))
    assert [len(s.text) for s in lines] == [31, 9]
    assert all(s.end_time - s.start_time < 8.0 for s in lines)
    assert "".join(s.text for s in lines) == "一" * 40


def test_span_cap_prefers_widest_internal_gap() -> None:
    words = contiguous("一二三四五六七八九十", 0.0) + contiguous("甲乙丙丁戊己庚辛壬癸", 2.8)
    words += contiguous("子丑寅卯", 5.6)  # total span 6.6 s
    lines = group(words, max_span=6.0)
    # The cut happens once the span crosses 6.0 s (at 丑). Both pauses are
    # 0.3 s; ties go to the later boundary so the first line stays as long
    # as allowed.
    assert [s.text for s in lines] == ["一二三四五六七八九十甲乙丙丁戊己庚辛壬癸", "子丑寅卯"]


def test_span_cap_widest_gap_beats_later_smaller_gap() -> None:
    words = contiguous("一二三四", 0.0) + contiguous("五六七八九十", 1.3)  # 0.3 s gap
    words += contiguous("甲乙丙丁", 3.0)  # 0.2 s gap, span 4.0 s
    lines = group(words, max_span=4.0)
    assert [s.text for s in lines] == ["一二三四", "五六七八九十甲乙丙丁"]


def test_span_cap_prefers_punctuation_when_no_gap() -> None:
    words = contiguous("我們先開始，", 0.0) + contiguous("然後再討論下一個議題吧", 1.5)
    lines = group(words, max_span=4.0)
    assert lines[0].text == "我們先開始，"
    assert "".join(s.text for s in lines) == "我們先開始，然後再討論下一個議題吧"


def test_single_overlong_word_is_emitted_alone() -> None:
    words = [W(0.0, 9.0, "嗯"), W(9.0, 9.3, "好")]
    assert [s.text for s in group(words)] == ["嗯", "好"]


def test_preserves_inner_spaces_and_strips_edges() -> None:
    words = [W(0.0, 0.2, "用"), W(0.2, 0.6, " iPhone"), W(0.6, 0.8, " 拍")]
    assert group(words)[0].text == "用 iPhone 拍"
    assert group([W(0.0, 0.3, " hello"), W(0.3, 0.6, " world")])[0].text == "hello world"


def test_whitespace_only_chunk_is_dropped() -> None:
    words = [W(0.0, 0.2, " "), W(1.0, 1.3, "好")]
    assert group(words) == [TimestampedSegment(1.0, 1.3, "好")]


def test_zero_or_negative_duration_words_never_invert_line() -> None:
    lines = group([W(2.0, 2.0, "a"), W(1.9, 1.95, "b")])
    assert all(s.end_time >= s.start_time for s in lines)


def test_numpy_float_timings_are_coerced_to_python_float() -> None:
    np = pytest.importorskip("numpy")
    line = group([W(np.float64(0.5), np.float64(0.8), "好")])[0]
    assert type(line.start_time) is float and type(line.end_time) is float


# --------------------------------------------------------------------------- #
# words_match_text
# --------------------------------------------------------------------------- #


def test_match_requires_exact_text_modulo_whitespace() -> None:
    assert words_match_text(contiguous("伊講", 0.0), "伊講")
    assert words_match_text([W(0, 1, " hello"), W(1, 2, " world")], "hello world")
    assert not words_match_text(contiguous("伊講好啦", 0.0), "。伊講")


def test_match_rejects_empty_or_untimed_words() -> None:
    assert not words_match_text([], "好")
    assert not words_match_text([W(None, 1.0, "好")], "好")
    assert not words_match_text([W(0.0, None, "好")], "好")


# --------------------------------------------------------------------------- #
# FasterWhisperEngine.transcribe wiring (fake pipeline, no model)
# --------------------------------------------------------------------------- #


@dataclass
class Seg:
    start: float
    end: float
    text: str
    words: list[W] | None = None


class FakeBatched:
    def __init__(self, segments: list[Seg]) -> None:
        self.segments = segments
        self.kwargs: dict = {}

    def transcribe(self, _audio, **kwargs):
        self.kwargs = kwargs
        return iter(self.segments), None


def make_engine(segments: list[Seg]) -> tuple[FasterWhisperEngine, FakeBatched]:
    engine = FasterWhisperEngine(device="cpu", compute_type="int8", batch_size=1)
    fake = FakeBatched(segments)
    engine._batched = fake
    engine._loaded = True
    return engine, fake


def long_segment() -> Seg:
    words = contiguous("早安", 0.0) + contiguous("大家好", 1.0)
    return Seg(0.0, 25.0, "早安大家好", words)


def test_engine_without_word_timestamps_keeps_segment_lines() -> None:
    engine, fake = make_engine([long_segment()])
    out = engine.transcribe("x.wav", word_timestamps=False)
    assert out == [TimestampedSegment(0.0, 25.0, "早安大家好")]
    assert fake.kwargs["word_timestamps"] is False


def test_engine_with_word_timestamps_splits_lines() -> None:
    engine, fake = make_engine([long_segment()])
    out = engine.transcribe("x.wav", word_timestamps=True)
    assert [s.text for s in out] == ["早安", "大家好"]
    assert fake.kwargs["word_timestamps"] is True


def test_engine_falls_back_when_words_were_shifted_between_segments() -> None:
    # faster-whisper distributes words by token count: a segment starting
    # with punctuation pushes words forward, leaving the last one empty.
    segs = [
        Seg(0.0, 1.0, "食飯", [W(0.0, 0.5, "食"), W(0.5, 1.0, "飯。")]),
        Seg(1.0, 2.0, "。伊講", [W(1.0, 1.4, "伊"), W(1.4, 1.8, "講"), W(1.8, 2.0, "好啦")]),
        Seg(2.0, 3.0, "好啦", []),
    ]
    engine, _ = make_engine(segs)
    out = engine.transcribe("x.wav", word_timestamps=True)
    assert [s.text for s in out] == ["食飯", "。伊講", "好啦"]  # no duplicated "好啦"


def test_engine_skips_blank_segments() -> None:
    engine, _ = make_engine([Seg(0.0, 1.0, "  ", []), long_segment()])
    assert [s.text for s in engine.transcribe("x.wav", word_timestamps=True)] == ["早安", "大家好"]
