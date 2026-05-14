"""Tests for taigi_asr.diarize (the pure-Python parts — no pyannote needed).

Covers attribute_speakers overlap math, RTTM round-trip, and TimestampedSegment
speaker-rendering. The DiarizationPipeline class itself requires HF auth + GPU
and is exercised by ``scripts/diarize_poc.py`` and the integration smoke test.
"""

from __future__ import annotations

import pytest

from taigi_asr.diarize import (
    SpeakerTurn,
    attribute_speakers,
    format_speaker_totals,
    parse_rttm,
    turns_to_rttm,
)
from taigi_asr.segments import TimestampedSegment


def _seg(s: float, e: float, txt: str = "x") -> TimestampedSegment:
    return TimestampedSegment(start_time=s, end_time=e, text=txt)


class TestAttributeSpeakers:
    def test_no_turns_returns_segments_unchanged(self) -> None:
        segs = [_seg(0, 5), _seg(5, 10)]
        out = attribute_speakers(segs, [])
        assert out == segs
        assert all(s.speaker is None for s in out)

    def test_single_speaker_dominant_overlap(self) -> None:
        segs = [_seg(0, 10)]
        turns = [SpeakerTurn(0, 8, "A"), SpeakerTurn(8, 10, "B")]
        out = attribute_speakers(segs, turns)
        assert out[0].speaker == "A"

    def test_tie_resolution_picks_first_in_sorted_order(self) -> None:
        # Equal overlap (5s each). The defensive sort by ``start`` makes the
        # iteration order deterministic, and ``max()`` over an insertion-
        # ordered dict returns the first-inserted key on ties. So "A" wins.
        segs = [_seg(0, 10)]
        turns = [SpeakerTurn(0, 5, "A"), SpeakerTurn(5, 10, "B")]
        out = attribute_speakers(segs, turns)
        assert out[0].speaker == "A"

    def test_tie_resolution_stable_under_input_reorder(self) -> None:
        # Same data, caller passes turns in reverse order. The defensive sort
        # must rescue the contract so the same speaker still wins.
        segs = [_seg(0, 10)]
        turns = [SpeakerTurn(5, 10, "B"), SpeakerTurn(0, 5, "A")]
        out = attribute_speakers(segs, turns)
        assert out[0].speaker == "A"

    def test_zero_duration_segment_gets_no_speaker(self) -> None:
        segs = [_seg(5.0, 5.0)]
        turns = [SpeakerTurn(0, 10, "A")]
        out = attribute_speakers(segs, turns)
        assert out[0].speaker is None  # no positive overlap

    def test_segment_outside_all_turns_keeps_none(self) -> None:
        segs = [_seg(100, 110)]
        turns = [SpeakerTurn(0, 50, "A"), SpeakerTurn(50, 90, "B")]
        out = attribute_speakers(segs, turns)
        assert out[0].speaker is None

    def test_unsorted_turns_handled_by_defensive_sort(self) -> None:
        # Caller violates sort contract; defensive sort makes the answer
        # correct anyway (early-break loop would otherwise drop turns).
        segs = [_seg(0, 100)]
        turns = [
            SpeakerTurn(50, 90, "B"),  # B should dominate (40s)
            SpeakerTurn(0, 30, "A"),   # A is shorter (30s)
        ]
        out = attribute_speakers(segs, turns)
        assert out[0].speaker == "B"

    def test_multiple_segments_independent_attribution(self) -> None:
        segs = [_seg(0, 10), _seg(20, 30), _seg(40, 50)]
        turns = [
            SpeakerTurn(0, 10, "A"),
            SpeakerTurn(20, 30, "B"),
            SpeakerTurn(40, 50, "C"),
        ]
        out = attribute_speakers(segs, turns)
        assert [s.speaker for s in out] == ["A", "B", "C"]


class TestRttmRoundtrip:
    def test_basic_roundtrip(self) -> None:
        turns = [
            SpeakerTurn(0.0, 5.5, "SPEAKER_00"),
            SpeakerTurn(5.5, 10.25, "SPEAKER_01"),
        ]
        rttm = turns_to_rttm(turns, uri="test_audio")
        out, tmp = [], []
        # Write/read via tmpfile equivalent — parse_rttm reads from disk, but
        # we can construct via splitlines pass-through with a tiny helper:
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".rttm", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(rttm)
            tmp_path = Path(fh.name)
        try:
            out = parse_rttm(tmp_path)
        finally:
            tmp_path.unlink()
        assert len(out) == 2
        assert out[0] == SpeakerTurn(0.0, 5.5, "SPEAKER_00")
        assert out[1] == SpeakerTurn(5.5, 10.25, "SPEAKER_01")

    def test_empty_turns_produces_empty_rttm(self) -> None:
        assert turns_to_rttm([], uri="x") == ""

    def test_invalid_uri_raises(self) -> None:
        with pytest.raises(ValueError):
            turns_to_rttm([], uri="has spaces")
        with pytest.raises(ValueError):
            turns_to_rttm([], uri="")

    def test_invalid_speaker_label_raises(self) -> None:
        turns = [SpeakerTurn(0, 1, "bad speaker")]
        with pytest.raises(ValueError):
            turns_to_rttm(turns, uri="ok")

    def test_non_positive_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            turns_to_rttm([SpeakerTurn(5.0, 5.0, "A")], uri="ok")
        with pytest.raises(ValueError, match="duration must be positive"):
            turns_to_rttm([SpeakerTurn(5.0, 3.0, "A")], uri="ok")

    def test_parse_rttm_skips_non_positive_duration(self) -> None:
        """Symmetric with turns_to_rttm: read-path drops bad rows silently."""
        from pathlib import Path
        import tempfile

        body = (
            "SPEAKER mtg 1 0.000 5.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER mtg 1 5.000 0.000 <NA> <NA> SPEAKER_BAD <NA> <NA>\n"
            "SPEAKER mtg 1 5.000 -3.000 <NA> <NA> SPEAKER_NEG <NA> <NA>\n"
            "SPEAKER mtg 1 6.000 2.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".rttm", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(body)
            tmp = Path(fh.name)
        try:
            turns = parse_rttm(tmp)
        finally:
            tmp.unlink()
        # Only the two well-formed rows survive.
        assert {t.speaker for t in turns} == {"SPEAKER_00", "SPEAKER_01"}

    def test_parse_rttm_handles_utf8_bom(self) -> None:
        """A BOM-prefixed RTTM (Windows tooling) must not silently drop cue #1."""
        from pathlib import Path
        import tempfile

        body = (
            "SPEAKER mtg 1 0.000 5.500 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER mtg 1 5.500 4.750 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
        )
        # Explicit BOM bytes, then UTF-8 body.
        raw = "﻿" + body
        with tempfile.NamedTemporaryFile(
            "w", suffix=".rttm", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(raw)
            tmp = Path(fh.name)
        try:
            turns = parse_rttm(tmp)
        finally:
            tmp.unlink()
        assert len(turns) == 2
        assert turns[0].speaker == "SPEAKER_00"


class TestFormatSpeakerTotals:
    def test_aggregates_and_sorts(self) -> None:
        turns = [
            SpeakerTurn(0, 30, "A"),
            SpeakerTurn(30, 40, "B"),
            SpeakerTurn(40, 100, "A"),  # A: 30 + 60 = 90s
        ]
        out = format_speaker_totals(turns, total_duration=100.0)
        assert out[0][0] == "A"
        assert out[0][1] == pytest.approx(90.0)
        assert out[0][2] == pytest.approx(90.0)
        assert out[1][0] == "B"
        assert out[1][2] == pytest.approx(10.0)


class TestSegmentSpeakerRendering:
    def test_with_speaker_returns_new_immutable(self) -> None:
        a = _seg(0, 1, "hi")
        b = a.with_speaker("S0")
        assert a.speaker is None  # original unchanged
        assert b.speaker == "S0"
        assert b.text == "hi"

    def test_srt_block_prefixes_speaker(self) -> None:
        s = _seg(0, 1, "hi").with_speaker("SPK0")
        block = s.to_srt_block(1)
        assert "[SPK0] hi" in block

    def test_txt_line_prefixes_speaker(self) -> None:
        s = _seg(0, 1, "hi").with_speaker("SPK0")
        assert "[SPK0] hi" in s.to_timestamp_line()

    def test_vtt_uses_voice_tag(self) -> None:
        s = _seg(0, 1, "hi").with_speaker("SPK0")
        block = s.to_vtt_block()
        assert "<v SPK0>hi" in block

    def test_vtt_escapes_unsafe_speaker(self) -> None:
        s = _seg(0, 1, "hi").with_speaker("a<b>&c")
        block = s.to_vtt_block()
        assert "<v a&lt;b&gt;&amp;c>hi" in block

    def test_json_includes_speaker_when_set(self) -> None:
        s = _seg(0, 1, "hi").with_speaker("SPK0")
        assert s.to_json_dict() == {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "speaker": "SPK0",
        }

    def test_json_omits_speaker_when_none(self) -> None:
        # Backward compat: existing consumers see the exact same shape.
        s = _seg(0, 1, "hi")
        assert s.to_json_dict() == {"start": 0.0, "end": 1.0, "text": "hi"}
