"""Tests for taigi_asr.diarize (the pure-Python parts — no pyannote needed).

Covers attribute_speakers overlap math, RTTM round-trip, and TimestampedSegment
speaker-rendering. The DiarizationPipeline class is exercised through fakes for
pyannote / huggingface_hub; real HF auth + GPU runs go through
``scripts/diarize_poc.py``.
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
            SpeakerTurn(0, 30, "A"),  # A is shorter (30s)
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
        # parse_rttm reads from disk; tmpfile is the simplest round-trip rig.
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False, encoding="utf-8") as fh:
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
        """Symmetric with turns_to_rttm: read-path drops bad rows silently.

        Also exercises three other skip branches in parse_rttm: non-SPEAKER
        rows (e.g. comments), too-few columns (malformed), and unparseable
        float timestamps. All four bad-row classes must be dropped silently
        without raising — third-party RTTM tolerance is the contract.
        """
        import tempfile
        from pathlib import Path

        body = (
            "# comment row — must be skipped (not SPEAKER prefix)\n"
            "SPEAKER mtg 1 0.000 5.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER too few cols\n"
            "SPEAKER mtg 1 not_a_float 1.0 <NA> <NA> SPEAKER_BAD <NA> <NA>\n"
            "SPEAKER mtg 1 5.000 0.000 <NA> <NA> SPEAKER_BAD <NA> <NA>\n"
            "SPEAKER mtg 1 5.000 -3.000 <NA> <NA> SPEAKER_NEG <NA> <NA>\n"
            "SPEAKER mtg 1 6.000 2.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False, encoding="utf-8") as fh:
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
        import tempfile
        from pathlib import Path

        body = (
            "SPEAKER mtg 1 0.000 5.500 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
            "SPEAKER mtg 1 5.500 4.750 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
        )
        # Explicit BOM bytes, then UTF-8 body.
        raw = "﻿" + body
        with tempfile.NamedTemporaryFile("w", suffix=".rttm", delete=False, encoding="utf-8") as fh:
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


class TestPyannoteVersionGate:
    """3.x must fail with an upgrade hint, not a deep TypeError/AttributeError."""

    def _fake_version(self, monkeypatch, versions: dict[str, str]) -> None:
        import importlib.metadata as md

        real = md.version

        def fake(name: str) -> str:
            if name in versions:
                return versions[name]
            if name == "pyannote.audio":
                raise md.PackageNotFoundError(name)
            return real(name)

        monkeypatch.setattr(md, "version", fake)

    def test_rejects_pyannote_3(self, monkeypatch) -> None:
        from taigi_asr.diarize import _check_pyannote_version
        from taigi_asr.errors import ModelLoadError

        self._fake_version(monkeypatch, {"pyannote.audio": "3.4.0"})
        with pytest.raises(ModelLoadError, match=r"3\.4\.0 is too old.*>=4\.0\.3"):
            _check_pyannote_version()

    def test_reports_missing_pyannote(self, monkeypatch) -> None:
        from taigi_asr.diarize import _check_pyannote_version
        from taigi_asr.errors import ModelLoadError

        self._fake_version(monkeypatch, {})
        with pytest.raises(ModelLoadError, match="not installed"):
            _check_pyannote_version()

    def test_accepts_pyannote_4(self, monkeypatch) -> None:
        from taigi_asr.diarize import _check_pyannote_version

        self._fake_version(monkeypatch, {"pyannote.audio": "4.0.7"})
        _check_pyannote_version()  # no exception


class TestPipelineAndTokenResolution:
    def test_pipeline_id_precedence(self, monkeypatch) -> None:
        from taigi_asr.diarize import DEFAULT_PIPELINE_ID, PIPELINE_ENV_VAR, resolve_pipeline_id

        monkeypatch.delenv(PIPELINE_ENV_VAR, raising=False)
        assert (
            resolve_pipeline_id()
            == DEFAULT_PIPELINE_ID
            == ("pyannote/speaker-diarization-community-1")
        )
        monkeypatch.setenv(PIPELINE_ENV_VAR, " pyannote/speaker-diarization-3.1 ")
        assert resolve_pipeline_id() == "pyannote/speaker-diarization-3.1"
        assert resolve_pipeline_id("org/custom") == "org/custom"

    def test_instance_exposes_resolved_pipeline_id(self, monkeypatch) -> None:
        from taigi_asr.diarize import PIPELINE_ENV_VAR, DiarizationPipeline

        monkeypatch.setenv(PIPELINE_ENV_VAR, "pyannote/speaker-diarization-3.1")
        assert DiarizationPipeline(device="cpu", hf_token="t").PIPELINE_ID == (
            "pyannote/speaker-diarization-3.1"
        )

    def test_gated_repos(self) -> None:
        from taigi_asr.diarize import gated_repos

        assert gated_repos("pyannote/speaker-diarization-3.1") == (
            "pyannote/speaker-diarization-3.1",
            "pyannote/segmentation-3.0",
        )
        assert gated_repos("pyannote/speaker-diarization-community-1") == (
            "pyannote/speaker-diarization-community-1",
        )

    def test_token_precedence(self, monkeypatch) -> None:
        import huggingface_hub

        from taigi_asr.diarize import _resolve_hf_token

        monkeypatch.setattr(huggingface_hub, "get_token", lambda: "stored")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        assert _resolve_hf_token(None) == "stored"  # `hf auth login`
        monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "legacy")
        assert _resolve_hf_token(None) == "legacy"
        monkeypatch.setenv("HF_TOKEN", "env")
        assert _resolve_hf_token(None) == "env"
        assert _resolve_hf_token("explicit") == "explicit"

    def test_device_defaults_to_cpu_without_cuda(self, monkeypatch) -> None:
        import torch

        from taigi_asr.diarize import DiarizationPipeline

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert DiarizationPipeline(hf_token="t").device == "cpu"
        assert DiarizationPipeline(device="cuda", hf_token="t").device == "cuda"


class TestWaveformAndOutput:
    def test_load_waveform_reads_and_downmixes(self, tmp_path) -> None:
        np = pytest.importorskip("numpy")
        sf = pytest.importorskip("soundfile")

        from taigi_asr.diarize import _load_waveform

        stereo = np.stack([np.full(1600, 0.2), np.full(1600, 0.4)], axis=1).astype(np.float32)
        wav = tmp_path / "st.wav"
        sf.write(wav, stereo, 16000)
        audio = _load_waveform(wav)
        assert audio["sample_rate"] == 16000
        assert tuple(audio["waveform"].shape) == (1, 1600)
        assert float(audio["waveform"][0, 0]) == pytest.approx(0.3, abs=1e-4)

    def test_load_waveform_maps_unreadable_file(self, tmp_path) -> None:
        from taigi_asr.diarize import _load_waveform
        from taigi_asr.errors import TranscriptionError

        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a wav")
        with pytest.raises(TranscriptionError, match="cannot read audio"):
            _load_waveform(bad)

    def test_output_unwrapping_prefers_exclusive(self) -> None:
        from taigi_asr.diarize import _annotation_from_output

        class _EmptyAnnotation:
            def __bool__(self) -> bool:  # pyannote's Annotation is falsy when empty
                return False

        exclusive, regular = _EmptyAnnotation(), object()

        class _Out:
            exclusive_speaker_diarization = exclusive
            speaker_diarization = regular

        class _OnlyRegular:
            speaker_diarization = regular

        bare = object()
        assert _annotation_from_output(_Out()) is exclusive  # even when empty
        assert _annotation_from_output(_OnlyRegular()) is regular
        assert _annotation_from_output(bare) is bare


class TestSpeechbrainWinPatch:
    """`_patch_speechbrain_lazy_module` must:
    - be a no-op on POSIX (so upstream speechbrain fixes aren't masked)
    - be idempotent on win32 (re-call doesn't double-patch)
    - leave the `_taigi_patched` sentinel on the class

    We exercise the public surface only — no LazyModule object construction,
    no monkey of speechbrain internals beyond what the patcher itself touches.
    """

    def test_noop_on_posix(self, monkeypatch) -> None:
        from taigi_asr.diarize import _patch_speechbrain_lazy_module

        monkeypatch.setattr("sys.platform", "linux")
        # Should not raise even when speechbrain isn't importable in this
        # subprocess, because the early-return runs before the import.
        _patch_speechbrain_lazy_module()  # no exception

    def test_idempotent_on_win32(self, monkeypatch) -> None:
        try:
            from speechbrain.utils import importutils as sb_iu
        except ImportError:
            pytest.skip("speechbrain not installed in this environment")

        import importlib.metadata as md

        from taigi_asr.diarize import _patch_speechbrain_lazy_module

        monkeypatch.setattr("sys.platform", "win32")
        real_version = md.version
        monkeypatch.setattr(
            md, "version", lambda n: "1.0.3" if n == "speechbrain" else real_version(n)
        )
        # Restore the real method + sentinel after the test.
        monkeypatch.setattr(sb_iu.LazyModule, "ensure_module", sb_iu.LazyModule.ensure_module)
        monkeypatch.setattr(sb_iu.LazyModule, "_taigi_patched", False, raising=False)

        _patch_speechbrain_lazy_module()
        assert getattr(sb_iu.LazyModule, "_taigi_patched", False) is True
        first_method = sb_iu.LazyModule.ensure_module

        # Second call: must short-circuit on the sentinel, leaving the method
        # object the same identity (not re-wrapped).
        _patch_speechbrain_lazy_module()
        assert sb_iu.LazyModule.ensure_module is first_method

    def test_noop_when_speechbrain_already_fixed(self, monkeypatch) -> None:
        import importlib.metadata as md

        from taigi_asr.diarize import _patch_speechbrain_lazy_module

        monkeypatch.setattr("sys.platform", "win32")
        real_version = md.version
        monkeypatch.setattr(
            md, "version", lambda n: "1.1.2" if n == "speechbrain" else real_version(n)
        )
        # Would raise ImportError past the gate in envs without speechbrain;
        # returning cleanly proves the version gate short-circuits first.
        _patch_speechbrain_lazy_module()


@pytest.fixture
def fake_pyannote(monkeypatch):
    """Install a fake ``pyannote.audio`` and skip the installed-version check.

    Returns a dict that the fake ``Pipeline.from_pretrained`` records its
    calls into; set ``behaviour`` to an exception to raise, or ``return_none``
    to mimic HF auth failure modes where ``from_pretrained`` returns None.
    """
    import sys

    import taigi_asr.diarize as dia_mod

    state: dict = {"calls": [], "behaviour": None, "return_none": False, "moved_to": None}

    class _Loaded:
        def to(self, device):
            state["moved_to"] = str(device)
            return self

    class _FakePipelineModule:
        class Pipeline:
            @staticmethod
            def from_pretrained(*a, **kw):
                state["calls"].append((a, kw))
                if state["return_none"]:
                    return None
                b = state["behaviour"]
                if isinstance(b, BaseException):
                    raise b
                return _Loaded() if b is None else b

    monkeypatch.setitem(sys.modules, "pyannote.audio", _FakePipelineModule)
    monkeypatch.setattr(dia_mod, "_check_pyannote_version", lambda: None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return state


class TestDiarizationPipelineErrorMapping:
    """`DiarizationPipeline.load()` should surface license-acceptance errors
    with a targeted message — distinct from the generic 'failed to load'."""

    @pytest.mark.parametrize(
        "upstream_msg",
        [
            "Cannot access gated repo for url ...",
            "401 Client Error: Unauthorized for url",
            "403 Client Error: Forbidden",
        ],
    )
    def test_gated_repo_error_gets_license_hint(self, fake_pyannote, upstream_msg: str) -> None:
        from taigi_asr.diarize import DiarizationPipeline
        from taigi_asr.errors import ModelLoadError

        fake_pyannote["behaviour"] = RuntimeError(upstream_msg)
        d = DiarizationPipeline(
            device="cpu", hf_token="hf_dummy_token", pipeline_id="pyannote/speaker-diarization-3.1"
        )
        with pytest.raises(ModelLoadError, match="License acceptance required") as exc_info:
            d.load()
        # 3.1 gates two repos; both pages must be named.
        assert "huggingface.co/pyannote/segmentation-3.0" in str(exc_info.value)

    def test_load_passes_token_kwarg_and_moves_to_device(self, fake_pyannote) -> None:
        from taigi_asr.diarize import DEFAULT_PIPELINE_ID, DiarizationPipeline

        d = DiarizationPipeline(device="cpu", hf_token="hf_x")
        d.load()
        # pyannote 4.x: ``token=`` (``use_auth_token=`` is a TypeError there).
        assert fake_pyannote["calls"] == [((DEFAULT_PIPELINE_ID,), {"token": "hf_x"})]
        assert fake_pyannote["moved_to"] == "cpu"
        assert d.is_loaded() is True
        d.load()  # idempotent
        assert len(fake_pyannote["calls"]) == 1

    def test_offline_mode_loads_without_token(self, fake_pyannote, monkeypatch) -> None:
        import huggingface_hub

        from taigi_asr.diarize import DiarizationPipeline

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        d = DiarizationPipeline(device="cpu")
        d.load()
        assert fake_pyannote["calls"][0][1] == {"token": None}

    def test_run_builds_kwargs_and_handles_empty_itertracks(self, monkeypatch) -> None:
        """Cover the Python-testable parts of DiarizationPipeline.run() that
        are NOT pyannote-GPU-bound: the kw-builder (passes through num/min/max
        speakers), the ``self._pipeline is None`` guard, and the post-loop
        sort + return on an empty diarization."""
        from taigi_asr.diarize import DiarizationPipeline
        from taigi_asr.errors import TranscriptionError

        captured_kwargs: list[dict] = []

        class _EmptyDiarization:
            def itertracks(self, yield_label=True):
                return iter([])

        def _fake_pipeline_call(audio, **kw):
            assert audio == {"waveform": "W", "sample_rate": 16000}
            captured_kwargs.append(kw)
            return _EmptyDiarization()

        import taigi_asr.diarize as dia_mod

        monkeypatch.setattr(
            dia_mod, "_load_waveform", lambda p: {"waveform": "W", "sample_rate": 16000}
        )
        d = DiarizationPipeline(device="cpu", hf_token="dummy")
        # Bypass load() — directly install a callable as the underlying
        # pipeline. This lets us hit run()'s pure-Python branches without
        # spinning up pyannote.
        d._pipeline = _fake_pipeline_call  # type: ignore[assignment]
        d._loaded = True

        # No kwargs — kw dict stays empty.
        assert d.run("/tmp/fake.wav") == []
        assert captured_kwargs[-1] == {}

        # All three flags — each branch of the kw-builder.
        d.run("/tmp/fake.wav", num_speakers=3, min_speakers=2, max_speakers=5)
        assert captured_kwargs[-1] == {
            "num_speakers": 3,
            "min_speakers": 2,
            "max_speakers": 5,
        }

        # Unloaded-during-run guard: simulate concurrent unload by setting
        # ``_pipeline = None`` after the load() bypass. ``run()`` must raise
        # TranscriptionError instead of crashing on ``None(...)`` call.
        d._pipeline = None
        with pytest.raises(TranscriptionError, match="pipeline unloaded"):
            d.run("/tmp/fake.wav")

    def test_is_loaded_false_before_load(self) -> None:
        from taigi_asr.diarize import DiarizationPipeline

        d = DiarizationPipeline(device="cpu", hf_token="dummy")
        assert d.is_loaded() is False

    def test_run_unwraps_diarize_output_and_sorts(self, monkeypatch) -> None:
        import taigi_asr.diarize as dia_mod
        from taigi_asr.diarize import DiarizationPipeline, SpeakerTurn

        class _Seg:
            def __init__(self, s, e):
                self.start, self.end = s, e

        class _Ann:
            def __init__(self, rows):
                self.rows = rows

            def itertracks(self, yield_label=True):
                return iter([(_Seg(s, e), "t", spk) for s, e, spk in self.rows])

        class _DiarizeOutput:  # pyannote 4.x return type
            speaker_diarization = _Ann([(0.0, 9.0, "WRONG")])
            exclusive_speaker_diarization = _Ann(
                [(2.0, 3.0, "SPEAKER_01"), (0.0, 2.0, "SPEAKER_00")]
            )

        monkeypatch.setattr(dia_mod, "_load_waveform", lambda p: {})
        d = DiarizationPipeline(device="cpu", hf_token="dummy")
        d._pipeline = lambda audio, **kw: _DiarizeOutput()  # type: ignore[assignment]
        d._loaded = True
        assert d.run("x.wav") == [
            SpeakerTurn(0.0, 2.0, "SPEAKER_00"),
            SpeakerTurn(2.0, 3.0, "SPEAKER_01"),
        ]

    def test_load_raises_when_pipeline_from_pretrained_returns_none(self, fake_pyannote) -> None:
        """Some HF auth failure modes (notably partial license acceptance) make
        ``Pipeline.from_pretrained`` return ``None`` rather than raise. The
        ``if pipeline is None: raise ModelLoadError(...)`` branch in
        ``DiarizationPipeline.load()`` exists for exactly this case; cover it
        by mocking the upstream call."""
        from taigi_asr.diarize import DiarizationPipeline
        from taigi_asr.errors import ModelLoadError

        fake_pyannote["return_none"] = True
        d = DiarizationPipeline(device="cpu", hf_token="dummy_token")
        with pytest.raises(ModelLoadError, match="Failed to load"):
            d.load()
        # Ensure the load() guard kept the pipeline unset on this failure path.
        assert d.is_loaded() is False

    def test_load_raises_when_no_hf_token(self, monkeypatch) -> None:
        import huggingface_hub

        from taigi_asr.diarize import DiarizationPipeline
        from taigi_asr.errors import ModelLoadError

        # Strip both env vars and the `hf auth login` token so the
        # constructor's resolution chain finds nothing.
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)

        d = DiarizationPipeline(device="cpu", hf_token=None)
        with pytest.raises(ModelLoadError, match="HF_TOKEN env var required"):
            d.load()

    def test_local_pipeline_path_loads_without_token(
        self, fake_pyannote, monkeypatch, tmp_path
    ) -> None:
        import huggingface_hub

        from taigi_asr.diarize import DiarizationPipeline

        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
        config = tmp_path / "config.yaml"
        config.write_text("pipeline: {}\n", encoding="utf-8")
        DiarizationPipeline(device="cpu", pipeline_id=str(config)).load()
        assert fake_pyannote["calls"] == [((str(config),), {"token": None})]

    def test_other_errors_use_generic_message(self, fake_pyannote) -> None:
        from taigi_asr.diarize import DiarizationPipeline
        from taigi_asr.errors import ModelLoadError

        fake_pyannote["behaviour"] = RuntimeError("disk full")
        d = DiarizationPipeline(device="cpu", hf_token="hf_dummy_token")
        with pytest.raises(ModelLoadError, match="Failed to load") as exc_info:
            d.load()
        # The targeted-license branch must NOT swallow unrelated errors.
        assert "License acceptance required" not in str(exc_info.value)
