"""Smoke tests for the --diarize CLI flag.

Exercises argparse wiring and pre-flight validation only (no actual inference).
Real end-to-end --diarize is validated by ``scripts/diarize_compare.py`` against
a known audio file.

Also covers the dia.load() failure fallback (CLI must still write the
un-attributed ASR transcripts so users don't lose the GPU time already spent
on the ASR pass) — via in-process monkeypatching.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from taigi_asr.errors import ModelLoadError
from taigi_asr.segments import TimestampedSegment


def _touch_wav(tmp_path: Path) -> Path:
    """Create a 0-byte .wav so the existence check passes and we exercise the
    later flag-validation branches without spinning up any model."""
    p = tmp_path / "fake.wav"
    p.write_bytes(b"")
    return p


def test_diarize_help_flag_present() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "taigi_asr.cli", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 0
    assert "--diarize" in result.stdout
    assert "--num-speakers" in result.stdout
    assert "--min-speakers" in result.stdout
    assert "--max-speakers" in result.stdout


def test_diarize_rejects_num_with_min(tmp_path) -> None:
    """--num-speakers is mutually exclusive with --min-speakers."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "taigi_asr.cli",
            str(_touch_wav(tmp_path)),
            "--diarize",
            "--num-speakers",
            "3",
            "--min-speakers",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 6
    assert "mutually exclusive" in result.stderr


def test_diarize_rejects_min_greater_than_max(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "taigi_asr.cli",
            str(_touch_wav(tmp_path)),
            "--diarize",
            "--min-speakers",
            "5",
            "--max-speakers",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    assert result.returncode == 6
    assert "--min-speakers (5) > --max-speakers (2)" in result.stderr


class _FailingDiarizationPipeline:
    """Drop-in replacement for ``taigi_asr.diarize.DiarizationPipeline`` whose
    ``load()`` raises ``ModelLoadError``. Used to force the dia-load-failure
    fallback branch in cli.main() without ever touching pyannote / CUDA."""

    PIPELINE_ID = "pyannote/speaker-diarization-3.1"

    def __init__(self, *a, **kw) -> None:  # noqa: D401, ARG002
        pass

    def load(self) -> None:
        raise ModelLoadError("simulated dia load failure for test")

    def is_loaded(self) -> bool:  # pragma: no cover - never reached
        return False

    def run(self, *a, **kw):  # pragma: no cover - never reached
        raise AssertionError("run() must not be called when load() failed")

    def unload(self) -> None:
        # cli.main() always calls dia.unload() in the outer finally; must be a
        # no-op rather than raise so it doesn't mask the load failure.
        pass


def test_diarize_load_failure_falls_back_to_unattributed_asr(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """When dia.load() fails, the CLI must still write the ASR transcripts
    (without speaker labels) and exit 4 so users don't lose ASR GPU time."""
    import taigi_asr.cli as cli_mod
    import taigi_asr.diarize as dia_mod
    import taigi_asr.engines.fake as fake_mod
    from taigi_asr.router import GPUInfo

    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00" * 32)

    # Stand-in 16k WAV (existence is all that matters — FakeEngine ignores it).
    fake_wav = tmp_path / "_fake_16k.wav"
    fake_wav.write_bytes(b"\x00" * 32)

    def fake_convert(src, out_dir=None):
        return fake_wav, 1.0

    monkeypatch.setattr(cli_mod.AudioConverter, "convert", staticmethod(fake_convert))
    monkeypatch.setattr(
        cli_mod.AudioConverter, "cleanup", staticmethod(lambda p: None)
    )

    monkeypatch.setattr(
        cli_mod.GPUProfiler,
        "detect",
        staticmethod(
            lambda: GPUInfo(
                name="FakeGPU", vram_gb=4.0, cuda_available=True, bf16_supported=False
            )
        ),
    )

    fake_engine = fake_mod.FakeEngine(
        script=[TimestampedSegment(start_time=0.0, end_time=1.0, text="台語句子")]
    )
    monkeypatch.setattr(cli_mod, "build_engine", lambda spec: fake_engine)

    # Patch DiarizationPipeline on the module the CLI's local import resolves
    # against. cli does ``from taigi_asr.diarize import DiarizationPipeline``
    # inside main(), which performs a module attribute lookup at call time —
    # so replacing the attribute here is observed by that import.
    monkeypatch.setattr(dia_mod, "DiarizationPipeline", _FailingDiarizationPipeline)

    out_path = tmp_path / "clip.srt"
    rc = cli_mod.main([str(audio), "--diarize", "--engine", "fw", "--format", "srt"])

    captured = capsys.readouterr()

    assert rc == 4, captured.err
    assert "diarize load failed" in captured.err
    assert "falling back" in captured.err
    # The un-attributed SRT must be on disk so the user doesn't have to re-ASR.
    assert out_path.exists(), captured.err
    content = out_path.read_text(encoding="utf-8")
    assert "台語句子" in content
    # No speaker prefix in the fallback output (proves the path is the
    # un-attributed branch, not the happy path).
    assert "[SPEAKER_" not in content


def test_has_any_speaker_flag_predicate() -> None:
    """The warning-without-diarize branch is gated by ``_has_any_speaker_flag``;
    exercise that predicate directly so the test doesn't need to run main()
    (which would drag past engine load and timeout)."""
    from taigi_asr.cli import _build_parser, _has_any_speaker_flag

    parser = _build_parser()

    # No speaker flag → no warning.
    assert _has_any_speaker_flag(parser.parse_args(["fake.wav"])) is False
    # Any of num/min/max → warning eligible.
    for flag, value in [
        ("--num-speakers", "3"),
        ("--min-speakers", "2"),
        ("--max-speakers", "4"),
    ]:
        args = parser.parse_args(["fake.wav", flag, value])
        assert _has_any_speaker_flag(args) is True, flag
