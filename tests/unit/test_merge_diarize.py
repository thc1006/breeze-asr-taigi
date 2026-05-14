"""Tests for ``scripts/merge_diarize.py``.

Focuses on the path-handling fixes that were added during P1/P2 review:
- ``Path.with_suffix`` collision (would have silently overwritten input SRT)
- Explicit clobber refusal when ``--out-prefix`` resolves to the input path
- UTF-8 BOM tolerance via ``utf-8-sig``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "merge_diarize.py"


@pytest.fixture(scope="module")
def merge_module():
    """Load ``scripts/merge_diarize.py`` as a module — it lives outside the
    package, so we go through importlib rather than rely on PYTHONPATH magic."""
    spec = importlib.util.spec_from_file_location("merge_diarize", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["merge_diarize"] = module
    spec.loader.exec_module(module)
    return module


def _write_srt(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_rttm(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


SAMPLE_SRT = (
    "1\n00:00:00,000 --> 00:00:03,000\n你好嗎\n\n2\n00:00:03,500 --> 00:00:06,000\n我很好\n"
)

SAMPLE_RTTM = (
    "SPEAKER demo 1 0.000 3.000 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
    "SPEAKER demo 1 3.500 2.500 <NA> <NA> SPEAKER_01 <NA> <NA>\n"
)


class TestMergeDiarizeOutputPaths:
    def test_default_prefix_writes_diarized_sidecars(self, tmp_path: Path, merge_module) -> None:
        srt = tmp_path / "audio.srt"
        rttm = tmp_path / "audio.rttm"
        _write_srt(srt, SAMPLE_SRT)
        _write_rttm(rttm, SAMPLE_RTTM)

        rc = merge_module.main.__wrapped__ if hasattr(merge_module.main, "__wrapped__") else None
        # The script's main() reads sys.argv, so monkeypatch via argv:
        argv_backup = sys.argv[:]
        sys.argv = ["merge_diarize", str(srt), str(rttm)]
        try:
            rc = merge_module.main()
        finally:
            sys.argv = argv_backup

        assert rc == 0
        # Outputs land at .diarized.{srt,txt,json}; input SRT is intact.
        assert (tmp_path / "audio.diarized.srt").exists()
        assert (tmp_path / "audio.diarized.txt").exists()
        assert (tmp_path / "audio.diarized.json").exists()
        assert srt.read_text(encoding="utf-8") == SAMPLE_SRT  # untouched
        assert "[SPEAKER_00]" in (tmp_path / "audio.diarized.srt").read_text(encoding="utf-8")

    def test_clobber_refused_when_out_prefix_collides_with_input(
        self, tmp_path: Path, merge_module
    ) -> None:
        """If --out-prefix resolves so that the eventual .srt output equals
        the input SRT, the script must refuse to write (return 4)."""
        srt = tmp_path / "audio.srt"
        rttm = tmp_path / "audio.rttm"
        _write_srt(srt, SAMPLE_SRT)
        _write_rttm(rttm, SAMPLE_RTTM)

        # --out-prefix without extension; the script appends ".srt" → equals input.
        bad_prefix = tmp_path / "audio"

        argv_backup = sys.argv[:]
        sys.argv = ["merge_diarize", str(srt), str(rttm), "--out-prefix", str(bad_prefix)]
        try:
            rc = merge_module.main()
        finally:
            sys.argv = argv_backup

        assert rc == 4
        # Input SRT must be unchanged.
        assert srt.read_text(encoding="utf-8") == SAMPLE_SRT

    def test_utf8_bom_in_srt_is_tolerated(self, tmp_path: Path, merge_module) -> None:
        """A UTF-8 BOM at SRT start must not silently drop cue #1."""
        srt = tmp_path / "audio.srt"
        rttm = tmp_path / "audio.rttm"
        # Write BOM + content; merge_diarize uses utf-8-sig to strip it.
        srt.write_bytes(b"\xef\xbb\xbf" + SAMPLE_SRT.encode("utf-8"))
        _write_rttm(rttm, SAMPLE_RTTM)

        argv_backup = sys.argv[:]
        sys.argv = ["merge_diarize", str(srt), str(rttm)]
        try:
            rc = merge_module.main()
        finally:
            sys.argv = argv_backup

        assert rc == 0
        out = (tmp_path / "audio.diarized.srt").read_text(encoding="utf-8")
        assert "你好嗎" in out  # cue #1 survived the BOM
        assert "我很好" in out  # cue #2 too
