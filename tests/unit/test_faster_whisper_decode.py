"""Regression guard: faster-whisper must be able to decode audio with the
installed PyAV.

faster-whisper <=1.2.1 calls ``av.open(..., metadata_errors="ignore")``; PyAV
19 removed that kwarg, so every file-path ``transcribe()`` raised
``TypeError`` on Python >=3.12 (the only interpreters PyAV 19 ships for).
The real-model integration tests are ``slow``-marked and skipped in CI, so
nothing else exercises this decode path. No GPU, model or system ffmpeg
needed: PyAV bundles its own FFmpeg libraries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

decode_audio = pytest.importorskip("faster_whisper").decode_audio


def test_decode_audio_reads_wav_with_installed_pyav(tmp_path: Path) -> None:
    sr = 16000
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, tone, sr)

    decoded = decode_audio(str(wav), sampling_rate=sr)

    assert decoded.dtype == np.float32
    assert abs(len(decoded) - sr) <= sr // 100  # resampler may add/drop a few samples
