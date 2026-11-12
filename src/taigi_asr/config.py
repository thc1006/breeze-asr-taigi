"""Project-wide constants and environment-variable overrides."""

from __future__ import annotations

import os
from pathlib import Path

# Model IDs — fixed, not user-overridable. Both point at MediaTek Breeze-ASR-26.
FASTER_WHISPER_MODEL_ID = "paulpengtw/faster-whisper-Breeze-ASR-26"
HUGGINGFACE_MODEL_ID = "MediaTek-Research/Breeze-ASR-26"

# HuggingFace cache: respect HF_HOME if user already configured one.
HF_CACHE_DIR: Path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

# Audio
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

# Audio/video container extensions the ``--input-dir`` glob auto-picks up.
# This is a UX whitelist for the directory-scan flow only — anything ffmpeg
# can decode still works when passed positionally on the command line.
# Lowercase by convention; matching against ``Path.suffix.lower()``.
SUPPORTED_AUDIO_EXTS: frozenset[str] = frozenset(
    {
        ".mp3",
        ".m4a",
        ".wav",
        ".flac",
        ".ogg",
        ".webm",
        ".mp4",
        ".mkv",
        ".mov",
        ".aac",
        ".opus",
        ".wma",
    }
)

# Inference defaults
DEFAULT_BEAM_SIZE = 5
# Practical ceiling for RTX 3050 4 GB + int8_float16 model (~2.9 GB). Beam widths
# above this eat the remaining VRAM and risk OOM.
MAX_BEAM_SIZE_4GB = 12
DEFAULT_BEST_OF = 5
DEFAULT_PATIENCE = 1.0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_VAD_MIN_SILENCE_MS = 500
DEFAULT_VAD_SPEECH_PAD_MS = 200
# Line splitting from word timings (faster-whisper, ``word_timestamps=True``).
# faster-whisper's own segments run 25 s+ on meeting audio -- long enough to
# span several speakers, so overlap-based diarization labels the whole line
# as whoever talked most. Start a new line at a pause and cap the line span.
# Word gaps only appear where VAD removed silence, so with the VAD settings
# above a 0.4 s gap corresponds to roughly 0.8 s of actual silence.
DEFAULT_WORD_SPLIT_GAP_S = 0.4
DEFAULT_WORD_SPLIT_MAX_S = 8.0
DEFAULT_LANGUAGE = "zh"  # Breeze-ASR-26 fine-tunes Whisper on Taiwanese Hokkien
# but uses Mandarin token vocabulary.
DEFAULT_CHUNK_LENGTH_S = 30
DEFAULT_STRIDE_LENGTH_S = (4, 2)
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_MAX_NEW_TOKENS = 440

# UI
GRADIO_HOST = os.environ.get("TAIGI_ASR_HOST", "127.0.0.1")


def _parse_port(raw: str | None, default: int = 7860) -> int:
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        import logging

        logging.getLogger(__name__).warning(
            "TAIGI_ASR_PORT=%r is not a valid integer, falling back to %d", raw, default
        )
        return default


GRADIO_PORT = _parse_port(os.environ.get("TAIGI_ASR_PORT"))
