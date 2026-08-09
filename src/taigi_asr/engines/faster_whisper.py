"""Faster-Whisper engine — the default path for RTX 3050 4GB.

Uses the community pre-converted CTranslate2 build of MediaTek Breeze-ASR-26:
``paulpengtw/faster-whisper-Breeze-ASR-26``. With int8_float16 and batch_size=4
the peak VRAM stays around 2.9 GB, leaving headroom on 4 GB laptop GPUs.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path

# Low-VRAM allocator hint — same rationale as HuggingFaceEngine.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from taigi_asr.config import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_BEST_OF,
    DEFAULT_LANGUAGE,
    DEFAULT_PATIENCE,
    DEFAULT_TEMPERATURE,
    DEFAULT_VAD_MIN_SILENCE_MS,
    DEFAULT_VAD_SPEECH_PAD_MS,
    FASTER_WHISPER_MODEL_ID,
)
from taigi_asr.errors import ModelLoadError, TranscriptionError
from taigi_asr.segments import TimestampedSegment

log = logging.getLogger(__name__)

# Word-grouping thresholds, used only when ``word_timestamps=True``.
# faster-whisper's own segments run 25 s+ on meeting audio — long enough to
# span several speakers, which makes downstream diarization label the whole
# line as whoever happened to talk most. Splitting on natural pauses (and
# capping the span) keeps each line inside a single speaker's turn.
WORD_SPLIT_GAP_S = 0.4
WORD_SPLIT_MAX_S = 8.0


def _group_words(words, max_gap: float, max_span: float) -> list[TimestampedSegment]:
    """Collapse per-word timings into short, speaker-alignable lines."""
    lines: list[TimestampedSegment] = []
    buf: list = []

    def emit(chunk: list) -> None:
        text = "".join(w.word for w in chunk).strip()
        if text:
            lines.append(TimestampedSegment(float(chunk[0].start), float(chunk[-1].end), text))

    def split_overlong() -> None:
        """Cut at the widest internal pause instead of mid-phrase at the cap.

        Hard-cutting the moment the span limit is hit lands in the middle of a
        word ("今天的會" / "議內容"), which reads badly in the SRT. The widest
        pause inside the buffer is the least-bad boundary available.
        """
        nonlocal buf
        if len(buf) < 2:
            emit(buf)
            buf = []
            return
        _, idx = max(
            (float(b.start) - float(a.end), i)
            for i, (a, b) in enumerate(zip(buf, buf[1:], strict=False))
        )
        emit(buf[: idx + 1])
        buf = buf[idx + 1 :]

    for w in words:
        if w.start is None or w.end is None:
            continue
        if buf and float(w.start) - float(buf[-1].end) >= max_gap:
            emit(buf)
            buf = []
        buf.append(w)
        while buf and float(buf[-1].end) - float(buf[0].start) >= max_span:
            split_overlong()
    if buf:
        emit(buf)
    return lines


class FasterWhisperEngine:
    """Wraps faster-whisper's :class:`BatchedInferencePipeline`."""

    MODEL_ID: str = FASTER_WHISPER_MODEL_ID  # fixed, not user-overridable

    def __init__(
        self,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        batch_size: int = 4,
        cpu_threads: int | None = None,
        *,
        beam_size: int = DEFAULT_BEAM_SIZE,
        best_of: int = DEFAULT_BEST_OF,
        patience: float = DEFAULT_PATIENCE,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.cpu_threads = cpu_threads or (os.cpu_count() or 4)
        # Decoder defaults applied by every transcribe() call; each call can
        # still override via kwargs.
        self.beam_size = beam_size
        self.best_of = best_of
        self.patience = patience
        self.temperature = temperature
        self._model = None
        self._batched = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                from faster_whisper import BatchedInferencePipeline, WhisperModel
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError(
                    "faster-whisper not installed. Run: pip install faster-whisper"
                ) from exc

            try:
                log.info(
                    "Loading %s on %s (compute_type=%s)…",
                    self.MODEL_ID,
                    self.device,
                    self.compute_type,
                )
                self._model = WhisperModel(
                    self.MODEL_ID,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                )
                self._batched = BatchedInferencePipeline(model=self._model)
                self._prewarm()
                self._loaded = True
                log.info("Faster-Whisper engine loaded.")
            except Exception as exc:
                raise ModelLoadError(f"Failed to load faster-whisper model: {exc}") from exc

    def _prewarm(self) -> None:
        """Force first-call CUDA / cuDNN / CTranslate2 kernel JIT to amortize.

        The first ``transcribe()`` on a fresh CUDA context spends 1-2 s
        compiling kernels and probing cuDNN heuristics. Running a 1 s dummy
        clip here shifts that cost into ``load()`` so the first real file
        already sees steady-state throughput. We use ``self.batch_size`` so
        the kernels JIT'd here are exactly the ones the real workload will
        hit (CT2 caches kernels per ``(compute_type, batch_size)`` shape).
        ``vad_filter=False`` keeps the silent dummy in the pipeline instead
        of being VAD-stripped to zero work.
        """
        if self.device == "cpu":
            return
        try:
            import numpy as np

            dummy = np.zeros(16000, dtype=np.float32)
            seg_iter, _ = self._batched.transcribe(
                dummy,
                batch_size=self.batch_size,
                language=DEFAULT_LANGUAGE,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=False,
                word_timestamps=False,
                without_timestamps=True,
            )
            # Generator is lazy — must drain to actually run the kernels.
            for _ in seg_iter:
                pass
            log.debug("Faster-Whisper pre-warm complete.")
        except Exception as exc:
            # Pre-warm failure must never abort load(). Surface as warning
            # (not debug) so the operator can correlate it with a slower
            # first transcribe — but the engine still loads and works.
            log.warning(
                "Pre-warm failed (%s); first transcribe will pay JIT cost.",
                exc,
            )

    def is_loaded(self) -> bool:
        return self._loaded

    def transcribe(
        self,
        wav_path: str | Path,
        *,
        word_timestamps: bool = False,
        language: str = DEFAULT_LANGUAGE,
        beam_size: int | None = None,
        best_of: int | None = None,
        patience: float | None = None,
        temperature: float | None = None,
        vad_filter: bool = True,
    ) -> list[TimestampedSegment]:
        # None -> use the per-engine defaults set at construction time.
        beam_size = beam_size if beam_size is not None else self.beam_size
        best_of = best_of if best_of is not None else self.best_of
        patience = patience if patience is not None else self.patience
        temperature = temperature if temperature is not None else self.temperature
        # Under the lock so a concurrent caller can't race past the loaded check.
        with self._lock:
            if not self._loaded:
                pass  # will load below, outside the lock, to avoid deadlock
        if not self._loaded:
            self.load()
        assert self._batched is not None

        # faster-whisper returns a *lazy* generator — decoding happens while the
        # caller iterates. Keep the consumption loop inside the try so OOM /
        # CTranslate2 errors during decode are caught and re-wrapped instead of
        # bubbling up as raw exceptions.
        try:
            segments_iter, _info = self._batched.transcribe(
                str(wav_path),
                batch_size=self.batch_size,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                temperature=temperature,
                vad_filter=vad_filter,
                vad_parameters={
                    "min_silence_duration_ms": DEFAULT_VAD_MIN_SILENCE_MS,
                    "speech_pad_ms": DEFAULT_VAD_SPEECH_PAD_MS,
                },
                word_timestamps=word_timestamps,
                without_timestamps=False,
            )
            out: list[TimestampedSegment] = []
            for seg in segments_iter:
                text = (seg.text or "").strip()
                if not text:
                    continue
                # With word_timestamps the model has already computed per-word
                # timings; use them instead of throwing them away and emitting
                # one 25-second line.
                words = getattr(seg, "words", None) if word_timestamps else None
                if words:
                    grouped = _group_words(words, WORD_SPLIT_GAP_S, WORD_SPLIT_MAX_S)
                    if grouped:
                        out.extend(grouped)
                        continue
                start = float(seg.start) if seg.start is not None else 0.0
                end = float(seg.end) if seg.end is not None else start + 1.0
                out.append(TimestampedSegment(start_time=start, end_time=end, text=text))
            return out
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper inference failed: {exc}") from exc

    def unload(self) -> None:
        """Release the CTranslate2 session and free VRAM."""
        with self._lock:
            if self._batched is not None:
                del self._batched
                self._batched = None
            if self._model is not None:
                del self._model
                self._model = None
            self._loaded = False
            try:
                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    @classmethod
    def preload(cls) -> None:
        """Convenience hook for install scripts: snapshot the CT2 model
        into the HuggingFace cache so the first real run isn't blocked by a
        ~2.9 GB download."""
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(cls.MODEL_ID)
            log.info("Faster-Whisper Breeze-ASR-26 model cached.")
        except Exception as exc:  # pragma: no cover
            log.warning("Preload failed (will download on first run): %s", exc)
