"""Speaker diarization via pyannote/speaker-diarization-3.1.

Wraps the pyannote pipeline with the same load/run/unload contract as the ASR
engines so the CLI can swap models cleanly on a 4 GB VRAM budget. Also exposes
``attribute_speakers`` (overlap-based speaker assignment) and ``parse_rttm`` /
``turns_to_rttm`` so the same code path serves both the integrated CLI and the
standalone ``scripts/merge_diarize.py`` / ``scripts/diarize_compare.py`` tools.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from taigi_asr.errors import ModelLoadError, TranscriptionError
from taigi_asr.segments import TimestampedSegment


# Serializes the global ``torch.load`` patch so two concurrent
# DiarizationPipeline.load() calls don't capture each other's patched
# version as the "original" and leak the patch on restore.
_torch_load_patch_lock = threading.Lock()


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """Temporarily force ``weights_only=False`` on ``torch.load``.

    PyTorch 2.6 flipped the default to ``True``, which breaks pyannote 3.4's
    Lightning checkpoint (contains non-tensor globals like ``TorchVersion``).
    We patch only for the duration of ``Pipeline.from_pretrained`` so other
    libraries (including faster-whisper's own checkpoint code) keep the safer
    default. Restoring in ``finally`` guarantees no leak even if pyannote
    raises mid-load. The module-level lock makes the patch safe across
    concurrent loaders (the typical CLI use is single-threaded, but library
    consumers may parallelize).
    """
    with _torch_load_patch_lock:
        original = torch.load

        def _patched(*args, **kwargs):
            kwargs["weights_only"] = False
            return original(*args, **kwargs)

        torch.load = _patched  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.load = original  # type: ignore[assignment]


# RTTM is space-delimited; URIs and speaker labels with whitespace would corrupt
# the file. Pyannote's own labels (SPEAKER_NN) are safe; user-supplied via
# external RTTM is the threat model.
#
# Trust boundary: we accept any non-whitespace string so non-ASCII filenames
# (e.g. ``5月6日 13-23.m4a`` → URI ``5月6日_13-23``) still round-trip. Callers
# passing untrusted strings as URI/speaker labels are responsible for their own
# allow-listing (e.g. shell-safe characters). This is fine for the project's
# threat model — URIs are always derived from local audio file names.
_RTTM_SAFE_RE = re.compile(r"^\S+$")


def _patch_speechbrain_lazy_module() -> None:
    """Windows-only fix for speechbrain's introspection-bypass check.

    Replaces ``ensure_module`` with a version that recognizes both ``/`` and
    ``\\`` as path separators when probing ``importer_frame.filename`` for
    ``inspect.py``. Gated on ``sys.platform == "win32"`` so POSIX runs pick
    up any future upstream speechbrain bug fix without our patch masking it.
    The ``_taigi_patched`` sentinel keys on class identity (no version
    check); long-lived Jupyter kernels that upgrade speechbrain mid-session
    should restart to pick up the new method.
    """
    if sys.platform != "win32":
        return
    try:
        from speechbrain.utils import importutils as _sb_iu
    except ImportError:  # pragma: no cover
        return
    if getattr(_sb_iu.LazyModule, "_taigi_patched", False):
        return
    import importlib as _importlib
    import inspect as _inspect
    import warnings as _warnings

    def patched_ensure_module(self, stacklevel: int):
        importer_frame = None
        try:
            importer_frame = _inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except (AttributeError, ValueError):  # pragma: no cover
            _warnings.warn(
                "Failed to inspect frame for speechbrain lazy import bypass."
            )
        if importer_frame is not None and importer_frame.filename.endswith(
            ("/inspect.py", "\\inspect.py")
        ):
            raise AttributeError()
        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = _importlib.import_module(self.target)
                else:
                    self.lazy_module = _importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as exc:  # pragma: no cover
                raise ImportError(
                    f"Lazy import of {repr(self)} failed"
                ) from exc
        return self.lazy_module

    _sb_iu.LazyModule.ensure_module = patched_ensure_module
    _sb_iu.LazyModule._taigi_patched = True

log = logging.getLogger(__name__)

PYANNOTE_PIPELINE_ID = "pyannote/speaker-diarization-3.1"


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """A contiguous interval attributed to a single speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiarizationPipeline:
    """pyannote/speaker-diarization-3.1 wrapper.

    Mirrors the ASR engine load/unload protocol so the CLI can sequence
    ``engine.unload()`` -> ``diarize.load()`` and stay under 4 GB VRAM.
    """

    PIPELINE_ID = PYANNOTE_PIPELINE_ID

    def __init__(self, device: str = "cuda", hf_token: str | None = None) -> None:
        self.device = device
        self.hf_token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )
        self._pipeline = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            if not self.hf_token:
                raise ModelLoadError(
                    "HF_TOKEN env var required for pyannote diarization. "
                    "Create a token at https://hf.co/settings/tokens and accept "
                    "the license for pyannote/speaker-diarization-3.1 + "
                    "pyannote/segmentation-3.0."
                )
            # speechbrain 1.0+ has a Windows path bug in
            # ``speechbrain.utils.importutils.LazyModule.ensure_module``: it
            # checks ``importer_frame.filename.endswith("/inspect.py")`` to
            # bypass lazy loads triggered by introspection (PyTorch op
            # registry, hasattr probes, etc.). On Windows the file path is
            # ``...\\inspect.py`` so the check never matches, and the lazy
            # import actually runs — which fails for integrations that
            # depend on k2 / unavailable packages, raising an opaque
            # ``Lazy import of LazyModule(...) failed``. We patch the method
            # in place to accept the Windows separator too. Idempotent via
            # the ``_taigi_patched`` sentinel.
            _patch_speechbrain_lazy_module()

            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError(
                    "pyannote.audio not installed. Run: pip install 'pyannote.audio<4'"
                ) from exc

            try:
                with _torch_load_weights_only_false():
                    pipeline = Pipeline.from_pretrained(
                        self.PIPELINE_ID, use_auth_token=self.hf_token
                    )
                if pipeline is None:
                    raise ModelLoadError(
                        f"Failed to load {self.PIPELINE_ID} — verify HF_TOKEN "
                        "and that you accepted the license for both "
                        "pyannote/speaker-diarization-3.1 and "
                        "pyannote/segmentation-3.0 at https://hf.co/."
                    )
                pipeline.to(torch.device(self.device))
                self._pipeline = pipeline
                self._loaded = True
                log.info("pyannote diarization pipeline loaded on %s", self.device)
            except ModelLoadError:
                raise
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load {self.PIPELINE_ID}: {exc}"
                ) from exc

    def is_loaded(self) -> bool:
        return self._loaded

    def run(
        self,
        wav_path: str | Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[SpeakerTurn]:
        # ``load()`` is idempotent and takes its own ``_lock``; call it
        # outside our lock so we don't deadlock on the non-reentrant
        # threading.Lock. After load, hold the lock for the inference call so
        # a concurrent ``unload()`` can't ``del`` the pipeline mid-flight.
        if not self._loaded:
            self.load()
        with self._lock:
            if self._pipeline is None:
                raise TranscriptionError(
                    "pipeline unloaded before run() could acquire it"
                )

            kw: dict = {}
            if num_speakers is not None:
                kw["num_speakers"] = num_speakers
            if min_speakers is not None:
                kw["min_speakers"] = min_speakers
            if max_speakers is not None:
                kw["max_speakers"] = max_speakers

            try:
                diarization = self._pipeline(str(wav_path), **kw)
            except Exception as exc:
                raise TranscriptionError(f"diarization failed: {exc}") from exc

            turns: list[SpeakerTurn] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(
                        start=float(turn.start),
                        end=float(turn.end),
                        speaker=str(speaker),
                    )
                )
            turns.sort(key=lambda t: t.start)
            return turns

    def unload(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                # pyannote's SpeakerDiarization pipeline keeps strong refs to
                # the segmentation + embedding sub-models via Inference
                # wrappers. ``del`` alone leaves ~600-900 MB resident on a
                # 4 GB GPU because cyclic refs from Inference -> Model ->
                # Inference's hooks survive a single GC pass. Null the leaf
                # nn.Modules first, then drop the pipeline reference, then
                # double-collect to break cycles. Wrapped in try/except so a
                # pyannote internal rename doesn't brick ``unload()``.
                try:
                    for attr_name in ("_segmentation", "_embedding"):
                        sub = getattr(self._pipeline, attr_name, None)
                        if sub is None:
                            continue
                        for inner in ("model", "model_"):
                            if hasattr(sub, inner):
                                setattr(sub, inner, None)
                except Exception as exc:  # pragma: no cover
                    log.debug("pyannote sub-model null-out skipped: %s", exc)
                del self._pipeline
                self._pipeline = None
            self._loaded = False
            gc.collect()
            gc.collect()  # second pass to break cycles missed on the first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def attribute_speakers(
    segments: list[TimestampedSegment],
    turns: list[SpeakerTurn],
) -> list[TimestampedSegment]:
    """For each ASR segment, find the diarization speaker with the largest
    time-overlap and attach it.

    Segments with no overlap (silence-only, or audio outside diarization window)
    keep ``speaker=None`` so callers can render them as UNKNOWN or fall back.

    Assumes ``turns`` is sorted by ``start`` — the public API in this module
    guarantees that; callers using their own RTTM should sort first.
    """
    if not turns:
        return list(segments)

    # Defensive re-sort: the public API in this module guarantees sorted turns,
    # but ``attribute_speakers`` also takes hand-built lists from callers
    # parsing third-party RTTM. The early-break loop below depends on sort
    # order, so paying O(n log n) on a typically <1000-turn list is cheap
    # insurance against a wrong-result-without-error bug.
    turns = sorted(turns, key=lambda t: t.start)

    out: list[TimestampedSegment] = []
    for seg in segments:
        overlaps: dict[str, float] = {}
        for turn in turns:
            if turn.end <= seg.start_time:
                continue
            if turn.start >= seg.end_time:
                # Turns are sorted; everything after is also past this segment.
                break
            ov = min(turn.end, seg.end_time) - max(turn.start, seg.start_time)
            if ov > 0:
                overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + ov
        if not overlaps:
            out.append(seg)
            continue
        best_spk = max(overlaps.items(), key=lambda kv: kv[1])[0]
        out.append(seg.with_speaker(best_spk))
    return out


def turns_to_rttm(turns: list[SpeakerTurn], uri: str) -> str:
    """NIST RTTM v1.3 — one ``SPEAKER`` line per turn.

    ``uri`` is the recording identifier; typically the basename without
    extension of the source audio. Must be non-empty and contain no
    whitespace (RTTM is space-delimited, so whitespace in URI or speaker
    label produces a file ``parse_rttm`` will silently truncate).
    """
    if not uri or not _RTTM_SAFE_RE.match(uri):
        raise ValueError(
            f"uri must be non-empty and whitespace-free, got {uri!r}"
        )
    lines = []
    for t in turns:
        if not _RTTM_SAFE_RE.match(t.speaker):
            raise ValueError(
                f"speaker label must be non-empty and whitespace-free, "
                f"got {t.speaker!r}"
            )
        if t.duration <= 0:
            # NIST RTTM consumers (dscore, pyannote.metrics) reject or silently
            # drop non-positive durations. Fail loud here so the bug surfaces
            # at write time instead of much later in someone else's tool.
            raise ValueError(
                f"turn duration must be positive, got {t.duration} for "
                f"speaker={t.speaker!r} start={t.start}"
            )
        lines.append(
            f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} "
            f"<NA> <NA> {t.speaker} <NA> <NA>"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def parse_rttm(path: str | Path) -> list[SpeakerTurn]:
    """Parse a NIST RTTM file. Lines that aren't ``SPEAKER`` rows are skipped.

    Uses ``utf-8-sig`` so a UTF-8 BOM at file start (Windows-edited RTTM,
    some MS tooling) doesn't cause the first SPEAKER row to be silently
    dropped because the leading byte sequence breaks the prefix match.

    Non-positive durations are skipped with a debug log line. Symmetrical
    with ``turns_to_rttm`` which raises on the same condition: write-path
    fails loud (so we never produce bad RTTM), read-path tolerates and
    drops (so we don't refuse to ingest a third-party file with one bad
    row). ``attribute_speakers``' overlap math also relies on positive
    durations, so silent drop is the safe default.
    """
    out: list[SpeakerTurn] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        parts = raw.strip().split()
        if not parts or parts[0] != "SPEAKER" or len(parts) < 8:
            continue
        try:
            start = float(parts[3])
            dur = float(parts[4])
        except ValueError:
            continue
        if dur <= 0:
            log.debug("parse_rttm: skipping non-positive duration row: %r", raw)
            continue
        out.append(SpeakerTurn(start=start, end=start + dur, speaker=parts[7]))
    out.sort(key=lambda t: t.start)
    return out


def format_speaker_totals(
    turns: list[SpeakerTurn], total_duration: float
) -> list[tuple[str, float, float]]:
    """Aggregate (speaker, seconds, percent) sorted desc by seconds."""
    totals: dict[str, float] = {}
    for t in turns:
        totals[t.speaker] = totals.get(t.speaker, 0.0) + t.duration
    rows = [
        (spk, secs, 100 * secs / max(total_duration, 1e-9))
        for spk, secs in totals.items()
    ]
    rows.sort(key=lambda r: -r[1])
    return rows
