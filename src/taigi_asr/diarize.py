"""Speaker diarization via pyannote.audio 4.x.

Default pipeline: ``pyannote/speaker-diarization-community-1``. Any pipeline
pyannote.audio 4.x can load (e.g. ``pyannote/speaker-diarization-3.1``) can be
selected with the ``TAIGI_ASR_DIARIZE_PIPELINE`` env var.

Wraps the pyannote pipeline with the same load/run/unload contract as the ASR
engines so the CLI can swap models cleanly on a 4 GB VRAM budget. Also exposes
``attribute_speakers`` (overlap-based speaker assignment) and ``parse_rttm`` /
``turns_to_rttm`` so the same code path serves both the integrated CLI and the
standalone ``scripts/merge_diarize.py`` / ``scripts/diarize_compare.py`` tools.

Why 4.x (``pyannote.audio>=4.0.3``) and not 3.x: on today's dependency set the
3.x line cannot work at all. It passes ``use_auth_token`` to
``huggingface_hub.hf_hub_download``, which huggingface_hub 1.0 removed (and
gradio 6, a base dependency, needs huggingface_hub >= 1.16); it crashes at
import on torchaudio >= 2.9 (``torchaudio.AudioMetaData`` was removed); and
Lightning >= 2.6 makes its checkpoint load hit PyTorch's ``weights_only=True``
default. 4.0.3+ passes ``weights_only=False`` itself, so no ``torch.load``
patching is needed any more.
"""

from __future__ import annotations

import gc
import importlib.metadata
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

log = logging.getLogger(__name__)

DEFAULT_PIPELINE_ID = "pyannote/speaker-diarization-community-1"
PIPELINE_ENV_VAR = "TAIGI_ASR_DIARIZE_PIPELINE"
MIN_PYANNOTE_MAJOR = 4

# Gated repos whose user conditions must be accepted, per pipeline. Pipelines
# not listed here only gate their own repo.
_GATED_REPOS: dict[str, tuple[str, ...]] = {
    "pyannote/speaker-diarization-3.1": (
        "pyannote/speaker-diarization-3.1",
        "pyannote/segmentation-3.0",
    ),
}


def resolve_pipeline_id(explicit: str | None = None) -> str:
    """Explicit argument > ``TAIGI_ASR_DIARIZE_PIPELINE`` > community-1."""
    return explicit or os.environ.get(PIPELINE_ENV_VAR, "").strip() or DEFAULT_PIPELINE_ID


def gated_repos(pipeline_id: str) -> tuple[str, ...]:
    return _GATED_REPOS.get(pipeline_id, (pipeline_id,))


def _resolve_hf_token(explicit: str | None) -> str | None:
    """Explicit token > ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN`` > the token
    saved by ``hf auth login`` (via ``huggingface_hub.get_token``)."""
    token = explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover - huggingface_hub is a base dependency
        return None
    try:
        return get_token()
    except Exception:  # pragma: no cover - corrupt token file etc.
        return None


def _hf_offline() -> bool:
    """True when ``HF_HUB_OFFLINE`` is set: cached gated models need no token."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", raw)[:3])


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

# speechbrain fixed the Windows ``inspect.py`` path check in 1.1.1
# (speechbrain/speechbrain#3065).
_SPEECHBRAIN_FIXED = (1, 1, 1)


def _patch_speechbrain_lazy_module() -> None:
    """Windows-only fix for speechbrain < 1.1.1's introspection-bypass check.

    pyannote.audio 4.x no longer depends on speechbrain, but still imports it
    when it happens to be installed. Older speechbrain's
    ``LazyModule.ensure_module`` checks ``filename.endswith("/inspect.py")``,
    which never matches ``...\\inspect.py`` on Windows, so introspection
    triggers lazy imports of optional integrations (k2, ...) and fails with an
    opaque ``Lazy import of LazyModule(...) failed``.

    Replaces ``ensure_module`` with a version that recognizes both ``/`` and
    ``\\``. Gated on ``sys.platform == "win32"`` and on the installed
    speechbrain being older than the upstream fix, so fixed versions are
    never masked. The ``_taigi_patched`` sentinel makes it idempotent.
    """
    if sys.platform != "win32":
        return
    try:
        installed = _version_tuple(importlib.metadata.version("speechbrain"))
    except importlib.metadata.PackageNotFoundError:
        return
    if installed >= _SPEECHBRAIN_FIXED:
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

    def patched_ensure_module(
        self, stacklevel: int
    ):  # pragma: no cover - exercised only by speechbrain's own LazyModule attribute access during pyannote load
        importer_frame = None
        try:
            importer_frame = _inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except (AttributeError, ValueError):
            _warnings.warn(
                "Failed to inspect frame for speechbrain lazy import bypass.",
                stacklevel=2,
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
                    self.lazy_module = _importlib.import_module(f".{self.target}", self.package)
            except Exception as exc:
                raise ImportError(f"Lazy import of {repr(self)} failed") from exc
        return self.lazy_module

    _sb_iu.LazyModule.ensure_module = patched_ensure_module
    _sb_iu.LazyModule._taigi_patched = True


def _check_pyannote_version() -> None:
    """Fail with an actionable message instead of a deep TypeError/AttributeError."""
    try:
        raw = importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModelLoadError(
            'pyannote.audio not installed. Run: pip install -e ".[diarize]" '
            "(see README: install a CUDA torch >= 2.8 first)."
        ) from exc
    if _version_tuple(raw)[:1] < (MIN_PYANNOTE_MAJOR,):
        raise ModelLoadError(
            f"pyannote.audio {raw} is too old: the 3.x line is incompatible with "
            "huggingface_hub >= 1.0 and torchaudio >= 2.9. Run: "
            'pip install -U "pyannote.audio>=4.0.3,<5"'
        )


def _load_waveform(wav_path: str | Path) -> dict:
    """Read a WAV into pyannote's in-memory input format.

    Passing ``{"waveform", "sample_rate"}`` instead of a path keeps pyannote
    4.x off its torchcodec decoder, which needs FFmpeg *shared libraries*
    (not just the ``ffmpeg`` binary this project uses) — often missing on
    Windows. The input is AudioConverter's 16 kHz mono WAV; multi-channel
    input is down-mixed defensively.
    """
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise TranscriptionError(f"cannot read audio for diarization: {exc}") from exc
    waveform = torch.from_numpy(data.T.copy())  # (channel, time)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _annotation_from_output(output):
    """Unwrap pyannote 4.x ``DiarizeOutput``.

    Prefers ``exclusive_speaker_diarization`` (one speaker at a time), which
    pyannote provides precisely for reconciling diarization with transcript
    timestamps; falls back to ``speaker_diarization``, then to the object
    itself (a bare ``Annotation``, as legacy pipelines return). Explicit
    ``None`` checks, not ``or``: an empty Annotation is falsy.
    """
    for attr in ("exclusive_speaker_diarization", "speaker_diarization"):
        annotation = getattr(output, attr, None)
        if annotation is not None:
            return annotation
    return output


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
    """pyannote.audio 4.x speaker-diarization wrapper.

    Mirrors the ASR engine load/unload protocol so the CLI can sequence
    ``engine.unload()`` -> ``diarize.load()`` and stay under 4 GB VRAM.
    """

    PIPELINE_ID = DEFAULT_PIPELINE_ID

    def __init__(
        self,
        device: str | None = None,
        hf_token: str | None = None,
        pipeline_id: str | None = None,
    ) -> None:
        # ``device=None`` -> CUDA when available, else CPU (pyannote runs on
        # CPU too, just slower), instead of failing on CPU-only machines.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = _resolve_hf_token(hf_token)
        # Instance attribute shadows the class default so callers can keep
        # reading ``pipeline.PIPELINE_ID``.
        self.PIPELINE_ID = resolve_pipeline_id(pipeline_id)
        self._pipeline = None
        self._loaded = False
        self._lock = threading.Lock()

    def _license_hint(self) -> str:
        pages = " AND ".join(
            f"https://huggingface.co/{repo}" for repo in gated_repos(self.PIPELINE_ID)
        )
        return (
            f"Visit {pages} with the same HF account that owns the token and "
            "accept the user conditions on each page, then retry."
        )

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            # A local pipeline directory / config needs no token; neither do
            # cached models in offline mode.
            local = Path(self.PIPELINE_ID).exists()
            if not self.hf_token and not _hf_offline() and not local:
                raise ModelLoadError(
                    f"HF_TOKEN env var required for pyannote diarization ({self.PIPELINE_ID}). "
                    "Create a read token at https://hf.co/settings/tokens and set HF_TOKEN "
                    "(or run `hf auth login`). " + self._license_hint()
                )
            _check_pyannote_version()
            _patch_speechbrain_lazy_module()

            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:  # pragma: no cover
                raise ModelLoadError(f"pyannote.audio failed to import: {exc}") from exc

            try:
                pipeline = Pipeline.from_pretrained(  # pragma: no cover - real HF network + checkpoint load
                    self.PIPELINE_ID, token=self.hf_token
                )
                if pipeline is None:
                    raise ModelLoadError(
                        f"Failed to load {self.PIPELINE_ID} — verify the HF token. "
                        + self._license_hint()
                    )
                pipeline.to(torch.device(self.device))  # pragma: no cover - real device move
                self._pipeline = pipeline  # pragma: no cover
                self._loaded = True  # pragma: no cover
                log.info(  # pragma: no cover
                    "pyannote pipeline %s loaded on %s", self.PIPELINE_ID, self.device
                )
            except ModelLoadError:
                raise
            except Exception as exc:
                msg = str(exc)
                lower = msg.lower()
                # Targeted hint when HF returned 401/403/gated. The upstream
                # exception class depends on huggingface_hub version (GatedRepoError,
                # HfHubHTTPError, requests.HTTPError); matching on the message
                # text is the version-stable signal.
                if "gated" in lower or "401" in msg or "403" in msg or "unauthorized" in lower:
                    raise ModelLoadError(
                        f"License acceptance required for {self.PIPELINE_ID}. "
                        f"{self._license_hint()} (Upstream error: {exc})"
                    ) from exc
                raise ModelLoadError(f"Failed to load {self.PIPELINE_ID}: {exc}") from exc

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
                raise TranscriptionError("pipeline unloaded before run() could acquire it")

            kw: dict = {}
            if num_speakers is not None:
                kw["num_speakers"] = num_speakers
            if min_speakers is not None:
                kw["min_speakers"] = min_speakers
            if max_speakers is not None:
                kw["max_speakers"] = max_speakers

            audio = _load_waveform(wav_path)
            try:
                output = self._pipeline(audio, **kw)
            except Exception as exc:  # pragma: no cover - real pyannote GPU error
                raise TranscriptionError(f"diarization failed: {exc}") from exc

            turns = [
                SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
                for turn, _, speaker in _annotation_from_output(output).itertracks(yield_label=True)
            ]
            turns.sort(key=lambda t: t.start)
            return turns

    def unload(self) -> None:  # pragma: no cover - exercises real pyannote internals
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
                except Exception as exc:
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
        raise ValueError(f"uri must be non-empty and whitespace-free, got {uri!r}")
    lines = []
    for t in turns:
        if not _RTTM_SAFE_RE.match(t.speaker):
            raise ValueError(
                f"speaker label must be non-empty and whitespace-free, got {t.speaker!r}"
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
            f"SPEAKER {uri} 1 {t.start:.3f} {t.duration:.3f} <NA> <NA> {t.speaker} <NA> <NA>"
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
    rows = [(spk, secs, 100 * secs / max(total_duration, 1e-9)) for spk, secs in totals.items()]
    rows.sort(key=lambda r: -r[1])
    return rows
