"""Run pyannote diarization with multiple constraint configs in one process.

Loads the model once and runs each variant back-to-back, so we save the
~3 s model-load cost per variant and the ~30 s audio-convert cost across all.

Output naming:
    <audio>.<tag>.rttm
    <audio>.<tag>.speakers.txt
where ``tag`` is the variant name (``binary``, ``ternary`` by default).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from taigi_asr.audio import AudioConverter  # noqa: E402
from taigi_asr.diarize import (  # noqa: E402
    DiarizationPipeline,
    format_speaker_totals,
    turns_to_rttm,
)

DEFAULT_VARIANTS: list[tuple[str, dict]] = [
    ("binary", {"min_speakers": 2, "max_speakers": 2}),
    ("ternary", {"min_speakers": 3, "max_speakers": 3}),
]


def _human(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sanitize_uri(name: str) -> str:
    """RTTM URI must be non-empty and whitespace-free.

    ``"_".join(name.split())`` collapses any whitespace; the fallback
    ``"audio"`` catches the pathological empty stem (e.g. a file literally
    named ``.m4a``) so we fail fast at the CLI surface rather than 5 minutes
    later inside ``turns_to_rttm``.
    """
    return "_".join(name.split()) or "audio"


def main() -> int:
    ap = argparse.ArgumentParser(prog="diarize_compare")
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        return 2

    # Validate URI before paying for audio conversion + model load. A bad URI
    # would otherwise raise inside the per-variant loop after ~5 minutes.
    uri = _sanitize_uri(args.audio.stem)

    print(f"Converting {args.audio.name}...", file=sys.stderr)
    wav_path, duration = AudioConverter.convert(args.audio)
    print(f"  -> {wav_path.name}  ({duration:.1f} s)", file=sys.stderr)

    pipeline = DiarizationPipeline()
    try:
        print(f"Loading {pipeline.PIPELINE_ID}...", file=sys.stderr)
        t0 = time.monotonic()
        pipeline.load()
        print(f"  -> loaded in {time.monotonic() - t0:.1f}s", file=sys.stderr)

        for tag, kw in DEFAULT_VARIANTS:
            print(f"\n--- variant '{tag}'  {kw} ---", file=sys.stderr)
            t0 = time.monotonic()
            turns = pipeline.run(wav_path, **kw)
            elapsed = time.monotonic() - t0
            xrt = duration / max(elapsed, 1e-3)
            print(
                f"  -> done in {elapsed:.1f}s (xRT {xrt:.1f}), {len(turns)} turns",
                file=sys.stderr,
            )

            rttm_path = args.audio.with_name(f"{args.audio.stem}.{tag}.rttm")
            rttm_path.write_text(turns_to_rttm(turns, uri), encoding="utf-8")

            timeline_path = args.audio.with_name(f"{args.audio.stem}.{tag}.speakers.txt")
            with timeline_path.open("w", encoding="utf-8") as fh:
                for t in turns:
                    fh.write(
                        f"[{_human(t.start)} - {_human(t.end)}] {t.speaker}  ({t.duration:.1f}s)\n"
                    )

            print(f"[OK] {rttm_path.name}", file=sys.stderr)
            print(f"[OK] {timeline_path.name}", file=sys.stderr)
            for spk, secs, pct in format_speaker_totals(turns, duration):
                print(
                    f"    {spk}: {_human(secs)} ({secs:.0f}s, {pct:.1f}%)",
                    file=sys.stderr,
                )
    finally:
        pipeline.unload()
        AudioConverter.cleanup(wav_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
