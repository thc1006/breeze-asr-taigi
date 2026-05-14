"""Single-shot diarization PoC.

Thin CLI wrapper over :class:`taigi_asr.diarize.DiarizationPipeline` — same
behaviour as the production CLI's ``--diarize`` pass, but skips the ASR stage
entirely (useful for sanity-checking speaker turns without re-running
faster-whisper). Writes ``<audio>.rttm`` + ``<audio>.speakers.txt`` next to
the input.

For multi-variant comparison (binary vs ternary speaker constraints), use
``scripts/diarize_compare.py`` instead — it loads the model once and runs
multiple configs back-to-back.
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


def _human(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _sanitize_uri(name: str) -> str:
    return "_".join(name.split()) or "audio"


def main() -> int:
    ap = argparse.ArgumentParser(prog="diarize_poc")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"ERROR: audio not found: {args.audio}", file=sys.stderr)
        return 2

    uri = _sanitize_uri(args.audio.stem)

    print(f"Converting {args.audio.name} to 16k mono WAV...", file=sys.stderr)
    wav_path, duration = AudioConverter.convert(args.audio)
    print(f"  -> {wav_path.name}  ({duration:.1f} s)", file=sys.stderr)

    pipeline = DiarizationPipeline()
    try:
        print(f"Loading {pipeline.PIPELINE_ID}...", file=sys.stderr)
        t0 = time.monotonic()
        pipeline.load()
        print(f"  -> loaded in {time.monotonic() - t0:.1f}s", file=sys.stderr)

        kw: dict = {}
        if args.num_speakers is not None:
            kw["num_speakers"] = args.num_speakers
        if args.min_speakers is not None:
            kw["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            kw["max_speakers"] = args.max_speakers

        print(f"Running diarization (kw={kw or 'auto'})...", file=sys.stderr)
        t0 = time.monotonic()
        turns = pipeline.run(wav_path, **kw)
        elapsed = time.monotonic() - t0
        xrt = duration / max(elapsed, 1e-3)
        print(
            f"  -> done in {elapsed:.1f}s (xRT {xrt:.1f}), {len(turns)} turns",
            file=sys.stderr,
        )

        rttm_path = args.audio.with_suffix(".rttm")
        rttm_path.write_text(turns_to_rttm(turns, uri), encoding="utf-8")
        print(f"[OK] RTTM: {rttm_path}", file=sys.stderr)

        timeline_path = args.audio.with_suffix(".speakers.txt")
        with timeline_path.open("w", encoding="utf-8") as fh:
            for t in turns:
                fh.write(
                    f"[{_human(t.start)} - {_human(t.end)}] "
                    f"{t.speaker}  ({t.duration:.1f}s)\n"
                )
        print(f"[OK] Timeline: {timeline_path}", file=sys.stderr)

        print("\nSpeaker totals:", file=sys.stderr)
        for spk, secs, pct in format_speaker_totals(turns, duration):
            print(
                f"  {spk}: {_human(secs)} ({secs:.0f}s, {pct:.1f}%)",
                file=sys.stderr,
            )
        print(f"Total speakers: {len({t.speaker for t in turns})}", file=sys.stderr)
    finally:
        pipeline.unload()
        AudioConverter.cleanup(wav_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
