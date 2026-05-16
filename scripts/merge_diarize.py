"""Merge an existing SRT + RTTM into a speaker-attributed transcript.

Standalone — does not call pyannote. Useful when you already have:
- a transcript SRT from ``taigi-asr`` (or any Whisper-compatible tool), and
- an RTTM produced by ``diarize_poc.py`` / ``diarize_compare.py``.

Each ASR segment is attributed to the diarization speaker that overlaps it
the most. Output formats: SRT, TXT, JSON (all with speaker labels).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from taigi_asr.diarize import (  # noqa: E402
    attribute_speakers,
    format_speaker_totals,
    parse_rttm,
)
from taigi_asr.formatters import to_json, to_srt, to_txt  # noqa: E402
from taigi_asr.segments import TimestampedSegment  # noqa: E402

SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.+?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _parse_srt_time(s: str) -> float:
    h, m, rest = s.split(":")
    sec, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000


def parse_srt(path: Path) -> list[TimestampedSegment]:
    # ``utf-8-sig`` transparently strips a UTF-8 BOM if present. Without this,
    # a Windows-edited SRT (Notepad, Subtitle Edit) would have ``﻿`` glued
    # to cue #1's index and the regex would silently drop it.
    text = path.read_text(encoding="utf-8-sig")
    out: list[TimestampedSegment] = []
    for m in SRT_BLOCK_RE.finditer(text):
        # Multi-line cues join with a space; preserves the on-screen text in a
        # plain-text-friendly single line.
        body = " ".join(line.strip() for line in m.group(4).splitlines() if line.strip())
        out.append(
            TimestampedSegment(
                start_time=_parse_srt_time(m.group(2)),
                end_time=_parse_srt_time(m.group(3)),
                text=body,
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="merge_diarize")
    ap.add_argument("srt", type=Path, help="Input SRT (from taigi-asr).")
    ap.add_argument("rttm", type=Path, help="Input RTTM (from diarize_poc.py).")
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help=(
            "Path prefix for outputs (extensions appended). "
            "Defaults to '<srt-stem>.diarized' alongside the input SRT."
        ),
    )
    args = ap.parse_args()

    if not args.srt.exists():
        print(f"ERROR: SRT not found: {args.srt}", file=sys.stderr)
        return 2
    if not args.rttm.exists():
        print(f"ERROR: RTTM not found: {args.rttm}", file=sys.stderr)
        return 2

    segments = parse_srt(args.srt)
    turns = parse_rttm(args.rttm)
    if not segments:
        print("ERROR: SRT parsed to zero segments — bad format?", file=sys.stderr)
        return 3
    if not turns:
        print("ERROR: RTTM parsed to zero turns — bad format?", file=sys.stderr)
        return 3

    audio_span = max(s.end_time for s in segments)
    print(
        f"SRT: {len(segments)} segments, span ~{audio_span:.0f}s | RTTM: {len(turns)} turns",
        file=sys.stderr,
    )

    attributed = attribute_speakers(segments, turns)

    seg_per_spk: dict[str, int] = {}
    sec_per_spk: dict[str, float] = {}
    for s in attributed:
        spk = s.speaker or "UNKNOWN"
        seg_per_spk[spk] = seg_per_spk.get(spk, 0) + 1
        sec_per_spk[spk] = sec_per_spk.get(spk, 0.0) + (s.end_time - s.start_time)

    prefix = args.out_prefix or args.srt.with_name(args.srt.stem + ".diarized")
    # NB: ``Path.with_suffix(".srt")`` would *replace* ``.diarized`` (treats it
    # as the existing suffix) and silently overwrite the input SRT. Append the
    # extension to ``.name`` instead so the prefix is preserved verbatim.
    srt_path = prefix.with_name(prefix.name + ".srt")
    txt_path = prefix.with_name(prefix.name + ".txt")
    json_path = prefix.with_name(prefix.name + ".json")

    # Clobber guard: if the user-supplied --out-prefix collides with the input
    # SRT path, refuse to write. ``resolve()`` handles ./../, symlinks, and
    # Windows backslash-vs-forward-slash normalization in one shot.
    try:
        in_resolved = args.srt.resolve()
    except OSError:
        in_resolved = args.srt
    for out_path in (srt_path, txt_path, json_path):
        try:
            out_resolved = out_path.resolve()
        except OSError:
            out_resolved = out_path
        if out_resolved == in_resolved:
            print(
                f"ERROR: refusing to overwrite input SRT: {args.srt} == "
                f"output {out_path}. Pass an explicit --out-prefix that "
                "does not collide.",
                file=sys.stderr,
            )
            return 4

    srt_path.write_text(to_srt(attributed), encoding="utf-8")
    txt_path.write_text(to_txt(attributed), encoding="utf-8")
    json_path.write_text(
        to_json(
            attributed,
            meta={
                "source_srt": str(args.srt),
                "source_rttm": str(args.rttm),
                "speakers": dict(seg_per_spk),
            },
        ),
        encoding="utf-8",
    )

    print(f"[OK] {srt_path}", file=sys.stderr)
    print(f"[OK] {txt_path}", file=sys.stderr)
    print(f"[OK] {json_path}", file=sys.stderr)

    print("\nSegment attribution (sorted desc by segment count):", file=sys.stderr)
    for spk in sorted(seg_per_spk, key=lambda k: -seg_per_spk[k]):
        secs = sec_per_spk[spk]
        pct = 100 * secs / max(audio_span, 1e-9)
        print(
            f"  {spk}: {seg_per_spk[spk]} segments  ({secs:.0f}s, {pct:.1f}%)",
            file=sys.stderr,
        )

    # RTTM-side totals to compare against the ASR-segment attribution above —
    # divergence between the two hints at where attribute_speakers had to break ties.
    print("\nRTTM-side speaker totals:", file=sys.stderr)
    for spk, secs, pct in format_speaker_totals(turns, audio_span):
        print(f"  {spk}: {secs:.0f}s ({pct:.1f}%)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
