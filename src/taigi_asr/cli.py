"""Command-line entry point: ``taigi-asr <audio> [<audio> ...] [options]``.

Uses only stdlib argparse to keep the dependency tree flat.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from taigi_asr import __version__
from taigi_asr.audio import AudioConverter
from taigi_asr.config import SUPPORTED_AUDIO_EXTS
from taigi_asr.engines import build_engine
from taigi_asr.errors import InsufficientVRAMError, TaigiASRError
from taigi_asr.formatters import to_json, to_srt, to_txt, to_vtt
from taigi_asr.router import EngineKind, EngineRouter, GPUProfiler

if TYPE_CHECKING:
    # Pulled in only for the asr_results annotation; importing at runtime
    # would force the segments module into the CLI's hot startup path.
    from taigi_asr.segments import TimestampedSegment


def _parse_engine(raw: str) -> EngineKind | None:
    mapping = {
        "auto": None,
        "fw": EngineKind.FASTER_WHISPER,
        "faster-whisper": EngineKind.FASTER_WHISPER,
        "hf": EngineKind.HUGGINGFACE,
        "huggingface": EngineKind.HUGGINGFACE,
    }
    if raw not in mapping:
        raise argparse.ArgumentTypeError(f"Unknown engine: {raw}")
    return mapping[raw]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taigi-asr",
        description="Taiwanese Hokkien ASR powered by MediaTek Breeze-ASR-26.",
    )
    # nargs="*" so users can rely solely on --input-dir without supplying
    # positional paths. Validation of "at least one resolved input" happens
    # in main() after the directory glob runs.
    parser.add_argument(
        "audio",
        type=Path,
        nargs="*",
        help="One or more audio/video files. Combine with --input-dir to add a directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Add every supported audio/video file in this directory to the batch "
            f"(extensions: {', '.join(sorted(SUPPORTED_AUDIO_EXTS))}). Non-recursive."
        ),
    )
    # TAIGI_ASR_DEFAULT_ENGINE lets the user flip the default engine per
    # machine without changing the router (e.g. set to "hf" on an Optimus
    # laptop where the 4 GB fp16 path is preferred over int8_float16).
    env_default = os.environ.get("TAIGI_ASR_DEFAULT_ENGINE", "auto").strip() or "auto"
    parser.add_argument(
        "--engine",
        default=env_default,
        type=_parse_engine,
        help="auto | fw (faster-whisper) | hf (huggingface). "
        "Default can be overridden via TAIGI_ASR_DEFAULT_ENGINE env var.",
    )
    parser.add_argument(
        "--format",
        default="srt",
        help="Output format(s). Single: srt|txt|vtt|json. Multiple: "
        "comma-separated (e.g. 'srt,txt,json') - each written alongside input.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path. Honored only when one input + one format are requested; "
        "otherwise outputs land alongside each input file.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Emit word-level timestamps (slower)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam search width (default 5; RTX 3050 4GB safe ceiling: 12)",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="Number of candidates for temperature fallback sampling (default 5)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help=(
            "Attach pyannote/speaker-diarization-3.1 speaker labels to every "
            "segment. Requires HF_TOKEN env var and license acceptance for "
            "pyannote/speaker-diarization-3.1 + pyannote/segmentation-3.0. "
            "Adds a sequential pass on the same GPU after ASR (ASR unloads "
            "first to keep peak VRAM under 4 GB)."
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact speaker count for diarization (overrides auto-detect).",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Lower bound on speaker count (with --diarize).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Upper bound on speaker count (with --diarize).",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"taigi-asr {__version__}")
    return parser


def _has_any_speaker_flag(args) -> bool:
    """True iff at least one diarization speaker-count knob was supplied.

    Extracted so the warning-without-diarize path can be unit-tested directly
    without spinning up the full main() (which would proceed to engine load).
    """
    return (
        args.num_speakers is not None
        or args.min_speakers is not None
        or args.max_speakers is not None
    )


def _resolve_inputs(positional: list[Path], input_dir: Path | None) -> list[Path]:
    """Merge positional file args with --input-dir glob.

    Order: positional first (preserved), then directory entries in sorted
    order. Duplicates collapsed by resolved absolute path so the same file
    isn't transcribed twice when both modes hit it.
    """
    resolved: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        try:
            key = p.resolve()
        except OSError:
            key = p.absolute()
        if key in seen:
            return
        seen.add(key)
        resolved.append(p)

    for p in positional:
        _add(p)

    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"--input-dir not found: {input_dir}")
        if not input_dir.is_dir():
            raise NotADirectoryError(f"--input-dir is not a directory: {input_dir}")
        for child in sorted(input_dir.iterdir()):
            if child.is_file() and child.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                _add(child)

    return resolved


def _render(segments, fmt: str, meta: dict) -> str:
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "txt":
        return to_txt(segments)
    if fmt == "vtt":
        return to_vtt(segments)
    if fmt == "json":
        return to_json(segments, meta=meta)
    raise ValueError(fmt)


def _write_outputs(
    audio: Path,
    segments,
    formats: list[str],
    single_out_path: Path | None,
    meta: dict,
) -> None:
    """Render + persist outputs for one input. Pure I/O — segments are already
    finalized (and possibly speaker-attributed) before this is called."""
    if single_out_path is not None:
        single_out_path.write_text(_render(segments, formats[0], meta), encoding="utf-8")
        print(f"[OK] Saved: {single_out_path}", file=sys.stderr)
    else:
        for fmt in formats:
            out_path = audio.with_suffix(f".{fmt}")
            out_path.write_text(_render(segments, fmt, meta), encoding="utf-8")
            print(f"[OK] Saved: {out_path}", file=sys.stderr)


def _transcribe_one(
    engine,
    audio: Path,
    formats: list[str],
    single_out_path: Path | None,
    transcribe_kwargs: dict,
    meta_base: dict,
) -> tuple[bool, float, float]:
    """Run convert + transcribe + render for one input.

    Returns ``(ok, duration_sec, elapsed_sec)``.

    Failed files preserve the real audio duration whenever it was determined
    (transcribe failures, empty transcripts) — only convert failures report
    0.0, since duration is undetermined when the source can't be decoded.
    The caller decides whether to credit failed files toward aggregate xRT.

    ``single_out_path`` is ``None`` for the multi-format / multi-file case
    (write each format alongside the input) or a concrete path for the
    single-input + single-format ``--out`` override case. The caller has
    already validated the latter — this function does not double-check.
    """
    t0 = time.monotonic()
    wav_path: Path | None = None
    duration = 0.0
    try:
        wav_path, duration = AudioConverter.convert(audio)
        print(f"Audio duration: {duration:.1f} s", file=sys.stderr)
        segments = engine.transcribe(wav_path, **transcribe_kwargs)
    except TaigiASRError as exc:
        print(f"ERROR [{audio.name}]: {exc}", file=sys.stderr)
        return False, duration, time.monotonic() - t0
    finally:
        if wav_path is not None:
            AudioConverter.cleanup(wav_path)

    if not segments:
        print(f"WARNING [{audio.name}]: empty transcript", file=sys.stderr)
        return False, duration, time.monotonic() - t0

    meta = {**meta_base, "duration_sec": round(duration, 2)}
    _write_outputs(audio, segments, formats, single_out_path, meta)
    return True, duration, time.monotonic() - t0


def _asr_keep_wav(
    engine,
    audio: Path,
    transcribe_kwargs: dict,
):
    """ASR-only variant of _transcribe_one for the --diarize pipeline.

    Converts the audio and runs the ASR engine, but does NOT clean up the WAV
    or write outputs — diarization needs the same 16 kHz WAV later. Returns
    ``(ok, segments, wav_path, duration, elapsed)``; on failure the WAV (if
    created) is cleaned up immediately and ``wav_path`` is None.
    """
    t0 = time.monotonic()
    wav_path = None
    duration = 0.0
    try:
        wav_path, duration = AudioConverter.convert(audio)
        print(f"Audio duration: {duration:.1f} s", file=sys.stderr)
        segments = engine.transcribe(wav_path, **transcribe_kwargs)
    except TaigiASRError as exc:
        print(f"ERROR [{audio.name}]: {exc}", file=sys.stderr)
        if wav_path is not None:
            AudioConverter.cleanup(wav_path)
        return False, [], None, duration, time.monotonic() - t0

    if not segments:
        print(f"WARNING [{audio.name}]: empty transcript", file=sys.stderr)
        AudioConverter.cleanup(wav_path)
        return False, [], None, duration, time.monotonic() - t0

    return True, segments, wav_path, duration, time.monotonic() - t0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        inputs = _resolve_inputs(list(args.audio), args.input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print(
            "ERROR: no audio inputs provided (pass paths positionally or use --input-dir).",
            file=sys.stderr,
        )
        return 2

    missing = [p for p in inputs if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: audio not found: {p}", file=sys.stderr)
        inputs = [p for p in inputs if p.exists()]
        if not inputs:
            return 2

    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    valid = {"srt", "txt", "vtt", "json"}
    bad = [f for f in formats if f not in valid]
    if bad:
        print(f"ERROR: unknown format(s): {bad}. Choose from {sorted(valid)}", file=sys.stderr)
        return 6

    # Diarization flag validation — surface user mistakes before we burn 5+
    # minutes of GPU time.
    speaker_flag_set = _has_any_speaker_flag(args)
    if speaker_flag_set and not args.diarize:
        print(
            "WARNING: --num-speakers/--min-speakers/--max-speakers are ignored without --diarize.",
            file=sys.stderr,
        )
    if (
        args.diarize
        and args.num_speakers is not None
        and (args.min_speakers is not None or args.max_speakers is not None)
    ):
        print(
            "ERROR: --num-speakers is mutually exclusive with --min-speakers/--max-speakers.",
            file=sys.stderr,
        )
        return 6
    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        print(
            f"ERROR: --min-speakers ({args.min_speakers}) > --max-speakers ({args.max_speakers})",
            file=sys.stderr,
        )
        return 6

    info = GPUProfiler.detect()
    try:
        spec = EngineRouter.select(info, prefer=args.engine)
    except InsufficientVRAMError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(
        f"Device: {info.name} | {info.vram_gb:.1f} GB | "
        f"Engine: {spec.kind.value} ({spec.compute_type}, batch={spec.batch_size})",
        file=sys.stderr,
    )

    # `--out` is the single-target sentinel: honored only when there's exactly
    # one input and exactly one format. Resolve once up-front so _transcribe_one
    # stays dumb (Path | None) and we don't re-derive the condition per file.
    single_out_path: Path | None = None
    if len(inputs) == 1 and len(formats) == 1 and args.out is not None:
        single_out_path = args.out
    elif args.out is not None:
        print(
            "WARNING: --out ignored when multiple inputs or formats are requested; "
            "outputs will be written alongside each input.",
            file=sys.stderr,
        )

    # Engine-specific knob filtering. Done ONCE here (not per-file) so a
    # 50-file batch with `--engine hf --beam-size 10` doesn't spam 50
    # identical "ignored on HF engine" warnings.
    transcribe_kwargs: dict = {"word_timestamps": args.word_timestamps}
    if spec.kind is EngineKind.FASTER_WHISPER:
        if args.beam_size is not None:
            transcribe_kwargs["beam_size"] = args.beam_size
        if args.best_of is not None:
            transcribe_kwargs["best_of"] = args.best_of
    elif args.beam_size is not None or args.best_of is not None:
        print(
            "WARNING: --beam-size / --best-of are ignored on the HuggingFace engine.",
            file=sys.stderr,
        )

    engine = build_engine(spec)
    try:
        engine.load()
    except InsufficientVRAMError as exc:
        # Auto-downgrade only when the user asked for `auto` AND the
        # original choice wasn't already Faster-Whisper.
        if args.engine is not None or spec.kind == EngineKind.FASTER_WHISPER:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 4
        print(f"WARNING: {exc}. Retrying with faster_whisper.", file=sys.stderr)
        try:
            engine.unload()
        except Exception:  # pragma: no cover
            pass
        spec = EngineRouter.select(info, prefer=EngineKind.FASTER_WHISPER)
        print(
            f"Device: {info.name} | {info.vram_gb:.1f} GB | "
            f"Engine: {spec.kind.value} ({spec.compute_type}, batch={spec.batch_size})",
            file=sys.stderr,
        )
        engine = build_engine(spec)
        try:
            engine.load()
        except TaigiASRError as exc2:
            print(f"ERROR: {exc2}", file=sys.stderr)
            return 4
    except TaigiASRError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 4

    meta_base = {"engine": spec.kind.value, "compute_type": spec.compute_type}

    failed: list[Path] = []
    # Aggregate stats count *successful* files only — failed files have
    # undefined or partial elapsed/duration that would distort xRT and
    # mislead the user about real throughput.
    success_duration = 0.0
    success_elapsed = 0.0

    if args.diarize:
        # Two-pass orchestration: ASR all files first, unload to free VRAM,
        # then diarize. Necessary on 4 GB cards where ASR (~2.9 GB peak) and
        # pyannote (~700-900 MB) can't co-reside.
        asr_results: list[tuple[Path, list[TimestampedSegment], Path, float, float]] = []
        # Tracked separately from asr_results so the cleanup loop runs even
        # when the diarize stage clears asr_results on load failure.
        wavs_to_cleanup: list[Path] = []
        try:
            try:
                for idx, audio in enumerate(inputs, 1):
                    if len(inputs) > 1:
                        print(f"\n[{idx}/{len(inputs)}] ASR: {audio}", file=sys.stderr)
                    ok, segs, wav_path, duration, elapsed = _asr_keep_wav(
                        engine, audio, transcribe_kwargs
                    )
                    if ok and wav_path is not None:
                        asr_results.append((audio, segs, wav_path, duration, elapsed))
                        wavs_to_cleanup.append(wav_path)
                    else:
                        failed.append(audio)
            finally:
                try:
                    engine.unload()
                except Exception:  # pragma: no cover
                    pass

            if asr_results:
                # Defer the diarize module import until after ASR has run and
                # been unloaded: pyannote.audio pulls in extra deps and using
                # it before unload races against the 4 GB VRAM budget.
                from taigi_asr.diarize import (
                    DiarizationPipeline,
                    attribute_speakers,
                    turns_to_rttm,
                )

                diarize_kwargs: dict = {}
                if args.num_speakers is not None:
                    diarize_kwargs["num_speakers"] = args.num_speakers
                if args.min_speakers is not None:
                    diarize_kwargs["min_speakers"] = args.min_speakers
                if args.max_speakers is not None:
                    diarize_kwargs["max_speakers"] = args.max_speakers

                dia = DiarizationPipeline()
                try:
                    print(
                        "\nLoading pyannote/speaker-diarization-3.1 on cuda...",
                        file=sys.stderr,
                    )
                    try:
                        dia.load()
                    except TaigiASRError as exc:
                        # Fallback: write un-attributed ASR transcripts so the
                        # user doesn't lose the GPU time already spent on the
                        # ASR pass. The CLI still exits non-zero (return 4
                        # below via the failed[] list) so callers / CI scripts
                        # can detect that --diarize didn't take effect, but
                        # the transcripts on disk are intact.
                        print(
                            f"ERROR: diarize load failed: {exc}\n"
                            "  -> falling back to un-attributed ASR transcripts "
                            "(no speaker labels). Files marked as failed; CLI "
                            "will exit with error code, but written outputs are "
                            "usable.",
                            file=sys.stderr,
                        )
                        for audio, segs, _wav_path, duration, asr_elapsed in asr_results:
                            meta = {
                                **meta_base,
                                "duration_sec": round(duration, 2),
                                "diarized": False,
                                "diarize_error": str(exc),
                            }
                            write_ok = True
                            try:
                                _write_outputs(audio, segs, formats, single_out_path, meta)
                            except Exception as werr:  # pragma: no cover
                                # Fallback is best-effort — any write failure
                                # (OSError, UnicodeEncodeError, _render's
                                # ValueError on unknown format, etc.) must
                                # NOT abort the loop for subsequent files.
                                # The user already lost diarize; we shouldn't
                                # additionally lose ASR work for files N+1..M.
                                write_ok = False
                                print(
                                    f"ERROR [{audio.name}] write fallback: {werr}",
                                    file=sys.stderr,
                                )
                            failed.append(audio)
                            # Credit the ASR cost even though we exit non-zero —
                            # the un-attributed transcript IS on disk and the
                            # GPU time was real work. Without this credit the
                            # Batch summary at end-of-run would print "no
                            # successful transcriptions" while N files sit on
                            # disk, contradicting itself.
                            if write_ok:
                                success_duration += duration
                                success_elapsed += asr_elapsed
                        asr_results = []
                    for audio, segs, wav_path, duration, asr_elapsed in asr_results:
                        try:
                            t0 = time.monotonic()
                            turns = dia.run(wav_path, **diarize_kwargs)
                            dia_elapsed = time.monotonic() - t0
                            attributed = attribute_speakers(segs, turns)
                            n_spk = len({s.speaker for s in attributed if s.speaker})
                            meta = {
                                **meta_base,
                                "duration_sec": round(duration, 2),
                                "diarized": True,
                                "num_speakers": n_spk,
                            }
                            _write_outputs(audio, attributed, formats, single_out_path, meta)
                            # Companion RTTM — same basename as the audio so
                            # downstream tooling (merger, NIST tools) finds it
                            # next to the transcript.
                            rttm_path = audio.with_suffix(".rttm")
                            # ``or "audio"`` mirrors scripts/diarize_compare.py:_sanitize_uri.
                            # Without the fallback, a pathological stem like
                            # ``"  "`` (e.g. file named ``"  .m4a"``) collapses
                            # to empty string after split-join, then
                            # ``turns_to_rttm`` raises ``ValueError`` — which is
                            # NOT ``TaigiASRError`` and would escape the per-file
                            # ``except`` below, aborting Phase 2 after pyannote
                            # already consumed GPU time for this file.
                            uri = "_".join(audio.stem.split()) or "audio"
                            rttm_path.write_text(turns_to_rttm(turns, uri), encoding="utf-8")
                            print(f"[OK] Saved: {rttm_path}", file=sys.stderr)
                            total_elapsed = asr_elapsed + dia_elapsed
                            success_duration += duration
                            success_elapsed += total_elapsed
                            if len(inputs) > 1:
                                xrt = duration / max(total_elapsed, 1e-3)
                                print(
                                    f"  -> ASR {asr_elapsed:.1f}s + "
                                    f"diarize {dia_elapsed:.1f}s "
                                    f"(xRT {xrt:.1f}, {n_spk} speakers)",
                                    file=sys.stderr,
                                )
                        except TaigiASRError as exc:
                            # Symmetric with the dia.load() failure branch
                            # above: write the un-attributed ASR transcript so
                            # the user doesn't lose the GPU time already spent
                            # transcribing this file. CLI still exits non-zero
                            # via failed[] so callers can detect the partial
                            # outcome.
                            print(
                                f"ERROR [{audio.name}] diarize: {exc}\n"
                                "  -> writing un-attributed ASR fallback for this file.",
                                file=sys.stderr,
                            )
                            fb_meta = {
                                **meta_base,
                                "duration_sec": round(duration, 2),
                                "diarized": False,
                                "diarize_error": str(exc),
                            }
                            try:
                                _write_outputs(audio, segs, formats, single_out_path, fb_meta)
                            except Exception as werr:  # pragma: no cover
                                # Same rationale as the dia.load() fallback
                                # write — broaden the catch so a unicode /
                                # render / OS failure on file N doesn't kill
                                # the diarize loop for files N+1..M.
                                print(
                                    f"ERROR [{audio.name}] write fallback: {werr}",
                                    file=sys.stderr,
                                )
                            failed.append(audio)
                finally:
                    try:
                        dia.unload()
                    except Exception:  # pragma: no cover
                        pass
        finally:
            for wav_path in wavs_to_cleanup:
                AudioConverter.cleanup(wav_path)
    else:
        try:
            for idx, audio in enumerate(inputs, 1):
                if len(inputs) > 1:
                    print(f"\n[{idx}/{len(inputs)}] {audio}", file=sys.stderr)
                ok, duration, elapsed = _transcribe_one(
                    engine,
                    audio,
                    formats,
                    single_out_path,
                    transcribe_kwargs,
                    meta_base,
                )
                if ok:
                    success_duration += duration
                    success_elapsed += elapsed
                    if len(inputs) > 1:
                        xrt = duration / max(elapsed, 1e-3)
                        print(f"  -> {elapsed:.1f}s (xRT {xrt:.1f})", file=sys.stderr)
                else:
                    failed.append(audio)
        finally:
            try:
                engine.unload()
            except Exception:  # pragma: no cover
                pass

    if len(inputs) > 1:
        ok_count = len(inputs) - len(failed)
        if success_duration > 0:
            agg_xrt = success_duration / max(success_elapsed, 1e-3)
            print(
                f"\nBatch summary: {ok_count}/{len(inputs)} OK | "
                f"audio {success_duration:.0f}s | wall {success_elapsed:.0f}s | "
                f"xRT {agg_xrt:.1f} (excl. model load)",
                file=sys.stderr,
            )
            if ok_count == 0:
                # All files counted as ``failed`` BUT success_duration > 0
                # means the dia.{load,run}() fallback wrote un-attributed
                # transcripts for them. Disambiguate so the user doesn't read
                # "0/N OK + positive audio time" as a contradiction.
                print(
                    "  (--diarize did not take effect for any file; "
                    "ASR transcripts were written as un-attributed fallback.)",
                    file=sys.stderr,
                )
        else:
            print(
                f"\nBatch summary: {ok_count}/{len(inputs)} OK (no successful transcriptions)",
                file=sys.stderr,
            )

    if failed:
        print(
            f"FAILED: {len(failed)} file(s): {', '.join(p.name for p in failed)}",
            file=sys.stderr,
        )
        return 4 if len(failed) == len(inputs) else 7

    return 0


if __name__ == "__main__":
    sys.exit(main())
