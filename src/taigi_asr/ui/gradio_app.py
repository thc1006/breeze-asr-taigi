"""Gradio UI — the zero-friction front door.

End-user flow: **drop audio -> click transcribe -> download SRT/TXT**. Advanced
options are collapsed by default; 99% of users never open them.
"""

from __future__ import annotations

import atexit
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.engines.base import ASREngine
from taigi_asr.errors import InsufficientVRAMError, ModelLoadError, TranscriptionError
from taigi_asr.formatters import to_json, to_srt, to_txt, to_vtt
from taigi_asr.router import EngineKind, EngineRouter, EngineSpec, GPUInfo, GPUProfiler
from taigi_asr.segments import TimestampedSegment

log = logging.getLogger(__name__)


# --------------------------- state / helpers ------------------------------- #


_engine_cache: dict[str, ASREngine] = {}
_engine_lock = threading.Lock()


def _spec_key(spec: EngineSpec) -> str:
    return f"{spec.kind.value}:{spec.device}:{spec.compute_type}:{spec.batch_size}"


def _get_or_build_engine(spec: EngineSpec) -> ASREngine:
    """LRU-of-1 engine cache. Swapping to a new spec unloads the old engine so
    two Whisper-large instances never coexist in VRAM (the 4 GB budget fits one).
    """
    key = _spec_key(spec)
    with _engine_lock:
        if key not in _engine_cache:
            # Evict every cached engine that doesn't match the new spec.
            for stale_key in list(_engine_cache.keys()):
                if stale_key == key:
                    continue
                try:
                    _engine_cache[stale_key].unload()
                except Exception:
                    log.exception("Failed to unload stale engine %s", stale_key)
                del _engine_cache[stale_key]
            _engine_cache[key] = build_engine(spec)
        return _engine_cache[key]


def _gpu_panel_text(info: GPUInfo, spec: EngineSpec) -> str:
    lines = [
        f"**裝置 / Device**：{info.name}",
        f"**VRAM**：{info.vram_gb:.1f} GB" if info.cuda_available else "**VRAM**：CPU-only",
        f"**自動選擇引擎**：{spec.kind.value} ({spec.compute_type}, batch={spec.batch_size})",
    ]
    if not info.cuda_available:
        lines.append("警告：未偵測到 CUDA GPU，將使用 CPU（較慢）")
    elif info.vram_gb < 4:
        lines.append("警告：VRAM < 4 GB，建議關閉其他 GPU 程式釋放記憶體")
    else:
        lines.append("狀態：硬體配置正常")
    return "\n\n".join(lines)


def _resolve_spec(info: GPUInfo, engine_choice: str) -> EngineSpec:
    mapping = {
        "Auto / 自動": None,
        "Faster-Whisper": EngineKind.FASTER_WHISPER,
        "HuggingFace Pipeline": EngineKind.HUGGINGFACE,
    }
    prefer = mapping.get(engine_choice)
    return EngineRouter.select(info, prefer=prefer)


# --------------------------- transcription pipeline ------------------------ #


def _transcribe(
    audio_path: str | None,
    engine_choice: str,
    formats: list[str],
    word_timestamps: bool,
    beam_size: int,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, str, str | None, str | None, str | None, str | None]:
    """Core callback. Returns (status_md, preview_md, srt_path, txt_path, vtt_path, json_path)."""
    if not audio_path:
        return "ERROR: 請先上傳音檔。", "", None, None, None, None

    progress(0.02, desc="偵測 GPU...")
    info = GPUProfiler.detect()
    try:
        spec = _resolve_spec(info, engine_choice)
    except InsufficientVRAMError as exc:
        return f"ERROR: VRAM 不足：{exc}", "", None, None, None, None

    wav_path: Path | None = None
    try:
        progress(0.08, desc="音檔前處理 (16 kHz / mono)...")
        wav_path, duration = AudioConverter.convert(audio_path)

        progress(0.15, desc=f"載入引擎 ({spec.kind.value})...")
        try:
            engine = _get_or_build_engine(spec)
            engine.load()
        except InsufficientVRAMError as exc:
            # HF path ran out of VRAM -> auto-downgrade to faster-whisper.
            log.warning("Downgrading to Faster-Whisper: %s", exc)
            fw_spec = EngineRouter.select(info, prefer=EngineKind.FASTER_WHISPER)
            engine = _get_or_build_engine(fw_spec)
            engine.load()
            spec = fw_spec

        progress(0.35, desc=f"轉錄中 (音檔長度 {duration:.1f} 秒)...")
        # beam_size only applies to the Faster-Whisper engine; HF pipeline
        # ignores it via its own generate_kwargs path.
        transcribe_kwargs: dict = {"word_timestamps": word_timestamps}
        if spec.kind is EngineKind.FASTER_WHISPER:
            transcribe_kwargs["beam_size"] = int(beam_size)
        segments = engine.transcribe(wav_path, **transcribe_kwargs)

        if not segments:
            return (
                "WARN: 轉錄結果為空 (可能是靜音或音量太小)",
                "",
                None,
                None,
                None,
                None,
            )

        progress(0.9, desc="輸出字幕檔...")
        outputs = _write_outputs(
            segments,
            formats,
            meta={
                "engine": spec.kind.value,
                "compute_type": spec.compute_type,
                "duration_sec": round(duration, 2),
            },
        )

        preview = "\n".join(seg.to_timestamp_line() for seg in segments[:20])
        if len(segments) > 20:
            preview += f"\n... (共 {len(segments)} 段)"

        status = (
            f"[OK] 完成！{len(segments)} 段 / {sum(len(s.text) for s in segments)} 字 "
            f"/ 引擎：{spec.kind.value} ({spec.compute_type})"
        )
        return (
            status,
            preview,
            outputs.get("srt"),
            outputs.get("txt"),
            outputs.get("vtt"),
            outputs.get("json"),
        )

    except (ModelLoadError, TranscriptionError) as exc:
        return f"ERROR: 引擎錯誤：{exc}", "", None, None, None, None
    except Exception as exc:  # pragma: no cover
        log.exception("Unexpected UI error")
        return f"ERROR: 未預期錯誤：{exc}", "", None, None, None, None
    finally:
        if wav_path is not None:
            AudioConverter.cleanup(wav_path)


_OUTPUT_DIRS: list[Path] = []
_OUTPUT_DIRS_LOCK = threading.Lock()


def _cleanup_output_dirs() -> None:
    with _OUTPUT_DIRS_LOCK:
        while _OUTPUT_DIRS:
            d = _OUTPUT_DIRS.pop()
            shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup_output_dirs)


def _write_outputs(
    segments: list[TimestampedSegment], formats: list[str], meta: dict[str, Any]
) -> dict[str, str]:
    """Persist requested formats to a temp dir so Gradio can serve them.

    Directories are tracked and deleted on process exit via ``atexit``. Also
    trims to the last 5 sessions so a long-running server doesn't fill disk.
    Gradio can serve concurrent requests -- guard append/pop with a lock so
    racing ``len()``/``pop(0)`` calls don't raise or delete the wrong dir.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="taigi_asr_"))
    with _OUTPUT_DIRS_LOCK:
        _OUTPUT_DIRS.append(out_dir)
        while len(_OUTPUT_DIRS) > 5:
            old = _OUTPUT_DIRS.pop(0)
            shutil.rmtree(old, ignore_errors=True)
    paths: dict[str, str] = {}
    if "SRT" in formats:
        p = out_dir / "transcript.srt"
        p.write_text(to_srt(segments), encoding="utf-8")
        paths["srt"] = str(p)
    if "TXT" in formats:
        p = out_dir / "transcript.txt"
        p.write_text(to_txt(segments), encoding="utf-8")
        paths["txt"] = str(p)
    if "VTT" in formats:
        p = out_dir / "transcript.vtt"
        p.write_text(to_vtt(segments), encoding="utf-8")
        paths["vtt"] = str(p)
    if "JSON" in formats:
        p = out_dir / "transcript.json"
        p.write_text(to_json(segments, meta=meta), encoding="utf-8")
        paths["json"] = str(p)
    return paths


# --------------------------- Blocks UI ------------------------------------- #


HEADER_MD = """# 台灣台語語音轉錄器 (Taigi ASR)

由 **MediaTek Breeze-ASR-26** 驅動，為 **RTX 3050 4GB** 等低 VRAM 環境最佳化。

**最簡用法**：拖放音檔 -> 點「開始轉錄」-> 下載字幕檔。"""


def build_ui() -> gr.Blocks:
    """Construct the Blocks layout. Separated so tests can import without launching."""
    info = GPUProfiler.detect()
    initial_spec = EngineRouter.select(info)

    with gr.Blocks(title="Taigi ASR · 台灣台語轉錄") as demo:
        gr.Markdown(HEADER_MD)

        with gr.Row():
            gr.Markdown(_gpu_panel_text(info, initial_spec))

        audio_input = gr.Audio(
            label="拖放或上傳音檔（支援 m4a / mp3 / wav / mp4 / mov / mkv / flac / ogg / webm）",
            type="filepath",
            sources=["upload"],
        )

        with gr.Accordion("進階選項（一般使用者不用打開）", open=False):
            engine_choice = gr.Radio(
                choices=["Auto / 自動", "Faster-Whisper", "HuggingFace Pipeline"],
                value="Auto / 自動",
                label="推論引擎",
                info="Auto 會依 VRAM 自動選擇。4GB 以下強烈建議 Faster-Whisper。",
            )
            formats = gr.CheckboxGroup(
                choices=["SRT", "TXT", "VTT", "JSON"],
                value=["SRT", "TXT"],
                label="輸出格式",
            )
            word_ts = gr.Checkbox(
                value=False,
                label="Word-level timestamps（依停頓切成短句，略慢）",
            )
            beam_slider = gr.Slider(
                minimum=1,
                maximum=12,
                value=5,
                step=1,
                label="Beam size（辨識精確度；大=準但慢 + 吃 VRAM）",
                info="RTX 3050 4GB 建議 5-10；>12 可能 OOM。僅影響 Faster-Whisper。",
            )

        run_btn = gr.Button("開始轉錄 / Transcribe", variant="primary", size="lg")

        status_md = gr.Markdown("", label="狀態")
        preview_md = gr.Textbox(
            label="結果預覽（前 20 段）",
            lines=12,
            interactive=False,
        )

        with gr.Row():
            dl_srt = gr.File(label="[下載] SRT 字幕", interactive=False)
            dl_txt = gr.File(label="[下載] TXT 文字", interactive=False)
            dl_vtt = gr.File(label="[下載] VTT 字幕", interactive=False)
            dl_json = gr.File(label="[下載] JSON", interactive=False)

        run_btn.click(
            fn=_transcribe,
            inputs=[audio_input, engine_choice, formats, word_ts, beam_slider],
            outputs=[status_md, preview_md, dl_srt, dl_txt, dl_vtt, dl_json],
        )

        gr.Markdown(
            "---\n\n*Model: `MediaTek-Research/Breeze-ASR-26` (HF) · "
            "`paulpengtw/faster-whisper-Breeze-ASR-26` (CT2) · MIT Licensed*"
        )

    return demo
