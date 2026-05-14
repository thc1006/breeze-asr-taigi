# taigi-asr

**台灣台語語音轉錄器 / Taiwanese Hokkien ASR Transcriber**

以 [MediaTek **Breeze-ASR-26**](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) 為核心，專為 **NVIDIA RTX 3050 Laptop 4GB VRAM** 等低顯存環境最佳化，支援句級時間戳記、SRT/VTT/TXT/JSON 多格式輸出、Gradio Web UI、CLI、WSL2/Docker。

---

## 特色

- **台語專用**：基於 Whisper-large-v2 微調，~10,000 小時台語資料（MediaTek 官方）。
- **RTX 3050 4GB 可跑**：int8_float16 量化下峰值 VRAM 約 2.9 GB，留出安全空間。
- **雙引擎自動路由**：依偵測 VRAM 自動選擇 Faster-Whisper (CTranslate2) 或 HuggingFace Pipeline。
- **時間戳記對齊**：句級（預設）與可選逐字（`--word-timestamps`）。
- **零摩擦 UI**：拖放音檔 -> 點「開始轉錄」-> 下載字幕檔。
- **廣泛格式**：`m4a / mp3 / wav / mp4 / mov / mkv / flac / ogg / webm` 全部透過 ffmpeg。
- **完整測試**：unit + smoke + integration（pytest），GitHub Actions CI Linux/Windows 多版本。
- **Docker + WSL2 支援**：CUDA 12.1 runtime + GPU passthrough + 模型 cache volume。
- **講者分群 (`--diarize`, 選用)**：序列載入 pyannote/speaker-diarization-3.1，4 GB VRAM 上 ASR 與 diarize 不共存。輸出帶 `[SPEAKER_xx]` 標籤的 SRT/TXT/VTT/JSON + 標準 RTTM。

## 模型來源（固定，不替代）

| 引擎 | HuggingFace Model ID |
|---|---|
| Faster-Whisper (CT2) | [`paulpengtw/faster-whisper-Breeze-ASR-26`](https://huggingface.co/paulpengtw/faster-whisper-Breeze-ASR-26) |
| HuggingFace Pipeline | [`MediaTek-Research/Breeze-ASR-26`](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) |

---

## 快速開始 / Quick Start

### Windows (native + GPU)

```batch
git clone https://github.com/thc1006/breeze-asr-taigi.git
cd breeze-asr-taigi
install.bat        REM 建 venv + 裝 CUDA 12.1 torch + 下載模型 (~2.9 GB)
start.bat          REM 啟動 Gradio UI + 自動開瀏覽器 http://127.0.0.1:7860
```

### Linux / WSL2 (native)

```bash
git clone https://github.com/thc1006/breeze-asr-taigi.git
cd breeze-asr-taigi
./install.sh        # 建 venv + 裝 CUDA 12.1 torch + 下載模型
./start.sh          # 啟動 Gradio UI
```

### Docker (WSL2 / Linux with NVIDIA Container Toolkit)

```bash
./install.sh --docker
# 或手動：
docker compose up -d
# 開啟 http://localhost:7860
```

---

## CLI 用法

```bash
# 單檔
taigi-asr data/test.m4a --format srt --out out.srt
taigi-asr long_audio.mp3 --engine fw --beam-size 10 --word-timestamps
taigi-asr interview.wav --format json --out interview.json -v

# 多檔批次（模型只載入一次，省 ~9 秒 / 檔）
taigi-asr a.mp3 b.m4a c.wav --format srt,txt
taigi-asr --input-dir music/ --format srt,json
taigi-asr clip1.mp3 --input-dir more_clips/ --format srt   # 兩種來源可混用
```

CLI 選項：
| 參數 | 預設 | 說明 |
|---|---|---|
| `audio` | — | 一或多個音檔路徑（多檔時模型只 load 一次） |
| `--input-dir` | — | 把目錄內所有支援副檔名的音檔加入批次（非遞迴） |
| `--engine` | `auto` | `auto` / `fw` (faster-whisper) / `hf` (huggingface)；可用 `TAIGI_ASR_DEFAULT_ENGINE` 環境變數覆蓋 |
| `--format` | `srt` | `srt` / `txt` / `vtt` / `json`，多格式以逗號串接（例：`srt,txt,json`）|
| `--out` | 自動 | 輸出路徑（**只在單檔 + 單格式時生效**；其他情況輸出落在輸入旁） |
| `--beam-size` | 5 | beam search 寬度（4GB GPU 建議 5-10）|
| `--best-of` | 5 | 溫度採樣候選數 |
| `--word-timestamps` | False | 逐字時間戳記（較慢）|
| `--diarize` | False | 接上 pyannote 講者分群（見下方專節，需 HF_TOKEN + license）|
| `--num-speakers` | — | 指定講者數（與 `--min/--max-speakers` 互斥）|
| `--min-speakers` | — | 講者數下限（搭配 `--diarize`）|
| `--max-speakers` | — | 講者數上限（搭配 `--diarize`）|
| `-v` / `-vv` | WARN | 增加 log 詳細度 |

`--input-dir` 自動撈的副檔名：`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.webm`, `.mp4`, `.mkv`, `.aac`, `.opus`, `.wma`。其他格式（如 `.aiff`）只要 ffmpeg 認得，仍可走 positional 直接傳。

退出碼：
| 代碼 | 含義 |
|---|---|
| `0` | 全部成功 |
| `2` | 找不到輸入檔 / `--input-dir` 不存在 / 沒給任何輸入 |
| `3` | 偵測到的 VRAM 不足以跑指定的 engine |
| `4` | 模型 load 失敗，或所有檔案皆失敗（含「轉錄為空」也計入失敗）|
| `6` | `--format` 指定了未知格式 |
| `7` | 多檔批次中部分檔案失敗（其他成功）|

## Python API

```python
from taigi_asr.audio import AudioConverter
from taigi_asr.engines import build_engine
from taigi_asr.formatters import to_srt
from taigi_asr.router import EngineKind, EngineRouter, GPUProfiler

info = GPUProfiler.detect()
spec = EngineRouter.select(info)           # 自動路由
wav, duration = AudioConverter.convert("audio.m4a")

engine = build_engine(spec)
engine.load()

# beam_size / best_of 只在 Faster-Whisper 引擎支援,
# HuggingFace 引擎的 transcribe() 簽章只吃 word_timestamps,
# 所以用 spec.kind 分流避免 TypeError。
if spec.kind is EngineKind.FASTER_WHISPER:
    segments = engine.transcribe(wav, beam_size=5)
else:
    segments = engine.transcribe(wav)

srt = to_srt(segments)
engine.unload()
```

或是要顯式鎖一個引擎時,直接建構 `FasterWhisperEngine`(不走 router):

```python
from taigi_asr.engines.faster_whisper import FasterWhisperEngine

engine = FasterWhisperEngine(
    device="cuda", compute_type="int8_float16", batch_size=4, beam_size=5
)
engine.load()
segments = engine.transcribe("audio.m4a")
```

---

## VRAM 決策表

偵測到的 VRAM 會自動選擇配置；亦可用 `--engine` 強制覆蓋。

| VRAM | 自動引擎 | compute_type | batch_size | 備註 |
|---|---|---|---|---|
| >= 22 GB (A100/L4) | HuggingFace | float16 | 16 | 最高吞吐 |
| >= 14 GB (4070+) | HuggingFace | float16 | 8 | 預設快 |
| >= 10 GB (3080+) | HuggingFace | float16 | 4 | |
| >= 6 GB (RTX 4060/A2000) | HuggingFace | int8 (bitsandbytes) | 2 | Linux only |
| >= 3.5 GB (**RTX 3050 4GB**) | **Faster-Whisper** | **int8_float16** | **4** | **主力路徑** |
| < 3.5 GB | Faster-Whisper | int8_float16 | 2 | 緊湊配置 |
| 無 CUDA | Faster-Whisper | int8 (CPU) | 1 | 純 CPU |

---

## 效能 (RTX 3050 Laptop 4GB)

在 `int8_float16` + `beam_size=5` + `batch_size=4` 配置下的實測：

| 測試音檔 | 長度 | Transcribe | Peak VRAM | xRT |
|---|---|---|---|---|
| `data/test.m4a` | 5.7 s | 1.9 s | ~2.0 GB | 2.93x |
| `data/test.mp3` | 54 min | **5 min 26 s** | **2.03 GB** | **9.94x** |

> 長音檔 xRT 顯著優於短音檔，因為 Silero VAD 跳過 60-70% 的訪談靜音、且 batched 解碼並行化顯著。Model load (~6-9s) 一次性。

更完整 benchmark 請見 [`docs/benchmarks.md`](docs/benchmarks.md)。

---

## 講者分群 / Speaker Diarization

`--diarize` 在 ASR 之後再跑一遍 [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1)，把每段 ASR 文字附上 `SPEAKER_NN` 標籤。整個 pipeline 是 **序列載入**：ASR 跑完 → `engine.unload()` 釋放 VRAM → pyannote 載入 → diarize → 對齊文字。這樣 4 GB 卡也能跑完整 stack，不會兩個模型一起擠 GPU 而 OOM。

### 一次性設定（HuggingFace token + license）

1. 在 [HuggingFace tokens](https://huggingface.co/settings/tokens) 建一個 read token，設環境變數 `HF_TOKEN`（或 `HUGGINGFACE_HUB_TOKEN`）。
2. 用同一個 HF 帳號開瀏覽器，在以下兩頁各按一次 **Agree and access repository**：
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

漏掉任一步，CLI 會在 `dia.load()` 時印 `HF_TOKEN env var required ...` 或 `Failed to load pyannote/speaker-diarization-3.1: ...` 並回 exit code 4，但已轉錄好的 ASR 文字仍會以無 speaker 標籤的形式寫出，不會白費 GPU 時間。

### 兩條常用路線

| 場景 | 指令 | 100min 預估 wall | 輸出 |
|---|---|---|---|
| **快路**（純文字，趕時間） | `taigi-asr <audio> --format txt` | ~10–12 min | `<audio>.txt`（`[時間] 內容`，無 speaker） |
| **完整 E2E**（含 diarize） | `taigi-asr <audio> --diarize --format srt,txt,json` | ~15–18 min | `.srt` / `.txt` / `.json`（每段前置 `[SPEAKER_xx]`）+ `.rttm` |

> **標點**：Breeze-ASR-26 對台語/中文輸出以空格分隔短語、**很少帶 ，。？！**。若要還原句末標點，需另外接 punctuation-restoration 模型（不在本專案範圍）。

### 講者數約束

| Flag | 用途 | 適用情境 |
|---|---|---|
| 預設（不加 flag） | pyannote 自動估計 | 不確定講者數 / 內容混雜，可能輕度過分群 |
| `--num-speakers N` | 強制 N 位 | 明確知道有幾人（訪談 1+1、會議名單） |
| `--min-speakers M --max-speakers N` | 範圍 | 知道大致範圍但不固定 |

**經驗法則（100min 演講錄音實測，1 主講 + 1 主持 + 零星觀眾發問）**：

| 變體 | 設定 | 結果 | 評價 |
|---|---|---|---|
| auto | 不加 flag | 5 講者，主講 74%、4 個 <30s 雜訊群 | 輕度過分群 |
| binary | `--num-speakers 2` | 2 講者，99.9% vs 0.1%（主持人被併入主講） | **不建議** |
| ternary | `--num-speakers 3` | 3 講者，74% / 0.7% / 0.1%（主講 / 主持 / 雜音） | **最貼近 ground truth** |

少數人對談（1 主 + 1 客）一般 `--num-speakers 2` 直接給即可；4 人以上互動，先試 auto 再依需要收斂。

### 端點 4 GB GPU 觀測值

90 秒 clip 的實測 GPU 使用率（`nvidia-smi -l 2`）：

| 階段 | GPU% | VRAM |
|---|---|---|
| ASR 推論高峰 | 100% | 3.0 GB |
| ASR unload 後 | — | 0.5 GB |
| Diarize embedding 高峰 | 99% | 2.4 GB |

100 min 音檔 E2E 約 15–18 分鐘完成（ASR ~10 min + diarize ~5 min + 載入/切換 ~1 min），peak VRAM 從未超過 3.1 GB。

### 獨立工具

只想 diarize（不重跑 ASR）、或想比較不同講者數約束：

```bash
# 把 ASR 跑出來的 SRT + 既有 RTTM 合併為帶 speaker 標籤的字幕
# 輸出：<audio>.diarized.srt / .txt / .json（與輸入同目錄；可用 --out-prefix 改）
python scripts/merge_diarize.py audio.srt audio.rttm

# 同一個音檔，比較 binary vs ternary 兩種約束（模型只 load 一次）
# 輸出：<audio>.binary.rttm / .ternary.rttm + .speakers.txt（與音檔同目錄）
python scripts/diarize_compare.py audio.m4a
```

### Diarize 退出碼

| 代碼 | 含義 |
|---|---|
| `0` | 全部成功（含 speaker 標籤） |
| `4` | 單檔或整批檔案都失敗（含 `dia.load()` token/license/網路錯，或所有檔案 diarize 都掛）。ASR 文字仍以無 speaker 形式寫出 |
| `7` | 多檔批次中部分檔案 diarize 失敗（其他成功） |

### 邊角情況

- **沒有 CUDA / CPU 環境**：`DiarizationPipeline` 目前固定 `device="cuda"`，CPU-only 環境會在 load 時拋出 CUDA error 並進入上方 exit 4 fallback；ASR 仍可以 CPU 跑（`--engine fw` 自動降級為 `int8` batch=1）。
- **多檔批次 + `--diarize`**：CLI 採 **兩 pass 編排** — Phase 1 把所有檔案 ASR 完，unload FW 釋放 VRAM；Phase 2 載入 pyannote、逐檔 diarize + 對齊。各檔的轉錄會緩存在記憶體直到 Phase 2 寫出（單檔 ~60 KB；100 檔 ~6 MB 可接受）。

---

## 專案結構

```
src/taigi_asr/
  segments.py         # TimestampedSegment dataclass (含 optional speaker)
  formatters.py       # to_txt / to_srt / to_vtt / to_json
  audio.py            # AudioConverter (16 kHz mono)
  router.py           # GPUProfiler + EngineRouter
  diarize.py          # DiarizationPipeline + attribute_speakers + RTTM I/O
  engines/
    base.py           # ASREngine Protocol
    faster_whisper.py # FasterWhisperEngine (CT2)
    huggingface.py    # HuggingFaceEngine (transformers)
    fake.py           # FakeEngine (tests)
  ui/
    gradio_app.py     # Gradio Blocks
    launcher.py       # python -m taigi_asr.ui.launcher
  cli.py              # python -m taigi_asr.cli（含 --diarize 兩 pass 編排）
scripts/
  merge_diarize.py    # SRT + RTTM → 帶 speaker 標籤的 SRT/TXT/JSON
  diarize_compare.py  # 一次跑 binary/ternary 兩種 speaker constraints
  diarize_poc.py      # 單次 diarize 試跑（auto-detect 講者數）
tests/
  unit/               # unit tests (CPU-only, <3s)
  smoke/              # CLI + UI smoke tests
  integration/        # Real model on test.m4a (marked slow)
```

---

## 開發

```bash
pip install -e ".[dev,hf]"
pytest tests/unit tests/smoke       # 快速
pytest -m slow                       # integration (需 GPU + 模型)
ruff check . && ruff format --check .
pre-commit install
```

如需開啟 `--diarize` 開發：

```bash
pip install "pyannote.audio<4"      # v3 系列相容 torch 2.6 CUDA
export HF_TOKEN=hf_xxx               # 接受 license（見「講者分群」節）
python -m taigi_asr.cli sample.m4a --diarize --format srt,txt,json
```

---

## 疑難排解 / FAQ

見 [`docs/faq.md`](docs/faq.md)：
- CUDA not found / WSL2 GPU passthrough
- OOM on 4GB
- bitsandbytes Windows 失敗
- torch.compile 錯誤
- 音檔格式不支援

---

## 致謝

- [MediaTek Research](https://huggingface.co/MediaTek-Research) - Breeze-ASR-26 官方模型
- [SYSTRAN / faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 推論框架
- [paulpengtw](https://huggingface.co/paulpengtw) - CT2 預轉換模型
- [OpenAI Whisper](https://github.com/openai/whisper) - 底層架構

## License

MIT. See [LICENSE](LICENSE). 模型授權請見各 HuggingFace 模型頁。
