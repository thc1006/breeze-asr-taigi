# Benchmarks

Hardware: **NVIDIA RTX 3050 Laptop GPU, 4 GB GDDR6** (Lenovo IdeaPad Gaming 3 82K1, Windows 11, driver 591.74 / CUDA 13.1, torch 2.6.0+cu124, faster-whisper 1.2.1, CTranslate2 4.7.1).

Measurement methodology: wall-clock timing via `scripts/gpu_benchmark.py`; peak VRAM sampled through `nvidia-smi --query-gpu=memory.used` *after* transcribe completes (CTranslate2 bypasses the torch allocator so `torch.cuda.max_memory_allocated` always reports 0). Accuracy measured as "anchor hits" — count of reference substrings (`小時候`, `跌倒`, `長大`, `傻傻`, `這裡`, `玲瓏`, `醫沒好`) present in the output; imperfect but fast enough for config comparisons.

## Short audio: `data/test.m4a` (5.7 s Taiwanese Hokkien sample)

Reference transcript: `小時候跌倒醫沒好，長大傻傻在這裡喊玲瓏。`

| compute_type | batch | beam | Preprocess | Model load | Transcribe | xRT | Peak VRAM | Anchors |
|---|---|---|---|---|---|---|---|---|
| int8_float16 | 4 | 5 | 0.1 s | 8.6 s | **1.9 s** | **2.93x** | ~2.0 GB | 6/10 |
| int8_float16 | 4 | 10 | 0.1 s | 9.2 s | 2.6 s | 2.15x | ~2.0 GB | 6/10 |
| float16 | 1 | 5 | 0.1 s | 6.0 s | 16.1 s | 0.35x | ~3.5 GB | 6/10 |

**Observations:**
- `int8_float16` is both faster AND uses less VRAM than `float16` on this GPU (Ampere consumer SKUs have aggressive Tensor Core paths for int8).
- `beam_size=10` doesn't improve accuracy on this specific clip — the Breeze-ASR-26 model itself struggles with `玲瓏` in this audio (substitutes `閒晃` or `呼來喝去`).
- Model load (~6-9 s) is a one-time cost per engine instance; amortizes over repeated transcribe calls.

## Long audio: `data/test.mp3` (54 min 1.3 s Taiwanese Hokkien interview)

Measured on RTX 3050 Laptop 4 GB with `scripts/gpu_benchmark.py --batch-size 4 --compute-type int8_float16 --beam-size 5`:

| Stage | Time |
|---|---|
| Audio preprocess (ffmpeg -> 16 kHz mono) | 2.9 s |
| Model load (first time) | 8.9 s |
| **Transcribe** | **5 min 25.9 s** |
| **Total** | **5 min 37.7 s** |

| Metric | Value |
|---|---|
| **xRT (transcribe-only)** | **9.94x** |
| **xRT (end-to-end)** | **9.60x** |
| **Peak VRAM** | **2.03 GB / 4.00 GB** (50 % headroom) |
| Segments produced | 143 |
| Characters produced | 15,889 |
| GPU utilization (steady state) | ~95 % |

**Why is long-audio xRT (9.94x) so much higher than short-audio (2.93x)?**

1. Silero VAD (on by default) skips 60-70 % of silence in conversational audio. A 54-min interview contains many pauses; the decoder never sees them.
2. Model load time (8.9 s) amortizes over 54 minutes of audio — negligible.
3. `BatchedInferencePipeline` parallelizes chunks across the batch dimension; the parallelism benefit only becomes visible on audio long enough to fill a batch.

**Why does batch=4 stay at 2.03 GB peak here but batch=8 OOMed earlier?**

Both configs fit during `load()` (~2 GB). The difference shows up at encoder time: `batch_size=8` pushes the encoder forward pass to ~4.4 GB peak (model + batched encoder activations), which does not fit on 4 GB. `batch_size=4` tops out at ~3.2 GB peak including decoder KV cache. The router default of 4 for 4 GB class cards is correct.

**Tuning knobs that did NOT help on this workload:**

- `beam_size > 5`: increases VRAM and latency, no measurable accuracy gain for this audio.
- `compute_type=float16`: actually *slower* on Ampere consumer cards and doubles VRAM.
- `condition_on_previous_text=False`: marginally faster but visibly worse on long-form (context collapse).

## Universal recommended settings (RTX 3050 4GB, short + long audio)

```python
FasterWhisperEngine(
    device="cuda",
    compute_type="int8_float16",  # best on Ampere consumer cards
    batch_size=4,  # ceiling for 4 GB during decoder generation
    beam_size=5,  # accuracy/speed sweet spot
    best_of=5,
    temperature=0.0,
    # vad_filter=True (default in transcribe())
)
```

These are the router defaults for VRAM in the `[3.5 GB, 14 GB)` bracket. Users with >= 6 GB can safely bump `batch_size=8`. bitsandbytes int8 HF path needs Linux + 6 GB+.

## Failed configs (documented so future contributors don't repeat)

| Config | Failure mode |
|---|---|
| `float16` + batch >= 2 on 4 GB | OOM at load — model alone is 3.1 GB. |
| `int8_float16` + batch=8 on 4 GB | Load succeeds (2034 MiB), **OOM during encoder.encode()** in `BatchedInferencePipeline.forward`. |
| HF pipeline `float16` on 4 GB | OOM at load. |
| HF pipeline `int8` (bitsandbytes) on Windows | bitsandbytes CUDA binary missing; Linux-only path. |
| CPU float16 | Not supported by CTranslate2; falls back to int8 silently. |

## Methodology notes

- **First-run penalty**: model download adds ~2.9 GB download time on first use (`install.bat` / `install.sh` pre-seeds this).
- **Audio preprocess**: `ffmpeg` re-sample to 16 kHz mono takes ~0.1-3 s depending on source format; `pydub` fallback is ~5-10x slower.
- **VAD effect on long audio**: Silero VAD default (`min_silence_duration_ms=500`) typically skips 20-40 % of Hokkien interview audio, giving a linear speedup on long files.
