# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZipVoice is a zero-shot text-to-speech (TTS) system based on flow matching with a Zipformer backbone (123M params). Supports Chinese and English, single-speaker and dialogue generation. Developed by Xiaomi Corp / k2-fsa.

This fork (`streaming/ZipVoice`) focuses on **streaming inference** — the active development area is the fixed-window cross-attention streaming variant.

## Running Commands

There is no `setup.py` — the project is used by adding the repo root to `PYTHONPATH`. All entry points are invoked as `python3 -m zipvoice.bin.<script>`.

### Training (streaming cross-attention variant, current focus)

```bash
# From repo root (or egs/zipvoice/ with PYTHONPATH=../../:$PYTHONPATH)
export PYTHONPATH="$PYTHONPATH:$(pwd)"
python3 -m zipvoice.bin.train_zipvoice_stream_fixedwindow_crossattn \
    --world-size 4 --num-epochs 200 --max-duration 250 \
    --model-config conf/zipvoice_base.json \
    --tokenizer libritts --token-file data/tokens_libritts.txt \
    --dataset libritts --manifest-dir aligned_data/fbank \
    --exp-dir exp/<experiment_name> --feat-scale 0.1
```

### Inference (streaming cross-attention variant)

```bash
python3 -m zipvoice.bin.infer_zipvoice_stream_fixed_window_crossattn \
    --model-name zipvoice --model-dir exp/<experiment_name> \
    --checkpoint-name <checkpoint>.pt --tokenizer libritts \
    --test-list test.tsv --res-dir res/<output_dir> \
    --num-step 16 --guidance-scale 2 --t-shift 0.7 --feat-scale 0.1
```

### Standard (non-streaming) training and inference

See `egs/zipvoice/run_libritts.sh` for the full 9-stage pipeline: data prep → train → checkpoint averaging → inference. Key stages:
- Stage 2: `python3 -m zipvoice.bin.train_zipvoice` (base model)
- Stage 3: `python3 -m zipvoice.bin.generate_averaged_model` (checkpoint averaging)
- Stages 4-7: Two-stage distillation via `zipvoice.bin.train_zipvoice_distill`
- Stages 8-9: Inference via `zipvoice.bin.infer_zipvoice`

### Developer scripts

`run/train.sh` and `run/test.sh` are the primary scripts for the streaming experiments. They set `CUDA_VISIBLE_DEVICES`, create experiment directories, and copy configs for reproducibility.

## Architecture

### Model hierarchy

All models are in `zipvoice/models/` and inherit from the base `ZipVoice` class:

- `zipvoice.py` → `ZipVoice` — base model: text encoder (small Zipformer) + flow-matching decoder (larger Zipformer) + Vocos vocoder
- `zipvoice_distill.py` → `ZipVoiceDistill(ZipVoice)` — distilled variant with guidance-scale embedding, fewer ODE steps (4 vs 8-16)
- `zipvoice_dialog.py` → `ZipVoiceDialog(ZipVoice)` — two-speaker dialogue using `TTSZipformerTwoStream`
- `zipvoice_stream_fixedwindow_crossattn.py` — **active streaming variant** using cross-attention Zipformer

### Core modules (`zipvoice/models/modules/`)

- `zipformer.py` → `TTSZipformer` — multi-scale Zipformer with downsampling/upsampling stacks, sinusoidal timestep embeddings
- `zipformer_crossattn.py` — cross-attention variant for streaming (attends to fixed windows of already-generated context)
- `zipformer_two_stream.py` → `TTSZipformerTwoStream` — two-stream variant for dialogue
- `solver.py` → `EulerSolver`, `DistillEulerSolver`, `DiffusionModel` — ODE solvers for flow-matching; `DiffusionModel` handles classifier-free guidance
- `solver_stream.py`, `solver_stream_pipeline.py` — streaming solver variants
- `scaling.py` — custom scaling ops from icefall/k2 (ScaledLinear, BiasNorm, Balancer, Whiten, SwooshR activation)

### Inference data flow

1. Text → tokenizer (phonemes/pinyin/characters) → text encoder (small Zipformer) → text conditioning
2. Gaussian noise initialized for mel-spectrogram shape
3. Euler solver iteratively denoises through FM decoder, conditioned on text + speech prompt
4. Vocos vocoder converts mel-spectrogram → waveform (24kHz)

### Data pipeline

Uses **lhotse** for data management: CutSets, manifests, feature extraction, dynamic bucketing samplers. Mel-spectrograms (100-dim fbank) are precomputed via `zipvoice.bin.compute_fbank`. Alignment data is stored in `aligned_data/fbank/`.

### Entry points (`zipvoice/bin/`)

Training: `train_zipvoice.py`, `train_zipvoice_distill.py`, `train_zipvoice_dialog.py`, `train_zipvoice_stream_fixedwindow_crossattn.py`
Inference: `infer_zipvoice.py`, `infer_zipvoice_stream_fixed_window_crossattn.py`, `infer_zipvoice_dialog.py`
Data prep: `compute_fbank.py`, `prepare_tokens.py`, `prepare_dataset.py`, `align_data_save.py`
Export: `onnx_export.py`, `tensorrt_export.py`

## Configuration

Model architecture is defined in JSON config files (e.g., `egs/zipvoice/conf/zipvoice_base.json`). Training hyperparameters are passed as CLI args. The config specifies FM decoder and text encoder dimensions, layer counts, kernel sizes, and feature settings (100-dim mel, 24kHz, Vocos vocoder).

## Code Style

- Formatting: black (line-length 88) + isort (black profile) per `pyproject.toml`
- No formal test suite — validation is done through inference scripts and evaluation pipelines in `egs/`

## Key Dependencies

- PyTorch + torchaudio (deep learning)
- lhotse (audio data management, CutSets, samplers)
- vocos (neural vocoder, mel → waveform)
- piper_phonemize, pypinyin, jieba (text tokenization for English/Chinese)
- safetensors, huggingface_hub (model checkpoints)

## Deployment

`runtime/nvidia_triton/` contains a complete Triton Inference Server deployment with TensorRT acceleration, Docker support, and HTTP/gRPC clients.
