#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script generates speech with our pre-trained ZipVoice or
    ZipVoice-Distill models. If no local model is specified,
    Required files will be automatically downloaded from HuggingFace.

Usage:

Note: If you having trouble connecting to HuggingFace,
    try switching endpoint to mirror site:
export HF_ENDPOINT=https://hf-mirror.com

(1) Inference of a single sentence:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav

(2) Inference of a list of sentences:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice \
    --test-list test.tsv \
    --res-dir results

`--model-name` can be `zipvoice` or `zipvoice_distill`,
    which are the models before and after distillation, respectively.

Each line of `test.tsv` is in the format of
    `{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}`.


(3) Inference with TensorRT:

python3 -m zipvoice.bin.infer_zipvoice \
    --model-name zipvoice_distill \
    --prompt-wav prompt.wav \
    --prompt-text "I am a prompt." \
    --text "I am a sentence." \
    --res-wav-path result.wav \
    --trt-engine-path models/zipvoice_distill_onnx_trt/fm_decoder.fp16.plan
"""

import argparse
import datetime as dt
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional
import re
import time
import matplotlib.pyplot as plt

import numpy as np
import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.word_pointer import WordPointer
from zipvoice.models.zipvoice_stream_fixedwindow_crossattn import ZipVoice
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, str2bool
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    batchify_tokens,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)
from zipvoice.utils.tensorrt import load_trt

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="The model used for inference",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="The model directory that contains model checkpoint, configuration "
        "file model.json, and tokens file tokens.txt. Will download pre-trained "
        "checkpoint from huggingface if not specified.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="model.pt",
        help="The name of model checkpoint.",
    )

    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="The vocoder checkpoint. "
        "Will download pre-trained vocoder from huggingface if not specified.",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="emilia",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="The list of prompt speech, prompt_transcription, "
        "and text to synthesizein the format of "
        "'{wav_name}\t{prompt_transcription}\t{prompt_wav}\t{text}'.",
    )

    parser.add_argument(
        "--prompt-wav",
        type=str,
        default=None,
        help="The prompt wav to mimic",
    )

    parser.add_argument(
        "--prompt-text",
        type=str,
        default=None,
        help="The transcription of the prompt wav",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="The text to synthesize",
    )

    parser.add_argument(
        "--res-dir",
        type=str,
        default="results",
        help="""
        Path name of the generated wavs dir,
        used when test-list is not None
        """,
    )

    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="result.wav",
        help="""
        Path name of the generated wav path,
        used when test-list is None
        """,
    )

    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="The scale of classifier-free guidance during inference.",
    )

    parser.add_argument(
        "--num-step",
        type=int,
        default=None,
        help="The number of sampling steps.",
    )

    parser.add_argument(
        "--feat-scale",
        type=float,
        default=0.1,
        help="The scale factor of fbank feature",
    )

    parser.add_argument(
        "--word-pointer-ckpt",
        type=Path,
        default=None,
        help="Path to word_pointer.pt. Required for streaming; the pointer "
             "head predicts how many target words the just-generated chunk "
             "covered, replacing the duration-ratio heuristic.",
    )

    parser.add_argument(
        "--word-pointer-max-pad",
        type=int,
        default=4,
        help="Should match the max_pad used at training time. Used only as "
             "a sanity check against the value stored in the checkpoint.",
    )

    parser.add_argument(
        "--word-pointer-min-frames",
        type=int,
        default=150,
        help="Minimum cumulative generated mel frames before invoking the "
             "WordPointer (it was trained on chunks of this many frames). "
             "Below this, fall back to the ratio heuristic.",
    )

    parser.add_argument(
        "--lookahead-words",
        type=int,
        default=3,
        help="Number of future target words exposed beyond the estimated "
             "current-window word position in each streaming chunk.",
    )

    parser.add_argument(
        "--history-context-chunks",
        type=int,
        default=-1,
        help="Number of previous generated streaming chunks kept as speech "
             "context for the next chunk. -1 keeps all generated history, "
             "0 uses only the original prompt speech.",
    )

    parser.add_argument(
        "--advance-mode",
        type=str,
        default="word_pointer",
        choices=[
            "word_pointer",
            "wp_right_only",
            "wp_ratio_clamped",
            "ratio",
            "fixed_words",
        ],
        help="Text advancement strategy for streaming inference. "
             "'word_pointer' uses the trained pointer; 'ratio' advances by "
             "generated-frame ratio; 'fixed_words' advances a fixed estimated "
             "number of words per chunk; 'wp_right_only' uses WordPointer only "
             "to estimate how many words at the right edge remain unspoken; "
             "'wp_ratio_clamped' uses ratio as the primary progress estimate "
             "and lets WordPointer adjust it within a small range.",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Control speech speed, 1.0 means normal, >1.0 means speed up",
    )

    parser.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="Shift t to smaller ones if t_shift < 1.0",
    )

    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.1,
        help="Target speech normalization rms value, set to 0 to disable normalization",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="Random seed",
    )

    parser.add_argument(
        "--num-thread",
        type=int,
        default=1,
        help="Number of threads to use for PyTorch on CPU.",
    )

    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index used for inference when CUDA is available.",
    )

    parser.add_argument(
        "--raw-evaluation",
        type=str2bool,
        default=False,
        help="Whether to use the 'raw' evaluation mode where provided "
        "prompts and text are fed to the model without pre-processing",
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=100,
        help="Maximum duration (seconds) in a single batch, including "
        "durations of the prompt and generated wavs. You can reduce it "
        "if it causes CUDA OOM.",
    )

    parser.add_argument(
        "--remove-long-sil",
        type=str2bool,
        default=False,
        help="Whether to remove long silences in the middle of the generated "
        "speech (edge silences will be removed by default).",
    )

    parser.add_argument(
        "--trt-engine-path",
        type=str,
        default=None,
        help="The path to the TensorRT engine file.",
    )

    parser.add_argument(
        "--debug-plot-dir",
        type=Path,
        default=None,
        help="Optional directory to save per-step debug plots.",
    )

    parser.add_argument(
        "--debug-plot-every",
        type=int,
        default=1,
        help="Save debug plots every N chunks when --debug-plot-dir is set.",
    )

    parser.add_argument(
        "--save-chunk-wavs",
        type=str2bool,
        default=False,
        help="Whether to save each decoded streaming chunk as a separate wav.",
    )

    parser.add_argument(
        "--first-chunk-only",
        type=str2bool,
        default=False,
        help="Stop after decoding the first streaming chunk. This is intended "
             "for measuring first-chunk synthesis latency.",
    )

    parser.add_argument(
        "--trim-tail-noise",
        type=str2bool,
        default=True,
        help="Trim trailing broadband hiss that can appear when the streaming "
             "window runs out of text.",
    )

    parser.add_argument(
        "--tail-noise-min-ms",
        type=float,
        default=120.0,
        help="Minimum detected trailing-noise duration before trimming.",
    )

    parser.add_argument(
        "--tail-noise-keep-ms",
        type=float,
        default=40.0,
        help="Keep this much audio before the detected trailing-noise region.",
    )

    return parser


def _safe_plot_name(text: str, max_len: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)[:max_len]


def _to_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


def _save_prompt_debug_plot(
    debug_dir: Optional[Path],
    sample_tag: str,
    prompt_wav: torch.Tensor,
    prompt_features: torch.Tensor,
    sampling_rate: int,
) -> None:
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"{sample_tag}_prompt_overview.png"

    prompt_wav_np = prompt_wav[0].detach().cpu().numpy()
    prompt_feat_rms = torch.sqrt(torch.mean(torch.exp(prompt_features) ** 2, dim=-1))
    prompt_feat_rms_np = prompt_feat_rms[0].detach().cpu().numpy()

    t_wave = np.arange(prompt_wav_np.shape[0]) / sampling_rate
    t_feat = np.arange(prompt_feat_rms_np.shape[0]) * 256 / sampling_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(t_wave, prompt_wav_np, linewidth=0.8)
    axes[0].set_title("Prompt waveform")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t_feat, prompt_feat_rms_np, linewidth=0.8)
    axes[1].set_title("Prompt mel RMS")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RMS")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_stream_debug_plot(
    debug_dir: Optional[Path],
    sample_tag: str,
    step_idx: int,
    target_words: list,
    text_chunk_str: str,
    expected_word_pos: int,
    lookahead_end: int,
    committed_word_pos: int,
    generated_new_frames: int,
    est_total_new_frames: int,
    fixed_chunk_frames: int,
    pred_features_lens: int,
    actual_chunk_frames: int,
    src: str,
) -> None:
    print(f"debug_dir:{debug_dir}, step_idx:{step_idx}, expected_word_pos:{expected_word_pos}, ")
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    words_total = len(target_words)
    ax0.axhline(words_total, color="gray", linestyle="--", linewidth=1.0, label="total words")
    ax0.axhline(committed_word_pos, color="tab:green", linestyle=":", linewidth=1.0)
    ax0.bar([step_idx - 0.25], [expected_word_pos], width=0.2, color="tab:blue", alpha=0.65, label="expected")
    ax0.bar([step_idx], [lookahead_end], width=0.2, color="tab:orange", alpha=0.65, label="lookahead_end")
    ax0.bar([step_idx + 0.25], [committed_word_pos], width=0.2, color="tab:green", alpha=0.75, label="committed")
    ax0.set_title(f"Word progression | step={step_idx} src={src} | chunk='{text_chunk_str}'")
    ax0.set_xlabel("Step index")
    ax0.set_ylabel("Word position")
    ax0.set_xlim(-0.5, max(3, step_idx + 1.5))
    ax0.set_ylim(0, max(words_total + 1, committed_word_pos + 3, lookahead_end + 3))
    ax0.legend(loc="upper left")
    ax0.grid(True, axis="y", alpha=0.2)

    ax1 = fig.add_subplot(gs[1, 0])
    xs = [0, 1, 2, 3]
    ys = [generated_new_frames, est_total_new_frames, fixed_chunk_frames, pred_features_lens]
    labels = ["generated", "estimated_total", "fixed_chunk", "pred_len"]
    colors = ["tab:green", "tab:red", "tab:blue", "tab:purple"]
    ax1.bar(xs, ys, color=colors, alpha=0.75)
    ax1.set_xticks(xs, labels)
    ax1.set_ylabel("Frames")
    ax1.set_title("Frame budget")
    ax1.grid(True, axis="y", alpha=0.2)
    ax1.text(
        0.02,
        0.95,
        f"actual_chunk_frames={actual_chunk_frames}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"),
    )

    fig.tight_layout()
    out_path = debug_dir / f"{sample_tag}_step_{step_idx:03d}_progress.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_word_pointer_heatmap(
    debug_dir: Optional[Path],
    sample_tag: str,
    step_idx: int,
    wp_probs: Optional[torch.Tensor],
    wp_max_pad: int,
    pred_l: Optional[int],
    pred_r: Optional[int],
    lookahead_end: int,
    committed_word_pos: int,
    text_chunk_str: str,
) -> None:
    if debug_dir is None or wp_probs is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)
    probs = wp_probs.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(probs, origin="lower", cmap="viridis")
    ax.set_xticks(range(wp_max_pad + 1))
    ax.set_yticks(range(wp_max_pad + 1))
    ax.set_xticklabels([str(i) for i in range(wp_max_pad + 1)])
    ax.set_yticklabels([str(i) for i in range(wp_max_pad + 1)])
    ax.set_xlabel("pad_right")
    ax.set_ylabel("pad_left")
    ax.set_title(f"WordPointer probs | step={step_idx} chunk='{text_chunk_str}'")
    fig.colorbar(im, ax=ax, shrink=0.85)

    if pred_l is not None and pred_r is not None:
        ax.scatter([pred_r], [pred_l], s=120, facecolors="none", edgecolors="red", linewidths=2)
        ax.text(
            pred_r + 0.05,
            pred_l + 0.05,
            f"({pred_l},{pred_r})",
            color="white",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.5),
        )

    ax.text(
        1.02,
        0.02,
        f"lookahead_end={lookahead_end}\ncommitted={committed_word_pos}",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"),
    )

    fig.tight_layout()
    out_path = debug_dir / f"{sample_tag}_step_{step_idx:03d}_wordpointer.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_chunk_debug_plot(
    debug_dir: Optional[Path],
    sample_tag: str,
    step_idx: int,
    model_chunk: torch.Tensor,
    wav: torch.Tensor,
    text_chunk_str: str,
    sampling_rate: int,
) -> None:
    if debug_dir is None:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)

    mel = model_chunk[0].detach().cpu().numpy()
    mel_for_display = (mel.T - mel.mean()) / (mel.std() + 1e-6)
    mel_rms = torch.sqrt(torch.mean(torch.exp(model_chunk) ** 2, dim=-1))
    mel_rms_np = mel_rms[0].detach().cpu().numpy()
    wav_np = wav[0].detach().cpu().numpy()

    t_mel = np.arange(mel.shape[0]) * 256 / sampling_rate
    t_wav = np.arange(wav_np.shape[0]) / sampling_rate

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    axes[0].imshow(mel_for_display, aspect="auto", origin="lower")
    axes[0].set_title(f"Generated mel chunk | step={step_idx} | chunk='{text_chunk_str}'")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Mel bin")

    axes[1].plot(t_mel, mel_rms_np, linewidth=0.8)
    axes[1].set_title("Generated mel RMS")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RMS")
    axes[1].grid(True, axis="y", alpha=0.2)

    axes[2].plot(t_wav, wav_np, linewidth=0.8)
    axes[2].set_title("Decoded waveform")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True, axis="y", alpha=0.2)

    fig.tight_layout()
    out_path = debug_dir / f"{sample_tag}_step_{step_idx:03d}_chunk.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_stream_summary_plot(
    debug_dir: Optional[Path],
    sample_tag: str,
    history: list,
    total_target_words: int,
    est_total_new_frames: int,
) -> None:
    if debug_dir is None or not history:
        return

    debug_dir.mkdir(parents=True, exist_ok=True)

    steps = [h["step"] for h in history]
    expected = [h["expected_word_pos"] for h in history]
    lookahead = [h["lookahead_end"] for h in history]
    committed = [h["committed_word_pos"] for h in history]
    generated_frames = [h["generated_new_frames"] for h in history]
    pred_lens = [h["pred_features_lens"] for h in history]
    src_labels = [h["src"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(steps, expected, marker="o", label="expected")
    axes[0].plot(steps, lookahead, marker="o", label="lookahead_end")
    axes[0].plot(steps, committed, marker="o", label="committed")
    axes[0].axhline(total_target_words, color="gray", linestyle="--", label="total words")
    axes[0].set_ylabel("Word position")
    axes[0].set_title("Streaming word progression")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper left")

    for step, y, src in zip(steps, committed, src_labels):
        axes[0].text(step, y + 0.15, src, fontsize=8, ha="center")

    axes[1].plot(steps, generated_frames, marker="o", label="generated frames")
    axes[1].plot(steps, pred_lens, marker="o", label="predicted chunk len")
    axes[1].axhline(est_total_new_frames, color="gray", linestyle="--", label="estimated total")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Frames")
    axes[1].set_title("Frame progression")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    out_path = debug_dir / f"{sample_tag}_summary.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def trim_trailing_noise(
    wav: torch.Tensor,
    mel: torch.Tensor,
    sampling_rate: int,
    min_noise_ms: float = 20.0,
    rms_tol: float = 1.0e-4,
) -> tuple[torch.Tensor, int]:
    """Trim from the first generated-mel RMS=1 plateau onward.

    Padded all-zero mel frames have ``sqrt(mean(exp(0)^2)) == 1`` and decode
    to a deterministic noise tail. Scan forward and cut the waveform at the
    first sufficiently long RMS=1 plateau.
    """
    if wav.numel() == 0 or mel.numel() == 0:
        return wav, 0

    squeeze = wav.dim() == 1
    x = wav.unsqueeze(0) if squeeze else wav

    mel_rms = torch.sqrt(torch.mean(torch.exp(mel.detach().float().cpu()) ** 2, dim=-1))
    plateau = torch.isclose(
        mel_rms[0],
        torch.ones((), dtype=mel_rms.dtype),
        rtol=0.0,
        atol=rms_tol,
    )
    min_frames = max(
        1,
        int(math.ceil(min_noise_ms * sampling_rate / (1000.0 * 256))),
    )

    plateau_start = None
    run = 0
    for idx, is_plateau in enumerate(plateau.tolist()):
        if is_plateau:
            run += 1
            if run >= min_frames:
                plateau_start = idx - run + 1
                break
        else:
            run = 0

    if plateau_start is None:
        return wav, 0

    samples_per_frame = x.size(-1) / max(1, mel.size(1))
    cut_sample = max(0, int(round(plateau_start * samples_per_frame)))
    trimmed_samples = x.size(-1) - cut_sample
    if trimmed_samples <= 0:
        return wav, 0

    trimmed = x[..., :cut_sample]
    if squeeze:
        trimmed = trimmed.squeeze(0)
    return trimmed.to(device=wav.device, dtype=wav.dtype), int(trimmed_samples)


def get_vocoder(vocos_local_path: Optional[str] = None):
    if vocos_local_path:
        vocoder = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_local_path}/pytorch_model.bin",
            weights_only=True,
            map_location="cpu",
        )
        vocoder.load_state_dict(state_dict)
    else:
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return vocoder


def generate_sentence_raw_evaluation(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function directly feed the prompt_text, prompt_wav and text to the model.
        It is not efficient and can have poor results for some inappropriate inputs.
        (e.g., prompt wav contains long silence, text to be generated is too long)
        This function can be used to evaluate the "raw" performance of the model.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)

    # Convert text to tokens
    tokens = tokenizer.texts_to_token_ids([text])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])

    # Start timing
    start_t = dt.datetime.now()

    # Generate features
    (
        pred_features,
        pred_features_lens,
        pred_prompt_features,
        pred_prompt_features_lens,
    ) = model.sample(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features=prompt_features,
        prompt_features_lens=prompt_features_lens,
        speed=speed,
        t_shift=t_shift,
        duration="predict",
        num_step=num_step,
        guidance_scale=guidance_scale,
    )

    # Postprocess predicted features
    pred_features = pred_features.permute(0, 2, 1) / feat_scale  # (B, C, T)

    # Start vocoder processing
    start_vocoder_t = dt.datetime.now()
    wav = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Calculate processing times and real-time factors
    t = (dt.datetime.now() - start_t).total_seconds()
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
    }

    # Adjust wav volume if necessary
    if prompt_rms < target_rms:
        wav = wav * prompt_rms / target_rms
    torchaudio.save(save_path, wav.cpu(), sample_rate=sampling_rate)

    return metrics

def stream_chunk_text(text, delay=0.1):
    words = re.findall(r"\b\w+\b", text)
    
    for i in range(0, len(words), 2):
        chunk = " ".join(words[i:i+2])
        if chunk:
            yield chunk
            time.sleep(delay)


def split_words(text: str):
    return re.findall(r"\b\w+\b", text)

def generate_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    max_duration: float = 100,
    remove_long_sil: bool = False,
    word_pointer: Optional[torch.nn.Module] = None,
    wp_max_pad: int = 4,
    wp_min_frames: int = 150,
    lookahead_words: int = 3,
    history_context_chunks: int = -1,
    advance_mode: str = "word_pointer",
    debug_plot_dir: Optional[Path] = None,
    debug_plot_every: int = 1,
    save_chunk_wavs: bool = False,
    first_chunk_only: bool = False,
    trim_tail_noise: bool = True,
    tail_noise_min_ms: float = 120.0,
    tail_noise_keep_ms: float = 40.0,
):
    """
    Generate waveform of a text based on a given prompt waveform and its transcription,
        this function will do the following to improve the generation quality:
        1. chunk the text according to punctuations.
        2. process chunked texts in batches.
        3. remove long silences in the prompt audio.
        4. add punctuation to the end of prompt text and text if there is not.

    Args:
        save_path (str): Path to save the generated wav.
        prompt_text (str): Transcription of the prompt wav.
        prompt_wav (str): Path to the prompt wav file.
        text (str): Text to be synthesized into a waveform.
        model (torch.nn.Module): The model used for generation.
        vocoder (torch.nn.Module): The vocoder used to convert features to waveforms.
        tokenizer (EmiliaTokenizer): The tokenizer used to convert text to tokens.
        feature_extractor (VocosFbank): The feature extractor used to
            extract acoustic features.
        device (torch.device): The device on which computations are performed.
        num_step (int, optional): Number of steps for decoding. Defaults to 16.
        guidance_scale (float, optional): Scale for classifier-free guidance.
            Defaults to 1.0.
        speed (float, optional): Speed control. Defaults to 1.0.
        t_shift (float, optional): Time shift. Defaults to 0.5.
        target_rms (float, optional): Target RMS for waveform normalization.
            Defaults to 0.1.
        feat_scale (float, optional): Scale for features.
            Defaults to 0.1.
        sampling_rate (int, optional): Sampling rate for the waveform.
            Defaults to 24000.
        max_duration (float, optional): The maximum duration to process in each
            batch. Used to control memory consumption when generating long audios.
        remove_long_sil (bool, optional): Whether to remove long silences in the
            middle of the generated speech (edge silences will be removed by default).
    Returns:
        metrics (dict): Dictionary containing time and real-time
            factor metrics for processing.
    """

    start_t = dt.datetime.now()
    # Load and process prompt wav
    prompt_wav = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)

    # Remove edge and long silences in the prompt wav.
    # Add 0.2s trailing silence to avoid leaking prompt to generated speech.
    prompt_wav = remove_silence(
        prompt_wav, sampling_rate, only_edge=False, trail_sil=200
    )

    prompt_wav, prompt_rms = rms_norm(prompt_wav, target_rms)

    prompt_duration = prompt_wav.shape[-1] / sampling_rate

    if prompt_duration > 20:
        logging.warning(
            f"Given prompt wav is too long ({prompt_duration}s). "
            f"Please provide a shorter one (1-3 seconds is recommended)."
        )
    elif prompt_duration > 10:
        logging.warning(
            f"Given prompt wav is long ({prompt_duration}s). "
            f"It will lead to slower inference speed and possibly worse speech quality."
        )

    # Extract features from prompt wav
    prompt_features = feature_extractor.extract(
        prompt_wav, sampling_rate=sampling_rate
    ).to(device)

    prompt_features = prompt_features.unsqueeze(0) * feat_scale
    prompt_wav = (
        vocoder.decode(prompt_features.permute(0, 2, 1) / feat_scale)
        .squeeze(1)
    )
    torchaudio.save(f"{save_path}_prompt.wav", prompt_wav.cpu(), sample_rate=sampling_rate)
    sample_tag = _safe_plot_name(Path(save_path).stem)
    _save_prompt_debug_plot(
        debug_dir=debug_plot_dir,
        sample_tag=sample_tag,
        prompt_wav=prompt_wav,
        prompt_features=prompt_features,
        sampling_rate=sampling_rate,
    )

    rmsprompt_features = torch.sqrt(torch.mean(torch.exp(prompt_features) ** 2, dim=-1))  # shape [T]
    rmsprompt_features_np = rmsprompt_features.detach().cpu().numpy()
        
    # 绘制 RMS 随时间变化图
    t_wave = np.arange(len(prompt_wav[0])) / sampling_rate   # 单位：秒
    t_feat = np.arange(rmsprompt_features_np.shape[-1]) * 256 / sampling_rate

    plt.figure()
    plt.plot(t_wave, prompt_wav[0].cpu().numpy()**2)
    plt.plot(t_feat, rmsprompt_features_np[0])
    plt.xlabel("Time step")
    plt.ylabel("RMS")
    plt.title("RMS over Time for promptwav")
    plt.savefig(f"rms_over_time_promptwav_0_.png")
    plt.close()

    # Add punctuation in the end if there is not
    text = add_punctuation(text)
    text = text[:-1] + "__."
    logging.debug(f"Target text: {text}")
    prompt_text = add_punctuation(prompt_text)

    # Fixed-window streaming setup.
    fixed_chunk_frames = 150
    frame_rate = 24000 / 256
    lookahead_words = max(0, int(lookahead_words))
    history_context_chunks = int(history_context_chunks)

    # Use text.split() (NOT a punctuation-stripping regex) so the words
    # passed to the WordPointer head match the trainer's tokenization
    # (zipvoice/bin/train_word_pointer.py:151). Punctuation stays glued
    # to its preceding word; downstream bookkeeping just counts words.
    target_words = text.split()
    total_target_words = len(target_words)

    prompt_tokens_for_est = tokenizer.texts_to_token_ids([prompt_text])[0]
    target_tokens_for_est = tokenizer.texts_to_token_ids([text])[0]
    prompt_token_len = max(1, len(prompt_tokens_for_est))
    target_token_len = max(1, len(target_tokens_for_est))
    # Use the same duration heuristic as model sampling to map synthesized frames to text progress.
    est_total_new_frames = int(
        torch.ceil(
            torch.tensor(
                [prompt_features.size(1) / prompt_token_len * target_token_len / speed],
                dtype=torch.float32,
            )
        ).item()
    )
    est_total_new_frames = max(1, est_total_new_frames)

    generated_new_frames = 0
    committed_word_pos = 0
    base_prompt_text = prompt_text
    base_prompt_features = prompt_features
    prompt_mel_len = base_prompt_features.size(1)
    generated_chunk_history = []

    output_wav = []
    output_mel = []
    debug_history = []

    if total_target_words == 0:
        logging.warning("Target text has no valid words after tokenization; skip generation.")
        return {
            "t": 0.0,
            "t_no_vocoder": 0.0,
            "t_vocoder": 0.0,
            "wav_seconds": 0.0,
            "rtf": 0.0,
            "rtf_no_vocoder": 0.0,
            "rtf_vocoder": 0.0,
            "first_chunk_t": 0.0,
            "first_chunk_stream_t": 0.0,
        }

    num_iter = max(1, int(np.ceil(est_total_new_frames / fixed_chunk_frames))) + 1

    # Words synthesized per chunk. WordPointer refines this estimate; the
    # baseline modes use it directly or combine it with frame-ratio progress.
    words_per_chunk = max(
        1,
        int(np.ceil(total_target_words * fixed_chunk_frames / est_total_new_frames)),
    )

    stream_start_t = dt.datetime.now()
    first_chunk_t = None
    first_chunk_stream_t = None
    for i in range(num_iter):
        # Expected synthesized position after this step. Pure extrapolation
        # from committed_word_pos (= what the prompt actually covers) plus
        # one chunk's worth of words. CTC at end-of-step will update
        # committed_word_pos for the *next* iteration.
        expected_word_pos = min(
            total_target_words,
            committed_word_pos + words_per_chunk + 1,
        )
        logging.debug(
            f"expected_word_pos={expected_word_pos} "
            f"total_target_words={total_target_words} "
            f"committed_word_pos={committed_word_pos}"
        )

        logging.debug(
            f"  step {i}: committed={committed_word_pos} "
            f"expected={expected_word_pos} (words_per_chunk={words_per_chunk})"
        )

        # Send configurable look-ahead words beyond the expected end-of-step
        # position so cross-attention has future context.
        lookahead_end = min(total_target_words, expected_word_pos + lookahead_words)
        if lookahead_end <= committed_word_pos:
            lookahead_end = min(total_target_words, committed_word_pos + 1)

        text_chunk_words = target_words[committed_word_pos:lookahead_end]
        if len(text_chunk_words) == 0:
            break

        chunk_start_word_pos = committed_word_pos
        text_chunk_str = " ".join(text_chunk_words)
        logging.debug(
            f"Processing chunk {i}: {text_chunk_str}, committed={committed_word_pos}, "
            f"expected={expected_word_pos}, lookahead_end={lookahead_end}"
        )

        if history_context_chunks < 0:
            context_history = generated_chunk_history
        elif history_context_chunks == 0:
            context_history = []
        else:
            context_history = generated_chunk_history[-history_context_chunks:]

        if context_history:
            context_features = [h["features"] for h in context_history]
            context_word_start = context_history[0]["start_word_pos"]
            context_words = target_words[context_word_start:committed_word_pos]
            prompt_features_for_model = torch.cat(
                [base_prompt_features] + context_features, dim=1
            )
            prompt_text_for_model = (
                base_prompt_text + " " + " ".join(context_words)
            ).strip()
        else:
            prompt_features_for_model = base_prompt_features
            prompt_text_for_model = base_prompt_text

        # Tokenize text (str tokens), punctuations will be preserved.
        tokens_str = tokenizer.texts_to_tokens([text_chunk_str])[0]
        prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text_for_model + " "])[0]

        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=1000)
        # Tokenize text (int tokens)
        tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
        prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

        # Start predicting features



        # print(f"tokens:{tokens} prompt_tokens:{prompt_tokens} shape of prompt_features: {prompt_features.shape}")
        # Generate features
        (
            pred_features,
            pred_features_lens,
            pred_prompt_features,
            pred_prompt_features_lens,
        ) = model.sample(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features_for_model,
            prompt_features_lens=torch.tensor(
                [prompt_features_for_model.size(1)],
                device=prompt_features_for_model.device,
            ),
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
        # print(f"pred_features:{pred_features}")

        pred_features_lens_int = _to_int(pred_features_lens)
        model_chunk = pred_features[:, : pred_features_lens_int, :]
        actual_chunk_frames = min(fixed_chunk_frames, model_chunk.size(1))
        model_chunk = model_chunk[:, :actual_chunk_frames, :]
        if actual_chunk_frames < fixed_chunk_frames:
            pad_frames = fixed_chunk_frames - actual_chunk_frames
            model_chunk = torch.nn.functional.pad(model_chunk, (0, 0, 0, pad_frames))
            actual_chunk_frames = fixed_chunk_frames

        # Postprocess predicted features
        pred_features = model_chunk.permute(0, 2, 1) / feat_scale  # (B, C, T)
        

        # Start vocoder processing
        start_vocoder_t = dt.datetime.now()
        # print(f"shape of pred_features: {pred_features.shape}, pred_features_lens: {pred_features_lens}, pred_features[0, :5, 0]: {pred_features[0, :5, 0]}")
        wav = (
            vocoder.decode(pred_features)
            .squeeze(1)
            .clamp(-1, 1)
        )
        # print(f"shape of prompt_features: {prompt_features.shape}")
        # wav = (
        #     vocoder.decode(prompt_features.permute(0, 2, 1) / feat_scale)
        #     .squeeze(1)
        #     .clamp(-1, 1)
        # )
        logging.debug(f"Chunk generated, duration: {wav.shape[-1] / sampling_rate}s")
        # Adjust wav volume if necessary
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        if first_chunk_t is None:
            now_t = dt.datetime.now()
            first_chunk_t = (now_t - start_t).total_seconds()
            first_chunk_stream_t = (now_t - stream_start_t).total_seconds()
        logging.debug(f"wav rms: {torch.sqrt(torch.mean(wav**2)).item():.6f}")
        if save_chunk_wavs:
            chunk_name = _safe_plot_name(text_chunk_str, max_len=120)
            torchaudio.save(
                f"{save_path}_chunk_{i:03d}_{chunk_name}.wav",
                wav.cpu(),
                sample_rate=sampling_rate,
            )
        output_wav.append(wav)
        output_mel.append(model_chunk)

        # Finish model generation
        t = (dt.datetime.now() - start_t).total_seconds()
        generated_new_frames += fixed_chunk_frames

        # End-of-step word-pointer decision: feed the most recent
        # ``wp_min_frames`` frames of generated mel together with this
        # step's text window into WordPointer. It predicts
        # ``(pred_left, pred_right)`` = (#words in the window already
        # past, #words at the right not yet spoken). We commit
        # ``lookahead_end - pred_right`` and fall back to the duration
        # ratio for the very first step (not enough mel yet).
        wp_pred = None
        wp_info = None
        wp_tokens_ids = []
        wp_mel = wp_mel_lens = wp_tokens = wp_token_lens = None
        gen_mel_full = torch.cat(output_mel, dim=1) if output_mel else None
        gen_mel_len = 0 if gen_mel_full is None else gen_mel_full.size(1)
        if (
            advance_mode in ("word_pointer", "wp_right_only", "wp_ratio_clamped")
            and word_pointer is not None
            and gen_mel_full is not None
            and gen_mel_len >= wp_min_frames
        ):
            wp_device = next(word_pointer.parameters()).device
            wp_mel = gen_mel_full[:, -wp_min_frames:, :].to(wp_device)
            wp_tokens_ids = tokenizer.texts_to_token_ids([text_chunk_str])[0]
            if len(wp_tokens_ids) > 0:
                wp_mel_lens = torch.tensor(
                    [wp_min_frames], dtype=torch.long, device=wp_device
                )
                wp_tokens = torch.tensor(
                    [wp_tokens_ids], dtype=torch.long, device=wp_device
                )
                wp_token_lens = torch.tensor(
                    [len(wp_tokens_ids)], dtype=torch.long, device=wp_device
                )
                with torch.inference_mode():
                    logits = word_pointer(wp_mel, wp_mel_lens, wp_tokens, wp_token_lens)
                probs = torch.softmax(logits[0], dim=-1)
                n_pad = wp_max_pad + 1
                if advance_mode == "wp_right_only":
                    right_probs = probs.reshape(n_pad, n_pad).sum(dim=0)
                    best_prob, best_r = right_probs.max(dim=-1)
                    pred_l = 0
                    pred_r = int(best_r.item())
                    label = pred_r
                    wp_conf = float(best_prob.item())
                    wp_decision = "right_marginal"
                else:
                    best_prob, best_label = probs.max(dim=-1)
                    label = int(best_label.item())
                    pred_l, pred_r = WordPointer.decode_label(label, max_pad=wp_max_pad)
                    wp_conf = float(best_prob.item())
                    wp_decision = "argmax"
                    fallback_l = 1
                    fallback_min_prob = 0.05
                    if (wp_conf < 0.7 and fallback_l <= wp_max_pad) or wp_conf < 0.4:
                        row_start = fallback_l * n_pad
                        row_probs = probs[row_start : row_start + n_pad]
                        row_best_prob, row_best_r = row_probs.max(dim=-1)
                        if float(row_best_prob.item()) > fallback_min_prob or wp_conf < 0.4:
                            pred_l = fallback_l
                            pred_r = int(row_best_r.item())
                            label = row_start + pred_r
                            wp_decision = "fallback_left1"
                            wp_conf = float(row_best_prob.item())
                wp_pred = (pred_l, pred_r)
                wp_info = {
                    "label": label,
                    "pred_left": pred_l,
                    "pred_right": pred_r,
                    "conf": wp_conf,
                    "decision": wp_decision,
                }

        ratio_pos = int(
            np.floor(total_target_words * generated_new_frames / est_total_new_frames)
        )
        ratio_pos = min(
            total_target_words,
            max(committed_word_pos, ratio_pos),
        )

        if advance_mode == "wp_ratio_clamped":
            if wp_pred is not None and wp_info is not None and wp_info["conf"] >= 0.7:
                _, pred_r = wp_pred
                wp_word_pos = lookahead_end - pred_r - 1
                clamped_pos = min(ratio_pos + 1, max(ratio_pos - 1, wp_word_pos))
                committed_word_pos = min(
                    total_target_words,
                    max(committed_word_pos, clamped_pos),
                )
                src = "wp_ratio_clamped"
            else:
                committed_word_pos = ratio_pos
                src = "ratio"
        elif advance_mode == "wp_right_only" and wp_pred is not None:
            _, pred_r = wp_pred
            wp_word_pos = lookahead_end - pred_r
            committed_word_pos = min(
                total_target_words,
                max(committed_word_pos, wp_word_pos),
            )
            src = "wp_right"
        elif advance_mode == "fixed_words":
            committed_word_pos = min(
                total_target_words,
                max(committed_word_pos, committed_word_pos + words_per_chunk),
            )
            src = "fixed_words"
        elif advance_mode == "ratio":
            committed_word_pos = ratio_pos
            src = "ratio"
        elif wp_pred is not None:
            _, pred_r = wp_pred
            wp_word_pos = lookahead_end - pred_r - 1
            committed_word_pos = min(
                total_target_words,
                max(committed_word_pos, wp_word_pos),
            )
            src = "wp"
        else:
            committed_word_pos = ratio_pos
            src = "ratio"

        generated_chunk_history.append(
            {
                "features": model_chunk,
                "start_word_pos": chunk_start_word_pos,
                "end_word_pos": committed_word_pos,
            }
        )

        debug_history.append(
            {
                "step": i,
                "expected_word_pos": expected_word_pos,
                "lookahead_end": lookahead_end,
                "committed_word_pos": committed_word_pos,
                "generated_new_frames": generated_new_frames,
                "pred_features_lens": pred_features_lens_int,
                "actual_chunk_frames": actual_chunk_frames,
                "src": src,
            }
        )

        if debug_plot_dir is not None and (i % max(1, debug_plot_every) == 0):
            wp_probs = None
            pred_l = pred_r = None
            if (
                word_pointer is not None
                and gen_mel_len >= wp_min_frames
                and len(wp_tokens_ids) > 0
                and wp_mel is not None
            ):
                with torch.inference_mode():
                    logits_for_plot = word_pointer(wp_mel, wp_mel_lens, wp_tokens, wp_token_lens)
                    wp_probs = torch.softmax(logits_for_plot[0], dim=-1).reshape(wp_max_pad + 1, wp_max_pad + 1)
                if wp_pred is not None:
                    pred_l, pred_r = wp_pred
            _save_stream_debug_plot(
                debug_dir=debug_plot_dir,
                sample_tag=sample_tag,
                step_idx=i,
                target_words=target_words,
                text_chunk_str=text_chunk_str,
                expected_word_pos=expected_word_pos,
                lookahead_end=lookahead_end,
                committed_word_pos=committed_word_pos,
                generated_new_frames=generated_new_frames,
                est_total_new_frames=est_total_new_frames,
                fixed_chunk_frames=fixed_chunk_frames,
                pred_features_lens=pred_features_lens_int,
                actual_chunk_frames=actual_chunk_frames,
                src=src,
            )
            _save_word_pointer_heatmap(
                debug_dir=debug_plot_dir,
                sample_tag=sample_tag,
                step_idx=i,
                wp_probs=wp_probs,
                wp_max_pad=wp_max_pad,
                pred_l=pred_l,
                pred_r=pred_r,
                lookahead_end=lookahead_end,
                committed_word_pos=committed_word_pos,
                text_chunk_str=text_chunk_str,
            )
            _save_chunk_debug_plot(
                debug_dir=debug_plot_dir,
                sample_tag=sample_tag,
                step_idx=i,
                model_chunk=model_chunk,
                wav=wav,
                text_chunk_str=text_chunk_str,
                sampling_rate=sampling_rate,
            )

        if wp_info is not None:
            logging.debug(
                f"  step {i} end: src={src} committed={committed_word_pos} "
                f"lookahead_end={lookahead_end} "
                f"pred_left={wp_info['pred_left']} pred_right={wp_info['pred_right']} "
                f"label={wp_info['label']} conf={wp_info['conf']:.4f} "
                f"decision={wp_info['decision']}"
            )
        else:
            logging.debug(
                f"  step {i} end: src={src} committed={committed_word_pos} (WP skipped)"
            )

        # Stop when all words are expected to be synthesized.
        if committed_word_pos >= total_target_words - 1:# and generated_new_frames >= est_total_new_frames:
            break
        if first_chunk_only:
            break

    final_wav = torch.cat(output_wav, dim=-1)
    if trim_tail_noise:
        final_mel = torch.cat(output_mel, dim=1)
        final_wav, trimmed_samples = trim_trailing_noise(
            final_wav,
            final_mel,
            sampling_rate=sampling_rate,
            min_noise_ms=tail_noise_min_ms,
        )
        if trimmed_samples > 0:
            logging.info(
                "Trimmed trailing RMS=1 mel plateau: %.3fs",
                trimmed_samples / sampling_rate,
            )
    _save_stream_summary_plot(
        debug_dir=debug_plot_dir,
        sample_tag=sample_tag,
        history=debug_history,
        total_target_words=total_target_words,
        est_total_new_frames=est_total_new_frames,
    )
    logging.debug(f"Final generated wav duration: {final_wav.shape[-1] / sampling_rate}s")
    # Calculate processing time metrics
    t_no_vocoder = (start_vocoder_t - start_t).total_seconds()
    t_vocoder = (dt.datetime.now() - start_vocoder_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate
    rtf = t / wav_seconds
    rtf_no_vocoder = t_no_vocoder / wav_seconds
    rtf_vocoder = t_vocoder / wav_seconds
    metrics = {
        "t": t,
        "t_no_vocoder": t_no_vocoder,
        "t_vocoder": t_vocoder,
        "wav_seconds": wav_seconds,
        "rtf": rtf,
        "rtf_no_vocoder": rtf_no_vocoder,
        "rtf_vocoder": rtf_vocoder,
        "first_chunk_t": first_chunk_t if first_chunk_t is not None else 0.0,
        "first_chunk_stream_t": (
            first_chunk_stream_t if first_chunk_stream_t is not None else 0.0
        ),
    }

    safe_prompt = re.sub(r"[^A-Za-z0-9._-]", "_", prompt_text)[:80]
    torchaudio.save(save_path, final_wav.cpu(), sample_rate=sampling_rate)
    # torchaudio.save(save_path + "_" + safe_prompt + ".wav", final_wav.cpu(), sample_rate=sampling_rate)
    return metrics


def generate_list(
    res_dir: str,
    test_list: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: EmiliaTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int = 16,
    guidance_scale: float = 1.0,
    speed: float = 1.0,
    t_shift: float = 0.5,
    target_rms: float = 0.1,
    feat_scale: float = 0.1,
    sampling_rate: int = 24000,
    raw_evaluation: bool = False,
    max_duration: float = 100,
    remove_long_sil: bool = False,
    word_pointer: Optional[torch.nn.Module] = None,
    wp_max_pad: int = 4,
    wp_min_frames: int = 150,
    lookahead_words: int = 3,
    history_context_chunks: int = -1,
    advance_mode: str = "word_pointer",
    debug_plot_dir: Optional[Path] = None,
    debug_plot_every: int = 1,
    save_chunk_wavs: bool = False,
    first_chunk_only: bool = False,
    trim_tail_noise: bool = True,
    tail_noise_min_ms: float = 120.0,
    tail_noise_keep_ms: float = 40.0,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []
    total_first_chunk_t = []
    total_first_chunk_stream_t = []

    with open(test_list, "r") as fr:
        lines = fr.readlines()

    for i, line in enumerate(lines):
        wav_name, prompt_text, prompt_wav, text = line.strip().split("\t")
        save_path = f"{res_dir}/{wav_name}.wav"

        common_params = {
            "save_path": save_path,
            "prompt_text": prompt_text,
            "prompt_wav": prompt_wav,
            "text": text,
            "model": model,
            "vocoder": vocoder,
            "tokenizer": tokenizer,
            "feature_extractor": feature_extractor,
            "device": device,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "t_shift": t_shift,
            "target_rms": target_rms,
            "feat_scale": feat_scale,
            "sampling_rate": sampling_rate,
        }

        if raw_evaluation:
            metrics = generate_sentence_raw_evaluation(**common_params)
        else:
            metrics = generate_sentence(
                **common_params,
                max_duration=max_duration,
                remove_long_sil=remove_long_sil,
                word_pointer=word_pointer,
                wp_max_pad=wp_max_pad,
                wp_min_frames=wp_min_frames,
                lookahead_words=lookahead_words,
                history_context_chunks=history_context_chunks,
                advance_mode=advance_mode,
                debug_plot_dir=debug_plot_dir,
                debug_plot_every=debug_plot_every,
                save_chunk_wavs=save_chunk_wavs,
                first_chunk_only=first_chunk_only,
                trim_tail_noise=trim_tail_noise,
                tail_noise_min_ms=tail_noise_min_ms,
                tail_noise_keep_ms=tail_noise_keep_ms,
            )
        logging.info(f"[Sentence: {i}] Saved to: {save_path}")
        logging.info(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
        if "first_chunk_t" in metrics:
            logging.info(
                f"[Sentence: {i}] First chunk latency: "
                f"{metrics['first_chunk_t']:.4f}s "
                f"(stream loop: {metrics['first_chunk_stream_t']:.4f}s)"
            )
            total_first_chunk_t.append(metrics["first_chunk_t"])
            total_first_chunk_stream_t.append(metrics["first_chunk_stream_t"])
        total_t.append(metrics["t"])
        total_t_no_vocoder.append(metrics["t_no_vocoder"])
        total_t_vocoder.append(metrics["t_vocoder"])
        total_wav_seconds.append(metrics["wav_seconds"])

    logging.info(f"Average RTF: {np.sum(total_t) / np.sum(total_wav_seconds):.4f}")
    logging.info(
        f"Average RTF w/o vocoder: "
        f"{np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds):.4f}"
    )
    logging.info(
        f"Average RTF vocoder: "
        f"{np.sum(total_t_vocoder) / np.sum(total_wav_seconds):.4f}"
    )
    if total_first_chunk_t:
        logging.info(
            f"Average first chunk latency: {np.mean(total_first_chunk_t):.4f}s"
        )
        logging.info(
            f"Average first chunk latency from stream loop: "
            f"{np.mean(total_first_chunk_stream_t):.4f}s"
        )

    summary = {
        "num_sentences": len(total_wav_seconds),
        "avg_rtf": float(np.sum(total_t) / np.sum(total_wav_seconds)),
        "avg_rtf_no_vocoder": float(
            np.sum(total_t_no_vocoder) / np.sum(total_wav_seconds)
        ),
        "avg_rtf_vocoder": float(np.sum(total_t_vocoder) / np.sum(total_wav_seconds)),
        "avg_first_chunk_latency": (
            float(np.mean(total_first_chunk_t)) if total_first_chunk_t else None
        ),
        "avg_first_chunk_stream_latency": (
            float(np.mean(total_first_chunk_stream_t))
            if total_first_chunk_stream_t
            else None
        ),
    }
    summary_path = Path(res_dir) / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(f"Saved metrics summary to: {summary_path}")


@torch.inference_mode()
def main():
    parser = get_parser()
    args = parser.parse_args()

    torch.set_num_threads(args.num_thread)
    torch.set_num_interop_threads(args.num_thread)

    params = AttributeDict()
    params.update(vars(args))
    fix_random_seed(params.seed)

    model_defaults = {
        "zipvoice": {
            "num_step": 16,
            "guidance_scale": 1.0,
        },
        "zipvoice_distill": {
            "num_step": 8,
            "guidance_scale": 3.0,
        },
    }

    model_specific_defaults = model_defaults.get(params.model_name, {})

    for param, value in model_specific_defaults.items():
        if getattr(params, param) is None:
            setattr(params, param, value)
            logging.info(f"Setting {param} to default value: {value}")

    assert (params.test_list is not None) ^ (
        (params.prompt_wav and params.prompt_text and params.text) is not None
    ), (
        "For inference, please provide prompts and text with either '--test-list'"
        " or '--prompt-wav, --prompt-text and --text'."
    )

    if params.model_dir is not None:
        params.model_dir = Path(params.model_dir)
        if not params.model_dir.is_dir():
            raise FileNotFoundError(f"{params.model_dir} does not exist")
        for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
            if not (params.model_dir / filename).is_file():
                raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
        model_ckpt = params.model_dir / params.checkpoint_name
        model_config = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
        logging.info(
            f"Using {params.model_name} in local model dir {params.model_dir}, "
            f"checkpoint {params.checkpoint_name}"
        )
    else:
        logging.info(f"Using pretrained {params.model_name} model from the Huggingface")
        model_ckpt = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.pt"
        )
        model_config = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json"
        )

        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt"
        )

    if params.tokenizer == "emilia":
        tokenizer = EmiliaTokenizer(token_file=token_file)
    elif params.tokenizer == "libritts":
        tokenizer = LibriTTSTokenizer(token_file=token_file)
    elif params.tokenizer == "espeak":
        tokenizer = EspeakTokenizer(token_file=token_file, lang=params.lang)
    else:
        assert params.tokenizer == "simple"
        tokenizer = SimpleTokenizer(token_file=token_file)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}

    with open(model_config, "r") as f:
        model_config = json.load(f)

    if params.model_name == "zipvoice":
        model = ZipVoice(
            **model_config["model"],
            **tokenizer_config,
        )
    else:
        assert params.model_name == "zipvoice_distill"
        model = ZipVoiceDistill(
            **model_config["model"],
            **tokenizer_config,
        )

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=False)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    if torch.cuda.is_available():
        params.device = torch.device("cuda", params.cuda_device)
    elif torch.backends.mps.is_available():
        params.device = torch.device("mps")
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    model = model.to(params.device)
    model.eval()

    if params.trt_engine_path:
        load_trt(model, params.trt_engine_path)

    word_pointer = None
    wp_max_pad = params.word_pointer_max_pad
    if params.advance_mode in ("word_pointer", "wp_right_only", "wp_ratio_clamped"):
        assert params.word_pointer_ckpt is not None, (
            "WordPointer-based mode requires --word-pointer-ckpt PATH."
        )
        logging.info(f"Loading WordPointer from {params.word_pointer_ckpt}")
        wp_ckpt = torch.load(
            params.word_pointer_ckpt, map_location="cpu", weights_only=False
        )
        wp_vocab_size = int(wp_ckpt.get("vocab_size", tokenizer.vocab_size))
        wp_params = wp_ckpt.get("params", {}) or {}
        wp_max_pad = int(wp_ckpt.get("max_pad", wp_params.get("max_pad", params.word_pointer_max_pad)))
        assert wp_max_pad == params.word_pointer_max_pad, (
            f"--word-pointer-max-pad={params.word_pointer_max_pad} disagrees with "
            f"checkpoint's max_pad={wp_max_pad}; pass the matching value."
        )
        wp_chunk_frames = int(wp_ckpt.get("chunk_frames", wp_params.get("chunk_frames", 150)))
        word_pointer = WordPointer(
            vocab_size=wp_vocab_size,
            max_pad=wp_max_pad,
            mel_in_dim=int(model_config["model"].get("feat_dim", 100)),
            dim=int(wp_params.get("dim", 128)),
            mel_encoder_layers=int(wp_params.get("mel_encoder_layers", 2)),
            text_encoder_layers=int(wp_params.get("text_encoder_layers", 2)),
            cross_attn_layers=int(wp_params.get("cross_attn_layers", 2)),
            num_heads=int(wp_params.get("num_heads", 4)),
            feedforward_dim=int(wp_params.get("feedforward_dim", 512)),
            dropout=float(wp_params.get("dropout", 0.0)),
        )
        word_pointer.load_state_dict(wp_ckpt["model"])
        word_pointer = word_pointer.to(params.device).eval()
        logging.info(
            f"WordPointer: vocab_size={wp_vocab_size} max_pad={wp_max_pad} "
            f"chunk_frames={wp_chunk_frames} "
            f"params={sum(p.numel() for p in word_pointer.parameters())}"
        )
    else:
        logging.info(f"Using advancement baseline: {params.advance_mode}")

    vocoder = get_vocoder(params.vocoder_path)
    vocoder = vocoder.to(params.device)
    vocoder.eval()

    if model_config["feature"]["type"] == "vocos":
        feature_extractor = VocosFbank()
    else:
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    params.sampling_rate = model_config["feature"]["sampling_rate"]

    logging.info("Start generating...")
    if params.test_list:
        res_dir = params.res_dir
        os.makedirs(res_dir, exist_ok=True)
        generate_list(
            res_dir=params.res_dir,
            test_list=params.test_list,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            raw_evaluation=params.raw_evaluation,
            max_duration=params.max_duration,
            remove_long_sil=params.remove_long_sil,
            word_pointer=word_pointer,
            wp_max_pad=wp_max_pad,
            wp_min_frames=params.word_pointer_min_frames,
            lookahead_words=params.lookahead_words,
            history_context_chunks=params.history_context_chunks,
            advance_mode=params.advance_mode,
            debug_plot_dir=params.debug_plot_dir,
            debug_plot_every=params.debug_plot_every,
            save_chunk_wavs=params.save_chunk_wavs,
            first_chunk_only=params.first_chunk_only,
            trim_tail_noise=params.trim_tail_noise,
            tail_noise_min_ms=params.tail_noise_min_ms,
            tail_noise_keep_ms=params.tail_noise_keep_ms,
        )
    else:
        assert (
            not params.raw_evaluation
        ), "Raw evaluation is only valid with --test-list"
        generate_sentence(
            save_path=params.res_wav_path,
            prompt_text=params.prompt_text,
            prompt_wav=params.prompt_wav,
            text=params.text,
            model=model,
            vocoder=vocoder,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            device=params.device,
            num_step=params.num_step,
            guidance_scale=params.guidance_scale,
            speed=params.speed,
            t_shift=params.t_shift,
            target_rms=params.target_rms,
            feat_scale=params.feat_scale,
            sampling_rate=params.sampling_rate,
            max_duration=params.max_duration,
            remove_long_sil=params.remove_long_sil,
            word_pointer=word_pointer,
            wp_max_pad=wp_max_pad,
            wp_min_frames=params.word_pointer_min_frames,
            lookahead_words=params.lookahead_words,
            history_context_chunks=params.history_context_chunks,
            advance_mode=params.advance_mode,
            debug_plot_dir=params.debug_plot_dir,
            debug_plot_every=params.debug_plot_every,
            save_chunk_wavs=params.save_chunk_wavs,
            first_chunk_only=params.first_chunk_only,
            trim_tail_noise=params.trim_tail_noise,
            tail_noise_min_ms=params.tail_noise_min_ms,
            tail_noise_keep_ms=params.tail_noise_keep_ms,
        )
        logging.info(f"Saved to: {params.res_wav_path}")
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    main()
