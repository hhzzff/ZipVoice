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
    sampling_rate: int,
    min_noise_ms: float = 120.0,
    keep_ms: float = 40.0,
) -> tuple[torch.Tensor, int]:
    """Trim stable low-energy broadband noise from the end of a waveform.

    The bad tail produced by text-starved streaming chunks is usually not
    silence: it has low but steady RMS, high zero-crossing rate, and high
    spectral flatness.  Work backwards from the end so normal internal
    fricatives are left untouched.
    """
    if wav.numel() == 0:
        return wav, 0

    squeeze = wav.dim() == 1
    x = wav.unsqueeze(0) if squeeze else wav
    mono = x.mean(dim=0).detach().float().cpu()

    win = max(16, int(round(0.04 * sampling_rate)))
    hop = max(1, int(round(0.01 * sampling_rate)))
    if mono.numel() < win * 4:
        return wav, 0

    frames = mono.unfold(0, win, hop)
    rms = frames.pow(2).mean(dim=-1).sqrt()
    zcr = ((frames[:, 1:] * frames[:, :-1]) < 0).float().mean(dim=-1)
    window = torch.hann_window(win, dtype=frames.dtype)
    spec = torch.fft.rfft(frames * window, dim=-1).abs().clamp_min(1.0e-8)
    flatness = spec.log().mean(dim=-1).exp() / spec.mean(dim=-1).clamp_min(1.0e-8)

    speech_rms_ref = rms.quantile(0.70).clamp_min(1.0e-4)
    # Hiss tails in observed failures sit around zcr ~= 0.17 and
    # flatness ~= 0.5, while voiced speech is much lower on both.
    noise_like = (
        (rms > 0.01 * speech_rms_ref)
        & (rms < 0.75 * speech_rms_ref)
        & (zcr > 0.11)
        & (flatness > 0.28)
    )

    last_good = len(noise_like) - 1
    min_frames = max(1, int(math.ceil(min_noise_ms / 10.0)))
    count = 0
    idx = len(noise_like) - 1
    while idx >= 0 and bool(noise_like[idx]):
        count += 1
        idx -= 1

    if count < min_frames:
        return wav, 0

    trim_start_frame = last_good - count + 1
    trim_sample = trim_start_frame * hop
    keep_samples = int(round(keep_ms * sampling_rate / 1000.0))
    cut_sample = max(0, trim_sample - keep_samples)
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
    debug_plot_dir: Optional[Path] = None,
    debug_plot_every: int = 1,
    save_chunk_wavs: bool = False,
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
    prompt_text = add_punctuation(prompt_text)

    # Fixed-window streaming setup.
    fixed_chunk_frames = 150
    frame_rate = 24000 / 256

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
    prompt_mel_len = prompt_features.size(1)

    output_wav = []
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
        }

    num_iter = max(1, int(np.ceil(est_total_new_frames / fixed_chunk_frames)))

    # Words synthesized per chunk (used to project committed → expected when
    # picking the text window for this step). It's an estimate; the next
    # step's CTC corrects any drift via committed_word_pos.
    words_per_chunk = max(
        1,
        int(np.ceil(total_target_words * fixed_chunk_frames / est_total_new_frames)),
    )

    for i in range(num_iter):
        # Expected synthesized position after this step. Pure extrapolation
        # from committed_word_pos (= what the prompt actually covers) plus
        # one chunk's worth of words. CTC at end-of-step will update
        # committed_word_pos for the *next* iteration.
        expected_word_pos = min(
            total_target_words,
            committed_word_pos + words_per_chunk,
        )

        logging.info(
            f"  step {i}: committed={committed_word_pos} "
            f"expected={expected_word_pos} (words_per_chunk={words_per_chunk})"
        )

        # Send a few words of look-ahead beyond the expected end-of-step
        # position so cross-attention has future context.
        lookahead_end = min(total_target_words, expected_word_pos + 2)
        if lookahead_end <= committed_word_pos:
            lookahead_end = min(total_target_words, committed_word_pos + 2)

        text_chunk_words = target_words[committed_word_pos:lookahead_end]
        if len(text_chunk_words) == 0:
            break

        text_chunk_str = " ".join(text_chunk_words)
        print(
            f"Processing chunk {i}: {text_chunk_str}, committed={committed_word_pos}, "
            f"expected={expected_word_pos}, lookahead_end={lookahead_end}"
        )
        # Tokenize text (str tokens), punctuations will be preserved.
        tokens_str = tokenizer.texts_to_tokens([text_chunk_str])[0]
        prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text + " "])[0]

        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=1000)
        # Tokenize text (int tokens)
        tokens = tokenizer.tokens_to_token_ids(chunked_tokens_str)
        prompt_tokens = tokenizer.tokens_to_token_ids([prompt_tokens_str])

        # Start predicting features
        start_t = dt.datetime.now()



        print(f"tokens:{tokens} prompt_tokens:{prompt_tokens} shape of prompt_features: {prompt_features.shape}")
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
            prompt_features_lens=torch.tensor(prompt_features.size(1)).unsqueeze(0).to(prompt_features.device),
            speed=speed,
            t_shift=t_shift,
            duration="predict",
            num_step=num_step,
            guidance_scale=guidance_scale,
        )
        print(f"pred_features:{pred_features}")

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
        print(f"shape of pred_features: {pred_features.shape}, pred_features_lens: {pred_features_lens}, pred_features[0, :5, 0]: {pred_features[0, :5, 0]}")
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
        print(f"Chunk generated, duration: {wav.shape[-1] / sampling_rate}s")
        # Adjust wav volume if necessary
        if prompt_rms < target_rms:
            wav = wav * prompt_rms / target_rms
        print(f"wav rms: {torch.sqrt(torch.mean(wav**2))} wav:{wav}")
        if save_chunk_wavs:
            chunk_name = _safe_plot_name(text_chunk_str, max_len=120)
            torchaudio.save(
                f"{save_path}_chunk_{i:03d}_{chunk_name}.wav",
                wav.cpu(),
                sample_rate=sampling_rate,
            )
        output_wav.append(wav)

        # Finish model generation
        t = (dt.datetime.now() - start_t).total_seconds()
        generated_new_frames += fixed_chunk_frames
        prompt_features = torch.cat([prompt_features, model_chunk], dim=1)

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
        gen_mel_full = prompt_features[:, prompt_mel_len:, :]
        gen_mel_len = gen_mel_full.size(1)
        if word_pointer is not None and gen_mel_len >= wp_min_frames:
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
                label = int(logits.argmax(dim=-1).item())
                pred_l, pred_r = WordPointer.decode_label(label, max_pad=wp_max_pad)
                wp_pred = (pred_l, pred_r)
                wp_info = {"label": label, "pred_left": pred_l, "pred_right": pred_r}

        if wp_pred is not None:
            _, pred_r = wp_pred
            wp_word_pos = lookahead_end - pred_r
            committed_word_pos = min(
                total_target_words,
                max(committed_word_pos, wp_word_pos),
            )
            src = "wp"
        else:
            # Ratio fallback: extrapolate from frames synthesized so far.
            ratio_pos = int(
                np.floor(total_target_words * generated_new_frames / est_total_new_frames)
            )
            committed_word_pos = min(
                total_target_words,
                max(committed_word_pos, ratio_pos),
            )
            src = "ratio"

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
            logging.info(
                f"  step {i} end: src={src} committed={committed_word_pos} "
                f"lookahead_end={lookahead_end} "
                f"pred_left={wp_info['pred_left']} pred_right={wp_info['pred_right']} "
                f"label={wp_info['label']}"
            )
        else:
            logging.info(
                f"  step {i} end: src={src} committed={committed_word_pos} (WP skipped)"
            )

        prompt_text = (
            base_prompt_text + " " + " ".join(target_words[:committed_word_pos])
        ).strip()

        # Stop when all words are expected to be synthesized.
        if committed_word_pos >= total_target_words and generated_new_frames >= est_total_new_frames:
            break

    final_wav = torch.cat(output_wav, dim=-1)
    if trim_tail_noise:
        final_wav, trimmed_samples = trim_trailing_noise(
            final_wav,
            sampling_rate=sampling_rate,
            min_noise_ms=tail_noise_min_ms,
            keep_ms=tail_noise_keep_ms,
        )
        if trimmed_samples > 0:
            logging.info(
                "Trimmed trailing broadband noise: %.3fs",
                trimmed_samples / sampling_rate,
            )
    _save_stream_summary_plot(
        debug_dir=debug_plot_dir,
        sample_tag=sample_tag,
        history=debug_history,
        total_target_words=total_target_words,
        est_total_new_frames=est_total_new_frames,
    )
    print(f"Final generated wav duration: {final_wav.shape[-1] / sampling_rate}s")
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
    debug_plot_dir: Optional[Path] = None,
    debug_plot_every: int = 1,
    save_chunk_wavs: bool = False,
    trim_tail_noise: bool = True,
    tail_noise_min_ms: float = 120.0,
    tail_noise_keep_ms: float = 40.0,
):
    total_t = []
    total_t_no_vocoder = []
    total_t_vocoder = []
    total_wav_seconds = []

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
                debug_plot_dir=debug_plot_dir,
                debug_plot_every=debug_plot_every,
                save_chunk_wavs=save_chunk_wavs,
                trim_tail_noise=trim_tail_noise,
                tail_noise_min_ms=tail_noise_min_ms,
                tail_noise_keep_ms=tail_noise_keep_ms,
            )
        logging.info(f"[Sentence: {i}] Saved to: {save_path}")
        logging.info(f"[Sentence: {i}] RTF: {metrics['rtf']:.4f}")
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
        params.device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        params.device = torch.device("mps")
    else:
        params.device = torch.device("cpu")
    logging.info(f"Device: {params.device}")

    model = model.to(params.device)
    model.eval()

    if params.trt_engine_path:
        load_trt(model, params.trt_engine_path)

    assert params.word_pointer_ckpt is not None, (
        "Streaming inference requires --word-pointer-ckpt PATH."
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
            debug_plot_dir=params.debug_plot_dir,
            debug_plot_every=params.debug_plot_every,
            save_chunk_wavs=params.save_chunk_wavs,
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
            debug_plot_dir=params.debug_plot_dir,
            debug_plot_every=params.debug_plot_every,
            save_chunk_wavs=params.save_chunk_wavs,
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
