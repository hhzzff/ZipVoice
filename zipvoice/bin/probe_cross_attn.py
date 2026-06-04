#!/usr/bin/env python3
# Copyright    2026    Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Quantitative probe for the FM decoder's cross-attention weights in the
fixed-window streaming ZipVoice model.

This script picks one sample from --test-list, runs model.sample for a
single streaming chunk, uses CrossAttnHook to capture every cross-attn
layer's weights at the final ODE step, and reports per-layer quality:

  - sharpness     = mean top-1 attention mass per Q frame (closer to 1 = sharper)
  - entropy_norm  = mean entropy(p) / log(T_k)  (closer to 0 = sharper)
  - monotonicity  = fraction of consecutive Q frames where argmax does not go backward
  - diagonality   = R^2 of linear fit: argmax ~ a * q_idx + b  (closer to 1 = cleanest pointer)
  - span          = argmax_max - argmax_min across the prediction window
  - top1_tok_idx  = argmax over Q frames (average, attended token)

Also saves:
  - probe_argmax_traces.png  — argmax trajectory per layer in the prediction window
  - probe_best_layer.png     — head-averaged attention matrix of the sharpest layer
  - cross_attention_target_tokens.png — paper-friendly target-token heatmap

Example:

    python3 -m zipvoice.bin.probe_cross_attn \\
        --model-dir exp/zipvoice_libritts_0427_1717_stream_alignmask_fixedwindow_crossattn \\
        --checkpoint-name epoch-37.pt \\
        --tokenizer libritts \\
        --test-list test.tsv \\
        --sample-idx 0 \\
        --out-dir probe_out
"""

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
import torchaudio
from vocos import Vocos

from zipvoice.models.zipvoice_stream_fixedwindow_crossattn import ZipVoice
from zipvoice.models.modules.zipformer_crossattn import CrossMultiheadAttentionWeights
from zipvoice.tokenizer.tokenizer_stream import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import add_punctuation, load_prompt_wav, rms_norm


class CrossAttnHook:
    """Capture cross-attention weights emitted by FM decoder layers."""

    def __init__(self, model: torch.nn.Module):
        self.attn_weights = []
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, CrossMultiheadAttentionWeights):
                self.handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                self.attn_weights.append((name, output.detach().cpu()))

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def get_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--checkpoint-name", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="libritts",
                   choices=["emilia", "libritts", "espeak", "simple"])
    p.add_argument("--lang", type=str, default="en-us")
    p.add_argument("--test-list", type=str, required=True,
                   help="TSV: id\\ttext\\tprompt_wav\\tprompt_text")
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Which row of --test-list to probe.")
    p.add_argument("--num-step", type=int, default=16)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--t-shift", type=float, default=0.7)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--feat-scale", type=float, default=0.1)
    p.add_argument("--target-rms", type=float, default=0.1)
    p.add_argument("--sampling-rate", type=int, default=24000)
    p.add_argument("--window-size", type=int, default=150,
                   help="Match fixed_chunk_frames in the infer script.")
    p.add_argument("--target-words", type=int, default=6,
                   help="Number of target words to feed in this single chunk.")
    p.add_argument("--vocoder-path", type=str, default="vocos-mel-24khz",
                   help="Local Vocos directory used to decode the probed chunk.")
    p.add_argument("--save-chunk-audio", type=int, default=1,
                   help="Whether to save the decoded single-chunk wav.")
    p.add_argument("--dump-all-heads", type=int, default=0,
                   help="Save target-token heatmaps for every layer/head "
                        "captured at the final ODE step.")
    p.add_argument("--dump-all-ode-steps", type=int, default=0,
                   help="Save target-token heatmaps for every ODE step, "
                        "layer and head captured by the hook.")
    p.add_argument("--ode-top-k", type=int, default=8,
                   help="When --dump-all-ode-steps is set, save only the "
                        "top-K ranked step/layer/head heatmaps.")
    p.add_argument("--out-dir", type=str, default="probe_out")
    return p


def build_tokenizer(args, token_file: Path):
    if args.tokenizer == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    if args.tokenizer == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    if args.tokenizer == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=args.lang)
    return SimpleTokenizer(token_file=token_file)


def load_model(args, tokenizer) -> ZipVoice:
    ckpt = args.model_dir / args.checkpoint_name
    cfg_path = args.model_dir / "model.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    model = ZipVoice(
        **cfg["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    if str(ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, str(ckpt))
    else:
        load_checkpoint(filename=str(ckpt), model=model, strict=False)
    return model


def load_vocoder(vocoder_path: str, device: torch.device) -> Vocos:
    vocoder = Vocos.from_hparams(f"{vocoder_path}/config.yaml")
    state_dict = torch.load(
        f"{vocoder_path}/pytorch_model.bin",
        weights_only=True,
        map_location="cpu",
    )
    vocoder.load_state_dict(state_dict)
    return vocoder.to(device).eval()


def read_sample(test_list: str, idx: int):
    test_list_path = Path(test_list)
    with open(test_list_path) as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    if idx >= len(rows):
        raise IndexError(f"sample-idx {idx} out of range (have {len(rows)})")
    r = rows[idx]
    if len(r) == 4:
        utt_id, prompt_text, prompt_wav, text = r
    elif len(r) == 3:
        utt_id, prompt_wav, prompt_text = r
        text = ""
    else:
        raise ValueError(f"unexpected tsv row: {r}")
    prompt_wav_path = Path(prompt_wav)
    if not prompt_wav_path.is_absolute():
        prompt_wav = str(test_list_path.parent / prompt_wav_path)
    return utt_id, text, prompt_wav, prompt_text


def layer_key(name: str) -> str:
    """Shorten a long dotted module path for display."""
    parts = name.split(".")
    # Keep last 3 segments.
    return "/".join(parts[-3:]) if len(parts) >= 3 else name


def compute_layer_stats(attn: torch.Tensor, pw_start: int, pw_len: int,
                        full_tq: int) -> dict:
    """attn: (H, T_q, T_k) -> head-averaged stats on prediction window rows.

    For downsampled layers the prediction window indices are rescaled so
    layers of different resolution are comparable.
    """
    H, Tq, Tk = attn.shape
    ds = max(1, round(full_tq / Tq))
    s = pw_start // ds
    e = s + max(1, pw_len // ds)
    e = min(e, Tq)
    if s >= Tq:
        s = max(0, Tq - max(1, pw_len // ds))
    w = attn.mean(dim=0)[s:e, :]  # (pw_frames, T_k)
    if w.numel() == 0:
        return {}
    # Renormalize defensively (should already sum to 1).
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    top1 = w.max(dim=-1).values                                 # (pw,)
    argmax = w.argmax(dim=-1).float()                           # (pw,)
    # Expected (soft) pointer: E[k | q] = Σ_k k · p(k | q)
    k_idx = torch.arange(Tk, dtype=w.dtype, device=w.device)
    expected = (w * k_idx.unsqueeze(0)).sum(dim=-1)             # (pw,)
    top3 = torch.topk(w, k=min(3, Tk), dim=-1).values.sum(-1)   # (pw,)
    logp = torch.log(w.clamp(min=1e-9))
    entropy = -(w * logp).sum(dim=-1)                           # (pw,)
    entropy_norm = entropy / max(float(np.log(max(Tk, 2))), 1e-9)

    def _mono_and_r2(trace: torch.Tensor):
        n = trace.numel()
        if n < 2:
            return float("nan"), float("nan"), float("nan")
        mono = float((trace[1:] >= trace[:-1]).float().mean())
        q = torch.arange(n, dtype=torch.float32)
        if trace.std() > 1e-6:
            slope = float(((q - q.mean()) * (trace - trace.mean())).sum() /
                          ((q - q.mean()) ** 2).sum().clamp(min=1e-9))
            intercept = float(trace.mean() - slope * q.mean())
            pred = slope * q + intercept
            ss_res = float(((trace - pred) ** 2).sum())
            ss_tot = float(((trace - trace.mean()) ** 2).sum())
            r2 = 1 - ss_res / max(ss_tot, 1e-9)
        else:
            slope = float("nan")
            r2 = float("nan")
        return mono, slope, r2

    mono_a, slope_a, r2_a = _mono_and_r2(argmax)
    mono_e, slope_e, r2_e = _mono_and_r2(expected)

    return {
        "Tq": int(Tq), "Tk": int(Tk), "ds": int(ds),
        "pw_rows": int(w.shape[0]),
        "sharpness": float(top1.mean()),
        "top3_mass": float(top3.mean()),
        "entropy_norm": float(entropy_norm.mean()),
        # argmax (hard) pointer
        "monotonicity": mono_a,
        "diag_slope": slope_a,
        "diag_r2": r2_a,
        "argmax_min": float(argmax.min()),
        "argmax_max": float(argmax.max()),
        "span": float(argmax.max() - argmax.min()),
        "argmax_trace": argmax.cpu().numpy(),
        # expected (soft) pointer
        "mono_exp": mono_e,
        "diag_slope_exp": slope_e,
        "diag_r2_exp": r2_e,
        "exp_min": float(expected.min()),
        "exp_max": float(expected.max()),
        "span_exp": float(expected.max() - expected.min()),
        "expected_trace": expected.cpu().numpy(),
        "weights_avg_hd": w.cpu().numpy(),
    }


def take_last_ode_step(hook: CrossAttnHook):
    """Return list of (layer_name, attn_tensor_first_batch) taken from the
    FINAL ODE step only."""
    if not hook.attn_weights:
        return []
    # Identify unique layer names in first-appearance order.
    uniq, seen = [], set()
    for name, _ in hook.attn_weights:
        if name not in seen:
            uniq.append(name)
            seen.add(name)
    L = len(uniq)
    last = hook.attn_weights[-L:]
    out = []
    for name, w in last:
        # w: (1, H, T_q, T_k) -> (H, T_q, T_k)
        out.append((name, w[0]))
    return out


def take_all_ode_steps(hook: CrossAttnHook):
    """Group captured cross-attention weights by decoder/ODE evaluation.

    Each FM decoder evaluation visits the cross-attention modules in a stable
    order. The hook stream is therefore split into consecutive groups with
    the same number of unique module names.
    """
    if not hook.attn_weights:
        return []
    uniq, seen = [], set()
    for name, _ in hook.attn_weights:
        if name not in seen:
            uniq.append(name)
            seen.add(name)
    L = len(uniq)
    if L == 0:
        return []

    groups = []
    usable = len(hook.attn_weights) - (len(hook.attn_weights) % L)
    if usable != len(hook.attn_weights):
        logging.warning(
            "Dropping %d trailing attention tensors that do not form a full group",
            len(hook.attn_weights) - usable,
        )
    for start in range(0, usable, L):
        group = []
        for name, w in hook.attn_weights[start : start + L]:
            group.append((name, w[0]))
        groups.append(group)
    return groups


@torch.no_grad()
def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer & model
    token_file = args.model_dir / "tokens.txt"
    tokenizer = build_tokenizer(args, token_file)
    model = load_model(args, tokenizer).to(device).eval()

    # sample
    utt_id, text, prompt_wav_path, prompt_text = read_sample(args.test_list, args.sample_idx)
    text = add_punctuation(text)
    prompt_text = add_punctuation(prompt_text)
    print(f"[probe] sample_id={utt_id}")
    print(f"[probe] prompt_text: {prompt_text!r}")
    print(f"[probe] text:        {text!r}")

    # features
    feat_ext = VocosFbank()
    prompt_wav = load_prompt_wav(prompt_wav_path, sampling_rate=args.sampling_rate)
    prompt_wav, _ = rms_norm(prompt_wav, args.target_rms)
    prompt_features = feat_ext.extract(prompt_wav, sampling_rate=args.sampling_rate).to(device)
    prompt_features = prompt_features.unsqueeze(0) * args.feat_scale
    prompt_features_lens = torch.tensor([prompt_features.size(1)], device=device)
    prompt_speech_frames = int(prompt_features.size(1))
    print(f"[probe] prompt_speech_frames = {prompt_speech_frames}")

    # Take the first target_words of text as the single-chunk text we probe.
    target_words = text.split()
    chunk_text = " ".join(target_words[:args.target_words])
    prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
    chunk_tokens_int = tokenizer.texts_to_token_ids([chunk_text])
    # Human-readable labels for the x-axis of the best-layer plot.
    prompt_tokens_str = tokenizer.texts_to_tokens([prompt_text])[0]
    chunk_tokens_str = tokenizer.texts_to_tokens([chunk_text])[0]
    all_labels = list(prompt_tokens_str) + list(chunk_tokens_str)
    prompt_token_len = len(prompt_tokens_str)
    print(f"[probe] chunk_text: {chunk_text!r}")
    print(f"[probe] tokens (prompt/target): {prompt_token_len}/{len(chunk_tokens_str)} "
          f"total_tokens={len(all_labels)}")

    # hook + generate
    hook = CrossAttnHook(model)
    try:
        (
            pred_features,
            pred_features_lens,
            _pred_prompt_features,
            _pred_prompt_features_lens,
        ) = model.sample(
            tokens=chunk_tokens_int,
            prompt_tokens=prompt_tokens,
            prompt_features=prompt_features,
            prompt_features_lens=prompt_features_lens,
            speed=args.speed,
            t_shift=args.t_shift,
            duration="predict",
            num_step=args.num_step,
            guidance_scale=args.guidance_scale,
        )
    finally:
        hook.remove()

    # process hook
    last_step = take_last_ode_step(hook)
    all_steps = take_all_ode_steps(hook)
    if not last_step:
        print("[probe] hook captured nothing — model has no CrossMultiheadAttentionWeights?")
        return

    # full-resolution T_q = max across layers
    full_tq = max(w.shape[1] for _, w in last_step)
    print(f"[probe] detected full-resolution T_q = {full_tq}, "
          f"prediction window = [{prompt_speech_frames}, "
          f"{prompt_speech_frames + args.window_size})")
    print(f"[probe] captured decoder/ODE evaluations = {len(all_steps)}")

    # per-layer stats
    rows = []
    for name, w in last_step:
        s = compute_layer_stats(w, prompt_speech_frames, args.window_size, full_tq)
        if not s:
            continue
        s["name"] = layer_key(name)
        rows.append(s)

    # Report: two tables — hard (argmax) pointer and soft (expected-value) pointer.
    def _print_table(rows_sorted, title, keys):
        print("\n" + "=" * 135)
        print(f"[{title}]")
        header = f"{'layer':<28} {'ds':>3} {'pw':>4} {'Tk':>4}   "
        for k, w in keys:
            header += f"{k:>{w}} "
        print(header)
        print("-" * 135)
        for r in rows_sorted:
            line = (f"{r['name']:<28} {r['ds']:>3} {r['pw_rows']:>4} {r['Tk']:>4}   ")
            for k, w in keys:
                val = r[k]
                if isinstance(val, float) and np.isnan(val):
                    line += f"{'nan':>{w}} "
                elif k == "span" or k == "span_exp":
                    line += f"{val:>{w}.0f} "
                else:
                    line += f"{val:>{w}.3f} "
            print(line)
        print("=" * 135)

    hard_keys = [
        ("sharp", 6), ("top3", 6), ("H_norm", 7),
        ("mono", 5), ("diag_r2", 7), ("span", 5),
    ]
    soft_keys = [
        ("sharp", 6), ("H_norm", 7),
        ("mono_exp", 8), ("diag_slope_exp", 14), ("diag_r2_exp", 11), ("span_exp", 8),
    ]
    # Unify column keys with the stored dict names.
    HARD_MAP = {"sharp": "sharpness", "top3": "top3_mass", "H_norm": "entropy_norm",
                "mono": "monotonicity", "diag_r2": "diag_r2", "span": "span"}
    SOFT_MAP = {"sharp": "sharpness", "H_norm": "entropy_norm",
                "mono_exp": "mono_exp", "diag_slope_exp": "diag_slope_exp",
                "diag_r2_exp": "diag_r2_exp", "span_exp": "span_exp"}

    def _remap(rows, mp):
        out = []
        for r in rows:
            o = dict(r)
            for k, src in mp.items():
                o[k] = r[src]
            out.append(o)
        return out

    rows_hard = sorted(_remap(rows, HARD_MAP), key=lambda r: -r["sharp"])
    rows_soft = sorted(_remap(rows, SOFT_MAP),
                       key=lambda r: (-(r["diag_r2_exp"] if not np.isnan(r["diag_r2_exp"]) else -1),
                                      -r["mono_exp"]))

    _print_table(rows_hard, "HARD pointer (argmax)", hard_keys)
    _print_table(rows_soft, "SOFT pointer (expected value)", soft_keys)

    arr = lambda k: np.array([r[k] for r in rows if not np.isnan(r[k])])
    print(
        f"\nMEAN over layers:\n"
        f"  sharpness      = {arr('sharpness').mean():.3f}\n"
        f"  entropy_norm   = {arr('entropy_norm').mean():.3f}\n"
        f"  mono  (argmax) = {arr('monotonicity').mean():.3f}    "
        f"mono_exp (soft) = {arr('mono_exp').mean():.3f}\n"
        f"  diag_r2 (arg)  = {arr('diag_r2').mean():.3f}    "
        f"diag_r2_exp     = {arr('diag_r2_exp').mean():.3f}"
    )

    # figures
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(pred_features_lens):
        pred_len = int(pred_features_lens.reshape(-1)[0].item())
    else:
        pred_len = int(pred_features_lens)
    # model.sample() returns x1_wo_prompt as the first value, so this tensor is
    # already the generated current chunk rather than prompt+chunk.
    chunk_start = 0
    chunk_end = min(pred_len, args.window_size, pred_features.size(1))
    chunk_mel = pred_features[:, chunk_start:chunk_end, :]
    if chunk_mel.size(1) == 0:
        raise RuntimeError(
            "Empty generated chunk after cropping. "
            f"pred_features={tuple(pred_features.shape)}, "
            f"pred_len={pred_len}, prompt_speech_frames={prompt_speech_frames}"
        )
    if chunk_mel.size(1) < args.window_size:
        chunk_mel = torch.nn.functional.pad(
            chunk_mel,
            (0, 0, 0, args.window_size - chunk_mel.size(1)),
        )
    chunk_wav_path = out_dir / "cross_attention_chunk.wav"
    if args.save_chunk_audio:
        vocoder = load_vocoder(args.vocoder_path, device)
        wav = (
            vocoder.decode(chunk_mel.permute(0, 2, 1) / args.feat_scale)
            .squeeze(1)
            .clamp(-1, 1)
        )
        torchaudio.save(
            str(chunk_wav_path),
            wav.detach().cpu(),
            sample_rate=args.sampling_rate,
        )
        print(f"[paper] saved chunk wav: {chunk_wav_path}")

    # --- Figure 1: argmax + expected-value trajectories per layer ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    for r in rows:
        q_a = np.arange(len(r["argmax_trace"]))
        q_e = np.arange(len(r["expected_trace"]))
        ax1.plot(q_a, r["argmax_trace"], alpha=0.55, label=r["name"])
        ax2.plot(q_e, r["expected_trace"], alpha=0.55, label=r["name"])
    for ax, title in [(ax1, "ARGMAX (hard)"),
                      (ax2, "EXPECTED VALUE (soft)")]:
        if 0 < prompt_token_len <= (rows[0]["Tk"] if rows else 1):
            ax.axhline(prompt_token_len - 0.5, color="red", linestyle="--",
                       linewidth=1, alpha=0.8,
                       label="prompt/target boundary")
        ax.set_xlabel("prediction-window Q frame (within-layer resolution)")
        ax.set_title(f"Per-layer pointer trace — {title}")
        ax.grid(alpha=0.3)
    ax1.set_ylabel("text token idx")
    ax1.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "probe_pointer_traces.png", dpi=130)
    plt.close(fig)

    # --- Figure 2: best layer per criterion — show argmax + expected overlay ---
    best_sharp = max(rows, key=lambda r: r["sharpness"])
    best_soft = max(rows, key=lambda r: (r["diag_r2_exp"]
                                          if not np.isnan(r["diag_r2_exp"]) else -1))

    def _plot_layer(ax, r):
        W = r["weights_avg_hd"]
        im = ax.imshow(W, aspect="auto", origin="lower", interpolation="nearest")
        ax.plot(r["argmax_trace"], np.arange(W.shape[0]),
                color="white", linewidth=1.0, alpha=0.85, label="argmax")
        ax.plot(r["expected_trace"], np.arange(W.shape[0]),
                color="cyan", linewidth=1.2, alpha=0.85, label="expected")
        if 0 < prompt_token_len < W.shape[1]:
            ax.axvline(prompt_token_len - 0.5, color="red",
                       linestyle="--", linewidth=1, label="prompt/target")
        if W.shape[1] <= 100:
            xt = list(range(W.shape[1]))
            lbls = all_labels + ["<pad>"] * max(0, W.shape[1] - len(all_labels))
            ax.set_xticks(xt)
            ax.set_xticklabels(lbls[:W.shape[1]], rotation=90, fontsize=7)
        ax.set_xlabel("text token")
        ax.set_ylabel("Q frame")
        ax.set_title(
            f"{r['name']}  sharp={r['sharpness']:.3f}  "
            f"H_n={r['entropy_norm']:.3f}  "
            f"mono_arg={r['monotonicity']:.2f}  r2_arg={r['diag_r2']:.2f}  "
            f"mono_exp={r['mono_exp']:.2f}  r2_exp={r['diag_r2_exp']:.2f}"
        )
        ax.legend(loc="upper right", fontsize=8)
        return im

    def _clean_token_label(token: str) -> str:
        if token in ("<blk>", "<pad>", "<unk>"):
            return token
        # SentencePiece-style word boundary.
        token = token.replace("▁", " ")
        # Some tokenizers use explicit whitespace markers.
        token = token.replace("<space>", " ")
        return token

    def _target_alignment_score(r):
        W_full = r["weights_avg_hd"]
        x0 = min(prompt_token_len, W_full.shape[1])
        n = min(len(chunk_tokens_str), max(0, W_full.shape[1] - x0))
        if n <= 1:
            return -1.0
        W = W_full[:, x0 : x0 + n]
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
        x = np.arange(n, dtype=np.float32)
        expected = (W * x[None, :]).sum(axis=1)
        q = np.arange(expected.shape[0], dtype=np.float32)
        if expected.std() < 1e-6:
            diag_r2 = 0.0
        else:
            slope = ((q - q.mean()) * (expected - expected.mean())).sum() / np.maximum(
                ((q - q.mean()) ** 2).sum(), 1e-9
            )
            pred = slope * q + (expected.mean() - slope * q.mean())
            ss_res = ((expected - pred) ** 2).sum()
            ss_tot = ((expected - expected.mean()) ** 2).sum()
            diag_r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
        mono = np.mean(expected[1:] >= expected[:-1]) if expected.shape[0] > 1 else 0.0
        sharp = W.max(axis=1).mean()
        target_mass = W_full[:, x0 : x0 + n].sum(axis=1).mean()
        return float(diag_r2 + 0.25 * mono + 0.15 * sharp + 0.05 * target_mass)

    def _get_prediction_rows(attn_3d: torch.Tensor):
        """attn_3d: (H, T_q, T_k), return current-window rows as numpy."""
        H, Tq, Tk = attn_3d.shape
        ds = max(1, round(full_tq / Tq))
        s = prompt_speech_frames // ds
        e = min(Tq, s + max(1, args.window_size // ds))
        if s >= Tq:
            s = max(0, Tq - max(1, args.window_size // ds))
        return attn_3d[:, s:e, :].detach().cpu().numpy()

    def _head_target_score(W_full: np.ndarray):
        x0 = min(prompt_token_len, W_full.shape[1])
        n = min(len(chunk_tokens_str), max(0, W_full.shape[1] - x0))
        if n <= 2:
            return -1.0, {}
        W_raw = W_full[:, x0 : x0 + n]
        target_mass = float(W_raw.sum(axis=1).mean())
        W = W_raw / np.maximum(W_raw.sum(axis=1, keepdims=True), 1e-9)
        x = np.arange(n, dtype=np.float32)
        expected = (W * x[None, :]).sum(axis=1)
        argmax = W.argmax(axis=1).astype(np.float32)
        q = np.arange(expected.shape[0], dtype=np.float32)

        def _fit(trace):
            if trace.shape[0] < 2 or trace.std() < 1e-6:
                return 0.0, 0.0, 0.0
            slope = ((q - q.mean()) * (trace - trace.mean())).sum() / np.maximum(
                ((q - q.mean()) ** 2).sum(), 1e-9
            )
            pred = slope * q + (trace.mean() - slope * q.mean())
            ss_res = ((trace - pred) ** 2).sum()
            ss_tot = ((trace - trace.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
            mono = float(np.mean(trace[1:] >= trace[:-1]))
            span = float(trace.max() - trace.min())
            return float(slope), float(r2), mono if slope >= -1e-4 else 0.0

        slope_exp, r2_exp, mono_exp = _fit(expected)
        slope_arg, r2_arg, mono_arg = _fit(argmax)
        span_exp = float(expected.max() - expected.min())
        span_arg = float(argmax.max() - argmax.min())
        sharp = float(W.max(axis=1).mean())
        entropy = -(W * np.log(np.maximum(W, 1e-9))).sum(axis=1)
        entropy_norm = float((entropy / max(np.log(max(n, 2)), 1e-9)).mean())
        coverage = min(1.0, span_exp / max(n * 0.45, 1.0))
        if slope_exp <= -1e-4:
            coverage *= 0.25
        score = (
            0.55 * max(r2_exp, 0.0)
            + 0.20 * mono_exp
            + 0.15 * coverage
            + 0.08 * sharp
            + 0.02 * target_mass
        )
        meta = {
            "target_mass": target_mass,
            "sharp": sharp,
            "entropy_norm": entropy_norm,
            "slope_exp": slope_exp,
            "r2_exp": r2_exp,
            "mono_exp": mono_exp,
            "span_exp": span_exp,
            "r2_arg": r2_arg,
            "mono_arg": mono_arg,
            "span_arg": span_arg,
        }
        return float(score), meta

    def _select_target_head():
        best = None
        for name, attn in last_step:
            pred = _get_prediction_rows(attn)
            for h in range(pred.shape[0]):
                score, meta = _head_target_score(pred[h])
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "meta": meta,
                        "name": layer_key(name),
                        "head": h,
                        "weights": pred[h],
                    }
        return best

    def _plot_paper_target_heatmap(best):
        W_full = best["weights"]
        # Keep only current-chunk visible target tokens. The prompt/target
        # boundary is known from tokenizer text tokens, and the model attends
        # over prompt+target internally.
        x0 = min(prompt_token_len, W_full.shape[1])
        target_labels = [_clean_token_label(x) for x in chunk_tokens_str]
        n = min(len(target_labels), max(0, W_full.shape[1] - x0))
        W = W_full[:, x0 : x0 + n]
        target_labels = target_labels[:n]
        if W.shape[1] == 0:
            return
        # Renormalize within visible target tokens. This emphasizes which
        # current-chunk tokens compete with one another after removing prompt
        # tokens from the display.
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)

        fig_w = min(22, max(11, W.shape[1] * 0.34))
        fig_h = 6.4
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        im = ax.imshow(
            W,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=float(np.quantile(W, 0.995)),
        )
        ax.set_xlabel("visible text token in current chunk")
        ax.set_ylabel("acoustic frame in current chunk")
        ax.set_title("Cross-attention alignment over visible tokens", pad=42)

        token_idx = np.arange(W.shape[1], dtype=np.float32)
        expected = (W * token_idx[None, :]).sum(axis=1)
        ax.plot(
            expected,
            np.arange(W.shape[0]),
            color="white",
            linewidth=2.0,
            alpha=0.92,
            label="expected attention position",
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

        xt = np.arange(W.shape[1])
        ax.set_xticks(xt)
        ax.set_xticklabels(target_labels, rotation=0, fontsize=10)
        ax.tick_params(axis="x", pad=8)

        # Word-boundary guides make character-level tokens easier to read.
        word_starts = []
        for idx, lab in enumerate(target_labels):
            if lab.startswith(" ") and idx > 0:
                word_starts.append(idx)
                ax.axvline(idx - 0.5, color="white", linewidth=0.8, alpha=0.55)

        # Put word spans above the heatmap so the character-level tokens can be
        # read together as the visible text.
        starts = [0] + word_starts
        ends = word_starts + [W.shape[1]]
        trans = ax.get_xaxis_transform()
        for s, e in zip(starts, ends):
            word = "".join(target_labels[s:e]).strip()
            if not word:
                continue
            ax.text(
                (s + e - 1) / 2,
                1.025,
                word,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )
            ax.hlines(
                1.015,
                s - 0.45,
                e - 0.55,
                transform=trans,
                color="#555555",
                linewidth=0.8,
                clip_on=False,
            )

        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("attention weight")
        meta = best["meta"]
        ax.text(
            0.0,
            -0.28,
            (
                f"visible text: {chunk_text}    "
                f"layer={best['name']}, head={best['head']}, "
                f"target mass={meta['target_mass']:.2f}, "
                f"monotonicity={meta['mono_exp']:.2f}, "
                f"R^2={meta['r2_exp']:.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#333333",
        )
        fig.subplots_adjust(top=0.82, bottom=0.25, left=0.075, right=0.92)
        fig.savefig(out_dir / "cross_attention_target_tokens.png", dpi=220)
        plt.close(fig)

    def _safe_file_part(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

    def _target_slice(W_full: np.ndarray):
        x0 = min(prompt_token_len, W_full.shape[1])
        target_labels = [_clean_token_label(x) for x in chunk_tokens_str]
        n = min(len(target_labels), max(0, W_full.shape[1] - x0))
        W = W_full[:, x0 : x0 + n]
        target_labels = target_labels[:n]
        if W.shape[1] > 0:
            W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
        return W, target_labels

    def _add_word_labels(ax, target_labels: list[str], show_words: bool = True):
        word_starts = []
        for idx, lab in enumerate(target_labels):
            if lab.startswith(" ") and idx > 0:
                word_starts.append(idx)
                ax.axvline(idx - 0.5, color="white", linewidth=0.5, alpha=0.45)
        if not show_words:
            return
        starts = [0] + word_starts
        ends = word_starts + [len(target_labels)]
        trans = ax.get_xaxis_transform()
        for s, e in zip(starts, ends):
            word = "".join(target_labels[s:e]).strip()
            if not word:
                continue
            ax.text(
                (s + e - 1) / 2,
                1.025,
                word,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#222222",
                clip_on=False,
            )
            ax.hlines(
                1.015,
                s - 0.45,
                e - 0.55,
                transform=trans,
                color="#555555",
                linewidth=0.7,
                clip_on=False,
            )

    def _plot_single_head(
        ax,
        W: np.ndarray,
        target_labels: list[str],
        title: str,
        show_words: bool = True,
        title_position: str = "top",
    ):
        im = ax.imshow(
            W,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=float(np.quantile(W, 0.995)) if W.size else 1.0,
        )
        if W.shape[1] > 0:
            token_idx = np.arange(W.shape[1], dtype=np.float32)
            expected = (W * token_idx[None, :]).sum(axis=1)
            ax.plot(
                expected,
                np.arange(W.shape[0]),
                color="white",
                linewidth=1.0,
                alpha=0.9,
                label="expected",
            )
            argmax = W.argmax(axis=1)
            ax.plot(
                argmax,
                np.arange(W.shape[0]),
                color="#ff4d4d",
                linewidth=0.9,
                linestyle="--",
                alpha=0.9,
                label="argmax",
            )
            ax.legend(loc="lower right", fontsize=6, framealpha=0.75)
        _add_word_labels(ax, target_labels, show_words=show_words)
        if title_position == "top":
            ax.set_title(title, fontsize=8, pad=24 if show_words else 6)
        elif title_position == "bottom":
            ax.text(
                0.0,
                -0.28,
                title,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#333333",
            )
        ax.set_xlabel("target token", fontsize=7)
        ax.set_ylabel("chunk frame", fontsize=7)
        if W.shape[1] <= 45:
            ax.set_xticks(np.arange(W.shape[1]))
            ax.set_xticklabels(target_labels, rotation=90, fontsize=5)
        else:
            ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=6)
        return im

    def _dump_all_head_heatmaps():
        all_dir = out_dir / "all_last_step_heads"
        single_dir = all_dir / "single_heads"
        grid_dir = all_dir / "layer_grids"
        single_dir.mkdir(parents=True, exist_ok=True)
        grid_dir.mkdir(parents=True, exist_ok=True)
        manifest = []

        for module_idx, (name, attn) in enumerate(last_step):
            layer = layer_key(name)
            pred = _get_prediction_rows(attn)
            H = pred.shape[0]
            fig, axes = plt.subplots(
                H,
                1,
                figsize=(max(10, len(chunk_tokens_str) * 0.22), max(2.4 * H, 3.0)),
                squeeze=False,
            )
            grid_im = None
            for head_idx in range(H):
                W, labels = _target_slice(pred[head_idx])
                score, meta = _head_target_score(pred[head_idx])
                title = (
                    f"module={module_idx:02d} {layer} head={head_idx} "
                    f"score={score:.3f} R2={meta.get('r2_exp', 0.0):.2f} "
                    f"mono={meta.get('mono_exp', 0.0):.2f}"
                )
                grid_im = _plot_single_head(axes[head_idx, 0], W, labels, title)

                fig_single, ax_single = plt.subplots(
                    1,
                    1,
                    figsize=(max(10, len(chunk_tokens_str) * 0.25), 4.2),
                )
                im = _plot_single_head(ax_single, W, labels, title)
                fig_single.colorbar(im, ax=ax_single, fraction=0.025, pad=0.02)
                fig_single.tight_layout()
                single_name = (
                    f"module_{module_idx:02d}_{_safe_file_part(layer)}"
                    f"_head_{head_idx:02d}.png"
                )
                fig_single.savefig(single_dir / single_name, dpi=180)
                plt.close(fig_single)

                manifest.append(
                    {
                        "module_idx": module_idx,
                        "layer": layer,
                        "head": head_idx,
                        "score": score,
                        **meta,
                        "file": f"single_heads/{single_name}",
                    }
                )

            if grid_im is not None:
                fig.colorbar(grid_im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
            fig.suptitle(
                f"Final ODE step target-token cross-attention | module={module_idx:02d} {layer}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 0.98, 0.98))
            grid_name = f"module_{module_idx:02d}_{_safe_file_part(layer)}_all_heads.png"
            fig.savefig(grid_dir / grid_name, dpi=180)
            plt.close(fig)

        manifest = sorted(manifest, key=lambda x: x["score"], reverse=True)
        manifest_path = all_dir / "manifest.csv"
        with open(manifest_path, "w", encoding="utf-8") as f:
            keys = [
                "module_idx", "layer", "head", "score", "target_mass",
                "sharp", "entropy_norm", "slope_exp", "r2_exp", "mono_exp",
                "span_exp", "r2_arg", "mono_arg", "span_arg", "file",
            ]
            f.write(",".join(keys) + "\n")
            for row in manifest:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(
            f"[all-heads] wrote {len(manifest)} single-head heatmaps, "
            f"{len(last_step)} layer grids, manifest={manifest_path}"
        )

    def _dump_all_ode_step_heatmaps():
        all_dir = out_dir / "top_ode_step_heads"
        single_dir = all_dir / "top_heads"
        single_dir.mkdir(parents=True, exist_ok=True)

        manifest = []
        for step_idx, step in enumerate(all_steps):
            for module_idx, (name, attn) in enumerate(step):
                layer = layer_key(name)
                pred = _get_prediction_rows(attn)
                for head_idx in range(pred.shape[0]):
                    W, labels = _target_slice(pred[head_idx])
                    score, meta = _head_target_score(pred[head_idx])
                    manifest.append(
                        {
                            "step": step_idx,
                            "module_idx": module_idx,
                            "layer": layer,
                            "head": head_idx,
                            "score": score,
                            **meta,
                        }
                    )
                    # Keep arrays outside the CSV row, only while ranking.
                    manifest[-1]["_weights"] = W
                    manifest[-1]["_labels"] = labels

        manifest = sorted(manifest, key=lambda x: x["score"], reverse=True)

        # Save only the best heads as individual inspectable figures.
        top_k = min(max(1, args.ode_top_k), len(manifest))
        for rank, row in enumerate(manifest[:top_k], start=1):
            title = (
                f"rank={rank} step={row['step']:03d} module={row['module_idx']:02d} "
                f"{row['layer']} head={row['head']} score={row['score']:.3f} "
                f"R2={row.get('r2_exp', 0.0):.2f} mono={row.get('mono_exp', 0.0):.2f}"
            )
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(max(11, len(chunk_tokens_str) * 0.32), 5.2),
            )
            im = _plot_single_head(
                ax,
                row["_weights"],
                row["_labels"],
                title,
                show_words=True,
                title_position="bottom",
            )
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            fig.subplots_adjust(top=0.78, bottom=0.28, left=0.08, right=0.92)
            fname = (
                f"rank_{rank:03d}_step_{row['step']:03d}_module_{row['module_idx']:02d}_"
                f"{_safe_file_part(row['layer'])}_head_{row['head']:02d}.png"
            )
            fig.savefig(single_dir / fname, dpi=180)
            plt.close(fig)
            row["file"] = f"single_heads/{fname}"

        # A compact overview of the saved best heads.
        if top_k:
            ncols = 4
            nrows = int(np.ceil(top_k / ncols))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.6 * ncols, 3.0 * nrows),
                squeeze=False,
            )
            last_im = None
            for idx in range(nrows * ncols):
                ax = axes[idx // ncols, idx % ncols]
                if idx >= top_k:
                    ax.axis("off")
                    continue
                row = manifest[idx]
                title = (
                    f"#{idx + 1} step {row['step']} m{row['module_idx']} h{row['head']}\n"
                    f"score={row['score']:.2f} R2={row.get('r2_exp', 0.0):.2f}"
                )
                last_im = _plot_single_head(
                    ax,
                    row["_weights"],
                    row["_labels"],
                    title,
                    show_words=False,
                    title_position="top",
                )
            if last_im is not None:
                fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.012, pad=0.01)
            fig.suptitle(
                f"Top {top_k} cross-attention heads across all ODE evaluations",
                fontsize=12,
            )
            fig.savefig(all_dir / f"top{top_k}_all_ode_steps.png", dpi=180)
            plt.close(fig)

        manifest_path = all_dir / "manifest.csv"
        keys = [
            "step", "module_idx", "layer", "head", "score", "target_mass",
            "sharp", "entropy_norm", "slope_exp", "r2_exp", "mono_exp",
            "span_exp", "r2_arg", "mono_arg", "span_arg", "file",
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for row in manifest:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(
            f"[all-ode] evaluated {len(manifest)} step/layer/head maps, "
            f"saved top {top_k} heatmaps only, manifest={manifest_path}"
        )

    if args.dump_all_heads:
        _dump_all_head_heatmaps()
    if args.dump_all_ode_steps:
        _dump_all_ode_step_heatmaps()

    fig, axes = plt.subplots(1, 2,
                             figsize=(max(16, len(all_labels) * 0.55), 5.4),
                             sharey=False)
    im0 = _plot_layer(axes[0], best_sharp)
    axes[0].set_title("Sharpest (by top-1): " + axes[0].get_title())
    im1 = _plot_layer(axes[1], best_soft)
    axes[1].set_title("Cleanest SOFT pointer (by diag_r2_exp): " + axes[1].get_title())
    fig.colorbar(im0, ax=axes[0], fraction=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "probe_best_layers.png", dpi=130)
    plt.close(fig)

    # --- Figure 3: paper-friendly single heatmap ---
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(9, len(all_labels) * 0.18), 4.8),
    )
    im = _plot_layer(ax, best_sharp)
    ax.set_title("Cross-attention heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_attention_heatmap.png", dpi=180)
    plt.close(fig)

    best_target = max(rows, key=_target_alignment_score)
    best_target_head = _select_target_head()
    case_md = out_dir / "cross_attention_case.md"
    print(
        f"[paper] selected_layer_avg={best_target['name']} "
        f"target_score={_target_alignment_score(best_target):.4f} "
        f"sample_id={utt_id} chunk_text={chunk_text!r}"
    )
    print(
        f"[paper] selected_layer_head={best_target_head['name']} "
        f"head={best_target_head['head']} "
        f"target_score={best_target_head['score']:.4f} "
        f"meta={best_target_head['meta']}"
    )
    _plot_paper_target_heatmap(best_target_head)

    if args.save_chunk_audio:
        case_md.write_text(
            "\n".join(
                [
                    "# Cross-Attention Chunk Case",
                    "",
                    f"- sample_id: `{utt_id}`",
                    f"- prompt_text: {prompt_text}",
                    f"- full target text: {text}",
                    f"- visible text in this probed chunk: {chunk_text}",
                    f"- chunk wav: `{chunk_wav_path.name}`",
                    "- heatmap: `cross_attention_target_tokens.png`",
                    "",
                    "This wav and heatmap are produced from the same `model.sample` call. "
                    "The heatmap visualizes the final-step cross-attention over the visible "
                    "target tokens, while the wav is decoded from the generated Mel chunk.",
                    "",
                    "Interpretation note: the white curve is the attention-weighted token "
                    "position for each acoustic frame. It is a smoothed reading guide, not "
                    "a hard word boundary. The heatmap colors remain the primary evidence.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[paper] wrote case note: {case_md}")
    else:
        print("[paper] skipped case note update because --save-chunk-audio=0")

    print(f"\nSaved figures under {out_dir}/")


if __name__ == "__main__":
    main()
