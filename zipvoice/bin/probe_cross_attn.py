#!/usr/bin/env python3
# Copyright    2026    Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Quantitative probe for the FM decoder's cross-attention weights in the
fixed-window streaming ZipVoice model.

Rationale: the FM decoder already learns an audio<->text alignment as a
byproduct of generation. If those weights are sharp and monotonic they could
be used as a "pointer" head (replacing the side-branch word-count predictor).
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
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch

from zipvoice.bin.infer_zipvoice_stream_fixed_window_crossattn import CrossAttnHook
from zipvoice.models.zipvoice_stream_fixedwindow_crossattn import ZipVoice
from zipvoice.tokenizer.tokenizer_stream import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import add_punctuation, load_prompt_wav, rms_norm


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


def read_sample(test_list: str, idx: int):
    with open(test_list) as f:
        rows = [r for r in csv.reader(f, delimiter="\t") if r]
    if idx >= len(rows):
        raise IndexError(f"sample-idx {idx} out of range (have {len(rows)})")
    r = rows[idx]
    if len(r) == 4:
        utt_id, text, prompt_wav, prompt_text = r
    elif len(r) == 3:
        utt_id, prompt_wav, prompt_text = r
        text = ""
    else:
        raise ValueError(f"unexpected tsv row: {r}")
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
        model.sample(
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
    if not last_step:
        print("[probe] hook captured nothing — model has no CrossMultiheadAttentionWeights?")
        return

    # full-resolution T_q = max across layers
    full_tq = max(w.shape[1] for _, w in last_step)
    print(f"[probe] detected full-resolution T_q = {full_tq}, "
          f"prediction window = [{prompt_speech_frames}, "
          f"{prompt_speech_frames + args.window_size})")

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

    print(f"\nSaved figures under {out_dir}/")


if __name__ == "__main__":
    main()
