#!/usr/bin/env python3
# Copyright    2026    Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Oracle feasibility probe for a mel-as-Q cross-attention pointer head.

Given the fixed-window streaming ZipVoice model, we freeze its text_encoder
and tokenizer, add a small learnable mel encoder + single cross-attention
layer whose Q=encoded mel and K/V=text_encoder output, and fit it for a
few hundred steps on a subset of LibriTTS dev cuts using GT alignments.
Evaluation then runs on held-out dev cuts and reports:

  - frame_acc:    fraction of frames whose argmax text token matches GT
  - sharpness:    mean top-1 attention mass (higher = sharper)
  - entropy_norm: mean entropy / log(T_k)
  - mono_arg:     argmax monotonicity (fraction of frames that don't go back)
  - diag_r2_arg:  R^2 of argmax trajectory vs frame index
  - mono_exp:     expected-value (soft) pointer monotonicity
  - diag_r2_exp:  R^2 of expected-value trajectory vs frame index

If these metrics clearly beat the FM cross-attn probe numbers
(sharpness 0.15, mono_exp 0.52, diag_r2 ~0), we have evidence that a
dedicated pointer head is trainable and the raw mel contains enough
alignment signal — no need to retrain FM.

Example:

    python3 -m zipvoice.bin.oracle_pointer \\
        --model-dir exp/zipvoice_libritts_0427_1717_stream_alignmask_fixedwindow_crossattn \\
        --checkpoint-name epoch-37.pt \\
        --tokenizer libritts \\
        --dataset libritts \\
        --manifest-dir aligned_data/fbank \\
        --max-duration 80 \\
        --num-train-steps 800 \\
        --num-eval-batches 20 \\
        --out-dir oracle_out
"""

import argparse
import math
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed

import safetensors.torch

from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.zipvoice_stream_fixedwindow_crossattn import ZipVoice
from zipvoice.tokenizer.tokenizer_stream import (
    EmiliaTokenizer,
    EspeakTokenizer,
    LibriTTSTokenizer,
    SimpleTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common_stream import AttributeDict, prepare_input


FPS = 24000 / 256  # 93.75


# ----------------------------- model loading -----------------------------

def build_tokenizer(tokenizer_name: str, token_file: Path, lang: str = "en-us"):
    if tokenizer_name == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    if tokenizer_name == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    if tokenizer_name == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=lang)
    return SimpleTokenizer(token_file=token_file)


def load_model(model_dir: Path, checkpoint_name: str, tokenizer) -> ZipVoice:
    ckpt = model_dir / checkpoint_name
    cfg_path = model_dir / "model.json"
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


def tokenize_text(c: Cut, tokenizer):
    if hasattr(c.supervisions[0], "tokens"):
        tokens = tokenizer.tokens_to_token_ids([c.supervisions[0].tokens])
    else:
        tokens = tokenizer.texts_to_token_ids([c.supervisions[0].text])
    c.supervisions[0].tokens = tokens[0]
    return c


# --------------------- GT per-frame token (oracle label) ---------------------

def _extract_words(ali):
    if ali is None:
        return []
    raw = ali.get("words", []) if isinstance(ali, dict) else getattr(ali, "words", [])
    out = []
    for w in raw:
        if isinstance(w, dict):
            out.append((str(w.get("symbol", "")),
                        float(w.get("start", 0.0)),
                        float(w.get("duration", 0.0))))
        else:
            out.append((str(getattr(w, "symbol", "")),
                        float(getattr(w, "start", 0.0)),
                        float(getattr(w, "duration", 0.0))))
    return out


def build_frame_targets(
    tokens_ids: List[int],
    alignment,
    id2text: dict,
    feat_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each frame return:
      - center_tk (long, -1 = unsupervised): center token of active word, for accuracy.
      - tk_start  (long, -1 = unsupervised): first token of active word.
      - tk_end    (long, -1 = unsupervised): one-past-last token of active word.

    Callers can build either a spike CE target (center) or a uniform KL
    target (over [tk_start, tk_end)) from these.
    """
    center = torch.full((feat_len,), -1, dtype=torch.long)
    tk_s = torch.full((feat_len,), -1, dtype=torch.long)
    tk_e = torch.full((feat_len,), -1, dtype=torch.long)
    words = _extract_words(alignment)
    if not words or not tokens_ids:
        return center, tk_s, tk_e

    full_text = "".join(id2text.get(int(t), "") for t in tokens_ids)
    full_text_upper = full_text.upper()

    cursor = 0
    for symbol, start_s, duration_s in words:
        sym_upper = symbol.upper()
        if not sym_upper:
            continue
        p = full_text_upper.find(sym_upper, cursor)
        if p < 0:
            continue
        a = p
        b = p + len(sym_upper)
        cursor = b
        center_tk = min(max(0, (a + b) // 2), len(tokens_ids) - 1)
        f_start = max(0, int(start_s * FPS))
        f_end = min(feat_len, int((start_s + duration_s) * FPS))
        if f_end > f_start:
            center[f_start:f_end] = center_tk
            tk_s[f_start:f_end] = a
            tk_e[f_start:f_end] = b
    return center, tk_s, tk_e


# ----------------------------- pointer head -----------------------------

class _ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad)
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, T, C)
        h = self.norm(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + self.act(h)


class PointerHead(nn.Module):
    """Mel encoder + cross-attention whose Q=encoded mel and
    K/V=frozen text_encoder output. Output is attention logits over
    text tokens for each mel frame.

    Larger pointer heads can be built by increasing `n_conv_blocks`,
    `hidden_dim`, and enabling `use_transformer`.
    """

    def __init__(self, mel_dim: int = 100, d_model: int = 100,
                 hidden_dim: int = 192, kernel: int = 5,
                 n_conv_blocks: int = 3, use_transformer: bool = True,
                 n_tf_layers: int = 2, tf_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Conv1d(mel_dim, hidden_dim, kernel_size=1)
        self.conv_stack = nn.ModuleList([
            _ConvBlock(hidden_dim, kernel) for _ in range(n_conv_blocks)
        ])
        self.use_transformer = use_transformer
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=tf_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                batch_first=True, activation="gelu", norm_first=True,
            )
            self.tf = nn.TransformerEncoder(layer, num_layers=n_tf_layers)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.log_scale = nn.Parameter(torch.tensor(math.log(d_model ** 0.5)))

    def forward(self, mel: torch.Tensor, text_cond: torch.Tensor,
                text_pad_mask: torch.Tensor,
                mel_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        mel:           (B, T_q, mel_dim)
        text_cond:     (B, T_k, d_model)
        text_pad_mask: (B, T_k), True = pad
        mel_pad_mask:  (B, T_q), True = pad  (optional, used by Transformer)
        Returns logits (B, T_q, T_k).
        """
        x = self.in_proj(mel.transpose(1, 2)).transpose(1, 2)  # (B, T, H)
        for blk in self.conv_stack:
            x = blk(x)
        if self.use_transformer:
            x = self.tf(x, src_key_padding_mask=mel_pad_mask)
        q = self.out_proj(x)                                   # (B, T, d_model)
        scale = torch.exp(-self.log_scale)
        logits = torch.einsum("bqd,bkd->bqk", q, text_cond) * scale
        logits = logits.masked_fill(text_pad_mask.unsqueeze(1), -1e4)
        return logits


# --------------------- dataset iteration helpers ---------------------

def _ok_dur(params):
    def _f(c: Cut) -> bool:
        return params.min_len <= c.duration <= params.max_len
    return _f


def build_dataloaders(params, tokenizer):
    """Return (train_dl, eval_dl).

    train_dl comes from `--train-source`:
      - "dev":            dev cuts (same as original oracle)
      - "libritts-train": train_libritts_cuts (the real train manifest)

    eval_dl always comes from dev cuts (held out when train-source != dev).
    """
    datamodule = TtsDataModule(argparse.Namespace(**params))
    tk_map = partial(tokenize_text, tokenizer=tokenizer)

    dev_cuts = (datamodule.dev_libritts_cuts() if params.dataset == "libritts"
                else datamodule.dev_custom_cuts(params.dev_manifest))
    dev_cuts = dev_cuts.filter(_ok_dur(params)).map(tk_map)
    eval_dl = datamodule.dev_dataloaders(dev_cuts)

    if params.train_source == "libritts-train":
        train_cuts = datamodule.train_libritts_cuts()
        train_cuts = train_cuts.filter(_ok_dur(params)).map(tk_map)
        # Use the simpler dev-style loader on train cuts so we don't need
        # to wire DynamicBucketingSampler epoch bookkeeping here.
        train_dl = datamodule.dev_dataloaders(train_cuts)
    else:
        train_dl = eval_dl

    return train_dl, eval_dl


# ----------------------------- eval metrics -----------------------------

def _mono_and_r2(trace: np.ndarray) -> Tuple[float, float]:
    n = len(trace)
    if n < 2:
        return float("nan"), float("nan")
    mono = float(np.mean(trace[1:] >= trace[:-1]))
    q = np.arange(n, dtype=np.float64)
    t = trace.astype(np.float64)
    if t.std() <= 1e-6:
        return mono, float("nan")
    slope = ((q - q.mean()) * (t - t.mean())).sum() / ((q - q.mean()) ** 2).sum()
    intercept = t.mean() - slope * q.mean()
    pred = slope * q + intercept
    ss_res = ((t - pred) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    return mono, float(r2)


def per_sample_metrics(
    attn_probs: np.ndarray,     # (T_q, T_k)
    center_gt: np.ndarray,       # (T_q,), -1 for unsupervised
    tk_s_gt: np.ndarray,         # (T_q,), -1 for unsupervised
    tk_e_gt: np.ndarray,         # (T_q,), -1 for unsupervised
    valid_mask: np.ndarray,
) -> dict:
    mask = valid_mask & (center_gt >= 0)
    if mask.sum() < 2:
        return {}

    w = attn_probs[mask]
    gtc = center_gt[mask]
    gts = tk_s_gt[mask]
    gte = tk_e_gt[mask]

    argmax = w.argmax(axis=-1)
    k_idx = np.arange(w.shape[-1], dtype=np.float64)
    expected = (w * k_idx[None, :]).sum(axis=-1)

    center_acc = float((argmax == gtc).mean())
    in_word = float(((argmax >= gts) & (argmax < gte)).mean())
    sharpness = float(w.max(axis=-1).mean())
    top3 = float(np.sort(w, axis=-1)[:, -3:].sum(axis=-1).mean()) if w.shape[-1] >= 3 else float("nan")
    logp = np.log(np.clip(w, 1e-9, None))
    entropy = -(w * logp).sum(axis=-1).mean()
    entropy_norm = float(entropy / max(math.log(max(w.shape[-1], 2)), 1e-9))

    mono_a, r2_a = _mono_and_r2(argmax)
    mono_e, r2_e = _mono_and_r2(expected)

    return {
        "center_acc": center_acc,
        "in_word": in_word,
        "sharpness": sharpness,
        "top3_mass": top3,
        "entropy_norm": entropy_norm,
        "mono_arg": mono_a,
        "diag_r2_arg": r2_a,
        "mono_exp": mono_e,
        "diag_r2_exp": r2_e,
        "argmax": argmax,
        "expected": expected,
        "gt": gtc,
        "attn": w,
    }


# ----------------------------- training -----------------------------

def encode_text_frozen(model: ZipVoice, tokens: List[List[int]]):
    """Run model.embed + model.text_encoder without gradients, return
    (text_cond, text_pad_mask). Device is inferred from model parameters."""
    with torch.no_grad():
        text_embed, tokens_lens = model.forward_text_embed(tokens)
        from zipvoice.utils.common import make_pad_mask
        text_pad_mask = make_pad_mask(tokens_lens, text_embed.size(1))
    return text_embed, text_pad_mask


def _build_gt_tensors(tokens, alignments, features_lens, id2text, device, T_q):
    B = features_lens.size(0)
    center_gt = torch.full((B, T_q), -1, dtype=torch.long, device=device)
    tk_s_gt = torch.full((B, T_q), -1, dtype=torch.long, device=device)
    tk_e_gt = torch.full((B, T_q), -1, dtype=torch.long, device=device)
    for i in range(B):
        flen = int(features_lens[i].item())
        ali = alignments[i] if alignments is not None and i < len(alignments) else None
        c_i, s_i, e_i = build_frame_targets(tokens[i], ali, id2text, flen)
        center_gt[i, :flen] = c_i.to(device)
        tk_s_gt[i, :flen] = s_i.to(device)
        tk_e_gt[i, :flen] = e_i.to(device)
    return center_gt, tk_s_gt, tk_e_gt


def _train_step(model, pointer, batch, params, device, id2text, opt):
    tokens, features, features_lens, alignments = prepare_input(
        params, batch, device, return_alignments=True,
    )
    T_q = features.size(1)

    center_gt, tk_s_gt, tk_e_gt = _build_gt_tensors(
        tokens, alignments, features_lens, id2text, device, T_q,
    )
    supervised = (tk_s_gt >= 0)

    text_cond, text_pad_mask = encode_text_frozen(model, tokens)
    T_k = text_cond.size(1)

    # mel padding mask — True where out-of-range (for transformer).
    seq = torch.arange(T_q, device=device)
    mel_pad_mask = seq[None, :] >= features_lens[:, None]

    logits = pointer(features, text_cond, text_pad_mask,
                     mel_pad_mask=mel_pad_mask)
    log_probs = F.log_softmax(logits, dim=-1)

    # CE on center token.
    ce_loss = F.cross_entropy(
        logits.reshape(-1, T_k), center_gt.reshape(-1), ignore_index=-1,
    )
    # Soft KL over [tk_start, tk_end).
    k_idx = torch.arange(T_k, device=device)
    in_range = (k_idx[None, None, :] >= tk_s_gt.unsqueeze(-1)) & (
        k_idx[None, None, :] < tk_e_gt.unsqueeze(-1)
    )
    soft_target = in_range.float()
    soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    frame_kl = -(soft_target * log_probs).sum(dim=-1)
    kl_loss = frame_kl[supervised].mean() if supervised.any() else frame_kl.sum() * 0.0

    loss = params.ce_weight * ce_loss + params.kl_weight * kl_loss

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(pointer.parameters(), 2.0)
    opt.step()

    with torch.no_grad():
        argmax = logits.argmax(dim=-1)
        if supervised.any():
            acc = (argmax[supervised] == center_gt[supervised]).float().mean().item()
            sup_idx = supervised.nonzero(as_tuple=False)
            aidx = argmax[sup_idx[:, 0], sup_idx[:, 1]]
            s_ref = tk_s_gt[sup_idx[:, 0], sup_idx[:, 1]]
            e_ref = tk_e_gt[sup_idx[:, 0], sup_idx[:, 1]]
            in_word = ((aidx >= s_ref) & (aidx < e_ref)).float().mean().item()
        else:
            acc = float("nan"); in_word = float("nan")
        sharp = log_probs.exp().max(dim=-1).values.mean().item()

    return {"loss": float(loss.item()),
            "ce": float(ce_loss.item()), "kl": float(kl_loss.item()),
            "center_acc": acc, "in_word": in_word, "sharp": sharp}


def _evaluate(model, pointer, eval_batches, params, device, id2text,
              num_plots: int = 0):
    pointer.eval()
    agg = {
        "center_acc": [], "in_word": [], "sharpness": [], "top3_mass": [],
        "entropy_norm": [], "mono_arg": [], "diag_r2_arg": [],
        "mono_exp": [], "diag_r2_exp": [],
    }
    plot_samples = []

    with torch.no_grad():
        for bi, batch in enumerate(eval_batches):
            tokens, features, features_lens, alignments = prepare_input(
                params, batch, device, return_alignments=True,
            )
            B, T_q, _ = features.shape
            text_cond, text_pad_mask = encode_text_frozen(model, tokens)
            seq = torch.arange(T_q, device=device)
            mel_pad_mask = seq[None, :] >= features_lens[:, None]
            logits = pointer(features, text_cond, text_pad_mask,
                             mel_pad_mask=mel_pad_mask)
            probs = F.softmax(logits, dim=-1)

            for i in range(B):
                flen = int(features_lens[i].item())
                ali = alignments[i] if alignments is not None and i < len(alignments) else None
                c_i, s_i, e_i = build_frame_targets(tokens[i], ali, id2text, flen)
                pad = T_q - flen
                c_i = np.concatenate([c_i.numpy(), np.full(pad, -1, dtype=np.int64)])
                s_i = np.concatenate([s_i.numpy(), np.full(pad, -1, dtype=np.int64)])
                e_i = np.concatenate([e_i.numpy(), np.full(pad, -1, dtype=np.int64)])
                valid = np.zeros(T_q, dtype=bool); valid[:flen] = True
                w_np = probs[i].detach().cpu().numpy()
                m = per_sample_metrics(w_np, c_i, s_i, e_i, valid)
                if not m:
                    continue
                for k in agg:
                    agg[k].append(m[k])
                if len(plot_samples) < num_plots:
                    plot_samples.append({
                        "batch": bi, "i": i, "feat_len": flen,
                        "w": m["attn"], "argmax": m["argmax"],
                        "expected": m["expected"], "gt": m["gt"],
                    })
    pointer.train()
    return agg, plot_samples


def _cycle(dataloader):
    while True:
        for b in dataloader:
            yield b


def run(params: AttributeDict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    token_file = params.model_dir / "tokens.txt"
    tokenizer = build_tokenizer(params.tokenizer, token_file, params.lang)
    id2text = tokenizer.id2token

    model = load_model(params.model_dir, params.checkpoint_name, tokenizer).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    pointer = PointerHead(
        mel_dim=100, d_model=100,
        hidden_dim=params.pointer_hidden,
        kernel=5,
        n_conv_blocks=params.pointer_conv,
        use_transformer=params.use_transformer,
        n_tf_layers=params.tf_layers,
        tf_heads=params.tf_heads,
    ).to(device)
    n_params = sum(p.numel() for p in pointer.parameters())
    logging.info(f"pointer head params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(pointer.parameters(), lr=params.lr,
                            weight_decay=params.weight_decay,
                            betas=(0.9, 0.95))
    # Cosine schedule with warmup.
    def _lr_lambda(step):
        if step < params.warmup_steps:
            return step / max(1, params.warmup_steps)
        t = (step - params.warmup_steps) / max(1, params.num_train_steps - params.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    fix_random_seed(params.seed)
    random.seed(params.seed)

    train_dl, eval_dl = build_dataloaders(params, tokenizer)

    # Pre-collect a fixed eval pool for repeatable mid-training metrics.
    eval_pool: List = []
    for b in eval_dl:
        eval_pool.append(b)
        if len(eval_pool) >= params.num_eval_batches:
            break
    logging.info(f"eval pool: {len(eval_pool)} batches")

    pointer.train()
    train_iter = _cycle(train_dl)

    log_every = 50
    eval_every = params.eval_every

    for step in range(params.num_train_steps):
        batch = next(train_iter)
        try:
            stats = _train_step(model, pointer, batch, params, device, id2text, opt)
        except RuntimeError as e:
            logging.warning(f"step {step} skipped: {e}")
            continue
        sched.step()

        if step % log_every == 0 or step == params.num_train_steps - 1:
            lr_now = sched.get_last_lr()[0]
            logging.info(
                f"step {step:5d}  loss={stats['loss']:.3f} "
                f"(ce={stats['ce']:.3f} kl={stats['kl']:.3f})  "
                f"center_acc={stats['center_acc']:.3f}  "
                f"in_word={stats['in_word']:.3f}  "
                f"sharp={stats['sharp']:.3f}  lr={lr_now:.2e}"
            )

        if eval_every > 0 and step > 0 and step % eval_every == 0:
            agg, _ = _evaluate(model, pointer, eval_pool, params, device,
                               id2text, num_plots=0)
            _log_eval(agg, tag=f"mid-eval @ step {step}")

    # final eval + plots
    agg, plots = _evaluate(model, pointer, eval_pool, params, device,
                           id2text, num_plots=params.num_plots)

    print("\n" + "=" * 70)
    print(f"ORACLE pointer head — held-out metrics "
          f"(train-source={params.train_source}, steps={params.num_train_steps})")
    print("=" * 70)
    print(f"eval cuts (supervised): {len(agg['center_acc'])}")
    for k in ["center_acc", "in_word", "sharpness", "top3_mass", "entropy_norm",
              "mono_arg", "diag_r2_arg", "mono_exp", "diag_r2_exp"]:
        vals = [v for v in agg[k] if not (isinstance(v, float) and math.isnan(v))]
        if vals:
            print(f"  {k:<16} mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")
    print("=" * 70)
    print("For reference — FM cross-attn probe (same model, no pointer training):")
    print("  sharpness ~0.15, mono_arg ~0.71, diag_r2_arg ~0.04, diag_r2_exp ~0.08")

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_sample_plots(plots, out_dir)
    print(f"\nSaved {len(plots)} attention plots to {out_dir}/")


def _log_eval(agg, tag: str) -> None:
    def _mean(k):
        vals = [v for v in agg[k] if not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    logging.info(
        f"[{tag}] center_acc={_mean('center_acc'):.3f} "
        f"in_word={_mean('in_word'):.3f} "
        f"sharp={_mean('sharpness'):.3f} "
        f"mono_arg={_mean('mono_arg'):.3f} "
        f"r2_arg={_mean('diag_r2_arg'):.3f} "
        f"r2_exp={_mean('diag_r2_exp'):.3f}"
    )


def _save_sample_plots(samples: List[dict], out_dir: Path) -> None:
    for s in samples:
        fig, ax = plt.subplots(figsize=(12, 4.4))
        ax.imshow(s["w"], aspect="auto", origin="lower",
                  interpolation="nearest", cmap="magma")
        q = np.arange(s["w"].shape[0])
        ax.plot(s["argmax"], q, color="white", linewidth=0.9,
                alpha=0.9, label="argmax")
        ax.plot(s["expected"], q, color="cyan", linewidth=1.1,
                alpha=0.9, label="expected")
        ax.plot(s["gt"], q, color="lime", linewidth=0.8,
                alpha=0.9, label="GT (center token)")
        ax.set_xlabel("text token idx")
        ax.set_ylabel("mel frame")
        ax.set_title(
            f"batch={s['batch']} sample={s['i']} feat_len={s['feat_len']}"
        )
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"oracle_b{s['batch']}_s{s['i']}.png", dpi=120)
        plt.close(fig)


# ----------------------------- main -----------------------------

def get_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--checkpoint-name", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="libritts",
                   choices=["emilia", "libritts", "espeak", "simple"])
    p.add_argument("--lang", type=str, default="en-us")
    p.add_argument("--dataset", type=str, default="libritts",
                   choices=["libritts", "custom"])
    p.add_argument("--dev-manifest", type=str, default=None)
    p.add_argument("--feat-scale", type=float, default=0.1)
    p.add_argument("--min-len", type=float, default=1.0)
    p.add_argument("--max-len", type=float, default=20.0)
    p.add_argument("--train-source", type=str, default="libritts-train",
                   choices=["dev", "libritts-train"],
                   help="Where training data comes from. dev = small-scale "
                        "oracle (same as earlier run); libritts-train = real "
                        "train manifest (held-out dev stays for eval).")
    p.add_argument("--num-eval-batches", type=int, default=20,
                   help="Held-out dev batches used for mid-training and "
                        "final evaluation.")
    p.add_argument("--num-train-steps", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=2000,
                   help="Run held-out eval every N training steps. 0 = off.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--ce-weight", type=float, default=1.0,
                   help="Weight on cross-entropy-to-center-token loss.")
    p.add_argument("--kl-weight", type=float, default=0.3,
                   help="Weight on KL-to-uniform-over-word loss.")
    p.add_argument("--pointer-hidden", type=int, default=192)
    p.add_argument("--pointer-conv", type=int, default=4,
                   help="Number of residual Conv blocks.")
    p.add_argument("--use-transformer", type=int, default=1,
                   help="0 or 1.")
    p.add_argument("--tf-layers", type=int, default=2)
    p.add_argument("--tf-heads", type=int, default=4)
    p.add_argument("--num-plots", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="oracle_out")

    TtsDataModule.add_arguments(p)
    return p


def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    params = AttributeDict()
    params.update(vars(args))
    params.model_dir = Path(params.model_dir)
    run(params)


if __name__ == "__main__":
    main()
