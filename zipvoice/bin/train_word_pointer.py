#!/usr/bin/env python3
"""Standalone trainer/evaluator for ``WordPointer``.

Each training sample is built on the fly from a full LibriTTS cut:
  1. Pick a random 150-frame mel chunk ``mel[t : t + 150]``.
  2. Determine which words are *covered* by the chunk via the
     midpoint rule: word ``w`` is covered iff
     ``0.5 * (w_start_frame + w_end_frame) ∈ [t, t + 150)``.
     Covered words form ``[w_lo, w_hi)``; if no word is covered
     ``w_lo = w_hi = k`` where ``k`` is the count of words whose
     midpoint precedes ``t + 150`` (i.e. the "split" between the words
     before vs. after the chunk).
  3. Sample ``pad_left, pad_right ∈ {0, .., max_pad}`` uniformly. Clamp
     each against the available preceding / trailing word counts.
  4. The provided text is
     ``target_words[w_lo - pad_left : w_hi + pad_right]``,
     tokenized with the tokenizer.
  5. The label is the joint index
     ``label = pad_left * (max_pad + 1) + pad_right``.

Loss: ``F.cross_entropy`` over ``(max_pad + 1) ** 2`` joint classes.
"""

import argparse
import json
import logging
import math
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from lhotse.cut import Cut
from lhotse.utils import fix_random_seed
from torch.utils.tensorboard import SummaryWriter

from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.word_pointer import WordPointer
from zipvoice.tokenizer.tokenizer import LibriTTSTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict, setup_logger, str2bool


FRAMES_PER_SECOND = 24000.0 / 256.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, default="libritts")
    parser.add_argument("--model-config", type=Path, default=None,
                        help="Optional ZipVoice JSON (only used to read feat_dim).")
    parser.add_argument("--pretrained-ckpt", type=Path, default=None,
                        help="Optional warm-start ckpt loaded with strict=False.")
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--chunk-frames", type=int, default=150,
                        help="Length of the mel chunk fed to the pointer.")
    parser.add_argument("--max-pad", type=int, default=4,
                        help="Pad words sampled from {0..max_pad} on each side.")
    parser.add_argument("--min-len", type=float, default=2.0)
    parser.add_argument("--max-len", type=float, default=20.0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--mel-encoder-layers", type=int, default=2)
    parser.add_argument("--text-encoder-layers", type=int, default=2)
    parser.add_argument("--cross-attn-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--num-eval-batches", type=int, default=50)
    parser.add_argument("--num-tb-samples", type=int, default=6,
                        help="Number of fixed dev examples to log to TensorBoard.")
    parser.add_argument("--tb-sample-topk", type=int, default=5,
                        help="How many WordPointer predictions to show per TB sample.")
    parser.add_argument("--tb-sample-seed", type=int, default=20260514,
                        help="Seed used to build fixed TensorBoard debug samples.")
    parser.add_argument("--augment-prob", type=float, default=0.0,
                        help="Probability of applying feature-domain augmentation.")
    parser.add_argument("--volume-augment-prob", type=float, default=0.0,
                        help="Probability of applying log-mel volume perturbation.")
    parser.add_argument("--volume-db-min", type=float, default=-6.0,
                        help="Minimum gain in dB for volume perturbation.")
    parser.add_argument("--volume-db-max", type=float, default=6.0,
                        help="Maximum gain in dB for volume perturbation.")
    parser.add_argument("--speed-augment-prob", type=float, default=0.0,
                        help="Probability of applying time-axis speed perturbation.")
    parser.add_argument("--speed-min", type=float, default=0.9,
                        help="Minimum speed factor. Values <1 stretch content.")
    parser.add_argument("--speed-max", type=float, default=1.1,
                        help="Maximum speed factor. Values >1 compress content.")
    parser.add_argument("--noise-augment-prob", type=float, default=0.0,
                        help="Probability of adding feature-domain Gaussian noise.")
    parser.add_argument("--noise-std-min", type=float, default=0.005,
                        help="Minimum noise std multiplier relative to sample std.")
    parser.add_argument("--noise-std-max", type=float, default=0.03,
                        help="Maximum noise std multiplier relative to sample std.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=20260507)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--tb", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    TtsDataModule.add_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------


def _extract_words(ali) -> List:
    if ali is None:
        return []
    if isinstance(ali, dict):
        return ali.get("words", [])
    if hasattr(ali, "words"):
        return ali.words
    return []


def _word_start_dur(w) -> Tuple[float, float]:
    if isinstance(w, dict):
        return float(w.get("start", 0.0)), float(w.get("duration", 0.0))
    return float(getattr(w, "start", 0.0)), float(getattr(w, "duration", 0.0))


def build_pointer_batch(
    batch,
    tokenizer: LibriTTSTokenizer,
    max_pad: int,
    chunk_frames: int,
    rng: random.Random,
    pad_id: int,
    return_debug: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Convert a TTS dataloader batch into a WordPointer training batch.

    Returns ``None`` when no sample in the batch is usable.
    """
    features = batch["features"]  # (B, T, feat_dim)
    features_lens = batch["features_lens"]
    texts = batch.get("text", [None] * features.size(0))
    alignments = batch.get("alignment", None)

    out_mel: List[torch.Tensor] = []
    out_mel_lens: List[int] = []
    out_tokens: List[List[int]] = []
    out_token_lens: List[int] = []
    out_labels: List[int] = []
    debug_items: List[Dict] = []

    n_classes = (max_pad + 1) ** 2

    for i in range(features.size(0)):
        T_mel = int(features_lens[i].item())
        if T_mel < chunk_frames:
            continue

        ali = alignments[i] if alignments is not None and i < len(alignments) else None
        words_ali = _extract_words(ali)
        text = texts[i] if texts is not None and i < len(texts) else None
        if not text:
            continue
        target_words = text.split()
        if not target_words or not words_ali:
            continue

        n_words = min(len(target_words), len(words_ali))
        if n_words == 0:
            continue
        target_words = target_words[:n_words]
        words_ali = words_ali[:n_words]

        mids = []
        for w in words_ali:
            ws, wd = _word_start_dur(w)
            mids.append((ws + wd / 2.0) * FRAMES_PER_SECOND)

        t0 = rng.randint(0, T_mel - chunk_frames)
        t1 = t0 + chunk_frames

        covered = [k for k, m in enumerate(mids) if t0 <= m < t1]
        if covered:
            w_lo = covered[0]
            w_hi = covered[-1] + 1
        else:
            split = sum(1 for m in mids if m < t1)
            w_lo = w_hi = split

        req_l = rng.randint(0, max_pad)
        req_r = rng.randint(0, max_pad)
        act_l = min(req_l, w_lo)
        act_r = min(req_r, n_words - w_hi)

        sel_lo = w_lo - act_l
        sel_hi = w_hi + act_r
        if sel_hi <= sel_lo:
            # Empty selection (no covered words and 0 padding both sides). Skip.
            continue

        text_sub = " ".join(target_words[sel_lo:sel_hi])
        token_ids = tokenizer.texts_to_token_ids([text_sub])[0]
        if not token_ids:
            continue

        out_mel.append(features[i, t0:t1, :])
        out_mel_lens.append(chunk_frames)
        out_tokens.append(token_ids)
        out_token_lens.append(len(token_ids))
        label = act_l * (max_pad + 1) + act_r
        assert 0 <= label < n_classes, (label, act_l, act_r, max_pad)
        out_labels.append(label)
        if return_debug:
            debug_items.append(
                {
                    "text_window": text_sub,
                    "full_text": text,
                    "t0": t0,
                    "t1": t1,
                    "w_lo": w_lo,
                    "w_hi": w_hi,
                    "sel_lo": sel_lo,
                    "sel_hi": sel_hi,
                    "label": label,
                    "gt_left": act_l,
                    "gt_right": act_r,
                    "covered_words": " ".join(target_words[w_lo:w_hi]),
                    "selected_words": " ".join(target_words[sel_lo:sel_hi]),
                    "token_len": len(token_ids),
                }
            )

    if not out_mel:
        return None

    mel = torch.stack(out_mel, dim=0)
    mel_lens = torch.tensor(out_mel_lens, dtype=torch.long)
    L_max = max(out_token_lens)
    tokens = torch.full((len(out_tokens), L_max), pad_id, dtype=torch.long)
    for j, ids in enumerate(out_tokens):
        tokens[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    token_lens = torch.tensor(out_token_lens, dtype=torch.long)
    labels = torch.tensor(out_labels, dtype=torch.long)

    out = {
        "mel": mel,
        "mel_lens": mel_lens,
        "tokens": tokens,
        "token_lens": token_lens,
        "labels": labels,
    }
    if return_debug:
        out["debug"] = debug_items
    return out


def _time_warp_fixed_len(mel: torch.Tensor, speed: float) -> torch.Tensor:
    """Speed-perturb a single ``(T, C)`` mel chunk and keep its original length."""
    if speed <= 0:
        return mel

    T = mel.size(0)
    new_t = max(2, int(round(T / speed)))
    x = mel.transpose(0, 1).unsqueeze(0)  # (1, C, T)
    x = F.interpolate(x, size=new_t, mode="linear", align_corners=False)
    warped = x.squeeze(0).transpose(0, 1)  # (new_t, C)

    if new_t == T:
        return warped
    if new_t > T:
        start = (new_t - T) // 2
        return warped[start:start + T]

    pad_left = (T - new_t) // 2
    pad_right = T - new_t - pad_left
    pieces = []
    if pad_left > 0:
        pieces.append(warped[:1].expand(pad_left, -1))
    pieces.append(warped)
    if pad_right > 0:
        pieces.append(warped[-1:].expand(pad_right, -1))
    return torch.cat(pieces, dim=0)


def augment_mel_batch(
    mel: torch.Tensor,
    params: AttributeDict,
    rng: random.Random,
) -> torch.Tensor:
    """Apply train-only feature-domain augmentation to unscaled log-mel chunks."""
    if params.augment_prob <= 0.0 or rng.random() >= params.augment_prob:
        return mel

    out = mel.clone()
    for i in range(out.size(0)):
        if params.volume_augment_prob > 0.0 and rng.random() < params.volume_augment_prob:
            gain_db = rng.uniform(params.volume_db_min, params.volume_db_max)
            # Vocos fbank is log-like; amplitude gain maps to an additive offset.
            out[i] = out[i] + gain_db * math.log(10.0) / 20.0

        if params.speed_augment_prob > 0.0 and rng.random() < params.speed_augment_prob:
            speed = rng.uniform(params.speed_min, params.speed_max)
            out[i] = _time_warp_fixed_len(out[i], speed)

        if params.noise_augment_prob > 0.0 and rng.random() < params.noise_augment_prob:
            noise_scale = rng.uniform(params.noise_std_min, params.noise_std_max)
            sample_std = out[i].std().clamp_min(1.0e-4)
            out[i] = out[i] + torch.randn_like(out[i]) * sample_std * noise_scale

    return out


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: WordPointer,
    dev_dl,
    params: AttributeDict,
    tokenizer: LibriTTSTokenizer,
    rng: random.Random,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    device = params.device
    n_pad = params.max_pad + 1

    correct = 0
    correct_l = 0
    correct_r = 0
    abs_err_l = 0.0
    abs_err_r = 0.0
    total = 0

    label_hist = {}

    for b_idx, batch in enumerate(dev_dl):
        if b_idx >= max_batches:
            break
        pb = build_pointer_batch(
            batch, tokenizer, params.max_pad, params.chunk_frames, rng, model.pad_id
        )
        if pb is None:
            continue
        mel = (pb["mel"] * params.feat_scale).to(device)
        mel_lens = pb["mel_lens"].to(device)
        tokens = pb["tokens"].to(device)
        token_lens = pb["token_lens"].to(device)
        labels = pb["labels"].to(device)

        logits = model(mel, mel_lens, tokens, token_lens)
        pred = logits.argmax(dim=-1)

        correct += int((pred == labels).sum().item())
        pred_l = (pred // n_pad).long()
        pred_r = (pred % n_pad).long()
        gt_l = (labels // n_pad).long()
        gt_r = (labels % n_pad).long()
        correct_l += int((pred_l == gt_l).sum().item())
        correct_r += int((pred_r == gt_r).sum().item())
        abs_err_l += float((pred_l - gt_l).abs().sum().item())
        abs_err_r += float((pred_r - gt_r).abs().sum().item())
        total += int(labels.numel())

        for v in labels.tolist():
            label_hist[v] = label_hist.get(v, 0) + 1

    out: Dict[str, float] = {"total": total}
    if total > 0:
        out["acc"] = correct / total
        out["acc_left"] = correct_l / total
        out["acc_right"] = correct_r / total
        out["mae_left"] = abs_err_l / total
        out["mae_right"] = abs_err_r / total
        # Diagnostic: most-frequent label fraction (random baseline = max class freq).
        if label_hist:
            out["majority_freq"] = max(label_hist.values()) / total
    return out


@torch.no_grad()
def log_tb_samples(
    model: WordPointer,
    dev_dl,
    params: AttributeDict,
    tokenizer: LibriTTSTokenizer,
    tb_writer: SummaryWriter,
    step: int,
) -> None:
    """Write fixed-looking dev examples to TensorBoard for qualitative debugging."""
    if params.num_tb_samples <= 0:
        return

    model.eval()
    device = params.device
    rng = random.Random(params.tb_sample_seed)
    rows = [
        "| idx | frames | gt | pred | top-k | text window | covered words | |",
        "|---:|---:|:---:|:---:|:---|:---|:---|:---|",
    ]
    logged = 0
    n_pad = params.max_pad + 1

    for batch in dev_dl:
        pb = build_pointer_batch(
            batch,
            tokenizer,
            params.max_pad,
            params.chunk_frames,
            rng,
            model.pad_id,
            return_debug=True,
        )
        if pb is None:
            continue

        mel = (pb["mel"] * params.feat_scale).to(device)
        mel_lens = pb["mel_lens"].to(device)
        tokens = pb["tokens"].to(device)
        token_lens = pb["token_lens"].to(device)
        labels = pb["labels"].to(device)

        logits = model(mel, mel_lens, tokens, token_lens)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)

        for j, info in enumerate(pb.get("debug", [])):
            if logged >= params.num_tb_samples:
                break

            gt_l = int(labels[j].item() // n_pad)
            gt_r = int(labels[j].item() % n_pad)
            pred_l = int(preds[j].item() // n_pad)
            pred_r = int(preds[j].item() % n_pad)
            vals, inds = torch.topk(
                probs[j], k=min(params.tb_sample_topk, probs.size(-1))
            )
            top_parts = []
            for prob, label in zip(vals.tolist(), inds.tolist()):
                l = int(label // n_pad)
                r = int(label % n_pad)
                top_parts.append(f"({l},{r}) {prob:.3f}")

            text_window = str(info["text_window"]).replace("|", "\\|")
            covered = str(info["covered_words"]).replace("|", "\\|")
            rows.append(
                f"| {logged} | {info['t0']}-{info['t1']} | "
                f"({gt_l},{gt_r}) | ({pred_l},{pred_r}) | "
                f"{'<br>'.join(top_parts)} | {text_window} | {covered} | |"
            )

            # TensorBoard image is C,H,W; normalize each mel chunk for display.
            mel_img = pb["mel"][j].transpose(0, 1).unsqueeze(0).float()
            mel_img = mel_img - mel_img.min()
            mel_img = mel_img / mel_img.max().clamp_min(1.0e-6)
            tb_writer.add_image(f"dev_samples/mel_{logged}", mel_img, step)
            tb_writer.add_text(
                f"dev_samples/sample_{logged}",
                "\n".join(
                    [
                        f"text_window: {info['text_window']}",
                        f"covered_words: {info['covered_words']}",
                        f"selected_words: {info['selected_words']}",
                        f"full_text: {info['full_text']}",
                        f"frames: {info['t0']}-{info['t1']}",
                        f"gt: ({gt_l},{gt_r}) label={int(labels[j].item())}",
                        f"pred: ({pred_l},{pred_r}) label={int(preds[j].item())}",
                        "topk: " + ", ".join(top_parts),
                    ]
                ),
                step,
            )
            logged += 1

        if logged >= params.num_tb_samples:
            break

    if logged > 0:
        tb_writer.add_text("dev_samples/summary", "\n".join(rows), step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    if total_steps <= 0:
        return base_lr
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * step / total_steps))


def remove_short_and_long_utt(c: Cut, min_len: float, max_len: float) -> bool:
    return min_len <= c.duration <= max_len


def tokenize_text(c: Cut, tokenizer):
    """Populate ``cut.supervisions[0].tokens`` so the TTS dataset's
    ``__getitem__`` can build a batch. The ids themselves are unused by
    WordPointer (we re-tokenize per-sample substrings), but the dataset
    still reads this attribute on every cut.
    """
    if hasattr(c.supervisions[0], "tokens") and c.supervisions[0].tokens is not None:
        c.supervisions[0].tokens = tokenizer.tokens_to_token_ids(
            [c.supervisions[0].tokens]
        )[0]
    else:
        c.supervisions[0].tokens = tokenizer.texts_to_token_ids(
            [c.supervisions[0].text]
        )[0]
    return c


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict(vars(args))
    params.exp_dir = Path(params.exp_dir)
    params.exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(f"{params.exp_dir}/log/log-train")

    fix_random_seed(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(params.seed)

    params.device = torch.device(params.device if torch.cuda.is_available() else "cpu")

    feat_dim = 100
    if params.model_config is not None:
        with open(params.model_config, "r") as f:
            model_cfg_full = json.load(f)
        feat_dim = int(model_cfg_full["model"].get("feat_dim", 100))

    tokenizer = LibriTTSTokenizer(token_file=str(params.token_file))
    vocab_size = tokenizer.vocab_size
    logging.info(
        f"vocab_size={vocab_size}  feat_dim={feat_dim}  "
        f"chunk_frames={params.chunk_frames}  max_pad={params.max_pad}"
    )

    model = WordPointer(
        vocab_size=vocab_size,
        max_pad=params.max_pad,
        mel_in_dim=feat_dim,
        dim=params.dim,
        mel_encoder_layers=params.mel_encoder_layers,
        text_encoder_layers=params.text_encoder_layers,
        cross_attn_layers=params.cross_attn_layers,
        num_heads=params.num_heads,
        feedforward_dim=params.feedforward_dim,
        dropout=params.dropout,
    ).to(params.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"WordPointer params: {n_params/1e6:.2f}M  num_classes={model.num_classes}")

    if params.pretrained_ckpt is not None:
        logging.info(f"Warm-starting from {params.pretrained_ckpt} (strict=False)")
        load_checkpoint(filename=params.pretrained_ckpt, model=model, strict=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )

    datamodule = TtsDataModule(args)
    train_cuts = datamodule.train_libritts_cuts()
    train_cuts = train_cuts.filter(
        partial(
            remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len
        )
    )
    dev_cuts = datamodule.dev_libritts_cuts()
    dev_cuts = dev_cuts.filter(
        partial(
            remove_short_and_long_utt, min_len=params.min_len, max_len=params.max_len
        )
    )

    _tokenize_text = partial(tokenize_text, tokenizer=tokenizer)
    train_cuts = train_cuts.map(_tokenize_text)
    dev_cuts = dev_cuts.map(_tokenize_text)

    train_dl = datamodule.train_dataloaders(train_cuts)
    dev_dl = datamodule.dev_dataloaders(dev_cuts)

    tb_writer = (
        SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard") if params.tb else None
    )

    train_rng = random.Random(params.seed)

    step = 0
    losses: List[float] = []
    accs: List[float] = []
    sample_counts: List[int] = []
    train_iter = iter(train_dl)
    while step < params.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_dl.sampler.set_epoch(step)
            train_iter = iter(train_dl)
            batch = next(train_iter)

        pb = build_pointer_batch(
            batch, tokenizer, params.max_pad, params.chunk_frames, train_rng, model.pad_id
        )
        if pb is None:
            step += 1
            continue

        model.train()

        mel_aug = augment_mel_batch(pb["mel"], params, train_rng)
        mel = (mel_aug * params.feat_scale).to(params.device)
        mel_lens = pb["mel_lens"].to(params.device)
        tokens = pb["tokens"].to(params.device)
        token_lens = pb["token_lens"].to(params.device)
        labels = pb["labels"].to(params.device)

        logits = model(mel, mel_lens, tokens, token_lens)
        loss = F.cross_entropy(logits, labels)

        for g in optimizer.param_groups:
            g["lr"] = cosine_lr(step, params.steps, params.lr)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.clip_grad_norm)
        optimizer.step()

        with torch.no_grad():
            acc = float((logits.argmax(dim=-1) == labels).float().mean().item())

        losses.append(float(loss.item()))
        accs.append(acc)
        sample_counts.append(int(labels.numel()))
        step += 1

        if step % params.log_interval == 0:
            mean_loss = sum(losses) / max(len(losses), 1)
            mean_acc = sum(accs) / max(len(accs), 1)
            mean_n = sum(sample_counts) / max(len(sample_counts), 1)
            cur_lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"step {step}/{params.steps}  loss={mean_loss:.4f} "
                f"acc={mean_acc:.3f}  n/batch={mean_n:.1f}  lr={cur_lr:.2e}"
            )
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", mean_loss, step)
                tb_writer.add_scalar("train/acc", mean_acc, step)
                tb_writer.add_scalar("train/n_per_batch", mean_n, step)
                tb_writer.add_scalar("train/lr", cur_lr, step)
            losses = []
            accs = []
            sample_counts = []

        if step % params.eval_every == 0 or step == params.steps:
            metrics = evaluate(
                model, dev_dl, params, tokenizer,
                rng=random.Random(params.eval_seed),
                max_batches=params.num_eval_batches,
            )
            logging.info("[eval @ step {}] ".format(step) + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            ))
            if tb_writer is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f"dev/{k}", v, step)
                log_tb_samples(
                    model=model,
                    dev_dl=dev_dl,
                    params=params,
                    tokenizer=tokenizer,
                    tb_writer=tb_writer,
                    step=step,
                )

    ckpt = {
        "model": model.state_dict(),
        "params": dict(params),
        "vocab_size": vocab_size,
        "max_pad": params.max_pad,
        "chunk_frames": params.chunk_frames,
        "tokenizer": {
            "name": params.tokenizer,
            "token_file": str(params.token_file),
        },
    }
    torch.save(ckpt, params.exp_dir / "word_pointer.pt")
    logging.info(f"Saved WordPointer ckpt to {params.exp_dir/'word_pointer.pt'}")

    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
