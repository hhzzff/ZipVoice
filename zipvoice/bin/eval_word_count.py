#!/usr/bin/env python3
# Copyright    2026    Xiaomi Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Standalone evaluator for the side-branch cumulative word-count predictor in
the fixed-window streaming ZipVoice model.

The predictor (`predict_word_count` in
zipvoice/models/zipvoice_stream_fixedwindow_crossattn.py) regresses the number
of words covered by the visible audio context (prompt + current chunk). It is
trained jointly with flow-matching via `wc_loss`. This script isolates it and
measures MAE / RMSE / bias / off-by-k hit rate on the dev set using GT mel
features and the same random-window distribution used during training.

Example:

    python3 -m zipvoice.bin.eval_word_count \\
        --model-name zipvoice \\
        --model-dir exp/zipvoice_libritts_0427_1717_stream_alignmask_fixedwindow_crossattn \\
        --checkpoint-name epoch-37.pt \\
        --tokenizer libritts \\
        --dataset libritts \\
        --manifest-dir aligned_data/fbank \\
        --max-duration 250 \\
        --feat-scale 0.1 \\
        --num-batches 200 \\
        --seed 42 \\
        --out-csv wc_eval.csv
"""

import argparse
import csv
import datetime as dt
import json
import logging
import random
from functools import partial
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate the cumulative-word-count predictor of the fixed-window "
            "streaming ZipVoice model on GT mel with training-style random windows."
        ),
    )

    parser.add_argument("--model-name", type=str, default="zipvoice",
                        choices=["zipvoice"])
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Experiment directory with checkpoint, model.json, tokens.txt.")
    parser.add_argument("--checkpoint-name", type=str, required=True)

    parser.add_argument("--tokenizer", type=str, default="libritts",
                        choices=["emilia", "libritts", "espeak", "simple"])
    parser.add_argument("--lang", type=str, default="en-us",
                        help="Language for espeak tokenizer.")

    parser.add_argument("--dataset", type=str, default="libritts",
                        choices=["libritts", "custom"])
    parser.add_argument("--dev-manifest", type=str, default=None,
                        help="Path to dev manifest (only for --dataset custom).")

    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--min-len", type=float, default=1.0,
                        help="Min cut duration (s) to include.")
    parser.add_argument("--max-len", type=float, default=20.0,
                        help="Max cut duration (s) to include.")

    parser.add_argument("--num-batches", type=int, default=-1,
                        help="Cap on batches to evaluate. -1 = all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=str, default=None,
                        help="Optional per-sample CSV dump path.")

    parser.add_argument("--num-plots", type=int, default=5,
                        help="How many samples to visualize (picked at abs-err "
                             "quantiles across a reservoir of ~100).")
    parser.add_argument("--plot-dir", type=str, default=None,
                        help="Directory to save plots. Defaults to "
                             "./wc_eval_plots/<timestamp>.")

    parser.add_argument("--mode", type=str, default="train-window",
                        choices=["train-window", "concat-infer"],
                        help="train-window: reproduce training's random mask window on "
                             "single utterances (with token truncation). "
                             "concat-infer: pair samples within a batch as "
                             "(prompt, target), concat their mel/tokens/alignment, "
                             "place mask_end inside the target region, and feed the "
                             "FULL untruncated prompt+target tokens to the text "
                             "encoder — mirrors the inference-time distribution.")
    parser.add_argument("--concat-window-size", type=int, default=30,
                        help="concat-infer: size of the 'current chunk' window "
                             "placed inside the target region.")

    TtsDataModule.add_arguments(parser)
    return parser


def build_tokenizer(params: AttributeDict, token_file: Path):
    if params.tokenizer == "emilia":
        return EmiliaTokenizer(token_file=token_file)
    if params.tokenizer == "libritts":
        return LibriTTSTokenizer(token_file=token_file)
    if params.tokenizer == "espeak":
        return EspeakTokenizer(token_file=token_file, lang=params.lang)
    assert params.tokenizer == "simple"
    return SimpleTokenizer(token_file=token_file)


def load_model(params: AttributeDict, tokenizer) -> ZipVoice:
    model_ckpt = params.model_dir / params.checkpoint_name
    model_config_path = params.model_dir / "model.json"
    for f in (model_ckpt, model_config_path):
        if not f.is_file():
            raise FileNotFoundError(f)

    with open(model_config_path) as f:
        model_config = json.load(f)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    model = ZipVoice(**model_config["model"], **tokenizer_config)

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, str(model_ckpt))
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=str(model_ckpt), model=model, strict=False)
    else:
        raise NotImplementedError(f"Unsupported checkpoint format: {model_ckpt}")

    return model


def tokenize_text(c: Cut, tokenizer):
    if hasattr(c.supervisions[0], "tokens"):
        tokens = tokenizer.tokens_to_token_ids([c.supervisions[0].tokens])
    else:
        tokens = tokenizer.texts_to_token_ids([c.supervisions[0].text])
    c.supervisions[0].tokens = tokens[0]
    return c


def build_dev_dataloader(params: AttributeDict, tokenizer, datamodule: TtsDataModule):
    if params.dataset == "libritts":
        dev_cuts = datamodule.dev_libritts_cuts()
    else:
        assert params.dataset == "custom"
        assert params.dev_manifest is not None
        dev_cuts = datamodule.dev_custom_cuts(params.dev_manifest)

    def _ok(c: Cut) -> bool:
        return params.min_len <= c.duration <= params.max_len

    dev_cuts = dev_cuts.filter(_ok).map(partial(tokenize_text, tokenizer=tokenizer))
    return datamodule.dev_dataloaders(dev_cuts)


def _extract_words_as_dicts(ali) -> List[dict]:
    """Normalize an alignment entry to a list of {symbol,start,duration} dicts."""
    if ali is None:
        return []
    if isinstance(ali, dict):
        raw = ali.get("words", [])
    elif hasattr(ali, "words"):
        raw = getattr(ali, "words")
    else:
        raw = []
    out = []
    for w in raw:
        if isinstance(w, dict):
            out.append({
                "symbol": str(w.get("symbol", "")),
                "start": float(w.get("start", 0.0)),
                "duration": float(w.get("duration", 0.0)),
            })
        else:
            out.append({
                "symbol": str(getattr(w, "symbol", "")),
                "start": float(getattr(w, "start", 0.0)),
                "duration": float(getattr(w, "duration", 0.0)),
            })
    return out


FPS = 24000 / 256


def _count_words_before(words: List[dict], frame_cutoff: float,
                        cap_frame: float = float("inf")) -> int:
    """Count words whose center frame is strictly less than `frame_cutoff`
    (optionally also < cap_frame). Mirrors training's GT rule."""
    n = 0
    for w in words:
        center = (w["start"] + w["duration"] / 2.0) * FPS
        if center < frame_cutoff and center < cap_frame:
            n += 1
    return n


def _build_train_window_batch(
    model,
    tokens: List[List[int]],
    features: torch.Tensor,
    features_lens: torch.Tensor,
    alignments,
    id2textDict: dict,
):
    """Reproduce training's random-window + token truncation."""
    prev_flag = model.training
    model.training = True
    try:
        (
            tokens_t,
            gt_features,
            feat_lens,
            speech_condition_mask,
            text_valid_lens,
            gt_cum,
            mask_ends,
        ) = model.condition_time_mask(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            alignments=alignments,
            id2textDict=id2textDict,
        )
    finally:
        model.training = prev_flag

    # mask_start from speech_condition_mask
    mask_bool = speech_condition_mask.detach().cpu()
    mask_starts = []
    for i in range(mask_bool.size(0)):
        idx = torch.nonzero(mask_bool[i], as_tuple=False)
        mask_starts.append(int(idx[0, 0]) if idx.numel() > 0 else 0)

    per_sample_plot = []
    for i in range(len(tokens_t)):
        per_sample_plot.append({
            "words": _extract_words_as_dicts(
                alignments[i] if alignments is not None and i < len(alignments) else None
            ),
            "prompt_len": None,  # no prompt in train-window mode
        })

    return {
        "tokens": tokens_t,
        "features_gt": gt_features,         # already zeroed past mask_end
        "features_vis": gt_features,         # same — the viz in train mode uses the un-masked clone from caller
        "features_lens": feat_lens,
        "text_valid_lens": text_valid_lens,
        "mask_starts": mask_starts,
        "mask_ends": mask_ends,
        "gt_cum": gt_cum,                    # tensor (B,) with -1 sentinel
        "per_sample_plot": per_sample_plot,
    }


def _build_concat_infer_batch(
    tokens: List[List[int]],
    features: torch.Tensor,
    features_lens: torch.Tensor,
    alignments,
    window_size: int,
    device: torch.device,
):
    """Pair up samples within a batch as (prompt, target). Concat mel +
    tokens + alignment, pick mask_end inside the target region, and return
    a padded batch ready for forward_text_train / predict_word_count.

    Text is *not* truncated — text_valid_lens spans the full prompt+target,
    mirroring the inference-time distribution where the text encoder sees
    prompt_text + target_text with no window-based cutoff.
    """
    B = features.size(0)
    feat_dim = features.size(-1)

    # Only keep samples with alignments so we can score.
    usable_idx = []
    aligned_words = []
    for i in range(B):
        words = _extract_words_as_dicts(
            alignments[i] if alignments is not None and i < len(alignments) else None
        )
        if len(words) > 0:
            usable_idx.append(i)
            aligned_words.append(words)
    if len(usable_idx) < 2:
        return None  # can't pair

    # Shuffle and pair up: (prompt, target). Drop odd tail.
    order = list(range(len(usable_idx)))
    random.shuffle(order)
    if len(order) % 2 == 1:
        order = order[:-1]
    pairs = [(order[k], order[k + 1]) for k in range(0, len(order), 2)]

    P = len(pairs)
    # Gather per-pair data.
    concat_tokens: List[List[int]] = []
    concat_lens = []
    mask_starts = []
    mask_ends = []
    prompt_lens = []
    gt_cum_list = []
    per_sample_plot = []
    max_total_len = 0
    for (pi, ti) in pairs:
        p_raw = usable_idx[pi]
        t_raw = usable_idx[ti]
        Tp = int(features_lens[p_raw].item())
        Tt = int(features_lens[t_raw].item())
        if Tt < 4:
            # target too short for a window; skip
            continue

        # Window inside the target region.
        ws = min(window_size, Tt)
        if Tt <= ws:
            ms_local, me_local = 0, Tt
        else:
            ms_local = random.randint(0, Tt - ws)
            me_local = ms_local + ws
        mask_start = Tp + ms_local
        mask_end = Tp + me_local

        concat_tokens.append(list(tokens[p_raw]) + list(tokens[t_raw]))
        total_len = Tp + Tt
        concat_lens.append(total_len)
        mask_starts.append(mask_start)
        mask_ends.append(mask_end)
        prompt_lens.append(Tp)
        max_total_len = max(max_total_len, total_len)

        p_words = aligned_words[pi]
        t_words = aligned_words[ti]
        # Count all prompt words that fall within the prompt cut,
        # plus target words whose center frame (relative to target) < me_local.
        wc = _count_words_before(p_words, frame_cutoff=float("inf"), cap_frame=Tp)
        wc += _count_words_before(t_words, frame_cutoff=me_local, cap_frame=Tt)
        gt_cum_list.append(float(wc))

        # For plotting: stitch words — prompt words stay as-is; target words
        # are shifted in time so that their "start" is relative to the concat
        # audio (so the plot's frame axis matches mel).
        t_words_shifted = [
            {"symbol": w["symbol"],
             "start": w["start"] + Tp / FPS,
             "duration": w["duration"]}
            for w in t_words
        ]
        per_sample_plot.append({
            "words": p_words + t_words_shifted,
            "prompt_len": Tp,
        })

    if not concat_tokens:
        return None

    # Build padded feature tensor on device. Copy prompt + target slices.
    P = len(concat_tokens)
    feat = torch.zeros(P, max_total_len, feat_dim,
                       dtype=features.dtype, device=device)
    for k, (pi, ti) in enumerate(pairs[:P]):
        p_raw = usable_idx[pi]
        t_raw = usable_idx[ti]
        Tp = int(features_lens[p_raw].item())
        Tt = int(features_lens[t_raw].item())
        feat[k, :Tp] = features[p_raw, :Tp]
        feat[k, Tp:Tp + Tt] = features[t_raw, :Tt]

    # Keep an un-masked copy for visualization.
    feat_vis = feat.clone()
    # Zero out frames past mask_end for the wc input (future = unknown).
    for k in range(P):
        feat[k, mask_ends[k]:, :] = 0.0

    feat_lens = torch.tensor(concat_lens, dtype=torch.long, device=device)
    tokens_lens = torch.tensor([len(t) for t in concat_tokens],
                               dtype=torch.long, device=device)
    gt_cum_t = torch.tensor(gt_cum_list, dtype=torch.float32, device=device)
    mask_ends_t = torch.tensor(mask_ends, dtype=torch.long, device=device)

    return {
        "tokens": concat_tokens,
        "features_gt": feat,                 # masked (post-mask_end zeroed)
        "features_vis": feat_vis,            # un-masked copy for plotting
        "features_lens": feat_lens,
        "text_valid_lens": tokens_lens,       # NO truncation
        "mask_starts": mask_starts,
        "mask_ends": mask_ends_t,
        "gt_cum": gt_cum_t,
        "per_sample_plot": per_sample_plot,
        "prompt_lens": prompt_lens,
    }


@torch.no_grad()
def run_eval(params: AttributeDict) -> Tuple[List[dict], dict, List[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer / model
    token_file = params.model_dir / "tokens.txt"
    tokenizer = build_tokenizer(params, token_file)
    id2textDict = tokenizer.id2token

    model = load_model(params, tokenizer).to(device)
    model.eval()

    # data
    datamodule = TtsDataModule(argparse.Namespace(**params))
    dev_dl = build_dev_dataloader(params, tokenizer, datamodule)

    # Seeds — fix python RNG because condition_time_mask uses random.randint / random.random
    fix_random_seed(params.seed)
    random.seed(params.seed)

    records: List[dict] = []
    plot_pool_cap = max(params.num_plots * 20, 100) if params.num_plots > 0 else 0
    plot_pool: List[dict] = []
    samples_seen = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dev_dl):
        if params.num_batches > 0 and n_batches >= params.num_batches:
            break

        tokens, features, features_lens, alignments = prepare_input(
            params, batch, device, return_alignments=True,
        )
        features_before_mask = features.detach().clone()

        if params.mode == "train-window":
            composite = _build_train_window_batch(
                model, tokens, features, features_lens,
                alignments, id2textDict,
            )
            features_vis = features_before_mask  # model zeroed `features` in place
        else:
            composite = _build_concat_infer_batch(
                tokens, features, features_lens, alignments,
                window_size=params.concat_window_size,
                device=device,
            )
            if composite is None:
                n_batches += 1
                continue
            features_vis = composite["features_vis"]

        tokens_t = composite["tokens"]
        gt_features = composite["features_gt"]
        feat_lens = composite["features_lens"]
        text_valid_lens = composite["text_valid_lens"]
        mask_starts = composite["mask_starts"]
        mask_ends = composite["mask_ends"]
        gt_cum_word_counts = composite["gt_cum"]
        per_sample_plot = composite["per_sample_plot"]

        text_cond, padding_mask, text_padding_mask = model.forward_text_train(
            tokens=tokens_t,
            features_lens=feat_lens,
            text_valid_lens=text_valid_lens,
        )

        seq = torch.arange(gt_features.size(1), device=device)
        context_mask = (seq[None, :] < mask_ends[:, None]) & (~padding_mask)

        pred_cum = model.predict_word_count(
            speech_features=gt_features,
            speech_valid_mask=context_mask,
            text_condition=text_cond,
            text_padding_mask=text_padding_mask,
        )

        pred_cum_cpu = pred_cum.detach().cpu().tolist()
        gt_cum_cpu = gt_cum_word_counts.detach().cpu().tolist()
        feat_lens_cpu = feat_lens.detach().cpu().tolist()
        mask_ends_cpu = mask_ends.detach().cpu().tolist()

        for i in range(len(gt_cum_cpu)):
            gt = gt_cum_cpu[i]
            if gt < 0:
                continue
            pred = pred_cum_cpu[i]
            feat_len = int(feat_lens_cpu[i])
            mask_end = int(mask_ends_cpu[i])
            mask_start = int(mask_starts[i])
            norm_pos = mask_end / max(feat_len, 1)

            rec = {
                "batch": batch_idx,
                "sample_in_batch": i,
                "feat_len": feat_len,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "norm_pos": norm_pos,
                "gt": float(gt),
                "pred": float(pred),
                "signed_err": float(pred - gt),
                "abs_err": abs(float(pred - gt)),
                "prompt_len": per_sample_plot[i].get("prompt_len"),
            }
            records.append(rec)

            if plot_pool_cap > 0:
                entry = {
                    **rec,
                    "feat_np": features_vis[i, :feat_len]
                        .detach().cpu().to(torch.float32).numpy(),
                    "words": per_sample_plot[i]["words"],
                }
                samples_seen += 1
                if len(plot_pool) < plot_pool_cap:
                    plot_pool.append(entry)
                else:
                    j = random.randint(0, samples_seen - 1)
                    if j < plot_pool_cap:
                        plot_pool[j] = entry

        n_batches += 1
        if n_batches % 10 == 0:
            logging.info(f"processed {n_batches} batches, {len(records)} scored samples")

    metrics = compute_metrics(records)
    return records, metrics, plot_pool


def compute_metrics(records: List[dict]) -> dict:
    if not records:
        return {"n": 0}

    preds = np.array([r["pred"] for r in records], dtype=np.float64)
    gts = np.array([r["gt"] for r in records], dtype=np.float64)
    err = preds - gts
    abs_err = np.abs(err)

    mae = float(abs_err.mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    bias = float(err.mean())

    gt_var = float(gts.var())
    r2 = 1.0 - float((err ** 2).mean() / gt_var) if gt_var > 0 else float("nan")

    def hit(k): return float((abs_err < k).mean())

    metrics = {
        "n": len(records),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "hit_<0.5": hit(0.5),
        "hit_<1.0": hit(1.0),
        "hit_<2.0": hit(2.0),
    }

    # Breakdown by normalized mask_end position
    norm_pos = np.array([r["norm_pos"] for r in records])
    pos_bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    pos_rows = []
    for lo, hi in pos_bins:
        m = (norm_pos >= lo) & (norm_pos < hi)
        if m.sum() > 0:
            pos_rows.append({
                "range": f"[{lo:.1f},{hi:.1f})",
                "n": int(m.sum()),
                "mae": float(abs_err[m].mean()),
                "rmse": float(np.sqrt((err[m] ** 2).mean())),
                "bias": float(err[m].mean()),
            })
    metrics["by_norm_pos"] = pos_rows

    # Breakdown by GT cumulative word count
    gt_bins = [(0, 3), (3, 6), (6, 11), (11, 21), (21, 10**9)]
    gt_rows = []
    for lo, hi in gt_bins:
        m = (gts >= lo) & (gts < hi)
        if m.sum() > 0:
            label = f"[{lo},{hi})" if hi < 10**8 else f"[{lo},inf)"
            gt_rows.append({
                "range": label,
                "n": int(m.sum()),
                "mae": float(abs_err[m].mean()),
                "rmse": float(np.sqrt((err[m] ** 2).mean())),
                "bias": float(err[m].mean()),
            })
    metrics["by_gt_cum"] = gt_rows

    return metrics


def print_metrics(metrics: dict) -> None:
    print("=" * 60)
    print("Word-count predictor evaluation")
    print("=" * 60)
    if metrics.get("n", 0) == 0:
        print("No scored samples (missing alignments in the batch?).")
        return

    print(f"samples:        {metrics['n']}")
    print(f"MAE:            {metrics['mae']:.3f}")
    print(f"RMSE:           {metrics['rmse']:.3f}")
    print(f"bias (pred-gt): {metrics['bias']:+.3f}")
    print(f"R^2:            {metrics['r2']:.3f}")
    print(f"hit |err|<0.5:  {metrics['hit_<0.5']:.1%}")
    print(f"hit |err|<1.0:  {metrics['hit_<1.0']:.1%}")
    print(f"hit |err|<2.0:  {metrics['hit_<2.0']:.1%}")

    print("\nBy normalized mask_end position:")
    print(f"  {'range':<12} {'n':>6} {'mae':>7} {'rmse':>7} {'bias':>7}")
    for r in metrics["by_norm_pos"]:
        print(f"  {r['range']:<12} {r['n']:>6} {r['mae']:>7.3f} {r['rmse']:>7.3f} {r['bias']:>+7.3f}")

    print("\nBy GT cumulative word count:")
    print(f"  {'range':<12} {'n':>6} {'mae':>7} {'rmse':>7} {'bias':>7}")
    for r in metrics["by_gt_cum"]:
        print(f"  {r['range']:<12} {r['n']:>6} {r['mae']:>7.3f} {r['rmse']:>7.3f} {r['bias']:>+7.3f}")


def _pick_plot_samples(pool: List[dict], num_plots: int) -> List[dict]:
    """Pick `num_plots` samples from the reservoir at evenly-spaced abs_err
    quantiles (min, …, max). Falls back to all if pool is smaller."""
    if not pool or num_plots <= 0:
        return []
    ordered = sorted(pool, key=lambda r: r["abs_err"])
    n = len(ordered)
    if num_plots >= n:
        return ordered
    idxs = [int(round(q * (n - 1))) for q in np.linspace(0, 1, num_plots)]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in idxs:
        if k not in seen:
            out.append(ordered[k])
            seen.add(k)
    return out


def _plot_sample(sample: dict, save_path: Path, feat_scale: float,
                 sampling_rate: int = 24000, hop: int = 256) -> None:
    """Plot mel + mask window + word boundaries + pred/gt for one sample."""
    feat = sample["feat_np"]                # (T, C)
    feat_len = sample["feat_len"]
    mask_start = sample["mask_start"]
    mask_end = sample["mask_end"]
    words = sample["words"]
    pred = sample["pred"]
    gt = sample["gt"]
    fps = sampling_rate / hop

    feat = feat[:feat_len]
    C = feat.shape[1] if feat.ndim == 2 else 0

    fig, ax = plt.subplots(figsize=(14, 4.2))
    if feat_len > 0 and C > 0:
        ax.imshow(
            feat.T / max(feat_scale, 1e-8),
            aspect="auto", origin="lower", cmap="magma",
        )
    ax.set_xlim(-0.5, max(feat_len - 0.5, 0.5))
    ax.set_ylim(-0.5, max(C - 0.5, 0.5))

    # Shade the prediction window [mask_start, mask_end).
    ax.axvspan(mask_start - 0.5, mask_end - 0.5,
               color="red", alpha=0.15, label="pred window")
    # Shade the masked-out tail [mask_end, feat_len) — future frames zeroed
    # for the wc head; shown here just so the user sees the visible region.
    if mask_end < feat_len:
        ax.axvspan(mask_end - 0.5, feat_len - 0.5,
                   color="gray", alpha=0.35, label="masked (future)")
    # Boundary lines.
    ax.axvline(mask_start - 0.5, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(mask_end - 0.5, color="red", linestyle="--", linewidth=1.2)

    # Prompt/target boundary (concat-infer mode only).
    prompt_len = sample.get("prompt_len")
    if prompt_len is not None and 0 < prompt_len < feat_len:
        ax.axvline(prompt_len - 0.5, color="deepskyblue",
                   linestyle="-", linewidth=1.5, label="prompt/target")
        ax.text(prompt_len - 0.5, max(C - 6, 1),
                "prompt | target", fontsize=8, color="deepskyblue",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.12", fc="black",
                          ec="deepskyblue", alpha=0.6))

    # Word boundaries and labels.
    y_label = max(C - 3, 1)
    counted = 0
    for w in words:
        f_start = w["start"] * fps
        f_end = (w["start"] + w["duration"]) * fps
        center = 0.5 * (f_start + f_end)
        center_frame = (w["start"] + w["duration"] / 2.0) * fps
        included = center_frame < mask_end
        if included:
            counted += 1
        color = "lime" if included else "lightgray"
        # Faint vertical line at word start for every word.
        ax.axvline(f_start, color=color, linestyle=":",
                   linewidth=0.6, alpha=0.55)
        if not w["symbol"]:
            continue
        ax.text(
            center, y_label, w["symbol"],
            fontsize=7.5, color=color, ha="center", va="top", rotation=90,
            clip_on=True,
            bbox=dict(boxstyle="round,pad=0.12", fc="black",
                      ec="none", alpha=0.55),
        )
    # Tail of last word.
    if words:
        last_end = (words[-1]["start"] + words[-1]["duration"]) * fps
        ax.axvline(last_end, color="white", linestyle=":",
                   linewidth=0.4, alpha=0.35)

    err = pred - gt
    ax.set_title(
        f"pred_cum={pred:.2f}  |  gt_cum={int(gt)}  |  err={err:+.2f}  "
        f"|  counted_in_window={counted}  "
        f"|  mask=[{mask_start},{mask_end})  |  feat_len={feat_len}"
    )
    ax.set_xlabel(f"frame  (fps={fps:.2f})")
    ax.set_ylabel("mel channel")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.75)

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def save_plots(pool: List[dict], params: AttributeDict) -> List[str]:
    picks = _pick_plot_samples(pool, params.num_plots)
    if not picks:
        print("\nNo samples cached for plotting.")
        return []

    if params.plot_dir:
        plot_dir = Path(params.plot_dir)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = Path("wc_eval_plots") / f"{ts}_{params.mode}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for rank, s in enumerate(picks):
        fname = (
            f"plot_{rank:02d}_q{rank+1}of{len(picks)}"
            f"_b{s['batch']}_s{s['sample_in_batch']}"
            f"_err{s['abs_err']:.2f}.png"
        )
        save_path = plot_dir / fname
        _plot_sample(s, save_path, feat_scale=params.feat_scale)
        paths.append(str(save_path))

    print(f"\nWrote {len(paths)} plots to {plot_dir}:")
    for p in paths:
        print(f"  {p}")
    return paths


def dump_csv(records: List[dict], path: str) -> None:
    fields = ["batch", "sample_in_batch", "feat_len", "mask_end",
              "norm_pos", "gt", "pred", "signed_err", "abs_err"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(r)
    print(f"\nWrote per-sample CSV: {path}")


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    params = AttributeDict()
    params.update(vars(args))
    params.model_dir = Path(params.model_dir)

    records, metrics, plot_pool = run_eval(params)
    print_metrics(metrics)

    if params.out_csv:
        dump_csv(records, params.out_csv)

    if params.num_plots > 0:
        save_plots(plot_pool, params)


if __name__ == "__main__":
    main()
