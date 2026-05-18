#!/usr/bin/env python3
"""Sanity-probe a trained WordPointer ckpt on a handful of dev samples.

Loads the ckpt, iterates a few dev batches, runs ``build_pointer_batch``
the same way the trainer does, and prints per-sample (pred_l, pred_r)
vs (gt_l, gt_r) together with the text window so a human can eyeball
whether the head's predictions track the chunk's actual word coverage.
"""

import argparse
import random
from pathlib import Path
from typing import List

import torch
from lhotse.utils import fix_random_seed

from zipvoice.bin.train_word_pointer import (
    build_pointer_batch,
    remove_short_and_long_utt,
    tokenize_text,
)
from zipvoice.dataset.datamodule import TtsDataModule
from zipvoice.models.word_pointer import WordPointer
from zipvoice.tokenizer.tokenizer import LibriTTSTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--chunk-frames", type=int, default=150)
    TtsDataModule.add_arguments(parser)
    args = parser.parse_args()

    fix_random_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    p = ckpt["params"]
    max_pad = int(ckpt["max_pad"])
    model = WordPointer(
        vocab_size=int(ckpt["vocab_size"]),
        max_pad=max_pad,
        mel_in_dim=100,
        dim=int(p["dim"]),
        mel_encoder_layers=int(p["mel_encoder_layers"]),
        text_encoder_layers=int(p["text_encoder_layers"]),
        cross_attn_layers=int(p["cross_attn_layers"]),
        num_heads=int(p["num_heads"]),
        feedforward_dim=int(p["feedforward_dim"]),
        dropout=0.0,
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded WordPointer: {sum(x.numel() for x in model.parameters())/1e6:.2f}M params")
    print(f"  vocab_size={ckpt['vocab_size']} max_pad={max_pad} chunk_frames={ckpt['chunk_frames']}")

    tokenizer = LibriTTSTokenizer(token_file=str(args.token_file))
    id2tok = {v: k for k, v in tokenizer.token2id.items()}

    dm = TtsDataModule(args)
    dev_cuts = dm.dev_libritts_cuts()
    dev_cuts = dev_cuts.filter(lambda c: remove_short_and_long_utt(c, 2.0, 20.0))
    dev_cuts = dev_cuts.map(lambda c: tokenize_text(c, tokenizer=tokenizer))
    dev_dl = dm.dev_dataloaders(dev_cuts)

    rng = random.Random(args.seed)
    n_done = 0
    correct = 0
    for batch in dev_dl:
        pb = build_pointer_batch(
            batch, tokenizer, max_pad, args.chunk_frames, rng, model.pad_id
        )
        if pb is None:
            continue
        mel = (pb["mel"] * args.feat_scale).to(device)
        mel_lens = pb["mel_lens"].to(device)
        tokens = pb["tokens"].to(device)
        token_lens = pb["token_lens"].to(device)
        labels = pb["labels"].to(device)

        with torch.inference_mode():
            logits = model(mel, mel_lens, tokens, token_lens)
        preds = logits.argmax(dim=-1)
        n_pad = max_pad + 1

        for i in range(labels.size(0)):
            if n_done >= args.num_samples:
                break
            gt = int(labels[i].item())
            pr = int(preds[i].item())
            gl, gr = gt // n_pad, gt % n_pad
            pl, prr = pr // n_pad, pr % n_pad
            tl = int(token_lens[i].item())
            ids: List[int] = tokens[i, :tl].cpu().tolist()
            txt = "".join(id2tok.get(t, "?") for t in ids)
            mark = "✓" if gt == pr else "✗"
            print(
                f"[{n_done:02d}] {mark} gt=({gl},{gr})  pred=({pl},{prr})  "
                f"text={txt!r}"
            )
            correct += int(gt == pr)
            n_done += 1
        if n_done >= args.num_samples:
            break

    print(f"\nExact-match {correct}/{n_done} = {correct/max(n_done,1):.3f}")


if __name__ == "__main__":
    main()
