"""Word-pointer predictor for streaming-window selection.

Given a fixed 150-frame mel chunk and a candidate word window
(target slice + ``pad_left`` extra preceding words + ``pad_right`` extra
trailing words), predicts the joint label
``pad_left * (max_pad + 1) + pad_right`` so the streaming loop can recover
the start/end word offsets.

This module is a designated replacement for the CTC-aligner pointer head
(``zipvoice/models/ctc_aligner.py``).
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from zipvoice.models.modules.zipformer import TTSZipformer


def _sinusoidal_pos_encoding(length: int, dim: int, device, dtype) -> torch.Tensor:
    """Return ``(length, dim)`` sinusoidal positional encoding."""
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype)


class _CrossAttnBlock(nn.Module):
    """A pre-norm cross-attention + FFN block.

    Q comes from ``x``; K, V come from ``ctx`` (text features here).
    """

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        ctx_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_kv(ctx)
        attn_out, _ = self.attn(
            q, kv, kv, key_padding_mask=ctx_padding_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x


class WordPointer(nn.Module):
    """Mel encoder + text encoder + cross-attention + joint classifier.

    The classifier emits ``num_classes = (max_pad + 1) ** 2`` logits over
    ``(pad_left, pad_right) ∈ [0, max_pad]^2``. Decode via
    ``label = pad_left * (max_pad + 1) + pad_right``.
    """

    def __init__(
        self,
        vocab_size: int,
        max_pad: int = 4,
        mel_in_dim: int = 100,
        dim: int = 128,
        mel_encoder_layers: int = 2,
        text_encoder_layers: int = 2,
        cross_attn_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_text_len: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = vocab_size
        self.max_pad = max_pad
        self.num_classes = (max_pad + 1) ** 2
        self.dim = dim

        self.mel_encoder = TTSZipformer(
            in_dim=mel_in_dim,
            out_dim=dim,
            downsampling_factor=1,
            num_encoder_layers=mel_encoder_layers,
            cnn_module_kernel=15,
            encoder_dim=dim,
            feedforward_dim=feedforward_dim,
            num_heads=num_heads,
            query_head_dim=24,
            pos_head_dim=4,
            value_head_dim=12,
            pos_dim=48,
            use_time_embed=False,
        )

        self.text_embed = nn.Embedding(vocab_size + 1, dim, padding_idx=self.pad_id)
        self.register_buffer(
            "text_pos",
            _sinusoidal_pos_encoding(max_text_len, dim, torch.device("cpu"), torch.float32),
            persistent=False,
        )
        text_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=text_encoder_layers)

        self.cross_blocks = nn.ModuleList(
            [
                _CrossAttnBlock(
                    dim=dim, num_heads=num_heads, ff_dim=feedforward_dim, dropout=dropout
                )
                for _ in range(cross_attn_layers)
            ]
        )

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, self.num_classes)

    def encode_text(
        self, tokens: torch.Tensor, text_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """``tokens``: (B, L) long; ``text_padding_mask``: (B, L) bool, True at PAD."""
        B, L = tokens.shape
        x = self.text_embed(tokens)
        if L > self.text_pos.size(0):
            pe = _sinusoidal_pos_encoding(L, self.dim, x.device, x.dtype)
        else:
            pe = self.text_pos[:L].to(dtype=x.dtype, device=x.device)
        x = x + pe.unsqueeze(0)
        x = self.text_encoder(x, src_key_padding_mask=text_padding_mask)
        return x

    def encode_mel(
        self, mel: torch.Tensor, mel_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.mel_encoder(x=mel, t=None, padding_mask=mel_padding_mask)

    def forward(
        self,
        mel: torch.Tensor,
        mel_lens: torch.Tensor,
        tokens: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            mel: ``(B, T_mel, mel_in_dim)`` — already × feat_scale.
            mel_lens: ``(B,)`` int.
            tokens: ``(B, L_text)`` long, padded with ``self.pad_id``.
            token_lens: ``(B,)`` int.

        Returns:
            logits: ``(B, num_classes)``.
        """
        B, T_mel, _ = mel.shape
        L_text = tokens.size(1)
        device = mel.device

        mel_arange = torch.arange(T_mel, device=device).unsqueeze(0)
        mel_padding_mask = mel_arange >= mel_lens.unsqueeze(1)
        text_arange = torch.arange(L_text, device=device).unsqueeze(0)
        text_padding_mask = text_arange >= token_lens.unsqueeze(1)

        mel_h = self.encode_mel(mel, mel_padding_mask)
        text_h = self.encode_text(tokens, text_padding_mask)

        x = mel_h
        for block in self.cross_blocks:
            x = block(x, text_h, ctx_padding_mask=text_padding_mask)

        x = self.norm(x)
        # Mean-pool over valid mel frames.
        mask = (~mel_padding_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        return self.classifier(pooled)

    @staticmethod
    def decode_label(label: int, max_pad: int = 4):
        n = max_pad + 1
        return label // n, label % n
