# Copyright    2024    Xiaomi Corp.        (authors:  Wei Kang
#                                                     Han Zhu)
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

from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import torch

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import matplotlib.pyplot as plt
import datetime as dt
import random

from zipvoice.models.modules.solver_stream import EulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.utils.common import (
    condition_time_mask,
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)

class ConcatSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        """
        Args:
            dim: 输入特征的维度 (D)，生成的 PE 也将是这个维度。
            max_len: 预计算的最大长度，不够会自动扩展，通常设大一点即可。
        """
        super().__init__()
        self.dim = dim
        
        # 1. 预计算 PE 矩阵
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算频率项 (div_term)
        # 公式: exp(arange(0, d, 2) * -(log(10000.0) / d))
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # 赋值 Sin 和 Cos
        # 注意处理奇数维度的情况
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 Batch 维度: [1, Max_Len, Dim]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer (不会作为参数更新，但会随模型保存)
        self.register_buffer('pe', pe)

    def forward(self, x, lens):
        """
        Args:
            x: [Batch, Time, Dim]
        Returns:
            out: [Batch, Time, Dim * 2] (特征 + 位置编码)
        """
        B, T, D = x.shape
        
        # 检查预计算的长度是否够用，不够则动态重新计算 (增强鲁棒性)
        if T > self.pe.size(1):
            raise ValueError(f"Input length {T} exceeds precomputed max_len {self.pe.size(1)}.")
            # self._reset_pe(T, x.device)/\
        current_pe = self.pe[:, :T, :]
        seq_range = torch.arange(T, device=x.device).unsqueeze(0)
        lens_expanded = lens.unsqueeze(1)
        mask = seq_range < lens_expanded
        mask_expanded = mask.unsqueeze(-1).float()
        masked_pe = current_pe * mask_expanded
        # out = torch.cat([x, masked_pe], dim=-1)
        
        return x, masked_pe

class ZipVoice(nn.Module):
    """The ZipVoice model."""

    def __init__(
        self,
        fm_decoder_downsampling_factor: List[int] = [1, 2, 4, 2, 1],
        fm_decoder_num_layers: List[int] = [2, 2, 4, 4, 4],
        fm_decoder_cnn_module_kernel: List[int] = [31, 15, 7, 15, 31],
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        vocab_size: int = 26,
        pad_id: int = 0,
    ):
        """
        Initialize the model with specified configuration parameters.

        Args:
            fm_decoder_downsampling_factor: List of downsampling factors for each layer
                in the flow-matching decoder.
            fm_decoder_num_layers: List of the number of layers for each block in the
                flow-matching decoder.
            fm_decoder_cnn_module_kernel: List of kernel sizes for CNN modules in the
                flow-matching decoder.
            fm_decoder_feedforward_dim: Dimension of the feedforward network in the
                flow-matching decoder.
            fm_decoder_num_heads: Number of attention heads in the flow-matching
                decoder.
            fm_decoder_dim: Hidden dimension of the flow-matching decoder.
            text_encoder_num_layers: Number of layers in the text encoder.
            text_encoder_feedforward_dim: Dimension of the feedforward network in the
                text encoder.
            text_encoder_cnn_module_kernel: Kernel size for the CNN module in the
                text encoder.
            text_encoder_num_heads: Number of attention heads in the text encoder.
            text_encoder_dim: Hidden dimension of the text encoder.
            time_embed_dim: Dimension of the time embedding.
            text_embed_dim: Dimension of the text embedding.
            query_head_dim: Dimension of the query attention head.
            value_head_dim: Dimension of the value attention head.
            pos_head_dim: Dimension of the position attention head.
            pos_dim: Dimension of the positional encoding.
            feat_dim: Dimension of the acoustic features.
            vocab_size: Size of the vocabulary.
            pad_id: ID used for padding tokens.
        """
        super().__init__()

        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
            # in_dim=feat_dim * 4,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
        )

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        # self.posemb = ConcatSinusoidalPositionalEmbedding(dim=feat_dim)
        self.embed = nn.Embedding(vocab_size, text_embed_dim)
        self.solver = EulerSolver(self, func_name="forward_fm_decoder")

    def forward_fm_decoder(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        # posemb_condition: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        guidance_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute velocity.
        Args:
            t:  A tensor of shape (N, 1, 1) or a tensor of a float,
                in the range of (0, 1).
            xt: the input of the current timestep, including condition
                embeddings and noisy acoustic features.
            text_condition: the text condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            speech_condition: the speech condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding, True means masked
                position, with the shape (N, T).
            guidance_scale: The guidance scale in classifier-free guidance,
                which is a tensor of shape (N, 1, 1) or a tensor of a float.

        Returns:
            predicted velocity, with the shape (batch, seq_len, emb_dim).
        """

        # xt = torch.cat([xt, text_condition, speech_condition, posemb_condition], dim=2)
        xt = torch.cat([xt, text_condition, speech_condition], dim=2)

        assert t.dim() in (0, 3)
        # Handle t with the shape (N, 1, 1):
        # squeeze the last dimension if it's size is 1.
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        # Handle t with a single value: expand to the size of batch size.
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(xt.shape[0])

            vt = self.fm_decoder(
                x=xt, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale
            )
        else:
            vt = self.fm_decoder(x=xt, t=t, padding_mask=padding_mask)
        return vt

    def forward_text_embed(
        self,
        tokens: List[List[int]],
    ):
        """
        Get the text embeddings.
        Args:
            tokens: a list of list of token ids.
        Returns:
            embed: the text embeddings, shape (batch, seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B, S)
        embed = self.embed(tokens_padded)  # (B, S, C)
        tokens_lens = torch.tensor(
            [len(token) for token in tokens], dtype=torch.int64, device=device
        )
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B, S)

        embed = self.text_encoder(
            x=embed, t=None, padding_mask=tokens_padding_mask
        )  # (B, S, C)
        return embed, tokens_lens

    def forward_text_condition(
        self,
        embed: torch.Tensor,
        tokens_lens: torch.Tensor,
        features_lens: torch.Tensor,
    ):
        """
        Get the text condition with the same length of the acoustic feature.
        Args:
            embed: the text embeddings, shape (batch, token_seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
            features_lens: the length of each acoustic feature sequence,
                shape (batch,).
        Returns:
            text_condition: the text condition, shape
                (batch, feature_seq_len, emb_dim).
            padding_mask: the padding mask of text condition, shape
                (batch, feature_seq_len).
        """

        num_frames = int(features_lens.max())

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)

        tokens_durations = prepare_avg_tokens_durations(features_lens, tokens_lens)

        tokens_index = get_tokens_index(tokens_durations, num_frames).to(
            embed.device
        )  # (B, T)

        text_condition = torch.gather(
            embed,
            dim=1,
            index=tokens_index.unsqueeze(-1).expand(
                embed.size(0), num_frames, embed.size(-1)
            ),
        )  # (B, T, F)
        return text_condition, padding_mask

    def forward_text_train(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
    ):
        """
        Process text for training, given text tokens and real feature lengths.
        """
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens
        )
        return (
            text_condition,
            padding_mask,
        )

    def forward_text_inference_gt_duration(
        self,
        tokens: List[List[int]],
        features_lens: torch.Tensor,
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
    ):
        """
        Process text for inference, given text tokens, real feature lengths and prompts.
        """
        tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]
        features_lens = prompt_features_lens + features_lens
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(
            embed, tokens_lens, features_lens
        )
        return text_condition, padding_mask

    def forward_text_inference_ratio_duration(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features_lens: torch.Tensor,
        speed: float,
    ):
        """
        Process text for inference, given text tokens and prompts,
        feature lengths are predicted with the ratio of token numbers.
        """
        device = (
            self.device if isinstance(self, DDP) else next(self.parameters()).device
        )

        cat_tokens = [
            prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens)
        ]

        prompt_tokens_lens = torch.tensor(
            [len(token) for token in prompt_tokens],
            dtype=torch.int64,
            device=device,
        )

        tokens_lens = torch.tensor(
            [len(token) for token in tokens],
            dtype=torch.int64,
            device=device,
        )

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)

        features_lens = prompt_features_lens + torch.ceil(
            (prompt_features_lens / prompt_tokens_lens * tokens_lens / speed)
        ).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(
            cat_embed, cat_tokens_lens, features_lens
        )
        return text_condition, padding_mask

    def condition_time_mask(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        alignments: Optional[List[Dict]] = None, # 类型提示变更为 List[Dict]
        mask_percent=None,
        max_len: int = 0,
        id2textDict: Optional[dict] = None,
    ):
        """
        Apply Time masking.
        Returns:
            features, features_lens, mask, tokens
        """

        if alignments is not None:
            print("aliments is not None, using alignment-based masking and cropping.", flush=True)
            # === 新增逻辑：基于 Alignment (Words) 进行 Mask 和裁剪 ===
            mask_starts_list = []
            mask_ends_list = []
            new_tokens = []
            new_feat_lens = []

            for i, (tok, ali_dict, feat_len) in enumerate(zip(tokens, alignments, features_lens)):
                # ali_dict 结构: {'words': [AlignmentItem(symbol='word', start=0.0, duration=0.1), ...]}
                if ali_dict is None:
                    # print(f"Sample {i} has None alignment!", flush=True) # 调试用
                    # print(f"Processing sample {i}: tokens={tok}, alignment={ali_dict}, feature length={feat_len}", flush=True)
                    words = []
                else:
                    # print(f"Processing sample {i}", flush=True)
                    words = ali_dict.get('words', [])
                num_words = len(words)
                feat_len_int = int(feat_len.item())

                if num_words == 0:
                    # 没有可用的对齐信息，退回到原始随机 mask 逻辑
                    if mask_percent is None:
                        raise ValueError(
                            "mask_percent must be provided when falling back to random masking."
                        )
                    if feat_len_int <= 0:
                        mask_start = 0
                    else:
                        mask_ratio = random.uniform(*mask_percent)
                        mask_size = int(mask_ratio * feat_len_int)
                        mask_size = min(mask_size, feat_len_int)
                        if feat_len_int - mask_size <= 0:
                            mask_start = 0
                        else:
                            mask_start = random.randint(0, feat_len_int - mask_size)
                    mask_starts_list.append(mask_start)
                    mask_ends_list.append(feat_len_int)
                    new_feat_lens.append(feat_len_int)
                    new_tokens.append(tok)
                    continue

                # 1. 计算当前样本的帧率 (FPS)
                # 使用 (特征总帧数 / 对齐总时长) 来进行时间到帧的映射
                fps = 24000/256 # 24kHz 采样率，256 帧每秒 (假设每帧对应 16ms 的音频)

                # 2. 确定 Mask 和 裁剪 的位置
                # 随机决定 mask 结尾的 1~3 个词
                num_to_mask = random.randint(1, 3)
                # print(f"num_words: {num_words}, num_to_mask: {num_to_mask}", flush=True)

                if num_words <= num_to_mask:
                    # 词数太少，全部 mask，保留全部
                    cut_idx = num_words
                    start_mask_idx = 0
                else:
                    # 随机选择一个“结尾点” (cut_idx)，模拟流式输出到这里结束
                    # cut_idx 的范围从 num_to_mask 到 num_words
                    cut_idx = random.randint(num_to_mask, num_words)
                    start_mask_idx = cut_idx - num_to_mask
                    print(f"cut_idx: {cut_idx}, start_mask_idx: {start_mask_idx}", flush=True)

                # 3. 获取时间戳并转为帧索引
                if num_words > 0:
                    # Mask 开始时间：start_mask_idx 对应单词的 start
                    t_start = words[start_mask_idx].start
                    # Mask 结束时间 (也是裁剪时间)：cut_idx-1 对应单词的 end
                    last_word = words[cut_idx - 1]
                    t_end = last_word.start + last_word.duration
                    
                    # 转帧 (向上取整或四舍五入均可，这里用 int)
                    f_start = int(t_start * fps)
                    f_end = int(t_end * fps)
                    
                    # 边界保护
                    f_end = min(f_end, feat_len_int)
                    f_start = min(f_start, f_end)
                    # print(f"t_start: {t_start:.3f}s, t_end: {t_end:.3f}s, f_start: {f_start}, f_end: {f_end}", flush=True)
                else:
                    # 异常情况处理
                    f_start = 0
                    f_end = feat_len_int
                    cut_idx = len(tok)

                mask_starts_list.append(f_start)
                mask_ends_list.append(f_end) # f_end 即为裁剪后的新长度
                new_feat_lens.append(f_end)
                
                # 裁剪 Token
                

                assert id2textDict is not None, "id2textDict cannot be None when using alignment-based masking."
                
                full_text = "".join([id2textDict[t] for t in tok])
                full_text_upper = full_text.upper()
                # print(f"full_text: {full_text} id2textDict[tok[0]]: {id2textDict[tok[0]]}", flush=True)

                cursor = 0
                cut_char_pos = 0


                # print(f"len(words):{num_words} cut_idx:{cut_idx}-", flush=True)
                for j in range(cut_idx):
                    if j >= len(words):
                        # 对齐失败保护
                        print(f"Warning: cut_idx {cut_idx} exceeds number of words {num_words}. Using full text length for cutting.", flush=True)   
                        print(f"tok: {tok}", flush=True)

                    word = words[j].symbol.upper()
                    # print(f"word:{word}", flush=True)
                    start = full_text_upper.find(word, cursor)
                    if start == -1:
                        # 对齐失败保护：尽量保留已匹配前缀，避免总是退回整句
                        if cursor > 0:
                            cut_char_pos = cursor
                        else:
                            est_cut = int(round(len(tok) * cut_idx / max(num_words, 1)))
                            cut_char_pos = max(1, min(len(tok), est_cut))
                        break
                    
                    end = start + len(word)
                    cursor = end
                    # print(f"{j} step : cursor:{cursor}, word:{word}, start:{start}, end:{end}", flush=True)
                    cut_char_pos = end
                new_tokens.append(tok[:cut_char_pos])
                # print(f"cut_idx:{cut_idx} -> cut_char_pos:{cut_char_pos}, original tokens: {tok}, new tokens: {tok[:cut_char_pos]}, len(tokens):{len(tok)}", flush=True)

            # 更新 Tensor 数据
            tokens = new_tokens
            mask_starts = torch.tensor(mask_starts_list, device=features.device, dtype=torch.long)
            mask_ends = torch.tensor(mask_ends_list, device=features.device, dtype=torch.long)
            features_lens = torch.tensor(new_feat_lens, device=features.device, dtype=torch.long)

        else:
            # === 原有逻辑：随机比例 Mask ===
            mask_size = (
                torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
                * features_lens
            ).to(torch.int64)
            mask_starts = (
                torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
            ).to(torch.int64)
            mask_ends = features_lens

        # === 公共处理：Mask 矩阵生成与应用 ===
        # 1. 生成 Pad Mask (用于处理变长序列)
        feature_mask = make_pad_mask(features_lens)
        
        # 2. 更新 max_len (因为做了裁剪，最大长度可能变小)
        max_len = features_lens.max().item()
        
        # 3. 物理裁剪 Features Tensor
        features = features[:, :max_len, :]
        feature_mask = feature_mask[:, :max_len]
        
        # 4. 将 Padding 部分置 0 (Pad Masking)
        features = torch.where(
            feature_mask.unsqueeze(-1),
            torch.zeros_like(features),
            features,
        )
        
        # 5. 生成 Mask 区域的 Boolean 矩阵
        seq_range = torch.arange(0, max_len, device=features_lens.device)
        # mask 为 True 的地方表示被 Mask 掉了（需要预测的部分）
        mask = (seq_range[None, :] >= mask_starts[:, None]) & (
            seq_range[None, :] < mask_ends[:, None]
        )
        
        # 6. 将 Mask 区域特征置 0 (In-painting Masking)
        # features = torch.where(
        #     mask.unsqueeze(-1),
        #     torch.zeros_like(features),
        #     features,
        # )

        return tokens, features, features_lens, mask


    def visualize_sample(
        self,
        orig_features,
        new_features,
        orig_tokens,
        new_tokens,
        mask,
        sample_idx=0,
        id2textDict=None,
        alignments: Optional[List[Dict]] = None,
        orig_features_lens: Optional[torch.Tensor] = None,
        new_features_lens: Optional[torch.Tensor] = None,
    ):
        """
        只画单个样本
        """

        def _get_word_fields(word_obj):
            if hasattr(word_obj, "symbol"):
                symbol = word_obj.symbol
                start = word_obj.start
                duration = word_obj.duration
            else:
                symbol = word_obj.get("symbol", "")
                start = word_obj.get("start", 0.0)
                duration = word_obj.get("duration", 0.0)
            return str(symbol), float(start), float(duration)

        def _draw_alignment_tokens(ax, alignment, valid_len, feat_bins):
            if alignment is None:
                return
            words = alignment.get("words", []) if isinstance(alignment, dict) else []
            if len(words) == 0 or valid_len <= 0:
                return

            fps = 24000 / 256
            y_text = max(feat_bins - 2, 1)
            for word in words:
                symbol, start, duration = _get_word_fields(word)
                f_start = int(start * fps)
                f_end = int((start + duration) * fps)

                if f_end <= 0 or f_start >= valid_len:
                    continue

                f_start = max(0, f_start)
                f_end = min(valid_len, max(f_end, f_start + 1))
                center = 0.5 * (f_start + f_end)

                ax.axvspan(f_start - 0.5, f_end - 0.5, color="cyan", alpha=0.08)
                ax.text(
                    center,
                    y_text,
                    symbol,
                    fontsize=8,
                    color="white",
                    ha="center",
                    va="top",
                    rotation=90,
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.35),
                )

        # 取单个样本
        orig_feat = orig_features[sample_idx].cpu().T  # 转为 [D, T] 方便imshow
        new_feat = new_features[sample_idx].cpu().T
        mask_sample = mask[sample_idx].cpu()
        orig_len = (
            int(orig_features_lens[sample_idx].item())
            if orig_features_lens is not None
            else orig_feat.shape[1]
        )
        new_len = (
            int(new_features_lens[sample_idx].item())
            if new_features_lens is not None
            else new_feat.shape[1]
        )
        alignment = alignments[sample_idx] if alignments is not None else None

        shared_xmax = max(orig_len, new_len)
        orig_for_scale = orig_feat[:, : max(orig_len, 1)]
        new_for_scale = new_feat[:, : max(new_len, 1)]
        global_min = min(orig_for_scale.min().item(), new_for_scale.min().item())
        global_max = max(orig_for_scale.max().item(), new_for_scale.max().item())

        if global_max <= global_min:
            global_max = global_min + 1e-6

        fig = plt.figure(figsize=(14, 10))

        # ===== 原始 features =====
        plt.subplot(3, 1, 1)
        plt.title("Original Features")
        plt.imshow(
            orig_feat,
            aspect="auto",
            origin="lower",
            vmin=global_min,
            vmax=global_max,
        )
        plt.xlim(-0.5, max(shared_xmax - 0.5, 0.5))
        if orig_len > 0:
            plt.axvline(orig_len - 0.5, color="yellow", linestyle="--", linewidth=1.2)
            plt.text(
                orig_len - 0.5,
                max(orig_feat.shape[0] - 2, 1),
                f"orig_len={orig_len}",
                color="yellow",
                fontsize=9,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45),
            )
        _draw_alignment_tokens(plt.gca(), alignment, orig_len, orig_feat.shape[0])
        plt.colorbar()

        # ===== 修改后 features =====
        plt.subplot(3, 1, 2)
        plt.title("Masked & Cropped Features")
        plt.imshow(
            new_feat,
            aspect="auto",
            origin="lower",
            vmin=global_min,
            vmax=global_max,
        )
        plt.xlim(-0.5, max(shared_xmax - 0.5, 0.5))
        if new_len > 0:
            plt.axvline(new_len - 0.5, color="yellow", linestyle="--", linewidth=1.2)
            plt.text(
                new_len - 0.5,
                max(new_feat.shape[0] - 2, 1),
                f"new_len={new_len}",
                color="yellow",
                fontsize=9,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45),
            )
        _draw_alignment_tokens(plt.gca(), alignment, new_len, new_feat.shape[0])
        plt.colorbar()

        # 叠加 mask 区域（红色半透明）
        T = min(new_feat.shape[1], new_len)
        for t in range(T):
            if mask_sample[t]:
                plt.axvspan(t - 0.5, t + 0.5, color='red', alpha=0.2)

        # ===== tokens 对比 =====
        plt.subplot(3, 1, 3)
        plt.axis("off")

        orig_text = " ".join([id2textDict[t] for t in orig_tokens[sample_idx]]) if id2textDict is not None else " ".join(map(str, orig_tokens[sample_idx]))
        new_text = " ".join([id2textDict[t] for t in new_tokens[sample_idx]]) if id2textDict is not None else " ".join(map(str, new_tokens[sample_idx]))

        text = f"Original Tokens:\n{orig_text}\n\nModified Tokens:\n{new_text}"
        # print(f"Original Tokens:\n{orig_text}\n\nModified Tokens:\n{new_text}")
        plt.text(0.01, 0.5, text, fontsize=12, verticalalignment='center')

        plt.tight_layout()
        plt.savefig(f"cut_feature_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.close()
        print("-----------------------------------------------------------")

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        alignments: Optional[List[torch.Tensor]] = None,
        condition_drop_ratio: float = 0.0,
        id2textDict: Optional[dict] = None,
    ) -> torch.Tensor:
        """Forward pass of the model for training.
        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            condition_drop_ratio: the ratio of dropped text condition.
        Returns:
            fm_loss: the flow-matching loss.
        """

        orig_features = features.clone().detach()
        orig_tokens = [t.copy() for t in tokens]
        orig_features_lens = features_lens.clone().detach()

        tokens, gt_features, features_lens, speech_condition_mask = self.condition_time_mask(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            mask_percent=(0.1, 0.3),
            alignments=alignments,
            id2textDict=id2textDict,
        )

        # self.visualize_sample(
        #     orig_features=orig_features,
        #     new_features=features,
        #     orig_tokens=orig_tokens,
        #     new_tokens=tokens,
        #     mask=speech_condition_mask,
        #     sample_idx=0,
        #     id2textDict=id2textDict,
        #     alignments=alignments,
        #     orig_features_lens=orig_features_lens,
        #     new_features_lens=features_lens,
        # )

        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, gt_features)
        # features, posemb_condition = self.posemb(features, features_lens)

        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        if condition_drop_ratio > 0.0:
            drop_mask = (
                torch.rand(text_condition.size(0), 1, 1).to(text_condition.device)
                > condition_drop_ratio
            )
            text_condition = text_condition * drop_mask

        noise = torch.randn_like(gt_features)

        xt = gt_features * t + noise * (1 - t)
        ut = gt_features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            # posemb_condition=posemb_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        return fm_loss

    def sample(
        self,
        tokens: List[List[int]],
        prompt_tokens: List[List[int]],
        prompt_features: torch.Tensor,
        prompt_features_lens: torch.Tensor,
        features_lens: Optional[torch.Tensor] = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate acoustic features, given text tokens, prompts feature
            and prompt transcription's text tokens.
        Args:
            tokens: a list of list of text tokens.
            prompt_tokens: a list of list of prompt tokens.
            prompt_features: the prompt feature with the shape
                (batch_size, seq_len, feat_dim).
            prompt_features_lens: the length of each prompt feature,
                with the shape (batch_size,).
            features_lens: the length of the predicted eature, with the
                shape (batch_size,). It is used only when duration is "real".
            duration: "real" or "predict". If "real", the predicted
                feature length is given by features_lens.
            num_step: the number of steps to use in the ODE solver.
            guidance_scale: the guidance scale for classifier-free guidance.
        """

        assert duration in ["real", "predict"]

        if duration == "predict":
            (
                text_condition,
                padding_mask,
            ) = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
            )
        else:
            assert features_lens is not None
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )
        batch_size, num_frames, _ = text_condition.shape

        # print(f"padding_mask.sum(dim=1): {padding_mask.sum(dim=1)}")

        # _, posemb_condition = self.posemb(text_condition, num_frames - padding_mask.sum(dim=1))

        speech_condition = torch.nn.functional.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B, T, F)

        # False means speech condition positions.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = torch.randn(
            batch_size,
            num_frames,
            prompt_features.size(-1),
            device=text_condition.device,
        )

        # print(f"shape of x0: {x0.shape}, shape of text_condition: {text_condition.shape}, shape of speech_condition: {speech_condition.shape}, shape of posemb_condition: {posemb_condition.shape}, shape of padding_mask: {padding_mask.shape}")

        # print(f"x0[0, :5, 0]: {x0[0, :5, 0]}, text_condition[0, :5, 0]: {text_condition[0, :5, 0]}, speech_condition[0, :5, 0]: {speech_condition[0, :5, 0]}, posemb_condition[0, :5, 0]: {posemb_condition[0, :5, 0]}")

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            # posemb_condition=posemb_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        # print(f"x1[0, :5, 0]: {x1[0, :5, 0]}")
        # rms = torch.sqrt(torch.mean(torch.exp(x1) ** 2, dim=-1))  # shape [T]
        # rms_np = rms.detach().cpu().numpy()
        # rmsx0 = torch.sqrt(torch.mean(torch.exp(x0) ** 2, dim=-1))  # shape [T]
        # rmsx0_np = rmsx0.detach().cpu().numpy()
        # rmsspeech_condition = torch.sqrt(torch.mean(torch.exp(speech_condition) ** 2, dim=-1))  # shape [T]
        # rmsspeech_condition_np = rmsspeech_condition.detach().cpu().numpy()
        # rmsprompt_features = torch.sqrt(torch.mean(torch.exp(prompt_features) ** 2, dim=-1))  # shape [T]
        # rmsprompt_features_np = rmsprompt_features.detach().cpu().numpy()
        # rmsposemb_condition = torch.sqrt(torch.mean(posemb_condition ** 2, dim=-1))  # shape [T]
        # rmsposemb_condition_np = rmsposemb_condition.detach().cpu().numpy()
        # rmstext_condition = torch.sqrt(torch.mean(text_condition ** 2, dim=-1))  # shape [T]
        # rmstext_condition_np = rmstext_condition.detach().cpu().numpy()

        # # 绘制 RMS 随时间变化图
        # plt.figure()
        # plt.plot(rms_np[0], label="x1", alpha=0.7)
        # plt.plot(rmsx0_np[0], label="x0", alpha=0.7)
        # plt.plot(rmsspeech_condition_np[0], label="speech_condition", alpha=0.7)
        # plt.plot(rmsprompt_features_np[0], label="prompt_features", alpha=0.7)
        # plt.plot(rmsposemb_condition_np[0], label="posemb_condition", alpha=0.7)
        # plt.plot(rmstext_condition_np[0], label="text_condition", alpha=0.7)
        # plt.legend()
        # plt.xlabel("Time step")
        # plt.ylabel("RMS")
        # plt.title("RMS over Time for x1[0]")
        # ts = dt.datetime.now().strftime("%H%M%S_%f")[:-3]  # 精确到毫秒
        # plt.savefig(f"rms_over_time_x1_0_{ts}.png")
        # plt.close()
        # print("-----------------------------------------------------------")
        x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
        x1_prompt = torch.zeros(
            x1.size(0), prompt_features_lens.max(), x1.size(2), device=x1.device
        )
        x1_wo_prompt = torch.zeros(
            x1.size(0), x1_wo_prompt_lens.max(), x1.size(2), device=x1.device
        )
        for i in range(x1.size(0)):
            x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                i,
                prompt_features_lens[i] : prompt_features_lens[i]
                + x1_wo_prompt_lens[i],
            ]
            x1_prompt[i, : prompt_features_lens[i], :] = x1[
                i, : prompt_features_lens[i]
            ]
        # print(f"x1_wo_prompt[0, :5, 0]: {x1_wo_prompt[0, :5, 0]}")
        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens

    def sample_intermediate(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        speech_condition_mask: torch.Tensor,
        t_start: float,
        t_end: float,
        num_step: int = 1,
        guidance_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Generate acoustic features in intermediate timesteps.
        Args:
            tokens: List of list of token ids.
            features: The acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: The length of each acoustic feature sequence,
                with the shape (batch,).
            noise: The initial noise, with the shape (batch, seq_len, feat_dim).
            speech_condition_mask: The mask for speech condition, True means
                non-condition positions, with the shape (batch, seq_len).
            t_start: The start timestep.
            t_end: The end timestep.
            num_step: The number of steps for sampling.
            guidance_scale: The scale for classifier-free guidance inference,
                with the shape (batch, 1, 1).
        """
        (text_condition, padding_mask,) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        x_t_end = self.solver.sample(
            x=noise,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_start=t_start,
            t_end=t_end,
        )
        x_t_end_lens = (~padding_mask).sum(-1)
        return x_t_end, x_t_end_lens
