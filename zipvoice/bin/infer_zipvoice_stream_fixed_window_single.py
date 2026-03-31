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
Minimal fixed-window streaming TTS inference script.

This script supports only one input:
- prompt wav
- prompt text
- target text

Example:
python3 -m zipvoice.bin.infer_zipvoice_stream_fixed_window\ copy \
  --model-name zipvoice \
  --prompt-wav prompt.wav \
  --prompt-text "I am a prompt." \
  --text "This is target text." \
  --res-wav-path result.wav
"""

import argparse
import datetime as dt
import json
import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from lhotse.utils import fix_random_seed
from vocos import Vocos

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.models.zipvoice_stream_fixedwindow import ZipVoice
from zipvoice.tokenizer.tokenizer_stream import (
    LibriTTSTokenizer,
)
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import add_punctuation, load_prompt_wav, remove_silence, rms_norm
from zipvoice.utils.tensorrt import load_trt

HUGGINGFACE_REPO = "k2-fsa/ZipVoice"
MODEL_DIR = {
    "zipvoice": "zipvoice",
    "zipvoice_distill": "zipvoice_distill",
}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="zipvoice",
        choices=["zipvoice", "zipvoice_distill"],
        help="Model name",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="exp/zipvoice_libritts_0326_1653_stream_alignmask_fixedwindow",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="epoch-170.pt",
        help="Checkpoint name in model-dir",
    )
    parser.add_argument(
        "--vocoder-path",
        type=str,
        default=None,
        help="Local vocoder dir (optional)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="libritts",
        choices=["emilia", "libritts", "espeak", "simple"],
        help="Tokenizer type",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en-us",
        help="Language id for espeak tokenizer",
    )
    parser.add_argument("--prompt-wav", type=str, default="download/librispeech_pc_testset/prompt_wavs/4992-23283-0015.wav")
    parser.add_argument("--prompt-text", type=str, default="Is she not afraid that I will thwart her inclinations?")
    parser.add_argument("--text", type=str, default="To ask any more questions of you, I believe, would be unfair.")
    parser.add_argument(
        "--res-wav-path",
        type=str,
        default="res/zipvoice_libritts_0326_1653_stream_alignmask_fixedwindow/result.wav",
        help="Output wav path",
    )
    parser.add_argument("--guidance-scale", type=float, default=3)
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--t-shift", type=float, default=0.7)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num-thread", type=int, default=1)
    parser.add_argument("--trt-engine-path", type=str, default=None)
    parser.add_argument(
        "--fixed-chunk-frames",
        type=int,
        default=30,
        help="Number of acoustic frames generated per streaming step",
    )
    return parser


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


def split_words(text: str):
    return re.findall(r"\b\w+\b", text)


def generate_streaming_single_sentence(
    save_path: str,
    prompt_text: str,
    prompt_wav: str,
    text: str,
    model: torch.nn.Module,
    vocoder: torch.nn.Module,
    tokenizer: LibriTTSTokenizer,
    feature_extractor: VocosFbank,
    device: torch.device,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    target_rms: float,
    feat_scale: float,
    sampling_rate: int,
    fixed_chunk_frames: int,
):
    prompt_wav_tensor = load_prompt_wav(prompt_wav, sampling_rate=sampling_rate)
    prompt_wav_tensor = remove_silence(
        prompt_wav_tensor, sampling_rate, only_edge=False, trail_sil=200
    )
    prompt_wav_tensor, prompt_rms = rms_norm(prompt_wav_tensor, target_rms)

    prompt_features = feature_extractor.extract(
        prompt_wav_tensor, sampling_rate=sampling_rate
    ).to(device)
    prompt_features = prompt_features.unsqueeze(0) * feat_scale

    prompt_text = add_punctuation(prompt_text)
    text = add_punctuation(text)

    target_words = split_words(text)
    if len(target_words) == 0:
        raise ValueError("Target text has no valid words")

    prompt_tokens_for_est = tokenizer.texts_to_token_ids([prompt_text])[0]
    target_tokens_for_est = tokenizer.texts_to_token_ids([text])[0]
    prompt_token_len = max(1, len(prompt_tokens_for_est))
    target_token_len = max(1, len(target_tokens_for_est))

    est_total_new_frames = int(
        torch.ceil(
            torch.tensor(
                [prompt_features.size(1) / prompt_token_len * target_token_len / speed],
                dtype=torch.float32,
            )
        ).item()
    )
    est_total_new_frames = max(1, est_total_new_frames)

    start_t = dt.datetime.now()
    model_seconds = 0.0
    vocoder_seconds = 0.0

    generated_new_frames = 0
    committed_word_pos = 0
    base_prompt_text = prompt_text
    output_wavs = []

    num_iter = max(1, int(np.ceil(est_total_new_frames / fixed_chunk_frames)))

    for i in range(num_iter):
        expected_after_this = min(
            est_total_new_frames,
            generated_new_frames + fixed_chunk_frames,
        )
        expected_word_pos = int(
            np.floor(len(target_words) * expected_after_this / est_total_new_frames)
        )
        expected_word_pos = min(
            len(target_words), max(committed_word_pos, expected_word_pos)
        )

        lookahead_end = min(len(target_words), expected_word_pos + 2)
        if lookahead_end <= committed_word_pos:
            lookahead_end = min(len(target_words), committed_word_pos + 2)

        text_chunk_words = target_words[committed_word_pos:lookahead_end]
        if len(text_chunk_words) == 0:
            break

        text_chunk = " ".join(text_chunk_words)
        logging.info(
            "Streaming step %d: chunk='%s' committed=%d expected=%d",
            i,
            text_chunk,
            committed_word_pos,
            expected_word_pos,
        )

        tokens = tokenizer.texts_to_token_ids([text_chunk])
        prompt_tokens = tokenizer.texts_to_token_ids([prompt_text])
        prompt_features_lens = torch.tensor([prompt_features.size(1)], device=prompt_features.device)

        model_t0 = dt.datetime.now()
        if i == 0:
            tokens = tokenizer.texts_to_token_ids(["To"])
            print(f"tokens: {tokens}")

        if i == 5:
            tokens = tokenizer.texts_to_token_ids(["of you step"])
            print(f"tokens: {tokens}")
            out_dir = Path(save_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Make tokens plottable even if tokenizer returns nested containers.
            ptoks = prompt_tokens[0] if len(prompt_tokens) > 0 else []
            ttoks = tokens[0] if len(tokens) > 0 else []
            if hasattr(ptoks, "tolist"):
                ptoks = ptoks.tolist()
            if hasattr(ttoks, "tolist"):
                ttoks = ttoks.tolist()
            print(f"ttoks: {ttoks}")
            ttoks = [tokenizer.id2token[ttoks[i]] for i in range(len(ttoks))]

            pf = prompt_features[0].detach().float().cpu().numpy()
            lens_v = int(prompt_features_lens[0].item())

            fig, axes = plt.subplots(2, 2, figsize=(14, 8))

            im = axes[0, 0].imshow(pf.T, aspect="auto", origin="lower")
            axes[0, 0].set_title("prompt_features")
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Channel")
            fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

            axes[0, 1].bar([0], [lens_v])
            axes[0, 1].set_xticks([0], ["prompt_features_lens"])
            axes[0, 1].set_title("prompt_features_lens")

            axes[1, 0].plot(ptoks)
            axes[1, 0].set_title(f"prompt_tokens (len={len(ptoks)})")
            axes[1, 0].set_xlabel("Index")
            axes[1, 0].set_ylabel("Token ID")

            axes[1, 1].plot(ttoks)
            axes[1, 1].set_title(f"tokens (len={len(ttoks)})")
            axes[1, 1].set_xlabel("Index")
            axes[1, 1].set_ylabel("Token ID")

            fig.tight_layout()
            debug_fig_path = out_dir / f"{Path(save_path).stem}_debug_i005.png"
            fig.savefig(debug_fig_path, dpi=150)
            plt.close(fig)
            logging.info("Saved debug plot to: %s", str(debug_fig_path))
            chunk_wav = vocoder.decode(prompt_features.permute(0, 2, 1) / feat_scale)
            chunk_wav = chunk_wav.squeeze(1).clamp(-1, 1)
            torchaudio.save(
                f"test_chunk_005_{text_chunk}.wav",
                chunk_wav.cpu(),
                sample_rate=sampling_rate,
            )

        pred_features, pred_features_lens, _, _ = model.sample(
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
        model_seconds += (dt.datetime.now() - model_t0).total_seconds()

        cur_len = pred_features_lens
        cur_len = max(1, cur_len)
        model_chunk = pred_features[:, :cur_len, :]

        keep_frames = min(fixed_chunk_frames, model_chunk.size(1))
        model_chunk = model_chunk[:, :keep_frames, :]

        vocoder_t0 = dt.datetime.now()
        chunk_wav = vocoder.decode(model_chunk.permute(0, 2, 1) / feat_scale)
        chunk_wav = chunk_wav.squeeze(1).clamp(-1, 1)
        vocoder_seconds += (dt.datetime.now() - vocoder_t0).total_seconds()

        if prompt_rms < target_rms:
            chunk_wav = chunk_wav * prompt_rms / target_rms

        torchaudio.save(
            f"{save_path}_chunk_{i:03d}_{text_chunk}.wav",
            chunk_wav.cpu(),
            sample_rate=sampling_rate,
        )

        output_wavs.append(chunk_wav)
        generated_new_frames += keep_frames
        committed_word_pos = expected_word_pos

        prompt_text = (
            base_prompt_text + " " + " ".join(target_words[:committed_word_pos])
        ).strip()
        prompt_features = torch.cat([prompt_features, model_chunk], dim=1)

        if (
            committed_word_pos >= len(target_words)
            and generated_new_frames >= est_total_new_frames
        ):
            break

    if len(output_wavs) == 0:
        raise RuntimeError("No waveform chunk generated")

    final_wav = torch.cat(output_wavs, dim=-1)
    torchaudio.save(
        save_path + "_" + prompt_text + ".wav",
        final_wav.cpu(),
        sample_rate=sampling_rate,
    )

    total_t = (dt.datetime.now() - start_t).total_seconds()
    wav_seconds = final_wav.shape[-1] / sampling_rate

    logging.info("Saved to: %s", save_path + "_" + prompt_text + ".wav")
    logging.info("Generated duration: %.2fs", wav_seconds)
    logging.info("RTF: %.4f", total_t / wav_seconds)
    logging.info("RTF (model): %.4f", model_seconds / wav_seconds)
    logging.info("RTF (vocoder): %.4f", vocoder_seconds / wav_seconds)


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
        "zipvoice": {"num_step": 16, "guidance_scale": 1.0},
        "zipvoice_distill": {"num_step": 8, "guidance_scale": 3.0},
    }
    for k, v in model_defaults[params.model_name].items():
        if getattr(params, k) is None:
            setattr(params, k, v)

    if params.model_dir is not None:
        params.model_dir = Path(params.model_dir)
        if not params.model_dir.is_dir():
            raise FileNotFoundError(f"{params.model_dir} does not exist")
        for filename in [params.checkpoint_name, "model.json", "tokens.txt"]:
            if not (params.model_dir / filename).is_file():
                raise FileNotFoundError(f"{params.model_dir / filename} does not exist")
        model_ckpt = params.model_dir / params.checkpoint_name
        model_config_path = params.model_dir / "model.json"
        token_file = params.model_dir / "tokens.txt"
    else:
        model_ckpt = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.pt"
        )
        model_config_path = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/model.json"
        )
        token_file = hf_hub_download(
            HUGGINGFACE_REPO, filename=f"{MODEL_DIR[params.model_name]}/tokens.txt"
        )

    tokenizer = LibriTTSTokenizer(token_file=token_file)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    tokenizer_config = {"vocab_size": tokenizer.vocab_size, "pad_id": tokenizer.pad_id}
    if params.model_name == "zipvoice":
        model = ZipVoice(**model_config["model"], **tokenizer_config)
    else:
        model = ZipVoiceDistill(**model_config["model"], **tokenizer_config)

    if str(model_ckpt).endswith(".safetensors"):
        safetensors.torch.load_model(model, model_ckpt)
    elif str(model_ckpt).endswith(".pt"):
        load_checkpoint(filename=model_ckpt, model=model, strict=True)
    else:
        raise NotImplementedError(f"Unsupported model checkpoint format: {model_ckpt}")

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("Device: %s", device)

    model = model.to(device)
    model.eval()

    if params.trt_engine_path:
        load_trt(model, params.trt_engine_path)

    vocoder = get_vocoder(params.vocoder_path).to(device)
    vocoder.eval()

    if model_config["feature"]["type"] != "vocos":
        raise NotImplementedError(
            f"Unsupported feature type: {model_config['feature']['type']}"
        )
    feature_extractor = VocosFbank()
    sampling_rate = model_config["feature"]["sampling_rate"]

    generate_streaming_single_sentence(
        save_path=params.res_wav_path,
        prompt_text=params.prompt_text,
        prompt_wav=params.prompt_wav,
        text=params.text,
        model=model,
        vocoder=vocoder,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device,
        num_step=params.num_step,
        guidance_scale=params.guidance_scale,
        speed=params.speed,
        t_shift=params.t_shift,
        target_rms=params.target_rms,
        feat_scale=params.feat_scale,
        sampling_rate=sampling_rate,
        fixed_chunk_frames=params.fixed_chunk_frames,
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    main()
