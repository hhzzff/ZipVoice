#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "docs" / "流式流匹配语音合成论文草稿.docx"
THU = ROOT / "docs" / "thuthesis-v7.7.0"

CITE_KEYS = {
    1: "shen2018tacotron2",
    2: "ren2019fastspeech",
    3: "ren2021fastspeech2",
    4: "kim2021vits",
    5: "wang2023valle",
    6: "popov2021gradtts",
    7: "lipman2023flowmatching",
    8: "mehta2023matchatts",
    9: "le2023voicebox",
    10: "zhu2025zipvoice",
    11: "du2024incrementalfastpitch",
    12: "dang2024livespeech",
    13: "bai2025speakstream",
    14: "siuzdak2024vocos",
}

BIB_ENTRIES = r"""@inproceedings{shen2018tacotron2,
  author    = {Shen, Jonathan and Pang, Ruoming and Weiss, Ron J. and Schuster, Mike and Jaitly, Navdeep and Yang, Zongheng and Chen, Zhifeng and Zhang, Yu and Wang, Yuxuan and Skerrv-Ryan, R. J. and Saurous, Rif A. and Agiomyrgiannakis, Yannis and Wu, Yonghui},
  title     = {Natural {TTS} Synthesis by Conditioning {WaveNet} on Mel Spectrogram Predictions},
  booktitle = {Proceedings of ICASSP},
  pages     = {4779--4783},
  year      = {2018}
}

@inproceedings{ren2019fastspeech,
  author    = {Ren, Yi and Ruan, Yangjun and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  title     = {{FastSpeech}: Fast, Robust and Controllable Text to Speech},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}

@inproceedings{ren2021fastspeech2,
  author    = {Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  title     = {{FastSpeech 2}: Fast and High-Quality End-to-End Text to Speech},
  booktitle = {International Conference on Learning Representations},
  year      = {2021}
}

@inproceedings{kim2021vits,
  author    = {Kim, Jaehyeon and Kong, Jungil and Son, Juhee},
  title     = {Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech},
  booktitle = {International Conference on Machine Learning},
  year      = {2021}
}

@article{wang2023valle,
  author  = {Wang, Chengyi and Chen, Sanyuan and Wu, Yu and Zhang, Ziqiang and Zhou, Long and Liu, Shujie and Chen, Zhuo and Liu, Yanqing and Wang, Huaming and Li, Jinyu and He, Lei and Zhao, Sheng},
  title   = {Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers},
  journal = {arXiv preprint arXiv:2301.02111},
  year    = {2023}
}

@inproceedings{popov2021gradtts,
  author    = {Popov, Vadim and Vovk, Ivan and Gogoryan, Vladimir and Sadekova, Tatiana and Kudinov, Mikhail},
  title     = {{Grad-TTS}: A Diffusion Probabilistic Model for Text-to-Speech},
  booktitle = {International Conference on Machine Learning},
  year      = {2021}
}

@inproceedings{lipman2023flowmatching,
  author    = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew},
  title     = {Flow Matching for Generative Modeling},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@article{mehta2023matchatts,
  author  = {Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  title   = {{Matcha-TTS}: A Fast {TTS} Architecture with Conditional Flow Matching},
  journal = {arXiv preprint arXiv:2309.03199},
  year    = {2023}
}

@article{le2023voicebox,
  author  = {Le, Matthew and Vyas, Apoorv and Shi, Bowen and Karrer, Brian and Sari, Leda and Moritz, Rashel and Williamson, Mary and Manohar, Vimal and Adi, Yossi and Mahadeokar, Jay and Hsu, Wei-Ning},
  title   = {{Voicebox}: Text-Guided Multilingual Universal Speech Generation at Scale},
  journal = {arXiv preprint arXiv:2306.15687},
  year    = {2023}
}

@article{zhu2025zipvoice,
  author  = {Zhu, Han and Kang, Wei and Yao, Zhe and others},
  title   = {{ZipVoice}: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
  journal = {arXiv preprint arXiv:2506.13053},
  year    = {2025}
}

@article{du2024incrementalfastpitch,
  author  = {Du, Min and Liu, Chang and Lai, Jun},
  title   = {Incremental {FastPitch}: Chunk-Based High Quality Text to Speech},
  journal = {arXiv preprint arXiv:2401.01755},
  year    = {2024}
}

@inproceedings{dang2024livespeech,
  author    = {Dang, Tu Anh and Aponte, Daniel and Tran, Dung and others},
  title     = {{LiveSpeech}: Low-Latency Zero-Shot Text-to-Speech via Autoregressive Modeling of Audio Discrete Codes},
  booktitle = {Proceedings of Interspeech},
  pages     = {3395--3399},
  year      = {2024}
}

@article{bai2025speakstream,
  author  = {Bai, R. H. and Gu, Z. and Likhomanenko, T. and others},
  title   = {{SpeakStream}: Streaming Text-to-Speech with Interleaved Data},
  journal = {arXiv preprint arXiv:2505.19206},
  year    = {2025}
}

@inproceedings{siuzdak2024vocos,
  author    = {Siuzdak, Hubert},
  title     = {{Vocos}: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis},
  booktitle = {International Conference on Learning Representations},
  year      = {2024}
}
"""


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def replace_citations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        keys: list[str] = []
        for part in match.group(1).split(","):
            part = part.strip()
            if "-" in part:
                start, end = [int(x) for x in part.split("-", 1)]
                if start not in CITE_KEYS or end not in CITE_KEYS:
                    return match.group(0)
                keys.extend(CITE_KEYS[i] for i in range(start, end + 1))
            else:
                idx = int(part)
                if idx not in CITE_KEYS:
                    return match.group(0)
                keys.append(CITE_KEYS[idx])
        return r"\cite{" + ",".join(keys) + "}"

    return re.sub(r"\[((?:\d+(?:-\d+)?)(?:,\s*\d+(?:-\d+)?)*)\]", repl, text)


MATH_REPLACEMENTS = [
    ("X ∈ R^{T×D}", r"$X \in \mathbb{R}^{T \times D}$"),
    ("y1, y2, ..., yK", r"$y_1, y_2, \ldots, y_K$"),
    ("y1:k-1", r"$y_{1:k-1}$"),
    ("xt = t x1 + (1 - t) x0", r"$x_t = t x_1 + (1 - t)x_0$"),
    ("ut = x1 - x0", r"$u_t = x_1 - x_0$"),
    ("LFM = ||vt - ut||²", r"$\mathcal{L}_{\mathrm{FM}} = \lVert v_t - u_t \rVert_2^2$"),
    ("vt = fθ(xt, t, c)", r"$v_t = f_\theta(x_t, t, c)$"),
    (
        "x_{i+1} = x_i + fθ(x_i, t_i, c)(t_{i+1} - t_i)",
        r"$x_{i+1} = x_i + f_\theta(x_i, t_i, c)(t_{i+1} - t_i)$",
    ),
    ("zt = t z1 + (1 - t) z0", r"$z_t = t z_1 + (1 - t)z_0$"),
    ("uz = z1 - z0", r"$u_z = z_1 - z_0$"),
    ("t0, t1, ..., tN", r"$t_0, t_1, \ldots, t_N$"),
    ("t ∈ [0, 1]", r"$t \in [0, 1]$"),
    ("[s, e)", r"$[s, e)$"),
    ("t = 0", r"$t = 0$"),
    ("t = 1", r"$t = 1$"),
    ("fθ", r"$f_\theta$"),
]

MATH_TOKEN_REPLACEMENTS = [
    (r"(?<![A-Za-z])x0(?![A-Za-z])", r"$x_0$"),
    (r"(?<![A-Za-z])x1(?![A-Za-z])", r"$x_1$"),
    (r"(?<![A-Za-z])xt(?![A-Za-z])", r"$x_t$"),
    (r"(?<![A-Za-z])ut(?![A-Za-z])", r"$u_t$"),
    (r"(?<![A-Za-z])vt(?![A-Za-z])", r"$v_t$"),
    (r"(?<![A-Za-z])xN(?![A-Za-z])", r"$x_N$"),
    (r"(?<![A-Za-z])z0(?![A-Za-z])", r"$z_0$"),
    (r"(?<![A-Za-z])z1(?![A-Za-z])", r"$z_1$"),
    (r"(?<![A-Za-z])zt(?![A-Za-z])", r"$z_t$"),
    (r"(?<![A-Za-z])uz(?![A-Za-z])", r"$u_z$"),
    (r"(?<![A-Za-z])vz(?![A-Za-z])", r"$v_z$"),
]


def normalize_heading(text: str) -> tuple[str, str]:
    chapter = re.match(r"第(\d+)章\s+(.+)", text)
    if chapter:
        return "chapter", chapter.group(2)
    subsection = re.match(r"\d+\.\d+\.\d+\s+(.+)", text)
    if subsection:
        return "subsection", subsection.group(1)
    section = re.match(r"\d+\.\d+\s+(.+)", text)
    if section:
        return "section", section.group(1)
    numbered_item = re.match(r"\d+\.\s+(.+)", text)
    if numbered_item:
        return "subsection", numbered_item.group(1)
    return "paragraph", text


def render_para(text: str) -> str:
    cited = replace_citations(text)
    protected: list[str] = []

    def protect(value: str) -> str:
        token = f"@@PROTECTED{len(protected)}@@"
        protected.append(value)
        return token

    content = re.sub(r"\\cite\{[^}]+\}", lambda match: protect(match.group(0)), cited)
    for source, target in MATH_REPLACEMENTS:
        content = content.replace(source, protect(target))
    for pattern, target in MATH_TOKEN_REPLACEMENTS:
        content = re.sub(pattern, lambda _match, value=target: protect(value), content)

    escaped = tex_escape(content)
    for index, value in enumerate(protected):
        escaped = escaped.replace(f"@@PROTECTED{index}@@", value)
    return escaped


def render_items(items: list[str | tuple[str, str]]) -> str:
    lines = ["% !TEX root = ../thuthesis-example.tex", ""]
    for item in items:
        if isinstance(item, tuple):
            text, style = item
        else:
            text, style = item, "论文正文段落"

        if style == "Heading 1":
            _, title = normalize_heading(text)
            lines.extend([rf"\chapter{{{tex_escape(title)}}}", ""])
        elif style == "Heading 2":
            _, title = normalize_heading(text)
            lines.extend([rf"\section{{{tex_escape(title)}}}", ""])
        elif style == "Heading 3":
            _, title = normalize_heading(text)
            lines.extend([rf"\subsection{{{tex_escape(title)}}}", ""])
        else:
            lines.extend([render_para(text), ""])
    return "\n".join(lines).rstrip() + "\n"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def extract_chapters_from_docx() -> dict[str, list[tuple[str, str]]]:
    doc = Document(DOCX)
    chapters: dict[str, list[tuple[str, str]]] = {}
    current: list[tuple[str, str]] | None = None
    chapter_no = 0

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style = paragraph.style.name
        if style == "Title" and text == "参考文献":
            break

        if style == "Heading 1" and re.match(r"第\d+章\s+", text):
            chapter_no += 1
            current = [(text, style)]
            chapters[f"chap{chapter_no:02d}.tex"] = current
            continue

        if current is None:
            continue

        if style in {"Heading 1", "Heading 2", "Heading 3"}:
            current.append((text, style))
        else:
            current.append((text, "论文正文段落"))

    if not chapters:
        raise RuntimeError(f"No thesis chapters found in {DOCX}")

    return chapters


def main() -> None:
    chapters = extract_chapters_from_docx()
    for filename, items in chapters.items():
        write(THU / "data" / filename, render_items(items))

    write(
        THU / "data" / "abstract.tex",
        r"""% !TEX root = ../thuthesis-example.tex

\begin{abstract}
  近年来，大语言模型在文本理解、任务规划和多轮对话等方面的能力快速提升，面向用户的实时交互助手逐渐被广泛使用。语音作为人机交互中最自然的输出形式之一，成为了用户与实时交互助手交流的最重要的媒介之一。在实时语音交互过程中，语音助手不仅需要生成自然、清晰的语音，还需要模拟用户与人类的语音交互习惯，在用户等待时间有限的情况下尽快开始发声。然而，传统非流式的文本到语音合成系统（Text-to-Speech, TTS）通常需要等待完整文本输入，等待整句声学特征生成和声码器解码完成后才能输出音频，在实时问答、双工对话等场景中容易造成延迟高、交互停顿明显等问题；同时，现有的开源流式 TTS 系统普遍存在首包延迟过高的问题，在实时交互中通常不够自然。为解决上述问题，本文面向低延迟实时语音交互需求，在 ZipVoice 条件流匹配语音合成框架基础上研究流式流匹配语音合成方法，探索如何在保持语音质量和内容正确性的同时实现流式合成与快速响应。

  对于非流式整句生成 TTS 模型难以直接满足低延迟输出的问题，本文提出固定窗长的流式流匹配语音合成框架。该框架将目标语音划分为连续的、长度相同的梅尔频谱窗口，每次只生成当前窗口对应的声学片段，并在窗口之间递推更新语音上下文。训练阶段，通过掩盖随机语音片段构造局部补全任务，使模型学习在提示语音和文本条件约束下自主选词生成下一段语音；在推理阶段，每一个合成窗口从随机噪声出发进行流匹配采样，并将生成结果加入历史缓存以保证声学连贯性，支持后续窗口生成。实验结果表明，该框架能够将条件流匹配模型组织为可逐窗口输出的流式系统，为降低首段音频输出等待时间提供了可行路径。

  针对固定声学窗口与文本覆盖范围不一致的问题，本文进一步研究流式生成中的文本进度估计与稳定性优化。固定窗长只能约束每轮生成的声学帧数，却不能直接确定当前语音实际覆盖的词数；若文本进度推进过快或过慢，后续窗口容易出现漏读、重复读或文本错位等问题。为此，本文引入词级指针机制，根据生成语音片段和文本上下文估计当前窗口对应的文本推进量，并结合历史声学上下文维护跨窗口连续性。实验结果表明，相比固定比例推进和固定词数推进，词级指针能够更有效地缓解文本进度偏移带来的重复和漏读问题。

  在隐空间建模方面，本文对基于波形（waveform, wav）和梅尔频谱（Mel spectrogram, Mel）的变分自编码器（Variational Autoencoder, VAE）隐变量（latent）表示进行了探索。隐空间方法的动机是降低声学建模维度，从而减轻流匹配模型直接生成高维声学特征的难度；但实际实验表明，直接将 VAE 隐变量用于流式流匹配会受到重建误差、时间同步误差和解码误差放大的影响，容易造成语音细节损失、发音模糊和窗口衔接不稳定。因此，本文将隐空间建模作为辅助探索而非主体贡献，并讨论了粗粒度隐变量（coarse latent）生成结合 Mel 残差细化的后续改进方向。

  综合来看，本文将固定窗长流式生成、时间掩码训练、历史语音上下文和词级文本进度估计结合起来，构建了面向实时交互场景的流式流匹配语音合成系统。实验分析表明，本文方法能够支持逐段语音输出，并在语音质量、文本准确性和生成延迟之间提供可分析、可调节的折中机制。该研究为大语言模型驱动的实时语音助手提供了低延迟语音输出的技术基础，也为后续进一步优化窗口边界连续性、文本进度控制和轻量化推理提供了参考。

  \thusetup{
    keywords = {语音合成, 流匹配, 流式生成, ZipVoice, 文本进度估计},
  }
\end{abstract}

\begin{abstract*}
  Recent advances in large language models have greatly improved text understanding, task planning, and multi-turn dialogue, making real-time interactive assistants increasingly common. As one of the most natural output modalities for human-computer interaction, speech generation is expected not only to be natural and intelligible, but also to start promptly under strict response-time constraints. However, conventional non-streaming text-to-speech systems usually wait for complete text processing, full-utterance acoustic generation, and vocoder decoding before playback, which leads to high first-packet latency and noticeable pauses in real-time dialogue and long-form reading. To address this problem, this thesis studies streaming flow-matching text-to-speech based on the ZipVoice conditional flow-matching framework, aiming to support segment-by-segment generation while preserving speech quality and content accuracy.

  To address the difficulty of applying utterance-level generation models to low-latency output, this thesis proposes a fixed-window streaming flow-matching text-to-speech (TTS) framework. The target speech is divided into consecutive Mel-spectrogram windows, and each step generates only the acoustic segment for the current window while recursively updating the speech context. During training, time masking constructs local completion tasks so that the model learns to generate the next speech segment conditioned on prompt speech, generated history, and text information. During inference, each window is sampled from random noise through flow matching, and the generated segment is added to the history cache for subsequent windows. Experimental results show that this framework organizes conditional flow matching into a streaming system with window-by-window output and provides a feasible way to reduce the waiting time before the first playable segment.

  To address the mismatch between fixed acoustic windows and variable text coverage, this thesis further studies text progress estimation and stability optimization. A fixed window constrains the number of acoustic frames generated in each step, but it cannot directly determine how many words have actually been covered by the current speech segment. If the text position advances too quickly or too slowly, later windows may suffer from omissions, repetitions, or content mismatch. Therefore, a word-level pointer is introduced to estimate the text advancement of each generated window from the generated speech segment and text context, while historical acoustic context is used to improve cross-window continuity. Experimental comparisons indicate that the word-level pointer is more effective than fixed-ratio or fixed-word advancement in reducing repetition and omission.

  This thesis also explores latent representations based on waveform and Mel-spectrogram features using a variational autoencoder (VAE). The motivation of latent-space modeling is to reduce the dimensionality of acoustic targets and alleviate the difficulty of directly generating high-dimensional acoustic features with flow matching. However, experiments show that direct VAE-latent generation is affected by reconstruction errors, temporal synchronization errors, and error amplification in the decoder, leading to weakened speech details, blurred pronunciation, and unstable window transitions. Therefore, latent modeling is treated as an auxiliary exploration rather than the main contribution, and a possible future direction is to combine coarse latent generation with Mel residual refinement.

  Overall, this work combines fixed-window streaming generation, time-mask training, historical speech context, and word-level text progress estimation into a streaming flow-matching text-to-speech system for real-time interaction. The analysis shows that the proposed method supports segment-by-segment speech output and provides an adjustable framework for balancing speech quality, text accuracy, and generation latency. This study provides a technical basis for low-latency speech output in large-language-model-driven interactive assistants and offers guidance for future improvements in boundary continuity, progress control, and efficient inference.

  \thusetup{
    keywords* = {text-to-speech, flow matching, streaming generation, ZipVoice, text progress prediction},
  }
\end{abstract*}
""",
    )

    write(
        THU / "data" / "denotation.tex",
        r"""% !TEX root = ../thuthesis-example.tex

\begin{denotation}[3cm]
  \item[TTS] 文本到语音（Text-to-Speech）
  \item[FM] 流匹配（Flow Matching）
  \item[CFM] 条件流匹配（Conditional Flow Matching）
  \item[Mel] Mel 频谱特征
  \item[VAE] 变分自编码器（Variational Autoencoder）
  \item[WER] 词错误率（Word Error Rate）
  \item[CER] 字符错误率（Character Error Rate）
  \item[MOS] 平均意见得分（Mean Opinion Score）
  \item[RTF] 实时率（Real-Time Factor）
  \item[$x_0$] 初始噪声声学特征
  \item[$x_1$] 真实声学特征
  \item[$x_t$] 流匹配路径上的中间状态
  \item[$u_t$] 真实速度场
  \item[$v_t$] 模型预测速度场
  \item[$W$] 固定窗口长度
\end{denotation}
""",
    )

    write(
        THU / "data" / "acknowledgements.tex",
        r"""% !TEX root = ../thuthesis-example.tex

\begin{acknowledgements}
  感谢导师在选题、实验设计和论文写作过程中给予的指导。感谢课题组同学在代码调试、实验讨论和结果分析中的帮助。本文工作基于开源 ZipVoice 框架展开，也感谢相关开源社区提供的研究基础。
\end{acknowledgements}
""",
    )

    write(
        THU / "data" / "appendix.tex",
        r"""% !TEX root = ../thuthesis-example.tex

\chapter{补充材料}

本附录用于记录与正文实验相关的补充材料。后续可根据实际实验情况加入详细配置文件、测试样例列表、主观评测问卷或更多可视化结果。
""",
    )

    write(THU / "ref" / "refs.bib", BIB_ENTRIES)


if __name__ == "__main__":
    main()
