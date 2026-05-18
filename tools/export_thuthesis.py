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
  低延迟语音合成要求系统在文本尚未全部处理完成时尽早输出可播放语音，并在后续片段中保持内容、音色和韵律连续。现有高质量流匹配语音合成模型通常面向非流式整句生成，能够使用完整目标文本和完整声学区间；若直接切分为短片段生成，容易出现文本进度错位、窗口边界不连续和多窗口误差累积等问题。针对上述问题，本文在 ZipVoice 条件流匹配语音合成框架基础上，研究固定窗口流式流匹配建模方法。

  本文保留 ZipVoice 的文本编码器、基础流匹配声学解码器和声码器，将研究重点放在流式训练任务与推理控制上。训练阶段通过时间掩码构造“给定历史语音条件，生成当前窗口”的局部补全任务，使模型学习与流式推理一致的窗口级条件分布；推理阶段将目标语音拆分为固定 Mel 窗口，逐段进行流匹配采样，并使用提示语音和历史生成语音作为声学上下文。针对固定声学窗口与文本覆盖范围不一致的问题，本文引入词级进度估计机制，用于更新后续窗口的文本输入位置，从而缓解重复读和漏读问题。

  此外，本文对基于 wav 和 Mel 的 VAE 隐空间建模进行了探索。实验和分析表明，直接使用隐空间表示会受到重建误差、时间同步误差和解码误差放大的影响，当前并不适合作为本文系统的主体方案。因此，本文将隐空间作为辅助探索，并讨论了粗声学生成结合 Mel 残差细化的后续方向。本文工作为条件流匹配语音合成模型向低延迟、逐段输出场景迁移提供了可实现的系统框架。

  \thusetup{
    keywords = {语音合成, 流匹配, 流式生成, ZipVoice, 文本进度估计},
  }
\end{abstract}

\begin{abstract*}
  Low-latency text-to-speech requires a system to emit playable speech before the whole utterance is fully generated, while preserving content accuracy, speaker consistency, and prosodic continuity across segments. Existing high-quality flow-matching TTS models are usually designed for offline utterance-level generation, where the complete target text and acoustic sequence are available. Directly splitting such models into short segments may lead to text progress mismatch, discontinuities at window boundaries, and accumulated errors during multi-window generation. This thesis studies fixed-window streaming flow matching based on the ZipVoice conditional flow-matching TTS framework.

  The proposed system keeps the text encoder, the base flow-matching acoustic decoder, and the vocoder from ZipVoice, and focuses on streaming-oriented training and inference control. During training, a time-mask strategy is used to construct local completion tasks, where the model learns to generate the current acoustic window conditioned on previous speech. During inference, the target speech is generated window by window in the Mel-spectrogram space, with both the prompt speech and generated history used as acoustic context. To address the mismatch between fixed acoustic windows and variable text coverage, a word-level progress predictor is introduced to update the text position after each generated window and reduce repetition or omission.

  This thesis also explores VAE-based latent representations derived from waveform or Mel features. The analysis shows that direct latent-space generation is affected by reconstruction errors, temporal synchronization errors, and error amplification in the decoder, and is therefore treated as an auxiliary exploration rather than the main method. A possible future direction is to combine coarse latent generation with Mel residual refinement. Overall, this work provides an implementable framework for adapting conditional flow-matching TTS models to low-latency streaming generation.

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
