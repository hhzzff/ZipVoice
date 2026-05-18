"""Tiny helpers for CTC greedy decoding.

Used by ``zipvoice/bin/train_ctc_aligner.py`` (eval) and by streaming
inference (``zipvoice/bin/infer_zipvoice_stream_fixed_window_crossattn.py``)
to anchor the next-window word position via CTC over the cumulative
generated mel.
"""

import bisect
from typing import Any, Dict, List, Optional, Tuple

import torch


def greedy_collapse(pred_ids: torch.Tensor, blank_id: int) -> List[int]:
    """CTC greedy decode for a single sample.

    Walks ``pred_ids`` (shape ``(T,)``), merges consecutive duplicates,
    then strips ``blank_id``. Returns the recognized character ids.
    """
    if pred_ids.numel() == 0:
        return []
    out: List[int] = []
    prev = -1
    for v in pred_ids.tolist():
        v = int(v)
        if v != prev:
            if v != blank_id:
                out.append(v)
            prev = v
    return out


def chars_recognized(
    log_probs: torch.Tensor,
    out_lens: torch.Tensor,
    blank_id: int,
) -> List[int]:
    """Count of recognized chars for each item in batch.

    Args:
        log_probs: ``(B, T, V+1)`` model output.
        out_lens: ``(B,)`` int — valid frame counts.
        blank_id: CTC blank id.

    Returns:
        ``List[int]`` of length B; entry ``i`` is
        ``len(greedy_collapse(log_probs[i, :out_lens[i]].argmax(-1)))``.
    """
    pred_ids = log_probs.argmax(dim=-1)
    counts: List[int] = []
    for i in range(pred_ids.size(0)):
        T_i = int(out_lens[i].item())
        counts.append(len(greedy_collapse(pred_ids[i, :T_i], blank_id)))
    return counts


def greedy_decode_batch(
    log_probs: torch.Tensor,
    out_lens: torch.Tensor,
    blank_id: int,
) -> List[List[int]]:
    """Greedy decode for a whole batch. Returns ``List[List[int]]``."""
    pred_ids = log_probs.argmax(dim=-1)
    out: List[List[int]] = []
    for i in range(pred_ids.size(0)):
        T_i = int(out_lens[i].item())
        out.append(greedy_collapse(pred_ids[i, :T_i], blank_id))
    return out


def _rfind_subseq(haystack: List[int], needle: List[int]) -> int:
    """Last index ``i`` such that ``haystack[i:i+len(needle)] == needle``,
    or ``-1`` if not found. Pure Python; needle is at most a few chars
    long so the O(N*M) walk is fine."""
    n = len(needle)
    if n == 0:
        return len(haystack)
    for i in range(len(haystack) - n, -1, -1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def locate_tail(
    decoded: List[int],
    ref: List[int],
    k: int = 3,
) -> Tuple[int, int]:
    """Anchor a pointer position by matching the last ``k`` decoded chars
    into ``ref`` (right-to-left search).

    Front-of-decode CTC errors are common (one substitution shifts the
    char count by 1 forever). This helper ignores them: it takes the
    last ``k`` chars of the (greedy-collapsed) ``decoded`` and finds the
    rightmost place in ``ref`` where they appear contiguously. The
    pointer position is then ``rfind_idx + k``. If the match fails, k is
    decreased by one and retried (down to k=1). If even k=1 fails, fall
    back to ``len(decoded)`` (the legacy "count chars" estimator).

    Decoded tokens that don't appear anywhere in ``ref`` are stripped
    before matching. This handles trailing sentence-end punctuation
    ("." / "!" / "?") that the FM decoder synthesizes but
    ``build_word_token_offsets`` doesn't include in ``ref`` (it uses
    ``" ".join(target_words)`` and ``split_words`` strips punctuation).
    Without the strip, every step's tail contains "." that isn't in
    ``ref`` and ``locate_tail`` falls all the way through to the
    ``len(decoded)`` fallback — defeating the purpose of the rfind.

    Args:
        decoded: greedy-collapsed CTC output for the visible mel prefix.
        ref: the reference token sequence (full sentence).
        k: tail length to match. Defaults to 3.

    Returns:
        ``(pred_pos, matched_k)`` where ``matched_k`` is the actual k
        used (== ``k`` on first hit, ``< k`` if we backed off, ``0`` if
        we fell back to the legacy estimator).
    """
    if len(decoded) == 0:
        return 0, 0

    ref_set = set(ref)
    filtered = [t for t in decoded if t in ref_set]
    if len(filtered) == 0:
        return 0, 0

    k_eff = min(k, len(filtered))
    while k_eff > 0:
        tail = filtered[-k_eff:]
        idx = _rfind_subseq(ref, tail)
        if idx >= 0:
            pred = idx + k_eff
            pred = max(0, min(len(ref), pred))
            return pred, k_eff
        k_eff -= 1

    # Fallback: legacy "len(decoded)" estimator, clamped. Use the
    # filtered length so out-of-vocab punctuation doesn't inflate it.
    pred = max(0, min(len(ref), len(filtered)))
    return pred, 0


def build_word_token_offsets(
    target_words: List[str],
    tokenizer,
) -> Tuple[List[int], List[int]]:
    """Build the CTC-side reference and a word→token-end mapping.

    The streaming inference works in **word indices** but the CTC head
    emits **char-token positions**. To bridge them we tokenize
    ``" ".join(target_words)`` once for the reference and, for each
    word ``i``, record the cumulative token count at the end of word
    ``i`` plus its trailing space (no trailing space for the final
    word). A later ``bisect_right(offsets, ctc_pos - 1)`` then returns
    "number of words whose last char is at or before ``ctc_pos - 1``".

    We re-tokenize each cumulative prefix (rather than per-word and
    summing) so the offsets faithfully reflect whatever normalization
    the tokenizer applies (case folding, apostrophes, punctuation).

    Args:
        target_words: list of words from ``split_words(text)``.
        tokenizer: a tokenizer with ``texts_to_token_ids`` returning
            ``List[List[int]]``.

    Returns:
        ``(ref_token_ids, word_end_offsets)``. If the per-prefix
        offsets disagree with the joint tokenization (rare — happens
        when normalization is non-additive across word boundaries),
        ``word_end_offsets`` is returned empty as a signal for the
        caller to fall back.
    """
    if len(target_words) == 0:
        return [], []

    full_str = " ".join(target_words)
    ref_token_ids = tokenizer.texts_to_token_ids([full_str])[0]

    word_end_offsets: List[int] = []
    for i in range(len(target_words)):
        if i + 1 < len(target_words):
            prefix = " ".join(target_words[: i + 1]) + " "
        else:
            prefix = " ".join(target_words)
        prefix_ids = tokenizer.texts_to_token_ids([prefix])[0]
        word_end_offsets.append(len(prefix_ids))

    # Sanity: monotonic, last offset ≤ len(ref) (could be less when
    # the joint string has trailing punctuation that the per-prefix
    # tokenization of the last word doesn't include — that's fine).
    if any(
        word_end_offsets[i] > word_end_offsets[i + 1]
        for i in range(len(word_end_offsets) - 1)
    ):
        return ref_token_ids, []
    if word_end_offsets[-1] > len(ref_token_ids):
        return ref_token_ids, []

    return ref_token_ids, word_end_offsets


def ctc_pointer_word_pos(
    cumulative_gen_mel: torch.Tensor,
    gen_mel_len: int,
    ref_token_ids: List[int],
    word_end_offsets: List[int],
    aligner,
    tail_k: int = 3,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Run CTC + ``locate_tail`` + bisect to map cumulative mel to a
    target-word index.

    Args:
        cumulative_gen_mel: ``(1, T, feat_dim)``, already scaled by
            the same ``feat_scale`` the CTC was trained on.
        gen_mel_len: actual frame count (``cumulative_gen_mel.size(1)``,
            but passed explicitly so the caller can pre-clamp).
        ref_token_ids: full-sentence token sequence (output of
            ``build_word_token_offsets``).
        word_end_offsets: cumulative token count at the end of each
            word (output of ``build_word_token_offsets``).
        aligner: a ``CTCAligner`` in eval mode on the right device.
        tail_k: tail length for ``locate_tail``.

    Returns:
        ``(word_pos_or_None, info)``. ``word_pos`` is ``None`` when
        CTC is low-confidence (caller should fall back to ratio).
        ``info`` always contains keys ``token_pos``, ``matched_k``,
        ``n_decoded``, ``used_fallback``.
    """
    info: Dict[str, Any] = {
        "token_pos": 0,
        "matched_k": 0,
        "n_decoded": 0,
        "used_fallback": False,
        "decoded": [],
    }
    if gen_mel_len <= 0 or len(ref_token_ids) == 0 or len(word_end_offsets) == 0:
        return None, info

    device = next(aligner.parameters()).device
    mel = cumulative_gen_mel.to(device)
    mel_lens = torch.tensor([gen_mel_len], dtype=torch.long, device=device)
    log_probs, out_lens = aligner(mel, mel_lens)
    T = int(out_lens[0].item())
    if T <= 0:
        return None, info

    argmax = log_probs[0, :T].argmax(dim=-1)
    decoded = greedy_collapse(argmax, blank_id=aligner.blank_id)
    info["n_decoded"] = len(decoded)
    info["decoded"] = decoded
    if len(decoded) < tail_k:
        return None, info

    token_pos, matched_k = locate_tail(decoded, ref_token_ids, k=tail_k)
    info["token_pos"] = token_pos
    info["matched_k"] = matched_k
    info["used_fallback"] = (matched_k == 0)

    # rfind failed (matched_k == 0). The legacy len(decoded) estimator
    # is only reliable when there's enough context.
    if matched_k == 0 and len(decoded) < tail_k * 2:
        return None, info

    # bisect_right(offsets, token_pos - 1) = number of words whose
    # last char is at or before token_pos - 1.
    word_pos = bisect.bisect_right(word_end_offsets, max(0, token_pos - 1))
    word_pos = max(0, min(len(word_end_offsets), word_pos))
    return word_pos, info


def ctc_forced_alignment(
    log_probs: torch.Tensor,
    out_lens: torch.Tensor,
    targets: List[List[int]],
    blank_id: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """CTC trellis Viterbi best-path forced alignment.

    For each sample, finds the maximum-probability path through the
    standard CTC ``2*L+1``-state trellis (interleaved blanks + target
    tokens) that emits exactly ``targets[i]`` in order. Returns a
    per-frame label tensor suitable for cross-entropy supervision.

    Used by ``train_ctc_aligner.py`` Phase-2 self-distillation: take the
    model's own posterior, force-align it, then train framewise CE on
    that path. Sharpens the spikes without depending on external
    alignments.

    Args:
        log_probs: ``(B, T, V+1)`` log-softmax model output.
        out_lens: ``(B,)`` valid frame counts.
        targets: list of length B; each ``List[int]`` is the target
            token sequence (no blanks).
        blank_id: CTC blank id.
        ignore_index: label written into pad positions ``t >= out_lens[i]``
            (and into samples where Viterbi has no valid path, e.g.
            ``out_lens[i] < target_len``). Pass to ``F.cross_entropy``
            via its ``ignore_index`` arg.

    Returns:
        ``(B, T)`` int64 tensor of forced labels (token ids in
        ``[0, V]``, ``ignore_index`` elsewhere). Same device as input.
    """
    B, T, _ = log_probs.shape
    device = log_probs.device
    out = torch.full((B, T), ignore_index, dtype=torch.long, device=device)

    lp_cpu = log_probs.detach().cpu()
    out_lens_cpu = out_lens.detach().cpu().tolist()

    NEG_INF = float("-inf")

    for b in range(B):
        T_b = int(out_lens_cpu[b])
        tgt = targets[b]
        L = len(tgt)
        if T_b <= 0 or L == 0:
            continue
        # State sequence: blank, t0, blank, t1, ..., blank, t_{L-1}, blank.
        S = 2 * L + 1
        if T_b < L:
            # No valid CTC path; leave row as ignore_index.
            continue

        state_ids = [blank_id] * S
        for j in range(L):
            state_ids[2 * j + 1] = int(tgt[j])

        lp_b = lp_cpu[b]  # (T, V+1)

        # Viterbi over states. dp[s] = best log-prob ending at state s
        # at current time. backptr[t, s] = predecessor state at t-1.
        dp = torch.full((S,), NEG_INF, dtype=torch.float32)
        dp[0] = float(lp_b[0, blank_id].item())
        if S >= 2:
            dp[1] = float(lp_b[0, state_ids[1]].item())
        backptr = torch.zeros((T_b, S), dtype=torch.int16)

        for t in range(1, T_b):
            new_dp = torch.full((S,), NEG_INF, dtype=torch.float32)
            for s in range(S):
                # Candidate predecessors: stay (s), step (s-1), or
                # skip-blank (s-2) when current state is a non-blank
                # target *and* differs from target two states back.
                best_prev = s
                best_val = dp[s]
                if s - 1 >= 0 and dp[s - 1] > best_val:
                    best_val = dp[s - 1]
                    best_prev = s - 1
                if (
                    s - 2 >= 0
                    and (s % 2 == 1)
                    and state_ids[s] != state_ids[s - 2]
                    and dp[s - 2] > best_val
                ):
                    best_val = dp[s - 2]
                    best_prev = s - 2
                emit = float(lp_b[t, state_ids[s]].item())
                new_dp[s] = best_val + emit
                backptr[t, s] = best_prev
            dp = new_dp

        # Best terminal state: last blank (S-1) or last target (S-2).
        cand = [S - 1] + ([S - 2] if S >= 2 else [])
        best_s = max(cand, key=lambda s: float(dp[s].item()))
        if dp[best_s].item() == NEG_INF:
            continue

        # Backtrace.
        path = [0] * T_b
        s = best_s
        for t in range(T_b - 1, -1, -1):
            path[t] = state_ids[s]
            s = int(backptr[t, s].item())

        out[b, :T_b] = torch.tensor(path, dtype=torch.long, device=device)

    return out
