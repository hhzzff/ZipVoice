"""Helper for converting an aligned word index into a token-character cutoff.

This logic is extracted from ``ZipVoice.condition_time_mask`` (in
``zipvoice/models/zipvoice_stream_fixedwindow_crossattn.py``) so that the
standalone window-pointer probe in
``zipvoice/bin/train_window_pointer.py`` can compute ground-truth
"next-window start position in token space" without depending on the
TTS forward pass.

The original loop walks the alignment word list, matches each word's
upper-cased symbol against the upper-cased text reconstructed from the
token-id sequence, and advances a cursor. The position of the cursor
after consuming the first ``w_count`` words is the token-character
cutoff we want.
"""

from typing import Dict, List


FRAMES_PER_SECOND = 24000.0 / 256.0


def _word_field(word, name: str, default):
    if isinstance(word, dict):
        return word.get(name, default)
    return getattr(word, name, default)


def words_done_at(words, mask_end: int, sr_over_hop: float = FRAMES_PER_SECOND) -> int:
    """Count how many words have finished (mid-point) before ``mask_end``.

    A word is "done" once its temporal mid-point falls before the last
    visible frame. This matches the streaming convention where the
    pointer head is asked: "given everything emitted so far, where do we
    pick up in token space?".
    """
    n = 0
    for w in words:
        start = float(_word_field(w, "start", 0.0))
        duration = float(_word_field(w, "duration", 0.0))
        mid_frame = (start + duration / 2.0) * sr_over_hop
        if mid_frame < mask_end:
            n += 1
        else:
            break
    return n


def words_to_token_cutoff(
    words,
    w_count: int,
    tokens_int: List[int],
    id2text: Dict[int, str],
    fallback: str = "none",
) -> int:
    """Compute the token-char cutoff after consuming the first ``w_count`` words.

    Args:
        fallback: behavior when the upper-case match against the
            reconstructed text fails:
              * ``"none"`` (default) — return -1 to signal "missing label";
                the caller should drop the sample from training/metrics.
              * ``"ratio"`` — estimate the cutoff by word ratio
                (``round(len(tokens) * (w_idx+1) / len(words))``). This
                mirrors the fallback in
                ``zipvoice_stream_fixedwindow_crossattn.condition_time_mask``
                so the prepared training inputs match the TTS recipe.

    Returns the cutoff index into ``tokens_int`` (i.e. the index of the
    first token that has *not* been consumed yet) or ``-1`` when
    ``fallback="none"`` and the matching loop fails.
    """
    if w_count <= 0 or len(tokens_int) == 0 or len(words) == 0:
        return 0

    full_text_upper = "".join(id2text.get(int(t), "") for t in tokens_int).upper()
    cursor = 0
    cutoff_char_pos = 0
    n_tok = len(tokens_int)
    n_words = len(words)

    for w_idx in range(min(w_count, n_words)):
        symbol = str(_word_field(words[w_idx], "symbol", "")).upper()
        if symbol == "":
            continue
        start = full_text_upper.find(symbol, cursor)
        if start == -1:
            if fallback == "ratio":
                est_cut = int(round(n_tok * (w_idx + 1) / max(n_words, 1)))
                cutoff_char_pos = max(cutoff_char_pos, min(n_tok, est_cut))
                break
            return -1
        end = start + len(symbol)
        cursor = end
        cutoff_char_pos = end

    return max(0, min(n_tok, cutoff_char_pos))
