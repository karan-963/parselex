"""Generic, model-first BIO repair for experience phrase segmentation.

The segmentation model drives all real boundary decisions (it learns entity and
style transitions such as ROLE -> COMP and SDATE -> EDATE directly). This layer
only enforces resume-agnostic structural invariants so the output is valid BIO
without any resume-specific tuning:

  1. pure-punctuation separators are not segment content            -> O
  2. the first content token of an entry opens a segment            -> B-SEG
  3. a content token immediately after a separator opens a segment  -> B-SEG
     (bullets, dashes, commas, pipes — matches training GT derivation)
  4. an I-SEG with no open segment is promoted                      -> B-SEG
  5. every other token keeps the model's own B-SEG / I-SEG / O prediction

No layout thresholds, coordinates, or per-resume rules are used here.
"""

from __future__ import annotations

O, B_SEG, I_SEG = 0, 1, 2

# Standalone punctuation tokens that act as visual separators (never content).
_SEPARATORS = frozenset({'"', "|", "•", ",", "—", "–", "-", "/", "●", "▪", "◦"})


def apply_structural_segmentation(
    entry_tokens: list[dict],
    pred_labels: list[int],
) -> list[int]:
    if not entry_tokens or len(pred_labels) != len(entry_tokens):
        return pred_labels

    out = list(pred_labels)
    prev_was_separator = False
    prev_content_label: int | None = None
    seen_content = False

    for idx, t in enumerate(entry_tokens):
        tok = (t.get("token") or "").strip()

        if tok in _SEPARATORS:
            out[idx] = O
            prev_was_separator = True
            continue

        if not seen_content or prev_was_separator:
            out[idx] = B_SEG
        elif out[idx] == I_SEG and prev_content_label is None:
            out[idx] = B_SEG
        # else: preserve the model's own B-SEG / I-SEG / O decision.

        prev_was_separator = False
        if out[idx] in (B_SEG, I_SEG):
            seen_content = True
            prev_content_label = out[idx]

    return out
