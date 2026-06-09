"""Pairwise agreement metrics between model transcriptions.

Agreement is reported as *dissimilarity* (CER / WER): 0.0 means identical, higher
means more disagreement. Metrics are computed on both raw and normalized text.
Word-level WER uses pythainlp tokenization (Thai has no word spaces).
"""

from dataclasses import (dataclass, field)
from itertools import (combinations)
from typing import (Dict, List, Optional, Sequence, Tuple)

import jiwer

from parallel_labeling.normalize import (normalize_text, strip_word_spaces, tokenize_thai)

# Number of decimal places all reported scores are rounded to.
SCORE_NDIGITS: int = 4


def round_score(value: float) -> float:
    """Round a score to ``SCORE_NDIGITS`` decimal places for consistent output."""
    return round(value, SCORE_NDIGITS)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level edit distance normalized by reference length.

    Both empty -> 0.0 (perfect agreement). Empty reference with non-empty
    hypothesis -> 1.0. Result is rounded to ``SCORE_NDIGITS`` places.
    """
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    return round_score(float(jiwer.cer(reference, hypothesis)))


def word_error_rate_thai(reference: str, hypothesis: str) -> float:
    """Word-level error rate after Thai tokenization.

    Operates on space-joined token strings so jiwer's word splitting aligns with
    pythainlp tokenization.
    """
    ref_tokens: List[str] = tokenize_thai(reference)
    hyp_tokens: List[str] = tokenize_thai(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 1.0
    return round_score(float(jiwer.wer(" ".join(ref_tokens), " ".join(hyp_tokens))))


@dataclass
class PairMetrics:
    """Agreement metrics for one ordered model pair (a, b)."""

    pair: Tuple[str, str]
    cer_raw: float
    cer_norm: float
    wer_norm: float


@dataclass
class FileComparison:
    """All pairwise + per-model agreement results for a single audio file."""

    pairs: List[PairMetrics] = field(default_factory=list)
    mean_agreement_per_model: Dict[str, float] = field(default_factory=dict)
    unanimous: bool = False
    # Consensus pick: the model whose output disagrees least with the others
    # (the medoid). ``best_model`` is None when no model produced usable output.
    best_model: Optional[str] = None
    best_text: Optional[str] = None


def _agreement_inputs(
    hypotheses: Dict[str, Optional[str]],
    model_keys: Sequence[str],
) -> List[str]:
    """Return model keys that produced a usable (non-None) hypothesis."""
    return [k for k in model_keys if hypotheses.get(k) is not None]


def compare_hypotheses(
    hypotheses: Dict[str, Optional[str]],
    model_keys: Sequence[str],
) -> FileComparison:
    """Compute pairwise metrics across all model pairs for one file.

    ``hypotheses`` maps model key -> transcription (or ``None`` if that model
    failed on this file). Pairs involving a failed model are skipped. ``unanimous``
    requires every model to have produced output and all normalized outputs equal.
    """
    usable: List[str] = _agreement_inputs(hypotheses, model_keys)
    norm_cache: Dict[str, str] = {k: normalize_text(hypotheses[k] or "") for k in usable}

    pairs: List[PairMetrics] = []
    # Accumulate normalized-CER per model to derive an outlier signal.
    per_model_sum: Dict[str, float] = {k: 0.0 for k in usable}
    per_model_count: Dict[str, int] = {k: 0 for k in usable}

    for a, b in combinations(usable, 2):
        cer_raw: float = character_error_rate(hypotheses[a] or "", hypotheses[b] or "")
        cer_norm: float = character_error_rate(norm_cache[a], norm_cache[b])
        wer_norm: float = word_error_rate_thai(norm_cache[a], norm_cache[b])
        pairs.append(PairMetrics(pair=(a, b), cer_raw=cer_raw, cer_norm=cer_norm, wer_norm=wer_norm))
        for key in (a, b):
            per_model_sum[key] += cer_norm
            per_model_count[key] += 1

    mean_per_model: Dict[str, float] = {
        k: round_score(per_model_sum[k] / per_model_count[k]) if per_model_count[k] else 0.0
        for k in usable
    }

    all_present: bool = len(usable) == len(model_keys) and len(model_keys) > 0
    unanimous: bool = all_present and len({norm_cache[k] for k in usable}) == 1

    # Consensus pick (medoid): the usable model with the lowest mean disagreement.
    # Ties broken by model_keys order so the choice is deterministic. With a
    # single usable model it is the pick by default; with none, there is no pick.
    best_model: Optional[str] = None
    best_text: Optional[str] = None
    if usable:
        best_model = min(usable, key=lambda k: (mean_per_model[k], model_keys.index(k)))
        # Clean pseudo-label: drop inserted word-spaces, keep original characters.
        best_text = strip_word_spaces(hypotheses[best_model] or "")

    return FileComparison(
        pairs=pairs,
        mean_agreement_per_model=mean_per_model,
        unanimous=unanimous,
        best_model=best_model,
        best_text=best_text,
    )
