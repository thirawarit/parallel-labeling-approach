"""Thai-aware text normalization and tokenization for metric computation.

Thai has no word delimiters, so word-level metrics require explicit tokenization
(via ``pythainlp``). Normalization is applied before any "normalized" metric.
"""

import re
import string
import unicodedata
from typing import (List)

from pythainlp.tokenize import (word_tokenize)

# Thai-specific punctuation in addition to the ASCII set.
_THAI_PUNCT: str = "ๆฯ"
_PUNCT_TABLE: dict = {ord(ch): " " for ch in (string.punctuation + _THAI_PUNCT)}

# Map Thai digits (๐-๙) to ASCII 0-9.
_THAI_DIGITS: dict = {ord("๐") + i: str(i) for i in range(10)}

_WHITESPACE_RE: "re.Pattern[str]" = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Steps: Unicode NFC, lowercase (for any Latin), Thai-digit -> ASCII-digit,
    strip punctuation, then collapse all whitespace.
    """
    if not text:
        return ""
    normalized: str = unicodedata.normalize("NFC", text)
    normalized = normalized.lower()
    normalized = normalized.translate(_THAI_DIGITS)
    normalized = normalized.translate(_PUNCT_TABLE)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def tokenize_thai(text: str) -> List[str]:
    """Tokenize Thai text into words for word-level metrics.

    Empty input yields an empty token list. Whitespace-only tokens are dropped.
    """
    if not text:
        return []
    tokens: List[str] = word_tokenize(text, engine="newmm", keep_whitespace=False)
    return [tok for tok in tokens if tok.strip()]
