"""Lightweight stand-ins for the heavy ML dependencies.

Installing these into ``sys.modules`` *before* importing ``parallel_labeling``
lets the synthetic smoke test exercise the dataset -> runner -> metrics -> report
pipeline without torch/transformers/pythainlp/jiwer/PyYAML present.
"""

import sys
import types
from typing import (Any, Dict, List)


def _install_yaml() -> None:
    yaml = types.ModuleType("yaml")
    yaml.safe_load = lambda stream: {}  # type: ignore[attr-defined]
    sys.modules.setdefault("yaml", yaml)


def _install_torch() -> None:
    torch = types.ModuleType("torch")
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]

    cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
    )
    torch.cuda = cuda  # type: ignore[attr-defined]
    sys.modules.setdefault("torch", torch)


def _install_transformers() -> None:
    transformers = types.ModuleType("transformers")

    def pipeline(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - not called
        raise RuntimeError("real pipeline must be monkeypatched in tests")

    transformers.pipeline = pipeline  # type: ignore[attr-defined]
    sys.modules.setdefault("transformers", transformers)


def _install_pythainlp() -> None:
    pythainlp = types.ModuleType("pythainlp")
    tokenize = types.ModuleType("pythainlp.tokenize")

    def word_tokenize(text: str, **kwargs: Any) -> List[str]:
        # Whitespace split is enough for deterministic test inputs.
        return text.split()

    tokenize.word_tokenize = word_tokenize  # type: ignore[attr-defined]
    pythainlp.tokenize = tokenize  # type: ignore[attr-defined]
    sys.modules.setdefault("pythainlp", pythainlp)
    sys.modules.setdefault("pythainlp.tokenize", tokenize)


def _install_jiwer() -> None:
    jiwer = types.ModuleType("jiwer")

    def _levenshtein(a: List[str], b: List[str]) -> int:
        prev: List[int] = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            cur: List[int] = [i]
            for j, cb in enumerate(b, start=1):
                cost: int = 0 if ca == cb else 1
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
            prev = cur
        return prev[-1]

    def cer(reference: str, hypothesis: str) -> float:
        ref: List[str] = list(reference)
        if not ref:
            return 0.0 if not hypothesis else 1.0
        return _levenshtein(ref, list(hypothesis)) / len(ref)

    def wer(reference: str, hypothesis: str) -> float:
        ref: List[str] = reference.split()
        if not ref:
            return 0.0 if not hypothesis.split() else 1.0
        return _levenshtein(ref, hypothesis.split()) / len(ref)

    jiwer.cer = cer  # type: ignore[attr-defined]
    jiwer.wer = wer  # type: ignore[attr-defined]
    sys.modules.setdefault("jiwer", jiwer)


def install_all() -> None:
    """Install every dependency stub. Call before importing parallel_labeling."""
    _install_yaml()
    _install_torch()
    _install_transformers()
    _install_pythainlp()
    _install_jiwer()


class FakeASR:
    """Deterministic stand-in for a transformers ASR pipeline.

    Returns a transcription looked up by audio file stem, so different "models"
    can be made to agree or disagree on a per-file basis. Raises for any stem in
    ``fail_on`` to exercise the per-cell error path.
    """

    def __init__(self, outputs: Dict[str, str], fail_on: List[str]) -> None:
        self._outputs: Dict[str, str] = outputs
        self._fail_on: List[str] = fail_on

    def __call__(self, audio_path: str, **kwargs: Any) -> Dict[str, str]:
        import os

        stem: str = os.path.splitext(os.path.basename(audio_path))[0]
        if stem in self._fail_on:
            raise RuntimeError(f"synthetic failure on {stem}")
        return {"text": self._outputs.get(stem, "")}
