"""Load the HuggingFace AudioFolder ``metadata.jsonl`` and resolve audio paths."""

import json
from dataclasses import (dataclass, replace)
from pathlib import (Path)
from typing import (Dict, List, Sequence)

from parallel_labeling.logging_utils import (get_logger)

logger = get_logger(__name__)

METADATA_FILENAME: str = "metadata.jsonl"


@dataclass
class AudioItem:
    """One row of the dataset."""

    file_name: str          # relative path, as written in metadata.jsonl
    audio_path: Path        # absolute, resolved against the dataset dir
    text: str               # reference/label passthrough (not used for scoring)


def load_dataset(dataset_dir: Path) -> List[AudioItem]:
    """Read ``metadata.jsonl`` from ``dataset_dir`` and resolve audio paths.

    Each line must contain ``file_name``; ``text`` is optional. Missing audio
    files are logged as warnings but kept so the run records the error per cell.
    """
    metadata_path: Path = dataset_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No {METADATA_FILENAME} found in {dataset_dir}")

    items: List[AudioItem] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row: dict = json.loads(line)
            file_name: str = row.get("file_name", "")
            if not file_name:
                raise ValueError(f"{metadata_path}:{lineno} missing 'file_name'")
            audio_path: Path = (dataset_dir / file_name).resolve()
            if not audio_path.is_file():
                logger.warning("Audio file not found: %s", audio_path)
            items.append(
                AudioItem(
                    file_name=file_name,
                    audio_path=audio_path,
                    text=row.get("text", "") or "",
                )
            )

    logger.info("Loaded %d items from %s", len(items), metadata_path)
    return items


def _dir_labels(dataset_dirs: Sequence[Path]) -> List[str]:
    """Return a short, unique label per dataset dir, preserving input order.

    Prefer the dir's own name; if two dirs share a name, fall back to a path that
    keeps enough parent components to disambiguate, and finally to a positional
    suffix so the labels are always unique.
    """
    names: List[str] = [d.name or str(d) for d in dataset_dirs]
    labels: List[str] = list(names)
    seen: Dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    for i, (d, name) in enumerate(zip(dataset_dirs, names)):
        if seen[name] > 1:
            parent: str = d.parent.name
            labels[i] = f"{parent}/{name}" if parent else f"{name}#{i}"
    # Guarantee uniqueness even if the parent-qualified labels still collide.
    used: Dict[str, int] = {}
    for i, label in enumerate(labels):
        if label in used:
            labels[i] = f"{label}#{i}"
        used[labels[i]] = i
    return labels


def merge_datasets(dataset_dirs: Sequence[Path]) -> List[AudioItem]:
    """Load and concatenate several datasets, disambiguating clashing file_names.

    ``file_name`` is the identity key used for resume dedup, per-file result
    aggregation, and report rows. When the same relative ``file_name`` appears in
    more than one dir, the colliding items are prefixed with a per-dir label
    (e.g. ``data_a/audio/0001.wav``) so each cell stays distinct. ``audio_path``
    is left untouched, so audio still loads from the original location.
    """
    per_dir: List[List[AudioItem]] = [load_dataset(d) for d in dataset_dirs]
    if len(per_dir) <= 1:
        return per_dir[0] if per_dir else []

    counts: Dict[str, int] = {}
    for items in per_dir:
        for item in items:
            counts[item.file_name] = counts.get(item.file_name, 0) + 1

    labels: List[str] = _dir_labels(dataset_dirs)
    merged: List[AudioItem] = []
    relabeled: int = 0
    for label, items in zip(labels, per_dir):
        for item in items:
            if counts[item.file_name] > 1:
                merged.append(replace(item, file_name=f"{label}/{item.file_name}"))
                relabeled += 1
            else:
                merged.append(item)

    if relabeled:
        logger.warning(
            "Disambiguated %d items whose file_name appeared in multiple dirs",
            relabeled,
        )
    return merged
