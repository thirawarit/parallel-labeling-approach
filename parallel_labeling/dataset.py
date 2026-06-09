"""Load the HuggingFace AudioFolder ``metadata.jsonl`` and resolve audio paths."""

import json
from dataclasses import (dataclass)
from pathlib import (Path)
from typing import (List)

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
