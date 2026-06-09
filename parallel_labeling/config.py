"""Configuration loading: YAML/JSON config file + model registry.

CLI flags override values loaded from the config file (merge handled in ``cli``).
"""

import json
from dataclasses import (dataclass, field)
from pathlib import (Path)
from typing import (Any, Dict, Optional)

import yaml

from parallel_labeling.logging_utils import (get_logger)

logger = get_logger(__name__)

DEFAULT_MODELS: Dict[str, Dict[str, Any]] = {
    "pathumma": {"model_id": "nectec/Pathumma-whisper-th-large-v3", "device": "cuda:0"},
    "typhoon": {"model_id": "typhoon-ai/typhoon-whisper-large-v3", "device": "cuda:1"},
    "distill": {"model_id": "biodatlab/distill-whisper-th-large-v3", "device": "cuda:2"},
}


@dataclass
class ModelConfig:
    """One entry in the model registry."""

    key: str
    model_id: str
    device: str
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Resolved run configuration."""

    models: Dict[str, ModelConfig]
    language: str = "th"
    task: str = "transcribe"
    sample_rate: int = 16000

    @property
    def model_keys(self) -> list:
        """Stable, sorted list of model keys (deterministic pair ordering)."""
        return sorted(self.models.keys())


def _read_config_file(path: Path) -> Dict[str, Any]:
    """Read a YAML or JSON config file into a dict."""
    text: str = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data: Any = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}: {path}")
    return data


def load_config(config_path: Optional[Path]) -> Config:
    """Load configuration from a file, falling back to built-in defaults.

    The model registry merges defaults with any ``models`` section in the file so
    a partial config still yields all three target models.
    """
    raw: Dict[str, Any] = {}
    if config_path is not None:
        raw = _read_config_file(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        logger.info("No config file given; using built-in defaults")

    raw_models: Dict[str, Any] = raw.get("models") or {}
    merged_keys = set(DEFAULT_MODELS) | set(raw_models)
    models: Dict[str, ModelConfig] = {}
    for key in merged_keys:
        base: Dict[str, Any] = dict(DEFAULT_MODELS.get(key, {}))
        base.update(raw_models.get(key, {}))
        if "model_id" not in base:
            raise ValueError(f"Model '{key}' is missing a model_id in config")
        models[key] = ModelConfig(
            key=key,
            model_id=base["model_id"],
            device=base.get("device", "cpu"),
            generate_kwargs=dict(base.get("generate_kwargs", {})),
        )

    return Config(
        models=models,
        language=raw.get("language", "th"),
        task=raw.get("task", "transcribe"),
        sample_rate=int(raw.get("sample_rate", 16000)),
    )
