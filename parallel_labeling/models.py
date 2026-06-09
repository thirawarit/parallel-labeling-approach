"""Model loading and transcription.

When models occupy separate GPUs, each runs in its own process (one per model)
via ``run_model_worker``, streaming results back through a queue. When models
share a device, the runner instead drives ``transcribe_items`` directly in a
single process (see ``runner._run_sequential``). Both paths share the same
load helper and per-file error handling.
"""

import logging
from dataclasses import (dataclass)
from typing import (Any, Dict, Iterator, List, Optional)

from parallel_labeling.config import (Config, ModelConfig)
from parallel_labeling.dataset import (AudioItem)
from parallel_labeling.logging_utils import (configure_logging, get_logger)

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """One model's output for one audio file."""

    model_key: str
    file_name: str
    text: Optional[str]      # None when transcription failed
    error: Optional[str]     # error message when failed, else None


def _resolve_device(requested: str) -> str:
    """Return the usable device, falling back to CPU when CUDA is unavailable."""
    import torch

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA unavailable; %s falls back to CPU", requested)
            return "cpu"
        index: int = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            logger.warning(
                "Device %s out of range (have %d); falling back to CPU",
                requested,
                torch.cuda.device_count(),
            )
            return "cpu"
    return requested


def _build_pipeline(model_cfg: ModelConfig, config: Config) -> Any:
    """Construct a transformers ASR pipeline pinned to the resolved device."""
    import torch
    from transformers import (pipeline)

    device: str = _resolve_device(model_cfg.device)
    torch_dtype: Any = torch.float16 if device.startswith("cuda") else torch.float32
    logger.info("Loading %s (%s) on %s", model_cfg.key, model_cfg.model_id, device)

    return pipeline(
        task="automatic-speech-recognition",
        model=model_cfg.model_id,
        device=device,
        torch_dtype=torch_dtype,
    )


def load_pipeline(model_cfg: ModelConfig, config: Config) -> Any:
    """Load an ASR pipeline for one model. Thin wrapper over ``_build_pipeline``."""
    return _build_pipeline(model_cfg, config)


def transcribe_items(
    asr: Any,
    model_cfg: ModelConfig,
    config: Config,
    items: List[AudioItem],
) -> Iterator[TranscriptionResult]:
    """Transcribe every item with a loaded pipeline, yielding one result each.

    Per-file failures are caught and reported as a ``TranscriptionResult`` with an
    ``error`` so the run never aborts. Used by both the subprocess worker and the
    single-process sequential path.
    """
    generate_kwargs: Dict[str, Any] = {
        "language": config.language,
        "task": config.task,
        **model_cfg.generate_kwargs,
    }
    for item in items:
        try:
            output: Dict[str, Any] = asr(
                str(item.audio_path),
                generate_kwargs=generate_kwargs,
            )
            text: str = (output.get("text") or "").strip()
            yield TranscriptionResult(
                model_key=model_cfg.key,
                file_name=item.file_name,
                text=text,
                error=None,
            )
        except Exception as exc:
            logger.warning("%s failed on %s: %s", model_cfg.key, item.file_name, exc)
            yield TranscriptionResult(
                model_key=model_cfg.key,
                file_name=item.file_name,
                text=None,
                error=str(exc),
            )


def run_model_worker(
    model_cfg: ModelConfig,
    config: Config,
    items: List[AudioItem],
    result_queue: "Any",
    log_level: int = logging.INFO,
) -> None:
    """Worker entrypoint: load the model, transcribe every item, enqueue results.

    Runs in a child process. A final ``None`` sentinel is enqueued to signal this
    worker is done. On load failure, every item is reported with a load error.
    """
    configure_logging(log_level)

    try:
        asr = load_pipeline(model_cfg, config)
    except Exception as exc:  # model failed to load -> every file errors out
        logger.exception("Failed to load model %s", model_cfg.key)
        for item in items:
            result_queue.put(
                TranscriptionResult(
                    model_key=model_cfg.key,
                    file_name=item.file_name,
                    text=None,
                    error=f"model_load_failed: {exc}",
                )
            )
        result_queue.put(None)
        return

    for result in transcribe_items(asr, model_cfg, config, items):
        result_queue.put(result)

    result_queue.put(None)
