"""Orchestrate model execution, stream results to JSONL, support resume.

Two execution modes, chosen automatically by whether models share a device:

- **Multi-process** (devices differ, e.g. cuda:0/1/2): one process per model
  transcribes the full file list concurrently; results arrive out of order via a
  shared queue.
- **Sequential single-process** (two or more models resolve to the same device,
  e.g. all cuda:0): models are loaded together in one process and run one after
  another per file, avoiding GPU contention.

Both modes collect all model outputs per ``file_name``, compute pairwise metrics,
and append the same row schema to ``results.jsonl`` immediately (crash-safe). On
rerun, file_names already present in the JSONL are skipped.
"""

import json
import multiprocessing as mp
from multiprocessing.process import (BaseProcess)
from pathlib import (Path)
from typing import (Any, Dict, List, Optional, Set, TextIO)

from parallel_labeling.config import (Config)
from parallel_labeling.dataset import (AudioItem)
from parallel_labeling.logging_utils import (get_logger)
from parallel_labeling.metrics import (FileComparison, PairMetrics, compare_hypotheses)
from parallel_labeling.models import (
    TranscriptionResult,
    _resolve_device,
    load_pipeline,
    run_model_worker,
    transcribe_items,
)

logger = get_logger(__name__)

RESULTS_FILENAME: str = "results.jsonl"


def _load_done_file_names(results_path: Path) -> Set[str]:
    """Return the set of file_names already recorded in an existing JSONL."""
    done: Set[str] = set()
    if not results_path.is_file():
        return done
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row: dict = json.loads(line)
            except json.JSONDecodeError:
                continue
            name: Optional[str] = row.get("file_name")
            if name:
                done.add(name)
    return done


def _comparison_to_dict(comparison: FileComparison) -> dict:
    """Serialize a FileComparison into JSON-friendly structures."""
    pairs: List[dict] = []
    pm: PairMetrics
    for pm in comparison.pairs:
        pairs.append(
            {
                "pair": list(pm.pair),
                "cer_raw": pm.cer_raw,
                "cer_norm": pm.cer_norm,
                "wer_norm": pm.wer_norm,
            }
        )
    return {
        "pairs": pairs,
        "mean_agreement_per_model": comparison.mean_agreement_per_model,
        "unanimous": comparison.unanimous,
        "best_model": comparison.best_model,
        "best_text": comparison.best_text,
    }


def _write_row(
    out: TextIO,
    file_name: str,
    text: str,
    bucket: Dict[str, Optional[str]],
    errors: Dict[str, str],
    model_keys: List[str],
) -> None:
    """Compute the comparison and append one result row to the open JSONL file."""
    comparison: FileComparison = compare_hypotheses(bucket, model_keys)
    row: dict = {
        "file_name": file_name,
        "text": text,
        "hypotheses": bucket,
        "errors": errors,
        "comparison": _comparison_to_dict(comparison),
    }
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    out.flush()
    logger.info("Wrote %s (unanimous=%s)", file_name, comparison.unanimous)


def _devices_coincide(config: Config) -> bool:
    """True if two or more models resolve to the same device (shared-card run)."""
    resolved: List[str] = [_resolve_device(config.models[k].device) for k in config.model_keys]
    return len(set(resolved)) < len(resolved)


def _run_multiprocess(
    config: Config,
    pending: List[AudioItem],
    results_path: Path,
    log_level: int,
) -> None:
    """One process per model, results merged per file via a shared queue."""
    model_keys: List[str] = config.model_keys
    text_by_name: Dict[str, str] = {it.file_name: it.text for it in pending}

    ctx = mp.get_context("spawn")
    result_queue: "mp.Queue" = ctx.Queue()
    processes: List[BaseProcess] = []
    for key in model_keys:
        proc = ctx.Process(
            target=run_model_worker,
            args=(config.models[key], config, pending, result_queue, log_level),
            name=f"worker-{key}",
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    # Collect per-file model outputs until each file has all models reported.
    expected_per_file: int = len(model_keys)
    collected: Dict[str, Dict[str, Optional[str]]] = {}
    errors: Dict[str, Dict[str, str]] = {}
    finished_workers: int = 0

    with results_path.open("a", encoding="utf-8") as out:
        while finished_workers < len(processes):
            msg = result_queue.get()
            if msg is None:
                finished_workers += 1
                continue
            res: TranscriptionResult = msg
            bucket: Dict[str, Optional[str]] = collected.setdefault(res.file_name, {})
            bucket[res.model_key] = res.text
            if res.error is not None:
                errors.setdefault(res.file_name, {})[res.model_key] = res.error

            if len(bucket) == expected_per_file:
                _write_row(
                    out,
                    res.file_name,
                    text_by_name.get(res.file_name, ""),
                    bucket,
                    errors.get(res.file_name, {}),
                    model_keys,
                )
                # Free memory for completed files.
                collected.pop(res.file_name, None)
                errors.pop(res.file_name, None)

    for proc in processes:
        proc.join()


def _run_sequential(
    config: Config,
    pending: List[AudioItem],
    results_path: Path,
) -> None:
    """Single process: load all models together, run them one-by-one per file.

    Used when models share a device. A model that fails to load is recorded as a
    ``model_load_failed`` error for every file (mirroring the subprocess worker)
    and the remaining models still run.
    """
    model_keys: List[str] = config.model_keys

    pipelines: Dict[str, Any] = {}
    load_errors: Dict[str, str] = {}
    for key in model_keys:
        try:
            pipelines[key] = load_pipeline(config.models[key], config)
        except Exception as exc:
            logger.exception("Failed to load model %s", key)
            load_errors[key] = f"model_load_failed: {exc}"

    with results_path.open("a", encoding="utf-8") as out:
        for item in pending:
            bucket: Dict[str, Optional[str]] = {}
            errors: Dict[str, str] = {}
            for key in model_keys:
                if key in load_errors:
                    bucket[key] = None
                    errors[key] = load_errors[key]
                    continue
                # transcribe_items handles per-file errors; one item at a time.
                result: TranscriptionResult = next(
                    transcribe_items(pipelines[key], config.models[key], config, [item])
                )
                bucket[key] = result.text
                if result.error is not None:
                    errors[key] = result.error
            _write_row(out, item.file_name, item.text, bucket, errors, model_keys)


def run(
    config: Config,
    items: List[AudioItem],
    output_dir: Path,
    log_level: int,
) -> Path:
    """Run all models and write ``results.jsonl``. Returns its path.

    Chooses the sequential single-process path when models share a device,
    otherwise the multi-process path. Skips items already present (resume).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path: Path = output_dir / RESULTS_FILENAME

    done: Set[str] = _load_done_file_names(results_path)
    pending: List[AudioItem] = [it for it in items if it.file_name not in done]
    if done:
        logger.info("Resume: %d already done, %d pending", len(done), len(pending))
    if not pending:
        logger.info("Nothing to do; all items already processed")
        return results_path

    if _devices_coincide(config):
        logger.info("Models share a device; running sequentially in one process")
        _run_sequential(config, pending, results_path)
    else:
        logger.info("Models on distinct devices; running one process per model")
        _run_multiprocess(config, pending, results_path, log_level)

    logger.info("All models finished; results at %s", results_path)
    return results_path
