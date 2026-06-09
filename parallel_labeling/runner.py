"""Orchestrate parallel model workers, stream results to JSONL, support resume.

One process per model transcribes the full file list concurrently. Results arrive
out of order via a shared queue; the runner collects all model outputs per
``file_name``, computes pairwise metrics once every model has reported, and appends
the row to ``results.jsonl`` immediately (crash-safe). On rerun, file_names already
present in the JSONL are skipped.
"""

import json
import multiprocessing as mp
from multiprocessing.process import (BaseProcess)
from pathlib import (Path)
from typing import (Dict, List, Optional, Set)

from parallel_labeling.config import (Config)
from parallel_labeling.dataset import (AudioItem)
from parallel_labeling.logging_utils import (get_logger)
from parallel_labeling.metrics import (FileComparison, PairMetrics, compare_hypotheses)
from parallel_labeling.models import (TranscriptionResult, run_model_worker)

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
    }


def run(
    config: Config,
    items: List[AudioItem],
    output_dir: Path,
    log_level: int,
) -> Path:
    """Run all model workers and write ``results.jsonl``. Returns its path.

    Skips items already present in an existing results file (resume).
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
                comparison: FileComparison = compare_hypotheses(bucket, model_keys)
                row: dict = {
                    "file_name": res.file_name,
                    "text": text_by_name.get(res.file_name, ""),
                    "hypotheses": bucket,
                    "errors": errors.get(res.file_name, {}),
                    "comparison": _comparison_to_dict(comparison),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                logger.info(
                    "Wrote %s (unanimous=%s)", res.file_name, comparison.unanimous
                )
                # Free memory for completed files.
                collected.pop(res.file_name, None)
                errors.pop(res.file_name, None)

    for proc in processes:
        proc.join()

    logger.info("All workers finished; results at %s", results_path)
    return results_path
