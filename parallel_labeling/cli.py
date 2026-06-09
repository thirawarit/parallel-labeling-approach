"""Command-line entrypoint for the parallel labeling tool.

Example:
    python -m parallel_labeling.cli --dataset-dir data/th --config config.yaml \
        --output-dir out
"""

import argparse
import logging
import sys
from pathlib import (Path)
from typing import (List, Optional, Sequence)

from parallel_labeling.config import (Config, load_config)
from parallel_labeling.dataset import (AudioItem, load_dataset)
from parallel_labeling.logging_utils import (configure_logging, get_logger)
from parallel_labeling.report import (write_reports)
from parallel_labeling.runner import (run)

logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """Build the argument parser. CLI flags override config-file values."""
    parser = argparse.ArgumentParser(
        prog="parallel_labeling",
        description="Run Thai Whisper models in parallel and compare transcriptions.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="HuggingFace AudioFolder dir containing metadata.jsonl.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML/JSON config file (model registry, language, sample_rate).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for results.jsonl, results.csv, report.html, summary.json.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override transcription language (default from config: th).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only produce results.jsonl; skip CSV/HTML/summary generation.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides onto a loaded Config."""
    if args.language is not None:
        config.language = args.language
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entrypoint. Returns a process exit code."""
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args(argv)

    log_level: int = getattr(logging, args.log_level)
    configure_logging(log_level)

    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_dir():
        logger.error("Dataset dir does not exist: %s", dataset_dir)
        return 2

    config: Config = load_config(args.config)
    config = _apply_overrides(config, args)
    logger.info("Models: %s", ", ".join(config.model_keys))

    items: List[AudioItem] = load_dataset(dataset_dir)
    if not items:
        logger.error("No items in dataset; nothing to do")
        return 1

    output_dir: Path = args.output_dir
    results_path: Path = run(config, items, output_dir, log_level)

    if not args.no_report:
        csv_path, html_path, summary_path = write_reports(
            results_path, output_dir, config.model_keys
        )
        logger.info("Reports: %s | %s | %s", csv_path, html_path, summary_path)

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
