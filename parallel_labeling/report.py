"""Generate CSV, HTML review report, and aggregate summary from results.jsonl."""

import csv
import html
import json
import statistics
from pathlib import (Path)
from typing import (Any, Dict, List, Tuple)

from parallel_labeling.logging_utils import (get_logger)
from parallel_labeling.metrics import (SCORE_NDIGITS)

logger = get_logger(__name__)

CSV_FILENAME: str = "results.csv"
HTML_FILENAME: str = "report.html"
SUMMARY_FILENAME: str = "summary.json"


def _read_rows(results_path: Path) -> List[dict]:
    """Read all rows from results.jsonl."""
    rows: List[dict] = []
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _pair_label(pair: List[str]) -> str:
    """Render a pair list as a stable 'a__b' label."""
    return "__".join(pair)


def _all_pair_labels(rows: List[dict]) -> List[str]:
    """Collect the set of pair labels seen across all rows, sorted."""
    labels: set = set()
    for row in rows:
        for pm in row.get("comparison", {}).get("pairs", []):
            labels.add(_pair_label(pm["pair"]))
    return sorted(labels)


def write_csv(rows: List[dict], output_dir: Path, model_keys: List[str]) -> Path:
    """Flatten rows into a spreadsheet-friendly CSV."""
    pair_labels: List[str] = _all_pair_labels(rows)
    path: Path = output_dir / CSV_FILENAME

    header: List[str] = ["file_name", "text", "unanimous", "best_model", "best_text"]
    header += [f"hyp_{k}" for k in model_keys]
    header += [f"error_{k}" for k in model_keys]
    for label in pair_labels:
        header += [f"cer_raw_{label}", f"cer_norm_{label}", f"wer_norm_{label}"]
    header += [f"mean_agreement_{k}" for k in model_keys]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            comp: Dict[str, Any] = row.get("comparison", {})
            hyp: Dict[str, Any] = row.get("hypotheses", {})
            err: Dict[str, Any] = row.get("errors", {})
            pair_map: Dict[str, dict] = {_pair_label(p["pair"]): p for p in comp.get("pairs", [])}
            mean_map: Dict[str, Any] = comp.get("mean_agreement_per_model", {})

            record: List[Any] = [
                row.get("file_name", ""),
                row.get("text", ""),
                comp.get("unanimous", ""),
                comp.get("best_model", "") or "",
                comp.get("best_text", "") or "",
            ]
            record += [hyp.get(k, "") if hyp.get(k) is not None else "" for k in model_keys]
            record += [err.get(k, "") for k in model_keys]
            for label in pair_labels:
                pm: dict = pair_map.get(label, {})
                record += [pm.get("cer_raw", ""), pm.get("cer_norm", ""), pm.get("wer_norm", "")]
            record += [mean_map.get(k, "") for k in model_keys]
            writer.writerow(record)

    logger.info("Wrote %s", path)
    return path


def write_summary(rows: List[dict], output_dir: Path, model_keys: List[str]) -> Path:
    """Compute and write the aggregate summary across the dataset."""
    pair_labels: List[str] = _all_pair_labels(rows)
    pair_cer: Dict[str, List[float]] = {p: [] for p in pair_labels}
    pair_wer: Dict[str, List[float]] = {p: [] for p in pair_labels}
    model_agreement: Dict[str, List[float]] = {k: [] for k in model_keys}
    model_errors: Dict[str, int] = {k: 0 for k in model_keys}
    unanimous_count: int = 0

    for row in rows:
        comp: Dict[str, Any] = row.get("comparison", {})
        if comp.get("unanimous"):
            unanimous_count += 1
        for pm in comp.get("pairs", []):
            label: str = _pair_label(pm["pair"])
            pair_cer[label].append(pm["cer_norm"])
            pair_wer[label].append(pm["wer_norm"])
        for key, val in comp.get("mean_agreement_per_model", {}).items():
            model_agreement.setdefault(key, []).append(val)
        for key in row.get("errors", {}):
            model_errors[key] = model_errors.get(key, 0) + 1

    def _mean(values: List[float]) -> float:
        return round(statistics.fmean(values), SCORE_NDIGITS) if values else 0.0

    total: int = len(rows)
    summary: Dict[str, Any] = {
        "total_files": total,
        "unanimous_files": unanimous_count,
        "unanimous_pct": round(unanimous_count / total * 100.0, SCORE_NDIGITS) if total else 0.0,
        "mean_cer_norm_per_pair": {p: _mean(v) for p, v in pair_cer.items()},
        "mean_wer_norm_per_pair": {p: _mean(v) for p, v in pair_wer.items()},
        "mean_agreement_per_model": {k: _mean(v) for k, v in model_agreement.items()},
        "error_counts_per_model": model_errors,
    }

    path: Path = output_dir / SUMMARY_FILENAME
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


_HTML_HEAD: str = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Parallel Labeling Review</title>
<style>
 body { font-family: sans-serif; margin: 1.5rem; }
 table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
 th, td { border: 1px solid #ccc; padding: .4rem .6rem; vertical-align: top; text-align: left; }
 th { background: #f3f3f3; }
 .unanimous { background: #e7f7e7; }
 .disagree { background: #fff3e0; }
 .err { color: #b00020; font-style: italic; }
 .file { font-family: monospace; font-size: .85rem; }
 .best { background: #d8ecff; font-weight: bold; }
 caption { font-weight: bold; text-align: left; margin-bottom: .5rem; }
</style></head><body>
"""


def write_html(rows: List[dict], output_dir: Path, model_keys: List[str]) -> Path:
    """Render a side-by-side human review report with disagreement highlighting."""
    parts: List[str] = [_HTML_HEAD, "<h1>Parallel Labeling Review</h1>"]
    parts.append(f"<p>{len(rows)} files. Models: {', '.join(model_keys)}.</p>")
    parts.append("<table><thead><tr><th>file</th><th>reference</th>")
    for key in model_keys:
        parts.append(f"<th>{html.escape(key)}</th>")
    parts.append("<th>best</th><th>unanimous</th></tr></thead><tbody>")

    for row in rows:
        comp: Dict[str, Any] = row.get("comparison", {})
        hyp: Dict[str, Any] = row.get("hypotheses", {})
        err: Dict[str, Any] = row.get("errors", {})
        unanimous: bool = bool(comp.get("unanimous"))
        best_model: str = comp.get("best_model") or ""
        best_text: str = comp.get("best_text") or ""
        row_cls: str = "unanimous" if unanimous else "disagree"
        parts.append(f'<tr class="{row_cls}">')
        parts.append(f'<td class="file">{html.escape(row.get("file_name", ""))}</td>')
        parts.append(f"<td>{html.escape(row.get('text', '') or '')}</td>")
        for key in model_keys:
            cell_err: str = err.get(key, "")
            cell_cls: str = ' class="best"' if key == best_model else ""
            if cell_err:
                parts.append(f'<td class="err">{html.escape(cell_err)}</td>')
            else:
                parts.append(f"<td{cell_cls}>{html.escape(hyp.get(key) or '')}</td>")
        parts.append(f'<td class="best">{html.escape(best_model)}<br>{html.escape(best_text)}</td>')
        parts.append(f"<td>{'✓' if unanimous else '✗'}</td>")
        parts.append("</tr>")

    parts.append("</tbody></table></body></html>")
    path: Path = output_dir / HTML_FILENAME
    path.write_text("".join(parts), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def write_reports(results_path: Path, output_dir: Path, model_keys: List[str]) -> Tuple[Path, Path, Path]:
    """Generate CSV, HTML, and summary from results.jsonl. Returns their paths."""
    rows: List[dict] = _read_rows(results_path)
    csv_path: Path = write_csv(rows, output_dir, model_keys)
    html_path: Path = write_html(rows, output_dir, model_keys)
    summary_path: Path = write_summary(rows, output_dir, model_keys)
    return csv_path, html_path, summary_path
