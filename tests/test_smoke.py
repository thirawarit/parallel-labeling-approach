"""Synthetic end-to-end smoke test for the parallel labeling pipeline.

Exercises dataset load -> runner (parallel workers) -> metrics -> reports without
any real ML dependency. The heavy libs are stubbed (see tests/stubs.py) and the
model pipeline is replaced with a deterministic FakeASR.

Run directly:  python -m tests.test_smoke
Or via pytest: pytest tests/test_smoke.py
"""

import json
import queue
import threading
import wave
from pathlib import (Path)
from tempfile import (TemporaryDirectory)
from typing import (Any, Callable, Dict, List)

from tests.stubs import (FakeASR, install_all)

# Stubs must be installed before importing the package modules.
install_all()

from parallel_labeling import models as models_mod  # noqa: E402
from parallel_labeling import runner as runner_mod  # noqa: E402
from parallel_labeling.config import (Config, ModelConfig)  # noqa: E402
from parallel_labeling.dataset import (load_dataset)  # noqa: E402
from parallel_labeling.report import (write_reports)  # noqa: E402
from parallel_labeling.runner import (run)  # noqa: E402


# --- Test fixtures ---------------------------------------------------------

# Per-model, per-file outputs keyed by audio stem.
# a & b agree on 0001 (unanimous), c disagrees on 0002, c fails on 0003.
MODEL_OUTPUTS: Dict[str, Dict[str, str]] = {
    "alpha": {"0001": "สวัสดี ครับ", "0002": "ฝน ตก", "0003": "กิน ข้าว"},
    "beta": {"0001": "สวัสดี ครับ", "0002": "ฝน ตก", "0003": "กิน ข้าว"},
    "gamma": {"0001": "สวัสดี ครับ", "0002": "ฝน ตก หนัก", "0003": "UNUSED"},
}
FAIL_ON: Dict[str, List[str]] = {"alpha": [], "beta": [], "gamma": ["0003"]}


def _write_silent_wav(path: Path) -> None:
    """Write a tiny valid mono 16kHz wav so the path exists and is readable."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)


def _make_dataset(root: Path) -> None:
    """Create an AudioFolder-layout dataset with metadata.jsonl + wavs."""
    audio_dir: Path = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for stem in ("0001", "0002", "0003"):
        _write_silent_wav(audio_dir / f"{stem}.wav")
        lines.append(json.dumps({"file_name": f"audio/{stem}.wav", "text": ""}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_config() -> Config:
    """Three synthetic models, all on CPU."""
    models: Dict[str, ModelConfig] = {
        key: ModelConfig(key=key, model_id=f"fake/{key}", device="cpu")
        for key in ("alpha", "beta", "gamma")
    }
    return Config(models=models, language="th", task="transcribe", sample_rate=16000)


# --- Thread-backed multiprocessing shim ------------------------------------
# spawn re-imports modules in child processes, which would lose our stubs and
# FakeASR patch. Run workers as threads instead so the test stays hermetic.

class _ThreadContext:
    """Minimal drop-in for an mp context using threads + a thread-safe queue."""

    @staticmethod
    def Queue() -> "queue.Queue[Any]":
        return queue.Queue()

    @staticmethod
    def Process(target: Callable[..., Any], args: tuple, name: str, daemon: bool) -> threading.Thread:
        return threading.Thread(target=target, args=args, name=name, daemon=daemon)


def _patch_pipeline() -> None:
    """Replace _build_pipeline with a FakeASR selected by model key."""
    def fake_build(model_cfg: ModelConfig, config: Config) -> FakeASR:
        return FakeASR(MODEL_OUTPUTS[model_cfg.key], FAIL_ON[model_cfg.key])

    models_mod._build_pipeline = fake_build  # type: ignore[assignment]


def run_smoke_test() -> None:
    """Drive the full pipeline and assert the comparison/report invariants."""
    _patch_pipeline()
    runner_mod.mp.get_context = lambda method=None: _ThreadContext()  # type: ignore[assignment]

    with TemporaryDirectory() as tmp:
        root: Path = Path(tmp)
        dataset_dir: Path = root / "ds"
        output_dir: Path = root / "out"
        _make_dataset(dataset_dir)

        config: Config = _make_config()
        items = load_dataset(dataset_dir)
        assert len(items) == 3, f"expected 3 items, got {len(items)}"

        import logging
        results_path: Path = run(config, items, output_dir, logging.WARNING)
        assert results_path.is_file(), "results.jsonl was not written"

        rows: List[dict] = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
        by_name: Dict[str, dict] = {r["file_name"]: r for r in rows}
        assert len(rows) == 3, f"expected 3 result rows, got {len(rows)}"

        # 0001: all three agree -> unanimous.
        r1 = by_name["audio/0001.wav"]
        assert r1["comparison"]["unanimous"] is True, "0001 should be unanimous"

        # 0002: gamma disagrees -> not unanimous, all models present.
        r2 = by_name["audio/0002.wav"]
        assert r2["comparison"]["unanimous"] is False, "0002 should not be unanimous"
        assert r2["errors"] == {}, "0002 should have no errors"

        # 0003: gamma fails -> error recorded, hypothesis None, not unanimous.
        r3 = by_name["audio/0003.wav"]
        assert "gamma" in r3["errors"], "0003 should record gamma error"
        assert r3["hypotheses"]["gamma"] is None, "failed model hypothesis must be None"
        assert r3["comparison"]["unanimous"] is False, "0003 cannot be unanimous"

        # Reports generate without error and produce all four artifacts.
        csv_path, html_path, summary_path = write_reports(
            results_path, output_dir, config.model_keys
        )
        for artifact in (csv_path, html_path, summary_path):
            assert artifact.is_file(), f"missing report artifact: {artifact}"

        summary: dict = json.loads(summary_path.read_text())
        assert summary["total_files"] == 3
        assert summary["unanimous_files"] == 1
        assert summary["error_counts_per_model"].get("gamma", 0) == 1

        # Resume: rerunning should add no new rows.
        run(config, items, output_dir, logging.WARNING)
        rows_after: List[dict] = [
            json.loads(l) for l in results_path.read_text().splitlines() if l.strip()
        ]
        assert len(rows_after) == 3, "resume should not duplicate rows"

    print("SMOKE TEST PASSED")


def test_smoke() -> None:
    """pytest entrypoint."""
    run_smoke_test()


if __name__ == "__main__":
    run_smoke_test()
