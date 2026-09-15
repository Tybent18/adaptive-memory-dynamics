import csv
import json

from adaptive_memory.config import ExperimentConfig
from adaptive_memory.experiment import run_single, run_suite


def test_single_run_produces_all_protocols(tmp_path) -> None:
    config = ExperimentConfig(
        mechanism="reinforced",
        seed=7,
        epochs=2,
        continual_epochs=2,
        hidden_size=12,
        idle_steps=3,
        output_dir=tmp_path,
    )
    history, summary = run_single(config)
    assert {row["phase"] for row in history} == {"supervised", "retention", "task_a", "task_b"}
    assert 0 <= summary["final_test_accuracy"] <= 1
    assert summary["catastrophic_forgetting"] >= 0


def test_suite_exports_machine_readable_evidence(tmp_path) -> None:
    _, summaries = run_suite(tmp_path, seeds=[7], epochs=1, continual_epochs=1)
    assert len(summaries) == 3
    assert len(list(csv.DictReader((tmp_path / "run_summary.csv").open()))) == 3
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["stage"] == "Stage One pipeline validation"
