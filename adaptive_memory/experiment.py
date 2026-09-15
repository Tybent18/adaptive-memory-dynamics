"""Supervised, retention, and continual-learning experiment protocols."""

from __future__ import annotations

import csv
import json
import platform
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import DatasetBundle, load_stage_one, task_subset
from .model import AdaptiveMLP

ProgressCallback = Callable[[str, float], None]


def _train(
    model: AdaptiveMLP,
    data: DatasetBundle,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    phase: str,
    callback: ProgressCallback | None = None,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(data.x_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            losses.append(
                model.train_batch(data.x_train[idx], data.y_train[idx], learning_rate).loss
            )
        row: dict[str, float | int | str] = {
            "phase": phase,
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_accuracy": model.accuracy(data.x_train, data.y_train),
            "test_accuracy": model.accuracy(data.x_test, data.y_test),
            "weight_norm": model.weight_norm(),
            "mean_retention": float(model.usage.mean()),
        }
        rows.append(row)
        if callback:
            callback(f"{phase}: epoch {epoch}/{epochs}", epoch / epochs)
    return rows


def run_single(
    config: ExperimentConfig,
    callback: ProgressCallback | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    data = load_stage_one(config.seed, config.test_size)
    model = AdaptiveMLP(
        data.x_train.shape[1],
        config.hidden_size,
        10,
        config.mechanism,
        config.forgetting_rate,
        config.reinforcement_strength,
        config.activation_threshold,
        config.seed,
    )
    history = _train(
        model,
        data,
        config.epochs,
        config.learning_rate,
        config.batch_size,
        config.seed,
        "supervised",
        callback,
    )
    final_train = model.accuracy(data.x_train, data.y_train)
    final_test = model.accuracy(data.x_test, data.y_test)

    retained = model.clone()
    retention_rows: list[dict[str, object]] = []
    checkpoints = sorted({0, 1, 5, 10, config.idle_steps})
    elapsed = 0
    for checkpoint in checkpoints:
        retained.apply_forgetting(checkpoint - elapsed)
        elapsed = checkpoint
        retention_rows.append(
            {
                "phase": "retention",
                "epoch": checkpoint,
                "loss": "",
                "train_accuracy": retained.accuracy(data.x_train, data.y_train),
                "test_accuracy": retained.accuracy(data.x_test, data.y_test),
                "weight_norm": retained.weight_norm(),
                "mean_retention": float(retained.usage.mean()),
            }
        )

    task_a = task_subset(data, range(0, 5))
    task_b = task_subset(data, range(5, 10))
    continual = AdaptiveMLP(
        data.x_train.shape[1],
        config.hidden_size,
        10,
        config.mechanism,
        config.forgetting_rate,
        config.reinforcement_strength,
        config.activation_threshold,
        config.seed,
    )
    a_history = _train(
        continual,
        task_a,
        config.continual_epochs,
        config.learning_rate,
        config.batch_size,
        config.seed + 101,
        "task_a",
        callback,
    )
    task_a_before = continual.accuracy(task_a.x_test, task_a.y_test)
    b_history = _train(
        continual,
        task_b,
        config.continual_epochs,
        config.learning_rate,
        config.batch_size,
        config.seed + 202,
        "task_b",
        callback,
    )
    task_a_after = continual.accuracy(task_a.x_test, task_a.y_test)
    task_b_after = continual.accuracy(task_b.x_test, task_b.y_test)
    b_accuracies = [float(row["test_accuracy"]) for row in b_history]
    threshold = 0.85 * max(b_accuracies)
    adaptation_epoch = next(
        (int(row["epoch"]) for row in b_history if float(row["test_accuracy"]) >= threshold),
        config.continual_epochs,
    )
    summary: dict[str, object] = {
        "dataset": data.name,
        "mechanism": config.mechanism,
        "seed": config.seed,
        "final_train_accuracy": final_train,
        "final_test_accuracy": final_test,
        "generalization_gap": final_train - final_test,
        "retained_accuracy": float(retention_rows[-1]["test_accuracy"]),
        "retention_loss": final_test - float(retention_rows[-1]["test_accuracy"]),
        "task_a_before": task_a_before,
        "task_a_after": task_a_after,
        "task_b_after": task_b_after,
        "catastrophic_forgetting": max(0.0, task_a_before - task_a_after),
        "adaptation_epoch": adaptation_epoch,
        "runtime_seconds": time.perf_counter() - started,
    }
    full_history: list[dict[str, object]] = [*history, *retention_rows, *a_history, *b_history]
    for row in full_history:
        row.update({"mechanism": config.mechanism, "seed": config.seed})
    return full_history, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(
    output_dir: Path,
    seeds: list[int] | None = None,
    epochs: int = 24,
    continual_epochs: int = 16,
    callback: ProgressCallback | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [7, 21, 42, 84, 101]
    histories: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    total = 3 * len(seeds)
    completed = 0
    for mechanism in ("baseline", "passive", "reinforced"):
        for seed in seeds:
            if stop_check and stop_check():
                break
            config = ExperimentConfig(
                mechanism=mechanism,
                seed=seed,
                epochs=epochs,
                continual_epochs=continual_epochs,
                output_dir=output_dir,
            )
            history, summary = run_single(config)
            histories.extend(history)
            summaries.append(summary)
            completed += 1
            if callback:
                callback(f"completed {mechanism}, seed {seed}", completed / total)
        if stop_check and stop_check():
            break
    _write_csv(output_dir / "raw_metrics.csv", histories)
    _write_csv(output_dir / "run_summary.csv", summaries)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "Stage One pipeline validation",
        "dataset": "sklearn-digits-8x8",
        "seeds": seeds,
        "mechanisms": ["baseline", "passive", "reinforced"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "parameters": {"epochs": epochs, "continual_epochs": continual_epochs},
        "claim_boundary": "Exploratory internal evidence; not a definitive MNIST/CIFAR benchmark.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return histories, summaries
