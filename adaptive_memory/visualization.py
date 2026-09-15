"""Publication-ready Stage One charts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"baseline": "#8fa6b8", "passive": "#f0b35a", "reinforced": "#35d0a4"}


def create_charts(
    histories: list[dict[str, object]], summaries: list[dict[str, object]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in histories:
        grouped[str(row["mechanism"])].append(row)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#08111f")
    ax.set_facecolor("#0d1929")
    for mechanism, rows in grouped.items():
        supervised = [r for r in rows if r["phase"] == "supervised"]
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for row in supervised:
            by_epoch[int(row["epoch"])].append(float(row["test_accuracy"]))
        epochs = sorted(by_epoch)
        values = [np.mean(by_epoch[e]) for e in epochs]
        ax.plot(epochs, values, lw=2.4, label=mechanism.title(), color=COLORS[mechanism])
    ax.set(title="Stage One Test Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax.grid(alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    accuracy = output_dir / "accuracy_curves.png"
    fig.savefig(accuracy, dpi=180)
    plt.close(fig)

    metrics = ["final_test_accuracy", "retained_accuracy", "task_b_after"]
    labels = ["Final test", "After idle decay", "Task B"]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#08111f")
    ax.set_facecolor("#0d1929")
    x = np.arange(len(metrics))
    width = 0.24
    for i, mechanism in enumerate(COLORS):
        subset = [r for r in summaries if r["mechanism"] == mechanism]
        means = [np.mean([float(r[m]) for r in subset]) for m in metrics]
        ax.bar(x + (i - 1) * width, means, width, label=mechanism.title(), color=COLORS[mechanism])
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Memory Mechanism Comparison")
    ax.grid(axis="y", alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    comparison = output_dir / "mechanism_comparison.png"
    fig.savefig(comparison, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#08111f")
    ax.set_facecolor("#0d1929")
    mechanisms = list(COLORS)
    means = [
        np.mean([float(r["catastrophic_forgetting"]) for r in summaries if r["mechanism"] == m])
        for m in mechanisms
    ]
    ax.bar([m.title() for m in mechanisms], means, color=[COLORS[m] for m in mechanisms])
    ax.set(title="Catastrophic Forgetting After Task B", ylabel="Task A accuracy lost")
    ax.grid(axis="y", alpha=0.12)
    fig.tight_layout()
    forgetting = output_dir / "continual_forgetting.png"
    fig.savefig(forgetting, dpi=180)
    plt.close(fig)
    return [accuracy, comparison, forgetting]
