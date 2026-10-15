"""Stage Two paired domain-incremental learning benchmark.

Each seed yields ONE stratified train/test split and ONE input transformation,
both shared across all memory mechanisms and hyperparameters. All ten class
labels are retained in both domains; unlike Stage One, this does not confound
forgetting with a change of label subset.

Results are exploratory offline sklearn-digits 8x8 proxy measurements,
NOT MNIST/Fashion-MNIST results or established continual-learning superiority.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .model import AdaptiveMLP

DOMAINS = ("rotate90", "permuted_pixels")
DEFAULT_SEEDS = (7, 21, 42, 84, 101)


@dataclass(frozen=True)
class Condition:
    name: str
    mechanism: str
    rate: float
    strength: float


CONDITIONS = (
    Condition("persistent", "baseline", 0.0, 0.0),
    Condition("passive_002", "passive", 0.002, 0.0),
    Condition("passive_008", "passive", 0.008, 0.0),
    Condition("reinforced_002_4", "reinforced", 0.002, 4.0),
    Condition("reinforced_002_8", "reinforced", 0.002, 8.0),
    Condition("reinforced_008_4", "reinforced", 0.008, 4.0),
    Condition("reinforced_008_8", "reinforced", 0.008, 8.0),
)


@dataclass(frozen=True)
class DomainPair:
    a_train: np.ndarray
    a_test: np.ndarray
    b_train: np.ndarray
    b_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    raw_data_sha256: str


def make_pair(seed: int, domain: str, test_size: float = 0.25) -> DomainPair:
    """Apply the SAME randomized split to both tasks; normalize raw pixels [0, 1]."""
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected {DOMAINS}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)
    digest = hashlib.sha256(digits.data.tobytes() + y.tobytes()).hexdigest()
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), stratify=y, test_size=test_size, random_state=seed
    )
    if domain == "rotate90":
        b = np.rot90(x.reshape(-1, 8, 8), k=1, axes=(1, 2)).reshape(-1, 64).copy()
    else:
        rng = np.random.default_rng(seed + 100_000)
        permutation = rng.permutation(64)
        b = x[:, permutation].copy()
    return DomainPair(
        x[train_idx], x[test_idx], b[train_idx], b[test_idx],
        y[train_idx], y[test_idx], digest
    )


def _train_epoch(
    model: AdaptiveMLP,
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float,
    batch_size: int,
    order: np.ndarray,
) -> float:
    losses = []
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        losses.append(model.train_batch(x[idx], y[idx], learning_rate).loss)
    return float(np.mean(losses))


def run_condition(
    seed: int,
    domain: str,
    condition: Condition,
    *,
    epochs: int = 8,
    hidden_size: int = 32,
    learning_rate: float = 0.08,
    batch_size: int = 64,
    idle_steps: int = 30,
    pair: DomainPair | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if epochs < 1 or hidden_size < 1 or batch_size < 1 or idle_steps < 0:
        raise ValueError("epoch and size settings must be positive; idle >= 0")
    if not 0 < learning_rate:
        raise ValueError("learning_rate must be positive")
    pair = pair if pair is not None else make_pair(seed, domain)
    model = AdaptiveMLP(
        64, hidden_size, 10, mechanism=condition.mechanism,
        forgetting_rate=condition.rate,
        reinforcement_strength=condition.strength,
        seed=seed,
    )
    # The shuffle stream does not depend on the memory mechanism.
    rng = np.random.default_rng(seed + 11_009)
    rows: list[dict[str, object]] = []
    a_before = 0.0
    b_before = 0.0
    for phase, x in (("task_a", pair.a_train), ("task_b", pair.b_train)):
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x))
            loss = _train_epoch(
                model, x, pair.y_train, learning_rate, batch_size, order
            )
            a_accuracy = model.accuracy(pair.a_test, pair.y_test)
            b_accuracy = model.accuracy(pair.b_test, pair.y_test)
            rows.append({
                "seed": seed, "domain": domain, "condition": condition.name,
                "mechanism": condition.mechanism, "rate": condition.rate,
                "strength": condition.strength, "phase": phase, "epoch": epoch,
                "loss": loss, "a_test_accuracy": a_accuracy,
                "b_test_accuracy": b_accuracy,
                "weight_norm": model.weight_norm(),
                "mean_usage": float(model.usage.mean()),
            })
        if phase == "task_a":
            a_before = model.accuracy(pair.a_test, pair.y_test)
            b_before = model.accuracy(pair.b_test, pair.y_test)
    a_after = model.accuracy(pair.a_test, pair.y_test)
    b_after = model.accuracy(pair.b_test, pair.y_test)
    idle = model.clone()
    idle.apply_forgetting(idle_steps)
    a_after_idle = idle.accuracy(pair.a_test, pair.y_test)
    b_after_idle = idle.accuracy(pair.b_test, pair.y_test)
    result: dict[str, object] = {
        "seed": seed, "domain": domain, "condition": condition.name,
        "mechanism": condition.mechanism, "rate": condition.rate,
        "strength": condition.strength, "train_epochs_per_task": epochs,
        "task_a_before_b": a_before, "task_b_before_b": b_before,
        "task_a_after_b": a_after, "task_b_after_b": b_after,
        "signed_forgetting": a_before - a_after,
        "nonnegative_forgetting": max(0.0, a_before - a_after),
        "backward_transfer": a_after - a_before,
        "task_b_learning_gain": b_after - b_before,
        "mean_final_accuracy": (a_after + b_after) / 2.0,
        "a_accuracy_after_idle": a_after_idle,
        "b_accuracy_after_idle": b_after_idle,
        "idle_retention_loss_a": a_after - a_after_idle,
        "idle_retention_loss_b": b_after - b_after_idle,
    }
    return rows, result


def paired_effects(
    summary: list[dict[str, object]], *,
    resamples: int = 2000,
) -> list[dict[str, object]]:
    """Paired seed-level differences from persistent, per domain and metric."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    metrics = ("task_a_after_b", "task_b_after_b",
               "nonnegative_forgetting", "mean_final_accuracy",
               "idle_retention_loss_a")
    by = {(str(r["domain"]), int(r["seed"]), str(r["condition"])): r
          for r in summary}
    effects: list[dict[str, object]] = []
    for domain in DOMAINS:
        seeds = sorted({
            int(r["seed"]) for r in summary if r["domain"] == domain
        })
        for condition in CONDITIONS[1:]:
            for metric in metrics:
                eligible = [seed for seed in seeds
                            if (domain, seed, "persistent") in by
                            and (domain, seed, condition.name) in by]
                delta = np.array([
                    float(by[domain, seed, condition.name][metric])
                    - float(by[domain, seed, "persistent"][metric])
                    for seed in eligible
                ])
                if len(delta) == 0:
                    continue
                lo: float | None = None
                hi: float | None = None
                if len(delta) >= 2:
                    rng = np.random.default_rng(8_000 + len(eligible))
                    boot = delta[rng.integers(len(delta), size=(resamples, len(delta)))]
                    lo, hi = (float(x) for x in np.percentile(
                        boot.mean(axis=1), [2.5, 97.5]
                    ))
                effects.append({
                    "domain": domain, "condition": condition.name,
                    "metric": metric, "n_paired_seeds": len(delta),
                    "mean_difference_vs_persistent": float(delta.mean()),
                    "ci95_lower": lo if lo is not None else "",
                    "ci95_upper": hi if hi is not None else "",
                    "ci_interpretation": "exploratory_unadjusted" if lo is not None
                    else "insufficient_seeds",
                })
    return effects


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figures(output: Path, summaries: list[dict[str, object]]) -> None:
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    for domain in DOMAINS:
        selected = [row for row in summaries if row["domain"] == domain]
        if not selected:
            continue
        labels = [c.name for c in CONDITIONS]
        a = [float(np.mean([
            float(row["task_a_after_b"]) for row in selected
            if row["condition"] == name
        ])) for name in labels]
        b = [float(np.mean([
            float(row["task_b_after_b"]) for row in selected
            if row["condition"] == name
        ])) for name in labels]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        pos = np.arange(len(labels))
        ax.bar(pos - .19, a, .38, label="A retained after B")
        ax.bar(pos + .19, b, .38, label="B learned after A")
        ax.set_xticks(pos, labels, rotation=28, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Held-out accuracy (0–1)")
        ax.set_title(f"Stage Two exploratory domain shift: {domain}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / f"{domain}_paired_accuracy.png", dpi=140)
        plt.close(fig)


def collect(
    output: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    domains: tuple[str, ...] = DOMAINS,
    epochs: int = 8,
    hidden_size: int = 32,
    batch_size: int = 64,
    learning_rate: float = 0.08,
    idle_steps: int = 30,
) -> dict[str, object]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    if not domains or any(d not in DOMAINS for d in domains):
        raise ValueError("domains must be nonempty and known")
    output.mkdir(parents=True, exist_ok=True)
    histories: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    hashes: dict[str, str] = {}
    for domain in domains:
        for seed in seeds:
            pair = make_pair(seed, domain)
            hashes[f"{domain}:{seed}"] = pair.raw_data_sha256
            for condition in CONDITIONS:
                rows, result = run_condition(
                    seed, domain, condition, epochs=epochs,
                    hidden_size=hidden_size, batch_size=batch_size,
                    learning_rate=learning_rate, idle_steps=idle_steps,
                    pair=pair,
                )
                histories.extend(rows)
                summaries.append(result)
    effects = paired_effects(summaries)
    generated = {
        "epoch_metrics.csv": histories,
        "run_summary.csv": summaries,
        "paired_effects.csv": effects,
    }
    for filename, records in generated.items():
        _csv(output / filename, records)
    _figures(output, summaries)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest: dict[str, object] = {
        "experiment": "stage_two_domain_shift_digits_8x8",
        "status": "exploratory_offline_proxy_not_confirmatory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "dataset": "scikit-learn load_digits, 8x8 handwritten digits; no MNIST",
        "dataset_sha256_by_domain_seed": hashes,
        "mechanisms": [asdict(condition) for condition in CONDITIONS],
        "seeds": list(seeds), "domains": list(domains),
        "model": {"input": 64, "hidden": hidden_size, "output": 10,
                  "hidden_activation": "relu", "optimizer": "numpy_sgd",
                  "learning_rate": learning_rate, "batch_size": batch_size},
        "epochs_per_task": epochs, "idle_steps": idle_steps,
        "randomization": "paired split, input mapping, weight initialization and "
                         "minibatch orders by seed within each domain",
        "transforms": {
            "task_a": "original raw digits pixels normalized to [0,1]",
            "rotate90": "numpy.rot90 k=1 on 8x8 images; same label",
            "permuted_pixels": "fixed seed-specific 64-feature permutation; same label",
        },
        "primary_metrics": ["task_a_after_b", "task_b_after_b"],
        "secondary_metrics": ["nonnegative_forgetting", "backward_transfer",
                              "mean_final_accuracy", "idle_retention_loss_a"],
        "statistics": "paired bootstrap seed-level 95% intervals when n>=2; "
                      "exploratory unadjusted multiple comparisons",
        "trials": len(summaries), "epochs_recorded": len(histories),
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "sklearn": sklearn.__version__},
        "limitations": [
            "Only 8x8 scikit-learn digits, not MNIST, Fashion-MNIST or CIFAR",
            "Synthetic deterministic domain transformations, not organic concept drift",
            "One-hidden-layer NumPy reference MLP, not PyTorch benchmark",
            "No rehearsal, EWC or strong continual-learning competitive baselines",
            "No task-order reversal, no class-incremental control in this suite",
            "Bootstraps are descriptive and not multiplicity-corrected",
            "No findings of improved forgetting without reviewing the actual results",
        ],
    }
    manifest["checksums_sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in generated
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/stage-two"))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--pilot", action="store_true",
                        help="short one-seed, two-domain, two-epoch pipeline check")
    args = parser.parse_args()
    seeds = (7,) if args.pilot else tuple(int(s) for s in args.seeds.split(","))
    epochs = 2 if args.pilot else args.epochs
    domains = tuple(args.domains.split(","))
    manifest = collect(args.output, seeds=seeds, domains=domains, epochs=epochs)
    print("Stage Two exploratory evidence:", manifest["trials"],
          "paired condition runs;", manifest["epochs_recorded"], "epoch rows")
    print("Data and figures:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
