"""Stage Two follow-up: matched selective versus uniform pathway consolidation.

Does preserving Task A pathways according to usage or gradient importance offer
a better A/B stability-plasticity tradeoff than uniform anchoring or no anchoring?

Use NEW held-out seeds. Original Stage Two experiment is never overwritten.
All evidence is exploratory: no claims of research superiority without review.
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

from .model import AdaptiveMLP
from .stage_two import DOMAINS, _train_epoch, make_pair

SEEDS = (1109, 1201, 1301, 1409, 1501)
PARAMETERS = ("w1", "b1", "w2", "b2")


@dataclass(frozen=True)
class Treatment:
    name: str
    importance: str
    penalty: float


TREATMENTS = (
    Treatment("persistent", "none", 0.0),
    Treatment("uniform_0p1", "uniform", 0.1),
    Treatment("uniform_0p5", "uniform", 0.5),
    Treatment("activity_0p1", "activity", 0.1),
    Treatment("activity_0p5", "activity", 0.5),
    Treatment("gradient_0p1", "gradient", 0.1),
    Treatment("gradient_0p5", "gradient", 0.5),
    Treatment("shuffled_activity_0p5", "shuffled_activity", 0.5),
)


def gradient_proxy(model: AdaptiveMLP, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Squared *minibatch* cross-entropy gradients, not empirical Fisher."""
    accum = {name: np.zeros_like(getattr(model, name)) for name in PARAMETERS}
    chunks = 0
    for ix in range(0, len(x), 64):
        bx, by = x[ix:ix + 64], y[ix:ix + 64]
        probs, h = model.forward(bx)
        glog = probs.copy()
        glog[np.arange(len(by)), by] -= 1.0
        glog /= len(by)
        gh = (glog @ model.w2.T) * (h > 0)
        grads = {
            "w2": h.T @ glog, "b2": glog.sum(axis=0),
            "w1": bx.T @ gh, "b1": gh.sum(axis=0),
        }
        for name in PARAMETERS:
            accum[name] += np.square(grads[name])
        chunks += 1
    return {name: array / max(1, chunks) for name, array in accum.items()}


def importance_masks(
    model: AdaptiveMLP, strategy: str,
    *,
    x_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    if strategy not in {"none", "uniform", "activity", "gradient", "shuffled_activity"}:
        raise ValueError(f"Unknown importance strategy: {strategy}")
    if strategy in ("none", "uniform"):
        value = 0.0 if strategy == "none" else 1.0
        return {key: np.full_like(getattr(model, key), value) for key in PARAMETERS}
    if strategy == "gradient":
        if x_train is None or y_train is None:
            raise ValueError("Gradient importance requires Task A TRAIN data")
        raw = gradient_proxy(model, x_train, y_train)
    else:
        usage = model.usage.copy()
        if strategy == "shuffled_activity":
            usage = np.random.default_rng(seed + 40_031).permutation(usage)
        # B output biases are not attached to a hidden pathway.
        raw = {
            "w1": np.broadcast_to(usage, model.w1.shape).copy(),
            "b1": usage.copy(),
            "w2": np.broadcast_to(usage[:, None], model.w2.shape).copy(),
            "b2": np.zeros_like(model.b2),
        }
    result = {}
    for name, arr in raw.items():
        avg = float(np.mean(arr))
        result[name] = (
            np.clip(arr / avg, 0.0, 3.0)
            if avg > 1e-12 else np.zeros_like(arr)
        )
    return result


def apply_anchor(
    model: AdaptiveMLP,
    reference: dict[str, np.ndarray],
    importance: dict[str, np.ndarray],
    *,
    penalty: float,
    learning_rate: float,
) -> None:
    """One SGD gradient step of quadratic anchoring after each task-B batch."""
    if penalty < 0 or learning_rate <= 0:
        raise ValueError("Penalty must be >= 0 and learning rate positive")
    for name in PARAMETERS:
        values = getattr(model, name)
        values -= learning_rate * penalty * importance[name] * (values - reference[name])


def evaluate(
    seed: int,
    domain: str,
    *,
    epochs: int = 8,
    hidden_size: int = 32,
    learning_rate: float = 0.08,
    batch_size: int = 64,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if domain not in DOMAINS or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid domain/epochs/batch size")
    pair = make_pair(seed, domain)
    model_a = AdaptiveMLP(64, hidden_size, 10, mechanism="baseline", seed=seed)
    rng_a = np.random.default_rng(seed + 11_009)
    first_task_rows = []
    for epoch in range(1, epochs + 1):
        order = rng_a.permutation(len(pair.y_train))
        loss = _train_epoch(model_a, pair.a_train, pair.y_train, learning_rate, batch_size, order)
        first_task_rows.append({
            "phase": "task_a", "epoch": epoch, "loss": loss,
            "a_accuracy": model_a.accuracy(pair.a_test, pair.y_test),
            "b_accuracy": model_a.accuracy(pair.b_test, pair.y_test),
        })
    anchor = {name: getattr(model_a, name).copy() for name in PARAMETERS}
    a_before = model_a.accuracy(pair.a_test, pair.y_test)
    b_before = model_a.accuracy(pair.b_test, pair.y_test)
    history = []
    summary = []
    for treatment in TREATMENTS:
        working = model_a.clone()
        masks = importance_masks(
            model_a, treatment.importance, x_train=pair.a_train,
            y_train=pair.y_train, seed=seed,
        )
        for record in first_task_rows:
            history.append({
                "seed": seed, "domain": domain, "condition": treatment.name,
                **record, "penalty": treatment.penalty,
            })
        rng_b = np.random.default_rng(seed + 22_009)
        for epoch in range(1, epochs + 1):
            shuffled = rng_b.permutation(len(pair.y_train))
            loss_values = []
            for start in range(0, len(shuffled), batch_size):
                ids = shuffled[start:start + batch_size]
                snap = working.train_batch(pair.b_train[ids], pair.y_train[ids], learning_rate)
                apply_anchor(working, anchor, masks,
                             penalty=treatment.penalty, learning_rate=learning_rate)
                loss_values.append(snap.loss)
            history.append({
                "seed": seed, "domain": domain, "condition": treatment.name,
                "phase": "task_b", "epoch": epoch, "loss": float(np.mean(loss_values)),
                "a_accuracy": working.accuracy(pair.a_test, pair.y_test),
                "b_accuracy": working.accuracy(pair.b_test, pair.y_test),
                "penalty": treatment.penalty,
            })
        final_a = working.accuracy(pair.a_test, pair.y_test)
        final_b = working.accuracy(pair.b_test, pair.y_test)
        summary.append({
            "seed": seed, "domain": domain, "condition": treatment.name,
            "importance": treatment.importance, "penalty": treatment.penalty,
            "task_a_before_b": a_before, "task_b_before_b": b_before,
            "task_a_after_b": final_a, "task_b_after_b": final_b,
            "forgetting": a_before - final_a,
            "mean_final_accuracy": (final_a + final_b) / 2,
            "task_b_learning_gain": final_b - b_before,
        })
    return history, summary


def paired_effects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by = {(str(r["domain"]), int(r["seed"]), str(r["condition"])): r for r in rows}
    result = []
    for domain in DOMAINS:
        for treatment in TREATMENTS[1:]:
            for metric in ("task_a_after_b", "task_b_after_b", "mean_final_accuracy"):
                seeds = sorted({
                    int(r["seed"]) for r in rows if r["domain"] == domain
                })
                differences = np.array([
                    float(by[domain, seed, treatment.name][metric])
                    - float(by[domain, seed, "persistent"][metric])
                    for seed in seeds
                ])
                rng = np.random.default_rng(77_777)
                samples = differences[
                    rng.integers(0, len(differences), size=(2000, len(differences)))
                ].mean(axis=1)
                ci = np.percentile(samples, (2.5, 97.5))
                result.append({
                    "domain": domain, "condition": treatment.name, "metric": metric,
                    "n_paired_seeds": len(seeds),
                    "paired_mean_difference": float(np.mean(differences)),
                    "ci95_lower": float(ci[0]), "ci95_upper": float(ci[1]),
                    "ci_status": "unadjusted_descriptive_only",
                })
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(output: Path, *, seeds: tuple[int, ...] = SEEDS,
            domains: tuple[str, ...] = DOMAINS, epochs: int = 8) -> dict[str, object]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Unique seeds required")
    output.mkdir(parents=True, exist_ok=True)
    history = []
    summary = []
    for domain in domains:
        for seed in seeds:
            partial_h, partial_s = evaluate(seed, domain, epochs=epochs)
            history.extend(partial_h)
            summary.extend(partial_s)
    effects = paired_effects(summary)
    for name, rows in (("epoch_metrics.csv", history),
                       ("run_summary.csv", summary),
                       ("paired_effects.csv", effects)):
        _write_csv(output / name, rows)
    charts = output / "charts"
    charts.mkdir(exist_ok=True)
    for domain in domains:
        entries = [r for r in summary if r["domain"] == domain]
        means = {
            c.name: [float(np.mean([float(r[k]) for r in entries
                     if r["condition"] == c.name]))
                     for k in ("task_a_after_b", "task_b_after_b")]
            for c in TREATMENTS
        }
        fig, ax = plt.subplots(figsize=(11.5, 6))
        x = np.arange(len(TREATMENTS))
        ax.bar(x - 0.19, [means[c.name][0] for c in TREATMENTS],
               width=0.38, label="Task A retained")
        ax.bar(x + 0.19, [means[c.name][1] for c in TREATMENTS],
               width=0.38, label="Task B learned")
        ax.set_xticks(x, [c.name for c in TREATMENTS], rotation=27, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Held-out accuracy")
        ax.set_title(f"Stage Two follow-up: {domain} (selective vs uniform)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(charts / f"{domain}_comparison.png", dpi=130)
        plt.close(fig)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "experiment": "stage_two_selective_consolidation_followup",
        "status": "exploratory_unadjusted_not_confirmatory",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "dataset": "offline scikit-learn 8x8 digits; all classes in both tasks",
        "seeds": list(seeds), "domains": list(domains), "epochs": epochs,
        "treatments": [asdict(t) for t in TREATMENTS],
        "evaluation": "All methods share PRETRAINED Task A weights, train/test splits "
                      "and Task B minibatch orders per seed/domain. Task A importance "
                      "computed only from train samples; no data replay.",
        "interpretation": "Quadratic anchoring applied after each SGD update in B. "
                          "Activity pathway masks, minibatch squared-gradient proxy "
                          "(NOT empirical Fisher), and shuffled-activity control.",
        "trials": len(summary), "epoch_records": len(history),
        "metrics": ["task_a_after_b", "task_b_after_b", "mean_final_accuracy"],
        "statistics": "paired seed-level descriptive bootstrap, unadjusted across "
                      "many treatments/domains/metrics",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
        "limitations": [
            "Not a broad continual learning benchmark or a mechanistic proof",
            "Only 8x8 digit transformations; 5 exploratory evaluation seeds",
            "No rehearsal, calibrated EWC, or capacity-matched random data replay",
            "Post-SGD regularization approximation, not an exactly integrated loss",
            "Multiple comparisons unadjusted and tuning must use new seeds",
        ],
    }
    manifest["sha256"] = {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ("epoch_metrics.csv", "run_summary.csv", "paired_effects.csv")
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Follow-up:",len(summary),"paired runs,",len(history),"epoch records")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/selection-followup"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    seeds = (SEEDS[0],) if args.pilot else tuple(int(s) for s in args.seeds.split(","))
    collect(args.output, seeds=seeds, epochs=2 if args.pilot else args.epochs)


if __name__ == "__main__":
    main()
