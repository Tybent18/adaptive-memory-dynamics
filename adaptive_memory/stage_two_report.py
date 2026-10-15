"""Write an evidence-grounded Stage Two results report from actual output CSVs.

Reports exploratory descriptive metrics only, never invents a win condition.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _format(values: list[float]) -> str:
    if not values:
        return "not measured"
    return f"{mean(values):.3f} +/- {stdev(values):.3f}" if len(values) > 1 else f"{values[0]:.3f} (n=1)"


def render(evidence: Path, destination: Path) -> str:
    manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
    records = _read_csv(evidence / "run_summary.csv")
    paired = _read_csv(evidence / "paired_effects.csv")
    expected = (
        len(manifest["seeds"]) * len(manifest["domains"]) *
        len(manifest["mechanisms"])
    )
    if expected != len(records):
        raise ValueError(f"Missing trials: expected {expected}, found {len(records)}")
    if any(row["condition"] not in {
        m["name"] for m in manifest["mechanisms"]
    } for row in records):
        raise ValueError("Unrecognized condition in summary")
    out = [
        "# Adaptive Memory Dynamics - Stage Two Domain-Shift Results",
        "",
        "**Status: preliminary exploratory software experiment, not peer reviewed.**",
        "",
        "This report is generated directly from the Stage Two CSV and manifest. "
        "Historical Stage One findings are preserved; this experiment addresses a "
        "**different question** (same labels, changed image domain).",
        "",
        "## Configuration and evidence",
        "",
        f"- Dataset: {manifest['dataset']}.",
        f"- Seeds: {', '.join(map(str,manifest['seeds']))} "
        f"({len(manifest['seeds'])} matched seeds per condition).",
        f"- Conditions: {len(manifest['mechanisms'])} "
        f"(persistent; passive decay; usage-reinforced decay).",
        f"- Domains: {', '.join(manifest['domains'])}.",
        f"- Training: {manifest['epochs_per_task']} epochs for each of A and B; "
        f"{manifest['model']['hidden']} hidden neurons; SGD "
        f"learning rate {manifest['model']['learning_rate']}.",
        f"- Complete condition runs: {len(records)}; "
        f"epoch metric rows: {manifest['epochs_recorded']}.",
        f"- Source commit: `{manifest.get('source_commit') or 'unknown'}`.",
        "- Input data, labels, splits, initialization seeds, and minibatch order "
        "are matched within each domain. Transformed test samples are held out "
        "from training, but the seeds/rates are exploratory rather than a "
        "preregistered independent confirmatory test.",
        "",
        "## Held-out accuracy and forgetting, by configuration",
        "",
        "Values below are the **mean +/- sample standard deviation across seeds** "
        "when at least two seeds are available. Higher A-after-B and B-after-B "
        "accuracy is preferable. Lower zero-clipped forgetting is preferable. "
        "These are proportions, not percentage points.",
        "",
    ]
    effects_by_key = {(r["domain"],r["condition"],r["metric"]):r for r in paired}
    for domain in manifest["domains"]:
        out.extend([
            f"### Domain B: `{domain}`",
            "",
            "| Condition | Task A after B | Task B after A | A forgetting | Mean final accuracy |",
            "|:--|--:|--:|--:|--:|",
        ])
        for condition in manifest["mechanisms"]:
            name = condition["name"]
            subset = [r for r in records if r["domain"] == domain
                      and r["condition"] == name]
            values = [
                _format([float(r[key]) for r in subset])
                for key in ("task_a_after_b","task_b_after_b",
                            "nonnegative_forgetting","mean_final_accuracy")
            ]
            out.append(f"| {name} | " + " | ".join(values) + " |")
        out.extend(["", "**Paired difference in Task A retention vs persistent** "
                    "(mean, exploratory 95% percentile bootstrap interval):", ""])
        for condition in manifest["mechanisms"][1:]:
            p = effects_by_key.get((domain,condition["name"],"task_a_after_b"))
            if p is None:
                continue
            ci = ("no interval with one seed" if p["ci95_lower"] == ""
                  else f"[{float(p['ci95_lower']):+.3f}, "
                       f"{float(p['ci95_upper']):+.3f}]")
            out.append(
                f"- `{condition['name']}`: "
                f"{float(p['mean_difference_vs_persistent']):+.3f} "
                f"({ci}; n={p['n_paired_seeds']})."
            )
        out.extend([
            "",
            f"![Stage Two paired condition comparison for {domain}]"
            f"(../data/stage_two_exploratory/runs/"
            f"2026-10-15_domain-shift-paired-v1/charts/{domain}_paired_accuracy.png)",
            "",
        ])
    out.extend([
        "## Interpretation and limitations",
        "",
        "This study compares *descriptive held-out accuracy* and paired differences "
        "on small, synthetic domain changes. A larger Task A score together with "
        "a lower Task B score would imply a stability-plasticity trade-off, "
        "not necessarily a better memory algorithm. "
        "Do not describe any observed advantage as a verified continual-learning "
        "breakthrough or as statistically significant without a separately "
        "designed multiple-comparison-controlled study.",
        "",
        "- The input is scikit-learn 8x8 digits, **not MNIST, Fashion-MNIST or CIFAR**.",
        "- All ten classes are present in both domains; this differs from the "
        "Stage One disjoint-label task.",
        "- The study does not include rehearsal/replay, EWC, task-order reversal, "
        "real-world concept drift or a larger neural architecture.",
        "- Bootstrap intervals treat the seed as the paired unit and are "
        "**unadjusted for multiple comparisons**.",
        "- Metrics measure classification accuracy, not human-like memory, "
        "physical energy, or neural data retention.",
        "- A five-seed exploratory result should not be treated as a publication-"
        "quality claim. The next study needs held-out evaluation seeds, more "
        "baselines, ablations, and a prespecified primary comparison.",
        "",
        "## Reproduce and inspect the full evidence",
        "",
        "- [Stage Two protocol](STAGE_TWO_DOMAIN_SHIFT.md)",
        "- [Frozen Stage One results]"
        "(../data/stage_one_baseline/runs/2026-09-15_digits-8x8-v1/RUN.md)",
        "- [Stage Two run CSVs, manifest and figures]"
        "(../data/stage_two_exploratory/runs/2026-10-15_domain-shift-paired-v1/)",
        "",
        "Run: `python -m adaptive_memory.stage_two --output results/stage-two`. "
        "The linked Stage Two run is a versioned exploratory evidence bundle; "
        "do not modify it in place after publication.",
        "",
    ])
    result = "\n".join(out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(result,encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path,required=True)
    parser.add_argument("--report", type=Path,required=True)
    args=parser.parse_args()
    result=render(args.evidence,args.report)
    print("Generated evidence-grounded Stage Two report:",args.report,
          "characters:",len(result))


if __name__=="__main__":
    main()
