# Evidence Guide

[← Project home](../README.md) · [Methods](METHODS.md) · [Roadmap →](ROADMAP.md)

## Evidence bundle contract

Every experiment directory contains raw observations, run summaries, a machine-readable manifest, and derived visualizations. A result is only reviewable when the manifest and raw CSV accompany the chart.

The manifest records the creation time, dataset, seeds, mechanism names, Python and platform versions, epoch budgets, and an explicit claim boundary. This prevents a polished chart from becoming detached from the conditions that produced it.

## Reproduce checked-in evidence

```bash
python -m adaptive_memory.cli \
  --seeds 7,21,42,84,101 \
  --epochs 24 \
  --continual-epochs 16 \
  --output results/reproduction
```

Compare `results/reproduction/run_summary.csv` with the checked-in [dated Stage One baseline](../data/stage_one_baseline/runs/2026-09-15_digits-8x8-v1/). Small floating-point differences across platforms are possible; directional claims should rely on aggregate comparisons, not exact final decimals.

The [data catalog](../data/README.md) indexes committed evidence by UTC collection date, dataset, and immutable run ID.

## Adding evidence

1. State a falsifiable hypothesis before executing the run.
2. Preserve control parity across mechanisms.
3. Use at least five independent seeds for exploratory comparison.
4. Commit raw data and its manifest with any derived figure.
5. Label exploratory, confirmatory, and externally replicated results separately.
