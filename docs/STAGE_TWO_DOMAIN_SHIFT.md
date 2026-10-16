# Stage Two — Paired Domain-Incremental Retention Study

**Status:** exploratory engineering study; not a confirmatory continual-learning result. [Repository](../README.md) · [Stage One frozen evidence](../data/stage_one_baseline/runs/2026-09-15_digits-8x8-v1/RUN.md).

## Research question

With **the same ten digit labels** in both tasks, do passive temporal decay or activity-protected decay improve the tradeoff between retaining Task A and adapting to a changed input domain B compared with a no-decay persistent model?

The frozen September 2026 Stage One class-split comparison yielded **less Task A forgetting for the persistent baseline than either decay mechanism**. That result is preserved. This Stage Two test is a *separate* domain-incremental task to isolate visual input shift from a change of class labels.

## Tasks, controls, and model

- Data: offline scikit-learn **8×8 handwritten digits only**, 1,797 images; raw intensities normalized to 0–1. **Not MNIST, Fashion-MNIST, or CIFAR.**
- Task A: learn all ten unmodified classes.
- Task B: same labels and images after either a fixed 90° rotation or a seed-specific 64-pixel permutation. Test inputs are transformed using the *same deterministic transformation* as training inputs.
- Train/test split is stratified and seed-dependent; **paired across all mechanisms and rates**, as are network initial weights and minibatch shuffle sequence.
- Mechanisms: one persistent control, passive decay at 0.002 or 0.008, and reinforced retention at both rates crossed with protection strength 4 or 8.
- Model: existing transparent one-hidden-layer NumPy MLP, 32 hidden neurons, SGD at learning rate 0.08, minibatch size 64, default eight epochs on A followed by eight on B.

## Outcomes

Primary: held-out Task A accuracy *after* B, and held-out Task B accuracy *after* B. Secondary: signed forgetting (A-before minus A-after), zero-clipped forgetting, backward transfer, B learning gain, balanced average accuracy, and accuracy loss after a fixed no-gradient idle interval.

Report all metrics per **paired seed**, with descriptive paired-bootstrap 95% intervals for changes from the persistent baseline. These intervals are **exploratory and unadjusted** for multiple comparisons. Do not claim significant improvement from favorable point estimates alone.

## Experimental workflow

```bash
pip install -e ".[dev]"
python -m adaptive_memory.stage_two --pilot --output results/stage-two-pilot
python -m adaptive_memory.stage_two --output results/stage-two
```

Outputs: `epoch_metrics.csv`, `run_summary.csv`, `paired_effects.csv`, `manifest.json`, and charts for both domains. The manifest records dataset digest, versions, all conditions, seeds, protocol settings, claim boundaries, and hashes of CSV files. CI runs the brief pilot; fuller five-seed sweeps should be reviewed before being committed as frozen evidence.

## Scientific boundaries and next phase

A rotation or feature permutation is a synthetic shift, not a natural real-world distribution. We are not yet evaluating larger image datasets, architectural alternatives, task-order reversal, replay, Elastic Weight Consolidation, calibration, energy consumption or class-incremental learning. This suite measures a limited, reproducible research *proxy*. Avoid retrospective cherry-picking of decay rates after examining outcomes; preregister a held-out design for any follow-up inferential study.

**Document replacement:** keep all frozen Stage One files and original research PDFs unaltered until Stage Two conclusions and reproducibility have been reviewed.
