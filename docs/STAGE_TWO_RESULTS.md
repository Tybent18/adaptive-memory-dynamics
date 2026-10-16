# Adaptive Memory Dynamics - Stage Two Domain-Shift Results

**Status: preliminary exploratory software experiment, not peer reviewed.**

This report is generated directly from the Stage Two CSV and manifest. Historical Stage One findings are preserved; this experiment addresses a **different question** (same labels, changed image domain).

## Configuration and evidence

- Dataset: scikit-learn load_digits, 8x8 handwritten digits; no MNIST.
- Seeds: 7, 21, 42, 84, 101 (5 matched seeds per condition).
- Conditions: 7 (persistent; passive decay; usage-reinforced decay).
- Domains: rotate90, permuted_pixels.
- Training: 8 epochs for each of A and B; 32 hidden neurons; SGD learning rate 0.08.
- Complete condition runs: 70; epoch metric rows: 1120.
- Source commit: `6ef2c501f111763c8b5a33c8fdd8d62b46072bad`.
- Input data, labels, splits, initialization seeds, and minibatch order are matched within each domain. Transformed test samples are held out from training, but the seeds/rates are exploratory rather than a preregistered independent confirmatory test.

## Held-out accuracy and forgetting, by configuration

Values below are the **mean +/- sample standard deviation across seeds** when at least two seeds are available. Higher A-after-B and B-after-B accuracy is preferable. Lower zero-clipped forgetting is preferable. These are proportions, not percentage points.

### Domain B: `rotate90`

| Condition | Task A after B | Task B after A | A forgetting | Mean final accuracy |
|:--|--:|--:|--:|--:|
| persistent | 0.587 +/- 0.065 | 0.879 +/- 0.013 | 0.220 +/- 0.077 | 0.733 +/- 0.031 |
| passive_002 | 0.362 +/- 0.032 | 0.812 +/- 0.025 | 0.426 +/- 0.064 | 0.587 +/- 0.027 |
| passive_008 | 0.201 +/- 0.025 | 0.603 +/- 0.048 | 0.512 +/- 0.055 | 0.402 +/- 0.026 |
| reinforced_002_4 | 0.452 +/- 0.050 | 0.848 +/- 0.014 | 0.340 +/- 0.085 | 0.650 +/- 0.028 |
| reinforced_002_8 | 0.463 +/- 0.055 | 0.852 +/- 0.014 | 0.330 +/- 0.085 | 0.658 +/- 0.030 |
| reinforced_008_4 | 0.344 +/- 0.033 | 0.772 +/- 0.034 | 0.427 +/- 0.057 | 0.558 +/- 0.033 |
| reinforced_008_8 | 0.400 +/- 0.045 | 0.801 +/- 0.026 | 0.384 +/- 0.083 | 0.600 +/- 0.033 |

**Paired difference in Task A retention vs persistent** (mean, exploratory 95% percentile bootstrap interval):

- `passive_002`: -0.225 ([-0.251, -0.200]; n=5).
- `passive_008`: -0.386 ([-0.424, -0.341]; n=5).
- `reinforced_002_4`: -0.135 ([-0.160, -0.114]; n=5).
- `reinforced_002_8`: -0.124 ([-0.149, -0.103]; n=5).
- `reinforced_008_4`: -0.244 ([-0.272, -0.213]; n=5).
- `reinforced_008_8`: -0.188 ([-0.216, -0.166]; n=5).

![Stage Two paired condition comparison for rotate90](../data/stage_two_exploratory/runs/2026-10-15_domain-shift-paired-v1/charts/rotate90_paired_accuracy.png)

### Domain B: `permuted_pixels`

| Condition | Task A after B | Task B after A | A forgetting | Mean final accuracy |
|:--|--:|--:|--:|--:|
| persistent | 0.607 +/- 0.066 | 0.885 +/- 0.020 | 0.201 +/- 0.098 | 0.746 +/- 0.028 |
| passive_002 | 0.499 +/- 0.100 | 0.830 +/- 0.021 | 0.289 +/- 0.131 | 0.665 +/- 0.046 |
| passive_008 | 0.235 +/- 0.058 | 0.649 +/- 0.032 | 0.479 +/- 0.117 | 0.442 +/- 0.023 |
| reinforced_002_4 | 0.580 +/- 0.081 | 0.861 +/- 0.014 | 0.212 +/- 0.120 | 0.720 +/- 0.040 |
| reinforced_002_8 | 0.591 +/- 0.080 | 0.866 +/- 0.011 | 0.202 +/- 0.119 | 0.729 +/- 0.040 |
| reinforced_008_4 | 0.491 +/- 0.103 | 0.806 +/- 0.032 | 0.279 +/- 0.136 | 0.649 +/- 0.043 |
| reinforced_008_8 | 0.536 +/- 0.096 | 0.831 +/- 0.023 | 0.248 +/- 0.135 | 0.683 +/- 0.041 |

**Paired difference in Task A retention vs persistent** (mean, exploratory 95% percentile bootstrap interval):

- `passive_002`: -0.108 ([-0.158, -0.057]; n=5).
- `passive_008`: -0.372 ([-0.400, -0.345]; n=5).
- `reinforced_002_4`: -0.027 ([-0.057, +0.006]; n=5).
- `reinforced_002_8`: -0.016 ([-0.042, +0.015]; n=5).
- `reinforced_008_4`: -0.116 ([-0.174, -0.060]; n=5).
- `reinforced_008_8`: -0.071 ([-0.127, -0.017]; n=5).

![Stage Two paired condition comparison for permuted_pixels](../data/stage_two_exploratory/runs/2026-10-15_domain-shift-paired-v1/charts/permuted_pixels_paired_accuracy.png)

## Interpretation and limitations

This study compares *descriptive held-out accuracy* and paired differences on small, synthetic domain changes. A larger Task A score together with a lower Task B score would imply a stability-plasticity trade-off, not necessarily a better memory algorithm. Do not describe any observed advantage as a verified continual-learning breakthrough or as statistically significant without a separately designed multiple-comparison-controlled study.

- The input is scikit-learn 8x8 digits, **not MNIST, Fashion-MNIST or CIFAR**.
- All ten classes are present in both domains; this differs from the Stage One disjoint-label task.
- The study does not include rehearsal/replay, EWC, task-order reversal, real-world concept drift or a larger neural architecture.
- Bootstrap intervals treat the seed as the paired unit and are **unadjusted for multiple comparisons**.
- Metrics measure classification accuracy, not human-like memory, physical energy, or neural data retention.
- A five-seed exploratory result should not be treated as a publication-quality claim. The next study needs held-out evaluation seeds, more baselines, ablations, and a prespecified primary comparison.

## Reproduce and inspect the full evidence

- [Stage Two protocol](STAGE_TWO_DOMAIN_SHIFT.md)
- [Frozen Stage One results](../data/stage_one_baseline/runs/2026-09-15_digits-8x8-v1/RUN.md)
- [Stage Two run CSVs, manifest and figures](../data/stage_two_exploratory/runs/2026-10-15_domain-shift-paired-v1/)

Run: `python -m adaptive_memory.stage_two --output results/stage-two`. The linked Stage Two run is a versioned exploratory evidence bundle; do not modify it in place after publication.
