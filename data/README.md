# Experimental data catalog

Committed evidence uses immutable dated run directories:

```text
data/<experiment_id>/runs/YYYY-MM-DD_<condition>-vN/
```

| UTC date | Run ID | Experiment | Dataset | Status | Evidence |
|---|---|---|---|---|---|
| 2026-09-15 | `2026-09-15_digits-8x8-v1` | Stage One memory-mechanism baseline | sklearn digits 8×8 | Frozen exploratory baseline | [Open run](stage_one_baseline/runs/2026-09-15_digits-8x8-v1/) |

| 2026-10-15 | `2026-10-15_domain-shift-paired-v1` | Stage Two paired domain-shift comparison: 5 seeds × 2 domains × 7 conditions = 70 runs | sklearn digits 8×8 | **Frozen exploratory, not confirmatory** | [Run, CSVs, figures and manifest](stage_two_exploratory/runs/2026-10-15_domain-shift-paired-v1/) · [Research results](../docs/STAGE_TWO_RESULTS.md) |

| 2026-10-15 | `2026-10-15_selective-consolidation-v1` | Selective versus uniform pathway consolidation; 5 fresh seeds × 2 domains × 8 conditions = 80 runs | sklearn digits 8×8 | **Frozen exploratory, not confirmatory** | [Evidence & figures](stage_two_exploratory/runs/2026-10-15_selective-consolidation-v1/) · [Study report](../docs/SELECTIVE_CONSOLIDATION_RESULTS.md) |

Dates come from generated manifests. New datasets, decay ablations, cost studies, and replications receive new run IDs; existing evidence is never overwritten. Quick checks and incomplete runs remain under ignored `results/` paths.
