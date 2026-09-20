# Experimental data catalog

Committed evidence uses immutable dated run directories:

```text
data/<experiment_id>/runs/YYYY-MM-DD_<condition>-vN/
```

| UTC date | Run ID | Experiment | Dataset | Status | Evidence |
|---|---|---|---|---|---|
| 2026-09-15 | `2026-09-15_digits-8x8-v1` | Stage One memory-mechanism baseline | sklearn digits 8×8 | Frozen exploratory baseline | [Open run](stage_one_baseline/runs/2026-09-15_digits-8x8-v1/) |

Dates come from generated manifests. New datasets, decay ablations, cost studies, and replications receive new run IDs; existing evidence is never overwritten. Quick checks and incomplete runs remain under ignored `results/` paths.
