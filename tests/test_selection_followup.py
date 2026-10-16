import numpy as np

from adaptive_memory.model import AdaptiveMLP
from adaptive_memory.selection_followup import (
    TREATMENTS, apply_anchor, collect, evaluate, importance_masks,
)


def test_importance_masks_and_shuffled_control() -> None:
    model = AdaptiveMLP(4, 5, 3, seed=9)
    model.usage[:] = [0.1, 0.2, 0.5, 0.8, 1.0]
    a = importance_masks(model, "activity")
    b = importance_masks(model, "shuffled_activity", seed=9)
    assert a["w1"].shape == model.w1.shape
    assert not np.array_equal(a["b1"], b["b1"])
    assert np.isclose(a["b1"].mean(), 1)


def test_anchor_preserves_reference_for_zero_displacement() -> None:
    model = AdaptiveMLP(4, 5, 3, seed=8)
    ref = {n: getattr(model, n).copy() for n in ("w1", "b1", "w2", "b2")}
    masks = importance_masks(model, "uniform")
    apply_anchor(model, ref, masks, penalty=0.5, learning_rate=0.08)
    for name, value in ref.items():
        np.testing.assert_array_equal(getattr(model, name), value)


def test_followup_determinism_and_common_task_a_pretrain() -> None:
    rows_a, summary_a = evaluate(1109, "rotate90", epochs=1)
    rows_b, summary_b = evaluate(1109, "rotate90", epochs=1)
    assert rows_a == rows_b
    assert summary_a == summary_b
    assert len(summary_a) == len(TREATMENTS)
    assert len({r["task_a_before_b"] for r in summary_a}) == 1


def test_followup_exports(tmp_path) -> None:
    manifest = collect(tmp_path, seeds=(1109,), epochs=1)
    assert manifest["trials"] == 2 * len(TREATMENTS)
    assert (tmp_path / "run_summary.csv").stat().st_size > 300
    assert (tmp_path / "charts/rotate90_comparison.png").stat().st_size > 1000
