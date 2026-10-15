import csv
import json

import numpy as np

from adaptive_memory.stage_two import (
    CONDITIONS,
    DOMAINS,
    collect,
    make_pair,
    paired_effects,
    run_condition,
)


def test_domain_transform_preserves_labels_and_split() -> None:
    rotated = make_pair(7, "rotate90")
    permuted = make_pair(7, "permuted_pixels")
    np.testing.assert_array_equal(rotated.y_train, permuted.y_train)
    np.testing.assert_array_equal(rotated.y_test, permuted.y_test)
    np.testing.assert_array_equal(rotated.a_train, permuted.a_train)
    assert not np.array_equal(rotated.a_train, rotated.b_train)
    assert not np.array_equal(permuted.a_train, permuted.b_train)


def test_repeated_stage_two_run_is_deterministic() -> None:
    a, r1 = run_condition(7, DOMAINS[0], CONDITIONS[3], epochs=2)
    b, r2 = run_condition(7, DOMAINS[0], CONDITIONS[3], epochs=2)
    assert a == b
    assert r1 == r2
    assert 0 <= r1["task_a_after_b"] <= 1
    assert 0 <= r1["task_b_after_b"] <= 1


def test_paired_effects_return_seed_level_differences() -> None:
    rows = []
    for seed in (7, 21):
        for condition in CONDITIONS:
            _, result = run_condition(seed, "rotate90", condition, epochs=1)
            rows.append(result)
    effects = paired_effects(rows, resamples=60)
    assert len(effects) == 6 * 5
    assert all(row["n_paired_seeds"] == 2 for row in effects)
    assert all(row["ci95_lower"] != "" for row in effects)


def test_stage_two_pilot_exports_with_integrity(tmp_path) -> None:
    manifest = collect(tmp_path, seeds=(7,), epochs=1)
    assert manifest["trials"] == len(CONDITIONS) * len(DOMAINS)
    assert manifest["status"].startswith("exploratory")
    assert (tmp_path / "charts/rotate90_paired_accuracy.png").exists()
    assert (tmp_path / "charts/permuted_pixels_paired_accuracy.png").exists()
    summary = list(csv.DictReader((tmp_path / "run_summary.csv").open()))
    assert len(summary) == len(CONDITIONS) * len(DOMAINS)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["checksums_sha256"] == manifest["checksums_sha256"]
