"""Deterministic Stage One dataset and task construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class DatasetBundle:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    name: str = "sklearn-digits-8x8"


def load_stage_one(seed: int = 42, test_size: float = 0.25) -> DatasetBundle:
    """Load the offline digits proxy used to validate the experimental pipeline."""
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data.astype(np.float64),
        digits.target.astype(np.int64),
        test_size=test_size,
        random_state=seed,
        stratify=digits.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return DatasetBundle(x_train, x_test, y_train, y_test)


def task_subset(bundle: DatasetBundle, classes: range) -> DatasetBundle:
    train_mask = np.isin(bundle.y_train, list(classes))
    test_mask = np.isin(bundle.y_test, list(classes))
    return DatasetBundle(
        bundle.x_train[train_mask],
        bundle.x_test[test_mask],
        bundle.y_train[train_mask],
        bundle.y_test[test_mask],
        name=f"{bundle.name}-classes-{classes.start}-{classes.stop - 1}",
    )
