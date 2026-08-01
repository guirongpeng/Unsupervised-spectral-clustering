from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import DatasetConfig


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)


def load_dataset(config: DatasetConfig) -> Dataset:
    X = np.loadtxt(config.x_path, delimiter=",", dtype=float)
    y = np.loadtxt(config.y_path, delimiter=",", dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"Invalid X/y shapes: {X.shape}, {y.shape}")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Dataset contains NaN or infinite values")
    if np.unique(y).size < 2:
        raise ValueError("Dataset must contain at least two classes")
    return Dataset(config.name, X, y)


def minmax_scale(X: np.ndarray) -> np.ndarray:
    minimum = X.min(axis=0)
    ranges = X.max(axis=0) - minimum
    return (X - minimum) / np.where(ranges == 0.0, 1.0, ranges)

