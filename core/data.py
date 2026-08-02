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
    if not config.path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {config.path}")
    with np.load(config.path, allow_pickle=False) as data:
        missing = {"X", "y"}.difference(data.files)
        if missing:
            raise ValueError(
                f"Dataset {config.path} is missing arrays: {sorted(missing)}"
            )
        X = np.asarray(data["X"])
        y = np.asarray(data["y"]).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size:
        raise ValueError(f"Invalid X/y shapes: {X.shape}, {y.shape}")
    if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(y.dtype, np.number):
        raise TypeError("Dataset X and y must be numeric")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Dataset contains NaN or infinite values")
    if np.unique(y).size < 2:
        raise ValueError("Dataset must contain at least two classes")
    return Dataset(config.name, X, y)


def minmax_scale(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    minimum = values.min(axis=0)
    ranges = values.max(axis=0) - minimum
    return (values - minimum) / np.where(ranges == 0.0, 1.0, ranges)
