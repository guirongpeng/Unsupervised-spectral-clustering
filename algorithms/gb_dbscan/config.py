from __future__ import annotations

"""Configuration for the source-compatible GB-DBSCAN implementation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GBDBSCANConfig:
    """``ratio`` is the source-paper Core-GB proportion parameter."""

    ratio: float
    n_neighbors: int | None = None
    neighbor_scale: float = 0.3
    neighbor_algorithm: str = "auto"
    leaf_size: int = 30

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError("ratio must be in the interval (0, 1]")
        if self.n_neighbors is not None and self.n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if self.neighbor_scale <= 0.0:
            raise ValueError("neighbor_scale must be positive")
        if self.neighbor_algorithm not in {"auto", "ball_tree", "kd_tree", "brute"}:
            raise ValueError("neighbor_algorithm must be auto, ball_tree, kd_tree, or brute")
        if self.leaf_size < 1:
            raise ValueError("leaf_size must be at least 1")

    def resolve_n_neighbors(self, n_samples: int) -> int:
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")
        if self.n_neighbors is not None:
            return self.n_neighbors
        return int(np.ceil(np.sqrt(n_samples)) * self.neighbor_scale)
