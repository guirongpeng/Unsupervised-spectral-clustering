from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MGAGCConfig:
    """Source-compatible MGAGC configuration."""

    n_clusters: int
    k: int
    beta: float
    lambda_init: float = 1.0
    min_points: int = 2
    max_iter: int = 20
    tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.n_clusters < 2 or self.k < 1 or self.min_points < 1 or self.max_iter < 1:
            raise ValueError("n_clusters, k, min_points and max_iter must be positive")
        if not math.isfinite(self.beta) or self.beta <= 0 or not math.isfinite(self.lambda_init) or self.lambda_init <= 0:
            raise ValueError("beta and lambda_init must be positive finite values")
        if not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be a finite non-negative value")
