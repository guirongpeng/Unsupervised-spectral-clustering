from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AGCILDConfig:
    """AGC-ILD fixed settings for one (m, beta) experiment."""

    n_anchors: int
    beta: float
    k_neighbors: int = 5
    max_iter: int = 100
    tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.n_anchors < 2 or self.n_anchors & (self.n_anchors - 1):
            raise ValueError("n_anchors must be a power of two and at least 2")
        if self.beta < 0 or self.k_neighbors < 1:
            raise ValueError("invalid AGC-ILD parameters")
