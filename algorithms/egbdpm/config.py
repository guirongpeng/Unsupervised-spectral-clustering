from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EGBDPMConfig:
    """EGBDPM 中密度和球面测地距离共用的近邻数 k。"""

    k_neighbors: int

    def __post_init__(self) -> None:
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
