from __future__ import annotations

"""GB-DP 参数配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GBDPConfig:
    """GB-DP 的聚类数及官方 2-means 参数。"""

    n_clusters: int
    random_state: int = 8
    n_init: int = 1

    def __post_init__(self) -> None:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer")
        if self.n_init < 1:
            raise ValueError("n_init must be at least 1")
