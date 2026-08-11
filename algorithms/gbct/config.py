from __future__ import annotations

"""GBCT 官方发布源码的参数配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GBCTConfig:
    """GBCT 聚类数、固定 K-Means 参数和源码降噪常量。"""

    n_clusters: int
    coarse_random_state: int = 5
    coarse_n_init: int | str = "auto"
    coarse_max_iter: int = 300
    fine_random_state: int = 0
    fine_n_init: int = 3
    fine_max_iter: int = 2
    minimum_ball_size: int = 2
    noise_density_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if not isinstance(self.coarse_random_state, int):
            raise TypeError("coarse_random_state must be an integer")
        if self.coarse_n_init != "auto" and (
            not isinstance(self.coarse_n_init, int)
            or self.coarse_n_init < 1
        ):
            raise ValueError("coarse_n_init must be 'auto' or a positive integer")
        if self.coarse_max_iter < 1:
            raise ValueError("coarse_max_iter must be at least 1")
        if not isinstance(self.fine_random_state, int):
            raise TypeError("fine_random_state must be an integer")
        if self.fine_n_init < 1:
            raise ValueError("fine_n_init must be at least 1")
        if self.fine_max_iter < 1:
            raise ValueError("fine_max_iter must be at least 1")
        if self.minimum_ball_size < 1:
            raise ValueError("minimum_ball_size must be at least 1")
        if self.noise_density_ratio < 0:
            raise ValueError("noise_density_ratio must be non-negative")

"""GBCT 官方源码语义的统一 Benchmark 实现。"""


