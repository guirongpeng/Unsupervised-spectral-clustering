from __future__ import annotations

"""Configuration for the source-first M3W command-line experiment path."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class M3WConfig:
    """Parameters used by the official ``run_m3w.py`` experiment."""

    k: int = 8
    levels: int = 3
    link_distance_expansion_factor: float = 1.6
    core_points_threshold: float = 0.6
    dvalue_threshold: float = 0.0
    border_percentile: float = 0.1
    mean_border_eps: float = 0.15
    stopping_percentile: float = 0.01
    min_cluster_size: int = 2
    convergence_constant: int = 0
    merge_core_points: bool = True

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.levels < 1:
            raise ValueError("levels must be at least 1")
        if (
            not math.isfinite(self.link_distance_expansion_factor)
            or self.link_distance_expansion_factor <= 0
        ):
            raise ValueError(
                "link_distance_expansion_factor must be a positive finite number"
            )
        if (
            not math.isfinite(self.core_points_threshold)
            or not 0 <= self.core_points_threshold <= 1
        ):
            raise ValueError(
                "core_points_threshold must be a finite number in [0, 1]"
            )
        if (
            not math.isfinite(self.dvalue_threshold)
            or self.dvalue_threshold < 0
        ):
            raise ValueError(
                "dvalue_threshold must be a non-negative finite number"
            )
        if (
            not math.isfinite(self.border_percentile)
            or not 0 < self.border_percentile < 1
        ):
            raise ValueError(
                "border_percentile must be a finite number in (0, 1)"
            )
        if (
            not math.isfinite(self.mean_border_eps)
            or self.mean_border_eps < 0
        ):
            raise ValueError(
                "mean_border_eps must be a non-negative finite number"
            )
        if (
            not math.isfinite(self.stopping_percentile)
            or not 0 <= self.stopping_percentile <= 1
        ):
            raise ValueError(
                "stopping_percentile must be a finite number in [0, 1]"
            )
        if self.min_cluster_size < 1:
            raise ValueError("min_cluster_size must be at least 1")
        if self.convergence_constant < 0:
            raise ValueError("convergence_constant must be non-negative")
        if not isinstance(self.merge_core_points, bool):
            raise TypeError("merge_core_points must be a boolean")
