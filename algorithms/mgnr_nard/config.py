from __future__ import annotations

"""Configuration for the four official NARD clustering applications."""

from dataclasses import dataclass
import math
from typing import Literal


NARDBackend = Literal["dpeak", "dbscan", "dadc", "hcdc"]
SUPPORTED_BACKENDS = frozenset({"dpeak", "dbscan", "dadc", "hcdc"})


@dataclass(frozen=True)
class NARDConfig:
    """Source constants and the density-clustering backend."""

    backend: NARDBackend
    radius_detection_factor: float = 2.0
    dbscan_core_factor: float = 0.4
    hcdc_small_cluster_fraction: float = 0.01

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            supported = ", ".join(sorted(SUPPORTED_BACKENDS))
            raise ValueError(
                f"Unsupported NARD backend {self.backend!r}; "
                f"expected: {supported}"
            )
        if (
            not math.isfinite(self.radius_detection_factor)
            or self.radius_detection_factor <= 0
        ):
            raise ValueError(
                "radius_detection_factor must be a positive finite number"
            )
        if (
            not math.isfinite(self.dbscan_core_factor)
            or self.dbscan_core_factor <= 0
        ):
            raise ValueError(
                "dbscan_core_factor must be a positive finite number"
            )
        if (
            not math.isfinite(self.hcdc_small_cluster_fraction)
            or not 0 <= self.hcdc_small_cluster_fraction < 1
        ):
            raise ValueError(
                "hcdc_small_cluster_fraction must be in [0, 1)"
            )
