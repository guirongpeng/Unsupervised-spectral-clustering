from __future__ import annotations

"""Configuration for the official GBSC UCI implementation path."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GBSCConfig:
    """GBSC source parameters; only ``sigma`` is a paper-level grid parameter."""

    sigma: float = 1.0
    minimum_split_size: int = 8
    radius_detection_factor: float = 2.0
    spectral_n_init: int = 10
    spectral_eigen_tol: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be a positive finite number")
        if self.minimum_split_size < 2:
            raise ValueError("minimum_split_size must be at least 2")
        if not math.isfinite(self.radius_detection_factor) or self.radius_detection_factor <= 0:
            raise ValueError("radius_detection_factor must be positive")
        if self.spectral_n_init < 1:
            raise ValueError("spectral_n_init must be at least 1")
        if not math.isfinite(self.spectral_eigen_tol) or self.spectral_eigen_tol < 0:
            raise ValueError("spectral_eigen_tol must be non-negative")
