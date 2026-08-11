from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class GBPOJGGBSCConfig:
    gamma: float = 2.0
    delta: float = 1.0
    sigma: float = 1.0
    n_init: int = 1
    max_iter: int = 100
    def __post_init__(self) -> None:
        if not math.isfinite(self.gamma) or self.gamma < 0: raise ValueError("gamma must be finite and non-negative")
        if not math.isfinite(self.delta) or not 0 < self.delta <= 1: raise ValueError("delta must be in (0, 1]")
        if not math.isfinite(self.sigma) or self.sigma <= 0: raise ValueError("sigma must be positive")
