from __future__ import annotations

"""GB-POJG + GBDPC configuration copied from the source-compatible Python reproduction."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GBPOJGGBDPCConfig:
    """Official ``main.m`` defaults for the GB-POJG + GBDPC pipeline.

    ``gamma`` is the POJG quality parameter and ``delta`` controls the
    initial granular-ball split threshold.  The MATLAB GBDPC stage itself
    has no additional tuned parameter.
    """

    gamma: float = 0.0
    delta: float = 0.5

    def __post_init__(self) -> None:
        if not math.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be a finite number in [0, +inf)")
        if not math.isfinite(self.delta) or not 0 < self.delta <= 1:
            raise ValueError("delta must be a finite number in (0, 1]")
