from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FIGBHCConfig:
    """FI-GBHC 的固定半径增长比例。"""

    ratio: float

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio:
            raise ValueError("ratio must be positive")
