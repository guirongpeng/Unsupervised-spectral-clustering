from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PBGBHCConfig:
    """PB-GBHC 的最近中心距离百分位 q，范围为 (0, 100]。"""

    q: float

    def __post_init__(self) -> None:
        if not 0.0 < self.q <= 100.0:
            raise ValueError("q must be in (0, 100]")
