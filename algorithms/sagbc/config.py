from __future__ import annotations

"""Configuration for the released SAGBC numerical path."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SAGBCConfig:
    """Official SAGBC constants plus the Benchmark sampling rule."""

    sample_size: int
    random_state: int = 1
    neighbor_count: int = 5
    max_ball_size: int = 8
    search_radius_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.sample_size < 2:
            raise ValueError("sample_size must be at least 2")
        if self.neighbor_count < 1:
            raise ValueError("neighbor_count must be at least 1")
        if self.max_ball_size < 1:
            raise ValueError("max_ball_size must be at least 1")
        if self.search_radius_scale <= 0:
            raise ValueError("search_radius_scale must be positive")
