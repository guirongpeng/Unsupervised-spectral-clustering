from __future__ import annotations

"""Unified Benchmark registration for M3W."""

from benchmark import AlgorithmRegistry

from .algorithm import M3W
from .config import M3WConfig


def register_m3w(
    registry: AlgorithmRegistry,
    config: M3WConfig | None = None,
) -> None:
    """Register M3W, optionally with dataset-specific ``k`` and ``levels``."""

    resolved_config = config or M3WConfig()
    registry.register(
        "m3w",
        lambda _n_clusters, _seed: M3W(resolved_config),
    )
