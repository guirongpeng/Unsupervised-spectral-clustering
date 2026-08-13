from __future__ import annotations

"""GB-DP 的统一注册入口。"""

from benchmark import AlgorithmRegistry

from .algorithm import GBDP
from .config import GBDPConfig


def register_gb_dp(registry: AlgorithmRegistry) -> None:
    """把 GB-DP 注册为 ``gb_dp``。

    Benchmark 提供聚类数。实验种子不会替换官方 2-means 固定随机种子 8。
    """

    registry.register(
        "gb_dp",
        lambda n_clusters, _seed: GBDP(
            GBDPConfig(n_clusters=n_clusters)
        ),
    )
