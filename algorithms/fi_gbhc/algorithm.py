from __future__ import annotations

"""Fixed Increment Granular Ball Hierarchical Clustering (FI-GBHC)."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from core.algorithm import Algorithm as BenchmarkAlgorithm

from .config import FIGBHCConfig


@dataclass(frozen=True, slots=True)
class GranularBall:
    sample_indices: np.ndarray
    center: np.ndarray
    radius: float


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size)); self.rank = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]; item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first == second: return
        if self.rank[first] < self.rank[second]: first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]: self.rank[first] += 1


def _make_ball(X: np.ndarray, indices: np.ndarray) -> GranularBall:
    points = X[indices]; center = np.mean(points, axis=0)
    return GranularBall(indices, center, float(np.mean(np.linalg.norm(points - center, axis=1))))


def run_fi_gbhc(X: np.ndarray, n_clusters: int, config: FIGBHCConfig) -> tuple[np.ndarray, tuple[GranularBall, ...]]:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not 2 <= n_clusters <= values.shape[0] or not np.all(np.isfinite(values)):
        raise ValueError("X must be finite 2D data and n_clusters must be in [2, n_samples]")
    balls = [_make_ball(values, np.asarray([index])) for index in range(values.shape[0])]
    # Official code uses the maximum sample-pair distance (equivalent to its cdist maximum).
    increment = float(np.max(cdist(values, values)) * config.ratio)
    while len(balls) > n_clusters:
        grown = [GranularBall(ball.sample_indices, ball.center, ball.radius + increment) for ball in balls]
        centers = np.asarray([ball.center for ball in grown]); tree = KDTree(centers); maximum = max(ball.radius for ball in grown)
        groups = _UnionFind(len(grown))
        for index, ball in enumerate(grown):
            for other in tree.query_ball_point(ball.center, ball.radius + maximum):
                if index < other and np.linalg.norm(ball.center - grown[other].center) <= ball.radius + grown[other].radius:
                    groups.union(index, other)
        merged: dict[int, list[int]] = {}
        for index in range(len(grown)): merged.setdefault(groups.find(index), []).append(index)
        balls = [grown[group[0]] if len(group) == 1 else _make_ball(values, np.concatenate([grown[item].sample_indices for item in group])) for group in merged.values()]
    labels = np.empty(values.shape[0], dtype=int)
    for label, ball in enumerate(balls): labels[ball.sample_indices] = label
    return labels, tuple(balls)


class FIGBHC(BenchmarkAlgorithm):
    def __init__(self, config: FIGBHCConfig, n_clusters: int) -> None:
        self.config, self.n_clusters = config, n_clusters

    def fit(self, X: np.ndarray) -> "FIGBHC":
        self.labels_, self.granular_balls_ = run_fi_gbhc(X, self.n_clusters, self.config)
        return self

    def get_params(self) -> dict[str, object]:
        return {"n_clusters": self.n_clusters, **asdict(self.config)}
