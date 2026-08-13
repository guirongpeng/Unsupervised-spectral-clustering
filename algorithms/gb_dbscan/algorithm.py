from __future__ import annotations

"""GB-DBSCAN, following Algorithms 1--2 and GB-DBSCAN-commit.py."""

from dataclasses import asdict

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from core.algorithm import Algorithm

from .config import GBDBSCANConfig


def _validate_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not all(values.shape) or not np.all(np.isfinite(values)):
        raise ValueError("X must be a finite, non-empty 2-D array")
    return values


def _generate_granular_balls(X: np.ndarray, n_neighbors: int, config: GBDBSCANConfig) -> tuple[np.ndarray, ...]:
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm=config.neighbor_algorithm, leaf_size=config.leaf_size, metric="euclidean").fit(X).kneighbors(X, return_distance=False)
    visited = np.zeros(X.shape[0], dtype=bool)
    balls: list[np.ndarray] = []
    for index in range(X.shape[0]):
        if not visited[index]:
            ball = np.asarray(neighbors[index], dtype=int)
            balls.append(ball)
            visited[ball] = True
    return tuple(balls)


def _centers_and_radii(X: np.ndarray, balls: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray([X[ball].mean(axis=0) for ball in balls])
    radii = np.asarray([np.max(np.linalg.norm(X[ball] - center, axis=1)) for ball, center in zip(balls, centers)], dtype=float)
    return centers, radii


def _split_core_balls(radii: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray, float]:
    core_count = int(radii.size * ratio)
    if core_count < 1:
        raise ValueError("ratio is too small for the generated granular-ball count")
    threshold = float(np.sort(radii)[core_count - 1])
    return np.flatnonzero(radii <= threshold), np.flatnonzero(radii > threshold), threshold


def _cluster_core_balls(core: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    unvisited = list(range(core.size))
    labels = np.full(core.size, -1, dtype=int)
    cluster_id = -1
    while unvisited:
        current = unvisited.pop(0)
        neighbors: list[int] = []
        for candidate in range(core.size):
            if candidate != current and np.linalg.norm(centers[core[candidate]] - centers[core[current]]) <= radii[core[candidate]] + radii[core[current]]:
                neighbors.append(candidate)
        cluster_id += 1
        labels[current] = cluster_id
        for neighbor in neighbors:
            if neighbor in unvisited:
                unvisited.remove(neighbor)
                for candidate in range(core.size):
                    if candidate != neighbor and np.linalg.norm(centers[core[candidate]] - centers[core[neighbor]]) <= radii[core[candidate]] + radii[core[neighbor]] and candidate not in neighbors:
                        neighbors.append(candidate)
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
    return labels


def _assign_non_core_balls(X: np.ndarray, labels: np.ndarray, balls: tuple[np.ndarray, ...], core: np.ndarray, non_core: np.ndarray, centers: np.ndarray) -> np.ndarray:
    core_centers = centers[core]
    for ball_index in non_core:
        ball = balls[ball_index]
        classified, unclassified = ball[labels[ball] != -1], ball[labels[ball] == -1]
        if classified.size and unclassified.size:
            labels[unclassified] = labels[classified[np.argmin(distance.cdist(X[unclassified], X[classified]), axis=1)]]
        elif not classified.size:
            nearest_core = int(np.argmin(distance.cdist([X[ball].mean(axis=0)], core_centers)))
            labels[ball] = labels[balls[core[nearest_core]]]
    return labels


class GBDBSCAN(Algorithm):
    def __init__(self, config: GBDBSCANConfig) -> None:
        self.config = config

    def fit(self, X: np.ndarray) -> "GBDBSCAN":
        values = _validate_features(X)
        n_neighbors = self.config.resolve_n_neighbors(values.shape[0])
        if not 1 <= n_neighbors <= values.shape[0]:
            raise ValueError("official K must be in [1, n_samples]")
        balls = _generate_granular_balls(values, n_neighbors, self.config)
        centers, radii = _centers_and_radii(values, balls)
        core, non_core, threshold = _split_core_balls(radii, self.config.ratio)
        core_labels = _cluster_core_balls(core, centers, radii)
        labels = np.full(values.shape[0], -1, dtype=int)
        for local_index, ball_index in enumerate(core):
            labels[balls[ball_index]] = core_labels[local_index]
        self.labels_ = _assign_non_core_balls(values, labels, balls, core, non_core, centers)
        if np.any(self.labels_ == -1):
            raise RuntimeError("GB-DBSCAN left samples unassigned")
        self.n_neighbors_, self.granular_balls_, self.centers_, self.radii_ = n_neighbors, balls, centers, radii
        self.radius_threshold_, self.core_ball_indices_, self.non_core_ball_indices_, self.core_ball_labels_ = threshold, core, non_core, core_labels
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)
