from __future__ import annotations

"""Source-compatible Structure-Aware Granular-Ball Clustering (SAGBC)."""

from collections import defaultdict
from dataclasses import asdict

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from core.algorithm import Algorithm as BenchmarkAlgorithm

from .config import SAGBCConfig


def _split_ball(sampled: np.ndarray, ball: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = sampled[ball]
    distances = squareform(pdist(data))
    first_seed, second_seed = np.unravel_index(np.argmax(distances), distances.shape)
    first_mask = np.linalg.norm(data - data[first_seed], axis=1) < np.linalg.norm(data - data[second_seed], axis=1)
    first, second = ball[first_mask], ball[~first_mask]
    if not first.size or not second.size:
        raise RuntimeError("SAGBC cannot split a ball of identical samples")
    return first, second


def _generate_anchors(X: np.ndarray, config: SAGBCConfig) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    rng = np.random.RandomState(config.random_state)
    sampled = X[rng.choice(X.shape[0], config.sample_size, replace=False)]
    balls: tuple[np.ndarray, ...] = (np.arange(config.sample_size),)
    while True:
        updated = tuple(child for ball in balls for child in (_split_ball(sampled, ball) if ball.size > config.max_ball_size else (ball,)))
        if len(updated) == len(balls):
            return np.asarray([sampled[ball].mean(axis=0) for ball in updated]), updated
        balls = updated


def _affiliation(X: np.ndarray, anchors: np.ndarray, neighbor_count: int) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    if anchors.shape[0] < neighbor_count:
        raise ValueError("SAGBC generated fewer anchors than neighbor_count")
    distances, indices = NearestNeighbors(n_neighbors=neighbor_count).fit(anchors).kneighbors(X)
    bandwidth = float(np.mean(distances))
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("SAGBC Gaussian bandwidth is not positive")
    weights = np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))
    weights[weights == 0] = np.finfo(float).eps
    covered: list[list[int]] = [[] for _ in range(anchors.shape[0])]
    for anchor_id, point_id in zip(indices.reshape(-1), np.repeat(np.arange(X.shape[0]), neighbor_count)):
        covered[int(anchor_id)].append(int(point_id))
    return indices, weights, tuple(np.asarray(points, dtype=int) for points in covered)


def _radii(X: np.ndarray, anchors: np.ndarray, indices: np.ndarray, weights: np.ndarray, covered: tuple[np.ndarray, ...]) -> np.ndarray:
    radii = np.zeros(anchors.shape[0])
    for anchor_id, points in enumerate(covered):
        if points.size:
            columns = np.argmax(indices[points] == anchor_id, axis=1)
            radii[anchor_id] = np.average(np.linalg.norm(X[points] - anchors[anchor_id], axis=1), weights=weights[points, columns])
    return radii


def _edges(anchors: np.ndarray, radii: np.ndarray, covered: tuple[np.ndarray, ...], scale: float) -> tuple[tuple[int, int, float], ...]:
    tree, sets = KDTree(anchors), tuple(set(points.tolist()) for points in covered)
    result: list[tuple[int, int, float]] = []
    for first, anchor in enumerate(anchors):
        for second in tree.query_ball_point(anchor, r=radii[first] * scale):
            if first >= second:
                continue
            shared = len(sets[first] & sets[second])
            if shared:
                result.append((first, int(second), 1.0 - shared / min(len(sets[first]), len(sets[second]))))
    return tuple(result)


def _components(edges: tuple[tuple[int, int, float], ...]) -> tuple[tuple[int, ...], ...]:
    involved = sorted({node for first, second, _ in edges for node in (first, second)})
    parent = {node: node for node in involved}
    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for first, second, _ in sorted(edges, key=lambda item: item[2]):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root
    groups: dict[int, list[int]] = defaultdict(list)
    for node in involved:
        groups[find(node)].append(node)
    return tuple(tuple(group) for group in groups.values())


def _assign(n_samples: int, components: tuple[tuple[int, ...], ...], covered: tuple[np.ndarray, ...], indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    labels, best_weights = np.full(n_samples, -1, dtype=int), np.zeros(n_samples)
    for label, component in enumerate(components):
        for anchor_id in component:
            points = covered[anchor_id]
            columns = np.argmax(indices[points] == anchor_id, axis=1)
            improved = weights[points, columns] > best_weights[points]
            labels[points[improved]], best_weights[points[improved]] = label, weights[points[improved], columns[improved]]
    return labels


class SAGBC(BenchmarkAlgorithm):
    """SAGBC adapter; connected components determine its cluster count."""

    def __init__(self, config: SAGBCConfig) -> None:
        self.config = config

    def fit(self, X: np.ndarray) -> "SAGBC":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] < self.config.sample_size or not np.all(np.isfinite(values)):
            raise ValueError("SAGBC requires finite X with at least sample_size rows")
        anchors, self.sampled_balls_ = _generate_anchors(values, self.config)
        indices, weights, covered = _affiliation(values, anchors, self.config.neighbor_count)
        radii = _radii(values, anchors, indices, weights, covered)
        edges = _edges(anchors, radii, covered, self.config.search_radius_scale)
        components = _components(edges)
        self.labels_ = _assign(values.shape[0], components, covered, indices, weights)
        self.anchors_, self.radii_, self.edges_, self.components_ = anchors, radii, edges, components
        self.detected_n_clusters_ = len(components)
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)
