from __future__ import annotations

"""EGBDPM: Efficient Granular-ball Density Peaks Clustering for Manifold Data."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from core.algorithm import Algorithm as BenchmarkAlgorithm

from .config import EGBDPMConfig


@dataclass(frozen=True, slots=True)
class GranularBall:
    sample_indices: np.ndarray
    center: np.ndarray
    radius: float
    quality: float


def _quality(points: np.ndarray) -> float:
    if points.shape[0] <= 2:
        return float("inf")
    distances = np.linalg.norm(points - np.mean(points, axis=0), axis=1)
    compactness = float(np.mean(distances))
    return compactness * (1.0 + float(np.std(distances)) / (compactness + 1e-10))


def _ball(X: np.ndarray, indices: np.ndarray) -> GranularBall:
    points = X[indices]; center = np.mean(points, axis=0)
    return GranularBall(indices, center, float(np.max(np.linalg.norm(points - center, axis=1))), _quality(points))


def _pca_split(X: np.ndarray, ball: GranularBall) -> tuple[np.ndarray, np.ndarray]:
    points = X[ball.sample_indices]; count, dimension = points.shape
    if count <= 2:
        return ball.sample_indices[:1], ball.sample_indices[1:]
    centered = points - np.mean(points, axis=0)
    if dimension >= 100:
        target = min(50, dimension // 4, count // 2)
        generator = np.random.RandomState(42)
        projection = generator.randn(dimension, target) / np.sqrt(target)
        left, _, _ = np.linalg.svd((centered @ projection).T, full_matrices=False)
        component = projection @ left[:, 0]; component /= np.linalg.norm(component)
    else:
        component = np.linalg.svd(centered.T, full_matrices=False)[0][:, 0]
    mask = centered @ component <= np.median(centered @ component)
    return ball.sample_indices[mask], ball.sample_indices[~mask]


def _initial_balls(X: np.ndarray) -> list[GranularBall]:
    count = int(np.ceil(np.sqrt(X.shape[0])))
    model = KMeans(n_clusters=count, random_state=42, n_init=1, init="k-means++", max_iter=1) if X.shape[0] < 7000 else MiniBatchKMeans(n_clusters=count, random_state=42, n_init=1, init="k-means++", max_iter=1, batch_size=1000)
    labels = model.fit_predict(X)
    return [_ball(X, np.flatnonzero(labels == label)) for label in range(count) if np.any(labels == label)]


def _quality_refine(X: np.ndarray, balls: list[GranularBall]) -> list[GranularBall]:
    changed = True
    while changed:
        changed = False; result: list[GranularBall] = []
        for ball in balls:
            if ball.sample_indices.size <= 8:
                result.append(ball); continue
            first, second = _pca_split(X, ball)
            if not first.size or not second.size:
                result.append(ball); continue
            left, right = _ball(X, first), _ball(X, second)
            weighted = (left.sample_indices.size * left.quality + right.sample_indices.size * right.quality) / ball.sample_indices.size
            if weighted < ball.quality:
                result.extend((left, right)); changed = True
            else: result.append(ball)
        balls = result
    return balls


def _radius_refine(X: np.ndarray, balls: list[GranularBall]) -> list[GranularBall]:
    radii = np.asarray([ball.radius for ball in balls]); q1, q3 = np.percentile(radii, [25, 75])
    threshold = float(np.clip(q3 + 1.5 * (q3 - q1), np.median(radii), np.percentile(radii, 95)))
    changed = True
    while changed:
        changed = False; result: list[GranularBall] = []
        for ball in balls:
            if ball.sample_indices.size <= 2 or ball.radius <= threshold:
                result.append(ball); continue
            first, second = _pca_split(X, ball)
            if first.size and second.size:
                result.extend((_ball(X, first), _ball(X, second))); changed = True
            else: result.append(ball)
        balls = result
    return balls


def generate_granular_balls(X: np.ndarray) -> tuple[GranularBall, ...]:
    balls = _radius_refine(X, _quality_refine(X, _initial_balls(X)))
    assigned = np.concatenate([ball.sample_indices for ball in balls])
    if not np.array_equal(np.sort(assigned), np.arange(X.shape[0])):
        raise RuntimeError("EGBDPM granular balls do not partition all samples")
    return tuple(balls)


def _sphere_arc(first: np.ndarray, second: np.ndarray, points: np.ndarray) -> float:
    matrix = np.hstack((2.0 * points, np.ones((points.shape[0], 1))))
    solution, _, _, _ = np.linalg.lstsq(matrix, np.sum(points**2, axis=1), rcond=None)
    center = solution[:-1]; radius = float(np.sqrt(max(0.0, np.sum(center**2) + solution[-1])))
    first_vector, second_vector = first - center, second - center
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(second_vector)
    if denominator == 0.0 or not np.isfinite(radius): return float(np.linalg.norm(first - second))
    return float(radius * np.arccos(np.clip(np.dot(first_vector, second_vector) / denominator, -1.0, 1.0)))


def _sphere_geodesic(centers: np.ndarray, k: int) -> np.ndarray:
    count = centers.shape[0]; neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(centers).kneighbors(centers, return_distance=False)[:, 1:]
    graph = lil_matrix((count, count))
    for index, adjacent in enumerate(neighbors):
        local = centers[np.concatenate(([index], adjacent))]
        for other in adjacent:
            distance = _sphere_arc(centers[index], centers[other], local)
            graph[index, other] = graph[other, index] = distance
    # Official source preserves Dijkstra's inf values for disconnected pairs.
    return dijkstra(graph.tocsr(), directed=False, return_predecessors=False)


def run_egbdpm(X: np.ndarray, n_clusters: int, config: EGBDPMConfig) -> tuple[np.ndarray, tuple[GranularBall, ...]]:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or not 2 <= n_clusters <= values.shape[0]:
        raise ValueError("X must be finite 2D data and n_clusters must be in [2, n_samples]")
    balls = generate_granular_balls(values); count = len(balls)
    if n_clusters > count: raise ValueError(f"n_clusters={n_clusters} exceeds generated granular balls ({count})")
    k = min(config.k_neighbors, count - 1)
    if k < 1: raise ValueError("EGBDPM requires at least two granular balls")
    centers = np.asarray([ball.center for ball in balls]); qualities = np.asarray([ball.quality for ball in balls]); sizes = np.asarray([ball.sample_indices.size for ball in balls])
    base_density = np.where(sizes > 2, qualities, 1000.0)
    neighbor_ids = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(centers).kneighbors(centers, return_distance=False)
    density = (k + 1.0) / np.sum(base_density[neighbor_ids], axis=1)
    distance = _sphere_geodesic(centers, k); ordered = np.argsort(-density); delta = np.zeros(count); parent = np.zeros(count, dtype=int)
    for position in range(1, count):
        current, higher = ordered[position], ordered[:position]
        nearest = int(np.argmin(distance[current, higher])); delta[current] = distance[current, higher[nearest]]; parent[current] = higher[nearest]
    delta[ordered[0]] = np.max(delta)
    if np.max(delta) < 1.0: delta *= 10.0
    peaks = np.argsort(-(density * delta))[:n_clusters]; ball_labels = np.full(count, -1, dtype=int); ball_labels[peaks] = np.arange(n_clusters)
    for index in ordered:
        if ball_labels[index] < 0: ball_labels[index] = ball_labels[parent[index]]
    labels = np.empty(values.shape[0], dtype=int)
    for ball, label in zip(balls, ball_labels): labels[ball.sample_indices] = label
    return labels, balls


class EGBDPM(BenchmarkAlgorithm):
    def __init__(self, config: EGBDPMConfig, n_clusters: int) -> None: self.config, self.n_clusters = config, n_clusters
    def fit(self, X: np.ndarray) -> "EGBDPM":
        self.labels_, self.granular_balls_ = run_egbdpm(X, self.n_clusters, self.config); return self
    def get_params(self) -> dict[str, object]: return {"n_clusters": self.n_clusters, **asdict(self.config)}
