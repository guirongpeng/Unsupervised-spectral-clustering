from __future__ import annotations

"""GBSC using the numerical path in the official UCI Python experiment."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering

from core.algorithm import Algorithm as BenchmarkAlgorithm

from .config import GBSCConfig


@dataclass(frozen=True, slots=True)
class GranularBall:
    sample_indices: np.ndarray


@dataclass(frozen=True)
class GBSCResult:
    labels: np.ndarray
    granular_balls: tuple[GranularBall, ...]
    centers: np.ndarray
    radii: np.ndarray
    affinity: np.ndarray
    ball_labels: np.ndarray


def _validate_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"X must have shape (n_samples >= 2, n_features >= 1), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    return values


def _center(X: np.ndarray, ball: GranularBall) -> np.ndarray:
    return np.mean(X[ball.sample_indices], axis=0)


def _radius(X: np.ndarray, ball: GranularBall) -> float:
    points = X[ball.sample_indices]
    return float(np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))


def _density(X: np.ndarray, ball: GranularBall) -> float:
    points = X[ball.sample_indices]
    distance_sum = float(np.sum(np.linalg.norm(points - np.mean(points, axis=0), axis=1)))
    return float(ball.sample_indices.size / distance_sum) if distance_sum else float(ball.sample_indices.size)


def _farthest_pair_split(X: np.ndarray, ball: GranularBall) -> tuple[GranularBall, GranularBall]:
    """Reproduce official ``spilt_ball_2`` including its strict tie branch."""

    points = X[ball.sample_indices]
    gram = points @ points.T
    diagonal = np.diag(gram)
    distances = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
    np.maximum(distances, 0.0, out=distances)
    np.sqrt(distances, out=distances)
    rows, columns = np.where(distances == np.max(distances))
    if rows.size < 2:
        raise RuntimeError("GBSC could not locate a symmetric farthest pair")
    first_endpoint, second_endpoint = int(rows[1]), int(columns[1])
    # Official code uses strict <; tied samples enter the second child.
    first_mask = distances[:, first_endpoint] < distances[:, second_endpoint]
    first = GranularBall(ball.sample_indices[first_mask])
    second = GranularBall(ball.sample_indices[~first_mask])
    if not first.sample_indices.size or not second.sample_indices.size:
        raise RuntimeError("GBSC cannot split a ball of identical samples")
    return first, second


def _quality_division(X: np.ndarray, balls: list[GranularBall], config: GBSCConfig) -> list[GranularBall]:
    while True:
        result: list[GranularBall] = []
        for ball in balls:
            if ball.sample_indices.size >= config.minimum_split_size:
                first, second = _farthest_pair_split(X, ball)
                child_density = (
                    first.sample_indices.size * _density(X, first)
                    + second.sample_indices.size * _density(X, second)
                ) / ball.sample_indices.size
                if child_density > _density(X, ball):
                    result.extend((first, second))
                else:
                    result.append(ball)
            else:
                result.append(ball)
        if len(result) == len(balls):
            return result
        balls = result


def _radius_division(X: np.ndarray, balls: list[GranularBall], config: GBSCConfig) -> list[GranularBall]:
    eligible = [_radius(X, ball) for ball in balls if ball.sample_indices.size >= 2]
    if not eligible:
        return balls
    threshold = config.radius_detection_factor * max(float(np.median(eligible)), float(np.mean(eligible)))
    while True:
        result: list[GranularBall] = []
        for ball in balls:
            if ball.sample_indices.size < 2:
                result.append(ball)
                continue
            first, second = _farthest_pair_split(X, ball)
            if _radius(X, ball) <= threshold:
                result.append(ball)
            else:
                result.extend((first, second))
        if len(result) == len(balls):
            return result
        balls = result


def _generate_granular_balls(X: np.ndarray, config: GBSCConfig) -> tuple[GranularBall, ...]:
    balls = _quality_division(X, [GranularBall(np.arange(X.shape[0], dtype=int))], config)
    balls = _radius_division(X, balls, config)
    assigned = np.concatenate([ball.sample_indices for ball in balls])
    if assigned.size != X.shape[0] or not np.array_equal(np.sort(assigned), np.arange(X.shape[0])):
        raise RuntimeError("GBSC granular balls do not partition all samples")
    return tuple(balls)


def _build_affinity(X: np.ndarray, balls: tuple[GranularBall, ...], sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([_center(X, ball) for ball in balls])
    radii = np.asarray([_radius(X, ball) for ball in balls])
    distance = (cdist(centers, centers) - radii[:, None] - radii[None, :]) / np.sqrt(X.shape[1])
    affinity = np.exp(-(distance**2) / (2.0 * sigma**2))
    np.fill_diagonal(affinity, 0.0)
    return affinity, centers, radii


def run_gbsc(X: np.ndarray, n_clusters: int, config: GBSCConfig, seed: int = 1) -> GBSCResult:
    values = _validate_features(X)
    if n_clusters < 2 or n_clusters > values.shape[0]:
        raise ValueError(f"n_clusters must be in [2, {values.shape[0]}]")
    balls = _generate_granular_balls(values, config)
    if len(balls) < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} exceeds granular balls ({len(balls)})")
    affinity, centers, radii = _build_affinity(values, balls, config.sigma)
    ball_labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=config.spectral_n_init,
        eigen_tol=config.spectral_eigen_tol,
    ).fit_predict(affinity)
    labels = np.full(values.shape[0], -1, dtype=int)
    for ball, label in zip(balls, ball_labels):
        labels[ball.sample_indices] = label
    if np.any(labels < 0):
        raise RuntimeError("GBSC left one or more samples unlabeled")
    return GBSCResult(labels, balls, centers, radii, affinity, np.asarray(ball_labels, dtype=int))


class GBSC(BenchmarkAlgorithm):
    """Unified Benchmark adapter for source-compatible GBSC."""

    def __init__(self, config: GBSCConfig, n_clusters: int, random_state: int = 1) -> None:
        self.config = config
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GBSC":
        self.result_ = run_gbsc(X, self.n_clusters, self.config, self.random_state)
        self.labels_ = self.result_.labels
        self.granular_balls_ = self.result_.granular_balls
        self.ball_centers_ = self.result_.centers
        self.ball_radii_ = self.result_.radii
        self.ball_affinity_ = self.result_.affinity
        return self

    def get_params(self) -> dict[str, object]:
        return {"n_clusters": self.n_clusters, "random_state": self.random_state, **asdict(self.config)}
