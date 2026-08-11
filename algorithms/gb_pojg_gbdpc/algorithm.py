from __future__ import annotations

"""Source-compatible GB-POJG + GBDPC clustering adapter."""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import cdist

from core.algorithm import Algorithm as BenchmarkAlgorithm

from .config import GBPOJGGBDPCConfig
from .generator import GBPOJGGenerationResult, GranularBall, generate_granular_balls


@dataclass(frozen=True)
class GBPOJGGBDPCResult:
    labels: np.ndarray
    densities: np.ndarray
    delta_distances: np.ndarray
    nearest_higher: np.ndarray
    decision_values: np.ndarray
    center_ball_indices: np.ndarray
    generation: GBPOJGGenerationResult


def _gbdpc_labels(
    granular_balls: tuple[GranularBall, ...], n_samples: int, n_clusters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Faithful translation of ``ClusteringMethod.GBDPC`` from MATLAB."""

    ball_count = len(granular_balls)
    if n_clusters > ball_count:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds generated granular balls ({ball_count})"
        )
    densities = np.zeros(ball_count, dtype=float)
    for index, ball in enumerate(granular_balls):
        if ball.max_radius != 0.0:
            densities[index] = ball.size / (ball.max_radius**2 * ball.average_radius)

    density_order = np.argsort(-densities, kind="stable")
    sorted_densities = densities[density_order]
    sorted_balls = tuple(granular_balls[index] for index in density_order)
    centers = np.asarray([ball.center for ball in sorted_balls])
    distances = cdist(centers, centers, metric="euclidean")

    delta_distances = np.zeros(ball_count, dtype=float)
    nearest_higher = np.arange(ball_count, dtype=int)
    for index in range(1, ball_count):
        candidates = distances[index, :index]
        nearest_higher[index] = int(np.argmin(candidates))
        delta_distances[index] = candidates[nearest_higher[index]]
    delta_distances[0] = float(np.max(delta_distances))

    decision_values = sorted_densities * delta_distances
    center_indices = np.argsort(-decision_values, kind="stable")[:n_clusters]
    ball_labels = np.full(ball_count, -1, dtype=int)
    for cluster_id, index in enumerate(center_indices, start=1):
        ball_labels[index] = cluster_id
    for index in range(ball_count):
        if ball_labels[index] == -1:
            ball_labels[index] = ball_labels[nearest_higher[index]]

    labels = np.zeros(n_samples, dtype=int)
    for ball, label in zip(sorted_balls, ball_labels):
        labels[ball.sample_indices] = label
    return (
        labels,
        sorted_densities,
        delta_distances,
        nearest_higher,
        decision_values,
        center_indices,
    )


def run_gb_pojg_gbdpc(
    X: np.ndarray,
    n_clusters: int,
    config: GBPOJGGBDPCConfig,
) -> GBPOJGGBDPCResult:
    """Run the official GB-POJG generation followed by official GBDPC."""

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"X must have shape (n_samples >= 2, n_features >= 1), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    if n_clusters < 2 or n_clusters > values.shape[0]:
        raise ValueError(f"n_clusters must be in [2, {values.shape[0]}]")

    generation = generate_granular_balls(values, config.gamma, config.delta)
    labels, densities, delta_distances, nearest_higher, decision_values, centers = (
        _gbdpc_labels(generation.granular_balls, values.shape[0], n_clusters)
    )
    return GBPOJGGBDPCResult(
        labels=labels,
        densities=densities,
        delta_distances=delta_distances,
        nearest_higher=nearest_higher,
        decision_values=decision_values,
        center_ball_indices=centers,
        generation=generation,
    )


class GBPOJGGBDPC(BenchmarkAlgorithm):
    """Unified Benchmark adapter for official GB-POJG + GBDPC."""

    def __init__(
        self,
        config: GBPOJGGBDPCConfig,
        n_clusters: int,
        random_state: int = 1,
    ) -> None:
        self.config = config
        self.n_clusters = n_clusters
        self.random_state = random_state  # GBDPC is deterministic; retained for the unified API.

    def fit(self, X: np.ndarray) -> "GBPOJGGBDPC":
        self.result_ = run_gb_pojg_gbdpc(X, self.n_clusters, self.config)
        self.labels_ = self.result_.labels
        self.granular_balls_ = self.result_.generation.granular_balls
        self.densities_ = self.result_.densities
        self.decision_values_ = self.result_.decision_values
        return self

    def get_params(self) -> dict[str, object]:
        return {"n_clusters": self.n_clusters, "random_state": self.random_state, **asdict(self.config)}
