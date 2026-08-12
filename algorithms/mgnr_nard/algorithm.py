from __future__ import annotations

"""Unified Algorithm adapter for the four MGNR/NARD clustering paths."""

from dataclasses import asdict

import numpy as np

from core.algorithm import Algorithm

from .backends import (
    dadc_nard,
    dbscan_nard,
    dpeak_nard,
    hcdc_nard,
)
from .config import NARDConfig
from .core import build_nard_state


def _validate_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {values.shape}")
    if values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(
            "X must contain at least two samples and one feature, "
            f"got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    return values


class MGNRNARD(Algorithm):
    """Source-first implementation of one NARD clustering application.

    The paper's four applications infer their own cluster structure. The
    Benchmark factory's known ``n_clusters`` and repeated ``seed`` values are
    therefore intentionally ignored.
    """

    def __init__(self, config: NARDConfig) -> None:
        self.config = config

    def fit(self, X: np.ndarray) -> "MGNRNARD":
        values = _validate_features(X)
        state = build_nard_state(
            values,
            self.config.radius_detection_factor,
        )
        if state.ball_centers.shape[0] < 2:
            raise ValueError(
                "NARD requires at least two granular-ball centers"
            )

        if self.config.backend == "dpeak":
            ball_labels = dpeak_nard(state)
        elif self.config.backend == "dbscan":
            ball_labels = dbscan_nard(
                state,
                self.config.dbscan_core_factor,
            )
        elif self.config.backend == "dadc":
            ball_labels = dadc_nard(state)
        else:
            ball_labels = hcdc_nard(
                state,
                self.config.hcdc_small_cluster_fraction,
            )
        ball_labels = np.asarray(ball_labels, dtype=int)
        if ball_labels.shape != (len(state.granular_balls),):
            raise RuntimeError(
                f"{self.config.backend}-NARD returned "
                f"{ball_labels.shape} labels for "
                f"{len(state.granular_balls)} granular balls"
            )

        labels = np.empty(values.shape[0], dtype=int)
        for ball, label in zip(state.granular_balls, ball_labels):
            labels[ball.sample_indices] = label

        if self.config.backend == "dbscan":
            cluster_labels = ball_labels[ball_labels >= 0]
            noise_count = int(np.count_nonzero(labels == -1))
        elif self.config.backend == "hcdc":
            cluster_labels = ball_labels[ball_labels > 0]
            noise_count = int(np.count_nonzero(labels == 0))
        else:
            cluster_labels = ball_labels
            noise_count = 0

        self.labels_ = labels
        self.ball_labels_ = ball_labels
        self.granular_balls_ = state.granular_balls
        self.ball_centers_ = state.ball_centers
        self.distance_matrix_ = state.distance_matrix
        self.nearest_neighbors_ = state.nearest_neighbors
        self.reverse_neighbors_ = state.reverse_neighbors
        self.natural_neighbors_ = state.natural_neighbors
        self.multi_granularity_neighbors_ = (
            state.multi_granularity_neighbors
        )
        self.sample_distribution_groups_ = (
            state.sample_distribution_groups
        )
        self.sample_distribution_count_ = (
            state.sample_distribution_count
        )
        self.expanded_neighbors_ = state.expanded_neighbors
        self.nard_ = state.density
        self.nard_centers_ = state.density_centers
        self.detected_n_clusters_ = int(np.unique(cluster_labels).size)
        self.noise_count_ = noise_count
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)

