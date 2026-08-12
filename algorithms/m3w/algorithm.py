from __future__ import annotations

"""Unified adapter around the released M3W Python implementation."""

from contextlib import redirect_stdout
from dataclasses import asdict
import io

import numpy as np

from core.algorithm import Algorithm

from . import BorderPeel as source_border_peel
from . import border_tools as source_border_tools

from .config import M3WConfig


class _OfficialUnionFind:
    """Union-find semantics supplied by M3W's declared PyPI dependency.

    The release imports ``python-algorithms==0.2.2``.  The local copy replaced
    that dependency with a size-weighted implementation, which changes tie
    representatives and exposes ``count`` as a property.  This small adapter
    preserves the official dependency's rank-based union and ``count()``
    method without adding an obsolete package to the Benchmark environment.
    """

    def __init__(self, size: int) -> None:
        self._id = list(range(size))
        self._count = size
        self._rank = [0] * size

    def find(self, item: int) -> int:
        while item != self._id[item]:
            self._id[item] = self._id[self._id[item]]
            item = self._id[item]
        return item

    def count(self) -> int:
        return self._count

    def connected(self, first: int, second: int) -> bool:
        return self.find(first) == self.find(second)

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return

        self._count -= 1
        if self._rank[first_root] < self._rank[second_root]:
            self._id[first_root] = second_root
        elif self._rank[first_root] > self._rank[second_root]:
            self._id[second_root] = first_root
        else:
            self._id[second_root] = first_root
            self._rank[first_root] += 1


def _validate_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"X must not be empty, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    return values


def _membership_to_labels(
    membership: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Reproduce the hard-label conversion in official ``run_m3w.py``."""

    values = np.asarray(membership)
    if values.ndim != 2 or values.shape[1] != n_samples:
        raise RuntimeError(
            "M3W returned an invalid membership matrix with shape "
            f"{values.shape}; expected (*, {n_samples})"
        )

    labels = np.full(n_samples, -1, dtype=int)
    nonzero_rows, nonzero_columns = np.nonzero(values)
    for column in range(n_samples):
        rows = nonzero_rows[nonzero_columns == column]
        if rows.size:
            # The official command-line script selects the numerically largest
            # membership row when a sample belongs to multiple clusters.
            labels[column] = int(np.max(rows))
    return labels


class M3W(Algorithm):
    """Source-faithful adapter for Multistep Three-Way Clustering.

    M3W estimates its own number of clusters and contains no random operation
    for fixed inputs and parameters.  Consequently the Benchmark
    ``n_clusters`` and ``seed`` factory arguments are intentionally ignored.
    """

    def __init__(self, config: M3WConfig | None = None) -> None:
        self.config = config or M3WConfig()

    def fit(self, X: np.ndarray) -> "M3W":
        values = _validate_features(X)
        if values.shape[0] <= self.config.k:
            raise ValueError(
                f"M3W requires more samples than k={self.config.k}; "
                f"received {values.shape[0]} samples"
            )

        # BorderPeel imports ``border_tools`` as a module, so replacing this
        # one compatibility symbol also affects the source's dynamic_3w path.
        source_border_tools.UF = _OfficialUnionFind
        lambda_estimate = float(
            source_border_tools.estimate_lambda(values, self.config.k)
        )
        estimator = source_border_peel.BorderPeel(
            method="exp_local_scaling",
            max_iterations=self.config.levels,
            mean_border_eps=self.config.mean_border_eps,
            k=self.config.k,
            plot_debug_output_dir=None,
            min_cluster_size=self.config.min_cluster_size,
            dist_threshold=lambda_estimate,
            convergence_constant=self.config.convergence_constant,
            link_dist_expansion_factor=(
                self.config.link_distance_expansion_factor
            ),
            verbose=False,
            border_precentile=self.config.border_percentile,
            stopping_precentile=self.config.stopping_percentile,
            merge_core_points=self.config.merge_core_points,
            core_points_threshold=self.config.core_points_threshold,
            dvalue_threshold=self.config.dvalue_threshold,
        )

        # The source Stopwatch prints even with verbose=False.  Silence only
        # those diagnostics so Benchmark result streams remain machine-readable.
        with redirect_stdout(io.StringIO()):
            membership = np.asarray(
                estimator.fit_predict(values),
                dtype=float,
            )

        labels = _membership_to_labels(membership, values.shape[0])
        membership_counts = np.count_nonzero(membership, axis=0)

        self.labels_ = labels
        self.membership_ = membership
        self.lambda_ = lambda_estimate
        self.detected_n_clusters_ = int(membership.shape[0])
        self.multi_membership_count_ = int(
            np.count_nonzero(membership_counts > 1)
        )
        self.noise_count_ = int(np.count_nonzero(membership_counts == 0))
        self.source_estimator_ = estimator
        self.core_points_ = estimator.core_points
        self.core_points_indices_ = estimator.core_points_indices
        self.non_merged_core_points_ = estimator.non_merged_core_points
        self.data_sets_by_iterations_ = estimator.data_sets_by_iterations
        self.associations_ = estimator.associations
        self.link_thresholds_ = estimator.link_thresholds
        self.border_values_per_iteration_ = (
            estimator.border_values_per_iteration
        )
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)
