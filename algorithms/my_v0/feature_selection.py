from __future__ import annotations

"""Gaussian-PDMF entropy based attribute reduction for MY-V0.

This module inherits the Gaussian-PDMF construction and the three-level
entropy definitions described in the research specification.  The reduction
rule itself is a MY-V0 design: attributes are ranked by their inner
significance with respect to the complete non-constant attribute set.  This
one-pass rule returns the fixed feature counts required by the PLGB-FSC
framework without using class labels or pseudo labels.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from .common.preprocessing import minmax_scale_like_matlab


_H0 = 0.3702632681
_SHAPE_GRID_SIZE = 513
_QUADRATURE_ORDER = 96


@lru_cache(maxsize=1)
def _shape_entropy_lookup() -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic lookup table for the Gaussian shape entropy."""

    nodes, weights = leggauss(_QUADRATURE_ORDER)
    t = 0.5 * (nodes + 1.0)
    quadrature_weights = 0.5 * weights
    transformed = np.tan(np.pi * t - np.pi / 2.0)

    mu_grid = np.linspace(-1.0, 1.0, _SHAPE_GRID_SIZE)
    membership = ndtr(transformed[None, :] - mu_grid[:, None])
    membership = np.clip(membership, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    binary_entropy = -(
        membership * np.log(membership)
        + (1.0 - membership) * np.log1p(-membership)
    )
    entropy_grid = binary_entropy @ quadrature_weights
    return mu_grid, entropy_grid


def gaussian_shape_entropy(mu: np.ndarray) -> np.ndarray:
    """Evaluate ``H(mu)`` for clarity values in ``[-1, 1]``."""

    values = np.asarray(mu, dtype=float)
    grid, entropy = _shape_entropy_lookup()
    interpolated = np.interp(np.clip(values, -1.0, 1.0).reshape(-1), grid, entropy)
    return interpolated.reshape(values.shape)


def _asymmetric_local_spreads(
    X: np.ndarray,
    neighbors: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the paper's left/right local spreads for every attribute."""

    n_samples, n_features = X.shape
    order = np.argsort(X, axis=0, kind="stable")
    sorted_values = np.take_along_axis(X, order, axis=0)
    prefix = np.vstack(
        [np.zeros((1, n_features), dtype=float), np.cumsum(sorted_values, axis=0)]
    )

    positions = np.arange(n_samples)
    left_counts = np.minimum(neighbors, positions)
    right_counts = np.minimum(neighbors, n_samples - 1 - positions)

    left_sorted = np.empty_like(sorted_values)
    has_left = left_counts > 0
    left_means = (
        prefix[positions[has_left]]
        - prefix[positions[has_left] - left_counts[has_left]]
    ) / left_counts[has_left, None]
    left_sorted[has_left] = sorted_values[has_left] - left_means

    right_sorted = np.empty_like(sorted_values)
    has_right = right_counts > 0
    right_means = (
        prefix[positions[has_right] + right_counts[has_right] + 1]
        - prefix[positions[has_right] + 1]
    ) / right_counts[has_right, None]
    right_sorted[has_right] = right_means - sorted_values[has_right]

    # At each boundary the unavailable side is copied from the available side.
    left_sorted[~has_left] = right_sorted[~has_left]
    right_sorted[~has_right] = left_sorted[~has_right]
    left_sorted = np.maximum(left_sorted, epsilon)
    right_sorted = np.maximum(right_sorted, epsilon)

    left = np.empty_like(left_sorted)
    right = np.empty_like(right_sorted)
    np.put_along_axis(left, order, left_sorted, axis=0)
    np.put_along_axis(right, order, right_sorted, axis=0)
    return left, right


def gaussian_pdmf_sample_entropies(
    X: np.ndarray,
    neighbors: int = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the sample-level Gaussian-PDMF entropy matrix.

    Returns the ``n_samples x n_features`` entropy matrix and a Boolean mask
    identifying non-constant attributes.  Constant attributes are retained in
    the matrix only so callers can preserve a fixed output dimension; they are
    excluded from the theoretical complete set and ranked last.
    """

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one attribute")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 1:
        raise ValueError("neighbors must be a positive integer")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")

    nonconstant = np.ptp(values, axis=0) > 0.0
    raw_left, raw_right = _asymmetric_local_spreads(values, neighbors, epsilon)
    spread_scale = np.maximum(np.maximum(raw_left, raw_right).max(axis=0), epsilon)
    spread_left = np.exp(raw_left / spread_scale[None, :])
    spread_right = np.exp(raw_right / spread_scale[None, :])

    means = values.mean(axis=0)
    population_stds = values.std(axis=0, ddof=0)
    z = np.abs(values - means[None, :]) / (population_stds[None, :] + epsilon)
    clarity = np.clip(1.0 - z, -1.0, 1.0)
    shape_entropy = gaussian_shape_entropy(clarity)

    sample_entropies = shape_entropy * (
        np.exp(-1.0 / spread_left) + np.exp(-1.0 / spread_right)
    ) / (2.0 * _H0)
    sample_entropies = np.clip(
        sample_entropies,
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
    )
    return sample_entropies, nonconstant


def _entropy_of_log_weights(log_weights: np.ndarray, axis: int = 0) -> np.ndarray:
    """Shannon entropy after normalizing positive weights in log space."""

    maxima = np.max(log_weights, axis=axis, keepdims=True)
    shifted = log_weights - maxima
    weights = np.exp(shifted)
    probabilities = weights / np.sum(weights, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    return -np.sum(terms, axis=axis)


def gaussian_pdmf_attribute_scores(
    X: np.ndarray,
    neighbors: int = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return MY-V0 inner-significance scores for all attributes.

    For the complete non-constant set ``C``, each score is
    ``E(C) - E(C - {a})``.  Products in the subset entropy are represented as
    sums of logarithms, so high-dimensional data do not underflow.
    """

    sample_entropies, nonconstant = gaussian_pdmf_sample_entropies(
        X, neighbors=neighbors, epsilon=epsilon
    )
    scores = np.full(sample_entropies.shape[1], -np.inf, dtype=float)
    single_entropies = np.full(sample_entropies.shape[1], -np.inf, dtype=float)
    active = np.flatnonzero(nonconstant)
    if active.size == 0:
        return scores, single_entropies, 0.0

    log_intensity = np.log1p(-sample_entropies[:, active])
    single_entropies[active] = _entropy_of_log_weights(log_intensity, axis=0)
    full_log_intensity = np.sum(log_intensity, axis=1)
    full_entropy = float(_entropy_of_log_weights(full_log_intensity, axis=0))

    if active.size == 1:
        # The research specification fixes E(emptyset) = 0.
        scores[active[0]] = full_entropy
    else:
        leave_one_out = full_log_intensity[:, None] - log_intensity
        leave_one_out_entropies = _entropy_of_log_weights(leave_one_out, axis=0)
        scores[active] = full_entropy - leave_one_out_entropies
    return scores, single_entropies, full_entropy


def select_features_by_gaussian_pdmf(
    X: np.ndarray,
    n_features: int,
    neighbors: int = 5,
    epsilon: float = 1e-8,
    *,
    scale_selected: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select an exact number of attributes using the MY-V0 ranking."""

    values = np.asarray(X, dtype=float)
    if isinstance(n_features, bool) or not isinstance(n_features, int):
        raise TypeError("n_features must be an integer")
    if n_features < 1 or n_features > values.shape[1]:
        raise ValueError(f"n_features must be in [1, {values.shape[1]}]")

    scores, single_entropies, _ = gaussian_pdmf_attribute_scores(
        values, neighbors=neighbors, epsilon=epsilon
    )
    feature_indices = np.arange(values.shape[1])
    # Primary key: descending inner significance.  Secondary key: descending
    # single-attribute entropy.  Original index makes ties reproducible.
    order = np.lexsort((feature_indices, -single_entropies, -scores))
    selected_indices = order[:n_features]
    selected = values[:, selected_indices]
    if scale_selected:
        selected = minmax_scale_like_matlab(selected)
    return selected, selected_indices, scores


def select_global_features_by_gaussian_pdmf(
    X: np.ndarray,
    p1: int,
    neighbors: int = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the global ``p1`` attributes without labels or pseudo labels."""

    return select_features_by_gaussian_pdmf(
        X, p1, neighbors=neighbors, epsilon=epsilon, scale_selected=False
    )


def select_local_features_by_gaussian_pdmf(
    X: np.ndarray,
    p2: int,
    neighbors: int = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select and Min-Max scale local attributes before granular-ball 2-Means."""

    values = np.asarray(X, dtype=float)
    local_count = max(1, min(int(p2), values.shape[1]))
    return select_features_by_gaussian_pdmf(
        values,
        local_count,
        neighbors=neighbors,
        epsilon=epsilon,
        scale_selected=True,
    )
