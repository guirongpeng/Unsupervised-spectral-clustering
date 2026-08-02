from __future__ import annotations

"""Gaussian-PDMF attribute reduction for MY-V1.

This module inherits the Gaussian-PDMF construction and the three-level
entropy definitions described in the research specification.  The reduction
Global attributes retain the MY-V0 entropy ranking.  Local granular-ball
attributes combine normalized ranks of entropy inner significance and sparse
Gaussian-PDMF graph importance.  Neither stage uses class or pseudo labels.
"""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr
from sklearn.neighbors import NearestNeighbors

from .common.preprocessing import minmax_scale_like_matlab


_H0 = 0.3702632681
_SHAPE_GRID_SIZE = 513
_SIMILARITY_SHAPE_GRID_SIZE = 1025
_QUADRATURE_ORDER = 96


@lru_cache(maxsize=2)
def _shape_entropy_lookup(
    limit: float = 1.0,
    grid_size: int = _SHAPE_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic lookup table for the Gaussian shape entropy."""

    nodes, weights = leggauss(_QUADRATURE_ORDER)
    t = 0.5 * (nodes + 1.0)
    quadrature_weights = 0.5 * weights
    transformed = np.tan(np.pi * t - np.pi / 2.0)

    mu_grid = np.linspace(-limit, limit, grid_size)
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


def gaussian_shape_entropy_difference(mu: np.ndarray) -> np.ndarray:
    """Evaluate ``H(mu)`` for clarity differences in ``[-2, 2]``."""

    values = np.asarray(mu, dtype=float)
    grid, entropy = _shape_entropy_lookup(2.0, _SIMILARITY_SHAPE_GRID_SIZE)
    interpolated = np.interp(np.clip(values, -2.0, 2.0).reshape(-1), grid, entropy)
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


def _gaussian_pdmf_components(
    X: np.ndarray,
    neighbors: int = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return entropy, active mask, spreads and clarity for Gaussian-PDMFs."""

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
    return sample_entropies, nonconstant, spread_left, spread_right, clarity


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

    sample_entropies, nonconstant, _, _, _ = _gaussian_pdmf_components(
        X, neighbors=neighbors, epsilon=epsilon
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


def _attribute_scores_from_sample_entropies(
    sample_entropies: np.ndarray,
    nonconstant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute inner significance from a precomputed sample-entropy matrix."""

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
        scores[active[0]] = full_entropy
    else:
        leave_one_out = full_log_intensity[:, None] - log_intensity
        leave_one_out_entropies = _entropy_of_log_weights(leave_one_out, axis=0)
        scores[active] = full_entropy - leave_one_out_entropies
    return scores, single_entropies, full_entropy


def gaussian_pdmf_attribute_scores(
    X: np.ndarray,
    neighbors: int = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Gaussian-PDMF inner-significance scores for all attributes.

    For the complete non-constant set ``C``, each score is
    ``E(C) - E(C - {a})``.  Products in the subset entropy are represented as
    sums of logarithms, so high-dimensional data do not underflow.
    """

    sample_entropies, nonconstant = gaussian_pdmf_sample_entropies(
        X, neighbors=neighbors, epsilon=epsilon
    )
    return _attribute_scores_from_sample_entropies(sample_entropies, nonconstant)


def _union_knn_edges(X: np.ndarray, neighbors: int) -> np.ndarray:
    """Return unique undirected edges from the union of directed KNN lists."""

    values = np.asarray(X, dtype=float)
    n_samples = values.shape[0]
    if n_samples < 2:
        return np.empty((0, 2), dtype=int)
    local_neighbors = min(int(neighbors), n_samples - 1)
    model = NearestNeighbors(
        n_neighbors=local_neighbors + 1,
        metric="euclidean",
    ).fit(values)
    neighbor_indices = model.kneighbors(values, return_distance=False)

    directed: list[tuple[int, int]] = []
    for sample_index, candidates in enumerate(neighbor_indices):
        selected = candidates[candidates != sample_index][:local_neighbors]
        directed.extend((sample_index, int(neighbor)) for neighbor in selected)
    if not directed:
        return np.empty((0, 2), dtype=int)

    edges = np.asarray(directed, dtype=int)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _edge_gaussian_pdmf_similarities(
    X: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    edges: np.ndarray,
    feature_indices: np.ndarray,
    similarity_lambda: float,
) -> np.ndarray:
    """Compute per-attribute Gaussian-PDMF similarities on sparse edges."""

    first = edges[:, 0]
    second = edges[:, 1]
    features = np.asarray(feature_indices, dtype=int)
    core_similarity = np.exp(
        -np.abs(X[first[:, None], features] - X[second[:, None], features])
    )
    shape_similarity = gaussian_shape_entropy_difference(
        clarity[first[:, None], features] - clarity[second[:, None], features]
    )
    left_similarity = np.exp(
        -np.abs(
            spread_left[first[:, None], features]
            - spread_left[second[:, None], features]
        )
    ) * shape_similarity
    right_similarity = np.exp(
        -np.abs(
            spread_right[first[:, None], features]
            - spread_right[second[:, None], features]
        )
    ) * shape_similarity
    return similarity_lambda * core_similarity + (
        (1.0 - similarity_lambda)
        * (left_similarity + right_similarity)
        / (2.0 * _H0)
    )


def _graph_attribute_scores_from_components(
    X: np.ndarray,
    nonconstant: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    graph_neighbors: int,
    similarity_lambda: float,
    epsilon: float,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-attribute-out sparse-graph importance scores."""

    scores = np.full(X.shape[1], -np.inf, dtype=float)
    active = np.flatnonzero(nonconstant)
    edges = _union_knn_edges(X, graph_neighbors)
    if active.size == 0 or edges.size == 0:
        return scores, edges
    if active.size == 1:
        scores[active[0]] = 1.0
        return scores, edges

    full_graph = np.zeros(edges.shape[0], dtype=float)
    for start in range(0, active.size, block_size):
        block = active[start : start + block_size]
        similarities = _edge_gaussian_pdmf_similarities(
            X,
            spread_left,
            spread_right,
            clarity,
            edges,
            block,
            similarity_lambda,
        )
        full_graph += np.sum(similarities, axis=1)
    full_graph /= active.size
    denominator = float(np.dot(full_graph, full_graph) + epsilon)
    removal_scale = float((active.size - 1) ** 2)

    for start in range(0, active.size, block_size):
        block = active[start : start + block_size]
        similarities = _edge_gaussian_pdmf_similarities(
            X,
            spread_left,
            spread_right,
            clarity,
            edges,
            block,
            similarity_lambda,
        )
        differences = similarities - full_graph[:, None]
        scores[block] = np.sum(differences * differences, axis=0) / (
            removal_scale * denominator
        )
    return scores, edges


def gaussian_pdmf_graph_attribute_scores(
    X: np.ndarray,
    neighbors: int = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int = 5,
    similarity_lambda: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MY-V1 sparse-graph importance and its fixed union-KNN edges."""

    if isinstance(graph_neighbors, bool) or not isinstance(graph_neighbors, int):
        raise TypeError("graph_neighbors must be an integer")
    if graph_neighbors < 1:
        raise ValueError("graph_neighbors must be at least 1")
    if not np.isfinite(similarity_lambda) or not 0 < similarity_lambda < 1:
        raise ValueError("similarity_lambda must be a finite number in (0, 1)")

    values = np.asarray(X, dtype=float)
    _, nonconstant, spread_left, spread_right, clarity = _gaussian_pdmf_components(
        values, neighbors=neighbors, epsilon=epsilon
    )
    return _graph_attribute_scores_from_components(
        values,
        nonconstant,
        spread_left,
        spread_right,
        clarity,
        graph_neighbors,
        similarity_lambda,
        epsilon,
    )


def _normalized_descending_ranks(
    scores: np.ndarray,
    secondary_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Map deterministic descending integer ranks to the interval ``[0, 1]``."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 1:
        return np.ones(1, dtype=float)
    indices = np.arange(values.size)
    if secondary_scores is None:
        order = np.lexsort((indices, -values))
    else:
        secondary = np.asarray(secondary_scores, dtype=float).reshape(-1)
        order = np.lexsort((indices, -secondary, -values))
    normalized = np.empty(values.size, dtype=float)
    normalized[order] = np.linspace(1.0, 0.0, values.size)
    return normalized


def select_features_by_gaussian_pdmf(
    X: np.ndarray,
    n_features: int,
    neighbors: int = 5,
    epsilon: float = 1e-8,
    *,
    scale_selected: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select an exact number of attributes using entropy significance."""

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


def select_local_features_by_gaussian_pdmf_graph(
    X: np.ndarray,
    p2: int,
    neighbors: int = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int = 5,
    similarity_lambda: float = 0.5,
    ranking_cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select local attributes by equal entropy and graph rank importance."""

    values = np.asarray(X, dtype=float)
    local_count = max(1, min(int(p2), values.shape[1]))
    if isinstance(graph_neighbors, bool) or not isinstance(graph_neighbors, int):
        raise TypeError("graph_neighbors must be an integer")
    if graph_neighbors < 1:
        raise ValueError("graph_neighbors must be at least 1")
    if not np.isfinite(similarity_lambda) or not 0 < similarity_lambda < 1:
        raise ValueError("similarity_lambda must be a finite number in (0, 1)")

    if ranking_cache is not None and {"order", "combined_scores"} <= ranking_cache.keys():
        order = np.asarray(ranking_cache["order"], dtype=int).reshape(-1).copy()
        combined_scores = (
            np.asarray(ranking_cache["combined_scores"], dtype=float)
            .reshape(-1)
            .copy()
        )
        if order.size != values.shape[1] or combined_scores.size != values.shape[1]:
            raise ValueError("cached local ranking size must match X features")
    else:
        sample_entropies, nonconstant, spread_left, spread_right, clarity = (
            _gaussian_pdmf_components(values, neighbors=neighbors, epsilon=epsilon)
        )
        entropy_scores, single_entropies, _ = _attribute_scores_from_sample_entropies(
            sample_entropies, nonconstant
        )
        graph_scores, _ = _graph_attribute_scores_from_components(
            values,
            nonconstant,
            spread_left,
            spread_right,
            clarity,
            graph_neighbors,
            similarity_lambda,
            epsilon,
        )
        entropy_ranks = _normalized_descending_ranks(entropy_scores, single_entropies)
        graph_ranks = _normalized_descending_ranks(graph_scores)
        combined_scores = 0.5 * (entropy_ranks + graph_ranks)
        feature_indices = np.arange(values.shape[1])
        order = np.lexsort(
            (
                feature_indices,
                -single_entropies,
                -graph_scores,
                -entropy_scores,
                -combined_scores,
            )
        )
        if ranking_cache is not None:
            ranking_cache["order"] = order.copy()
            ranking_cache["combined_scores"] = combined_scores.copy()

    selected_indices = order[:local_count]
    selected = minmax_scale_like_matlab(values[:, selected_indices])
    return selected, selected_indices, combined_scores
