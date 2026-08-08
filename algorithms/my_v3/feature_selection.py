from __future__ import annotations

"""Gaussian-PDMF attribute reduction for MY-V3.

This module inherits the Gaussian-PDMF construction and the three-level
entropy definitions described in the research specification.  The reduction
Global attributes retain the MY-V0 entropy ranking.  Local granular-ball
attributes combine normalized ranks of entropy inner significance and sparse
Gaussian-PDMF graph importance.  Neither stage uses class or pseudo labels.
"""

import math
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


def _resolve_neighbor_count(
    value: int | float,
    n_samples: int,
    name: str,
) -> int:
    """Resolve a fixed count or a sample-size ratio to an effective K."""

    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an integer count or float ratio")
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} count must be at least 1")
        count = value
    else:
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} ratio must be a finite number in (0, 1]")
        count = math.ceil((n_samples - 1) * value)
    return min(max(1, count), n_samples - 1)


def resolve_pdmf_neighbor_count(value: int | float, n_samples: int) -> int:
    """Resolve the Gaussian-PDMF neighborhood for the current sample set."""

    return _resolve_neighbor_count(value, n_samples, "pdmf_neighbors")


def resolve_graph_neighbor_count(value: int | float, n_samples: int) -> int:
    """Resolve the sparse-graph KNN neighborhood for the current sample set."""

    return _resolve_neighbor_count(value, n_samples, "graph_neighbors")


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
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return entropy, active mask, spreads and clarity for Gaussian-PDMFs."""

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one attribute")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    neighbor_count = resolve_pdmf_neighbor_count(neighbors, values.shape[0])
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")

    nonconstant = np.ptp(values, axis=0) > 0.0
    raw_left, raw_right = _asymmetric_local_spreads(
        values, neighbor_count, epsilon
    )
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
    neighbors: int | float = 5,
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
    neighbors: int | float = 5,
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


def _union_knn_edges(
    X: np.ndarray,
    neighbors: int | float,
    *,
    mutual: bool = True,
) -> np.ndarray:
    """Return undirected mutual-KNN edges (or union-KNN when disabled)."""

    edges, _ = _knn_edges_and_scales(X, neighbors, mutual=mutual)
    return edges


def _knn_edges_and_scales(
    X: np.ndarray,
    neighbors: int | float,
    *,
    mutual: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN edges and local scales in one neighbor-search pass."""

    values = np.asarray(X, dtype=float)
    n_samples = values.shape[0]
    if n_samples < 2:
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=float)
    local_neighbors = resolve_graph_neighbor_count(neighbors, n_samples)
    model = NearestNeighbors(
        n_neighbors=local_neighbors + 1,
        metric="euclidean",
    ).fit(values)
    distances, neighbor_indices = model.kneighbors(values, return_distance=True)
    scales = np.maximum(distances[:, -1], 1e-12)

    neighbor_sets = [set(map(int, row[row != i][:local_neighbors])) for i, row in enumerate(neighbor_indices)]
    directed: list[tuple[int, int]] = []
    for sample_index, candidates in enumerate(neighbor_indices):
        selected = candidates[candidates != sample_index][:local_neighbors]
        directed.extend(
            (sample_index, int(neighbor))
            for neighbor in selected
            if not mutual or sample_index in neighbor_sets[int(neighbor)]
        )
    if not directed:
        return np.empty((0, 2), dtype=int), scales

    edges = np.asarray(directed, dtype=int)
    edges.sort(axis=1)
    return np.unique(edges, axis=0), scales


def _edge_gaussian_pdmf_similarities(
    X: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    edges: np.ndarray,
    feature_indices: np.ndarray,
    similarity_lambda: float,
    edge_weights: np.ndarray | None = None,
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
    similarities = similarity_lambda * core_similarity + (
        (1.0 - similarity_lambda)
        * (left_similarity + right_similarity)
        / (2.0 * _H0)
    )
    if edge_weights is not None:
        similarities *= np.asarray(edge_weights, dtype=float)[:, None]
    return similarities


def _self_tuning_edge_weights(
    X: np.ndarray,
    edges: np.ndarray,
    neighbors: int | float,
    epsilon: float,
) -> np.ndarray:
    """Compute Zelnik-Manor style local-scale weights on graph edges."""

    values = np.asarray(X, dtype=float)
    k = resolve_graph_neighbor_count(neighbors, values.shape[0])
    distances = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(values)
    knn_distances = distances.kneighbors(values, return_distance=True)[0]
    sigma = np.maximum(knn_distances[:, -1], epsilon)
    first, second = edges[:, 0], edges[:, 1]
    edge_distance = np.linalg.norm(values[first] - values[second], axis=1)
    return np.exp(-(edge_distance * edge_distance) / (sigma[first] * sigma[second] + epsilon))


def _graph_attribute_scores_from_components(
    X: np.ndarray,
    nonconstant: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    graph_neighbors: int | float,
    similarity_lambda: float,
    epsilon: float,
    block_size: int = 128,
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-attribute-out sparse-graph importance scores."""

    scores = np.full(X.shape[1], -np.inf, dtype=float)
    active = np.flatnonzero(nonconstant)
    edges, local_scales = _knn_edges_and_scales(
        X, graph_neighbors, mutual=mutual_knn
    )
    if active.size == 0 or edges.size == 0:
        return scores, edges
    if active.size == 1:
        scores[active[0]] = 1.0
        return scores, edges

    if self_tuning_graph:
        first, second = edges[:, 0], edges[:, 1]
        edge_distance = np.linalg.norm(X[first] - X[second], axis=1)
        edge_weights = np.exp(
            -(edge_distance * edge_distance)
            / (local_scales[first] * local_scales[second] + epsilon)
        )
    else:
        edge_weights = None
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
            edge_weights,
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
            edge_weights,
        )
        differences = similarities - full_graph[:, None]
        scores[block] = np.sum(differences * differences, axis=0) / (
            removal_scale * denominator
        )
    return scores, edges


def gaussian_pdmf_graph_attribute_scores(
    X: np.ndarray,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    similarity_lambda: float = 0.5,
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MY-V3 sparse-graph importance and its mutual-KNN edges."""

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
        mutual_knn=mutual_knn,
        self_tuning_graph=self_tuning_graph,
    )


def _normalized_descending_ranks(
    scores: np.ndarray,
    secondary_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Map deterministic descending integer ranks to the interval ``[0, 1]``."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 1:
        return np.ones(1, dtype=float)
    if not np.any(np.isfinite(values)):
        return np.zeros(values.size, dtype=float)
    indices = np.arange(values.size)
    if secondary_scores is None:
        order = np.lexsort((indices, -values))
    else:
        secondary = np.asarray(secondary_scores, dtype=float).reshape(-1)
        order = np.lexsort((indices, -secondary, -values))
    normalized = np.empty(values.size, dtype=float)
    normalized[order] = np.linspace(1.0, 0.0, values.size)
    return normalized


def _redundancy_aware_order(
    X: np.ndarray,
    base_order: np.ndarray,
    scores: np.ndarray,
    n_select: int,
    beta: float,
) -> np.ndarray:
    """Re-rank a bounded candidate pool with a correlation redundancy penalty."""

    order = np.asarray(base_order, dtype=int).reshape(-1)
    if beta <= 0 or n_select <= 1 or order.size <= 1:
        return order
    pool_size = min(order.size, max(64, 4 * int(n_select)))
    candidates = order[:pool_size]
    candidate_values = np.asarray(X, dtype=float)[:, candidates]
    centered = candidate_values - candidate_values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    normalized = centered / np.maximum(norms, 1e-12)
    remaining = np.ones(pool_size, dtype=bool)
    max_redundancy = np.zeros(pool_size, dtype=float)
    selected: list[int] = []
    for _ in range(min(int(n_select), pool_size)):
        if not np.any(remaining):
            break
        adjusted = np.full(pool_size, -np.inf, dtype=float)
        adjusted[remaining] = scores[candidates[remaining]] - beta * max_redundancy[remaining]
        best = int(np.argmax(adjusted))
        selected.append(best)
        remaining[best] = False
        similarities = np.abs(normalized.T @ normalized[:, best])
        max_redundancy = np.maximum(max_redundancy, similarities)
    selected_indices = [int(candidates[index]) for index in selected]
    selected_set = set(selected_indices)
    return np.asarray(selected_indices + [int(i) for i in order if int(i) not in selected_set], dtype=int)


def select_features_by_gaussian_pdmf(
    X: np.ndarray,
    n_features: int,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    *,
    scale_selected: bool = False,
    redundancy_beta: float = 0.0,
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
    order = _redundancy_aware_order(
        values, order, scores, n_features, redundancy_beta
    )
    selected_indices = order[:n_features]
    selected = values[:, selected_indices]
    if scale_selected:
        selected = minmax_scale_like_matlab(selected)
    return selected, selected_indices, scores


def select_global_features_by_gaussian_pdmf(
    X: np.ndarray,
    p1: int,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    redundancy_beta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the global ``p1`` attributes without labels or pseudo labels."""

    return select_features_by_gaussian_pdmf(
        X,
        p1,
        neighbors=neighbors,
        epsilon=epsilon,
        scale_selected=False,
        redundancy_beta=redundancy_beta,
    )


def select_local_features_by_gaussian_pdmf_graph(
    X: np.ndarray,
    p2: int,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    similarity_lambda: float = 0.5,
    ranking_cache: dict[str, np.ndarray] | None = None,
    redundancy_beta: float = 0.1,
    fusion_alpha_mode: str = "adaptive",
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select local attributes using adaptive entropy/graph fusion and redundancy control."""

    values = np.asarray(X, dtype=float)
    local_count = max(1, min(int(p2), values.shape[1]))
    resolved_graph_neighbors = resolve_graph_neighbor_count(
        graph_neighbors, values.shape[0]
    )
    if not np.isfinite(similarity_lambda) or not 0 < similarity_lambda < 1:
        raise ValueError("similarity_lambda must be a finite number in (0, 1)")
    if not np.isfinite(redundancy_beta) or redundancy_beta < 0:
        raise ValueError("redundancy_beta must be a finite non-negative number")
    if fusion_alpha_mode not in {"adaptive", "equal"}:
        raise ValueError("fusion_alpha_mode must be 'adaptive' or 'equal'")

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
            resolved_graph_neighbors,
            similarity_lambda,
            epsilon,
            mutual_knn=mutual_knn,
            self_tuning_graph=self_tuning_graph,
        )
        entropy_ranks = _normalized_descending_ranks(entropy_scores, single_entropies)
        graph_ranks = _normalized_descending_ranks(graph_scores)
        if fusion_alpha_mode == "adaptive":
            entropy_spread = float(np.std(entropy_ranks))
            graph_spread = float(np.std(graph_ranks))
            alpha = entropy_spread / (entropy_spread + graph_spread + epsilon)
        else:
            alpha = 0.5
        combined_scores = alpha * entropy_ranks + (1.0 - alpha) * graph_ranks
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
        if redundancy_beta > 0:
            order = _redundancy_aware_order(
                values, order, combined_scores, local_count, redundancy_beta
            )
        if ranking_cache is not None:
            ranking_cache["order"] = order.copy()
            ranking_cache["combined_scores"] = combined_scores.copy()

    selected_indices = order[:local_count]
    selected = minmax_scale_like_matlab(values[:, selected_indices])
    return selected, selected_indices, combined_scores
