from __future__ import annotations

"""Adaptive entropy-graph-stable attribute reduction for MY-V2.

Global attributes follow the Gaussian-PDMF entropy ranking.  Local attributes
follow the equal entropy-graph ranking introduced by MY-V1.  In both stages,
the smallest prefix preserving full-set entropy and the sparse Gaussian-PDMF
graph is selected without labels or pseudo labels.
"""

import hashlib
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


def _union_knn_edges(X: np.ndarray, neighbors: int | float) -> np.ndarray:
    """Return unique undirected edges from the union of directed KNN lists."""

    values = np.asarray(X, dtype=float)
    n_samples = values.shape[0]
    if n_samples < 2:
        return np.empty((0, 2), dtype=int)
    local_neighbors = resolve_graph_neighbor_count(neighbors, n_samples)
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
    graph_neighbors: int | float,
    similarity_lambda: float,
    epsilon: float,
    block_size: int = 128,
    *,
    precomputed_edges: np.ndarray | None = None,
    precomputed_full_graph: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return leave-one-attribute-out sparse-graph importance scores."""

    scores = np.full(X.shape[1], -np.inf, dtype=float)
    active = np.flatnonzero(nonconstant)
    if (precomputed_edges is None) != (precomputed_full_graph is None):
        raise ValueError("precomputed_edges and precomputed_full_graph must be paired")
    if precomputed_edges is None:
        edges = _union_knn_edges(X, graph_neighbors)
        full_graph: np.ndarray | None = None
    else:
        edges = np.asarray(precomputed_edges, dtype=int)
        full_graph = np.asarray(precomputed_full_graph, dtype=float).reshape(-1)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("precomputed_edges must have shape (n_edges, 2)")
        if full_graph.size != edges.shape[0]:
            raise ValueError("precomputed_full_graph length must match edges")
    if active.size == 0 or edges.size == 0:
        return scores, edges
    if active.size == 1:
        scores[active[0]] = 1.0
        return scores, edges

    if full_graph is None:
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


def _mean_graph_from_components(
    X: np.ndarray,
    nonconstant: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    graph_neighbors: int | float,
    similarity_lambda: float,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full-attribute mean affinity on fixed sparse KNN edges."""

    active = np.flatnonzero(nonconstant)
    edges = _union_knn_edges(X, graph_neighbors)
    full_graph = np.zeros(edges.shape[0], dtype=float)
    if active.size == 0 or edges.size == 0:
        return edges, full_graph
    for start in range(0, active.size, block_size):
        block = active[start : start + block_size]
        full_graph += np.sum(
            _edge_gaussian_pdmf_similarities(
                X,
                spread_left,
                spread_right,
                clarity,
                edges,
                block,
                similarity_lambda,
            ),
            axis=1,
        )
    return edges, full_graph / active.size


def _stable_prefix_count(
    X: np.ndarray,
    order: np.ndarray,
    sample_entropies: np.ndarray,
    nonconstant: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    edges: np.ndarray,
    full_graph: np.ndarray,
    full_entropy: float,
    stability_delta: float,
    similarity_lambda: float,
    epsilon: float,
    minimum: int,
    block_size: int = 128,
) -> tuple[np.ndarray, float, float]:
    """Select the smallest ranked prefix satisfying both stability losses."""

    active_order = np.asarray(order, dtype=int)[nonconstant[order]]
    if active_order.size == 0:
        return np.array([int(order[0])]), 0.0, 0.0

    minimum = min(max(1, minimum), active_order.size)
    log_intensity = np.log1p(-sample_entropies[:, active_order])
    entropy_denominator = max(abs(full_entropy), epsilon)
    graph_denominator = float(np.linalg.norm(full_graph) + epsilon)
    prefix_log = np.zeros(X.shape[0], dtype=float)
    prefix_graph = np.zeros(edges.shape[0], dtype=float)

    for start in range(0, active_order.size, block_size):
        block = active_order[start : start + block_size]
        cumulative_log = prefix_log[:, None] + np.cumsum(
            log_intensity[:, start : start + block.size], axis=1
        )
        prefix_entropies = _entropy_of_log_weights(cumulative_log, axis=0)
        entropy_losses = np.abs(prefix_entropies - full_entropy) / entropy_denominator

        if edges.size:
            similarities = _edge_gaussian_pdmf_similarities(
                X,
                spread_left,
                spread_right,
                clarity,
                edges,
                block,
                similarity_lambda,
            )
            cumulative_graph = prefix_graph[:, None] + np.cumsum(
                similarities, axis=1
            )
            counts = np.arange(start + 1, start + block.size + 1, dtype=float)
            prefix_means = cumulative_graph / counts[None, :]
            graph_losses = np.linalg.norm(
                prefix_means - full_graph[:, None], axis=0
            ) / graph_denominator
        else:
            similarities = np.empty((0, block.size), dtype=float)
            graph_losses = np.zeros(block.size, dtype=float)

        counts = np.arange(start + 1, start + block.size + 1)
        if counts[-1] == active_order.size:
            entropy_losses[-1] = 0.0
            graph_losses[-1] = 0.0
        feasible = np.flatnonzero(
            (counts >= minimum)
            & (entropy_losses <= stability_delta)
            & (graph_losses <= stability_delta)
        )
        if feasible.size:
            position = int(feasible[0])
            count = int(counts[position])
            return (
                active_order[:count],
                float(entropy_losses[position]),
                float(graph_losses[position]),
            )

        prefix_log = cumulative_log[:, -1]
        if edges.size:
            prefix_graph = cumulative_graph[:, -1]

    return active_order, 0.0, 0.0


def _full_stability_curve(
    X: np.ndarray,
    order: np.ndarray,
    sample_entropies: np.ndarray,
    nonconstant: np.ndarray,
    spread_left: np.ndarray,
    spread_right: np.ndarray,
    clarity: np.ndarray,
    edges: np.ndarray,
    full_graph: np.ndarray,
    full_entropy: float,
    similarity_lambda: float,
    epsilon: float,
    block_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return entropy and graph losses for every ranked attribute prefix."""

    active_order = np.asarray(order, dtype=int)[nonconstant[order]]
    if active_order.size == 0:
        return np.array([int(order[0])]), np.zeros(1), np.zeros(1)

    log_intensity = np.log1p(-sample_entropies[:, active_order])
    entropy_denominator = max(abs(full_entropy), epsilon)
    graph_denominator = float(np.linalg.norm(full_graph) + epsilon)
    entropy_curve = np.empty(active_order.size, dtype=float)
    graph_curve = np.empty(active_order.size, dtype=float)
    prefix_log = np.zeros(X.shape[0], dtype=float)
    prefix_graph = np.zeros(edges.shape[0], dtype=float)

    for start in range(0, active_order.size, block_size):
        block = active_order[start : start + block_size]
        stop = start + block.size
        cumulative_log = prefix_log[:, None] + np.cumsum(
            log_intensity[:, start:stop], axis=1
        )
        prefix_entropies = _entropy_of_log_weights(cumulative_log, axis=0)
        entropy_curve[start:stop] = (
            np.abs(prefix_entropies - full_entropy) / entropy_denominator
        )

        if edges.size:
            similarities = _edge_gaussian_pdmf_similarities(
                X,
                spread_left,
                spread_right,
                clarity,
                edges,
                block,
                similarity_lambda,
            )
            cumulative_graph = prefix_graph[:, None] + np.cumsum(
                similarities, axis=1
            )
            counts = np.arange(start + 1, stop + 1, dtype=float)
            prefix_means = cumulative_graph / counts[None, :]
            graph_curve[start:stop] = np.linalg.norm(
                prefix_means - full_graph[:, None], axis=0
            ) / graph_denominator
            prefix_graph = cumulative_graph[:, -1]
        else:
            graph_curve[start:stop] = 0.0
        prefix_log = cumulative_log[:, -1]

    entropy_curve[-1] = 0.0
    graph_curve[-1] = 0.0
    return active_order, entropy_curve, graph_curve


def _select_from_stability_curve(
    active_order: np.ndarray,
    entropy_curve: np.ndarray,
    graph_curve: np.ndarray,
    stability_delta: float,
    minimum: int,
) -> tuple[np.ndarray, float, float]:
    """Select the first prefix satisfying a requested stability threshold."""

    minimum = min(max(1, minimum), active_order.size)
    counts = np.arange(1, active_order.size + 1)
    feasible = np.flatnonzero(
        (counts >= minimum)
        & (entropy_curve <= stability_delta)
        & (graph_curve <= stability_delta)
    )
    position = int(feasible[0]) if feasible.size else active_order.size - 1
    return (
        active_order[: position + 1],
        float(entropy_curve[position]),
        float(graph_curve[position]),
    )


def _adaptive_stable_selection(
    X: np.ndarray,
    stability_delta: float,
    neighbors: int | float,
    epsilon: float,
    graph_neighbors: int | float,
    similarity_lambda: float,
    *,
    local_ranking: bool,
    scale_selected: bool,
    ranking_cache: dict[str, object] | None = None,
    stability_curve_cache: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Rank attributes and select the smallest entropy-graph-stable prefix."""

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("X must contain at least two samples and one attribute")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    if not np.isfinite(stability_delta) or stability_delta < 0:
        raise ValueError("stability_delta must be a finite non-negative number")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")
    if not np.isfinite(similarity_lambda) or not 0 < similarity_lambda < 1:
        raise ValueError("similarity_lambda must be a finite number in (0, 1)")

    resolved_neighbors = resolve_pdmf_neighbor_count(neighbors, values.shape[0])
    resolved_graph_neighbors = resolve_graph_neighbor_count(
        graph_neighbors, values.shape[0]
    )
    contiguous = np.ascontiguousarray(values)
    base_cache_signature = (
        values.shape,
        hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest(),
        resolved_neighbors,
        float(epsilon),
        resolved_graph_neighbors,
        float(similarity_lambda),
        local_ranking,
    )
    cache_signature = (*base_cache_signature, float(stability_delta))

    required = {
        "selected_indices",
        "scores",
        "entropy_loss",
        "graph_loss",
        "cache_signature",
    }
    if ranking_cache is not None and required <= ranking_cache.keys():
        selected_indices = np.asarray(
            ranking_cache["selected_indices"], dtype=int
        ).reshape(-1)
        scores = np.asarray(ranking_cache["scores"], dtype=float).reshape(-1)
        if (
            ranking_cache["cache_signature"] == cache_signature
            and selected_indices.size >= 1
            and np.all((0 <= selected_indices) & (selected_indices < values.shape[1]))
            and scores.size == values.shape[1]
        ):
            selected = values[:, selected_indices]
            if scale_selected:
                selected = minmax_scale_like_matlab(selected)
            return (
                selected,
                selected_indices.copy(),
                scores.copy(),
                float(np.asarray(ranking_cache["entropy_loss"]).item()),
                float(np.asarray(ranking_cache["graph_loss"]).item()),
            )

    curve_required = {
        "active_order",
        "scores",
        "entropy_curve",
        "graph_curve",
        "base_cache_signature",
    }
    if (
        stability_curve_cache is not None
        and curve_required <= stability_curve_cache.keys()
        and stability_curve_cache["base_cache_signature"] == base_cache_signature
    ):
        active_order = np.asarray(
            stability_curve_cache["active_order"], dtype=int
        ).reshape(-1)
        scores = np.asarray(stability_curve_cache["scores"], dtype=float).reshape(-1)
        entropy_curve = np.asarray(
            stability_curve_cache["entropy_curve"], dtype=float
        ).reshape(-1)
        graph_curve = np.asarray(
            stability_curve_cache["graph_curve"], dtype=float
        ).reshape(-1)
        if (
            active_order.size >= 1
            and np.all((0 <= active_order) & (active_order < values.shape[1]))
            and scores.size == values.shape[1]
            and entropy_curve.size == active_order.size
            and graph_curve.size == active_order.size
        ):
            selected_indices, entropy_loss, graph_loss = (
                _select_from_stability_curve(
                    active_order,
                    entropy_curve,
                    graph_curve,
                    stability_delta,
                    minimum=1,
                )
            )
            selected = values[:, selected_indices]
            if scale_selected:
                selected = minmax_scale_like_matlab(selected)
            return selected, selected_indices, scores, entropy_loss, graph_loss

    sample_entropies, nonconstant, spread_left, spread_right, clarity = (
        _gaussian_pdmf_components(
            values, neighbors=resolved_neighbors, epsilon=epsilon
        )
    )
    entropy_scores, single_entropies, full_entropy = (
        _attribute_scores_from_sample_entropies(sample_entropies, nonconstant)
    )
    edges, full_graph = _mean_graph_from_components(
        values,
        nonconstant,
        spread_left,
        spread_right,
        clarity,
        resolved_graph_neighbors,
        similarity_lambda,
    )

    feature_indices = np.arange(values.shape[1])
    if local_ranking:
        graph_scores, _ = _graph_attribute_scores_from_components(
            values,
            nonconstant,
            spread_left,
            spread_right,
            clarity,
            resolved_graph_neighbors,
            similarity_lambda,
            epsilon,
            precomputed_edges=edges,
            precomputed_full_graph=full_graph,
        )
        entropy_ranks = _normalized_descending_ranks(
            entropy_scores, single_entropies
        )
        graph_ranks = _normalized_descending_ranks(graph_scores)
        scores = 0.5 * (entropy_ranks + graph_ranks)
        order = np.lexsort(
            (
                feature_indices,
                -single_entropies,
                -graph_scores,
                -entropy_scores,
                -scores,
            )
        )
        minimum = 1
    else:
        scores = entropy_scores
        order = np.lexsort((feature_indices, -single_entropies, -entropy_scores))
        minimum = 1

    if stability_curve_cache is None:
        selected_indices, entropy_loss, graph_loss = _stable_prefix_count(
            values,
            order,
            sample_entropies,
            nonconstant,
            spread_left,
            spread_right,
            clarity,
            edges,
            full_graph,
            full_entropy,
            stability_delta,
            similarity_lambda,
            epsilon,
            minimum,
        )
    else:
        active_order, entropy_curve, graph_curve = _full_stability_curve(
            values,
            order,
            sample_entropies,
            nonconstant,
            spread_left,
            spread_right,
            clarity,
            edges,
            full_graph,
            full_entropy,
            similarity_lambda,
            epsilon,
        )
        selected_indices, entropy_loss, graph_loss = _select_from_stability_curve(
            active_order,
            entropy_curve,
            graph_curve,
            stability_delta,
            minimum,
        )
        stability_curve_cache["active_order"] = active_order.copy()
        stability_curve_cache["scores"] = scores.copy()
        stability_curve_cache["entropy_curve"] = entropy_curve.copy()
        stability_curve_cache["graph_curve"] = graph_curve.copy()
        stability_curve_cache["base_cache_signature"] = base_cache_signature
    selected = values[:, selected_indices]
    if scale_selected:
        selected = minmax_scale_like_matlab(selected)

    if ranking_cache is not None:
        ranking_cache["selected_indices"] = selected_indices.copy()
        ranking_cache["scores"] = scores.copy()
        ranking_cache["entropy_loss"] = np.asarray(entropy_loss)
        ranking_cache["graph_loss"] = np.asarray(graph_loss)
        ranking_cache["cache_signature"] = cache_signature
    return selected, selected_indices, scores, entropy_loss, graph_loss


def gaussian_pdmf_graph_attribute_scores(
    X: np.ndarray,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    similarity_lambda: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MY-V1 sparse-graph importance and its fixed union-KNN edges."""

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
    neighbors: int | float = 5,
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
    stability_delta: float = 0.05,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    similarity_lambda: float = 0.5,
    stability_curve_cache: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Select the smallest globally entropy-graph-stable attribute prefix."""

    return _adaptive_stable_selection(
        X,
        stability_delta,
        neighbors,
        epsilon,
        graph_neighbors=graph_neighbors,
        similarity_lambda=similarity_lambda,
        local_ranking=False,
        scale_selected=False,
        stability_curve_cache=stability_curve_cache,
    )


def select_local_features_by_gaussian_pdmf_graph(
    X: np.ndarray,
    stability_delta: float = 0.05,
    neighbors: int | float = 5,
    epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    similarity_lambda: float = 0.5,
    ranking_cache: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Select the smallest locally entropy-graph-stable attribute prefix."""

    return _adaptive_stable_selection(
        X,
        stability_delta,
        neighbors,
        epsilon,
        graph_neighbors,
        similarity_lambda,
        local_ranking=True,
        scale_selected=True,
        ranking_cache=ranking_cache,
    )
