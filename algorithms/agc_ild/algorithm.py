from __future__ import annotations

"""Source-faithful Python migration of the official AGC-ILD MATLAB code.

Only dataset preprocessing remains the unified Benchmark responsibility.
"""

from dataclasses import asdict

import numpy as np

from core.algorithm import Algorithm as BenchmarkAlgorithm

from .config import AGCILDConfig


def _balanced_ekm(X: np.ndarray, alpha: float, config: AGCILDConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """BalancedEKM_with_one_XX.m, including its source repulsion heuristic."""
    count = X.shape[0]
    first_center, second_center = X[rng.integers(count)].copy(), X[rng.integers(count)].copy()
    labels = np.empty(count, dtype=int)
    for _ in range(config.max_iter):
        first_distance = 0.5 * np.sum((X - first_center) ** 2, axis=1)
        second_distance = 0.5 * np.sum((X - second_center) ** 2, axis=1)
        weights = np.exp(-alpha * np.column_stack((first_distance, second_distance)))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), np.finfo(float).eps)
        repulsion = np.exp(-alpha * abs(float(first_distance.mean() - second_distance.mean())))
        adjusted_first = np.maximum(weights[:, 0] * (1.0 + repulsion * (first_distance - (first_distance + second_distance) / 2.0)), np.finfo(float).eps)
        adjusted_second = np.maximum(weights[:, 1] * (1.0 + repulsion * (second_distance - (first_distance + second_distance) / 2.0)), np.finfo(float).eps)
        order = np.argsort(adjusted_first - adjusted_second, kind="mergesort")
        first, second = order[: count // 2], order[count // 2 :]
        updated_first = adjusted_first[first] @ X[first] / (adjusted_first[first].sum() + np.finfo(float).eps)
        updated_second = adjusted_second[second] @ X[second] / (adjusted_second[second].sum() + np.finfo(float).eps)
        labels[first], labels[second] = 0, 1
        if np.linalg.norm(updated_first - first_center) < 1e-5 and np.linalg.norm(updated_second - second_center) < 1e-5:
            break
        first_center, second_center = updated_first, updated_second
    return np.vstack((first_center, second_center)), labels


def _anchor_generation(X: np.ndarray, indices: np.ndarray, depth: int, alpha: float, config: AGCILDConfig, rng: np.random.Generator) -> np.ndarray:
    """AnchorGeneration.m recursion; returned column order is irrelevant to clustering."""
    centers, labels = _balanced_ekm(X[indices], alpha, config, rng)
    if depth == 1:
        return centers
    return np.vstack((
        _anchor_generation(X, indices[labels == 0], depth - 1, alpha, config, rng),
        _anchor_generation(X, indices[labels == 1], depth - 1, alpha, config, rng),
    ))


def generate_anchors(X: np.ndarray, config: AGCILDConfig, rng: np.random.Generator) -> np.ndarray:
    """Official BWHK implementation: source heuristic, not the paper's idealized Eq. (10)."""
    if config.n_anchors > X.shape[0]:
        raise ValueError("n_anchors cannot exceed n_samples")
    depth = config.n_anchors.bit_length() - 1
    alpha = 2.0 / max(float(np.mean(np.var(X, axis=0))), np.finfo(float).eps)  # paper bbb=2
    return _anchor_generation(X, np.arange(X.shape[0]), depth, alpha, config, rng)


def construct_anchor_graph(X: np.ndarray, anchors: np.ndarray, config: AGCILDConfig) -> np.ndarray:
    """AnchorGraphConstruction.m, including density-adaptive neighbour count."""
    count = anchors.shape[0]
    requested_k = min(config.k_neighbors, count)
    graph = np.zeros((X.shape[0], count), dtype=float)
    eps = np.finfo(float).eps
    for index, sample in enumerate(X):
        distances = np.sum((anchors - sample) ** 2, axis=1)
        order = np.argsort(distances)
        sorted_distances = distances[order]
        density_weight = float(np.mean(sorted_distances[:requested_k]) + eps)
        adaptive_k = min(requested_k + int(np.floor(density_weight / np.mean(sorted_distances))), count)
        if adaptive_k == count:
            boundary = sorted_distances[-1] + eps
        else:
            boundary = sorted_distances[adaptive_k]
        local = sorted_distances[:adaptive_k] / density_weight
        adjusted_boundary = boundary / density_weight
        denominator = float(np.sum(adjusted_boundary - local))
        if denominator <= 0:
            graph[index, order[:adaptive_k]] = 1.0 / adaptive_k
        else:
            graph[index, order[:adaptive_k]] = (adjusted_boundary - local) / denominator
    # Source: Delta=diag(q./sqrt(q)); B=Z*sqrt(inv(Delta)) = Z*q^(-1/4).
    degree = graph.sum(axis=0)
    return graph / np.power(degree + eps, 0.25)[None, :]


def propagate_labels(B: np.ndarray, n_clusters: int, config: AGCILDConfig, rng: np.random.Generator) -> np.ndarray:
    """AGCILD.m coordinate updates (source limit: 30 outer iterations)."""
    anchors = B.shape[1]
    labels = rng.integers(n_clusters, size=anchors)
    U = np.eye(n_clusters, dtype=float)[labels]
    counts = U.sum(axis=0)
    double_gram = 2.0 * (B.T @ B)
    diagonal = np.diag(double_gram) / 2.0
    gram_labels = double_gram @ U
    energy = np.diag(U.T @ gram_labels / 2.0).copy()
    objectives = [float(np.sqrt(np.maximum(energy, 0.0)).sum() + config.beta * np.reciprocal(counts).sum())]
    for iteration in range(30):  # exactly AGCILD.m
        for anchor in range(anchors):
            old = int(labels[anchor])
            if counts[old] == 1:
                continue
            add_energy = energy + (gram_labels[anchor] + diagonal[anchor]) * (1.0 - U[anchor])
            remove_energy = energy - (gram_labels[anchor] - diagonal[anchor]) * U[anchor]
            score = np.sqrt(np.maximum(add_energy, 0.0)) - np.sqrt(np.maximum(remove_energy, 0.0))
            penalty = config.beta * (np.reciprocal(counts + 1.0) - np.reciprocal(counts))
            penalty[old] = config.beta * (1.0 / counts[old] - 1.0 / (counts[old] - 1.0))
            target = int(np.argmax(score + penalty))
            if target != old:
                counts[target] += 1.0; counts[old] -= 1.0
                energy[old] = remove_energy[old]; energy[target] = add_energy[target]
                U[anchor, old] = 0.0; U[anchor, target] = 1.0; labels[anchor] = target
                gram_labels[:, old] -= double_gram[:, anchor]
                gram_labels[:, target] += double_gram[:, anchor]
        objectives.append(float(np.sqrt(np.maximum(energy, 0.0)).sum() + config.beta * np.reciprocal(counts).sum()))
        if iteration > 0 and abs(objectives[-2] - objectives[-3]) < config.tolerance:
            break
    return np.argmax(B @ U, axis=1)


class AGCILD(BenchmarkAlgorithm):
    def __init__(self, config: AGCILDConfig, n_clusters: int, random_state: int | None = None) -> None:
        self.config, self.n_clusters, self.random_state = config, n_clusters, random_state

    def fit(self, X: np.ndarray) -> "AGCILD":
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not np.all(np.isfinite(values)) or not 2 <= self.n_clusters <= values.shape[0]:
            raise ValueError("X must be finite 2D data and n_clusters must be in [2, n_samples]")
        rng = np.random.default_rng(self.random_state)
        self.anchors_ = generate_anchors(values, self.config, rng)
        self.anchor_graph_ = construct_anchor_graph(values, self.anchors_, self.config)
        self.labels_ = propagate_labels(self.anchor_graph_, self.n_clusters, self.config, rng)
        return self

    def get_params(self) -> dict[str, object]:
        return {"n_clusters": self.n_clusters, "random_state": self.random_state, **asdict(self.config)}
