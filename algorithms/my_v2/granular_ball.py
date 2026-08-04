from __future__ import annotations

"""Granular-ball division used by MY-V2."""

from dataclasses import dataclass

import numpy as np

from .feature_selection import select_local_features_by_gaussian_pdmf_graph
from .weighted_kmeans import two_means_labels


@dataclass(frozen=True)
class GranularBall:
    """Samples and unsupervised pseudo labels contained in one granular ball."""

    X: np.ndarray
    pseudo_labels: np.ndarray

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


def pseudo_purity(labels: np.ndarray) -> float:
    """Return the proportion of the most frequent pseudo label."""

    values = np.asarray(labels).reshape(-1)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    return float(counts.max() / values.size)


def split_ball_with_2means(
    ball: GranularBall,
    stability_delta: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    max_iter: int = 3,
    seed: int | None = None,
    ranking_cache: dict[str, object] | None = None,
) -> tuple[GranularBall, GranularBall, int]:
    """Split one ball using its smallest entropy-graph-stable prefix."""

    if ball.size < 2:
        empty = GranularBall(ball.X[:0].copy(), ball.pseudo_labels[:0].copy())
        return ball, empty, 0

    split_X, selected_indices, _, _, _ = (
        select_local_features_by_gaussian_pdmf_graph(
            ball.X,
            stability_delta,
            neighbors=pdmf_neighbors,
            epsilon=pdmf_epsilon,
            graph_neighbors=graph_neighbors,
            similarity_lambda=pdmf_similarity_lambda,
            ranking_cache=ranking_cache,
        )
    )
    labels = two_means_labels(split_X, max_iter=max_iter, seed=seed)
    first = labels == 0
    second = labels == 1
    if not np.any(first) or not np.any(second):
        midpoint = ball.size // 2
        first = np.zeros(ball.size, dtype=bool)
        first[:midpoint] = True
        second = ~first
    return (
        GranularBall(ball.X[first], ball.pseudo_labels[first]),
        GranularBall(ball.X[second], ball.pseudo_labels[second]),
        int(selected_indices.size),
    )


def should_keep_ball(
    ball: GranularBall,
    purity_threshold: float,
    keep_matlab_split_rule: bool = True,
) -> bool:
    """Apply the retained PLGB-FSC stopping rule."""

    purity = pseudo_purity(ball.pseudo_labels)
    if ball.size < 2:
        return True
    if keep_matlab_split_rule:
        return bool(purity >= purity_threshold and ball.size < 8)
    return bool(purity >= purity_threshold)


def split_granular_balls(
    balls: list[GranularBall],
    purity_threshold: float,
    stability_delta: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    root_ranking_cache: dict[str, object] | None = None,
) -> tuple[list[GranularBall], list[int]]:
    """Perform one scan and record each split's selected attribute count."""

    new_balls: list[GranularBall] = []
    selected_counts: list[int] = []
    for index, ball in enumerate(balls):
        if should_keep_ball(ball, purity_threshold, keep_matlab_split_rule):
            new_balls.append(ball)
            continue

        split_seed = None if seed is None else seed + index
        ball_1, ball_2, selected_count = split_ball_with_2means(
            ball,
            stability_delta,
            pdmf_neighbors=pdmf_neighbors,
            pdmf_epsilon=pdmf_epsilon,
            graph_neighbors=graph_neighbors,
            pdmf_similarity_lambda=pdmf_similarity_lambda,
            max_iter=split_kmeans_max_iter,
            seed=split_seed,
            ranking_cache=root_ranking_cache if index == 0 else None,
        )
        if ball_2.size == 0:
            new_balls.append(ball_1)
        else:
            new_balls.extend([ball_1, ball_2])
            selected_counts.append(selected_count)
    return new_balls, selected_counts


def generate_granular_balls(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    stability_delta: float,
    purity_threshold: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    max_rounds: int = 10_000,
    root_ranking_cache: dict[str, object] | None = None,
) -> tuple[list[GranularBall], tuple[int, ...]]:
    """Recursively divide balls and retain all adaptive local counts."""

    values = np.asarray(X, dtype=float)
    pseudo = np.asarray(pseudo_labels).reshape(-1)
    balls = [GranularBall(values, pseudo)]
    selected_counts: list[int] = []
    for round_index in range(max_rounds):
        old_count = len(balls)
        balls, round_counts = split_granular_balls(
            balls,
            purity_threshold,
            stability_delta,
            pdmf_neighbors=pdmf_neighbors,
            pdmf_epsilon=pdmf_epsilon,
            graph_neighbors=graph_neighbors,
            pdmf_similarity_lambda=pdmf_similarity_lambda,
            split_kmeans_max_iter=split_kmeans_max_iter,
            seed=seed,
            keep_matlab_split_rule=keep_matlab_split_rule,
            root_ranking_cache=root_ranking_cache if round_index == 0 else None,
        )
        selected_counts.extend(round_counts)
        if len(balls) == old_count:
            break
    else:
        raise RuntimeError(
            f"Granular-ball splitting did not converge within {max_rounds} rounds"
        )
    return balls, tuple(selected_counts)


def anchors_from_balls(balls: list[GranularBall]) -> np.ndarray:
    """Use each final granular-ball mean as an anchor."""

    anchors = [ball.X[0] if ball.size == 1 else ball.X.mean(axis=0) for ball in balls]
    return np.vstack(anchors)


def generate_anchors(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    stability_delta: float,
    purity_threshold: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    root_ranking_cache: dict[str, object] | None = None,
) -> tuple[np.ndarray, list[GranularBall], tuple[int, ...]]:
    """Generate anchors, final balls and adaptive local attribute counts."""

    balls, selected_counts = generate_granular_balls(
        X,
        pseudo_labels,
        stability_delta,
        purity_threshold,
        pdmf_neighbors=pdmf_neighbors,
        pdmf_epsilon=pdmf_epsilon,
        graph_neighbors=graph_neighbors,
        pdmf_similarity_lambda=pdmf_similarity_lambda,
        split_kmeans_max_iter=split_kmeans_max_iter,
        seed=seed,
        keep_matlab_split_rule=keep_matlab_split_rule,
        root_ranking_cache=root_ranking_cache,
    )
    return anchors_from_balls(balls), balls, selected_counts
