from __future__ import annotations

"""Granular-ball division used by MY-V0."""

from dataclasses import dataclass

import numpy as np

from .feature_selection import select_local_features_by_gaussian_pdmf
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
    p2: int,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    max_iter: int = 3,
    seed: int | None = None,
) -> tuple[GranularBall, GranularBall]:
    """Split one ball after MY-V0 local Gaussian-PDMF reduction."""

    if ball.size < 2:
        return ball, GranularBall(ball.X[:0].copy(), ball.pseudo_labels[:0].copy())

    split_X, _, _ = select_local_features_by_gaussian_pdmf(
        ball.X,
        p2,
        neighbors=pdmf_neighbors,
        epsilon=pdmf_epsilon,
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
    p2: int,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
) -> list[GranularBall]:
    """Perform one scan of granular-ball division."""

    new_balls: list[GranularBall] = []
    for index, ball in enumerate(balls):
        if should_keep_ball(ball, purity_threshold, keep_matlab_split_rule):
            new_balls.append(ball)
            continue

        split_seed = None if seed is None else seed + index
        ball_1, ball_2 = split_ball_with_2means(
            ball,
            p2,
            pdmf_neighbors=pdmf_neighbors,
            pdmf_epsilon=pdmf_epsilon,
            max_iter=split_kmeans_max_iter,
            seed=split_seed,
        )
        if ball_2.size == 0:
            new_balls.append(ball_1)
        else:
            new_balls.extend([ball_1, ball_2])
    return new_balls


def generate_granular_balls(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    p2: int,
    purity_threshold: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    max_rounds: int = 10_000,
) -> list[GranularBall]:
    """Recursively divide the initial ball until no ball is split."""

    values = np.asarray(X, dtype=float)
    pseudo = np.asarray(pseudo_labels).reshape(-1)
    balls = [GranularBall(values, pseudo)]
    for _ in range(max_rounds):
        old_count = len(balls)
        balls = split_granular_balls(
            balls,
            purity_threshold,
            p2,
            pdmf_neighbors=pdmf_neighbors,
            pdmf_epsilon=pdmf_epsilon,
            split_kmeans_max_iter=split_kmeans_max_iter,
            seed=seed,
            keep_matlab_split_rule=keep_matlab_split_rule,
        )
        if len(balls) == old_count:
            break
    else:
        raise RuntimeError(
            f"Granular-ball splitting did not converge within {max_rounds} rounds"
        )
    return balls


def anchors_from_balls(balls: list[GranularBall]) -> np.ndarray:
    """Use each final granular-ball mean as an anchor."""

    anchors = [ball.X[0] if ball.size == 1 else ball.X.mean(axis=0) for ball in balls]
    return np.vstack(anchors)


def generate_anchors(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    p2: int,
    purity_threshold: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
) -> tuple[np.ndarray, list[GranularBall]]:
    """Generate the final anchor matrix and granular-ball list."""

    balls = generate_granular_balls(
        X,
        pseudo_labels,
        p2,
        purity_threshold,
        pdmf_neighbors=pdmf_neighbors,
        pdmf_epsilon=pdmf_epsilon,
        split_kmeans_max_iter=split_kmeans_max_iter,
        seed=seed,
        keep_matlab_split_rule=keep_matlab_split_rule,
    )
    return anchors_from_balls(balls), balls
