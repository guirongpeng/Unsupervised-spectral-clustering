from __future__ import annotations

"""Granular-ball division used by MY-V3."""

from concurrent.futures import ThreadPoolExecutor
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
    p2: int,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    redundancy_beta: float = 0.1,
    fusion_alpha_mode: str = "adaptive",
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
    max_iter: int = 3,
    seed: int | None = None,
    ranking_cache: dict[str, np.ndarray] | None = None,
    attribute_parallel_jobs: int = 1,
) -> tuple[GranularBall, GranularBall]:
    """Split one ball after MY-V3 local entropy-graph reduction."""

    if ball.size < 2:
        return ball, GranularBall(ball.X[:0].copy(), ball.pseudo_labels[:0].copy())

    split_X, _, _ = select_local_features_by_gaussian_pdmf_graph(
        ball.X,
        p2,
        neighbors=pdmf_neighbors,
        epsilon=pdmf_epsilon,
        graph_neighbors=graph_neighbors,
        similarity_lambda=pdmf_similarity_lambda,
        redundancy_beta=redundancy_beta,
        fusion_alpha_mode=fusion_alpha_mode,
        mutual_knn=mutual_knn,
        self_tuning_graph=self_tuning_graph,
        ranking_cache=ranking_cache,
        attribute_parallel_jobs=attribute_parallel_jobs,
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
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    redundancy_beta: float = 0.1,
    fusion_alpha_mode: str = "adaptive",
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    root_ranking_cache: dict[str, np.ndarray] | None = None,
    executor: ThreadPoolExecutor | None = None,
    ball_parallel_jobs: int = 1,
) -> list[GranularBall]:
    """Perform one scan of granular-ball division."""

    should_split = tuple(
        not should_keep_ball(ball, purity_threshold, keep_matlab_split_rule)
        for ball in balls
    )
    attribute_parallel_jobs = ball_parallel_jobs if sum(should_split) == 1 else 1

    def process(item: tuple[int, GranularBall]) -> list[GranularBall]:
        index, ball = item
        if not should_split[index]:
            return [ball]

        split_seed = None if seed is None else seed + index
        ball_1, ball_2 = split_ball_with_2means(
            ball,
            p2,
            pdmf_neighbors=pdmf_neighbors,
            pdmf_epsilon=pdmf_epsilon,
            graph_neighbors=graph_neighbors,
            pdmf_similarity_lambda=pdmf_similarity_lambda,
            redundancy_beta=redundancy_beta,
            fusion_alpha_mode=fusion_alpha_mode,
            mutual_knn=mutual_knn,
            self_tuning_graph=self_tuning_graph,
            max_iter=split_kmeans_max_iter,
            seed=split_seed,
            ranking_cache=root_ranking_cache if index == 0 else None,
            attribute_parallel_jobs=attribute_parallel_jobs,
        )
        if ball_2.size == 0:
            return [ball_1]
        return [ball_1, ball_2]

    results = (
        executor.map(process, enumerate(balls))
        if executor is not None
        else map(process, enumerate(balls))
    )
    new_balls: list[GranularBall] = []
    for children in results:
        new_balls.extend(children)
    return new_balls


def generate_granular_balls(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    p2: int,
    purity_threshold: float,
    pdmf_neighbors: int | float = 5,
    pdmf_epsilon: float = 1e-8,
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    redundancy_beta: float = 0.1,
    fusion_alpha_mode: str = "adaptive",
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    max_rounds: int = 10_000,
    root_ranking_cache: dict[str, np.ndarray] | None = None,
    ball_parallel_jobs: int = 1,
) -> list[GranularBall]:
    """Recursively divide the initial ball until no ball is split."""

    values = np.asarray(X, dtype=float)
    pseudo = np.asarray(pseudo_labels).reshape(-1)
    if isinstance(ball_parallel_jobs, bool) or not isinstance(ball_parallel_jobs, int):
        raise TypeError("ball_parallel_jobs must be an integer")
    if ball_parallel_jobs < 1:
        raise ValueError("ball_parallel_jobs must be at least 1")
    balls = [GranularBall(values, pseudo)]
    executor = (
        ThreadPoolExecutor(max_workers=ball_parallel_jobs, thread_name_prefix="my_v3_ball")
        if ball_parallel_jobs > 1
        else None
    )
    try:
        for round_index in range(max_rounds):
            old_count = len(balls)
            balls = split_granular_balls(
                balls,
                purity_threshold,
                p2,
                pdmf_neighbors=pdmf_neighbors,
                pdmf_epsilon=pdmf_epsilon,
                graph_neighbors=graph_neighbors,
                pdmf_similarity_lambda=pdmf_similarity_lambda,
                redundancy_beta=redundancy_beta,
                fusion_alpha_mode=fusion_alpha_mode,
                mutual_knn=mutual_knn,
                self_tuning_graph=self_tuning_graph,
                split_kmeans_max_iter=split_kmeans_max_iter,
                seed=seed,
                keep_matlab_split_rule=keep_matlab_split_rule,
                root_ranking_cache=root_ranking_cache if round_index == 0 else None,
                executor=executor,
                ball_parallel_jobs=ball_parallel_jobs,
            )
            if len(balls) == old_count:
                break
        else:
            raise RuntimeError(
                f"Granular-ball splitting did not converge within {max_rounds} rounds"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
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
    graph_neighbors: int | float = 5,
    pdmf_similarity_lambda: float = 0.5,
    redundancy_beta: float = 0.1,
    fusion_alpha_mode: str = "adaptive",
    mutual_knn: bool = True,
    self_tuning_graph: bool = True,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    root_ranking_cache: dict[str, np.ndarray] | None = None,
    ball_parallel_jobs: int = 1,
) -> tuple[np.ndarray, list[GranularBall]]:
    """Generate the final anchor matrix and granular-ball list."""

    balls = generate_granular_balls(
        X,
        pseudo_labels,
        p2,
        purity_threshold,
        pdmf_neighbors=pdmf_neighbors,
        pdmf_epsilon=pdmf_epsilon,
        graph_neighbors=graph_neighbors,
        pdmf_similarity_lambda=pdmf_similarity_lambda,
        redundancy_beta=redundancy_beta,
        fusion_alpha_mode=fusion_alpha_mode,
        mutual_knn=mutual_knn,
        self_tuning_graph=self_tuning_graph,
        split_kmeans_max_iter=split_kmeans_max_iter,
        seed=seed,
        keep_matlab_split_rule=keep_matlab_split_rule,
        root_ranking_cache=root_ranking_cache,
        ball_parallel_jobs=ball_parallel_jobs,
    )
    return anchors_from_balls(balls), balls
