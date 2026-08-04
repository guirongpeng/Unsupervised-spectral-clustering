from __future__ import annotations

"""Granular-ball division used by MY-V2."""

import hashlib
from _thread import LockType
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import numpy as np

from .common.preprocessing import minmax_scale_like_matlab
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


LocalFeatureSelectionCache = dict[tuple[object, ...], np.ndarray]
_LOCAL_CACHE_MAX_ENTRIES = 1024


def _local_selection_cache_key(
    X: np.ndarray,
    stability_delta: float,
    pdmf_neighbors: int | float,
    pdmf_epsilon: float,
    graph_neighbors: int | float,
    pdmf_similarity_lambda: float,
) -> tuple[object, ...]:
    values = np.ascontiguousarray(X, dtype=float)
    return (
        values.shape,
        hashlib.blake2b(values.view(np.uint8), digest_size=16).digest(),
        type(pdmf_neighbors).__name__,
        float(pdmf_neighbors),
        float(pdmf_epsilon),
        type(graph_neighbors).__name__,
        float(graph_neighbors),
        float(pdmf_similarity_lambda),
        float(stability_delta),
    )


def _get_cached_indices(
    cache: LocalFeatureSelectionCache | None,
    key: tuple[object, ...] | None,
    lock: LockType | None,
) -> np.ndarray | None:
    if cache is None or key is None:
        return None
    if lock is None:
        selected = cache.pop(key, None)
        if selected is not None:
            cache[key] = selected
        return selected
    with lock:
        selected = cache.pop(key, None)
        if selected is not None:
            cache[key] = selected
        return selected


def _store_cached_indices(
    cache: LocalFeatureSelectionCache | None,
    key: tuple[object, ...] | None,
    selected: np.ndarray,
    lock: LockType | None,
) -> None:
    if cache is None or key is None:
        return

    def store() -> None:
        if len(cache) >= _LOCAL_CACHE_MAX_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = selected.copy()

    if lock is None:
        store()
    else:
        with lock:
            store()


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
    local_feature_selection_cache: LocalFeatureSelectionCache | None = None,
    cache_lock: LockType | None = None,
) -> tuple[GranularBall, GranularBall, int]:
    """Split one ball using its smallest entropy-graph-stable prefix."""

    if ball.size < 2:
        empty = GranularBall(ball.X[:0].copy(), ball.pseudo_labels[:0].copy())
        return ball, empty, 0

    cache_key = (
        _local_selection_cache_key(
            ball.X,
            stability_delta,
            pdmf_neighbors,
            pdmf_epsilon,
            graph_neighbors,
            pdmf_similarity_lambda,
        )
        if local_feature_selection_cache is not None
        else None
    )
    selected_indices = _get_cached_indices(
        local_feature_selection_cache, cache_key, cache_lock
    )
    if selected_indices is not None:
        split_X = minmax_scale_like_matlab(ball.X[:, selected_indices])
    else:
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
        _store_cached_indices(
            local_feature_selection_cache,
            cache_key,
            selected_indices,
            cache_lock,
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
    local_feature_selection_cache: LocalFeatureSelectionCache | None = None,
    executor: ThreadPoolExecutor | None = None,
    cache_lock: LockType | None = None,
) -> tuple[list[GranularBall], list[int]]:
    """Perform one scan and record each split's selected attribute count."""

    def process(
        item: tuple[int, GranularBall],
    ) -> tuple[list[GranularBall], int | None]:
        index, ball = item
        if should_keep_ball(ball, purity_threshold, keep_matlab_split_rule):
            return [ball], None

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
            local_feature_selection_cache=local_feature_selection_cache,
            cache_lock=cache_lock,
        )
        if ball_2.size == 0:
            return [ball_1], None
        return [ball_1, ball_2], selected_count

    items = enumerate(balls)
    results = executor.map(process, items) if executor is not None else map(process, items)
    new_balls: list[GranularBall] = []
    selected_counts: list[int] = []
    for children, selected_count in results:
        new_balls.extend(children)
        if selected_count is not None:
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
    local_feature_selection_cache: LocalFeatureSelectionCache | None = None,
    ball_parallel_jobs: int = 1,
) -> tuple[list[GranularBall], tuple[int, ...]]:
    """Recursively divide balls and retain all adaptive local counts."""

    values = np.asarray(X, dtype=float)
    pseudo = np.asarray(pseudo_labels).reshape(-1)
    if isinstance(ball_parallel_jobs, bool) or not isinstance(ball_parallel_jobs, int):
        raise TypeError("ball_parallel_jobs must be an integer")
    if ball_parallel_jobs < 1:
        raise ValueError("ball_parallel_jobs must be at least 1")
    balls = [GranularBall(values, pseudo)]
    selected_counts: list[int] = []
    executor = (
        ThreadPoolExecutor(
            max_workers=ball_parallel_jobs,
            thread_name_prefix="my_v2_ball",
        )
        if ball_parallel_jobs > 1
        else None
    )
    cache_lock = Lock() if executor is not None else None
    try:
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
                root_ranking_cache=(
                    root_ranking_cache if round_index == 0 else None
                ),
                local_feature_selection_cache=local_feature_selection_cache,
                executor=executor,
                cache_lock=cache_lock,
            )
            selected_counts.extend(round_counts)
            if len(balls) == old_count:
                break
        else:
            raise RuntimeError(
                f"Granular-ball splitting did not converge within {max_rounds} rounds"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
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
    local_feature_selection_cache: LocalFeatureSelectionCache | None = None,
    ball_parallel_jobs: int = 1,
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
        local_feature_selection_cache=local_feature_selection_cache,
        ball_parallel_jobs=ball_parallel_jobs,
    )
    return anchors_from_balls(balls), balls, selected_counts
