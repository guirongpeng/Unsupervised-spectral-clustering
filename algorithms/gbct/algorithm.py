from __future__ import annotations

"""GBCT 官方 Python 源码的统一 Benchmark 实现。"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from core.algorithm import Algorithm

from .config import GBCTConfig


@dataclass(frozen=True, slots=True)
class GranularBall:
    """用样本下标保存粒球，避免官方浮点元组回查丢失重复样本。"""

    sample_indices: np.ndarray


class _UnionFind:
    """对应官方 ``UnionFind``，保留根节点合并方向。"""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        if value not in self.parent:
            self.parent[value] = value
            return value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            self.parent[first_root] = second_root


def _validate_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"X must not be empty, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    return values


def _radius(X: np.ndarray, ball: GranularBall) -> float:
    points = X[ball.sample_indices]
    center = np.mean(points, axis=0)
    return float(np.max(np.linalg.norm(points - center, axis=1)))


def _center(X: np.ndarray, ball: GranularBall) -> np.ndarray:
    return np.mean(X[ball.sample_indices], axis=0)


def _divide(number: float, denominator: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(number, denominator))


def _coarse_division(
    X: np.ndarray,
    config: GBCTConfig,
) -> list[GranularBall]:
    ball_count = int(np.sqrt(X.shape[0]))
    if ball_count < 1:
        raise RuntimeError("GBCT could not create an initial granular ball")
    labels = KMeans(
        n_clusters=ball_count,
        random_state=config.coarse_random_state,
        n_init=config.coarse_n_init,
        max_iter=config.coarse_max_iter,
    ).fit_predict(X)
    return [
        GranularBall(np.flatnonzero(labels == label))
        for label in range(ball_count)
    ]


def _split_two_means(
    X: np.ndarray,
    ball: GranularBall,
    config: GBCTConfig,
) -> tuple[GranularBall, GranularBall]:
    local_labels = KMeans(
        n_clusters=2,
        random_state=config.fine_random_state,
        n_init=config.fine_n_init,
        max_iter=config.fine_max_iter,
    ).fit_predict(X[ball.sample_indices])
    return (
        GranularBall(ball.sample_indices[local_labels == 0]),
        GranularBall(ball.sample_indices[local_labels == 1]),
    )


def _consistency_ratio(
    X: np.ndarray,
    ball: GranularBall,
    *,
    strong: bool,
) -> float:
    points = X[ball.sample_indices]
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    maximum_radius = float(np.max(distances))
    dimension = X.shape[1]

    if strong:
        inner_radius = maximum_radius / 4.0
    else:
        inner_radius = float(np.mean(distances))
    inner_count = int(np.count_nonzero(distances <= inner_radius))
    inner_density = _divide(inner_count, inner_radius**dimension)
    maximum_density = _divide(ball.sample_indices.size, maximum_radius**dimension)
    return _divide(inner_density, maximum_density)


def _should_split_from_ratio(ratio: float) -> bool:
    # 官方源码接受 [1, 1.3)，区间之外继续分裂。
    return bool(ratio >= 1.30 or ratio < 1.0)


def _density_or_radius(X: np.ndarray, ball: GranularBall) -> float:
    radius = _radius(X, ball)
    if ball.sample_indices.size > 2:
        return _divide(ball.sample_indices.size, radius ** X.shape[1])
    return radius


def _strong_consistency_division(
    X: np.ndarray,
    active: list[GranularBall],
    finalized: list[GranularBall],
    config: GBCTConfig,
) -> tuple[list[GranularBall], list[GranularBall]]:
    split_balls: list[GranularBall] = []
    for ball in active:
        if ball.sample_indices.size > 1:
            first, second = _split_two_means(X, ball, config)
            ratio = _consistency_ratio(X, ball, strong=True)
            large_enough = (
                first.sample_indices.size > 2
                and second.sample_indices.size > 2
            )
            if _should_split_from_ratio(ratio) and large_enough:
                split_balls.extend((first, second))
            else:
                finalized.append(ball)
        else:
            finalized.append(ball)
    return split_balls, finalized


def _consistency_division(
    X: np.ndarray,
    active: list[GranularBall],
    finalized: list[GranularBall],
    config: GBCTConfig,
) -> tuple[list[GranularBall], list[GranularBall]]:
    split_balls: list[GranularBall] = []
    for ball in active:
        if ball.sample_indices.size <= 1:
            finalized.append(ball)
            continue

        first, second = _split_two_means(X, ball, config)
        if first.sample_indices.size == 0 or second.sample_indices.size == 0:
            finalized.append(ball)
            continue

        ratio = _consistency_ratio(X, ball, strong=False)
        parent_density = _density_or_radius(X, ball)
        first_density = _density_or_radius(X, first)
        second_density = _density_or_radius(X, second)
        large_enough = (
            first.sample_indices.size > 2
            and second.sample_indices.size > 2
        )
        singleton_exception = (
            first_density >= parent_density
            or second_density >= parent_density
        ) and (
            first.sample_indices.size == 1
            or second.sample_indices.size == 1
        )

        if (
            _should_split_from_ratio(ratio)
            and (large_enough or singleton_exception)
        ):
            split_balls.extend((first, second))
        else:
            finalized.append(ball)
    return split_balls, finalized


def _farthest_point_split(
    X: np.ndarray,
    ball: GranularBall,
) -> tuple[GranularBall, GranularBall]:
    points = X[ball.sample_indices]
    distances = squareform(pdist(points, metric="euclidean"))
    rows, columns = np.where(distances == np.max(distances))
    first_endpoint = int(rows[1])
    second_endpoint = int(columns[1])

    first_mask = (
        distances[:, first_endpoint] < distances[:, second_endpoint]
    )
    return (
        GranularBall(ball.sample_indices[first_mask]),
        GranularBall(ball.sample_indices[~first_mask]),
    )


def _split_sparse_balls(
    X: np.ndarray,
    balls: list[GranularBall],
) -> list[GranularBall]:
    average_radius_per_sample = float(
        np.mean(
            [
                _radius(X, ball) / ball.sample_indices.size
                for ball in balls
            ]
        )
    )
    average_radius = float(np.mean([_radius(X, ball) for ball in balls]))

    retained: list[GranularBall] = []
    to_split: list[GranularBall] = []
    for ball in balls:
        radius = _radius(X, ball)
        if (
            radius / ball.sample_indices.size > average_radius_per_sample
            and radius > average_radius
        ):
            to_split.append(ball)
        else:
            retained.append(ball)

    for ball in to_split:
        if ball.sample_indices.size > 1:
            retained.extend(_farthest_point_split(X, ball))
        else:
            retained.append(ball)
    return retained


def _generate_granular_balls(
    X: np.ndarray,
    config: GBCTConfig,
) -> tuple[tuple[GranularBall, ...], int]:
    active = _coarse_division(X, config)
    coarse_ball_count = len(active)

    active, finalized = _strong_consistency_division(
        X,
        active,
        [],
        config,
    )
    active = active + finalized
    finalized = []

    while True:
        old_count = len(active) + len(finalized)
        active, finalized = _consistency_division(
            X,
            active,
            finalized,
            config,
        )
        new_count = len(active) + len(finalized)
        if new_count == old_count:
            active = finalized
            break

    active = [ball for ball in active if ball.sample_indices.size != 0]
    active = _split_sparse_balls(X, active)
    active = [ball for ball in active if ball.sample_indices.size != 0]

    assigned = np.concatenate([ball.sample_indices for ball in active])
    if assigned.size != X.shape[0] or not np.array_equal(
        np.sort(assigned),
        np.arange(X.shape[0]),
    ):
        raise RuntimeError("GBCT granular balls do not partition all samples")
    return tuple(active), coarse_ball_count


def _noise_division(
    X: np.ndarray,
    balls: tuple[GranularBall, ...],
    config: GBCTConfig,
) -> tuple[tuple[GranularBall, ...], tuple[GranularBall, ...]]:
    small_noise = [
        ball
        for ball in balls
        if ball.sample_indices.size < config.minimum_ball_size
    ]
    candidates = [
        ball
        for ball in balls
        if ball.sample_indices.size >= config.minimum_ball_size
    ]
    if not candidates:
        raise RuntimeError("GBCT removed every granular ball by size")

    densities = np.asarray(
        [
            _divide(ball.sample_indices.size, _radius(X, ball))
            for ball in candidates
        ],
        dtype=float,
    )
    average_density = float(np.mean(densities))
    threshold = config.noise_density_ratio * average_density
    sparse_noise = [
        ball
        for ball, density in zip(candidates, densities)
        if density < threshold
    ]
    retained = [
        ball
        for ball, density in zip(candidates, densities)
        if density >= threshold
    ]
    if not retained:
        raise RuntimeError("GBCT removed every granular ball as noise")
    return tuple(retained), tuple(small_noise + sparse_noise)


def _similarity_matrix(
    X: np.ndarray,
    balls: tuple[GranularBall, ...],
) -> np.ndarray:
    centers = np.asarray([_center(X, ball) for ball in balls])
    radii = np.asarray([_radius(X, ball) for ball in balls])
    center_distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )
    boundary_distances = (
        center_distances - radii[:, None] - radii[None, :]
    )
    minimum = float(np.min(boundary_distances))
    if minimum < 0:
        boundary_distances += 2.0 * abs(minimum)
    np.fill_diagonal(boundary_distances, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = 1.0 / boundary_distances
    similarity[np.isinf(similarity)] = 0.0
    return similarity


def _merge_sets(sets: list[set[int]]) -> list[set[int]]:
    union_find = _UnionFind()
    for values in sets:
        iterator = iter(values)
        first = next(iterator)
        for value in iterator:
            union_find.union(first, value)

    result: dict[int, set[int]] = {}
    for values in sets:
        for value in values:
            root = union_find.find(value)
            if root not in result:
                result[root] = set()
            result[root].add(value)
    return list(result.values())


def _cluster_similarity(
    clusters: list[list[int]],
    ball_similarity: np.ndarray,
) -> np.ndarray:
    matrix: list[list[float]] = []
    for first_cluster in clusters:
        row: list[float] = []
        for second_cluster in clusters:
            maximum = 0.0
            for first_ball in first_cluster:
                for second_ball in second_cluster:
                    maximum = max(
                        float(ball_similarity[first_ball, second_ball]),
                        maximum,
                    )
            row.append(maximum)
        matrix.append(row)
    result = np.asarray(matrix, dtype=float)
    np.fill_diagonal(result, 0.0)
    return result


def _map_merged_clusters(
    merged_cluster_indices: list[set[int]],
    current: dict[int, set[int]],
) -> list[set[int]]:
    mapped: list[set[int]] = []
    for cluster_indices in merged_cluster_indices:
        original_ball_indices: set[int] = set()
        for cluster_index in cluster_indices:
            original_ball_indices = original_ball_indices.union(
                current[cluster_index]
            )
        mapped.append(original_ball_indices)
    return mapped


def _merge_all_nearest(similarity: np.ndarray) -> list[set[int]]:
    sets: list[set[int]] = []
    for cluster_index, row in enumerate(similarity):
        sets.append({cluster_index, int(np.argmax(row))})
    return _merge_sets(sets)


def _merge_top_nearest(
    similarity: np.ndarray,
    n_clusters: int,
) -> list[set[int]]:
    ranked: list[tuple[int, int, float]] = []
    for cluster_index, row in enumerate(similarity):
        nearest = int(np.argmax(row))
        ranked.append(
            (cluster_index, nearest, float(similarity[cluster_index, nearest]))
        )
    ranked.sort(key=lambda item: item[2], reverse=True)

    merge_count = similarity.shape[0] - n_clusters
    sets = [
        {source, target}
        for source, target, _value in ranked[:merge_count]
    ]
    sets.extend({source} for source, _target, _value in ranked[merge_count:])
    return _merge_sets(sets)


def _pairwise_merge_to_k(
    current: dict[int, set[int]],
    ball_similarity: np.ndarray,
    n_clusters: int,
) -> dict[int, set[int]]:
    while len(current) > n_clusters:
        clusters = [list(values) for values in current.values()]
        similarity = _cluster_similarity(clusters, ball_similarity)
        flat_index = int(np.argmax(similarity))
        first, second = np.unravel_index(flat_index, similarity.shape)
        if first == second:
            raise RuntimeError("GBCT merging made no progress")

        merged_indices = [{int(first), int(second)}]
        merged_indices.extend(
            {index}
            for index in range(similarity.shape[0])
            if index not in {first, second}
        )
        merged = _map_merged_clusters(merged_indices, current)
        current = {
            index: values for index, values in enumerate(merged)
        }
    return current


def _merge_granular_balls(
    ball_similarity: np.ndarray,
    n_clusters: int,
) -> dict[int, set[int]]:
    ball_count = ball_similarity.shape[0]
    if ball_count < n_clusters:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the number of non-noise "
            f"granular balls ({ball_count})"
        )

    first_merged = _merge_all_nearest(ball_similarity)
    current = {
        index: values for index, values in enumerate(first_merged)
    }
    clusters = [list(values) for values in first_merged]
    epoch = 0
    changed = False
    previous: dict[int, set[int]] | None = None

    while len(clusters) > n_clusters:
        similarity = _cluster_similarity(clusters, ball_similarity)
        if epoch < 1:
            merged_indices = _merge_all_nearest(similarity)
        else:
            merged_indices = _merge_top_nearest(similarity, n_clusters)
        merged = _map_merged_clusters(merged_indices, current)

        epoch += 1
        previous = current
        changed = True
        current = {
            index: values for index, values in enumerate(merged)
        }
        clusters = [list(values) for values in merged]
        if previous == current:
            break

    if not changed:
        previous = {
            index: {index} for index in range(ball_count)
        }
    if len(clusters) <= n_clusters:
        if previous is None:
            raise RuntimeError("GBCT is missing its rollback partition")
        current = _pairwise_merge_to_k(
            previous,
            ball_similarity,
            n_clusters,
        )
    if len(current) != n_clusters:
        raise RuntimeError(
            f"GBCT produced {len(current)} clusters; expected {n_clusters}"
        )
    return current


class GBCT(Algorithm):
    """GBCT 的统一 Benchmark 接口。输入应由 Benchmark 完成 MinMax。"""

    def __init__(self, config: GBCTConfig) -> None:
        self.config = config

    def fit(self, X: np.ndarray) -> "GBCT":
        values = _validate_features(X)
        granular_balls, coarse_ball_count = _generate_granular_balls(
            values,
            self.config,
        )
        core_balls, noise_balls = _noise_division(
            values,
            granular_balls,
            self.config,
        )
        similarity = _similarity_matrix(values, core_balls)
        clusters = _merge_granular_balls(
            similarity,
            self.config.n_clusters,
        )

        core_ball_labels = np.full(len(core_balls), -1, dtype=int)
        for label, ball_indices in clusters.items():
            for ball_index in ball_indices:
                core_ball_labels[ball_index] = label
        if np.any(core_ball_labels < 0):
            raise RuntimeError("GBCT left a non-noise granular ball unlabeled")

        labels = np.full(values.shape[0], -1, dtype=int)
        for ball, label in zip(core_balls, core_ball_labels):
            labels[ball.sample_indices] = label

        core_centers = np.asarray([_center(values, ball) for ball in core_balls])
        noise_labels = np.empty(len(noise_balls), dtype=int)
        for noise_index, ball in enumerate(noise_balls):
            distances = np.linalg.norm(
                core_centers - _center(values, ball),
                axis=1,
            )
            nearest_ball = int(np.argmin(distances))
            noise_labels[noise_index] = core_ball_labels[nearest_ball]
            labels[ball.sample_indices] = noise_labels[noise_index]

        if np.any(labels < 0):
            raise RuntimeError("GBCT left one or more samples unlabeled")

        self.labels_ = labels
        self.coarse_ball_count_ = coarse_ball_count
        self.granular_balls_ = granular_balls
        self.core_balls_ = core_balls
        self.noise_balls_ = noise_balls
        self.ball_similarity_ = similarity
        self.core_ball_labels_ = core_ball_labels
        self.noise_ball_labels_ = noise_labels
        self.cluster_ball_indices_ = tuple(
            np.asarray(list(ball_indices), dtype=int)
            for ball_indices in clusters.values()
        )
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)


