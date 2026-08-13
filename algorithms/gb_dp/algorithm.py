from __future__ import annotations

"""GB-DP 主算法。

粒球生成、密度、delta 距离和标签传播遵循官方 ``GB-DP.py``。原实现需要
人工从决策图选择中心；统一 Benchmark 按已确认方案选择 ``rho * delta``
最大的前 ``n_clusters`` 个粒球。
"""

from dataclasses import asdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import k_means

from core.algorithm import Algorithm

from .config import GBDPConfig


def _validate_features(X: np.ndarray) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"X must not be empty, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    return values


def _split_ball(
    X: np.ndarray,
    sample_indices: np.ndarray,
    config: GBDPConfig,
) -> tuple[np.ndarray, ...]:
    """按官方 ``splits_ball`` 使用固定种子的 2-means 划分一个粒球。"""

    unique_count = np.unique(X[sample_indices], axis=0).shape[0]
    split_count = min(2, unique_count)
    labels = k_means(
        X=X[sample_indices],
        n_clusters=split_count,
        n_init=config.n_init,
        random_state=config.random_state,
    )[1]
    return tuple(
        sample_indices[labels == label] for label in range(split_count)
    )


def _generate_granular_balls(
    X: np.ndarray,
    config: GBDPConfig,
) -> tuple[tuple[np.ndarray, ...], int]:
    """对应论文 Algorithm 1 和官方循环，生成无监督粒球。"""

    max_ball_size = int(np.ceil(np.sqrt(X.shape[0])))
    granular_balls: tuple[np.ndarray, ...] = (
        np.arange(X.shape[0], dtype=int),
    )

    while True:
        updated: list[np.ndarray] = []
        for ball in granular_balls:
            if ball.size < max_ball_size:
                updated.append(ball)
            else:
                updated.extend(_split_ball(X, ball, config))
        if len(updated) == len(granular_balls):
            return tuple(updated), max_ball_size
        granular_balls = tuple(updated)


def _ball_statistics(
    X: np.ndarray,
    granular_balls: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算官方代码中的中心、半径、质量和 ``mean_r``。"""

    centers = np.asarray([X[ball].mean(axis=0) for ball in granular_balls])
    radii = np.asarray(
        [
            np.max(np.linalg.norm(X[ball] - center, axis=1))
            for ball, center in zip(granular_balls, centers)
        ],
        dtype=float,
    )
    masses = np.asarray([ball.size for ball in granular_balls], dtype=float)

    # 官方代码使用所有坐标绝对偏差的总体均值，而非论文式 (5) 的
    # 样本到中心的欧氏距离均值。此处按“官方代码优先”保留源码行为。
    mean_radii = np.asarray(
        [
            np.mean(np.abs(X[ball] - center))
            for ball, center in zip(granular_balls, centers)
        ],
        dtype=float,
    )
    return centers, radii, masses, mean_radii


def _ball_densities(
    radii: np.ndarray,
    masses: np.ndarray,
    mean_radii: np.ndarray,
) -> np.ndarray:
    """对应官方 ``ball_density2``。"""

    densities = np.zeros(radii.size, dtype=float)
    nonzero = radii != 0.0
    densities[nonzero] = masses[nonzero] / (
        radii[nonzero] ** 2 * mean_radii[nonzero]
    )
    return densities


def _delta_distances(
    centers: np.ndarray,
    densities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对应论文式 (7)-(9) 和官方 ``ball_min_dist``。"""

    distances = squareform(pdist(centers, metric="euclidean"))
    density_order = np.argsort(-densities)
    deltas = np.zeros(densities.size, dtype=float)
    nearest_higher = np.zeros(densities.size, dtype=int)

    for position, ball_index in enumerate(density_order):
        if position == 0:
            continue
        higher_density = density_order[:position]
        candidate_distances = distances[ball_index, higher_density]
        nearest_position = int(np.argmin(candidate_distances))
        deltas[ball_index] = candidate_distances[nearest_position]
        nearest_higher[ball_index] = higher_density[nearest_position]

    deltas[density_order[0]] = np.max(deltas)
    if np.max(deltas) < 1.0:
        deltas = deltas * 10.0
    return deltas, nearest_higher, density_order


def _select_centers(
    densities: np.ndarray,
    deltas: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """自动选择 ``rho * delta`` 最大的前 K 个粒球。"""

    if n_clusters > densities.size:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the number of generated "
            f"granular balls ({densities.size})"
        )
    decision_scores = densities * deltas
    center_indices = np.argsort(
        -decision_scores,
        kind="stable",
    )[:n_clusters]
    return center_indices, decision_scores


def _cluster_granular_balls(
    center_indices: np.ndarray,
    nearest_higher: np.ndarray,
    density_order: np.ndarray,
) -> np.ndarray:
    """按官方 DP 标签传播顺序为全部粒球分配类别。"""

    ball_labels = np.full(nearest_higher.size, -1, dtype=int)
    for cluster_id, ball_index in enumerate(center_indices, start=1):
        ball_labels[ball_index] = cluster_id

    for ball_index in density_order:
        if ball_labels[ball_index] == -1:
            parent = nearest_higher[ball_index]
            if parent == ball_index or ball_labels[parent] == -1:
                raise RuntimeError(
                    "GB-DP could not propagate a label from a higher-density ball"
                )
            ball_labels[ball_index] = ball_labels[parent]
    return ball_labels


class GBDP(Algorithm):
    """GB-DP 的统一 Benchmark 接口。"""

    def __init__(self, config: GBDPConfig) -> None:
        self.config = config

    def fit(self, X: np.ndarray) -> "GBDP":
        values = _validate_features(X)
        granular_balls, max_ball_size = _generate_granular_balls(
            values,
            self.config,
        )
        centers, radii, masses, mean_radii = _ball_statistics(
            values,
            granular_balls,
        )
        densities = _ball_densities(radii, masses, mean_radii)
        deltas, nearest_higher, density_order = _delta_distances(
            centers,
            densities,
        )
        center_indices, decision_scores = _select_centers(
            densities,
            deltas,
            self.config.n_clusters,
        )
        ball_labels = _cluster_granular_balls(
            center_indices,
            nearest_higher,
            density_order,
        )

        labels = np.empty(values.shape[0], dtype=int)
        for ball_index, sample_indices in enumerate(granular_balls):
            labels[sample_indices] = ball_labels[ball_index]

        self.labels_ = labels
        self.max_ball_size_ = max_ball_size
        self.granular_balls_ = granular_balls
        self.centers_ = centers
        self.radii_ = radii
        self.masses_ = masses
        self.mean_radii_ = mean_radii
        self.densities_ = densities
        self.deltas_ = deltas
        self.nearest_higher_ = nearest_higher
        self.density_order_ = density_order
        self.decision_scores_ = decision_scores
        self.center_ball_indices_ = center_indices
        self.ball_labels_ = ball_labels
        return self

    def get_params(self) -> dict[str, object]:
        return asdict(self.config)
