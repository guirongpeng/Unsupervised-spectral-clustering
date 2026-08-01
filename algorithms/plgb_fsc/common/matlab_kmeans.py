from __future__ import annotations

"""MATLAB 风格 K-Means 辅助实现。

该模块用于 PLGB-FSC 的源码兼容路径，尽量贴近 `litekmeans.m` 中
sqEuclidean + sample 初始化 + 缺失簇修复的行为。普通基线 KMeans 仍使用 sklearn。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatlabKMeansResult:
    """MATLAB-like KMeans 的返回结果。"""

    labels: np.ndarray  # 0-based 聚类标签。
    centers: np.ndarray  # 聚类中心矩阵，形状为 k x d。
    converged: bool  # 是否在 max_iter 之前收敛。
    distances: np.ndarray  # 每个样本到每个中心的欧式距离。
    sum_distances: np.ndarray  # 每个簇内部距离和，用于选择最佳 replicate。


def squared_euclidean_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """向量化计算样本到中心的平方欧式距离矩阵。"""

    X_norm = np.sum(X * X, axis=1)[:, None]
    c_norm = np.sum(centers * centers, axis=1)[None, :]
    distances = X_norm + c_norm - 2.0 * X @ centers.T
    return np.maximum(distances, 0.0)  # 数值误差可能导致极小负数，截断为 0。


def litekmeans_like(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    replicates: int = 1,
    seed: int | None = None,
) -> MatlabKMeansResult:
    """接近 MATLAB `litekmeans.m` 的 KMeans。

    主要用于两个位置：
    1. PLGB-FSC 粒球拆分中的 2-Means；
    2. Transfer Cut 低维嵌入后的最终 KMeans。
    """

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n_samples = X.shape[0]
    if n_clusters <= 0 or n_clusters > n_samples:
        raise ValueError(f"n_clusters={n_clusters} must be in [1, {n_samples}]")

    rng = np.random.default_rng(seed)
    best: MatlabKMeansResult | None = None
    best_total = np.inf
    for _ in range(max(1, int(replicates))):
        # sample 初始化：随机抽 k 个样本作为初始中心，对应 litekmeans 的 Start='sample' 分支。
        centers = X[rng.choice(n_samples, size=n_clusters, replace=False)].copy()
        labels = np.ones(n_samples, dtype=int) * -1
        last = np.zeros(n_samples, dtype=int) * -2
        distances = squared_euclidean_distances(X, centers)

        iteration = 0
        # 标签不再变化或达到最大迭代次数时停止。
        while np.any(labels != last) and iteration < max_iter:
            last = labels.copy()
            distances = squared_euclidean_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            nearest = distances[np.arange(n_samples), labels]
            # MATLAB litekmeans 会修复空簇：把距离大的样本分配给缺失簇。
            labels = repair_missing_clusters(labels, nearest, n_clusters)
            centers = recompute_centers(X, labels, n_clusters)
            iteration += 1
        converged = iteration < max_iter

        # replicate 选择使用簇内欧式距离和，贴近 litekmeans 的 bestsumD 逻辑。
        distances = np.sqrt(squared_euclidean_distances(X, centers))
        sum_distances = np.array([distances[labels == i, i].sum() for i in range(n_clusters)])
        total = float(sum_distances.sum())
        result = MatlabKMeansResult(
            labels=labels,
            centers=centers,
            converged=converged,
            distances=distances,
            sum_distances=sum_distances,
        )
        if total < best_total:
            best_total = total
            best = result

    if best is None:
        raise RuntimeError("litekmeans_like failed to produce labels")
    return best


def repair_missing_clusters(labels: np.ndarray, nearest_distances: np.ndarray, n_clusters: int) -> np.ndarray:
    """修复空簇，把当前误差最大的样本分配给缺失簇。"""

    present = np.unique(labels)
    if present.size == n_clusters:
        return labels
    repaired = labels.copy()
    missing = [cluster for cluster in range(n_clusters) if cluster not in present]
    farthest = np.argsort(nearest_distances)[::-1]
    for sample_index, cluster in zip(farthest, missing):
        repaired[sample_index] = cluster
    return repaired


def recompute_centers(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """根据当前标签重新计算每个簇的均值中心。"""

    centers = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for cluster in range(n_clusters):
        members = X[labels == cluster]
        if members.size:
            centers[cluster] = members.mean(axis=0)
    return centers
