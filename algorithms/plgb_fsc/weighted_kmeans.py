from __future__ import annotations

"""PLGB-FSC 专属加权 KMeans。

该模块复刻 MATLAB `wkmeans.m` 中的加权 KMeans，用于生成伪标签。
它不是普通 KMeans：每轮会根据各维特征的簇内离散度更新特征权重。
"""

import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from .common.matlab_kmeans import litekmeans_like


def make_rng(seed: int | None = None) -> np.random.Generator:
    """创建随机数生成器，统一控制 Python 端随机性。"""

    return np.random.default_rng(seed)


def initialize_weighted_centroids(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """按 `wkmeans.m` 的策略初始化中心。

    第一个中心随机选；第二个中心取离第一个中心最远的样本；第三个及以后
    采用类似 kmeans++ 的距离平方概率抽样。
    """

    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    if k > n_samples:
        raise ValueError(f"k={k} cannot exceed n_samples={n_samples}")

    centers = np.zeros((k, n_features), dtype=float)
    first_idx = int(rng.integers(0, n_samples))
    centers[0] = X[first_idx]
    if k == 1:
        return centers

    # 第二个中心：选择距离第一个中心最远的样本。
    distances = np.linalg.norm(X - centers[0], axis=1)
    centers[1] = X[int(np.argmax(distances))]

    for cluster_idx in range(2, k):
        # 后续中心：按到最近已有中心的距离平方进行概率采样。
        _, dist_to_centers = pairwise_distances_argmin_min(X, centers[:cluster_idx], metric="euclidean")
        squared = dist_to_centers**2
        total = float(squared.sum())
        if total <= 0 or not np.isfinite(total):
            candidate = int(rng.integers(0, n_samples))
        else:
            candidate = int(rng.choice(n_samples, p=squared / total))
        centers[cluster_idx] = X[candidate]
    return centers


def assign_weighted_clusters(X: np.ndarray, centers: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """根据加权平方距离，把每个样本分配给最近中心。"""

    diff = X[:, None, :] - centers[None, :, :]
    # 对每个维度乘以特征权重 w_j，再求和。
    distances = np.sum(weights[None, None, :] * diff * diff, axis=2)
    distances = np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)
    return np.argmin(distances, axis=1)


def update_centers(X: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """根据当前标签更新每个簇中心。"""

    centers = np.zeros((k, X.shape[1]), dtype=float)
    for cluster_idx in range(k):
        mask = labels == cluster_idx
        if np.any(mask):
            centers[cluster_idx] = X[mask].mean(axis=0)
        elif rng is not None:
            # 理论上 wkmeans 没有显式空簇修复；这里给空簇随机中心以避免 NaN 中断。
            centers[cluster_idx] = X[int(rng.integers(0, X.shape[0]))]
    return centers


def update_feature_weights(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    beta: float,
) -> np.ndarray:
    """根据每个特征的簇内离散度更新权重。

    对应 MATLAB `computeWeight`：离散度越小的特征权重越大，说明它更有助于
    当前聚类划分。
    """

    k = centers.shape[0]
    dispersions = np.zeros(X.shape[1], dtype=float)
    for cluster_idx in range(k):
        cluster = X[labels == cluster_idx]
        if cluster.size:
            dispersions += np.sum((cluster - centers[cluster_idx]) ** 2, axis=0)

    if np.allclose(dispersions, 0):
        return np.ones(X.shape[1], dtype=float) / X.shape[1]

    safe = np.where(dispersions <= 0, np.finfo(float).eps, dispersions)
    exponent = 1.0 / (beta - 1.0)
    weights = np.zeros_like(safe)
    for j, value in enumerate(safe):
        ratios = value / safe
        weights[j] = 1.0 / np.sum(ratios**exponent)
    return np.nan_to_num(weights, nan=1.0 / X.shape[1], posinf=1.0, neginf=0.0)


def weighted_kmeans(
    X: np.ndarray,
    k: int,
    beta: float = 3.0,
    initial_weights: np.ndarray | None = None,
    max_iter: int = 20,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """执行加权 KMeans，返回伪标签、中心和最终特征权重。"""

    X = np.asarray(X, dtype=float)
    rng = make_rng(seed)
    weights = (
        np.ones(X.shape[1], dtype=float)
        if initial_weights is None
        else np.asarray(initial_weights, dtype=float).copy()
    )
    centers = initialize_weighted_centroids(X, k, rng)
    costs: list[float] = []
    labels = np.zeros(X.shape[0], dtype=int)

    for iteration in range(max_iter):
        labels = assign_weighted_clusters(X, centers, weights)
        centers = update_centers(X, labels, k, rng)
        weights = update_feature_weights(X, labels, centers, beta)
        cost = weighted_kmeans_cost(X, labels, centers, weights, beta)
        costs.append(round(cost, 4))
        # MATLAB 收敛条件：连续三次 round(cost,4) 相同，或达到最大迭代次数。
        if iteration >= 2 and costs[-1] == costs[-2] == costs[-3]:
            break
    return labels, centers, weights


def weighted_kmeans_cost(
    X: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    beta: float,
) -> float:
    """计算加权 KMeans 目标函数值。"""

    dispersions = np.zeros(X.shape[1], dtype=float)
    for cluster_idx in range(centers.shape[0]):
        cluster = X[labels == cluster_idx]
        if cluster.size:
            dispersions += np.sum((cluster - centers[cluster_idx]) ** 2, axis=0)
    return float(np.sum((weights**beta) * dispersions))


def two_means_labels(X: np.ndarray, max_iter: int = 3, seed: int | None = None) -> np.ndarray:
    """粒球拆分专用 2-Means，贴近 MATLAB `litekmeans(..., 2)`。"""

    return litekmeans_like(X, 2, max_iter=max_iter, replicates=1, seed=seed).labels
