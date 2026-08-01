from __future__ import annotations

"""Transfer Cut 二分图降阶工具。

PLGB-FSC 和 LSC 都会先构造“样本-锚点”二分图，再通过 Transfer Cut
把 n 个样本的谱分解问题压缩为 m 个锚点的谱分解问题，从而降低复杂度。
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .matlab_kmeans import litekmeans_like


@dataclass(frozen=True)
class TransferCutResult:
    """Transfer Cut 的输出。"""

    labels: np.ndarray  # 最终聚类标签。
    embedding: np.ndarray  # 每个样本映射后的低维谱嵌入。
    graph: sparse.csr_matrix  # 样本-锚点稀疏相似度矩阵 B。
    sigma: float  # 高斯核带宽，取近邻距离均值。


def build_sample_anchor_graph(
    X: np.ndarray,
    anchors: np.ndarray,
    k_neighbors: int = 5,
) -> tuple[sparse.csr_matrix, float]:
    """构建样本-锚点稀疏二分图。

    对每个样本只连接最近的 k 个锚点，边权使用高斯核，带宽 sigma
    对应 MATLAB `Tcut.m` 中的 `mean(knnDist(:))`。
    """

    X = np.asarray(X, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    n_samples = X.shape[0]
    n_anchors = anchors.shape[0]
    if n_anchors == 0:
        raise ValueError("At least one anchor is required")
    k = max(1, min(int(k_neighbors), n_anchors))

    # 查找每个样本最近的 k 个锚点，等价于 MATLAB knnsearch(centerDist, fea, 'k', K)。
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    neighbors.fit(anchors)
    distances, indices = neighbors.kneighbors(X, return_distance=True)

    sigma = float(np.mean(distances))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = np.finfo(float).eps

    # 高斯核权重：exp(-dist^2 / (2*sigma^2))。
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    weights[weights == 0] = np.finfo(float).eps
    rows = np.repeat(np.arange(n_samples), k)
    graph = sparse.csr_matrix((weights.reshape(-1), (rows, indices.reshape(-1))), shape=(n_samples, n_anchors))
    return graph, sigma


def transfer_cut_embedding(B: sparse.csr_matrix, n_clusters: int) -> np.ndarray:
    """执行 Transfer Cut 谱嵌入。

    公式对应 MATLAB `Tcut.m`：
    ``Wy = B' * Dx * B``，再对 ``D * Wy * D`` 做特征分解，最后把锚点
    特征向量映射回样本空间。
    """

    _, n_anchors = B.shape
    if n_anchors < n_clusters:
        raise ValueError(f"Need at least {n_clusters} anchors, got {n_anchors}")

    # Dx 是样本度矩阵的逆，对应 Dx = diag(1 ./ sum(B,2))。
    dx = np.asarray(B.sum(axis=1)).reshape(-1)
    dx = np.where(dx == 0, 1e-10, dx)
    inv_dx = 1.0 / dx
    Dx = sparse.diags(inv_dx)
    # Wy 是锚点侧的压缩图，规模为 m x m，避免直接分解 n x n 矩阵。
    Wy = B.T @ Dx @ B

    # 对锚点图做对称归一化。
    d = np.asarray(Wy.sum(axis=1)).reshape(-1)
    d = np.where(d <= 0, np.finfo(float).eps, d)
    inv_sqrt_d = 1.0 / np.sqrt(d)
    D = sparse.diags(inv_sqrt_d)
    nWy = D @ Wy @ D
    nWy_dense = np.asarray(((nWy + nWy.T) * 0.5).toarray())

    # MATLAB 中按特征值降序取前 k 个特征向量。
    eigenvalues, eigenvectors = np.linalg.eigh(nWy_dense)
    order = np.argsort(eigenvalues)[::-1]
    anchor_embedding = D @ eigenvectors[:, order[:n_clusters]]
    # 把锚点嵌入映射回样本嵌入：evec = Dx * B * Ncut_evec。
    sample_embedding = Dx @ B @ anchor_embedding
    sample_embedding = np.asarray(sample_embedding)
    # 行归一化，防止向量长度影响最终 KMeans。
    norms = np.sqrt(np.sum(sample_embedding * sample_embedding, axis=1, keepdims=True)) + 1e-10
    return sample_embedding / norms


def run_transfer_cut(
    X: np.ndarray,
    anchors: np.ndarray,
    n_clusters: int,
    k_neighbors: int = 5,
    kmeans_max_iter: int | None = None,
    kmeans_n_init: int | None = None,
    seed: int | None = None,
    clusterer: str = "sklearn",
) -> TransferCutResult:
    """完整执行二分图构建、谱嵌入和最终聚类。"""

    graph, sigma = build_sample_anchor_graph(X, anchors, k_neighbors=k_neighbors)
    embedding = transfer_cut_embedding(graph, n_clusters)
    if clusterer == "litekmeans":
        # PLGB-FSC 源码兼容路径：使用 MATLAB-like KMeans。
        litekmeans_kwargs = {"X": embedding, "n_clusters": n_clusters, "seed": seed}
        if kmeans_max_iter is not None:
            litekmeans_kwargs["max_iter"] = kmeans_max_iter
        if kmeans_n_init is not None:
            litekmeans_kwargs["replicates"] = kmeans_n_init
        labels = litekmeans_like(**litekmeans_kwargs).labels
    elif clusterer == "sklearn":
        # 普通基线可使用 sklearn 标准 KMeans。
        kmeans_kwargs = {"n_clusters": n_clusters, "random_state": seed}
        if kmeans_max_iter is not None:
            kmeans_kwargs["max_iter"] = kmeans_max_iter
        if kmeans_n_init is not None:
            kmeans_kwargs["n_init"] = kmeans_n_init
        labels = KMeans(**kmeans_kwargs).fit_predict(embedding)
    else:
        raise ValueError(f"Unknown Transfer Cut clusterer: {clusterer}")
    return TransferCutResult(labels=labels, embedding=embedding, graph=graph, sigma=sigma)
