from __future__ import annotations

"""PLGB-FSC source path and unified Benchmark adapter."""

from dataclasses import asdict, dataclass

import numpy as np

from core.algorithm import Algorithm as BenchmarkAlgorithm
from .common.preprocessing import minmax_scale_like_matlab
from .common.transfer_cut import TransferCutResult, run_transfer_cut
from .config import PLGBFSCConfig
from .feature_selection import select_global_features_by_pseudo_label
from .granular_ball import GranularBall, generate_anchors
from .weighted_kmeans import weighted_kmeans


@dataclass(frozen=True)
class PLGBFSCResult:
    """PLGB-FSC 一次运行的完整中间结果和最终标签。"""

    labels: np.ndarray  # 最终聚类标签。
    pseudo_labels: np.ndarray  # 加权 KMeans 产生的伪标签。
    selected_feature_indices: np.ndarray  # 第一阶段全局特征选择保留的原始特征下标。
    mutual_info_scores: np.ndarray  # 每个原始特征与伪标签的互信息得分。
    anchors: np.ndarray  # 粒球均值形成的锚点矩阵。
    balls: list[GranularBall]  # 最终粒球列表，便于检查锚点来源。
    embedding: np.ndarray  # Transfer Cut 得到的样本低维嵌入。
    graph_sigma: float  # 样本-锚点高斯核带宽。
    p1: int  # 实际使用的全局特征数。
    p2: int  # 实际使用的局部特征数。
    transfer_cut: TransferCutResult  # Transfer Cut 的完整返回结果。

    @property
    def n_anchors(self) -> int:
        """锚点数量 m，也就是最终粒球数量。"""

        return int(self.anchors.shape[0])


def run_algorithm(
    X: np.ndarray,
    n_clusters: int,
    config: PLGBFSCConfig | None = None,
    seed: int | None = None,
) -> PLGBFSCResult:
    """Backward-compatible functional entry point."""

    if config is None:
        raise ValueError(
            "PLGB-FSC requires an explicit PLGBFSCConfig(p1=..., p2=...)"
        )
    return run_plgb_fsc(X, n_clusters, config, seed=seed)


def run_plgb_fsc(
    X: np.ndarray,
    n_clusters: int,
    config: PLGBFSCConfig,
    seed: int | None = None,
) -> PLGBFSCResult:
    """执行完整 PLGB-FSC 流程。"""

    X = _validate_features(X, n_clusters, config)
    p1 = config.p1
    p2 = config.p2

    # This scaling is inside pure_ball_hu.m and is used only by weighted
    # K-Means.  Global feature selection and Transfer Cut retain raw values.
    scaled = minmax_scale_like_matlab(X)
    pseudo_labels, _, _ = weighted_kmeans(
        scaled,
        n_clusters,
        beta=config.weighted_kmeans_beta,
        initial_weights=np.ones(X.shape[1], dtype=float),
        max_iter=config.weighted_kmeans_max_iter,
        seed=seed,
    )

    # 2) 用伪标签给每个原始特征打互信息分数，保留前 p1 个特征。
    X_selected, feature_indices, mi_scores = select_global_features_by_pseudo_label(X, pseudo_labels, p1)
    # 3) 在筛选后的特征空间中递归拆分粒球，并用每个粒球均值作为锚点。
    anchors, balls = generate_anchors(
        X_selected,
        pseudo_labels,
        p2,
        purity_threshold=config.purity,
        split_kmeans_max_iter=config.split_kmeans_max_iter,
        seed=seed,
        keep_matlab_split_rule=config.keep_matlab_split_rule,
    )
    # 4) 构建样本-锚点二分图，并用 Transfer Cut 得到最终聚类标签。
    tcut = run_transfer_cut(
        X_selected,
        anchors,
        n_clusters,
        k_neighbors=config.anchor_neighbors,
        kmeans_max_iter=config.tcut_kmeans_max_iter,
        kmeans_n_init=config.tcut_kmeans_n_init,
        seed=seed,
        clusterer="litekmeans",  # PLGB-FSC 源码兼容路径，贴近 MATLAB kmeans/litekmeans。
    )
    return PLGBFSCResult(
        labels=tcut.labels,
        pseudo_labels=pseudo_labels,
        selected_feature_indices=feature_indices,
        mutual_info_scores=mi_scores,
        anchors=anchors,
        balls=balls,
        embedding=tcut.embedding,
        graph_sigma=tcut.sigma,
        p1=p1,
        p2=p2,
        transfer_cut=tcut,
    )


def _validate_features(
    X: np.ndarray,
    n_clusters: int,
    config: PLGBFSCConfig,
) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"X must not be empty, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise TypeError("n_clusters must be an integer")
    if n_clusters < 2 or n_clusters > values.shape[0]:
        raise ValueError(
            "n_clusters must be in "
            f"[2, {values.shape[0]}], got {n_clusters}"
        )
    if config.p1 > values.shape[1]:
        raise ValueError(
            f"p1={config.p1} exceeds the {values.shape[1]} input features"
        )
    return values


class PLGBFSC(BenchmarkAlgorithm):
    """Unified adapter for the released ICDE 2025 PLGB-FSC algorithm."""

    def __init__(
        self,
        config: PLGBFSCConfig,
        n_clusters: int,
        random_state: int = 1,
    ) -> None:
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        self.config = config
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "PLGBFSC":
        result = run_plgb_fsc(
            X,
            self.n_clusters,
            self.config,
            seed=self.random_state,
        )
        self.labels_ = np.asarray(result.labels, dtype=int).reshape(-1)
        self.result_ = result
        self.pseudo_labels_ = result.pseudo_labels
        self.selected_feature_indices_ = result.selected_feature_indices
        self.mutual_info_scores_ = result.mutual_info_scores
        self.anchors_ = result.anchors
        self.granular_balls_ = result.balls
        self.embedding_ = result.embedding
        self.graph_sigma_ = result.graph_sigma
        return self

    def get_params(self) -> dict[str, object]:
        return {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            **asdict(self.config),
        }
