from __future__ import annotations

"""MY-V3 clustering pipeline and unified Benchmark adapter."""

from dataclasses import asdict, dataclass

import numpy as np

from core.algorithm import Algorithm as BenchmarkAlgorithm
from .common.preprocessing import minmax_scale_like_matlab
from .common.transfer_cut import TransferCutResult, run_transfer_cut
from .config import MYV3Config
from .feature_selection import select_global_features_by_gaussian_pdmf
from .granular_ball import GranularBall, generate_anchors
from .weighted_kmeans import weighted_kmeans


@dataclass(frozen=True)
class MYV3Result:
    """Complete result and inspectable intermediate state of one MY-V3 run."""

    labels: np.ndarray
    pseudo_labels: np.ndarray
    selected_feature_indices: np.ndarray
    attribute_scores: np.ndarray
    anchors: np.ndarray
    balls: list[GranularBall]
    embedding: np.ndarray
    graph_sigma: float
    p1: int
    p2: int
    transfer_cut: TransferCutResult

    @property
    def n_anchors(self) -> int:
        return int(self.anchors.shape[0])


def run_algorithm(
    X: np.ndarray,
    n_clusters: int,
    config: MYV3Config | None = None,
    seed: int | None = None,
) -> MYV3Result:
    """Conventional functional entry point used by direct tests."""

    if config is None:
        raise ValueError("MY-V3 requires an explicit MYV3Config(p1=..., p2=...)")
    return run_my_v3(X, n_clusters, config, seed=seed)


def run_my_v3(
    X: np.ndarray,
    n_clusters: int,
    config: MYV3Config,
    seed: int | None = None,
    precomputed_pseudo_labels: np.ndarray | None = None,
    precomputed_global_selection: tuple[np.ndarray, np.ndarray] | None = None,
    root_feature_ranking_cache: dict[str, np.ndarray] | None = None,
) -> MYV3Result:
    """Run the complete MY-V3 clustering pipeline.

    The PLGB-FSC framework is retained, while both the global and the
    global reduction uses Gaussian-PDMF entropy, while granular-ball-local
    reduction combines entropy and sparse Gaussian-PDMF graph importance.
    """

    values = _validate_features(X, n_clusters, config)

    # Component 1: weighted K-Means produces pseudo labels without true labels.
    if precomputed_pseudo_labels is None:
        scaled = minmax_scale_like_matlab(values)
        pseudo_labels, _, _ = weighted_kmeans(
            scaled,
            n_clusters,
            beta=config.weighted_kmeans_beta,
            initial_weights=np.ones(values.shape[1], dtype=float),
            max_iter=config.weighted_kmeans_max_iter,
            seed=seed,
            compute_device=config.compute_device,
            gpu_chunk_size=config.gpu_chunk_size,
        )
    else:
        pseudo_labels = np.asarray(precomputed_pseudo_labels, dtype=int).reshape(-1)
        if pseudo_labels.size != values.shape[0]:
            raise ValueError("precomputed_pseudo_labels length must match X")
        pseudo_labels = pseudo_labels.copy()

    # Component 2: label-free global Gaussian-PDMF entropy reduction.
    if precomputed_global_selection is None:
        X_selected, feature_indices, attribute_scores = (
            select_global_features_by_gaussian_pdmf(
                values,
                config.p1,
                neighbors=config.pdmf_neighbors,
                epsilon=config.pdmf_epsilon,
                redundancy_beta=config.redundancy_beta,
            )
        )
    else:
        feature_indices, attribute_scores = precomputed_global_selection
        feature_indices = np.asarray(feature_indices, dtype=int).reshape(-1).copy()
        attribute_scores = np.asarray(attribute_scores, dtype=float).reshape(-1).copy()
        if feature_indices.size != config.p1:
            raise ValueError("cached global feature count must equal p1")
        if attribute_scores.size != values.shape[1]:
            raise ValueError("cached global score count must match X features")
        X_selected = values[:, feature_indices]

    # Component 3: each split combines local Gaussian-PDMF entropy importance
    # and sparse-graph importance before 2-Means.  Pseudo labels are used only
    # by the retained pseudo-purity stopping rule.
    anchors, balls = generate_anchors(
        X_selected,
        pseudo_labels,
        config.p2,
        purity_threshold=config.purity,
        pdmf_neighbors=config.pdmf_neighbors,
        pdmf_epsilon=config.pdmf_epsilon,
        graph_neighbors=config.graph_neighbors,
        pdmf_similarity_lambda=config.pdmf_similarity_lambda,
        redundancy_beta=config.redundancy_beta,
        fusion_alpha_mode=config.fusion_alpha_mode,
        mutual_knn=config.mutual_knn,
        self_tuning_graph=config.self_tuning_graph,
        split_kmeans_max_iter=config.split_kmeans_max_iter,
        seed=seed,
        keep_matlab_split_rule=config.keep_matlab_split_rule,
        root_ranking_cache=root_feature_ranking_cache,
        compute_device=config.compute_device,
        gpu_chunk_size=config.gpu_chunk_size,
    )

    # Component 4: retain the PLGB-FSC sample-anchor graph and Transfer Cut.
    tcut = run_transfer_cut(
        X_selected,
        anchors,
        n_clusters,
        k_neighbors=config.anchor_neighbors,
        kmeans_max_iter=config.tcut_kmeans_max_iter,
        kmeans_n_init=config.tcut_kmeans_n_init,
        seed=seed,
        clusterer="litekmeans",
        compute_device=config.compute_device,
        gpu_chunk_size=config.gpu_chunk_size,
    )
    return MYV3Result(
        labels=tcut.labels,
        pseudo_labels=pseudo_labels,
        selected_feature_indices=feature_indices,
        attribute_scores=attribute_scores,
        anchors=anchors,
        balls=balls,
        embedding=tcut.embedding,
        graph_sigma=tcut.sigma,
        p1=config.p1,
        p2=config.p2,
        transfer_cut=tcut,
    )


def _validate_features(
    X: np.ndarray,
    n_clusters: int,
    config: MYV3Config,
) -> np.ndarray:
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be a 2-D array, got shape {values.shape}")
    if values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError(
            f"X must contain at least two samples and one feature, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
        raise TypeError("n_clusters must be an integer")
    if n_clusters < 2 or n_clusters > values.shape[0]:
        raise ValueError(
            f"n_clusters must be in [2, {values.shape[0]}], got {n_clusters}"
        )
    if config.p1 > values.shape[1]:
        raise ValueError(
            f"p1={config.p1} exceeds the {values.shape[1]} input features"
        )
    return values


class MYV3(BenchmarkAlgorithm):
    """Unified algorithm adapter for MY-V3."""

    def __init__(
        self,
        config: MYV3Config,
        n_clusters: int,
        random_state: int = 1,
        precomputed_pseudo_labels: np.ndarray | None = None,
        precomputed_global_selection: tuple[np.ndarray, np.ndarray] | None = None,
        root_feature_ranking_cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        if isinstance(random_state, bool) or not isinstance(random_state, int):
            raise TypeError("random_state must be an integer")
        self.config = config
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._precomputed_pseudo_labels = precomputed_pseudo_labels
        self._precomputed_global_selection = precomputed_global_selection
        self._root_feature_ranking_cache = root_feature_ranking_cache

    def fit(self, X: np.ndarray) -> "MYV3":
        result = run_my_v3(
            X,
            self.n_clusters,
            self.config,
            seed=self.random_state,
            precomputed_pseudo_labels=self._precomputed_pseudo_labels,
            precomputed_global_selection=self._precomputed_global_selection,
            root_feature_ranking_cache=self._root_feature_ranking_cache,
        )
        self.labels_ = np.asarray(result.labels, dtype=int).reshape(-1)
        self.result_ = result
        self.pseudo_labels_ = result.pseudo_labels
        self.selected_feature_indices_ = result.selected_feature_indices
        self.attribute_scores_ = result.attribute_scores
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
