from __future__ import annotations

"""Configuration for the MY-V1 clustering algorithm."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MYV1Config:
    """MY-V1 parameters.

    ``p1`` and ``p2`` keep the same fixed-cardinality roles as in PLGB-FSC.
    ``pdmf_neighbors`` controls the local Gaussian-PDMF spread construction;
    ``graph_neighbors`` and ``pdmf_similarity_lambda`` control the sparse
    Gaussian-PDMF graph used only by local attribute reduction.  No true or
    pseudo labels are used by either attribute-reduction stage.
    """

    p1: int
    p2: int
    purity: float = 0.95
    pdmf_neighbors: int = 5
    pdmf_epsilon: float = 1e-8
    graph_neighbors: int = 5
    pdmf_similarity_lambda: float = 0.5
    anchor_neighbors: int = 5
    weighted_kmeans_beta: float = 3.0
    weighted_kmeans_max_iter: int = 20
    split_kmeans_max_iter: int = 3
    tcut_kmeans_max_iter: int = 100
    tcut_kmeans_n_init: int = 3
    keep_matlab_split_rule: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.p1, bool) or not isinstance(self.p1, int):
            raise TypeError("p1 must be an integer")
        if isinstance(self.p2, bool) or not isinstance(self.p2, int):
            raise TypeError("p2 must be an integer")
        if self.p1 < 2:
            raise ValueError("p1 must be at least 2")
        if self.p2 < 1 or self.p2 >= self.p1:
            raise ValueError("p2 must satisfy 1 <= p2 < p1")
        if not math.isfinite(self.purity) or not 0 < self.purity <= 1:
            raise ValueError("purity must be a finite number in (0, 1]")
        if self.pdmf_neighbors < 1:
            raise ValueError("pdmf_neighbors must be at least 1")
        if not math.isfinite(self.pdmf_epsilon) or self.pdmf_epsilon <= 0:
            raise ValueError("pdmf_epsilon must be a finite positive number")
        if self.graph_neighbors < 1:
            raise ValueError("graph_neighbors must be at least 1")
        if (
            not math.isfinite(self.pdmf_similarity_lambda)
            or not 0 < self.pdmf_similarity_lambda < 1
        ):
            raise ValueError(
                "pdmf_similarity_lambda must be a finite number in (0, 1)"
            )
        if self.anchor_neighbors < 1:
            raise ValueError("anchor_neighbors must be at least 1")
        if (
            not math.isfinite(self.weighted_kmeans_beta)
            or self.weighted_kmeans_beta <= 1
        ):
            raise ValueError(
                "weighted_kmeans_beta must be a finite number greater than 1"
            )
        if self.weighted_kmeans_max_iter < 1:
            raise ValueError("weighted_kmeans_max_iter must be at least 1")
        if self.split_kmeans_max_iter < 1:
            raise ValueError("split_kmeans_max_iter must be at least 1")
        if self.tcut_kmeans_max_iter < 1:
            raise ValueError("tcut_kmeans_max_iter must be at least 1")
        if self.tcut_kmeans_n_init < 1:
            raise ValueError("tcut_kmeans_n_init must be at least 1")
        if not isinstance(self.keep_matlab_split_rule, bool):
            raise TypeError("keep_matlab_split_rule must be a bool")
