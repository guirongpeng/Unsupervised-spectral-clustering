from __future__ import annotations

"""Configuration for the MY-V4 clustering algorithm."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MYV4Config:
    """MY-V4 parameters.

    ``p1`` and ``p2`` keep the same fixed-cardinality roles as in PLGB-FSC.
    ``pdmf_neighbors`` controls the local Gaussian-PDMF spread construction;
    ``graph_neighbors`` and ``pdmf_similarity_lambda`` control the sparse
    Gaussian-PDMF graph.  Pseudo-label confidence is computed internally and
    is not an additional tunable parameter.
    """

    p1: int
    p2: int
    purity: float = 0.95
    pdmf_neighbors: int | float = 5
    pdmf_epsilon: float = 1e-8
    graph_neighbors: int | float = 5
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
        self._validate_neighbor_setting("pdmf_neighbors", self.pdmf_neighbors)
        if not math.isfinite(self.pdmf_epsilon) or self.pdmf_epsilon <= 0:
            raise ValueError("pdmf_epsilon must be a finite positive number")
        self._validate_neighbor_setting("graph_neighbors", self.graph_neighbors)
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

    @staticmethod
    def _validate_neighbor_setting(name: str, value: int | float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an integer count or float ratio")
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"{name} count must be at least 1")
        elif not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} ratio must be a finite number in (0, 1]")
