from __future__ import annotations

import numpy as np

from algorithms.my_v0 import MYV0
from algorithms.my_v0.feature_selection import resolve_pdmf_neighbor_count
from algorithms.my_v1 import MYV1
from algorithms.my_v1.feature_selection import (
    resolve_graph_neighbor_count,
    resolve_pdmf_neighbor_count as resolve_my_v1_pdmf_neighbor_count,
)
from algorithms.my_v2 import MYV2
from algorithms.plgb_fsc import PLGBFSC
from config import (
    DATASETS,
    MY_V0_PARAMS,
    MY_V1_PARAMS,
    MY_V2_PARAMS,
    PLGB_FSC_PARAMS,
    ExperimentConfig,
)
from core.data import load_dataset
from core.metrics import evaluate_clustering
from run import (
    _create_model,
    _resolve_p1_values,
    _resolve_p2_values,
    _resolve_pdmf_neighbor_settings,
    _resolve_graph_neighbor_settings,
    _resolve_similarity_lambda_settings,
    _resolve_stability_delta_settings,
)


def test_sucancer_protocol_can_be_configured_with_three_seeds() -> None:
    config = ExperimentConfig(datasets=("SuCancer",), seeds=(1, 2, 3))
    assert config.seeds == (1, 2, 3)
    assert _resolve_p1_values("plgb_fsc", 7909) == (5932,)
    assert all(p2 < 5932 for p2 in PLGB_FSC_PARAMS["p2_values"])


def test_plgb_p1_supports_counts_and_ratios(monkeypatch) -> None:
    monkeypatch.setitem(PLGB_FSC_PARAMS, "p1_counts", (150,))
    monkeypatch.setitem(PLGB_FSC_PARAMS, "p1_ratios", (0.1, 0.2))
    assert _resolve_p1_values("plgb_fsc", 1000) == (100, 150, 200)


def test_my_v0_grid_supports_counts_and_ratios(monkeypatch) -> None:
    monkeypatch.setitem(MY_V0_PARAMS, "p1_counts", (6,))
    monkeypatch.setitem(MY_V0_PARAMS, "p1_ratios", (0.5,))
    monkeypatch.setitem(MY_V0_PARAMS, "p2_counts", (2,))
    monkeypatch.setitem(MY_V0_PARAMS, "p2_ratios", (0.5,))
    monkeypatch.setitem(MY_V0_PARAMS, "pdmf_neighbors_counts", (3,))
    monkeypatch.setitem(MY_V0_PARAMS, "pdmf_neighbors_ratios", (0.25,))
    assert _resolve_p1_values("my_v0", 20) == (6, 10)
    assert _resolve_p2_values("my_v0", 10) == (2, 5)
    assert _resolve_pdmf_neighbor_settings("my_v0") == (3, 0.25)
    assert resolve_pdmf_neighbor_count(0.25, 21) == 5


def test_my_v1_grid_supports_counts_and_ratios(monkeypatch) -> None:
    monkeypatch.setitem(MY_V1_PARAMS, "p1_counts", (6,))
    monkeypatch.setitem(MY_V1_PARAMS, "p1_ratios", (0.5,))
    monkeypatch.setitem(MY_V1_PARAMS, "p2_counts", (2,))
    monkeypatch.setitem(MY_V1_PARAMS, "p2_ratios", (0.5,))
    monkeypatch.setitem(MY_V1_PARAMS, "pdmf_neighbors_counts", (3,))
    monkeypatch.setitem(MY_V1_PARAMS, "pdmf_neighbors_ratios", (0.25,))
    monkeypatch.setitem(MY_V1_PARAMS, "graph_neighbors_counts", (4,))
    monkeypatch.setitem(MY_V1_PARAMS, "graph_neighbors_ratios", (0.2,))
    monkeypatch.setitem(MY_V1_PARAMS, "pdmf_similarity_lambda_ratios", (0.3, 0.7))
    assert _resolve_p1_values("my_v1", 20) == (6, 10)
    assert _resolve_p2_values("my_v1", 10) == (2, 5)
    assert _resolve_pdmf_neighbor_settings("my_v1") == (3, 0.25)
    assert _resolve_graph_neighbor_settings("my_v1") == (4, 0.2)
    assert _resolve_similarity_lambda_settings("my_v1") == (0.3, 0.7)
    assert resolve_my_v1_pdmf_neighbor_count(0.25, 21) == 5
    assert resolve_graph_neighbor_count(0.2, 21) == 4


def test_my_v2_grid_uses_adaptive_counts(monkeypatch) -> None:
    monkeypatch.setitem(MY_V2_PARAMS, "stability_delta_values", (0.02, 0.05))
    monkeypatch.setitem(MY_V2_PARAMS, "pdmf_neighbors_counts", (3,))
    monkeypatch.setitem(MY_V2_PARAMS, "pdmf_neighbors_ratios", (0.25,))
    monkeypatch.setitem(MY_V2_PARAMS, "graph_neighbors_counts", (4,))
    monkeypatch.setitem(MY_V2_PARAMS, "graph_neighbors_ratios", (0.2,))
    monkeypatch.setitem(MY_V2_PARAMS, "pdmf_similarity_lambda_ratios", (0.3, 0.7))
    assert _resolve_stability_delta_settings("my_v2") == (0.02, 0.05)
    assert _resolve_pdmf_neighbor_settings("my_v2") == (3, 0.25)
    assert _resolve_graph_neighbor_settings("my_v2") == (4, 0.2)
    assert _resolve_similarity_lambda_settings("my_v2") == (0.3, 0.7)


def test_run_can_create_all_algorithms() -> None:
    config = ExperimentConfig()
    plgb = _create_model(
        "plgb_fsc", config, p1=20, p2=5, theta=0.95, n_clusters=2, seed=1
    )
    my_v0 = _create_model(
        "my_v0", config, p1=20, p2=5, theta=0.95, n_clusters=2, seed=1
    )
    my_v1 = _create_model(
        "my_v1", config, p1=20, p2=5, theta=0.95, n_clusters=2, seed=1
    )
    my_v2 = _create_model(
        "my_v2",
        config,
        p1=None,
        p2=None,
        theta=0.95,
        n_clusters=2,
        seed=1,
        stability_delta=0.05,
    )
    assert isinstance(plgb, PLGBFSC)
    assert isinstance(my_v0, MYV0)
    assert isinstance(my_v1, MYV1)
    assert isinstance(my_v2, MYV2)


def test_dataset_catalog_contains_both_papers_datasets() -> None:
    assert set(DATASETS) == {
        "COIL20",
        "ORL",
        "SuCancer",
        "USPS",
        "Yale",
        "warpPIE10P",
        "GLIOMA",
        "TOX_171",
        "ALLAML",
        "D3",
        "T4",
        "E6",
        "PenDigits",
        "Letter",
        "Covertype",
    }
    assert all(config.path.suffix == ".npz" for config in DATASETS.values())


def test_new_standardized_dataset_can_be_loaded() -> None:
    yale = load_dataset(DATASETS["Yale"])
    assert yale.X.shape == (165, 1024)
    assert yale.n_classes == 15


def test_metrics_are_perfect_for_identical_labels() -> None:
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate_clustering(labels, labels)
    assert metrics.acc == 1.0
    assert metrics.nmi == 1.0
    assert metrics.f_measure == 1.0
