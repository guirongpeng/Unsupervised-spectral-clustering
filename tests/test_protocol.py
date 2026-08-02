from __future__ import annotations

import math

import numpy as np

from algorithms.my_v0 import MYV0
from algorithms.my_v1 import MYV1
from algorithms.plgb_fsc import PLGBFSC
from config import DATASETS, PLGB_FSC_PARAMS, ExperimentConfig
from core.data import load_dataset
from core.metrics import evaluate_clustering
from run import _create_model


def test_sucancer_protocol_can_be_configured_with_three_seeds() -> None:
    config = ExperimentConfig(datasets=("SuCancer",), seeds=(1, 2, 3))
    assert config.seeds == (1, 2, 3)
    assert math.ceil(7909 * PLGB_FSC_PARAMS["p1_ratio"]) == 5932
    assert all(p2 < 5932 for p2 in PLGB_FSC_PARAMS["p2_values"])


def test_run_can_create_both_algorithms() -> None:
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
    assert isinstance(plgb, PLGBFSC)
    assert isinstance(my_v0, MYV0)
    assert isinstance(my_v1, MYV1)


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
