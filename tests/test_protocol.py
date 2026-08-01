from __future__ import annotations

import math

import numpy as np

from config import DATASETS, EXPERIMENT
from core.metrics import evaluate_clustering


def test_initial_protocol_is_sucancer_with_three_seeds() -> None:
    assert EXPERIMENT.datasets == ("SuCancer",)
    assert EXPERIMENT.seeds == (1, 2, 3)
    assert math.ceil(7909 * EXPERIMENT.p1_ratio) == 5932
    assert all(p2 < 5932 for p2 in EXPERIMENT.p2_values)


def test_dataset_catalog_contains_the_three_plgb_datasets() -> None:
    assert set(DATASETS) == {"COIL20", "ORL", "SuCancer"}


def test_metrics_are_perfect_for_identical_labels() -> None:
    labels = np.array([0, 0, 1, 1])
    metrics = evaluate_clustering(labels, labels)
    assert metrics.acc == 1.0
    assert metrics.nmi == 1.0
    assert metrics.f_measure == 1.0

