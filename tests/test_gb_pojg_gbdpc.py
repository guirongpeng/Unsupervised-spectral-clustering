from __future__ import annotations

import numpy as np

from algorithms.gb_pojg_gbdpc import GBPOJGGBDPC, GBPOJGGBDPCConfig
from run import _create_model, _validate_algorithm_config
from config import ExperimentConfig


def test_gb_pojg_gbdpc_generates_a_label_for_every_sample() -> None:
    X = np.array([[0.0], [0.1], [0.2], [4.0], [4.1], [4.2]])
    model = GBPOJGGBDPC(GBPOJGGBDPCConfig(gamma=0.0, delta=0.5), n_clusters=2)
    labels = model.fit_predict(X)
    assert labels.shape == (6,)
    assert set(labels) == {1, 2}
    assigned = np.concatenate([ball.sample_indices for ball in model.granular_balls_])
    assert np.array_equal(np.sort(assigned), np.arange(X.shape[0]))


def test_gb_pojg_gbdpc_is_available_to_the_benchmark() -> None:
    _validate_algorithm_config("gb_pojg_gbdpc")
    model = _create_model(
        "gb_pojg_gbdpc",
        ExperimentConfig(),
        p1=None,
        p2=None,
        theta=0.0,
        n_clusters=2,
        seed=1,
        gamma=0.0,
        delta=0.5,
    )
    assert isinstance(model, GBPOJGGBDPC)
