from __future__ import annotations

import numpy as np

from algorithms.gbsc import GBSC, GBSCConfig
from config import ExperimentConfig
from run import _create_model, _validate_algorithm_config


def test_gbsc_assigns_a_label_to_every_sample() -> None:
    X = np.array([[0.0], [0.1], [0.2], [0.3], [4.0], [4.1], [4.2], [4.3]])
    model = GBSC(GBSCConfig(sigma=1.0), n_clusters=2, random_state=1)
    labels = model.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert np.all(labels >= 0)
    assigned = np.concatenate([ball.sample_indices for ball in model.granular_balls_])
    assert np.array_equal(np.sort(assigned), np.arange(X.shape[0]))


def test_gbsc_is_available_to_the_benchmark() -> None:
    _validate_algorithm_config("gbsc")
    model = _create_model(
        "gbsc", ExperimentConfig(), p1=None, p2=None, theta=0.0,
        n_clusters=2, seed=1, sigma=1.0,
    )
    assert isinstance(model, GBSC)
