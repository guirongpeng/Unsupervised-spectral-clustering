from __future__ import annotations

import numpy as np

from algorithms.sagbc import SAGBC, SAGBCConfig
from config import ExperimentConfig
from run import _create_model, _validate_algorithm_config


def test_sagbc_produces_one_label_per_sample() -> None:
    X = np.r_[np.linspace(0, 0.7, 32)[:, None], np.linspace(4, 4.7, 32)[:, None]]
    labels = SAGBC(SAGBCConfig(sample_size=64, random_state=1)).fit_predict(X)
    assert labels.shape == (64,)
    assert np.all(labels >= -1)


def test_sagbc_is_available_to_the_benchmark() -> None:
    _validate_algorithm_config("sagbc")
    model = _create_model("sagbc", ExperimentConfig(), p1=None, p2=None, theta=0.0, n_clusters=2, seed=1, sagbc_sample_size=8)
    assert isinstance(model, SAGBC)

