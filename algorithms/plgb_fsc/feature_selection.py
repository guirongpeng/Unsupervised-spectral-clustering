from __future__ import annotations

"""The two feature-selection stages in the released PLGB-FSC source."""

import numpy as np

from .common.preprocessing import minmax_scale_like_matlab


def mutual_info_source_compatible(x: np.ndarray, y: np.ndarray) -> float:
    """Reproduce the indexing behavior of ``pure_ball_hu.m/mutual_info``.

    The release first removes zero cells from ``p_xy`` and then reuses the
    shortened logical vector to index the two full marginal grids.  This is
    not the conventional histogram mutual-information formula, but retaining
    it is necessary to reproduce the published SuCancer path.
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size == 0:
        raise ValueError("x and y must not be empty")

    num_bins = min(max(10, x.size // 10), 100)
    x_counts, x_edges = np.histogram(x, bins=num_bins, density=False)
    y_counts, y_edges = np.histogram(y, bins=num_bins, density=False)
    xy_counts, _, _ = np.histogram2d(
        x,
        y,
        bins=[x_edges, y_edges],
        density=False,
    )

    sample_count = float(x.size)
    p_x = x_counts / sample_count
    p_y = y_counts / sample_count
    p_xy = xy_counts / sample_count
    positive_joint = p_xy[p_xy > 0]

    # MATLAB linear indexing is column-major.  After p_xy is shortened, the
    # source's ``p_xy > 0`` mask is all true and selects the first L elements
    # of each full marginal grid.
    p_x_grid, p_y_grid = np.meshgrid(p_x, p_y, indexing="ij")
    cell_count = positive_joint.size
    selected_p_x = p_x_grid.ravel(order="F")[:cell_count]
    selected_p_y = p_y_grid.ravel(order="F")[:cell_count]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = positive_joint * np.log2(
            positive_joint / (selected_p_x * selected_p_y)
        )
    return float(np.sum(terms))


def mutual_info_scores(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
) -> np.ndarray:
    """Score every feature with the source-compatible global criterion."""

    values = np.asarray(X, dtype=float)
    pseudo = np.asarray(pseudo_labels).reshape(-1)
    return np.array(
        [
            mutual_info_source_compatible(values[:, index], pseudo)
            for index in range(values.shape[1])
        ],
        dtype=float,
    )


def select_global_features_by_pseudo_label(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    p1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the ``p1`` largest source-compatible global scores."""

    scores = mutual_info_scores(X, pseudo_labels)
    indices = np.argsort(scores)[::-1][:p1]
    return X[:, indices], indices, scores


def select_local_features_by_discernibility(
    X: np.ndarray,
    p2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the released local discernibility score before 2-Means."""

    X = np.asarray(X, dtype=float)
    if X.shape[1] == 0:
        raise ValueError(
            "Cannot select local features from an empty feature matrix"
        )
    p2 = max(1, min(int(p2), X.shape[1]))

    # MATLAB std uses the sample standard deviation by default.
    ddof = 1 if X.shape[0] > 1 else 0
    stds = np.std(X, axis=0, ddof=ddof)

    if X.shape[0] < 2 or X.shape[1] == 1:
        corr_abs_sum = np.ones(X.shape[1], dtype=float)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(
            corr,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        np.fill_diagonal(corr, 1.0)
        corr_abs_sum = np.sum(np.abs(corr), axis=0)
        corr_abs_sum = np.where(
            corr_abs_sum == 0,
            np.finfo(float).eps,
            corr_abs_sum,
        )

    independence = 1.0 / corr_abs_sum
    max_std_index = int(np.argmax(stds))
    min_abs_sum = float(np.min(np.abs(corr_abs_sum)))
    if min_abs_sum <= 0:
        min_abs_sum = np.finfo(float).eps
    independence[max_std_index] = 1.0 / min_abs_sum

    scores = independence * stds
    indices = np.argsort(scores)[::-1][:p2]
    selected = minmax_scale_like_matlab(X[:, indices])
    return selected, indices, scores
