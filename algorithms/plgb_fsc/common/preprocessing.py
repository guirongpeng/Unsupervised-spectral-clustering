from __future__ import annotations

"""公共预处理函数。

这里放多个算法都会用到的轻量预处理。当前主要是 MATLAB 风格的
min-max 归一化，用于 PLGB-FSC 伪标签、谱聚类基线和密度/层次聚类基线。
"""

import numpy as np


def minmax_scale_like_matlab(X: np.ndarray) -> np.ndarray:
    """按列做 min-max 缩放，尽量对应 MATLAB 中 ``(X-min)/(max-min)``。

    若某列是常数列，MATLAB 会产生 NaN；为了让实验流程继续可运行，
    这里把常数列分母替换为 1，使该列缩放后全为 0。
    """

    X = np.asarray(X, dtype=float)
    mins = np.min(X, axis=0)  # 每个特征维度的最小值。
    maxs = np.max(X, axis=0)  # 每个特征维度的最大值。
    ranges = maxs - mins
    safe_ranges = np.where(ranges == 0, 1.0, ranges)  # 避免常数列除零。
    return (X - mins) / safe_ranges
