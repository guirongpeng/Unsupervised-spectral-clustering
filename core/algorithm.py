from __future__ import annotations

"""所有 Benchmark 算法必须遵循的最小接口。"""

from abc import ABC, abstractmethod

import numpy as np


class Algorithm(ABC):
    """聚类算法统一接口。

    算法的配置应在构造函数中传入并由算法自己的 Config 管理。算法只接收
    特征矩阵 ``X``，不得读取数据文件、真实标签或计算评价指标。
    """

    labels_: np.ndarray

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Algorithm":
        """在特征矩阵 ``X`` 上执行聚类并设置 ``labels_``。"""

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """执行聚类并返回一维标签数组。"""

        fitted = self.fit(X)
        if not hasattr(fitted, "labels_"):
            raise RuntimeError(
                f"{type(self).__name__}.fit() must set the labels_ attribute"
            )
        return np.asarray(fitted.labels_).reshape(-1)

    @abstractmethod
    def get_params(self) -> dict[str, object]:
        """返回当前算法参数，供实验记录使用。"""
