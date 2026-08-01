from __future__ import annotations

"""统一聚类评价指标。"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
    rand_score,
)
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix


@dataclass(frozen=True)
class MetricResult:
    """一次聚类实验的统一外部评价指标。"""

    acc: float
    nmi: float
    ari: float
    ami: float
    f_measure: float
    macro_f1: float
    pairwise_f1: float
    fmi: float
    purity: float
    rand_index: float

    def as_dict(self) -> dict[str, float]:
        return {
            "acc": self.acc,
            "nmi": self.nmi,
            "ari": self.ari,
            "ami": self.ami,
            "f_measure": self.f_measure,
            "macro_f1": self.macro_f1,
            "pairwise_f1": self.pairwise_f1,
            "fmi": self.fmi,
            "purity": self.purity,
            "rand_index": self.rand_index,
        }


def _validate_labels(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(true_labels).reshape(-1)
    pred = np.asarray(pred_labels).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Label shape mismatch: {true.shape} != {pred.shape}")
    if true.size == 0:
        raise ValueError("Labels must not be empty")
    if np.issubdtype(true.dtype, np.number) and not np.all(np.isfinite(true)):
        raise ValueError("True labels contain NaN or infinite values")
    if np.issubdtype(pred.dtype, np.number) and not np.all(np.isfinite(pred)):
        raise ValueError("Predicted labels contain NaN or infinite values")
    return true, pred


def clustering_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """通过匈牙利算法完成最优标签匹配后计算 ACC。"""

    true, pred = _validate_labels(true_labels, pred_labels)
    counts = contingency_matrix(true, pred, sparse=False)
    rows, columns = linear_sum_assignment(-counts)
    return float(counts[rows, columns].sum() / true.size)


def clustering_f_measure(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> float:
    """计算 PLGB-FSC 官方代码使用的类别加权 F-measure。

    对每个真实类别选择 F 值最高的预测簇，再按照真实类别样本占比加权。
    该指标不同于 Macro-F1、Pairwise-F1 和 FMI。
    """

    true, pred = _validate_labels(true_labels, pred_labels)
    counts = contingency_matrix(true, pred, sparse=False).astype(float)
    true_sizes = counts.sum(axis=1)
    pred_sizes = counts.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = counts / pred_sizes[None, :]
        recall = counts / true_sizes[:, None]
        values = 2.0 * precision * recall / (precision + recall)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    weights = true_sizes / true.size
    return float(np.sum(weights * values.max(axis=1)))


def _labels_after_optimal_mapping(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把预测簇映射到真实类别的整数编码，未匹配簇保留为独立类别。"""

    true, pred = _validate_labels(true_labels, pred_labels)
    _, true_encoded = np.unique(true, return_inverse=True)
    _, pred_encoded = np.unique(pred, return_inverse=True)
    counts = contingency_matrix(true_encoded, pred_encoded, sparse=False)
    rows, columns = linear_sum_assignment(-counts)

    n_true = counts.shape[0]
    mapping = np.arange(counts.shape[1], dtype=int) + n_true
    mapping[columns] = rows
    return true_encoded, mapping[pred_encoded]


def mapped_macro_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """计算最优标签映射后的 Macro-F1。"""

    true, mapped = _labels_after_optimal_mapping(true_labels, pred_labels)
    return float(f1_score(true, mapped, average="macro", zero_division=0))


def pairwise_f1(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """计算样本对定义下的 Pairwise-F1。"""

    true, pred = _validate_labels(true_labels, pred_labels)
    pair_counts = pair_confusion_matrix(true, pred)
    false_positive = float(pair_counts[0, 1])
    false_negative = float(pair_counts[1, 0])
    true_positive = float(pair_counts[1, 1])
    denominator = 2.0 * true_positive + false_positive + false_negative
    if denominator == 0.0:
        return 1.0
    return float(2.0 * true_positive / denominator)


def clustering_purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """计算聚类 Purity。"""

    true, pred = _validate_labels(true_labels, pred_labels)
    counts = contingency_matrix(true, pred, sparse=False)
    return float(counts.max(axis=0).sum() / true.size)


def evaluate_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    *,
    nmi_average_method: str = "geometric",
) -> MetricResult:
    """计算统一的外部聚类评价指标。

    ``geometric`` 对应 PLGB-FSC 官方 ``nmi.m`` 的
    ``I/sqrt(H_true*H_pred)``；``arithmetic`` 对应部分粒球论文使用的
    ``2I/(H_true+H_pred)``。
    """

    true, pred = _validate_labels(true_labels, pred_labels)
    return MetricResult(
        acc=clustering_accuracy(true, pred),
        nmi=float(
            normalized_mutual_info_score(
                true,
                pred,
                average_method=nmi_average_method,
            )
        ),
        ari=float(adjusted_rand_score(true, pred)),
        ami=float(adjusted_mutual_info_score(true, pred)),
        f_measure=clustering_f_measure(true, pred),
        macro_f1=mapped_macro_f1(true, pred),
        pairwise_f1=pairwise_f1(true, pred),
        fmi=float(fowlkes_mallows_score(true, pred)),
        purity=clustering_purity(true, pred),
        rand_index=float(rand_score(true, pred)),
    )


def summarize_metric_rows(
    rows: Iterable[dict[str, float]],
    metric_names: Iterable[str],
) -> dict[str, float]:
    """按指标计算总体均值和总体标准差。"""

    materialized = list(rows)
    if not materialized:
        return {}

    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in materialized], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
    return summary
