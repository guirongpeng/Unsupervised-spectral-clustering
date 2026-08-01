from __future__ import annotations

"""PLGB-FSC 粒球生成与锚点生成。

粒球的核心作用是把大量样本压缩为少量锚点：每个最终粒球的均值作为一个
锚点，后续 Transfer Cut 只需要处理样本-锚点二分图。
"""

from dataclasses import dataclass

import numpy as np

try:
    from .feature_selection import select_local_features_by_discernibility
    from .weighted_kmeans import two_means_labels
except ImportError:
    from feature_selection import select_local_features_by_discernibility
    from weighted_kmeans import two_means_labels


@dataclass(frozen=True)
class GranularBall:
    """一个粒球，包含该粒球内样本及其伪标签。"""

    X: np.ndarray
    pseudo_labels: np.ndarray

    @property
    def size(self) -> int:
        """粒球内样本数量。"""

        return int(self.X.shape[0])


def pseudo_purity(labels: np.ndarray) -> float:
    """计算伪纯度：粒球内占比最高的伪标签比例。"""

    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / labels.size)


def split_ball_with_2means(
    ball: GranularBall,
    p2: int,
    max_iter: int = 3,
    seed: int | None = None,
) -> tuple[GranularBall, GranularBall]:
    """用局部特征选择 + 2-Means 将一个粒球二分。"""

    if ball.size < 2:
        # 单样本粒球无法再拆，返回自身和一个空粒球占位。
        return ball, GranularBall(ball.X[:0].copy(), ball.pseudo_labels[:0].copy())

    # 先在粒球内部选择 p2 个最适合拆分的局部特征，再做 2-Means。
    split_X, _, _ = select_local_features_by_discernibility(ball.X, p2)
    labels = two_means_labels(split_X, max_iter=max_iter, seed=seed)
    first = labels == 0
    second = labels == 1
    if not np.any(first) or not np.any(second):
        # 极端情况下如果 2-Means 仍产生空簇，用简单二分兜底，避免后续矩阵为空。
        midpoint = ball.size // 2
        first = np.zeros(ball.size, dtype=bool)
        first[:midpoint] = True
        second = ~first
    return (
        GranularBall(ball.X[first], ball.pseudo_labels[first]),
        GranularBall(ball.X[second], ball.pseudo_labels[second]),
    )


def should_keep_ball(
    ball: GranularBall,
    purity_threshold: float,
    keep_matlab_split_rule: bool = True,
) -> bool:
    """判断粒球是否停止拆分。

    MATLAB 源码中的条件是 `p >= purity && m < 8`，即高纯度且小于 8 个样本
    才保留；这个规则比较特殊，但源码兼容模式下必须保留。
    """

    purity = pseudo_purity(ball.pseudo_labels)
    if ball.size < 2:
        return True
    if keep_matlab_split_rule:
        return bool(purity >= purity_threshold and ball.size < 8)
    return bool(purity >= purity_threshold)


def split_granular_balls(
    balls: list[GranularBall],
    purity_threshold: float,
    p2: int,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
) -> list[GranularBall]:
    """对当前粒球列表做一轮扫描拆分。"""

    new_balls: list[GranularBall] = []
    for index, ball in enumerate(balls):
        if should_keep_ball(ball, purity_threshold, keep_matlab_split_rule):
            new_balls.append(ball)
        else:
            # 给不同粒球拆分使用不同 seed，减少完全相同初始化。
            split_seed = None if seed is None else seed + index
            ball_1, ball_2 = split_ball_with_2means(ball, p2, split_kmeans_max_iter, split_seed)
            if ball_2.size == 0:
                new_balls.append(ball_1)
            else:
                new_balls.extend([ball_1, ball_2])
    return new_balls


def generate_granular_balls(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    p2: int,
    purity_threshold: float,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
    max_rounds: int = 10_000,
) -> list[GranularBall]:
    """从单个大粒球开始，迭代拆分直到一轮后粒球数量不再变化。"""

    balls = [GranularBall(np.asarray(X, dtype=float), np.asarray(pseudo_labels).reshape(-1))]
    for _ in range(max_rounds):
        old_count = len(balls)
        balls = split_granular_balls(
            balls,
            purity_threshold,
            p2,
            split_kmeans_max_iter=split_kmeans_max_iter,
            seed=seed,
            keep_matlab_split_rule=keep_matlab_split_rule,
        )
        if len(balls) == old_count:
            break
    else:
        raise RuntimeError(f"Granular-ball splitting did not converge within {max_rounds} rounds")
    return balls


def anchors_from_balls(balls: list[GranularBall]) -> np.ndarray:
    """把每个粒球转换为一个锚点：多样本取均值，单样本取自身。"""

    anchors = []
    for ball in balls:
        if ball.size == 1:
            anchors.append(ball.X[0])
        else:
            anchors.append(ball.X.mean(axis=0))
    return np.vstack(anchors)


def generate_anchors(
    X: np.ndarray,
    pseudo_labels: np.ndarray,
    p2: int,
    purity_threshold: float,
    split_kmeans_max_iter: int = 3,
    seed: int | None = None,
    keep_matlab_split_rule: bool = True,
) -> tuple[np.ndarray, list[GranularBall]]:
    """生成最终锚点矩阵和粒球列表。"""

    balls = generate_granular_balls(
        X,
        pseudo_labels,
        p2,
        purity_threshold,
        split_kmeans_max_iter=split_kmeans_max_iter,
        seed=seed,
        keep_matlab_split_rule=keep_matlab_split_rule,
    )
    return anchors_from_balls(balls), balls
