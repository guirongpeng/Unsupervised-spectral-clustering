from __future__ import annotations

"""Faithful translation of MATLAB ``GenerationMethodGBs.GBPOJG``."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True, slots=True)
class GranularBall:
    sample_indices: np.ndarray
    center: np.ndarray
    average_radius: float
    max_radius: float
    size: int
    num_in_ball: int


@dataclass(slots=True)
class _TreeNode:
    node_id: int
    parent_id: int
    ball: GranularBall
    best_combination: list[int] = field(default_factory=list)
    best_value: float = -1.0

    def __post_init__(self) -> None:
        self.best_combination = [self.node_id]


@dataclass(frozen=True, slots=True)
class GBPOJGGenerationResult:
    granular_balls: tuple[GranularBall, ...]
    tree_node_count: int
    split_threshold: float
    pruned_ball_count: int


def _make_ball(X: np.ndarray, indices: np.ndarray) -> GranularBall:
    points = X[indices]
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    average_radius = float(np.mean(distances))
    return GranularBall(
        sample_indices=np.asarray(indices, dtype=int),
        center=np.asarray(center, dtype=float),
        average_radius=average_radius,
        max_radius=float(np.max(distances)),
        size=int(indices.size),
        num_in_ball=int(np.count_nonzero(distances <= average_radius)),
    )


def _matlab_farthest_endpoints(distances: np.ndarray) -> tuple[int, int]:
    """Match MATLAB ``find(max(max(Distance)) == Distance)`` column order."""

    positions = np.flatnonzero(
        np.equal(distances, np.max(distances)).ravel(order="F")
    )
    rows = positions % distances.shape[0]
    if rows.size < 2 or rows[0] == rows[1]:
        raise RuntimeError("GB-POJG could not identify two farthest endpoints")
    return int(rows[0]), int(rows[1])


def _split_ball(
    X: np.ndarray, ball: GranularBall, full_distances: np.ndarray
) -> tuple[GranularBall, GranularBall]:
    indices = ball.sample_indices
    local = full_distances[np.ix_(indices, indices)]
    first, second = _matlab_farthest_endpoints(local)
    # MATLAB uses <=, so ties go to the first endpoint.
    first_mask = local[first, :] <= local[second, :]
    first_mask[first] = True
    first_mask[second] = False
    left, right = indices[first_mask], indices[~first_mask]
    if not left.size or not right.size:
        raise RuntimeError("GB-POJG generated an empty child granular ball")
    return _make_ball(X, left), _make_ball(X, right)


def _quality(ball: GranularBall, gamma: float) -> float:
    return float(ball.num_in_ball * np.exp(-gamma * ball.average_radius))


def _build_binary_tree(
    X: np.ndarray, distances: np.ndarray, gamma: float, delta: float
) -> tuple[list[_TreeNode | None], list[bool], float]:
    n_samples = X.shape[0]
    threshold = float(max(delta * np.sqrt(n_samples), n_samples**0.25))
    root = _TreeNode(1, -1, _make_ball(X, np.arange(n_samples, dtype=int)))
    nodes: list[_TreeNode | None] = [None, root]
    is_leaf = [False, False]
    queue: deque[_TreeNode] = deque([root])
    next_id = 2

    while queue:
        node = queue.popleft()
        if node.ball.size > threshold:
            left, right = _split_ball(X, node.ball, distances)
            left_node = _TreeNode(next_id, node.node_id, left, best_value=_quality(left, gamma))
            right_node = _TreeNode(next_id + 1, node.node_id, right, best_value=_quality(right, gamma))
            nodes.extend((left_node, right_node))
            is_leaf.extend((False, False))
            queue.extend((left_node, right_node))
            next_id += 2
        else:
            is_leaf[node.node_id] = True
    return nodes, is_leaf, threshold


def _prune_binary_tree(nodes: list[_TreeNode | None], is_leaf: list[bool]) -> list[int]:
    queue: deque[int] = deque(i for i, leaf in enumerate(is_leaf) if leaf)
    while len(queue) > 1:
        node_id = queue.popleft()
        if not is_leaf[node_id]:
            continue
        sibling_id = node_id + 1 if node_id % 2 == 0 else node_id - 1
        if not is_leaf[sibling_id]:
            queue.append(node_id)
            continue
        node, sibling = nodes[node_id], nodes[sibling_id]
        if node is None or sibling is None:
            raise RuntimeError("GB-POJG tree contains a missing node")
        parent = nodes[node.parent_id]
        if parent is None:
            raise RuntimeError("GB-POJG tree contains a missing parent")
        is_leaf[node_id] = is_leaf[sibling_id] = False
        is_leaf[parent.node_id] = True
        queue.append(parent.node_id)
        children_value = node.best_value + sibling.best_value
        if children_value > parent.best_value:
            parent.best_value = children_value
            parent.best_combination = node.best_combination + sibling.best_combination
    root = nodes[1]
    if root is None:
        raise RuntimeError("GB-POJG tree is missing its root")
    return root.best_combination


def _source_anomaly_split(
    X: np.ndarray, distances: np.ndarray, selected: tuple[GranularBall, ...]
) -> tuple[GranularBall, ...]:
    """Preserve source behavior: ``MeanNumAll`` is a sum, not a mean."""

    mean_radius = float(np.mean([ball.average_radius for ball in selected]))
    summed_size = int(sum(ball.size for ball in selected))
    queue: deque[GranularBall] = deque(selected)
    result: list[GranularBall] = []
    while queue:
        ball = queue.popleft()
        if (
            ball.average_radius > 2.0 * mean_radius
            and ball.size >= 2
            and ball.size <= 0.5 * summed_size
        ):
            queue.extend(_split_ball(X, ball, distances))
        else:
            result.append(ball)
    return tuple(result)


def generate_granular_balls(
    X: np.ndarray, gamma: float, delta: float
) -> GBPOJGGenerationResult:
    """Run official GB-POJG using one source-style full distance matrix."""

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError(f"X must be a non-empty 2-D matrix, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("X contains NaN or infinite values")
    if gamma < 0 or not 0 < delta <= 1:
        raise ValueError("gamma must be >= 0 and delta must be in (0, 1]")

    distances = cdist(values, values, metric="euclidean")
    nodes, is_leaf, threshold = _build_binary_tree(values, distances, gamma, delta)
    best_combination = _prune_binary_tree(nodes, is_leaf)
    selected = tuple(nodes[index].ball for index in best_combination if nodes[index] is not None)
    balls = _source_anomaly_split(values, distances, selected)
    assigned = np.concatenate([ball.sample_indices for ball in balls])
    if assigned.size != values.shape[0] or not np.array_equal(
        np.sort(assigned), np.arange(values.shape[0])
    ):
        raise RuntimeError("GB-POJG granular balls do not partition all samples")
    return GBPOJGGenerationResult(
        granular_balls=balls,
        tree_node_count=len(nodes) - 1,
        split_threshold=threshold,
        pruned_ball_count=len(selected),
    )
