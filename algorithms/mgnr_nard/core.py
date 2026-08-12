from __future__ import annotations

"""Source-faithful granular-ball, MGN, SDGS and NARD calculations."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True, slots=True)
class GranularBall:
    """One official granular ball represented by original row indices."""

    sample_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class NARDState:
    """Artifacts shared by all four NARD clustering applications."""

    granular_balls: tuple[GranularBall, ...]
    ball_centers: np.ndarray
    distance_matrix: np.ndarray
    nearest_neighbors: tuple[tuple[int, ...], ...]
    reverse_neighbors: tuple[tuple[int, ...], ...]
    natural_neighbors: tuple[frozenset[int], ...]
    multi_granularity_neighbors: tuple[frozenset[int], ...]
    sample_distribution_groups: tuple[np.ndarray, ...]
    expanded_neighbors: tuple[tuple[int, ...], ...]
    density: np.ndarray
    density_centers: tuple[int, ...]

    @property
    def sample_distribution_count(self) -> int:
        return len(self.sample_distribution_groups)


def _distribution_measure(
    X: np.ndarray,
    ball: GranularBall,
) -> float:
    """Return the released ``get_dm`` value, including its small-ball rule."""

    count = int(ball.sample_indices.size)
    if count == 0:
        return 0.0

    points = X[ball.sample_indices]
    center = np.mean(points, axis=0)
    distances = np.sqrt(np.sum((center - points) ** 2, axis=1))
    distance_sum = 0.0
    for distance in distances:
        distance_sum += float(distance)
    mean_radius = distance_sum / count
    return mean_radius if count > 2 else 1.0


def _radius(X: np.ndarray, ball: GranularBall) -> float:
    points = X[ball.sample_indices]
    center = np.mean(points, axis=0)
    return float(np.max(np.sqrt(np.sum((center - points) ** 2, axis=1))))


def _split_ball(
    X: np.ndarray,
    ball: GranularBall,
) -> tuple[GranularBall, GranularBall]:
    """Reproduce the released farthest-from-center/farthest-point split."""

    points = X[ball.sample_indices]
    center = np.mean(points, axis=0)

    first_distance = 0.0
    first_index = 0
    for index in range(points.shape[0]):
        distance = float(sum((points[index] - center) ** 2))
        if first_distance < distance:
            first_distance = distance
            first_index = index

    second_distance = 0.0
    second_index = 0
    for index in range(points.shape[0]):
        distance = float(sum((points[index] - points[first_index]) ** 2))
        if second_distance < distance:
            second_distance = distance
            second_index = index

    first_members: list[int] = []
    second_members: list[int] = []
    for local_index, original_index in enumerate(ball.sample_indices):
        distance_to_first = float(
            sum((points[local_index] - points[first_index]) ** 2)
        )
        distance_to_second = float(
            sum((points[local_index] - points[second_index]) ** 2)
        )
        if distance_to_first < distance_to_second:
            first_members.append(int(original_index))
        else:
            second_members.append(int(original_index))

    return (
        GranularBall(np.asarray(first_members, dtype=int)),
        GranularBall(np.asarray(second_members, dtype=int)),
    )


def _quality_division(
    X: np.ndarray,
    balls: list[GranularBall],
) -> list[GranularBall]:
    finalized: list[GranularBall] = []
    active = balls

    while True:
        old_count = len(active) + len(finalized)
        new_active: list[GranularBall] = []
        for ball in active:
            if ball.sample_indices.size > 1:
                first, second = _split_ball(X, ball)
                count = first.sample_indices.size + second.sample_indices.size
                child_measure = (
                    first.sample_indices.size / count
                    * _distribution_measure(X, first)
                    + second.sample_indices.size / count
                    * _distribution_measure(X, second)
                )
                if child_measure < _distribution_measure(X, ball):
                    new_active.extend((first, second))
                else:
                    finalized.append(ball)
            else:
                finalized.append(ball)

        new_count = len(new_active) + len(finalized)
        if new_count == old_count:
            return finalized
        active = new_active


def _radius_division(
    X: np.ndarray,
    balls: list[GranularBall],
    radius_detection_factor: float,
) -> list[GranularBall]:
    radii = [
        _radius(X, ball)
        for ball in balls
        if ball.sample_indices.size >= 2
    ]
    if not radii:
        return balls

    radius_detection = max(
        float(np.median(radii)),
        float(np.mean(radii)),
    )
    finalized: list[GranularBall] = []
    active = balls

    while True:
        old_count = len(active) + len(finalized)
        new_active: list[GranularBall] = []
        for ball in active:
            if ball.sample_indices.size < 2:
                finalized.append(ball)
            elif _radius(X, ball) <= (
                radius_detection_factor * radius_detection
            ):
                finalized.append(ball)
            else:
                new_active.extend(_split_ball(X, ball))

        new_count = len(new_active) + len(finalized)
        if new_count == old_count:
            return finalized
        active = new_active


def generate_granular_balls(
    X: np.ndarray,
    radius_detection_factor: float,
) -> tuple[GranularBall, ...]:
    balls = [
        GranularBall(np.arange(X.shape[0], dtype=int)),
    ]
    balls = _quality_division(X, balls)
    balls = _radius_division(X, balls, radius_detection_factor)

    assigned = np.concatenate([ball.sample_indices for ball in balls])
    if assigned.size != X.shape[0] or not np.array_equal(
        np.sort(assigned),
        np.arange(X.shape[0]),
    ):
        raise RuntimeError("NARD granular balls do not partition all samples")
    if any(ball.sample_indices.size == 0 for ball in balls):
        raise RuntimeError("NARD generated an empty granular ball")
    return tuple(balls)


def _natural_neighbor_search(
    distance_matrix: np.ndarray,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[frozenset[int], ...],
]:
    count = distance_matrix.shape[0]
    sorted_indices = np.argsort(distance_matrix, axis=1)
    nearest: list[list[int]] = [[] for _ in range(count)]
    reverse: list[list[int]] = [[] for _ in range(count)]
    reverse_counts = [0] * count
    level = 0
    previous_without_reverse = 0

    while level + 1 < count:
        for index in range(count):
            neighbor = int(sorted_indices[index, level + 1])
            nearest[index].append(neighbor)
            reverse[neighbor].append(index)
            reverse_counts[neighbor] += 1

        without_reverse = reverse_counts.count(0)
        if without_reverse != previous_without_reverse:
            previous_without_reverse = without_reverse
        else:
            break
        level += 1

    natural = tuple(
        frozenset(set(nearest[index]) & set(reverse[index]))
        for index in range(count)
    )
    return (
        tuple(tuple(values) for values in nearest),
        tuple(tuple(values) for values in reverse),
        natural,
    )


def _merge_neighbor_groups(
    groups: list[set[int]],
) -> list[set[int]]:
    merged: list[set[int]] = []
    iterated = [False] * len(groups)
    for first_index in range(len(groups)):
        if iterated[first_index]:
            continue
        current = set(groups[first_index])
        for second_index in range(first_index, len(groups)):
            if (
                len(current & groups[second_index]) > 1
                and not iterated[second_index]
            ):
                iterated[second_index] = True
                current |= groups[second_index]
        merged.append(current)
    return merged


def _sample_distribution_groups(
    natural_neighbors: tuple[frozenset[int], ...],
) -> tuple[
    tuple[frozenset[int], ...],
    tuple[np.ndarray, ...],
]:
    multi_granularity = tuple(
        frozenset(set(neighbors) | {index})
        for index, neighbors in enumerate(natural_neighbors)
    )
    current = [set(group) for group in multi_granularity]
    while True:
        merged = _merge_neighbor_groups(current)
        if len(merged) == len(current):
            break
        current = merged
    groups = tuple(
        np.asarray(list(group), dtype=int)
        for group in merged
    )
    return multi_granularity, groups


def _expanded_neighbors(
    natural_neighbors: tuple[frozenset[int], ...],
) -> tuple[tuple[int, ...], ...]:
    expanded: list[tuple[int, ...]] = []
    for index in range(len(natural_neighbors)):
        first_layer = set(natural_neighbors[index])
        for neighbor in natural_neighbors[index]:
            first_layer |= set(natural_neighbors[neighbor])

        second_layer = set(first_layer)
        for neighbor in first_layer:
            second_layer |= set(natural_neighbors[neighbor])
        second_layer.discard(index)
        expanded.append(tuple(second_layer))
    return tuple(expanded)


def _nard_density(
    ball_centers: np.ndarray,
    groups: tuple[np.ndarray, ...],
    expanded: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, tuple[int, ...]]:
    density = np.zeros(ball_centers.shape[0], dtype=float)
    centers: list[int] = []

    for group_indices in groups:
        adaptive_neighborhoods: list[np.ndarray] = []
        for original_index in group_indices:
            mapped = np.where(
                np.isin(group_indices, expanded[int(original_index)])
            )[0]
            adaptive_neighborhoods.append(mapped)

        group_data = ball_centers[group_indices]
        group_distances = squareform(
            pdist(group_data, metric="euclidean")
        )
        neighbor_distances: list[np.ndarray] = []
        multi_granularity_density: list[float] = []
        for local_index, neighbors in enumerate(adaptive_neighborhoods):
            distances = group_distances[local_index][neighbors]
            neighbor_distances.append(distances)
            if neighbors.size:
                multi_granularity_density.append(
                    1.0 / float(np.average(distances))
                )
            else:
                multi_granularity_density.append(0.0000001)

        natural_domain_density: list[float] = []
        for local_index, neighbors in enumerate(adaptive_neighborhoods):
            value = multi_granularity_density[local_index]
            for neighbor_position, neighbor_index in enumerate(neighbors):
                value += (
                    multi_granularity_density[int(neighbor_index)]
                    * (1.0 / neighbor_distances[local_index][neighbor_position])
                )
            natural_domain_density.append(value)

        local_peak = float(np.max(natural_domain_density))
        for local_index, original_index in enumerate(group_indices):
            relative_density = (
                natural_domain_density[local_index] / local_peak
            )
            if relative_density == 1:
                centers.append(int(original_index))
            density[int(original_index)] = relative_density

    return density, tuple(centers)


def build_nard_state(
    X: np.ndarray,
    radius_detection_factor: float,
) -> NARDState:
    balls = generate_granular_balls(X, radius_detection_factor)
    ball_centers = np.asarray(
        [np.mean(X[ball.sample_indices], axis=0) for ball in balls],
        dtype=float,
    )
    distance_matrix = squareform(pdist(ball_centers, metric="euclidean"))
    nearest, reverse, natural = _natural_neighbor_search(distance_matrix)
    multi_granularity, groups = _sample_distribution_groups(natural)
    expanded = _expanded_neighbors(natural)
    density, density_centers = _nard_density(
        ball_centers,
        groups,
        expanded,
    )
    return NARDState(
        granular_balls=balls,
        ball_centers=ball_centers,
        distance_matrix=distance_matrix,
        nearest_neighbors=nearest,
        reverse_neighbors=reverse,
        natural_neighbors=natural,
        multi_granularity_neighbors=multi_granularity,
        sample_distribution_groups=groups,
        expanded_neighbors=expanded,
        density=density,
        density_centers=density_centers,
    )
