from __future__ import annotations

"""The four released density-clustering applications of NARD."""

import numpy as np

from .core import NARDState


def dpeak_nard(state: NARDState) -> np.ndarray:
    """Run the active numerical path of ``DensityPeak_Auto_Adaptive``."""

    density = state.density
    distance_matrix = state.distance_matrix
    sample_count = density.size
    sorted_density_indices = np.argsort(density)[::-1]

    distance_to_master = np.zeros(sample_count, dtype=float)
    distance_to_master[sorted_density_indices[0]] = float("inf")
    masters = np.zeros(sample_count, dtype=int)
    # Preserve the released assignment, even when the density maximum is not
    # row zero. The maximum is always selected as a center before propagation.
    masters[sorted_density_indices[0]] = 0
    for position in range(1, sample_count):
        sample_index = int(sorted_density_indices[position])
        higher_density = sorted_density_indices[:position]
        nearest_position = int(
            np.argmin(distance_matrix[sample_index][higher_density])
        )
        masters[sample_index] = int(higher_density[nearest_position])
        distance_to_master[sample_index] = float(
            np.min(distance_matrix[sample_index][higher_density])
        )

    priority = np.multiply(density, distance_to_master)
    cluster_count = state.sample_distribution_count
    cluster_centers = np.argsort(priority)[-cluster_count:][::-1]
    labels = np.zeros(sample_count, dtype=int)
    for cluster_index, center_index in enumerate(cluster_centers):
        labels[int(center_index)] = -cluster_index - 1

    def resolve_label(sample_index: int) -> int:
        if labels[sample_index] < 0:
            return int(labels[sample_index])
        return resolve_label(int(masters[sample_index]))

    for sample_index in range(sample_count):
        if labels[sample_index] >= 0:
            labels[sample_index] = resolve_label(
                int(masters[sample_index])
            )
    return np.abs(labels)


def dbscan_nard(
    state: NARDState,
    core_factor: float,
) -> np.ndarray:
    """Run the released natural-neighbor DBSCAN-NARD path."""

    threshold = float(np.mean(state.density)) * core_factor
    core_indices = np.where(state.density >= threshold)[0]
    core_set = set(int(index) for index in core_indices)
    labels = np.full(state.density.size, -1, dtype=int)
    cluster_id = 0

    for point_id_value in core_indices:
        point_id = int(point_id_value)
        if labels[point_id] != -1:
            continue
        labels[point_id] = cluster_id
        seeds = set(state.expanded_neighbors[point_id])
        while seeds:
            new_point = int(seeds.pop())
            labels[new_point] = cluster_id
            if new_point in core_set:
                for neighbor in state.expanded_neighbors[new_point]:
                    if labels[neighbor] == -1:
                        seeds.add(int(neighbor))
        cluster_id += 1
    return labels


def dadc_nard(state: NARDState) -> np.ndarray:
    """Run the active numerical path of ``DADC_Auto_Adaptive``."""

    points = state.ball_centers
    sample_count = points.shape[0]
    distances = np.zeros((sample_count, sample_count), dtype=float)
    for first in range(sample_count - 1):
        for second in range(first + 1, sample_count):
            distance = float(np.linalg.norm(points[first] - points[second]))
            distances[first, second] = distance
            distances[second, first] = distance

    labels = np.full(sample_count, -1, dtype=int)
    for cluster_index, center_index in enumerate(state.density_centers):
        labels[center_index] = cluster_index

    distances[np.diag_indices(sample_count)] = 1_000_000.0

    def nearest_higher_density(sample_index: int) -> int:
        nearest_distance = 1_000_000.0
        nearest_index = -1
        for candidate in range(sample_count):
            if (
                distances[sample_index, candidate] < nearest_distance
                and state.density[sample_index] < state.density[candidate]
            ):
                nearest_distance = distances[sample_index, candidate]
                nearest_index = candidate
        if labels[nearest_index] == -1:
            labels[nearest_index] = nearest_higher_density(nearest_index)
        return int(labels[nearest_index])

    for sample_index in range(sample_count):
        if labels[sample_index] == -1:
            labels[sample_index] = nearest_higher_density(sample_index)
    return labels


def _hcdc_natural_neighbors(
    sorted_distance_indices: np.ndarray,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    sample_count = sorted_distance_indices.shape[0]
    level = 1
    finished = False
    reverse_counts = np.zeros(sample_count)
    stable_rounds = 0
    previous_without_reverse = 0
    nearest = [[] for _ in range(sample_count)]
    reverse = [[] for _ in range(sample_count)]
    natural: list[list[int]] = []

    while not finished:
        without_reverse = 0
        natural = []
        for sample_index in range(sample_count):
            neighbor = int(sorted_distance_indices[sample_index, level])
            nearest[sample_index].append(neighbor)
            reverse[neighbor].append(sample_index)
            reverse_counts[neighbor] += 1
        for sample_index in range(sample_count):
            natural.append(
                list(set(nearest[sample_index]) & set(reverse[sample_index]))
            )
            if reverse_counts[sample_index] == 0:
                without_reverse += 1

        level += 1
        if previous_without_reverse == without_reverse:
            stable_rounds += 1
        else:
            stable_rounds = 1
        if without_reverse == 0 or stable_rounds >= 2:
            finished = True
        previous_without_reverse = without_reverse
    return nearest, reverse, natural


def _hcdc_density(
    distances: np.ndarray,
    natural: list[list[int]],
    nearest: list[list[int]],
) -> np.ndarray:
    density = np.zeros(len(natural), dtype=float)
    for sample_index in range(len(natural)):
        for neighbor in natural[sample_index]:
            shared = list(
                set(nearest[sample_index]) & set(nearest[neighbor])
            )
            if shared:
                density[sample_index] += (
                    len(shared) ** 2
                    / (
                        (
                            np.sum(distances[sample_index, shared])
                            + np.sum(distances[neighbor, shared])
                        )
                        * (distances[sample_index, neighbor] + 0.01)
                    )
                )
    return density


def _hcdc_core_children(
    cores: list[int],
    representatives: np.ndarray,
    natural: list[list[int]],
) -> list[list[int]]:
    children_by_core: list[list[int]] = []
    for core in cores:
        children = list(np.where(representatives == core)[0])
        neighbors: set[int] = set()
        if len(children) != 1:
            for child in children:
                neighbors |= set(natural[child])
        else:
            neighbors = set(natural[children[0]]) | set(
                natural[children[0]]
            )
        children_by_core.append(
            list(set(children) | neighbors)
        )
    return children_by_core


def _hcdc_representatives_and_cores(
    nearest: list[list[int]],
    density: np.ndarray,
    natural: list[list[int]],
    distances: np.ndarray,
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    sample_count = density.size
    representatives = np.full(sample_count, -1, dtype=int)
    representative_candidates = [[] for _ in range(sample_count)]

    for sample_index_value in np.argsort(density):
        sample_index = int(sample_index_value)
        neighborhood = nearest[sample_index]
        maximum_position = int(np.argmax(density[neighborhood]))
        maximum_index = int(neighborhood[maximum_position])
        maximum_density = float(np.max(density[neighborhood]))
        if density[sample_index] > maximum_density:
            maximum_index = sample_index
        for neighbor in neighborhood:
            representative_candidates[neighbor].append(maximum_index)

    for sample_index in range(sample_count):
        candidates = list(set(representative_candidates[sample_index]))
        selected = -1
        selected_count = 0
        for candidate in candidates:
            count = representative_candidates[sample_index].count(candidate)
            if count > selected_count:
                selected = candidate
                selected_count = count
            elif count == selected_count:
                candidate_score = (
                    distances[sample_index, candidate]
                    * abs(density[sample_index] - density[candidate])
                )
                selected_score = (
                    distances[sample_index, selected]
                    * abs(density[sample_index] - density[selected])
                )
                if candidate_score < selected_score:
                    selected = candidate
                    selected_count = count
        representatives[sample_index] = selected

    visited = np.zeros(sample_count)
    round_index = 0
    for sample_index in range(sample_count):
        if visited[sample_index] != 0:
            continue
        parent = sample_index
        round_index += 1
        while representatives[parent] != parent:
            visited[parent] = round_index
            parent = int(representatives[parent])
        representatives[np.where(visited == round_index)[0]] = parent

    cores = [
        sample_index
        for sample_index in range(sample_count)
        if representatives[sample_index] == sample_index
    ]
    deleted_cores: list[int] = []
    children_by_core = _hcdc_core_children(
        cores,
        representatives,
        natural,
    )
    original_core_positions = {
        core: position for position, core in enumerate(cores)
    }
    sorted_cores = [
        cores[int(position)]
        for position in np.argsort(density[cores])
    ]

    for first_core in sorted_cores:
        if first_core in deleted_cores:
            continue
        for second_core in sorted_cores:
            if first_core in deleted_cores:
                break
            if second_core in deleted_cores:
                continue
            first_position = original_core_positions[first_core]
            second_position = original_core_positions[second_core]
            if (
                second_core in children_by_core[first_position]
                or first_core in children_by_core[second_position]
            ):
                if density[first_core] < density[second_core]:
                    representatives[
                        np.where(representatives == first_core)[0]
                    ] = second_core
                    cores.remove(first_core)
                    deleted_cores.append(first_core)
                    children_by_core[first_position] = list(
                        set(children_by_core[first_position])
                        | set(children_by_core[second_position])
                    )
                    break
                if density[first_core] > density[second_core]:
                    representatives[
                        np.where(representatives == second_core)[0]
                    ] = first_core
                    cores.remove(second_core)
                    children_by_core[second_position] = list(
                        set(children_by_core[first_position])
                        | set(children_by_core[second_position])
                    )
                    deleted_cores.append(second_core)
                    continue

    contained_points = [
        np.where(representatives == core)[0]
        for core in cores
    ]
    return representatives, cores, contained_points


def _hcdc_similarity(
    cores: list[int],
    density: np.ndarray,
    children_by_core: list[list[int]],
    distances: np.ndarray,
    representatives: np.ndarray,
) -> np.ndarray:
    core_count = len(cores)
    similarity = np.zeros((core_count, core_count), dtype=float)
    maximum_core_distance = 0.0
    for first in range(core_count):
        for second in range(first + 1, core_count):
            maximum_core_distance = max(
                maximum_core_distance,
                float(distances[cores[first], cores[second]]),
            )

    maximum_penalty = 1.0
    for first in range(core_count):
        for second in range(first + 1, core_count):
            shared = list(
                set(children_by_core[first])
                & set(children_by_core[second])
            )
            first_members = np.where(
                representatives == cores[first]
            )[0]
            second_members = np.where(
                representatives == cores[second]
            )[0]
            first_mean = float(np.mean(density[first_members]))
            second_mean = float(np.mean(density[second_members]))
            if shared:
                value = (
                    distances[cores[first], cores[second]]
                    * max(
                        len(children_by_core[first]),
                        len(children_by_core[second]),
                    )
                    * abs(first_mean + second_mean)
                    / (
                        len(shared) ** 2
                        * (2 * np.sqrt(first_mean * second_mean))
                    )
                )
                maximum_penalty = max(
                    maximum_penalty,
                    len(children_by_core[first])
                    * len(children_by_core[second])
                    / (len(shared) ** 2),
                )
                similarity[first, second] = value
                similarity[second, first] = value

    for first in range(core_count):
        for second in range(first + 1, core_count):
            shared = (
                set(children_by_core[first])
                & set(children_by_core[second])
            )
            if not shared:
                value = (
                    maximum_penalty
                    * distances[cores[first], cores[second]]
                    * maximum_core_distance
                )
                similarity[first, second] = value
                similarity[second, first] = value
    return similarity


def _minimum_cluster_distance(
    first_cluster: list[int],
    second_cluster: list[int],
    distances: np.ndarray,
) -> float:
    minimum = float("inf")
    for first in first_cluster:
        for second in second_cluster:
            if distances[first, second] < minimum:
                minimum = float(distances[first, second])
    return minimum


def _hcdc_agglomerative_clustering(
    distances: np.ndarray,
    cluster_count: int,
    contained_points: list[np.ndarray],
    sample_count: int,
    small_cluster_fraction: float,
) -> list[list[int]]:
    clusters = [[index] for index in range(len(contained_points))]
    represented_counts = [
        int(len(points)) for points in contained_points
    ]

    while len(clusters) > cluster_count:
        minimum = _minimum_cluster_distance(
            clusters[0],
            clusters[1],
            distances,
        )
        merged_indices = clusters[0] + clusters[1]
        remove_high = 1
        remove_low = 0

        for first_index, first_cluster in enumerate(clusters):
            for second_index in range(len(clusters)):
                distance = _minimum_cluster_distance(
                    first_cluster,
                    clusters[second_index],
                    distances,
                )
                if distance < minimum and first_index != second_index:
                    minimum = distance
                    merged_indices = (
                        clusters[first_index] + clusters[second_index]
                    )
                    if first_index < second_index:
                        remove_high = second_index
                        remove_low = first_index
                    else:
                        remove_high = first_index
                        remove_low = second_index

        represented_count = (
            represented_counts[remove_high]
            + represented_counts[remove_low]
        )
        clusters.append(merged_indices)
        represented_counts.append(represented_count)
        del clusters[remove_high]
        del clusters[remove_low]
        del represented_counts[remove_high]
        del represented_counts[remove_low]

    index = 0
    while index < len(clusters):
        if represented_counts[index] < (
            small_cluster_fraction * sample_count
        ):
            del clusters[index]
            del represented_counts[index]
        else:
            index += 1
    return clusters


def hcdc_nard(
    state: NARDState,
    small_cluster_fraction: float,
) -> np.ndarray:
    """Run the released ``HCDC_Auto_Adaptive`` numerical path."""

    points = state.ball_centers
    sample_count = points.shape[0]
    distances = np.zeros((sample_count, sample_count), dtype=float)
    for first in range(sample_count):
        for second in range(first + 1, sample_count):
            distance = float(np.linalg.norm(points[first] - points[second]))
            distances[first, second] = distance
            distances[second, first] = distance

    sorted_indices = np.argsort(distances, axis=1)
    nearest, _reverse, natural = _hcdc_natural_neighbors(sorted_indices)
    for sample_index in range(sample_count):
        nearest[sample_index].append(sample_index)
    density = _hcdc_density(distances, natural, nearest)
    representatives, cores, contained_points = (
        _hcdc_representatives_and_cores(
            nearest,
            density,
            natural,
            distances,
        )
    )
    children_by_core = _hcdc_core_children(
        cores,
        representatives,
        natural,
    )
    core_distances = _hcdc_similarity(
        cores,
        density,
        children_by_core,
        distances,
        representatives,
    )
    clusters = _hcdc_agglomerative_clustering(
        core_distances,
        state.sample_distribution_count,
        contained_points,
        sample_count,
        small_cluster_fraction,
    )

    labels = np.zeros(sample_count, dtype=float)
    for cluster_index, cluster in enumerate(clusters):
        for core_position in cluster:
            labels[cores[core_position]] = cluster_index + 1
    for sample_index in range(sample_count):
        if labels[sample_index] == 0:
            labels[sample_index] = labels[representatives[sample_index]]
    return labels.astype(int)
