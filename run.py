from __future__ import annotations

"""Single entry point for the clean clustering benchmark."""

import csv
import json
import math
import random
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.my_v0 import MYV0, MYV0Config
from algorithms.my_v1 import MYV1, MYV1Config
from algorithms.my_v2 import MYV2, MYV2Config
from algorithms.plgb_fsc import PLGBFSC, PLGBFSCConfig
from config import (
    DATASETS,
    EXPERIMENT,
    MY_V0_PARAMS,
    MY_V1_PARAMS,
    MY_V2_PARAMS,
    PLGB_FSC_PARAMS,
    ExperimentConfig,
)
from core.data import Dataset, load_dataset, minmax_scale
from core.metrics import evaluate_clustering


METRICS = (
    "acc",
    "nmi",
    "ari",
    "ami",
    "f_measure",
    "macro_f1",
    "pairwise_f1",
    "fmi",
    "purity",
    "rand_index",
)
RUN_FIELDS = (
    "algorithm",
    "dataset",
    "seed",
    "p1",
    "p2",
    "pdmf_neighbors",
    "graph_neighbors",
    "pdmf_similarity_lambda",
    "theta",
    "status",
    *METRICS,
    "runtime_seconds",
    "prediction_path",
    "error_type",
    "error_message",
)
V2_RUN_FIELDS = (
    *RUN_FIELDS,
    "stability_delta",
    "selected_p1",
    "local_feature_count_mean",
    "local_feature_count_min",
    "local_feature_count_max",
    "global_entropy_loss",
    "global_graph_loss",
)


def _get_algorithm_config(algorithm: str) -> dict[str, object]:
    if algorithm == "plgb_fsc":
        return PLGB_FSC_PARAMS
    if algorithm == "my_v0":
        return MY_V0_PARAMS
    if algorithm == "my_v1":
        return MY_V1_PARAMS
    if algorithm == "my_v2":
        return MY_V2_PARAMS
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _validate_algorithm_config(algorithm: str) -> None:
    params = _get_algorithm_config(algorithm)
    theta_values = tuple(params["theta_values"])
    if algorithm != "my_v2":
        _validate_count_ratio_options(algorithm, params, "p1", minimum_count=2)
        if algorithm in {"my_v0", "my_v1"}:
            _validate_count_ratio_options(
                algorithm,
                params,
                "p2",
                minimum_count=1,
                allow_ratio_one=False,
            )
        else:
            p2_values = tuple(params["p2_values"])
            if not p2_values or any(p2 <= 0 for p2 in p2_values):
                raise ValueError(f"{algorithm}: p2_values must contain positive integers")
    if not theta_values or any(not 0.0 < theta <= 1.0 for theta in theta_values):
        raise ValueError(f"{algorithm}: theta_values must be in (0, 1]")
    if algorithm in {"my_v0", "my_v1", "my_v2"}:
        _validate_count_ratio_options(
            algorithm, params, "pdmf_neighbors", minimum_count=1
        )
    if algorithm in {"my_v0", "my_v1", "my_v2"}:
        epsilon = float(params["pdmf_epsilon"])
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"{algorithm}: pdmf_epsilon must be positive")
    if algorithm in {"my_v1", "my_v2"}:
        _validate_count_ratio_options(
            algorithm, params, "graph_neighbors", minimum_count=1
        )
        similarity_lambdas = tuple(
            float(value) for value in params["pdmf_similarity_lambda_ratios"]
        )
        if not similarity_lambdas or any(
            not math.isfinite(value) or not 0 < value < 1
            for value in similarity_lambdas
        ):
            raise ValueError(
                f"{algorithm}: pdmf_similarity_lambda_ratios must be in (0, 1)"
            )
    if algorithm == "my_v2":
        deltas = tuple(float(value) for value in params["stability_delta_values"])
        if not deltas or any(
            not math.isfinite(value) or value < 0 for value in deltas
        ):
            raise ValueError("my_v2: stability_delta_values must be in [0, +inf)")


def _validate_count_ratio_options(
    algorithm: str,
    params: dict[str, object],
    name: str,
    *,
    minimum_count: int,
    allow_ratio_one: bool = True,
) -> None:
    counts = tuple(params[f"{name}_counts"])
    ratios = tuple(float(value) for value in params[f"{name}_ratios"])
    if not counts and not ratios:
        raise ValueError(f"{algorithm}: {name}_counts and {name}_ratios cannot both be empty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum_count
        for value in counts
    ):
        raise ValueError(
            f"{algorithm}: {name}_counts must contain integers >= {minimum_count}"
        )
    if any(
        not math.isfinite(value)
        or value <= 0.0
        or value > 1.0
        or (value == 1.0 and not allow_ratio_one)
        for value in ratios
    ):
        upper = "(0, 1]" if allow_ratio_one else "(0, 1)"
        raise ValueError(f"{algorithm}: {name}_ratios must be in {upper}")


def _resolve_p1_values(algorithm: str, n_features: int) -> tuple[int, ...]:
    params = _get_algorithm_config(algorithm)
    candidates = [int(value) for value in params["p1_counts"]]
    candidates.extend(
        math.ceil(float(ratio) * n_features) for ratio in params["p1_ratios"]
    )

    values = tuple(sorted({value for value in candidates if 2 <= value <= n_features}))
    if not values:
        raise ValueError(
            f"{algorithm}: no valid p1 value for dataset with {n_features} features"
        )
    return values


def _resolve_p2_values(algorithm: str, p1: int) -> tuple[int, ...]:
    params = _get_algorithm_config(algorithm)
    if algorithm in {"my_v0", "my_v1"}:
        candidates = [int(value) for value in params["p2_counts"]]
        candidates.extend(
            math.ceil(float(ratio) * p1) for ratio in params["p2_ratios"]
        )
    else:
        candidates = [int(value) for value in params["p2_values"]]
    return tuple(sorted({value for value in candidates if 1 <= value < p1}))


def _resolve_pdmf_neighbor_settings(
    algorithm: str,
) -> tuple[int | float | None, ...]:
    if algorithm not in {"my_v0", "my_v1", "my_v2"}:
        return (None,)
    return _resolve_neighbor_settings(algorithm, "pdmf_neighbors")


def _resolve_neighbor_settings(
    algorithm: str,
    name: str,
) -> tuple[int | float, ...]:
    params = _get_algorithm_config(algorithm)
    settings: list[int | float] = []
    seen: set[tuple[str, int | float]] = set()
    for mode, values in (
        ("count", params[f"{name}_counts"]),
        ("ratio", params[f"{name}_ratios"]),
    ):
        for raw_value in values:
            value = int(raw_value) if mode == "count" else float(raw_value)
            key = (mode, value)
            if key not in seen:
                seen.add(key)
                settings.append(value)
    return tuple(settings)


def _resolve_graph_neighbor_settings(
    algorithm: str,
) -> tuple[int | float | None, ...]:
    if algorithm not in {"my_v1", "my_v2"}:
        return (None,)
    return _resolve_neighbor_settings(algorithm, "graph_neighbors")


def _resolve_similarity_lambda_settings(
    algorithm: str,
) -> tuple[float | None, ...]:
    if algorithm not in {"my_v1", "my_v2"}:
        return (None,)
    return tuple(
        dict.fromkeys(
            float(value)
            for value in _get_algorithm_config(algorithm)[
                "pdmf_similarity_lambda_ratios"
            ]
        )
    )


def _resolve_stability_delta_settings(
    algorithm: str,
) -> tuple[float | None, ...]:
    if algorithm != "my_v2":
        return (None,)
    return tuple(
        dict.fromkeys(
            float(value)
            for value in _get_algorithm_config(algorithm)["stability_delta_values"]
        )
    )


def _format_pdmf_neighbors(value: int | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return f"count:{value}"
    return f"ratio:{value:.12g}"


def _format_graph_neighbors(value: int | float | None) -> str:
    return _format_pdmf_neighbors(value)


def _format_similarity_lambda(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _format_stability_delta(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _mean_std(rows: list[dict[str, str]], field: str) -> tuple[float, float]:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def _benchmark_summary_row(
    algorithm: str,
    best: dict[str, object],
    best_params: dict[str, object],
) -> dict[str, object]:
    excluded = {"dataset", *best_params}
    return {
        "algorithm": algorithm,
        "dataset": best["dataset"],
        "best_params": json.dumps(best_params, ensure_ascii=False),
        **{key: value for key, value in best.items() if key not in excluded},
    }


def _summarize(
    output_dir: Path,
    config: ExperimentConfig,
    algorithm: str,
    parameter_combinations: tuple[
        tuple[
            int | None,
            int | None,
            int | float | None,
            int | float | None,
            float | None,
            float | None,
        ],
        ...,
    ],
) -> dict[str, object]:
    algorithm_config = _get_algorithm_config(algorithm)
    theta_values = tuple(algorithm_config["theta_values"])
    rows = _read_csv(output_dir / "all_runs.csv")
    parameter_fields = (
        (
            "stability_delta",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
        )
        if algorithm == "my_v2"
        else (
            "p1",
            "p2",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
        )
    )
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[tuple(row[field] for field in parameter_fields)].append(row)

    adaptive_fields = (
        "selected_p1_mean",
        "selected_p1_std",
        "local_feature_count_mean",
        "local_feature_count_std",
        "local_feature_count_min",
        "local_feature_count_max",
        "global_entropy_loss_mean",
        "global_entropy_loss_std",
        "global_graph_loss_mean",
        "global_graph_loss_std",
    ) if algorithm == "my_v2" else ()
    fields = (
        "dataset",
        *parameter_fields,
        "success_runs",
        *adaptive_fields,
        *(name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")),
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "runtime_seconds_sum",
    )
    summaries: list[dict[str, object]] = []
    dataset_name = rows[0]["dataset"] if rows else ""
    for (
        p1,
        p2,
        pdmf_neighbors,
        graph_neighbors,
        similarity_lambda,
        stability_delta,
    ) in parameter_combinations:
        pdmf_neighbors_text = _format_pdmf_neighbors(pdmf_neighbors)
        graph_neighbors_text = _format_graph_neighbors(graph_neighbors)
        similarity_lambda_text = _format_similarity_lambda(similarity_lambda)
        stability_delta_text = _format_stability_delta(stability_delta)
        for theta in theta_values:
            parameter_values: dict[str, object] = {
                "pdmf_neighbors": pdmf_neighbors_text,
                "graph_neighbors": graph_neighbors_text,
                "pdmf_similarity_lambda": similarity_lambda_text,
                "theta": f"{theta:.2f}",
            }
            if algorithm == "my_v2":
                parameter_values["stability_delta"] = stability_delta_text
            else:
                parameter_values.update(p1=p1, p2=p2)
            key = tuple(str(parameter_values[field]) for field in parameter_fields)
            success = grouped.get(key, [])
            item: dict[str, object] = {
                "dataset": dataset_name,
                **parameter_values,
                "success_runs": len(success),
            }
            if algorithm == "my_v2":
                for source, mean_field, std_field in (
                    ("selected_p1", "selected_p1_mean", "selected_p1_std"),
                    (
                        "local_feature_count_mean",
                        "local_feature_count_mean",
                        "local_feature_count_std",
                    ),
                    (
                        "global_entropy_loss",
                        "global_entropy_loss_mean",
                        "global_entropy_loss_std",
                    ),
                    (
                        "global_graph_loss",
                        "global_graph_loss_mean",
                        "global_graph_loss_std",
                    ),
                ):
                    available = [row for row in success if row.get(source, "") != ""]
                    if available:
                        mean, std = _mean_std(available, source)
                    else:
                        mean = std = float("nan")
                    item[mean_field] = mean
                    item[std_field] = std
                local_mins = [
                    float(row["local_feature_count_min"])
                    for row in success
                    if row.get("local_feature_count_min", "") != ""
                ]
                local_maxs = [
                    float(row["local_feature_count_max"])
                    for row in success
                    if row.get("local_feature_count_max", "") != ""
                ]
                item["local_feature_count_min"] = (
                    min(local_mins) if local_mins else float("nan")
                )
                item["local_feature_count_max"] = (
                    max(local_maxs) if local_maxs else float("nan")
                )
            for metric in METRICS:
                if success:
                    mean, std = _mean_std(success, metric)
                else:
                    mean = std = float("nan")
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
            if success:
                runtime_mean, runtime_std = _mean_std(success, "runtime_seconds")
                runtime_sum = sum(float(row["runtime_seconds"]) for row in success)
            else:
                runtime_mean = runtime_std = float("nan")
                runtime_sum = 0.0
            item.update(
                runtime_seconds_mean=runtime_mean,
                runtime_seconds_std=runtime_std,
                runtime_seconds_sum=runtime_sum,
            )
            summaries.append(item)
    _write_csv(output_dir / "grid_summary.csv", summaries, fields)

    candidates = [
        row
        for row in summaries
        if int(row["success_runs"]) == len(config.seeds)
        and math.isfinite(float(row["acc_mean"]))
    ]
    if not candidates:
        raise RuntimeError("No parameter combination completed every seed")
    best = min(
        candidates,
        key=lambda row: (
            -float(row["acc_mean"]),
            -float(row["nmi_mean"]),
            -float(row["f_measure_mean"]),
            float(row["runtime_seconds_mean"]),
            *(str(row[field]) for field in parameter_fields),
        ),
    )
    best = {
        **best,
        "selection_metric": "acc_mean",
        "grid_runtime_seconds": sum(
            float(row["runtime_seconds"])
            for row in rows
            if row["status"] == "success"
        ),
    }
    best_fields = (*fields, "selection_metric", "grid_runtime_seconds")
    _write_csv(output_dir / "best_parameter_combination.csv", [best], best_fields)

    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    result = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "planned_runs": len(parameter_combinations)
        * len(theta_values)
        * len(config.seeds),
        "completed_rows": len(rows),
        "status_counts": dict(status_counts),
        "best_parameter_combination": best,
    }
    _write_json(output_dir / "experiment_summary.json", result)
    return best


def _create_model(
    algorithm: str,
    config: ExperimentConfig,
    *,
    p1: int | None,
    p2: int | None,
    theta: float,
    n_clusters: int,
    seed: int,
    pdmf_neighbors: int | float | None = None,
    graph_neighbors: int | float | None = None,
    pdmf_similarity_lambda: float | None = None,
    stability_delta: float | None = None,
    precomputed_pseudo_labels: np.ndarray | None = None,
    precomputed_global_selection: tuple[object, ...] | None = None,
    root_feature_ranking_cache: dict[str, object] | None = None,
) -> PLGBFSC | MYV0 | MYV1 | MYV2:
    algorithm_config = _get_algorithm_config(algorithm)
    if algorithm != "my_v2" and (p1 is None or p2 is None):
        raise ValueError(f"{algorithm}: p1 and p2 are required")
    if algorithm == "plgb_fsc":
        return PLGBFSC(
            PLGBFSCConfig(p1=p1, p2=p2, purity=theta),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
        )
    if algorithm == "my_v0":
        if pdmf_neighbors is None:
            pdmf_neighbors = _resolve_pdmf_neighbor_settings(algorithm)[0]
        return MYV0(
            MYV0Config(
                p1=p1,
                p2=p2,
                purity=theta,
                pdmf_neighbors=pdmf_neighbors,
                pdmf_epsilon=float(algorithm_config["pdmf_epsilon"]),
            ),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
            precomputed_global_selection=precomputed_global_selection,
        )
    if algorithm == "my_v1":
        if pdmf_neighbors is None:
            pdmf_neighbors = _resolve_pdmf_neighbor_settings(algorithm)[0]
        if graph_neighbors is None:
            graph_neighbors = _resolve_graph_neighbor_settings(algorithm)[0]
        if pdmf_similarity_lambda is None:
            pdmf_similarity_lambda = _resolve_similarity_lambda_settings(algorithm)[0]
        return MYV1(
            MYV1Config(
                p1=p1,
                p2=p2,
                purity=theta,
                pdmf_neighbors=pdmf_neighbors,
                pdmf_epsilon=float(algorithm_config["pdmf_epsilon"]),
                graph_neighbors=graph_neighbors,
                pdmf_similarity_lambda=pdmf_similarity_lambda,
            ),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
            precomputed_global_selection=precomputed_global_selection,
            root_feature_ranking_cache=root_feature_ranking_cache,
        )
    if algorithm == "my_v2":
        if pdmf_neighbors is None:
            pdmf_neighbors = _resolve_pdmf_neighbor_settings(algorithm)[0]
        if graph_neighbors is None:
            graph_neighbors = _resolve_graph_neighbor_settings(algorithm)[0]
        if pdmf_similarity_lambda is None:
            pdmf_similarity_lambda = _resolve_similarity_lambda_settings(algorithm)[0]
        if stability_delta is None:
            stability_delta = _resolve_stability_delta_settings(algorithm)[0]
        if stability_delta is None:
            raise ValueError("my_v2: stability_delta is required")
        return MYV2(
            MYV2Config(
                stability_delta=stability_delta,
                purity=theta,
                pdmf_neighbors=pdmf_neighbors,
                pdmf_epsilon=float(algorithm_config["pdmf_epsilon"]),
                graph_neighbors=graph_neighbors,
                pdmf_similarity_lambda=pdmf_similarity_lambda,
            ),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
            precomputed_global_selection=precomputed_global_selection,
            root_feature_ranking_cache=root_feature_ranking_cache,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _algorithm_parameters(
    algorithm: str,
    config: ExperimentConfig,
) -> dict[str, object]:
    algorithm_config = _get_algorithm_config(algorithm)
    if algorithm == "plgb_fsc":
        return {
            "global_selection": "source-compatible pseudo-label mutual information",
            "local_selection": "source discernibility score",
        }
    if algorithm == "my_v0":
        return {
            "global_selection": "Gaussian-PDMF inner-significance ranking",
            "local_selection": "local Gaussian-PDMF inner-significance ranking",
            "p2_counts": algorithm_config["p2_counts"],
            "p2_ratios": algorithm_config["p2_ratios"],
            "pdmf_neighbors_counts": algorithm_config["pdmf_neighbors_counts"],
            "pdmf_neighbors_ratios": algorithm_config["pdmf_neighbors_ratios"],
            "pdmf_epsilon": algorithm_config["pdmf_epsilon"],
        }
    if algorithm == "my_v1":
        return {
            "global_selection": "Gaussian-PDMF inner-significance ranking",
            "local_selection": "equal-ranked local entropy and sparse-graph importance",
            "p2_counts": algorithm_config["p2_counts"],
            "p2_ratios": algorithm_config["p2_ratios"],
            "pdmf_neighbors_counts": algorithm_config["pdmf_neighbors_counts"],
            "pdmf_neighbors_ratios": algorithm_config["pdmf_neighbors_ratios"],
            "pdmf_epsilon": algorithm_config["pdmf_epsilon"],
            "graph_neighbors_counts": algorithm_config["graph_neighbors_counts"],
            "graph_neighbors_ratios": algorithm_config["graph_neighbors_ratios"],
            "pdmf_similarity_lambda_ratios": algorithm_config[
                "pdmf_similarity_lambda_ratios"
            ],
        }
    if algorithm == "my_v2":
        return {
            "global_selection": "smallest entropy-graph-stable global prefix",
            "local_selection": "smallest entropy-graph-stable local prefix",
            "stability_delta_values": algorithm_config[
                "stability_delta_values"
            ],
            "pdmf_neighbors_counts": algorithm_config["pdmf_neighbors_counts"],
            "pdmf_neighbors_ratios": algorithm_config["pdmf_neighbors_ratios"],
            "pdmf_epsilon": algorithm_config["pdmf_epsilon"],
            "graph_neighbors_counts": algorithm_config["graph_neighbors_counts"],
            "graph_neighbors_ratios": algorithm_config["graph_neighbors_ratios"],
            "pdmf_similarity_lambda_ratios": algorithm_config[
                "pdmf_similarity_lambda_ratios"
            ],
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _run_algorithm_grid(
    dataset: Dataset,
    config: ExperimentConfig,
    run_id: str,
    algorithm: str,
) -> dict[str, object]:
    algorithm_config = _get_algorithm_config(algorithm)
    theta_values = tuple(algorithm_config["theta_values"])
    if algorithm == "my_v2":
        p1_values: tuple[int, ...] = ()
        parameter_pairs: tuple[tuple[int | None, int | None], ...] = ((None, None),)
    else:
        p1_values = _resolve_p1_values(algorithm, dataset.n_features)
        parameter_pairs = tuple(
            (p1, p2)
            for p1 in p1_values
            for p2 in _resolve_p2_values(algorithm, p1)
        )
        if not parameter_pairs:
            raise ValueError(f"{dataset.name}: no p2 value is smaller than any p1")
    pdmf_neighbor_settings = _resolve_pdmf_neighbor_settings(algorithm)
    graph_neighbor_settings = _resolve_graph_neighbor_settings(algorithm)
    similarity_lambda_settings = _resolve_similarity_lambda_settings(algorithm)
    stability_delta_settings = _resolve_stability_delta_settings(algorithm)
    parameter_combinations = tuple(
        (
            p1,
            p2,
            pdmf_neighbors,
            graph_neighbors,
            similarity_lambda,
            stability_delta,
        )
        for p1, p2 in parameter_pairs
        for pdmf_neighbors in pdmf_neighbor_settings
        for graph_neighbors in graph_neighbor_settings
        for similarity_lambda in similarity_lambda_settings
        for stability_delta in stability_delta_settings
    )

    output_dir = config.output_root / run_id / dataset.name / algorithm
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    all_runs_path = output_dir / "all_runs.csv"
    previous = _read_csv(all_runs_path) if config.resume else []
    completed_fields = (
        (
            "stability_delta",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
        )
        if algorithm == "my_v2"
        else (
            "p1",
            "p2",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
        )
    )
    completed = {
        (int(row["seed"]), *(row[field] for field in completed_fields))
        for row in previous
    }

    selection_parameters = (
        {
            "p1_rule": "adaptive smallest entropy-graph-stable prefix",
            "p2_rule": "adaptive per granular ball",
            "stability_delta_values": tuple(stability_delta_settings),
        }
        if algorithm == "my_v2"
        else {
            "p1_rule": "p1_counts plus ceil(p1_ratios * n_features)",
            "p1_values": p1_values,
            "p1_counts": tuple(algorithm_config["p1_counts"]),
            "p1_ratios": tuple(algorithm_config["p1_ratios"]),
            "p1_p2_pairs": parameter_pairs,
        }
    )

    _write_json(
        output_dir / "experiment_config.json",
        {
            "algorithm": algorithm,
            "dataset": dataset.name,
            "seeds": config.seeds,
            **selection_parameters,
            "pdmf_neighbors_settings": tuple(
                _format_pdmf_neighbors(value) for value in pdmf_neighbor_settings
            ),
            "graph_neighbors_settings": tuple(
                _format_graph_neighbors(value) for value in graph_neighbor_settings
            ),
            "pdmf_similarity_lambda_settings": tuple(
                _format_similarity_lambda(value)
                for value in similarity_lambda_settings
            ),
            "theta_values": theta_values,
            "preprocessing": "global_featurewise_minmax_to_0_1",
            "stop_rule": "pseudo_purity >= theta and ball_size < 8",
            "selection_rule": "maximize mean ACC across seeds",
            "algorithm_parameters": _algorithm_parameters(algorithm, config),
        },
    )

    mode = "a" if previous else "w"
    X = minmax_scale(dataset.X)
    run_fields = V2_RUN_FIELDS if algorithm == "my_v2" else RUN_FIELDS
    pseudo_labels_by_seed: dict[int, np.ndarray] = {}
    global_selection_cache: dict[tuple[str, ...], tuple[object, ...]] = {}
    root_feature_ranking_caches: dict[tuple[str, ...], dict[str, object]] = {}
    planned = len(config.seeds) * len(parameter_combinations) * len(theta_values)
    with all_runs_path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_fields)
        if mode == "w":
            writer.writeheader()
        done = len(completed)
        for seed in config.seeds:
            for (
                p1,
                p2,
                pdmf_neighbors,
                graph_neighbors,
                similarity_lambda,
                stability_delta,
            ) in parameter_combinations:
                pdmf_neighbors_text = _format_pdmf_neighbors(pdmf_neighbors)
                graph_neighbors_text = _format_graph_neighbors(graph_neighbors)
                similarity_lambda_text = _format_similarity_lambda(similarity_lambda)
                stability_delta_text = _format_stability_delta(stability_delta)
                parameter_output = ""
                if algorithm in {"my_v0", "my_v1", "my_v2"}:
                    parameter_output += f"pdmf_neighbors={pdmf_neighbors_text} "
                if algorithm in {"my_v1", "my_v2"}:
                    parameter_output += (
                        f"graph_neighbors={graph_neighbors_text} "
                        f"lambda={similarity_lambda_text} "
                    )
                if algorithm == "my_v2":
                    parameter_output += f"delta={stability_delta_text} "
                    global_cache_key = (
                        stability_delta_text,
                        pdmf_neighbors_text,
                        graph_neighbors_text,
                        similarity_lambda_text,
                    )
                    root_cache_key = global_cache_key
                else:
                    global_cache_key = (str(p1), pdmf_neighbors_text)
                    root_cache_key = (
                        str(p1),
                        pdmf_neighbors_text,
                        graph_neighbors_text,
                        similarity_lambda_text,
                    )
                root_feature_ranking_cache = (
                    root_feature_ranking_caches.setdefault(root_cache_key, {})
                    if algorithm in {"my_v1", "my_v2"}
                    else None
                )
                for theta in theta_values:
                    key = (
                        (
                            seed,
                            stability_delta_text,
                            pdmf_neighbors_text,
                            graph_neighbors_text,
                            similarity_lambda_text,
                            f"{theta:.2f}",
                        )
                        if algorithm == "my_v2"
                        else (
                            seed,
                            str(p1),
                            str(p2),
                            pdmf_neighbors_text,
                            graph_neighbors_text,
                            similarity_lambda_text,
                            f"{theta:.2f}",
                        )
                    )
                    if key in completed:
                        continue
                    row: dict[str, object] = {field: "" for field in run_fields}
                    row.update(
                        algorithm=algorithm,
                        dataset=dataset.name,
                        seed=seed,
                        p1="" if p1 is None else p1,
                        p2="" if p2 is None else p2,
                        pdmf_neighbors=pdmf_neighbors_text,
                        graph_neighbors=graph_neighbors_text,
                        pdmf_similarity_lambda=similarity_lambda_text,
                        theta=f"{theta:.2f}",
                    )
                    if algorithm == "my_v2":
                        row["stability_delta"] = stability_delta_text
                    random.seed(seed)
                    np.random.seed(seed)
                    start = time.perf_counter()
                    try:
                        model = _create_model(
                            algorithm,
                            config,
                            p1=p1,
                            p2=p2,
                            theta=theta,
                            n_clusters=dataset.n_classes,
                            seed=seed,
                            pdmf_neighbors=pdmf_neighbors,
                            graph_neighbors=graph_neighbors,
                            pdmf_similarity_lambda=similarity_lambda,
                            stability_delta=stability_delta,
                            precomputed_pseudo_labels=pseudo_labels_by_seed.get(seed),
                            precomputed_global_selection=global_selection_cache.get(
                                global_cache_key
                            ),
                            root_feature_ranking_cache=root_feature_ranking_cache,
                        )
                        labels = model.fit_predict(X.copy())
                        if seed not in pseudo_labels_by_seed:
                            pseudo_labels_by_seed[seed] = model.pseudo_labels_.copy()
                        if global_cache_key not in global_selection_cache:
                            if algorithm in {"my_v0", "my_v1"}:
                                global_selection_cache[global_cache_key] = (
                                    model.selected_feature_indices_.copy(),
                                    model.attribute_scores_.copy(),
                                )
                            elif algorithm == "my_v2":
                                global_selection_cache[global_cache_key] = (
                                    model.selected_feature_indices_.copy(),
                                    model.attribute_scores_.copy(),
                                    model.global_entropy_loss_,
                                    model.global_graph_loss_,
                                )
                        if algorithm == "my_v2":
                            local_counts = np.asarray(
                                model.local_feature_counts_, dtype=float
                            )
                            row.update(
                                selected_p1=model.p1_,
                                global_entropy_loss=model.global_entropy_loss_,
                                global_graph_loss=model.global_graph_loss_,
                            )
                            if local_counts.size:
                                row.update(
                                    local_feature_count_mean=float(local_counts.mean()),
                                    local_feature_count_min=int(local_counts.min()),
                                    local_feature_count_max=int(local_counts.max()),
                                )
                        runtime = time.perf_counter() - start
                        metrics = evaluate_clustering(
                            dataset.y,
                            labels,
                            nmi_average_method=config.nmi_average_method,
                        ).as_dict()
                        pdmf_suffix = (
                            ""
                            if not pdmf_neighbors_text
                            else "_pdmf_"
                            + pdmf_neighbors_text.replace(":", "_").replace(".", "p")
                        )
                        graph_suffix = (
                            ""
                            if not graph_neighbors_text
                            else "_graph_"
                            + graph_neighbors_text.replace(":", "_").replace(".", "p")
                        )
                        lambda_suffix = (
                            ""
                            if not similarity_lambda_text
                            else "_lambda_" + similarity_lambda_text.replace(".", "p")
                        )
                        if algorithm == "my_v2":
                            delta_suffix = stability_delta_text.replace(".", "p")
                            prediction_name = (
                                f"seed_{seed}_delta_{delta_suffix}{pdmf_suffix}"
                                f"{graph_suffix}{lambda_suffix}_theta_{theta:.2f}.npy"
                            )
                        else:
                            prediction_name = (
                                f"seed_{seed}_p1_{p1}_p2_{p2}{pdmf_suffix}"
                                f"{graph_suffix}{lambda_suffix}_theta_{theta:.2f}.npy"
                            )
                        prediction = labels_dir / prediction_name
                        np.save(prediction, labels)
                        row.update(
                            status="success",
                            runtime_seconds=runtime,
                            prediction_path=prediction.relative_to(output_dir).as_posix(),
                            **metrics,
                        )
                    except Exception as exc:
                        row.update(
                            status="failed",
                            runtime_seconds=time.perf_counter() - start,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                        (output_dir / "last_error.txt").write_text(
                            traceback.format_exc(), encoding="utf-8"
                        )
                    writer.writerow(row)
                    stream.flush()
                    done += 1
                    metric_text = (
                        f"NMI={float(row['nmi']):.4f} "
                        f"ACC={float(row['acc']):.4f} "
                        f"F-measure={float(row['f_measure']):.4f}"
                        if row["status"] == "success"
                        else "NMI=N/A ACC=N/A F-measure=N/A"
                    )
                    fixed_counts = (
                        f"selected_p1={row['selected_p1']} "
                        f"local_mean={row['local_feature_count_mean']} "
                        if algorithm == "my_v2" and row["status"] == "success"
                        else f"p1={p1} p2={p2} "
                        if algorithm != "my_v2"
                        else ""
                    )
                    print(
                        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] "
                        f"[{done}/{planned}] {algorithm} {dataset.name} seed={seed} "
                        f"{fixed_counts}{parameter_output}"
                        f"theta={theta:.2f} {row['status']} "
                        f"runtime={float(row['runtime_seconds']):.3f}s {metric_text}",
                        flush=True,
                    )
    return _summarize(output_dir, config, algorithm, parameter_combinations)


def main() -> int:
    if not EXPERIMENT.datasets:
        raise ValueError("At least one dataset is required")
    if not EXPERIMENT.seeds:
        raise ValueError("At least one seed is required")
    if not EXPERIMENT.algorithms:
        raise ValueError("At least one algorithm is required")
    unknown = sorted(set(EXPERIMENT.datasets).difference(DATASETS))
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}")
    if len(EXPERIMENT.seeds) != len(set(EXPERIMENT.seeds)):
        raise ValueError("Seeds must be unique")
    supported_algorithms = {"plgb_fsc", "my_v0", "my_v1", "my_v2"}
    unknown_algorithms = sorted(set(EXPERIMENT.algorithms).difference(supported_algorithms))
    if unknown_algorithms:
        raise KeyError(f"Unknown algorithms: {unknown_algorithms}")
    if len(EXPERIMENT.algorithms) != len(set(EXPERIMENT.algorithms)):
        raise ValueError("Algorithms must be unique")
    for algorithm in EXPERIMENT.algorithms:
        _validate_algorithm_config(algorithm)
    run_id = EXPERIMENT.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    best_rows = []
    for name in EXPERIMENT.datasets:
        dataset = load_dataset(DATASETS[name])
        for algorithm in EXPERIMENT.algorithms:
            algorithm_config = _get_algorithm_config(algorithm)
            best = _run_algorithm_grid(
                dataset,
                EXPERIMENT,
                run_id,
                algorithm,
            )
            best_params = {"theta": float(best["theta"])}
            if algorithm != "my_v2":
                best_params.update(
                    p1=int(best["p1"]),
                    p2=int(best["p2"]),
                )
            if algorithm == "my_v0":
                best_params.update(
                    pdmf_neighbors=best["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                )
            if algorithm == "my_v1":
                best_params.update(
                    pdmf_neighbors=best["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                    graph_neighbors=best["graph_neighbors"],
                    pdmf_similarity_lambda=best["pdmf_similarity_lambda"],
                )
            if algorithm == "my_v2":
                best_params.update(
                    stability_delta=float(best["stability_delta"]),
                    pdmf_neighbors=best["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                    graph_neighbors=best["graph_neighbors"],
                    pdmf_similarity_lambda=best["pdmf_similarity_lambda"],
                )
            best_rows.append(
                _benchmark_summary_row(
                    algorithm,
                    best,
                    best_params=best_params,
                )
            )
    summary_path = EXPERIMENT.output_root / run_id / "benchmark_summary.csv"
    fields = tuple(dict.fromkeys(field for row in best_rows for field in row))
    _write_csv(summary_path, best_rows, fields)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# cd D:\Projects\zhengchuang\paper_team\liqiu\algorithm\unified_benchmark
# ..\python\.venv\Scripts\python.exe .\run.py
# ..\python\.venv\Scripts\python.exe .\run.py *> run.log
# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > run-0802-2105.log 2>&1" -WindowStyle Hidden
