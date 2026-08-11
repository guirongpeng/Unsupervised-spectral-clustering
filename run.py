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
from algorithms.my_v3 import MYV3, MYV3Config
from algorithms.my_v4 import MYV4, MYV4Config
from algorithms.plgb_fsc import PLGBFSC, PLGBFSCConfig
from algorithms.gb_pojg_gbdpc import GBPOJGGBDPC, GBPOJGGBDPCConfig
from algorithms.gb_pojg_gbsc import GBPOJGGBSC, GBPOJGGBSCConfig
from algorithms.gbsc import GBSC, GBSCConfig
from algorithms.sagbc import SAGBC, SAGBCConfig
from algorithms.gbct import GBCT, GBCTConfig
from config import (
    DATASETS,
    EXPERIMENT,
    GB_POJG_GBDPC_PARAMS,
    GB_POJG_GBSC_PARAMS,
    GBSC_PARAMS,
    SAGBC_PARAMS,
    GBCT_PARAMS,
    MY_V0_PARAMS,
    MY_V1_PARAMS,
    MY_V2_PARAMS,
    MY_V3_PARAMS,
    MY_V4_PARAMS,
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
V3_RUN_FIELDS = (
    *RUN_FIELDS,
    "redundancy_beta",
    "fusion_alpha_mode",
    "mutual_knn",
    "self_tuning_graph",
)
GB_POJG_GBDPC_RUN_FIELDS = (
    "algorithm",
    "dataset",
    "seed",
    "gamma",
    "delta",
    "status",
    *METRICS,
    "runtime_seconds",
    "prediction_path",
    "error_type",
    "error_message",
)
GBSC_RUN_FIELDS = (
    "algorithm", "dataset", "seed", "sigma", "status", *METRICS,
    "runtime_seconds", "prediction_path", "error_type", "error_message",
)
GB_POJG_GBSC_RUN_FIELDS = (
    "algorithm", "dataset", "seed", "gamma", "delta", "sigma", "status", *METRICS,
    "runtime_seconds", "prediction_path", "error_type", "error_message",
)
SAGBC_RUN_FIELDS = (
    "algorithm", "dataset", "seed", "sample_size", "status", *METRICS,
    "runtime_seconds", "prediction_path", "error_type", "error_message",
)
GBCT_RUN_FIELDS = ("algorithm", "dataset", "seed", "n_clusters", "status", *METRICS, "runtime_seconds", "prediction_path", "error_type", "error_message")


def _get_algorithm_config(algorithm: str) -> dict[str, object]:
    if algorithm == "gbct":
        return GBCT_PARAMS
    if algorithm == "sagbc":
        return SAGBC_PARAMS
    if algorithm == "gbsc":
        return GBSC_PARAMS
    if algorithm == "gb_pojg_gbdpc":
        return GB_POJG_GBDPC_PARAMS
    if algorithm == "gb_pojg_gbsc":
        return GB_POJG_GBSC_PARAMS
    if algorithm == "plgb_fsc":
        return PLGB_FSC_PARAMS
    if algorithm == "my_v0":
        return MY_V0_PARAMS
    if algorithm == "my_v1":
        return MY_V1_PARAMS
    if algorithm == "my_v2":
        return MY_V2_PARAMS
    if algorithm == "my_v3":
        return MY_V3_PARAMS
    if algorithm == "my_v4":
        return MY_V4_PARAMS
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _validate_algorithm_config(algorithm: str) -> None:
    params = _get_algorithm_config(algorithm)
    if algorithm == "gbct":
        if not 0 <= float(params["noise_density_ratio"]): raise ValueError("gbct: noise_density_ratio must be non-negative")
        return
    if algorithm == "sagbc":
        sample_size = params["sample_size"]
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 2:
            raise ValueError("sagbc: sample_size must be an integer >= 2")
        return
    if algorithm == "gb_pojg_gbdpc":
        gammas = tuple(float(value) for value in params["gamma_values"])
        deltas = tuple(float(value) for value in params["delta_values"])
        if not gammas or any(not math.isfinite(value) or value < 0 for value in gammas):
            raise ValueError("gb_pojg_gbdpc: gamma_values must be finite values in [0, +inf)")
        if not deltas or any(not math.isfinite(value) or not 0 < value <= 1 for value in deltas):
            raise ValueError("gb_pojg_gbdpc: delta_values must be finite values in (0, 1]")
        return
    if algorithm == "gb_pojg_gbsc":
        for name, lower, strict in (("gamma_values", 0.0, False), ("delta_values", 0.0, True), ("sigma_values", 0.0, True)):
            values = tuple(float(value) for value in params[name])
            if not values or any(not math.isfinite(value) or (value <= lower if strict else value < lower) for value in values):
                relation = "> 0" if strict else ">= 0"
                raise ValueError(f"gb_pojg_gbsc: {name} must contain finite values {relation}")
        if any(float(value) > 1.0 for value in params["delta_values"]):
            raise ValueError("gb_pojg_gbsc: delta_values must be in (0, 1]")
        return
    if algorithm == "gbsc":
        sigmas = tuple(float(value) for value in params["sigma_values"])
        if not sigmas or any(not math.isfinite(value) or value <= 0 for value in sigmas):
            raise ValueError("gbsc: sigma_values must be finite positive values")
        return
    theta_values = tuple(params["theta_values"])
    if algorithm != "my_v2":
        _validate_count_ratio_options(algorithm, params, "p1", minimum_count=2)
        if algorithm in {"my_v0", "my_v1", "my_v3", "my_v4"}:
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
    if algorithm in {"my_v0", "my_v1", "my_v2", "my_v3", "my_v4"}:
        _validate_count_ratio_options(
            algorithm, params, "pdmf_neighbors", minimum_count=1
        )
    if algorithm in {"my_v0", "my_v1", "my_v2", "my_v3", "my_v4"}:
        epsilon = float(params["pdmf_epsilon"])
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"{algorithm}: pdmf_epsilon must be positive")
    if algorithm in {"my_v1", "my_v2", "my_v3", "my_v4"}:
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
        jobs = params["ball_parallel_jobs"]
        if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
            raise ValueError("my_v2: ball_parallel_jobs must be an integer >= 1")
    if algorithm == "my_v3":
        betas = tuple(float(value) for value in params["redundancy_beta_values"])
        if not betas or any(not math.isfinite(value) or value < 0 for value in betas):
            raise ValueError("my_v3: redundancy_beta_values must be non-negative")
        if params["fusion_alpha_mode"] not in {"adaptive", "equal"}:
            raise ValueError("my_v3: fusion_alpha_mode must be adaptive or equal")
        if not isinstance(params["mutual_knn"], bool):
            raise ValueError("my_v3: mutual_knn must be boolean")
        if not isinstance(params["self_tuning_graph"], bool):
            raise ValueError("my_v3: self_tuning_graph must be boolean")


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
    if algorithm in {"my_v0", "my_v1", "my_v3", "my_v4"}:
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
    if algorithm not in {"my_v0", "my_v1", "my_v2", "my_v3", "my_v4"}:
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
    if algorithm not in {"my_v1", "my_v2", "my_v3", "my_v4"}:
        return (None,)
    return _resolve_neighbor_settings(algorithm, "graph_neighbors")


def _resolve_similarity_lambda_settings(
    algorithm: str,
) -> tuple[float | None, ...]:
    if algorithm not in {"my_v1", "my_v2", "my_v3", "my_v4"}:
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


def _resolve_redundancy_beta_settings(
    algorithm: str,
) -> tuple[float | None, ...]:
    if algorithm != "my_v3":
        return (None,)
    return tuple(
        dict.fromkeys(
            float(value)
            for value in _get_algorithm_config(algorithm)[
                "redundancy_beta_values"
            ]
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
    parameter_combinations: tuple[tuple[object, ...], ...],
) -> dict[str, object]:
    algorithm_config = _get_algorithm_config(algorithm)
    theta_values = tuple(algorithm_config["theta_values"])
    rows = _read_csv(output_dir / "all_runs.csv")
    if algorithm == "my_v2":
        parameter_fields = (
            "stability_delta",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
        )
    elif algorithm == "my_v3":
        parameter_fields = (
            "p1",
            "p2",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
            "redundancy_beta",
        )
    else:
        parameter_fields = (
            "p1",
            "p2",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
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
        redundancy_beta,
    ) in parameter_combinations:
        pdmf_neighbors_text = _format_pdmf_neighbors(pdmf_neighbors)
        graph_neighbors_text = _format_graph_neighbors(graph_neighbors)
        similarity_lambda_text = _format_similarity_lambda(similarity_lambda)
        stability_delta_text = _format_stability_delta(stability_delta)
        redundancy_beta_text = "" if redundancy_beta is None else f"{float(redundancy_beta):.12g}"
        for theta in theta_values:
            parameter_values: dict[str, object] = {
                "pdmf_neighbors": pdmf_neighbors_text,
                "graph_neighbors": graph_neighbors_text,
                "pdmf_similarity_lambda": similarity_lambda_text,
                "theta": f"{theta:.2f}",
            }
            if algorithm == "my_v2":
                parameter_values["stability_delta"] = stability_delta_text
            elif algorithm == "my_v3":
                parameter_values.update(
                    p1=p1,
                    p2=p2,
                    redundancy_beta=redundancy_beta_text,
                )
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
    redundancy_beta: float | None = None,
    gamma: float | None = None,
    delta: float | None = None,
    sigma: float | None = None,
    sagbc_sample_size: int | None = None,
    precomputed_pseudo_labels: np.ndarray | None = None,
    precomputed_global_selection: tuple[object, ...] | None = None,
    global_stability_curve_cache: dict[str, object] | None = None,
    root_feature_ranking_cache: dict[str, object] | None = None,
    local_feature_selection_cache: dict[tuple[object, ...], np.ndarray] | None = None,
) -> PLGBFSC | MYV0 | MYV1 | MYV2 | MYV3 | MYV4 | GBPOJGGBDPC | GBPOJGGBSC | GBSC | SAGBC | GBCT:
    algorithm_config = _get_algorithm_config(algorithm)
    if algorithm == "gbct":
        return GBCT(GBCTConfig(n_clusters=n_clusters, noise_density_ratio=float(algorithm_config["noise_density_ratio"])))
    if algorithm == "sagbc":
        if sagbc_sample_size is None:
            raise ValueError("sagbc: resolved sample_size is required")
        return SAGBC(SAGBCConfig(
            sample_size=sagbc_sample_size,
            random_state=seed,
        ))
    if algorithm == "gbsc":
        if sigma is None:
            raise ValueError("gbsc: sigma is required")
        return GBSC(GBSCConfig(sigma=float(sigma)), n_clusters=n_clusters, random_state=seed)
    if algorithm == "gb_pojg_gbdpc":
        if gamma is None or delta is None:
            raise ValueError("gb_pojg_gbdpc: gamma and delta are required")
        return GBPOJGGBDPC(
            GBPOJGGBDPCConfig(gamma=float(gamma), delta=float(delta)),
            n_clusters=n_clusters,
            random_state=seed,
        )
    if algorithm == "gb_pojg_gbsc":
        if gamma is None or delta is None or sigma is None:
            raise ValueError("gb_pojg_gbsc: gamma, delta and sigma are required")
        return GBPOJGGBSC(
            GBPOJGGBSCConfig(gamma=float(gamma), delta=float(delta), sigma=float(sigma)),
            n_clusters=n_clusters,
            random_state=seed,
        )
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
                ball_parallel_jobs=int(algorithm_config["ball_parallel_jobs"]),
            ),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
            precomputed_global_selection=precomputed_global_selection,
            global_stability_curve_cache=global_stability_curve_cache,
            root_feature_ranking_cache=root_feature_ranking_cache,
            local_feature_selection_cache=local_feature_selection_cache,
        )
    if algorithm == "my_v3":
        if pdmf_neighbors is None:
            pdmf_neighbors = _resolve_pdmf_neighbor_settings(algorithm)[0]
        if graph_neighbors is None:
            graph_neighbors = _resolve_graph_neighbor_settings(algorithm)[0]
        if pdmf_similarity_lambda is None:
            pdmf_similarity_lambda = _resolve_similarity_lambda_settings(algorithm)[0]
        if redundancy_beta is None:
            redundancy_beta = _resolve_redundancy_beta_settings(algorithm)[0]
        return MYV3(
            MYV3Config(
                p1=p1,
                p2=p2,
                purity=theta,
                pdmf_neighbors=pdmf_neighbors,
                pdmf_epsilon=float(algorithm_config["pdmf_epsilon"]),
                graph_neighbors=graph_neighbors,
                pdmf_similarity_lambda=pdmf_similarity_lambda,
                redundancy_beta=float(redundancy_beta),
                fusion_alpha_mode=str(algorithm_config["fusion_alpha_mode"]),
                mutual_knn=bool(algorithm_config["mutual_knn"]),
                self_tuning_graph=bool(algorithm_config["self_tuning_graph"]),
            ),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
            precomputed_global_selection=precomputed_global_selection,
            root_feature_ranking_cache=root_feature_ranking_cache,
        )
    if algorithm == "my_v4":
        if pdmf_neighbors is None:
            pdmf_neighbors = _resolve_pdmf_neighbor_settings(algorithm)[0]
        if graph_neighbors is None:
            graph_neighbors = _resolve_graph_neighbor_settings(algorithm)[0]
        if pdmf_similarity_lambda is None:
            pdmf_similarity_lambda = _resolve_similarity_lambda_settings(algorithm)[0]
        return MYV4(
            MYV4Config(
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
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _algorithm_parameters(
    algorithm: str,
    config: ExperimentConfig,
) -> dict[str, object]:
    algorithm_config = _get_algorithm_config(algorithm)
    if algorithm == "sagbc":
        return {
            "anchor_sampling": "official random representative-point sampling; min(config.sample_size, n_samples)",
            "granular_ball_generation": "official farthest-pair division until ball size <= 8",
            "structure_graph": "official 5-nearest-anchor Gaussian affiliation and 2r shared-neighbor connection",
            "sample_size": algorithm_config["sample_size"],
            "neighbor_count": 5,
            "max_ball_size": 8,
            "search_radius_scale": 2.0,
        }
    if algorithm == "gbsc":
        return {
            "granular_ball_generation": "official weighted-density division plus fixed-radius normalization",
            "spectral_clustering": "official UCI boundary-distance Gaussian affinity",
            "sigma_values": algorithm_config["sigma_values"],
            "minimum_split_size": 8,
            "radius_detection_factor": 2.0,
        }
    if algorithm == "gb_pojg_gbdpc":
        return {
            "granular_ball_generation": "official GB-POJG binary-tree pruning and source-style anomaly split",
            "clustering": "official GBDPC density * delta decision value",
            "gamma_values": algorithm_config["gamma_values"],
            "delta_values": algorithm_config["delta_values"],
        }
    if algorithm == "gb_pojg_gbsc":
        return {
            "granular_ball_generation": "official GB-POJG binary-tree pruning and source-style anomaly split",
            "spectral_clustering": "official GBSC boundary-distance Gaussian affinity and normalized Laplacian",
            "gamma_values": algorithm_config["gamma_values"],
            "delta_values": algorithm_config["delta_values"],
            "sigma_values": algorithm_config["sigma_values"],
        }
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
            "ball_parallel_jobs": algorithm_config["ball_parallel_jobs"],
            "graph_neighbors_counts": algorithm_config["graph_neighbors_counts"],
            "graph_neighbors_ratios": algorithm_config["graph_neighbors_ratios"],
            "pdmf_similarity_lambda_ratios": algorithm_config[
                "pdmf_similarity_lambda_ratios"
            ],
        }
    if algorithm == "my_v3":
        return {
            "global_selection": "Gaussian-PDMF importance with redundancy penalty",
            "local_selection": "adaptive entropy-graph fusion with redundancy penalty",
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
            "redundancy_beta_values": algorithm_config["redundancy_beta_values"],
            "fusion_alpha_mode": algorithm_config["fusion_alpha_mode"],
            "mutual_knn": algorithm_config["mutual_knn"],
            "self_tuning_graph": algorithm_config["self_tuning_graph"],
        }
    if algorithm == "my_v4":
        return {
            "global_selection": "Gaussian-PDMF plus confidence-weighted pseudo-label mutual information",
            "local_selection": "entropy-graph importance plus confidence-weighted pseudo-label consistency",
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
            "confidence_weighting": "automatic from nearest-vs-second-nearest distance margin",
        }
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _summarize_gb_pojg_gbdpc(
    output_dir: Path,
    config: ExperimentConfig,
    parameter_combinations: tuple[tuple[float, float], ...],
) -> dict[str, object]:
    rows = _read_csv(output_dir / "all_runs.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[(row["gamma"], row["delta"])].append(row)

    fields = (
        "dataset", "gamma", "delta", "success_runs",
        *(name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")),
        "runtime_seconds_mean", "runtime_seconds_std", "runtime_seconds_sum",
    )
    dataset_name = rows[0]["dataset"] if rows else ""
    summaries: list[dict[str, object]] = []
    for gamma, delta in parameter_combinations:
        key = (f"{gamma:.12g}", f"{delta:.12g}")
        success = grouped.get(key, [])
        item: dict[str, object] = {
            "dataset": dataset_name,
            "gamma": key[0],
            "delta": key[1],
            "success_runs": len(success),
        }
        for metric in METRICS:
            mean, std = _mean_std(success, metric) if success else (float("nan"), float("nan"))
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
        row for row in summaries
        if int(row["success_runs"]) == len(config.seeds) and math.isfinite(float(row["acc_mean"]))
    ]
    if not candidates:
        raise RuntimeError("No GB-POJG-GBDPC parameter combination completed every seed")
    best = min(
        candidates,
        key=lambda row: (
            -float(row["acc_mean"]), -float(row["nmi_mean"]),
            -float(row["f_measure_mean"]), float(row["runtime_seconds_mean"]),
            str(row["gamma"]), str(row["delta"]),
        ),
    )
    best = {
        **best,
        "selection_metric": "acc_mean",
        "grid_runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows if row["status"] == "success"),
    }
    _write_csv(
        output_dir / "best_parameter_combination.csv", [best],
        (*fields, "selection_metric", "grid_runtime_seconds"),
    )
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    _write_json(
        output_dir / "experiment_summary.json",
        {
            "dataset": dataset_name,
            "algorithm": "gb_pojg_gbdpc",
            "planned_runs": len(config.seeds) * len(parameter_combinations),
            "completed_rows": len(rows),
            "status_counts": dict(status_counts),
            "best_parameter_combination": best,
        },
    )
    return best


def _run_gb_pojg_gbdpc_grid(
    dataset: Dataset, config: ExperimentConfig, run_id: str
) -> dict[str, object]:
    params = _get_algorithm_config("gb_pojg_gbdpc")
    combinations = tuple(
        (float(gamma), float(delta))
        for gamma in dict.fromkeys(params["gamma_values"])
        for delta in dict.fromkeys(params["delta_values"])
    )
    output_dir = config.output_root / run_id / dataset.name / "gb_pojg_gbdpc"
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    all_runs_path = output_dir / "all_runs.csv"
    previous = _read_csv(all_runs_path) if config.resume else []
    completed = {
        (int(row["seed"]), row["gamma"], row["delta"])
        for row in previous
    }
    _write_json(
        output_dir / "experiment_config.json",
        {
            "algorithm": "gb_pojg_gbdpc",
            "dataset": dataset.name,
            "seeds": config.seeds,
            "gamma_values": tuple(gamma for gamma, _ in combinations),
            "delta_values": tuple(delta for _, delta in combinations),
            "preprocessing": "global_featurewise_minmax_to_0_1",
            "selection_rule": "maximize mean ACC across seeds",
            "algorithm_parameters": _algorithm_parameters("gb_pojg_gbdpc", config),
        },
    )
    X = minmax_scale(dataset.X)
    mode = "a" if previous else "w"
    planned = len(config.seeds) * len(combinations)
    with all_runs_path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GB_POJG_GBDPC_RUN_FIELDS)
        if mode == "w":
            writer.writeheader()
        done = len(completed)
        for seed in config.seeds:
            for gamma, delta in combinations:
                gamma_text, delta_text = f"{gamma:.12g}", f"{delta:.12g}"
                key = (seed, gamma_text, delta_text)
                if key in completed:
                    continue
                row: dict[str, object] = {field: "" for field in GB_POJG_GBDPC_RUN_FIELDS}
                row.update(
                    algorithm="gb_pojg_gbdpc", dataset=dataset.name, seed=seed,
                    gamma=gamma_text, delta=delta_text,
                )
                start = time.perf_counter()
                try:
                    model = _create_model(
                        "gb_pojg_gbdpc", config, p1=None, p2=None, theta=0.0,
                        n_clusters=dataset.n_classes, seed=seed, gamma=gamma, delta=delta,
                    )
                    labels = model.fit_predict(X.copy())
                    runtime = time.perf_counter() - start
                    metrics = evaluate_clustering(
                        dataset.y, labels, nmi_average_method=config.nmi_average_method
                    ).as_dict()
                    prediction = labels_dir / f"seed_{seed}_gamma_{gamma_text.replace('.', 'p')}_delta_{delta_text.replace('.', 'p')}.npy"
                    np.save(prediction, labels)
                    row.update(
                        status="success", runtime_seconds=runtime,
                        prediction_path=prediction.relative_to(output_dir).as_posix(), **metrics,
                    )
                except Exception as exc:
                    row.update(
                        status="failed", runtime_seconds=time.perf_counter() - start,
                        error_type=type(exc).__name__, error_message=str(exc),
                    )
                    (output_dir / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                writer.writerow(row)
                stream.flush()
                done += 1
                metric_text = (
                    f"NMI={float(row['nmi']):.4f} ACC={float(row['acc']):.4f} F-measure={float(row['f_measure']):.4f}"
                    if row["status"] == "success" else "NMI=N/A ACC=N/A F-measure=N/A"
                )
                print(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{done}/{planned}] gb_pojg_gbdpc "
                    f"{dataset.name} seed={seed} gamma={gamma_text} delta={delta_text} "
                    f"{row['status']} runtime={float(row['runtime_seconds']):.3f}s {metric_text}",
                    flush=True,
                )
    return _summarize_gb_pojg_gbdpc(output_dir, config, combinations)


def _summarize_gb_pojg_gbsc(
    output_dir: Path,
    config: ExperimentConfig,
    parameter_combinations: tuple[tuple[float, float, float], ...],
) -> dict[str, object]:
    rows = _read_csv(output_dir / "all_runs.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[(row["gamma"], row["delta"], row["sigma"])].append(row)
    fields = (
        "dataset", "gamma", "delta", "sigma", "success_runs",
        *(name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")),
        "runtime_seconds_mean", "runtime_seconds_std", "runtime_seconds_sum",
    )
    dataset_name = rows[0]["dataset"] if rows else ""
    summaries: list[dict[str, object]] = []
    for gamma, delta, sigma in parameter_combinations:
        key = (f"{gamma:.12g}", f"{delta:.12g}", f"{sigma:.12g}")
        success = grouped.get(key, [])
        item: dict[str, object] = {"dataset": dataset_name, "gamma": key[0], "delta": key[1], "sigma": key[2], "success_runs": len(success)}
        for metric in METRICS:
            item[f"{metric}_mean"], item[f"{metric}_std"] = _mean_std(success, metric) if success else (float("nan"), float("nan"))
        runtime_mean, runtime_std = _mean_std(success, "runtime_seconds") if success else (float("nan"), float("nan"))
        item.update(runtime_seconds_mean=runtime_mean, runtime_seconds_std=runtime_std, runtime_seconds_sum=sum(float(row["runtime_seconds"]) for row in success))
        summaries.append(item)
    _write_csv(output_dir / "grid_summary.csv", summaries, fields)
    candidates = [row for row in summaries if int(row["success_runs"]) == len(config.seeds) and math.isfinite(float(row["acc_mean"]))]
    if not candidates:
        raise RuntimeError("No GB-POJG-GBSC parameter combination completed every seed")
    best = min(candidates, key=lambda row: (-float(row["acc_mean"]), -float(row["nmi_mean"]), -float(row["f_measure_mean"]), float(row["runtime_seconds_mean"]), str(row["gamma"]), str(row["delta"]), str(row["sigma"])))
    best = {**best, "selection_metric": "acc_mean", "grid_runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows if row["status"] == "success")}
    _write_csv(output_dir / "best_parameter_combination.csv", [best], (*fields, "selection_metric", "grid_runtime_seconds"))
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    _write_json(output_dir / "experiment_summary.json", {"dataset": dataset_name, "algorithm": "gb_pojg_gbsc", "planned_runs": len(config.seeds) * len(parameter_combinations), "completed_rows": len(rows), "status_counts": dict(status_counts), "best_parameter_combination": best})
    return best


def _run_gb_pojg_gbsc_grid(dataset: Dataset, config: ExperimentConfig, run_id: str) -> dict[str, object]:
    params = _get_algorithm_config("gb_pojg_gbsc")
    combinations = tuple((float(gamma), float(delta), float(sigma)) for gamma in dict.fromkeys(params["gamma_values"]) for delta in dict.fromkeys(params["delta_values"]) for sigma in dict.fromkeys(params["sigma_values"]))
    output_dir = config.output_root / run_id / dataset.name / "gb_pojg_gbsc"
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir = output_dir / "labels"; labels_dir.mkdir(exist_ok=True)
    all_runs_path = output_dir / "all_runs.csv"
    previous = _read_csv(all_runs_path) if config.resume else []
    completed = {(int(row["seed"]), row["gamma"], row["delta"], row["sigma"]) for row in previous}
    _write_json(output_dir / "experiment_config.json", {"algorithm": "gb_pojg_gbsc", "dataset": dataset.name, "seeds": config.seeds, "gamma_values": params["gamma_values"], "delta_values": params["delta_values"], "sigma_values": params["sigma_values"], "preprocessing": "global_featurewise_minmax_to_0_1", "selection_rule": "maximize mean ACC across seeds", "algorithm_parameters": _algorithm_parameters("gb_pojg_gbsc", config)})
    X = minmax_scale(dataset.X)
    mode = "a" if previous else "w"; planned = len(config.seeds) * len(combinations)
    with all_runs_path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GB_POJG_GBSC_RUN_FIELDS)
        if mode == "w": writer.writeheader()
        done = len(completed)
        for seed in config.seeds:
            for gamma, delta, sigma in combinations:
                gamma_text, delta_text, sigma_text = f"{gamma:.12g}", f"{delta:.12g}", f"{sigma:.12g}"
                if (seed, gamma_text, delta_text, sigma_text) in completed: continue
                row: dict[str, object] = {field: "" for field in GB_POJG_GBSC_RUN_FIELDS}
                row.update(algorithm="gb_pojg_gbsc", dataset=dataset.name, seed=seed, gamma=gamma_text, delta=delta_text, sigma=sigma_text)
                start = time.perf_counter()
                try:
                    labels = _create_model("gb_pojg_gbsc", config, p1=None, p2=None, theta=0.0, n_clusters=dataset.n_classes, seed=seed, gamma=gamma, delta=delta, sigma=sigma).fit_predict(X.copy())
                    runtime = time.perf_counter() - start
                    metrics = evaluate_clustering(dataset.y, labels, nmi_average_method=config.nmi_average_method).as_dict()
                    prediction = labels_dir / f"seed_{seed}_gamma_{gamma_text.replace('.', 'p')}_delta_{delta_text.replace('.', 'p')}_sigma_{sigma_text.replace('.', 'p')}.npy"
                    np.save(prediction, labels)
                    row.update(status="success", runtime_seconds=runtime, prediction_path=prediction.relative_to(output_dir).as_posix(), **metrics)
                except Exception as exc:
                    row.update(status="failed", runtime_seconds=time.perf_counter() - start, error_type=type(exc).__name__, error_message=str(exc))
                    (output_dir / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                writer.writerow(row); stream.flush(); done += 1
                metric_text = f"NMI={float(row['nmi']):.4f} ACC={float(row['acc']):.4f} F-measure={float(row['f_measure']):.4f}" if row["status"] == "success" else "NMI=N/A ACC=N/A F-measure=N/A"
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{done}/{planned}] gb_pojg_gbsc {dataset.name} seed={seed} gamma={gamma_text} delta={delta_text} sigma={sigma_text} {row['status']} runtime={float(row['runtime_seconds']):.3f}s {metric_text}", flush=True)
    return _summarize_gb_pojg_gbsc(output_dir, config, combinations)


def _summarize_gbsc(
    output_dir: Path, config: ExperimentConfig, sigma_values: tuple[float, ...]
) -> dict[str, object]:
    rows = _read_csv(output_dir / "all_runs.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[row["sigma"]].append(row)
    fields = (
        "dataset", "sigma", "success_runs",
        *(name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")),
        "runtime_seconds_mean", "runtime_seconds_std", "runtime_seconds_sum",
    )
    summaries: list[dict[str, object]] = []
    dataset_name = rows[0]["dataset"] if rows else ""
    for sigma in sigma_values:
        sigma_text = f"{sigma:.12g}"
        success = grouped.get(sigma_text, [])
        item: dict[str, object] = {"dataset": dataset_name, "sigma": sigma_text, "success_runs": len(success)}
        for metric in METRICS:
            mean, std = _mean_std(success, metric) if success else (float("nan"), float("nan"))
            item[f"{metric}_mean"], item[f"{metric}_std"] = mean, std
        if success:
            runtime_mean, runtime_std = _mean_std(success, "runtime_seconds")
            runtime_sum = sum(float(row["runtime_seconds"]) for row in success)
        else:
            runtime_mean = runtime_std = float("nan")
            runtime_sum = 0.0
        item.update(runtime_seconds_mean=runtime_mean, runtime_seconds_std=runtime_std, runtime_seconds_sum=runtime_sum)
        summaries.append(item)
    _write_csv(output_dir / "grid_summary.csv", summaries, fields)
    candidates = [row for row in summaries if int(row["success_runs"]) == len(config.seeds) and math.isfinite(float(row["acc_mean"]))]
    if not candidates:
        raise RuntimeError("No GBSC sigma value completed every seed")
    best = min(candidates, key=lambda row: (-float(row["acc_mean"]), -float(row["nmi_mean"]), -float(row["f_measure_mean"]), float(row["runtime_seconds_mean"]), str(row["sigma"])))
    best = {
        **best,
        "selection_metric": "acc_mean",
        "grid_runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows if row["status"] == "success"),
    }
    _write_csv(output_dir / "best_parameter_combination.csv", [best], (*fields, "selection_metric", "grid_runtime_seconds"))
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    _write_json(output_dir / "experiment_summary.json", {
        "dataset": dataset_name, "algorithm": "gbsc",
        "planned_runs": len(config.seeds) * len(sigma_values),
        "completed_rows": len(rows), "status_counts": dict(status_counts),
        "best_parameter_combination": best,
    })
    return best


def _run_gbsc_grid(dataset: Dataset, config: ExperimentConfig, run_id: str) -> dict[str, object]:
    params = _get_algorithm_config("gbsc")
    sigma_values = tuple(dict.fromkeys(float(value) for value in params["sigma_values"]))
    output_dir = config.output_root / run_id / dataset.name / "gbsc"
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    all_runs_path = output_dir / "all_runs.csv"
    previous = _read_csv(all_runs_path) if config.resume else []
    completed = {(int(row["seed"]), row["sigma"]) for row in previous}
    _write_json(output_dir / "experiment_config.json", {
        "algorithm": "gbsc", "dataset": dataset.name, "seeds": config.seeds,
        "sigma_values": sigma_values,
        "preprocessing": "global_featurewise_minmax_to_0_1",
        "selection_rule": "maximize mean ACC across seeds",
        "algorithm_parameters": _algorithm_parameters("gbsc", config),
    })
    X = minmax_scale(dataset.X)
    mode = "a" if previous else "w"
    planned = len(config.seeds) * len(sigma_values)
    with all_runs_path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GBSC_RUN_FIELDS)
        if mode == "w":
            writer.writeheader()
        done = len(completed)
        for seed in config.seeds:
            for sigma in sigma_values:
                sigma_text = f"{sigma:.12g}"
                if (seed, sigma_text) in completed:
                    continue
                row: dict[str, object] = {field: "" for field in GBSC_RUN_FIELDS}
                row.update(algorithm="gbsc", dataset=dataset.name, seed=seed, sigma=sigma_text)
                start = time.perf_counter()
                try:
                    model = _create_model("gbsc", config, p1=None, p2=None, theta=0.0, n_clusters=dataset.n_classes, seed=seed, sigma=sigma)
                    labels = model.fit_predict(X.copy())
                    runtime = time.perf_counter() - start
                    metrics = evaluate_clustering(dataset.y, labels, nmi_average_method=config.nmi_average_method).as_dict()
                    prediction = labels_dir / f"seed_{seed}_sigma_{sigma_text.replace('.', 'p')}.npy"
                    np.save(prediction, labels)
                    row.update(status="success", runtime_seconds=runtime, prediction_path=prediction.relative_to(output_dir).as_posix(), **metrics)
                except Exception as exc:
                    row.update(status="failed", runtime_seconds=time.perf_counter() - start, error_type=type(exc).__name__, error_message=str(exc))
                    (output_dir / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                writer.writerow(row)
                stream.flush()
                done += 1
                metric_text = f"NMI={float(row['nmi']):.4f} ACC={float(row['acc']):.4f} F-measure={float(row['f_measure']):.4f}" if row["status"] == "success" else "NMI=N/A ACC=N/A F-measure=N/A"
                print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{done}/{planned}] gbsc {dataset.name} seed={seed} sigma={sigma_text} {row['status']} runtime={float(row['runtime_seconds']):.3f}s {metric_text}", flush=True)
    return _summarize_gbsc(output_dir, config, sigma_values)


def _summarize_sagbc(output_dir: Path, config: ExperimentConfig, sample_size: int) -> dict[str, object]:
    rows = _read_csv(output_dir / "all_runs.csv")
    success = [row for row in rows if row["status"] == "success"]
    fields = (
        "dataset", "sample_size", "success_runs",
        *(name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")),
        "runtime_seconds_mean", "runtime_seconds_std", "runtime_seconds_sum",
    )
    best: dict[str, object] = {
        "dataset": rows[0]["dataset"] if rows else "", "sample_size": sample_size,
        "success_runs": len(success),
    }
    for metric in METRICS:
        best[f"{metric}_mean"], best[f"{metric}_std"] = _mean_std(success, metric) if success else (float("nan"), float("nan"))
    runtime_mean, runtime_std = _mean_std(success, "runtime_seconds") if success else (float("nan"), float("nan"))
    best.update(runtime_seconds_mean=runtime_mean, runtime_seconds_std=runtime_std, runtime_seconds_sum=sum(float(row["runtime_seconds"]) for row in success))
    _write_csv(output_dir / "grid_summary.csv", [best], fields)
    best = {**best, "selection_metric": "not_applicable_single_source_configuration", "grid_runtime_seconds": sum(float(row["runtime_seconds"]) for row in success)}
    _write_csv(output_dir / "best_parameter_combination.csv", [best], (*fields, "selection_metric", "grid_runtime_seconds"))
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    _write_json(output_dir / "experiment_summary.json", {
        "dataset": best["dataset"], "algorithm": "sagbc", "planned_runs": len(config.seeds),
        "completed_rows": len(rows), "status_counts": dict(status_counts), "best_parameter_combination": best,
    })
    return best


def _run_sagbc(dataset: Dataset, config: ExperimentConfig, run_id: str) -> dict[str, object]:
    requested = int(_get_algorithm_config("sagbc")["sample_size"])
    sample_size = min(requested, dataset.n_samples)
    output_dir = config.output_root / run_id / dataset.name / "sagbc"
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir, all_runs_path = output_dir / "labels", output_dir / "all_runs.csv"
    labels_dir.mkdir(exist_ok=True)
    previous = _read_csv(all_runs_path) if config.resume else []
    completed = {int(row["seed"]) for row in previous}
    _write_json(output_dir / "experiment_config.json", {
        "algorithm": "sagbc", "dataset": dataset.name, "seeds": config.seeds,
        "sample_size_requested": requested, "sample_size_used": sample_size,
        "preprocessing": "none_official_source", "selection_rule": "not_applicable_single_source_configuration",
        "algorithm_parameters": _algorithm_parameters("sagbc", config),
    })
    with all_runs_path.open("a" if previous else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAGBC_RUN_FIELDS)
        if not previous:
            writer.writeheader()
        for done, seed in enumerate(config.seeds, start=1):
            if seed in completed:
                continue
            row: dict[str, object] = {field: "" for field in SAGBC_RUN_FIELDS}
            row.update(algorithm="sagbc", dataset=dataset.name, seed=seed, sample_size=sample_size)
            start = time.perf_counter()
            try:
                labels = _create_model("sagbc", config, p1=None, p2=None, theta=0.0, n_clusters=dataset.n_classes, seed=seed, sagbc_sample_size=sample_size).fit_predict(dataset.X.copy())
                runtime = time.perf_counter() - start
                prediction = labels_dir / f"seed_{seed}.npy"
                np.save(prediction, labels)
                row.update(status="success", runtime_seconds=runtime, prediction_path=prediction.relative_to(output_dir).as_posix(), **evaluate_clustering(dataset.y, labels, nmi_average_method=config.nmi_average_method).as_dict())
            except Exception as exc:
                row.update(status="failed", runtime_seconds=time.perf_counter() - start, error_type=type(exc).__name__, error_message=str(exc))
                (output_dir / "last_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            writer.writerow(row)
            stream.flush()
            metric_text = f"NMI={float(row['nmi']):.4f} ACC={float(row['acc']):.4f} F-measure={float(row['f_measure']):.4f}" if row["status"] == "success" else "NMI=N/A ACC=N/A F-measure=N/A"
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [{done}/{len(config.seeds)}] sagbc {dataset.name} seed={seed} sample_size={sample_size} {row['status']} runtime={float(row['runtime_seconds']):.3f}s {metric_text}", flush=True)
    return _summarize_sagbc(output_dir, config, sample_size)


def _run_gbct(dataset: Dataset, config: ExperimentConfig, run_id: str) -> dict[str, object]:
    output = config.output_root / run_id / dataset.name / "gbct"
    if output.exists() and not config.resume: raise FileExistsError(f"Result directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=config.resume); labels_dir = output / "labels"; labels_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        row = {field: "" for field in GBCT_RUN_FIELDS}; row.update(algorithm="gbct", dataset=dataset.name, seed=seed, n_clusters=dataset.n_classes)
        start = time.perf_counter()
        try:
            labels = _create_model("gbct", config, p1=None, p2=None, theta=0.0, n_clusters=dataset.n_classes, seed=seed).fit_predict(minmax_scale(dataset.X))
            path = labels_dir / f"seed_{seed}.npy"; np.save(path, labels)
            row.update(status="success", runtime_seconds=time.perf_counter()-start, prediction_path=path.relative_to(output).as_posix(), **evaluate_clustering(dataset.y, labels, nmi_average_method=config.nmi_average_method).as_dict())
        except Exception as exc: row.update(status="failed", runtime_seconds=time.perf_counter()-start, error_type=type(exc).__name__, error_message=str(exc))
        rows.append(row)
    _write_csv(output / "all_runs.csv", rows, GBCT_RUN_FIELDS); success = [row for row in rows if row["status"] == "success"]
    best: dict[str, object] = {"dataset": dataset.name, "n_clusters": dataset.n_classes, "success_runs": len(success)}
    for metric in METRICS: best[f"{metric}_mean"], best[f"{metric}_std"] = _mean_std(success, metric) if success else (float("nan"), float("nan"))
    best.update(runtime_seconds_mean=_mean_std(success,"runtime_seconds")[0] if success else float("nan"), runtime_seconds_std=_mean_std(success,"runtime_seconds")[1] if success else float("nan"), runtime_seconds_sum=sum(float(row["runtime_seconds"]) for row in success))
    _write_csv(output / "grid_summary.csv", [best], tuple(best)); _write_csv(output / "best_parameter_combination.csv", [best], tuple(best)); _write_json(output / "experiment_summary.json", {"dataset":dataset.name,"algorithm":"gbct","planned_runs":len(config.seeds),"completed_rows":len(rows)})
    return best


def _run_algorithm_grid(
    dataset: Dataset,
    config: ExperimentConfig,
    run_id: str,
    algorithm: str,
) -> dict[str, object]:
    if algorithm == "gbct": return _run_gbct(dataset, config, run_id)
    if algorithm == "sagbc":
        return _run_sagbc(dataset, config, run_id)
    if algorithm == "gbsc":
        return _run_gbsc_grid(dataset, config, run_id)
    if algorithm == "gb_pojg_gbdpc":
        return _run_gb_pojg_gbdpc_grid(dataset, config, run_id)
    if algorithm == "gb_pojg_gbsc":
        return _run_gb_pojg_gbsc_grid(dataset, config, run_id)
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
    redundancy_beta_settings = _resolve_redundancy_beta_settings(algorithm)
    parameter_combinations = tuple(
        (
            p1,
            p2,
            pdmf_neighbors,
            graph_neighbors,
            similarity_lambda,
            stability_delta,
            redundancy_beta,
        )
        for p1, p2 in parameter_pairs
        for pdmf_neighbors in pdmf_neighbor_settings
        for graph_neighbors in graph_neighbor_settings
        for similarity_lambda in similarity_lambda_settings
        for stability_delta in stability_delta_settings
        for redundancy_beta in redundancy_beta_settings
    )

    output_dir = config.output_root / run_id / dataset.name / algorithm
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    all_runs_path = output_dir / "all_runs.csv"
    previous = _read_csv(all_runs_path) if config.resume else []
    if algorithm == "my_v2":
        completed_fields = (
            "stability_delta",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
        )
    elif algorithm == "my_v3":
        completed_fields = (
            "p1",
            "p2",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
            "redundancy_beta",
        )
    else:
        completed_fields = (
            "p1",
            "p2",
            "pdmf_neighbors",
            "graph_neighbors",
            "pdmf_similarity_lambda",
            "theta",
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
    if algorithm == "my_v3":
        selection_parameters["redundancy_beta_values"] = tuple(redundancy_beta_settings)

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
    run_fields = (
        V2_RUN_FIELDS if algorithm == "my_v2"
        else V3_RUN_FIELDS if algorithm == "my_v3"
        else RUN_FIELDS
    )
    pseudo_labels_by_seed: dict[int, np.ndarray] = {}
    global_selection_cache: dict[tuple[str, ...], tuple[object, ...]] = {}
    global_stability_curve_caches: dict[
        tuple[str, ...], dict[str, object]
    ] = {}
    root_feature_ranking_caches: dict[tuple[str, ...], dict[str, object]] = {}
    local_feature_selection_caches: dict[
        tuple[str, ...], dict[tuple[object, ...], np.ndarray]
    ] = {}
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
                redundancy_beta,
            ) in parameter_combinations:
                pdmf_neighbors_text = _format_pdmf_neighbors(pdmf_neighbors)
                graph_neighbors_text = _format_graph_neighbors(graph_neighbors)
                similarity_lambda_text = _format_similarity_lambda(similarity_lambda)
                stability_delta_text = _format_stability_delta(stability_delta)
                redundancy_beta_text = "" if redundancy_beta is None else f"{float(redundancy_beta):.12g}"
                parameter_output = ""
                if algorithm in {"my_v0", "my_v1", "my_v2", "my_v3", "my_v4"}:
                    parameter_output += f"pdmf_neighbors={pdmf_neighbors_text} "
                if algorithm in {"my_v1", "my_v2", "my_v3", "my_v4"}:
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
                    curve_cache_key = (
                        pdmf_neighbors_text,
                        graph_neighbors_text,
                        similarity_lambda_text,
                    )
                elif algorithm == "my_v3":
                    parameter_output += f"beta={redundancy_beta_text} "
                    global_cache_key = (str(p1), pdmf_neighbors_text, graph_neighbors_text, similarity_lambda_text, redundancy_beta_text)
                    root_cache_key = global_cache_key
                    curve_cache_key = ()
                elif algorithm == "my_v4":
                    # V4 全局评分依赖伪标签，缓存必须按 seed 隔离。
                    global_cache_key = (
                        str(seed),
                        str(p1),
                        pdmf_neighbors_text,
                        graph_neighbors_text,
                        similarity_lambda_text,
                    )
                    root_cache_key = global_cache_key
                    curve_cache_key = ()
                else:
                    global_cache_key = (str(p1), pdmf_neighbors_text)
                    root_cache_key = (
                        str(p1),
                        pdmf_neighbors_text,
                        graph_neighbors_text,
                        similarity_lambda_text,
                    )
                    curve_cache_key = ()
                global_stability_curve_cache = (
                    global_stability_curve_caches.setdefault(curve_cache_key, {})
                    if algorithm == "my_v2"
                    and len(stability_delta_settings) > 1
                    else None
                )
                root_feature_ranking_cache = (
                    root_feature_ranking_caches.setdefault(root_cache_key, {})
                    if algorithm in {"my_v1", "my_v2", "my_v3", "my_v4"}
                    else None
                )
                local_feature_selection_cache = (
                    local_feature_selection_caches.setdefault(root_cache_key, {})
                    if algorithm == "my_v2"
                    else None
                )
                for theta in theta_values:
                    if algorithm == "my_v2":
                        key = (
                            seed,
                            stability_delta_text,
                            pdmf_neighbors_text,
                            graph_neighbors_text,
                            similarity_lambda_text,
                            f"{theta:.2f}",
                        )
                    elif algorithm == "my_v3":
                        key = (
                            seed,
                            str(p1),
                            str(p2),
                            pdmf_neighbors_text,
                            graph_neighbors_text,
                            similarity_lambda_text,
                            f"{theta:.2f}",
                            redundancy_beta_text,
                        )
                    else:
                        key = (
                            seed,
                            str(p1),
                            str(p2),
                            pdmf_neighbors_text,
                            graph_neighbors_text,
                            similarity_lambda_text,
                            f"{theta:.2f}",
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
                    if algorithm == "my_v3":
                        row.update(
                            redundancy_beta=redundancy_beta_text,
                            fusion_alpha_mode=algorithm_config["fusion_alpha_mode"],
                            mutual_knn=algorithm_config["mutual_knn"],
                            self_tuning_graph=algorithm_config["self_tuning_graph"],
                        )
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
                            redundancy_beta=redundancy_beta,
                            precomputed_pseudo_labels=pseudo_labels_by_seed.get(seed),
                            precomputed_global_selection=global_selection_cache.get(
                                global_cache_key
                            ),
                            global_stability_curve_cache=(
                                global_stability_curve_cache
                            ),
                            root_feature_ranking_cache=root_feature_ranking_cache,
                            local_feature_selection_cache=(
                                local_feature_selection_cache
                            ),
                        )
                        labels = model.fit_predict(X.copy())
                        if seed not in pseudo_labels_by_seed:
                            pseudo_labels_by_seed[seed] = model.pseudo_labels_.copy()
                        if global_cache_key not in global_selection_cache:
                            if algorithm in {"my_v0", "my_v1", "my_v3", "my_v4"}:
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
                        elif algorithm == "my_v3":
                            prediction_name = (
                                f"seed_{seed}_p1_{p1}_p2_{p2}{pdmf_suffix}"
                                f"{graph_suffix}{lambda_suffix}_beta_{redundancy_beta_text.replace('.', 'p')}"
                                f"_theta_{theta:.2f}.npy"
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
    supported_algorithms = {"plgb_fsc", "my_v0", "my_v1", "my_v2", "my_v3", "my_v4", "gb_pojg_gbdpc", "gb_pojg_gbsc", "gbsc", "sagbc", "gbct"}
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
            best_params = (
                {"gamma": float(best["gamma"]), "delta": float(best["delta"])}
                if algorithm == "gb_pojg_gbdpc"
                else {"gamma": float(best["gamma"]), "delta": float(best["delta"]), "sigma": float(best["sigma"])}
                if algorithm == "gb_pojg_gbsc"
                else {"sigma": float(best["sigma"])}
                if algorithm == "gbsc"
                else {"sample_size": int(best["sample_size"])}
                if algorithm == "sagbc"
                else {"n_clusters": int(best["n_clusters"])}
                if algorithm == "gbct"
                else {"theta": float(best["theta"])}
            )
            if algorithm not in {"my_v2", "gb_pojg_gbdpc", "gb_pojg_gbsc", "gbsc", "sagbc", "gbct"}:
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
                    ball_parallel_jobs=algorithm_config["ball_parallel_jobs"],
                    graph_neighbors=best["graph_neighbors"],
                    pdmf_similarity_lambda=best["pdmf_similarity_lambda"],
                )
            if algorithm == "my_v3":
                best_params.update(
                    pdmf_neighbors=best["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                    graph_neighbors=best["graph_neighbors"],
                    pdmf_similarity_lambda=best["pdmf_similarity_lambda"],
                    redundancy_beta=float(best["redundancy_beta"]),
                    fusion_alpha_mode=algorithm_config["fusion_alpha_mode"],
                    mutual_knn=algorithm_config["mutual_knn"],
                    self_tuning_graph=algorithm_config["self_tuning_graph"],
                )
            if algorithm == "my_v4":
                best_params.update(
                    pdmf_neighbors=best["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                    graph_neighbors=best["graph_neighbors"],
                    pdmf_similarity_lambda=best["pdmf_similarity_lambda"],
                    confidence_weighting="automatic",
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

# python3 run.py > only_plgb_fsc_with_COIL20.log 2>&1 &
