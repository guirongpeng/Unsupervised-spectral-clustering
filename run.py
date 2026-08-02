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
from algorithms.plgb_fsc import PLGBFSC, PLGBFSCConfig
from config import (
    DATASETS,
    EXPERIMENT,
    MY_V0_PARAMS,
    MY_V1_PARAMS,
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
    "theta",
    "status",
    *METRICS,
    "runtime_seconds",
    "prediction_path",
    "error_type",
    "error_message",
)


def _get_algorithm_config(algorithm: str) -> dict[str, object]:
    if algorithm == "plgb_fsc":
        return PLGB_FSC_PARAMS
    if algorithm == "my_v0":
        return MY_V0_PARAMS
    if algorithm == "my_v1":
        return MY_V1_PARAMS
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _validate_algorithm_config(algorithm: str) -> None:
    params = _get_algorithm_config(algorithm)
    p1_ratio = float(params["p1_ratio"])
    p2_values = tuple(params["p2_values"])
    theta_values = tuple(params["theta_values"])
    if not 0.0 < p1_ratio <= 1.0:
        raise ValueError(f"{algorithm}: p1_ratio must be in (0, 1]")
    if not p2_values or any(p2 <= 0 for p2 in p2_values):
        raise ValueError(f"{algorithm}: p2_values must contain positive integers")
    if not theta_values or any(not 0.0 <= theta <= 1.0 for theta in theta_values):
        raise ValueError(f"{algorithm}: theta_values must be in [0, 1]")
    if algorithm in {"my_v0", "my_v1"}:
        if int(params["pdmf_neighbors"]) < 1:
            raise ValueError(f"{algorithm}: pdmf_neighbors must be at least 1")
        epsilon = float(params["pdmf_epsilon"])
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"{algorithm}: pdmf_epsilon must be positive")
    if algorithm == "my_v1":
        if int(params["graph_neighbors"]) < 1:
            raise ValueError("my_v1: graph_neighbors must be at least 1")
        similarity_lambda = float(params["pdmf_similarity_lambda"])
        if not math.isfinite(similarity_lambda) or not 0 < similarity_lambda < 1:
            raise ValueError("my_v1: pdmf_similarity_lambda must be in (0, 1)")


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
    p1: int,
    valid_p2: tuple[int, ...],
) -> dict[str, object]:
    algorithm_config = _get_algorithm_config(algorithm)
    theta_values = tuple(algorithm_config["theta_values"])
    rows = _read_csv(output_dir / "all_runs.csv")
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "success":
            grouped[(int(row["p2"]), f"{float(row['theta']):.2f}")].append(row)

    fields = (
        "dataset",
        "p1",
        "p2",
        "theta",
        "success_runs",
        *(name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std")),
        "runtime_seconds_mean",
        "runtime_seconds_std",
        "runtime_seconds_sum",
    )
    summaries: list[dict[str, object]] = []
    dataset_name = rows[0]["dataset"] if rows else ""
    for p2 in valid_p2:
        for theta in theta_values:
            success = grouped.get((p2, f"{theta:.2f}"), [])
            item: dict[str, object] = {
                "dataset": dataset_name,
                "p1": p1,
                "p2": p2,
                "theta": f"{theta:.2f}",
                "success_runs": len(success),
            }
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
            int(row["p2"]),
            float(row["theta"]),
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
        "planned_runs": len(valid_p2) * len(theta_values) * len(config.seeds),
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
    p1: int,
    p2: int,
    theta: float,
    n_clusters: int,
    seed: int,
    precomputed_pseudo_labels: np.ndarray | None = None,
    precomputed_global_selection: tuple[np.ndarray, np.ndarray] | None = None,
    root_feature_ranking_cache: dict[str, np.ndarray] | None = None,
) -> PLGBFSC | MYV0 | MYV1:
    algorithm_config = _get_algorithm_config(algorithm)
    if algorithm == "plgb_fsc":
        return PLGBFSC(
            PLGBFSCConfig(p1=p1, p2=p2, purity=theta),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
        )
    if algorithm == "my_v0":
        return MYV0(
            MYV0Config(
                p1=p1,
                p2=p2,
                purity=theta,
                pdmf_neighbors=int(algorithm_config["pdmf_neighbors"]),
                pdmf_epsilon=float(algorithm_config["pdmf_epsilon"]),
            ),
            n_clusters=n_clusters,
            random_state=seed,
            precomputed_pseudo_labels=precomputed_pseudo_labels,
            precomputed_global_selection=precomputed_global_selection,
        )
    if algorithm == "my_v1":
        return MYV1(
            MYV1Config(
                p1=p1,
                p2=p2,
                purity=theta,
                pdmf_neighbors=int(algorithm_config["pdmf_neighbors"]),
                pdmf_epsilon=float(algorithm_config["pdmf_epsilon"]),
                graph_neighbors=int(algorithm_config["graph_neighbors"]),
                pdmf_similarity_lambda=float(
                    algorithm_config["pdmf_similarity_lambda"]
                ),
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
            "pdmf_neighbors": algorithm_config["pdmf_neighbors"],
            "pdmf_epsilon": algorithm_config["pdmf_epsilon"],
        }
    if algorithm == "my_v1":
        return {
            "global_selection": "Gaussian-PDMF inner-significance ranking",
            "local_selection": "equal-ranked local entropy and sparse-graph importance",
            "pdmf_neighbors": algorithm_config["pdmf_neighbors"],
            "pdmf_epsilon": algorithm_config["pdmf_epsilon"],
            "graph_neighbors": algorithm_config["graph_neighbors"],
            "pdmf_similarity_lambda": algorithm_config[
                "pdmf_similarity_lambda"
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
    p1_ratio = float(algorithm_config["p1_ratio"])
    p2_values = tuple(algorithm_config["p2_values"])
    theta_values = tuple(algorithm_config["theta_values"])
    p1 = int(math.ceil(p1_ratio * dataset.n_features))
    valid_p2 = tuple(p2 for p2 in p2_values if p2 < p1)
    if not valid_p2:
        raise ValueError(f"{dataset.name}: no p2 value is smaller than p1={p1}")

    output_dir = config.output_root / run_id / dataset.name / algorithm
    if output_dir.exists() and not config.resume:
        raise FileExistsError(f"Result directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=config.resume)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    all_runs_path = output_dir / "all_runs.csv"
    previous = _read_csv(all_runs_path) if config.resume else []
    completed = {
        (int(row["seed"]), int(row["p2"]), f"{float(row['theta']):.2f}")
        for row in previous
    }

    _write_json(
        output_dir / "experiment_config.json",
        {
            "algorithm": algorithm,
            "dataset": dataset.name,
            "seeds": config.seeds,
            "p1_rule": f"ceil({p1_ratio} * n_features)",
            "p1": p1,
            "p2_values": valid_p2,
            "theta_values": theta_values,
            "preprocessing": "global_featurewise_minmax_to_0_1",
            "stop_rule": "pseudo_purity >= theta and ball_size < 8",
            "selection_rule": "maximize mean ACC across seeds",
            "algorithm_parameters": _algorithm_parameters(algorithm, config),
        },
    )

    mode = "a" if previous else "w"
    X = minmax_scale(dataset.X)
    pseudo_labels_by_seed: dict[int, np.ndarray] = {}
    global_selection_cache: tuple[np.ndarray, np.ndarray] | None = None
    root_feature_ranking_cache = {} if algorithm == "my_v1" else None
    planned = len(config.seeds) * len(valid_p2) * len(theta_values)
    with all_runs_path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        if mode == "w":
            writer.writeheader()
        done = len(completed)
        for seed in config.seeds:
            for p2 in valid_p2:
                for theta in theta_values:
                    key = (seed, p2, f"{theta:.2f}")
                    if key in completed:
                        continue
                    row: dict[str, object] = {field: "" for field in RUN_FIELDS}
                    row.update(
                        algorithm=algorithm,
                        dataset=dataset.name,
                        seed=seed,
                        p1=p1,
                        p2=p2,
                        theta=f"{theta:.2f}",
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
                            precomputed_pseudo_labels=pseudo_labels_by_seed.get(seed),
                            precomputed_global_selection=global_selection_cache,
                            root_feature_ranking_cache=root_feature_ranking_cache,
                        )
                        labels = model.fit_predict(X.copy())
                        if seed not in pseudo_labels_by_seed:
                            pseudo_labels_by_seed[seed] = model.pseudo_labels_.copy()
                        if (
                            algorithm in {"my_v0", "my_v1"}
                            and global_selection_cache is None
                        ):
                            global_selection_cache = (
                                model.selected_feature_indices_.copy(),
                                model.attribute_scores_.copy(),
                            )
                        runtime = time.perf_counter() - start
                        metrics = evaluate_clustering(
                            dataset.y,
                            labels,
                            nmi_average_method=config.nmi_average_method,
                        ).as_dict()
                        prediction = labels_dir / (
                            f"seed_{seed}_p2_{p2}_theta_{theta:.2f}.npy"
                        )
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
                    print(
                        f"[{done}/{planned}] {algorithm} {dataset.name} seed={seed} "
                        f"p2={p2} theta={theta:.2f} {row['status']} "
                        f"runtime={float(row['runtime_seconds']):.3f}s",
                        flush=True,
                    )
    return _summarize(output_dir, config, algorithm, p1, valid_p2)


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
    supported_algorithms = {"plgb_fsc", "my_v0", "my_v1"}
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
            best_params = {
                "p1": int(best["p1"]),
                "p2": int(best["p2"]),
                "theta": float(best["theta"]),
            }
            if algorithm == "my_v0":
                best_params.update(
                    pdmf_neighbors=algorithm_config["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                )
            if algorithm == "my_v1":
                best_params.update(
                    pdmf_neighbors=algorithm_config["pdmf_neighbors"],
                    pdmf_epsilon=algorithm_config["pdmf_epsilon"],
                    graph_neighbors=algorithm_config["graph_neighbors"],
                    pdmf_similarity_lambda=algorithm_config[
                        "pdmf_similarity_lambda"
                    ],
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
# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > run-0801-2143.log 2>&1" -WindowStyle Hidden
