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

from algorithms.plgb_fsc import PLGBFSC, PLGBFSCConfig
from config import DATASETS, EXPERIMENT, ExperimentConfig
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
    p1: int,
    valid_p2: tuple[int, ...],
) -> dict[str, object]:
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
        for theta in config.theta_values:
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
        "algorithm": "plgb_fsc",
        "planned_runs": len(valid_p2) * len(config.theta_values) * len(config.seeds),
        "completed_rows": len(rows),
        "status_counts": dict(status_counts),
        "best_parameter_combination": best,
    }
    _write_json(output_dir / "experiment_summary.json", result)
    return best


def _run_plgb_fsc(
    dataset: Dataset,
    config: ExperimentConfig,
    run_id: str,
) -> dict[str, object]:
    p1 = int(math.ceil(config.p1_ratio * dataset.n_features))
    valid_p2 = tuple(p2 for p2 in config.p2_values if p2 < p1)
    if not valid_p2:
        raise ValueError(f"{dataset.name}: no p2 value is smaller than p1={p1}")

    output_dir = config.output_root / dataset.name / run_id / "plgb_fsc"
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
            "algorithm": "plgb_fsc",
            "dataset": dataset.name,
            "seeds": config.seeds,
            "p1_rule": f"ceil({config.p1_ratio} * n_features)",
            "p1": p1,
            "p2_values": valid_p2,
            "theta_values": config.theta_values,
            "preprocessing": "global_featurewise_minmax_to_0_1",
            "stop_rule": "pseudo_purity >= theta and ball_size < 8",
            "selection_rule": "maximize mean ACC across seeds",
        },
    )

    mode = "a" if previous else "w"
    X = minmax_scale(dataset.X)
    planned = len(config.seeds) * len(valid_p2) * len(config.theta_values)
    with all_runs_path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        if mode == "w":
            writer.writeheader()
        done = len(completed)
        for seed in config.seeds:
            for p2 in valid_p2:
                for theta in config.theta_values:
                    key = (seed, p2, f"{theta:.2f}")
                    if key in completed:
                        continue
                    row: dict[str, object] = {field: "" for field in RUN_FIELDS}
                    row.update(
                        algorithm="plgb_fsc",
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
                        model = PLGBFSC(
                            PLGBFSCConfig(p1=p1, p2=p2, purity=theta),
                            n_clusters=dataset.n_classes,
                            random_state=seed,
                        )
                        labels = model.fit_predict(X.copy())
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
                        f"[{done}/{planned}] {dataset.name} seed={seed} "
                        f"p2={p2} theta={theta:.2f} {row['status']}",
                        flush=True,
                    )
    return _summarize(output_dir, config, p1, valid_p2)


def main() -> int:
    if not EXPERIMENT.datasets:
        raise ValueError("At least one dataset is required")
    if not EXPERIMENT.seeds:
        raise ValueError("At least one seed is required")
    if not 0.0 < EXPERIMENT.p1_ratio <= 1.0:
        raise ValueError("p1_ratio must be in (0, 1]")
    if not EXPERIMENT.p2_values or any(p2 <= 0 for p2 in EXPERIMENT.p2_values):
        raise ValueError("p2_values must contain positive integers")
    if not EXPERIMENT.theta_values or any(
        not 0.0 <= theta <= 1.0 for theta in EXPERIMENT.theta_values
    ):
        raise ValueError("theta_values must be in [0, 1]")
    unknown = sorted(set(EXPERIMENT.datasets).difference(DATASETS))
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}")
    if len(EXPERIMENT.seeds) != len(set(EXPERIMENT.seeds)):
        raise ValueError("Seeds must be unique")
    run_id = EXPERIMENT.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    best_rows = []
    for name in EXPERIMENT.datasets:
        dataset = load_dataset(DATASETS[name])
        best = _run_plgb_fsc(dataset, EXPERIMENT, run_id)
        best_rows.append(
            _benchmark_summary_row(
                "plgb_fsc",
                best,
                best_params={
                    "p1": int(best["p1"]),
                    "p2": int(best["p2"]),
                    "theta": float(best["theta"]),
                },
            )
        )
    summary_path = EXPERIMENT.output_root / f"benchmark_summary_{run_id}.csv"
    fields = tuple(dict.fromkeys(field for row in best_rows for field in row))
    _write_csv(summary_path, best_rows, fields)
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# cd D:\Projects\zhengchuang\paper_team\liqiu\algorithm\unified_benchmark
# ..\python\.venv\Scripts\python.exe .\run.py
