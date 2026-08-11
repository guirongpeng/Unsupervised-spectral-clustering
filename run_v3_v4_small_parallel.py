from __future__ import annotations

"""并行运行原论文 9 个数据集上的 MY-V3 与 MY-V4。

每个数据集在独立进程中运行，并写入独立日志；不修改 config.py 和 run.py。
"""

import argparse
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_DATASETS = (

    "COIL20",
    "USPS",
    "TOX_171",
    "GLIOMA",
    "warpPIE10P",
    "Yale",
    "ALLAML",
    # # "COIL20",
    # "ORL",
    # "SuCancer",
    # # "USPS",
    # "Yale",
    # "warpPIE10P",
    # "GLIOMA",
    # "TOX_171",
    # "ALLAML",
)


def _select_datasets(names: list[str] | None) -> list[str]:
    # 延迟导入，避免子进程启动时重复加载 benchmark 模块。
    from config import DATASETS

    selected = list(PAPER_DATASETS if names is None else names)
    unknown = sorted(set(selected).difference(DATASETS))
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}")
    return selected


def _run_one(dataset: str, run_id: str, log_dir: Path) -> tuple[str, bool, str]:
    log_path = log_dir / f"{dataset}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        import run as benchmark_run
        from config import EXPERIMENT

        # 每个进程只处理一个数据集，且使用独立 run_id，避免并行写 benchmark_summary.csv。
        benchmark_run.EXPERIMENT = replace(
            EXPERIMENT,
            algorithms=("my_v3", "my_v4"),
            datasets=(dataset,),
            run_id=f"{run_id}_{dataset}",
        )
        with log_path.open("w", encoding="utf-8", buffering=1) as stream:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                print(f"dataset={dataset} algorithms=my_v3,my_v4 pid={os.getpid()}", flush=True)
                benchmark_run.main()
        return dataset, True, str(log_path)
    except Exception as exc:  # noqa: BLE001 - 错误已记录到对应日志
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\nFAILED: {type(exc).__name__}: {exc}\n")
        return dataset, False, str(log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", help="覆盖默认的原论文 9 个数据集")
    parser.add_argument("--workers", type=int, default=len(PAPER_DATASETS), help="并行进程数")
    parser.add_argument("--run-id", default=None, help="公共运行标识；默认使用时间戳")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    datasets = _select_datasets(args.datasets)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "results" / "parallel_logs" / run_id
    print(f"Selected datasets: {', '.join(datasets)}")
    print("Algorithms: my_v3, my_v4")
    print(f"Logs: {log_dir}")

    with ProcessPoolExecutor(max_workers=min(args.workers, len(datasets))) as pool:
        futures = [pool.submit(_run_one, name, run_id, log_dir) for name in datasets]
        for future in as_completed(futures):
            dataset, success, log_path = future.result()
            print(f"[{dataset}] {'success' if success else 'failed'} log={log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    # ..\python\.venv\Scripts\python.exe .\run_v3_v4_small_parallel.py --workers 2
