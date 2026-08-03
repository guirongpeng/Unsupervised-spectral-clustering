from __future__ import annotations

"""
Only edit this file when configuring an experiment.

| 值                | NMI 分母 | 说明 |
|------------------ |-----------|-----------|
| `"min"`           | `min(H真值, H预测)` | 分数通常偏高 |
| `"geometric"`     | `sqrt(H真值 × H预测)` | 几何平均，当前默认 |
| `"arithmetic"`    | `(H真值 + H预测) / 2` | 算术平均，文献中常见 |
| `"max"`           | `max(H真值, H预测)` | 分数通常较保守 |

run_id + resume:
    1. 某个 seed 失败后，失败记录仍写入 all_runs.csv。下次续跑时，该 (seed, p2, theta) 被视为已记录，所以不会重新运行。
    2. 对于部分 seed 成功的参数组合，程序仍会用成功 seed 计算均值和标准差，并写入 grid_summary.csv。

"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
STANDARDIZED_DATA_ROOT = DATA_ROOT / "standardized"

# 每个算法独立维护实验参数；即使当前取值相同，也不要相互复用。
PLGB_FSC_PARAMS = {
    "p1_ratio": 0.75,
    "p2_values": tuple(range(4, 56, 4)),        # (50, 201, 10)
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),# (70, 100, 5):(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
}

MY_V0_PARAMS = {
    "p1_ratio": 0.75,
    "p2_values": tuple(range(4, 56, 4)),
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    "pdmf_neighbors": 5,
    "pdmf_epsilon": 1e-8,
}

MY_V1_PARAMS = {
    "p1_ratio": 0.75,
    "p2_values": tuple(range(50, 201, 10)),
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    "pdmf_neighbors": 5,
    "pdmf_epsilon": 1e-8,
    "graph_neighbors": 5,
    "pdmf_similarity_lambda": 0.5,
}

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


@dataclass(frozen=True)
class ExperimentConfig:
    algorithms: tuple[str, ...] = ( "my_v1",)  # 指定运行算法:("plgb_fsc", "my_v0","my_v1")
    datasets: tuple[str, ...] = ("COIL20","ORL","SuCancer","USPS","Yale","warpPIE10P","GLIOMA","TOX_171","ALLAML",)
                                # "PenDigits","Letter","Covertype")    # 指定运行数据集名
    seeds: tuple[int, ...] = (1,2)         # 指定运行种子
    nmi_average_method: str = "geometric"  # 指定运行NMI平均方法
    output_root: Path = ROOT / "results"
    run_id: str | None = None              # None: 自动生成，指定：使用指定ID
    resume: bool = False                   # False: 重新运行，True: 覆盖已存在的结果


DATASETS = {
    name: DatasetConfig(name, STANDARDIZED_DATA_ROOT / f"{name}.npz")
    for name in (
        "COIL20",
        "ORL",
        "SuCancer",
        "USPS",
        "Yale",
        "warpPIE10P",
        "GLIOMA",
        "TOX_171",
        "ALLAML",
        "D3",
        "T4",
        "E6",
        "PenDigits",
        "Letter",
        "Covertype",
    )
}

# 当前先验证 SuCancer，完整遍历参数网格并运行 3 个 seed。
EXPERIMENT = ExperimentConfig()
