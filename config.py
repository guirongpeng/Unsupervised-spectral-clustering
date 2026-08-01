from __future__ import annotations

"""
Only edit this file when configuring an experiment.

| 值                | NMI 分母 | 说明 |
|------------------ |-----------|-----------|
| `"min"`           | `min(H真值, H预测)` | 分数通常偏高 |
| `"geometric"`     | `sqrt(H真值 × H预测)` | 几何平均，当前默认 |
| `"arithmetic"`    | `(H真值 + H预测) / 2` | 算术平均，文献中常见 |
| `"max"`           | `max(H真值, H预测)` | 分数通常较保守 |
"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "unified_benchmark" / "data"

print(ROOT.parent)

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    x_path: Path
    y_path: Path


@dataclass(frozen=True)
class ExperimentConfig:
    datasets: tuple[str, ...] = ("COIL20",)    # 指定运行数据集名
    seeds: tuple[int, ...] = (1,)         # 指定运行种子
    p1_ratio: float = 0.78125                  # 0.75d
    p2_values: tuple[int, ...] = tuple(range(50, 201, 10))           # 指定运行p2参数：(50, 201, 10)
    theta_values: tuple[float, ...] = tuple([i / 100 for i in range(70, 100, 5)])    # 指定运行theta参数：(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    nmi_average_method: str = "geometric"  # 指定运行NMI平均方法
    output_root: Path = ROOT / "results"
    run_id: str | None = None
    resume: bool = False


DATASETS = {
    "COIL20": DatasetConfig(
        "COIL20",
        DATA_ROOT / "COIL20_X_1_0_.csv",
        DATA_ROOT / "COIL20_Y_1_0_.csv",
    ),
    "ORL": DatasetConfig(
        "ORL",
        DATA_ROOT / "ORL_X_1_0_.csv",
        DATA_ROOT / "ORL_Y_1_0_.csv",
    ),
    "SuCancer": DatasetConfig(
        "SuCancer",
        DATA_ROOT / "SuCancer_X_1_0_.csv",
        DATA_ROOT / "SuCancer_Y_1_0_.csv",
    ),
}

# 当前先验证 SuCancer，完整遍历参数网格并运行 3 个 seed。
EXPERIMENT = ExperimentConfig()

