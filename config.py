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
    1. 某个 seed 失败后，失败记录仍写入 all_runs.csv。MY-V0 使用 (seed, p1, p2, pdmf_neighbors, theta) 作为断点键。
    2. 对于部分 seed 成功的参数组合，程序仍会用成功 seed 计算均值和标准差，并写入 grid_summary.csv。

"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
STANDARDIZED_DATA_ROOT = DATA_ROOT / "standardized"

# 每个算法独立维护实验参数；即使当前取值相同，也不要相互复用。
PLGB_FSC_PARAMS = {
    # 组件2：全局保留属性数。counts 有效范围 [2, d]；ratios 按 ceil(d * ratio) 换算，范围 (0, 1]。
    "p1_counts": (),
    "p1_ratios": (0.75,),                                   # 默认比例0.75

    "p2_values": tuple(range(4, 56, 4)),          # 默认属性区间(50, 201, 10)
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
                                                            # 默认阈值(70, 100, 5):(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
}

MY_V0_PARAMS = {
    # 组件2：全局保留属性数。counts 有效范围 [2, d]；ratios 按 ceil(d * ratio) 换算，范围 (0, 1]。
    "p1_counts": (),
    "p1_ratios": (0.25, 0.50, 0.75),
    # 组件3：粒球局部保留属性数。counts 范围 [1, p1-1]；ratios 按 ceil(p1 * ratio) 换算，范围 (0, 1)。
    "p2_counts": (),
    "p2_ratios": (0.05, 0.10, 0.25, 0.50, 0.75),
    # 组件3：伪纯度停止阈值，满足 pseudo_purity >= theta 且 ball_size < 8 时停止划分，范围 (0, 1]。
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    # PDMF 左右邻域d。counts >= 1 并截断到 m-1；ratios 按 ceil((m-1) * ratio) 换算，范围 (0, 1]。
    "pdmf_neighbors_counts": (5,10,),
    "pdmf_neighbors_ratios": (),
    # 数值稳定项，必须大于 0，不参与网格搜索。
    "pdmf_epsilon": 1e-8,
}

MY_V1_PARAMS = {
    # 组件2：全局保留属性数。counts 有效范围 [2, d]；ratios 按 ceil(d * ratio) 换算，范围 (0, 1]。
    "p1_counts": (),
    "p1_ratios": (0.25, 0.50, 0.75),
    # 组件3：粒球局部保留属性数。counts 范围 [1, p1-1]；ratios 按 ceil(p1 * ratio) 换算，范围 (0, 1)。
    "p2_counts": (),
    "p2_ratios": (0.05, 0.10, 0.25, 0.50, 0.75),
    # 组件3：伪纯度停止阈值，满足 pseudo_purity >= theta 且 ball_size < 8 时停止划分，范围 (0, 1]。
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    # Gaussian-PDMF 左右局部邻域数。counts >= 1 并截断到 m-1；ratios 按 ceil((m-1) * ratio) 换算，范围 (0, 1]。
    "pdmf_neighbors_counts": (5,10),
    "pdmf_neighbors_ratios": (),
    # 稀疏 KNN 图每个样本的邻居数。counts >= 1 并截断到 m-1；ratios 按 ceil((m-1) * ratio) 换算，范围 (0, 1]。
    "graph_neighbors_counts": (3,5,10),
    "graph_neighbors_ratios": (),
    # 边相似度中原始属性相似度的权重 lambda；1-lambda 为 PDMF 形状与展宽相似度权重，范围 (0, 1)。
    "pdmf_similarity_lambda_ratios": (0.1,0.5,0.9),
    # 数值稳定项，必须大于 0，不参与网格搜索。
    "pdmf_epsilon": 1e-8,
}

MY_V2_PARAMS = {
    # 组件2/3：熵相对损失和稀疏图相对损失共同使用的稳定阈值，范围 [0, +inf)。
    "stability_delta_values": (0.001,0.01,0.05,0.1,0.3,0.5,0.7,0.9),
    # 组件3：伪纯度停止阈值，满足 pseudo_purity >= theta 且 ball_size < 8 时停止划分，范围 (0, 1]。
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    # Gaussian-PDMF 左右局部邻域数。counts >= 1；ratios 按 ceil((m-1) * ratio) 换算，范围 (0, 1]。
    "pdmf_neighbors_counts": (5,10),
    "pdmf_neighbors_ratios": (),
    # 稀疏 KNN 图邻居数。counts >= 1；ratios 按 ceil((m-1) * ratio) 换算，范围 (0, 1]。
    "graph_neighbors_counts": (5,10),
    "graph_neighbors_ratios": (),
    # Gaussian-PDMF 相似度中核心值项的权重，范围 (0, 1)。
    "pdmf_similarity_lambda_ratios": (0.5,),
    # 数值稳定项，必须大于 0，不参与网格搜索。
    "pdmf_epsilon": 1e-8,
    # 同一轮粒球划分的线程数；1 为串行，>=2 为并行，不参与参数网格。
    "ball_parallel_jobs": 4,
}

MY_V3_PARAMS = {
    # 组件2：全局保留属性数；counts 使用具体数量，ratios 使用特征比例。
    "p1_counts": (),
    "p1_ratios": (0.25, 0.50, 0.75),
    # 组件3：每个粒球保留的局部属性数。
    "p2_counts": (),
    "p2_ratios": (0.05, 0.10, 0.25, 0.50, 0.75),
    # 伪纯度停止阈值：pseudo_purity >= theta 且 ball_size < 8。
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    # Gaussian-PDMF 邻域和局部互惠 KNN 图邻域。
    "pdmf_neighbors_counts": (5, 10),
    "pdmf_neighbors_ratios": (),
    "graph_neighbors_counts": (3, 5, 10),
    "graph_neighbors_ratios": (),
    # PDMF 边相似度中原始相似度的权重 lambda。
    "pdmf_similarity_lambda_ratios": (0.1, 0.5, 0.9),
    # 属性冗余惩罚 beta；beta=0 可作为 V1 风格消融。
    "redundancy_beta_values": (0.0, 0.1, 0.3, 0.5),
    # V3 默认自适应融合熵重要性和图重要性；图结构选项固定开启。
    "fusion_alpha_mode": "adaptive",
    "mutual_knn": True,
    "self_tuning_graph": True,
    "pdmf_epsilon": 1e-8,
}

MY_V4_PARAMS = {
    # 组件2：全局属性数量；伪标签互信息权重由置信度自动确定。
    "p1_counts": (),
    "p1_ratios": (0.25, 0.50, 0.75),
    # 组件3：每个粒球的局部属性数量。
    "p2_counts": (),
    "p2_ratios": (0.05, 0.10, 0.25, 0.50, 0.75),
    # 伪纯度停止阈值。
    "theta_values": tuple(i / 100 for i in range(70, 100, 5)),
    "pdmf_neighbors_counts": (5, 10),
    "pdmf_neighbors_ratios": (),
    "graph_neighbors_counts": (3, 5, 10),
    "graph_neighbors_ratios": (),
    # PDMF 边相似度中的原始相似度权重。
    "pdmf_similarity_lambda_ratios": (0.1, 0.5, 0.9),
    # V4 不设置 alpha/beta/gamma；伪标签置信度自动生成融合权重。
    "pdmf_epsilon": 1e-8,
}

GB_POJG_GBDPC_PARAMS = {
    # GB-POJG 合理粒度质量 BQ(G)=NumInBall*exp(-gamma*AveRadius) 中的 gamma，范围 [0,+inf)。
    # 原论文网格：{0,0.05,...,1,2,...,10}，共 30 个候选值。
    "gamma_values": (
        *tuple(round(i * 0.05, 2) for i in range(21)),
        # *tuple(range(2, 11)), 原文建议[0,1]，故去掉该范围值
    ),
    # 初始粒球分裂阈值 max(delta*sqrt(n), n**(1/4)) 中的 delta，范围 (0,1]。
    # 原论文网格：{0.1,0.2,...,1.0}，共 10 个候选值。
    "delta_values": tuple(round(i / 10, 1) for i in range(4, 11)),
    # "delta_values": tuple(round(i / 10, 1) for i in range(1, 11)),  # 原文建议[0.4,1.0]，故去掉该范围值

}

GBSC_PARAMS = {
    # 粒球边界距离高斯相似度 exp(-d_boundary^2/(2*sigma^2)) 的带宽。
    # 官方 UCI 可执行源码固定使用 sigma=1.0；论文 UCI 表中记为 0.1。
    "sigma_values": (1.0,),
}

SAGBC_PARAMS = {
    # 官方源码 CC 实验固定抽取 5000 个代表样本；小数据集使用全部样本 min(5000, n)。
    "sample_size": 5000,
}

GBCT_PARAMS = {
    # 官方源码固定：初始 sqrt(n) 粗划分、细分 2-Means、噪声密度阈值 0.2；不做网格搜索。
    "noise_density_ratio": 0.2,
}

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path


@dataclass(frozen=True)
class ExperimentConfig:
    algorithms: tuple[str, ...] = ("my_v2",)  # 可选:("plgb_fsc", "my_v0", "my_v1", "my_v2", "my_v3", "my_v4", "gb_pojg_gbdpc", "gbsc", "sagbc")
    datasets: tuple[str, ...] = ("SuCancer",)#("COIL20","ORL","SuCancer","USPS","Yale","warpPIE10P","GLIOMA","TOX_171","ALLAML",)
                                # "PenDigits","Letter","Covertype")    # 指定运行数据集名
    seeds: tuple[int, ...] = (1,2,3)         # 指定运行种子
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

# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_COIL20.log 2>&1" -WindowStyle Hidden

# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_ORL.log 2>&1" -WindowStyle Hidden


# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_Yale.log 2>&1" -WindowStyle Hidden

# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_warpPIE10P.log 2>&1" -WindowStyle Hidden

# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_GLIOMA.log 2>&1" -WindowStyle Hidden

# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_ALLAML.log 2>&1" -WindowStyle Hidden

# Start-Process -FilePath "cmd.exe" -ArgumentList "/c ..\python\.venv\Scripts\python.exe .\run.py > only_plgb_fsc_with_SuCancer.log 2>&1" -WindowStyle Hidden


# Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*only_plgb_fsc_with_SuCancer.log*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
