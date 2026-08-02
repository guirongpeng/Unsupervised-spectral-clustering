# Unified Clustering Benchmark


## 当前统一协议

- 输入逐特征 Min-Max 到 `[0,1]`；
- `p1 = ceil(0.75 * d)`；
- `p2 = {50, 60, ..., 200}`，且 `p2 < p1`；
- `theta = {0.70, 0.75, ..., 0.95}`；
- 停止条件：`pseudo_purity >= theta && ball_size < 8`；
- 每组参数跨 seed 求均值，以平均 ACC 选择唯一组合；
- NMI、F-measure、其他指标和耗时均报告同一组合。

## 配置

只修改 `config.py`。例如同时运行两个算法：

```python
algorithms=("plgb_fsc", "my_v0")
datasets=("SuCancer",)
seeds=(1, 2, 3)
```

只运行 MY-V0 时设置 `algorithms=("my_v0",)`；只运行 PLGB-FSC 时设置
`algorithms=("plgb_fsc",)`。正式论文实验时再把 `seeds` 改为
`tuple(range(1, 11))`。

可用数据集：

```python
datasets=(
    "COIL20", "ORL", "SuCancer", "USPS", "Yale", "warpPIE10P",
    "GLIOMA", "TOX_171", "ALLAML", "D3", "T4", "E6",
    "PenDigits", "Letter", "Covertype",
)
```

数据统一保存在 `data/standardized/<数据集>.npz`，键名固定为 `X` 和
`y`。原始下载文件位于 `data/raw/`。

注意：PLGB-FSC 原论文的 `p2={50,...,200}` 只适合高维数据。运行 D3、T4、
E6、PenDigits、Letter 或 Covertype 时，必须在 `config.py` 中设置满足
`p2 < ceil(0.75*d)` 的较小 `p2_values`；这属于跨论文测试，不是
PLGB-FSC 原论文参数协议。

## 运行

```powershell
cd D:\Projects\zhengchuang\paper_team\liqiu\algorithm\unified_benchmark
..\python\.venv\Scripts\python.exe .\run.py
```

结果保存在：

```text
results/<时间戳>/<数据集>/<算法名>/
├── experiment_config.json
├── all_runs.csv
├── grid_summary.csv
├── best_parameter_combination.csv
├── experiment_summary.json
└── labels/
```

同一个数据集同时运行两个算法时，`plgb_fsc/` 和 `my_v0/` 各自保存完整
参数网格结果；时间戳目录的 `benchmark_summary.csv` 每个算法一行，
算法专属参数保存在 `best_params` JSON 字典中。

若实验中断，在 `config.py` 中把 `run_id` 改成已有时间戳，并设置
`resume=True`，然后再次运行同一入口。

## 边界

- `algorithms/plgb_fsc` 只负责算法，不读取数据或计算评价指标；
- `core` 统一负责数据和指标；
- `run.py` 只负责实验循环、参数选择和结果保存；
- 真实标签只用于确定聚类数、计算指标以及实验层的最优参数选择。
