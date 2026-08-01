# Unified Clustering Benchmark


## PLGB-FSC当前协议

- 输入逐特征 Min-Max 到 `[0,1]`；
- `p1 = ceil(0.75 * d)`；
- `p2 = {50, 60, ..., 200}`，且 `p2 < p1`；
- `theta = {0.70, 0.75, ..., 0.95}`；
- 停止条件：`pseudo_purity >= theta && ball_size < 8`；
- 每组参数跨 seed 求均值，以平均 ACC 选择唯一组合；
- NMI、F-measure、其他指标和耗时均报告同一组合。

## 配置

只修改 `config.py`。当前默认配置为：

```python
datasets=("SuCancer",)
seeds=(1, 2, 3)
```

因此首次验证执行 `1 × 16 × 6 × 3 = 288` 次聚类。正式论文实验时再把
`seeds` 改为 `tuple(range(1, 11))`。

## 运行

```powershell
cd D:\Projects\zhengchuang\paper_team\liqiu\algorithm\unified_benchmark
..\python\.venv\Scripts\python.exe .\run.py
```

结果保存在：

```text
results/<数据集>/<时间戳>/plgb_fsc/
├── experiment_config.json
├── all_runs.csv
├── grid_summary.csv
├── best_parameter_combination.csv
├── experiment_summary.json
└── labels/
```

若实验中断，在 `config.py` 中把 `run_id` 改成已有时间戳，并设置
`resume=True`，然后再次运行同一入口。

## 边界

- `algorithms/plgb_fsc` 只负责算法，不读取数据或计算评价指标；
- `core` 统一负责数据和指标；
- `run.py` 只负责实验循环、参数选择和结果保存；
- 真实标签只用于确定聚类数、计算指标以及实验层的最优参数选择。

