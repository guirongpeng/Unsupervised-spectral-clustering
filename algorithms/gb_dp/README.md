# GB-DP Benchmark 实现

本包根据论文 Algorithm 1、Algorithm 2 和官方 `GB-DP.py` 接入统一
Benchmark。

## 保留的官方行为

- 粒球最大规模使用 `ceil(sqrt(n))`。
- 粒球通过 `n_init=1`、`random_state=8` 的 2-means 递归划分。
- 密度计算保留官方代码的坐标绝对偏差均值。
- 粒球间距离、delta 距离和最近高密度粒球遵循官方实现。
- 聚类标签按密度降序传播，再扩展到粒球中的全部样本。
- 数据读取、预处理和指标计算全部由 Benchmark 完成。

## 自动中心选择

论文和官方代码要求在决策图上人工选择中心，没有给出自动阈值。按项目确认
的自动化方案，本实现计算：

```text
decision_score = rho * delta
```

并选择分数最高的前 `n_clusters` 个粒球。`n_clusters` 由统一 Benchmark
提供。这是本实现相对官方算法的唯一流程替换。

## 单独运行

从 `algorithm/python` 目录运行：

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("GB-DP-main").resolve()))

from gb_dp import GBDP, GBDPConfig

algorithm = GBDP(GBDPConfig(n_clusters=5))
labels = algorithm.fit_predict(X)
```

## 接入统一 Benchmark

```python
from benchmark import AlgorithmRegistry, ExperimentRunner
from gb_dp import register_gb_dp

registry = AlgorithmRegistry()
register_gb_dp(registry)
result = ExperimentRunner(registry).run(
    "gb_dp",
    dataset,
    experiment_config,
)
```

论文实验使用 ACC、NMI 和运行时间评价 GB-DP。
