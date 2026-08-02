# 论文数据集格式与来源

## 目录结构

```text
data/
├── standardized/              # Benchmark 实际读取
│   └── <Dataset>.npz          # 固定包含 X 和 y
└── raw/                       # 下载源，仅用于追溯和重新转换
    ├── plgb_fsc_icde2025/
    └── fsc_plgb_tkde2026/
```

统一格式约束：

- `X`：二维数值矩阵，形状为 `n_samples × n_features`；
- `y`：一维真实标签，形状为 `n_samples`；
- 不在数据文件中执行归一化；运行时由 Benchmark 逐特征 Min-Max；
- 真实标签只供实验层确定聚类数和计算指标，算法只接收 `X`。

## PLGB-FSC（ICDE 2025，高维数据）

| 数据集 | 样本 × 属性 | 类别 | Benchmark 文件 |
|---|---:|---:|---|
| COIL20 | 1440 × 1024 | 20 | `standardized/COIL20.npz` |
| ORL | 400 × 1024 | 40 | `standardized/ORL.npz` |
| SuCancer | 174 × 7909 | 2 | `standardized/SuCancer.npz` |
| USPS | 7291 × 256 | 10 | `standardized/USPS.npz` |
| Yale | 165 × 1024 | 15 | `standardized/Yale.npz` |
| warpPIE10P | 210 × 2420 | 10 | `standardized/warpPIE10P.npz` |
| GLIOMA | 50 × 4434 | 4 | `standardized/GLIOMA.npz` |
| TOX_171 | 171 × 5748 | 4 | `standardized/TOX_171.npz` |
| ALLAML | 72 × 7129 | 2 | `standardized/ALLAML.npz` |

来源：

- 作者代码：https://github.com/DongdongCheng/PLGB-FSC
- 论文脚注数据页：https://jundongl.github.io/scikit-feature/datasets.html

ASU 当前的 `USPS.mat` 是 9298 个样本的训练、测试合并版；论文使用 7291
个训练样本。原始目录中的 `USPS_7291.mat` 保存其前 7291 行，标签顺序已与
标准 `zip.train` 核对一致。

## FSC-PLGB（TKDE 2026，大规模数据）

下列原始文件来自作者仓库的 `som datasets.rar`。

| 数据集 | 样本 × 属性 | 类别 | Benchmark 文件 |
|---|---:|---:|---|
| D3 | 1741 × 2 | 6 | `standardized/D3.npz` |
| T4 | 7201 × 2 | 6 | `standardized/T4.npz` |
| E6 | 7856 × 2 | 7 | `standardized/E6.npz` |
| PenDigits | 10992 × 16 | 10 | `standardized/PenDigits.npz` |
| Letter | 20000 × 16 | 26 | `standardized/Letter.npz` |
| Covertype | 581012 × 54 | 7 | `standardized/Covertype.npz` |
| USPS | 7291 × 256 | 10 | 共用 `standardized/USPS.npz` |

作者代码和数据：https://github.com/DongdongCheng/FSC-PLGB

## 尚未下载

| 数据集 | 原因 / 下载入口 |
|---|---|
| TB、SF、CC、CG、Flower | 作者 FSC-PLGB 压缩包未包含；原始 U-SPEC 数据入口：https://www.researchgate.net/publication/330760669 |
| TB_10M、CC_10M | 尚未找到能确认与论文完全一致的公开文件 |
| MNIST | 70000 × 784，体积较大：http://yann.lecun.com/exdb/mnist/ |
| Balanced | EMNIST Balanced，131600 × 784、47 类：https://www.nist.gov/itl/products-and-services/emnist-dataset |
| PaviaU、Ground Truth | 高光谱数据：https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes |

大规模数据没有使用名称相同但规模或生成规则无法核对的文件替代。
