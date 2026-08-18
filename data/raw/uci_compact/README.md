# Additional compact UCI datasets

| Benchmark name | Source | Shape used | Classes | Conversion |
|---|---|---:|---:|---|
| `Parkinsons` | <https://archive.ics.uci.edu/dataset/174/parkinsons> | 195 x 22 | 2 | Drop the recording identifier `name`; use `status` as the label. |
| `LSVT` | <https://archive.ics.uci.edu/dataset/282/lsvt+voice+rehabilitation> | 126 x 310 | 2 | Use the complete `Data` worksheet; use the separate `Binary response` worksheet as labels. |

The LSVT repository page describes 309 features, while the official Excel `Data` worksheet contains 310 numeric columns (including `Data_length`). The benchmark retains the complete official worksheet and does not remove or alter any source feature. No normalization or feature selection is applied during conversion.
