# Additional UCI clustering-evaluation datasets

All source files were downloaded from the official UCI Machine Learning Repository on 2026-08-18. Each corresponding file in `../../standardized/` stores `X` as `float64` and `y` as integer labels. Feature Min-Max scaling remains the responsibility of the benchmark runner.

| Benchmark name | Official source | Raw-file handling | `X` shape | Classes |
|---|---|---|---:|---:|
| BalanceScale | https://archive.ics.uci.edu/dataset/12/balance+scale | First comma-separated column is the label; remaining four columns are features. | 625 x 4 | 3 |
| Banknote | https://archive.ics.uci.edu/dataset/267/banknote+authentication | Last comma-separated column is the label. | 1372 x 4 | 2 |
| Haberman | https://archive.ics.uci.edu/dataset/43/haberman+s+survival | Last comma-separated column is the label. | 306 x 3 | 2 |
| Yeast | https://archive.ics.uci.edu/dataset/110/yeast | First whitespace-separated field is the sequence identifier and is discarded; final field is the label. | 1484 x 8 | 10 |
| Landsat | https://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite | Official `sat.trn` and `sat.tst` are concatenated; final field is the label. | 6435 x 36 | 6 |
| RiceCammeoOsmancik | https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik | Official ARFF data rows; final field is the label. | 3810 x 7 | 2 |
