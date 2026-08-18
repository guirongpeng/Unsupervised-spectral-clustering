# Selected moderate-size UCI datasets

| Benchmark name | Source | Shape | Classes | Conversion |
|---|---|---:|---:|---|
| `MuskV1` | <https://archive.ics.uci.edu/dataset/74/musk+version+1> | 476 x 166 | 2 | Drop molecule and conformation identifiers; final field is the label. |
| `Semeion` | <https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit> | 1,593 x 256 | 10 | First 256 binary pixel attributes; final 10 fields are one-hot digit labels. |

`MuskV1/clean1.data.Z` is the official compressed source. `clean1.data` is its decompressed copy, retained solely to make conversion runnable without an extra decompression dependency. No normalization or feature selection is applied here.
