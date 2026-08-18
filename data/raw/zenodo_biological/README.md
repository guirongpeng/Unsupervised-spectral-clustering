# Zenodo biological / microarray datasets

Source record: <https://doi.org/10.5281/zenodo.2709491>.

The downloaded MATLAB files use `X` for the sample-by-feature matrix and `Y` for class labels. `prepare_zenodo_biological.py` only converts those arrays to the benchmark's `X`/`y` NPZ format; it does not perform normalization or feature selection.

| Benchmark name | Source file | Shape | Classes |
|---|---|---:|---:|
| `CLL_SUB_111` | `CLL_SUB_111.mat` | 111 x 11,340 | 3 |
| `LUNG` | `LUNG.mat` | 203 x 3,312 | 5 |
| `CARCINOM` | `CARCINOM.mat` | 174 x 9,182 | 11 |
| `LEUKEMIA` | `LEUKEMIA.mat` | 72 x 7,070 | 2 |
