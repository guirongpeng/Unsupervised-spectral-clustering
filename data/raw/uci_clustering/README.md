# UCI clustering datasets

Files were downloaded from the UCI Machine Learning Repository on 2026-08-13.

| Raw file | Standardized output | Source |
|---|---|---|
| `Wine.data` | `Wine.npz` | https://archive.ics.uci.edu/dataset/109/wine |
| `Glass.data` | `Glass.npz` | https://archive.ics.uci.edu/dataset/42/glass+identification |
| `Seeds.txt` | `Seeds.npz` | https://archive.ics.uci.edu/dataset/236/seeds |
| `Ionosphere.data` | `Ionosphere.npz` | https://archive.ics.uci.edu/dataset/52/ionosphere |
| `WDBC.data` | `WDBC.npz` | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| `Iris.data` | `Iris.npz` | https://archive.ics.uci.edu/dataset/53/iris |
| `Ecoli.data` | `Ecoli.npz` | https://archive.ics.uci.edu/dataset/39/ecoli |
| `Libras/movement_libras.data` | `Libras.npz` | https://archive.ics.uci.edu/dataset/181/libras+movement |
| `Optdigits/optdigits.{tra,tes}` | `Optdigits.npz` | https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits |
| `Sonar/sonar.all-data` | `Sonar.npz` | https://archive.ics.uci.edu/dataset/151/connectionist+bench |
| `Segment/segment.dat` | `Segment.npz` | https://archive.ics.uci.edu/dataset/147/statlog+image+segmentation |
| `Vehicle/x*.dat` | `Vehicle.npz` | https://archive.ics.uci.edu/dataset/149/statlog+vehicle+silhouettes |

The first column or class field is retained only as `y` for Benchmark evaluation.
`X` contains numeric input features only. No scaling is saved in these files;
the Benchmark applies its common feature-wise Min-Max preprocessing at runtime.
