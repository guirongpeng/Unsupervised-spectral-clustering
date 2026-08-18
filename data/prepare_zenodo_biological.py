"""Convert the downloaded Zenodo microarray datasets to benchmark NPZ files."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "zenodo_biological"
OUTPUT = ROOT / "standardized"
DATASETS = {
    "CLL_SUB_111": "CLL_SUB_111.mat",
    "LUNG": "LUNG.mat",
    "CARCINOM": "CARCINOM.mat",
    "LEUKEMIA": "LEUKEMIA.mat",
}


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for name, filename in DATASETS.items():
        source = loadmat(RAW / filename)
        X = np.asarray(source["X"], dtype=float)
        y = np.asarray(source["Y"]).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or not np.all(np.isfinite(X)):
            raise ValueError(f"Invalid {name} data")
        np.savez_compressed(OUTPUT / f"{name}.npz", X=X, y=y)
        print(f"{name}: X={X.shape}, classes={np.unique(y).size}")


if __name__ == "__main__":
    main()
