"""Convert selected moderate-size UCI datasets to benchmark NPZ files."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "uci_moderate"
OUTPUT = ROOT / "standardized"


def _save(name: str, X: np.ndarray, y: np.ndarray) -> None:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or not np.all(np.isfinite(X)):
        raise ValueError(f"Invalid {name} data")
    np.savez_compressed(OUTPUT / f"{name}.npz", X=X, y=y)
    print(f"{name}: X={X.shape}, classes={np.unique(y).size}")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)

    musk = np.genfromtxt(RAW / "MuskV1" / "clean1.data", delimiter=",", dtype=str)
    _save("MuskV1", musk[:, 2:-1].astype(float), musk[:, -1].astype(float).astype(int))

    semeion = np.loadtxt(RAW / "Semeion" / "semeion.data")
    _save("Semeion", semeion[:, :256], np.argmax(semeion[:, 256:], axis=1))


if __name__ == "__main__":
    main()
