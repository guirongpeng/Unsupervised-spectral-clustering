from __future__ import annotations

"""One-time conversion of checked-in UCI raw files to Benchmark X/y NPZ files."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "uci_clustering"
OUTPUT = ROOT / "standardized"


def save(name: str, X: np.ndarray, y: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float64)
    y = np.unique(np.asarray(y), return_inverse=True)[1].astype(np.int64)
    if X.ndim != 2 or X.shape[0] != y.size or not np.all(np.isfinite(X)):
        raise ValueError(f"Invalid {name} data")
    np.savez_compressed(OUTPUT / f"{name}.npz", X=X, y=y)


def main() -> None:
    wine = np.loadtxt(RAW / "Wine.data", delimiter=",")
    save("Wine", wine[:, 1:], wine[:, 0])

    glass = np.loadtxt(RAW / "Glass.data", delimiter=",")
    save("Glass", glass[:, 1:-1], glass[:, -1])  # discard UCI sample identifier

    seeds = np.loadtxt(RAW / "Seeds.txt")
    save("Seeds", seeds[:, :-1], seeds[:, -1])

    ionosphere = np.genfromtxt(RAW / "Ionosphere.data", delimiter=",", dtype=str)
    save("Ionosphere", ionosphere[:, :-1].astype(float), ionosphere[:, -1])

    wdbc = np.genfromtxt(RAW / "WDBC.data", delimiter=",", dtype=str)
    save("WDBC", wdbc[:, 2:].astype(float), wdbc[:, 1])  # discard UCI identifier

    iris = np.genfromtxt(RAW / "Iris.data", delimiter=",", dtype=str)
    iris = iris[iris[:, 0] != ""]
    save("Iris", iris[:, :-1].astype(float), iris[:, -1])

    ecoli = np.genfromtxt(RAW / "Ecoli.data", dtype=str)
    save("Ecoli", ecoli[:, 1:-1].astype(float), ecoli[:, -1])  # discard sequence name

    libras = np.loadtxt(RAW / "Libras" / "movement_libras.data", delimiter=",")
    save("Libras", libras[:, :-1], libras[:, -1])

    optdigits = np.vstack((
        np.loadtxt(RAW / "Optdigits" / "optdigits.tra", delimiter=","),
        np.loadtxt(RAW / "Optdigits" / "optdigits.tes", delimiter=","),
    ))
    save("Optdigits", optdigits[:, :-1], optdigits[:, -1])

    sonar = np.genfromtxt(RAW / "Sonar" / "sonar.all-data", delimiter=",", dtype=str)
    save("Sonar", sonar[:, :-1].astype(float), sonar[:, -1])

    segment = np.loadtxt(RAW / "Segment" / "segment.dat")
    save("Segment", segment[:, :-1], segment[:, -1])

    vehicle_rows = [np.genfromtxt(path, dtype=str) for path in sorted((RAW / "Vehicle").glob("x*.dat"))]
    vehicle = np.vstack(vehicle_rows)
    save("Vehicle", vehicle[:, :-1].astype(float), vehicle[:, -1])


if __name__ == "__main__":
    main()
