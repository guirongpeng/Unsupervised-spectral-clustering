"""Convert the MNIST and EMNIST Balanced files used by FSC-PLGB."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "fsc_plgb_real"
OUTPUT = ROOT / "standardized"


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, columns = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid IDX image file: {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count * rows * columns:
        raise ValueError(f"Unexpected image count: {path}")
    return values.reshape(count, rows * columns)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid IDX label file: {path}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"Unexpected label count: {path}")
    return values


def _save(
    name: str,
    directory: Path,
    train_name: str = "train",
    test_name: str = "test",
) -> None:
    X = np.vstack((
        _read_images(directory / f"{train_name}-images-idx3-ubyte.gz"),
        _read_images(directory / f"{test_name}-images-idx3-ubyte.gz"),
    ))
    y = np.concatenate((
        _read_labels(directory / f"{train_name}-labels-idx1-ubyte.gz"),
        _read_labels(directory / f"{test_name}-labels-idx1-ubyte.gz"),
    ))
    if X.shape[0] != y.size:
        raise ValueError(f"Mismatched X/y size for {name}")
    np.savez_compressed(OUTPUT / f"{name}.npz", X=X, y=y)
    print(f"{name}: X={X.shape}, classes={np.unique(y).size}")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    _save("MNIST", RAW / "MNIST", test_name="t10k")
    _save(
        "Balanced",
        RAW / "Balanced",
        train_name="emnist-balanced-train",
        test_name="emnist-balanced-test",
    )


if __name__ == "__main__":
    main()
