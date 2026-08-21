"""Optional CuPy backend used only by PLGB-FSC numerical kernels."""

from __future__ import annotations


def resolve_cupy(compute_device: str):
    """Return CuPy for an available CUDA device, or ``None`` for CPU fallback."""

    if compute_device == "cpu":
        return None
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No CUDA device is available")
        return cp
    except Exception as error:
        if compute_device == "gpu":
            raise RuntimeError(
                "PLGB-FSC GPU mode requires a working CuPy/CUDA installation"
            ) from error
        return None
