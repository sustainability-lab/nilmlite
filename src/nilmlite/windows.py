"""Sliding-window extraction for seq2point / seq2seq models.

These produce exactly the tensors a trained PyTorch/ONNX model consumes, so the
same windowing runs at train time (Python) and inference time (browser/WASM).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["sliding_windows", "seq2point_xy"]


def sliding_windows(x, width: int, stride: int = 1) -> np.ndarray:
    """(N,) -> (num_windows, width) view. Zero-copy when stride == 1."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("expected a 1-D array")
    if width > x.shape[0]:
        raise ValueError(f"width {width} larger than series length {x.shape[0]}")
    w = sliding_window_view(x, width)
    return w[::stride] if stride > 1 else w


def seq2point_xy(mains, target, width: int, stride: int = 1):
    """Windows of `mains` mapped to the midpoint sample of `target`.

    Returns (X, y) with X shape (n, width) and y shape (n,).
    """
    X = sliding_windows(mains, width, stride)
    half = width // 2
    y = np.asarray(target)[half: half + X.shape[0] * stride: stride]
    y = y[: X.shape[0]]
    return X, y
