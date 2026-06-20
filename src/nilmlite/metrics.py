"""NILM disaggregation metrics (pure NumPy).

Regression metrics treat the appliance power signal directly; classification
metrics first binarise on an on/off threshold (W). All functions accept 1-D
array-likes of equal length and return plain floats.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "mae", "rmse", "sae", "nde", "nep",
    "f1", "precision", "recall", "accuracy", "matthews",
    "report",
]


def _pair(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    return yt, yp


# ---- regression ---------------------------------------------------------------

def mae(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def sae(y_true, y_pred) -> float:
    """Signal Aggregate Error: |sum(pred) - sum(true)| / sum(true)."""
    yt, yp = _pair(y_true, y_pred)
    denom = np.sum(yt)
    if denom == 0:
        return float("nan")
    return float(abs(np.sum(yp) - denom) / denom)


def nde(y_true, y_pred) -> float:
    """Normalised Disaggregation Error: sqrt(sum((t-p)^2) / sum(t^2))."""
    yt, yp = _pair(y_true, y_pred)
    denom = np.sum(yt ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sqrt(np.sum((yt - yp) ** 2) / denom))


def nep(y_true, y_pred) -> float:
    """Normalised Error in assigned Power: sum|t-p| / sum(t)."""
    yt, yp = _pair(y_true, y_pred)
    denom = np.sum(yt)
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom)


# ---- classification (on/off) --------------------------------------------------

def _confusion(y_true, y_pred, threshold):
    yt, yp = _pair(y_true, y_pred)
    a = yt >= threshold
    b = yp >= threshold
    tp = float(np.sum(a & b))
    fp = float(np.sum(~a & b))
    fn = float(np.sum(a & ~b))
    tn = float(np.sum(~a & ~b))
    return tp, fp, fn, tn


def precision(y_true, y_pred, threshold: float = 15.0) -> float:
    tp, fp, _, _ = _confusion(y_true, y_pred, threshold)
    return float(tp / (tp + fp)) if (tp + fp) else float("nan")


def recall(y_true, y_pred, threshold: float = 15.0) -> float:
    tp, _, fn, _ = _confusion(y_true, y_pred, threshold)
    return float(tp / (tp + fn)) if (tp + fn) else float("nan")


def f1(y_true, y_pred, threshold: float = 15.0) -> float:
    p = precision(y_true, y_pred, threshold)
    r = recall(y_true, y_pred, threshold)
    if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def accuracy(y_true, y_pred, threshold: float = 15.0) -> float:
    tp, fp, fn, tn = _confusion(y_true, y_pred, threshold)
    n = tp + fp + fn + tn
    return float((tp + tn) / n) if n else float("nan")


def matthews(y_true, y_pred, threshold: float = 15.0) -> float:
    """Matthews correlation coefficient — robust to class imbalance."""
    tp, fp, fn, tn = _confusion(y_true, y_pred, threshold)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return float("nan")
    return float((tp * tn - fp * fn) / denom)


def report(y_true, y_pred, threshold: float = 15.0) -> dict[str, float]:
    """Full metric suite as a dict — the canonical leaderboard row."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "sae": sae(y_true, y_pred),
        "nde": nde(y_true, y_pred),
        "nep": nep(y_true, y_pred),
        "f1": f1(y_true, y_pred, threshold),
        "precision": precision(y_true, y_pred, threshold),
        "recall": recall(y_true, y_pred, threshold),
        "accuracy": accuracy(y_true, y_pred, threshold),
        "mcc": matthews(y_true, y_pred, threshold),
    }
