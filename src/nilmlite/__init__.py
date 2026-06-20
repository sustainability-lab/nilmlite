"""nilmlite — a lightweight, dependency-free NILM data layer.

Parquet + Polars in place of NILMTK's HDF5/PyTables stack. Pip-installable in
seconds, no conda, no Docker, no native HDF5. The clean seam (standard
windowing + columnar I/O) feeds a future Rust/WASM core and in-browser ONNX
inference.
"""
from __future__ import annotations

from . import baselines, convert, evaluate as _evaluate, io, leaderboard, metrics, resample, tasks, windows
from .evaluate import evaluate
from .io import Dataset, load_building, load_manifest, save_building, save_manifest
from .leaderboard import leaderboard_html, leaderboard_text
from .metrics import report
from .resample import fill_gaps, resample as resample_to
from .schema import MAINS_COL, TIME_COL, Manifest
from .tasks import Split, Task
from .windows import seq2point_xy, sliding_windows

__version__ = "0.0.1"

__all__ = [
    "Dataset", "Manifest", "Task", "Split",
    "load_building", "save_building", "load_manifest", "save_manifest",
    "resample_to", "fill_gaps", "sliding_windows", "seq2point_xy",
    "evaluate", "leaderboard_html", "leaderboard_text", "report",
    "metrics", "baselines", "convert", "io", "resample", "tasks", "windows", "leaderboard",
    "TIME_COL", "MAINS_COL", "__version__",
]
