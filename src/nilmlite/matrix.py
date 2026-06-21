"""Cross-dataset generalization matrix.

For one appliance and one model family, train on each dataset and test on every
dataset — the N×N table whose off-diagonal is the cross-dataset gap. The diagonal
is within-dataset (cross-building when train/test buildings differ).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_building
from .metrics import f1 as _f1
from .metrics import mae as _mae
from .resample import resample
from .schema import MAINS_COL
from .windows import seq2point_xy

__all__ = ["cross_dataset_matrix", "matrix_text", "matrix_html"]


def _xy(path, building, appliance, period_s, window):
    df = resample(load_building(Path(path) / f"building{building}.parquet"), period_s)
    if appliance not in df.columns:
        return None
    return seq2point_xy(df[MAINS_COL].to_numpy().astype(np.float64),
                        df[appliance].to_numpy().astype(np.float64), window)


def _concat(path, buildings, appliance, period_s, window):
    parts = [p for b in buildings if (p := _xy(path, b, appliance, period_s, window))]
    if not parts:
        return None
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def cross_dataset_matrix(specs, appliance, model_factory, period_s=60, window=99,
                         metric="mae"):
    """`specs`: list of {name, path, train:[buildings], test:building}.

    Returns {appliance, model, metric, names, matrix:[[...]]} where matrix[i][j]
    is train-on-specs[i] / test-on-specs[j].
    """
    score = _mae if metric == "mae" else _f1
    names = [s["name"] for s in specs]
    test_xy = {s["name"]: _concat(s["path"], [s["test"]], appliance, period_s, window)
               for s in specs}
    M = []
    for src in specs:
        tr = _concat(src["path"], src["train"], appliance, period_s, window)
        row = []
        if tr is None:
            M.append([None] * len(specs)); continue
        model = model_factory().fit(*tr)
        for tgt in specs:
            te = test_xy[tgt["name"]]
            row.append(None if te is None else round(float(score(te[1], model.predict(te[0]))), 2))
        M.append(row)
    return {"appliance": appliance, "metric": metric, "names": names, "matrix": M}


def matrix_text(result):
    names, M = result["names"], result["matrix"]
    w = max(10, *(len(n) for n in names))
    head = " " * (w + 2) + "".join(f"{('→'+n)[:w]:>{w+2}}" for n in names)
    out = [f"\n{result['appliance']} · {result['metric'].upper()}  (train ↓ / test →)", head]
    for i, n in enumerate(names):
        cells = "".join(f"{('—' if v is None else f'{v:.1f}'):>{w+2}}" for v in M[i])
        out.append(f"{n:<{w+2}}{cells}")
    return "\n".join(out)


def matrix_html(result):
    """One cross-dataset matrix as a colour-scaled HTML heatmap table."""
    names, M = result["names"], result["matrix"]
    vals = [v for row in M for v in row if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)

    def cell(v):
        if v is None:
            return '<td style="background:#1a2030">—</td>'
        t = (v - lo) / (hi - lo) if hi > lo else 0
        return f'<td style="background:hsl({120*(1-t):.0f},55%,28%);color:#fff;text-align:center">{v:.1f}</td>'

    head = f'<tr><th>{result.get("model","")}</th>' + "".join(f"<th>→ {n}</th>" for n in names) + "</tr>"
    rows = "".join(f'<tr><td style="text-align:left;font-weight:700">{n}</td>'
                   + "".join(cell(v) for v in M[i]) + "</tr>" for i, n in enumerate(names))
    return f'<table class="mtx"><thead>{head}</thead><tbody>{rows}</tbody></table>'
