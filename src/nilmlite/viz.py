"""Plotting helpers (optional `[viz]` extra: matplotlib).

Every function takes a Polars DataFrame (a loaded building) or plain arrays and
returns a matplotlib Figure, so callers control saving/showing.
"""
from __future__ import annotations

import numpy as np

from .schema import MAINS_COL, TIME_COL

__all__ = ["plot_power", "plot_day", "plot_disaggregation", "plot_signature"]

_PALETTE = ["#e4634f", "#4f9de4", "#5fbf6f", "#e0a93b", "#9b6fe0", "#46c2c2"]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _slice(df, time_col, start, end):
    import polars as pl
    if start is not None:
        df = df.filter(pl.col(time_col) >= pl.lit(start).str.to_datetime())
    if end is not None:
        df = df.filter(pl.col(time_col) <= pl.lit(end).str.to_datetime())
    return df


def plot_power(df, appliances=None, start=None, end=None, time_col=TIME_COL,
               mains_col=MAINS_COL, title="Power", figsize=(11, 5)):
    """Mains (filled) with appliances overlaid — the canonical NILM trace view."""
    plt = _plt()
    df = _slice(df, time_col, start, end)
    cols = appliances or [c for c in df.columns if c not in (time_col, mains_col)]
    t = df[time_col].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(t, df[mains_col].to_numpy(), color="#26324a", alpha=.7,
                    label="mains (aggregate)", zorder=1)
    for i, c in enumerate(cols):
        ax.plot(t, df[c].to_numpy(), color=_PALETTE[i % len(_PALETTE)],
                lw=1.3, label=c, zorder=2)
    ax.set_ylabel("power (W)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def plot_day(df, appliance, time_col=TIME_COL, title=None, figsize=(10, 4.5)):
    """Day-vs-time-of-day heatmap of one appliance — exposes usage routines."""
    plt = _plt()
    import polars as pl
    d = df.select([
        pl.col(time_col).dt.date().alias("day"),
        (pl.col(time_col).dt.hour() + pl.col(time_col).dt.minute() / 60).alias("tod"),
        pl.col(appliance),
    ])
    # bucket to 15-min slots × day, mean power
    d = d.with_columns((pl.col("tod") * 4).floor().cast(pl.Int32).alias("slot"))
    piv = d.group_by(["day", "slot"]).agg(pl.col(appliance).mean())
    days = piv["day"].unique().sort().to_list()
    day_ix = {dd: i for i, dd in enumerate(days)}
    grid = np.full((len(days), 96), np.nan)
    for row in piv.iter_rows(named=True):
        s = row["slot"]
        if 0 <= s < 96:
            grid[day_ix[row["day"]], s] = row[appliance]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, 24, 0, len(days)], interpolation="nearest")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("day")
    ax.set_xticks(range(0, 25, 3))
    ax.set_title(title or f"{appliance} — daily routine")
    fig.colorbar(im, ax=ax, label="power (W)")
    fig.tight_layout()
    return fig


def plot_disaggregation(y_true, y_pred, mains=None, title="Disaggregation",
                        labels=("ground truth", "prediction"), figsize=(11, 4.5)):
    """Overlay predicted vs true appliance power (with optional mains context)."""
    plt = _plt()
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    x = np.arange(y_true.shape[0])

    fig, ax = plt.subplots(figsize=figsize)
    if mains is not None:
        m = np.asarray(mains).ravel()[: y_true.shape[0]]
        ax.fill_between(x, m, color="#26324a", alpha=.5, label="mains", zorder=1)
    ax.plot(x, y_true, color="#5fbf6f", lw=1.6, label=labels[0], zorder=2)
    ax.plot(x, y_pred, color="#e4634f", lw=1.3, ls="--", label=labels[1], zorder=3)
    ax.set_ylabel("power (W)")
    ax.set_xlabel("timestep")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def plot_signature(df, appliance, top=6, pad=20, time_col=TIME_COL,
                   figsize=(10, 4.5)):
    """Overlay the strongest activation cycles of an appliance (its 'signature')."""
    plt = _plt()
    x = df[appliance].to_numpy().astype(float)
    on = x > max(15.0, 0.1 * x.max())
    # find contiguous on-segments
    edges = np.diff(on.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if on[0]:
        starts = np.r_[0, starts]
    if on[-1]:
        ends = np.r_[ends, len(x)]
    segs = sorted(zip(starts, ends), key=lambda se: -x[se[0]:se[1]].sum())[:top]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (s, e) in enumerate(segs):
        a, b = max(0, s - pad), min(len(x), e + pad)
        ax.plot(np.arange(b - a), x[a:b], color=_PALETTE[i % len(_PALETTE)],
                lw=1.4, alpha=.85)
    ax.set_ylabel("power (W)")
    ax.set_xlabel("timesteps (aligned)")
    ax.set_title(f"{appliance} — top {len(segs)} activation signatures")
    ax.margins(x=0)
    fig.tight_layout()
    return fig
