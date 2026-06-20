"""Time-series resampling and alignment with Polars."""
from __future__ import annotations

import polars as pl

from .schema import TIME_COL

__all__ = ["resample", "fill_gaps"]


def resample(df: pl.DataFrame, period_s: int, time_col: str = TIME_COL,
             how: str = "mean") -> pl.DataFrame:
    """Downsample every value column to `period_s` seconds.

    `how` is any Polars aggregation name: mean, sum, max, min, median.

    Implemented as truncate-to-bucket + ``group_by`` rather than
    ``group_by_dynamic``: identical results for fixed-period downsampling and
    noticeably faster because it runs the hashed aggregation in parallel without
    the ordered-window bookkeeping.
    """
    value_cols = [c for c in df.columns if c != time_col]
    agg = [getattr(pl.col(c), how)().alias(c) for c in value_cols]
    return (
        df.with_columns(pl.col(time_col).dt.truncate(f"{period_s}s"))
          .group_by(time_col)
          .agg(agg)
          .sort(time_col)
    )


def fill_gaps(df: pl.DataFrame, period_s: int, time_col: str = TIME_COL,
              value: float = 0.0) -> pl.DataFrame:
    """Insert missing timestamps on a regular grid and fill holes with `value`."""
    df = df.sort(time_col)
    grid = df.select(
        pl.datetime_range(pl.col(time_col).min(), pl.col(time_col).max(),
                          interval=f"{period_s}s", time_unit="us").alias(time_col)
    )
    out = grid.join(df, on=time_col, how="left")
    value_cols = [c for c in df.columns if c != time_col]
    return out.with_columns([pl.col(c).fill_null(value) for c in value_cols])
