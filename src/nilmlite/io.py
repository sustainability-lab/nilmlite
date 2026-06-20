"""Read/write NILM-Parquet datasets."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .schema import MANIFEST_NAME, TIME_COL, Manifest

__all__ = ["save_building", "load_building", "save_manifest", "load_manifest", "Dataset"]


def save_manifest(directory: str | Path, manifest: Manifest) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


def load_manifest(directory: str | Path) -> Manifest:
    path = Path(directory) / MANIFEST_NAME
    return Manifest.from_dict(json.loads(path.read_text()))


def save_building(df: pl.DataFrame, path: str | Path) -> Path:
    """Write one building's wide table to Parquet (zstd)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if TIME_COL not in df.columns:
        raise ValueError(f"missing '{TIME_COL}' column; have {df.columns}")
    df.sort(TIME_COL).write_parquet(path, compression="zstd")
    return path


def load_building(path: str | Path, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Read one building. `lazy=True` returns a LazyFrame for query pushdown."""
    return pl.scan_parquet(path) if lazy else pl.read_parquet(path)


class Dataset:
    """A directory of NILM-Parquet buildings + manifest."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.manifest = load_manifest(self.dir)

    @property
    def buildings(self) -> list[int]:
        return self.manifest.buildings

    def building_path(self, b: int) -> Path:
        return self.dir / f"building{b}.parquet"

    def load(self, b: int, lazy: bool = False):
        return load_building(self.building_path(b), lazy=lazy)

    def __repr__(self) -> str:
        m = self.manifest
        return (f"Dataset({m.name!r}, buildings={m.buildings}, "
                f"{m.sample_period_s}s, appliances={m.appliances})")
