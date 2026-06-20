"""The NILM-Parquet on-disk contract.

A *dataset* is a directory:

    mydataset/
        manifest.json          # describes buildings, rate, appliances, units
        building1.parquet      # wide table: timestamp, mains, <appliance>, ...
        building2.parquet
        ...

Each building parquet is a single **wide, time-aligned** table:

    timestamp : Datetime(us)   (UTC, strictly increasing)
    mains     : Float32        (site aggregate, watts)
    <appliance columns> : Float32 (watts)

This deliberately replaces NILMTK's per-meter HDF5 hierarchy with one flat,
columnar, self-describing file per building — readable by anything that speaks
Parquet/Arrow (Polars, pandas, DuckDB, Rust, the browser), with no PyTables,
no custom schema, and no native HDF5 dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

TIME_COL = "timestamp"
MAINS_COL = "mains"
MANIFEST_NAME = "manifest.json"


@dataclass
class Manifest:
    name: str
    sample_period_s: int                 # base sampling period, seconds
    appliances: list[str]                # appliance column names (excludes mains)
    buildings: list[int] = field(default_factory=list)
    units: str = "W"
    timezone: str = "UTC"
    source: str = ""                     # provenance, e.g. "REDD" / "synthetic"
    version: str = "nilm-parquet/1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def columns(self) -> list[str]:
        return [TIME_COL, MAINS_COL, *self.appliances]
