"""Tiny CLI: `nilmlite synth` and `nilmlite info`."""
from __future__ import annotations

import argparse
from pathlib import Path

from .convert import make_synthetic
from .io import Dataset, save_building, save_manifest


def _synth(args) -> None:
    df, man = make_synthetic(days=args.days, period_s=args.period, seed=args.seed)
    out = Path(args.out)
    save_manifest(out, man)
    save_building(df, out / "building1.parquet")
    size = (out / "building1.parquet").stat().st_size / 1e6
    print(f"wrote {out}/building1.parquet  ({len(df):,} rows, {size:.1f} MB)")


def _info(args) -> None:
    ds = Dataset(args.path)
    print(ds)
    for b in ds.buildings:
        df = ds.load(b)
        print(f"  building{b}: {len(df):,} rows  cols={df.columns}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="nilmlite", description="lightweight NILM data layer")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate a synthetic dataset")
    s.add_argument("--out", default="data/synth")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--period", type=int, default=6)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_synth)

    i = sub.add_parser("info", help="describe a dataset directory")
    i.add_argument("path")
    i.set_defaults(func=_info)

    sv = sub.add_parser("serve", help="serve the no-code studio + real PyTorch training")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--docs", default="docs")
    sv.set_defaults(func=lambda a: __import__("nilmlite.server", fromlist=["serve"]).serve(a.port, a.docs))

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
