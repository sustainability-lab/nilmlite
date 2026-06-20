# Benchmark: data layer

Identical pipeline (read → align → resample → window → metrics) over the same
synthetic trace. See `bench/bench_load.py`.

| backend | read+align | resample | window | metrics | total |
|---|---|---|---|---|---|
| nilmlite (Polars/Parquet) | 14.3 ms | 51.3 ms | 0.1 ms | 16.5 ms | 82.1 ms |
| pandas/HDF5 table (NILMTK) | 135.8 ms | 89.6 ms | 0.0 ms | 18.1 ms | 243.5 ms |
| pandas/HDF5 fixed | 72.5 ms | 81.0 ms | 0.0 ms | 17.6 ms | 171.1 ms |

**nilmlite is 3.0× faster end-to-end** than the NILMTK-style pandas/PyTables(table) layer.

| format | on-disk |
|---|---|
| Parquet (zstd) | 42.5 MB |
| HDF5 table (NILMTK-style) | 285.1 MB |
| HDF5 fixed | 252.3 MB |
