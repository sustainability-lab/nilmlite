# Systems comparison: `nilmlite` vs NILMTK / nilmtk-contrib

The reproducibility/systems contribution, measured **on one machine, same data**
(ramanujan, inside the working nilmtk-contrib container — so the CPU and the REDD
source are identical for both stacks). Reproduce with `bench/systems_compare.py`.

| Dimension | NILMTK / nilmtk-contrib | **nilmlite** | Δ |
|---|---|---|---|
| **Install** | not on PyPI — `uv pip install nilmtk` → *"not found in the package registry"*; needs conda/GitHub + pinned legacy deps. Reference env = a **1.89 GB Docker image** we had to build. | `pip install nilmlite` → **4 s, 264 MB**, no conda/Docker/HDF5 | pip-only |
| **Data access** (load → resample → window, REDD b1, same machine) | 4.25 s (NILMTK `DataSet`/`MeterGroup` API) | **0.069 s** | **61.6× faster** |
| **Storage** (REDD, all buildings) | 401 MB HDF5 | **18 MB** Parquet | **21.8× smaller** |
| **Determinism** | — | **bit-identical** across runs (hash-stable data layer + seeded models) | reproducible by construction |
| **Verify results** | install the full stack | **run the exported ONNX in a browser — zero install** | zero-install verification |

### Honest framing (for the paper)
- The **61.6×** is *repeated-access* speed: nilmlite reads analysis-ready Parquet
  while NILMTK re-parses and re-resamples native HDF5 on every load. The one-time
  migration (`tools/nilmtk_to_parquet.py`) pays the resample cost once; every
  subsequent load — the common case when benchmarking — is ~free. This is the
  realistic comparison for running a benchmark, not a microbenchmark trick.
- The **install** result is not a slow install — `nilmtk` is simply **not
  pip-installable** (absent from PyPI), which is the crux of NILM's reproducibility
  problem. nilmlite is `pip install` / `pip install git+…`, 4 s.
- **Zero-install verification** is, to our knowledge, new for NILM: anyone can
  re-run the leaderboard's models in a browser (onnxruntime-web) without installing
  anything — the benchmark is verifiable from a URL.

### Reproduce
```bash
# inside the nilmtk container, REDD HDF5 at /in, migrated Parquet at /pq:
uv pip install --system polars numpy git+https://github.com/sustainability-lab/nilmlite
python bench/systems_compare.py
```
