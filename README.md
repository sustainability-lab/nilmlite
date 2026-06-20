# nilmlite

**A lightweight, dependency-free data layer for NILM** — Parquet + Polars in place
of NILMTK's HDF5/PyTables stack.

Inspired by [`geolibre-rust`](https://github.com/opengeos/geolibre-rust), which
deletes the GDAL/native-dependency tax from geospatial work by shipping a pure
core that runs anywhere. `nilmlite` does the same for energy disaggregation: the
brittle, heavy plumbing (dataset I/O, alignment, resampling, windowing, metrics)
becomes a tiny, portable, columnar core.

```bash
pip install nilmlite        # seconds. no conda, no Docker, no HDF5/GDAL, no TF/torch
```

> This is **Phase 0**: a pure-Python reference implementation. It establishes the
> on-disk contract (NILM-Parquet) and the clean API seam that a future Rust/WASM
> core (`nilmlite-wasm`, the direct geolibre analog) and in-browser ONNX
> inference plug into. See [Roadmap](#roadmap).

## Why

Standing up `nilmtk` + `nilmtk-contrib` means conda pins, a TF/Keras-vs-PyTorch
split, multi-GB CUDA wheels, and — in practice — a Docker image to make any of it
reproducible. That is NILM's version of "GDAL hell." `nilmlite` removes it:

| | NILMTK data layer | nilmlite |
|---|---|---|
| Install | conda + PyTables + (often) Docker | `pip install`, seconds |
| Storage | custom HDF5 meter hierarchy | one wide **Parquet** per building |
| Engine | pandas | **Polars** (multi-threaded, lazy) |
| Native deps | HDF5 | none (Arrow wheels) |
| Reads from | NILMTK only | Polars, pandas, DuckDB, Rust, the browser |
| Reproducible | "works on my machine" | same bytes every OS |

## Quickstart

```bash
python examples/quickstart.py
# or:
nilmlite synth --out data/synth --days 10
nilmlite info data/synth
```

```python
import nilmlite as nl
from nilmlite.convert import make_synthetic

df, manifest = make_synthetic(days=10, period_s=6)
nl.save_manifest("data/h1", manifest)
nl.save_building(df, "data/h1/building1.parquet")

data  = nl.Dataset("data/h1").load(1)
rs    = nl.resample_to(data, 60)                 # 6s -> 60s, multi-threaded
X, y  = nl.seq2point_xy(rs["mains"].to_numpy(),  # tensors a model consumes
                        rs["fridge"].to_numpy(), width=99)
print(nl.report(y, y))                           # full metric suite
```

## NILM-Parquet format

A dataset is just a directory:

```
mydataset/
  manifest.json        # name, sample_period_s, appliances, units, source
  building1.parquet    # timestamp | mains | fridge | microwave | ...  (wide, Float32, UTC)
  building2.parquet
```

No custom reader required — anything that speaks Parquet/Arrow can consume it.

## Benchmark

Identical pipeline (read → align → resample → window → metrics) over the same
synthetic trace, `nilmlite` vs the NILMTK-style pandas/PyTables layer:

```bash
python bench/gen_data.py 365 6  # writes parquet + HDF5(table) + HDF5(fixed)
python bench/bench_load.py      # prints the table, writes bench/results.md
```

Measured on a 365-day @ 6 s trace (**5.26 M rows**), median of 3 runs (Apple M-series):

| backend | read+align | resample | total |
|---|---|---|---|
| **nilmlite** (Polars/Parquet) | **14.3 ms** | **51.3 ms** | **82.1 ms** |
| pandas/HDF5 *table* (NILMTK-style) | 135.8 ms | 89.6 ms | 243.5 ms |
| pandas/HDF5 *fixed* | 72.5 ms | 81.0 ms | 171.1 ms |

→ **3.0× faster end-to-end**, **9.5× faster** to read+align, and on disk
**42.5 MB vs 285 MB (6.7× smaller)**. Numbers land in [`bench/results.md`](bench/results.md).

### Install footprint

```
core:  pip install nilmlite   →  3 deps (numpy, polars), ~209 MB, ~0.2 s (warm)
```

versus a conda `nilmtk` + `nilmtk-contrib` environment (TF/Keras or the PyTorch
fork, frequently multi-GB with CUDA wheels, and in practice a Docker image to be
reproducible at all).

## Real datasets — migrate once, never look back

Convert any NILMTK HDF5 dataset (REDD, UK-DALE, REFIT, iAWE, ECO…) to NILM-Parquet
with a one-time tool that lives **outside** the package (it's the only thing that
needs the legacy nilmtk/PyTables stack):

```bash
# run once, in your existing nilmtk environment
python tools/nilmtk_to_parquet.py redd.h5 data/redd --period 6
```

REDD's 383 MB HDF5 becomes ~18 MB of Parquet (6 buildings); afterwards `nilmlite`
reads it with no HDF5/conda/Docker, ever.

## Benchmark harness

```python
import nilmlite as nl
from nilmlite.baselines import Linear, Mean

t = nl.Task(name="REDD · cross-building", kind="cross_building",
            appliances=["fridge", "microwave", "dish_washer"],
            train=[nl.Split("data/redd", 1), nl.Split("data/redd", 2)],
            test=[nl.Split("data/redd", 3)], window=99)

result = nl.evaluate({"Mean": Mean, "Linear": lambda: Linear(l2=5)}, t)
print(nl.leaderboard_text(result))
nl.leaderboard_html(result, "leaderboard.html")
```

`Task` covers same-building / cross-building / cross-dataset; `evaluate` reports
MAE/F1/params/inference per model; `leaderboard_html` renders a standalone page
with a generalization-gap callout.

## Deep models + portable ONNX  (`pip install -e ".[dl,onnx]"`)

```python
from nilmlite.models_torch import Seq2Point
m = Seq2Point(window=99).fit(Xtr, ytr)       # same fit/predict interface
m.export_onnx("seq2point_fridge.onnx")       # self-contained: raw watts in/out
```

On **real REDD** (train homes 1–2, test home 3) Seq2Point reaches single-digit-W
MAE where Mean/Linear sit in the 30–50 W range. The exported `.onnx` runs
unchanged in the browser (onnxruntime-web) — training in Python and in-browser
inference share one artifact.

## Visualization  (`pip install -e ".[viz]"`)

`viz.plot_power` (mains + appliances), `viz.plot_day` (daily-routine heatmap),
`viz.plot_signature` (top activation cycles), `viz.plot_disaggregation`
(prediction vs truth). Each returns a matplotlib `Figure`.

## API

- `convert.make_synthetic` / `write_synthetic_dataset` — realistic traces, zero downloads
- `io` — `save_building` / `load_building` / `Dataset` / manifest
- `resample.resample`, `resample.fill_gaps` — Polars time-series ops
- `windows.seq2point_xy`, `windows.sliding_windows` — model-ready tensors
- `tasks.Task` / `evaluate` / `leaderboard_*` — the benchmark spine
- `baselines.Mean`, `baselines.Linear`; `models_torch.Seq2Point` (+ ONNX export)
- `metrics.report` — MAE, RMSE, SAE, NDE, NEP, F1, precision, recall, accuracy, MCC
- `viz` — power, daily heatmap, signature, disaggregation plots

## Roadmap

- **Phase 0 (this repo):** Parquet/Polars data layer, metrics, baselines, benchmark harness, NILMTK migration tool, Seq2Point + ONNX export, viz. Real REDD working. *No Rust.*
- **Phase 1 — `nilmlite-wasm`:** port the hot/portable core (reader, resample, metrics, classic baselines CO/FHMM) to Rust → WASM. Ship to PyPI (via `wasmtime`) and npm — the geolibre move.
- **Phase 2 — browser leaderboard:** run baselines *and trained models* client-side. Train natively in PyTorch → `torch.onnx.export` → run the `.onnx` in-browser with `onnxruntime-web` (or Rust `tract`/`candle` in WASM). Upload mains, watch disaggregation + leaderboard live, zero install — the "ImageNet moment."

Deep-learning **training** stays native PyTorch; only **inference** crosses into WASM.

## License

MIT
