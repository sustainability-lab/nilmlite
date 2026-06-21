# Modern stack: a Rust → WASM NILM core

The contribution beyond "pip-only Python": the NILM **compute hot path** — metrics,
mean-resampling, and combinatorial-optimization (CO) disaggregation — is a single
Rust crate (`rust/nilm-core`) that compiles to:

- an **18 KB WebAssembly module** that runs **in the browser** (client-side), and
- a **native** library (`rlib`/`cdylib`),
- (ahead) a **Python wheel via wasmtime** — one core, every runtime.

This is the geolibre-rust move applied to NILM: nobody has a Rust/WASM NILM engine.

## Measured (Apple M-series)

| op | native Rust | NumPy (C/SIMD) | **WASM (in browser)** |
|---|---|---|---|
| MAE, 2,000,000 pts | 5.3 ms | 1.5 ms | **9.6 ms** |
| resample ÷10, 2M | 1.3 ms | 1.3 ms | **5.2 ms** |
| CO disaggregation, 200k, 4 appliances | **2.9 ms** | 7.3 ms | **15 ms** |

Honest reading:
- The claim is **not** "Rust beats NumPy" — NumPy is hand-tuned C/SIMD and wins the
  SIMD-friendly reduction (MAE). On the **algorithmic kernel (CO)** Rust is **2.5×
  faster** than vectorized NumPy.
- The real win is **portability at native-class speed**: the exact same core runs
  **in a browser tab** (CO over 200k samples in 15 ms) from an **18 KB** artifact —
  something the Python/NILMTK stack fundamentally cannot do.

## Why it beats the NILMTK / nilmtk-contrib stack

| | NILMTK / -contrib | nilmlite core |
|---|---|---|
| Runs in a browser | ✗ | ✓ (WASM) |
| Install to run the compute | not pip-installable; 1.89 GB Docker | 18 KB module / `pip install` |
| Portability | one platform (Python) | browser · native · (wasmtime) Python · edge |
| Verify a result | install the stack | open a URL |

Live demo: [`/wasm.html`](https://sustainability-lab.github.io/nilmlite/wasm.html) —
press the button and the Rust core runs in your tab.

## Build

```bash
cd rust/nilm-core
cargo build --release            # native + bench:  ./target/release/bench
wasm-pack build --target web     # -> pkg/  (copy *.js + *_bg.wasm into docs/wasm/)
```
