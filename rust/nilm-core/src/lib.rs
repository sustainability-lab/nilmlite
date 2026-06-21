//! nilm-core — the hot path of NILM in Rust, compiled to WASM (browser) and
//! native. Metrics, mean-resampling, and combinatorial-optimization
//! disaggregation. The geolibre-rust move: one portable core, no Python at
//! inference/eval time.
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn mae(truth: &[f64], pred: &[f64]) -> f64 {
    let n = truth.len().min(pred.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut s = 0.0;
    for i in 0..n {
        s += (truth[i] - pred[i]).abs();
    }
    s / n as f64
}

#[wasm_bindgen]
pub fn f1(truth: &[f64], pred: &[f64], thr: f64) -> f64 {
    let (mut tp, mut fp, mut fnn) = (0.0f64, 0.0f64, 0.0f64);
    let n = truth.len().min(pred.len());
    for i in 0..n {
        let a = truth[i] >= thr;
        let b = pred[i] >= thr;
        if a && b {
            tp += 1.0;
        } else if !a && b {
            fp += 1.0;
        } else if a && !b {
            fnn += 1.0;
        }
    }
    let p = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let r = if tp + fnn > 0.0 { tp / (tp + fnn) } else { 0.0 };
    if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    }
}

/// Mean-downsample `x` by integer `factor` (analysis-ready resampling).
#[wasm_bindgen]
pub fn resample_mean(x: &[f64], factor: usize) -> Vec<f64> {
    if factor <= 1 {
        return x.to_vec();
    }
    let mut out = Vec::with_capacity(x.len() / factor + 1);
    let mut i = 0;
    while i < x.len() {
        let end = (i + factor).min(x.len());
        let mut s = 0.0;
        for v in &x[i..end] {
            s += v;
        }
        out.push(s / (end - i) as f64);
        i += factor;
    }
    out
}

/// Combinatorial Optimization (Hart 1985), joint over appliances.
/// `states_flat` holds `n_app * n_states` power levels (appliance-major).
/// Returns flattened `n_samples * n_app` predictions (sample-major).
#[wasm_bindgen]
pub fn co_disaggregate(mains: &[f64], states_flat: &[f64], n_app: usize, n_states: usize) -> Vec<f64> {
    let n_combo = n_states.pow(n_app as u32);
    let mut sums = vec![0.0f64; n_combo];
    let mut combos = vec![0.0f64; n_combo * n_app];
    for c in 0..n_combo {
        let mut rem = c;
        let mut s = 0.0;
        for a in 0..n_app {
            let si = rem % n_states;
            rem /= n_states;
            let val = states_flat[a * n_states + si];
            combos[c * n_app + a] = val;
            s += val;
        }
        sums[c] = s;
    }
    let mut out = Vec::with_capacity(mains.len() * n_app);
    for &m in mains {
        let mut bi = 0usize;
        let mut bd = f64::INFINITY;
        for (k, &su) in sums.iter().enumerate() {
            let d = (m - su).abs();
            if d < bd {
                bd = d;
                bi = k;
            }
        }
        for a in 0..n_app {
            out.push(combos[bi * n_app + a]);
        }
    }
    out
}
