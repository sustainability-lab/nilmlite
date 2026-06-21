//! Native Rust benchmark for the hot path (compare against the Python numbers).
use nilm_core::{co_disaggregate, mae, resample_mean};
use std::time::Instant;

fn lcg(n: usize) -> Vec<f64> {
    let mut s = 12345u64;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 200000) as f64 / 100.0
        })
        .collect()
}

fn main() {
    let n = 2_000_000usize;
    let x = lcg(n);
    let pred: Vec<f64> = x.iter().map(|v| v * 0.9).collect();

    let t = Instant::now();
    let m = mae(&x, &pred);
    println!("mae         n={n}  -> {m:.3}   {:?}", t.elapsed());

    let t = Instant::now();
    let r = resample_mean(&x, 10);
    println!("resample/10 n={n}  -> {} pts {:?}", r.len(), t.elapsed());

    // CO: 4 appliances x 2 states over 200k samples
    let states = [0.0, 150.0, 0.0, 1500.0, 0.0, 2000.0, 0.0, 100.0];
    let nco = 200_000usize;
    let t = Instant::now();
    let o = co_disaggregate(&x[..nco], &states, 4, 2);
    println!("co (4ap/2st) n={nco} -> {} vals {:?}", o.len(), t.elapsed());
}
