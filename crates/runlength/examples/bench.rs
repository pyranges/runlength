//! Timings on large random inputs: `cargo run --release --example bench`.
//! `scripts/bench.py` runs the same workload with the Python package and
//! pyrle, and `scripts/bench_r.R` with Bioconductor. Each timing is the best
//! of three runs.

use std::time::Instant;

use runlength::prelude::*;
use runlength::{coverage_count, coverage_sum_by_key};

/// A small deterministic generator, so both benchmarks see similar data.
struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: u64) -> i64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n) as i64
    }
}

fn time<T>(label: &str, f: impl Fn() -> T) -> T {
    let mut best = f64::INFINITY;
    for _ in 0..2 {
        let start = Instant::now();
        std::hint::black_box(f());
        best = best.min(start.elapsed().as_secs_f64());
    }
    let start = Instant::now();
    let out = std::hint::black_box(f());
    best = best.min(start.elapsed().as_secs_f64());
    println!("{label:<40} {:>8.0} ms", best * 1e3);
    out
}

fn main() {
    let n = 1_000_000;
    let mut rng = Lcg(42);
    let starts: Vec<i64> = (0..n).map(|_| rng.below(100_000_000)).collect();
    let ends: Vec<i64> = starts.iter().map(|&s| s + 100).collect();

    let a = time("coverage_count, 1M intervals", || {
        coverage_count(&starts, &ends, None).unwrap()
    });
    let starts2: Vec<i64> = (0..n).map(|_| rng.below(100_000_000)).collect();
    let ends2: Vec<i64> = starts2.iter().map(|&s| s + 100).collect();
    let b = coverage_count(&starts2, &ends2, None).unwrap();
    println!("  runs: {} and {}", a.num_runs(), b.num_runs());

    let sum = time("a + b", || &a + &b);
    println!("  runs: {}", sum.num_runs());
    time("a.elem_lt(b)", || a.elem_lt(&b));
    time("a.mean()", || a.mean());

    let slice_starts: Vec<i64> = (0..n).map(|_| rng.below(99_000_000)).collect();
    let slice_ends: Vec<i64> = slice_starts.iter().map(|&s| s + 1000).collect();
    time("getitems, 1M slices of 1000", || {
        a.getitems(&slice_starts, &slice_ends).unwrap()
    });
    time("slice_reduce sum, same slices", || {
        a.slice_reduce(&slice_starts, &slice_ends, |s| s.sum())
            .unwrap()
    });
    time("slice_reduce mean, same slices", || {
        a.slice_reduce(&slice_starts, &slice_ends, |s| s.mean())
            .unwrap()
    });

    let keys: Vec<u32> = (0..n).map(|_| rng.below(25) as u32).collect();
    let weights = vec![1i64; n];
    let batch = time("coverage_sum_by_key, 25 keys", || {
        coverage_sum_by_key(&keys, &starts, &ends, &weights).unwrap()
    });
    time("batch + batch, 25 keys", || &batch + &batch);
}
