//! Rough timings on large random inputs: `cargo run --release --example bench`.
//! `scripts/bench_pyrle.py` runs the same workload with pyrle.

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

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = std::hint::black_box(f());
    println!(
        "{label:<40} {:>8.1} ms",
        start.elapsed().as_secs_f64() * 1e3
    );
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
    let b = time("coverage_count, 1M more intervals", || {
        let starts: Vec<i64> = (0..n).map(|_| rng.below(100_000_000)).collect();
        let ends: Vec<i64> = starts.iter().map(|&s| s + 100).collect();
        coverage_count(&starts, &ends, None).unwrap()
    });
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
