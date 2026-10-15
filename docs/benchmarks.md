# Benchmarks

runlength compared with Bioconductor's
[IRanges](https://bioconductor.org/packages/IRanges/) and S4Vectors, the
standard run-length encoding in R, and with pyrle.

## Results

Times in milliseconds; each is the best of three runs.

| Operation | runlength (Python) | runlength (Rust) | IRanges (R) | pyrle |
|---|--:|--:|--:|--:|
| Coverage of 1M intervals | 38 | 32 | 334 | 271 |
| `a + b` | 20 | 17 | 197 | 86 |
| `a < b` | 18 | 14 | 139 | 124 |
| `a.mean()` | 2 | 1 | 18 | 0.5¹ |
| Sum over 1M slices | 34 | 26 | 125 | – |
| 1M slices, copied out | 94 | 90 | –² | – |
| Coverage of 1M intervals, 25 keys | 34 | 17 | 351 | 390 |
| `batch + batch`, 25 keys | 10 | 3 | 59 | 21 |

- **The workload:** 1M random intervals of length 100 on a sequence of length
  100M, giving coverage with about 2M runs (`a`); a second, independent
  coverage (`b`); 1M slices of length 1000; and for the keyed rows, the same
  intervals spread over 25 keys. `a + b` has about 3.9M runs.
- **Machine:** Apple M3 Pro (12 cores), runlength from master (after 0.1.0), R 4.5.0 with
  IRanges 2.42.0, pyrle 0.0.41 on Python 3.12.
- **runlength (Rust)** is the core crate with `i64` values and integer keys.
  The Python package goes through the Arrow layer, which handles any value
  type. For numbers without nulls its operators use the same single-pass
  loop as the core, so the two are close; coverage by key also hashes the
  string keys, which come from a pandas DataFrame column.
- ¹ pyrle's `mean()` is quick because it ignores run lengths, which makes it
  wrong: see [Coming from pyrle](pyrle.md).
- ² IRanges doesn't copy slices: converting `Views` to an `RleList` takes 0 ms
  because the list still points into the original. The closest comparison is
  the sum over slices (`viewSums`), which does visit the data.
- pyrle has no batch slicing; "–" means the library has no matching
  operation.

## Running them

```sh
cargo run --release --example bench                    # runlength (Rust)
python scripts/bench.py                                # runlength (Python), and pyrle if importable
PYTHONPATH=~/code/pyrle python3.12 scripts/bench.py    # with pyrle built in place
Rscript scripts/bench_r.R                              # IRanges
```

The R script needs the Bioconductor packages S4Vectors, IRanges and
GenomicRanges.
