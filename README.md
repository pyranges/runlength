# runlength

Run-length encoded sequences of any type, in Rust and Python.

A run-length encoded sequence stores each run of repeated values once:
`1 1 1 2 2` becomes the runs `[3, 2]` with the values `[1, 2]`. Long
sequences with few changes, such as read coverage along a genome, become
small, and arithmetic on them is fast because it works run by run.

runlength is the successor to [pyrle](https://github.com/pyranges/pyrle). It
holds values of any type (numbers, strings, booleans, dates, with nulls),
combines sequences with any function, and keeps many sequences under keys of
any type, processed in parallel. In Python it works with polars, pandas,
numpy and pyarrow, and requires none of them.

**Documentation: <https://pyranges.github.io/runlength/>**

```sh
pip install runlength
cargo add runlength
```

## Python

```python
>>> import runlength as rl
>>> a = rl.Rle([3, 2], [1, 2])        # the sequence 1 1 1 2 2
>>> a
Rle(length=5, runs=2, dtype=Int64)
  runs:   [3, 2]
  values: [1, 2]
>>> b = rl.Rle([1, 4], [10, 20])      # 10 20 20 20 20
>>> a + b
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
>>> a.mean(), a[3]
(1.4, 2)
>>> a[1:4]                             # slices start at 0
Rle(length=3, runs=2, dtype=Int64)
  runs:   [2, 1]
  values: [1, 2]
```

Coverage turns intervals into a sequence, here one per chromosome and
strand, and back into intervals:

```python
>>> import polars as pl
>>> reads = pl.DataFrame({
...     "chrom": ["chr1", "chr1", "chr2"],
...     "strand": ["+", "-", "+"],
...     "start": [0, 2, 5],
...     "end": [3, 4, 7],
... })
>>> cov = rl.coverage_by_key(reads.select("chrom", "strand"), reads["start"], reads["end"])
>>> cov
RleBatch(keys=3, runs=5, dtype=Int64)
  ('chr1', '+'): length=3, runs=[3], values=[1]
  ('chr1', '-'): length=4, runs=[2, 2], values=[0, 1]
  ('chr2', '+'): length=7, runs=[5, 2], values=[0, 1]
>>> per_chrom = cov.collapse(drop="strand")       # add up the strands
>>> per_chrom.to_polars().filter(pl.col("value") > 0)
shape: (4, 4)
┌───────┬───────┬─────┬───────┐
│ chrom ┆ start ┆ end ┆ value │
│ ---   ┆ ---   ┆ --- ┆ ---   │
│ str   ┆ i64   ┆ i64 ┆ i64   │
╞═══════╪═══════╪═════╪═══════╡
│ chr1  ┆ 0     ┆ 2   ┆ 1     │
│ chr1  ┆ 2     ┆ 3   ┆ 2     │
│ chr1  ┆ 3     ┆ 4   ┆ 1     │
│ chr2  ┆ 5     ┆ 7   ┆ 1     │
└───────┴───────┴─────┴───────┘
```

Mean coverage over many intervals at once, without copying them:

```python
>>> genes = pl.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 4], "end": [4, 7]})
>>> per_chrom.slice_agg(genes.select("chrom"), genes["start"], genes["end"], agg="mean").to_pylist()
[1.25, 0.6666666666666666]
```

Any function can combine sequences, written with numpy, polars or pyarrow, on
any value type. runlength works out where the output runs are; your function
only sees one value per run:

```python
>>> names = rl.Rle([3, 2], ["x", "y"])
>>> names.zip_with(b, lambda n, v: n + "=" + v.cast(pl.String), backend="polars")
Rle(length=5, runs=3, dtype=Utf8View)
  runs:   [1, 2, 2]
  values: ["x=10", "x=20", "y=20"]
```

## Rust

```rust
use runlength::prelude::*;

fn main() -> runlength::Result<()> {
    let a = Rle::from_runs(&[3, 2], vec![1, 2])?;
    let b = Rle::from_runs(&[1, 4], vec![10, 20])?;
    assert_eq!(format!("{:?}", &a + &b), "Rle { runs: [1, 2, 2], values: [11, 21, 22] }");

    // The plan says where the output runs end, and which run of each input
    // covers each one, without looking at values...
    let plan = a.plan_with(&b);
    assert_eq!(plan.ends, vec![1, 3, 5]);
    assert_eq!((plan.idx_a, plan.idx_b), (vec![0, 0, 1], vec![0, 1, 1]));

    // ...so any function works on any value types.
    let labels = a.zip_with(&b, None, None, |x, y| format!("{x}/{y}"))?;
    assert_eq!(labels.values(), &["1/10", "1/20", "2/20"]);
    Ok(())
}
```

| Crate | Contents |
|---|---|
| [`runlength`](https://docs.rs/runlength) | The core: `Rle<V>` and `RleBatch<K, V>` for any value and key types, plans, merging, coverage. Depends only on rayon and rustc-hash. |
| [`runlength-arrow`](https://docs.rs/runlength-arrow) | The same over Apache Arrow arrays: values of any Arrow type, batch keys as Arrow tables. |
| `runlength-py` | The `runlength` Python package (pyo3). |

## Benchmarks

The Python package against Bioconductor's
[IRanges](https://bioconductor.org/packages/IRanges/) (R), the standard
run-length encoding in genomics, on 1M random intervals over a sequence of
length 100M (about 2M runs). Best of three runs, on an Apple M3 Pro.

| Operation | runlength | IRanges | Speed-up |
|---|--:|--:|--:|
| Coverage of 1M intervals | 38 ms | 334 ms | 8.8× |
| `a + b` (3.9M runs out) | 20 ms | 197 ms | 9.9× |
| `a < b` | 18 ms | 139 ms | 7.7× |
| `a.mean()` | 2 ms | 18 ms | 9× |
| Sum over 1M slices of length 1000 | 34 ms | 125 ms | 3.7× |
| Coverage of 1M intervals over 25 keys | 34 ms | 351 ms | 10× |
| `batch + batch`, 25 keys | 10 ms | 59 ms | 5.9× |

The Rust core is faster still, and pyrle slower; see the
[benchmarks page](https://pyranges.github.io/runlength/benchmarks/) for all
of them and how to run them.

## Development

```sh
cargo test                                        # unit, property and doc tests
cargo run --example tour                          # the examples on the docs' Rust page
PROPTEST_CASES=5000 cargo test --release          # more property-test cases
cd crates/runlength-py && maturin develop --release && cd ../..
pytest crates/runlength-py/tests                  # including every example in the docs
mkdocs serve                                      # the documentation site
```

The examples in this README, `docs/` and the Python docstrings are tested.
After changing one (or a repr), regenerate their output, and after changing a
doc comment in `crates/runlength-py/src`, copy it into the type stub:

```sh
python scripts/update_doc_examples.py
python scripts/sync_stub.py
```

Benchmarks: `cargo run --release --example bench`, `python scripts/bench.py`
and `Rscript scripts/bench_r.R`. `scripts/gen_pyrle_fixtures.py` regenerates
the pyrle cross-check cases, and `scripts/pyranges_port.py` checks the port of
pyranges' `to_rle` and `to_bigwig`.

## Releasing

Versions of all three crates move together. Bump `version` in each
`crates/*/Cargo.toml` (and the `runlength` version that `runlength-arrow`
requires), commit, then push a tag:

```sh
git tag py-v0.2.0 && git push origin py-v0.2.0       # wheels + sdist to PyPI
git tag rust-v0.2.0 && git push origin rust-v0.2.0   # runlength, runlength-arrow to crates.io
```

`.github/workflows/release.yml` checks that the tag matches the crate
versions, builds wheels for Linux (x86_64 and aarch64, glibc and musl), macOS
(universal2) and Windows, tests them, and publishes. Run it by hand from the
Actions tab to build and test the wheels without publishing.
