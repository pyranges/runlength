# runlength

Run-length encoded sequences of any type, in Rust and Python.

A run-length encoded sequence stores each run of repeated values once:
`1 1 1 2 2` becomes the runs `[3, 2]` with the values `[1, 2]`. Long
sequences with few changes, such as read coverage along a genome, become
small, and arithmetic on them is fast because it works run by run.

```python
>>> import runlength as rl
>>> rl.Rle([3, 2], [1, 2])
Rle(length=5, runs=2, dtype=Int64)
  runs:   [3, 2]
  values: [1, 2]
>>> rl.coverage(starts=[0, 2, 8], ends=[3, 4, 10])
Rle(length=10, runs=5, dtype=Int64)
  runs:   [2, 1, 1, 4, 2]
  values: [1, 2, 1, 0, 1]
```

runlength is the successor to [pyrle](https://github.com/pyranges/pyrle), and
computes coverage for [pyranges](https://github.com/pyranges/pyranges1). It
differs from pyrle in three ways:

- **Any value type.** Values can be numbers, booleans, strings, dates, or
  anything else Apache Arrow can hold, with nulls. pyrle only held `float64`.
- **Plans.** Combining two sequences is split into a *plan*, which works out
  where the output runs end from the run boundaries alone, and a separate
  step that computes the values. So any function can combine sequences,
  written in polars, numpy, pyarrow or plain Python. See
  [Combining sequences](guide/combining.md).
- **Batches.** An `RleBatch` holds many sequences under keys of any type, for
  example one per chromosome and strand, and combines batches key by key in
  parallel.

It doesn't depend on polars, pandas or pyarrow, but accepts input from all of
them and numpy, and its results convert to any of them without copying.

## Install

```sh
pip install runlength              # Python 3.11 or newer
cargo add runlength                # Rust: the generic core
cargo add runlength-arrow          # Rust: the same over Arrow arrays
```

Wheels are available for Linux, macOS and Windows.

## Where to go next

- [Tutorial](tutorial.md): the basics in ten minutes.
- User guide: [combining sequences](guide/combining.md),
  [coverage and intervals](guide/coverage.md), [batches](guide/batches.md),
  [merging runs](guide/merging.md),
  [polars, pandas, numpy and pyarrow](guide/interop.md), and
  [working with plans](guide/plans.md) for advanced use.
- Genomics recipes: [homopolymers](genomics/homopolymers.md),
  [comparing chromatin states](genomics/chromatin-states.md), and
  [annotation tracks and transcripts](genomics/transcripts.md).
- [Coming from pyrle](pyrle.md): what changed, with a translation table.
- [Benchmarks](benchmarks.md) against Bioconductor's IRanges and pyrle.
- [Rust](rust.md): using the crates directly.
- [API reference](reference.md) for the Python package.
