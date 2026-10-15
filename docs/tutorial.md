# Tutorial

This page covers what most users need: making sequences, looking inside
them, arithmetic, reductions, slicing and getting the results out. The
examples assume `import runlength as rl`.

## Making a sequence

An `Rle` is made from run lengths and one value per run:

```python
>>> import runlength as rl
>>> r = rl.Rle([3, 2, 4], [1.0, 2.5, 1.0])
>>> r
Rle(length=9, runs=3, dtype=Float64)
  runs:   [3, 2, 4]
  values: [1.0, 2.5, 1.0]
```

The repr shows the length (the number of positions), the number of runs and
the value type, then the runs and their values. `r` is the sequence
`1.0 1.0 1.0 2.5 2.5 1.0 1.0 1.0 1.0`:

```python
>>> r.to_dense().to_pylist()
[1.0, 1.0, 1.0, 2.5, 2.5, 1.0, 1.0, 1.0, 1.0]
```

From one value per position, `from_dense` finds the runs:

```python
>>> rl.Rle.from_dense(["a", "a", "b", None, None])
Rle(length=5, runs=3, dtype=Utf8)
  runs:   [2, 1, 2]
  values: ["a", "b", null]
```

Values can be any type, including strings, booleans and nulls. Run lengths and
values can be lists, numpy arrays, or polars, pandas or pyarrow columns.

## Looking inside

```python
>>> len(r), r.num_runs, r.dtype
(9, 3, 'Float64')
>>> r.runs.to_pylist()
[3, 2, 4]
>>> r.values.to_pylist()
[1.0, 2.5, 1.0]
>>> r.starts.to_pylist(), r.ends.to_pylist()
([0, 3, 5], [3, 5, 9])
```

`runs`, `values`, `starts` and `ends` are Arrow arrays. Pass them to
`pl.Series`, `np.asarray` or `pa.array` to use them in another library (no
data is copied), or call `.to_pylist()` as above. As a table, one row per
run:

```python
>>> r.to_polars()
shape: (3, 3)
┌───────┬─────┬───────┐
│ start ┆ end ┆ value │
│ ---   ┆ --- ┆ ---   │
│ i64   ┆ i64 ┆ f64   │
╞═══════╪═════╪═══════╡
│ 0     ┆ 3   ┆ 1.0   │
│ 3     ┆ 5   ┆ 2.5   │
│ 5     ┆ 9   ┆ 1.0   │
└───────┴─────┴───────┘
```

`to_pandas()` and `to_arrow()` give the same table as pandas and pyarrow.

## Positions and slices

Positions start at 0. Indexing gives the value at a position, and slicing
gives a new sequence that starts at 0:

```python
>>> r[3], r[-1]
(2.5, 1.0)
>>> r[2:6]
Rle(length=4, runs=3, dtype=Float64)
  runs:   [1, 2, 1]
  values: [1.0, 2.5, 1.0]
```

## Arithmetic and comparisons

Operators work position by position, with a single value or another sequence.
The result is merged: neighbouring runs with the same value become one run.

```python
>>> a = rl.Rle([3, 2], [1, 2])
>>> b = rl.Rle([1, 4], [10, 20])
>>> a + b
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
>>> a * 2
Rle(length=5, runs=2, dtype=Int64)
  runs:   [3, 2]
  values: [2, 4]
>>> a / 2
Rle(length=5, runs=2, dtype=Float64)
  runs:   [3, 2]
  values: [0.5, 1.0]
```

Integers divided with `/` become floats, as in Python. Comparisons give a
boolean sequence, and `&`, `|` and `~` combine those:

```python
>>> a == 1
Rle(length=5, runs=2, dtype=Boolean)
  runs:   [3, 2]
  values: [true, false]
>>> (a > 1) | (b < 20)
Rle(length=5, runs=3, dtype=Boolean)
  runs:   [1, 2, 2]
  values: [true, false, true]
```

If the two sequences have different lengths, the shorter one is padded with
zero:

```python
>>> a + rl.Rle([7], [100])
Rle(length=7, runs=3, dtype=Int64)
  runs:   [3, 2, 2]
  values: [101, 102, 100]
```

Because `==` works position by position, use `equals` to ask whether two
sequences are the same. Their runs don't need to be split the same way:

```python
>>> rl.Rle([2, 1], [1, 2]).equals(rl.Rle([1, 1, 1], [1, 1, 2]))
True
```

## Reductions

Reductions count every position, so a run of length 3 counts three times:

```python
>>> a.sum(), a.mean(), a.min(), a.max()
(7, 1.4, 1, 2)
>>> a.var(), a.std(ddof=0)
(0.3, 0.4898979485566356)
```

## Coverage

`coverage` turns intervals into a sequence: how many intervals cover each
position, or the sum of a value per interval. Intervals are half-open, so
`[0, 3)` covers positions 0, 1 and 2.

```python
>>> cov = rl.coverage(starts=[0, 2, 8], ends=[3, 4, 10])
>>> cov
Rle(length=10, runs=5, dtype=Int64)
  runs:   [2, 1, 1, 4, 2]
  values: [1, 2, 1, 0, 1]
>>> rl.coverage([0, 2, 8], [3, 4, 10], values=[0.5, 2.0, 1.0])
Rle(length=10, runs=5, dtype=Float64)
  runs:   [2, 1, 1, 4, 2]
  values: [0.5, 2.5, 2.0, 0.0, 1.0]
```

A comparison followed by `ranges` turns a sequence back into intervals, here
the stretches with coverage:

```python
>>> starts, ends = (cov > 0).ranges()
>>> starts.to_pylist(), ends.to_pylist()
([0, 8], [4, 10])
```

[Coverage and intervals](guide/coverage.md) covers this in depth, including
coverage per chromosome and statistics over many intervals at once.

## Many sequences at once

`coverage_by_key` computes one sequence per key in parallel, and returns an
`RleBatch`. The keys can be several columns of any type:

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
>>> cov.get(("chr1", "-"))
Rle(length=4, runs=2, dtype=Int64)
  runs:   [2, 2]
  values: [0, 1]
```

Operators combine two batches key by key. [Batches](guide/batches.md) has
the details.

## Next

- [Combining sequences](guide/combining.md) with your own functions.
- [Merging runs](guide/merging.md), and when runs are considered equal.
- [polars, pandas, numpy and pyarrow](guide/interop.md): what goes in and
  what comes out.
