# Batches

An `RleBatch` holds many sequences, each identified by a key: a row of one or
more key columns of any type, such as chromosome and strand. It replaces
pyrle's `RleDict`. The sequences are stored together in shared arrays, and
operations on a batch run in parallel, one key per thread.

## Making a batch

Most batches come from `coverage_by_key` (see
[Coverage](coverage.md#one-sequence-per-key)). To build one from data you
already have, use a table with one row per run: the key columns, where each
run starts and ends, and its value. Any polars or pandas DataFrame, pyarrow
Table or dict of columns works:

```python
>>> import runlength as rl
>>> import polars as pl
>>> runs = pl.DataFrame({
...     "chrom": ["chr1", "chr1", "chr1", "chr2"],
...     "strand": ["+", "+", "-", "+"],
...     "start": [0, 3, 0, 0],
...     "end": [3, 5, 4, 2],
...     "value": [1, 2, 7, 0],
... })
>>> batch = rl.RleBatch.from_table(runs, ["chrom", "strand"])
>>> batch
RleBatch(keys=3, runs=4, dtype=Int64)
  ('chr1', '+'): length=5, runs=[3, 2], values=[1, 2]
  ('chr1', '-'): length=4, runs=[4], values=[7]
  ('chr2', '+'): length=2, runs=[2], values=[0]
```

The whole table is read in one pass, without making a Python object per
sequence, so this is the fast way to build a batch with many keys. Keys
appear in the order they first occur. A key's rows don't have to be next to
each other, but they must be in position order.

Each key's runs must start at 0 and follow each other without gaps; a gap is
an error, since `from_table` can't know what lies in it:

```python
>>> rl.RleBatch.from_table(runs.filter(pl.col("end") != 3), ["chrom", "strand"])
Traceback (most recent call last):
    ...
ValueError: row 0 starts at 3, expected 0: each key's runs must start at 0 and follow each other without gaps
```

For intervals with gaps, such as a bedGraph, use `coverage_by_key`
instead, which fills them (with 0 by default).

The column names default to those of `to_polars()`, so a batch survives the
round trip, for example through a Parquet file:

```python
>>> rl.RleBatch.from_table(batch.to_polars(), ["chrom", "strand"]).to_polars().equals(batch.to_polars())
True
```

Other names are given with `values`, `starts` and `ends`. If the table has
only ends, pass `starts=None`; if it has run lengths, pass their column as
`runs`:

```python
>>> lengths = pl.DataFrame({"contig": ["a", "a", "b"], "width": [2, 3, 4], "base": ["G", "T", "A"]})
>>> rl.RleBatch.from_table(lengths, "contig", values="base", runs="width")
RleBatch(keys=2, runs=3, dtype=Utf8View)
  ('a',): length=5, runs=[2, 3], values=["G", "T"]
  ('b',): length=4, runs=[4], values=["A"]
```

If you already have the sequences, `from_rles` takes a key table and a list
with one sequence per key row:

```python
>>> batch = rl.RleBatch.from_rles(
...     {"chrom": ["chr1", "chr1", "chr2"], "strand": ["+", "-", "+"]},
...     [rl.Rle([3, 2], [1, 2]), rl.Rle([4], [7]), rl.Rle([2, 2], [0, 5])],
... )
>>> batch
RleBatch(keys=3, runs=5, dtype=Int64)
  ('chr1', '+'): length=5, runs=[3, 2], values=[1, 2]
  ('chr1', '-'): length=4, runs=[4], values=[7]
  ('chr2', '+'): length=4, runs=[2, 2], values=[0, 5]
```

The repr shows each key, then the length, runs and values of its sequence.

## Getting sequences out

Look a sequence up by key with `get`, as a tuple with one value per key
column. With a single key column, the value alone also works.

```python
>>> batch.get(("chr1", "-"))
Rle(length=4, runs=1, dtype=Int64)
  runs:   [4]
  values: [7]
>>> batch.get(("chr3", "+")) is None
True
```

`len`, indexing by position and `items` work as for a list of
`(key, sequence)` pairs:

```python
>>> len(batch)
3
>>> batch[0]
Rle(length=5, runs=2, dtype=Int64)
  runs:   [3, 2]
  values: [1, 2]
>>> [key for key, rle in batch.items()]
[('chr1', '+'), ('chr1', '-'), ('chr2', '+')]
```

The keys are a table, and the whole batch converts to a table with one row
per run:

```python
>>> import polars as pl
>>> pl.DataFrame(batch.keys)
shape: (3, 2)
┌───────┬────────┐
│ chrom ┆ strand │
│ ---   ┆ ---    │
│ str   ┆ str    │
╞═══════╪════════╡
│ chr1  ┆ +      │
│ chr1  ┆ -      │
│ chr2  ┆ +      │
└───────┴────────┘
>>> batch.to_polars()
shape: (5, 5)
┌───────┬────────┬───────┬─────┬───────┐
│ chrom ┆ strand ┆ start ┆ end ┆ value │
│ ---   ┆ ---    ┆ ---   ┆ --- ┆ ---   │
│ str   ┆ str    ┆ i64   ┆ i64 ┆ i64   │
╞═══════╪════════╪═══════╪═════╪═══════╡
│ chr1  ┆ +      ┆ 0     ┆ 3   ┆ 1     │
│ chr1  ┆ +      ┆ 3     ┆ 5   ┆ 2     │
│ chr1  ┆ -      ┆ 0     ┆ 4   ┆ 7     │
│ chr2  ┆ +      ┆ 0     ┆ 2   ┆ 0     │
│ chr2  ┆ +      ┆ 2     ┆ 4   ┆ 5     │
└───────┴────────┴───────┴─────┴───────┘
```

## Reductions

Reductions give one row per key:

```python
>>> pl.DataFrame(batch.sum())
shape: (3, 3)
┌───────┬────────┬─────┐
│ chrom ┆ strand ┆ sum │
│ ---   ┆ ---    ┆ --- │
│ str   ┆ str    ┆ i64 │
╞═══════╪════════╪═════╡
│ chr1  ┆ +      ┆ 7   │
│ chr1  ┆ -      ┆ 28  │
│ chr2  ┆ +      ┆ 10  │
└───────┴────────┴─────┘
>>> pl.DataFrame(batch.mean())
shape: (3, 3)
┌───────┬────────┬──────┐
│ chrom ┆ strand ┆ mean │
│ ---   ┆ ---    ┆ ---  │
│ str   ┆ str    ┆ f64  │
╞═══════╪════════╪══════╡
│ chr1  ┆ +      ┆ 1.4  │
│ chr1  ┆ -      ┆ 7.0  │
│ chr2  ┆ +      ┆ 2.5  │
└───────┴────────┴──────┘
```

`var`, `std`, `min` and `max` work the same way.

## Combining batches

Operators combine two batches key by key, and a batch with a single value:

```python
>>> other = rl.RleBatch.from_rles(
...     {"chrom": ["chr2", "chr3"], "strand": ["+", "+"]},
...     [rl.Rle([4], [1]), rl.Rle([2], [9])],
... )
>>> batch + other
RleBatch(keys=4, runs=6, dtype=Int64)
  ('chr1', '+'): length=5, runs=[3, 2], values=[1, 2]
  ('chr1', '-'): length=4, runs=[4], values=[7]
  ('chr2', '+'): length=4, runs=[2, 2], values=[1, 6]
  ('chr3', '+'): length=2, runs=[2], values=[9]
>>> batch * 10
RleBatch(keys=3, runs=5, dtype=Int64)
  ('chr1', '+'): length=5, runs=[3, 2], values=[10, 20]
  ('chr1', '-'): length=4, runs=[4], values=[70]
  ('chr2', '+'): length=4, runs=[2, 2], values=[0, 50]
```

Operators use an outer join: the result has every key from either side. A
key missing on one side counts as an empty sequence there, padded with zero.
Keys come in the left batch's order, followed by keys found only on the
right.

`zip_with` takes any function, as for [single sequences](combining.md), and
lets you choose the join with `how`: `"outer"`, `"inner"`, `"left"` or
`"right"`. Fill values say what to pad with; without them, different lengths
or missing keys are an error.

```python
>>> batch.zip_with(other, lambda x, y: x * y, how="inner", backend="numpy")
RleBatch(keys=1, runs=2, dtype=Int64)
  ('chr2', '+'): length=4, runs=[2, 2], values=[0, 5]
>>> batch.zip_with(other, lambda x, y: x - y, how="left", fill_other=0, backend="numpy")
RleBatch(keys=3, runs=5, dtype=Int64)
  ('chr1', '+'): length=5, runs=[3, 2], values=[1, 2]
  ('chr1', '-'): length=4, runs=[4], values=[7]
  ('chr2', '+'): length=4, runs=[2, 2], values=[-1, 4]
```

## Collapsing keys

`collapse` combines the sequences whose keys agree on the remaining columns.
Name the columns to `drop` or to `keep`:

```python
>>> batch.collapse(drop="strand")
RleBatch(keys=2, runs=5, dtype=Int64)
  ('chr1',): length=5, runs=[3, 1, 1], values=[8, 9, 2]
  ('chr2',): length=4, runs=[2, 2], values=[0, 5]
>>> batch.collapse(keep=["chrom"], op="mul")
RleBatch(keys=2, runs=5, dtype=Int64)
  ('chr1',): length=5, runs=[3, 1, 1], values=[7, 14, 0]
  ('chr2',): length=4, runs=[2, 2], values=[0, 5]
```

`op` is `"add"` (the default), `"sub"`, `"mul"`, `"div"`, `"and"` or `"or"`,
and shorter sequences are padded with zero.

## Other operations

`map`, `shift`, `reverse` and `slice_agg` work as for single sequences, on
every key at once. `rl.concat` joins batches key by key:

```python
>>> batch.map(lambda v: v > 1, backend="numpy")
RleBatch(keys=3, runs=5, dtype=Boolean)
  ('chr1', '+'): length=5, runs=[3, 2], values=[false, true]
  ('chr1', '-'): length=4, runs=[4], values=[true]
  ('chr2', '+'): length=4, runs=[2, 2], values=[false, true]
>>> rl.concat([batch, other])
RleBatch(keys=4, runs=7, dtype=Int64)
  ('chr1', '+'): length=5, runs=[3, 2], values=[1, 2]
  ('chr1', '-'): length=4, runs=[4], values=[7]
  ('chr2', '+'): length=8, runs=[2, 2, 4], values=[0, 5, 1]
  ('chr3', '+'): length=2, runs=[2], values=[9]
```
