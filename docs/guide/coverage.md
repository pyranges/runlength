# Coverage and intervals

Coverage turns a set of intervals into a sequence: at each position, how
many intervals cover it, or a sum, maximum or minimum of a value per
interval. In genomics this is read depth or a signal track, but nothing here
is specific to genomics.

Intervals are half-open: `[start, end)` covers `start` up to but not
including `end`, as in BED files and Python slices.

## One sequence

```python
>>> import runlength as rl
>>> starts, ends = [0, 2, 8], [3, 4, 10]
>>> rl.coverage(starts, ends)
Rle(length=10, runs=5, dtype=Int64)
  runs:   [2, 1, 1, 4, 2]
  values: [1, 2, 1, 0, 1]
```

The result covers positions `0` up to the largest end. Pass `length` to make
it longer, for example the length of a chromosome:

```python
>>> rl.coverage(starts, ends, length=12)
Rle(length=12, runs=6, dtype=Int64)
  runs:   [2, 1, 1, 4, 2, 2]
  values: [1, 2, 1, 0, 1, 0]
```

With `values`, coverage sums one value per interval. `agg` picks another way
to combine them: `"max"`, `"min"` or `"count"`.

```python
>>> scores = [0.5, 2.0, 1.0]
>>> rl.coverage(starts, ends, values=scores)
Rle(length=10, runs=5, dtype=Float64)
  runs:   [2, 1, 1, 4, 2]
  values: [0.5, 2.5, 2.0, 0.0, 1.0]
>>> rl.coverage(starts, ends, values=scores, agg="max")
Rle(length=10, runs=4, dtype=Float64)
  runs:   [2, 2, 4, 2]
  values: [0.5, 2.0, 0.0, 1.0]
```

Positions that no interval covers hold `fill`, which is 0 by default.
`fill=None` leaves them null, which tells "no data" apart from a real zero:

```python
>>> rl.coverage(starts, ends, values=scores, agg="max", fill=None)
Rle(length=10, runs=4, dtype=Float64)
  runs:   [2, 2, 4, 2]
  values: [0.5, 2.0, null, 1.0]
```

`"max"` and `"min"` work on strings too, and ignore NaN.

## One sequence per key

`coverage_by_key` takes key columns with one row per interval, and computes
one sequence per distinct key, in parallel. Keys can be a polars or pandas
DataFrame, a pyarrow Table, or a dict of columns, and can be of any type:

```python
>>> import polars as pl
>>> reads = pl.DataFrame({
...     "chrom": ["chr1", "chr1", "chr1", "chr2"],
...     "strand": ["+", "+", "-", "+"],
...     "start": [0, 2, 8, 5],
...     "end": [3, 4, 10, 7],
... })
>>> cov = rl.coverage_by_key(reads.select("chrom", "strand"), reads["start"], reads["end"])
>>> cov
RleBatch(keys=3, runs=7, dtype=Int64)
  ('chr1', '+'): length=4, runs=[2, 1, 1], values=[1, 2, 1]
  ('chr1', '-'): length=10, runs=[8, 2], values=[0, 1]
  ('chr2', '+'): length=7, runs=[5, 2], values=[0, 1]
```

Keys appear in the order they first appear in. Each sequence runs from 0 to
the largest end of its own intervals. It takes `values`, `agg` and `fill` like
`coverage`. The result is an `RleBatch`; see [Batches](batches.md).

To add up the strands, collapse the batch:

```python
>>> cov.collapse(drop="strand")
RleBatch(keys=2, runs=7, dtype=Int64)
  ('chr1',): length=10, runs=[2, 1, 1, 4, 2], values=[1, 2, 1, 0, 1]
  ('chr2',): length=7, runs=[5, 2], values=[0, 1]
```

## From coverage back to intervals

`ranges` gives the stretches where a boolean sequence is true. Combined with
a comparison, it finds regions by their coverage, for example where at least
two intervals overlap:

```python
>>> per_chrom = cov.collapse(drop="strand")
>>> starts, ends = (per_chrom.get("chr1") >= 2).ranges()
>>> starts.to_pylist(), ends.to_pylist()
([2], [3])
```

Or as a table of every run, keys included, to filter with polars:

```python
>>> per_chrom.to_polars().filter(pl.col("value") > 0)
shape: (5, 4)
┌───────┬───────┬─────┬───────┐
│ chrom ┆ start ┆ end ┆ value │
│ ---   ┆ ---   ┆ --- ┆ ---   │
│ str   ┆ i64   ┆ i64 ┆ i64   │
╞═══════╪═══════╪═════╪═══════╡
│ chr1  ┆ 0     ┆ 2   ┆ 1     │
│ chr1  ┆ 2     ┆ 3   ┆ 2     │
│ chr1  ┆ 3     ┆ 4   ┆ 1     │
│ chr1  ┆ 8     ┆ 10  ┆ 1     │
│ chr2  ┆ 5     ┆ 7   ┆ 1     │
└───────┴───────┴─────┴───────┘
```

## Statistics over many intervals

`slice_agg` computes a sum, count, mean, variance, standard deviation,
minimum or maximum over many slices at once, without copying them. It runs in
parallel, and is much faster than slicing in a loop:

```python
>>> track = rl.coverage(starts=[0, 2, 8], ends=[3, 4, 10], values=[0.5, 2.0, 1.0])
>>> track.slice_agg(starts=[0, 4, 0], ends=[4, 10, 10], agg="mean").to_pylist()
[1.375, 0.3333333333333333, 0.75]
>>> track.slice_agg(starts=[0, 4, 0], ends=[4, 10, 10], agg="max").to_pylist()
[2.5, 1.0, 2.5]
```

On a batch, each slice also names its key, here one row per gene:

```python
>>> genes = pl.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 4], "end": [4, 7]})
>>> per_chrom.slice_agg(genes.select("chrom"), genes["start"], genes["end"], agg="mean").to_pylist()
[1.25, 0.6666666666666666]
```

A key that isn't in the batch gives an empty slice, so its sum is 0 and its
mean is null.

If you need the slices themselves, `getitems` returns them as a batch keyed
by slice number:

```python
>>> track.getitems(starts=[0, 4], ends=[4, 10])
RleBatch(keys=2, runs=5, dtype=Float64)
  (0,): length=4, runs=[2, 1, 1], values=[0.5, 2.5, 2.0]
  (1,): length=6, runs=[4, 2], values=[0.0, 1.0]
```

## Any other aggregation: overlap plans

`coverage` handles sums, counts, maxima and minima. For anything else, such
as a median, a weighted mean, or the set of interval names at each position,
`overlap_plan` says which intervals cover each segment, and you aggregate
however you like:

```python
>>> seg_ends, offsets, ids = rl.overlap_plan([0, 2, 8], [3, 4, 10])
>>> seg_ends.to_pylist()
[2, 3, 4, 8, 10]
>>> offsets.to_pylist(), ids.to_pylist()
([0, 1, 3, 4, 4, 5], [0, 0, 1, 1, 2])
```

Segment `s` ends at `seg_ends[s]` and is covered by the intervals
`ids[offsets[s]:offsets[s + 1]]`: segment 1 (positions 2 up to 3) by
intervals 0 and 1, and segment 3 (positions 4 up to 8) by none.

`overlap_plan_by_key` does the same per key and returns one table, with the
covering intervals as a list column of row numbers. polars can look up the
interval values through those row numbers and aggregate them, here the names
and the median score of the intervals covering each segment:

```python
>>> peaks = pl.DataFrame({
...     "chrom": ["chr1", "chr1", "chr1"],
...     "start": [0, 2, 8],
...     "end": [3, 4, 10],
...     "name": ["a", "b", "c"],
...     "score": [1.0, 4.0, 2.0],
... })
>>> plan = pl.DataFrame(rl.overlap_plan_by_key(peaks.select("chrom"), peaks["start"], peaks["end"]))
>>> plan
shape: (5, 4)
┌───────┬───────┬─────┬───────────┐
│ chrom ┆ start ┆ end ┆ ids       │
│ ---   ┆ ---   ┆ --- ┆ ---       │
│ str   ┆ i64   ┆ i64 ┆ list[i64] │
╞═══════╪═══════╪═════╪═══════════╡
│ chr1  ┆ 0     ┆ 2   ┆ [0]       │
│ chr1  ┆ 2     ┆ 3   ┆ [0, 1]    │
│ chr1  ┆ 3     ┆ 4   ┆ [1]       │
│ chr1  ┆ 4     ┆ 8   ┆ []        │
│ chr1  ┆ 8     ┆ 10  ┆ [2]       │
└───────┴───────┴─────┴───────────┘
>>> def lookup(column):
...     return pl.col("ids").list.eval(pl.lit(peaks[column]).gather(pl.element()))
>>> plan.select(
...     "chrom", "start", "end",
...     names=lookup("name"),
...     median=lookup("score").list.median(),
... )
shape: (5, 5)
┌───────┬───────┬─────┬────────────┬────────┐
│ chrom ┆ start ┆ end ┆ names      ┆ median │
│ ---   ┆ ---   ┆ --- ┆ ---        ┆ ---    │
│ str   ┆ i64   ┆ i64 ┆ list[str]  ┆ f64    │
╞═══════╪═══════╪═════╪════════════╪════════╡
│ chr1  ┆ 0     ┆ 2   ┆ ["a"]      ┆ 1.0    │
│ chr1  ┆ 2     ┆ 3   ┆ ["a", "b"] ┆ 2.5    │
│ chr1  ┆ 3     ┆ 4   ┆ ["b"]      ┆ 4.0    │
│ chr1  ┆ 4     ┆ 8   ┆ []         ┆ null   │
│ chr1  ┆ 8     ┆ 10  ┆ ["c"]      ┆ 2.0    │
└───────┴───────┴─────┴────────────┴────────┘
```

Segments that no interval covers get an empty list, and a null median.
