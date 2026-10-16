# Comparing chromatin states

Tools such as ChromHMM and Segway split the genome into segments, each with a
chromatin state like *Active*, *Poised* or *Repressed*. A segmentation is a
run-length encoding of strings: every segment is a run. So comparing two
samples, for example finding regions that switch from active to repressed
during differentiation, is a matter of combining two string sequences.

## Two segmentations

Segmentations usually come as BED-like tables, one row per segment. Since the
segments tile the chromosome, each one's end and state make a sequence:

```python
>>> import polars as pl
>>> import runlength as rl
>>> stem = pl.DataFrame({
...     "start": [0, 300, 500, 900, 1200],
...     "end": [300, 500, 900, 1200, 1500],
...     "state": ["Repressed", "Poised", "Active", "Poised", "Repressed"],
... })
>>> neuron = pl.DataFrame({
...     "start": [0, 400, 700, 1000],
...     "end": [400, 700, 1000, 1500],
...     "state": ["Repressed", "Active", "Repressed", "Active"],
... })
>>> a = rl.Rle.from_ends(stem["end"], stem["state"])
>>> b = rl.Rle.from_ends(neuron["end"], neuron["state"])
>>> a
Rle(length=1500, runs=5, dtype=Utf8View)
  runs:   [300, 200, 400, 300, 300]
  values: ["Repressed", "Poised", "Active", "Poised", "Repressed"]
>>> b
Rle(length=1500, runs=4, dtype=Utf8View)
  runs:   [400, 300, 300, 500]
  values: ["Repressed", "Active", "Repressed", "Active"]
```

## Where the state changed

`!=` compares the two position by position, and `ranges` gives the regions
where they differ:

```python
>>> changed = a != b
>>> starts, ends = changed.ranges()
>>> list(zip(starts.to_pylist(), ends.to_pylist()))
[(300, 500), (700, 1500)]
```

## Which transitions happened

To see *what* changed, combine the two states into one label per position.
`zip_with` runs a function on the states of each segment where neither
sequence changes; here polars joins the strings:

```python
>>> transitions = a.zip_with(b, lambda x, y: x + " → " + y, backend="polars")
>>> transitions
Rle(length=1500, runs=8, dtype=Utf8View)
  runs:   [300, 100, 100, 200, 200, 100, 200, 300]
  values: ["Repressed → Repressed", "Poised → Repressed", "Poised → Active", "Active → Active", "Active → Repressed", "Poised → Repressed", "Poised → Active", "Repressed → Active"]
```

Each run is now a region with one transition. Count the bases of each,
leaving out regions that kept their state:

```python
>>> (
...     pl.DataFrame(transitions)
...     .with_columns(bases=pl.col("end") - pl.col("start"))
...     .filter(pl.col("value").str.split(" → ").list.n_unique() > 1)
...     .group_by("value")
...     .agg(pl.col("bases").sum(), regions=pl.len())
...     .sort(["bases", "value"], descending=[True, False])
... )
shape: (4, 3)
┌────────────────────┬───────┬─────────┐
│ value              ┆ bases ┆ regions │
│ ---                ┆ ---   ┆ ---     │
│ str                ┆ i64   ┆ u32     │
╞════════════════════╪═══════╪═════════╡
│ Poised → Active    ┆ 300   ┆ 2       │
│ Repressed → Active ┆ 300   ┆ 1       │
│ Active → Repressed ┆ 200   ┆ 1       │
│ Poised → Repressed ┆ 200   ┆ 2       │
└────────────────────┴───────┴─────────┘
```

The regions of one transition, for example the ones silenced during
differentiation:

```python
>>> silenced = transitions == "Active → Repressed"
>>> starts, ends = silenced.ranges()
>>> list(zip(starts.to_pylist(), ends.to_pylist()))
[(700, 900)]
```

## Many chromosomes

With one sequence per chromosome in a batch, the same code runs on every
chromosome at once, in parallel. `from_table` builds a batch from each
sample's table, one sequence per chromosome. The tables have no start
column, since each segment starts where the previous one ended:

```python
>>> def segmentation(df):
...     return rl.RleBatch.from_table(df, "chrom", values="state", starts=None)
>>> stem_all = segmentation(pl.DataFrame({
...     "chrom": ["chr1", "chr1", "chr2", "chr2"],
...     "end": [500, 1000, 200, 800],
...     "state": ["Active", "Poised", "Repressed", "Active"],
... }))
>>> neuron_all = segmentation(pl.DataFrame({
...     "chrom": ["chr1", "chr1", "chr2"],
...     "end": [700, 1000, 800],
...     "state": ["Active", "Repressed", "Active"],
... }))
>>> per_chrom = stem_all.zip_with(neuron_all, lambda x, y: x + " → " + y, how="inner", backend="polars")
>>> per_chrom
RleBatch(keys=2, runs=5, dtype=Utf8View)
  ('chr1',): length=1000, runs=[500, 200, 300], values=["Active → Active", "Poised → Active", "Poised → Repressed"]
  ('chr2',): length=800, runs=[200, 600], values=["Repressed → Active", "Active → Active"]
>>> per_chrom.to_polars().filter(pl.col("value").str.split(" → ").list.n_unique() > 1)
shape: (3, 4)
┌───────┬───────┬──────┬────────────────────┐
│ chrom ┆ start ┆ end  ┆ value              │
│ ---   ┆ ---   ┆ ---  ┆ ---                │
│ str   ┆ i64   ┆ i64  ┆ str                │
╞═══════╪═══════╪══════╪════════════════════╡
│ chr1  ┆ 500   ┆ 700  ┆ Poised → Active    │
│ chr1  ┆ 700   ┆ 1000 ┆ Poised → Repressed │
│ chr2  ┆ 0     ┆ 200  ┆ Repressed → Active │
└───────┴───────┴──────┴────────────────────┘
```

`how="inner"` keeps only chromosomes segmented in both samples. String
sequences have no default fill value, so if the two segmentations of a
chromosome have different lengths, `zip_with` raises an error unless you pass
`fill_self` and `fill_other`, for example `"Unknown"`.
