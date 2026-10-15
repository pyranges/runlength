# Homopolymers

A homopolymer is a run of one repeated base, such as `AAAAA`. Long
homopolymers cause insertion and deletion errors in most sequencing
technologies, so variant callers often mask them. Run-length encoding a DNA
sequence finds every homopolymer directly: each run *is* one.

## Finding them

Encode the sequence one base per position:

```python
>>> import numpy as np
>>> import polars as pl
>>> import runlength as rl
>>> seq = "GATTTTACCCCCGGAAAAATTGCAAAAAAAC"
>>> bases = rl.Rle.from_dense(list(seq))
>>> bases
Rle(length=31, runs=12, dtype=Utf8)
  runs:   [1, 1, 4, 1, 5, ..., 2, 1, 1, 7, 1]
  values: ["G", "A", "T", "A", "C", ..., "T", "G", "C", "A", "C"]
```

As a table, each row is one homopolymer with its position and base. Keep the
ones of length 5 or more:

```python
>>> runs = pl.DataFrame(bases).with_columns(length=pl.col("end") - pl.col("start"))
>>> runs.filter(pl.col("length") >= 5)
shape: (3, 4)
┌───────┬─────┬───────┬────────┐
│ start ┆ end ┆ value ┆ length │
│ ---   ┆ --- ┆ ---   ┆ ---    │
│ i64   ┆ i64 ┆ str   ┆ i64    │
╞═══════╪═════╪═══════╪════════╡
│ 7     ┆ 12  ┆ C     ┆ 5      │
│ 14    ┆ 19  ┆ A     ┆ 5      │
│ 23    ┆ 30  ┆ A     ┆ 7      │
└───────┴─────┴───────┴────────┘
```

That table is already in BED's coordinates (0-based, half-open), so it can be
written out as a mask for a variant caller.

## The length distribution

How many homopolymers of each length there are, and per base:

```python
>>> runs.group_by("length").len().sort("length")
shape: (5, 2)
┌────────┬─────┐
│ length ┆ len │
│ ---    ┆ --- │
│ i64    ┆ u32 │
╞════════╪═════╡
│ 1      ┆ 6   │
│ 2      ┆ 2   │
│ 4      ┆ 1   │
│ 5      ┆ 2   │
│ 7      ┆ 1   │
└────────┴─────┘
>>> runs.filter(pl.col("length") >= 3).group_by("value").agg(
...     count=pl.len(), longest=pl.col("length").max()
... ).sort("value")
shape: (3, 3)
┌───────┬───────┬─────────┐
│ value ┆ count ┆ longest │
│ ---   ┆ ---   ┆ ---     │
│ str   ┆ u32   ┆ i64     │
╞═══════╪═══════╪═════════╡
│ A     ┆ 2     ┆ 7       │
│ C     ┆ 1     ┆ 5       │
│ T     ┆ 1     ┆ 4       │
└───────┴───────┴─────────┘
```

## Whole chromosomes

Passing a chromosome as a list of one-letter strings is slow, because every
base becomes a Python object. Pass its bytes as a numpy array instead: each
base becomes its ASCII code, and a 50-million-base chromosome encodes in about
a quarter of a second.

```python
>>> codes = np.frombuffer(seq.encode(), dtype=np.uint8)
>>> bases = rl.Rle.from_dense(codes)
>>> bases.values.to_pylist()[:4]
[71, 65, 84, 65]
```

Turn the codes back into letters when you're done:

```python
>>> letters = np.asarray(bases.values).view("S1").astype(str)
>>> letters[:4]
array(['G', 'A', 'T', 'A'], dtype='<U1')
>>> runs = pl.DataFrame({
...     "start": bases.starts,
...     "end": bases.ends,
...     "base": letters,
... }).filter(pl.col("end") - pl.col("start") >= 5)
>>> runs
shape: (3, 3)
┌───────┬─────┬──────┐
│ start ┆ end ┆ base │
│ ---   ┆ --- ┆ ---  │
│ i64   ┆ i64 ┆ str  │
╞═══════╪═════╪══════╡
│ 7     ┆ 12  ┆ C    │
│ 14    ┆ 19  ┆ A    │
│ 23    ┆ 30  ┆ A    │
└───────┴─────┴──────┘
```

For several contigs, encode each one and keep them together in a batch,
keyed by contig name. One table then covers all of them:

```python
>>> contigs = {"chr1": "GATTTTACCCCCG", "chr2": "AAAAAAGT"}
>>> batch = rl.RleBatch.from_rles(
...     {"chrom": list(contigs)},
...     [rl.Rle.from_dense(np.frombuffer(s.encode(), dtype=np.uint8)) for s in contigs.values()],
... )
>>> batch.to_polars().filter(pl.col("end") - pl.col("start") >= 4).with_columns(
...     base=pl.col("value").replace_strict({65: "A", 67: "C", 71: "G", 84: "T"})
... )
shape: (3, 5)
┌───────┬───────┬─────┬───────┬──────┐
│ chrom ┆ start ┆ end ┆ value ┆ base │
│ ---   ┆ ---   ┆ --- ┆ ---   ┆ ---  │
│ str   ┆ i64   ┆ i64 ┆ u8    ┆ str  │
╞═══════╪═══════╪═════╪═══════╪══════╡
│ chr1  ┆ 2     ┆ 6   ┆ 84    ┆ T    │
│ chr1  ┆ 7     ┆ 12  ┆ 67    ┆ C    │
│ chr2  ┆ 0     ┆ 6   ┆ 65    ┆ A    │
└───────┴───────┴─────┴───────┴──────┘
```

## Masking around homopolymers

Errors near a homopolymer are more common too, so masks often extend a few
bases past each end. A boolean track of "inside a long homopolymer" can be
widened with `shift`: a position is masked if it, or a neighbouring
position, lies in one.

```python
>>> bases = rl.Rle.from_dense(list(seq))
>>> long_runs = rl.Rle.from_ends(bases.ends, np.asarray(bases.runs) >= 5)
>>> masked = long_runs | long_runs.shift(1) | long_runs.shift(-1)
>>> starts, ends = masked.ranges()
>>> list(zip(starts.to_pylist(), ends.to_pylist()))
[(6, 20), (22, 31)]
```
