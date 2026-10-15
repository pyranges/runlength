# Annotation tracks and transcripts

Two things that come up when working with gene annotation: labelling every
base as exon, intron or intergenic, and looking at coverage along a spliced
transcript rather than along the genome.

The examples use one gene on the minus strand with three exons, and read
coverage over the region:

```python
>>> import numpy as np
>>> import polars as pl
>>> import runlength as rl
>>> gene = {"start": 10, "end": 60, "strand": "-"}
>>> exons = pl.DataFrame({"start": [10, 30, 50], "end": [18, 36, 60]})
>>> reads = pl.DataFrame({"start": [12, 14, 31, 33, 52, 55, 40], "end": [17, 20, 35, 36, 58, 60, 45]})
>>> cov = rl.coverage(reads["start"], reads["end"], length=70)
>>> cov
Rle(length=70, runs=15, dtype=Int64)
  runs:   [12, 2, 3, 3, 11, ..., 7, 3, 3, 2, 10]
  values: [0, 1, 2, 1, 0, ..., 0, 1, 2, 1, 0]
```

## A per-base annotation track

To label each base, give each kind of feature a priority (exon over intron
over intergenic) and take the highest priority covering each position.
`coverage` with `agg="max"` does exactly that, and positions nothing covers
get the fill value, 0:

```python
>>> starts = [gene["start"], *exons["start"]]
>>> ends = [gene["end"], *exons["end"]]
>>> priority = [1, 2, 2, 2]
>>> codes = rl.coverage(starts, ends, values=priority, agg="max", length=70)
>>> codes
Rle(length=70, runs=7, dtype=Int64)
  runs:   [10, 8, 12, 6, 14, 10, 10]
  values: [0, 2, 1, 2, 1, 2, 0]
>>> labels = np.array(["intergenic", "intron", "exon"])
>>> annotation = codes.map(lambda v: labels[v], backend="numpy")
>>> annotation
Rle(length=70, runs=7, dtype=Utf8)
  runs:   [10, 8, 12, 6, 14, 10, 10]
  values: ["intergenic", "exon", "intron", "exon", "intron", "exon", "intergenic"]
```

Combined with coverage, this answers questions about where reads fall. A
comparison and `&` mark the exonic bases that have coverage, and `sum` counts
the true positions. `zip_with` can also pick out the coverage over introns:

```python
>>> ((annotation == "exon") & (cov > 0)).sum()
19
>>> cov.zip_with(annotation, lambda c, a: np.where(a == "intron", c, 0), backend="numpy").sum()
7
```

Intronic coverage adds up to 7: five bases from the read at 40 to 45, which
lies entirely in an intron, and two from the read at 14 to 20, which runs past
the end of the first exon. [Working with plans](../guide/plans.md#segments-as-a-table)
shows how to get every annotation class at once as a table.

## Coverage along a transcript

A transcript's coordinates run through its exons only, skipping the introns.
To get coverage in transcript coordinates, cut out each exon with `getitems`
and join the pieces with `concat`:

```python
>>> pieces = cov.getitems(exons["start"], exons["end"])
>>> pieces
RleBatch(keys=3, runs=12, dtype=Int64)
  (0,): length=8, runs=[2, 2, 3, 1], values=[0, 1, 2, 1]
  (1,): length=6, runs=[1, 2, 2, 1], values=[0, 1, 2, 1]
  (2,): length=10, runs=[2, 3, 3, 2], values=[0, 1, 2, 1]
>>> transcript = rl.concat([rle for _, rle in pieces.items()])
>>> transcript
Rle(length=24, runs=12, dtype=Int64)
  runs:   [2, 2, 3, 1, 1, ..., 1, 2, 3, 3, 2]
  values: [0, 1, 2, 1, 0, ..., 1, 0, 1, 2, 1]
```

The gene is on the minus strand, so its transcript reads from right to left
along the genome. `reverse` puts position 0 at the transcript's 5' end:

```python
>>> if gene["strand"] == "-":
...     transcript = transcript.reverse()
>>> transcript
Rle(length=24, runs=12, dtype=Int64)
  runs:   [2, 3, 3, 2, 1, ..., 1, 1, 3, 2, 2]
  values: [1, 2, 1, 0, 1, ..., 0, 1, 2, 1, 0]
```

Now transcript-level questions are simple. Mean coverage over the whole
transcript, and over its first and last 5 bases, for example to spot 3' bias:

```python
>>> transcript.mean()
1.125
>>> transcript.slice_agg(starts=[0, len(transcript) - 5], ends=[5, len(transcript)], agg="mean").to_pylist()
[1.6, 0.8]
```

## Many transcripts

For a whole annotation, keep the coverage in a batch with one sequence per
chromosome, and let `slice_agg` compute a statistic for every exon at once,
without copying any of them. polars then combines the exons of each
transcript, here into a mean weighted by exon length:

```python
>>> cov_by_chrom = rl.coverage_by_key(
...     {"chrom": ["chr1"] * 7}, reads["start"], reads["end"]
... )
>>> all_exons = pl.DataFrame({
...     "transcript": ["tx1", "tx1", "tx1", "tx2", "tx2"],
...     "chrom": ["chr1"] * 5,
...     "start": [10, 30, 50, 10, 50],
...     "end": [18, 36, 60, 18, 60],
... })
>>> means = cov_by_chrom.slice_agg(all_exons.select("chrom"), all_exons["start"], all_exons["end"], agg="mean")
>>> per_exon = all_exons.with_columns(mean=pl.Series(means), length=pl.col("end") - pl.col("start"))
>>> per_exon
shape: (5, 6)
┌────────────┬───────┬───────┬─────┬──────────┬────────┐
│ transcript ┆ chrom ┆ start ┆ end ┆ mean     ┆ length │
│ ---        ┆ ---   ┆ ---   ┆ --- ┆ ---      ┆ ---    │
│ str        ┆ str   ┆ i64   ┆ i64 ┆ f64      ┆ i64    │
╞════════════╪═══════╪═══════╪═════╪══════════╪════════╡
│ tx1        ┆ chr1  ┆ 10    ┆ 18  ┆ 1.125    ┆ 8      │
│ tx1        ┆ chr1  ┆ 30    ┆ 36  ┆ 1.166667 ┆ 6      │
│ tx1        ┆ chr1  ┆ 50    ┆ 60  ┆ 1.1      ┆ 10     │
│ tx2        ┆ chr1  ┆ 10    ┆ 18  ┆ 1.125    ┆ 8      │
│ tx2        ┆ chr1  ┆ 50    ┆ 60  ┆ 1.1      ┆ 10     │
└────────────┴───────┴───────┴─────┴──────────┴────────┘
>>> per_exon.group_by("transcript", maintain_order=True).agg(
...     mean_coverage=(pl.col("mean") * pl.col("length")).sum() / pl.col("length").sum()
... )
shape: (2, 2)
┌────────────┬───────────────┐
│ transcript ┆ mean_coverage │
│ ---        ┆ ---           │
│ str        ┆ f64           │
╞════════════╪═══════════════╡
│ tx1        ┆ 1.125         │
│ tx2        ┆ 1.111111      │
└────────────┴───────────────┘
```

To build each transcript's coverage as a sequence, as in the previous section,
loop over the transcripts and concatenate their exons.
