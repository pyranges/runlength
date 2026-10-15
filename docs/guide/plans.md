# Working with plans

Every operation that combines sequences runs in three steps: a *plan* works
out where the output runs end and which run of each input covers them, the
values are gathered and combined, and equal neighbours are merged. The
operators and `zip_with` do all three for you. This page is about using the
intermediate results directly: to compute several things from one plan, to
analyse segments as a table, or to merge runs by your own rule.

The examples use two coverage tracks from a ChIP-seq experiment,
`treatment` and `control`, and an annotation track that labels each position
of a 20-base region as exon, intron or intergenic.

```python
>>> import numpy as np
>>> import polars as pl
>>> import runlength as rl
>>> treatment = rl.coverage(starts=[2, 4, 5, 12], ends=[9, 9, 11, 16], length=20)
>>> control = rl.coverage(starts=[0, 6], ends=[8, 14], length=20)
>>> region = rl.Rle([3, 5, 4, 6, 2], ["intergenic", "exon", "intron", "exon", "intergenic"])
>>> treatment
Rle(length=20, runs=8, dtype=Int64)
  runs:   [2, 2, 1, 4, 2, 1, 4, 4]
  values: [0, 1, 2, 3, 1, 0, 1, 0]
>>> control
Rle(length=20, runs=4, dtype=Int64)
  runs:   [6, 2, 6, 6]
  values: [1, 2, 1, 0]
```

## What a plan is

`a.plan(b)` returns three arrays with one entry per output segment: where
the segment ends, and which run of `a` and of `b` covers it.

```python
>>> ends, idx_t, idx_c = treatment.plan(control)
>>> ends.to_pylist()
[2, 4, 5, 6, 8, 9, 11, 12, 14, 16, 20]
>>> idx_t.to_pylist()
[0, 1, 2, 3, 3, 3, 4, 5, 6, 6, 7]
>>> idx_c.to_pylist()
[0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3]
```

Segment 1 ends at position 4, and is covered by run 1 of `treatment` and run
0 of `control`. A plan only looks at where runs end, never at values, so it
works the same for numbers, strings or anything else.

## Several results from one plan

Gathering each side's values through its indices lines the two tracks up
segment by segment. From there, any number of results can be computed with
plain numpy, without planning again:

```python
>>> t = np.asarray(treatment.values)[np.asarray(idx_t)]
>>> c = np.asarray(control.values)[np.asarray(idx_c)]
>>> t, c
(array([0, 1, 2, 3, 3, 3, 1, 0, 1, 1, 0]), array([1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0]))
>>> log2_ratio = rl.Rle.from_ends(ends, np.log2((t + 1) / (c + 1)).round(2)).merge()
>>> log2_ratio
Rle(length=20, runs=11, dtype=Float64)
  runs:   [2, 2, 1, 1, 2, ..., 2, 1, 2, 2, 4]
  values: [-1.0, 0.0, 0.58, 1.0, 0.42, ..., 0.0, -1.0, 0.0, 1.0, 0.0]
>>> enriched = rl.Rle.from_ends(ends, (t >= 2) & (t > c)).merge()
>>> starts, stops = enriched.ranges()
>>> starts.to_pylist(), stops.to_pylist()
([4], [9])
```

`Rle.from_ends` builds a sequence from segment ends and one value per
segment, and `merge` joins segments that came out equal. `t > c` gives the
same result as `treatment > control`, but here the plan was computed once for
all three results.

## Segments as a table

A plan of several sequences is a table with one row per segment, which polars
or pandas can then filter, group and aggregate. `rl.plan_many` plans any number
of sequences at once; here, the two tracks and the annotation:

```python
>>> ends, (it, ic, ir) = rl.plan_many([treatment, control, region])
>>> ends = pl.Series("end", ends)
>>> segments = pl.DataFrame({
...     "start": ends.shift(1, fill_value=0),
...     "end": ends,
...     "treatment": pl.Series(treatment.values).gather(pl.Series(it)),
...     "control": pl.Series(control.values).gather(pl.Series(ic)),
...     "region": pl.Series(region.values).gather(pl.Series(ir)),
... }).with_columns(length=pl.col("end") - pl.col("start"))
>>> segments
shape: (13, 6)
┌───────┬─────┬───────────┬─────────┬────────────┬────────┐
│ start ┆ end ┆ treatment ┆ control ┆ region     ┆ length │
│ ---   ┆ --- ┆ ---       ┆ ---     ┆ ---        ┆ ---    │
│ i64   ┆ i64 ┆ i64       ┆ i64     ┆ str        ┆ i64    │
╞═══════╪═════╪═══════════╪═════════╪════════════╪════════╡
│ 0     ┆ 2   ┆ 0         ┆ 1       ┆ intergenic ┆ 2      │
│ 2     ┆ 3   ┆ 1         ┆ 1       ┆ intergenic ┆ 1      │
│ 3     ┆ 4   ┆ 1         ┆ 1       ┆ exon       ┆ 1      │
│ 4     ┆ 5   ┆ 2         ┆ 1       ┆ exon       ┆ 1      │
│ 5     ┆ 6   ┆ 3         ┆ 1       ┆ exon       ┆ 1      │
│ …     ┆ …   ┆ …         ┆ …       ┆ …          ┆ …      │
│ 11    ┆ 12  ┆ 0         ┆ 1       ┆ intron     ┆ 1      │
│ 12    ┆ 14  ┆ 1         ┆ 1       ┆ exon       ┆ 2      │
│ 14    ┆ 16  ┆ 1         ┆ 0       ┆ exon       ┆ 2      │
│ 16    ┆ 18  ┆ 0         ┆ 0       ┆ exon       ┆ 2      │
│ 18    ┆ 20  ┆ 0         ┆ 0       ┆ intergenic ┆ 2      │
└───────┴─────┴───────────┴─────────┴────────────┴────────┘
```

Questions about the tracks are now ordinary queries. For example, the mean
coverage of each annotation class, counting every base, and how many bases
of each class the treatment covers more deeply than the control:

```python
>>> segments.group_by("region", maintain_order=True).agg(
...     bases=pl.col("length").sum(),
...     treatment=(pl.col("treatment") * pl.col("length")).sum() / pl.col("length").sum(),
...     control=(pl.col("control") * pl.col("length")).sum() / pl.col("length").sum(),
...     enriched_bases=pl.col("length").filter(pl.col("treatment") > pl.col("control")).sum(),
... )
shape: (3, 5)
┌────────────┬───────┬───────────┬──────────┬────────────────┐
│ region     ┆ bases ┆ treatment ┆ control  ┆ enriched_bases │
│ ---        ┆ ---   ┆ ---       ┆ ---      ┆ ---            │
│ str        ┆ i64   ┆ f64       ┆ f64      ┆ i64            │
╞════════════╪═══════╪═══════════╪══════════╪════════════════╡
│ intergenic ┆ 5     ┆ 0.2       ┆ 0.6      ┆ 0              │
│ exon       ┆ 11    ┆ 1.454545  ┆ 0.818182 ┆ 6              │
│ intron     ┆ 4     ┆ 1.25      ┆ 1.0      ┆ 1              │
└────────────┴───────┴───────────┴──────────┴────────────────┘
```

The table has one row per segment, not per base, so this stays fast for whole
genomes: the number of segments is at most the total number of runs.

## Sequences of different lengths

When one sequence is shorter, the plan marks segments past its end with an
index equal to its number of runs, one past its last run. Append a fill value
to its values before gathering, and those segments get the fill value:

```python
>>> short = rl.Rle([3, 2], [1, 2])
>>> long = rl.Rle([2, 6], [10, 20])
>>> ends, i_short, i_long = short.plan(long)
>>> ends.to_pylist(), i_short.to_pylist()
([2, 3, 5, 8], [0, 0, 1, 2])
>>> values = np.append(np.asarray(short.values), 0)
>>> values[np.asarray(i_short)]
array([1, 1, 2, 0])
```

This is what the operators do, with zero as the fill value. `zip_with` takes
the fill values as `fill_self` and `fill_other`.

## Replicates: one value from many sequences

`plan_many` lines up any number of sequences, which gives a matrix with one
row per replicate and one column per segment. Any function over replicates,
such as a median or a variance, then runs on that matrix:

```python
>>> replicates = [
...     rl.coverage([0, 3], [6, 9], length=10),
...     rl.coverage([1, 3], [5, 10], length=10),
...     rl.coverage([0, 2, 4], [4, 8, 10], length=10),
... ]
>>> ends, idx = rl.plan_many(replicates)
>>> matrix = np.stack([np.asarray(r.values)[np.asarray(i)] for r, i in zip(replicates, idx)])
>>> matrix
array([[1, 1, 1, 2, 2, 1, 1, 0],
       [0, 1, 1, 2, 1, 1, 1, 1],
       [1, 1, 2, 2, 2, 2, 1, 1]])
>>> rl.Rle.from_ends(ends, np.median(matrix, axis=0)).merge()
Rle(length=10, runs=3, dtype=Float64)
  runs:   [3, 3, 4]
  values: [1.0, 2.0, 1.0]
>>> rl.Rle.from_ends(ends, matrix.var(axis=0).round(2)).merge()
Rle(length=10, runs=7, dtype=Float64)
  runs:   [1, 1, 1, 2, 3, 1, 1]
  values: [0.22, 0.0, 0.22, 0.0, 0.22, 0.0, 0.22]
```

`rl.zip_many` does the same in one call when you only need one result, for
example `rl.zip_many(replicates, lambda *cols: np.median(np.stack(cols), axis=0), backend="numpy")`.

## Merging by your own rule

The last step, merging, has a plan of its own. `merge_plan` takes one boolean
per pair of neighbouring runs, saying whether they join, and returns which
runs form each group, without touching the values. That separates *which*
runs to merge from *what value* the merged run gets.

For example, coverage often has short dips and spikes. To absorb runs shorter
than 3 bases into the run before them, and give each merged run its mean
coverage:

```python
>>> cov = rl.Rle([5, 1, 4, 2, 6], [10.0, 2.0, 12.0, 30.0, 11.0])
>>> runs = np.asarray(cov.runs)
>>> mask = runs[1:] < 3
>>> mask
array([ True, False,  True, False])
>>> ends, offsets = cov.merge_plan(mask)
>>> ends.to_pylist(), offsets.to_pylist()
([6, 12, 18], [0, 2, 4, 5])
>>> first = np.asarray(offsets)[:-1]
>>> weighted = np.add.reduceat(runs * np.asarray(cov.values), first)
>>> rl.Rle.from_ends(ends, (weighted / np.add.reduceat(runs, first)).round(2))
Rle(length=18, runs=3, dtype=Float64)
  runs:   [6, 6, 6]
  values: [8.67, 18.0, 11.0]
```

Group `g` holds runs `offsets[g]` up to `offsets[g + 1]`, so `reduceat`
aggregates each group in one call. The mean counts every base: the first
group is five bases of 10 and one of 2, so its mean is 52 / 6, about 8.67.

## Keeping segments unmerged

`zip_with` and `zip_many` take `merge=False` to keep one run per plan
segment, even where neighbours came out equal. Then several results computed
separately still line up run by run, and can be put side by side:

```python
>>> diff = treatment.zip_with(control, lambda t, c: t - c, backend="numpy", merge=False)
>>> both = treatment.zip_with(control, lambda t, c: (t > 0) & (c > 0), backend="numpy", merge=False)
>>> diff.num_runs == both.num_runs
True
>>> pl.DataFrame({"end": diff.ends, "diff": diff.values, "both": both.values}).head(4)
shape: (4, 3)
┌─────┬──────┬───────┐
│ end ┆ diff ┆ both  │
│ --- ┆ ---  ┆ ---   │
│ i64 ┆ i64  ┆ bool  │
╞═════╪══════╪═══════╡
│ 2   ┆ -1   ┆ false │
│ 4   ┆ 0    ┆ true  │
│ 5   ┆ 1    ┆ true  │
│ 6   ┆ 2    ┆ true  │
└─────┴──────┴───────┘
```

Call `merge()` on a result once you no longer need it aligned.

## See also

- [Combining sequences](combining.md) for `zip_with` and `zip_many`, which
  run the whole plan, gather and merge for you.
- [Coverage and intervals](coverage.md#any-other-aggregation-overlap-plans)
  for `overlap_plan`, the plan of a set of intervals: which intervals cover
  each segment.
- [Merging runs](merging.md) for when two values count as equal.
