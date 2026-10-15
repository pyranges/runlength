# Coming from pyrle

runlength replaces [pyrle](https://github.com/pyranges/pyrle). It's a new
library rather than a new version, so the API differs. This page lists what
changed and how to translate pyrle code.

## What's different

- **Any value type.** pyrle stored every value as `float64`. runlength keeps
  the type you give it: integer coverage stays integer, and strings, booleans,
  dates and nulls work.
- **Positions start at 0**, and every sequence covers `0` up to its length,
  as in pyrle.
- **Batches have keys of any type.** pyrle's `RleDict` was keyed by
  chromosome, or by chromosome and strand. An `RleBatch` takes any key
  columns. Strands are never combined implicitly: pyrle added up the strands
  when a stranded `RleDict` met an unstranded one, while runlength asks for an
  explicit `collapse(drop="Strand")`.
- **Your own operations.** pyrle supported a fixed set of operators. In
  runlength, `zip_with`, `map` and plans let any function combine sequences;
  see [Combining sequences](guide/combining.md).
- **Parallelism is automatic.** pyrle's `nb_cpu` argument and its ray backend
  are gone. Batch operations use all cores.
- **Correct statistics.** pyrle's `mean()` and `std()` ignored run lengths:
  for the sequence `0 1 1 1`, pyrle's mean is 0.25. runlength counts every
  position, so it's 0.75.
- **NaN merges with NaN**, so a stretch of NaN is one run instead of one run
  per original interval.
- **No pandas requirement.** pyrle needed pandas; runlength needs none of
  polars, pandas, numpy or pyarrow, and works with all of them.

## Translation table

| pyrle | runlength |
|---|---|
| `Rle(runs, values)` | `rl.Rle(runs, values)` (keeps the value type) |
| `rle.runs`, `rle.values` | `rle.runs`, `rle.values` (Arrow arrays; `np.asarray(...)` for numpy) |
| `rle.length` | `len(rle)` |
| `len(rle)` (the number of runs) | `rle.num_runs` |
| `rle[i]`, `rle[a:b]` | `rle[i]`, `rle[a:b]` |
| `rle + other`, `-`, `*`, `/`, comparisons | the same operators |
| `rle.defragment()` | `rle.merge()` (most results are already merged) |
| `rle.apply_values(f)` | `rle.map(f, backend="numpy")` |
| `rle.apply_runs(f)` | `rl.Rle(f(np.asarray(rle.runs)), rle.values)` |
| `rle.numbers_only()` | `rle.map(np.nan_to_num, backend="numpy")` |
| `rle.shift(dist, fill=0)` | `rle.shift(dist, fill=0)` |
| `rle.mean()`, `rle.std()` | `rle.mean()`, `rle.std()` (weighted by run length) |
| `rle.to_frame()` | `rle.to_pandas()` (columns `start`, `end`, `value`) |
| `rle.to_csv(...)` | `rle.to_pandas().to_csv(...)` |
| `RleDict(df)` / `RleDict(df, stranded=True)` | `rl.coverage_by_key(df[["Chromosome"]], ...)` / `df[["Chromosome", "Strand"]]` |
| `RleDict(df, value_col="Score")` | `rl.coverage_by_key(keys, df["Start"], df["End"], values=df["Score"])` |
| `RleDict({...})` | `rl.RleBatch.from_rles(keys, rles)` |
| `rledict["chr1"]`, `rledict["chr1", "+"]` | `batch.get("chr1")`, `batch.get(("chr1", "+"))` |
| `rledict.chromosomes`, `rledict.keys()` | `batch.keys` (a table) |
| `rledict.items()` | `batch.items()` |
| `rledict + other` (and other operators) | the same operators (an outer join on keys) |
| stranded + unstranded `RleDict` | `stranded.collapse(drop="Strand") + unstranded` |
| `rledict.make_strands_same_length()` | not needed: shorter sequences are padded |
| `rledict.add_pseudocounts(0.01)` | `batch.map(lambda v: np.where(v == 0, 0.01, v), backend="numpy")` |
| `rledict.to_table()` | `batch.to_pandas()` |
| `rledict.to_ranges()` | `batch.to_pandas()`, then drop rows where `value == 0` |
| `nb_cpu=...` | not needed |

## An example

Coverage per chromosome and strand, in reads per million, as pyranges
computes it. With pyrle:

```text
from pyrle import RleDict

cov = RleDict(df, stranded=True) * (1e6 / len(df))
```

With runlength:

```python
>>> import pandas as pd
>>> import runlength as rl
>>> df = pd.DataFrame({
...     "Chromosome": ["chr1", "chr1", "chr2"],
...     "Strand": ["+", "-", "+"],
...     "Start": [0, 2, 5],
...     "End": [3, 4, 7],
... })
>>> cov = rl.coverage_by_key(df[["Chromosome", "Strand"]], df["Start"], df["End"]) * (1e6 / len(df))
>>> cov
RleBatch(keys=3, runs=5, dtype=Float64)
  ('chr1', '+'): length=3, runs=[3], values=[333333.3333333333]
  ('chr1', '-'): length=4, runs=[2, 2], values=[0.0, 333333.3333333333]
  ('chr2', '+'): length=7, runs=[5, 2], values=[0.0, 333333.3333333333]
>>> cov.to_pandas().query("value != 0")
  Chromosome Strand  start  end          value
0       chr1      +      0    3  333333.333333
2       chr1      -      2    4  333333.333333
4       chr2      +      5    7  333333.333333
```
