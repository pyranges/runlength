# polars, pandas, numpy and pyarrow

runlength stores values as [Apache Arrow](https://arrow.apache.org) arrays,
which polars and pyarrow use too, and pandas and numpy can read. None of
these libraries is required: install the ones you use.

## What goes in

Anywhere runlength takes an array (run lengths, values, starts, ends), you
can pass:

- a polars Series, a pandas Series or a pyarrow array,
- a numpy array,
- a Python list,
- anything else that implements the
  [Arrow PyCapsule interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html).

```python
>>> import numpy as np
>>> import pandas as pd
>>> import polars as pl
>>> import pyarrow as pa
>>> import runlength as rl
>>> rl.Rle(np.array([3, 2]), pl.Series([1.5, 2.5]))
Rle(length=5, runs=2, dtype=Float64)
  runs:   [3, 2]
  values: [1.5, 2.5]
>>> rl.Rle(pd.Series([3, 2]), pa.array(["x", "y"]))
Rle(length=5, runs=2, dtype=Utf8)
  runs:   [3, 2]
  values: ["x", "y"]
```

Key tables (for `coverage_by_key` and batches) can be a polars or pandas
DataFrame, a pyarrow Table, or a dict of columns:

```python
>>> df = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 5], "end": [3, 7]})
>>> rl.coverage_by_key(df[["chrom"]], df["start"], df["end"])
RleBatch(keys=2, runs=3, dtype=Int64)
  ('chr1',): length=3, runs=[3], values=[1]
  ('chr2',): length=7, runs=[5, 2], values=[0, 1]
```

Missing values become nulls: `None` in lists, nulls in polars and pyarrow,
and `None`, `NaN` or `pd.NA` in pandas columns that can hold missing values.
NaN in a numpy float array stays NaN.

## What comes out

Arrays from runlength (`rle.values`, `rle.runs`, ...) and tables
(`batch.keys`, `batch.mean()`, ...) are small Arrow wrappers. Every library
above reads them directly, without copying data:

```python
>>> r = rl.Rle([3, 2], [1.5, 2.5])
>>> pl.Series(r.values)
shape: (2,)
Series: '' [f64]
[
	1.5
	2.5
]
>>> np.asarray(r.values)
array([1.5, 2.5])
>>> pa.array(r.values).type
DataType(double)
```

Tables work the same way with `pl.DataFrame(...)` and `pa.table(...)`, and an
`Rle` or `RleBatch` is itself a table of its runs:

```python
>>> pl.DataFrame(r)
shape: (2, 3)
┌───────┬─────┬───────┐
│ start ┆ end ┆ value │
│ ---   ┆ --- ┆ ---   │
│ i64   ┆ i64 ┆ f64   │
╞═══════╪═════╪═══════╡
│ 0     ┆ 3   ┆ 1.5   │
│ 3     ┆ 5   ┆ 2.5   │
└───────┴─────┴───────┘
```

For a named conversion, use `to_polars()`, `to_pandas()` or `to_arrow()` to
get the table of runs, or `to_numpy()` and `to_dense()` to get one value per
position:

```python
>>> r.to_pandas()
   start  end  value
0      0    3    1.5
1      3    5    2.5
>>> r.to_numpy()
array([1.5, 1.5, 1.5, 2.5, 2.5])
```

## Your own functions

Functions you pass to `zip_with`, `map` and `zip_many` receive numpy arrays,
polars Series or pyarrow arrays, whichever you name with `backend`. See
[Choosing a backend](combining.md#choosing-a-backend).

## Value types

Values keep their Arrow type through every operation, and `rle.dtype` shows
it. Integer and float types of every width, booleans, strings, dates,
timestamps and other Arrow types all work. Arithmetic needs numbers (or
booleans for `&`, `|` and `~`), comparisons work on any type with an order,
and everything else (slicing, merging, `zip_with`, batches) works on any type.

```python
>>> import datetime
>>> days = rl.Rle([2, 3], [datetime.date(2026, 1, 1), datetime.date(2026, 2, 1)])
>>> days
Rle(length=5, runs=2, dtype=Date32)
  runs:   [2, 3]
  values: [2026-01-01, 2026-02-01]
>>> days.max()
datetime.date(2026, 2, 1)
```

Lists of bools, ints, floats and strings convert directly. Lists of other
values, such as the dates above, are converted by pyarrow if it's installed;
otherwise pass a polars Series.

`cast` converts to another type, by Arrow's name for it or the lowercase
names numpy uses:

```python
>>> rl.Rle([3, 2], [1, 2]).cast("float32")
Rle(length=5, runs=2, dtype=Float32)
  runs:   [3, 2]
  values: [1.0, 2.0]
```

### Mixing numeric types

Combining two numeric types converts both to a common one first: `Float64`
if either is a float, `UInt64` if both are unsigned, and `Int64` otherwise.
Values are never silently changed on the way: a value that doesn't fit in
the common type, or in the type you `cast` to, is an error rather than null:

```python
>>> import pyarrow as pa
>>> big = rl.Rle([1], pa.array([2**64 - 1], pa.uint64()))
>>> big + rl.Rle([1], pa.array([1], pa.uint8()))
Traceback (most recent call last):
    ...
ValueError: Arithmetic overflow: Overflow happened on: 18446744073709551615 + 1
>>> big + rl.Rle([1], [0])
Traceback (most recent call last):
    ...
ValueError: Cast error: Can't cast value 18446744073709551615 to type Int64
```

Lists of Python ints become `Int64`, or `UInt64` if a value is too large
for that. Fill values and lookup keys must also convert exactly:
`batch.get(1.0)` finds the integer key 1, but `batch.get(1.5)` is an error.

Within one type, arithmetic that overflows is an error. Reductions such as
`sum` wrap around instead, so for small integer types, cast to a wider
type first:

```python
>>> small = rl.Rle([2], pa.array([100], pa.int8()))
>>> small.sum()
-56
>>> small.cast("int64").sum()
200
```
