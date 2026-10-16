# Combining sequences

The operators (`+`, `<`, `&`, ...) cover the common cases. This page is about
everything else: combining sequences with your own function, combining more
than two at once, and the *plan* underneath all of it.

## The plan

To combine two sequences, runlength first works out where the output runs
end. It does this from the run boundaries alone, without looking at any
values:

```python
>>> import runlength as rl
>>> a = rl.Rle([3, 2], [1, 2])          # 1 1 1 2 2
>>> b = rl.Rle([1, 4], [10, 20])        # 10 20 20 20 20
>>> ends, idx_a, idx_b = a.plan(b)
>>> ends.to_pylist()
[1, 3, 5]
>>> idx_a.to_pylist(), idx_b.to_pylist()
([0, 0, 1], [0, 1, 1])
```

The plan has one entry per output segment. Segment 1 ends at position 3, and
is covered by run 0 of `a` (value 1) and run 1 of `b` (value 20). Every
operation then has the same shape: take the values through the indices,
apply a function, and merge neighbours that came out equal. That's what
`a + b` does:

```python
>>> import numpy as np
>>> va, vb = np.asarray(a.values), np.asarray(b.values)
>>> rl.Rle.from_ends(ends, va[idx_a] + vb[idx_b]).merge()
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
>>> a + b
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
```

Because the plan doesn't involve values, it works for any value types, and
you can compute it once and use it for several operations on the same pair.
[Working with plans](plans.md) shows how, and how to analyse segments as a
table.

## Your own function: `zip_with`

`zip_with` does the plan, the take and the merge for you. Your function gets
the values of both sides, one entry per segment, and returns one value per
segment:

```python
>>> a.zip_with(b, lambda x, y: np.maximum(x, y) - np.minimum(x, y), backend="numpy")
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [9, 19, 18]
```

### Choosing a backend

runlength can't tell which library your function is written in, so
`backend` says what to pass it:

| `backend` | your function receives | write it with |
| --- | --- | --- |
| `"numpy"` | numpy arrays | numpy operators and functions |
| `"polars"` | polars Series | polars operators and methods |
| `"pyarrow"` | pyarrow arrays | `pyarrow.compute` |
| `"arrow"` (default) | bare Arrow arrays | conversion to one of the above |

Here is the same function written for each:

```python
>>> import polars as pl
>>> import pyarrow.compute as pc
>>> a.zip_with(b, lambda x, y: x + y, backend="numpy")
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
>>> a.zip_with(b, lambda x, y: x + y, backend="polars")
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
>>> a.zip_with(b, lambda x, y: pc.add(x, y), backend="pyarrow")
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
>>> a.zip_with(b, lambda x, y: x.to_numpy() + y.to_numpy())
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [11, 21, 22]
```

The default objects have no operators or compute methods, only conversions
such as `to_numpy()` and `to_pylist()`; any of the libraries above can also
read them directly (`pl.Series(x)`, `pa.array(x)`). Use the default when your
function does its own conversion, or when it only passes the arrays on.

If the function is written for a different library than `backend` names, you
get that library's error:

```python
>>> a.zip_with(b, lambda x, y: x + y)
Traceback (most recent call last):
    ...
TypeError: unsupported operand type(s) for +: 'arro3.core._core.Array' and 'arro3.core._core.Array'
```

Use numpy for numbers. numpy arrays can't hold nulls, so sequences with
missing values fail before your function is called; polars handles them:

```python
>>> gappy = rl.Rle([2, 1], [1, None])
>>> gappy.map(lambda v: v * 10, backend="numpy")
Traceback (most recent call last):
    ...
ValueError: Cannot create numpy array from arrow array with nulls.
>>> gappy.map(lambda v: v * 10, backend="polars")
Rle(length=3, runs=2, dtype=Int64)
  runs:   [2, 1]
  values: [10, null]
```

Use polars for strings, casts, nulls and anything else polars has an
expression for:

```python
>>> names = rl.Rle([2, 1], ["ann", "bob"])
>>> ages = rl.Rle([1, 2], [31, 45])
>>> names.zip_with(ages, lambda n, a: n + " (" + a.cast(pl.String) + ")", backend="polars")
Rle(length=3, runs=3, dtype=Utf8View)
  runs:   [1, 1, 1]
  values: ["ann (31)", "ann (45)", "bob (45)"]
```

Use pyarrow if you already know `pyarrow.compute`. The library you name must
be installed; runlength itself needs none of them.

Whichever backend you pick, the function can return a list, or an array or
Series from any of these libraries, and of any value type. It must return one
value per segment:

```python
>>> a.zip_with(b, lambda x, y: [f"{i}/{j}" for i, j in zip(x, y)], backend="numpy")
Rle(length=5, runs=3, dtype=Utf8)
  runs:   [1, 2, 2]
  values: ["1/10", "1/20", "2/20"]
>>> a.zip_with(b, lambda x, y: x[:1], backend="numpy")
Traceback (most recent call last):
    ...
ValueError: operation returned 1 values for 3 segments
```

`map`, `zip_many` and the batch versions of `map` and `zip_with` take
`backend` the same way.

### Different lengths

`zip_with` doesn't guess what lies past the end of the shorter sequence.
Without fill values, different lengths are an error:

```python
>>> short = rl.Rle([2], [5])
>>> a.zip_with(short, lambda x, y: x * y, backend="numpy")
Traceback (most recent call last):
    ...
ValueError: lengths differ (5 and 2) and the shorter side has no fill value
```

Give each side the value to pad it with:

```python
>>> a.zip_with(short, lambda x, y: x * y, fill_other=1, backend="numpy")
Rle(length=5, runs=3, dtype=Int64)
  runs:   [2, 1, 2]
  values: [5, 1, 2]
```

The operators use zero as the fill value on both sides.

### Without merging

Pass `merge=False` to keep one run per plan segment, even where neighbours
came out equal. This is useful when the segments themselves matter, for
example to line two sequences up run by run:

```python
>>> a.zip_with(b, lambda x, y: x * 0, backend="numpy", merge=False)
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [0, 0, 0]
```

## One sequence: `map`

`map` applies a function to the run values. It's called once per run, not
once per position:

```python
>>> cov = rl.coverage([0, 2, 8], [3, 4, 10])
>>> cov.map(lambda v: np.log2(v + 1), backend="numpy")
Rle(length=10, runs=5, dtype=Float64)
  runs:   [2, 1, 1, 4, 2]
  values: [1.0, 1.584962500721156, 1.0, 0.0, 1.0]
```

`cast` changes the value type:

```python
>>> cov.cast("float32")
Rle(length=10, runs=5, dtype=Float32)
  runs:   [2, 1, 1, 4, 2]
  values: [1.0, 2.0, 1.0, 0.0, 1.0]
```

## More than two: `zip_many`

`zip_many` combines any number of sequences in one pass, and `plan_many`
gives its plan:

```python
>>> x = rl.Rle([2, 2], [1, 2])
>>> y = rl.Rle([3, 1], [10, 20])
>>> z = rl.Rle([1, 3], [100, 200])
>>> rl.zip_many([x, y, z], lambda a, b, c: a + b + c, backend="numpy")
Rle(length=4, runs=4, dtype=Int64)
  runs:   [1, 1, 1, 1]
  values: [111, 211, 212, 222]
>>> rl.zip_many([x, y, z], lambda *cols: np.median(np.stack(cols), axis=0), backend="numpy")
Rle(length=4, runs=2, dtype=Float64)
  runs:   [3, 1]
  values: [10.0, 20.0]
>>> ends, (ix, iy, iz) = rl.plan_many([x, y, z])
>>> ends.to_pylist(), ix.to_pylist(), iy.to_pylist(), iz.to_pylist()
([1, 2, 3, 4], [0, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 1])
```

## Rearranging

`shift` moves values right (or left, with a negative distance) and keeps the
length, `reverse` flips the sequence, and `concat` joins sequences end to
end:

```python
>>> a.shift(2)
Rle(length=5, runs=2, dtype=Int64)
  runs:   [2, 3]
  values: [0, 1]
>>> a.shift(-2, fill=None)
Rle(length=5, runs=3, dtype=Int64)
  runs:   [1, 2, 2]
  values: [1, 2, null]
>>> a.reverse()
Rle(length=5, runs=2, dtype=Int64)
  runs:   [2, 3]
  values: [2, 1]
>>> rl.concat([a, rl.Rle([1], [2]), a])
Rle(length=11, runs=4, dtype=Int64)
  runs:   [3, 3, 3, 2]
  values: [1, 2, 1, 2]
```

By default, `shift` fills with 0 for numbers, false for booleans and null for
other types.
