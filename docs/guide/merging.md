# Merging runs

A sequence is *merged* when no two neighbouring runs hold the same value.
Operations merge their results, so `1 1 2` is stored as two runs, not three.
This page covers when values count as the same, sequences that aren't
merged, and merging by your own rule.

## When are two values the same?

Neighbouring runs merge when their values are *identical*, which differs from
`==` in two places:

- `NaN` is identical to `NaN`, so a stretch of NaN is one run. (With `==`,
  NaN never equals anything, and long NaN stretches would stay in pieces.)
- `-0.0` is not identical to `0.0`, because they are different values.

Nulls are identical to each other, and never to NaN.

```python
>>> import runlength as rl
>>> rl.Rle.from_dense([float("nan"), float("nan"), 0.0, -0.0, None, None])
Rle(length=6, runs=4, dtype=Float64)
  runs:   [2, 1, 1, 2]
  values: [NaN, 0.0, -0.0, null]
```

## Unmerged sequences

The constructor keeps runs as you give them, without merging:

```python
>>> r = rl.Rle([1, 1, 2], [5, 5, 7])
>>> r
Rle(length=4, runs=3, dtype=Int64)
  runs:   [1, 1, 2]
  values: [5, 5, 7]
>>> r.is_merged()
False
>>> r.merge()
Rle(length=4, runs=2, dtype=Int64)
  runs:   [2, 2]
  values: [5, 7]
```

Unmerged sequences behave the same in every operation; they just use more
runs than needed. `equals` compares the positions, not the runs, so it doesn't
matter how a sequence is split:

```python
>>> r.equals(r.merge())
True
```

Plans and `zip_with(..., merge=False)` also produce unmerged sequences, with
one run per segment.

## Merging by your own rule

`merge` also takes a mask, with one boolean per pair of neighbouring runs:
`mask[i]` says whether run `i + 1` merges into run `i`. A group of merged runs
keeps the value of its `"first"` or `"last"` run.

For example, merge neighbouring runs whose values differ by less than 0.5:

```python
>>> import numpy as np
>>> signal = rl.Rle([2, 2, 2, 2], [1.0, 1.25, 5.0, 5.25])
>>> mask = np.abs(np.diff(np.asarray(signal.values))) < 0.5
>>> mask
array([ True, False,  True])
>>> signal.merge(mask, keep="first")
Rle(length=8, runs=2, dtype=Float64)
  runs:   [4, 4]
  values: [1.0, 5.0]
```

To give each group a value of your choosing, such as its mean, `merge_plan`
returns the groups without touching the values: group `g` ends at `ends[g]`
and holds runs `offsets[g]` up to `offsets[g + 1]`.

```python
>>> ends, offsets = signal.merge_plan(mask)
>>> ends.to_pylist(), offsets.to_pylist()
([4, 8], [0, 2, 4])
>>> runs = np.asarray(signal.runs)
>>> values = np.asarray(signal.values)
>>> starts = np.asarray(offsets)[:-1]
>>> sums = np.add.reduceat(runs * values, starts)
>>> lengths = np.add.reduceat(runs, starts)
>>> rl.Rle.from_ends(ends, sums / lengths)
Rle(length=8, runs=2, dtype=Float64)
  runs:   [4, 4]
  values: [1.125, 5.125]
```

The result holds the mean of each group, counting every position.
