# Design

These decisions were made before writing the code. They explain why the
library looks the way it does, and what is intentionally left for later.

## Plan, then compute

pyrle had one merge loop per operator (`add_rles`, `sub_rles`, …) that differed
in a single line. Here, combining two sequences is split up:

1. **Plan** (`plan`): from run ends alone, compute the output segments (the
   union of both inputs' breakpoints) and, for each segment, which run of each
   input covers it. No values are involved, so the plan works for any types
   and can be reused for several operations on the same pair.
2. **Compute**: gather values through the plan and apply any function.
3. **Merge**: join neighbouring runs with identical values. This must be a
   separate step, because it depends on the computed values.

`zip_with` fuses all three for speed. `Plan` exposes step 1 so a caller (for
example Python with polars or numpy) can do step 2 itself.

## Storage

A sequence stores cumulative run *ends*, not run lengths. The plan is then a
merge of two sorted lists, and slicing uses binary search. The API still
accepts and returns run lengths.

Positions are `i64` everywhere: that is what Arrow, polars, pandas and numpy
use. Input is validated once, at construction. Negative runs and mismatched
lengths are errors; zero-length runs are dropped.

## Different lengths

When lengths differ, the shorter side is padded with a fill value. Operators
pad with zero (`V::default()`); `zip_with` takes an explicit `Option` per side,
and `None` makes unequal lengths an error. Which part was filled is not
tracked: it is always the tail `[min_len, max_len)`, and plans mark it with an
index equal to the input's run count. If "no data" must differ from 0, use
`NaN` or `Option<V>` with `None` as the fill.

## Coordinates

Every sequence starts at 0, and slices restart at 0. Genomic coordinates,
gaps and spliced transcripts belong to the layer above (pyranges). A
positioned wrapper (`start` + `Rle`) can be added later without changing the
core.

## Merging

The default rule is `Identical`: `NaN` merges with `NaN`, `None` with `None`,
and `-0.0` stays separate from `0.0`. Merged form is *not* an invariant:
operations merge by default, but raw variants and custom merges exist.
Custom merges use a predicate (`merge_by`) or a mask over neighbouring pairs
(`merge_mask`, which keeps the first or last value). `merge_plan` returns the
groups without values, for any aggregation.

`PartialEq` on sequences compares what they represent, regardless of how runs
are split. Element-wise comparisons are `elem_eq`, `elem_lt`, … and return
`Rle<bool>`.

## Batches

`RleBatch<K, V>` stores many sequences back to back, like an Arrow list column,
under arbitrary hashable keys (tuples for multi-part keys). This replaces
pyrle's `RleDict` and its process-based parallelism: batch operations run in
parallel with rayon. Combining batches is a join (`How::Outer` by default); a
key missing on one side is treated as a sequence of length 0, which the fill
rule then pads. Keys keep first-appearance order. The core knows nothing about
chromosomes or strands.

## Coverage

Sum and count use a sweep (add at starts, subtract at ends), keeping the value
type. When no interval is active, the running sum resets to exactly zero, so
float rounding cannot leave noise in gaps. The length is the largest interval
end, including empty intervals.

## Later

- Arrow layer: generic gather and merge over any Arrow type.
- `runlength-py`: pyo3 bindings holding Arrow arrays, with `to_polars`,
  `to_pandas`, `to_numpy` and `to_arrow`. Neither polars nor pandas is the
  primary backend.
- `shift`, `concat`, `reverse`, `to_ranges`, `collapse`, n-ary plans.
- Coverage max/min (sweep with a heap) and a generic `overlap_plan`: segments
  plus the ids of the intervals covering each one, in list-column layout, for
  any aggregation.

## Dropped from pyrle

`__array_ufunc__`, `to_csv`, and `nb_cpu`/ray.
