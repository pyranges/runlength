# API reference

The Python package, `runlength`. The examples assume `import runlength as rl`,
and some use `import polars as pl`.

Wherever a parameter takes an array, it accepts a polars or pandas Series, a
pyarrow or numpy array, or a list; see
[polars, pandas, numpy and pyarrow](guide/interop.md). Arrays and tables that
runlength returns can be passed to any of those libraries.

## Sequences

::: runlength.Rle

## Batches

::: runlength.RleBatch

## Coverage

::: runlength.coverage

::: runlength.coverage_by_key

::: runlength.overlap_plan

::: runlength.overlap_plan_by_key

## Combining

::: runlength.concat

::: runlength.plan_many

::: runlength.zip_many
