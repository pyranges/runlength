# Rust

The Python package is a thin layer over two Rust crates, which you can use
directly:

| Crate | What it is |
|---|---|
| [`runlength`](https://docs.rs/runlength) | The core. `Rle<V>` holds values of any Rust type, `RleBatch<K, V>` keys of any hashable type. It depends only on rayon and rustc-hash. |
| [`runlength-arrow`](https://docs.rs/runlength-arrow) | The same over Apache Arrow arrays: values of any Arrow type, and batch keys as Arrow tables. The Python package uses this one. |

```sh
cargo add runlength
```

The API reference for both is on docs.rs. The examples below come from
[`crates/runlength/examples/tour.rs`](https://github.com/pyranges/runlength/blob/master/crates/runlength/examples/tour.rs),
which CI runs; each `assert_eq!` shows what the line before it gives.

## Sequences

`Rle<V>` owns a sequence and `RleRef<'_, V>` borrows one. Their shared
methods are on the `RleLike` trait, which the prelude brings in.

```rust
--8<-- "crates/runlength/examples/tour.rs:basics"
```

## Combining sequences

Operators work on numbers. `plan_with` and `zip_with` work on any types, as
described in [Combining sequences](guide/combining.md).

```rust
--8<-- "crates/runlength/examples/tour.rs:combine"
```

The `_raw` variants (`zip_with_raw`, ...) skip the final merge.

## Coverage

```rust
--8<-- "crates/runlength/examples/tour.rs:coverage"
```

`coverage_max` and `coverage_min` take the largest or smallest value per
position instead of the sum, and `coverage_sum_opt` leaves uncovered
positions as `None`.

## Batches

```rust
--8<-- "crates/runlength/examples/tour.rs:batch"
```

`collapse` merges keys, `getitems` and `slice_reduce` slice many sequences
by key, and `map_rles` and `reduce` apply a function to every sequence, all in
parallel.

## Conventions

- Positions and lengths are `i64`, every sequence starts at 0, and intervals
  are half-open, `[start, end)`.
- Neighbouring runs merge when their values are `Identical`: `NaN` merges
  with `NaN`, and `-0.0` stays apart from `0.0`. Implement `Identical` for
  your own value types (it's implemented for numbers, `bool`, `char`,
  strings, `Option`, tuples, `Vec` and more).
- Functions that can fail return `runlength::Result`. Input is checked once,
  where it enters (run lengths must not be negative, `runs` and `values`
  must have the same length); after that, operations can't fail on bad
  data.
