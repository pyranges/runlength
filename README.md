# runlength

Generic run-length encoded sequences in Rust, with plan-based arithmetic. It is
the successor to [pyrle](https://github.com/pyranges/pyrle), with a new API and
any value type, not just `f64`. Python bindings will come later as a separate
crate (`runlength-py`).

```rust
use runlength::prelude::*;

let a = Rle::from_runs(&[3, 2], vec![1, 2])?;
let b = Rle::from_runs(&[1, 4], vec![10, 20])?;

// The plan only looks at run boundaries...
let p = a.plan_with(&b);
assert_eq!(p.ends, vec![1, 3, 5]);
assert_eq!((p.idx_a, p.idx_b), (vec![0, 0, 1], vec![0, 1, 1]));

// ...so any operation works on any value types.
let labels = a.zip_with(&b, None, None, |x, y| format!("{x}/{y}"))?;

// Operators pad the shorter side with zero and merge the result.
assert_eq!((&a + &b).to_dense(), vec![11, 21, 21, 22, 22]);
```

## What's in it

| Area | API |
|---|---|
| Sequences | `Rle<V>` (owned), `RleRef<V>` (borrowed), shared methods on `RleLike` |
| Combining | `plan`, `zip_with`, `zip_with_raw`, `+ - * / & \| ^`, unary `-` and `!`, `elem_eq/ne/lt/le/gt/ge` |
| Merging | `merge`, `merge_by(pred)`, `merge_mask(mask, Keep)`, `merge_plan(mask)`, `is_merged` |
| Slicing | `get`, `slice`, `getitems` (many slices → batch) |
| Reductions | `sum`, `mean`, `variance(ddof)`, `std(ddof)`, `min`, `max` (all weighted by run length) |
| Batches | `RleBatch<K, V>`: many sequences under any hashable keys, joins with `How::{Outer, Inner, Left, Right}`, parallel with rayon |
| Coverage | `coverage_sum`, `coverage_sum_opt`, `coverage_count`, and `_by_key` variants |

## Conventions

- Positions are `i64`; every sequence starts at 0; intervals are `[start, end)`.
- Neighbouring runs merge when values are *identical*: `NaN` merges with `NaN`,
  `-0.0` does not merge with `0.0`.
- Operations merge by default; `_raw` variants and plans do not.
- Combining sequences of different lengths pads the shorter one with a fill
  value. Operators use zero. With `zip_with`, passing `None` makes unequal
  lengths an error.

See [docs/design.md](docs/design.md) for the reasoning behind these choices.

## Development

```sh
cargo test                                        # unit, property and doc tests
PROPTEST_CASES=5000 cargo test --release          # more property-test cases
cargo run --release --example bench               # rough timings

# Regenerate the pyrle cross-check cases (needs a Python that imports pyrle):
PYTHONPATH=~/code/pyrle python3.12 scripts/gen_pyrle_fixtures.py
```
