# runlength

Generic run-length encoded sequences in Rust, with plan-based arithmetic. It is
the successor to [pyrle](https://github.com/pyranges/pyrle), with a new API and
any value type, not just `f64`.

| Crate | Contents |
|---|---|
| `runlength` | The core: generic `Rle<V>`, plans, merging, batches, coverage. Depends only on rayon. |
| `runlength-arrow` | The same over Apache Arrow arrays, so values can be any Arrow type, and batch keys are Arrow tables. |
| `runlength-py` | The `runlength` Python package (pyo3). Works with polars, pandas, pyarrow and numpy; none of them is required. |

## Python

```python
import polars as pl
import runlength as rl

df = pl.DataFrame({
    "Chromosome": ["chr1", "chr1", "chr2"],
    "Strand": ["+", "-", "+"],
    "Start": [0, 2, 5],
    "End": [3, 4, 7],
    "Score": [0.5, 2.0, 1.0],
})
keys = df.select("Chromosome", "Strand")
cov = rl.coverage_by_key(keys, df["Start"], df["End"])  # an RleBatch
cov.get(("chr1", "+"))           # Rle(length=3, runs=1, dtype=Int64) ...
(cov + cov).to_polars()          # Chromosome, Strand, start, end, value
pl.DataFrame(cov.mean())         # Chromosome, Strand, mean
per_chrom = cov.collapse(drop="Strand")          # add the strands together
starts, ends = (per_chrom.get("chr1") > 0).ranges()   # back to intervals
best = rl.coverage_by_key(keys, df["Start"], df["End"], df["Score"], agg="max", fill=None)

a = rl.Rle([3, 2], ["x", "y"])              # values of any type
b = rl.Rle([1, 4], pl.Series(["p", "q"]))
a.zip_with(b, lambda x, y: x + y, backend="polars")   # any function, any library
a.plan(b)                                    # or just the plan: (ends, idx_a, idx_b)
```

Inputs can be polars or pandas Series and DataFrames, pyarrow or numpy arrays,
or lists. Outputs (`rle.values`, `rle.runs`, `batch.keys`, ...) speak the Arrow
PyCapsule interface, so `pl.Series(x)`, `pa.array(x)` and `np.asarray(x)` read
them without copying. `==`, `<` etc. are element-wise and return boolean Rles;
`a.equals(b)` compares whole sequences.

Build it locally with `cd crates/runlength-py && maturin develop --release`.

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
| Rearranging | `shift`, `reversed`, `concat`, `ranges_where` |
| Many inputs | `plan_many`, `zip_many` |
| Reductions | `sum`, `mean`, `variance(ddof)`, `std(ddof)`, `min`, `max` (all weighted by run length) |
| Batches | `RleBatch<K, V>`: many sequences under any hashable keys, joins with `How::{Outer, Inner, Left, Right}`, `collapse`, parallel with rayon |
| Coverage | `coverage_sum`, `coverage_count`, `coverage_max`, `coverage_min` (with `_by_key` variants), and `overlap_plan` for any other aggregation |

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
cd crates/runlength-py && maturin develop --release && pytest tests

# Regenerate the pyrle cross-check cases (needs a Python that imports pyrle):
PYTHONPATH=~/code/pyrle python3.12 scripts/gen_pyrle_fixtures.py
```
