//! A tour of the Rust API, shown on the documentation site's Rust page:
//! `cargo run --example tour`. Each `assert_eq!` shows what a line prints.

use runlength::prelude::*;
use runlength::{coverage_count, coverage_count_by_key, coverage_sum, overlap_plan};

fn main() -> runlength::Result<()> {
    // --8<-- [start:basics]
    // Runs of length 3 and 2, holding 1 and 2: the sequence 1 1 1 2 2.
    let a = Rle::from_runs(&[3, 2], vec![1, 2])?;
    assert_eq!(format!("{a:?}"), "Rle { runs: [3, 2], values: [1, 2] }");
    assert_eq!(a.to_dense(), vec![1, 1, 1, 2, 2]);
    assert_eq!((a.len(), a.num_runs(), a.ends()), (5, 2, &[3, 5][..]));
    assert_eq!(a.get(3), Some(&2));

    // Values can be any type; neighbours merge when they are identical.
    let labels = Rle::from_dense(["x", "x", "y"]);
    assert_eq!(
        format!("{labels:?}"),
        r#"Rle { runs: [2, 1], values: ["x", "y"] }"#
    );
    // --8<-- [end:basics]

    // --8<-- [start:combine]
    let b = Rle::from_runs(&[1, 4], vec![10, 20])?;

    // Operators work position by position and merge the result.
    let sum = &a + &b;
    assert_eq!(
        format!("{sum:?}"),
        "Rle { runs: [1, 2, 2], values: [11, 21, 22] }"
    );
    let less = a.elem_lt(&b);
    assert_eq!(format!("{less:?}"), "Rle { runs: [5], values: [true] }");

    // The plan: where the output segments end, and which run of each input
    // covers each segment. It never looks at values.
    let plan = a.plan_with(&b);
    assert_eq!(plan.ends, vec![1, 3, 5]);
    assert_eq!((plan.idx_a, plan.idx_b), (vec![0, 0, 1], vec![0, 1, 1]));

    // So any function works, on any value types. No fill values: the lengths
    // must match.
    let pairs = a.zip_with(&b, None, None, |x, y| format!("{x}/{y}"))?;
    assert_eq!(
        format!("{pairs:?}"),
        r#"Rle { runs: [1, 2, 2], values: ["1/10", "1/20", "2/20"] }"#
    );
    // --8<-- [end:combine]

    // --8<-- [start:coverage]
    // How many of the intervals [0, 3), [2, 4) and [8, 10) cover each position.
    let starts = [0, 2, 8];
    let ends = [3, 4, 10];
    let cov = coverage_count(&starts, &ends, None)?;
    assert_eq!(
        format!("{cov:?}"),
        "Rle { runs: [2, 1, 1, 4, 2], values: [1, 2, 1, 0, 1] }"
    );
    // Or the sum of a value per interval.
    let scores = coverage_sum(&starts, &ends, &[0.5, 2.0, 1.0], None)?;
    assert_eq!(scores.max(), Some(&2.5));

    // Mean coverage over many slices at once, without copying them.
    let means = cov.slice_reduce(&[0, 4], &[4, 10], |s| s.mean())?;
    assert_eq!(means, vec![Some(1.25), Some(1.0 / 3.0)]);

    // Which intervals cover each segment, for any other aggregation.
    let plan = overlap_plan(&starts, &ends, None)?;
    assert_eq!(plan.ends, vec![2, 3, 4, 8, 10]);
    assert_eq!(plan.ids(1), &[0, 1]);
    // --8<-- [end:coverage]

    // --8<-- [start:batch]
    // One sequence per key; keys can be any hashable type.
    let keys = ["chr1", "chr1", "chr2"];
    let batch = coverage_count_by_key(&keys, &[0, 2, 5], &[3, 4, 7])?;
    assert_eq!(
        format!("{batch:?}"),
        r#"{"chr1": RleRef { runs: [2, 1, 1], values: [1, 2, 1] }, "chr2": RleRef { runs: [5, 2], values: [0, 1] }}"#
    );
    assert_eq!(batch.get("chr2").map(|r| r.len()), Some(7));

    // Operators join two batches on their keys (an outer join) in parallel.
    let doubled = &batch + &batch;
    assert_eq!(doubled.get("chr1").unwrap().to_dense(), vec![2, 2, 4, 2]);

    // Or pick the join and the function yourself.
    let other = coverage_count_by_key(&["chr2"], &[0], &[7])?;
    let both = batch.zip_with(&other, How::Inner, Some(&0), Some(&0), |x, y| x * y)?;
    assert_eq!(both.keys(), &["chr2"]);
    // --8<-- [end:batch]

    println!("All examples ran.");
    Ok(())
}
