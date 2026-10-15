//! Combining any number of sequences in one pass.

use crate::error::{Error, Result};
use crate::identical::Identical;
use crate::rle::{Rle, RleLike, RleRef};

/// How to combine `k` sequences, computed from their run ends alone: the
/// `k`-input version of [`Plan`](crate::Plan).
///
/// Segment `s` ends at `ends[s]` and is covered by run `idx[i][s]` of input
/// `i`. An index equal to input `i`'s run count means the segment lies past
/// its end.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanMany {
    /// Cumulative ends of the output segments.
    pub ends: Vec<i64>,
    /// One index vector per input.
    pub idx: Vec<Vec<usize>>,
}

/// Walks the union of the breakpoints of several inputs, calling
/// `visit(end, positions)` once per segment.
fn walk(inputs: &[&[i64]], mut visit: impl FnMut(i64, &[usize])) {
    let mut pos = vec![0usize; inputs.len()];
    loop {
        let end = inputs
            .iter()
            .zip(&pos)
            .filter_map(|(ends, &p)| ends.get(p).copied())
            .min();
        let Some(end) = end else { return };
        visit(end, &pos);
        for (ends, p) in inputs.iter().zip(pos.iter_mut()) {
            if ends.get(*p) == Some(&end) {
                *p += 1;
            }
        }
    }
}

/// Computes the [`PlanMany`] for sequences with the given run ends.
pub fn plan_many(inputs: &[&[i64]]) -> PlanMany {
    let mut out = PlanMany {
        ends: Vec::new(),
        idx: vec![Vec::new(); inputs.len()],
    };
    walk(inputs, |end, pos| {
        out.ends.push(end);
        for (column, &p) in out.idx.iter_mut().zip(pos) {
            column.push(p);
        }
    });
    out
}

/// Combines several sequences position by position with `f`, which receives
/// one value per input, and merges the result.
///
/// Shorter inputs are padded with `fill`; with no fill value, unequal lengths
/// are an error.
///
/// ```
/// use runlength::{zip_many, Rle, RleLike};
///
/// let a = Rle::from_runs(&[2, 2], vec![1, 2]).unwrap();
/// let b = Rle::from_runs(&[1, 3], vec![10, 20]).unwrap();
/// let c = Rle::from_runs(&[3], vec![100]).unwrap();
/// let total = zip_many(&[a.view(), b.view(), c.view()], Some(&0), |v| v.iter().copied().sum::<i32>()).unwrap();
/// assert_eq!(total.to_dense(), vec![111, 121, 122, 22]);
/// ```
pub fn zip_many<V, C, F>(inputs: &[RleRef<'_, V>], fill: Option<&V>, mut f: F) -> Result<Rle<C>>
where
    C: Identical,
    F: FnMut(&[&V]) -> C,
{
    let lengths = inputs.iter().map(|r| r.len());
    let (min, max) = (
        lengths.clone().min().unwrap_or(0),
        lengths.max().unwrap_or(0),
    );
    if min != max && fill.is_none() {
        return Err(Error::UnequalLength {
            left: min,
            right: max,
        });
    }
    let ends: Vec<&[i64]> = inputs.iter().map(|r| r.ends).collect();
    let mut out = Rle::new();
    let mut values: Vec<&V> = Vec::with_capacity(inputs.len());
    walk(&ends, |end, pos| {
        values.clear();
        values.extend(inputs.iter().zip(pos).map(|(r, &p)| {
            r.values
                .get(p)
                .or(fill)
                .expect("the length check guarantees a fill value")
        }));
        out.push_merged(end, f(&values));
    });
    Ok(out)
}
