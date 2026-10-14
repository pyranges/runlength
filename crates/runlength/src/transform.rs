//! Rearranging sequences: shifting, reversing, joining, and finding ranges.

use crate::identical::Identical;
use crate::rle::{Rle, RleLike, RleRef};

/// Joins sequences end to end. Identical neighbours are merged, including at
/// the joins.
///
/// ```
/// use runlength::{concat, Rle, RleLike};
///
/// let a = Rle::from_runs(&[2], vec![1]).unwrap();
/// let b = Rle::from_runs(&[1, 1], vec![1, 2]).unwrap();
/// let c = concat([a.view(), b.view()]);
/// assert_eq!(c.runs(), vec![3, 1]);
/// ```
pub fn concat<'a, V, I>(rles: I) -> Rle<V>
where
    V: Clone + Identical + 'a,
    I: IntoIterator<Item = RleRef<'a, V>>,
{
    let mut out = Rle::new();
    for rle in rles {
        let offset = out.len();
        for (&end, value) in rle.ends.iter().zip(rle.values) {
            out.push_merged(offset + end, value.clone());
        }
    }
    out
}

pub(crate) fn shift<V: Clone + Identical>(rle: RleRef<'_, V>, distance: i64, fill: V) -> Rle<V> {
    let len = rle.len();
    if distance == 0 {
        return rle.merged();
    }
    if distance.unsigned_abs() >= len as u64 {
        return Rle::repeat(fill, len).expect("length is not negative");
    }
    if distance > 0 {
        let mut out = Rle::new();
        out.push_merged(distance, fill);
        for (_, end, value) in rle.slice(0, len - distance).iter_runs() {
            out.push_merged(end + distance, value.clone());
        }
        out
    } else {
        let mut out = rle.slice(-distance, len).merge();
        out.extend_to(len, fill)
            .expect("the slice is shorter than the sequence");
        out
    }
}

pub(crate) fn reversed<V: Clone>(rle: RleRef<'_, V>) -> Rle<V> {
    let len = rle.len();
    let n = rle.ends.len();
    let ends = (0..n)
        .rev()
        .map(|i| len - if i == 0 { 0 } else { rle.ends[i - 1] })
        .collect();
    let values = rle.values.iter().rev().cloned().collect();
    Rle { ends, values }
}

pub(crate) fn ranges_where<V>(
    rle: RleRef<'_, V>,
    mut pred: impl FnMut(&V) -> bool,
) -> (Vec<i64>, Vec<i64>) {
    let (mut starts, mut ends) = (Vec::new(), Vec::new());
    for (start, end, value) in rle.iter_runs() {
        if !pred(value) {
            continue;
        }
        match ends.last_mut() {
            Some(last) if *last == start => *last = end,
            _ => {
                starts.push(start);
                ends.push(end);
            }
        }
    }
    (starts, ends)
}
