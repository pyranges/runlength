//! Length-weighted reductions over `(run length, value)` pairs, shared by
//! whole sequences and [`SliceView`](crate::SliceView)s.

use crate::num::Num;
use crate::rle::RleRef;

/// `(run length, value)` for each run.
pub(crate) fn weighted<'a, V: Copy>(
    rle: RleRef<'a, V>,
) -> impl Iterator<Item = (i64, V)> + Clone + 'a {
    let mut prev = 0;
    rle.ends.iter().zip(rle.values).map(move |(&end, &value)| {
        let run = end - prev;
        prev = end;
        (run, value)
    })
}

pub(crate) fn sum<V: Num>(runs: impl Iterator<Item = (i64, V)>) -> V {
    runs.fold(V::default(), |acc, (run, value)| {
        acc + value * V::from_i64(run)
    })
}

/// The mean of runs covering `len` positions in total.
pub(crate) fn mean<V: Num>(runs: impl Iterator<Item = (i64, V)>, len: i64) -> Option<f64> {
    if len == 0 {
        return None;
    }
    let total: f64 = runs.map(|(run, value)| run as f64 * value.to_f64()).sum();
    Some(total / len as f64)
}

pub(crate) fn variance<V: Num>(
    runs: impl Iterator<Item = (i64, V)> + Clone,
    len: i64,
    ddof: i64,
) -> Option<f64> {
    let denominator = len - ddof;
    if denominator <= 0 {
        return None;
    }
    let mean = mean(runs.clone(), len)?;
    let squares: f64 = runs
        .map(|(run, value)| run as f64 * (value.to_f64() - mean).powi(2))
        .sum();
    Some(squares / denominator as f64)
}

/// The value `v` for which `better(v, other)` holds against every other
/// comparable value. Values not comparable with themselves (`NaN`) are skipped.
pub(crate) fn extreme<V: PartialOrd>(values: &[V], better: impl Fn(&V, &V) -> bool) -> Option<&V> {
    values
        .iter()
        .filter(|v| v.partial_cmp(v).is_some())
        .fold(None, |best, v| match best {
            Some(b) if !better(v, b) => Some(b),
            _ => Some(v),
        })
}
