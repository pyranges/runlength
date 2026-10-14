use crate::num::Num;
use crate::rle::{RleLike, RleRef};

/// `(run length, value)` for each run.
fn weighted<V: Copy>(rle: RleRef<'_, V>) -> impl Iterator<Item = (i64, V)> + '_ {
    let mut prev = 0;
    rle.ends.iter().zip(rle.values).map(move |(&end, &value)| {
        let run = end - prev;
        prev = end;
        (run, value)
    })
}

pub(crate) fn sum<V: Num>(rle: RleRef<'_, V>) -> V {
    weighted(rle).fold(V::default(), |acc, (run, value)| acc + value * V::from_i64(run))
}

pub(crate) fn mean<V: Num>(rle: RleRef<'_, V>) -> Option<f64> {
    if rle.is_empty() {
        return None;
    }
    let total: f64 = weighted(rle).map(|(run, value)| run as f64 * value.to_f64()).sum();
    Some(total / rle.len() as f64)
}

pub(crate) fn variance<V: Num>(rle: RleRef<'_, V>, ddof: i64) -> Option<f64> {
    let denominator = rle.len() - ddof;
    if denominator <= 0 {
        return None;
    }
    let mean = mean(rle)?;
    let squares: f64 = weighted(rle)
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
