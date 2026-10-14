//! Building sequences from intervals.
//!
//! Intervals are half-open, `[start, end)`. Positions no interval covers are
//! 0 (or `None` in the `_opt` variants). Empty intervals (`start == end`)
//! cover nothing, but still count towards the length, which is the largest
//! interval end.

use std::collections::HashMap;
use std::hash::Hash;

use rayon::prelude::*;

use crate::batch::RleBatch;
use crate::error::{Error, Result};
use crate::identical::Identical;
use crate::num::Num;
use crate::rle::Rle;
use crate::slice::validate_intervals;

/// The raw output of a sweep: one segment per distinct breakpoint, with the
/// summed weight and the number of intervals covering it.
struct Sweep<V> {
    /// The largest interval end, including empty intervals.
    length: i64,
    ends: Vec<i64>,
    sums: Vec<V>,
    counts: Vec<i64>,
}

/// Sweeps over the intervals in `idx`, adding each interval's weight where it
/// starts and subtracting it where it ends.
///
/// When no interval is active the running sum is reset to exactly zero, so
/// floating-point rounding cannot leave small non-zero values in gaps.
fn sweep<V: Num>(starts: &[i64], ends: &[i64], idx: &[usize], weight: impl Fn(usize) -> V) -> Sweep<V> {
    let mut by_start: Vec<usize> = idx.iter().copied().filter(|&i| starts[i] < ends[i]).collect();
    let mut by_end = by_start.clone();
    by_start.sort_unstable_by_key(|&i| starts[i]);
    by_end.sort_unstable_by_key(|&i| ends[i]);

    let mut out = Sweep {
        length: idx.iter().map(|&i| ends[i]).max().unwrap_or(0),
        ends: Vec::with_capacity(2 * by_start.len()),
        sums: Vec::with_capacity(2 * by_start.len()),
        counts: Vec::with_capacity(2 * by_start.len()),
    };
    let (mut s, mut e) = (0, 0);
    let mut pos = 0;
    let mut sum = V::default();
    let mut count = 0i64;
    while e < by_end.len() {
        let next_start = by_start.get(s).map_or(i64::MAX, |&i| starts[i]);
        let p = next_start.min(ends[by_end[e]]);
        if p > pos {
            out.ends.push(p);
            out.sums.push(sum);
            out.counts.push(count);
            pos = p;
        }
        while s < by_start.len() && starts[by_start[s]] == p {
            sum = sum + weight(by_start[s]);
            count += 1;
            s += 1;
        }
        while e < by_end.len() && ends[by_end[e]] == p {
            sum = sum - weight(by_end[e]);
            count -= 1;
            e += 1;
        }
        if count == 0 {
            sum = V::default();
        }
    }
    out
}

/// Turns a sweep into a sequence; `value(sum, count)` gives each segment's
/// value. Empty intervals past the last covered position are padded with the
/// uncovered value.
fn finish<S: Default, V: Identical>(sweep: Sweep<S>, mut value: impl FnMut(S, i64) -> V) -> Rle<V> {
    let mut out = Rle::new();
    for ((end, sum), count) in sweep.ends.into_iter().zip(sweep.sums).zip(sweep.counts) {
        out.push_merged(end, value(sum, count));
    }
    out.extend_to(sweep.length, value(S::default(), 0))
        .expect("the sweep never goes past the largest end");
    out
}

fn pad<V: Identical>(mut rle: Rle<V>, length: Option<i64>, uncovered: V) -> Result<Rle<V>> {
    if let Some(length) = length {
        rle.extend_to(length, uncovered)?;
    }
    Ok(rle)
}

fn check_values<V>(starts: &[i64], values: &[V]) -> Result<()> {
    if values.len() != starts.len() {
        return Err(Error::LengthMismatch {
            what: "values and starts",
            left: values.len(),
            right: starts.len(),
        });
    }
    Ok(())
}

fn all(n: usize) -> Vec<usize> {
    (0..n).collect()
}

/// The sum of `values` over the intervals covering each position.
///
/// The result covers `0..max(end)`, or `0..length` if given (which must not
/// be shorter). Uncovered positions are 0.
///
/// ```
/// use runlength::{coverage_sum, RleLike};
///
/// let cov = coverage_sum(&[0, 2], &[3, 4], &[1, 10], None).unwrap();
/// assert_eq!(cov.to_dense(), vec![1, 1, 11, 10]);
/// ```
pub fn coverage_sum<V: Num + Identical>(
    starts: &[i64],
    ends: &[i64],
    values: &[V],
    length: Option<i64>,
) -> Result<Rle<V>> {
    validate_intervals(starts, ends)?;
    check_values(starts, values)?;
    let rle = finish(sweep(starts, ends, &all(starts.len()), |i| values[i]), |sum, _| sum);
    pad(rle, length, V::default())
}

/// Like [`coverage_sum`], but uncovered positions are `None` instead of 0.
pub fn coverage_sum_opt<V: Num + Identical>(
    starts: &[i64],
    ends: &[i64],
    values: &[V],
    length: Option<i64>,
) -> Result<Rle<Option<V>>> {
    validate_intervals(starts, ends)?;
    check_values(starts, values)?;
    let sweep = sweep(starts, ends, &all(starts.len()), |i| values[i]);
    let rle = finish(sweep, |sum, count| (count > 0).then_some(sum));
    pad(rle, length, None)
}

/// The number of intervals covering each position.
pub fn coverage_count(starts: &[i64], ends: &[i64], length: Option<i64>) -> Result<Rle<i64>> {
    validate_intervals(starts, ends)?;
    let rle = finish(sweep(starts, ends, &all(starts.len()), |_| 1i64), |_, count| count);
    pad(rle, length, 0)
}

/// Groups interval indices by key, in order of first appearance.
fn group_by_key<K: Hash + Eq + Clone>(keys: &[K]) -> (Vec<K>, Vec<Vec<usize>>) {
    let mut positions: HashMap<&K, usize> = HashMap::new();
    let mut unique = Vec::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        let g = *positions.entry(key).or_insert_with(|| {
            unique.push(key.clone());
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    (unique, groups)
}

fn check_keys<K>(keys: &[K], starts: &[i64]) -> Result<()> {
    if keys.len() != starts.len() {
        return Err(Error::LengthMismatch {
            what: "keys and starts",
            left: keys.len(),
            right: starts.len(),
        });
    }
    Ok(())
}

/// [`coverage_sum`] per key, computed in parallel. Keys appear in order of
/// first appearance; each sequence covers `0..max(end)` of its intervals.
pub fn coverage_sum_by_key<K, V>(keys: &[K], starts: &[i64], ends: &[i64], values: &[V]) -> Result<RleBatch<K, V>>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Num + Identical,
{
    validate_intervals(starts, ends)?;
    check_keys(keys, starts)?;
    check_values(starts, values)?;
    let (unique, groups) = group_by_key(keys);
    let rles = groups
        .par_iter()
        .map(|idx| finish(sweep(starts, ends, idx, |i| values[i]), |sum, _| sum))
        .collect();
    Ok(RleBatch::from_rles_unchecked(unique, rles))
}

/// [`coverage_count`] per key, computed in parallel. Keys appear in order of
/// first appearance; each sequence covers `0..max(end)` of its intervals.
pub fn coverage_count_by_key<K>(keys: &[K], starts: &[i64], ends: &[i64]) -> Result<RleBatch<K, i64>>
where
    K: Hash + Eq + Clone + Send + Sync,
{
    validate_intervals(starts, ends)?;
    check_keys(keys, starts)?;
    let (unique, groups) = group_by_key(keys);
    let rles = groups
        .par_iter()
        .map(|idx| finish(sweep(starts, ends, idx, |_| 1i64), |_, count| count))
        .collect();
    Ok(RleBatch::from_rles_unchecked(unique, rles))
}
