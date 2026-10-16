//! Building sequences from intervals.
//!
//! Intervals are half-open, `[start, end)`. Positions no interval covers are
//! 0 (or `None` in the `_opt` variants). Empty intervals (`start == end`)
//! cover nothing, but still count towards the length, which is the largest
//! interval end.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;
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
/// Intervals ending at a position are removed before those starting there
/// are added, and when no interval is active the running sum is reset to
/// exactly zero, so floating-point rounding cannot carry one interval's value
/// into a gap or into an interval that merely touches it. NaN and infinite
/// weights are counted rather than summed, since subtracting them again would
/// give NaN: a segment is NaN while a NaN (or both infinities) cover it, and
/// infinite while one infinity does.
fn sweep<V: Num>(
    starts: &[i64],
    ends: &[i64],
    idx: &[usize],
    weight: impl Fn(usize) -> V,
) -> Sweep<V> {
    // Integers are always finite, and so are most floats: skip the
    // bookkeeping then.
    if idx.iter().all(|&i| weight(i).to_f64().is_finite()) {
        sweep_impl::<V, false>(starts, ends, idx, weight)
    } else {
        sweep_impl::<V, true>(starts, ends, idx, weight)
    }
}

fn sweep_impl<V: Num, const NONFINITE: bool>(
    starts: &[i64],
    ends: &[i64],
    idx: &[usize],
    weight: impl Fn(usize) -> V,
) -> Sweep<V> {
    // Sort (position, interval) pairs rather than indices by an indirect key:
    // contiguous keys sort much faster. Ties break by interval, so float sums
    // always add up in the same order.
    let mut by_start: Vec<(i64, usize)> = idx
        .iter()
        .filter(|&&i| starts[i] < ends[i])
        .map(|&i| (starts[i], i))
        .collect();
    let mut by_end: Vec<(i64, usize)> = by_start.iter().map(|&(_, i)| (ends[i], i)).collect();
    sort_pairs(&mut by_start);
    sort_pairs(&mut by_end);

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
    let mut nonfinite = NonFinite::default();
    while e < by_end.len() {
        let next_start = by_start.get(s).map_or(i64::MAX, |&(p, _)| p);
        let p = next_start.min(by_end[e].0);
        if p > pos {
            out.ends.push(p);
            let value = if NONFINITE { nonfinite.value() } else { None };
            out.sums.push(value.unwrap_or(sum));
            out.counts.push(count);
            pos = p;
        }
        while e < by_end.len() && by_end[e].0 == p {
            let w = weight(by_end[e].1);
            if !(NONFINITE && nonfinite.update(w, -1)) {
                sum = sum - w;
            }
            count -= 1;
            e += 1;
        }
        if count == 0 {
            sum = V::default();
        }
        while s < by_start.len() && by_start[s].0 == p {
            let w = weight(by_start[s].1);
            if !(NONFINITE && nonfinite.update(w, 1)) {
                sum = sum + w;
            }
            count += 1;
            s += 1;
        }
    }
    out
}

/// The NaN and infinite weights covering the current position.
struct NonFinite<V> {
    nan: i64,
    pos_inf: i64,
    neg_inf: i64,
    /// One weight of each kind, to return as the segment's value.
    nan_value: Option<V>,
    pos_inf_value: Option<V>,
    neg_inf_value: Option<V>,
}

impl<V> Default for NonFinite<V> {
    fn default() -> Self {
        NonFinite {
            nan: 0,
            pos_inf: 0,
            neg_inf: 0,
            nan_value: None,
            pos_inf_value: None,
            neg_inf_value: None,
        }
    }
}

impl<V: Num> NonFinite<V> {
    /// Counts `w` in (`delta` 1) or out (-1) if it is NaN or infinite, and
    /// returns whether it was.
    #[inline]
    fn update(&mut self, w: V, delta: i64) -> bool {
        let f = w.to_f64();
        if f.is_finite() {
            return false;
        }
        let (count, value) = if f.is_nan() {
            (&mut self.nan, &mut self.nan_value)
        } else if f > 0.0 {
            (&mut self.pos_inf, &mut self.pos_inf_value)
        } else {
            (&mut self.neg_inf, &mut self.neg_inf_value)
        };
        *count += delta;
        value.get_or_insert(w);
        true
    }

    /// The segment's value if a NaN or infinity covers it.
    fn value(&self) -> Option<V> {
        let pos = (self.pos_inf > 0).then_some(self.pos_inf_value).flatten();
        let neg = (self.neg_inf > 0).then_some(self.neg_inf_value).flatten();
        if self.nan > 0 {
            return self.nan_value;
        }
        match (pos, neg) {
            (Some(p), Some(n)) => Some(p + n),
            (p, n) => p.or(n),
        }
    }
}

/// Sorts in parallel when there is enough work to split.
fn sort_pairs(pairs: &mut [(i64, usize)]) {
    if pairs.len() > 1 << 16 {
        pairs.par_sort_unstable();
    } else {
        pairs.sort_unstable();
    }
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
    let rle = finish(
        sweep(starts, ends, &all(starts.len()), |i| values[i]),
        |sum, _| sum,
    );
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
    let rle = finish(
        sweep(starts, ends, &all(starts.len()), |_| 1i64),
        |_, count| count,
    );
    pad(rle, length, 0)
}

/// Groups interval indices by key, in order of first appearance.
fn group_by_key<K: Hash + Eq + Clone>(keys: &[K]) -> (Vec<K>, Vec<Vec<usize>>) {
    let mut positions: FxHashMap<&K, usize> = FxHashMap::default();
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
pub fn coverage_sum_by_key<K, V>(
    keys: &[K],
    starts: &[i64],
    ends: &[i64],
    values: &[V],
) -> Result<RleBatch<K, V>>
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
pub fn coverage_count_by_key<K>(
    keys: &[K],
    starts: &[i64],
    ends: &[i64],
) -> Result<RleBatch<K, i64>>
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

/// Orders values for max/min coverage: comparable values by their order
/// (reversed for min), and values not comparable with themselves (`NaN`)
/// below every comparable value, so they only win when nothing else covers.
struct Ranked<'a, V> {
    value: &'a V,
    local: usize,
    max: bool,
}

impl<V: PartialOrd> Ranked<'_, V> {
    fn rank(&self, other: &Self) -> Ordering {
        let comparable = |v: &V| v.partial_cmp(v).is_some();
        let by_value = match (comparable(self.value), comparable(other.value)) {
            (true, true) => {
                let order = self
                    .value
                    .partial_cmp(other.value)
                    .unwrap_or(Ordering::Equal);
                if self.max { order } else { order.reverse() }
            }
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => Ordering::Equal,
        };
        by_value.then(self.local.cmp(&other.local))
    }
}

impl<V: PartialOrd> PartialEq for Ranked<'_, V> {
    fn eq(&self, other: &Self) -> bool {
        self.rank(other) == Ordering::Equal
    }
}
impl<V: PartialOrd> Eq for Ranked<'_, V> {}
impl<V: PartialOrd> PartialOrd for Ranked<'_, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<V: PartialOrd> Ord for Ranked<'_, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank(other)
    }
}

/// The intervals of `idx` sorted by start and by end, as local positions
/// into `idx`, skipping empty intervals.
fn sorted_locals(starts: &[i64], ends: &[i64], idx: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut by_start: Vec<usize> = (0..idx.len())
        .filter(|&k| starts[idx[k]] < ends[idx[k]])
        .collect();
    let mut by_end = by_start.clone();
    by_start.sort_unstable_by_key(|&k| starts[idx[k]]);
    by_end.sort_unstable_by_key(|&k| ends[idx[k]]);
    (by_start, by_end)
}

/// Receives the events of [`sweep_events`].
trait Visitor {
    /// A segment ends at `end`; called before the intervals that start or
    /// stop there are applied.
    fn emit(&mut self, end: i64);
    /// Interval `k` (a local position in `idx`) starts.
    fn start(&mut self, k: usize);
    /// Interval `k` stops.
    fn stop(&mut self, k: usize);
}

/// Sweeps over the non-empty intervals in `idx` in position order.
fn sweep_events(starts: &[i64], ends: &[i64], idx: &[usize], visitor: &mut impl Visitor) {
    let (by_start, by_end) = sorted_locals(starts, ends, idx);
    let (mut s, mut e) = (0, 0);
    let mut pos = 0;
    while e < by_end.len() {
        let next_start = by_start.get(s).map_or(i64::MAX, |&k| starts[idx[k]]);
        let p = next_start.min(ends[idx[by_end[e]]]);
        if p > pos {
            visitor.emit(p);
            pos = p;
        }
        while s < by_start.len() && starts[idx[by_start[s]]] == p {
            visitor.start(by_start[s]);
            s += 1;
        }
        while e < by_end.len() && ends[idx[by_end[e]]] == p {
            visitor.stop(by_end[e]);
            e += 1;
        }
    }
}

fn max_end(ends: &[i64], idx: &[usize]) -> i64 {
    idx.iter().map(|&i| ends[i]).max().unwrap_or(0)
}

/// Tracks the largest (or smallest) active value with a heap, removing
/// stopped intervals lazily when they reach the top.
struct Extreme<'a, V> {
    idx: &'a [usize],
    values: &'a [V],
    max: bool,
    active: Vec<bool>,
    heap: BinaryHeap<Ranked<'a, V>>,
    out: Rle<Option<V>>,
}

impl<V: PartialOrd + Clone + Identical> Visitor for Extreme<'_, V> {
    fn emit(&mut self, end: i64) {
        while self.heap.peek().is_some_and(|top| !self.active[top.local]) {
            self.heap.pop();
        }
        let value = self.heap.peek().map(|top| top.value.clone());
        self.out.push_merged(end, value);
    }

    fn start(&mut self, k: usize) {
        self.active[k] = true;
        self.heap.push(Ranked {
            value: &self.values[self.idx[k]],
            local: k,
            max: self.max,
        });
    }

    fn stop(&mut self, k: usize) {
        self.active[k] = false;
    }
}

fn extreme<V>(
    starts: &[i64],
    ends: &[i64],
    idx: &[usize],
    values: &[V],
    max: bool,
) -> Rle<Option<V>>
where
    V: PartialOrd + Clone + Identical,
{
    let mut visitor = Extreme {
        idx,
        values,
        max,
        active: vec![false; idx.len()],
        heap: BinaryHeap::new(),
        out: Rle::new(),
    };
    sweep_events(starts, ends, idx, &mut visitor);
    let mut out = visitor.out;
    out.extend_to(max_end(ends, idx), None)
        .expect("the sweep never passes the largest end");
    out
}

/// The largest of `values` over the intervals covering each position, or
/// `None` where no interval covers it. Works for any ordered value type;
/// `NaN` only wins where nothing else covers the position.
pub fn coverage_max<V>(
    starts: &[i64],
    ends: &[i64],
    values: &[V],
    length: Option<i64>,
) -> Result<Rle<Option<V>>>
where
    V: PartialOrd + Clone + Identical,
{
    validate_intervals(starts, ends)?;
    check_values(starts, values)?;
    pad(
        extreme(starts, ends, &all(starts.len()), values, true),
        length,
        None,
    )
}

/// The smallest of `values` over the intervals covering each position; see
/// [`coverage_max`].
pub fn coverage_min<V>(
    starts: &[i64],
    ends: &[i64],
    values: &[V],
    length: Option<i64>,
) -> Result<Rle<Option<V>>>
where
    V: PartialOrd + Clone + Identical,
{
    validate_intervals(starts, ends)?;
    check_values(starts, values)?;
    pad(
        extreme(starts, ends, &all(starts.len()), values, false),
        length,
        None,
    )
}

fn extreme_by_key<K, V>(
    keys: &[K],
    starts: &[i64],
    ends: &[i64],
    values: &[V],
    max: bool,
) -> Result<RleBatch<K, Option<V>>>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: PartialOrd + Clone + Identical + Send + Sync,
{
    validate_intervals(starts, ends)?;
    check_keys(keys, starts)?;
    check_values(starts, values)?;
    let (unique, groups) = group_by_key(keys);
    let rles = groups
        .par_iter()
        .map(|idx| extreme(starts, ends, idx, values, max))
        .collect();
    Ok(RleBatch::from_rles_unchecked(unique, rles))
}

/// [`coverage_max`] per key, computed in parallel.
pub fn coverage_max_by_key<K, V>(
    keys: &[K],
    starts: &[i64],
    ends: &[i64],
    values: &[V],
) -> Result<RleBatch<K, Option<V>>>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: PartialOrd + Clone + Identical + Send + Sync,
{
    extreme_by_key(keys, starts, ends, values, true)
}

/// [`coverage_min`] per key, computed in parallel.
pub fn coverage_min_by_key<K, V>(
    keys: &[K],
    starts: &[i64],
    ends: &[i64],
    values: &[V],
) -> Result<RleBatch<K, Option<V>>>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: PartialOrd + Clone + Identical + Send + Sync,
{
    extreme_by_key(keys, starts, ends, values, false)
}

/// Which intervals cover each segment, for aggregating interval values in
/// any way.
///
/// Segments are cut at every interval start and end, and cover
/// `0..max(end)` (or `0..length`). Segment `s` ends at `ends[s]` and is
/// covered by intervals `interval_ids[offsets[s]..offsets[s + 1]]` (sorted;
/// none where uncovered). This is the same layout as an Arrow list column.
///
/// The ids can be large when many intervals overlap deeply; for sums, counts,
/// maxima and minima, the `coverage_*` functions avoid building them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlapPlan {
    /// Cumulative ends of the segments.
    pub ends: Vec<i64>,
    /// Where each segment's ids start in `interval_ids`; `num_segments() + 1`
    /// entries.
    pub offsets: Vec<usize>,
    /// The intervals covering each segment, as positions in the input.
    pub interval_ids: Vec<usize>,
}

impl OverlapPlan {
    /// The number of segments.
    pub fn num_segments(&self) -> usize {
        self.ends.len()
    }

    /// The ids of the intervals covering segment `s`.
    pub fn ids(&self, s: usize) -> &[usize] {
        &self.interval_ids[self.offsets[s]..self.offsets[s + 1]]
    }
}

/// Tracks the active intervals, with each one's slot for O(1) removal.
struct Overlap<'a> {
    idx: &'a [usize],
    active: Vec<usize>,
    slot: Vec<usize>,
    out: OverlapPlan,
}

impl Visitor for Overlap<'_> {
    fn emit(&mut self, end: i64) {
        let first = self.out.interval_ids.len();
        self.out
            .interval_ids
            .extend(self.active.iter().map(|&k| self.idx[k]));
        self.out.interval_ids[first..].sort_unstable();
        self.out.ends.push(end);
        self.out.offsets.push(self.out.interval_ids.len());
    }

    fn start(&mut self, k: usize) {
        self.slot[k] = self.active.len();
        self.active.push(k);
    }

    fn stop(&mut self, k: usize) {
        let i = self.slot[k];
        self.active.swap_remove(i);
        if let Some(&moved) = self.active.get(i) {
            self.slot[moved] = i;
        }
    }
}

fn overlap(starts: &[i64], ends: &[i64], idx: &[usize]) -> OverlapPlan {
    let mut visitor = Overlap {
        idx,
        active: Vec::new(),
        slot: vec![usize::MAX; idx.len()],
        out: OverlapPlan {
            ends: Vec::new(),
            offsets: vec![0],
            interval_ids: Vec::new(),
        },
    };
    sweep_events(starts, ends, idx, &mut visitor);
    let mut out = visitor.out;
    let length = max_end(ends, idx);
    if length > out.ends.last().copied().unwrap_or(0) {
        out.ends.push(length);
        out.offsets.push(out.interval_ids.len());
    }
    out
}

/// Computes the [`OverlapPlan`] for intervals `[start, end)`.
///
/// ```
/// use runlength::overlap_plan;
///
/// let p = overlap_plan(&[0, 2, 3], &[5, 7, 4], None).unwrap();
/// assert_eq!(p.ends, vec![2, 3, 4, 5, 7]);
/// assert_eq!(p.ids(2), &[0, 1, 2]);
/// ```
pub fn overlap_plan(starts: &[i64], ends: &[i64], length: Option<i64>) -> Result<OverlapPlan> {
    validate_intervals(starts, ends)?;
    let mut plan = overlap(starts, ends, &all(starts.len()));
    if let Some(length) = length {
        let current = plan.ends.last().copied().unwrap_or(0);
        if length < current {
            return Err(Error::LengthTooShort {
                requested: length,
                required: current,
            });
        }
        if length > current {
            plan.ends.push(length);
            plan.offsets.push(plan.interval_ids.len());
        }
    }
    Ok(plan)
}

/// [`overlap_plan`] per key, computed in parallel. Returns the keys in order
/// of first appearance and one plan per key, covering `0..max(end)` of that
/// key's intervals. Interval ids are positions in the input, not per key.
///
/// ```
/// use runlength::overlap_plan_by_key;
///
/// let (keys, plans) = overlap_plan_by_key(&["a", "b", "a"], &[0, 0, 1], &[2, 1, 3]).unwrap();
/// assert_eq!(keys, vec!["a", "b"]);
/// assert_eq!(plans[0].ends, vec![1, 2, 3]);
/// assert_eq!(plans[0].ids(1), &[0, 2]);
/// assert_eq!(plans[1].ids(0), &[1]);
/// ```
pub fn overlap_plan_by_key<K>(
    keys: &[K],
    starts: &[i64],
    ends: &[i64],
) -> Result<(Vec<K>, Vec<OverlapPlan>)>
where
    K: Hash + Eq + Clone + Send + Sync,
{
    validate_intervals(starts, ends)?;
    check_keys(keys, starts)?;
    let (unique, groups) = group_by_key(keys);
    let plans = groups
        .par_iter()
        .map(|idx| overlap(starts, ends, idx))
        .collect();
    Ok((unique, plans))
}
