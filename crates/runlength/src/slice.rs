use std::fmt;

use rayon::prelude::*;

use crate::batch::RleBatch;
use crate::error::{Error, Result};
use crate::num::Num;
use crate::reduce;
use crate::rle::{Rle, RleLike, RleRef};

pub(crate) fn run_index(ends: &[i64], pos: i64) -> Option<usize> {
    let len = ends.last().copied().unwrap_or(0);
    if pos < 0 || pos >= len {
        return None;
    }
    Some(ends.partition_point(|&end| end <= pos))
}

/// Checks that every interval has `0 <= start <= end`.
pub(crate) fn validate_intervals(starts: &[i64], ends: &[i64]) -> Result<()> {
    if starts.len() != ends.len() {
        return Err(Error::LengthMismatch {
            what: "starts and ends",
            left: starts.len(),
            right: ends.len(),
        });
    }
    for (index, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start < 0 || end < start {
            return Err(Error::InvalidInterval { index, start, end });
        }
    }
    Ok(())
}

/// Appends the runs of `rle[start..end]`, shifted to start at 0, to `ends`
/// and `values`. The range is clamped to the sequence.
pub(crate) fn slice_into<V: Clone>(
    rle: RleRef<'_, V>,
    start: i64,
    end: i64,
    ends: &mut Vec<i64>,
    values: &mut Vec<V>,
) {
    let (start, end) = (start.max(0), end.min(rle.len()));
    if start >= end {
        return;
    }
    let first = rle.ends.partition_point(|&e| e <= start);
    let last = rle.ends.partition_point(|&e| e < end);
    ends.extend(rle.ends[first..last].iter().map(|&e| e - start));
    ends.push(end - start);
    values.extend_from_slice(&rle.values[first..=last]);
}

pub(crate) fn slice<V: Clone>(rle: RleRef<'_, V>, start: i64, end: i64) -> Rle<V> {
    let mut out = Rle::new();
    slice_into(rle, start, end, &mut out.ends, &mut out.values);
    out
}

pub(crate) fn getitems<V: Clone>(
    rle: RleRef<'_, V>,
    starts: &[i64],
    ends: &[i64],
) -> Result<RleBatch<usize, V>> {
    validate_intervals(starts, ends)?;
    Ok(RleBatch::from_slices(
        (0..starts.len()).collect(),
        |k, out_ends, out_values| slice_into(rle, starts[k], ends[k], out_ends, out_values),
    ))
}

/// A borrowed slice `start..end` of a sequence that is not copied, for
/// reducing many slices cheaply. Positions are relative to `start`.
///
/// ```
/// use runlength::{Rle, RleLike};
///
/// let rle = Rle::from_runs(&[3, 2, 4], vec![1, 5, 2]).unwrap();
/// let view = rle.slice_view(2, 6);
/// assert_eq!(view.len(), 4);
/// assert_eq!(view.sum(), 1 + 5 + 5 + 2);
/// assert_eq!(view.to_rle().to_dense(), vec![1, 5, 5, 2]);
/// ```
pub struct SliceView<'a, V> {
    /// The ends of the runs the slice touches, at the sequence's positions.
    ends: &'a [i64],
    values: &'a [V],
    start: i64,
    end: i64,
}

impl<V> Clone for SliceView<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for SliceView<'_, V> {}

impl<'a, V> SliceView<'a, V> {
    /// The positions `start..end` of `rle`, clamped to the sequence.
    pub(crate) fn new(rle: RleRef<'a, V>, start: i64, end: i64) -> Self {
        let (start, end) = (start.max(0), end.min(rle.len()));
        if start >= end {
            return SliceView {
                ends: &[],
                values: &[],
                start: 0,
                end: 0,
            };
        }
        let first = rle.ends.partition_point(|&e| e <= start);
        let last = rle.ends.partition_point(|&e| e < end);
        SliceView {
            ends: &rle.ends[first..=last],
            values: &rle.values[first..=last],
            start,
            end,
        }
    }

    /// The number of positions.
    pub fn len(&self) -> i64 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// The number of runs the slice touches.
    pub fn num_runs(&self) -> usize {
        self.values.len()
    }

    /// One value per run.
    pub fn values(&self) -> &'a [V] {
        self.values
    }

    /// Iterates over `(start, end, value)` for each run, relative to the
    /// start of the slice.
    pub fn iter_runs(&self) -> impl Iterator<Item = (i64, i64, &'a V)> + Clone + 'a {
        let (start, end) = (self.start, self.end);
        let mut prev = start;
        self.ends.iter().zip(self.values).map(move |(&e, value)| {
            let (run_start, run_end) = (prev - start, e.min(end) - start);
            prev = e;
            (run_start, run_end, value)
        })
    }

    /// `(run length, value)` for each run.
    fn weighted(&self) -> impl Iterator<Item = (i64, V)> + Clone + 'a
    where
        V: Copy,
    {
        self.iter_runs().map(|(s, e, &v)| (e - s, v))
    }

    /// Copies the slice into an owned sequence starting at 0.
    pub fn to_rle(&self) -> Rle<V>
    where
        V: Clone,
    {
        let mut out = Rle::new();
        for (_, end, value) in self.iter_runs() {
            out.ends.push(end);
            out.values.push(value.clone());
        }
        out
    }

    /// The sum over all positions.
    pub fn sum(&self) -> V
    where
        V: Num,
    {
        reduce::sum(self.weighted())
    }

    /// The mean over all positions, or `None` if the slice is empty.
    pub fn mean(&self) -> Option<f64>
    where
        V: Num,
    {
        reduce::mean(self.weighted(), self.len())
    }

    /// The variance with `ddof` delta degrees of freedom, or `None` if
    /// `len() <= ddof`.
    pub fn variance(&self, ddof: i64) -> Option<f64>
    where
        V: Num,
    {
        reduce::variance(self.weighted(), self.len(), ddof)
    }

    /// The square root of [`variance`](SliceView::variance).
    pub fn std(&self, ddof: i64) -> Option<f64>
    where
        V: Num,
    {
        self.variance(ddof).map(f64::sqrt)
    }

    /// The smallest value, ignoring values not comparable with themselves.
    pub fn min(&self) -> Option<&'a V>
    where
        V: PartialOrd,
    {
        reduce::extreme(self.values, |a, b| a < b)
    }

    /// The largest value, ignoring values not comparable with themselves.
    pub fn max(&self) -> Option<&'a V>
    where
        V: PartialOrd,
    {
        reduce::extreme(self.values, |a, b| a > b)
    }
}

impl<V: fmt::Debug> fmt::Debug for SliceView<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.iter_runs().map(|(s, e, v)| (e - s, v)))
            .finish()
    }
}

/// Minimum slices per rayon task; reducing one slice is cheap.
const SLICES_PER_TASK: usize = 1024;

/// Reduces slice `k` of the sequence `rle_of(k)` with `f`, in parallel.
pub(crate) fn slice_reduce<'a, V, T, R, F>(
    starts: &[i64],
    ends: &[i64],
    rle_of: R,
    f: F,
) -> Result<Vec<T>>
where
    V: Sync + 'a,
    T: Send,
    R: Fn(usize) -> RleRef<'a, V> + Sync,
    F: Fn(SliceView<'a, V>) -> T + Sync,
{
    validate_intervals(starts, ends)?;
    Ok((0..starts.len())
        .into_par_iter()
        .with_min_len(SLICES_PER_TASK)
        .map(|k| f(SliceView::new(rle_of(k), starts[k], ends[k])))
        .collect())
}
