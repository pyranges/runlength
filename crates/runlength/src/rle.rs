use std::fmt;

use crate::batch::RleBatch;
use crate::error::{Error, Result};
use crate::identical::Identical;
use crate::merge::{self, MergePlan};
use crate::num::Num;
use crate::ops;
use crate::plan::{self, Plan, Segments};
use crate::reduce;
use crate::slice::{self, SliceView};
use crate::transform;

/// A run-length encoded sequence that owns its data.
///
/// Internally a sequence is stored as the cumulative *ends* of its runs plus
/// one value per run: runs `[3, 2]` with values `[a, b]` are stored as ends
/// `[3, 5]`. Run `i` covers positions `ends[i - 1]..ends[i]` (the first run
/// starts at 0). Every sequence starts at position 0.
///
/// Invariants: ends are strictly increasing and positive, so there are no
/// zero-length runs, and there is one value per run. Neighbouring runs are
/// *not* guaranteed to hold different values; see [`Rle::merge`].
#[derive(Clone)]
pub struct Rle<V> {
    pub(crate) ends: Vec<i64>,
    pub(crate) values: Vec<V>,
}

/// A borrowed run-length encoded sequence, for example one entry of an
/// [`RleBatch`]. It has the same invariants as [`Rle`].
pub struct RleRef<'a, V> {
    pub(crate) ends: &'a [i64],
    pub(crate) values: &'a [V],
}

impl<V> Clone for RleRef<'_, V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V> Copy for RleRef<'_, V> {}

pub(crate) fn validate_ends(ends: &[i64]) -> Result<()> {
    let mut prev = 0;
    for (index, &end) in ends.iter().enumerate() {
        if end <= prev {
            return Err(Error::InvalidEnds { index });
        }
        prev = end;
    }
    Ok(())
}

fn check_same_len(what: &'static str, left: usize, right: usize) -> Result<()> {
    if left != right {
        return Err(Error::LengthMismatch { what, left, right });
    }
    Ok(())
}

impl<V> Rle<V> {
    /// An empty sequence of length 0.
    pub fn new() -> Self {
        Rle {
            ends: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Builds a sequence from run lengths and one value per run.
    ///
    /// Zero-length runs are dropped. Negative runs and mismatched lengths are
    /// errors. Neighbouring runs with identical values are kept as they are.
    pub fn from_runs(runs: &[i64], values: Vec<V>) -> Result<Self> {
        check_same_len("runs and values", runs.len(), values.len())?;
        let mut out = Rle {
            ends: Vec::with_capacity(runs.len()),
            values: Vec::with_capacity(values.len()),
        };
        let mut total: i64 = 0;
        for (index, (&run, value)) in runs.iter().zip(values).enumerate() {
            if run < 0 {
                return Err(Error::NegativeRun { index, run });
            }
            if run == 0 {
                continue;
            }
            total = total.checked_add(run).ok_or(Error::Overflow)?;
            out.ends.push(total);
            out.values.push(value);
        }
        Ok(out)
    }

    /// Builds a sequence from cumulative run ends and one value per run.
    pub fn from_ends(ends: Vec<i64>, values: Vec<V>) -> Result<Self> {
        check_same_len("ends and values", ends.len(), values.len())?;
        validate_ends(&ends)?;
        Ok(Rle { ends, values })
    }

    /// A sequence of `length` copies of `value`.
    pub fn repeat(value: V, length: i64) -> Result<Self> {
        Self::from_runs(&[length], vec![value])
    }

    /// Splits the sequence into its ends and values.
    pub fn into_parts(self) -> (Vec<i64>, Vec<V>) {
        (self.ends, self.values)
    }

    /// Mutable access to the values. Changing values cannot break the
    /// invariants, but may leave identical neighbours unmerged.
    pub fn values_mut(&mut self) -> &mut [V] {
        &mut self.values
    }

    /// Appends a run. Zero-length runs are ignored; neighbours are not merged.
    pub fn push_run(&mut self, run: i64, value: V) -> Result<()> {
        if run < 0 {
            return Err(Error::NegativeRun {
                index: self.ends.len(),
                run,
            });
        }
        if run > 0 {
            let end = self.len().checked_add(run).ok_or(Error::Overflow)?;
            self.ends.push(end);
            self.values.push(value);
        }
        Ok(())
    }

    /// Pads the sequence with `value` up to `length`, merging with the last
    /// run if it holds an identical value.
    pub fn extend_to(&mut self, length: i64, value: V) -> Result<()>
    where
        V: Identical,
    {
        let current = self.len();
        if length < current {
            return Err(Error::LengthTooShort {
                requested: length,
                required: current,
            });
        }
        if length > current {
            self.push_merged(length, value);
        }
        Ok(())
    }

    /// Appends a run ending at `end`, extending the last run instead if its
    /// value is identical. `end` must be greater than the current length.
    #[inline]
    pub(crate) fn push_merged(&mut self, end: i64, value: V)
    where
        V: Identical,
    {
        debug_assert!(end > self.len());
        match self.values.last() {
            Some(last) if last.identical(&value) => {
                *self
                    .ends
                    .last_mut()
                    .expect("ends and values have equal length") = end;
            }
            _ => {
                self.ends.push(end);
                self.values.push(value);
            }
        }
    }
}

impl<V: Identical> Rle<V> {
    /// Encodes a dense sequence, merging identical neighbours.
    pub fn from_dense<I: IntoIterator<Item = V>>(values: I) -> Self {
        let mut out = Rle::new();
        for (i, value) in values.into_iter().enumerate() {
            out.push_merged(i as i64 + 1, value);
        }
        out
    }
}

impl<V> Default for Rle<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, V> RleRef<'a, V> {
    /// Borrows cumulative ends and values as a sequence, validating them.
    pub fn from_ends(ends: &'a [i64], values: &'a [V]) -> Result<Self> {
        check_same_len("ends and values", ends.len(), values.len())?;
        validate_ends(ends)?;
        Ok(RleRef { ends, values })
    }

    /// An empty sequence.
    pub fn empty() -> Self {
        RleRef {
            ends: &[],
            values: &[],
        }
    }

    pub(crate) fn new_unchecked(ends: &'a [i64], values: &'a [V]) -> Self {
        debug_assert_eq!(ends.len(), values.len());
        debug_assert!(validate_ends(ends).is_ok());
        RleRef { ends, values }
    }
}

/// Read-only operations shared by [`Rle`] and [`RleRef`].
///
/// Import it (or [`prelude`](crate::prelude)) to call these methods.
pub trait RleLike<V> {
    /// The cumulative end of each run.
    fn ends(&self) -> &[i64];
    /// One value per run.
    fn values(&self) -> &[V];

    /// A borrowed view of this sequence.
    fn view(&self) -> RleRef<'_, V> {
        RleRef::new_unchecked(self.ends(), self.values())
    }

    /// The number of positions the sequence covers.
    fn len(&self) -> i64 {
        self.ends().last().copied().unwrap_or(0)
    }

    /// Whether the sequence has no positions.
    fn is_empty(&self) -> bool {
        self.ends().is_empty()
    }

    /// The number of runs.
    fn num_runs(&self) -> usize {
        self.ends().len()
    }

    /// The length of each run.
    fn runs(&self) -> Vec<i64> {
        let mut prev = 0;
        self.ends()
            .iter()
            .map(|&end| {
                let run = end - prev;
                prev = end;
                run
            })
            .collect()
    }

    /// Iterates over `(start, end, value)` for each run.
    fn iter_runs<'a>(&'a self) -> impl Iterator<Item = (i64, i64, &'a V)>
    where
        V: 'a,
    {
        let ends = self.ends();
        ends.iter()
            .zip(self.values())
            .enumerate()
            .map(move |(i, (&end, value))| (if i == 0 { 0 } else { ends[i - 1] }, end, value))
    }

    /// Expands the sequence into one value per position.
    fn to_dense(&self) -> Vec<V>
    where
        V: Clone,
    {
        let mut out = Vec::with_capacity(self.len() as usize);
        for (start, end, value) in self.iter_runs() {
            out.extend(std::iter::repeat_n(value.clone(), (end - start) as usize));
        }
        out
    }

    /// Copies the sequence into an owned [`Rle`].
    fn to_rle(&self) -> Rle<V>
    where
        V: Clone,
    {
        Rle {
            ends: self.ends().to_vec(),
            values: self.values().to_vec(),
        }
    }

    /// The index of the run containing `pos`, if `pos` is in range.
    fn run_index(&self, pos: i64) -> Option<usize> {
        slice::run_index(self.ends(), pos)
    }

    /// The value at `pos`, if `pos` is in range.
    fn get(&self, pos: i64) -> Option<&V> {
        self.run_index(pos).map(|i| &self.values()[i])
    }

    /// The positions `start..end` as a new sequence starting at 0.
    ///
    /// The range is clamped to the sequence, like slicing in Python, so an
    /// out-of-range slice is shorter or empty rather than an error.
    fn slice(&self, start: i64, end: i64) -> Rle<V>
    where
        V: Clone,
    {
        slice::slice(self.view(), start, end)
    }

    /// Many slices at once, returned as a batch keyed by slice number.
    ///
    /// Each interval must have `0 <= start <= end`; ends past the sequence are
    /// clamped.
    fn getitems(&self, starts: &[i64], ends: &[i64]) -> Result<RleBatch<usize, V>>
    where
        V: Clone,
    {
        slice::getitems(self.view(), starts, ends)
    }

    /// The positions `start..end` without copying them, clamped like
    /// [`slice`](RleLike::slice). See [`SliceView`].
    fn slice_view(&self, start: i64, end: i64) -> SliceView<'_, V> {
        SliceView::new(self.view(), start, end)
    }

    /// Reduces many slices with `f` without copying them, in parallel, for
    /// example `rle.slice_reduce(&starts, &ends, |s| s.mean())`.
    ///
    /// Each interval must have `0 <= start <= end`; ends past the sequence are
    /// clamped.
    fn slice_reduce<'a, T, F>(&'a self, starts: &[i64], ends: &[i64], f: F) -> Result<Vec<T>>
    where
        V: Sync + 'a,
        T: Send,
        F: Fn(SliceView<'a, V>) -> T + Sync,
    {
        let rle = self.view();
        slice::slice_reduce(starts, ends, |_| rle, f)
    }

    /// Whether no two neighbouring runs hold identical values.
    fn is_merged(&self) -> bool
    where
        V: Identical,
    {
        merge::is_merged(self.values())
    }

    /// A copy with identical neighbours merged.
    fn merged(&self) -> Rle<V>
    where
        V: Clone + Identical,
    {
        merge::merged(self.view())
    }

    /// Groups runs by a mask over neighbouring pairs without touching values.
    /// See [`MergePlan`].
    fn merge_plan(&self, mask: &[bool]) -> Result<MergePlan> {
        merge::merge_plan(self.ends(), mask)
    }

    /// Whether both sequences represent the same values at every position,
    /// regardless of how the runs are split.
    fn equals<W: RleLike<V> + ?Sized>(&self, other: &W) -> bool
    where
        V: Identical,
    {
        if self.len() != other.len() {
            return false;
        }
        let (a, b) = (self.values(), other.values());
        Segments::new(self.ends(), other.ends()).all(|(_, i, j)| a[i].identical(&b[j]))
    }

    /// The plan for combining this sequence with `other`. See [`Plan`].
    fn plan_with<B, W: RleLike<B> + ?Sized>(&self, other: &W) -> Plan {
        plan::plan(self.ends(), other.ends())
    }

    /// Combines two sequences position by position and merges the result.
    /// See [`zip_with`](crate::zip_with).
    fn zip_with<B, C, W, F>(
        &self,
        other: &W,
        fill_a: Option<&V>,
        fill_b: Option<&B>,
        f: F,
    ) -> Result<Rle<C>>
    where
        W: RleLike<B> + ?Sized,
        C: Identical,
        F: FnMut(&V, &B) -> C,
    {
        ops::zip_with(self.view(), other.view(), fill_a, fill_b, f)
    }

    /// Like [`zip_with`](RleLike::zip_with) but without merging the result.
    fn zip_with_raw<B, C, W, F>(
        &self,
        other: &W,
        fill_a: Option<&V>,
        fill_b: Option<&B>,
        f: F,
    ) -> Result<Rle<C>>
    where
        W: RleLike<B> + ?Sized,
        F: FnMut(&V, &B) -> C,
    {
        ops::zip_with_raw(self.view(), other.view(), fill_a, fill_b, f)
    }

    /// Applies `f` to every run value and merges the result.
    fn map<W: Identical, F: FnMut(&V) -> W>(&self, mut f: F) -> Rle<W> {
        let mut out = Rle::new();
        for (&end, value) in self.ends().iter().zip(self.values()) {
            out.push_merged(end, f(value));
        }
        out
    }

    /// Applies `f` to every run value without merging the result.
    fn map_raw<W, F: FnMut(&V) -> W>(&self, f: F) -> Rle<W> {
        Rle {
            ends: self.ends().to_vec(),
            values: self.values().iter().map(f).collect(),
        }
    }

    /// Element-wise `==`, padding the shorter side with `V::default()`.
    fn elem_eq<W: RleLike<V> + ?Sized>(&self, other: &W) -> Rle<bool>
    where
        V: PartialEq + Default,
    {
        ops::compare(self.view(), other.view(), |a, b| a == b)
    }

    /// Element-wise `!=`, padding the shorter side with `V::default()`.
    fn elem_ne<W: RleLike<V> + ?Sized>(&self, other: &W) -> Rle<bool>
    where
        V: PartialEq + Default,
    {
        ops::compare(self.view(), other.view(), |a, b| a != b)
    }

    /// Element-wise `<`, padding the shorter side with `V::default()`.
    fn elem_lt<W: RleLike<V> + ?Sized>(&self, other: &W) -> Rle<bool>
    where
        V: PartialOrd + Default,
    {
        ops::compare(self.view(), other.view(), |a, b| a < b)
    }

    /// Element-wise `<=`, padding the shorter side with `V::default()`.
    fn elem_le<W: RleLike<V> + ?Sized>(&self, other: &W) -> Rle<bool>
    where
        V: PartialOrd + Default,
    {
        ops::compare(self.view(), other.view(), |a, b| a <= b)
    }

    /// Element-wise `>`, padding the shorter side with `V::default()`.
    fn elem_gt<W: RleLike<V> + ?Sized>(&self, other: &W) -> Rle<bool>
    where
        V: PartialOrd + Default,
    {
        ops::compare(self.view(), other.view(), |a, b| a > b)
    }

    /// Element-wise `>=`, padding the shorter side with `V::default()`.
    fn elem_ge<W: RleLike<V> + ?Sized>(&self, other: &W) -> Rle<bool>
    where
        V: PartialOrd + Default,
    {
        ops::compare(self.view(), other.view(), |a, b| a >= b)
    }

    /// Shifts values `distance` positions to the right (left if negative),
    /// keeping the length: positions shifted in hold `fill`, values shifted
    /// past either end are dropped. The result is merged.
    fn shift(&self, distance: i64, fill: V) -> Rle<V>
    where
        V: Clone + Identical,
    {
        transform::shift(self.view(), distance, fill)
    }

    /// The sequence back to front.
    fn reversed(&self) -> Rle<V>
    where
        V: Clone,
    {
        transform::reversed(self.view())
    }

    /// The half-open ranges `(starts, ends)` where `pred` holds, with
    /// touching ranges joined. For example, where coverage is positive.
    fn ranges_where<F: FnMut(&V) -> bool>(&self, pred: F) -> (Vec<i64>, Vec<i64>) {
        transform::ranges_where(self.view(), pred)
    }

    /// The sum over all positions (each value times its run length).
    fn sum(&self) -> V
    where
        V: Num,
    {
        reduce::sum(reduce::weighted(self.view()))
    }

    /// The mean over all positions, or `None` if the sequence is empty.
    fn mean(&self) -> Option<f64>
    where
        V: Num,
    {
        reduce::mean(reduce::weighted(self.view()), self.len())
    }

    /// The variance over all positions with `ddof` delta degrees of freedom
    /// (0 for population, 1 for sample). `None` if `len() <= ddof`.
    fn variance(&self, ddof: i64) -> Option<f64>
    where
        V: Num,
    {
        reduce::variance(reduce::weighted(self.view()), self.len(), ddof)
    }

    /// The square root of [`variance`](RleLike::variance).
    fn std(&self, ddof: i64) -> Option<f64>
    where
        V: Num,
    {
        self.variance(ddof).map(f64::sqrt)
    }

    /// The smallest value, ignoring values that are not comparable with
    /// themselves (such as `NaN`).
    fn min(&self) -> Option<&V>
    where
        V: PartialOrd,
    {
        reduce::extreme(self.values(), |a, b| a < b)
    }

    /// The largest value, ignoring values that are not comparable with
    /// themselves (such as `NaN`).
    fn max(&self) -> Option<&V>
    where
        V: PartialOrd,
    {
        reduce::extreme(self.values(), |a, b| a > b)
    }
}

impl<V> RleLike<V> for Rle<V> {
    fn ends(&self) -> &[i64] {
        &self.ends
    }
    fn values(&self) -> &[V] {
        &self.values
    }
}

impl<V> RleLike<V> for RleRef<'_, V> {
    fn ends(&self) -> &[i64] {
        self.ends
    }
    fn values(&self) -> &[V] {
        self.values
    }
}

/// Equality of the represented sequences; see [`RleLike::equals`].
impl<V: Identical> PartialEq for Rle<V> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl<V: Identical> PartialEq for RleRef<'_, V> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

fn debug_rle<V: fmt::Debug>(
    name: &str,
    rle: RleRef<'_, V>,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    f.debug_struct(name)
        .field("runs", &rle.runs())
        .field("values", &rle.values)
        .finish()
}

impl<V: fmt::Debug> fmt::Debug for Rle<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_rle("Rle", self.view(), f)
    }
}

impl<V: fmt::Debug> fmt::Debug for RleRef<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        debug_rle("RleRef", *self, f)
    }
}
